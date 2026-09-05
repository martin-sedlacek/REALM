#!/usr/bin/env python3
"""Re-score a recorded REALM rollout with a Robometer server, causally, and render the reward
unrolling as a graph above the video.

    uv run --with "imageio[pyav]" --with pyarrow --with requests \
        python scripts/robometer_replay_video.py <run_dir> --task put_banana_into_box \
        --port 8010 --out banana_robometer.mp4 [--repeat 0] [--every 8] [--camera base]

<run_dir> is a realm.paths.run_log_dir() directory: it has reports/{task}_Default.csv and
videos/{task}.parquet. Nothing about the simulator is needed; only the Robometer server.

WHY REPLAY RATHER THAN --robometer. On one GPU the policy server, Isaac and Robometer-4B do not
fit together (pi0.5 alone reserves half the card). Recording with the rubric first and scoring the
recording afterwards needs only two of the three at a time, and the model sees the SAME pixels the
live scorer would have, up to H.264 round-tripping and the recorder's height-480 scaling:

  * cameras   the base exterior tile (the view the policy is shown) AND the wrist tile by default
              (VideoRecorder tiles base | wrist side by side in every run; --multi-view adds a 2x2
              layout, handled), fused by --fusion (max by default), as the live scorer does;
  * cadence   one query per `--every` frames, each over the PREFIX frames[0:t+1] -- causal, like
              the live scorer's one query per action chunk (horizon 8), never the whole clip;
  * clip      each prefix linspace-subsampled to --max-frames (16, the training clip length; first
              and current frame kept), as realm.progress_scorer does -- the server does not
              subsample, and un-subsampled prefixes OOM'd it at ~440 frames;
  * downscale realm.progress_scorer.downscale_frame's rule: longest side 256, bilinear;
  * prompt    the report's `instruction` column, i.e. what the policy was given.

Every frame of the recording is one control step when the run used --no-render_on_demand (the
recorder adds a frame per fresh observation), so the x axis is the control step. Under
render-on-demand a frame is one action chunk instead; pass --every 1 then. The plotted curve is the
RAW Robometer score; the dashed line is the task's calibrated ceiling (raw at which --robometer
counts success, from realm/config/robometer_calibration.yaml) and the dotted one its floor; green
ticks are the RUBRIC's stage completions from `task_progression_timestamps`, the privileged-state
ground truth, so the eye can compare when the rubric says a stage completed with when the reward
model noticed.

The scored trace is also written next to the video as <out>.json (steps, progress, success_prob,
instruction, rubric stage timestamps) so the numbers can be inspected without the video.
"""
import argparse
import ast
import io
import json
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "robometer-client" / "src"))
sys.path.insert(0, str(REPO_ROOT))
from robometer_client import RobometerClient, subsample_frames  # noqa: E402

from realm.robometer_calibration import (  # noqa: E402  (host-importable: no omnigibson)
    DEFAULT_CALIBRATION_PATH,
    calibration_for,
    load_calibration,
)

FRAME_SIZE = 256          # realm.progress_scorer.DEFAULT_ROBOMETER_FRAME_SIZE


def parse_list(s):
    s = str(s or "").strip()
    if not s or s == "nan":
        return []
    try:
        return [float(x) for x in ast.literal_eval(s)]
    except Exception:
        return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]


def downscale(frame, longest=FRAME_SIZE):
    """realm.progress_scorer.downscale_frame, duplicated so this stays container-free."""
    frame = np.ascontiguousarray(np.asarray(frame)[..., :3])
    h, w = frame.shape[:2]
    s = longest / max(h, w)
    if s >= 1.0:
        return frame.astype(np.uint8, copy=False)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return np.asarray(Image.fromarray(frame.astype(np.uint8, copy=False)).resize(size, Image.BILINEAR))


def split_tiles(frames):
    """{camera: frames} for either VideoRecorder layout, dispatched on aspect ratio (16:9 tiles:
    base|wrist lands near 3.55, the 2x2 multi-view grid near 1.78)."""
    h, w = frames.shape[1:3]
    if w / h > 2.6:
        return {"base": frames[:, :, : w // 2], "wrist": frames[:, :, w // 2:]}
    top, bot = frames[:, : h // 2], frames[:, h // 2:]
    return {"base": top[:, :, : w // 2], "base2": top[:, :, w // 2:], "wrist": bot[:, :, : w // 2]}


def load_rollout(run_dir, task, repeat):
    df = pd.read_parquet(run_dir / "videos" / f"{task}.parquet")
    idx = {int(r): i for i, r in enumerate(df["repeat"].tolist())}   # last occurrence wins
    if repeat not in idx:
        raise SystemExit(f"repeat {repeat} not in {task}.parquet (has {sorted(idx)})")
    frames = np.asarray(iio.imread(io.BytesIO(bytes(df.iloc[idx[repeat]]["video"])),
                                   plugin="pyav", format="rgb24"))
    rows = pd.read_csv(run_dir / "reports" / f"{task}_Default.csv")
    row = rows[rows["run_id"] == repeat].iloc[0]
    return frames, row


FUSIONS = {"max": max, "min": min, "mean": lambda xs: sum(xs) / len(xs)}   # realm.progress_scorer.FUSIONS


def score_causally(client, cams, instruction, every, max_frames, fusion):
    """Query the prefix frames[0:t+1] of every camera at t = every-1, 2*every-1, ... and the final
    frame, each prefix linspace-subsampled to `max_frames`, all cameras of a step in one request,
    fused by `fusion` -- exactly what realm.progress_scorer.RobometerScorer sends and records.
    Returns (steps, fused raw, fused success, {camera: raw trace})."""
    names = list(cams)
    n = len(cams[names[0]])
    small = {c: np.stack([downscale(f) for f in cams[c]]) for c in names}
    ts = list(range(every - 1, n, every))
    if not ts or ts[-1] != n - 1:
        ts.append(n - 1)
    fuse = FUSIONS[fusion]
    steps, progress, success, per_cam = [], [], [], {c: [] for c in names}
    for t in ts:
        results = client.progress_batch([subsample_frames(small[c][: t + 1], max_frames) for c in names],
                                        [instruction] * len(names))
        for c, r in zip(names, results):
            per_cam[c].append(r.reward)
        raw = fuse([r.reward for r in results])
        succ = [r.success_prob for r in results]
        steps.append(t)
        progress.append(raw)
        success.append(None if any(s is None for s in succ) else float(fuse(succ)))
        print(f"  t={t:4d}  clip={t + 1:4d}  " + "  ".join(f"{c}={r.reward:.3f}" for c, r in zip(names, results))
              + f"  {fusion}={raw:.3f}", flush=True)
    return steps, progress, success, per_cam


CAMERA_COLORS = {"base": "#2980b9", "wrist": "#e67e22", "base2": "#8e44ad"}


def render(frames, steps, progress, stages, cal, title, out, fps, per_cam=None):
    """Video frames with the RAW fused trace (bold) and each camera's own trace (thin, coloured)
    revealed up to each frame drawn above them, plus the task's calibration lines (ceiling =
    success, floor = start state) and the calibrated value."""
    per_cam = per_cam or {}
    H, W = frames.shape[1:3]
    plot_h = max(240, H // 2)
    dpi = 100
    n = len(frames)
    xmax = n - 1
    # For frame i, the last query at or before i is what the live scorer would have recorded.
    q_upto = np.searchsorted(np.asarray(steps), np.arange(n), side="right")
    panels = []
    for i in range(n):
        k = int(q_upto[i])
        fig = plt.figure(figsize=(W / dpi, plot_h / dpi), dpi=dpi)
        ax = fig.add_axes([0.07, 0.20, 0.90, 0.66])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.0)
        if cal.calibrated:
            ax.axhline(cal.ceiling, ls="--", lw=1.2, color="#c0392b")
            ax.text(xmax * 0.995, cal.ceiling + 0.02, f"success: raw >= ceiling {cal.ceiling:g}",
                    ha="right", va="bottom", fontsize=8, color="#c0392b")
            if cal.floor > 0:
                ax.axhline(cal.floor, ls=":", lw=1.0, color="#7f8c8d")
                ax.text(xmax * 0.995, cal.floor + 0.02, f"floor {cal.floor:g}", ha="right",
                        va="bottom", fontsize=8, color="#7f8c8d")
        else:
            ax.text(xmax * 0.995, 0.93, "no calibration entry for this task", ha="right",
                    va="top", fontsize=8, color="#c0392b")
        for s in stages:
            if s <= i:
                ax.axvline(s, color="#27ae60", lw=1.2, alpha=0.7)
        if k:
            for c, tr in per_cam.items():
                ax.plot(steps[:k], tr[:k], lw=1.1, alpha=0.8, color=CAMERA_COLORS.get(c, "#95a5a6"), label=c)
            ax.plot(steps[:k], progress[:k], lw=2.2, color="#2c3e50", label="fused" if per_cam else None)
            ax.scatter([steps[k - 1]], [progress[k - 1]], s=34, color="#2c3e50", zorder=5)
            if per_cam:
                ax.legend(loc="lower right", fontsize=7, ncol=len(per_cam) + 1, framealpha=0.7)
            label = f"raw {progress[k-1]:.3f}   running max {max(progress[:k]):.3f}"
            if cal.calibrated:
                label += f"   calibrated TP {max(cal.apply(p) for p in progress[:k]):.2f}"
            ax.text(0.015, 0.93, label, transform=ax.transAxes, fontsize=11, va="top",
                    fontweight="bold")
        ax.axvline(i, color="#7f8c8d", lw=0.8, alpha=0.6)
        ax.set_xlabel("control step", fontsize=9)
        ax.set_ylabel("Robometer raw progress", fontsize=9)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if buf.shape[1] != W:
            buf = np.asarray(Image.fromarray(buf).resize((W, buf.shape[0])))
        panel = np.vstack([buf, frames[i]])
        # libx264 + yuv420p need even dimensions; a 16:9 tile at height 480 is 853 wide, and
        # avcodec_open2 fails opaquely ("Generic error in an external library") on the odd width.
        panel = panel[: panel.shape[0] - panel.shape[0] % 2, : panel.shape[1] - panel.shape[1] % 2]
        panels.append(np.ascontiguousarray(panel))
    write_mp4(out, panels, fps)


def write_mp4(path, panels, fps):
    """Encode with pyav directly: imageio's pyav plugin re-derives the stream size per frame and
    dies with "Cannot change width after codec is open" on this layout."""
    import av

    h, w = panels[0].shape[:2]
    assert all(p.shape == panels[0].shape for p in panels), "panels differ in shape"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for panel in panels:
            frame = av.VideoFrame.from_ndarray(panel, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--every", type=int, default=8, help="query every N frames (the action-chunk horizon)")
    ap.add_argument("--cameras", default="base,wrist",
                    help="comma-separated cameras scored per query: base, wrist, base2 (default base,wrist)")
    ap.add_argument("--fusion", default="max", choices=sorted(FUSIONS),
                    help="how the cameras' raw scores combine before calibration (default max)")
    ap.add_argument("--show-camera", default="base", help="which camera tile to show in the video (default base)")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION_PATH,
                    help="per-task floor/ceiling table drawn on the plot (realm/config/robometer_calibration.yaml)")
    ap.add_argument("--fps", type=int, default=15, help="output fps; 15 = real time at CONTROL_HZ")
    ap.add_argument("--max-frames", type=int, default=16,
                    help="linspace-subsample each prefix clip to this many frames before sending, "
                         "first and last kept (robometer's own training rule; 0 = send every frame)")
    ap.add_argument("--from-json", action="store_true",
                    help="skip scoring and re-render from <out>.json written by an earlier run")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    frames, row = load_rollout(a.run_dir, a.task, a.repeat)
    cams = split_tiles(frames)
    scored = [c.strip() for c in a.cameras.split(",") if c.strip()]
    missing = [c for c in scored + [a.show_camera] if c not in cams]
    if missing:
        raise SystemExit(f"camera(s) {missing} not in this recording ({sorted(cams)})")
    instruction = str(row["instruction"])
    stages = [int(v) for v in parse_list(row.get("task_progression_timestamps"))]
    rubric = float(row["task_progression"])
    print(f"{a.task} repeat {a.repeat}: {len(frames)} frames, cameras {sorted(cams)}, "
          f"instruction={instruction!r}, rubric TP={rubric:.2f} ({row['stage']}), "
          f"stage steps={stages}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = Path(str(a.out) + ".json")
    if a.from_json and trace_path.exists():
        rec = json.load(open(trace_path))
        steps, progress, per_cam = rec["steps"], rec["progress"], rec.get("per_camera", {})
        print(f"reusing {len(steps)} scored queries from {trace_path} "
              f"(cameras {rec.get('cameras')}, fusion {rec.get('fusion')})", flush=True)
    else:
        client = RobometerClient(host=a.host, port=a.port, timeout_s=600)
        client.wait_until_healthy(timeout_s=1800)
        steps, progress, success, per_cam = score_causally(
            client, {c: cams[c] for c in scored}, instruction, a.every, a.max_frames, a.fusion)
        # Written BEFORE rendering: the queries are the expensive part, and a rendering hiccup
        # must not cost them. --from-json picks this file up.
        json.dump({"task": a.task, "repeat": a.repeat, "instruction": instruction,
                   "cameras": scored, "fusion": a.fusion, "every": a.every, "max_frames": a.max_frames,
                   "steps": steps, "progress": progress, "success_prob": success,
                   "per_camera": per_cam,
                   "rubric_task_progression": rubric, "rubric_stage": str(row["stage"]),
                   "rubric_stage_steps": stages, "robometer_max": max(progress),
                   "robometer_final": progress[-1]},
                  open(trace_path, "w"), indent=1)

    cal = calibration_for(a.task, load_calibration(a.calibration))
    title = (f"{a.task}  \"{instruction}\"  |  rubric {row['stage']} (TP {rubric:.2f})  |  "
             f"green ticks: rubric stages")
    render(cams[a.show_camera], steps, progress, stages, cal, title, str(a.out), a.fps, per_cam)
    cal_max = max(cal.apply(p) for p in progress)
    print(f"wrote {a.out}  ({len(frames)} frames @ {a.fps} fps)  raw max={max(progress):.3f} "
          f"final={progress[-1]:.3f}  calibrated TP={cal_max:.2f} "
          f"(floor {cal.floor:g}, ceiling {cal.ceiling:g}{'' if cal.calibrated else ', NO ENTRY'})  "
          f"rubric={rubric:.2f}")


if __name__ == "__main__":
    main()
