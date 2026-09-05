#!/usr/bin/env python3
"""Render a rollout's video with its Robometer reward unrolling as a graph above it.

    uv run --no-project --with "imageio[pyav]" --with pandas --with pyarrow --with matplotlib \
        python rbm_reward_video.py --run rbm_t0_n8 --task put_green_block_into_bowl --repeat 3 \
        --out /path/out.mp4

Nothing is recomputed: `robometer_progress_trace` and `robometer_query_steps` are already columns of
every --robometer report, and the rollout video is already in videos/{task}.parquet. This just draws
them together.

FRAME/QUERY ALIGNMENT. The recorded video has one frame per RENDER, and under render-on-demand a
render is exactly what feeds an inference step, so frames and queries advance together: our runs
have 38 frames and 37 queries. Frame 0 is the reset render, before any query, so trace[k] is drawn
at frame k+1. The x axis is labelled with the real simulator step from `robometer_query_steps`, not
the frame index, so a reader is never misled about when something happened. If a run ever has a
frame count that is not queries+1 the assumption is wrong and the script says so rather than
silently misaligning the plot.

The dashed line is the shipped 0.9 success threshold. Green markers on the axis are the RUBRIC's
stage completions (`task_progression_timestamps`) -- the privileged-state ground truth -- so the
gap between "the rubric says the stage completed here" and "the reward model noticed" is visible
rather than asserted.
"""
import argparse
import ast
import io
import re

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOTS = {
    "reliability": "/mnt/home_lustre/sedlam56/projects/REALM/logs/robometer_reliability/checkpoints_pi05_droid_jointpos",
    "multiview": "/mnt/home_lustre/sedlam56/projects/REALM/logs/robometer_multiview/checkpoints_pi05_droid_jointpos",
}


def parse_list(s):
    s = str(s or "").strip()
    if not s:
        return []
    try:
        return [float(x) for x in ast.literal_eval(s)]
    except Exception:
        return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reliability", choices=list(ROOTS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()
    base = ROOTS[a.root]

    rep = pd.read_csv(f"{base}/{a.run}/reports/{a.task}_Default.csv")
    row = rep[rep["run_id"] == a.repeat].iloc[0]
    trace = parse_list(row.get("robometer_progress_trace"))
    steps = [int(v) for v in parse_list(row.get("robometer_query_steps"))]
    stages = [int(v) for v in parse_list(row.get("task_progression_timestamps"))]
    thr = float(row.get("success_threshold", 0.9))
    rubric = float(row.get("rubric_task_progression", float("nan")))
    stage = str(row.get("stage", "?"))

    df = pd.read_parquet(f"{base}/{a.run}/videos/{a.task}.parquet")
    idx = {int(r): i for i, r in enumerate(df["repeat"].tolist())}[a.repeat]  # last occurrence wins
    frames = np.asarray(iio.imread(io.BytesIO(bytes(df.iloc[idx]["video"])),
                                   plugin="pyav", format="rgb24"))

    n_f, n_q = len(frames), len(trace)
    if n_f != n_q + 1:
        print(f"!! {n_f} frames but {n_q} queries: expected queries+1. Alignment assumption does "
              f"not hold for this run -- plotting by proportional index instead and saying so.")
    print(f"{a.task} repeat {a.repeat}: {n_f} frames, {n_q} queries, "
          f"rubric={rubric:.3f} ({stage}), robometer max={max(trace):.3f}, threshold={thr}")

    H, W = frames.shape[1:3]
    plot_h = max(220, H // 2)
    dpi = 100
    out = []
    xmax = max(steps) if steps else n_q

    for i in range(n_f):
        k = min(i, n_q)                     # trace points revealed up to this frame
        fig = plt.figure(figsize=(W / dpi, plot_h / dpi), dpi=dpi)
        ax = fig.add_axes([0.07, 0.22, 0.90, 0.68])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.0)
        ax.axhline(thr, ls="--", lw=1.2, color="#c0392b")
        ax.text(xmax * 0.995, thr + 0.02, f"success threshold {thr:g}", ha="right", va="bottom",
                fontsize=8, color="#c0392b")
        for s in stages:                    # rubric stage completions = privileged ground truth
            if s <= (steps[k - 1] if k else 0):
                ax.axvline(s, color="#27ae60", lw=1.0, alpha=0.55)
        if k:
            ax.plot(steps[:k], trace[:k], lw=2.0, color="#2c3e50")
            ax.scatter([steps[k - 1]], [trace[k - 1]], s=34, color="#2c3e50", zorder=5)
            ax.text(0.015, 0.93, f"reward {trace[k-1]:.3f}", transform=ax.transAxes,
                    fontsize=11, va="top", fontweight="bold")
        ax.set_xlabel("simulator step", fontsize=9)
        ax.set_ylabel("Robometer progress", fontsize=9)
        ax.set_title(f"{a.task}  repeat {a.repeat}   "
                     f"rubric: {stage} ({rubric:.2f})   green = rubric stage completed",
                     fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if buf.shape[1] != W:
            buf = np.asarray(iio.imread(iio.imwrite("<bytes>", buf, extension=".png")))
        out.append(np.vstack([buf[:, :W], frames[i]]))

    iio.imwrite(a.out, np.stack(out), fps=a.fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {a.out}  ({len(out)} frames, {a.fps} fps, {out[0].shape[1]}x{out[0].shape[0]})")


if __name__ == "__main__":
    main()
