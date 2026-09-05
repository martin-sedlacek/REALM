#!/usr/bin/env python3
"""Re-score already-recorded REALM rollouts with a Robometer server, varying camera and prompt.

    uv run --no-project --with "imageio[pyav]" --with pandas --with pyarrow --with requests \
        python rbm_replay.py --host <node> --port 9100 --out replay.json [--tasks pick_spoon,...]

WHY OFFLINE REPLAY. The question "does a more specific prompt help, and is the wrist camera any
use?" does not need new rollouts: `videos/{task}.parquet` already holds every rollout, and
`realm_logging.VideoRecorder._build_frame` tiles **base | wrist side by side** in EVERY run --
`--multi-view` is not required for the wrist view to be recorded. So one robometer server and the
existing artifacts answer both, at zero simulator cost, and every condition sees literally the same
pixels, which a re-run could never guarantee (pi0.5 does not reproduce; see the stream).

WHAT IS AND IS NOT COMPARABLE TO THE LIVE RUN. The live scorer sent RAW frames; these are H.264
round-tripped and scaled to height 480, so absolute values will drift slightly. The control
condition (base camera + original prompt, whole clip) is compared against the live run's LAST trace
entry -- NOT against `task_progression`, which is the running MAX over queries and is a different
quantity. If the control tracks the live last-entry, the pipeline is sound and the other conditions
can be read.

Rows in a videos parquet can outnumber the report's rollouts when a failed run wrote partial rows
before a rerun appended under the same run_id (rbm_t4_n8: order [0,1,2,3,0,1,2,3,4,5,6,7], the
first four from the crashed job 204598). LAST occurrence per repeat wins, which is the rerun's.
"""
import argparse
import io
import json
import sys

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/mnt/home_lustre/sedlam56/projects/wt/realm_robometer/packages/robometer-client/src")
from robometer_client import RobometerClient  # noqa: E402

BASE = "/mnt/home_lustre/sedlam56/projects/REALM/logs/robometer_reliability/checkpoints_pi05_droid_jointpos"
MV_BASE = "/mnt/home_lustre/sedlam56/projects/REALM/logs/robometer_multiview/checkpoints_pi05_droid_jointpos"

# run_id -> (task file stem, original instruction, enhanced instruction)
#
# The enhanced prompts are written from the ACTUAL rubric in realm/config/tasks/task_progressions.yaml
# and the thresholds in realm/environments/task_progression.py, not from the task name:
#   pick   = REACH, GRASP, LIFT_LARGE   (check_lift_and_distance_condition lift_threshold=0.075)
#   put    = REACH, GRASP, LIFT_SLIGHT, MOVE_CLOSE, PLACE_INTO   (lift_threshold=0.01)
#   rotate = REACH, GRASP, ROTATED      (rot_threshold=1.1 rad ~ 63 deg)
#   push   = REACH, TOUCH, TOGGLED_ON   (ToggledOn object state)
# so "success" in the prompt means the same event the rubric scores.
RUNS = {
    "rbm_t4_n8": ("pick_spoon", "pick up the spoon",
                  "The robot arm reaches the spoon, closes its gripper on it, and lifts it more "
                  "than 7 centimetres straight off the table so the spoon hangs clearly in the air."),
    "rbm_t1_n8": ("put_banana_into_box", "put the banana in the box",
                  "The robot arm reaches the banana, grasps it, lifts it off the table, carries it "
                  "over the box and releases it so the banana ends up resting inside the box."),
    "rbm_t3_n8": ("rotate_mug", "rotate the mug",
                  "The robot arm reaches the mug, grasps it, and turns it in place by more than 60 "
                  "degrees without lifting it away."),
    "rbm_t0_n8": ("put_green_block_into_bowl", "put the green block in the bowl",
                  "The robot arm reaches the green block, grasps it, lifts it off the table, "
                  "carries it over the bowl and releases it so the block ends up inside the bowl."),
    "rbm_t7_n8_ctl": ("push_switch", "push the light switch",
                      "The robot arm reaches the light switch on the wall, touches it, and pushes "
                      "it far enough to flip the switch into the on position."),
}


def downscale(frames, longest=256):
    """Match realm/progress_scorer.py::downscale_frame -- longest side `longest`, BILINEAR, no upscale."""
    out = []
    for f in frames:
        f = np.ascontiguousarray(np.asarray(f)[..., :3])
        h, w = f.shape[:2]
        s = longest / max(h, w)
        if s >= 1.0:
            out.append(f.astype(np.uint8, copy=False))
        else:
            size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
            out.append(np.asarray(Image.fromarray(f.astype(np.uint8, copy=False)).resize(size, Image.BILINEAR)))
    return np.stack(out)


def split_tiles(a):
    """{camera: frames} for either VideoRecorder layout, chosen by aspect ratio.

    _build_frame produces, before the VIDEO_TARGET_HEIGHT=480 cap:
      no second exterior : hstack(base, wrist)                      -> aspect 2 * tile_aspect
      --multi-view       : vstack(hstack(base, base2),
                                  hstack(wrist, black))             -> aspect 1 * tile_aspect
    Tiles are 16:9, so the single row lands near 3.55 and the 2x2 near 1.78 -- far enough apart to
    dispatch on, and asserted rather than assumed.
    """
    h, w = a.shape[1:3]
    ratio = w / h
    if ratio > 2.6:                                   # base | wrist
        return {"base": a[:, :, : w // 2], "wrist": a[:, :, w // 2:]}
    # base | base2  over  wrist | black
    top, bot = a[:, : h // 2], a[:, h // 2:]
    black = bot[:, :, w // 2:]
    if black.mean() > 8.0:
        raise AssertionError(f"expected a black bottom-right pad in the 2x2 layout, "
                             f"got mean {black.mean():.1f} -- layout guess is wrong")
    return {"base": top[:, :, : w // 2], "base2": top[:, :, w // 2:], "wrist": bot[:, :, : w // 2]}


def load_rollouts(run_id, task):
    """[(repeat, {camera: frames})] for the winning rows."""
    df = pd.read_parquet(f"{BASE}/{run_id}/videos/{task}.parquet")
    keep = {}
    for i, rep in enumerate(df["repeat"].tolist()):
        keep[int(rep)] = i                      # last occurrence wins
    rollouts = []
    for rep in sorted(keep):
        vid = bytes(df.iloc[keep[rep]]["video"])
        a = np.asarray(iio.imread(io.BytesIO(vid), plugin="pyav", format="rgb24"))
        rollouts.append((rep, split_tiles(a)))
    return rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9100)
    # NOT ",".join(RUNS) as a default: argparse evaluates it before --multiview renames the keys,
    # so the default would name the 2-tile runs and every lookup would miss.
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", default="replay.json")
    ap.add_argument("--multiview", action="store_true",
                    help="read the robometer_multiview experiment (rbm_mv_t*_n8, 3 cameras)")
    a = ap.parse_args()
    global BASE
    if a.multiview:
        BASE = MV_BASE
        for k in list(RUNS):
            RUNS[k.replace("rbm_t", "rbm_mv_t").replace("_ctl", "")] = RUNS.pop(k)

    client = RobometerClient(host=a.host, port=a.port, timeout_s=600)
    print("health:", client.wait_until_healthy(timeout_s=1800), flush=True)

    results = []
    for run_id in (a.tasks.split(",") if a.tasks else list(RUNS)):
        run_id = run_id.strip()
        if run_id not in RUNS:
            print(f"!! unknown run_id {run_id}", file=sys.stderr)
            continue
        task, orig, enh = RUNS[run_id]
        # The report keys rollouts by `run_id` (0..repeats-1), NOT by a `repeat` column -- that name
        # only exists in the parquets. Same index space, different column name.
        _rep = pd.read_csv(f"{BASE}/{run_id}/reports/{task}_Default.csv").to_dict("records")
        rep_rows = {int(r["run_id"]): r for r in _rep}
        rollouts = load_rollouts(run_id, task)
        _cams = rollouts[0][1]
        print(f"\n=== {run_id} / {task}: {len(rollouts)} rollouts, "
              f"{next(iter(_cams.values())).shape[0]} frames, cameras={sorted(_cams)}, "
              f"tile {next(iter(_cams.values())).shape[1:3]}", flush=True)
        for rep, cams in rollouts:
            row = rep_rows.get(rep, {})
            rec = {"run_id": run_id, "task": task, "repeat": rep,
                   "rubric": float(row.get("rubric_task_progression", float("nan"))),
                   "live_progress_max": float(row.get("task_progression", float("nan"))),
                   "live_trace_last": None,
                   "n_frames": int(next(iter(cams.values())).shape[0]),
                   "cameras": sorted(cams)}
            tr = str(row.get("robometer_progress_trace", "") or "")
            try:
                vals = [float(x) for x in tr.strip("[]").replace(",", " ").split()]
                rec["live_trace_last"] = vals[-1] if vals else None
            except Exception:
                pass
            for cam, frames in sorted(cams.items()):
                small = downscale(frames)
                for pname, prompt in (("orig", orig), ("enh", enh)):
                    r = client.progress(small, prompt)
                    rec[f"{cam}_{pname}"] = round(r.reward, 6)
                    rec[f"{cam}_{pname}_succ"] = (None if r.success_prob is None
                                                  else round(float(r.success_prob), 6))
            print(f"  rep {rep}: rubric={rec['rubric']:.3f} " +
                  "  ".join(f"{c}={rec[c + '_enh']:.3f}" for c in sorted(cams)), flush=True)
            results.append(rec)

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(results)} rows -> {a.out}")


if __name__ == "__main__":
    main()
