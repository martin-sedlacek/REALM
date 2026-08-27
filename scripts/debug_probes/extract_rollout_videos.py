#!/usr/bin/env python
"""Host-side (no container, no GPU): unpack REALM rollout videos from parquet into .mp4 files.

REALM stores each rollout's encoded mp4 as a blob in `videos/<task>.parquet` (one row per repeat).
That is the right shape for archiving and the wrong shape for putting a clip on a slide, so this
writes each row out as its own file and names it with the outcome, joining against the run's
`reports/<task>_<pert>.csv` on (task, perturbation, repeat):

    <out>/<SR|FAIL>/<task>__<pert>__rep<NN>__<stage>.mp4

Sorting successes and failures into separate directories is the point -- for a localization slide
you generally want a couple of each, labelled, and picking them out of 175 unlabelled blobs by hand
is the part that wastes an afternoon.

    python scripts/debug_probes/extract_rollout_videos.py \
        --run <REALM_LOGS>/<experiment>/<model>/<run_id> [--out <dir>] [--only SR] [--limit 3]

Requires pyarrow (uv run --with pyarrow ...). Frame counts are reported per task so a run recorded
under --og_lite -- one frame per action chunk, ~1/8 of real time -- is obvious rather than silent.
"""
import argparse
import csv
import glob
import os

import pyarrow.parquet as pq


def load_outcomes(run_dir):
    """{(task, perturbation, repeat): (binary_SR, stage)} from every report CSV in the run.

    `repeat` is the row index within a (task, perturbation) report: eval.py appends one row per
    rollout in order and rewrites the file whole, so row order IS repeat order -- the same
    assumption the videos parquet makes with its own `repeat` column.
    """
    out = {}
    for path in glob.glob(os.path.join(run_dir, "reports", "*.csv")):
        with open(path) as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                key = (row["task"], row["perturbation"], i)
                try:
                    sr = float(row.get("binary_SR", 0)) > 0.5
                except ValueError:
                    sr = False
                out[key] = (sr, (row.get("stage") or "").strip().replace(" | ", "+") or "NA")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir holding videos/ and reports/")
    p.add_argument("--out", default=None, help="output dir (default: <run>/mp4)")
    p.add_argument("--only", choices=["SR", "FAIL"], default=None, help="extract just one outcome")
    p.add_argument("--limit", type=int, default=None, help="at most N clips per task per outcome")
    args = p.parse_args()

    out_root = args.out or os.path.join(args.run, "mp4")
    outcomes = load_outcomes(args.run)
    if not outcomes:
        print(f"WARNING: no reports under {args.run}/reports -- every clip will be filed as UNKNOWN")

    total, kept = 0, 0
    for path in sorted(glob.glob(os.path.join(args.run, "videos", "*.parquet"))):
        table = pq.read_table(path).to_pydict()
        n = len(table["video"])
        per_bucket = {}
        sizes = []
        for i in range(n):
            total += 1
            task, pert, rep = table["task"][i], table["perturbation"][i], table["repeat"][i]
            sr, stage = outcomes.get((task, pert, rep), (None, "UNKNOWN"))
            bucket = "UNKNOWN" if sr is None else ("SR" if sr else "FAIL")
            if args.only and bucket != args.only:
                continue
            if args.limit and per_bucket.get(bucket, 0) >= args.limit:
                continue
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
            blob = bytes(table["video"][i])
            sizes.append(len(blob))
            d = os.path.join(out_root, bucket)
            os.makedirs(d, exist_ok=True)
            name = f"{task}__{pert}__rep{rep:02d}__{stage}.mp4"
            with open(os.path.join(d, name), "wb") as fh:
                fh.write(blob)
            kept += 1
        task_name = os.path.basename(path).removesuffix(".parquet")
        mean_kb = sum(sizes) / len(sizes) / 1e3 if sizes else 0
        print(f"{task_name:28s} {n:3d} rollouts -> {sum(per_bucket.values()):3d} written "
              f"{dict(sorted(per_bucket.items()))}  mean {mean_kb:.0f} kB")

    print(f"\n{kept} of {total} clips -> {out_root}")


if __name__ == "__main__":
    main()
