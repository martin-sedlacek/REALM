"""Extract the mp4s stored in a REALM results dir's videos/*.parquet.

02_evaluate.py encodes each rollout to mp4 and stores the resulting bytes in a parquet column
(realm_logging.py writes via a NamedTemporaryFile and keeps the blob), so the `video` column holds
complete MP4 files -- not frame arrays. Extracting them is a byte copy, no re-encoding.

The mp4s under videos_mp4/ in older run dirs were produced ad hoc; this makes the step reproducible.

Usage -- needs only pandas, so the container is not required:
    python scripts/videos_parquet_to_mp4.py <run_dir> [--prefix name]

<run_dir> contains videos/, e.g.
    REALM/logs/og391_gravfix_robolab/checkpoints_pi05_droid_jointpos/20260811_141731

Output names carry the repeat index plus task_progression and final stage from
reports/<task>_<pert>.csv when present, so `ls` tells you which rollouts are worth watching.
"""

import argparse
import csv
import os
import sys

import pandas as pd


def load_report(run_dir):

    out = {}
    rep_dir = os.path.join(run_dir, "reports")
    if not os.path.isdir(rep_dir):
        return out
    for fn in sorted(os.listdir(rep_dir)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(rep_dir, fn)) as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row["run_id"])] = (float(row["task_progression"]), row["stage"])
                except (KeyError, ValueError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    vdir = os.path.join(args.run_dir, "videos")
    if not os.path.isdir(vdir):
        sys.exit(f"no videos/ under {args.run_dir}")
    outdir = os.path.join(args.run_dir, "videos_mp4")
    os.makedirs(outdir, exist_ok=True)
    report = load_report(args.run_dir)

    written = 0
    for pq in sorted(os.listdir(vdir)):
        if not pq.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(vdir, pq))
        for _, row in df.iterrows():
            blob = row["video"]
            if not isinstance(blob, (bytes, bytearray)):
                print(f"  {pq}: 'video' is {type(blob).__name__}, not bytes -- skipped")
                continue
            if bytes(blob[4:8]) != b"ftyp":
                print(f"  {pq}: not an MP4 container (magic {bytes(blob[:8])!r}) -- skipped")
                continue
            idx = int(row.get("repeat", written))
            tp, stage = report.get(idx, (None, None))
            tag = f"_TP{tp:.2f}_{stage}" if tp is not None else ""
            name = f"{args.prefix}{row['task']}_{row['perturbation']}_run{idx:03d}{tag}.mp4"
            with open(os.path.join(outdir, name), "wb") as f:
                f.write(blob)
            print(f"  wrote {name}  ({len(blob) / 1e6:.1f} MB)")
            written += 1
    print(f"\n{written} mp4(s) under {outdir}")


if __name__ == "__main__":
    main()
