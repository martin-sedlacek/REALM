#!/usr/bin/env python
"""Extract viewable PNG/MP4s from REALM's video parquets, one per (task, perturbation, repeat).

REALM stores rollout videos as mp4 bytes inside a parquet, one row per rollout, appended across
runs -- so they cannot be opened directly and the newest rollout is the *last* row, not the first.
This unpacks them for eyeballing, which is the quickest way to check that each perturbation is
actually doing what it claims (lighting changed, distractors present, object swapped, ...).

    python scripts/extract_perturbation_frames.py logs/pert_integrity_test_tmp
    python scripts/extract_perturbation_frames.py logs/pert_integrity_test_tmp --video
"""

import argparse
import io
import os
import sys

import av
import pandas as pd
from PIL import Image


def extract(parquet_path, out_dir, want_video=False):
    df = pd.read_parquet(parquet_path)
    written = []
    for i, row in df.iterrows():
        stem = f"{row['task']}_{row['perturbation']}_rep{row['repeat']}_row{i}"
        container = av.open(io.BytesIO(row["video"]))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if not frames:
            print(f"  !! {stem}: no decodable frames")
            continue
        # Last frame: with max_steps=1 the rollout is ~1 frame anyway, and for longer rollouts the
        # final frame shows the outcome. Perturbations are visible from the first frame too.
        png = os.path.join(out_dir, f"{stem}.png")
        Image.fromarray(frames[-1]).save(png)
        written.append(png)
        if want_video:
            mp4 = os.path.join(out_dir, f"{stem}.mp4")
            with open(mp4, "wb") as f:
                f.write(row["video"])
            written.append(mp4)
        print(f"  {stem}: {len(frames)} frame(s) {frames[0].shape[1]}x{frames[0].shape[0]} -> {os.path.basename(png)}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", help="directory to search for videos/*.parquet")
    parser.add_argument("--out", default=None, help="output directory (default: <log_dir>/frames)")
    parser.add_argument("--video", action="store_true", help="also write the mp4 alongside each PNG")
    args = parser.parse_args()

    parquets = []
    for dirpath, _, files in os.walk(args.log_dir):
        if os.path.basename(dirpath) != "videos":
            continue
        parquets.extend(os.path.join(dirpath, f) for f in sorted(files) if f.endswith(".parquet"))

    if not parquets:
        sys.exit(f"error: no videos/*.parquet found under {args.log_dir}")

    out_dir = args.out or os.path.join(args.log_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)

    total = []
    for p in parquets:
        print(f"\n{p}")
        total += extract(p, out_dir, want_video=args.video)

    print(f"\nwrote {len(total)} file(s) to {out_dir}")


if __name__ == "__main__":
    main()
