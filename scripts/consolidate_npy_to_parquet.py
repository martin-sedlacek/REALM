"""
Retroactively consolidate per-repeat .npy files in actions/ and qpos/ directories
into a single data.parquet per run directory.

Filename pattern: {task}_{perturbation}_{repeat}.npy
  e.g. put_banana_into_box_V-SC_22.npy  →  task=put_banana_into_box, perturbation=V-SC, repeat=22

Perturbation names never contain underscores (they use hyphens), so rsplit('_', 2) is safe.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def parse_npy_filename(stem: str):
    """Parse '{task}_{perturbation}_{repeat}' into (task, perturbation, repeat)."""
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    task, perturbation, repeat_str = parts
    try:
        repeat = int(repeat_str)
    except ValueError:
        return None
    return task, perturbation, repeat


def consolidate_dir(npy_dir: Path, dry_run: bool = False):
    npy_files = sorted(npy_dir.glob("*.npy"))
    if not npy_files:
        return 0, 0  # converted, errors

    rows = []
    errors = []
    for npy_file in npy_files:
        parsed = parse_npy_filename(npy_file.stem)
        if parsed is None:
            errors.append(f"Could not parse: {npy_file.name}")
            continue
        task, perturbation, repeat = parsed
        try:
            arr = np.load(npy_file)
            rows.append({"task": task, "perturbation": perturbation, "repeat": repeat, "data": arr.tolist()})
        except Exception as e:
            errors.append(f"Failed to load {npy_file.name}: {e}")

    if not rows:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 0, len(errors)

    parquet_path = npy_dir / "data.parquet"
    if dry_run:
        print(f"  [DRY RUN] Would write {len(rows)} rows → {parquet_path.relative_to(npy_dir.parent.parent.parent.parent.parent)}")
        return len(rows), len(errors)

    df = pd.DataFrame(rows).sort_values(["task", "perturbation", "repeat"]).reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)

    # Delete the original npy files after successful write
    for npy_file in npy_files:
        if parse_npy_filename(npy_file.stem) is not None:
            npy_file.unlink()

    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)

    return len(rows), len(errors)


def consolidate_logs(logs_root: str, dry_run: bool = False):
    logs_path = Path(logs_root)
    total_converted = 0
    total_errors = 0
    dirs_processed = 0

    for subdir_name in ["actions", "qpos"]:
        npy_dirs = [p.parent for p in logs_path.rglob(f"{subdir_name}/*.npy")]
        npy_dirs = sorted(set(npy_dirs))

        for npy_dir in npy_dirs:
            rel = npy_dir.relative_to(logs_path)
            npy_count = len(list(npy_dir.glob("*.npy")))
            print(f"Processing {rel}/ ({npy_count} npy files)")
            converted, errors = consolidate_dir(npy_dir, dry_run=dry_run)
            total_converted += converted
            total_errors += errors
            dirs_processed += 1

    print(f"\nDone. Directories processed: {dirs_processed}, Rows written: {total_converted}, Errors: {total_errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate npy files into parquets")
    parser.add_argument("--logs", default="/Users/sedlam/PycharmProjects/REALM/logs", help="Path to logs directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing")
    args = parser.parse_args()

    consolidate_logs(args.logs, dry_run=args.dry_run)
