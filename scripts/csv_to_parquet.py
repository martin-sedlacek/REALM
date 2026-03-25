"""
Retrospectively convert all CSV report files in logs/ to parquet format,
following the logic in realm/logging.py (save_results with use_parquet=True).
"""
import os
import sys
import pandas as pd
from pathlib import Path


def convert_csv_to_parquet(logs_root: str, dry_run: bool = False, skip_existing: bool = True):
    logs_path = Path(logs_root)
    csv_files = list(logs_path.rglob("reports/*.csv"))

    print(f"Found {len(csv_files)} CSV files under {logs_root}")

    converted = 0
    skipped = 0
    errors = 0

    for csv_file in sorted(csv_files):
        parquet_file = csv_file.with_suffix(".parquet")

        if skip_existing and parquet_file.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY RUN] Would convert: {csv_file.relative_to(logs_path)}")
            converted += 1
            continue

        try:
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print(f"Converted: {csv_file.relative_to(logs_path)}")
            converted += 1
        except Exception as e:
            print(f"ERROR converting {csv_file.relative_to(logs_path)}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone. Converted: {converted}, Skipped (already exist): {skipped}, Errors: {errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV reports to parquet")
    parser.add_argument("--logs", default="/Users/sedlam/PycharmProjects/REALM/logs", help="Path to logs directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be converted without writing")
    parser.add_argument("--no-skip", action="store_true", help="Overwrite existing parquet files")
    args = parser.parse_args()

    convert_csv_to_parquet(args.logs, dry_run=args.dry_run, skip_existing=not args.no_skip)
