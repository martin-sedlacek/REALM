"""Apply the full REALM pass criteria to a run directory + its log.

Exit code 0 is NOT sufficient evidence that an OG-lite run worked. Two independent reasons:

  * the OG-lite failure mode asserts inside an Isaac callback and then segfaults, which can still
    leave artifacts half-written;
  * Isaac's SimulationApp.close() HARD-EXITS the process with status 0, so an unhandled Python
    exception -- a bad --robot, a missing asset, anything that raises before the rollout loop --
    still leaves `$?` at 0. Job 190683 (2026-08-13) died on `AssertionError: droid_robolab_v2 is
    not a registered robot` after ~6 minutes, wrote zero results, reported exit 0, and Slurm logged
    it COMPLETED.

Pass requires all of:

  1. no 'row mismatch' / Traceback / Segmentation fault / AssertionError in the log
  2. all four artifacts present -- reports/*.csv, actions/*.parquet, qpos/*.parquet, videos/*.parquet
  3. each artifact carries at least one populated data row
  4. (--repeats N) the results CSV carries exactly N rollout rows, so a run that died half way
     through is not mistaken for a complete one
  5. (--newer-than EPOCH) the artifacts were written by THIS run, not left behind by an earlier one
     that happened to use the same RUN_ID

    python scripts/clara/interactive/check_run.py <run_dir> [log_file] [--repeats N] [--newer-than EPOCH]

Runs on the host, where pandas/pyarrow may be absent, so parquet reads degrade to a size check.
The reports CSV is counted with the stdlib `csv` module instead, because criterion 4 is the whole
point of the check and must not silently degrade to "the file is non-empty".
"""
import argparse
import csv
import datetime
import os
import re
import sys

FAIL_PATTERNS = [
    "row mismatch",
    "Traceback (most recent call last)",
    "Segmentation fault",
    "AssertionError",
    "core dumped",
    "CUDA out of memory",
]

# Slack on --newer-than. Artifact mtimes come from whichever compute node wrote them, and the
# timestamp we compare against is taken on the submitting node, so a couple of seconds of clock
# skew is possible. Evals run for tens of minutes, so two minutes of grace cannot hide a stale
# artifact from a previous job while comfortably absorbing any realistic skew.
MTIME_GRACE_S = 120


def check_log(path):
    print(f"\n=== log scan: {path} ===")
    if not os.path.exists(path):
        print("  MISSING LOG -- cannot verify")
        return False
    text = open(path, errors="replace").read()
    ok = True
    for pat in FAIL_PATTERNS:
        n = len(re.findall(re.escape(pat), text, flags=re.IGNORECASE))
        status = "ok" if n == 0 else "FAIL"
        print(f"  [{status}] {pat!r}: {n}")
        if n:
            ok = False
            for m in re.finditer(re.escape(pat), text, flags=re.IGNORECASE):
                start = max(0, m.start() - 200)
                print(f"        ...{text[start:m.end() + 400]}...")
                break
    m = re.search(r"### EXIT_CODE=(\d+)", text)
    if m:
        code = int(m.group(1))
        print(f"  [{'ok' if code == 0 else 'FAIL'}] EXIT_CODE={code}")
        ok = ok and code == 0
    else:
        print("  [warn] no EXIT_CODE marker found (run may still be in flight)")
        ok = False
    return ok


def rows(path):
    """Number of data rows, or None if it cannot be counted on this host."""
    if path.endswith(".csv"):
        # Deliberately stdlib rather than pandas: the host has no guaranteed pandas, and the
        # rollout-count criterion is gated on this number. Degrading it to "not None bytes" is
        # exactly the silent pass this script exists to prevent.
        with open(path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # header
            except StopIteration:
                return 0
            return sum(1 for r in reader if any(field.strip() for field in r))
    try:
        import pandas as pd
        return len(pd.read_parquet(path))
    except ImportError:
        return None
    except Exception as e:
        print(f"      read error: {type(e).__name__}: {e}")
        return -1


def _stamp(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def check_artifacts(run_dir, expected_rows=None, newer_than=None):
    print(f"\n=== artifacts: {run_dir} ===")
    if expected_rows is not None:
        print(f"  expecting {expected_rows} rollout row(s) in reports/*.csv")
    if newer_than is not None:
        print(f"  expecting mtime >= {_stamp(newer_than)} (grace {MTIME_GRACE_S}s)")
    if not os.path.isdir(run_dir):
        print("  MISSING RUN DIR")
        return False
    ok = True
    for sub, ext in (("reports", ".csv"), ("actions", ".parquet"),
                     ("qpos", ".parquet"), ("videos", ".parquet")):
        d = os.path.join(run_dir, sub)
        files = sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []
        files = [f for f in files if f.endswith(ext)]
        if not files:
            print(f"  [FAIL] {sub}/: no {ext} file")
            ok = False
            continue
        for f in files:
            p = os.path.join(d, f)
            size = os.path.getsize(p)
            n = rows(p)
            if n is None:
                verdict = "ok?" if size > 0 else "FAIL"
                detail = f"{size} bytes (no pandas here, size-only check)"
            else:
                verdict = "ok" if n > 0 else "FAIL"
                detail = f"{n} rows, {size} bytes"
            print(f"  [{verdict}] {sub}/{f}: {detail}")
            if verdict == "FAIL":
                ok = False

            # -- criterion 5: written by THIS run ------------------------------------------------
            if newer_than is not None:
                mtime = os.path.getmtime(p)
                if mtime < newer_than - MTIME_GRACE_S:
                    print(f"  [FAIL] {sub}/{f}: STALE -- last written {_stamp(mtime)}, before this "
                          f"run started at {_stamp(newer_than)}. These are a previous run's "
                          f"artifacts, not this one's.")
                    ok = False

            # -- criterion 4: the full rollout count ---------------------------------------------
            if expected_rows is not None and sub == "reports" and n is not None and n >= 0:
                if n < expected_rows:
                    print(f"  [FAIL] {sub}/{f}: TRUNCATED -- {n} rollout rows, expected "
                          f"{expected_rows}. The run died part way through; the numbers in this "
                          f"file are a prefix of the requested eval, not the eval.")
                    ok = False
                elif n > expected_rows:
                    print(f"  [FAIL] {sub}/{f}: {n} rollout rows, expected {expected_rows}. More "
                          f"rows than were requested means this CSV is not this run's -- a resumed "
                          f"or reused RUN_ID appended to someone else's results.")
                    ok = False
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the $EXPERIMENT/$MODEL_NAME/$RUN_ID results directory")
    p.add_argument("log", nargs="?", default=None, help="the run's log, scanned for crash markers")
    p.add_argument("--repeats", type=int, default=None,
                   help="required number of rollout rows in reports/*.csv (i.e. REPEATS)")
    p.add_argument("--newer-than", type=float, default=None, metavar="EPOCH",
                   help="unix time the run started; artifacts older than this are a previous run's")
    a = p.parse_args()

    artifacts_ok = check_artifacts(a.run_dir, expected_rows=a.repeats, newer_than=a.newer_than)
    log_ok = check_log(a.log) if a.log else True
    print(f"\n########## VERDICT: {'PASS' if (artifacts_ok and log_ok) else 'FAIL'} ##########")
    print(f"  artifacts: {'pass' if artifacts_ok else 'FAIL'}")
    if a.log:
        print(f"  log:       {'pass' if log_ok else 'FAIL'}")
    sys.exit(0 if (artifacts_ok and log_ok) else 1)
