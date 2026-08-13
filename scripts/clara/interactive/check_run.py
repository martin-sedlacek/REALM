"""Apply the full REALM pass criteria to a run directory + its log.

Exit code 0 is NOT sufficient evidence that an OG-lite run worked: the failure mode asserts inside
an Isaac callback and then segfaults, which can still leave artifacts half-written. Pass requires
all of:

  1. no 'row mismatch' / Traceback / Segmentation fault / AssertionError in the log
  2. all four artifacts present -- reports/*.csv, actions/*.parquet, qpos/*.parquet, videos/*.parquet
  3. each artifact carries at least one populated data row

    python scripts/clara/interactive/check_run.py <run_dir> [log_file]

Runs on the host (pandas/pyarrow may be absent there), so parquet reads degrade to a size check
rather than failing the whole report.
"""
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
    """Row count, or None if it cannot be read here."""
    try:
        import pandas as pd
        return len(pd.read_parquet(path)) if path.endswith(".parquet") else len(pd.read_csv(path))
    except ImportError:
        return None
    except Exception as e:
        print(f"      read error: {type(e).__name__}: {e}")
        return -1


def check_artifacts(run_dir):
    print(f"\n=== artifacts: {run_dir} ===")
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
    return ok


if __name__ == "__main__":
    run_dir = sys.argv[1]
    log = sys.argv[2] if len(sys.argv) > 2 else None
    a = check_artifacts(run_dir)
    b = check_log(log) if log else True
    print(f"\n########## VERDICT: {'PASS' if (a and b) else 'FAIL'} ##########")
    print(f"  artifacts: {'pass' if a else 'FAIL'}")
    if log:
        print(f"  log:       {'pass' if b else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)
