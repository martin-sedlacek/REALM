"""Classify ALREADY-COMPLETED grasp-gate cells using test_vector_integrity.py's OWN criteria.

Why this exists: `logs/ship_gate_<robot>/` directories were described as holding "partial artifacts"
from earlier attempts. Whether a cell passed is decided by three things -- a log free of crash
markers (minus Isaac's unconditional teardown noise), all four artifacts present and non-empty, and
a report with exactly `--repeats` rows -- and none of those needs a GPU to check. So a directory
that is already on disk can be classified now instead of being re-run on faith.

The regexes and the artifact layout are IMPORTED from the test module, never re-typed here: a
re-implementation that drifted from the real one would produce a verdict that looks authoritative
and is not. Importing tests/test_vector_integrity.py is safe -- it reads realm/eval.py with
ast.parse specifically to avoid booting Isaac just to read two lists of strings.

    python scripts/debug_probes/val_gate_classify.py --log-dir /logs \
        --experiment ship_gate_DROID_robolab_v2 --repeats 2
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import test_vector_integrity as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", default="/logs")
ap.add_argument("--experiment", action="append", required=True,
                help="experiment_name, i.e. the ship_gate_<robot> directory. Repeat.")
ap.add_argument("--repeats", type=int, default=2)
args = ap.parse_args()

RC = 0
for exp in args.experiment:
    root = Path(args.log_dir) / exp
    print(f"\n{'=' * 100}\n{exp}\n{'=' * 100}")
    if not root.is_dir():
        print("  *** directory does not exist ***")
        RC = 1
        continue
    runlogs = sorted((root / "_runlogs").glob("*.log"))
    if not runlogs:
        print("  *** no _runlogs/*.log -- no cell was ever launched here ***")
        RC = 1
        continue
    for log_path in runlogs:
        run_id = log_path.stem                       # e.g. t0_Default
        d = root / "debug" / run_id
        text = log_path.read_text(errors="replace")
        crash_lines = [ln for ln in text.splitlines()
                       if T.CRASH_MARKERS.search(ln) and not T.TEARDOWN_NOISE.search(ln)]
        reports = sorted((d / "reports").glob("*.csv")) if (d / "reports").is_dir() else []
        report = reports[0] if reports else None
        task = report.name.rsplit("_", 1)[0] if report else None
        artifacts = {"report": report} if report else {"report": d / "reports" / "MISSING.csv"}
        for kind in ("qpos", "actions", "videos"):
            artifacts[kind] = (d / kind / f"{task}.parquet") if task else (d / kind / "MISSING")
        missing = [k for k, p in artifacts.items()
                   if p is None or not p.exists() or p.stat().st_size == 0]
        n_rows = None
        if report and report.exists():
            with open(report) as fh:
                n_rows = sum(1 for _ in csv.DictReader(fh))

        if crash_lines:
            status, detail = "CRASH", crash_lines[0].strip()[:150]
        elif missing:
            status, detail = "NO_ARTIFACTS", f"missing={','.join(missing)}"
        elif n_rows != args.repeats:
            status, detail = "PARTIAL", f"report has {n_rows} rows, expected {args.repeats}"
        else:
            status, detail = "PASS", f"{n_rows} rollouts, all artifacts present"
        if status != "PASS":
            RC = 1
        mt = os.path.getmtime(log_path)
        import datetime
        print(f"  {run_id:16s} {status:14s} {detail}")
        print(f"  {'':16s} log {log_path.name} "
              f"({datetime.datetime.fromtimestamp(mt).isoformat(timespec='seconds')}, "
              f"{len(text.splitlines())} lines), task={task}")
        # The columns a changed pad response could plausibly move. Printed for every cell so the
        # gate is not just "the process survived" -- but see the caveat below.
        if report and report.exists():
            with open(report) as fh:
                for row in csv.DictReader(fh):
                    print(f"  {'':16s}   run {row['run_id']}: stage={row['stage']:10s} "
                          f"progression={row['task_progression']:6s} SR={row['binary_SR']:5s} "
                          f"self_col={row['collisions_self']} env_col={row['collisions_env']} "
                          f"drops={row['object_drops']}")

print(f"\n{'=' * 100}")
print("CAVEAT, and it matters for what this gate can and cannot show: these cells run "
      "--model_type debug,\nwhich is a canned motion, not a policy. Every row below lands "
      "stage=REACH / progression=0.0 / SR=0.0 on\nEVERY build, so the gate answers 'does the "
      "vectorized eval path still run end to end with this robot',\nNOT 'does this build grasp "
      "better'. is_grasping is only exercised if the rollout reaches an object.")
print(f"{'=' * 100}")
print("VAL_GATE_CLASSIFY_COMPLETE rc=%d" % RC)
sys.exit(0)
