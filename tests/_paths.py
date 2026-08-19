"""Where a test's scratch artifacts go, and how a test decides a child eval really ran.

Shared by the four script-style tests that drive examples/02_evaluate.py in a subprocess
(test_integrity, test_perturbations_integrity, test_single_task, test_pi0_integration) so the two
rules below are stated once instead of four times, and cannot drift apart.
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def scratch_log_root(name):
    """Absolute path for a test's throwaway log tree.

    Order: an explicit REALM_TEST_LOG_DIR wins, then the container's bound log tree at /logs, then
    PROJECT_ROOT/logs for a plain host checkout. NOT `/app/logs`: in this checkout `logs` is a
    symlink whose target the clara `rr` binds do not mount, so `/app/logs` dangles in the
    container and the first makedirs dies (measured; see docs/code_archaeology.md).

    TWO SUITE RUNS CANNOT SHARE THIS PATH: the name has no per-invocation discriminator, and the
    parquets are appended to, so a concurrent run makes whichever finishes second report
    FAIL_ROWS(2!=1) -- which reads exactly like a regression and is not one (measured on jobs
    191494/191495; details in docs/code_archaeology.md). Set REALM_TEST_LOG_DIR to a distinct path
    per concurrent invocation, or serialize; test_vector_integrity has no override and needs a
    distinct --experiment_name instead. Do not "fix" this by relaxing the exact-rows check -- it
    is what made the collision visible, and what stops a half-finished sweep reading as complete.
    """
    override = os.environ.get("REALM_TEST_LOG_DIR")
    if override:
        return os.path.join(override, name)
    if os.path.isdir("/logs"):
        return os.path.join("/logs", name)
    return os.path.join(PROJECT_ROOT, "logs", name)


# Signatures that mean a child eval died even if its exit status says otherwise. Same list as
# tests/test_vector_integrity.py's CRASH_MARKERS -- kept identical on purpose.
CRASH_MARKERS = re.compile(
    r"Traceback \(most recent call last\)|AssertionError|AttributeError|KeyError|TypeError|"
    r"IndexError|RuntimeError|Segmentation fault|CUDA error|out of memory", re.I)
# Isaac tears down with a segfault after all work is done, on passing runs as well as failing ones.
# Counting that as a crash would fail every cell.
TEARDOWN_NOISE = re.compile(r"Fatal Python error: Segmentation fault|"
                            r"srun: error:.*Segmentation fault|core dumped")


def crash_lines(output):
    """Lines in a child's combined output that indicate a real failure, teardown noise removed."""
    return [ln for ln in (output or "").splitlines()
            if CRASH_MARKERS.search(ln) and not TEARDOWN_NOISE.search(ln)]


def check_artifacts(task_log_dir, task, perturbation, repeats):
    """Status of the four artifacts one (task, perturbation) cell must produce.

    ROW COUNTS, not just "the file exists and is non-empty". Two reasons, both of which make the
    weaker check unable to fail:

      1. The parquets are APPENDED to. realm_logging.append_trajectory/append_video write ONE file
         per task -- qpos/<task>.parquet -- with a row per (perturbation, repeat). So in a sweep
         over perturbations against a single task, "qpos/<task>.parquet is non-empty" is satisfied
         for perturbation 15 by whatever perturbation 0 wrote. Once any one cell succeeds, that
         check can never fail again, for any later cell, no matter what it does. A sweep can report
         16/16 green with 15 cells having written nothing.
      2. realm_logging.save_results rewrites the report after EVERY repeat, so a run that dies half
         way leaves a complete-LOOKING prefix. Row count is what separates it from a finished run.

    So: filter each parquet to THIS cell's perturbation, and require exactly @repeats rows in both
    the parquets and the report.

    Returns {artifact_name: status}, where status is "PASS" or a FAIL_* string naming the reason.
    """
    import pandas as pd

    paths = {
        "report_csv": (os.path.join(task_log_dir, "reports", f"{task}_{perturbation}.csv"), "csv"),
        "qpos_parquet": (os.path.join(task_log_dir, "qpos", f"{task}.parquet"), "parquet"),
        "actions_parquet": (os.path.join(task_log_dir, "actions", f"{task}.parquet"), "parquet"),
        "video_parquet": (os.path.join(task_log_dir, "videos", f"{task}.parquet"), "parquet"),
    }

    out = {}
    for key, (path, kind) in paths.items():
        if not os.path.exists(path):
            out[key] = "FAIL_MISSING"
            continue
        try:
            df = pd.read_csv(path) if kind == "csv" else pd.read_parquet(path)
        except Exception as exc:
            out[key] = f"FAIL_UNREADABLE({type(exc).__name__})"
            continue
        if df.empty:
            out[key] = "FAIL_EMPTY"
            continue
        # The report is already per-perturbation (its filename carries it); the parquets are not.
        if kind == "csv":
            rows = len(df)
        elif "perturbation" not in df.columns:
            # realm_logging changed its parquet schema. Say so; do not fall back to a row count
            # that would silently start counting other perturbations' rows as this one's.
            out[key] = f"FAIL_NO_PERTURBATION_COLUMN({list(df.columns)})"
            continue
        else:
            rows = int((df["perturbation"] == perturbation).sum())
        out[key] = "PASS" if rows == repeats else f"FAIL_ROWS({rows}!={repeats})"
    return out


def summarize(cell_results):
    """(all_passed, one-line detail) for a dict of {artifact: status}."""
    ok = all(v == "PASS" for v in cell_results.values())
    return ok, ", ".join(f"{k}: {v}" for k, v in cell_results.items())
