"""Where a test's scratch artifacts go, and how a test decides a child eval really ran.

Shared by the script-style tests that drive an eval example in a subprocess (test_integrity,
test_perturbations_integrity, test_single_task, test_pi0_integration, test_vector_integrity) so
the rules below are stated once instead of five times, and cannot drift apart. run_eval_cell is
the whole single-env cell -- launch, log, crash-classify, artifact-check -- because the three
single-env drivers used to carry three copies of it and the copies had already drifted (their
progress-line phrasing disagreed, which run_suite.py's `cells` regexes then had to track
per-file).

Import-light on purpose: nothing here imports omnigibson (pandas is deferred into
check_artifacts), so the DRIVERS stay host-importable and boot no Isaac -- only their child
processes need the container. eval_const_list is what makes that possible for the drivers'
SUPPORTED_TASKS / SUPPORTED_PERTURBATIONS needs.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.paths import run_log_dir  # noqa: E402  (import-light: realm/__init__.py is empty)

#: The script every single-env cell drives. A module constant so a test harness can substitute a
#: stub (the real one cannot run outside the container), and so the choice is stated once.
EVALUATE_SCRIPT = PROJECT_ROOT / "examples" / "02_evaluate.py"


def eval_const_list(name):
    """Read a top-level list literal out of realm/eval.py WITHOUT importing it.

    `from realm.eval import SUPPORTED_TASKS` looks harmless but realm/eval.py imports omnigibson
    at module scope, which boots a full Isaac instance -- in the DRIVER, purely to read a list of
    strings, while every cell then boots another one in its child process. Two Isaac instances in
    one process tree is a needless risk on top of the wasted minute per invocation. Parsing the
    literal keeps the drivers dependency-free; it is a plain list of string constants, and
    ast.parse fails loudly if that ever stops being true.
    """
    import ast
    tree = ast.parse((PROJECT_ROOT / "realm" / "eval.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError(f"{name} not found as a top-level list in realm/eval.py")


def scratch_log_root(name):
    """Absolute path for a test's throwaway log tree.

    Order: an explicit REALM_TEST_LOG_DIR wins, then the container's bound log tree at /logs, then
    PROJECT_ROOT/logs for a plain host checkout. NOT `/app/logs`: in this checkout `logs` is a
    symlink whose target some container launchers do not mount, so `/app/logs` can dangle in the
    container and the first makedirs dies.

    TWO SUITE RUNS CANNOT SHARE THIS PATH: the name has no per-invocation discriminator, and the
    parquets are appended to, so a concurrent run makes whichever finishes second report
    FAIL_ROWS(2!=1) -- which reads exactly like a regression and is not one. Set
    REALM_TEST_LOG_DIR to a distinct path
    per concurrent invocation, or serialize; test_vector_integrity has no override and needs a
    distinct --experiment_name instead -- run_suite's four vector entries now each pass one
    (`suite_vector_*`), because they previously all defaulted to "vector_integrity" and the _tasks
    matrix and _drawers entry write the SAME t8/t9:Default cells, so _drawers reported FAIL_ROWS in
    every suite run where both ran. test_vector_integrity also clears each cell's own tree before
    writing it now, which is what stops rows surviving from one sweep into the next. Do not "fix" this by relaxing the exact-rows check -- it
    is what made the collision visible, and what stops a half-finished sweep reading as complete.
    """
    return os.path.join(scratch_log_base(), name)


def scratch_log_base():
    """The ROOT that scratch_log_root(name) puts `name` under, resolved the same way.

    test_vector_integrity needs the root itself rather than a named subtree, because it composes
    `<root>/<experiment_name>/debug/<run_id>` and `<root>/<experiment_name>/_runlogs`. It used to
    hardcode `/logs`, which scripts/run_apptainer.sh does NOT bind -- so every per-cell child log
    landed in the container's --writable-tmpfs overlay (64 MB, discarded on exit) and a crashed
    cell could not be diagnosed at all: the parent keeps only the first traceback line. Resolving
    it here means the artifacts and logs reach real disk under the repo.
    """
    override = os.environ.get("REALM_TEST_LOG_DIR")
    if override:
        return override
    if os.path.isdir("/logs"):
        return "/logs"
    return os.path.join(PROJECT_ROOT, "logs")


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

    ok = all(v == "PASS" for v in cell_results.values())
    return ok, ", ".join(f"{k}: {v}" for k, v in cell_results.items())


def run_eval_cell(label, base_log_dir, *, task_id, pert_id, task_name, pert_name,
                  repeats, max_steps, experiment_name, robot, no_render,
                  run_id="test_run", model_name="debug", model_type="debug", port=8000):
    """Run one (task, perturbation) cell through EVALUATE_SCRIPT and check what it produced.

    The shared body of test_integrity (sweeps tasks), test_perturbations_integrity (sweeps
    perturbations) and test_single_task (one cell): launch the child, save its combined output to
    ``<base_log_dir>/<label>.log``, decide crash-vs-ran from the output (NOT the exit status --
    Isaac tears down with a segfault after all work is done, on passing runs as much as failing
    ones, so gating on returncode marked every cell failed regardless of what it produced), then
    row-count the four artifacts with check_artifacts.

    Returns "EXECUTION_FAILED" or check_artifacts' {artifact: status} dict.

    The progress lines printed here are extracted by run_suite.py's `cells` regexes for its detail
    column ("CRASHED during evaluation for ...", "Ran evaluation for ...", "  <artifact>: <status>")
    -- change them together.
    """
    cmd = [
        sys.executable, "-u", str(EVALUATE_SCRIPT),
        "--task_id", str(task_id),
        "--perturbation_id", str(pert_id),
        "--repeats", str(repeats),
        "--max_steps", str(max_steps),
        "--model_name", model_name,
        "--model_type", model_type,
        "--port", str(port),
        "--experiment_name", experiment_name,
        "--run_id", run_id,
        "--log_dir", base_log_dir,
        "--robot", robot,
    ]
    if no_render:
        cmd.append("--no_render")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    child_log = os.path.join(base_log_dir, f"{label}.log")
    with open(child_log, "w") as fh:
        fh.write(proc.stdout or "")
        fh.write(proc.stderr or "")

    crashes = crash_lines((proc.stdout or "") + (proc.stderr or ""))
    if crashes:
        print(f"CRASHED during evaluation for {label} (exit={proc.returncode})")
        print(f"  first crash line: {crashes[0].strip()[:200]}")
        print(f"  full child log: {child_log}")
        return "EXECUTION_FAILED"
    print(f"Ran evaluation for {label} (exit={proc.returncode}, no crash signature)")

    task_log_dir = run_log_dir(base_log_dir, experiment_name, model_name, run_id)
    cell_results = check_artifacts(task_log_dir, task_name, pert_name, repeats)
    for key, status in cell_results.items():
        print(f"  {key}: {status}")
    if any(v != "PASS" for v in cell_results.values()):
        print(f"  full child log: {child_log}")
    return cell_results
