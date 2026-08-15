"""One task's slice of tests/test_integrity.py, for when you only care about one task.

    ./scripts/clara/interactive/rr python -u tests/test_single_task.py --task_id 4

Same coverage limits as tests/test_integrity.py -- read that module's docstring. In short:
`--model_type debug` commands a hold-still action with the gripper open, `--max_steps 1` skips
every metric formula, and `--no_render` drops the EXTERNAL sensors only, so the recorded frame is
extract_from_obs's synthetic black fallback for base_im beside a real wrist view. This checks that
the plumbing runs, nothing else.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from tests._paths import check_artifacts, crash_lines, scratch_log_root, summarize

from realm.eval import SUPPORTED_TASKS

REPEATS = 1
MAX_STEPS = 1


def run_test(task_id=0):
    task_name = SUPPORTED_TASKS[task_id]
    experiment_name = "single_task_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    base_log_dir = scratch_log_root("single_task_test_tmp")

    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Running single-task integrity test: task {task_id} ({task_name})")
    print(f"log root: {base_log_dir}", flush=True)

    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "examples/02_evaluate.py"),
        "--task_id", str(task_id),
        "--perturbation_id", "0",
        "--repeats", str(REPEATS),
        "--max_steps", str(MAX_STEPS),
        "--model_name", model_name,
        "--model_type", model_type,
        "--port", str(port),
        "--experiment_name", experiment_name,
        "--run_id", run_id,
        "--log_dir", base_log_dir,
        "--no_render",
    ]

    # NOT check=True -- Isaac segfaults at teardown on passing runs too, so the exit status carries
    # no information about the test. See tests/test_integrity.py.
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    child_log = os.path.join(base_log_dir, f"{task_name}.log")
    with open(child_log, "w") as fh:
        fh.write(proc.stdout or "")
        fh.write(proc.stderr or "")

    crashes = crash_lines((proc.stdout or "") + (proc.stderr or ""))
    if crashes:
        print(f"Evaluation failed for {task_name} (exit={proc.returncode})")
        print(f"  first crash line: {crashes[0].strip()[:200]}")
        print(f"  full child log: {child_log}")
        print(f"\nTask {task_id} ({task_name}) FAILED!")
        sys.exit(1)
    print(f"Evaluation ran for {task_name} (exit={proc.returncode}, no crash signature)")

    task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
    checks = check_artifacts(task_log_dir, task_name, "Default", REPEATS)
    for key, status in checks.items():
        print(f"  {key}: {status}")

    all_pass, _ = summarize(checks)
    if all_pass:
        print(f"\nTask {task_id} ({task_name}) PASSED!")
    else:
        print(f"  full child log: {child_log}")
        print(f"\nTask {task_id} ({task_name}) FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-task integrity test")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID to test (0-9)")
    args = parser.parse_args()
    run_test(args.task_id)
