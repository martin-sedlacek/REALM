import os
import subprocess
import pandas as pd
import shutil
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.eval import SUPPORTED_TASKS


def run_test(task_id=0):
    """Run a single-task integrity test (1 step, 1 repeat)."""
    task_name = SUPPORTED_TASKS[task_id]
    experiment_name = "single_task_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    base_log_dir = os.path.join(PROJECT_ROOT, "logs/single_task_test_tmp")

    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Running single-task integrity test: task {task_id} ({task_name})")

    cmd = [
        "python", str(PROJECT_ROOT / "examples/02_evaluate.py"),
        "--task_id", str(task_id),
        "--perturbation_id", "0",
        "--repeats", "1",
        "--max_steps", "1",
        "--model_name", model_name,
        "--model_type", model_type,
        "--port", str(port),
        "--experiment_name", experiment_name,
        "--run_id", run_id,
        "--log_dir", base_log_dir,
        "--no_render",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        print(f"Evaluation succeeded for {task_name}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {task_name}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

    task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
    checks = {
        "report_csv": os.path.join(task_log_dir, "reports", f"{task_name}_Default.csv"),
        "qpos_parquet": os.path.join(task_log_dir, "qpos", f"{task_name}.parquet"),
        "actions_parquet": os.path.join(task_log_dir, "actions", f"{task_name}.parquet"),
        "video_parquet": os.path.join(task_log_dir, "videos", f"{task_name}.parquet"),
    }

    all_pass = True
    for key, path in checks.items():
        exists = os.path.exists(path)
        valid = False
        if exists:
            try:
                df = pd.read_csv(path) if key.endswith("_csv") else pd.read_parquet(path)
                if not df.empty:
                    valid = True
            except Exception as e:
                print(f"Error reading {key} at {path}: {e}")

        status = "PASS" if valid else ("FAIL_EMPTY" if exists else "FAIL_MISSING")
        print(f"  {key}: {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nTask {task_id} ({task_name}) PASSED!")
    else:
        print(f"\nTask {task_id} ({task_name}) FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-task integrity test")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID to test (0-9)")
    args = parser.parse_args()
    run_test(args.task_id)
