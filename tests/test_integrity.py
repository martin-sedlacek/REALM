import os
import subprocess
import pandas as pd
import shutil
import sys
from pathlib import Path

# Add project root to path so we can import realm
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.eval import SUPPORTED_TASKS

def run_test():
    """
    Integrity test that runs a single step for every task to ensure 
    that data logging (reports, trajectories, videos) works correctly.
    """
    experiment_name = "integrity_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    
    # Logs will be placed in /app/logs by default in apptainer, 
    # but we use a tmp dir for the test
    base_log_dir = os.path.join(PROJECT_ROOT, "logs/integrity_test_tmp")
    
    # Clean up previous test runs
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Starting integrity test for tasks 0-9...")
    
    results = {}

    for task_id in range(10):
        task_name = SUPPORTED_TASKS[task_id]
        print(f"\n--- Testing Task {task_id}: {task_name} ---")
        
        # Run 02_evaluate.py for 1 step, 1 repeat
        # We run it from the PROJECT_ROOT
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
            "--no_render" 
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            print(f"Successfully ran evaluation for {task_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run evaluation for {task_name}")
            print(f"Error: {e.stderr}")
            results[task_name] = "EXECUTION_FAILED"
            continue

        # Paths to check
        task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
        
        checks = {
            "report_csv": os.path.join(task_log_dir, "reports", f"{task_name}_Default.csv"),
            "qpos_parquet": os.path.join(task_log_dir, "qpos", f"{task_name}.parquet"),
            "actions_parquet": os.path.join(task_log_dir, "actions", f"{task_name}.parquet"),
            "video_parquet": os.path.join(task_log_dir, "videos", f"{task_name}.parquet"),
        }

        task_results = {}
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
            
            task_results[key] = "PASS" if valid else ("FAIL_EMPTY" if exists else "FAIL_MISSING")
            print(f"  {key}: {task_results[key]} ({path})")
            
        results[task_name] = task_results

    # Summary
    print("\n" + "="*50)
    print("INTEGRITY TEST SUMMARY")
    print("="*50)
    all_pass = True
    for task, status in results.items():
        if status == "EXECUTION_FAILED":
            print(f"{task}: FAILED EXECUTION")
            all_pass = False
        else:
            task_pass = all(v == "PASS" for v in status.values())
            if not task_pass:
                all_pass = False
            status_str = ", ".join([f"{k}: {v}" for k, v in status.items()])
            print(f"{task}: {'PASS' if task_pass else 'FAIL'} ({status_str})")
    
    if all_pass:
        print("\nALL TASKS PASSED INTEGRITY CHECK!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_test()
