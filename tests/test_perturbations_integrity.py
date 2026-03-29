import os
import subprocess
import pandas as pd
import shutil
import sys
from pathlib import Path

# Add project root to path so we can import realm
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS

def run_test():
    """
    Integrity test that runs a single step for all perturbations (on Task 0)
    to ensure that data logging works correctly.
    """
    experiment_name = "pert_integrity_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    task_id = 0
    task_name = SUPPORTED_TASKS[task_id]
    
    base_log_dir = os.path.join(PROJECT_ROOT, "logs/pert_integrity_test_tmp")
    
    # Clean up previous test runs
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Starting perturbation integrity test for Task {task_id} ({task_name})...")
    print(f"Testing {len(SUPPORTED_PERTURBATIONS)} perturbations...")
    
    results = {}

    for pert_id, pert_name in enumerate(SUPPORTED_PERTURBATIONS):
        print(f"\n--- Testing Perturbation {pert_id}: {pert_name} ---")
        
        # Run 02_evaluate.py for 1 step, 1 repeat
        cmd = [
            "python", str(PROJECT_ROOT / "examples/02_evaluate.py"),
            "--task_id", str(task_id),
            "--perturbation_id", str(pert_id),
            "--repeats", "1",
            "--max_steps", "1",
            "--model_name", model_name,
            "--model_type", model_type,
            "--port", str(port),
            "--experiment_name", experiment_name,
            "--run_id", run_id,
            "--log_dir", base_log_dir
        ] # "--no_render"
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            print(f"Successfully ran evaluation for {pert_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run evaluation for {pert_name}")
            print(f"Error: {e.stderr}")
            results[pert_name] = "EXECUTION_FAILED"
            continue

        # Paths to check
        task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
        
        checks = {
            "report_csv": os.path.join(task_log_dir, "reports", f"{task_name}_{pert_name}.csv"),
            "qpos_parquet": os.path.join(task_log_dir, "qpos", f"{task_name}.parquet"),
            "actions_parquet": os.path.join(task_log_dir, "actions", f"{task_name}.parquet"),
            "video_parquet": os.path.join(task_log_dir, "videos", f"{task_name}.parquet"),
        }

        pert_results = {}
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
            
            pert_results[key] = "PASS" if valid else ("FAIL_EMPTY" if exists else "FAIL_MISSING")
            print(f"  {key}: {pert_results[key]} ({path})")
            
        results[pert_name] = pert_results

    # Summary
    print("\n" + "="*50)
    print("PERTURBATION INTEGRITY TEST SUMMARY")
    print("="*50)
    all_pass = True
    for pert, status in results.items():
        if status == "EXECUTION_FAILED":
            print(f"{pert}: FAILED EXECUTION")
            all_pass = False
        else:
            pert_pass = all(v == "PASS" for v in status.values())
            if not pert_pass:
                all_pass = False
            status_str = ", ".join([f"{k}: {v}" for k, v in status.items()])
            print(f"{pert}: {'PASS' if pert_pass else 'FAIL'} ({status_str})")
    
    if all_pass:
        print("\nALL PERTURBATIONS PASSED INTEGRITY CHECK!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_test()
