import os
import subprocess
import pandas as pd
import shutil
from realm.eval import SUPPORTED_TASKS

def run_test():
    experiment_name = "integrity_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    base_log_dir = "/app/logs/integrity_test_tmp"
    
    # Clean up previous test runs
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Starting integrity test for {len(SUPPORTED_TASKS)} tasks...")
    
    results = {}

    for task_id, task_name in enumerate(SUPPORTED_TASKS):
        print(f"\n--- Testing Task {task_id}: {task_name} ---")
        
        # Run 02_evaluate.py for 1 step, 1 repeat
        cmd = [
            "python", "examples/02_evaluate.py",
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
        
        # ... (execution code)
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully ran evaluation for {task_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run evaluation for {task_name}")
            print(f"Error: {e.stderr}")
            results[task_name] = "EXECUTION_FAILED"
            continue

        # Paths to check
        # log_dir = base_log_dir / experiment_name / model_name / run_id
        task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
        
        checks = {
            "report_parquet": os.path.join(task_log_dir, "reports", f"{task_name}_Default.parquet"),
            "qpos_parquet": os.path.join(task_log_dir, "qpos", "data.parquet"),
            "actions_parquet": os.path.join(task_log_dir, "actions", "data.parquet"),
            "video_parquet": os.path.join(task_log_dir, "videos", "data.parquet"),
        }
        
        task_results = {}
        for key, path in checks.items():
            exists = os.path.exists(path)
            valid = False
            if exists:
                try:
                    df = pd.read_parquet(path)
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
        print("\nSOME TASKS FAILED INTEGRITY CHECK.")

if __name__ == "__main__":
    run_test()
