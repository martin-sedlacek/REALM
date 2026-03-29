import os
import subprocess
import pandas as pd
import shutil
import sys
import time
from pathlib import Path

# Add project root to path so we can import realm
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

def run_test():
    """
    Integration test for Pi0-FAST.
    1. Runs examples/01_pi0_eval.py (which uses task_id=1, perturbation_id=0, model_type='openpi', port=8000)
    2. Checks that reports/put_banana_into_box_Default.csv exists
    3. Checks that task_progression > 0
    """
    experiment_name = "pi0_integration_test"
    # Note: examples/01_pi0_eval.py doesn't take args, it uses these defaults:
    # task_id=1 -> put_banana_into_box
    # perturbation_id=0 -> Default
    # model_type="openpi"
    # port=8000
    
    # We'll use a specific log dir by setting an environment variable if evaluate supported it, 
    # but 01_pi0_eval.py uses the default /app/logs. 
    # In the bash script, we will bind a local directory to /app/logs.
    
    log_dir = "/app/logs"
    report_path = os.path.join(log_dir, "reports", "put_banana_into_box_Default.csv")
    
    print("Starting Pi0-FAST integration test...")
    
    # Run 01_pi0_eval.py
    cmd = ["python", "examples/01_pi0_eval.py"]
    
    try:
        # We assume the policy server is already running on port 8000 (started by the bash script)
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        print("Successfully ran 01_pi0_eval.py")
    except subprocess.CalledProcessError as e:
        print("Failed to run 01_pi0_eval.py")
        print(f"Error: {e.stderr}")
        sys.exit(1)

    # Verification
    if not os.path.exists(report_path):
        print(f"FAIL: Report not found at {report_path}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(report_path)
        if df.empty:
            print("FAIL: Report is empty")
            sys.exit(1)

        # Check task progression
        progression = df['task_progression'].iloc[-1]
        print(f"Task progression: {progression}")
        
        if progression <= 0:
            print("FAIL: Task progression is 0 or less")
            sys.exit(1)
            
        print("PASS: Pi0-FAST integration test successful!")
        
    except Exception as e:
        print(f"FAIL: Error reading or validating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
