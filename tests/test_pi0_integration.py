"""End-to-end check that a REAL policy drives the task somewhere: Pi0-FAST on put_banana_into_box.

This is the ONLY test in tests/ that runs a real policy. Everything else uses
`--model_type debug`, which returns a hardcoded zero action with the gripper open, so this is the
only place the suite can observe task_progression > 0 -- i.e. the only place any of the grasping,
placement or success machinery is exercised at all.

    # in one shell: bring up the openpi policy server on :8000
    # in another:
    ./scripts/run_apptainer.sh python -u tests/test_pi0_integration.py

PRECONDITIONS, both of which this test now checks up front instead of discovering 15 minutes in:

  1. A live openpi websocket policy server on 127.0.0.1:8000. There is no such server in this
     repo and no fixture that starts one; it is brought up by hand from the model checkout. With
     no server the run used to boot a full Isaac instance and 500 steps before failing on the
     connection, which is why the socket is probed first.

  2. examples/01_pi0_eval.py takes NO arguments and writes to its `evaluate()` default of
     "/app/logs". The container must bind that path to writable storage. Nothing this test can
     do fixes a missing bind from the outside -- 01_pi0_eval.py accepts no --log_dir -- so the condition is
     reported, loudly, rather than worked around. `/app/logs` being resolvable is checked first.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from tests._paths import crash_lines

HOST, PORT = "127.0.0.1", 8000
# 01_pi0_eval.py's hardcoded destination -- evaluate()'s default log_dir. Not configurable.
PI0_LOG_DIR = "/app/logs"


def preflight():
    """Report every reason this test cannot run, before spending an Isaac boot to find out."""
    blockers = []

    if not os.path.isdir(PI0_LOG_DIR):
        target = os.path.realpath(PI0_LOG_DIR)
        blockers.append(
            f"{PI0_LOG_DIR} is not a directory (islink={os.path.islink(PI0_LOG_DIR)}, "
            f"resolves to {target!r}). examples/01_pi0_eval.py writes there and takes no "
            f"--log_dir, so it will die in os.makedirs before loading the task.")

    try:
        with socket.create_connection((HOST, PORT), timeout=5):
            pass
    except OSError as exc:
        blockers.append(f"no policy server accepting connections on {HOST}:{PORT} ({exc}). "
                        f"Start the openpi server for the Pi0-FAST checkpoint first.")
    return blockers


def run_test():
    print("Starting Pi0-FAST integration test...")

    blockers = preflight()
    if blockers:
        # NOT a pass and NOT a failure of the code under test. A distinct verdict, so a suite
        # summary cannot quietly count it as either.
        print("SKIP: preconditions not met, the policy was never run:")
        for b in blockers:
            print(f"  - {b}")
        sys.exit(2)

    report_path = os.path.join(PI0_LOG_DIR, "reports", "put_banana_into_box_Default.csv")

    # NOT check=True: Isaac segfaults at teardown on passing runs too. See tests/test_integrity.py.
    proc = subprocess.run([sys.executable, "-u", "examples/01_pi0_eval.py"],
                          capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    combined = (proc.stdout or "") + (proc.stderr or "")
    crashes = crash_lines(combined)
    if crashes:
        print(f"Failed to run 01_pi0_eval.py (exit={proc.returncode})")
        print(f"  first crash line: {crashes[0].strip()[:200]}")
        sys.exit(1)
    print(f"Ran 01_pi0_eval.py (exit={proc.returncode}, no crash signature)")

    if not os.path.exists(report_path):
        print(f"FAIL: Report not found at {report_path}")
        sys.exit(1)

    try:
        import pandas as pd
        df = pd.read_csv(report_path)
        if df.empty:
            print("FAIL: Report is empty")
            sys.exit(1)

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
