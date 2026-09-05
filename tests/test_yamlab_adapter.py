"""End-to-end: `--model_type yamlab --robot YAM_bimanual` against the reference sweep server. Container + GPU.

    ./scripts/run_apptainer.sh python -u tests/test_yamlab_adapter.py

Starts tests/yamlab_sweep_server.py (which validates every observation against the YAMLab LeRobot
contract and answers with a joint sweep), runs one 90-step rollout through examples/02_evaluate.py,
and checks the four artifacts plus that the recorded qpos actually followed the sweep: both arms'
joints move, the left and right arms move in opposite directions, and both grippers visit the open
AND the closed state. The debug policy cannot show any of that -- it holds the zero pose.
"""
import glob
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from tests._paths import eval_const_list, run_eval_cell, scratch_log_root, summarize  # noqa: E402
from realm.paths import run_log_dir  # noqa: E402

SUPPORTED_TASKS = eval_const_list("SUPPORTED_TASKS")
PORT = 8123
MAX_STEPS = 90


def _wait_port(port, proc, timeout=60):
    for _ in range(timeout):
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def main(task_id=0, robot="YAM_bimanual"):
    import pandas as pd

    task_name = SUPPORTED_TASKS[task_id]
    experiment_name = "yamlab_adapter_test"
    run_id = "test_run"
    base_log_dir = scratch_log_root("yamlab_adapter_test_tmp")
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)
    server_log = open(os.path.join(base_log_dir, "sweep_server.log"), "w")
    server = subprocess.Popen([sys.executable, "-u", str(PROJECT_ROOT / "tests" / "yamlab_sweep_server.py"),
                               "--port", str(PORT)], stdout=server_log, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT), env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    problems = []
    try:
        if not _wait_port(PORT, server):
            print(f"FAILED -- 1 problem(s): sweep server did not come up (see {server_log.name})")
            sys.exit(1)
        print(f"[0] sweep server up on {PORT}; log root: {base_log_dir}", flush=True)

        checks = run_eval_cell(
            f"{task_name}_yamlab", base_log_dir,
            task_id=task_id, pert_id=0, task_name=task_name, pert_name="Default",
            repeats=1, max_steps=MAX_STEPS, experiment_name=experiment_name, robot=robot,
            no_render=False, run_id=run_id, model_name="yamlab", model_type="yamlab", port=PORT,
        )
        if checks == "EXECUTION_FAILED":
            problems.append("eval crashed")
        else:
            ok, detail = summarize(checks)
            if not ok:
                problems.append(f"artifacts: {detail}")

        server_log.flush()
        srv = open(server_log.name).read()
        bad = [ln for ln in srv.splitlines() if "BAD OBSERVATION" in ln]
        print(f"[1] server saw {'NO' if not bad else len(bad)} contract violations; "
              f"{'first observation OK' if 'first observation OK' in srv else 'never received a valid observation'}")
        if bad or "first observation OK" not in srv:
            problems.append(f"observation contract: {bad[:1] or 'no valid observation received'}")

        if checks != "EXECUTION_FAILED":
            task_log_dir = run_log_dir(base_log_dir, experiment_name, "yamlab", run_id)
            q = np.stack([np.asarray(r, dtype=float) for r in
                          pd.read_parquet(glob.glob(f"{task_log_dir}/qpos/*.parquet")[0])["data"].iloc[0]])
            left, right, gl, gr = q[:, 0:6], q[:, 6:12], q[:, 12], q[:, 13]
            moved_l, moved_r = np.ptp(left, axis=0), np.ptp(right, axis=0)
            corr = float(np.corrcoef(left[:, 0], right[:, 0])[0, 1]) if moved_l[0] > 1e-3 else float("nan")
            print(f"[2] qpos {q.shape}: left joint ranges {moved_l.round(3)} right {moved_r.round(3)} | "
                  f"left/right joint1 correlation {corr:+.2f} (sweep mirrors them)")
            print(f"[3] grippers: left min {gl.min():.2f} max {gl.max():.2f}, right min {gr.min():.2f} max {gr.max():.2f}")
            if q.shape[1] != 14:
                problems.append(f"qpos width {q.shape[1]} != 14")
            if not (moved_l > 0.1).all() or not (moved_r > 0.1).all():
                problems.append(f"an arm joint did not follow the sweep: left {moved_l.round(3)} right {moved_r.round(3)}")
            if not (corr < -0.5):
                problems.append(f"left/right arms not mirrored (corr {corr:+.2f}): the right-arm columns may not reach the right arm")
            for name, g in (("left", gl), ("right", gr)):
                if not (g.min() < 0.15 and g.max() > 0.85):
                    problems.append(f"{name} gripper did not visit both states (min {g.min():.2f}, max {g.max():.2f})")
    finally:
        server.terminate()
        server.wait(timeout=10)
        server_log.close()

    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("\nPASSED -- yamlab adapter: contract accepted by the reference server, both arms and grippers follow the sweep")


if __name__ == "__main__":
    main()
