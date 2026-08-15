"""Does EVERY task still run end to end and produce its four log artifacts?

Sweeps all 10 tasks under Default through examples/02_evaluate.py, one process each, and checks
the reports csv plus the qpos/actions/videos parquets.

    MODE=oglite ./scripts/clara/interactive/rr python -u tests/test_integrity.py

THE MODE MATTERS AND rr's DEFAULT IS NOT ENOUGH FOR ALL TEN TASKS. Under `MODE=stock` -- what you
get if you do not say otherwise -- tasks 8 and 9 (open_drawer / close_drawer) die while loading
`custom_assets/impact_drawer/usd/cabinet.usd` with

    TypeError: missing a required argument: 'preset_name'
      omnigibson/prims/material_prim.py, via OmniSurfaceMaterialPrim.__init__

They are the only two tasks whose main object needs it. That default is one of the three things
OG-lite and `stock_patch` supply and the image does not (see the MODE=stockfix block in
scripts/clara/interactive/rr). Measured 2026-08-16 as a controlled pair, same task, same code:

    MODE=stock   task 8 -> crash, the TypeError above     (this sweep, 8/10 tasks passed)
    MODE=oglite  task 8 -> PASSED, all four artifacts     (tests/run_suite.py --only
                                                           test_single_task_drawer)

So a red result here is not evidence of a code regression until the mode is checked. Run under
`MODE=oglite` or `MODE=stockfix` for a full-ten pass.

WHAT THIS DOES NOT COVER -- read before treating a green run as evidence
-----------------------------------------------------------------------
  * `--model_type debug` is not a policy. InferenceClient returns a hardcoded `np.zeros(8)` every
    call, whose last element becomes gripper=-1 (open). The arm is commanded to hold still and the
    gripper never closes, so NOTHING about grasping, placement, task progression or success is
    exercised. Every report row this test produces has task_progression 0.
  * `--max_steps 1` means one control step per rollout. eval.py's own metric block takes its
    `len(qpos_joints) > 4` false branch, so joint_vel_var / joint_acc_var / joint_jerk /
    joint_path_length / cart_* are all written as literal 0.0 without their formulas ever running.
  * `--repeats 1` means the per-repeat reset path is never entered.
  * `--no_render` drops the EXTERNAL sensors only -- env_config.py adds them under
    `if not env.no_rendering`, while the robot's own wrist camera is part of the robot and keeps
    rendering. So `extract_from_obs` takes its "external sensors are missing" fallback and hands
    the recorder a synthetic black 128x128 for base_im, next to a REAL wrist view. Measured on
    this gate's own artifact (task 0, 2026-08-16): the recorded 128x256 frame is max=2 mean=0.002
    on the left (base) half and max=214 mean=100 on the right (wrist) half. So the video check
    below proves append_video ran and the wrist camera rendered; it proves NOTHING about the
    external camera -- which is the view the policies are actually conditioned on. Use
    tests/test_vector_integrity.py (rendering on, --extract-videos) for that.

So: this is a smoke test for the eval plumbing and the task configs loading. It is not evidence
that a task works.
"""
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


def run_test():
    experiment_name = "integrity_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"

    base_log_dir = scratch_log_root("integrity_test_tmp")

    # Clean up previous test runs
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Starting integrity test for tasks 0-9...")
    print(f"log root: {base_log_dir}", flush=True)

    results = {}

    for task_id in range(10):
        task_name = SUPPORTED_TASKS[task_id]
        print(f"\n--- Testing Task {task_id}: {task_name} ---", flush=True)

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
            "--no_render"
        ]

        # NOT check=True. Isaac tears down with a segfault after all work is done, on passing runs
        # as much as on failing ones, so a non-zero status here is the NORM and gating on it marked
        # every task EXECUTION_FAILED regardless of what it produced. The child's own output is
        # what decides: a real traceback, or artifacts that are missing / the wrong length.
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        child_log = os.path.join(base_log_dir, f"{task_name}.log")
        with open(child_log, "w") as fh:
            fh.write(proc.stdout or "")
            fh.write(proc.stderr or "")

        crashes = crash_lines((proc.stdout or "") + (proc.stderr or ""))
        if crashes:
            print(f"CRASHED during evaluation for {task_name} (exit={proc.returncode})")
            print(f"  first crash line: {crashes[0].strip()[:200]}")
            print(f"  full child log: {child_log}")
            results[task_name] = "EXECUTION_FAILED"
            continue
        print(f"Ran evaluation for {task_name} (exit={proc.returncode}, no crash signature)")

        task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
        task_results = check_artifacts(task_log_dir, task_name, "Default", REPEATS)
        for key, status in task_results.items():
            print(f"  {key}: {status}")
        if any(v != "PASS" for v in task_results.values()):
            print(f"  full child log: {child_log}")
        results[task_name] = task_results

    # Summary
    print("\n" + "=" * 50)
    print("INTEGRITY TEST SUMMARY")
    print("=" * 50)
    all_pass = True
    for task, status in results.items():
        if status == "EXECUTION_FAILED":
            print(f"{task}: FAILED EXECUTION")
            all_pass = False
        else:
            task_pass, detail = summarize(status)
            if not task_pass:
                all_pass = False
            print(f"{task}: {'PASS' if task_pass else 'FAIL'} ({detail})")

    if all_pass:
        print("\nALL TASKS PASSED INTEGRITY CHECK!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_test()
