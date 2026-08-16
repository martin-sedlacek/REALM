"""Does EVERY perturbation still apply cleanly, on one task?

Sweeps all 16 entries of SUPPORTED_PERTURBATIONS through examples/02_evaluate.py, one process
each. The task-sweeping counterpart is tests/test_integrity.py.

    ./scripts/clara/interactive/rr python -u tests/test_perturbations_integrity.py --task_id 0

Rendering is left ON here (unlike test_integrity.py), so videos/<task>.parquet holds real frames.

WHAT THIS DOES NOT COVER
------------------------
This checks that a perturbation APPLIES WITHOUT CRASHING and that the run logs. It does NOT check
that the perturbation changed anything: a perturbation that silently no-ops passes every check
below, and one in this repo did exactly that. Nothing here compares a perturbed rollout against
its Default. `--model_type debug` also means the rollout is a hold-still action with the gripper
open, so no perturbation is ever exercised against actual manipulation.

    * `--task_id 0`'s main object is a PrimitiveObject, so VB-MOBJ takes its rescale branch here;
      pass `--task_id 4` (pick_spoon, a DatasetObject) to exercise its remove/add branch instead.
      Neither is covered by a default run.
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

from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS


def run_test(task_id=0, repeats=1, max_steps=1):
    """Run a short rollout under every perturbation.

    Args:
        task_id (int): which task from SUPPORTED_TASKS to exercise
        repeats (int): rollouts per perturbation -- >1 also exercises the per-repeat reset path
        max_steps (int): steps per rollout
    """
    experiment_name = "pert_integrity_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    task_name = SUPPORTED_TASKS[task_id]

    base_log_dir = scratch_log_root("pert_integrity_test_tmp")

    # Clean up previous test runs
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Starting perturbation integrity test for Task {task_id} ({task_name}), "
          f"{repeats} repeat(s) x {max_steps} step(s)...")
    print(f"Testing {len(SUPPORTED_PERTURBATIONS)} perturbations...")
    print(f"log root: {base_log_dir}", flush=True)

    results = {}

    for pert_id, pert_name in enumerate(SUPPORTED_PERTURBATIONS):
        print(f"\n--- Testing Perturbation {pert_id}: {pert_name} ---", flush=True)

        cmd = [
            sys.executable, "-u", str(PROJECT_ROOT / "examples/02_evaluate.py"),
            "--task_id", str(task_id),
            "--perturbation_id", str(pert_id),
            "--repeats", str(repeats),
            "--max_steps", str(max_steps),
            "--model_name", model_name,
            "--model_type", model_type,
            "--port", str(port),
            "--experiment_name", experiment_name,
            "--run_id", run_id,
            "--log_dir", base_log_dir
        ]

        # NOT check=True: Isaac segfaults at teardown on passing runs too. See test_integrity.py.
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        child_log = os.path.join(base_log_dir, f"{pert_name}.log")
        with open(child_log, "w") as fh:
            fh.write(proc.stdout or "")
            fh.write(proc.stderr or "")

        crashes = crash_lines((proc.stdout or "") + (proc.stderr or ""))
        if crashes:
            print(f"CRASHED during evaluation for {pert_name} (exit={proc.returncode})")
            print(f"  first crash line: {crashes[0].strip()[:200]}")
            print(f"  full child log: {child_log}")
            results[pert_name] = "EXECUTION_FAILED"
            continue
        print(f"Ran evaluation for {pert_name} (exit={proc.returncode}, no crash signature)")

        task_log_dir = os.path.join(base_log_dir, experiment_name, model_name, run_id)
        # check_artifacts filters the parquets to THIS perturbation and counts rows. It has to:
        # every perturbation in this sweep appends into the SAME qpos/<task>.parquet, so the old
        # "file exists and is non-empty" check was satisfied for perturbations 1..15 by whatever
        # perturbation 0 wrote, and could not fail once any one cell had succeeded.
        pert_results = check_artifacts(task_log_dir, task_name, pert_name, repeats)
        for key, status in pert_results.items():
            print(f"  {key}: {status}")
        if any(v != "PASS" for v in pert_results.values()):
            print(f"  full child log: {child_log}")

        results[pert_name] = pert_results

    # Summary
    print("\n" + "=" * 50)
    print("PERTURBATION INTEGRITY TEST SUMMARY")
    print("=" * 50)
    all_pass = True
    for pert, status in results.items():
        if status == "EXECUTION_FAILED":
            print(f"{pert}: FAILED EXECUTION")
            all_pass = False
        else:
            pert_pass, detail = summarize(status)
            if not pert_pass:
                all_pass = False
            print(f"{pert}: {'PASS' if pert_pass else 'FAIL'} ({detail})")

    if all_pass:
        print("\nALL PERTURBATIONS PASSED INTEGRITY CHECK!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run every perturbation for a short rollout.")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    run_test(task_id=args.task_id, repeats=args.repeats, max_steps=args.max_steps)
