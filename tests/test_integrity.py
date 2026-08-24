"""Does EVERY task still run end to end and produce its four log artifacts?

Sweeps all 10 tasks under Default through examples/02_evaluate.py, one process each, and checks
the reports csv plus the qpos/actions/videos parquets.

    ./scripts/clara/interactive/rr python -u tests/test_integrity.py

WHICH IMAGE YOU ARE ON DECIDES WHETHER THIS PASSES. Tasks 8 and 9 (open_drawer / close_drawer) are
the only two whose main object is `custom_assets/impact_drawer/usd/cabinet.usd`, and loading it
needs a `preset_name` default in `OmniSurfaceMaterialPrim.__init__`
(omnigibson/prims/material_prim.py). Without it they die with

    TypeError: missing a required argument: 'preset_name'

Three measured runs, 2026-08-16, differing only in image and bind:

    realm_og391.sif     MODE=stock    8/10 -- tasks 8 and 9 crash with the TypeError above
    realm_og391.sif     MODE=oglite   task 8 PASSES  (the fork carries the default)
    realm_og391_v2.sif  MODE=stock    10/10 ALL TASKS PASSED

`realm_og391_v2.sif` (built 2026-08-14) ships that patch and six others, and
`scripts/clara/lib/paths.sh` now selects it by default, so plain `rr` is sufficient. If you are
reproducing a result recorded before 2026-08-16, pin the old image --
`REALM_SIF_OG391=$REALM_SHARED/realm_og391.sif` -- and expect 8/10 here.

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
import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from tests._paths import eval_const_list, run_eval_cell, scratch_log_root, summarize

# ast-parsed, not imported: `from realm.eval import SUPPORTED_TASKS` boots Isaac in this DRIVER,
# on top of the one each child boots. See eval_const_list.
SUPPORTED_TASKS = eval_const_list("SUPPORTED_TASKS")

REPEATS = 1
MAX_STEPS = 1

#: The robot asset this sweep drives. Passed EXPLICITLY rather than inherited from
#: examples/02_evaluate.py's own default, which is what happened until 2026-08-16 -- so half of
#: tests/ ran DROID_mounted (test_vector_integrity, test_scene_object_placement, which set it)
#: and half ran stock DROID (this file, test_single_task), and nothing in either said so. The
#: value is unchanged: "DROID" is exactly what 02_evaluate.py's default gave this sweep before.
#: What changes is that the choice is visible and overridable with --robot.
DEFAULT_ROBOT = "DROID"


def run_test(robot=DEFAULT_ROBOT):
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
        results[task_name] = run_eval_cell(
            task_name, base_log_dir,
            task_id=task_id, pert_id=0, task_name=task_name, pert_name="Default",
            repeats=REPEATS, max_steps=MAX_STEPS,
            experiment_name=experiment_name, robot=robot, no_render=True,
            run_id=run_id, model_name=model_name, model_type=model_type, port=port,
        )

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
    parser = argparse.ArgumentParser(description="10-task integrity sweep")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT,
                        help="robot config to evaluate (default: %(default)s)")
    run_test(parser.parse_args().robot)
