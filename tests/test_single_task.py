"""One task's slice of tests/test_integrity.py, for when you only care about one task.

    ./scripts/run_apptainer.sh python -u tests/test_single_task.py --task_id 4

Same coverage limits as tests/test_integrity.py -- read that module's docstring. In short:
`--model_type debug` commands a hold-still action with the gripper open, `--max_steps 1` skips
every metric formula, and `--no_render` drops the EXTERNAL sensors only, so the recorded frame is
extract_from_obs's synthetic black fallback for base_im beside a real wrist view. This checks that
the plumbing runs, nothing else.
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
# on top of the one the child boots. See eval_const_list.
SUPPORTED_TASKS = eval_const_list("SUPPORTED_TASKS")

REPEATS = 1
MAX_STEPS = 1

#: The robot asset this test drives. Passed EXPLICITLY rather than inherited from
#: examples/02_evaluate.py's own default, which is what happened until 2026-08-16 -- so half of
#: tests/ ran DROID_mounted (test_vector_integrity, test_scene_object_placement, which set it)
#: and half ran stock DROID (this file, test_integrity), and nothing in either said so. The value
#: is unchanged: "DROID" is exactly what 02_evaluate.py's default gave this test before. What
#: changes is that the choice is now visible and overridable with --robot.
#:
#: Which asset this SHOULD be is a separate question, and not one this test can settle: with
#: --model_type debug and --max_steps 1 no rollout moves, so no success condition is reachable on
#: any asset. This threshold was measured during the OmniGibson 3.9.1 port; this test cannot tell
#: the assets apart.
DEFAULT_ROBOT = "DROID"


def run_test(task_id=0, robot=DEFAULT_ROBOT):
    task_name = SUPPORTED_TASKS[task_id]
    experiment_name = "single_task_test"
    model_name = "debug"
    model_type = "debug"
    port = 8000
    run_id = "test_run"
    base_log_dir = scratch_log_root("single_task_test_tmp")

    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Running single-task integrity test: task {task_id} ({task_name})")
    print(f"log root: {base_log_dir}", flush=True)

    checks = run_eval_cell(
        task_name, base_log_dir,
        task_id=task_id, pert_id=0, task_name=task_name, pert_name="Default",
        repeats=REPEATS, max_steps=MAX_STEPS,
        experiment_name=experiment_name, robot=robot, no_render=True,
        run_id=run_id, model_name=model_name, model_type=model_type, port=port,
    )

    if checks != "EXECUTION_FAILED":
        all_pass, _ = summarize(checks)
        if all_pass:
            print(f"\nTask {task_id} ({task_name}) PASSED!")
            return
    print(f"\nTask {task_id} ({task_name}) FAILED!")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-task integrity test")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID to test (0-9)")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT,
                        help="robot config to evaluate (default: %(default)s)")
    args = parser.parse_args()
    run_test(args.task_id, args.robot)
