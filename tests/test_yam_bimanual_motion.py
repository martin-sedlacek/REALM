"""Does each column of the 14-D YAM_bimanual action drive the joint it is supposed to? Container + GPU.

    ./scripts/run_apptainer.sh python -u tests/test_yam_bimanual_motion.py

WHY THIS EXISTS. OmniGibson orders the action by `raw_controller_order` (arm_left, gripper_left,
arm_right, gripper_right) and PhysX orders the DOFs breadth-first (left_joint1, right_joint1, ...),
so nothing about the action layout can be read off the proprio layout, and the debug policy's
zero action holds the reset pose without ever exercising the mapping. This boots one environment,
steps scripted actions with a single non-zero column at a time, and checks -- by joint NAME through
ROBOT_OBS_PROFILES -- that exactly the intended joint moved to the target and everything else stayed
put; then opens/closes the two grippers asymmetrically to pin columns 6 and 13 to their arms.

Script-style with printed verdicts (PASSED -- / FAILED -- N problem(s)), like the other tier-2 tests.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

TARGET_RAD = 0.3          # well inside every YAM joint range; large enough to see
STEPS_PER_PHASE = 45      # 1.5 s at 30 Hz: the previous joint must be back at zero too
JOINT_TOL_RAD = 0.05
GRIPPER_OPEN_MAX = 0.15   # normalised (0 open, 1 closed)
GRIPPER_CLOSED_MIN = 0.85


def main(task_cfg_path, robot):
    from realm.eval import CONFIG_ROOT
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.inference.utils import extract_from_obs, get_robot_obs_profile
    from realm.sim_config import set_sim_config

    set_sim_config(robot=robot)
    env = RealmEnvironmentDynamic(config_path=CONFIG_ROOT, task_cfg_path=task_cfg_path,
                                  perturbations=["Default"], robot=robot, no_rendering=True)
    profile = get_robot_obs_profile(env.robot.name)
    arms = list(profile["arms"])
    arm_dof = profile["arm_dof"]
    action_dim = len(arms) * (arm_dof + 1)
    # expected layout: [arm_0 joints, gripper_0, arm_1 joints, gripper_1, ...] in `arms` order
    expected_col_joint = {}
    gripper_cols = {}
    for a, arm in enumerate(arms):
        base = a * (arm_dof + 1)
        for j, joint in enumerate(profile["arm_joint_names"][arm]):
            expected_col_joint[base + j] = joint
        gripper_cols[arm] = base + arm_dof
    assert tuple(gripper_cols.values()) == tuple(profile["gripper_action_idx"]), (gripper_cols, profile["gripper_action_idx"])

    obs, _ = env.reset()
    obs, *_ = env.warmup(obs)

    # Target sign per joint from the built robot's limits (DOF order), so a joint whose zero sits near
    # one limit is driven the other way; the limits themselves are printed for the record.
    dof_names = list(env.robot.dof_names_ordered)
    lower = np.asarray(env.robot.joint_lower_limits.cpu().numpy() if hasattr(env.robot.joint_lower_limits, "cpu")
                       else env.robot.joint_lower_limits, dtype=float)
    upper = np.asarray(env.robot.joint_upper_limits.cpu().numpy() if hasattr(env.robot.joint_upper_limits, "cpu")
                       else env.robot.joint_upper_limits, dtype=float)
    limits = {n: (float(lower[i]), float(upper[i])) for i, n in enumerate(dof_names)}
    print("joint limits (rad / m): " + ", ".join(f"{n} [{lo:+.2f}, {hi:+.2f}]" for n, (lo, hi) in limits.items()), flush=True)

    def target_for(joint):
        lo, hi = limits[joint]
        if hi >= TARGET_RAD + JOINT_TOL_RAD:
            return TARGET_RAD
        assert lo <= -TARGET_RAD - JOINT_TOL_RAD, f"{joint} range [{lo}, {hi}] too narrow for the test"
        return -TARGET_RAD

    def step_n(action, n):
        nonlocal obs
        for _ in range(n):
            obs, *_ = env.step(action)
        return obs

    def state():
        _, _, _, _, _, robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
        joints = {}
        for a, arm in enumerate(arms):
            for j, joint in enumerate(profile["arm_joint_names"][arm]):
                joints[joint] = float(robot_state[a * arm_dof + j])
        return joints, {arm: float(gripper_state[a]) for a, arm in enumerate(arms)}

    problems = []
    n_cell = 0

    # --- one arm-joint column at a time ------------------------------------------------------
    for col, joint in expected_col_joint.items():
        action = np.zeros(action_dim)
        action[list(gripper_cols.values())] = 1.0  # grippers open throughout
        target = target_for(joint)
        action[col] = target
        step_n(action, STEPS_PER_PHASE)
        joints, _ = state()
        reached = joints[joint]
        others = {k: v for k, v in joints.items() if k != joint}
        worst = max(others, key=lambda k: abs(others[k]))
        ok = abs(reached - target) < JOINT_TOL_RAD and abs(others[worst]) < JOINT_TOL_RAD
        print(f"[{n_cell}] action[{col:2d}] -> {joint:14s} target {target:+.2f} reached {reached:+.3f} | "
              f"largest other {worst} {others[worst]:+.3f} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            problems.append(f"action column {col} ({joint}): reached {reached:+.3f}, largest other {worst}={others[worst]:+.3f}")
        n_cell += 1

    # --- grippers: same, then opposite, so the two columns cannot be swapped -----------------
    for want in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]:
        action = np.zeros(action_dim)
        for arm, cmd in zip(arms, want):
            action[gripper_cols[arm]] = cmd
        step_n(action, 30)
        joints, grippers = state()
        ok = True
        desc = []
        for arm, cmd in zip(arms, want):
            g = grippers[arm]
            good = g < GRIPPER_OPEN_MAX if cmd > 0 else g > GRIPPER_CLOSED_MIN
            ok &= good
            desc.append(f"{arm} cmd {cmd:+.0f} -> {g:.3f}")
        arm_drift = max(abs(v) for v in joints.values())
        ok &= arm_drift < JOINT_TOL_RAD
        print(f"[{n_cell}] grippers {', '.join(desc)} | arm drift {arm_drift:.3f} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            problems.append(f"gripper command {want}: {desc}, arm drift {arm_drift:.3f}")
        n_cell += 1

    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print(f"\nPASSED -- {n_cell} cells: every action column drives its own joint; grippers map to their arms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_cfg_path", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
    parser.add_argument("--robot", default="YAM_bimanual")
    main(parser.parse_args().task_cfg_path, parser.parse_args().robot)
