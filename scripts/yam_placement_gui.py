#!/usr/bin/env python
"""Nudge the YAM around a REALM scene in the Isaac GUI and read off its offset from the DROID spawn pose.

    OMNIGIBSON_HEADLESS=0 python /app/scripts/yam_placement_gui.py --robot YAM_single_arm --task_id 1

Builds the real evaluation environment (same scene, table, objects and exterior cameras a rollout
would get), holds the reset pose, and moves the WHOLE robot with the keyboard while the Kit window has
focus. Every move prints the robot's offset from the pose the config currently gives it (the DROID
arm-base pose, scene spawn + `mount_height`, shifted by the config's existing `spawn_offset`) in the
robot frame:

    dx forward (+x of the robot), dy left (+y), dz up, dyaw about the robot's own z,

and ENTER prints the TOTAL offset from the DROID pose (existing `spawn_offset` + this session's nudges)
as the YAML block to paste into realm/config/robots/<robot>.yaml.

Keys (the Kit viewport must be focused; the viewport's own mouse navigation keeps working):

    UP / DOWN          +x / -x           (forward / back, robot frame)
    LEFT / RIGHT       +y / -y           (left / right, robot frame)
    PAGE_UP / PAGE_DOWN  +z / -z
    Q / E              yaw +5 deg / -5 deg (about the robot's z)
    [ / ]              halve / double the translation step (starts at 2 cm)
    R                  back to the nominal pose
    P                  print the current offset again
    ENTER              print the YAML snippet for realm/config/robots/<robot>.yaml

The viewer camera starts at the task's first exterior camera (`cam1`), so the view is the one the policy
gets; orbit/pan it with the mouse as usual. Ctrl-C in the terminal quits.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STEP_M = 0.02
YAW_STEP_DEG = 5.0
STATUS_EVERY_S = 10.0


class Placement:
    """Offset of the robot from its nominal spawn, in the nominal frame; applies itself to the sim."""

    def __init__(self, env):
        from scipy.spatial.transform import Rotation as R

        self.env = env
        self.R = R
        self.nominal_pos = np.asarray(env.robot_pos, dtype=float) + np.array([0.0, 0.0, env.base_height])
        self.nominal_rot = R.from_euler("xyz", np.asarray(env.robot_rot_rad, dtype=float))
        self.d = np.zeros(3)
        self.dyaw_deg = 0.0
        self.step_m = DEFAULT_STEP_M
        self.dirty = False

    # -- edits (called from keyboard callbacks; applied in the sim loop) --------------------------
    def translate(self, axis, sign):
        self.d[axis] += sign * self.step_m
        self.dirty = True

    def yaw(self, sign):
        self.dyaw_deg += sign * YAW_STEP_DEG
        self.dirty = True

    def scale_step(self, factor):
        self.step_m = float(np.clip(self.step_m * factor, 0.0025, 0.32))
        print(f"[placement] translation step = {self.step_m * 100:.2f} cm")

    def reset(self):
        self.d[:] = 0.0
        self.dyaw_deg = 0.0
        self.dirty = True

    # -- sim ----------------------------------------------------------------------------------------
    def world_pose(self):
        pos = self.nominal_pos + self.nominal_rot.apply(self.d)
        rot = self.nominal_rot * self.R.from_euler("z", np.radians(self.dyaw_deg))
        return pos, rot.as_quat()  # xyzw, OmniGibson's convention

    def apply(self):
        import torch as th

        pos, quat = self.world_pose()
        self.env.robot.set_position_orientation(th.tensor(pos, dtype=th.float32), th.tensor(quat, dtype=th.float32))
        self.dirty = False
        self.report()

    # -- output -------------------------------------------------------------------------------------
    def report(self):
        pos, quat = self.env.robot.get_position_orientation()
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
        # Measured offset, so a move PhysX refused (or clamped) shows up as a mismatch with the command.
        rel = self.nominal_rot.inv().apply(pos - self.nominal_pos)
        yaw_rel = np.degrees((self.nominal_rot.inv() * self.R.from_quat(quat)).as_euler("xyz")[2])
        print(f"[placement] offset from the config's spawn pose (robot frame): dx={rel[0]:+.3f} dy={rel[1]:+.3f} "
              f"dz={rel[2]:+.3f} m  dyaw={yaw_rel:+.1f} deg   | commanded dx={self.d[0]:+.3f} dy={self.d[1]:+.3f} "
              f"dz={self.d[2]:+.3f} dyaw={self.dyaw_deg:+.1f}   | world pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
              flush=True)

    def yaml_snippet(self):
        # The config's existing offset is already inside nominal_pos/nominal_rot (env_config applied it),
        # so the total offset from the DROID pose composes the two: rotate this session's nudge by the
        # existing yaw and add.
        existing = getattr(self.env, "spawn_offset", {"pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0})
        R_existing = self.R.from_euler("z", np.radians(existing["yaw_deg"]))
        total = np.asarray(existing["pos"], dtype=float) + R_existing.apply(self.d)
        total_yaw = existing["yaw_deg"] + self.dyaw_deg
        print("\n# realm/config/robots/<robot>.yaml -- total offset from the DROID spawn pose (robot frame:\n"
              "# x forward, y left, z up; yaw about the robot's z), replacing the current spawn_offset block:\n"
              f"    spawn_offset:\n"
              f"      pos: [{total[0]:.4f}, {total[1]:.4f}, {total[2]:.4f}]\n"
              f"      yaw_deg: {total_yaw:.1f}\n", flush=True)


def install_keys(placement):
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    K = lazy.carb.input.KeyboardInput
    binds = {
        K.UP: lambda: placement.translate(0, +1),
        K.DOWN: lambda: placement.translate(0, -1),
        K.LEFT: lambda: placement.translate(1, +1),
        K.RIGHT: lambda: placement.translate(1, -1),
        K.PAGE_UP: lambda: placement.translate(2, +1),
        K.PAGE_DOWN: lambda: placement.translate(2, -1),
        K.Q: lambda: placement.yaw(+1),
        K.E: lambda: placement.yaw(-1),
        K.LEFT_BRACKET: lambda: placement.scale_step(0.5),
        K.RIGHT_BRACKET: lambda: placement.scale_step(2.0),
        K.R: placement.reset,
        K.P: placement.report,
        K.ENTER: placement.yaml_snippet,
    }
    for key, fn in binds.items():
        KeyboardEventHandler.add_keyboard_callback(key, fn)


def point_viewer_at_cam1(env):
    """Start the GUI viewer where the policy's first exterior camera is (scene 0 == world frame)."""
    import omnigibson as og
    import torch as th

    sensors = env.cfg.get("env", {}).get("external_sensors") or []
    if not sensors or og.sim.viewer_camera is None:
        return
    cam = sensors[0]
    og.sim.viewer_camera.set_position_orientation(th.tensor(cam["position"], dtype=th.float32),
                                                  th.tensor(cam["orientation"], dtype=th.float32))


def smoke(env, placement, hold, steps=15, tol_m=0.005, tol_deg=0.5):
    """Headless proof that a live pose write moves the fixed-base robot and that the arm stays put on it.

    Commands one nudge (+10 cm forward, +5 cm left, +2 cm up, +10 deg yaw), steps physics, and checks the
    measured offset against the command and the arm joints against the reset pose. Exits 0/1 with a
    printed verdict; nothing is written.
    """
    import torch as th

    q0 = env.robot.get_joint_positions().cpu().numpy().copy()
    placement.d[:] = (0.10, 0.05, 0.02)
    placement.dyaw_deg = 10.0
    placement.apply()
    for _ in range(steps):
        env.step(hold)
    pos, quat = env.robot.get_position_orientation()
    rel = placement.nominal_rot.inv().apply(pos.cpu().numpy() - placement.nominal_pos)
    yaw_rel = np.degrees((placement.nominal_rot.inv() * placement.R.from_quat(quat.cpu().numpy())).as_euler("xyz")[2])
    q1 = env.robot.get_joint_positions().cpu().numpy()
    problems = []
    if np.max(np.abs(rel - placement.d)) > tol_m:
        problems.append(f"measured offset {rel} != commanded {placement.d} (tol {tol_m} m)")
    if abs(yaw_rel - placement.dyaw_deg) > tol_deg:
        problems.append(f"measured yaw {yaw_rel:.2f} != commanded {placement.dyaw_deg} deg")
    n_arm = sum(len(env.robot.arm_joint_names[a]) for a in env.robot.arm_names)
    drift = float(np.max(np.abs(q1 - q0)[:len(q0)]))
    if drift > 0.02:
        problems.append(f"joints drifted {drift:.3f} rad across the move (arm joints {n_arm}); the robot should hold its pose")
    placement.report()
    print(f"[placement] joint drift across the move: {drift:.4f} rad", flush=True)
    if problems:
        print("[placement] SMOKE FAILED --\n  " + "\n  ".join(problems), flush=True)
        # Isaac exits 0 on unhandled exceptions; the verdict line is what tests read.
        th.cuda.synchronize() if th.cuda.is_available() else None
        os._exit(1)
    print("[placement] SMOKE PASSED -- live pose write moves the fixed-base robot", flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="YAM_single_arm", help="realm/config/robots/<robot>.yaml (YAM_single_arm, YAM_bimanual, YAM_molmoact2, ...)")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--task_cfg_path", default=None, help="overrides --task_id")
    parser.add_argument("--multi-view", dest="multi_view", action="store_true", help="also spawn the second exterior camera")
    parser.add_argument("--rendering_mode", default="rt", choices=("rt", "pt", "r"))
    parser.add_argument("--smoke", action="store_true",
                        help="headless self-check: apply a fixed nudge, step, verify the robot moved, exit")
    args = parser.parse_args()

    if not args.smoke and os.environ.get("OMNIGIBSON_HEADLESS", "1") != "0":
        print("[placement] WARNING: OMNIGIBSON_HEADLESS is not 0 -- no window will open and no key will be received. "
              "Run through scripts/run_docker.sh WITHOUT --headless (it exports OMNIGIBSON_HEADLESS=0).", flush=True)

    from realm.eval import CONFIG_ROOT, SUPPORTED_TASKS
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.rollout import resolve_task
    from realm.sim_config import set_sim_config

    set_sim_config(robot=args.robot)
    task, task_cfg_path = resolve_task(args.task_id, args.task_cfg_path, SUPPORTED_TASKS, name_includes_config=True)
    env = RealmEnvironmentDynamic(config_path=CONFIG_ROOT, task_cfg_path=task_cfg_path, perturbations=["Default"],
                                  robot=args.robot, multi_view=args.multi_view, rendering_mode=args.rendering_mode)
    env.reset()
    placement = Placement(env)
    hold = env.warmup_action(0, env.warmup_ee_cmd())  # reset pose, gripper open

    if args.smoke:
        smoke(env, placement, hold)
        return

    point_viewer_at_cam1(env)
    install_keys(placement)

    print(f"\n[placement] task {task}, robot {args.robot}. Nominal (DROID) arm-base pose in world: "
          f"({placement.nominal_pos[0]:.3f}, {placement.nominal_pos[1]:.3f}, {placement.nominal_pos[2]:.3f}), "
          f"yaw {np.degrees(env.robot_rot_rad[2]):.1f} deg. Click the viewport, then use the keys in the docstring.",
          flush=True)
    placement.report()

    last_status = time.time()
    while True:
        if placement.dirty:
            placement.apply()
        env.step(hold)
        if time.time() - last_status > STATUS_EVERY_S:
            placement.report()
            last_status = time.time()


if __name__ == "__main__":
    main()
