"""Boot ONE REALM env under EE control and dump everything needed to judge the joint mapping.

Answers, at runtime rather than by reading:
  * which OmniGibson joints the arm controller's dof_idx actually selects, and in what order,
    compared with the MJCF chain the IK solves on (panda_joint1..7);
  * what frame the controller's eef pose is expressed in (i.e. whether height_offset=0.87 is
    right for this asset), by comparing the controller's relative pose against the world pose
    and against env._world2robot();
  * whether stepping with the debug EE action moves the arm toward the commanded pose.

    REALM_ROBOT=DROID_ee_control python -u /app/tmp/ee_env_probe.py
"""
import os
import sys

import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as Rot

np.set_printoptions(precision=5, suppress=True, linewidth=200)

ROBOT = os.environ.get("REALM_ROBOT", "DROID_ee_control")
TASK_CFG = os.environ.get("REALM_TASK_CFG", "REALM_DROID10/put_green_block_into_bowl/default.yaml")
STEPS = int(os.environ.get("REALM_STEPS", "8"))


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def _np(x):
    """torch tensor (any device) / cb array / list -> numpy."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


import omnigibson as og
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

from realm.sim_config import set_sim_config
from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.environments.constants import DROID_BASE_HEIGHT

print(f"[probe] robot={ROBOT} task={TASK_CFG}", flush=True)
set_sim_config(robot=ROBOT)

env = RealmEnvironmentDynamic(
    config_path="/app/realm/config",
    task_cfg_path=TASK_CFG,
    perturbations=["Default"],
    multi_view=False,
    no_rendering=False,
    rendering_mode="rt",
    robot=ROBOT,
)
obs, _ = env.reset()
robot = env.robot

hdr("A. ROBOT / ENV IDENTITY")
print(f"  robot.name          = {robot.name}")
print(f"  robot.model_name    = {getattr(robot, 'model_name', '<none>')}")
print(f"  robot.prim_path     = {robot.prim_path}")
print(f"  env.ee_control      = {env.ee_control}")
print(f"  env.robot_name      = {env.robot_name}")
print(f"  env.use_droid_with_base = {env.use_droid_with_base}   DROID_BASE_HEIGHT = {DROID_BASE_HEIGHT}")
print(f"  env.robot_pos       = {env.robot_pos}   env.robot_rot_rad = {env.robot_rot_rad}")
print(f"  env.reset_qpos      = {np.asarray(env.reset_qpos)}")
print(f"  wrist_camera_key    = {getattr(env, 'wrist_camera_key', '<unset>')}")

hdr("B. LINKS / ROOT FRAME")
link_names = list(robot.links.keys())
print(f"  n_links = {len(link_names)}")
print(f"  links   = {link_names}")
try:
    print(f"  robot.root_link.body_name = {robot.root_link.body_name}")
except Exception as e:
    print(f"  robot.root_link -> {type(e).__name__}: {e}")
print(f"  robot.articulation_root_path = {robot.articulation_root_path}")
print(f"  og.sim.device = {og.sim.device!r}   (numpy conversions in droid_ee_controller assume cpu)")
print(f"  robot.eef_link_names = {robot.eef_link_names}")
print(f"  robot.default_arm    = {robot.default_arm}")

hdr("C. JOINT / DOF ORDER  (the joint-mapping check)")
joint_names = list(robot.joints.keys())
print(f"  robot.joints order ({len(joint_names)}):")
for i, jn in enumerate(joint_names):
    print(f"     dof {i:>2}: {jn}")
print(f"  robot.arm_joint_names = {robot.arm_joint_names}")
# OG 3.9.1: robot._controllers[name] is (group_key, controller_idx); the instance lives in the
# ControllerView registry, shared by every robot with the same tree pattern + identical config.
from omnigibson.controllers.controller_view import ControllerView
arm_key, arm_member = robot._controllers["arm_0"]
arm_ctrl = ControllerView._controller_groups[arm_key]
print(f"\n  arm group_key        = {arm_key}")
print(f"  arm member idx       = {arm_member}   (group has "
      f"{len(getattr(arm_ctrl, '_articulation_root_paths', []))} member(s))")
print(f"  arm controller       = {type(arm_ctrl).__name__}")
print(f"  arm.dof_idx          = {_np(arm_ctrl.dof_idx)}")
print(f"  arm.control_dim      = {arm_ctrl.control_dim}")
print(f"  arm.command_dim      = {arm_ctrl.command_dim}")
print(f"  arm.mode             = {getattr(arm_ctrl, 'mode', '<n/a>')}")
print(f"  arm.height_offset    = {getattr(arm_ctrl, 'height_offset', '<n/a>')}")
print(f"  arm._link_name       = {getattr(arm_ctrl, '_link_name', '<n/a>')}")
print(f"  arm.routing_path     = {getattr(arm_ctrl, 'routing_path', '<n/a>')}")
try:
    print(f"  arm.view_row_indices = {arm_ctrl.view_row_indices}")
except Exception as e:
    print(f"  arm.view_row_indices -> {type(e).__name__}: {e}")
print(f"  arm._use_cc_compensation      = {getattr(arm_ctrl, '_use_cc_compensation', '<n/a>')!r}")
print(f"  arm._use_gravity_compensation = {getattr(arm_ctrl, '_use_gravity_compensation', '<n/a>')!r}")
dof_joint_names = [joint_names[int(i)] for i in _np(arm_ctrl.dof_idx)]
print(f"\n  OG joints selected by dof_idx, IN ORDER:")
for k, jn in enumerate(dof_joint_names):
    print(f"     ctrl slot {k}: {jn}")

ik = getattr(arm_ctrl, "_ik_solver", None)
if ik is not None:
    mjcf_joints = [j.name for j in ik._arm.joints]
    print(f"\n  MJCF joints the IK solves on, IN ORDER:")
    for k, jn in enumerate(mjcf_joints):
        print(f"     ik   slot {k}: {jn}")
    match = (len(mjcf_joints) == len(dof_joint_names)) and all(
        a == b for a, b in zip(mjcf_joints, dof_joint_names))
    print(f"\n  JOINT_MAPPING_MATCH = {match}")
    if not match:
        print(f"  !! MISMATCH: IK order {mjcf_joints} vs OG order {dof_joint_names}")
else:
    print("  (no _ik_solver on this controller -- not an EE-control config)")

grip_key, grip_member = robot._controllers["gripper_0"]
gripper_ctrl = ControllerView._controller_groups[grip_key]
print(f"\n  gripper controller   = {type(gripper_ctrl).__name__}")
print(f"  gripper.dof_idx      = {_np(gripper_ctrl.dof_idx)}")
print(f"  gripper dof joints   = {[joint_names[int(i)] for i in _np(gripper_ctrl.dof_idx)]}")
print(f"  robot.action_dim     = {robot.action_dim}")

hdr("D. FRAMES: is height_offset=0.87 right for this asset?")
rel_pos, rel_quat = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
    arm_ctrl.routing_path, arm_ctrl._link_name)
rel_pos = _np(rel_pos)[arm_member]
rel_quat = _np(rel_quat)[arm_member]
ee_pos_w, ee_quat_w = env.get_ee_pose()
ee_pos_w = _np(ee_pos_w)
ee_quat_w = _np(ee_quat_w)
ee_pose_w = np.concatenate([ee_pos_w, Rot.from_quat(ee_quat_w).as_euler("xyz")])
ee_pose_robotframe = env._world2robot(np.concatenate([ee_pose_w, [0.0]]))[:6]

root_pos, root_quat = robot.get_position_orientation()
root_pos = _np(root_pos)
l0 = robot.links.get("panda_link0")
if l0 is not None:
    l0p, _ = l0.get_position_orientation()
    l0p = _np(l0p)
else:
    l0p = None

print(f"  robot root world pos                 = {root_pos}")
print(f"  panda_link0 world pos                = {l0p}")
print(f"  eef ({arm_ctrl._link_name}) world pose      = {ee_pose_w}")
print(f"  eef pose RELATIVE (controller input) = {np.concatenate([rel_pos, Rot.from_quat(rel_quat).as_euler('xyz')])}")
print(f"  env._world2robot(eef world)          = {ee_pose_robotframe}")
print(f"\n  relative_z                              = {rel_pos[2]:.5f}")
print(f"  _world2robot_z (what the policy sees)   = {ee_pose_robotframe[2]:.5f}")
print(f"  difference (should equal height_offset) = {rel_pos[2] - ee_pose_robotframe[2]:.5f}")
print(f"  controller height_offset                = {arm_ctrl.height_offset}")
print(f"  HEIGHT_OFFSET_CONSISTENT = "
      f"{abs((rel_pos[2] - ee_pose_robotframe[2]) - arm_ctrl.height_offset) < 0.02}")

hdr("E. DEBUG-POLICY ACTION -> COMMANDED TARGET")
client = InferenceClient("debug", host="127.0.0.1", port=8000)
(base_im, _, base_im_second, _, wrist_im, robot_state, gripper_state) = extract_from_obs(
    obs, robot_name=robot.name)
print(f"  robot_state (proprio[:7]) = {_np(robot_state)}")
print(f"  gripper_state (normalised) = {gripper_state}")
print(f"  wrist_im shape = {np.asarray(wrist_im).shape}  base_im shape = {np.asarray(base_im).shape}")
action = client.infer(env.instruction, base_im, base_im_second, wrist_im, robot_state,
                      gripper_state, ee_control=env.ee_control,
                      cartesian_position=ee_pose_robotframe.astype(np.float32))
action = np.asarray(action, dtype=float)
print(f"  debug action (robot frame, xyz+rpy+grip) = {action}")
print(f"  -> controller target_pos (z + height_offset) = "
      f"{np.array([action[0], action[1], action[2] + arm_ctrl.height_offset])}")
print(f"  -> current relative pos                      = {rel_pos}")
# Ask the controller itself, rather than reproducing its arithmetic. _update_goal is pure -- it
# returns a dict and does not write self._goals -- so calling it here disturbs nothing.
try:
    goal = arm_ctrl._update_goal(arm_member, th.tensor(action[:6], dtype=th.float32))
    print(f"  -> controller _update_goal target_pos         = {_np(goal['target_pos'])}")
    print(f"  -> controller _update_goal target_pos_relative= {_np(goal['target_pos_relative'])}")
    print(f"  -> controller _update_goal target_rpy         = {_np(goal['target_rpy'])}")
except Exception as e:
    import traceback as _tb
    print(f"  -> _update_goal FAILED: {type(e).__name__}: {e}")
    _tb.print_exc()
print(f"  -> |target - current| = {np.linalg.norm(np.array([action[0], action[1], action[2] + arm_ctrl.height_offset]) - rel_pos):.5f} m")

hdr(f"F. STEPPING {STEPS} STEPS WITH THE DEBUG ACTION")
new_action = action.copy()
new_action[-1] = 1 if action[-1] > 0.5 else -1
for t in range(STEPS):
    rel_pos, rel_quat = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
        arm_ctrl.routing_path, arm_ctrl._link_name)
    rel_pos_t = _np(rel_pos)[arm_member]
    q = _np(obs[robot.name]["proprio"])[:7]
    tgt = np.array([action[0], action[1], action[2] + arm_ctrl.height_offset])
    print(f"  t={t:>2}  relpos={rel_pos_t}  |err|={np.linalg.norm(tgt - rel_pos_t):.5f}  qpos={q}")
    if not np.all(np.isfinite(q)):
        print("  !! NON-FINITE qpos -- sim has blown up")
        break
    obs, prog, terminated, truncated, info = env.step(new_action)

rel_pos, _ = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
    arm_ctrl.routing_path, arm_ctrl._link_name)
rel_pos_f = _np(rel_pos)[arm_member]
q = _np(obs[robot.name]["proprio"])[:7]
tgt = np.array([action[0], action[1], action[2] + arm_ctrl.height_offset])
print(f"  final relpos={rel_pos_f}  |err|={np.linalg.norm(tgt - rel_pos_f):.5f}  qpos={q}")
print(f"  qpos finite = {np.all(np.isfinite(q))}")

hdr("VERDICT")
print("EE_ENV_PROBE_OK")
sys.stdout.flush()
og.shutdown()
