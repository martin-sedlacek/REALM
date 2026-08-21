from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson.utils.transform_utils as omnigibson_transform_utils

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Camera jitter: uniform position offset (metres, per axis) and pitch/yaw offsets (radians).
MAX_POS_DEVIATION = 0.2
MAX_PITCH_DEVIATION = 0.2
MAX_YAW_DEVIATION = 0.2


def _perturb_camera_pose(cam_pos: list[float], cam_orientation: list[float]) -> tuple[list[float], list[float]]:
    """@cam_pos/@cam_orientation with uniform position jitter and pitch/yaw jitter added."""
    cam_pos = np.array(cam_pos)
    delta_pos = np.random.uniform(-MAX_POS_DEVIATION, MAX_POS_DEVIATION, 3)
    cam_pos += delta_pos
    cam_pos = cam_pos.tolist()

    cam_orientation = torch.tensor(cam_orientation)
    cam_rpy = omnigibson_transform_utils.quat2euler(cam_orientation)
    cam_rpy[0] += (torch.rand(()) * 2 - 1) * MAX_PITCH_DEVIATION
    cam_rpy[2] += (torch.rand(()) * 2 - 1) * MAX_YAW_DEVIATION
    cam_orientation = omnigibson_transform_utils.euler2quat(cam_rpy)
    cam_orientation = cam_orientation.cpu().numpy().tolist()

    return cam_pos, cam_orientation


def _opposite_side_keys(
    keys: list[str],
    extrinsics: dict[str, dict[str, list[float]]],
    first_camera_y: float,
) -> list[str]:
    """Return poses whose robot-base-frame Y is strictly opposite @first_camera_y."""
    return [key for key in keys if extrinsics[key]["pos"][1] * first_camera_y < 0]


def v_view(env: "RealmEnvironmentDynamic") -> None:
    """Re-draw both external cameras' poses from the extrinsics catalogue, plus jitter.

    Each external sensor gets a pose picked from env.cfg_camera_extrinsics (drawer tasks pin the
    standard episode pair) with _perturb_camera_pose's jitter on top. Mutates only the external
    VisionSensor prims; nothing else on @env changes.
    """
    # TODO: in some cases, the objects are not fully visible - add a look_at or similar to minimize these cases
    #
    # No og.sim.stop()/play() around this loop. That cycle is GLOBAL while REALM applies
    # perturbations per member inside reset(), so vectorized it tore down every other member's scene
    # mid-reset -- the failure mode measured for VB-POSE in job 190555, where three of four members
    # lost their main object from the contact view and silently scored zero. Nothing here needs a
    # stopped sim: these are external VisionSensor prims, i.e. XForms rather than rigid bodies, so a
    # pose write is a USD transform update that applies whether or not physics is running, and there
    # is no velocity to zero afterwards (contrast vb_pose._place, which must keep_still()).
    sensor_count = len(env.omnigibson_env.external_sensors)
    robot_pos = env.cfg["robots"][0]["position"]
    robot_rot = env.cfg["robots"][0]["orientation"]
    robot_rot = omnigibson_transform_utils.quat2euler(
        torch.tensor(robot_rot, dtype=torch.float32)
    ).tolist()
    first_camera_y = None

    for i in range(sensor_count):

        cam_pose_keys = list(env.cfg_camera_extrinsics.keys())
        filtered_cam_pose_keys = [
            key for key in cam_pose_keys
            if (
                    not key.startswith('CP') and
                    not (i == 0 and 'cam2' in key) and
                    not (i == 1 and 'cam1' in key) and
                    not (sensor_count > 1 and i == 0 and env.cfg_camera_extrinsics[key]["pos"][1] == 0)
            )
        ]
        if env.task_type in ["open_drawer", "close_drawer"]:
            cam_pose_name = "ep_001042_cam1" if i == 0 else "ep_001042_cam2" # TODO: scene specific, just get the extrinsic key dynamically
        else:
            if i == 1:
                filtered_cam_pose_keys = _opposite_side_keys(
                    filtered_cam_pose_keys,
                    env.cfg_camera_extrinsics,
                    first_camera_y,
                )
                if not filtered_cam_pose_keys:
                    raise ValueError("V-VIEW has no camera extrinsic on the opposite side of the robot")
            cam_pose_name = np.random.choice(filtered_cam_pose_keys)
        if i == 0:
            first_camera_y = env.cfg_camera_extrinsics[cam_pose_name]["pos"][1]
        cam_pos, cam_orientation = env.construct_ext_cam_pose_by_name(cam_pose_name, robot_pos, robot_rot)
        new_cam_pos, new_cam_orientation = _perturb_camera_pose(cam_pos, cam_orientation)
        base_cam_config = env.cfg["env"]["external_sensors"][i]
        pose_frame = base_cam_config["pose_frame"]
        env.omnigibson_env.external_sensors[base_cam_config["name"]].set_position_orientation(new_cam_pos, new_cam_orientation, pose_frame)
    # No og.Environment.reset() here: it would restore nothing (reset_pre_perturbation() already
    # reset this member, and the loop above writes only external-sensor prims, which reset() does
    # not touch) while costing a GLOBAL sim step plus three renders PER MEMBER in a vector env.
    # reset_joints() is kept so this function's post-state matches the old stop/play version; it is
    # a no-op on every task except the drawer pair.
    env.reset_joints()
