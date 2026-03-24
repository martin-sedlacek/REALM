from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og
import omnigibson.utils.transform_utils as omnigibson_transform_utils

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def v_view(env: "RealmEnvironmentDynamic") -> None:
    def perturb_camera_pose(cam_pos: list[float], cam_orientation: list[float]) -> tuple[list[float], list[float]]:
        MAX_POS_DEVIATION = 0.2
        MAX_PITCH_DEVIATION = 0.2
        MAX_YAW_DEVIATION = 0.2
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

    # TODO: in some cases, the objects are not fully visible - add a look_at or similar to minimize these cases
    og.sim.stop()
    for i in range(len(env.omnigibson_env.external_sensors)):
        robot_pos = env.cfg["robots"][0]["position"]
        robot_rot = env.cfg["robots"][0]["orientation"]
        robot_rot = omnigibson_transform_utils.quat2euler(torch.tensor(robot_rot, dtype=torch.float32)).tolist()

        cam_pose_keys = list(env.cfg_camera_extrinsics.keys())
        filtered_cam_pose_keys = [
            key for key in cam_pose_keys
            if (
                    not key.startswith('CP') and
                    not (i == 0 and 'cam2' in key) and
                    not (i == 1 and 'cam1' in key)
            )
        ]
        if env.task_type in ["open_drawer", "close_drawer"]:
            cam_pose_name = "ep_001042_cam1" if i == 0 else "ep_001042_cam2" # TODO: scene specific, just get the extrinsic key dynamically
        else:
            cam_pose_name = np.random.choice(filtered_cam_pose_keys)
        cam_pos, cam_orientation = env.construct_ext_cam_pose_by_name(cam_pose_name, robot_pos, robot_rot)
        new_cam_pos, new_cam_orientation = perturb_camera_pose(cam_pos, cam_orientation)
        base_cam_config = env.cfg["env"]["external_sensors"][i]
        pose_frame = base_cam_config["pose_frame"]
        env.omnigibson_env.external_sensors[base_cam_config["name"]].set_position_orientation(new_cam_pos, new_cam_orientation, pose_frame)
    og.sim.play()
    obs, _ = env.omnigibson_env.reset()
    env.reset_joints()
