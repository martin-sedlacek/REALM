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
    #
    # No og.sim.stop()/play() around this loop. That cycle is GLOBAL while REALM applies
    # perturbations per member inside reset(), so vectorized it tore down every other member's scene
    # mid-reset -- the failure mode measured for VB-POSE in job 190555, where three of four members
    # lost their main object from the contact view and silently scored zero. Nothing here needs a
    # stopped sim: these are external VisionSensor prims, i.e. XForms rather than rigid bodies, so a
    # pose write is a USD transform update that applies whether or not physics is running, and there
    # is no velocity to zero afterwards (contrast vb_pose._place, which must keep_still()).
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
    # An og.Environment.reset() used to sit here. It existed to restore scene state that
    # og.sim.stop() had clobbered; with the stop gone it restored nothing that was disturbed --
    # reset_pre_perturbation() already reset this member immediately before the perturbation ran,
    # and the loop above writes only external-sensor poses, which are env-level prims that
    # og.Environment.reset() does not touch (if it did, V-VIEW would already be a no-op and the
    # harness's `cameras` check could not pass).
    #
    # It was not free. og.Environment.reset(get_obs=True) issues a GLOBAL og.sim.step() plus three
    # og.sim.render()s, and REALM applies perturbations per member, so a vector reset paid it once
    # per member ON TOP of the one reset_pre_perturbation() already does -- measured 2N: 4 global
    # steps at Vec=2 against Default's 2, 8 at Vec=4 against 4. Same class as the per-member settle
    # loops removed from the other perturbations, one step per member instead of thirty.
    #
    # env.reset_joints() below is kept. It is the same "post-state identical to the stop/play
    # version" line vb_pose.py carries, it is a two-line no-op on every task that is not
    # open_drawer/close_drawer, and on those it is now batched across members anyway.
    env.reset_joints()
