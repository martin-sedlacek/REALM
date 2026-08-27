"""VB-POSE object-pose perturbation."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from realm.environments.perturbations._helpers import backfill_object_cfgs, settle
from realm.geometry import add_rotation_noise
from realm.placement import place_within

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


SWITCH_DZ_RANGE = 0.15
SWITCH_DXY_RANGE = 0.075

DRAWER_YAW_NOISE_STD = (0, 0, 0.12)
DRAWER_YAW_NOISE_MEAN = (0, 0, 0.25)
DRAWER_YAW_CLIP_MIN = [-3.14, -3.14, 0]
DRAWER_YAW_CLIP_MAX = [3.14, 3.14, 0.57]

TABLETOP_YAW_NOISE_STD = (0, 0, 3.14)

DRAWER_Z_OFFSET = 0.3


def _place(obj, position=None, orientation=None, frame="scene"):
    """Teleport an object and clear residual velocity; positions default to scene frame."""
    obj.set_position_orientation(position=position, orientation=orientation, frame=frame)
    obj.keep_still()


def vb_pose(env: "RealmEnvironmentDynamic") -> None:
    if env.task_type == "push":
        _perturb_switch(env)
    else:
        _perturb_tabletop(env)

    settle(env)


def _perturb_switch(env: "RealmEnvironmentDynamic") -> None:
    """Nudge the light switch across its wall plane."""
    delta_z = np.random.uniform(-SWITCH_DZ_RANGE, SWITCH_DZ_RANGE)
    delta_xy = np.random.uniform(-SWITCH_DXY_RANGE, SWITCH_DXY_RANGE)
    for obj_cfg in env.cfg["objects"]:
        if obj_cfg["name"] == "electric_switch":
            obj = env.omnigibson_env.scene.object_registry("name", obj_cfg["name"])
            # Clone so repeated resets do not mutate the stored initial pose.
            new_pos = env.init_poses[obj._relative_prim_path]["pos"].clone()
            new_pos[2] += delta_z
            new_pos[0] += delta_xy
            _place(obj, position=new_pos, frame="world")


def _perturb_tabletop(env: "RealmEnvironmentDynamic") -> None:
    """Re-place the movable objects collision-free, then add rotation noise to the main objects."""
    backfill_object_cfgs(env.main_objects + env.distractors + env.target_objects,
                         env.cfg["objects"])

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        env.cfg["objects"],
        objects_to_skip=[obj.name for obj in env.distractors + env.target_objects],
        main_object_names=[],
        max_attempts_per_object=25000
    )

    for obj_cfg in env.cfg["objects"]:
        if env.task_type in ["open_drawer", "close_drawer"] and obj_cfg["name"] == "drawer":
            obj_cfg["position"][-1] -= DRAWER_Z_OFFSET
        _place(env.omnigibson_env.scene.object_registry("name", obj_cfg["name"]),
               position=obj_cfg["position"])

    for o in env.main_objects:
        if env.task_type in ["open_drawer", "close_drawer"]:
            drawer_cfg = next((c for c in env.cfg["objects"] if c["name"] == "drawer"), None)
            if drawer_cfg is None:
                raise RuntimeError(
                    "VB-POSE on a drawer task found no 'drawer' entry in cfg['objects']")
            current = drawer_cfg["orientation"] if "orientation" in drawer_cfg else [0, 0, 0, 1]
            new_rot = add_rotation_noise(current, DRAWER_YAW_NOISE_STD,
                                         DRAWER_YAW_CLIP_MIN, DRAWER_YAW_CLIP_MAX,
                                         DRAWER_YAW_NOISE_MEAN)
            _place(o, orientation=new_rot)
        else:
            current = o.get_position_orientation(frame="scene")[1]
            _place(o, orientation=add_rotation_noise(current, TABLETOP_YAW_NOISE_STD))

    env.reset_joints()
