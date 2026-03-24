from __future__ import annotations

import copy
import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.helpers import (
    get_non_colliding_positions_for_objects,
    get_droid_categories_by_theme,
    get_objects_by_names,
    get_default_objects_cfg,
)
from realm.environments.perturbations._helpers import replace_obj

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def v_sc(env: "RealmEnvironmentDynamic") -> None:
    # --------------- Translation ---------------
    og.sim.stop()

    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    num_mo_to = len(env.target_objects + env.main_objects)

    for scene_obj in env.target_objects + env.main_objects:
        for cfg in obj_cfgs:
            if cfg["name"] == scene_obj.name:
                if "position" not in cfg:
                    cfg["position"] = scene_obj.get_position_orientation()[0].tolist()
                if "bounding_box" not in cfg:
                    cfg["bounding_box"] = scene_obj.aabb_extent.tolist()

    env.cfg["objects"] = None
    num_distractors = len(obj_cfgs) - num_mo_to

    env.cfg["objects"] = get_non_colliding_positions_for_objects(
        xmin=env.spawn_bbox[0],
        xmax=env.spawn_bbox[1],
        ymin=env.spawn_bbox[2],
        ymax=env.spawn_bbox[3],
        z=env.spawn_bbox[4],
        obj_cfg=obj_cfgs[:num_mo_to + num_distractors],
        objects_to_skip=[obj.name for obj in env.target_objects + env.main_objects],
        main_object_names=[o["name"] for o in obj_cfgs[:num_mo_to]],
        maximum_dim=0.12,
    )

    env.distractors = [env.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in env.cfg["objects"][num_mo_to:]]

    # TODO: check if this works properly in the edge cases where it should trigger
    if num_distractors < len(env.distractors):
        for dist_cfg in env.cfg["objects"][num_mo_to + num_distractors:]:
            obj = env.omnigibson_env.scene.object_registry("name", dist_cfg["name"])
            env.omnigibson_env.scene.remove_object(obj)
        env.cfg["objects"] = env.cfg["objects"][:num_mo_to + num_distractors]

    # --------------- Set Position ---------------
    for obj in env.cfg["objects"]:
        env.omnigibson_env.scene.object_registry("name", obj["name"]).set_position(obj["position"])

    # --------------- Replace the objects models ---------------
    distractor_obj_cfgs = get_default_objects_cfg(env.omnigibson_env.scene, [obj.name for obj in env.distractors])
    distractor_objs = get_objects_by_names(env.omnigibson_env.scene, list(distractor_obj_cfgs.keys()))
    excluded_categories = [obj.category for obj in env.main_objects + env.target_objects]
    for distractor in distractor_objs:
        cat_dict = get_droid_categories_by_theme()
        t = [k for k, v in cat_dict.items() if any(distractor.category in c for c in v.values())]
        if t:
            cat_dict.pop(t[0])
        l = [o for v in cat_dict.values() for c in v.values() for o in c]
        l = [c for c in l if c not in excluded_categories]
        _, _ = replace_obj(env, distractor, included_categories=l, maximum_dim=0.12)

    og.sim.play()
    env.reset_joints()
    # fake rest to get to original pose after stopping sim
    for _ in range(30):
        env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
