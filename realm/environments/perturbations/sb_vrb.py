from __future__ import annotations

import copy
import random
import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og
from omnigibson.objects import DatasetObject
from realm.helpers import get_non_colliding_positions_for_objects
from realm.environments.utils import load_task_progressions
TASK_PROGRESSIONS = load_task_progressions()
from realm.environments.perturbations._helpers import replace_obj, sample_objects

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def sb_vrb(env: "RealmEnvironmentDynamic") -> None:
    compatibility_matrix = {
        "put": ["pick", "rotate", "stack"],
        "push": [], #["put", "pick", "rotate", "stack"],
        "pick": ["put", "rotate", "stack"],
        "rotate": ["put", "pick", "stack"],
        "stack": ["put", "pick", "rotate"],
        "open": ["close"],
        "close": ["open"]
    }

    available_task_types = compatibility_matrix[env.task_type]

    new_verb_for_task = random.choice(available_task_types)
    env.task_type = new_verb_for_task
    env.task_progression = TASK_PROGRESSIONS[env.task_type]

    included_categories = None
    if env.task_type == "put":
        included_categories = ["bowl", "wineglass"]

    if len(env.target_objects) == 0:
        nobj_cfg = sample_objects(env, num_objects=1, included_categories=included_categories)[0]
        env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
        nobj_cfg["name"] = "receiver"

        new_obj = DatasetObject(
            name="receiver",
            relative_prim_path="/receiver",
            category=nobj_cfg["category"],
            model=nobj_cfg["model"],
        )
        env.omnigibson_env.scene.add_object(new_obj)
        env.target_objects = [new_obj]

        bbox_center, bbox_orn, bbox_extent, bbox_center_in_frame = new_obj.get_base_aligned_bbox()
        nobj_cfg["bounding_box"] = bbox_center

        max_dim = np.max(bbox_extent.numpy())
        new_scale_factor = 0.185 / max_dim
        if new_scale_factor < 1.0:
            new_obj.scale = new_scale_factor
            nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor

        env.cfg["objects"].append(nobj_cfg)

        # --------------- Translation ---------------
        obj_cfgs = copy.deepcopy(env.cfg["objects"])
        num_mo_to = len(obj_cfgs) - 1

        for scene_obj in env.main_objects + env.distractors + env.target_objects:
            for cfg in obj_cfgs:
                if cfg["name"] == scene_obj.name:
                    if "position" not in cfg:
                        cfg["position"] = scene_obj.get_position_orientation()[0].tolist()
                    if "bounding_box" not in cfg:
                        cfg["bounding_box"] = scene_obj.aabb_extent.tolist()

        env.cfg["objects"] = get_non_colliding_positions_for_objects(
            xmin=env.spawn_bbox[0],
            xmax=env.spawn_bbox[1],
            ymin=env.spawn_bbox[2],
            ymax=env.spawn_bbox[3],
            z=env.spawn_bbox[4],
            obj_cfg=obj_cfgs,
            objects_to_skip=[obj.name for obj in env.main_objects + env.distractors],
            main_object_names=[o["name"] for o in obj_cfgs[:num_mo_to]],
        )

        pos = torch.tensor(env.cfg["objects"][-1]["position"])
        rot = torch.tensor(env.cfg["objects"][-1]["orientation"] if "orientation" in env.cfg["objects"][-1] else [0,0,0,1])
        new_obj.set_bbox_center_position_orientation(pos, rot)

        env.init_poses[new_obj._relative_prim_path] = {}
        env.init_poses[new_obj._relative_prim_path]["pos"] = pos
        env.init_poses[new_obj._relative_prim_path]["rot"] = rot

        # --------------- Set Position ---------------
        for obj in env.cfg["objects"]:
            env.omnigibson_env.scene.object_registry("name", obj["name"]).set_position(obj["position"])

    og.sim.step()

    if env.task_type in ["put", "stack"]:
        og.sim.stop()
        nobj, nobj_cfg = replace_obj(env, env.target_objects[0], included_categories=included_categories, maximum_dim=0.185)
        env.target_objects = [nobj]
        env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
        og.sim.play()
        # fake rest to get to original pose after stopping sim
        for _ in range(30):
            env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))

    if new_verb_for_task in ["rotate", "push", "pick", "open", "close"]:
        tmp = "pick up" if new_verb_for_task == "pick" else new_verb_for_task
        env.instruction = f"{tmp} the {env.cfg['instruction_obj_to_replace']}"
    elif new_verb_for_task == "stack":
        env.instruction = f"stack the {env.cfg['instruction_obj_to_replace']} on top of the {env.cfg['instruction_target_to_replace']}"
    elif new_verb_for_task == "put":
        env.instruction = f"put the {env.cfg['instruction_obj_to_replace']} into the {env.cfg['instruction_target_to_replace']}"
    else:
        raise NotImplementedError()
    env.instruction = env.instruction.replace("_", " ")
    og.log.info(f"New instruction: {env.instruction}")