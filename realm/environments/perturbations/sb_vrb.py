
from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

import torch

import omnigibson as og
from omnigibson.objects import DatasetObject

from realm.environments.perturbations._helpers import (
    backfill_object_cfgs,
    rebase_after_play,
    set_scene_positions,
    settle,
    sim_play,
    sim_step,
    sim_stop,
)
from realm.environments.perturbations.object_sampling import (
    replace_obj,
    rescale_to_max_dim,
    sample_objects,
)
from realm.config.shared import (
    COMPATIBILITY_MATRIX,
    UNSUPPORTED_BY_PERTURBATION,
    VERB_PHRASE,
)
from realm.environments.utils import load_task_progressions
from realm.placement import place_within

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

TASK_PROGRESSIONS = load_task_progressions()

RECEIVER_MAX_DIM = 0.185


def sb_vrb(env: "RealmEnvironmentDynamic") -> None:
    if env.task_type in UNSUPPORTED_BY_PERTURBATION["SB-VRB"]:
        raise NotImplementedError(
            f"SB-VRB does not support task_type {env.task_type!r}: the drawer configs declare "
            f"target_objects: [], so the perturbation would inject a 'receiver' object that has no "
            f"placeable position in these scenes and gets dropped from the air. Deliberate refusal, "
            f"not an unimplemented branch -- do not 'fix' this by making it a no-op."
        )

    # Missing means a configuration bug; an empty list is an intentional no-op.
    if env.task_type not in COMPATIBILITY_MATRIX:
        raise KeyError(
            f"SB-VRB: task_type {env.task_type!r} is not in COMPATIBILITY_MATRIX "
            f"(known: {sorted(COMPATIBILITY_MATRIX)}). Add it -- with an empty list if the "
            f"perturbation genuinely does not apply -- rather than leaving it absent."
        )
    available_task_types = COMPATIBILITY_MATRIX[env.task_type]
    if not available_task_types:
        og.log.info(
            f"SB-VRB: no-op, task_type {env.task_type!r} is a deliberate opt-out (empty candidate list)"
        )
        return

    new_task_type = _draw_new_task_type(env, available_task_types)

    included_categories = None
    if env.task_type == "put":
        included_categories = ["bowl", "wineglass"]

    if len(env.target_objects) == 0:
        _spawn_receiver(env, included_categories)

    sim_step(env)

    if env.task_type in ["put", "stack"]:
        _swap_target(env, included_categories)

    rebase_after_play(env, vec_only_rebase=True)

    _rebuild_instruction(env, new_task_type)


def _draw_new_task_type(env: "RealmEnvironmentDynamic", available_task_types) -> str:

    new_task_type = random.choice(available_task_types)
    env.task_type = new_task_type
    # Progression dictionaries are mutated per environment.
    env.task_progression = copy.deepcopy(TASK_PROGRESSIONS[env.task_type])
    return new_task_type


def _spawn_receiver(env: "RealmEnvironmentDynamic", included_categories) -> None:

    sampled = sample_objects(num_objects=1, included_categories=included_categories)
    if not sampled:
        raise RuntimeError(
            f"SB-VRB could not sample a receiver: no installed object model matches "
            f"included_categories={included_categories}")
    nobj_cfg = sampled[0]
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

    rescale_to_max_dim(new_obj, nobj_cfg, RECEIVER_MAX_DIM)

    env.cfg["objects"].append(nobj_cfg)

    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    n_non_receiver = len(obj_cfgs) - 1

    backfill_object_cfgs(env.main_objects + env.distractors + env.target_objects, obj_cfgs)

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        obj_cfgs,
        objects_to_skip=[obj.name for obj in env.main_objects + env.distractors],
        main_object_names=[o["name"] for o in obj_cfgs[:n_non_receiver]],
    )

    _record_receiver_pose(env, new_obj)

    set_scene_positions(env, env.cfg["objects"])


def _record_receiver_pose(env: "RealmEnvironmentDynamic", new_obj) -> None:

    receiver_cfg = env.cfg["objects"][-1]
    pos = torch.tensor(receiver_cfg["position"])
    rot = torch.tensor(receiver_cfg["orientation"] if "orientation" in receiver_cfg else [0, 0, 0, 1])
    pos_world, rot_world = env.omnigibson_env.scene.convert_scene_relative_pose_to_world(pos, rot)
    new_obj.set_bbox_center_position_orientation(pos_world, rot_world)
    env.init_poses[new_obj._relative_prim_path] = {"pos": pos_world, "rot": rot_world}


def _swap_target(env: "RealmEnvironmentDynamic", included_categories) -> None:

    sim_stop(env)
    nobj, nobj_cfg = replace_obj(env, env.target_objects[0],
                                 included_categories=included_categories,
                                 maximum_dim=RECEIVER_MAX_DIM)
    env.target_objects = [nobj]
    env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
    sim_play(env)
    settle(env)


def _rebuild_instruction(env: "RealmEnvironmentDynamic", new_task_type: str) -> None:

    if new_task_type in ("rotate", "push", "pick", "open_drawer", "close_drawer"):
        env.instruction = f"{VERB_PHRASE[new_task_type]} the {env.cfg['instruction_obj_to_replace']}"
    elif new_task_type == "stack":
        env.instruction = f"stack the {env.cfg['instruction_obj_to_replace']} on top of the {env.cfg['instruction_target_to_replace']}"
    elif new_task_type == "put":
        env.instruction = f"put the {env.cfg['instruction_obj_to_replace']} into the {env.cfg['instruction_target_to_replace']}"
    else:
        raise NotImplementedError(
            f"SB-VRB: no instruction phrasing for task_type {new_task_type!r}. It is reachable "
            f"from COMPATIBILITY_MATRIX, so add a branch here rather than widening the matrix alone."
        )
    env.instruction = env.instruction.replace("_", " ")
    og.log.info(f"New instruction: {env.instruction}")
