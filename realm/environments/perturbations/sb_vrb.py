from __future__ import annotations

import copy
import random
import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og
from omnigibson.objects import DatasetObject
from realm.placement import get_non_colliding_positions_for_objects
from realm.environments.utils import load_task_progressions
TASK_PROGRESSIONS = load_task_progressions()
from realm.environments.perturbations._helpers import (
    after_play,
    replace_obj,
    sample_objects,
    settle,
    sim_play,
    sim_step,
    sim_stop,
)

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
    # deepcopy is load-bearing, for the same reason as in env_base.py: TASK_PROGRESSIONS is built ONCE
    # at module import and recompute_task_progression MUTATES this dict in place
    # (self.task_progression[stage] = True), short-circuiting on stages already marked True. Assigning
    # it directly gave every member that drew the same new verb ONE SHARED progression dict, so one
    # member grasping marked GRASP done for all of them. That is precisely the bug that invalidated
    # the SR 0.960 run; SB-VRB would have reintroduced it after env_base.py was fixed.
    env.task_progression = copy.deepcopy(TASK_PROGRESSIONS[env.task_type])

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
                        # Scene frame, matching the scene-relative spawn_bbox that
                        # get_non_colliding_positions_for_objects rewrites this from just below.
                        # See vb_pose._place: world frame agrees only for scene 0.
                        cfg["position"] = scene_obj.get_position_orientation(frame="scene")[0].tolist()
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
        # pos/rot are SCENE-relative (they come from the scene-relative spawn_bbox above), but
        # set_bbox_center_position_orientation is world-frame by construction -- unlike
        # set_position_orientation it takes no frame argument at all -- so convert explicitly.
        # In scene 0 this conversion is the identity, which is why the world-frame version was
        # correct single-env and wrong for every other member of a vector env.
        pos_world, rot_world = env.omnigibson_env.scene.convert_scene_relative_pose_to_world(pos, rot)
        new_obj.set_bbox_center_position_orientation(pos_world, rot_world)

        # env.init_poses holds WORLD poses everywhere else (env_dynamic.py builds it from
        # get_position_orientation()'s world default, and _helpers/env_dynamic restore from it with
        # the world-frame bbox setter). Store the world pose here so the frames cannot disagree --
        # this used to store the scene-relative one, which would restore to the wrong place.
        env.init_poses[new_obj._relative_prim_path] = {}
        env.init_poses[new_obj._relative_prim_path]["pos"] = pos_world
        env.init_poses[new_obj._relative_prim_path]["rot"] = rot_world

        # --------------- Set Position ---------------
        # frame="scene" to match obj["position"]'s frame; the old set_position() was deprecated and
        # world-frame-only, with no way to express the correct frame.
        for obj in env.cfg["objects"]:
            env.omnigibson_env.scene.object_registry("name", obj["name"]).set_position_orientation(
                position=obj["position"], frame="scene")

    # sim_step: og.sim.step() asserts a playing sim. Single-env that holds here; in a vector env the
    # shared stop is already in force for the whole perturbation, so this no-ops and the shared
    # settle after the shared play does the equivalent work.
    sim_step(env)

    if env.task_type in ["put", "stack"]:
        # Genuinely needs the stopped sim: replace_obj removes and adds an object. No-ops in a
        # vector env, where RealmVectorEnvironment.reset() has already stopped once for all members.
        sim_stop(env)
        nobj, nobj_cfg = replace_obj(env, env.target_objects[0], included_categories=included_categories, maximum_dim=0.185)
        env.target_objects = [nobj]
        env.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
        sim_play(env)
        # Let the replaced target come to rest; no-op in a vector env, which settles once for all.
        settle(env)

    # SB-VRB adds a "receiver" and/or replaces the target, so like V-SC and vsb_nobj/vb_mobj it MUST
    # rebase the scene's initial file. Without it the next scene.reset() has to undo those object
    # changes, and og.sim.dump_state() -- which batch_remove_objects calls -- walks EVERY scene
    # (simulator.py:2093), so one member's half-restored scene makes a sibling's reset assert
    # "Object must be initialized before dumping state!". V-SC failed exactly that way before this
    # call was added to it; SB-VRB was the only add/replace perturbation still missing it.
    # vec-only, for the same reason as in v_sc.py: single-env SB-VRB has never called this and works,
    # so adding it there would be an unverified change to a working path.
    #
    # UNTESTED and expected to hit the same wall as V-SC: update_initial_file() -> scene.dump_state()
    # asserts "Object must be initialized before dumping state!" for objects added while the sim was
    # stopped. See the long note in v_sc.py; whatever fixes it there applies here unchanged.
    def _post_play():
        og.sim.step()
        if env.in_vec_env:
            env.omnigibson_env.scene.update_initial_file()
        env.reset_joints()

    after_play(env, _post_play)

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