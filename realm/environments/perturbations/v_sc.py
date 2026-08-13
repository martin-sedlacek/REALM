from __future__ import annotations

import copy
import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.categories import get_droid_categories_by_theme
from realm.placement import (
    get_non_colliding_positions_for_objects,
    get_objects_by_names,
    get_default_objects_cfg,
)
from realm.environments.perturbations._helpers import (
    after_play,
    replace_obj,
    settle,
    sim_play,
    sim_stop,
)

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def v_sc(env: "RealmEnvironmentDynamic") -> None:
    # --------------- Translation ---------------
    # sim_stop/sim_play rather than og.sim.stop()/play(): those are global, so in a vector env
    # RealmVectorEnvironment.reset() does ONE cycle for every member and these no-op. V-SC genuinely
    # needs the stopped sim -- it calls scene.remove_object() and replace_obj().
    sim_stop(env)

    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    num_mo_to = len(env.target_objects + env.main_objects)

    for scene_obj in env.target_objects + env.main_objects:
        for cfg in obj_cfgs:
            if cfg["name"] == scene_obj.name:
                if "position" not in cfg:
                    # Scene frame: this backfills the same cfg["position"] that
                    # get_non_colliding_positions_for_objects rewrites from the scene-relative
                    # spawn_bbox below, so it must be in that frame. See vb_pose._place for the
                    # full write-up -- reading this in world frame agrees with scene frame only
                    # for scene 0, whose origin IS the world origin, so it is invisible single-env
                    # and silently wrong for every other member of a vector env.
                    cfg["position"] = scene_obj.get_position_orientation(frame="scene")[0].tolist()
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
    # frame="scene" because obj["position"] came from the scene-relative spawn_bbox above. NOTE the
    # old call was set_position(), which is deprecated AND world-frame-only (it forwards to
    # set_position_orientation(position=...) with no frame), so there was no way to express the
    # right frame through it.
    for obj in env.cfg["objects"]:
        env.omnigibson_env.scene.object_registry("name", obj["name"]).set_position_orientation(
            position=obj["position"], frame="scene")

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

    sim_play(env)

    # update_initial_file() is REQUIRED here, not an optimisation, and V-SC was missing it while
    # vsb_nobj/vb_mobj both have it. Without it the scene's initial file still describes the objects
    # this perturbation just replaced, so the NEXT scene.reset() has to remove the new ones and re-add
    # the old ones. Those re-added objects are uninitialised until the sim steps, and
    # og.sim.dump_state() -- which batch_remove_objects calls -- iterates EVERY scene
    # (simulator.py:2093). So in a vector env member 0's half-restored scene made member 1's reset
    # assert "Object must be initialized before dumping state!". Single-env never saw it: with one
    # scene there is no sibling to trip over. Capturing the post-perturbation state as the new
    # baseline means restore() has nothing to add or remove at all.
    # DELIBERATELY vec-only. Single-env V-SC has never called update_initial_file() and works, so
    # adding it there would be an unverified change to a working path. It is only needed because of
    # the multi-scene coupling described above.
    #
    # This used to assert "Object must be initialized before dumping state!" here even though the
    # shared og.sim.play() and an og.sim.step() had both already run. The cause was NOT that objects
    # added while stopped miss the simulator's init queue -- they do reach it -- but that
    # Simulator._pre_remove_object() prunes that GLOBAL queue by NAME ALONE, and every member of a
    # vector env is built from the same task config, so member 1's remove_object("corkscrew")
    # evicted member 0's freshly-added "corkscrew" and nothing ever initialised it. Fixed in
    # RealmVectorEnvironment._initialize_evicted_objects(), which re-queues the orphans right after
    # the shared play(); read that docstring for the full write-up. Nothing is needed here.
    def _post_play():
        og.sim.step()
        if env.in_vec_env:
            env.omnigibson_env.scene.update_initial_file()
        env.reset_joints()

    after_play(env, _post_play)
    # Let the re-placed objects come to rest. No-op in a vector env, where the shared settle runs
    # once for all members instead of stepping the global sim 30 times per member.
    settle(env)
