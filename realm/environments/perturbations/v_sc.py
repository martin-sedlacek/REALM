"""V-SC: re-place the movable objects and swap every distractor's model for a random one.

What one call mutates on @env: ``cfg["objects"]`` (fresh positions; the list is rebuilt by the
placement pass), ``distractors`` (re-bound to the live objects), and the scene itself (each
distractor removed and re-added as a different model via replace_obj). Main and target objects
keep their authored poses; only distractors move and change identity. Requires ``env.spawn_bbox``
(a task with a spawn region).
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from realm.categories import droid_categories_excluding_theme
from realm.environments.perturbations._helpers import (
    backfill_object_cfgs,
    rebase_after_play,
    set_scene_positions,
    settle,
    sim_play,
    sim_stop,
)
from realm.environments.perturbations.object_sampling import replace_obj
from realm.placement import (
    get_default_objects_cfg,
    get_objects_by_names,
    place_within,
)

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Largest bbox dimension (metres) a distractor may have after the swap.
DISTRACTOR_MAX_DIM = 0.12


def v_sc(env: "RealmEnvironmentDynamic") -> None:
    # sim_stop/sim_play rather than og.sim.stop()/play(): those are global, so in a vector env
    # RealmVectorEnvironment.reset() does ONE cycle for every member and these no-op. V-SC genuinely
    # needs the stopped sim -- it calls replace_obj(), which removes and adds objects.
    sim_stop(env)

    # --------------- Translation ---------------
    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    num_mo_to = len(env.target_objects + env.main_objects)

    backfill_object_cfgs(env.target_objects + env.main_objects, obj_cfgs)

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        obj_cfgs,
        objects_to_skip=[obj.name for obj in env.target_objects + env.main_objects],
        main_object_names=[o["name"] for o in obj_cfgs[:num_mo_to]],
        maximum_dim=DISTRACTOR_MAX_DIM,
    )

    env.distractors = [env.omnigibson_env.scene.object_registry("name", dist["name"])
                       for dist in env.cfg["objects"][num_mo_to:]]

    # --------------- Set Position ---------------
    set_scene_positions(env, env.cfg["objects"])

    _swap_distractor_models(env)

    sim_play(env)

    # The rebase is REQUIRED, not an optimisation, for a perturbation that replaces objects -- see
    # rebase_after_play for the vector-env failure mode it prevents (and vec_init_queue.py for the
    # related init-queue fix). vec-only because single-env V-SC has never rebased and works, so
    # rebasing there would be an unverified change to a working path.
    rebase_after_play(env, vec_only_rebase=True)

    # Let the re-placed objects come to rest. No-op in a vector env, where the shared settle runs
    # once for all members instead of stepping the global sim 30 times per member.
    settle(env)


def _swap_distractor_models(env: "RealmEnvironmentDynamic") -> None:

    distractor_obj_cfgs = get_default_objects_cfg(env.omnigibson_env.scene,
                                                  [obj.name for obj in env.distractors])
    distractor_objs = get_objects_by_names(env.omnigibson_env.scene,
                                           list(distractor_obj_cfgs.keys()))
    excluded_categories = [obj.category for obj in env.main_objects + env.target_objects]
    for distractor in distractor_objs:
        candidates = [c for c in droid_categories_excluding_theme(distractor.category)
                      if c not in excluded_categories]
        replace_obj(env, distractor, included_categories=candidates,
                    maximum_dim=DISTRACTOR_MAX_DIM)
