"""V-SC: re-place the movable objects and swap every distractor's model for a random one.

What one call mutates on @env: ``cfg["objects"]`` (fresh positions; the list is rebuilt by the
placement pass), ``distractors`` (re-bound to the live objects), and the scene itself (each
distractor removed and re-added as a different model via replace_obj). Main and target objects
keep their authored poses, and so do the task's ``immutables`` -- authored fixtures such as a
support surface or a light, which ride in ``env.distractors`` but are neither re-placed nor
re-modelled here. Only declared distractors move and change identity. Requires ``env.spawn_bbox``
(a task with a spawn region).
"""
from __future__ import annotations
from realm.config.shared import DISTRACTOR_MAX_DIM

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
from realm.environments.foam_ball_reset import refresh_foam_ball_cfg_positions
from realm.environments.perturbations.object_sampling import replace_obj
from realm.placement import (
    get_default_objects_cfg,
    get_objects_by_names,
    place_within,
)

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Largest bbox dimension (metres) a distractor may have after the swap.


def v_sc(env: "RealmEnvironmentDynamic") -> None:
    # sim_stop/sim_play rather than og.sim.stop()/play(): those are global, so in a vector env
    # RealmVectorEnvironment.reset() does ONE cycle for every member and these no-op. V-SC genuinely
    # needs the stopped sim -- it calls replace_obj(), which removes and adds objects.
    sim_stop(env)

    # --------------- Translation ---------------
    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    num_mo_to = len(env.target_objects + env.main_objects)

    backfill_object_cfgs(env.target_objects + env.main_objects, obj_cfgs)

    # Immutables are pinned alongside main/target, exactly as the initial build pins them
    # (env_config._apply_object_cfg passes EVERY authored object as main_object_names). The slice
    # below covers main+target only, so without the immutable names a task's authored fixtures fall
    # through to placement's random-placement bucket: open_drawer's breakfast_table_support is a
    # 1.0x1.0 m footprint that cannot fit the 0.70x0.80 m Drawers_Near_Table spawn region at all,
    # so every attempt collided and it was dropped from DROP_HEIGHT, while light_over_table -- small
    # enough to fit -- was silently relocated from 1.5 m above the table down to tabletop z.
    #
    # Pinning, not skipping: objects_to_skip RESCALES cfg["bounding_box"] to maximum_dim, and that
    # key is what DatasetObject scales the asset by, so skipping would shrink the real fixture on
    # the next build. main_object_names only reads position and bounding_box.
    pinned_names = ([o["name"] for o in obj_cfgs[:num_mo_to]]
                    + list(env.immutable_names) + list(env.foam_ball_names))

    # pour_proxy's foam balls sit INSIDE the source bottle, and pinning alone would still let
    # set_scene_positions below write their provisional spawn column back over that.
    refresh_foam_ball_cfg_positions(env, obj_cfgs)

    env.cfg["objects"] = place_within(
        env.spawn_bbox,
        obj_cfgs,
        objects_to_skip=[obj.name for obj in env.target_objects + env.main_objects],
        main_object_names=pinned_names,
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
    # related init-queue fix).
    #
    # It runs in BOTH paths. It was vec-only on the grounds that single-env V-SC had never rebased
    # and worked -- true only at repeats=1, the one shape that never resets a second time. At
    # repeats>1 the single-env path restored a scene file describing the PRE-swap distractors onto
    # the post-swap objects and died in scene.reset() with
    #     KeyError: 'joint_pos'
    # from entity_prim._load_state -- the saved state has no joints, the replacement is articulated
    # -- followed by a segfault. Measured 2026-08-28 on task 0 with --repeats 3.
    #
    # This matches vb_mobj and vsb_nobj, the other two object-replacing perturbations, which have
    # always passed False. sb_vrb.py replaces objects and still passes True, so it carries the same
    # latent defect; not touched here because it is a separate perturbation and a separate call.
    #
    # KNOWN ISSUE -- MOVES NUMBERS, needs the VERSION gate: rebasing makes repeats 2..N start from
    # the perturbed scene rather than the authored one, so single-env V-SC numbers recorded before
    # this are not comparable. The vector path has always behaved this way.
    rebase_after_play(env, vec_only_rebase=False)

    # Let the re-placed objects come to rest. No-op in a vector env, where the shared settle runs
    # once for all members instead of stepping the global sim 30 times per member.
    settle(env)


def _swap_distractor_models(env: "RealmEnvironmentDynamic") -> None:

    # env.distractors carries the task's immutables (env_config._apply_object_cfg folds them in)
    # and pour_proxy's foam balls (RealmEnvironmentDynamic injects them there for scene handles),
    # and replace_obj would swap a support surface, a light or a ball being poured for a random
    # <=DISTRACTOR_MAX_DIM object. Swap only what the task actually declared as clutter.
    protected = set(env.immutable_names) | set(env.foam_ball_names)
    swappable = [obj for obj in env.distractors if obj.name not in protected]
    distractor_obj_cfgs = get_default_objects_cfg(env.omnigibson_env.scene,
                                                  [obj.name for obj in swappable])
    distractor_objs = get_objects_by_names(env.omnigibson_env.scene,
                                           list(distractor_obj_cfgs.keys()))
    excluded_categories = [obj.category for obj in env.main_objects + env.target_objects]
    for distractor in distractor_objs:
        candidates = [c for c in droid_categories_excluding_theme(distractor.category)
                      if c not in excluded_categories]
        replace_obj(env, distractor, included_categories=candidates,
                    maximum_dim=DISTRACTOR_MAX_DIM)
