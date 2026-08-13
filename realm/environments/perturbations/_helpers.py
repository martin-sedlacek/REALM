from __future__ import annotations

import os
import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og

from realm.categories import get_non_droid_categories
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_models

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


# ============================== [SIM STATE, SHARED ACROSS MEMBERS] ==============================
# og.sim.stop()/play() and og.sim.step() are GLOBAL: they act on every scene in the simulator, not
# on one member. REALM applies perturbations per member inside reset(), so a perturbation that calls
# them directly does N times the work in a vector env AND disturbs its siblings mid-reset. Measured
# (job 190555, VB-POSE Vec=4): three of four members lost their main object from the contact view,
# scored TP=0.00 with zero collisions, and the job still exited 0.
#
# So perturbations must never call them directly. They call the wrappers below, which no-op when the
# member belongs to a vector env; RealmVectorEnvironment.reset() then performs ONE stop/play cycle
# and ONE settle for all members together. Single-env behaviour is byte-identical to before.
SETTLE_STEPS = 30

# Perturbations that genuinely cannot run on a live sim, because they add or remove objects --
# OmniGibson requires a stopped sim for that (see the note in realm/placement.py). Everything else
# only writes poses, which works fine while playing, so it must NOT be in here: VB-POSE and V-VIEW
# were moved off the stopped path precisely because a pose write never needed it.
NEEDS_STOPPED_SIM = frozenset({"V-SC", "VB-MOBJ", "VSB-NOBJ", "SB-VRB"})


def sim_stop(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.stop(), unless a vector env is batching the cycle for every member."""
    if not env.in_vec_env:
        og.sim.stop()


def sim_play(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.play(), unless a vector env is batching the cycle for every member."""
    if not env.in_vec_env:
        og.sim.play()


def sim_step(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.step(), unless a vector env is batching -- see after_play for why it must be deferred."""
    if not env.in_vec_env:
        og.sim.step()


def after_play(env: "RealmEnvironmentDynamic", fn) -> None:
    """Run @fn once the simulator is playing again.

    Single env: og.sim.play() has already happened by the time a perturbation calls this, so run it
    now and behaviour is unchanged. Vector env: sim_play() was a no-op and the shared play does not
    happen until every member's perturbation has run, so @fn would execute against a STOPPED sim --
    og.Environment.step() asserts is_playing(), and update_initial_file() would capture the wrong
    state. Defer it instead; RealmVectorEnvironment.reset() drains these right after its single
    og.sim.play().
    """
    if env.in_vec_env:
        env.deferred_post_play.append(fn)
    else:
        fn()


def settle_action(env: "RealmEnvironmentDynamic"):
    """The hold-still action used to let a perturbed scene come to rest."""
    return np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1]))))


def settle(env: "RealmEnvironmentDynamic", steps: int = SETTLE_STEPS) -> None:
    """Let this member's scene come to rest after a perturbation.

    In a vector env this only RAISES A FLAG: og.sim.step() advances EVERY scene, so running the loop
    per member would step the shared sim N*steps times and, worse, advance the other members while
    feeding them no action at all. RealmVectorEnvironment.reset() reads the flag and runs the
    equivalent loop once for all members. The flag matters because settling is not unconditional --
    Default perturbs nothing and never calls this, and a vector env must not start settling resets
    that a single env would not.
    """
    if env.in_vec_env:
        env.wants_settle = True
        return
    # Nothing reads a camera here, so skip the render pass on each step. gm.HEADLESS does NOT do
    # this -- og.sim.step() renders every call regardless; only this context actually suppresses it.
    # Object states and contact caching still update normally inside it.
    with og.sim.render_on_step(False):
        for _ in range(steps):
            env.omnigibson_env.step(settle_action(env))


def apply_cached_semantic_perturbations(env: "RealmEnvironmentDynamic", perturbation: str) -> None:
    tmp = env.cfg["cached_semantic_perturbations"][perturbation]
    idx = np.random.randint(0, len(tmp))
    env.instruction = tmp[idx]


def sample_objects(env: "RealmEnvironmentDynamic", num_objects=3, included_categories=None, excluded_categories=None):
    assert not (included_categories is not None and excluded_categories is not None)

    # TODO: this can be pre-computed once, no need to parse the whole thing every call
    available_object_paths = []
    whitelisted_categories = get_non_droid_categories()

    if included_categories is not None:
        whitelisted_categories = included_categories
    elif excluded_categories is not None:
        for cat in excluded_categories:
            if cat in whitelisted_categories:
                whitelisted_categories.remove(cat)

    for model_path in get_all_object_models():
        if os.path.exists(model_path):
            category = model_path.split("/")[-2]
            if category in whitelisted_categories:
                available_object_paths.append(model_path)

    if not available_object_paths:
        return []

    if len(available_object_paths) < num_objects:
        import omnigibson as og
        og.log.info(
            f"Warning: Only {len(available_object_paths)} suitable objects found, less than requested {num_objects}.")
        num_objects = len(available_object_paths)

    sampled_indices = np.random.choice(len(available_object_paths), size=num_objects, replace=False)
    sampled_objects = []
    for i in sampled_indices:
        category = available_object_paths[i].split("/")[-2]
        model_id = available_object_paths[i].split("/")[-1]
        name = f"distractor_{i}"
        obj_cfg = {
            "type": "DatasetObject",
            "name": name,
            "category": category,
            "model": model_id,
        }
        sampled_objects.append(obj_cfg)

    return sampled_objects


def replace_obj(env: "RealmEnvironmentDynamic", obj: DatasetObject, included_categories=None, maximum_dim=0.2, fixed_base=False, preserve_ori=True):
    obj_name = obj.name

    env.omnigibson_env.scene.remove_object(obj)

    if not (included_categories is None) and len(included_categories) == 1 and "bottom_cabinet" in included_categories:
        bottom_cabinet_models = [
            "bamfsz",
            "dsbcxl",
            "ilofmb",
            # "jhymlr", two top drawers
            "lhucjo",
            "mbmbpa",
            "nddvba",
            "immwzb",
            "pkdnbu",
            "plccav",
            #"pllcur", opens bottom for some reason
            "rntwkg",
            # "ttmejh", not leveled
            "slgzfc",
            "rvpunw",
            "wesxdp",
            "rhdbzv"
        ]
        sampled_idx = np.random.choice(len(bottom_cabinet_models), size=1, replace=False)[0]
        nobj_cfg = {
            "type": "DatasetObject",
            "name": obj_name,
            "category": "bottom_cabinet",
            "model": bottom_cabinet_models[sampled_idx],
        }
    else:
        nobj_cfg = sample_objects(env, num_objects=1, included_categories=included_categories)[0]

    new_obj = DatasetObject(
        name=obj_name,
        relative_prim_path=obj._relative_prim_path,
        category=nobj_cfg["category"],
        model=nobj_cfg["model"],
        fixed_base=fixed_base
    )
    env.omnigibson_env.scene.add_object(new_obj)

    if preserve_ori:
        new_obj.set_bbox_center_position_orientation(torch.tensor(env.init_poses[new_obj._relative_prim_path]["pos"]),
                                                     torch.tensor(env.init_poses[new_obj._relative_prim_path]["rot"]))
    else:
        new_obj.set_bbox_center_position_orientation(torch.tensor(env.init_poses[new_obj._relative_prim_path]["pos"]),
                                                     torch.tensor([0, 0, 0, 1]))

    bbox_center, bbox_orn, bbox_extent, bbox_center_in_frame = new_obj.get_base_aligned_bbox()
    nobj_cfg["bounding_box"] = bbox_center

    max_dim = np.max(bbox_extent.numpy())
    new_scale_factor = maximum_dim / max_dim
    if new_scale_factor < 1.0:
        new_obj.scale = new_scale_factor
        nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor
    nobj_cfg["fixed_base"] = fixed_base

    return new_obj, nobj_cfg
