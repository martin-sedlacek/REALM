"""Drawing a random object out of the dataset, and swapping one into a live scene.

Used by every perturbation that changes WHICH objects a scene contains -- V-SC (distractors),
VSB-NOBJ and VB-MOBJ (the main object), SB-VRB (the receiver) -- and by env_config.py when a task
asks for sampled distractors at build time.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_models

from realm.categories import get_non_droid_categories

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


# bottom_cabinet models with exactly the drawer layout the drawer tasks assume. Commented-out
# entries are models that load but are unusable for the task, and the reason each was dropped.
DRAWER_CABINET_MODELS = [
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


def sample_objects(num_objects=3, included_categories=None, excluded_categories=None):
    """@num_objects object configs drawn without replacement from the installed models."""
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
    """Swap @obj for a freshly sampled one at the same name, prim path and initial pose.

    Removing and adding an object needs a STOPPED sim, so every caller is in NEEDS_STOPPED_SIM.
    Returns (new object, its config).
    """
    obj_name = obj.name

    env.omnigibson_env.scene.remove_object(obj)

    if not (included_categories is None) and len(included_categories) == 1 and "bottom_cabinet" in included_categories:
        sampled_idx = np.random.choice(len(DRAWER_CABINET_MODELS), size=1, replace=False)[0]
        nobj_cfg = {
            "type": "DatasetObject",
            "name": obj_name,
            "category": "bottom_cabinet",
            "model": DRAWER_CABINET_MODELS[sampled_idx],
        }
    else:
        nobj_cfg = sample_objects(num_objects=1, included_categories=included_categories)[0]

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
    # EXTENT, not centre. "bounding_box" is a SIZE everywhere it is consumed -- the task YAMLs write
    # [0.20, 0.20, 0.07] and get_non_colliding_positions_for_objects reads cfg["bounding_box"][0] / 2
    # as a half-width -- while get_base_aligned_bbox returns the centre in WORLD frame. The scale
    # multiply just below, which shrinks the value along with the object, only makes sense for a size.
    #
    # LATENT rather than live: no caller reads this key today (v_sc.py discards the cfg, vsb_nobj.py
    # and sb_vrb.py read only "category"/"model"), and the dict is freshly built by sample_objects()
    # or by the literal above, so it is never aliased into env.cfg["objects"] either. Fixed rather
    # than merely flagged because the identical line in sb_vrb.py DID feed placement and was
    # measured: a world-frame centre reads ~0 in scene 0, whose origin is the world origin, but
    # ~25 m per tile in every other member of a vector env, so the "half-width" came out ~12.5 m, no
    # candidate position could clear it, and the receiver was dropped ~12 m off the table.
    nobj_cfg["bounding_box"] = bbox_extent

    max_dim = np.max(bbox_extent.numpy())
    new_scale_factor = maximum_dim / max_dim
    if new_scale_factor < 1.0:
        new_obj.scale = new_scale_factor
        nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor
    nobj_cfg["fixed_base"] = fixed_base

    return new_obj, nobj_cfg
