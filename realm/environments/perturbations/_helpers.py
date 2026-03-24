from __future__ import annotations

import os
import numpy as np
import torch
from typing import TYPE_CHECKING

from realm.helpers import get_non_droid_categories
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_models

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


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
