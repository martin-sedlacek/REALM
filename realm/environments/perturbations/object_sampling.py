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


# bottom_cabinet models the drawer tasks may draw from. The commented-out entries record models
# that were tried and rejected, each with its reason.
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


#: The dataset DatasetObject.get_usd_path loads from. get_all_object_models() globs every dataset
#: directory under DATA_PATH, so candidates from any other one cannot be loaded.
LOADABLE_DATASET = "behavior-1k-assets"


def sample_objects(num_objects=3, included_categories=None, excluded_categories=None):
    """@num_objects object configs drawn without replacement from the installed models.

    May return FEWER than requested -- including [] -- when not enough installed models match the
    category filter; a warning is logged but nothing raises, so callers that index the result must
    guard for empty. At most one of @included_categories / @excluded_categories may be given; both
    filter against the non-DROID catalogue's category names.
    """
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

    # get_all_object_models() globs `<DATA_PATH>/*/objects/*/*` -- EVERY dataset directory under
    # /data, not just the one objects load from. DatasetObject.get_usd_path pins loading to
    # `behavior-1k-assets`, so a leftover dataset dir (og_dataset, the 1.1.1-era tree) contributes
    # candidates whose USD does not exist at the path the loader will use, and the swap dies with
    #     FileNotFoundError: .../objects/jar/mefezc/usd/mefezc.encrypted.usd
    # deep inside scene.add_object(), after the object it replaces has already been removed.
    # `jar/mefezc` is real in og_dataset and absent from behavior-1k-assets, which renamed the
    # category to hingeless_jar.
    #
    # So the guard is on the file the LOADER will open, not on the sampled directory, and a
    # candidate that fails it is dropped from the pool rather than drawn and raised on. Dedup by
    # (category, model) because two dataset dirs can offer the same pair, which would otherwise
    # weight that object twice in the draw. Order stays deterministic: get_all_object_models()
    # returns sorted paths and first occurrence wins.
    seen = set()
    for model_path in get_all_object_models():
        if not os.path.exists(model_path):
            continue
        category = model_path.split("/")[-2]
        model_id = model_path.split("/")[-1]
        if category not in whitelisted_categories or (category, model_id) in seen:
            continue
        # The loader opens <DATA_PATH>/behavior-1k-assets/objects/<cat>/<model>/usd/
        # <model>.encrypted.usd, so a candidate is only usable if it lives in THAT dataset and
        # carries that file. og_dataset/jar/mefezc has its own encrypted usd, so testing the
        # sampled path alone is not enough -- the dataset root is what decides.
        if f"/{LOADABLE_DATASET}/objects/" not in model_path:
            continue
        if not os.path.exists(os.path.join(model_path, "usd", f"{model_id}.encrypted.usd")):
            continue
        seen.add((category, model_id))
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


def rescale_to_max_dim(obj, cfg, maximum_dim):
    """Shrink @obj so its largest bbox dimension is at most @maximum_dim, recording the extent.

    Writes cfg["bounding_box"] as the (possibly shrunk) base-aligned bbox EXTENT -- a size, which
    is what every consumer of that key expects: the task YAMLs write sizes, and the placement pass
    reads element/2 as a half-width. Feeding it get_base_aligned_bbox's CENTER instead was measured
    to break vector-env placement -- the center is world-frame, ~25 m per scene tile, so the
    "half-width" came out ~12.5 m and nothing could ever be placed. An object already small enough
    is left alone.
    """
    _, _, bbox_extent, _ = obj.get_base_aligned_bbox()
    cfg["bounding_box"] = bbox_extent

    max_dim = np.max(bbox_extent.numpy())
    new_scale_factor = maximum_dim / max_dim
    if new_scale_factor < 1.0:
        obj.scale = new_scale_factor
        cfg["bounding_box"] = cfg["bounding_box"] * new_scale_factor


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
        sampled = sample_objects(num_objects=1, included_categories=included_categories)
        if not sampled:
            raise RuntimeError(
                f"replace_obj could not sample a replacement for '{obj_name}': no installed "
                f"object model matches included_categories={included_categories}")
        nobj_cfg = sampled[0]

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

    rescale_to_max_dim(new_obj, nobj_cfg, maximum_dim)
    nobj_cfg["fixed_base"] = fixed_base

    return new_obj, nobj_cfg
