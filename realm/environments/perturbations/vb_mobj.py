"""VB-MOBJ: rescale the main object by a random per-axis factor.

What one call mutates on @env: ``main_objects[0]`` (a PrimitiveObject is rescaled in place; a
DatasetObject is removed and re-added at the new bounding box, same name/prim path/pose) and the
scene itself. Drawer tasks clip to cabinet-sized bounds and fix the base. USDObjects are not
supported.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from realm.placement import get_default_objects_cfg
from realm.config.shared import UNSUPPORTED_BY_PERTURBATION
from realm.environments.perturbations._helpers import rebase_after_play, settle, sim_play, sim_stop

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Per-axis scale factor draw, and the cap on the factors' product -- rejecting draws whose
#: combined volume factor exceeds it keeps the object recognisable. P(1000 consecutive rejections)
#: is effectively zero; if it ever happened, the last (rejected) draw would be used as-is.
RESCALE_RANGE = (0.5, 1.5)
MAX_VOLUME_FACTOR = 1.5
RESCALE_MAX_TRIES = 1000

#: Post-scale bbox clips (metres): cabinets must stay reachable with usable drawers, everything
#: else must remain graspable.
DRAWER_BBOX_CLIP = (0.4, 0.75)
TABLETOP_BBOX_CLIP = (0.02, 0.175)


def vb_mobj(env: "RealmEnvironmentDynamic") -> None:
    for _ in range(RESCALE_MAX_TRIES):
        s1 = np.random.uniform(*RESCALE_RANGE)
        s2 = np.random.uniform(*RESCALE_RANGE)
        s3 = np.random.uniform(*RESCALE_RANGE)
        if s1 * s2 * s3 <= MAX_VOLUME_FACTOR:
            break

    scene = env.omnigibson_env.scene
    mo = env.main_objects[0]

    if type(mo) is PrimitiveObject:
        # assumes the primitives have a default scale 1,1,1 hence the orig bbox can be used as replacement
        # sim_stop/sim_play rather than og.sim.stop()/play(): those are global, so a vector env
        # batches ONE cycle for every member and these no-op. Rescaling a prim needs the sim stopped.
        sim_stop(env)
        scale = torch.tensor([s1, s2, s3])
        mo.scale = torch.tensor(env.mo_bbox_orig) * scale
        sim_play(env)
        # Let the rescaled object come to rest; no-op in a vector env, which settles once for all.
        settle(env)
    else:
        obj_name = mo.name
        obj_relative_prim_path = mo._relative_prim_path
        new_bbox = env.mo_bbox_orig * np.array([s1, s2, s3])

        obj_cfg = None
        if type(mo) is DatasetObject:
            obj_cfg = get_default_objects_cfg(env.omnigibson_env.scene, [mo.name])[obj_name]

        # Genuinely needs the stopped sim: this removes and re-adds an object.
        sim_stop(env)
        scene.remove_object(mo)

        if env.task_type in ["open_drawer", "close_drawer"]:
            new_bbox = np.clip(new_bbox, a_min=DRAWER_BBOX_CLIP[0], a_max=DRAWER_BBOX_CLIP[1])
            fix_base = True
        else:
            new_bbox = np.clip(new_bbox, a_min=TABLETOP_BBOX_CLIP[0], a_max=TABLETOP_BBOX_CLIP[1])
            fix_base = False

        if type(mo) is DatasetObject:
            new_obj = DatasetObject(
                name=obj_name,
                relative_prim_path=obj_relative_prim_path,
                category=mo.category,
                model=mo.model,
                bounding_box=torch.tensor(new_bbox, dtype=torch.float32),
                fixed_base=fix_base
            )
            scene.add_object(new_obj)
            new_obj.set_bbox_center_position_orientation(obj_cfg["pos"], obj_cfg["ori"])
        else:
            assert type(mo) is USDObject
            if env.task_type in UNSUPPORTED_BY_PERTURBATION["VB-MOBJ"]:
                raise NotImplementedError(
                    f"VB-MOBJ does not support task_type {env.task_type!r}: its main object is a USD asset"
                )
            raise NotImplementedError()

        env.main_objects = [new_obj]
        sim_play(env)

        # Replaced an object, so the reset baseline must be rebased -- see rebase_after_play.
        # VB-MOBJ has always rebased unconditionally (unlike V-SC/SB-VRB), hence vec_only=False.
        rebase_after_play(env, vec_only_rebase=False)
