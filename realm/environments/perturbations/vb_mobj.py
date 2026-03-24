from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og
from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from realm.helpers import get_default_objects_cfg

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def vb_mobj(env: "RealmEnvironmentDynamic") -> None:
    # sample rescaling of the bbox
    for _ in range(1000):
        s1 = np.random.uniform(0.5, 1.5)
        s2 = np.random.uniform(0.5, 1.5)
        s3 = np.random.uniform(0.5, 1.5)
        if s1 * s2 * s3 <= 1.5:
            break

    scene = env.omnigibson_env.scene
    mo = env.main_objects[0]

    if type(mo) == PrimitiveObject:
        # assumes the primitives have a default scale 1,1,1 hence the orig bbox can be used as replacement
        og.sim.stop()
        scale = torch.tensor([s1, s2, s3])
        mo.scale = torch.tensor(env.mo_bbox_orig) * scale
        og.sim.play()
        for _ in range(30):
            env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
    else:
        obj_name = mo.name
        obj_relative_prim_path = mo._relative_prim_path
        new_bbox = env.mo_bbox_orig * np.array([s1, s2, s3])

        obj_cfg = None
        if type(mo) == DatasetObject:
            obj_cfg = get_default_objects_cfg(env.omnigibson_env.scene, [mo.name])[obj_name]

        og.sim.stop()
        scene.remove_object(mo)

        if env.task_type in ["open_drawer", "close_drawer"]:
            new_bbox = np.clip(new_bbox, a_min=0.4, a_max=0.75)
            fix_base = True
        else:
            new_bbox = np.clip(new_bbox, a_min=0.02, a_max=0.175)
            fix_base = False

        if type(mo) == DatasetObject:
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
            assert type(mo) == USDObject
            raise NotImplementedError()

        env.main_objects = [new_obj]
        og.sim.play()
        og.sim.step()
        env.omnigibson_env.scene.update_initial_state()
        env.reset_joints()
