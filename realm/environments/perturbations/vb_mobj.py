from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import omnigibson as og
from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from realm.placement import get_default_objects_cfg
from realm.environments.perturbations._helpers import after_play, settle, sim_play, sim_stop

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
        if type(mo) == DatasetObject:
            obj_cfg = get_default_objects_cfg(env.omnigibson_env.scene, [mo.name])[obj_name]

        # Genuinely needs the stopped sim: this removes and re-adds an object.
        sim_stop(env)
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
        sim_play(env)

        # Needs a PLAYING sim -- og.sim.step() asserts it and update_initial_file() must capture the
        # post-play state -- so in a vector env this waits for the shared play.
        def _post_play():
            og.sim.step()
            env.omnigibson_env.scene.update_initial_file()  # renamed from update_initial_state() in OG 3.9.1
            env.reset_joints()

        after_play(env, _post_play)
