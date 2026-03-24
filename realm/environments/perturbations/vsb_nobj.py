from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.environments.perturbations._helpers import replace_obj

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def vsb_nobj(env: "RealmEnvironmentDynamic") -> None:
    included_categories = None
    if env.task_type == "push":
        included_categories = ["electric_switch", "thermostat"] # TODO: microwave, monitor buttons (maybe more)?
    elif env.task_type in ["open_drawer", "close_drawer"]:
        included_categories = ["bottom_cabinet"]

    og.sim.stop()
    fixed_base_loc = True if env.task_type in ["push", "open_drawer", "close_drawer"] else False
    preserve_ori = False if env.task_type in ["push", "open_drawer", "close_drawer"] else True
    max_dim = 0.5 if env.task_type in ["open_drawer", "close_drawer"] else 0.15
    nobj, nobj_cfg = replace_obj(env, env.main_objects[0], included_categories=included_categories, maximum_dim=max_dim, fixed_base=fixed_base_loc, preserve_ori=preserve_ori)
    env.main_objects = [nobj]

    env.instruction = env.cfg["instruction"].replace(env.cfg["instruction_obj_to_replace"], nobj_cfg["category"].replace("_", " "))
    og.log.info(f"New instruction: {env.instruction}")
    if nobj_cfg["model"] in ["strbnw", "gashan", "qxhtct", "wseglt"]:
        env.main_objects[0].set_orientation(np.array([0, 0, 0.7071068, 0.7071068]))
    og.sim.play()
    og.sim.step()
    env.omnigibson_env.scene.update_initial_state()
    env.reset_joints()

    if og.object_states.ToggledOn in nobj.states:
        nobj.states[og.object_states.ToggledOn].visual_marker.visible = False

    # fake rest to get to original pose after stopping sim
    for _ in range(30):
        env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
