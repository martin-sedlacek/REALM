from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.environments.perturbations._helpers import (
    after_play,
    replace_obj,
    settle,
    sim_play,
    sim_stop,
)

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def vsb_nobj(env: "RealmEnvironmentDynamic") -> None:
    included_categories = None
    if env.task_type == "push":
        included_categories = ["electric_switch", "thermostat"] # TODO: microwave, monitor buttons (maybe more)?
    elif env.task_type in ["open_drawer", "close_drawer"]:
        included_categories = ["bottom_cabinet"]

    # sim_stop/sim_play rather than og.sim.stop()/play(): those are global, so in a vector env
    # RealmVectorEnvironment.reset() does ONE cycle for every member and these no-op. This
    # perturbation genuinely needs the stopped sim -- replace_obj removes and adds an object.
    sim_stop(env)
    fixed_base_loc = True if env.task_type in ["push", "open_drawer", "close_drawer"] else False
    preserve_ori = False if env.task_type in ["push", "open_drawer", "close_drawer"] else True
    max_dim = 0.5 if env.task_type in ["open_drawer", "close_drawer"] else 0.15
    nobj, nobj_cfg = replace_obj(env, env.main_objects[0], included_categories=included_categories, maximum_dim=max_dim, fixed_base=fixed_base_loc, preserve_ori=preserve_ori)
    env.main_objects = [nobj]

    env.instruction = env.cfg["instruction"].replace(env.cfg["instruction_obj_to_replace"], nobj_cfg["category"].replace("_", " "))
    og.log.info(f"New instruction: {env.instruction}")
    if nobj_cfg["model"] in ["strbnw", "gashan", "qxhtct", "wseglt"]:
        env.main_objects[0].set_orientation(np.array([0, 0, 0.7071068, 0.7071068]))
    sim_play(env)

    # Needs a PLAYING sim: og.sim.step() asserts it, and update_initial_file() must capture the
    # post-play state. In a vector env the shared play has not happened yet, so this is deferred and
    # drained by RealmVectorEnvironment.reset() immediately after it.
    def _post_play():
        og.sim.step()
        env.omnigibson_env.scene.update_initial_file()  # renamed from update_initial_state() in OG 3.9.1
        env.reset_joints()

    after_play(env, _post_play)

    if og.object_states.ToggledOn in nobj.states:
        nobj.states[og.object_states.ToggledOn].visual_marker.visible = False

    # Let the replaced object come to rest. No-op in a vector env, where the shared settle runs once
    # for all members instead of stepping the global sim 30 times per member.
    settle(env)
