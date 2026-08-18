"""VSB-NOBJ: replace the main object with one drawn from an unseen category.

What one call mutates on @env: ``main_objects[0]`` (a freshly sampled object at the same name,
prim path and recorded pose), ``instruction`` (the object noun swapped in), and the scene itself.
Push and drawer tasks constrain the draw to categories that keep the task doable, and fix the new
object's base so it cannot be knocked over.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.environments.perturbations._helpers import (
    rebase_after_play,
    settle,
    sim_play,
    sim_stop,
)
from realm.environments.perturbations.object_sampling import replace_obj

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic

#: Task types whose replacement object must be fixed to the wall/floor (and whose recorded
#: orientation must NOT be preserved -- the new asset's own upright pose is used instead).
FIXED_BASE_TASK_TYPES = ("push", "open_drawer", "close_drawer")

#: Largest bbox dimension (metres) for the replacement: cabinets need to stay cabinet-sized,
#: everything else has to remain graspable.
DRAWER_MAX_DIM = 0.5
GRASPABLE_MAX_DIM = 0.15

#: Models that need this fixed upright orientation after replacement (provenance unrecorded --
#: kept exactly as the original hardcoded list).
UPRIGHT_MODELS = ("strbnw", "gashan", "qxhtct", "wseglt")


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
    fixed_task = env.task_type in FIXED_BASE_TASK_TYPES
    max_dim = DRAWER_MAX_DIM if env.task_type in ["open_drawer", "close_drawer"] else GRASPABLE_MAX_DIM
    nobj, nobj_cfg = replace_obj(env, env.main_objects[0], included_categories=included_categories,
                                 maximum_dim=max_dim, fixed_base=fixed_task,
                                 preserve_ori=not fixed_task)
    env.main_objects = [nobj]

    env.instruction = env.cfg["instruction"].replace(env.cfg["instruction_obj_to_replace"],
                                                     nobj_cfg["category"].replace("_", " "))
    og.log.info(f"New instruction: {env.instruction}")
    if nobj_cfg["model"] in UPRIGHT_MODELS:
        env.main_objects[0].set_orientation(np.array([0, 0, 0.7071068, 0.7071068]))
    sim_play(env)

    def _hide_toggle_marker():
        # MUST run inside the deferred post-play block, not after it: ToggleState.visual_marker is
        # None until the state's _initialize() runs, which needs a playing sim. Single-env this was
        # invisible -- sim_play() really played and the block ran inline. In a vector env the whole
        # block is deferred to the shared play, so running this line outside it touched a brand-new
        # object on a stopped sim: AttributeError: 'NoneType' object has no attribute 'visible'
        # (measured on task 4, pick_spoon, where the sampler drew a toggleable replacement).
        if og.object_states.ToggledOn in nobj.states:
            nobj.states[og.object_states.ToggledOn].visual_marker.visible = False

    rebase_after_play(env, vec_only_rebase=False, extra=_hide_toggle_marker)

    # Let the replaced object come to rest. No-op in a vector env, where the shared settle runs once
    # for all members instead of stepping the global sim 30 times per member.
    settle(env)
