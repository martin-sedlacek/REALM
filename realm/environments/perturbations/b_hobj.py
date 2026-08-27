from __future__ import annotations
from realm.config.shared import MASS_CLIP_KG

import numpy as np
from typing import TYPE_CHECKING

from omnigibson.prims.joint_prim import JointPrim

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic




def _baselines(env: "RealmEnvironmentDynamic", obj) -> dict:

    if not hasattr(env, "b_hobj_baselines"):
        env.b_hobj_baselines = {}
    key = obj._relative_prim_path  # relative prim path as unique id, as in init_poses
    base = env.b_hobj_baselines.get(key)
    if base is None or set(base["mass"]) != set(obj._links) or set(base["joints"]) != set(obj.joints):
        base = {
            "mass": {name: float(link.mass) for name, link in obj._links.items()},
            "joints": {name: (float(j.max_velocity), float(j.max_effort), float(j.stiffness),
                              float(j.damping), float(j.friction))
                       for name, j in obj.joints.items()},
        }
        env.b_hobj_baselines[key] = base
    return base


def b_hobj(env: "RealmEnvironmentDynamic") -> None:

    s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
    for obj in env.main_objects:
        base = _baselines(env, obj)

        for name, link in obj._links.items():
            # Isaac's articulation views require float32-compatible scalar inputs.
            link.mass = float(min(base["mass"][name] * s_mass, MASS_CLIP_KG))  # clip at 2.0kg payload

        for name, joint in obj.joints.items():
            joint: JointPrim
            base_mvel, base_meff, base_stif, base_damp, base_fric = base["joints"][name]
            joint.max_velocity = float(base_mvel * s_mvel)
            joint.max_effort = float(base_meff * s_meff)
            joint.stiffness = float(base_stif * s_stif)
            joint.damping = float(base_damp * s_damp)
            joint.friction = float(base_fric * s_fric)
