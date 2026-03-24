from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from omnigibson.prims.joint_prim import JointPrim

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


def b_hobj(env: "RealmEnvironmentDynamic") -> None:
    s = np.random.uniform(0.25, 3)
    s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
    for obj in env.main_objects:
        for link in obj._links.values():
            link.mass = min(link.mass * s, 2.0) # clip at 2.0kg payload

        for joint in obj.joints.values():
            joint: JointPrim
            joint.max_effort = joint.max_effort * float(s_meff)
            joint.stiffness = joint.stiffness * s_stif
            joint.damping = joint.damping * s_damp
            joint._articulation_view.set_max_efforts(torch.tensor([[joint.max_effort]], dtype=torch.float32), joint_indices=joint.dof_indices)
            joint._articulation_view.set_gains(kps=torch.tensor([[joint.stiffness]]), joint_indices=joint.dof_indices)
            joint._articulation_view.set_gains(kds=torch.tensor([[joint.damping]]), joint_indices=joint.dof_indices)
