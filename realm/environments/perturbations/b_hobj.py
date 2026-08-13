from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from omnigibson.prims.joint_prim import JointPrim

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


MASS_CLIP_KG = 2.0  # heaviest payload B-HOBJ is allowed to hand the policy


def _baselines(env: "RealmEnvironmentDynamic", obj) -> dict:
    """@obj's PRISTINE physical properties, captured the first time B-HOBJ touches it.

    Why this exists. Every property B-HOBJ scales is read back from the live simulator --
    JointPrim.max_effort/stiffness/damping go through the articulation view and RigidPrim.mass
    through the rigid-prim view (OG-lite prims/joint_prim.py, prims/rigid_dynamic_prim.py). And
    og.Environment.reset() restores pose and velocity state, NOT physical properties. So the
    original `joint.stiffness = joint.stiffness * s` read the value the PREVIOUS reset had written
    and the per-reset factors multiplied.

    Measured on task 0 (put_green_block_into_bowl) before this fix, with
    scripts/clara/interactive/t10_bhobj_props.py --resets 10:

        reset  1: mass 0.0270 -> 0.0569 kg   (2.1x baseline)
        reset  5: mass 0.0875 -> 0.2272 kg   (8.4x)
        reset  9: mass 0.3371 -> 0.7393 kg   (27.4x)

    and every reset's "pre" value was exactly the previous reset's "post" -- 9 of 9 properties
    carried over, 0 restored. Over a 25-rollout eval the 3 cm block ends up pinned at the 2 kg cap:
    every B-HOBJ number collected that way is a measurement of a drifting object, and no rollout
    carries the perturbation that was drawn for it. stiffness/damping/max_effort have no cap at
    all, so they just keep going.

    The fix is the same shape as RealmEnvironmentDynamic.bind_scene_handles()'s `init_poses`, which
    solves the identical problem for pose: snapshot the pristine value once, keyed by relative prim
    path, and always scale from the snapshot. Captured lazily here rather than in
    bind_scene_handles() so nothing pays for it unless B-HOBJ is actually active; the first call
    happens inside the first reset(), before any perturbation has run, so what it sees is pristine.

    The link/joint name check re-captures when the object underneath the prim path has been swapped
    out (VB-MOBJ / VSB-NOBJ / SB-VRB call replace_obj(), which reuses the relative prim path for a
    DIFFERENT asset). Scaling a new asset's mass by an old asset's baseline would be meaningless.
    """
    if not hasattr(env, "b_hobj_baselines"):
        env.b_hobj_baselines = {}
    key = obj._relative_prim_path  # relative prim path as unique id, as in init_poses
    base = env.b_hobj_baselines.get(key)
    if base is None or set(base["mass"]) != set(obj._links) or set(base["joints"]) != set(obj.joints):
        base = {
            "mass": {name: float(link.mass) for name, link in obj._links.items()},
            # (max_effort, stiffness, damping) per joint.
            "joints": {name: (float(j.max_effort), float(j.stiffness), float(j.damping))
                       for name, j in obj.joints.items()},
        }
        env.b_hobj_baselines[key] = base
    return base


def b_hobj(env: "RealmEnvironmentDynamic") -> None:
    s = np.random.uniform(0.25, 3)
    s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
    for obj in env.main_objects:
        base = _baselines(env, obj)

        for name, link in obj._links.items():
            link.mass = min(base["mass"][name] * s, MASS_CLIP_KG)  # clip at 2.0kg payload

        for name, joint in obj.joints.items():
            joint: JointPrim
            base_effort, base_stiffness, base_damping = base["joints"][name]
            # float() rather than leaving the numpy scalar in: the setters build the tensor
            # themselves, and a np.float64 would make it a float64 tensor where the articulation
            # view expects float32.
            #
            # The setters ARE the writes -- JointPrim.max_effort/stiffness/damping call
            # set_max_efforts/set_gains on the articulation view. This used to repeat all three
            # calls by hand immediately afterwards with the same values; that was a no-op (each
            # write is an unconditional assignment, so applying the group twice lands in the same
            # place) and only obscured which line was doing the work.
            joint.max_effort = float(base_effort * s_meff)
            joint.stiffness = float(base_stiffness * s_stif)
            joint.damping = float(base_damping * s_damp)
