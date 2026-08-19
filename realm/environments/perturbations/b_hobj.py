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
            # (max_velocity, max_effort, stiffness, damping, friction) per joint -- the five
            # JointPrim properties b_hobj scales, snapshot in the order it applies them.
            "joints": {name: (float(j.max_velocity), float(j.max_effort), float(j.stiffness),
                              float(j.damping), float(j.friction))
                       for name, j in obj.joints.items()},
        }
        env.b_hobj_baselines[key] = base
    return base


def b_hobj(env: "RealmEnvironmentDynamic") -> None:
    """Scale the main object's mass and joint properties from their pristine baselines.

    Six independent log-uniform factors, e^U(-1,1) i.e. ~[0.37, 2.72], scale mass and the five
    JointPrim properties (max velocity, max effort, stiffness, damping, friction). Mass is clipped
    at MASS_CLIP_KG. All six setters exist and write through the articulation view in OmniGibson
    3.9.1 (prims/joint_prim.py).

    FIXED 2026-08-19 in the versioned number-moving batch (VERSION 1.0.0, owner call: recompute
    rather than preserve): previously only s_meff / s_stif / s_damp were applied -- mass was scaled
    by an UNRELATED U(0.25, 3) draw and s_mass / s_mvel / s_fric were discarded, so max velocity
    and friction were never perturbed at all. That extra uniform draw is now REMOVED, which shifts
    the shared RNG stream: B-HOBJ numbers recorded before 1.0.0 are not comparable and must be
    recomputed. See CHANGE_LEDGER.md.
    """
    s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
    for obj in env.main_objects:
        base = _baselines(env, obj)

        for name, link in obj._links.items():
            # float() for the same reason as the joint setters below -- and it is NOT redundant here
            # even though min() looks like it returns a plain float. `base["mass"][name] * s_mass` is
            # an np.float64 (s_mass comes from np.random), and min() returns whichever OPERAND is
            # smaller, keeping its type: below the clip it hands back the np.float64, at the clip it
            # hands back the Python MASS_CLIP_KG. So the dtype of what gets written depends on the
            # DRAW, and only the under-clip branch is wrong.
            #
            # Uncast, that branch reaches RigidDynamicPrim.mass -> th.tensor([np.float64]) -> a float64
            # tensor -> set_masses' `dst[indices] = src` against a float32 destination:
            #   RuntimeError: Index put requires the source and destination dtypes match,
            #                 got Float for the destination and Double for the source.
            # then a SIGSEGV at teardown. Measured 2026-08-19, first live B-HOBJ run after the 1.0.0
            # batch (logs/todo_clara/item3_logs); object masses here sit well under 2 kg, so the bad
            # branch is the usual one. No number moves: Isaac stores masses as float32 either way.
            link.mass = float(min(base["mass"][name] * s_mass, MASS_CLIP_KG))  # clip at 2.0kg payload

        for name, joint in obj.joints.items():
            joint: JointPrim
            base_mvel, base_meff, base_stif, base_damp, base_fric = base["joints"][name]
            # float() rather than leaving the numpy scalar in: the setters build the tensor
            # themselves, and a np.float64 would make it a float64 tensor where the articulation
            # view expects float32.
            #
            # The setters ARE the writes -- JointPrim's property setters call set_max_velocities /
            # set_max_efforts / set_gains / set_friction_coefficients on the articulation view.
            joint.max_velocity = float(base_mvel * s_mvel)
            joint.max_effort = float(base_meff * s_meff)
            joint.stiffness = float(base_stif * s_stif)
            joint.damping = float(base_damp * s_damp)
            joint.friction = float(base_fric * s_fric)
