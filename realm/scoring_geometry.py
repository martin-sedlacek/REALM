"""Scoring on current geometry: the rubric's object poses from the physics engine on steps that do not render.

Why. A moving link's pose (`RigidDynamicPrim.get_position_orientation`) is read from the PhysX view and is
current on every step. But `obj.aabb`, and with it OmniGibson's `AABB`, `VerticalAdjacency`, `OnTop` and
`Inside` states, is built from `XFormPrim.scaled_transform`, which reads Fabric's world transform
(`omnigibson.utils.usd_utils.get_world_pose_with_scale`); Fabric is synced from PhysX only on a step that
renders. With render-on-demand, the OnTop/Inside reads (PLACE_ONTO, PLACE_INTO, POURED_PROXY and
`is_placed_on_target`, behind the `object_drops` column) would otherwise see the geometry of the last rendered
step.

`physx_link_poses(env)`: within the context, `get_world_pose_with_scale` returns, for the prim path of a moving
rigid link (`RigidDynamicPrim`) of any object in the env's scene, the matrix built from that link's PhysX world
pose and the scale read from Fabric's own matrix; every other path is passed through unchanged. Kinematic and
fixed bodies do not move, so Fabric is right for them. The robot's links are covered too, but no rubric checker
reads a robot AABB (REACH and GRASP use finger link poses and contacts, both PhysX). Nothing is written to Fabric
or PhysX; the function is restored on exit. If OmniGibson's function is missing or has another signature (an
image upgrade), entering the context raises instead of silently scoring on Fabric.

`current_geometry(env)`: what scoring uses (`RealmEnvironmentDynamic.step` / `post_step`, `Rollout.record_step`).
A no-op when the env's last control step rendered (`env.last_step_rendered`, recorded inside the step): Fabric
was just synced then, and with every step rendered nothing changes. Otherwise it runs the body on emptied object-
state caches, which are put back afterwards so that nothing outside scoring sees a value computed here, with
`physx_link_poses`.
"""
import contextlib
import inspect


def _scene_states(env):
    states = []
    for obj in env.omnigibson_env.scene.objects:
        states.extend(getattr(obj, "states", {}).values())
    return states


@contextlib.contextmanager
def _empty_caches(env):
    """Run the body on emptied object-state caches; put the originals back after."""
    saved = []
    for state in _scene_states(env):
        cache, changed = getattr(state, "_cache", None), getattr(state, "_changed", None)
        saved.append((state, cache, changed))
        if cache is not None:
            state._cache = {}
        if changed is not None:
            state._changed = {}
    try:
        yield
    finally:
        for state, cache, changed in saved:
            state._cache, state._changed = cache, changed


def _check_patch_target(usd_utils, xform_prim):
    """Raise unless OmniGibson's pose function is where and what this module patches."""
    original = getattr(usd_utils, "get_world_pose_with_scale", None)
    if original is None or not callable(original):
        raise RuntimeError("realm.scoring_geometry: omnigibson.utils.usd_utils.get_world_pose_with_scale is "
                           "missing; the current-geometry patch cannot be applied to this OmniGibson")
    params = list(inspect.signature(original).parameters)
    if params != ["prim_path"]:
        raise RuntimeError("realm.scoring_geometry: get_world_pose_with_scale has parameters "
                           f"{params}, expected ['prim_path']; review the patch for this OmniGibson")
    if getattr(xform_prim, "get_world_pose_with_scale", None) is not original:
        raise RuntimeError("realm.scoring_geometry: omnigibson.prims.xform_prim no longer uses "
                           "usd_utils.get_world_pose_with_scale; review the patch for this OmniGibson")
    return original


@contextlib.contextmanager
def physx_link_poses(env):
    """Within the body, Fabric world transforms of the scene's moving rigid links are their PhysX poses."""
    import torch as th
    import omnigibson.prims.xform_prim as xform_prim
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
    from omnigibson.utils import transform_utils as T
    from omnigibson.utils import usd_utils

    original = _check_patch_target(usd_utils, xform_prim)
    links = {}
    for obj in env.omnigibson_env.scene.objects:
        for link in getattr(obj, "links", {}).values():
            if isinstance(link, RigidDynamicPrim):
                links[link.prim_path] = link

    def physx_pose_with_scale(prim_path):
        fabric = original(prim_path)
        link = links.get(prim_path)
        if link is None:
            return fabric
        position, quaternion = link.get_position_orientation()     # world frame, like Fabric's matrix
        scale = th.linalg.norm(fabric[:3, :3], dim=0)               # column-vector convention: R @ diag(s)
        matrix = th.eye(4, dtype=fabric.dtype)
        matrix[:3, :3] = T.quat2mat(quaternion).to(fabric.dtype) * scale
        matrix[:3, 3] = position.to(fabric.dtype)
        return matrix

    usd_utils.get_world_pose_with_scale = physx_pose_with_scale
    xform_prim.get_world_pose_with_scale = physx_pose_with_scale
    try:
        yield
    finally:
        usd_utils.get_world_pose_with_scale = original
        xform_prim.get_world_pose_with_scale = original


@contextlib.contextmanager
def current_geometry(env):
    """Scoring context: see the module docstring. A no-op after a rendered step."""
    rendered = getattr(env, "last_step_rendered", None)
    if rendered is None:                    # before the env's first step: the simulator's flag
        import omnigibson as og
        rendered = bool(og.sim._render_on_step)
    if rendered:
        yield
        return
    with _empty_caches(env), physx_link_poses(env):
        yield
