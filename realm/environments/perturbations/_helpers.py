"""The sim-state contract every perturbation goes through.

``og.sim.stop()``/``play()`` and ``og.sim.step()`` are GLOBAL: they act on every scene in the
simulator, not on one member. REALM applies perturbations per member inside ``reset()``, so a
perturbation that calls them directly does N times the work in a vector env AND disturbs its
siblings mid-reset. Measured (job 190555, VB-POSE Vec=4): three of four members lost their main
object from the contact view and scored TP=0.00, 18 of 25 rollouts logged zero environment
collisions and never left REACH, and the job still exited 0.

So perturbations never call them directly. They call the wrappers below, which no-op (or defer, or
raise a flag) when the member belongs to a vector env; ``RealmVectorEnvironment.reset()`` then
performs ONE stop/play cycle and ONE settle for all members together. Single-env behaviour is
byte-identical to before.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import omnigibson as og

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


SETTLE_STEPS = 30

# Perturbations that genuinely cannot run on a live sim, because they add or remove objects --
# OmniGibson requires a stopped sim for that (see the note in realm/placement.py). Everything else
# only writes poses, which works fine while playing, so it must NOT be in here: VB-POSE and V-VIEW
# were moved off the stopped path precisely because a pose write never needed it.
NEEDS_STOPPED_SIM = frozenset({"V-SC", "VB-MOBJ", "VSB-NOBJ", "SB-VRB"})


def sim_stop(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.stop(), unless a vector env is batching the cycle for every member."""
    if not env.in_vec_env:
        og.sim.stop()


def sim_play(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.play(), unless a vector env is batching the cycle for every member."""
    if not env.in_vec_env:
        og.sim.play()


def sim_step(env: "RealmEnvironmentDynamic") -> None:
    """og.sim.step(), unless a vector env is batching -- see after_play for why it must be deferred."""
    if not env.in_vec_env:
        og.sim.step()


def after_play(env: "RealmEnvironmentDynamic", fn) -> None:
    """Run @fn once the simulator is playing again.

    Single env: og.sim.play() has already happened by the time a perturbation calls this, so run it
    now and behaviour is unchanged. Vector env: sim_play() was a no-op and the shared play does not
    happen until every member's perturbation has run, so @fn would execute against a STOPPED sim --
    og.Environment.step() asserts is_playing(), and update_initial_file() would capture the wrong
    state. Defer it instead; RealmVectorEnvironment.reset() drains these right after its single
    og.sim.play().
    """
    if env.in_vec_env:
        env.deferred_post_play.append(fn)
    else:
        fn()


def settle_action(env: "RealmEnvironmentDynamic"):
    """The hold-still action used to let a perturbed scene come to rest."""
    return np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1]))))


def settle(env: "RealmEnvironmentDynamic", steps: int = SETTLE_STEPS) -> None:
    """Let this member's scene come to rest after a perturbation.

    In a vector env this only RAISES A FLAG: og.sim.step() advances EVERY scene, so running the loop
    per member would step the shared sim N*steps times and, worse, advance the other members while
    feeding them no action at all. RealmVectorEnvironment.reset() reads the flag and runs the
    equivalent loop once for all members. The flag matters because settling is not unconditional --
    Default perturbs nothing and never calls this, and a vector env must not start settling resets
    that a single env would not.
    """
    if env.in_vec_env:
        env.wants_settle = True
        return
    # Nothing reads a camera here, so skip the render pass on each step. gm.HEADLESS does NOT do
    # this -- og.sim.step() renders every call regardless; only this context actually suppresses it.
    # Object states and contact caching still update normally inside it.
    with og.sim.render_on_step(False):
        for _ in range(steps):
            env.omnigibson_env.step(settle_action(env))


def backfill_object_cfgs(scene_objects, cfgs) -> None:
    """Fill in "position" and "bounding_box" on @cfgs from the live objects, where missing.

    Task YAMLs may omit both; the placement pass needs them. The frames matter: "position" is
    written in the SCENE frame, because it feeds `realm.placement`'s placement pass, which rewrites
    it from the scene-relative spawn_bbox -- and the caller then writes it back with frame="scene".
    Reading it in world frame agrees only for scene 0, whose origin IS the world origin, so a
    world-frame read is invisible single-env and silently wrong for every other member of a vector
    env (vb_pose._place has the measured failure). "bounding_box" is an EXTENT (a size), matching
    what the task YAMLs write.

    Mutates @cfgs in place; only entries whose "name" matches one of @scene_objects are touched.
    """
    for scene_obj in scene_objects:
        for cfg in cfgs:
            if cfg["name"] == scene_obj.name:
                if "position" not in cfg:
                    cfg["position"] = scene_obj.get_position_orientation(frame="scene")[0].tolist()
                if "bounding_box" not in cfg:
                    cfg["bounding_box"] = scene_obj.aabb_extent.tolist()


def set_scene_positions(env: "RealmEnvironmentDynamic", cfgs) -> None:
    """Write each cfg's "position" onto its live object, in the scene frame.

    frame="scene" because the positions come from the scene-relative spawn_bbox (see
    backfill_object_cfgs). The old set_position() call this replaced was deprecated AND
    world-frame-only, with no way to express the right frame.
    """
    for cfg in cfgs:
        env.omnigibson_env.scene.object_registry("name", cfg["name"]).set_position_orientation(
            position=cfg["position"], frame="scene")


def rebase_after_play(env: "RealmEnvironmentDynamic", vec_only_rebase: bool, extra=None) -> None:
    """Step once, rebase the scene's reset baseline, and re-run the joint reset -- after play.

    Every perturbation that adds, removes or replaces an object must do this. Without the
    update_initial_file() rebase, the next scene.reset() has to undo the object changes, and
    og.sim.dump_state() walks EVERY scene -- so in a vector env one member's half-restored scene
    makes a sibling's reset assert "Object must be initialized before dumping state!". (The related
    init-queue eviction bug is fixed in environments/vec_init_queue.py; see its docstring.)

    @vec_only_rebase=True skips the rebase in a single env. V-SC and SB-VRB have never rebased
    single-env and work, so rebasing there would be an unverified change to a working path;
    VSB-NOBJ and VB-MOBJ have always rebased unconditionally. The flag preserves each
    perturbation's historical behaviour rather than expressing a rule.

    @extra runs LAST, still inside the deferred block. Code that touches state which only exists
    once the sim is playing (e.g. a ToggledOn visual_marker) must go here, not after this call --
    in a vector env everything after this call still runs on a STOPPED sim.

    Runs inline in a single env; in a vector env the whole block is deferred until the shared
    og.sim.play() (see after_play).
    """
    def _post_play():
        og.sim.step()
        if not vec_only_rebase or env.in_vec_env:
            env.omnigibson_env.scene.update_initial_file()
        env.reset_joints()
        if extra is not None:
            extra()

    after_play(env, _post_play)
