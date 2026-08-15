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
