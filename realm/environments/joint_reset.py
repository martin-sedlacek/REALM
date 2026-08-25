"""Drawer-task joint resets, batched across the members of a vector env.

Only ``open_drawer`` / ``close_drawer`` reach any of this; every other task type takes an early
return that costs nothing.

A drawer reset costs ~55 ``og.sim.step()`` calls: 10 + 5 inside :func:`utils.reset_joints_batched`
driving the joints home, then ``JOINT_SETTLE_STEPS`` + ``JOINT_HOLD_STEPS`` free-running here.
``og.sim.step()`` is GLOBAL -- it advances every scene in the simulator -- so a vector env that let
each member run its own loop would pay 55*N global steps per reset and advance every member's scene
N times over while driving only one member's joints. That is the same defect the settle loop and the
per-member ``og.sim.stop()``/``play()`` cycles were hoisted out of ``RealmVectorEnvironment.reset()``
for.

So the work is cut in two. :meth:`JointResetMixin.reset_joints` does the per-member half -- pick the
joint, set its drive -- and records the rest as a :class:`JointResetPlan`; :func:`run_joint_resets`
drives every recorded plan off ONE shared step loop. A single env runs its own plan inline and emits
exactly the call sequence the pre-batching straight-line version did.

Recording rather than no-oping is deliberate, for the reason ``perturbations/_helpers.settle()``
raises a flag: a member that never asked for a joint reset must not silently acquire one, and a plan
that is never drained must fail loudly -- ``RealmVectorEnvironment.reset()`` asserts nothing is left
pending -- rather than quietly leave a drawer in the wrong start state and score the rollout against
it.

VERIFIED END TO END 2026-08-14, once ``open_drawer``/``close_drawer`` started loading. Measured on
task 8, Default, with the openness of every drawer read back after each reset:

    num_envs=2  reset issues 57 og.sim.step() calls  (2 per-member reset obs + 55 shared)
    num_envs=1  reset issues 56                      (1 + 55)

57 rather than 110 is the batching. The outcome is right too, not just the count: member 1 lands
every one of its five drawer joints on the commanded normalized -1.0000, and member 0 lands on
exactly the state a num_envs=1 run of the same task produces, joint for joint. Both halves matter --
the count alone would also be satisfied by a loop that stepped once and wrote nothing.
``tests/test_joint_reset_batching.py`` pins the schedule against a stubbed simulator.

Found through this path but NOT caused by it: scene 0's cabinet used to be placed lying on its back,
so its drawers slid vertically and jammed instead of reaching the commanded position. Fixed in
OG-lite.
"""
from collections import namedtuple

import torch

import omnigibson as og
from omnigibson.prims.joint_prim import JointPrim

from realm.environments.utils import (
    get_openable_joints,
    get_target_drawer_joint,
    reset_joints_batched,
)

# Normalized openness close_drawer's target drawer starts from; every other joint starts at -1.
INIT_OPENNESS_FRACTION = 1.0

# The free-run half of a drawer reset: let the cabinet come to rest, tell every one of its joints to
# hold, then let that take effect.
JOINT_SETTLE_STEPS = 30
JOINT_HOLD_STEPS = 10

JointResetPlan = namedtuple("JointResetPlan", ["cabinet", "joints", "reset_states"])
JointResetPlan.__doc__ = """One member's pending drawer reset, minus every og.sim.step().

The steps are left out because they are global: run_joint_resets() issues them once for all members.
"""


def run_joint_resets(envs):
    """Work every pending joint reset in @envs off ONE shared set of og.sim.step() calls.

    Each member still experiences exactly the sequence a single env gives it: its own writes, then a
    step, N members' writes at a time. With one member the emitted calls are identical to the
    pre-batching straight-line version, so single-env behaviour is unchanged.
    """
    pending = [env for env in envs if env.pending_joint_reset is not None]
    if not pending:
        return

    reset_joints_batched(
        [(env.pending_joint_reset.joints, env.pending_joint_reset.reset_states) for env in pending]
    )
    # Between the two loops, exactly where the straight-line version read it.
    for env in pending:
        env._record_joint_openness()

    # Pure settle -- no camera is read, so skip the render pass on all 40 steps.
    # (gm.HEADLESS only removes the window; step() still renders without this context.)
    with og.sim.render_on_step(False):
        for _ in range(JOINT_SETTLE_STEPS):
            og.sim.step()
        for env in pending:
            for j in env.pending_joint_reset.cabinet.joints.values():
                j: JointPrim
                j.keep_still()
        for _ in range(JOINT_HOLD_STEPS):
            og.sim.step()

    # Clear last: the loop above reads pending_joint_reset.cabinet, and a member whose plan is still
    # set after this returns is the "recorded but never run" case RealmVectorEnvironment asserts on.
    for env in pending:
        env.pending_joint_reset = None


class JointResetMixin:
    """Drawer-reset half of a REALM environment.

    Expects the host to provide ``task_type``, ``main_objects`` and ``in_vec_env``, and sets
    ``mo_joint``, ``joint_range`` and ``init_openness_fraction`` on it.
    """

    # A JointResetPlan recorded by reset_joints() and drained by RealmVectorEnvironment. Always None
    # outside a vector env, where reset_joints() runs the plan inline before returning.
    pending_joint_reset = None

    def reset_joints(self, target_drawer_loc: str = "top"):
        """Put this member's cabinet back to the task's starting drawer state.

        In a vector env this only RECORDS the plan and returns; RealmVectorEnvironment drains it and
        runs one shared step loop for every member. See the module docstring for why.
        """
        if self.task_type not in ("open_drawer", "close_drawer"):
            self.mo_joint = None
            return

        self.pending_joint_reset = self._prepare_joint_reset(target_drawer_loc)
        if not self.in_vec_env:
            run_joint_resets([self])

    def _prepare_joint_reset(self, target_drawer_loc: str) -> JointResetPlan:
        """The half of reset_joints() that touches only THIS member: pick the joint, set its drive.

        Deliberately contains no og.sim.step(): that is what lets a vector env run this for every
        member up front and then step once for all of them. It stays at the reset_joints() call site
        rather than being deferred with the stepping, so a caller that reads self.mo_joint straight
        afterwards still sees the joint this reset selected.
        """
        cabinet = self.main_objects[0]
        init_state_open = self.task_type == "close_drawer"
        self.mo_joint = get_target_drawer_joint(cabinet, target_drawer_loc=target_drawer_loc)

        self.mo_joint._articulation_view.set_max_efforts(torch.tensor([[1.0e8]], dtype=torch.float32), joint_indices=self.mo_joint.dof_indices)
        self.mo_joint._articulation_view.set_gains(kps=torch.tensor([[0.0]]), joint_indices=self.mo_joint.dof_indices)
        self.mo_joint._articulation_view.set_gains(kds=torch.tensor([[1000.0]]), joint_indices=self.mo_joint.dof_indices)

        openable_joints = get_openable_joints(cabinet)
        reset_states = [-1 for _ in openable_joints]
        target_joint_ind = openable_joints.index(self.mo_joint)
        reset_states[target_joint_ind] = INIT_OPENNESS_FRACTION if init_state_open else -1
        return JointResetPlan(cabinet=cabinet, joints=openable_joints, reset_states=reset_states)

    def _record_joint_openness(self):
        """Capture the openness reference the joint progression stages are measured against.

        This is the openness the drawer actually SETTLED into once driven home, not the one it was
        commanded to. Called by run_joint_resets() between the driving loop and the free-run loop,
        which is where the pre-batching straight-line version read it.
        """
        self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
        self.init_openness_fraction = (self.mo_joint.get_state()[0][
                                           0] - self.mo_joint.lower_limit) / self.joint_range
