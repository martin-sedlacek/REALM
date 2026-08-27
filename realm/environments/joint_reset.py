
from collections import namedtuple

import torch

import omnigibson as og
from omnigibson.prims.joint_prim import JointPrim

from realm.environments.utils import (
    get_openable_joints,
    get_target_drawer_joint,
    reset_joints_batched,
)

INIT_OPENNESS_FRACTION = 1.0

JOINT_SETTLE_STEPS = 30
JOINT_HOLD_STEPS = 10

JointResetPlan = namedtuple("JointResetPlan", ["cabinet", "joints", "reset_states"])
JointResetPlan.__doc__ = """One member's reset state, excluding global simulator steps."""


def run_joint_resets(envs):

    pending = [env for env in envs if env.pending_joint_reset is not None]
    if not pending:
        return

    reset_joints_batched(
        [(env.pending_joint_reset.joints, env.pending_joint_reset.reset_states) for env in pending]
    )
    for env in pending:
        env._record_joint_openness()

    # HEADLESS does not disable rendering during simulator steps.
    with og.sim.render_on_step(False):
        for _ in range(JOINT_SETTLE_STEPS):
            og.sim.step()
        for env in pending:
            for j in env.pending_joint_reset.cabinet.joints.values():
                j: JointPrim
                j.keep_still()
        for _ in range(JOINT_HOLD_STEPS):
            og.sim.step()

    for env in pending:
        env.pending_joint_reset = None


class JointResetMixin:


    pending_joint_reset = None

    def reset_joints(self, target_drawer_loc: str = "top"):

        if self.task_type not in ("open_drawer", "close_drawer"):
            self.mo_joint = None
            return

        self.pending_joint_reset = self._prepare_joint_reset(target_drawer_loc)
        if not self.in_vec_env:
            run_joint_resets([self])

    def _prepare_joint_reset(self, target_drawer_loc: str) -> JointResetPlan:

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

        self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
        self.init_openness_fraction = (self.mo_joint.get_state()[0][
                                           0] - self.mo_joint.lower_limit) / self.joint_range
