import torch as th
from omnigibson.controllers.controller_base import (
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
import omnigibson as og  # For og.sim.device


class IndividualJointPDController(LocomotionController, ManipulationController, GripperController):


    def __init__(
            self,
            control_freq,
            motor_type,
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            kp=50,
            kd=1,
            use_impedances=False,
            use_gravity_compensation=False,
            use_cc_compensation=True,
            use_delta_commands=False,
            compute_delta_in_quat_space=None,
            max_effort=None,
            min_effort=None,
            **kwargs,
    ):
        motor_type = "effort"

        self.kp = kp
        self.kd = kd
        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def _get_joint_positions(self):
        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][:, self.dof_idx]

    def _get_joint_velocities(self):

        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_velocities(
            self.routing_path, estimate=False
        )[rows, :][
            :, self.dof_idx
        ]

    def _update_goal(self, controller_idx, command):
        target_joint_pos = cb.to_torch(command).to(og.sim.device)

        target_joint_pos = target_joint_pos.clip(
            cb.to_torch(self._control_limits[ControlType.get_type("position")][0][self.dof_idx]).to(og.sim.device),
            cb.to_torch(self._control_limits[ControlType.get_type("position")][1][self.dof_idx]).to(og.sim.device),
        )

        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(
            target_joint_pos=cb.from_torch(target_joint_pos),
            target_joint_vel=cb.from_torch(target_joint_vel),
        )

    def compute_control(self, goals):
        current_joint_pos = cb.to_torch(self._get_joint_positions()).to(og.sim.device)
        current_joint_vel = cb.to_torch(self._get_joint_velocities()).to(og.sim.device)

        joint_pos_desired = cb.to_torch(goals["target_joint_pos"]).to(og.sim.device)
        joint_vel_desired = cb.to_torch(goals["target_joint_vel"]).to(og.sim.device)

        u = self.kp * (joint_pos_desired - current_joint_pos) + self.kd * (joint_vel_desired - current_joint_vel)

        if self.min_effort is not None and self.max_effort is not None:
            assert u.shape[-1] == self.max_effort.shape[-1] == self.min_effort.shape[-1]
            u = u.clip(self.min_effort, self.max_effort)

        return cb.from_torch(u)

    def compute_no_op_goal(self, controller_idx):
        target_joint_pos = cb.to_torch(self._get_joint_positions()[controller_idx]).to(og.sim.device)
        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(
            target_joint_pos=cb.from_torch(target_joint_pos),
            target_joint_vel=cb.from_torch(target_joint_vel),
        )

    def _compute_no_op_command(self, controller_idx):
        return cb.zeros(self.command_dim)

    def _get_goal_shapes(self):
        return dict(
            target_joint_pos=(self.control_dim,),
            target_joint_vel=(self.control_dim,)
        )

    def is_grasping(self, controller_idx):
        return IsGraspingState.UNKNOWN

    @property
    def motor_type(self):
        return self._motor_type

    @property
    def control_type(self):
        return ControlType.EFFORT

    @property
    def command_dim(self):
        return len(self.dof_idx)
