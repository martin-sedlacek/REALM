import torch as th
from omnigibson.controllers.controller_base import (
    BaseController,
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
import omnigibson as og  # For og.sim.device
from omnigibson.macros import gm
import numpy as np

# Create module logger
log = create_module_logger(module_name=__name__)


class IndividualJointPDController(LocomotionController, ManipulationController, GripperController):
    def __init__(
            self,
            control_freq,
            motor_type,  # This will be forced to 'effort' for hybrid control
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            kp=50,
            kd=1,
            use_impedances=False,
            use_gravity_compensation=False,
            use_cc_compensation=True,
            use_delta_commands=False,  # Delta commands are less common for torque control
            compute_delta_in_quat_space=None,  # Delta commands are less common for torque control
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

        self.cached_torque = None

    def _get_joint_positions(self):
        """(N, control_dim) current positions of this controller's DOFs, one row per group member."""
        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][:, self.dof_idx]

    def _get_joint_velocities(self):
        """(N, control_dim) current velocities of this controller's DOFs, one row per group member.

        estimate=False on purpose -- see the long note in droid_joint_controller.py. This is an
        effort impedance law whose damping term consumes the velocity directly, and the pre-3.9.1
        version read the reported `control_dict["joint_velocity"]`. The stock 3.9.1 idiom
        (estimate=True, a one-step finite difference) leaves a ~95x larger standing error.
        """
        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_velocities(
            self.routing_path, estimate=False  # reported velocity, not the finite-difference
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
        """
        Scalar-gain joint PD. Unlike the DROID controller this needs no Jacobian, so the whole
        batch is computed at once.

        Args:
            goals (Dict[str, Array]): batched goals of shape (N, control_dim); must include
                "target_joint_pos" and "target_joint_vel"

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        current_joint_pos = cb.to_torch(self._get_joint_positions()).to(og.sim.device)  # (N, ctrl_dim)
        current_joint_vel = cb.to_torch(self._get_joint_velocities()).to(og.sim.device)  # (N, ctrl_dim)

        joint_pos_desired = cb.to_torch(goals["target_joint_pos"]).to(og.sim.device)  # (N, ctrl_dim)
        joint_vel_desired = cb.to_torch(goals["target_joint_vel"]).to(og.sim.device)  # (N, ctrl_dim)

        u = self.kp * (joint_pos_desired - current_joint_pos) + self.kd * (joint_vel_desired - current_joint_vel)

        if self.min_effort is not None and self.max_effort is not None:
            assert u.shape[-1] == self.max_effort.shape[-1] == self.min_effort.shape[-1]
            u = u.clip(self.min_effort, self.max_effort)

        return cb.from_torch(u)  # (N, control_dim)

    # NOTE: the pre-3.9.1 clip_control override clipped to the same control limits and copied every
    # index back, making it equivalent to the (now batched) base implementation. Dropped.

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

    def _to_tensor(self, input):
        if th.is_tensor(input):
            return input.to(th.Tensor())
        else:
            return th.tensor(input).to(th.Tensor())

    def _diagonalize_gain(self, gain: th.Tensor) -> th.Tensor:
        if gain.dim() == 1:
            return th.diag(gain)
        elif gain.dim() == 2:
            return gain
        else:
            raise ValueError(f"Gain tensor must be 1D or 2D, but got {gain.dim()}D.")

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