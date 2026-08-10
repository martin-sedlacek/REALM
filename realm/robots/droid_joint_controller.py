import torch as th
from omnigibson.controllers.controller_base import (
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


# Create module logger
log = create_module_logger(module_name=__name__)


class IndividualJointPDController(LocomotionController, ManipulationController, GripperController):
    """Task-space-weighted joint PD controller used by the DROID platform.

    Ported to the OG 3.9.1 controller contract, which is batched: one controller instance serves N
    group members, `_update_goal`/`compute_no_op_goal` take a member index, `compute_control` receives
    goals of shape (N, ...) and returns (N, control_dim), and robot state comes from
    `ControllableObjectViewAPI` keyed on `self.routing_path` rather than a per-step `control_dict`.

    The control law itself is unchanged from the pre-3.9.1 version:

        Kp = J^T Kx J + Kq;  Kd = J^T Kxd J + Kqd
        u  = Kp (q* - q) + Kd (qd* - qd)  [+ Coriolis/centrifugal compensation]

    REALM runs a single robot per environment, so the batch dimension is handled by looping over the
    view's rows; that keeps the per-robot math identical to what was validated in the paper rather
    than re-deriving it in batched form.
    """

    def __init__(
            self,
            control_freq,
            motor_type,  # This will be forced to 'effort' for hybrid control
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            link_name=None,  # eef link, needed to look up the task-space Jacobian
            Kq=None,  # Kq: Can be scalar, list, or torch.Tensor
            Kqd=None,  # For Kqd: Can be scalar, list, or torch.Tensor
            Kx=None,  # Kx: Cartesian P gain (scalar, list (for diagonal), or 6x6 tensor)
            Kxd=None,  # Kxd: Cartesian D gain (scalar, list (for diagonal), or 6x6 tensor)
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
        self._motor_type = motor_type.lower()
        self._use_impedances = True
        self._link_name = link_name

        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

        Kq = self._diagonalize_gain(self._to_tensor(Kq))
        Kqd = self._diagonalize_gain(self._to_tensor(Kqd))
        assert Kq.shape == Kqd.shape
        Kx = self._diagonalize_gain(self._to_tensor(Kx))
        Kxd = self._diagonalize_gain(self._to_tensor(Kxd))
        assert Kx.shape == th.Size([6, 6])
        assert Kxd.shape == th.Size([6, 6])

        # Plain tensors, not th.nn.Parameter: nothing here is ever optimized, and under OG 3.9.1 the
        # compute backend converts controls with `Tensor.numpy()`, which raises on grad-tracking
        # tensors. Values are identical -- only requires_grad differs.
        self.Kq = Kq.detach().to(og.sim.device)
        self.Kqd = Kqd.detach().to(og.sim.device)
        self.Kx = Kx.detach().to(og.sim.device)
        self.Kxd = Kxd.detach().to(og.sim.device)

        self.time_tracker = -1 # we update at the very beginning of compute_control, so this is 0 when controller is queried for the very first time
        self.cached_torque = None

    def _get_joint_positions(self):
        """(N, control_dim) current positions of this controller's DOFs, one row per group member."""
        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][:, self.dof_idx]

    def _get_joint_velocities(self):
        """(N, control_dim) current velocities of this controller's DOFs, one row per group member.

        `estimate=False` on purpose, and NOT the 3.9.1 idiom: every stock OmniGibson controller
        (joint, OSC, multi-finger gripper) passes `estimate=True`, which returns the one-step finite
        difference (pos - last_pos)/physics_dt rather than the physics engine's reported velocity.

        This controller is an *effort* impedance law, u = Kp(q* - q) + Kd(qd* - qd), so its damping
        term consumes the velocity directly, and the pre-3.9.1 version read the reported velocity via
        `control_dict["joint_velocity"]`. Feeding it the finite-difference estimate instead leaves a
        standing position error: measured on a 0.25 rad joint step, steady-state |q - cmd| is
        0.00858 rad with estimate=True vs 0.00009 rad with estimate=False -- a 95x degradation, worth
        several mm at the fingertips, which is the difference between closing on a 3 cm block and
        hovering beside it. Keep this False to preserve the validated 1.1.1 dynamics.
        """
        rows = self.view_row_indices
        return ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=False)[rows, :][
            :, self.dof_idx
        ]

    def _get_relative_jacobians(self):
        """(N, 6, control_dim) base-relative Jacobian of the eef link for this controller's DOFs."""
        rows = self.view_row_indices
        eef_body_idx = ControllableObjectViewAPI.get_link_index(self.routing_path, self._link_name)
        jac_all = ControllableObjectViewAPI.get_all_relative_jacobians(self.routing_path)  # (N, n_links, 6, n_dof)
        jac = jac_all[rows, eef_body_idx]  # (N, 6, n_dof_total)
        # Generalized DoFs may include extra base DoFs ahead of the actuated joints (floating base).
        base_dof_offset = max(jac.shape[-1] - ControllableObjectViewAPI.get_all_joint_positions(self.routing_path).shape[-1], 0)
        return jac[..., [idx + base_dof_offset for idx in self.dof_idx]]  # (N, 6, control_dim)

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
        Args:
            goals (Dict[str, Array]): batched goals, each of shape (N, control_dim). Must include
                "target_joint_pos" and "target_joint_vel".

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        self.time_tracker += 1

        current_joint_pos = cb.to_torch(self._get_joint_positions()).to(og.sim.device)  # (N, ctrl_dim)
        current_joint_vel = cb.to_torch(self._get_joint_velocities()).to(og.sim.device)  # (N, ctrl_dim)
        jacobians = cb.to_torch(self._get_relative_jacobians()).to(og.sim.device)  # (N, 6, ctrl_dim)

        joint_pos_desired = cb.to_torch(goals["target_joint_pos"]).to(og.sim.device)  # (N, ctrl_dim)
        joint_vel_desired = cb.to_torch(goals["target_joint_vel"]).to(og.sim.device)  # (N, ctrl_dim)

        if self._use_cc_compensation:
            rows = self.view_row_indices
            cc_forces = cb.to_torch(
                ControllableObjectViewAPI.get_all_coriolis_and_centrifugal_compensation_forces(self.routing_path)
            ).to(og.sim.device)[rows, :]

        # Gravity compensation. The pre-3.9.1 controller accepted `use_gravity_compensation` but never
        # applied it -- only the Coriolis term was ever added -- so with it left False (as the stock
        # DROID config does) a pure PD law carries a steady-state droop proportional to the gravity
        # torque. That is tolerable for droid.usd's link masses but not for heavier assets: the
        # robolab arm settles 0.2968 rad off the commanded panda_joint5 without this.
        if self._use_gravity_compensation:
            rows = self.view_row_indices
            grav_forces = cb.to_torch(
                ControllableObjectViewAPI.get_all_gravity_compensation_forces(self.routing_path)
            ).to(og.sim.device)[rows, :]

        # Indices into the generalized-force vectors for this controller's DOFs. The offset accounts
        # only for extra *base* DOFs on floating-base robots, so it is measured against the robot's
        # total joint count -- NOT against control_dim, which would shift the arm's slice by
        # (n_joints - n_arm_joints) and silently feed it the gripper's terms instead. On a 13-DOF
        # robolab DROID that mis-indexing demanded torques far past the wrist's +-12 Nm limit, so the
        # wrist saturated and settled ~0.29 rad off command.
        n_joint_dof = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path).shape[-1]
        if self._use_cc_compensation or self._use_gravity_compensation:
            ref = cc_forces if self._use_cc_compensation else grav_forces
            base_dof_offset = max(ref.shape[-1] - n_joint_dof, 0)
            comp_idx = [idx + base_dof_offset for idx in self.dof_idx]

        us = []
        for i in range(current_joint_pos.shape[0]):
            jacobian = jacobians[i]
            assert jacobian.shape == (6, self.control_dim), (
                f"Expected a (6, {self.control_dim}) Jacobian for link {self._link_name}, got {tuple(jacobian.shape)}"
            )

            Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
            Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd

            u = Kp @ (joint_pos_desired[i] - current_joint_pos[i]) + Kd @ (
                joint_vel_desired[i] - current_joint_vel[i]
            )

            # Add Coriolis / centrifugal compensation
            if self._use_cc_compensation:
                u = u + cc_forces[i][comp_idx]

            # Add gravity compensation
            if self._use_gravity_compensation:
                u = u + grav_forces[i][comp_idx]

            if self.min_effort is not None and self.max_effort is not None:
                assert u.shape == self.max_effort.shape == self.min_effort.shape
                u = u.clip(self.min_effort, self.max_effort)

            us.append(u)

        return cb.from_torch(th.stack(us))  # (N, control_dim)

    # NOTE: the pre-3.9.1 version overrode clip_control, but that override clipped to exactly the
    # same control limits and then copied every index back, making it equivalent to the base
    # implementation. 3.9.1's base version does the same clip on the batched (N, control_dim)
    # signal (and correctly leaves limitless position joints alone), so the override is dropped.

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
