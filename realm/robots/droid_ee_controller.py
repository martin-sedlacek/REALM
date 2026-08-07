from math import floor

import torch as th
from omnigibson.controllers.controller_base import (
    BaseController,
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.utils.ui_utils import create_module_logger
import omnigibson as og  # For og.sim.device
from omnigibson.macros import gm
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.transform_utils import orientation_error  # moved from control_utils in OG 3.9.1
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
import omnigibson.utils.transform_utils as T
import numpy as np
from realm.helpers import add_poses, pose_diff
from scipy.spatial.transform import Rotation as R
from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver

# Create module logger
log = create_module_logger(module_name=__name__)

IK_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
    "cartesian_velocity": 6
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())


class DroidEndEffectorController(LocomotionController, ManipulationController, GripperController):
    def __init__(
            self,
            control_freq,
            motor_type,  # This will be forced to 'effort' for hybrid control
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            Kq=None,  # Kq: Can be scalar, list, or torch.Tensor
            Kqd=None,  # For Kqd: Can be scalar, list, or torch.Tensor
            Kx=None,  # Kx: Cartesian P gain (scalar, list (for diagonal), or 6x6 tensor)
            Kxd=None,  # Kxd: Cartesian D gain (scalar, list (for diagonal), or 6x6 tensor)
            use_impedances=False,
            use_gravity_compensation=False,
            use_cc_compensation=True,
            use_delta_commands=False,  # Delta commands are less common for torque control
            compute_delta_in_quat_space=None,  # Delta commands are less common for torque control
            mode="pose_delta_ori",
            workspace_pose_limiter=None,
            max_effort=None,
            min_effort=None,
            height_offset=0.87,
            link_name=None,  # eef link; OG 3.9.1 reads eef state/Jacobian by link name
            **kwargs,
    ):
        self._link_name = link_name
        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        self.height_offset = height_offset

        assert mode in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = f"eef_0"
        self.mode = mode

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

        # Plain tensors, not th.nn.Parameter: nothing is optimized here, and OG 3.9.1's compute
        # backend converts controls with Tensor.numpy(), which raises on grad-tracking tensors.
        self.Kq = Kq.detach().to(og.sim.device)
        self.Kqd = Kqd.detach().to(og.sim.device)
        self.Kx = Kx.detach().to(og.sim.device)
        self.Kxd = Kxd.detach().to(og.sim.device)

        urdf_path = f"/app/realm/robots/panda_robotiq/panda_arm.urdf"
        self.time_tracker = -1 # we update at the very beginning of compute_control, so this is 0 when controller is queried for the very first time
        self.cached_torque = None

        self._ik_solver = RobotIKSolver()

    # ---- OG 3.9.1 state access -------------------------------------------------------------
    # The per-step `control_dict` is gone; robot state is read from ControllableObjectViewAPI,
    # batched over the controller group's members and indexed by self.view_row_indices.

    def _get_joint_positions(self):
        rows = self.view_row_indices
        return cb.to_torch(
            ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][:, self.dof_idx]
        ).to(og.sim.device)

    def _get_joint_velocities(self):
        rows = self.view_row_indices
        return cb.to_torch(
            ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)[rows, :][
                :, self.dof_idx
            ]
        ).to(og.sim.device)

    def _get_eef_pose_relative(self):
        """((N, 3), (N, 4)) eef position/quaternion in the robot base frame."""
        rows = self.view_row_indices
        pos, quat = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
            self.routing_path, self._link_name
        )
        return cb.to_torch(pos)[rows], cb.to_torch(quat)[rows]

    def _get_relative_jacobians(self):
        """(N, 6, control_dim) base-relative Jacobian of the eef link for this controller's DOFs."""
        rows = self.view_row_indices
        eef_body_idx = ControllableObjectViewAPI.get_link_index(self.routing_path, self._link_name)
        jac = cb.to_torch(ControllableObjectViewAPI.get_all_relative_jacobians(self.routing_path))[rows, eef_body_idx]
        n_joint_dof = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path).shape[-1]
        base_dof_offset = max(jac.shape[-1] - n_joint_dof, 0)
        return jac[..., [idx + base_dof_offset for idx in self.dof_idx]].to(og.sim.device)

    def _get_cc_forces(self):
        rows = self.view_row_indices
        forces = cb.to_torch(
            ControllableObjectViewAPI.get_all_coriolis_and_centrifugal_compensation_forces(self.routing_path)
        )[rows]
        n_joint_dof = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path).shape[-1]
        base_dof_offset = max(forces.shape[-1] - n_joint_dof, 0)
        return forces[:, [idx + base_dof_offset for idx in self.dof_idx]].to(og.sim.device)

    def _update_goal(self, controller_idx, command):
        # Grab important info from the view (was the per-step control_dict pre-3.9.1)
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]
        command = cb.to_torch(command).to(og.sim.device)

        #command[:3], command[3:6] = self._scale_cartesian_6d_velocity(command[:3], command[3:6])

        # Convert position command to absolute values if needed
        if self.mode == "absolute_pose":
            target_pos = command[:3]
            target_pos[-1] += self.height_offset
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        target_rpy_relative = None
        target_rpy = None
        target_cartesian_pos_vel = None
        target_cartesian_rot_vel = None
        target_quat = None
        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._fixed_quat_target is None:
                self._fixed_quat_target = quat_relative if (self._goal is None) else self._goal["target_quat"]
            target_quat = self._fixed_quat_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori" or self.mode == "absolute_pose":
            if command.shape[-1] < 6:
                raise ValueError(
                    f"Command for mode {self.mode} has fewer than 6 dimensions ({command.shape[-1]}). "
                    "Expected 6 dimensions (x,y,z,ax,ay,az) but RPY components are missing."
                )
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.euler2quat(command[3:6])
            target_rpy = command[3:6]
        elif self.mode == "cartesian_velocity":
            target_cartesian_pos_vel = command[:3]
            target_cartesian_rot_vel = command[3:6]
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            target_rpy_relative = command[3:6]
            dori = T.quat2mat(T.euler2quat(command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            # No control_dict in OG 3.9.1; limiters get the eef pose they would have read from it.
            target_pos, target_quat = self.workspace_pose_limiter(
                target_pos, target_quat, dict(pos_relative=pos_relative, quat_relative=quat_relative)
            )

        goal_dict = dict(
            target_pos=target_pos,
            target_quat=target_quat,
            target_rpy=target_rpy,
            target_pos_relative=pos_relative,
            target_quat_relative=quat_relative,
            target_rpy_relative=target_rpy_relative,
            target_cartesian_pos_vel=target_cartesian_pos_vel,
            target_cartesian_rot_vel=target_cartesian_rot_vel,
        )
        # OG 3.9.1 stores goals in preallocated per-member buffers (`_goals[k][idx] = cb.copy(v)`),
        # so every declared goal key must carry a real array. The pre-3.9.1 contract handed the dict
        # over as-is and tolerated the Nones this method leaves for keys the active mode does not
        # use; fill those with zeros of the declared shape instead. compute_control only reads the
        # keys its own mode sets, so the zeros are never consumed.
        goal_shapes = self._get_goal_shapes()
        return {
            k: (cb.zeros(goal_shapes[k]) if v is None else cb.from_torch(v))
            for k, v in goal_dict.items()
        }

    def compute_control(self, goals):
        """
        Args:
            goals (Dict[str, Array]): batched goals, each of shape (N, *goal_shape)

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        self.time_tracker += 1

        all_current_joint_pos = self._get_joint_positions()  # (N, ctrl_dim)
        all_current_joint_vel = self._get_joint_velocities()  # (N, ctrl_dim)
        all_jacobians = self._get_relative_jacobians()  # (N, 6, ctrl_dim)
        all_pos_current, all_quat_current = self._get_eef_pose_relative()
        all_cc_forces = self._get_cc_forces() if self._use_cc_compensation else None

        # The IK solver and the pose helpers below are inherently single-robot, so each group member
        # is solved in turn. REALM runs one robot per environment, so N is 1 in practice.
        us = []
        for i in range(all_current_joint_pos.shape[0]):
            current_joint_pos = all_current_joint_pos[i]
            current_joint_vel = all_current_joint_vel[i]
            jacobian = all_jacobians[i]
            assert jacobian.shape == (6, self.control_dim), (
                f"Expected a (6, {self.control_dim}) Jacobian for link {self._link_name}, got {tuple(jacobian.shape)}"
            )

            goal = {k: (None if v is None else cb.to_torch(v[i]).to(og.sim.device)) for k, v in goals.items()}

            pos_current = all_pos_current[i]
            quat_current = all_quat_current[i]
            rpy_current = th.from_numpy(R.from_quat(quat_current.cpu().numpy()).as_euler("xyz")).to(og.sim.device)

            # If the delta is really small, we just keep the current joint position. This avoids joint
            # drift caused by IK solver inaccuracy even when zero delta actions are provided.
            if (
                self.mode not in ["cartesian_velocity"]
                and th.allclose(pos_current, goal["target_pos"], atol=1e-4)
                and th.allclose(quat_current, goal["target_quat"], atol=1e-4)
            ):
                joint_pos_desired = current_joint_pos
            else:
                action_dict = {}
                if self.mode == "cartesian_velocity":
                    action_dict["cartesian_velocity"] = th.cat(
                        [goal["target_cartesian_pos_vel"], goal["target_cartesian_rot_vel"]]
                    )
                    action_dict["cartesian_delta"] = self._ik_solver.cartesian_velocity_to_delta(
                        action_dict["cartesian_velocity"]
                    )
                elif self.mode == "pose_delta_ori":
                    dpos = goal["target_pos"] - goal["target_pos_relative"]
                    action_dict["cartesian_delta"] = th.cat([dpos, goal["target_rpy_relative"]])
                    cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(action_dict["cartesian_delta"])
                    action_dict["cartesian_velocity"] = cartesian_velocity.tolist()
                elif self.mode == "absolute_pose":
                    action_dict["cartesian_position"] = th.cat([goal["target_pos"], goal["target_rpy"]])
                    current_cartesian_position = th.cat([pos_current, rpy_current])
                    cartesian_delta = th.from_numpy(
                        pose_diff(action_dict["cartesian_position"], current_cartesian_position)
                    )
                    cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                    action_dict["cartesian_velocity"] = cartesian_velocity.tolist()
                else:
                    raise NotImplementedError()

                action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                    action_dict["cartesian_velocity"],
                    robot_state={
                        "joint_positions": current_joint_pos,
                        "joint_velocities": current_joint_vel,
                    },
                ).tolist()
                joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
                action_dict["joint_position"] = (joint_delta + np.array(current_joint_pos.cpu())).tolist()
                joint_pos_desired = th.tensor(action_dict["joint_position"], dtype=th.float32, device=og.sim.device)

            joint_vel_desired = th.zeros(self.control_dim).to(og.sim.device)

            Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
            Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd

            u = Kp @ (joint_pos_desired - current_joint_pos) + Kd @ (joint_vel_desired - current_joint_vel)

            # Add Coriolis / centrifugal compensation
            if self._use_cc_compensation:
                u = u + all_cc_forces[i]

            if self.min_effort is not None and self.max_effort is not None:
                assert u.shape == self.max_effort.shape == self.min_effort.shape
                u = u.clip(self.min_effort, self.max_effort)

            us.append(u)

        return cb.from_torch(th.stack(us))  # (N, control_dim)

    # NOTE: the pre-3.9.1 clip_control override clipped to the same limits and copied every index
    # back, making it equivalent to the (now batched) base implementation. Dropped.

    def compute_no_op_goal(self, controller_idx):
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]
        rpy_relative = th.from_numpy(R.from_quat(quat_relative.cpu().numpy()).as_euler('xyz')).to(pos_relative.device)

        goal_dict = dict(
            target_pos=pos_relative,
            target_quat=quat_relative,
            target_rpy=rpy_relative,
            target_pos_relative=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_quat_relative=quat_relative,
            target_rpy_relative=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_pos_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_rot_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
        )
        # Goals are stored in the backend array type, not torch.
        return {k: cb.from_torch(v) for k, v in goal_dict.items()}

    def _compute_no_op_command(self, controller_idx):
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]

        command = th.zeros(6, dtype=th.float32, device=pos_relative.device)

        # Handle position
        if self.mode == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass

        # Handle orientation
        if self.mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = T.quat2axisangle(quat_relative)
        else:
            # For these modes, we don't need to add orientation to the command
            pass

        return cb.from_torch(command)

    def _get_goal_shapes(self):
        return dict(
            target_pos=(3,),
            target_quat=(4,),
            target_rpy=(3,),
            target_pos_relative=(3,),
            target_quat_relative=(4,),
            target_rpy_relative=(3,),
            target_cartesian_pos_vel=(3,),
            target_cartesian_rot_vel=(3,),
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
        return IK_MODE_COMMAND_DIMS[self.mode]

    def _scale_cartesian_6d_velocity(self, lin_vel, rot_vel):
        max_lin_delta = 0.075
        max_rot_delta = 0.15
        lin_vel_norm = th.linalg.norm(lin_vel)
        rot_vel_norm = th.linalg.norm(rot_vel)
        if lin_vel_norm > max_lin_delta:
            lin_vel = lin_vel * max_lin_delta / lin_vel_norm
        if rot_vel_norm > max_rot_delta:
            rot_vel = rot_vel * max_rot_delta / rot_vel_norm
        return lin_vel, rot_vel