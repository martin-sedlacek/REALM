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
from omnigibson.utils.control_utils import orientation_error
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
            height_offset=0.87
    ):
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

        self.Kq = th.nn.Parameter(Kq).to(og.sim.device)
        self.Kqd = th.nn.Parameter(Kqd).to(og.sim.device)
        self.Kx = th.nn.Parameter(Kx).to(og.sim.device)
        self.Kxd = th.nn.Parameter(Kxd).to(og.sim.device)

        urdf_path = f"/app/realm/robots/panda_robotiq/panda_arm.urdf"
        self.time_tracker = -1 # we update at the very beginning of compute_control, so this is 0 when controller is queried for the very first time
        self.cached_torque = None

        self._ik_solver = RobotIKSolver()

    def _update_goal(self, command, control_dict):
        # Grab important info from control dict
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

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
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

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
        return goal_dict

    def compute_control(self, goal_dict, control_dict):
        self.time_tracker += 1
        current_joint_pos = control_dict["joint_position"][self.dof_idx].to(og.sim.device)
        current_joint_vel = control_dict["joint_velocity"][self.dof_idx].to(og.sim.device)

        # Assuming arm name is 0 and there is only one arm
        jacobian = control_dict["eef_0_jacobian_relative"].to(og.sim.device)[:, :7]

        assert jacobian.shape == (6, 7)
        # for n in control_dict.get_fcn_names():
        #     print(n, control_dict[n])

        #--------------------------------------------------------------------------------
        pos_current = control_dict[f"{self.task_name}_pos_relative"]

        quat_current = control_dict[f"{self.task_name}_quat_relative"]
        rpy_current = th.from_numpy(R.from_quat(quat_current.numpy()).as_euler('xyz'))

        # If the delta is really small, we just keep the current joint position. This avoids joint
        # drift caused by IK solver inaccuracy even when zero delta actions are provided.
        if self.mode not in ["cartesian_velocity"] and th.allclose(pos_current, goal_dict["target_pos"], atol=1e-4) and th.allclose(quat_current, goal_dict["target_quat"], atol=1e-4):
            joint_pos_desired = current_joint_pos
        else:
            action_dict = {}
            if self.mode == "cartesian_velocity":
                action_dict["cartesian_velocity"] = th.cat([goal_dict["target_cartesian_pos_vel"], goal_dict["target_cartesian_rot_vel"]])
                action_dict["cartesian_delta"] = self._ik_solver.cartesian_velocity_to_delta(action_dict["cartesian_velocity"])
            elif self.mode == "pose_delta_ori":
                dpos = goal_dict["target_pos"] - goal_dict["target_pos_relative"]
                action_dict["cartesian_delta"] = th.cat([dpos, goal_dict["target_rpy_relative"]])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(action_dict["cartesian_delta"])
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()
            elif self.mode == "absolute_pose":
                action_dict["cartesian_position"] = th.cat([goal_dict["target_pos"], goal_dict["target_rpy"]])
                current_cartesian_position = th.cat([pos_current, rpy_current])
                cartesian_delta = th.from_numpy(pose_diff(action_dict["cartesian_position"], current_cartesian_position))
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()
            else:
                raise NotImplementedError()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state={
                    "joint_positions": current_joint_pos,
                    "joint_velocities": current_joint_vel
                }
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (joint_delta + np.array(current_joint_pos)).tolist()
            joint_pos_desired = th.tensor(action_dict["joint_position"], dtype=th.float32, device=og.sim.device)

        #--------------------------------------------------------------------------------
        joint_vel_desired = th.zeros(7).to(og.sim.device)

        Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
        Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd

        # Ensure current_joint_vel is a tensor on the correct device
        if isinstance(current_joint_vel, list):
             current_joint_vel = th.tensor(current_joint_vel, dtype=th.float32, device=og.sim.device)

        u_feedback = Kp @ (joint_pos_desired - current_joint_pos) + Kd @ (joint_vel_desired - current_joint_vel)
        u_feedforward = th.zeros_like(u_feedback)
        u = u_feedback + self._to_tensor(u_feedforward[:7]).to(og.sim.device)

        # # Add Coriolis / centrifugal compensation
        if self._use_cc_compensation:
            u += control_dict["cc_force"][self.dof_idx].to(og.sim.device)

        if self.min_effort is not None and self.max_effort is not None:
            assert u.shape == self.max_effort.shape == self.min_effort.shape
            u = u.clip(self.min_effort, self.max_effort)

        return u

    def clip_control(self, control):
        clipped_control = control.clip(
            self._control_limits[self.control_type][0][self.dof_idx],
            self._control_limits[self.control_type][1][self.dof_idx],
        )

        idx = [True] * self.control_dim

        control_copy = control.clone()
        control_copy[idx] = clipped_control[idx]
        return control_copy

    def compute_no_op_goal(self, control_dict):
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]
        rpy_relative = th.from_numpy(R.from_quat(quat_relative.cpu().numpy()).as_euler('xyz')).to(pos_relative.device)

        return dict(
            target_pos=pos_relative,
            target_quat=quat_relative,
            target_rpy=rpy_relative,
            target_pos_relative=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_quat_relative=quat_relative,
            target_rpy_relative=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_pos_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_rot_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
        )

    def _compute_no_op_action(self, control_dict):
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

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

        return command

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

    def is_grasping(self):
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