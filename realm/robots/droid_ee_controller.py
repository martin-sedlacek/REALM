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


# Create module logger
log = create_module_logger(module_name=__name__)

IK_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
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
            workspace_pose_limiter=None
    ):
        motor_type = "effort"
        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        assert mode in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.mode = mode
        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = f"eef_0"

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

    def _update_goal(self, command, control_dict):
        # Grab important info from control dict
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

        # Convert position command to absolute values if needed
        if self.mode == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

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
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.axisangle2quat(command[3:6])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = T.quat2mat(T.axisangle2quat(command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

        goal_dict = dict(
            target_pos=target_pos,
            target_quat=target_quat,
        )

        return goal_dict

    def compute_control(self, goal_dict, control_dict):
        self.time_tracker += 1
        # if self.time_tracker % gm.DEFAULT_SIM_STEP_FREQ != 0:
        #     return self.cached_torque
        current_joint_pos = control_dict["joint_position"][self.dof_idx].to(og.sim.device)
        current_joint_vel = control_dict["joint_velocity"][self.dof_idx].to(og.sim.device)

        #--------------------------------------------------------------------------------
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]
        target_pos = goal_dict["target_pos"]
        target_quat = goal_dict["target_quat"]

        # If the delta is really small, we just keep the current joint position. This avoids joint
        # drift caused by IK solver inaccuracy even when zero delta actions are provided.
        if th.allclose(pos_relative, target_pos, atol=1e-4) and th.allclose(quat_relative, target_quat, atol=1e-4):
            joint_pos_desired = current_joint_pos
        else:
            # Compute the pose error. Note that this is computed NOT in the EEF frame but still
            # in the base frame.
            pos_err = target_pos - pos_relative
            ori_err = orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))
            err = th.cat([pos_err, ori_err])

            # Use the jacobian to compute a local approximation
            j_eef = control_dict[f"{self.task_name}_jacobian_relative"][:, self.dof_idx]
            j_eef_pinv = th.linalg.pinv(j_eef)
            delta_j = j_eef_pinv @ err
            target_joint_pos = current_joint_pos + delta_j

            # Clip values to be within the joint limits
            joint_pos_desired = target_joint_pos.clamp(
                min=self._control_limits[ControlType.get_type("position")][0][self.dof_idx],
                max=self._control_limits[ControlType.get_type("position")][1][self.dof_idx],
            )
        #--------------------------------------------------------------------------------
        # Assuming arm name is 0 and there is only one arm
        jacobian = control_dict["eef_0_jacobian_relative"].to(og.sim.device)[:, :7]

        assert jacobian.shape == (6, 7)

        #joint_pos_desired = goal_dict["target_joint_pos"].to(og.sim.device)
        #joint_vel_desired = goal_dict["target_joint_vel"].to(og.sim.device)
        joint_vel_desired = th.zeros(7)  # TODO: maybe this also needs to have gripper

        Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
        Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd

        u_feedback = Kp @ (joint_pos_desired - current_joint_pos) + Kd @ (joint_vel_desired - current_joint_vel)
        u_feedforward = th.zeros_like(u_feedback)
        u = u_feedback + self._to_tensor(u_feedforward[:7]).to(og.sim.device)

        # # Add Coriolis / centrifugal compensation
        if self._use_cc_compensation:
            u += control_dict["cc_force"][self.dof_idx].to(og.sim.device)

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
        return dict(
            target_pos=control_dict[f"{self.task_name}_pos_relative"],
            target_quat=control_dict[f"{self.task_name}_quat_relative"],
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