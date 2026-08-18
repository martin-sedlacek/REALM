import numpy as np
import omnigibson as og  # For og.sim.device
import omnigibson.utils.transform_utils as T
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
from scipy.spatial.transform import Rotation as R

from realm.geometry import pose_diff
from realm.robots.gains import prepare_gain
from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver

IK_MODE_COMMAND_DIMS = {
    # 6DOF (x,y,z,ax,ay,az) control of pose, both position and orientation absolute
    "absolute_pose": 6,
    # 6DOF (dx,dy,dz,ax,ay,az), orientation given in absolute axis-angle coordinates
    "pose_absolute_ori": 6,
    # 6DOF (dx,dy,dz,dax,day,daz)
    "pose_delta_ori": 6,
    # 3DOF (dx,dy,dz), orientation held at its initial absolute value
    "position_fixed_ori": 3,
    # 3DOF (dx,dy,dz), orientation commanded as 0s, so it can drift over time
    "position_compliant_ori": 3,
    "cartesian_velocity": 6,
}

#: The modes that actually EXECUTE. The other four in IK_MODE_COMMAND_DIMS above are declared-only
#: and each dies mid-episode if selected: pose_absolute_ori / position_compliant_ori raise
#: NotImplementedError in _cartesian_velocity_command, position_fixed_ori reads a
#: `_fixed_quat_target` attribute nothing initialises (AttributeError), and cartesian_velocity
#: returns a torch tensor down a path the IK solver expects a list on. __init__ therefore rejects
#: them at CONSTRUCTION, where the config is in hand -- not on the first control step. Implement
#: and verify a mode before adding it here.
SUPPORTED_MODES = ("absolute_pose", "pose_delta_ori")


class DroidEndEffectorController(LocomotionController, ManipulationController, GripperController):
    """Cartesian end-effector controller for the DROID platform: IK, then joint-space impedance.

    A command is turned into a cartesian velocity for `RobotIKSolver` -- a dm_control MuJoCo model
    of the Franka arm under `realm/robots/robot_ik/`, NOT the URDF next to this file -- whose joint
    solution is then tracked by the same task-space-weighted impedance law as
    `droid_joint_controller.py`:

        Kp = J^T Kx J + Kq;  Kd = J^T Kxd J + Kqd
        u  = Kp (q* - q) + Kd (qd* - qd)  [+ Coriolis/centrifugal compensation]

    Only `absolute_pose` and `pose_delta_ori` are used by any config in `realm/config/robots/`.

    Ported to the OG 3.9.1 controller contract, which is batched: one controller instance serves N
    group members, `_update_goal` / `compute_no_op_goal` take a member index, `compute_control`
    receives goals of shape (N, ...) and returns (N, control_dim), and robot state comes from
    `ControllableObjectViewAPI` keyed on `self.routing_path` rather than a per-step `control_dict`.
    The IK solver and the pose helpers are inherently single-robot, so members are solved in turn;
    REALM runs one robot per environment, so N is 1 in practice.

    Two things this does NOT do, both deliberate:

    * It accepts `use_gravity_compensation` and never applies it, unlike
      `droid_joint_controller.py`, which does. Every EE-control config sets it False, so the flag
      has never had an effect on this controller.

      CHECKED AGAINST PRE-PORT AND DELIBERATELY LEFT ALONE (2026-08-16). The 1.1.1 controller
      (`~/projects/REALM/realm/robots/droid_ee_controller.py:46,62`) also only ASSIGNS
      `self._use_gravity_compensation` and never reads it -- the two occurrences in that file are
      the constructor default and the assignment, exactly as here. So this controller's behaviour
      is identical to the reference implementation, which is the definition of correct for
      anything under `realm/robots/`. Neither implementing the flag nor rejecting the key is a
      documentation fix: both would change a working controller. The same key on
      `droid_joint_controller.py` DOES have an effect there (that controller gained a real gravity
      term during the 3.9.1 port; pre-port it, too, only stored the flag), so one YAML key means
      two different things depending on which controller reads it. That asymmetry is recorded
      here rather than removed.
    * It does not pre-clamp the commanded cartesian delta. `RobotIKSolver` already clamps it to the
      same limits this controller would use -- 0.075 m linear, 0.15 rad angular -- on the way
      through `cartesian_delta_to_velocity` / `cartesian_velocity_to_delta`.

    The pre-3.9.1 version also overrode `clip_control`, but that override clipped to exactly the
    same control limits and then copied every index back, making it equivalent to the (now batched)
    base implementation. Dropped.
    """

    def __init__(
            self,
            control_freq,
            motor_type,  # Stored as given; control_type is EFFORT regardless (base-class clipping
                         # keys off control_type, so the two disagreeing is currently harmless --
                         # unlike the joint controllers, which force motor_type to 'effort').
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
        """See `realm/config/robots/DROID_ee_control.yaml` for the values these take in practice.

        `height_offset` is a property of the ASSET, not of this controller: it is the height of the
        arm base above the robot prim, and it is added to a commanded z before that z is compared
        against the eef pose relative to the robot prim. The default suits `droid_mounted.usd`
        (panda_link0 at z = 0.86444 above /panda); the bare robolab arm has panda_link0 at the robot
        prim, so its config sets 0.0. See DROID_robolab_v2_ee_control.yaml for the measurements.
        """
        self._link_name = link_name
        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        self.height_offset = height_offset

        assert mode in SUPPORTED_MODES, (
            f"Unsupported ik mode {mode!r}. Implemented and verified modes: {SUPPORTED_MODES}. "
            f"({sorted(set(IK_MODE_COMMAND_DIMS) - set(SUPPORTED_MODES))} are declared but broken "
            f"-- see SUPPORTED_MODES in realm/robots/droid_ee_controller.py.)"
        )

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.workspace_pose_limiter = workspace_pose_limiter
        self.mode = mode

        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

        self.Kq = prepare_gain(Kq)
        self.Kqd = prepare_gain(Kqd)
        self.Kx = prepare_gain(Kx)
        self.Kxd = prepare_gain(Kxd)
        assert self.Kq.shape == self.Kqd.shape
        assert self.Kx.shape == th.Size([6, 6])
        assert self.Kxd.shape == th.Size([6, 6])

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
            ControllableObjectViewAPI.get_all_joint_velocities(
            self.routing_path, estimate=False  # reported velocity, not the finite-difference
        )[rows, :][
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
        """Turn one member's command into the goal `compute_control` will track.

        Every declared goal key must carry a real array: OG 3.9.1 stores goals in preallocated
        per-member buffers (`_goals[k][idx] = cb.copy(v)`). The pre-3.9.1 contract handed the dict
        over as-is and tolerated Nones for the keys the active mode does not use, so those are
        filled with zeros of the declared shape instead. `compute_control` only reads the keys its
        own mode sets, so the zeros are never consumed.

        `target_pos = command[:3]` IS A VIEW, AND THAT IS LEFT AS IT IS (checked 2026-08-16).
        `target_pos[-1] += self.height_offset` therefore writes the offset back into `command`,
        and `command` is the array `BaseController.update_goal` handed down --
        `preprocessed = self._preprocess_command(command)`, which returns its input UNCHANGED when
        `self._command_input_limits is None` (`controller_base.py:318`). `absolute_pose` mode
        asserts exactly that (see `__init__`), so in the one mode where the `+=` executes, the
        write does reach the caller's command array. The aliasing is real, not theoretical.

        It is also pre-port behaviour, byte for byte:
        `~/projects/REALM/realm/robots/droid_ee_controller.py:113-115` is the same two lines with
        the same view semantics. Whatever depends on it has depended on it since before the 3.9.1
        port, and this controller is a reference implementation that worked as intended. Copying
        (`command[:3].clone()`) would be a behaviour change to a controller, undertaken to fix a
        smell rather than an observed defect, so it is NOT made here. Anyone who does change it
        must first establish what reads `command` after `update_goal` returns.
        """
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]
        command = cb.to_torch(command).to(og.sim.device)

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
        all_joint_pos = self._get_joint_positions()  # (N, ctrl_dim)
        all_joint_vel = self._get_joint_velocities()  # (N, ctrl_dim)
        all_jacobians = self._get_relative_jacobians()  # (N, 6, ctrl_dim)
        all_pos_current, all_quat_current = self._get_eef_pose_relative()
        all_cc_forces = self._get_cc_forces() if self._use_cc_compensation else None

        us = []
        for i in range(all_joint_pos.shape[0]):
            jacobian = all_jacobians[i]
            assert jacobian.shape == (6, self.control_dim), (
                f"Expected a (6, {self.control_dim}) Jacobian for link {self._link_name}, got {tuple(jacobian.shape)}"
            )
            goal = {k: (None if v is None else cb.to_torch(v[i]).to(og.sim.device)) for k, v in goals.items()}

            joint_pos_desired = self._desired_joint_positions(
                goal, all_joint_pos[i], all_joint_vel[i], all_pos_current[i], all_quat_current[i]
            )
            joint_vel_desired = th.zeros(self.control_dim).to(og.sim.device)

            Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
            Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd
            u = Kp @ (joint_pos_desired - all_joint_pos[i]) + Kd @ (joint_vel_desired - all_joint_vel[i])

            # Add Coriolis / centrifugal compensation
            if self._use_cc_compensation:
                u = u + all_cc_forces[i]

            if self.min_effort is not None and self.max_effort is not None:
                assert u.shape == self.max_effort.shape == self.min_effort.shape
                u = u.clip(self.min_effort, self.max_effort)

            us.append(u)

        return cb.from_torch(th.stack(us))  # (N, control_dim)

    def _desired_joint_positions(self, goal, current_joint_pos, current_joint_vel,
                                 pos_current, quat_current):
        """Joint targets that carry this member's eef to its goal pose, through the IK solver.

        A pose delta below the IK solver's own accuracy is held rather than solved: solving it makes
        the joints drift even when the commanded delta is zero.
        """
        if (
            self.mode not in ["cartesian_velocity"]
            and th.allclose(pos_current, goal["target_pos"], atol=1e-4)
            and th.allclose(quat_current, goal["target_quat"], atol=1e-4)
        ):
            return current_joint_pos

        cartesian_velocity = self._cartesian_velocity_command(goal, pos_current, quat_current)
        joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(
            cartesian_velocity,
            robot_state={
                "joint_positions": current_joint_pos,
                "joint_velocities": current_joint_vel,
            },
        ).tolist()
        joint_delta = self._ik_solver.joint_velocity_to_delta(joint_velocity)
        joint_position = (joint_delta + np.array(current_joint_pos.cpu())).tolist()
        return th.tensor(joint_position, dtype=th.float32, device=og.sim.device)

    def _cartesian_velocity_command(self, goal, pos_current, quat_current):
        """The 6-D cartesian velocity the IK solver should track, for this controller's mode."""
        if self.mode == "cartesian_velocity":
            return th.cat([goal["target_cartesian_pos_vel"], goal["target_cartesian_rot_vel"]])

        if self.mode == "pose_delta_ori":
            dpos = goal["target_pos"] - goal["target_pos_relative"]
            cartesian_delta = th.cat([dpos, goal["target_rpy_relative"]])
        elif self.mode == "absolute_pose":
            rpy_current = th.from_numpy(
                R.from_quat(quat_current.cpu().numpy()).as_euler("xyz")).to(og.sim.device)
            cartesian_position = th.cat([goal["target_pos"], goal["target_rpy"]])
            current_cartesian_position = th.cat([pos_current, rpy_current])
            cartesian_delta = th.from_numpy(pose_diff(cartesian_position, current_cartesian_position))
        else:
            raise NotImplementedError()

        return self._ik_solver.cartesian_delta_to_velocity(cartesian_delta).tolist()

    def compute_no_op_goal(self, controller_idx):
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]
        rpy_relative = th.from_numpy(R.from_quat(quat_relative.cpu().numpy()).as_euler('xyz')).to(pos_relative.device)

        goal_dict = dict(
            target_pos=pos_relative,
            target_quat=quat_relative,
            target_rpy=rpy_relative,
            target_pos_relative=pos_relative.clone(),
            target_quat_relative=quat_relative,
            target_rpy_relative=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_pos_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
            target_cartesian_rot_vel=th.zeros(3, dtype=th.float32, device=pos_relative.device),
        )
        # Goals are stored in the backend array type, not torch.
        return {k: cb.from_torch(v) for k, v in goal_dict.items()}

    def _compute_no_op_command(self, controller_idx):
        """A command that asks for no motion: zero deltas, or the current pose in absolute modes."""
        all_pos_relative, all_quat_relative = self._get_eef_pose_relative()
        pos_relative = all_pos_relative[controller_idx]
        quat_relative = all_quat_relative[controller_idx]

        command = th.zeros(6, dtype=th.float32, device=pos_relative.device)
        if self.mode == "absolute_pose":
            command[:3] = pos_relative
        if self.mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = T.quat2axisangle(quat_relative)
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
