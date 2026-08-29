"""Differential IK for the Franka arm, via dm_robotics' Cartesian6dVelocityEffector.

UNIT SYSTEM -- read this before touching any conversion below. "Velocity" here does NOT mean m/s
or rad/s: every velocity is NORMALISED to [-1, 1], the range the dm_robotics effector takes as
`ctrl`. A "delta" is the physical per-control-step displacement (metres / radians). The two are
related by fixed scalings:

    cartesian_delta = cartesian_velocity * [MAX_LIN_DELTA x3, MAX_ROT_DELTA x3]
    joint_delta     = joint_velocity * MAX_JOINT_DELTA

so every `*_velocity_to_delta` / `*_delta_to_velocity` method is one multiply or divide, plus
norm-clipping into the unit ball. The solve itself (`cartesian_velocity_to_joint_velocity`) is:
velocity -> delta -> MuJoCo effector ctrl -> back to velocity; its caller
(droid_ee_controller._desired_joint_positions) then converts that velocity back to a delta and
integrates it onto the current joint positions.

The MuJoCo model under `robot_ik/franka/` is the solver's OWN copy of the arm, updated from the
live OmniGibson joint state each solve -- it is not the simulated robot.
"""
import numpy as np
from dm_control import mjcf
from dm_robotics.moma.effectors import arm_effector, cartesian_6d_velocity_effector

from realm.robots.robot_ik.arm import FrankaArm

#: Largest per-control-step joint displacement (radians), per joint. Uniform across the seven
#: joints; MAX_JOINT_DELTA (its max) is the single scale used for the delta<->velocity mapping.
RELATIVE_MAX_JOINT_DELTA = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
MAX_JOINT_DELTA = RELATIVE_MAX_JOINT_DELTA.max()

#: Largest per-control-step end-effector displacement: metres / radians. These two are the safety
#: clamp on every commanded cartesian motion -- droid_ee_controller deliberately does NOT pre-clamp
#: because everything passes through these on the way to the solver.
MAX_LIN_DELTA = 0.075
MAX_ROT_DELTA = 0.15

#: Must match `control_freq` in the EE-control robot YAMLs (DROID_ee_control.yaml: 15). Only sets
#: the effector's integration timestep; nothing checks the two agree, so a 5 Hz or 30 Hz platform
#: needs this changed alongside its YAML.
CONTROL_HZ = 15

#: dm_robotics ControlParams tuning, inherited from the DROID codebase this was lifted from and
#: never re-tuned for REALM. Provenance unrecorded: treat as "works for the Franka at 15 Hz".
NULLSPACE_GAIN = 0.025
REGULARIZATION_WEIGHT = 1e-2
MIN_DISTANCE_FROM_JOINT_LIMIT = 0.3  # radians
JOINT_LIMIT_VELOCITY_SCALE = 0.95
MAX_CONTROL_ITERATIONS = 300  # per solve; on non-convergence the effector returns its best ctrl


class RobotIKSolver:


    def __init__(self):
        self.relative_max_joint_delta = RELATIVE_MAX_JOINT_DELTA
        self.max_joint_delta = MAX_JOINT_DELTA
        self.max_lin_delta = MAX_LIN_DELTA
        self.max_rot_delta = MAX_ROT_DELTA
        self.control_hz = CONTROL_HZ

        self._arm = FrankaArm()
        self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)
        self._effector = arm_effector.ArmEffector(arm=self._arm, action_range_override=None, robot_name=self._arm.name)

        self._effector_model = cartesian_6d_velocity_effector.ModelParams(self._arm.wrist_site, self._arm.joints)

        self._effector_control = cartesian_6d_velocity_effector.ControlParams(
            control_timestep_seconds=1 / self.control_hz,
            max_lin_vel=self.max_lin_delta,
            max_rot_vel=self.max_rot_delta,
            joint_velocity_limits=self.relative_max_joint_delta,
            nullspace_joint_position_reference=[0] * 7,
            nullspace_gain=NULLSPACE_GAIN,
            regularization_weight=REGULARIZATION_WEIGHT,
            enable_joint_position_limits=True,
            minimum_distance_from_joint_position_limit=MIN_DISTANCE_FROM_JOINT_LIMIT,
            joint_position_limit_velocity_scale=JOINT_LIMIT_VELOCITY_SCALE,
            max_cartesian_velocity_control_iterations=MAX_CONTROL_ITERATIONS,
            max_nullspace_control_iterations=MAX_CONTROL_ITERATIONS,
        )

        self._cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
            self._arm.name, self._effector, self._effector_model, self._effector_control
        )
        self._cart_effector_6d.after_compile(self._arm.mjcf_model, self._physics)

    def cartesian_velocity_to_joint_velocity(self, cartesian_velocity, robot_state):
        """Solve one differential-IK step: normalised cartesian velocity -> normalised joint velocity.

        @robot_state carries "joint_positions"/"joint_velocities" of the LIVE robot, written onto
        the solver's own MuJoCo model before the solve.
        """
        cartesian_delta = self.cartesian_velocity_to_delta(cartesian_velocity)
        qpos = np.array(robot_state["joint_positions"])
        qvel = np.array(robot_state["joint_velocities"])

        self._arm.update_state(self._physics, qpos, qvel)
        self._cart_effector_6d.set_control(self._physics, cartesian_delta)
        joint_delta = self._physics.bind(self._arm.actuators).ctrl.copy()

        return self.joint_delta_to_velocity(joint_delta)

    def cartesian_velocity_to_delta(self, cartesian_velocity):

        cartesian_velocity = np.asarray(cartesian_velocity)

        lin_vel, rot_vel = cartesian_velocity[:3], cartesian_velocity[3:6]

        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)

        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm

        lin_delta = lin_vel * self.max_lin_delta
        rot_delta = rot_vel * self.max_rot_delta

        return np.concatenate([lin_delta, rot_delta])

    def joint_velocity_to_delta(self, joint_velocity):

        joint_velocity = np.asarray(joint_velocity)

        relative_max_joint_vel = self.joint_delta_to_velocity(self.relative_max_joint_delta)
        max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()

        if max_joint_vel_norm > 1:
            joint_velocity = joint_velocity / max_joint_vel_norm

        return joint_velocity * self.max_joint_delta

    def cartesian_delta_to_velocity(self, cartesian_delta):

        cartesian_delta = np.asarray(cartesian_delta)

        cartesian_velocity = np.zeros_like(cartesian_delta)
        cartesian_velocity[:3] = cartesian_delta[:3] / self.max_lin_delta
        cartesian_velocity[3:6] = cartesian_delta[3:6] / self.max_rot_delta

        return cartesian_velocity

    def joint_delta_to_velocity(self, joint_delta):

        joint_delta = np.asarray(joint_delta)
        return joint_delta / self.max_joint_delta
