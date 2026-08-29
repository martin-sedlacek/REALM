import os

import numpy as np
from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models.robots.robot_arms import robot_arm


class RobotArm(robot_arm.RobotArm):
    """dm_robotics arm entity backed by an MJCF model. Instantiate a subclass (FrankaArm), whose
    _build sets _name/_model_file/_mjcf_root and calls _create_body -- the base class deliberately
    defines no _build of its own (the one it used to have read an attribute nothing had set)."""

    def _create_body(self):
        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all("joint")
        self._bodies = self.mjcf_model.find_all("body")
        self._actuators = self.mjcf_model.find_all("actuator")
        self._wrist_site = self.mjcf_model.find("site", "wrist_site")
        self._base_site = self.mjcf_model.find("site", "base_site")

    @property
    def name(self) -> str:
        # A property, matching the dm_robotics RobotArm contract. Without the decorator,
        # `arm.name` was a BOUND METHOD, and RobotIKSolver passed it as robot_name into the
        # effectors -- so their prefix strings embedded `<bound method ...>` instead of "franka".
        return self._name

    @property
    def joints(self):

        return self._joints

    @property
    def actuators(self):

        return self._actuators

    @property
    def mjcf_model(self):

        return self._mjcf_root

    def update_state(self, physics: mjcf.Physics, qpos: np.ndarray, qvel: np.ndarray) -> None:
        physics.bind(self._joints).qpos[:] = qpos
        physics.bind(self._joints).qvel[:] = qvel

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        physics.bind(self._joints).qpos[:] = qpos

    @property
    def base_site(self) -> types.MjcfElement:
        return self._base_site

    @property
    def wrist_site(self) -> types.MjcfElement:
        return self._wrist_site

    def initialize_episode(self, physics: mjcf.Physics, random_state: np.random.RandomState):

        del random_state  # Unused.
        return


class FrankaArm(RobotArm):
    def _build(self):
        self._name = "franka"
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._model_file = os.path.join(dir_path, "franka", "{0}.xml".format("panda"))
        self._mjcf_root = mjcf.from_path(self._model_file)
        self._create_body()
