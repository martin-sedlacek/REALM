import os
import torch as th

from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.transform_utils import euler2quat


class FrankaPandaRobotiq(ManipulationRobot):
    """
    The Franka Emika Panda robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        # Unique to Franka
        end_effector="gripper",
        controller_name="CustomJointController",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            end_effector (str): type of end effector to use. One of {"gripper", "allegro", "leap_right", "leap_left", "inspire"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # store end effector information
        self.end_effector = end_effector
        if end_effector == "gripper":
            self._model_name = "franka_panda"
            self._gripper_control_idx = th.arange(7, 11) #(7, 9)
            self._eef_link_names = "panda_link8"
            self._finger_link_names = [
                "gripper_link_left_inner_finger",
                "gripper_link_right_inner_finger"
            ]
            self._finger_joint_names = [
                "left_inner_finger_prismatic_joint",
                "right_inner_finger_prismatic_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint"
            ]
            self._default_robot_model_joint_pos = th.tensor([
                1.43628523e-01, #0.00,
                -5.97331285e-01, #np.pi / 8,
                2.01173499e-03, #0.00,
                -2.38619041e+00, #-np.pi * 1 / 2 + 0.25,
                2.67941624e-01, #0.00,
                1.97807848e+00, #np.pi * 1 / 2,
                -8.98026153e-02, #0.00,
                0.00,
                0.00,
                0.00,
                0.00
            ])
            self._teleop_rotation_offset = th.tensor([-1, 0, 0, 0])
            self._ag_start_points = [
                GraspingPoint(link_name="panda_leftfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="panda_rightfigner", position=th.tensor([0.0, 0.001, 0.045])),
            ]

        self.controller_name = controller_name #"JointController"
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            grasping_direction="lower",  # gripper grasps in the opposite direction
            **kwargs,
        )

    @property
    def model_name(self):
        # Override based on specified Franka variant
        return self._model_name

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Franka does not support discrete actions!")

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = self.controller_name
        controllers["gripper_{}".format(self.default_arm)] = "CustomGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        return self._default_robot_model_joint_pos

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint{i+1}" for i in range(7)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: self._eef_link_names}

    @property
    def finger_link_names(self):
        return {self.default_arm: self._finger_link_names}

    @property
    def finger_joint_names(self):
        return {self.default_arm: self._finger_joint_names}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, f"/app/realm/robots/panda_robotiq/droid_mounted.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        #return {self.default_arm: os.path.join(gm.ASSET_PATH, f"models/panda_robotiq/panda_with_cam_description.yaml")}
        # TODO: do not hardcode paths and set the macro instead
        return {self.default_arm: os.path.join(gm.ASSET_PATH, f"/app/realm/robots/panda_robotiq/panda_robotiq_description.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/franka/panda_robotiq_85.urdf")

    @property
    def curobo_path(self):
        # Only supported for normal franka now
        assert (
            self._model_name == "franka_panda"
        ), f"Only franka_panda is currently supported for curobo. Got: {self._model_name}"
        # TODO:
        return None #os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_description_curobo.yaml")

    @property
    def eef_usd_path(self):
        # TODO:
        return None #{self.default_arm: os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_eef.usd")}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: self._teleop_rotation_offset}

    @property
    def assisted_grasp_start_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def assisted_grasp_end_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def disabled_collision_pairs(self):
        # some dexhand has self collisions that needs to be filtered outg
        if self.end_effector == "allegro":
            return [["link_12_0", "part_studio_link"]]
        elif self.end_effector == "inspire":
            return [["base_link", "link12"]]
        else:
            return []

    @property
    def _default_controller_config(self):
        controllers = {}
        controllers.update(
            {
                f"arm_{arm_name}": {self.controller_name: self._default_arm_joint_controller_configs[arm_name]}
                for arm_name in self.arm_names
            }
        )
        controllers.update(
            {
                f"gripper_{arm_name}": {
                    "CustomGripperController": self._default_gripper_multi_finger_controller_configs[arm_name]
                }
                for arm_name in self.arm_names
            }
        )
        return controllers

    @property
    def _default_arm_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config to control that
                robot's arm. Uses velocity control by default.
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": self.controller_name,
                "control_freq": self._control_freq,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_input_limits": None,
                "command_output_limits": None,
                "motor_type": "position",
                "use_delta_commands": False,
                "use_impedances": True,
            }
        return dic