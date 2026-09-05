from realm.config.shared import WARMUP_STEPS

import copy
from functools import partial

import numpy as np
import yaml

import omnigibson as og
from omnigibson.controllers.controller_view import ControllerView
from scipy.spatial.transform import Rotation as R

from realm.environments.constants import DEFAULT_RESET_JOINTPOS, DROID_BASE_HEIGHT
from realm.config.shared import UNSUPPORTED_BY_PERTURBATION
from realm.environments.env_base import RealmEnvironmentBase
from realm.environments.env_config import build_environment_config
from realm.environments.foam_ball_reset import FoamBallMixin, foam_ball_cfgs
from realm.environments.perturbations.registry import PERTURBATION_FNS
from realm.environments.perturbations.v_aug import ALPHA_RANGE, SIGMA_RANGE, apply_blur_and_contrast
from realm.environments.scene_setup import SceneSetupMixin
from realm.geometry import (
    calculate_new_camera_pose_mixed_rotations,
    robot_to_world,
    world_to_robot,
)
from realm.inference.utils import assert_wrist_camera
from realm.sim_config import set_rendering_mode



class RealmEnvironmentDynamic(SceneSetupMixin, FoamBallMixin, RealmEnvironmentBase):
    def __init__(
        self,
        config_path="/app/realm/config",
        scene_model=None,
        scene_part=None,
        reset_qpos=None,
        task_cfg_path="REALM_DROID10/put_green_block_into_bowl/default.cfg",
        perturbations=None,
        common_freq: int = None,
        no_rendering: bool = False,
        multi_view: bool = False,
        rendering_mode: str = "rt",
        robot: str = "DROID",
        in_vec_env: bool = False,
    ) -> None:
        assert not (multi_view and no_rendering), f"Multi-view rendering was enabled during no_rendering mode. Either one is likely a mistake."
        self.task_cfg_path = "/".join(task_cfg_path.split("/")[-3:])
        self.use_droid_with_base = robot.startswith("DROID")
        self.robot_name = robot
        self.multi_view = multi_view
        self.no_rendering = no_rendering
        self.rendering_mode = rendering_mode
        self.config_path = config_path
        self.scene_model = scene_model
        self.scene_part = scene_part
        self.reset_qpos = reset_qpos if reset_qpos is not None else DEFAULT_RESET_JOINTPOS
        self.common_freq = common_freq
        self.supported_pertrubations = {
            name: partial(fn, self) for name, fn in PERTURBATION_FNS.items()
        }

        self.active_perturbations = perturbations
        for perturbation in self.active_perturbations:
            assert perturbation in self.supported_pertrubations.keys()

        camera_extrinsics_path = f"{self.config_path}/env/external_sensors/camera_extrinsics.yaml"
        self.cfg_camera_extrinsics = yaml.load(open(camera_extrinsics_path, "r"), Loader=yaml.FullLoader)

        cfg, mo_cfgs, to_cfgs, dist_cfgs = self.construct_environment_config()
        assert len(mo_cfgs) == 1
        assert len(to_cfgs) <= 1
        assert "position" in mo_cfgs[0], "mo must have a specified position"
        if (cfg["task_type"] in UNSUPPORTED_BY_PERTURBATION["SB-NOUN"]
                and "SB-NOUN" in self.active_perturbations):
            raise NotImplementedError()

        if common_freq is not None:
            cfg["env"]["rendering_frequency"] = common_freq
            cfg["env"]["action_frequency"] = common_freq

        # Needed before the deferred base-class initialization.
        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        # `bidirectional: true` (IMPACT/stack_plates) means the task is satisfied by EITHER object
        # ending up on the other -- two plates are interchangeable in a way a green block and a
        # yellow block are not. The stage checks in TaskProgressionMixin then have to score the
        # target object as well, which needs its authored pose here beside the main object's.
        self.bidirectional = bool(cfg.get("bidirectional", False)) and len(to_cfgs) > 0
        if self.bidirectional:
            self.to_pos_orig = np.array(to_cfgs[0]["position"])
            self.to_rot_orig = np.array(to_cfgs[0]["orientation"] if "orientation" in to_cfgs[0] else [0, 0, 0, 1])
            self.to_bbox_orig = np.array(to_cfgs[0]["bounding_box"])
        else:
            self.to_pos_orig = None
            self.to_rot_orig = None
            self.to_bbox_orig = None

        self.cfg = copy.deepcopy(cfg)
        self.task_type = self.cfg["task_type"]
        self.instruction = self.cfg["instruction"]

        # After build_environment_config, so the balls never reach placement.place_within: they
        # belong inside the source bottle, not scattered over the spawn region as distractors.
        # They ride in dist_cfgs from here so they get scene handles and init_poses like any other
        # object -- V-SC is told to leave them alone by name (foam_ball_names).
        for ball_cfg in foam_ball_cfgs(cfg, mo_cfgs):
            cfg["objects"].append(ball_cfg)
            dist_cfgs.append(ball_cfg)

        self.in_vec_env = in_vec_env
        self.deferred_post_play = []
        self.wants_settle = False
        self._mo_cfgs = mo_cfgs
        self._to_cfgs = to_cfgs
        self._dist_cfgs = dist_cfgs
        self.omnigibson_env = og.Environment(configs=[cfg], in_vec_env=in_vec_env)

        if not in_vec_env:
            self.post_play_setup()

    def post_play_setup(self):

        self.bind_scene_handles()
        self.apply_scene_fixes_from_cfg()
        self.rebase_initial_file()
        self.finalize_setup()

    def bind_scene_handles(self):

        mo_cfgs, to_cfgs, dist_cfgs = self._mo_cfgs, self._to_cfgs, self._dist_cfgs

        assert len(self.omnigibson_env.robots) == 1
        self.robot = self.omnigibson_env.robots[0]
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}
        self.wrist_camera_key = assert_wrist_camera(self.robot)

        self.main_objects = [self.omnigibson_env.scene.object_registry("name", mo["name"]) for mo in mo_cfgs]
        self.target_objects = [self.omnigibson_env.scene.object_registry("name", to["name"]) for to in to_cfgs]
        self.distractors = [self.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in dist_cfgs]

        self.init_poses = {obj._relative_prim_path: {
            "pos": obj.get_position_orientation()[0],
            "rot": obj.get_position_orientation()[1]
        } for obj in self.main_objects + self.target_objects + self.distractors}

        if "VSB-NOBJ" in self.active_perturbations and self.task_type in ["open_drawer", "close_drawer"]:
            self.init_poses[self.main_objects[0]._relative_prim_path]["pos"][-1] += 0.3

        self.bind_foam_balls()

        self.v_aug_sigma = None
        self.v_aug_alpha = None

        self.update_robot_physics()
        self.restore_double_duty_render_purpose()

    def finalize_setup(self):

        self.disable_visual_toggles()
        set_rendering_mode(self.rendering_mode)
        self.place_foam_balls()

        super().__init__(
            main_objects=self.main_objects,
            target_objects=self.target_objects,
            task_type=self.task_type,
            robot=self.robot,
            mo_cfgs=self._mo_cfgs
        )

    def construct_environment_config(self):
        return build_environment_config(self)

    def construct_ext_cam_pose_by_name(self, pose_name, robot_pos, robot_rot):
        if isinstance(pose_name, dict):
            assert set(pose_name) >= {"pos", "rot"}
            base_cam_pos = pose_name["pos"]
            base_cam_rot = pose_name["rot"]
        else:
            assert pose_name in self.cfg_camera_extrinsics
            base_cam_pos = self.cfg_camera_extrinsics[pose_name]["pos"]
            base_cam_rot = self.cfg_camera_extrinsics[pose_name]["rot"]
        base_cam_pos, base_cam_rot = calculate_new_camera_pose_mixed_rotations(
            base_cam_pos, base_cam_rot,
            robot_pos, robot_rot
        )
        base_cam_pos[-1] += DROID_BASE_HEIGHT if self.use_droid_with_base else 0
        return base_cam_pos, base_cam_rot

    def reset(self):

        obs, info = self.reset_pre_perturbation()
        obs = self.apply_perturbations(obs)
        return obs, info

    def reset_pre_perturbation(self):

        obs, info = self.omnigibson_env.reset()
        self.reset_joints()

        self.was_lifted = False
        for k in self.task_progression.keys():
            self.task_progression[k] = False

        self.deferred_post_play.clear()
        self.wants_settle = False
        return obs, info

    def apply_perturbations(self, obs):

        for p in self.active_perturbations:
            self.supported_pertrubations[p]()
        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(*SIGMA_RANGE)
            self.v_aug_alpha = np.random.uniform(*ALPHA_RANGE)
        obs = self._distort_if_v_aug(obs)

        # Replacements change which object anchors progression metrics.
        if not self.in_vec_env:
            self.capture_mo_reference()
            self.capture_foam_ball_reference()
        return obs

    def step(self, action, n_render_iterations=1):

        obs, rew, terminated, truncated, info = self.omnigibson_env.step(
            action, n_render_iterations=n_render_iterations
        )
        task_progression = self.recompute_task_progression(obs)
        return self._distort_if_v_aug(obs), task_progression, terminated, truncated, info

    def pre_step(self, action):

        self.omnigibson_env._pre_step(action)

    def post_step(self, action):

        obs, rew, terminated, truncated, info = self.omnigibson_env._post_step(action)
        task_progression = self.recompute_task_progression(obs)
        return self._distort_if_v_aug(obs), task_progression, terminated, truncated, info

    def _distort_if_v_aug(self, obs):

        if "V-AUG" not in self.active_perturbations:
            return obs
        return apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha,
                                       robot_name=self.robot.name)

    def warmup(self, obs=None):
        og.log.info("Starting warmup...")
        for _ in range(30):
            og.sim.render()

        if obs is None:
            obs, _ = self.reset()

        ee_cmd = self.warmup_ee_cmd()

        for t in range(WARMUP_STEPS):
            obs, rew, terminated, truncated, info = self.step(self.warmup_action(t, ee_cmd))

        self.capture_mo_reference()
        og.log.info("Warmup finished.")
        return obs, rew, terminated, truncated, info

    def warmup_ee_cmd(self):

        if not self.ee_control:
            return None
        entry = self.robot._controllers.get("arm_0")
        if entry is not None and ControllerView.get_mode(entry[0]) != "absolute_pose":
            return np.zeros(6)
        ee_pos, ee_quat = self.get_ee_pose()
        ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
        ee_euler = R.from_quat(ee_quat.cpu().numpy()).as_euler('xyz')
        return self._world2robot(np.concatenate([ee_pos, ee_euler]))

    def warmup_action(self, t, ee_cmd):

        gripper_val = np.atleast_1d(1.0 if t < WARMUP_STEPS // 2 else -1.0)
        base = ee_cmd if self.ee_control else self.reset_qpos[:7]
        return np.concatenate((base, gripper_val))

    def _robot2world(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return robot_to_world(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def _world2robot(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return world_to_robot(action, self.robot_pos, self.robot_rot_rad[2], base_height)
