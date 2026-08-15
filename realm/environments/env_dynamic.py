"""One REALM environment: a scene built from the task config, and the rollout loop over it.

RealmEnvironmentDynamic owns construction and stepping. What it delegates:

    env_config.py               assembling the OmniGibson config from REALM's YAML layers
    scene_setup.py              the fixes applied to the loaded scene and robot
    perturbations/registry.py   which callable a perturbation name resolves to
    env_base.py                 scoring reference, contact sensing, progression, joint reset

Construction is two-phase because og.sim.play() is global. A single env plays for itself and runs
post_play_setup() straight through; a member of a vector env stops after the scene loads and lets
RealmVectorEnvironment play once for everyone and then drive the four pieces of post_play_setup()
itself. The per-member ordering is identical either way.
"""
import copy
from functools import partial

import numpy as np
import yaml

import omnigibson as og
from omnigibson.controllers.controller_view import ControllerView
from scipy.spatial.transform import Rotation as R

from realm.environments.constants import DEFAULT_RESET_JOINTPOS, DROID_BASE_HEIGHT
from realm.environments.env_base import RealmEnvironmentBase
from realm.environments.env_config import build_environment_config
from realm.environments.perturbations.registry import PERTURBATION_FNS
from realm.environments.perturbations.v_aug import apply_blur_and_contrast
from realm.environments.scene_setup import SceneSetupMixin
from realm.geometry import (
    calculate_new_camera_pose_mixed_rotations,
    robot_to_world,
    world_to_robot,
)
from realm.inference.utils import assert_wrist_camera
from realm.sim_config import set_rendering_mode

WARMUP_STEPS = 30


class RealmEnvironmentDynamic(SceneSetupMixin, RealmEnvironmentBase):
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
        self.use_droid_with_base = True if self.task_cfg_path.split("/")[0] == "REALM_DROID10" else False # TODO: infer properly from the task/scene config yaml
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
        if "SB-NOUN" in self.active_perturbations and cfg["task_type"] == "push":
            raise NotImplementedError() # TODO: move this to some compatibility matrix / exclusion list

        if common_freq is not None:
            cfg["env"]["rendering_frequency"] = common_freq
            cfg["env"]["action_frequency"] = common_freq

        # Duplicated from RealmEnvironmentBase.__init__, which finalize_setup() runs later and which
        # overwrites all three with the same values -- a vector env defers that far enough that a
        # perturbation could read them first. Read capture_mo_reference() before touching these, in
        # particular for why mo_bbox_orig is an anchor on the task config and must NOT track the
        # live object.
        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.cfg = copy.deepcopy(cfg)
        self.task_type = self.cfg["task_type"]
        self.instruction = self.cfg["instruction"]

        # in_vec_env defers og.sim.play() and everything that depends on a playing simulator to
        # RealmVectorEnvironment, which plays once for all members. See environments/env_vector.py.
        self.in_vec_env = in_vec_env
        # Work a perturbation deferred because it needs a playing sim, which in a vector env does
        # not exist until every member's perturbation has run. Drained by
        # RealmVectorEnvironment.reset() right after its single og.sim.play(); always empty in a
        # single env, where perturbations/_helpers.after_play() runs the work inline instead.
        self.deferred_post_play = []
        # Set by perturbations/_helpers.settle() in a vector env to request the shared settle loop.
        self.wants_settle = False
        self._mo_cfgs = mo_cfgs
        self._to_cfgs = to_cfgs
        self._dist_cfgs = dist_cfgs
        self.omnigibson_env = og.Environment(configs=[cfg], in_vec_env=in_vec_env)

        if not in_vec_env:
            self.post_play_setup()

    # ============================== [CONSTRUCTION] ==============================
    def post_play_setup(self):
        """Everything that requires a playing simulator. Single-env path; run straight through.

        A vector env cannot call this directly: apply_scene_fixes_from_cfg() cycles og.sim.stop()/
        play(), which are global, so one member cannot cycle them without disturbing the others.
        RealmVectorEnvironment instead calls the four pieces below itself, batching the stop/play
        across all members. The per-member ordering is identical either way.
        """
        self.bind_scene_handles()
        self.apply_scene_fixes_from_cfg()
        self.rebase_initial_file()
        self.finalize_setup()

    def bind_scene_handles(self):
        """Resolve robot/object handles and apply robot physics overrides (pre scene-fix half)."""
        mo_cfgs, to_cfgs, dist_cfgs = self._mo_cfgs, self._to_cfgs, self._dist_cfgs

        assert len(self.omnigibson_env.robots) == 1  # assumes single robot, single arm
        self.robot = self.omnigibson_env.robots[0]
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}
        # Which physical camera the wrist observation comes from is decided by a creation-order index
        # in ROBOT_OBS_PROFILES, and getting it wrong costs a silent warning rather than a crash.
        # Check it here, once per member, before anything steps.
        self.wrist_camera_key = assert_wrist_camera(self.robot)

        self.main_objects = [self.omnigibson_env.scene.object_registry("name", mo["name"]) for mo in mo_cfgs]
        self.target_objects = [self.omnigibson_env.scene.object_registry("name", to["name"]) for to in to_cfgs]
        self.distractors = [self.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in dist_cfgs]

        self.init_poses = {obj._relative_prim_path: { # using relative prim path as unique id
            "pos": obj.get_position_orientation()[0],
            "rot": obj.get_position_orientation()[1]
        } for obj in self.main_objects + self.target_objects + self.distractors}

        if "VSB-NOBJ" in self.active_perturbations and self.task_type in ["open_drawer", "close_drawer"]:
            self.init_poses[self.main_objects[0]._relative_prim_path]["pos"][-1] += 0.3

        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 3.0)
            self.v_aug_alpha = np.random.uniform(0.5, 2.0)

        self.update_robot_physics()

    def finalize_setup(self):
        """Visual toggles, render mode and base-class init (post scene-fix half)."""
        self.disable_visual_toggles()
        set_rendering_mode(self.rendering_mode)

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
        assert pose_name in self.cfg_camera_extrinsics
        base_cam_pos = self.cfg_camera_extrinsics[pose_name]["pos"]
        base_cam_rot = self.cfg_camera_extrinsics[pose_name]["rot"]
        base_cam_pos, base_cam_rot = calculate_new_camera_pose_mixed_rotations(
            base_cam_pos, base_cam_rot,
            robot_pos, robot_rot
        )
        base_cam_pos[-1] += DROID_BASE_HEIGHT if self.use_droid_with_base else 0  # height of the robot base
        return base_cam_pos, base_cam_rot

    # ============================== [RESET] ==============================
    def reset(self):
        """Single-env reset. Vector envs must drive the two phases below directly instead.

        The split exists because a perturbation's sim-state management is GLOBAL: see
        RealmVectorEnvironment.reset(), which interleaves one shared stop/play/settle around all
        members' perturbations rather than letting each member cycle the simulator on its own.
        """
        obs, info = self.reset_pre_perturbation()
        obs = self.apply_perturbations(obs)
        return obs, info

    def reset_pre_perturbation(self):
        """Phase 1 of reset: restore this member's scene and clear its task bookkeeping.

        Touches no global simulator state, so a vector env can run it for every member up front.
        """
        obs, info = self.omnigibson_env.reset()
        self.reset_joints()

        self.was_lifted = False
        for k in self.task_progression.keys():
            self.task_progression[k] = False

        self.deferred_post_play.clear()
        self.wants_settle = False
        return obs, info

    def apply_perturbations(self, obs):
        """Phase 2 of reset: run this member's perturbations.

        In a vector env the caller is responsible for the surrounding sim state -- perturbations
        route their stop/play/step/settle through perturbations/_helpers, which no-op or defer here
        and let RealmVectorEnvironment do each of them exactly once for all members.
        """
        for p in self.active_perturbations:
            self.supported_pertrubations[p]()
        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 2.5)
            self.v_aug_alpha = np.random.uniform(0.25, 1.5)
        obs = self._distort_if_v_aug(obs)

        # LAST, once every perturbation has run: SB-NOUN, VSB-NOBJ and VB-MOBJ all re-point
        # main_objects[0], so the lift/distance/rotation reference has to be re-taken from whatever
        # the target now is. Guarded exactly like _helpers.sim_play()/settle(): in a vector env this
        # phase runs with the sim still STOPPED and a replaced object not yet initialized, so a pose
        # read here would be invalid -- RealmVectorEnvironment.reset() makes the equivalent call for
        # every member once its shared play and settle are done.
        #
        # Here rather than at the end of reset() because reset() is only one of the two ways this
        # phase is driven (a vector env and the probe scripts call the phases directly), and the
        # invariant belongs to the phase that breaks it. See capture_mo_reference().
        if not self.in_vec_env:
            self.capture_mo_reference()
        return obs

    # ============================== [ROLLOUT] ==============================
    def step(self, action, n_render_iterations=1):
        """Advance this env one action.

        @n_render_iterations is passed straight through to OmniGibson: it issues that many
        og.sim.render() calls before observations are read, which is how a render step flushes the
        rendering pipeline after a run of non-rendering (blind) steps. See realm/eval.py's
        render_on_demand path.
        """
        obs, rew, terminated, truncated, info = self.omnigibson_env.step(
            action, n_render_iterations=n_render_iterations
        )
        task_progression = self.recompute_task_progression(obs)
        return self._distort_if_v_aug(obs), task_progression, terminated, truncated, info

    def pre_step(self, action):
        """Apply @action to this member's robot without advancing physics.

        og.sim.step() advances every scene at once, so a vector env must apply all members' actions
        first and step once. Single-env stepping goes through step() instead.
        """
        self.omnigibson_env._pre_step(action)

    def post_step(self, action):
        """Read observations for this member after a shared og.sim.step(), mirroring step()."""
        obs, rew, terminated, truncated, info = self.omnigibson_env._post_step(action)
        task_progression = self.recompute_task_progression(obs)
        return self._distort_if_v_aug(obs), task_progression, terminated, truncated, info

    def _distort_if_v_aug(self, obs):
        """Apply V-AUG's blur/contrast to @obs, or return it untouched when V-AUG is not active."""
        if "V-AUG" not in self.active_perturbations:
            return obs
        # obs is keyed by robot.name, which is NOT always "DROID" -- see v_aug.py.
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

        # Re-take the reference now that the arm has settled, so the rollout is scored against where
        # the object actually sits rather than where reset() left it mid-settle. reset() has already
        # taken it once; this refines that value, it does not repair a different object's pose.
        self.capture_mo_reference()
        og.log.info("Warmup finished.")
        return obs, rew, terminated, truncated, info

    def warmup_ee_cmd(self):
        """Hold-still EE command for warmup: the current pose in robot frame. None if joint-control.

        OG 3.9.1: robot._controllers[name] is a (group_key, controller_idx) TUPLE, not the controller
        object -- instances live in the ControllerView registry, shared by every robot whose
        kinematic-tree pattern and controller config hash match. So the mode is read off the registry
        rather than off the entry.
        """
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
        """Warmup action for step @t: hold the arm still, open the gripper then close it."""
        gripper_val = np.atleast_1d(1.0 if t < WARMUP_STEPS // 2 else -1.0)
        base = ee_cmd if self.ee_control else self.reset_qpos[:7]
        return np.concatenate((base, gripper_val))

    # ============================== [FRAMES] ==============================
    def _robot2world(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return robot_to_world(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def _world2robot(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return world_to_robot(action, self.robot_pos, self.robot_rot_rad[2], base_height)
