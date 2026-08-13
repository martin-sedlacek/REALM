import numpy as np
import yaml
import copy
import os

from realm.environments.env_base import RealmEnvironmentBase
from realm.environments.env_config import build_environment_config
from realm.environments.constants import DEFAULT_RESET_JOINTPOS, DROID_BASE_HEIGHT
from realm.environments.perturbations.default import default as _pert_default
from realm.environments.perturbations.v_light import v_light as _pert_v_light
from realm.environments.perturbations.v_view import v_view as _pert_v_view
from realm.environments.perturbations.v_sc import v_sc as _pert_v_sc
from realm.environments.perturbations.semantic import s_prop as _pert_s_prop, s_lang as _pert_s_lang, s_mo as _pert_s_mo, s_aff as _pert_s_aff, s_int as _pert_s_int
from realm.environments.perturbations.b_hobj import b_hobj as _pert_b_hobj
from realm.environments.perturbations.sb_noun import sb_noun as _pert_sb_noun
from realm.environments.perturbations.sb_vrb import sb_vrb as _pert_sb_vrb
from realm.environments.perturbations.vb_pose import vb_pose as _pert_vb_pose
from realm.environments.perturbations.vb_mobj import vb_mobj as _pert_vb_mobj
from realm.environments.perturbations.vsb_nobj import vsb_nobj as _pert_vsb_nobj
# OG 3.9.1: robots are no longer Python classes. DROID/UR are declared as RobotDefinition YAMLs
# under realm/robots/definitions/ and instantiated by OmniGibson's single Robot class via
# `model: <name>`; WidowX uses OmniGibson's stock `vx300s` definition. Nothing to import here.
from realm.categories import get_non_droid_categories
from realm.environments.perturbations.v_aug import apply_blur_and_contrast
from realm.geometry import (
    calculate_new_camera_pose_mixed_rotations,
    robot_to_world,
    world_to_robot,
)
from realm.inference.utils import assert_wrist_camera
from realm.sim_config import set_rendering_mode

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.asset_utils import get_all_object_models
from omnigibson.utils.usd_utils import create_joint
from scipy.spatial.transform import Rotation as R



MISSING_PERTURBATIONS = ["V-OBJ", "VB-ISC", "VS-PROP", "SB-ADV", "SB-SMO"]
SUPPORTED_TASK_TYPES = ["put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer"]
WARMUP_STEPS = 30
SKILL_COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [],  # ["put", "pick", "rotate", "stack"],
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open": ["close"],
    "close": ["open"]
}

class RealmEnvironmentDynamic(RealmEnvironmentBase):
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
            'Default':  lambda: _pert_default(self),
            "V-AUG":    lambda: _pert_default(self),  # V-AUG is applied when distorting the images in obs
            "V-VIEW":   lambda: _pert_v_view(self),
            "V-SC":     lambda: _pert_v_sc(self),
            "V-LIGHT":  lambda: _pert_v_light(self),
            "S-PROP":   lambda: _pert_s_prop(self),
            "S-LANG":   lambda: _pert_s_lang(self),
            "S-MO":     lambda: _pert_s_mo(self),
            "S-AFF":    lambda: _pert_s_aff(self),
            "S-INT":    lambda: _pert_s_int(self),
            "B-HOBJ":   lambda: _pert_b_hobj(self),
            "SB-NOUN":  lambda: _pert_sb_noun(self),
            "SB-VRB":   lambda: _pert_sb_vrb(self),
            "VB-POSE":  lambda: _pert_vb_pose(self),
            "VB-MOBJ":  lambda: _pert_vb_mobj(self),
            "VSB-NOBJ": lambda: _pert_vsb_nobj(self),
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
        # overwrites all three with the same values. Read the write-up there (and in
        # capture_mo_reference()) before touching these -- in particular why mo_bbox_orig is an
        # anchor on the task config and must NOT track the live object.
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

    def post_play_setup(self):
        """Everything that requires a playing simulator. Single-env path; run straight through.

        A vector env cannot call this directly: apply_scene_fixes_from_cfg() cycles og.sim.stop()/
        play(), which are global, so one member cannot cycle them without disturbing the others.
        RealmVectorEnvironment instead calls the three pieces below itself, batching the stop/play
        across all members. The per-member ordering is identical either way.
        """
        self.bind_scene_handles()
        self.apply_scene_fixes_from_cfg()
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

        # ---------- apply fixes to the env ----------
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

    # ============================== [VECTOR ENV HOOKS] ==============================
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
        if "V-AUG" in self.active_perturbations:
            # obs is keyed by robot.name, which is NOT always "DROID" -- see v_aug.py.
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha,
                                          robot_name=self.robot.name)
        return obs, task_progression, terminated, truncated, info

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

    def update_robot_physics(self):
        # Every DROID variant, not just the config literally named "DROID". The robolab asset was
        # silently skipped here, so its arm ran with zero armature -- no rotor inertia against a
        # stiff impedance law -- and the wrist would not hold a commanded pose.
        if not self.robot_name.startswith("DROID"):
            return

        friction = np.array(self.cfg["robots"][0]["friction"])
        armature = np.array(self.cfg["robots"][0]["armature"])

        joint_names = self.robot.arm_joint_names
        # OG 3.9.1 runs under the Fabric Scene Delegate, where raw USD edits are not automatically
        # propagated to Fabric; every USD write must happen inside this context (which synchronizes
        # on exit) or OmniGibson aborts. The context must not be nested, so all edits go in one block.
        with og.sim.editing_usd():
            for idx in range(7):
                prim_path = f"{self.robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
                joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
                assert joint_prim.IsValid(), f"no joint prim at {prim_path}"
                # Create the attributes if the asset never authored them -- droid.usd ships them,
                # the robolab asset does not, and GetAttribute(...).Set() on a missing attribute is
                # a silent no-op, which would leave armature at zero without any error.
                lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)
                for attr_name, value in (("physxJoint:jointFriction", friction[idx]),
                                         ("physxJoint:armature", armature[idx])):
                    attr = joint_prim.GetAttribute(attr_name)
                    if not attr:
                        attr = joint_prim.CreateAttribute(attr_name, lazy.pxr.Sdf.ValueTypeNames.Float)
                    attr.Set(float(value))

            # Fix triangle mesh collision approximation for dynamic bodies
            for link_name, link in self.robot.links.items():
                for collision_mesh in link.collision_meshes.values():
                    prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                    if prim.IsValid() and prim.HasAttribute("physxMeshCollision:approximation"):
                        approx = prim.GetAttribute("physxMeshCollision:approximation").Get()
                        if approx in ["none", "meshSimplification"]:
                            prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")

    def apply_scene_fixes_from_cfg(self, manage_sim_state=True):
        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)

        if self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]:
            scene_data = spawn_cfg[self.scene_model][self.scene_part]
            if manage_sim_state:
                og.sim.stop()
            for obj in self.omnigibson_env.scene.objects:
                if obj.name in scene_data.get("to_fix", []):
                    obj.fixed_base = True
                    create_joint(
                        prim_path=f"{obj.prim_path}/rootJoint",
                        joint_type="FixedJoint",
                        body1=f"{obj.prim_path}/{obj._root_link_name}",
                    )
                elif obj.name in scene_data.get("to_remove", []):
                    obj_to_remove = self.omnigibson_env.scene.object_registry("name", obj.name)
                    self.omnigibson_env.scene.remove_object(obj_to_remove)
                # elif obj.name in special_prims[self.scene_model][self.scene_part].get("drawer", []):
                #     drawer_to_modify = self.omnigibson_env.scene.object_registry("name", obj.name)

            if manage_sim_state:
                og.sim.play()

    def disable_visual_toggles(self):
        for obj in self.omnigibson_env.scene.objects:
            # TODO: (martin) for pre-baked OG switches on walls their rotation seems off so we cannot use those without the visual toggle...
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False

    # ============================== [ROLLOUT UTILS] ==============================
    def warmup_ee_cmd(self):
        """Hold-still EE command for warmup: the current pose in robot frame. None if joint-control."""
        if not self.ee_control:
            return None
        arm_controller = self.robot._controllers.get("arm_0")
        if arm_controller is not None and arm_controller.mode != "absolute_pose":
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
            # obs is keyed by robot.name, which is NOT always "DROID" -- see v_aug.py.
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha,
                                          robot_name=self.robot.name)

        # LAST, once every perturbation has run: SB-NOUN re-points main_objects[0] at a distractor
        # and VSB-NOBJ/VB-MOBJ replace it, so the lift/distance/rotation reference has to be re-taken
        # from whatever the target now is. Guarded exactly like _helpers.sim_play()/settle(): in a
        # vector env this phase runs with the sim still STOPPED and a replaced object not yet
        # initialized, so a pose read here would be invalid -- RealmVectorEnvironment.reset() makes
        # the equivalent call for every member once its shared play and settle are done.
        #
        # Here rather than at the end of reset() because reset() is only one of the two ways this
        # phase is driven (a vector env and the probe scripts call the phases directly), and the
        # invariant belongs to the phase that breaks it. See capture_mo_reference().
        if not self.in_vec_env:
            self.capture_mo_reference()
        return obs

    def _robot2world(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return robot_to_world(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def _world2robot(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return world_to_robot(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def step(self, action, n_render_iterations=1):
        # if self.ee_control:
        #     action = self._robot2world(action)

        # n_render_iterations is passed straight through to OmniGibson: it issues that many
        # og.sim.render() calls before observations are read, which is how a render step flushes
        # the rendering pipeline after a run of non-rendering (blind) steps. See
        # realm/eval.py's render_on_demand path.
        obs, rew, terminated, truncated, info = self.omnigibson_env.step(
            action, n_render_iterations=n_render_iterations
        )

        task_progression = self.recompute_task_progression(obs)

        if "V-AUG" in self.active_perturbations:
            # obs is keyed by robot.name, which is NOT always "DROID" -- see v_aug.py.
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha,
                                          robot_name=self.robot.name)

        return obs, task_progression, terminated, truncated, info

    # ============================== [INIT HELPERS] ==============================
    def sample_objects(self, num_objects=3, included_categories=None, excluded_categories=None, ):
        assert not (included_categories is not None and excluded_categories is not None)

        # TODO: this can be pre-computed once, no need to parse the whole thing every call
        available_object_paths = []
        whitelisted_categories = get_non_droid_categories()

        if included_categories is not None:
          whitelisted_categories = included_categories
        elif excluded_categories is not None:
            for cat in excluded_categories:
                if cat in whitelisted_categories:
                    whitelisted_categories.remove(cat)

        for model_path in get_all_object_models():
            if os.path.exists(model_path):
                category = model_path.split("/")[-2]
                if category in whitelisted_categories:
                    available_object_paths.append(model_path)

        if not available_object_paths:
            return []

        if len(available_object_paths) < num_objects:
            og.log.info(
                f"Warning: Only {len(available_object_paths)} suitable objects found, less than requested {num_objects}.")
            num_objects = len(available_object_paths)

        # Randomly sample unique objects
        sampled_indices = np.random.choice(len(available_object_paths), size=num_objects, replace=False)
        sampled_objects = []
        for i in sampled_indices:
            category = available_object_paths[i].split("/")[-2]
            model_id = available_object_paths[i].split("/")[-1]
            name = f"distractor_{i}"
            obj_cfg = {
                "type": "DatasetObject",
                "name": name,
                "category": category,
                "model": model_id,
            }
            sampled_objects.append(obj_cfg)

        return sampled_objects

    # NOTE: RealmEnvironmentDynamic.replace_obj() used to live here. It was a pre-refactor duplicate
    # of perturbations/_helpers.replace_obj() with ZERO call sites left (every perturbation imports
    # the _helpers one), and it still carried the bbox-centre-as-extent bug that _helpers and
    # sb_vrb.py have since fixed. Deleted rather than repaired so there is only one copy to keep
    # correct -- the next person to wire up "replace an object" must find the live one.