import math
import numpy as np
import torch
import yaml
import random
import copy
import os

from realm.environments.env_base import RealmEnvironmentBase, TASK_PROGRESS_RUBRICS
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
from realm.robots.widowx import WidowX
from realm.robots.ur import UR
from realm.helpers import (
    calculate_new_camera_pose_mixed_rotations,
    add_rotation_noise,
    get_non_colliding_positions_for_objects,
    apply_blur_and_contrast,
    get_non_droid_categories,
    get_droid_categories_by_theme,
    get_objects_by_names,
    get_default_objects_cfg,
    robot_to_world,
    world_to_robot,
)

import omnigibson as og
import omnigibson.utils.transform_utils as omnigibson_transform_utils
import omnigibson.lazy as lazy
from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.asset_utils import get_all_object_models
from omnigibson.utils.usd_utils import create_joint
from omnigibson.prims.joint_prim import JointPrim
from scipy.spatial.transform import Rotation as R



MISSING_PERTURBATIONS = ["V-OBJ", "VB-ISC", "VS-PROP", "SB-ADV", "SB-SMO"]
SUPPORTED_TASK_TYPES = ["put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer"]
SKILL_COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [],  # ["put", "pick", "rotate", "stack"],
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open": ["close"],
    "close": ["open"]
}
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
DROID_BASE_HEIGHT = 0.86244
MAX_CAMERA_POS_DEVIATION = 0.2
MAX_CAMERA_PITCH_DEVIATION = 0.2
MAX_CAMERA_YAW_DEVIATION = 0.2
DROID_DEFAULT_DOF = 11

# Panda joint origins from panda_robotiq_85.urdf: (xyz, rpy) per joint
_PANDA_JOINT_ORIGINS = [
    ([0,       0,      0.333], [0,        0, 0]),
    ([0,       0,      0    ], [-np.pi/2, 0, 0]),
    ([0,      -0.316,  0    ], [ np.pi/2, 0, 0]),
    ([0.0825,  0,      0    ], [ np.pi/2, 0, 0]),
    ([-0.0825, 0.384,  0    ], [-np.pi/2, 0, 0]),
    ([0,       0,      0    ], [ np.pi/2, 0, 0]),
    ([0.088,   0,      0    ], [ np.pi/2, 0, 0]),
]
_PANDA_EE_OFFSET = [0, 0, 0.107]  # panda_link8 fixed offset from panda_link7 (panda_arm.urdf)


def _panda_fk(q):
    """Forward kinematics for Panda arm using URDF parameters.
    q: array of 7 joint angles (radians).
    Returns (pos, quat_xyzw) of panda_link8 in the robot base (link0) frame.
    Each joint transform: T = Translate(xyz) @ RPY(rpy) @ Rz(q_i)
    """
    def _rot3(a, axis):
        ca, sa = np.cos(a), np.sin(a)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        if axis == 'y':
            return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])  # z

    def _ht(xyz, rpy, qi):
        # Fixed part: Translate(xyz) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)
        r_fixed = _rot3(rpy[2], 'z') @ _rot3(rpy[1], 'y') @ _rot3(rpy[0], 'x')
        r_total = r_fixed @ _rot3(qi, 'z')
        m = np.eye(4)
        m[:3, :3] = r_total
        m[:3, 3] = xyz
        return m

    m = np.eye(4)
    for (xyz, rpy), qi in zip(_PANDA_JOINT_ORIGINS, q):
        m = m @ _ht(xyz, rpy, qi)

    m_ee = np.eye(4)
    m_ee[:3, 3] = _PANDA_EE_OFFSET
    m = m @ m_ee

    from scipy.spatial.transform import Rotation as _R
    return m[:3, 3].copy(), _R.from_matrix(m[:3, :3]).as_quat()


def set_rendering_mode(rendering_mode):
    carb_settings = lazy.carb.settings.get_settings()
    if rendering_mode == "pt":
        def enable_interactive_path_tracing(carb_settings, samples_per_pixel=8):
            carb_settings.set("/rtx/rendermode", "PathTracing")
            if samples_per_pixel is not None:
                carb_settings.set_int("/rtx/pathtracing/spp", samples_per_pixel)
                carb_settings.set_int("/rtx/pathtracing/totalSpp", samples_per_pixel)
                carb_settings.set_int(
                    "/rtx/pathtracing/useDirectLightingCache", False
                )
            carb_settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)

        #carb_settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
        enable_interactive_path_tracing(carb_settings, samples_per_pixel=8)
    elif rendering_mode == "r":
        carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")
        carb_settings.set_bool("/rtx/translucency/enabled", True)
        carb_settings.set_bool("/rtx/reflections/enabled", False)
        carb_settings.set_bool("/rtx/indirectDiffuse/enabled", False)
        carb_settings.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
        carb_settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 1)
        carb_settings.set_bool("/rtx/shadows/enabled", False)
        carb_settings.set_int("/rtx/post/dlss/execMode", 0)
        carb_settings.set_bool("/rtx/ambientOcclusion/enabled", False)
        carb_settings.set_bool("/rtx-transient/dlssg/enabled", False)
        carb_settings.set_float("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.6)
        carb_settings.set_float("/rtx/sceneDb/ambientLightIntensity", 1.0)
        carb_settings.set_bool("/exts/omni.renderer.core/present/enabled", False)
        carb_settings.set_string("/isaaclab/rendering/rendering_mode", "performance")
    else:
        assert rendering_mode == "rt", f"rendering mode must be 'pt', 'rt', or 'r'"


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
        robot: str = "DROID"
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

        if self.use_droid_with_base:
            from realm.robots.droid_arm_mounted import DROID
        else:
            from realm.robots.droid_arm import DROID

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

        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.cfg = copy.deepcopy(cfg)
        self.task_type = self.cfg["task_type"]
        self.instruction = self.cfg["instruction"]

        self.omnigibson_env = og.Environment(configs=[cfg])

        assert len(self.omnigibson_env.robots) == 1  # assumes single robot, single arm
        self.robot = self.omnigibson_env.robots[0]
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

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
        self.apply_scene_fixes_from_cfg()
        self.disable_visual_toggles()
        set_rendering_mode(rendering_mode)

        super().__init__(
            main_objects=self.main_objects,
            target_objects=self.target_objects,
            task_type=self.task_type,
            robot=self.robot,
            mo_cfgs=mo_cfgs
        )

    def construct_environment_config(self):
        cfg = dict()
        task_cfg = yaml.load(open(f"{self.config_path}/tasks/{self.task_cfg_path}", "r"), Loader=yaml.FullLoader)
        cfg.update(task_cfg)

        # ---------------------------------------- scene config ----------------------------------------
        for k in ["external_sensors", "robots"]:
            assert k not in cfg, f"{k} should be defined outside the scene file!"

        if self.scene_model is None:
            assert self.scene_part is None
            self.scene_model = list(task_cfg["supported_scenes"].keys())[0]
            self.scene_part = task_cfg["supported_scenes"][self.scene_model][0]
        assert self.scene_model in task_cfg["supported_scenes"]
        assert self.scene_part in task_cfg["supported_scenes"][self.scene_model]
        cfg.update(task_cfg["task"])

        scene_cfg_path = f"{self.config_path}/scenes/{self.scene_model}/{self.scene_part}/scene_definition.yaml"
        scene_cfg = None
        if os.path.exists(scene_cfg_path):
            scene_cfg = yaml.load(open(scene_cfg_path, "r"), Loader=yaml.FullLoader)
            cfg["scene"] = copy.deepcopy(scene_cfg["scene"])
        else:
            cfg["scene"] = {
                "type": "InteractiveTraversableScene",
                "scene_model": self.scene_model,
            }

        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)
        assert self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]
        scene_data = spawn_cfg[self.scene_model][self.scene_part]
        if all(k in scene_data for k in ["x_min", "x_max", "y_min", "y_max", "z"]):
            x_min = scene_data["x_min"]
            x_max = scene_data["x_max"]
            y_min = scene_data["y_min"]
            y_max = scene_data["y_max"]
            z = scene_data["z"]
            self.spawn_bbox = np.array([x_min, x_max, y_min, y_max, z])
        else:
            self.spawn_bbox = None

        # ---------------------------------------- robot config ----------------------------------------
        assert "pos" in scene_data and "rot" in scene_data
        robot_pos = scene_data['pos']
        robot_rot = [math.radians(angle_deg) for angle_deg in scene_data['rot']]
        self.robot_pos = np.array(robot_pos, dtype=float)
        self.robot_rot_rad = np.array(robot_rot, dtype=float)

        cfg_robot = yaml.load(open(f"{self.config_path}/robots/{self.robot_name}.yaml", "r"), Loader=yaml.FullLoader)
        self.ee_control = cfg_robot["robots"][0].get("ee_control", False)
        cfg_robot["robots"][0]["position"] = robot_pos
        cfg_robot["robots"][0]["orientation"] = omnigibson_transform_utils.euler2quat(
            torch.tensor(robot_rot, dtype=torch.float32)).tolist()
        cfg_robot["robots"][0]["fixed_base"] = True

        reset_joint_pos = np.zeros(cfg_robot["robots"][0]["dof"] if "dof" in cfg_robot["robots"][0] else DROID_DEFAULT_DOF)
        if "DROID" in self.robot_name:
            if "reset_joint_pos" in task_cfg:
                reset_joint_pos[:7] = np.array(task_cfg['reset_joint_pos'])
            elif "reset_joint_pos" in scene_data:
                reset_joint_pos[:7] = np.array(scene_data['reset_joint_pos'])
            else:
                reset_joint_pos[:7] = DEFAULT_RESET_JOINTPOS
        elif self.robot_name == "WidowX":
            reset_joint_pos[:6] = np.zeros(6) #np.array([0.0, -0.849879, 0.258767, 0.0, 1.2831712, 0.0])
        cfg_robot["robots"][0]["reset_joint_pos"] = reset_joint_pos

        if self.common_freq is not None:
            cfg_robot["robots"][0]["control_freq"] = self.common_freq
            cfg_robot["robots"][0]["controller_config"]["arm_0"]["control_freq"] = self.common_freq

        cfg.update(cfg_robot)
        self.reset_qpos = reset_joint_pos

        # ---------------------------------------- object config ----------------------------------------
        obj_list = task_cfg["main_objects"] + task_cfg["target_objects"]
        if "distractors" in task_cfg:
            obj_list += task_cfg["distractors"]
        if "immutables" in task_cfg:
            obj_list += task_cfg["immutables"]
        if scene_cfg is not None:
            obj_list += scene_cfg["objects"]

        robot_rot_deg_z = scene_data['rot'][-1]
        assert robot_rot_deg_z >= 0
        obj_pos_modifier_x = 1
        if 90 <= robot_rot_deg_z <= 270:
            obj_pos_modifier_x = -1

        if self.spawn_bbox is not None:
            for obj in obj_list:
                obj["relative_bbox_position"][0] *= obj_pos_modifier_x
                if obj_pos_modifier_x != 1:
                    if obj["relative_bbox_position"][0] < 0:
                        obj["relative_bbox_position"][0] -= obj_pos_modifier_x * (self.spawn_bbox[1] - self.spawn_bbox[0])
                    else:
                        obj["relative_bbox_position"][0] += obj_pos_modifier_x * (self.spawn_bbox[1] - self.spawn_bbox[0])
                obj["position"] = [x + y for x, y in zip(obj["relative_bbox_position"], [self.spawn_bbox[0], self.spawn_bbox[2], self.spawn_bbox[4]])]

            # TODO: the pipeline is broken for dynamically reducing # objects when there are too many distractors and
            # they become unplaceable - 3 is always fine and easy to place so we use that for now as maximum
            num_distractors = 3 if any(p in self.active_perturbations for p in ["V-SC"]) else 0 #"VB-ISC" #"SB-NOUN"
            cfg["objects"] = None
            excluded_categories = []
            for obj in task_cfg["main_objects"] + task_cfg["target_objects"]:
                if "category" in obj:
                    excluded_categories.append(obj["category"])
            distractors = self.sample_objects(num_objects=num_distractors, excluded_categories=excluded_categories)

            cfg["objects"] = get_non_colliding_positions_for_objects(
                xmin=self.spawn_bbox[0],
                xmax=self.spawn_bbox[1],
                ymin=self.spawn_bbox[2],
                ymax=self.spawn_bbox[3],
                z=self.spawn_bbox[4],
                obj_cfg=obj_list + distractors,
                max_attempts_per_object=25000,
                main_object_names=[o["name"] for o in obj_list],
            )
        else:
            cfg["objects"] = obj_list
            distractors = []

        if "distractors" in task_cfg:
            distractors += task_cfg["distractors"]
        if "immutables" in task_cfg:
            distractors += task_cfg["immutables"] # immutables go here because the distractor list above is meant to be replaceable objects

        for obj in cfg["objects"]:
            assert "position" in obj

        # ---------------------------------------- external camera config ----------------------------------------
        if "env" not in cfg:
            cfg["env"] = {
                "initial_pos_z_offset": 0.2
            }
        if not self.no_rendering:
            ext_cam1_pose = task_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in task_cfg else "default"
            if "camera_extrinsics" in task_cfg and "cam2" in task_cfg["camera_extrinsics"]:
                ext_cam2_pose = task_cfg["camera_extrinsics"]["cam2"]
            else:
                ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"

            base_cam_pos, base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam1_pose, robot_pos, robot_rot)

            cfg_external_sensors = yaml.load(open(f"{self.config_path}/env/external_sensors/camera_config.yaml", "r"), Loader=yaml.FullLoader)
            cfg_external_sensors["external_sensors"][0]["position"] = base_cam_pos
            cfg_external_sensors["external_sensors"][0]["orientation"] = base_cam_rot

            if self.multi_view:
                second_base_cam_pos, second_base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam2_pose, robot_pos,
                                                                                               robot_rot)
                cfg_external_sensors["external_sensors"][1]["position"] = second_base_cam_pos
                cfg_external_sensors["external_sensors"][1]["orientation"] = second_base_cam_rot
            else:
                del cfg_external_sensors["external_sensors"][1]

            cfg["env"].update(cfg_external_sensors)

        return (copy.deepcopy(cfg),
                copy.deepcopy([o for o in task_cfg["main_objects"]]),
                copy.deepcopy([o for o in task_cfg["target_objects"]]),
                copy.deepcopy([o for o in distractors])
                )

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
        if not self.robot_name == "DROID":
            return

        friction = np.array(self.cfg["robots"][0]["friction"])
        armature = np.array(self.cfg["robots"][0]["armature"])

        joint_names = self.robot.arm_joint_names
        for idx in range(7):
            prim_path = f"{self.robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
            joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            assert joint_prim.IsValid()
            joint_prim.GetAttribute("physxJoint:jointFriction").Set(friction[idx])
            joint_prim.GetAttribute("physxJoint:armature").Set(armature[idx])

        # Fix triangle mesh collision approximation for dynamic bodies
        for link_name, link in self.robot.links.items():
            for collision_mesh in link.collision_meshes.values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                if prim.IsValid() and prim.HasAttribute("physxMeshCollision:approximation"):
                    approx = prim.GetAttribute("physxMeshCollision:approximation").Get()
                    if approx in ["none", "meshSimplification"]:
                        prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")

    def apply_scene_fixes_from_cfg(self):
        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)

        if self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]:
            scene_data = spawn_cfg[self.scene_model][self.scene_part]
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

            og.sim.play()

    def disable_visual_toggles(self):
        for obj in self.omnigibson_env.scene.objects:
            # TODO: (martin) for pre-baked OG switches on walls their rotation seems off so we cannot use those without the visual toggle...
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False

    # ============================== [ROLLOUT UTILS] ==============================
    def warmup(self, obs=None):
        og.log.info("Starting warmup...")
        for _ in range(30):
            og.sim.render()

        if obs is None:
            obs, _ = self.reset()

        if self.ee_control:
            arm_controller = self.robot._controllers.get("arm_0")
            if arm_controller is not None and arm_controller.mode != "absolute_pose":
                ee_cmd = np.zeros(6)
            else:
                ee_pos, ee_quat = self.get_ee_pose()
                ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
                ee_euler = R.from_quat(ee_quat.cpu().numpy()).as_euler('xyz')
                ee_cmd = self._world2robot(np.concatenate([ee_pos, ee_euler]))

        for t in range(30):
            gripper_val = np.atleast_1d(1.0 if t < 15 else -1.0)
            if self.ee_control:
                new_action = np.concatenate((ee_cmd, gripper_val))
            else:
                new_action = np.concatenate((self.reset_qpos[:7], gripper_val))

            obs, rew, terminated, truncated, info = self.step(new_action)

            # if self.ee_control:
            #     # Sanity check: robot must not drift during warmup.
            #     # Compare FK position at each step against FK at step 0 (not against ee_cmd,
            #     # which may have a constant calibration offset from the URDF model).
            #     q_current = obs[self.robot.name]['proprio'].cpu().numpy()[:7]
            #     fk_pos, fk_quat = _panda_fk(q_current)
            #     drift = np.linalg.norm(fk_pos - ee_cmd[:3])
            #     assert drift < 0.05, (
            #         f"Robot drifted {drift:.4f}m during EE warmup (step {t}). "
            #         f"fk_pos_initial={ee_cmd}, fk_pos_now={fk_pos}"
            #     )

        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()
        og.log.info("Warmup finished.")
        return obs, rew, terminated, truncated, info

    def reset(self):
        obs, _ = self.omnigibson_env.reset()
        self.reset_joints()

        self.was_lifted = False
        for k in self.task_progression.keys():
            self.task_progression[k] = False

        for p in self.active_perturbations:
            self.supported_pertrubations[p]()
        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 2.5)
            self.v_aug_alpha = np.random.uniform(0.25, 1.5)
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)
        return obs, _

    def _robot2world(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return robot_to_world(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def _world2robot(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return world_to_robot(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def step(self, action):
        # if self.ee_control:
        #     action = self._robot2world(action)

        obs, rew, terminated, truncated, info = self.omnigibson_env.step(action)

        task_progression = self.recompute_task_progression(obs)

        if "V-AUG" in self.active_perturbations:
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)

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

    def replace_obj(self, obj: DatasetObject, included_categories=None, maximum_dim=0.2, fixed_base=False, preserve_ori=True):
        obj_name = obj.name

        if not (included_categories is None) and len(included_categories) == 1 and "bottom_cabinet" in included_categories:
            bottom_cabinet_models = [
                "bamfsz",
                "dsbcxl",
                "ilofmb",
                # "jhymlr", two top drawers
                "lhucjo",
                "mbmbpa",
                "nddvba",
                "immwzb",
                "pkdnbu",
                "plccav",
                #"pllcur", opens bottom for some reason
                "rntwkg",
                # "ttmejh", not leveled
                "slgzfc",
                "rvpunw",
                "wesxdp",
                "rhdbzv"
            ]
            sampled_idx = np.random.choice(len(bottom_cabinet_models), size=1, replace=False)[0]
            nobj_cfg = {
                "type": "DatasetObject",
                "name": obj_name,
                "category": "bottom_cabinet",
                "model": bottom_cabinet_models[sampled_idx],
            }
        else:
            candidates = self.sample_objects(num_objects=1, included_categories=included_categories)
            if not candidates:
                raise ValueError(f"replace_obj: No suitable objects found for categories: {included_categories}")
            nobj_cfg = candidates[0]

        self.omnigibson_env.scene.remove_object(obj)

        new_obj = DatasetObject(
            name=obj_name,
            relative_prim_path=obj._relative_prim_path,
            category=nobj_cfg["category"],
            model=nobj_cfg["model"],
            fixed_base=fixed_base
        )
        self.omnigibson_env.scene.add_object(new_obj)

        if preserve_ori:
            new_obj.set_bbox_center_position_orientation(torch.tensor(self.init_poses[new_obj._relative_prim_path]["pos"]),
                                                        torch.tensor(self.init_poses[new_obj._relative_prim_path]["rot"]))
        else:
            new_obj.set_bbox_center_position_orientation(torch.tensor(self.init_poses[new_obj._relative_prim_path]["pos"]),
                                                        torch.tensor([0, 0, 0, 1]))

        bbox_center, bbox_orn, bbox_extent, bbox_center_in_frame = new_obj.get_base_aligned_bbox()
        nobj_cfg["bounding_box"] = bbox_center

        max_dim = np.max(bbox_extent.numpy())
        new_scale_factor = maximum_dim / max_dim
        if new_scale_factor < 1.0:
            new_obj.scale = new_scale_factor # TODO: explain method code in comments
            nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor
        nobj_cfg["fixed_base"] = fixed_base

        return new_obj, nobj_cfg