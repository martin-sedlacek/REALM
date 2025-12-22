import math
import numpy as np
import torch
import yaml
import random
import copy
import os
import time

from realm.robots.droid_joint_controller import IndividualJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController

from realm.tasks.task_progressions import TASK_PROGRESSIONS
from realm.helpers import (calculate_new_camera_pose_mixed_rotations, add_rotation_noise,
                           get_non_colliding_positions_for_objects_v2, apply_blur_and_contrast,
                           get_non_droid_categories, get_droid_categories_by_theme,
                           find_and_remove_category, process_droid_categories,
                           get_objects_by_names, get_default_objects_cfg)

import omnigibson as og
import omnigibson.utils.transform_utils as omnigibson_transform_utils
import omnigibson.lazy as lazy
from realm.environments.realm_environtment_base import RealmEnvironmentBase
from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.asset_utils import get_all_object_models
from omnigibson.utils.usd_utils import create_joint
from omnigibson.prims.joint_prim import JointPrim

MISSING_PERTURBATIONS = ["V-OBJ", "VB-ISC", "VS-PROP", "SB-ADV", "SB-SMO"]
SUPPORTED_TASK_TYPES = ["put", "pick", "rotate", "push", "stack"]# TODO: "open_close_drawer", "turn_faucet"


class RealmEnvironmentDynamic(RealmEnvironmentBase):
    def __init__(
        self,
        config_path="/app/realm/config",
        scene_model=None,
        scene_part=None,
        reset_qpos=None,
        task="put_green_block_in_bowl",
        perturbations=None,
        use_droid_with_base=True,
        common_freq: int = None
    ) -> None:
        self.use_droid_with_base = use_droid_with_base # TODO: infer from task / scene config
        if self.use_droid_with_base:
            from realm.robots.franka_robotiq_mounted import FrankaPandaRobotiq
        else:
            from realm.robots.franka_robotiq import FrankaPandaRobotiq

        self.task = task
        self.config_path = config_path
        self.scene_model = scene_model
        self.scene_part = scene_part
        self.reset_qpos = reset_qpos

        self.supported_pertrubations = {
            'Default': self.default,
            "V-AUG": self.default, # V-AUG is applied when distorting the images in obs
            "V-VIEW": self.v_view,
            "V-SC": self.v_sc,
            "V-LIGHT": self.v_light,
            "V-SCENE": None, # TODO
            "S-PROP": self.s_prop,
            "S-LANG": self.s_lang,
            "S-MO": self.s_mo,
            "S-AFF": self.s_aff,
            "S-INT": self.s_int,
            "B-HOBJ": self.b_hobj,
            "SB-NOUN": self.sb_noun,
            "SB-VRB": self.sb_vrb,
            "VB-POSE": self.vb_pose,
            "VB-MOBJ": self.vb_mobj,
            "VSB-NOBJ": self.vsb_nobj
        }

        self.active_perturbations = perturbations
        for perturbation in self.active_perturbations:
            assert perturbation in self.supported_pertrubations.keys()

        self.common_freq = common_freq

        camera_extrinsics_path = f"{self.config_path}/env/external_sensors/camera_extrinsics.yaml"
        self.cfg_camera_extrinsics = yaml.load(open(camera_extrinsics_path, "r"), Loader=yaml.FullLoader)

        cfg, mo_cfgs, to_cfgs, dist_cfgs = self.construct_environment_config()
        test_mo_cfg = copy.deepcopy(mo_cfgs)
        assert len(mo_cfgs) == 1
        assert len(to_cfgs) <= 1
        assert "position" in mo_cfgs[0], "mo must have a specified position"

        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        if common_freq is not None:
            cfg["env"]["rendering_frequency"] = common_freq
            cfg["env"]["action_frequency"] = common_freq

        self.cfg = copy.deepcopy(cfg)

        self.task_type = self.cfg["task_type"]

        # TODO: move this to some compatibility matrix / exclusion list
        # assert self.task_type in SUPPORTED_TASK_TYPES, (self.task_type, SUPPORTED_TASK_TYPES)
        if "SB-NOUN" in self.active_perturbations and self.task_type == "push":
            raise NotImplementedError()

        self.omnigibson_env = og.Environment(configs=[cfg])

        assert len(self.omnigibson_env.robots) == 1  # assumes single robot, single arm
        self.robot = self.omnigibson_env.robots[0]
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

        self.instruction = self.cfg["instruction"]
        self.main_objects = [self.omnigibson_env.scene.object_registry("name", mo["name"]) for mo in mo_cfgs]
        self.target_objects = [self.omnigibson_env.scene.object_registry("name", to["name"]) for to in to_cfgs]
        self.distractors = [self.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in dist_cfgs]

        self.init_poses = {obj._relative_prim_path: { # using relative prim path as unique id
            "pos": obj.get_position_orientation()[0],
            "rot": obj.get_position_orientation()[1]
        } for obj in self.main_objects + self.target_objects + self.distractors}

        # TODO: only for our custom drawer offset
        if "VSB-NOBJ" in self.active_perturbations:
            self.init_poses[self.main_objects[0]._relative_prim_path]["pos"][-1] += 0.3

        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 3.0)
            self.v_aug_alpha = np.random.uniform(0.5, 2.0)

        # TODO: move to a config file
        self.friction = np.array([0.05, 0.10, 0.05, 0.10, 0.75, 0.425, 0.20])
        self.armature = np.array([0.25, 0.50, 0.25, 0.50, 0.25, 0.150, 0.00])
        self.update_robot_physics()

        self.apply_scene_fixes_from_cfg()
        self.disable_visual_toggles()

        super().__init__(
            main_objects=self.main_objects,
            target_objects=self.target_objects,
            task=self.task,
            task_type=self.task_type,
            robot=self.robot,
            use_droid_with_base=use_droid_with_base,
            mo_cfgs=test_mo_cfg
        )

    def construct_environment_config(self):
        cfg = dict()

        scene_config_path = f"{self.config_path}/tasks/{self.task}.yaml"
        comprehensive_cfg = yaml.load(open(scene_config_path, "r"), Loader=yaml.FullLoader)
        cfg.update(comprehensive_cfg)
        # ---------------------------------------- scene config ----------------------------------------
        for k in ["external_sensors", "robots"]:
            assert k not in cfg, f"{k} should be defined outside the scene file!"

        if self.scene_model is None:
            assert self.scene_part is None
            self.scene_model = list(comprehensive_cfg["supported_scenes"].keys())[0]
            self.scene_part = comprehensive_cfg["supported_scenes"][self.scene_model][0]
        assert self.scene_model in comprehensive_cfg["supported_scenes"]
        assert self.scene_part in comprehensive_cfg["supported_scenes"][self.scene_model]
        cfg.update(comprehensive_cfg["task"])

        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": self.scene_model
        }

        spawn_config_path = f"{self.config_path}/scenes/scenes.yaml"
        spawn_cfg = yaml.load(open(spawn_config_path, "r"), Loader=yaml.FullLoader)
        assert self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]
        scene_data = spawn_cfg[self.scene_model][self.scene_part]
        x_min = scene_data["x_min"]
        x_max = scene_data["x_max"]
        y_min = scene_data["y_min"]
        y_max = scene_data["y_max"]
        z = scene_data["z"]
        self.spawn_bbox = np.array([x_min, x_max, y_min, y_max, z])

        # ---------------------------------------- robot config ----------------------------------------
        assert "pos" in scene_data and "rot" in scene_data
        robot_pos = scene_data['pos']
        robot_rot = [math.radians(angle_deg) for angle_deg in scene_data['rot']]
        reset_joint_pos = np.zeros(11)
        if "reset_joint_pos" in comprehensive_cfg:
            reset_joint_pos[:7] = np.array(comprehensive_cfg['reset_joint_pos'])
        elif "reset_joint_pos" in scene_data:
            reset_joint_pos[:7] = np.array(scene_data['reset_joint_pos'])
        else:
            reset_joint_pos[:7] = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])

        print(reset_joint_pos)
        cfg_robot = yaml.load(open(f"{self.config_path}/robots/franka_robotiq.yaml", "r"), Loader=yaml.FullLoader)
        cfg_robot["robots"][0]["position"] = robot_pos
        cfg_robot["robots"][0]["orientation"] = omnigibson_transform_utils.euler2quat(
            torch.tensor(robot_rot, dtype=torch.float32)).tolist()
        cfg_robot["robots"][0]["fixed_base"] = True
        cfg_robot["robots"][0]["reset_joint_pos"] = reset_joint_pos

        if self.common_freq is not None:
            cfg_robot["robots"][0]["control_freq"] = self.common_freq
            cfg_robot["robots"][0]["controller_config"]["arm_0"]["control_freq"] = self.common_freq

        cfg.update(cfg_robot)
        self.reset_qpos = reset_joint_pos

        # ---------------------------------------- object config ----------------------------------------
        obj_list = comprehensive_cfg["main_objects"] + comprehensive_cfg["target_objects"]
        if "distractors" in comprehensive_cfg:
            obj_list += comprehensive_cfg["distractors"]
        if "immutables" in comprehensive_cfg:
            obj_list += comprehensive_cfg["immutables"]

        robot_rot_deg_z = scene_data['rot'][-1]
        assert robot_rot_deg_z >= 0
        obj_pos_modifier_x = 1
        obj_pos_modifier_y = 1
        if 90 <= robot_rot_deg_z <= 270:
            obj_pos_modifier_x = -1

        for obj in obj_list:
            obj["relative_bbox_position"][0] *= obj_pos_modifier_x
            if obj_pos_modifier_x != 1:
                if obj["relative_bbox_position"][0] < 0:
                    obj["relative_bbox_position"][0] -= obj_pos_modifier_x * (x_max - x_min)
                else:
                    obj["relative_bbox_position"][0] += obj_pos_modifier_x * (x_max - x_min)
            obj["position"] = [x + y for x, y in zip(obj["relative_bbox_position"], [x_min, y_min, z])]

        # TODO: the pipeline is broken for dynamically reducing # objects when there are too many distractors and
        # they become unplaceable - 3 is always fine and easy to place so we use that for now as maximum
        num_distractors = 3 if any(p in self.active_perturbations for p in ["V-SC"]) else 0 #"VB-ISC" #"SB-NOUN"
        cfg["objects"] = None
        excluded_categories = []
        for obj in comprehensive_cfg["main_objects"] + comprehensive_cfg["target_objects"]:
            if "category" in obj:
                excluded_categories.append(obj["category"])
        distractors = self.sample_objects(num_objects=num_distractors, excluded_categories=excluded_categories)

        cfg["objects"] = get_non_colliding_positions_for_objects_v2(
                xmin=x_min,
                xmax=x_max,
                ymin=y_min,
                ymax=y_max,
                z=z,
                obj_cfg=obj_list + distractors,
                max_attempts_per_object=25000,
                main_object_names=[o["name"] for o in obj_list]
            )

        if "distractors" in comprehensive_cfg:
            distractors += comprehensive_cfg["distractors"]
        if "immutables" in comprehensive_cfg:
            distractors += comprehensive_cfg["immutables"] # immutables go here because the distractor list above is meant to be replaceable objects

        # make sure positions exist
        for obj in cfg["objects"]:
            assert "position" in obj

        # ---------------------------------------- external camera config ----------------------------------------
        ext_cam1_pose = comprehensive_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in comprehensive_cfg else "default"
        if "camera_extrinsics" in comprehensive_cfg:
            ext_cam2_pose = comprehensive_cfg["camera_extrinsics"]["cam2"]
        else:
            ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"

        base_cam_pos, base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam1_pose, robot_pos, robot_rot)
        second_base_cam_pos, second_base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam2_pose, robot_pos, robot_rot)

        cfg_external_sensors = yaml.load(open(f"{self.config_path}/env/external_sensors/camera_config.yaml", "r"), Loader=yaml.FullLoader)
        cfg_external_sensors["external_sensors"][0]["position"] = base_cam_pos
        cfg_external_sensors["external_sensors"][0]["orientation"] = base_cam_rot
        cfg_external_sensors["external_sensors"][1]["position"] = second_base_cam_pos
        cfg_external_sensors["external_sensors"][1]["orientation"] = second_base_cam_rot
        # Debugging camera view:
        # cfg_external_sensors["external_sensors"][1]["position"] = [-0.64, -1.7, 1.0625] #second_base_cam_pos
        # cfg_external_sensors["external_sensors"][1]["orientation"] = [ 0.2847762, -0.4648792, -0.7096336, 0.4463295 ] #second_base_cam_rot

        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"].update(cfg_external_sensors)

        return (cfg,
                [o for o in comprehensive_cfg["main_objects"]],
                [o for o in comprehensive_cfg["target_objects"]],
                [o for o in distractors]
                )

    def construct_ext_cam_pose_by_name(self, pose_name, robot_pos, robot_rot):
        assert pose_name in self.cfg_camera_extrinsics
        base_cam_pos = self.cfg_camera_extrinsics[pose_name]["pos"]
        base_cam_rot = self.cfg_camera_extrinsics[pose_name]["rot"]
        print("-----DEBUG")
        print(len(base_cam_pos), len(base_cam_rot))
        print(len(robot_pos), len(robot_rot))
        base_cam_pos, base_cam_rot = calculate_new_camera_pose_mixed_rotations(
            base_cam_pos, base_cam_rot,
            robot_pos, robot_rot
        )
        base_cam_pos[-1] += 0.86244 if self.use_droid_with_base else 0  # height of the robot base
        return base_cam_pos, base_cam_rot

    def update_robot_physics(self):
        robot = self.omnigibson_env.robots[0]
        joint_names = robot.arm_joint_names
        for idx in range(7):
            prim_path = f"{robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
            joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            assert joint_prim.IsValid()
            joint_prim.GetAttribute("physxJoint:jointFriction").Set(self.friction[idx])
            joint_prim.GetAttribute("physxJoint:armature").Set(self.armature[idx])

    def apply_scene_fixes_from_cfg(self):
        spawn_config_path = f"{self.config_path}/scenes/scenes.yaml"
        spawn_cfg = yaml.load(open(spawn_config_path, "r"), Loader=yaml.FullLoader)

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

    # ============================== [PERTURBATIONS] ==============================
    def default(self):
        return

    def v_light(self, intensity=None):
        if intensity is None:
            intensity =  np.random.uniform(20000, 750000)

        def find_lights_recursive(obj): # TODO: move the search to new scene instantiation, pointless to call it everytime unless we are swapping scene
            lights = []
            if "light" in obj.name:
                lights.append(obj)

            if hasattr(obj, "_links"):
                for link in obj._links.values():
                    lights.extend(find_lights_recursive(link))

            return lights

        # def find_light_prim(light_object):
        #     object_prim = light_object.root_prim
        #     for child in object_prim.GetChildren():
        #         if child.GetTypeName() == "Xform":
        #             for grand_child in child.GetChildren():
        #                 if grand_child.IsA(lazy.pxr.UsdLux.Light):
        #                     return grand_child
        #     return None

        all_lights = []
        for obj in self.omnigibson_env.scene.objects:
            all_lights.extend(find_lights_recursive(obj))

        col_mean = np.array([255, 214, 170])
        col_std = 15
        world_path = "/World/scene_0" # TODO: is this always the case? what about vectorized envs
        for light in all_lights:
            light_prim_path = world_path + light._relative_prim_path + "/light_0" # TODO: ^^^
            light_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(light_prim_path)
            if light_prim is None or not light_prim.IsValid(): # the recursive search also takes links that do not contain the light object, these are skipped here
                continue
            #assert light_prim.IsValid()

            light_prim.GetAttribute("inputs:intensity").Set(intensity)

            color = np.random.normal(loc=col_mean, scale=col_std, size=(3,))
            color = np.clip(color, 0, 255).astype(float) / 255.0
            light_prim.GetAttribute("inputs:color").Set(lazy.pxr.Gf.Vec3f(*color))

        # for light in all_lights:
        #     light_prim = find_light_prim(light)
        #     if not light_prim.IsValid(): # the recursive search also takes links that do not contain the light object, these are skipped here
        #         continue
        #     #assert light_prim and light_prim.IsValid()
        #
        #     # light_prim_path = light_prim.GetPath().pathString
        #     # light_prim = lazy.omni.isaac.core.utils.prims.lazy_prims_utils.get_prim_at_path(light_prim_path)
        #     # assert light_prim.IsValid()
        #
        #     light_prim.GetAttribute("inputs:intensity").Set(intensity)
        #
        #     color = np.random.normal(loc=col_mean, scale=col_std, size=(3,))
        #     color = np.clip(color, 0, 255).astype(float) / 255.0
        #     light_prim.GetAttribute("inputs:color").Set(lazy.pxr.Gf.Vec3f(*color))

    def v_view(self):
        def perturb_camera_pose(cam_pos: list[float], cam_orientation: list[float]) -> tuple[list[float], list[float]]:
            MAX_POS_DEVIATION = 0.2
            MAX_PITCH_DEVIATION = 0.2
            MAX_YAW_DEVIATION = 0.2
            cam_pos = np.array(cam_pos)
            delta_pos = np.random.uniform(-MAX_POS_DEVIATION, MAX_POS_DEVIATION, 3)
            cam_pos += delta_pos
            cam_pos = cam_pos.tolist()

            cam_orientation = torch.tensor(cam_orientation)
            cam_rpy = omnigibson_transform_utils.quat2euler(cam_orientation)
            cam_rpy[0] += (torch.rand(()) * 2 - 1) * MAX_PITCH_DEVIATION
            cam_rpy[2] += (torch.rand(()) * 2 - 1) * MAX_YAW_DEVIATION
            cam_orientation = omnigibson_transform_utils.euler2quat(cam_rpy)
            cam_orientation = cam_orientation.cpu().numpy().tolist()

            return cam_pos, cam_orientation

        # TODO: in some cases, the objects are not fully visible - add a look_at or similar to minimize these cases
        og.sim.stop()
        for i in range(len(self.omnigibson_env.external_sensors)):
            robot_pos = self.cfg["robots"][0]["position"]
            robot_rot = self.cfg["robots"][0]["orientation"]
            robot_rot = omnigibson_transform_utils.quat2euler(torch.tensor(robot_rot, dtype=torch.float32)).tolist()

            cam_pose_keys = list(self.cfg_camera_extrinsics.keys())
            filtered_cam_pose_keys = [
                key for key in cam_pose_keys
                if (
                        not key.startswith('CP') and
                        not (i == 0 and 'cam2' in key) and
                        not (i == 1 and 'cam1' in key)
                )
            ]
            if self.task_type in ["open_drawer", "close_drawer"]:
                cam_pose_name = "ep_001042_cam1" if i == 0 else "ep_001042_cam2" # TODO: scene specific, just get the extrinsic key dynamically
            else:
                cam_pose_name = np.random.choice(filtered_cam_pose_keys)
            cam_pos, cam_orientation = self.construct_ext_cam_pose_by_name(cam_pose_name, robot_pos, robot_rot)
            new_cam_pos, new_cam_orientation = perturb_camera_pose(cam_pos, cam_orientation)
            base_cam_config = self.cfg["env"]["external_sensors"][i]
            pose_frame = base_cam_config["pose_frame"]
            self.omnigibson_env.external_sensors[base_cam_config["name"]].set_position_orientation(new_cam_pos, new_cam_orientation, pose_frame)
        og.sim.play()
        obs, _ = self.omnigibson_env.reset()
        self.reset_joints()

    def vsb_nobj(self):
        included_categories = None
        if self.task_type == "push":
            included_categories = ["electric_switch", "thermostat"] # TODO: microwave, monitor buttons (maybe more)?
        elif self.task_type in ["open_drawer", "close_drawer"]:
            included_categories = ["bottom_cabinet"]

        og.sim.stop()
        fixed_base_loc = True if self.task_type in ["push", "open_drawer", "close_drawer"] else False
        preserve_ori = False if self.task_type in ["push", "open_drawer", "close_drawer"] else True
        max_dim = 0.5 if self.task_type in ["open_drawer", "close_drawer"] else 0.15
        nobj, nobj_cfg = self.replace_obj(self.main_objects[0], included_categories=included_categories, maximum_dim=max_dim, fixed_base=fixed_base_loc, preserve_ori=preserve_ori)
        self.main_objects = [nobj]

        self.instruction = self.cfg["instruction"].replace(self.cfg["instruction_obj_to_replace"], nobj_cfg["category"].replace("_", " "))
        if nobj_cfg["model"] in ["strbnw", "gashan", "qxhtct", "wseglt"]:
            self.main_objects[0].set_orientation(np.array([0, 0, 0.7071068, 0.7071068]))
        # elif nobj_cfg["model"] in ["hpowgy", "hrwnhp", "jophec"]:
        #     self.main_objects[0].set_orientation(np.array([0, 0, 1, 0]))  # wall flip 180
            #self.main_objects[0].set_orientation(np.array([-0.4330127, -0.4330127, 0.25, 0.75])) # tabletop flip 180
        og.sim.play()
        og.sim.step()
        self.omnigibson_env.scene.update_initial_state()
        self.reset_joints()

        if og.object_states.ToggledOn in nobj.states:
            nobj.states[og.object_states.ToggledOn].visual_marker.visible = False

        # fake rest to get to original pose after stopping sim
        for _ in range(30):
            self.omnigibson_env.step(np.concatenate((self.reset_qpos[:7], np.atleast_1d(np.array([-1])))))

    def vb_pose(self):
        # --------------- Translation ---------------
        if self.task_type == "push":
            delta_z = np.random.uniform(-0.15, 0.15)
            delta_xy = np.random.uniform(-0.075, 0.075)
            for obj_cfg in self.cfg["objects"]:
                if obj_cfg["name"] == "electric_switch":
                    obj = self.omnigibson_env.scene.object_registry("name", obj_cfg["name"])
                    init_pos = self.init_poses[obj._relative_prim_path]["pos"]
                    init_pos[2] += delta_z
                    init_pos[0] += delta_xy # TODO: this is only for pomaria light switch, elsewhere it might be y axis on the wall...
                    og.sim.stop()
                    obj.set_position_orientation(init_pos)
                    og.sim.play()
        else:
            self.cfg["objects"] = get_non_colliding_positions_for_objects_v2(
                xmin=self.spawn_bbox[0],
                xmax=self.spawn_bbox[1],
                ymin=self.spawn_bbox[2],
                ymax=self.spawn_bbox[3],
                z=self.spawn_bbox[4],
                obj_cfg=self.cfg["objects"],
                objects_to_skip=[obj.name for obj in self.distractors + self.target_objects],
                main_object_names=[],
                max_attempts_per_object=25000 # TODO: this must be successful, careful what we do here...
            )

            # obj_cfgs = copy.deepcopy(self.cfg["objects"])
            # num_mo_to = len(self.target_objects + self.main_objects)
            #
            # self.cfg["objects"] = None
            # num_distractors = len(obj_cfgs) - num_mo_to
            #
            # while self.cfg["objects"] is None and num_distractors >= 0:
            #     # TODO: this placement algo is naive and super bad actually, improve this
            #     self.cfg["objects"] = get_non_colliding_positions_for_objects(
            #             xmin=self.spawn_bbox[0],
            #             xmax=self.spawn_bbox[1],
            #             ymin=self.spawn_bbox[2],
            #             ymax=self.spawn_bbox[3],
            #             z=self.spawn_bbox[4],
            #             obj_cfg=obj_cfgs[:num_mo_to + num_distractors],
            #             #obj_cfg=self.cfg["objects"],
            #             objects_to_skip=[obj.name for obj in self.distractors],
            #             main_object_names=[]
            #         )
            #     num_distractors -= 1
            # assert num_distractors > -1, "Failed to place task objects with 0 distractors. This is not expected - investigate position config in your task or reach out to us."

            og.sim.stop()
            for obj_cfg in self.cfg["objects"]:
                if self.task_type in ["open_drawer", "close_drawer"] and obj_cfg["name"] == "drawer":
                    obj_cfg["position"][-1] -= 0.3
                self.omnigibson_env.scene.object_registry("name", obj_cfg["name"]).set_position_orientation(obj_cfg["position"])

            # --------------- Rotation ---------------
            for o in self.main_objects:
                if self.task_type in ["open_drawer", "close_drawer"]:
                    for obj_cfg in self.cfg["objects"]:
                        if obj_cfg["name"] == "drawer":
                            tmp_obj_cfg = obj_cfg
                    tmp = tmp_obj_cfg["orientation"] if "orientation" in tmp_obj_cfg else [0, 0, 0, 1]
                    new_rot = add_rotation_noise(tmp, (0, 0, 0.12), [-3.14, -3.14, 0], [3.14, 3.14, 0.57], (0, 0, 0.25))
                    o.set_orientation(new_rot)
                else:
                    tmp = o.get_orientation() # TODO: also from orig rot?
                    o.set_orientation(add_rotation_noise(tmp, (0, 0, 3.14)))
            og.sim.play()
            self.reset_joints()

        # fake rest to get to original pose after stopping sim
        for _ in range(30):
            self.omnigibson_env.step(np.concatenate((self.reset_qpos[:7], np.atleast_1d(np.array([-1])))))


    def b_hobj(self):
        s = np.random.uniform(0.25, 3)
        s_mass, s_mvel, s_meff, s_stif, s_damp, s_fric = np.exp(np.random.uniform(-1, 1, size=(6,)))
        for obj in self.main_objects:
            for link in obj._links.values():
                link.mass = min(link.mass * s, 2.0) # clip at 2.0kg payload
                link.mass *= s

            for joint in obj.joints.values():
                joint: JointPrim
                joint.max_effort = joint.max_effort * float(s_meff)
                joint.stiffness = joint.stiffness * s_stif
                joint.damping = joint.damping * s_damp
                joint._articulation_view.set_max_efforts(torch.tensor([[joint.max_effort]], dtype=torch.float32), joint_indices=joint.dof_indices)
                joint._articulation_view.set_gains(kps=torch.tensor([[joint.stiffness]]), joint_indices=joint.dof_indices)
                joint._articulation_view.set_gains(kds=torch.tensor([[joint.damping]]), joint_indices=joint.dof_indices)

                # TODO: add frictions
                # print(type(link))
                # print(link)
                # link_name = link.name
                # mat_name = f"{link_name}_physics_mat"
                # physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                #     prim_path=f"{link.prim_path}/Looks/{mat_name}",
                #     name=mat_name,
                #     **material_info,
                # )
                # for msh in self.links[link_name].collision_meshes.values():
                #     msh.apply_physics_material(physics_mat)


                # link_prim = link._prim
                # if not link_prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
                #     lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(link_prim)
                # og.sim.step()
                # link_prim.GetAttribute("physxRigidBody:dynamicFriction").Set(0.5)
                # link_prim.GetAttribute("physxRigidBody:staticFriction").Set(0.5)
                # print(link.get_attribute("physxRigidBody:dynamicFriction"))
                # print(link.get_attribute("physxRigidBody:staticFriction"))

    def apply_cached_semantic_perturbations(self, perturbation):
        tmp = self.cfg["cached_semantic_perturbations"][perturbation]
        idx = np.random.randint(0, len(tmp))
        self.instruction = tmp[idx]

    def s_prop(self):
        self.apply_cached_semantic_perturbations("S-PROP")

    def s_lang(self):
        synonyms: dict[str, list[str]] = self.cfg.get("synonyms", None)
        if synonyms is None:
            self.apply_cached_semantic_perturbations("S-LANG")
        n_synonyms_comb = np.prod([(len(v) + 1) for v in synonyms.values()]) - 1
        s_langs = self.cfg["cached_semantic_perturbations"].get("S-LANG", None)
        if s_langs is not None:
            n_s_langs = len(s_langs)
            if np.random.random() < n_s_langs / (n_synonyms_comb + n_s_langs):
                self.apply_cached_semantic_perturbations("S-LANG")

        orig_instruction: str = self.cfg["instruction"]
        instruction = orig_instruction.lower()
        instruction_words = instruction.split()

        synonyms: dict[str, list[str]] = self.cfg["synonyms"]
        number_words_which_can_be_replaced = len(synonyms)
        # Picking with 50% which words to replace with synonyms
        word_idx_to_replace = np.random.randint(2, size=number_words_which_can_be_replaced)
        # Making sure that at least one word will be replaced
        guaranteed_replaced_word_idx = np.random.randint(number_words_which_can_be_replaced)
        word_idx_to_replace[guaranteed_replaced_word_idx] = 1

        for word_idx, (word, syns) in enumerate(synonyms.items()):
            if not word_idx_to_replace[word_idx]:
                continue
            for i, w in enumerate(instruction_words):
                if w == word:
                    s = np.random.choice(syns)
                    instruction_words[i] = s

        self.instruction = " ".join(instruction_words).capitalize()

    def s_mo(self):
        self.apply_cached_semantic_perturbations("S-MO")

    def s_aff(self):
        self.apply_cached_semantic_perturbations("S-AFF")

    def s_int(self):
        self.apply_cached_semantic_perturbations("S-INT")

    def sb_noun(self):
        if self.task_type in ["open_drawer", "close_drawer"]:
            adjective = random.choice(["middle", "top"])
            #adjective = random.choice(["middle", "bottom"])
            self.instruction = self.cfg["instruction"].replace("top", adjective)
            self.reset_joints(target_drawer_loc=adjective)
            return

        i = np.random.randint(len(self.distractors))
        new_mo = self.distractors.pop(i)
        new_obj_for_task = new_mo.category
        self.instruction = self.cfg["instruction"].replace(self.cfg["instruction_obj_to_replace"], new_obj_for_task)
        self.instruction = self.instruction.replace("_", " ")

        self.distractors.append(self.main_objects[0])
        self.main_objects[0] = new_mo
        print([obj.name for obj in self.main_objects])
        print([obj.name for obj in self.distractors])


    def sb_vrb(self):
        #all_available_task_types = [task for task in SUPPORTED_TASK_TYPES if task != self.task_type]

        compatibility_matrix = {
            "put": ["pick", "rotate", "stack"],
            "push": [], #["put", "pick", "rotate", "stack"],
            "pick": ["put", "rotate", "stack"],
            "rotate": ["put", "pick", "stack"],
            "stack": ["put", "pick", "rotate"],
            "open": ["close"],
            "close": ["open"]
        }

        available_task_types = compatibility_matrix[self.task_type]

        new_verb_for_task = random.choice(available_task_types)
        self.task_type = new_verb_for_task

        if new_verb_for_task in ["rotate", "push", "pick", "open", "close"]:
            tmp = "pick up" if new_verb_for_task == "pick" else new_verb_for_task
            self.instruction = f"{tmp} the {self.cfg['instruction_obj_to_replace']}"
        elif new_verb_for_task == "stack":
            self.instruction = f"stack the {self.cfg['instruction_obj_to_replace']} on top of the {self.cfg['instruction_target_to_replace']}"
        elif new_verb_for_task == "put":
            self.instruction = f"put the {self.cfg['instruction_obj_to_replace']} into the {self.cfg['instruction_target_to_replace']}"
        else:
            raise NotImplementedError()
        self.task_progression = TASK_PROGRESSIONS[self.task_type]

        included_categories = None
        if self.task_type == "put":
            included_categories = ["bowl", "wineglass"]

        if len(self.target_objects) == 0:
            nobj_cfg = self.sample_objects(num_objects=1, included_categories=included_categories)[0]
            self.cfg['instruction_target_to_replace'] = nobj_cfg["category"]
            nobj_cfg["name"] = "receiver"

            new_obj = DatasetObject(
                name="receiver",
                relative_prim_path="/receiver",
                category=nobj_cfg["category"],
                model=nobj_cfg["model"],
            )
            self.omnigibson_env.scene.add_object(new_obj)
            self.target_objects = [new_obj]

            bbox_center, bbox_orn, bbox_extent, bbox_center_in_frame = new_obj.get_base_aligned_bbox()
            nobj_cfg["bounding_box"] = bbox_center

            max_dim = np.max(bbox_extent.numpy())
            new_scale_factor = 0.185 / max_dim
            if new_scale_factor < 1.0:
                new_obj.scale = new_scale_factor
                nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor

            self.cfg["objects"].append(nobj_cfg)

            # --------------- Translation ---------------
            obj_cfgs = copy.deepcopy(self.cfg["objects"])
            num_mo_to = len(obj_cfgs) - 1

            self.cfg["objects"] = get_non_colliding_positions_for_objects_v2(
                xmin=self.spawn_bbox[0],
                xmax=self.spawn_bbox[1],
                ymin=self.spawn_bbox[2],
                ymax=self.spawn_bbox[3],
                z=self.spawn_bbox[4],
                obj_cfg=obj_cfgs,
                objects_to_skip=[obj.name for obj in self.main_objects + self.distractors],
                main_object_names=[o["name"] for o in obj_cfgs[:num_mo_to]]
            )

            pos = torch.tensor(self.cfg["objects"][-1]["position"])
            rot = torch.tensor(self.cfg["objects"][-1]["orientation"] if "orientation" in self.cfg["objects"][-1] else [0,0,0,1])
            new_obj.set_bbox_center_position_orientation(pos, rot)

            self.init_poses[new_obj._relative_prim_path] = {}
            self.init_poses[new_obj._relative_prim_path]["pos"] = pos
            self.init_poses[new_obj._relative_prim_path]["rot"] = rot

            # --------------- Set Position ---------------
            for obj in self.cfg["objects"]:
                self.omnigibson_env.scene.object_registry("name", obj["name"]).set_position(obj["position"])

        og.sim.step()
        if self.task_type in ["put", "stack"]:
            og.sim.stop()
            # --------------- Replace the objects models ---------------
            nobj, _ = self.replace_obj(self.target_objects[0], included_categories=included_categories, maximum_dim=0.185)
            self.target_objects = [nobj]
            og.sim.play()
            # fake rest to get to original pose after stopping sim
            for _ in range(30):
                self.omnigibson_env.step(np.concatenate((self.reset_qpos[:7], np.atleast_1d(np.array([-1])))))

    def vb_mobj(self):
        # sample rescaling of the bbox
        for _ in range(1000):
            s1 = np.random.uniform(0.5, 1.5)
            s2 = np.random.uniform(0.5, 1.5)
            s3 = np.random.uniform(0.5, 1.5)
            if s1 * s2 * s3 <= 1.5:
                break

        scene = self.omnigibson_env.scene
        mo = self.main_objects[0]

        #if type(mo) != DatasetObject:
        if type(mo) == PrimitiveObject:
            # assumes the primitives have a defautl scale 1,1,1 hence the orig bbox can be used as replacement
            og.sim.stop()
            scale = torch.tensor([s1, s2, s3])
            # if self.task_type in ["open_drawer", "close_drawer"]:
            #     scale[-1] = 1
            mo.scale = torch.tensor(self.mo_bbox_orig) * scale
            og.sim.play()
            for _ in range(30):
                self.omnigibson_env.step(np.concatenate((self.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
        else:
            obj_name = mo.name
            obj_relative_prim_path = mo._relative_prim_path
            new_bbox = self.mo_bbox_orig * np.array([s1, s2, s3])

            og.sim.stop()
            scene.remove_object(mo)

            if self.task_type in ["open_drawer", "close_drawer"]:
                new_bbox = np.clip(new_bbox, a_min=0.4, a_max=0.75)
                fix_base = True
            else:
                new_bbox = np.clip(new_bbox, a_min=0.02, a_max=0.175)
                fix_base = False

            if type(mo) == DatasetObject:
                obj_cfg = get_default_objects_cfg(self.omnigibson_env.scene, [mo.name])[obj_name]
                new_obj = DatasetObject(
                    name=obj_name,
                    relative_prim_path=obj_relative_prim_path, #obj_cfg["relative_prim_path"],
                    category=mo.category,
                    model=mo.model,
                    bounding_box=torch.tensor(new_bbox, dtype=torch.float32),
                    fixed_base=fix_base
                )
                scene.add_object(new_obj)
                new_obj.set_bbox_center_position_orientation(obj_cfg["pos"], obj_cfg["ori"])
            else:
                assert type(mo) == USDObject
                raise NotImplementedError()
                # new_obj = USDObject(
                #     name=obj_name,
                #     relative_prim_path=obj_relative_prim_path,
                #     usd_path="/app/custom_assets/impact_drawer/usd/test_v123.usd",
                #     bounding_box = torch.tensor(new_bbox, dtype=torch.float32),
                #     fixed_base = fix_base
                # )
                # scene.add_object(new_obj)
                # new_obj.set_position_orientation(torch.tensor(self.init_poses[new_obj._relative_prim_path]["pos"]),
                #                        torch.tensor(self.init_poses[new_obj._relative_prim_path]["rot"]))

            self.main_objects = [new_obj]
            og.sim.play()
            og.sim.step()
            self.omnigibson_env.scene.update_initial_state()
            self.reset_joints()

    def v_sc(self):
        # --------------- Translation ---------------
        og.sim.stop()

        obj_cfgs = copy.deepcopy(self.cfg["objects"])
        num_mo_to = len(self.target_objects + self.main_objects)

        self.cfg["objects"] = None
        num_distractors = len(obj_cfgs) - num_mo_to

        self.cfg["objects"] = get_non_colliding_positions_for_objects_v2(
                xmin=self.spawn_bbox[0],
                xmax=self.spawn_bbox[1],
                ymin=self.spawn_bbox[2],
                ymax=self.spawn_bbox[3],
                z=self.spawn_bbox[4],
                obj_cfg=obj_cfgs[:num_mo_to + num_distractors],
                objects_to_skip=[obj.name for obj in self.target_objects + self.main_objects],
                main_object_names=[o["name"] for o in obj_cfgs[:num_mo_to]],
                maximum_dim=0.12,
            )

        self.distractors = [self.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in self.cfg["objects"][num_mo_to:]]

        # TODO: check if this works properly in the edge cases where it should trigger
        if num_distractors < len(self.distractors):
            for dist_cfg in self.cfg["objects"][num_mo_to + num_distractors:]:
                obj = self.omnigibson_env.scene.object_registry("name", dist_cfg["name"])
                self.omnigibson_env.scene.remove_object(obj)
            self.cfg["objects"] = self.cfg["objects"][:num_mo_to + num_distractors]

        # --------------- Set Position ---------------
        for obj in self.cfg["objects"]:
            self.omnigibson_env.scene.object_registry("name", obj["name"]).set_position(obj["position"])

        # TODO: support this again? rn we just use default rot for the objects
        # # --------------- Set Rotation ---------------
        # for o in self.distractors:
        #     tmp = o.get_orientation()
        #     o.set_orientation(add_rotation_noise(tmp, (3.14, 3.14, 3.14)))

        # --------------- Replace the objects models ---------------
        distractor_obj_cfgs = get_default_objects_cfg(self.omnigibson_env.scene, [obj.name for obj in self.distractors])
        distractor_objs = get_objects_by_names(self.omnigibson_env.scene, list(distractor_obj_cfgs.keys()))
        excluded_categories = [obj.category for obj in self.main_objects + self.target_objects]
        for distractor in distractor_objs:
            cat_dict = get_droid_categories_by_theme()
            t = [k for k, v in cat_dict.items() if any(distractor.category in c for c in v.values())]
            if t:
                cat_dict.pop(t[0])
            l = [o for v in cat_dict.values() for c in v.values() for o in c]
            l = [c for c in l if c not in excluded_categories]
            _, _ = self.replace_obj(distractor, included_categories=l, maximum_dim=0.12)

        og.sim.play()
        self.reset_joints()
        # fake rest to get to original pose after stopping sim
        for _ in range(30):
            self.omnigibson_env.step(np.concatenate((self.reset_qpos[:7], np.atleast_1d(np.array([-1])))))


    # ============================== [ROLLOUT UTILS] ==============================
    def warmup(self, obs=None):
        print("Starting warmup...")
        for _ in range(30):
            og.sim.render()

        if obs is None:
            obs, _ = self.reset()

        is_gripper_closed = True
        for t in range(19):
            new_action = np.concatenate((
                obs['franka']['proprio'][:7].cpu().numpy(),
                np.atleast_1d(np.zeros(1))
            ))
            if t != 0 and t % 10 == 0:
                is_gripper_closed = not is_gripper_closed
            new_action[-1] = 1 if is_gripper_closed else -1

            obs, rew, terminated, truncated, info = self.step(new_action)

        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()
        print("Warmup finished.")
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

    def step(self, action):
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
            print(
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

        self.omnigibson_env.scene.remove_object(obj)

        if len(included_categories) == 1 and "bottom_cabinet" in included_categories:
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
            nobj_cfg = self.sample_objects(num_objects=1, included_categories=included_categories)[0]

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