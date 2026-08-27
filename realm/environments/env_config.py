
import copy
import math
import os

import numpy as np
import torch
import yaml

import omnigibson.utils.transform_utils as omnigibson_transform_utils

from realm.environments.constants import (
    DEFAULT_RESET_JOINTPOS,
    DROID_BASE_HEIGHT,
    DROID_DEFAULT_DOF,
)
from realm.environments.perturbations.object_sampling import sample_objects
from realm.placement import place_within


def build_environment_config(env):

    cfg = dict()
    task_cfg = yaml.load(open(f"{env.config_path}/tasks/{env.task_cfg_path}", "r"), Loader=yaml.FullLoader)
    cfg.update(task_cfg)
    scene_cfg, scene_data = _apply_scene_cfg(env, cfg, task_cfg)
    robot_pos, robot_rot = _apply_robot_cfg(env, cfg, task_cfg, scene_data)
    distractors = _apply_object_cfg(env, cfg, task_cfg, scene_cfg, scene_data)
    _apply_env_cfg(cfg)
    _apply_camera_cfg(env, cfg, task_cfg, robot_pos, robot_rot)

    return (copy.deepcopy(cfg),
            copy.deepcopy(task_cfg["main_objects"]),
            copy.deepcopy(task_cfg["target_objects"]),
            copy.deepcopy(distractors))


def _apply_scene_cfg(env, cfg, task_cfg):

    for k in ["external_sensors", "robots"]:
        assert k not in cfg, f"{k} should be defined outside the scene file!"

    if env.scene_model is None:
        assert env.scene_part is None
        env.scene_model = list(task_cfg["supported_scenes"].keys())[0]
        env.scene_part = task_cfg["supported_scenes"][env.scene_model][0]
    assert env.scene_model in task_cfg["supported_scenes"]
    assert env.scene_part in task_cfg["supported_scenes"][env.scene_model]
    cfg.update(task_cfg["task"])

    scene_cfg_path = f"{env.config_path}/scenes/{env.scene_model}/{env.scene_part}/scene_definition.yaml"
    scene_cfg = None
    if os.path.exists(scene_cfg_path):
        scene_cfg = yaml.load(open(scene_cfg_path, "r"), Loader=yaml.FullLoader)
        cfg["scene"] = copy.deepcopy(scene_cfg["scene"])
    else:
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": env.scene_model,
        }

    spawn_cfg = yaml.load(open(f"{env.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)
    assert env.scene_model in spawn_cfg and env.scene_part in spawn_cfg[env.scene_model]
    scene_data = spawn_cfg[env.scene_model][env.scene_part]
    if all(k in scene_data for k in ["x_min", "x_max", "y_min", "y_max", "z"]):
        x_min = scene_data["x_min"]
        x_max = scene_data["x_max"]
        y_min = scene_data["y_min"]
        y_max = scene_data["y_max"]
        z = scene_data["z"]
        env.spawn_bbox = np.array([x_min, x_max, y_min, y_max, z])
    else:
        env.spawn_bbox = None
    return scene_cfg, scene_data


def _apply_robot_cfg(env, cfg, task_cfg, scene_data):

    assert "pos" in scene_data and "rot" in scene_data
    robot_pos = scene_data['pos']
    robot_rot = [math.radians(angle_deg) for angle_deg in scene_data['rot']]
    env.robot_pos = np.array(robot_pos, dtype=float)
    env.robot_rot_rad = np.array(robot_rot, dtype=float)

    cfg_robot = yaml.load(open(f"{env.config_path}/robots/{env.robot_name}.yaml", "r"), Loader=yaml.FullLoader)
    env.ee_control = cfg_robot["robots"][0].get("ee_control", False)
    # Column-free assets have their origin at the arm base.
    spawn_pos = list(robot_pos)
    if env.use_droid_with_base and not cfg_robot["robots"][0].pop("has_base_column", True):
        spawn_pos[2] += DROID_BASE_HEIGHT
    cfg_robot["robots"][0]["position"] = spawn_pos
    cfg_robot["robots"][0]["orientation"] = omnigibson_transform_utils.euler2quat(
        torch.tensor(robot_rot, dtype=torch.float32)).tolist()
    cfg_robot["robots"][0]["fixed_base"] = True

    reset_joint_pos = np.zeros(cfg_robot["robots"][0]["dof"] if "dof" in cfg_robot["robots"][0] else DROID_DEFAULT_DOF)
    if "DROID" in env.robot_name:
        if "reset_joint_pos" in task_cfg:
            reset_joint_pos[:7] = np.array(task_cfg['reset_joint_pos'])
        elif "reset_joint_pos" in scene_data:
            reset_joint_pos[:7] = np.array(scene_data['reset_joint_pos'])
        else:
            reset_joint_pos[:7] = DEFAULT_RESET_JOINTPOS
    elif env.robot_name == "WidowX":
        reset_joint_pos[:6] = np.zeros(6)
    cfg_robot["robots"][0]["reset_joint_pos"] = reset_joint_pos

    if env.common_freq is not None:
        cfg_robot["robots"][0]["control_freq"] = env.common_freq
        cfg_robot["robots"][0]["controller_config"]["arm_0"]["control_freq"] = env.common_freq

    cfg.update(cfg_robot)
    env.reset_qpos = reset_joint_pos
    return robot_pos, robot_rot


def _apply_object_cfg(env, cfg, task_cfg, scene_cfg, scene_data):

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

    if env.spawn_bbox is not None:
        for obj in obj_list:
            obj["relative_bbox_position"][0] *= obj_pos_modifier_x
            if obj_pos_modifier_x != 1:
                if obj["relative_bbox_position"][0] < 0:
                    obj["relative_bbox_position"][0] -= obj_pos_modifier_x * (env.spawn_bbox[1] - env.spawn_bbox[0])
                else:
                    obj["relative_bbox_position"][0] += obj_pos_modifier_x * (env.spawn_bbox[1] - env.spawn_bbox[0])
            obj["position"] = [x + y for x, y in zip(obj["relative_bbox_position"], [env.spawn_bbox[0], env.spawn_bbox[2], env.spawn_bbox[4]])]

        num_distractors = 3 if any(p in env.active_perturbations for p in ["V-SC"]) else 0
        cfg["objects"] = None
        excluded_categories = []
        for obj in task_cfg["main_objects"] + task_cfg["target_objects"]:
            if "category" in obj:
                excluded_categories.append(obj["category"])
        distractors = sample_objects(num_objects=num_distractors, excluded_categories=excluded_categories)

        cfg["objects"] = place_within(
            env.spawn_bbox,
            obj_list + distractors,
            max_attempts_per_object=25000,
            main_object_names=[o["name"] for o in obj_list],
        )
    else:
        cfg["objects"] = obj_list
        distractors = []

    if "distractors" in task_cfg:
        distractors += task_cfg["distractors"]
    if "immutables" in task_cfg:
        distractors += task_cfg["immutables"]

    for obj in cfg["objects"]:
        assert "position" in obj
    return distractors


def _apply_env_cfg(cfg):

    if "env" not in cfg:
        cfg["env"] = {
            "initial_pos_z_offset": 0.2
        }
    if os.environ.get("REALM_GPU_DYNAMICS") == "1":
        cfg["env"]["device"] = os.environ.get("REALM_TORCH_DEVICE", "cuda:0")


def _apply_camera_cfg(env, cfg, task_cfg, robot_pos, robot_rot):

    if not env.no_rendering:
        ext_cam1_pose = task_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in task_cfg else "default"
        if "camera_extrinsics" in task_cfg and "cam2" in task_cfg["camera_extrinsics"]:
            ext_cam2_pose = task_cfg["camera_extrinsics"]["cam2"]
        else:
            ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"

        base_cam_pos, base_cam_rot = env.construct_ext_cam_pose_by_name(ext_cam1_pose, robot_pos, robot_rot)

        cfg_external_sensors = yaml.load(open(f"{env.config_path}/env/external_sensors/camera_config.yaml", "r"), Loader=yaml.FullLoader)
        cfg_external_sensors["external_sensors"][0]["position"] = base_cam_pos
        cfg_external_sensors["external_sensors"][0]["orientation"] = base_cam_rot

        if env.multi_view:
            second_base_cam_pos, second_base_cam_rot = env.construct_ext_cam_pose_by_name(ext_cam2_pose, robot_pos,
                                                                                           robot_rot)
            cfg_external_sensors["external_sensors"][1]["position"] = second_base_cam_pos
            cfg_external_sensors["external_sensors"][1]["orientation"] = second_base_cam_rot
        else:
            del cfg_external_sensors["external_sensors"][1]

        cfg["env"].update(cfg_external_sensors)
