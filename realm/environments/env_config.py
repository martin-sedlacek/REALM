"""Assembly of the OmniGibson environment config from REALM's YAML layers.

One REALM environment is stitched together from five config files:
  config/tasks/<suite>/<task>/<variant>.yaml        task, objects, instruction, camera choice
  config/scenes/<model>/<part>/scene_definition.yaml   optional per-part scene override
  config/scenes/scenes.yaml                        robot spawn pose + object spawn bbox per part
  config/robots/<robot>.yaml                       robot model, DOF, controller config
  config/env/external_sensors/camera_config.yaml   external camera sensor spec

The section builders take the RealmEnvironmentDynamic instance because they also populate
attributes the environment needs later: spawn_bbox, robot_pos, robot_rot_rad, ee_control,
reset_qpos, and the resolved scene_model/scene_part.
"""
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
    """Build the OmniGibson env config. Returns (cfg, mo_cfgs, to_cfgs, dist_cfgs)."""
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
    """Resolve scene_model/scene_part, set cfg["scene"], and read the object spawn bbox.

    Returns (scene_cfg, scene_data): the optional per-part scene override (None when the part
    has no scene_definition.yaml) and that part's entry in scenes.yaml.
    """
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
    """Merge config/robots/<robot>.yaml into cfg, resolving spawn pose and reset joint pos.

    Returns the raw (pos, rot_radians) from scenes.yaml, which camera poses are relative to.
    """
    assert "pos" in scene_data and "rot" in scene_data
    robot_pos = scene_data['pos']
    robot_rot = [math.radians(angle_deg) for angle_deg in scene_data['rot']]
    env.robot_pos = np.array(robot_pos, dtype=float)
    env.robot_rot_rad = np.array(robot_rot, dtype=float)

    cfg_robot = yaml.load(open(f"{env.config_path}/robots/{env.robot_name}.yaml", "r"), Loader=yaml.FullLoader)
    env.ee_control = cfg_robot["robots"][0].get("ee_control", False)
    # Assets without the DROID base column have their origin at the arm base rather than at the
    # bottom of the column, so on base-mounted tasks they must be raised by the column's height.
    # env.robot_pos deliberately stays at the scene value: _robot2world/_world2robot already add
    # DROID_BASE_HEIGHT themselves, so only the spawn point needs adjusting.
    spawn_pos = list(robot_pos)
    if env.use_droid_with_base and not cfg_robot["robots"][0].pop("has_base_column", True):
        spawn_pos[2] += DROID_BASE_HEIGHT
    cfg_robot["robots"][0]["position"] = spawn_pos
    cfg_robot["robots"][0]["orientation"] = omnigibson_transform_utils.euler2quat(
        torch.tensor(robot_rot, dtype=torch.float32)).tolist()
    cfg_robot["robots"][0]["fixed_base"] = True

    # OG 3.9.1 selects a robot by `model` (a RobotDefinition YAML name), not by Python class:
    # DROID and UR are declared as RobotDefinition YAMLs under realm/robots/definitions/ and
    # instantiated by OmniGibson's single Robot class, and WidowX uses OmniGibson's stock `vx300s`
    # definition -- so there is no robot class for the environment to import any more.
    # The base-mounted DROID used to be chosen by importing a different module; it is now a
    # separate definition. `type` in the REALM robot configs is still accepted -- OmniGibson
    # lowercases it into `model` -- but we set `model` explicitly so the mounted variant works.
    # A config that names its own `model` (e.g. DROID_robolab.yaml) is left alone -- only the
    # stock DROID configs, which still carry the legacy `type: DROID`, get the mounted/unmounted
    # definition chosen for them here.
    if "DROID" in env.robot_name and "model" not in cfg_robot["robots"][0]:
        cfg_robot["robots"][0].pop("type", None)
        cfg_robot["robots"][0]["model"] = "droid_mounted" if env.use_droid_with_base else "droid"

    reset_joint_pos = np.zeros(cfg_robot["robots"][0]["dof"] if "dof" in cfg_robot["robots"][0] else DROID_DEFAULT_DOF)
    if "DROID" in env.robot_name:
        if "reset_joint_pos" in task_cfg:
            reset_joint_pos[:7] = np.array(task_cfg['reset_joint_pos'])
        elif "reset_joint_pos" in scene_data:
            reset_joint_pos[:7] = np.array(scene_data['reset_joint_pos'])
        else:
            reset_joint_pos[:7] = DEFAULT_RESET_JOINTPOS
    elif env.robot_name == "WidowX":
        reset_joint_pos[:6] = np.zeros(6) #np.array([0.0, -0.849879, 0.258767, 0.0, 1.2831712, 0.0])
    cfg_robot["robots"][0]["reset_joint_pos"] = reset_joint_pos

    if env.common_freq is not None:
        cfg_robot["robots"][0]["control_freq"] = env.common_freq
        cfg_robot["robots"][0]["controller_config"]["arm_0"]["control_freq"] = env.common_freq

    cfg.update(cfg_robot)
    env.reset_qpos = reset_joint_pos
    return robot_pos, robot_rot


def _apply_object_cfg(env, cfg, task_cfg, scene_cfg, scene_data):
    """Place task objects, distractors and immutables into cfg["objects"].

    Returns the distractor configs (sampled + declared), reported separately by the caller.
    """
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

        # TODO: the pipeline is broken for dynamically reducing # objects when there are too many distractors and
        # they become unplaceable - 3 is always fine and easy to place so we use that for now as maximum
        num_distractors = 3 if any(p in env.active_perturbations for p in ["V-SC"]) else 0 #"VB-ISC" #"SB-NOUN"
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
        distractors += task_cfg["immutables"] # immutables go here because the distractor list above is meant to be replaceable objects

    for obj in cfg["objects"]:
        assert "position" in obj
    return distractors


def _apply_env_cfg(cfg):
    """Create cfg["env"] and pin the torch backend device when GPU dynamics is on.

    The device has to move with gm.USE_GPU_DYNAMICS -- both are driven by the same
    REALM_GPU_DYNAMICS knob (realm/sim_config.py). With GPU dynamics on but the backend left on
    OmniGibson's "cpu" default, the PhysX articulation view lives on the GPU and
    ArticulationView.get_joint_positions() returns None to a CPU reader, so the first get_obs()
    after reset dies with `AttributeError: 'NoneType' object has no attribute 'view'`
    (entity_prim.py:864), which Isaac then turns into a segfault. Measured 2026-08-13, job 190243.
    """
    if "env" not in cfg:
        cfg["env"] = {
            "initial_pos_z_offset": 0.2
        }
    if os.environ.get("REALM_GPU_DYNAMICS") == "1":
        cfg["env"]["device"] = os.environ.get("REALM_TORCH_DEVICE", "cuda:0")


def _apply_camera_cfg(env, cfg, task_cfg, robot_pos, robot_rot):
    """Add the external camera sensors to cfg["env"] (skipped when rendering is off)."""
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
