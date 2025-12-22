import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes

import omnigibson.utils.transform_utils as T
import numpy as np
import torch as th
import math
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

USE_DROID_WITH_BASE = True
if USE_DROID_WITH_BASE:
    from realm.robots.franka_robotiq_mounted import FrankaPandaRobotiq
else:
    from realm.robots.franka_robotiq import FrankaPandaRobotiq

from omnigibson.controllers import REGISTERED_CONTROLLERS
from realm.robots.droid_joint_controller import IndividualJointPDController
if "CustomJointController" not in REGISTERED_CONTROLLERS:
    REGISTERED_CONTROLLERS["CustomJointController"] = IndividualJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController
if "CustomGripperController" not in REGISTERED_CONTROLLERS:
    REGISTERED_CONTROLLERS["CustomGripperController"] = MultiFingerGripperController

freq = 15 #60
gm.DEFAULT_SIM_STEP_FREQ = freq
gm.DEFAULT_RENDERING_FREQ = freq
gm.DEFAULT_PHYSICS_FREQ = 120
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = False #True
gm.ENABLE_FLATCACHE = False #True


cfg = dict()

# Define scene
scene_id = 0
scenes = get_available_og_scenes()
scene_model = list(scenes)[scene_id]
cfg["scene"] = {
     "type": "Scene",
     "floor_plane_visible": True,
}

# kp_values = [20, 50, 25, 25, 15, 10, 10]
# damping_ratios = [1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5]

# Define robots
cfg["robots"] = [
    {
        "type": "FrankaPandaRobotiq",
        "name": "franka",
        "obs_modalities": ["proprio"], #"rgb",
        "proprio_obs": ["joint_qpos"],
        "position": [0, 0, 0], #0.87],
        "orientation": T.euler2quat(th.tensor([0, 0, -math.pi], dtype=th.float32)).tolist(),
        "control_freq": freq,
        "action_normalize": False,
        "controller_name": "CustomJointController", #"JointController",
        "controller_config": {
            "arm_0": {
                "name": "CustomJointController", # "JointController"
                "motor_type": "position",
                "control_freq": 15,
                "use_delta_commands": False,
                "use_impedances": True,
                "use_gravity_compensation": False,
                "use_cc_compensation": False,
                "Kq": [40, 30, 50, 35, 35, 25, 10],
                "Kqd": [4, 6, 5, 5, 3, 2, 1],
                "Kx": [400, 400, 400, 15, 15, 15],
                "Kxd": [37, 37, 37, 2, 2, 2],
                "command_output_limits": None,
                "command_input_limits": None
            },
            # "arm_0": {
            #     "name": "JointController",
            #     "motor_type": "position",
            #     "control_freq": freq,
            #     "command_input_limits": None,
            #     "use_delta_commands": False
            # },
            "gripper_0": {
                "name": "CustomGripperController",
                "mode": "binary",
            }
        }
    }
]

# Define task
cfg["task"] = {
    "type": "DummyTask",
    "termination_config": dict(),
    "reward_config": dict(),
}

cfg["env"] = {
    "external_sensors": [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": 720,
                "image_width": 1280,
            },
            "position": th.tensor([-1.15716, 0, 0.73043], dtype=th.float32),
            #"position": th.tensor([-0.1, -3.3, 1.3675], dtype=th.float32),
            "orientation": th.tensor([ 0.5, -0.5, -0.5, 0.5 ], dtype=th.float32),
            #"orientation": th.tensor([ 0.6421758, 0.0334144, 0.028038, 0.7653153 ], dtype=th.float32),
            "pose_frame": "parent"
        },
    ],
}

cfg["objects"] = [
    {
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "lcsizg",
        "position": [-0.825, -0., 0.55], #[-0.25, -2.15, .9],
        #"orientation": [ 0.6903455, 0.1530459, 0.1530459, 0.6903455 ]
    },
    {
        "type": "PrimitiveObject",
        "name": "obj0",
        "primitive_type": "Cube",
        "fixed_base": False,
        #"kinematic_only": True,
        "scale": [0.025, 0.025, 0.025],
        #"position": [-0.475, -0.02, 0.73],
        "position": [-0.465, -0.01, 0.715],
        "orientation": [0, 0, 0, 1],
    },
    # {
    #     "type": "DatasetObject",
    #     "name": "obj0",
    #     "category": "banana",
    #     "model": "vvyyyv",
    #     "position": [-0.46, -0.01, 0.715],
    # },
]

# position: [-0.2, -2.1, 0.88]
# orientation: [-0.7071068, 0, 0, 0.7071068]
# scale: [0.7, 0.7, 0.7]

# Create the environment
env = og.Environment(cfg)

# Allow camera teleoperation
og.sim.enable_viewer_camera_teleoperation()

obs, _ = env.reset()

# target_objects = [env.scene.object_registry("name", "obj0")]
# for obj in target_objects:
#     for link in obj._links.values():
#         link.mass = 0.1

traj_id = "traj_0"
#qpos = np.load(f"/examples/{traj_id}/action_qpos.npy")

grasp_state = np.array([
    0., #0.25,
    1.0, #0.95
    0.,
    -2,
    0.,
    2.75, #2.825,
    0.
])

reach_state = grasp_state.copy()
reach_state[1] = 0
reach_state[6] += .3

video = []

close = False

# for t in range(1000):
#     if t % 50 == 0:
#         close = not close
#
#     robot_state = obs['franka']['proprio'][:7].cpu().numpy()
#
#     sample = env.action_space.sample()
#     sample['franka'][:-1] = reach_state
#     sample['franka'][-1] = 1 if close else -1
#     obs, rew, terminated, truncated, info = env.step(sample)

for t in range(300):
    sample = env.action_space.sample()
    robot_state = obs['franka']['proprio'][:7].cpu().numpy()

    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    video.append(base_im)

    if t < 30:
        intended_state = reach_state
    else:
        intended_state = grasp_state
    #print(t, np.round(intended_state - robot_state, 4))
    #print(np.round(robot_state, 4))
    sample['franka'][:-1] = intended_state #robot_state
    #sample['franka'][:-1] = np.zeros_like(sample['franka'])[:-1]
    sample['franka'][-1] = -1

    if t > 60:
        sample['franka'][0] -= 0.001 * (t - 100)
    if t > 90:
        sample['franka'][-1] = 1
    if t > 250:
        sample['franka'][1] -= 0.15

    print(t, sample)
    obs, rew, terminated, truncated, info = env.step(th.from_numpy(sample['franka']))

video = np.stack(video)
save_filename = f"/app/logs/debug_weird_grippe"
ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")

#og.shutdown()
print("Done!")
