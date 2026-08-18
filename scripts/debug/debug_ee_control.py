import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes  # renamed from get_available_og_scenes in OG 3.9.1

import omnigibson.utils.transform_utils as T
import numpy as np
import torch as th

USE_DROID_WITH_BASE = True
# OG 3.9.1: no DROID class to import -- select the RobotDefinition by model name instead.
DROID_MODEL = "droid_mounted" if USE_DROID_WITH_BASE else "droid"

# Registers REALM's controllers *and* their default configs, which OG 3.9.1 requires separately.
from realm.robots.controller_registry import register_realm_controllers
register_realm_controllers()

freq = 15 #60
gm.DEFAULT_SIM_STEP_FREQ = freq
gm.DEFAULT_RENDERING_FREQ = freq
gm.DEFAULT_PHYSICS_FREQ = 120
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = False #True
gm.ENABLE_FLATCACHE = False #True

ep_id = "episode_000001"
action_cartesian_pos = np.load(f"data/droid_1.0.1/extracted_eps/chunk-000/{ep_id}/action_cartesian_position.npy", allow_pickle=True)
action_qpos = np.load(f"data/droid_1.0.1/extracted_eps/chunk-000/{ep_id}/action_joint_position.npy", allow_pickle=True)

state_cartesian_pos = np.load(f"data/droid_1.0.1/extracted_eps/chunk-000/{ep_id}/observation_state_cartesian_position.npy", allow_pickle=True)
state_qpos = np.load(f"data/droid_1.0.1/extracted_eps/chunk-000/{ep_id}/observation_state_joint_position.npy", allow_pickle=True)

cfg = dict()

# Define scene
scene_id = 0
scenes = get_available_behavior_1k_scenes()
scene_model = list(scenes)[scene_id]
cfg["scene"] = {
     "type": "Scene",
     "floor_plane_visible": True,
}

cfg["robots"] = [
    {
        "name": "DROID",
        "model": DROID_MODEL,
        "obs_modalities": ["proprio"], #"rgb",
        "proprio_obs": ["joint_qpos"],
        "position": [0, 0, 0], #0.87],
        "reset_joint_pos": list(state_qpos[0]) + [0, 0, 0, 0],
        #"reset_joint_pos": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #list(state_qpos[0]) + [0, 0, 0, 0],
        "orientation": T.euler2quat(th.tensor([0, 0, 0], dtype=th.float32)).tolist(),
        "control_freq": freq,
        "action_normalize": False,
        "controller_name": "EEController", #"JointController",
        "controller_config": {
            "arm_0": {
                "name": "EEController",
                "motor_type": "effort",
                "mode": "absolute_pose",
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
            "orientation": th.tensor([ 0.5, -0.5, -0.5, 0.5 ], dtype=th.float32),
            "pose_frame": "parent"
        },
    ],
}

# Create the environment
env = og.Environment(cfg)

# Allow camera teleoperation
og.sim.enable_viewer_camera_teleoperation()

obs, _ = env.reset()
video = []
close = False
flip = True

from scipy.spatial.transform import Rotation as R
for t in range(175):
    robot_state = obs['DROID']['proprio'][:7].cpu().numpy()

    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    video.append(base_im)

    # a = np.zeros(7)
    # if t % 30 == 0:
    #     flip = not flip
    #a = np.array([0.3, 0.3, 0.1, 0.0, 0.0, 0.0, -1.0])
    def get_ee(og_env):
        ee_link_name = og_env.robots[0].eef_link_names[og_env.robots[0].default_arm]
        ee_link = og_env.robots[0].links[ee_link_name]
        return ee_link.get_position_orientation()

    ee_pos, ee_rot = get_ee(env)
    ee_rot = R.from_quat(ee_rot.cpu().numpy()).as_euler('xyz')

    #a[3:6] = flip_pose_pointing_down(a[3:6])
    a = np.concatenate([ee_pos, ee_rot, np.atleast_1d(np.array(-1.0))])
    a[2] -= 0.87

    obs, rew, terminated, truncated, info = env.step(th.from_numpy(a))

# video = np.stack(video)
# save_filename = f"/app/logs/debug_ee_control"
# ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")

#og.shutdown()
print("Done!")
