import numpy as np

# Per-robot observation conventions. Keyed by the robot's name in the scene (robot.name, which comes
# from `name:` in realm/config/robots/*.yaml). Anything not listed falls back to "DROID".
#
#   wrist_camera_link  -- the link the wrist camera hangs off; the obs key is
#                         "<robot>:<link>:Camera:<idx>". droid.usd mounts it on `gripper_link_camera`;
#                         the robolab asset mounts two on the gripper's `base_link`.
#   wrist_camera_idx   -- which camera on that link (the robolab asset ships `wrist_camera` as :0 and
#                         `wrist_camera_flipped` as :1).
#   gripper_open_qpos / gripper_closed_qpos
#                      -- value of proprio[gripper_proprio_idx] at each extreme, used to normalise
#                         gripper_state to the DROID convention (0 = open, 1 = closed). The robolab
#                         gripper is a single revolute finger_joint that runs the *other* way:
#                         0 rad = closed, 0.7854 rad = fully open (measured, 85 mm pad separation).
ROBOT_OBS_PROFILES = {
    "DROID": dict(wrist_camera_link="gripper_link_camera", wrist_camera_idx=0,
                  gripper_proprio_idx=7, gripper_open_qpos=0.0, gripper_closed_qpos=0.05),
    # Camera:1 is `wrist_camera_flipped`, whose framing matches the stock DROID wrist view (fingers
    # entering symmetrically from the bottom); Camera:0 (`wrist_camera`) is rotated/offset, which
    # would hand the policy a view unlike anything in its training distribution.
    "DROID_robolab": dict(wrist_camera_link="base_link", wrist_camera_idx=1,
                          gripper_proprio_idx=7, gripper_open_qpos=0.7853982, gripper_closed_qpos=0.0),
}


def get_robot_obs_profile(robot_name):
    return ROBOT_OBS_PROFILES.get(robot_name, ROBOT_OBS_PROFILES["DROID"])


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    # Fallback to zeros if external sensors are missing (e.g. during no_render)
    if 'external' in obs and 'external_sensor0' in obs['external']:
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        # Dummy 128x128 image
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
        base_depth = np.zeros((128, 128), dtype=np.float32) if enable_depth else None

    if 'external' in obs and 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    # Handle wrist camera. The key depends on which link the asset mounts the camera on, so it comes
    # from the robot profile rather than being hardcoded -- a mismatch here fails silently as a black
    # image, which the policy consumes without complaint.
    profile = get_robot_obs_profile(robot_name)
    wrist_cam_key = f"{robot_name}:{profile['wrist_camera_link']}:Camera:{profile['wrist_camera_idx']}"
    if robot_name in obs and wrist_cam_key in obs[robot_name]:
        wrist_im = obs[robot_name][wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
    else:
        # Last resort: any camera on the robot, so a renamed link degrades to the wrong view rather
        # than to a black one. Warn, because either case means the profile needs updating.
        cam_keys = [k for k in obs.get(robot_name, {}) if ":Camera:" in k]
        if cam_keys:
            print(f"[extract_from_obs] WARNING: no '{wrist_cam_key}' in obs; "
                  f"falling back to '{cam_keys[0]}'. Update ROBOT_OBS_PROFILES for '{robot_name}'.")
            wrist_im = obs[robot_name][cam_keys[0]]['rgb'].cpu().numpy()[..., :3]
        else:
            wrist_im = np.zeros((128, 128, 3), dtype=np.uint8)

    # Proprio is always present in DROID and other robots
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    # Normalise to the DROID convention the policies expect: 0 = open, 1 = closed.
    _open, _closed = profile["gripper_open_qpos"], profile["gripper_closed_qpos"]
    gripper_state = float(np.clip((proprio[profile["gripper_proprio_idx"]] - _open) / (_closed - _open), 0.0, 1.0))

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
