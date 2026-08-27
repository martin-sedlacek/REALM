import numpy as np

# Camera indices follow sensor creation order and must match each robot config's sensor filter.
ROBOT_OBS_PROFILES = {
    # Robotiq finger_joint: 0 rad is open; 0.7854 rad is closed.
    "DROID_mounted": dict(wrist_camera_link="base_link", wrist_camera_idx=0,
                             wrist_camera_prim="wrist_camera_flipped",
                             gripper_proprio_idx=7, gripper_open_qpos=0.0,
                             gripper_closed_qpos=0.7853982),
}
ROBOT_OBS_PROFILES["DROID"] = dict(ROBOT_OBS_PROFILES["DROID_mounted"])


def get_robot_obs_profile(robot_name):
    return ROBOT_OBS_PROFILES.get(robot_name, ROBOT_OBS_PROFILES["DROID_mounted"])


def wrist_camera_obs_key(robot_name):
    """The obs key extract_from_obs() will read the wrist image from, per the robot's profile."""
    profile = get_robot_obs_profile(robot_name)
    return f"{robot_name}:{profile['wrist_camera_link']}:Camera:{profile['wrist_camera_idx']}"


def assert_wrist_camera(robot):
    """Verify that the configured wrist-camera index resolves to the intended prim."""
    if robot.name not in ROBOT_OBS_PROFILES:
        return None

    profile = get_robot_obs_profile(robot.name)
    want_prim = profile.get("wrist_camera_prim")
    cameras = {k: s for k, s in robot.sensors.items() if ":Camera:" in k}
    if not cameras:
        assert want_prim is None, (
            f"[wrist camera] {robot.name} has NO cameras, but ROBOT_OBS_PROFILES expects "
            f"'{want_prim}'. The include_sensor_names/exclude_sensor_names filter in "
            f"realm/config/robots/*.yaml has matched nothing -- note that the match is a SUBSTRING "
            f"test, so an exclude on a prefix of the wanted name removes the wanted camera as well."
        )
        return None

    key = wrist_camera_obs_key(robot.name)
    sensor = cameras.get(key)
    assert sensor is not None, (
        f"[wrist camera] {robot.name} has no '{key}'; extract_from_obs would silently fall back to "
        f"'{sorted(cameras)[0]}'. Cameras on this robot: "
        f"{ {k: s.prim_path.rsplit('/', 1)[-1] for k, s in sorted(cameras.items())} }. "
        f"OmniGibson numbers a link's cameras by creation order and skips filtered-out ones, so "
        f"ROBOT_OBS_PROFILES['{robot.name}'].wrist_camera_idx and the include_sensor_names filter in "
        f"realm/config/robots/*.yaml have to be changed together."
    )
    if want_prim is not None:
        got_prim = sensor.prim_path.rsplit("/", 1)[-1]
        assert got_prim == want_prim, (
            f"[wrist camera] '{key}' resolves to prim '{got_prim}', but "
            f"ROBOT_OBS_PROFILES['{robot.name}'] says the wrist camera is '{want_prim}'. The index "
            f"is pointing at the wrong physical camera -- the policy would get a view it was never "
            f"trained on, with no error. Full path: {sensor.prim_path}"
        )
    return key


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    # Rendering-disabled runs have no external images.
    if 'external' in obs and 'external_sensor0' in obs['external']:
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
        base_depth = np.zeros((128, 128), dtype=np.float32) if enable_depth else None

    if 'external' in obs and 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    profile = get_robot_obs_profile(robot_name)
    wrist_cam_key = wrist_camera_obs_key(robot_name)
    if robot_name in obs and wrist_cam_key in obs[robot_name]:
        wrist_im = obs[robot_name][wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
    else:
        cam_keys = [k for k in obs.get(robot_name, {}) if ":Camera:" in k]
        if cam_keys:
            print(f"[extract_from_obs] WARNING: no '{wrist_cam_key}' in obs; "
                  f"falling back to '{cam_keys[0]}'. Update ROBOT_OBS_PROFILES for '{robot_name}'.")
            wrist_im = obs[robot_name][cam_keys[0]]['rgb'].cpu().numpy()[..., :3]
        else:
            wrist_im = np.zeros((128, 128, 3), dtype=np.uint8)

    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    # Policies expect gripper state 0=open, 1=closed.
    _open, _closed = profile["gripper_open_qpos"], profile["gripper_closed_qpos"]
    gripper_state = float(np.clip((proprio[profile["gripper_proprio_idx"]] - _open) / (_closed - _open), 0.0, 1.0))

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
