import numpy as np

# Per-robot observation conventions. Keyed by the robot's name in the scene (robot.name, which comes
# from `name:` in realm/config/robots/*.yaml). Anything not listed falls back to "DROID".
#
#   wrist_camera_link  -- the link the wrist camera hangs off; the obs key is
#                         "<robot>:<link>:Camera:<idx>". droid.usd mounts it on `gripper_link_camera`;
#                         the robolab asset mounts two on the gripper's `base_link`.
#   wrist_camera_idx   -- which camera on that link. This is a CREATION-ORDER index, not a property
#                         of the asset: Robot._load_sensors numbers a link's cameras 0,1,2... in prim
#                         order and only counts the ones it actually instantiates, so filtering a
#                         camera out RENUMBERS the survivors. It therefore has to be kept in step
#                         with the include_sensor_names filter in realm/config/robots/<robot>.yaml.
#   wrist_camera_prim  -- the USD prim the index above must land on, or None if unknown. This is what
#                         makes the coupling above checkable: assert_wrist_camera() resolves the index
#                         against the live robot and fails the build if it points at a different
#                         prim. Without it a stale index degrades silently -- extract_from_obs's
#                         fallback below would hand the policy whichever camera happens to be first.
#   gripper_open_qpos / gripper_closed_qpos
#                      -- value of proprio[gripper_proprio_idx] at each extreme, used to normalise
#                         gripper_state to the DROID convention (0 = open, 1 = closed). The robolab
#                         gripper is a single revolute finger_joint following the standard Robotiq
#                         2F-85 convention: 0 rad = fully OPEN, 0.7854 rad = CLOSED.
ROBOT_OBS_PROFILES = {
    # Both droid.usd and droid_mounted.usd hold exactly ONE camera prim, /panda/gripper_link_camera/
    # Camera, so there is nothing here to renumber -- but the prim name is recorded anyway so the
    # guard covers the stock asset too.
    "DROID": dict(wrist_camera_link="gripper_link_camera", wrist_camera_idx=0,
                  wrist_camera_prim="Camera",
                  gripper_proprio_idx=7, gripper_open_qpos=0.0, gripper_closed_qpos=0.05),
    # `wrist_camera_flipped` is the camera whose framing matches the stock DROID wrist view (fingers
    # entering symmetrically from the bottom); the asset's other camera, `wrist_camera`, is rotated
    # 180 deg in yaw and looks at the floor and the wall behind the table -- a view unlike anything in
    # the policy's training distribution.
    #
    # It was `Camera:1` until 2026-08-13, when realm/config/robots/DROID_robolab.yaml started
    # filtering the dead camera out with include_sensor_names to save a render product per env.
    # OmniGibson does not count filtered-out cameras, so the survivor is now `Camera:0`. Measured on
    # a live build, not inferred: prim path /..._DROID_robolab/base_link/wrist_camera_flipped, at
    # rel_pos [-0.0740, 0.0310, 0.0292] / rel_rpy [160.5, 0.2, -91.4] deg in the panda_link8 frame,
    # which is the same pose the camera had when it was Camera:1.
    # THE INDEX AND THE YAML FILTER MUST CHANGE TOGETHER -- see wrist_camera_idx above.
    #
    # finger_joint follows the standard Robotiq 2F-85 convention: 0 rad = OPEN, 0.7854 rad = CLOSED.
    # Measured 2026-08-11 (job 189066) from the separation of the two inner_finger links once
    # The shipped RoboLab asset has its finger origins on the pad centroids:
    #     0.0 rad -> 116.2 mm apart (open)     0.7854 rad -> 33.0 mm apart (shut)
    # These were the other way round until 2026-08-11, which fed pi0.5 an INVERTED gripper_state --
    # it is closed-loop on that signal, so it was told "closed" whenever the hand was open.
    # Do not re-derive this from knuckle or link-origin separation: the four-bar linkage swings the
    # knuckles apart as the pads close, so any such measurement reports the exact opposite.
    "DROID_robolab": dict(wrist_camera_link="base_link", wrist_camera_idx=0,
                          wrist_camera_prim="wrist_camera_flipped",
                          gripper_proprio_idx=7, gripper_open_qpos=0.0, gripper_closed_qpos=0.7853982),
}
# v2 of the robolab asset differs only in geometry, so it shares v1's observation conventions.
# Without an entry here it would silently fall back to the stock DROID profile: a wrist camera key
# that does not exist (black image) and the prismatic gripper normalisation (inverted state).
# Sharing the entry means sharing wrist_camera_idx=0, which is only correct while
# realm/config/robots/DROID_robolab_v2.yaml also filters the dead camera out -- it does, and
# assert_wrist_camera() is what stops the pair from drifting apart unnoticed.
ROBOT_OBS_PROFILES["DROID_robolab_v2"] = dict(ROBOT_OBS_PROFILES["DROID_robolab"])


def get_robot_obs_profile(robot_name):
    return ROBOT_OBS_PROFILES.get(robot_name, ROBOT_OBS_PROFILES["DROID"])


def wrist_camera_obs_key(robot_name):
    """The obs key extract_from_obs() will read the wrist image from, per the robot's profile."""
    profile = get_robot_obs_profile(robot_name)
    return f"{robot_name}:{profile['wrist_camera_link']}:Camera:{profile['wrist_camera_idx']}"


def assert_wrist_camera(robot):
    """Fail the build if @robot's wrist observation would not come from the intended camera.

    This is the guard on a failure that is otherwise silent and expensive. `wrist_camera_idx` is a
    creation-order index, so anything that changes which cameras get instantiated -- an
    include_sensor_names/exclude_sensor_names filter in realm/config/robots/<robot>.yaml, a renamed
    link, a re-exported asset -- can renumber the survivors. When that happens, extract_from_obs
    finds no key, prints a warning, and reads "the first camera on the robot" instead. The rollout
    then completes with exit code 0 while the policy is fed a camera pointing somewhere else.

    Called from RealmEnvironmentDynamic.bind_scene_handles(), i.e. once per member before any
    stepping, so a mismatch surfaces at build time rather than as a mysteriously bad success rate.

    Returns the verified obs key, or None when there is nothing to check (see the two early exits).
    """
    if robot.name not in ROBOT_OBS_PROFILES:
        # No profile of its own, so get_robot_obs_profile falls back to the stock DROID one, whose
        # camera link need not exist on this asset at all (UR5, WidowX). That path has always
        # degraded to extract_from_obs's warn-and-fallback; do not promote it to a hard failure here.
        return None

    profile = get_robot_obs_profile(robot.name)
    want_prim = profile.get("wrist_camera_prim")
    cameras = {k: s for k, s in robot.sensors.items() if ":Camera:" in k}
    if not cameras:
        # A profile that names the prim is asserting the asset HAS that camera, so zero cameras means
        # the include/exclude filter matched nothing -- the exact outcome of writing
        # `exclude_sensor_names: ["wrist_camera"]`, which substring-matches "wrist_camera_flipped"
        # too. Fail; otherwise extract_from_obs quietly returns a black 128x128 image.
        assert want_prim is None, (
            f"[wrist camera] {robot.name} has NO cameras, but ROBOT_OBS_PROFILES expects "
            f"'{want_prim}'. The include_sensor_names/exclude_sensor_names filter in "
            f"realm/config/robots/*.yaml has matched nothing -- note that the match is a SUBSTRING "
            f"test, so an exclude on a prefix of the wanted name removes the wanted camera as well."
        )
        # Otherwise: no cameras instantiated at all (e.g. "rgb" dropped from obs_modalities), and
        # extract_from_obs's black image is the documented behaviour for a render-free run.
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
    wrist_cam_key = wrist_camera_obs_key(robot_name)
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
