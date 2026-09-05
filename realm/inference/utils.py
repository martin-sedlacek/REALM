import numpy as np

from realm.robots.yam import YamBimanualRobot, YamRobot

# Camera indices follow sensor creation order and must match each robot config's sensor filter.
# `arm_dof` (default DEFAULT_ARM_DOF when absent) is how many leading proprio entries are arm joints:
# the policy's joint state is proprio[:arm_dof] and REALM's finger predicates read
# proprio[arm_dof:arm_dof + 2]. `gripper_proprio_idx` is the proxy joint for the gripper state.
#
# Multi-arm profiles carry `arms` (OmniGibson arm names in action order) plus per-arm tables
# (`arm_joint_names`, `finger_joint_names`, `gripper_proxy_joints`, `wrist_cameras`) and `dof_order`,
# the articulation DOF order the joints are looked up in. The policy state is then the concatenation
# of the arms' joint positions in `arms` order, the gripper state one normalised value per arm, and
# `gripper_action_idx` says which action columns carry a gripper command. The single-arm keys describe
# the FIRST arm so every single-wrist / single-gripper caller keeps working on the default arm.
DEFAULT_ARM_DOF = 7
ROBOT_OBS_PROFILES = {
    # Robotiq finger_joint: 0 rad is open; 0.7854 rad is closed.
    "DROID_mounted": dict(wrist_camera_link="base_link", wrist_camera_idx=0,
                             wrist_camera_prim="wrist_camera_flipped",
                             gripper_proprio_idx=7, gripper_open_qpos=0.0,
                             gripper_closed_qpos=0.7853982),
    # YAM (YAMLab port): 6 arm joints, prismatic left_finger -0.0475 m open / 0.0 closed.
    YamRobot.NAME: YamRobot.obs_profile(),
    # YAM bimanual workstation: two of the above as one articulation, YAMLab's 14-D action layout.
    YamBimanualRobot.NAME: YamBimanualRobot.obs_profile(),
}
ROBOT_OBS_PROFILES["DROID"] = dict(ROBOT_OBS_PROFILES["DROID_mounted"])


def get_robot_obs_profile(robot_name):
    return ROBOT_OBS_PROFILES.get(robot_name, ROBOT_OBS_PROFILES["DROID_mounted"])


def is_multi_arm(robot_name):
    """True when the robot's profile drives more than one arm (`arms` present)."""
    return "arms" in get_robot_obs_profile(robot_name)


def arm_names(robot_name):
    """The profile's arm names in action order; a single-arm profile has one unnamed arm (None)."""
    return tuple(get_robot_obs_profile(robot_name).get("arms", (None,)))


def arm_dof(robot_name):
    """Number of arm joints at the front of the robot's proprio vector (7 for every DROID profile).

    For a multi-arm profile this is the per-arm count; the policy state has `n_arm_joints` entries.
    """
    return get_robot_obs_profile(robot_name).get("arm_dof", DEFAULT_ARM_DOF)


def n_arm_joints(robot_name):
    """Width of the policy's joint state: arm_dof summed over the arms (7 DROID, 6 YAM, 12 bimanual)."""
    return arm_dof(robot_name) * len(arm_names(robot_name))


def gripper_action_idx(robot_name):
    """Action columns carrying a gripper command, to be binarised by realm.rollout. Single-arm robots
    keep the gripper as the LAST entry, which is what every existing caller assumed."""
    return tuple(get_robot_obs_profile(robot_name).get("gripper_action_idx", (-1,)))


def _arm_dof_indices(profile, arm):
    """Proprio indices of `arm`'s joints, looked up by name in the profile's `dof_order`."""
    order = list(profile["dof_order"])
    return [order.index(j) for j in profile["arm_joint_names"][arm]]


def _gripper_proprio_idx(profile, arm):
    return list(profile["dof_order"]).index(profile["gripper_proxy_joints"][arm])


def finger_proprio_indices(robot_name, arm=None):
    """Proprio indices of the two finger DOFs of `arm` (default arm when None).

    Single-arm profiles: the two DOFs right after the arm joints (7-8 DROID, 6-7 YAM), unchanged.
    Multi-arm: each arm's finger joints looked up by name in `dof_order`.
    """
    profile = get_robot_obs_profile(robot_name)
    if "arms" not in profile:
        n = profile.get("arm_dof", DEFAULT_ARM_DOF)
        return [n, n + 1]
    arm = profile["arms"][0] if arm is None else arm
    order = list(profile["dof_order"])
    return [order.index(j) for j in profile["finger_joint_names"][arm]]


def assert_proprio_layout(robot):
    """Fail loudly if the profile's proprio indices do not line up with the robot's DOF order.

    `proprio_obs: ["joint_qpos"]` is OmniGibson's joint positions in articulation DOF order, and
    the profile hard-codes where the arm ends and which entry is the gripper proxy. A wrong index
    would hand the policy a finger position as an arm joint (or vice versa) without any error, so
    this checks the two facts the indices rest on: the arm joints are the first `arm_dof` DOFs, and
    the gripper proxy DOF is one of the definition's finger joints.

    Multi-arm profiles pin the WHOLE DOF order instead (the profile looks every index up by name in
    `dof_order`, so that list has to be exactly what PhysX built), plus each arm's joint and finger
    names against the loaded definition.
    """
    if robot.name not in ROBOT_OBS_PROFILES:
        return
    profile = get_robot_obs_profile(robot.name)
    dof_names = list(robot.dof_names_ordered)
    if "arms" in profile:
        assert dof_names == list(profile["dof_order"]), (
            f"[proprio layout] {robot.name}: ROBOT_OBS_PROFILES.dof_order is {list(profile['dof_order'])}, "
            f"but the built articulation reports {dof_names}. Every arm/gripper index is looked up in "
            f"dof_order, so fix YamBimanualRobot.dof_order() (and the definition's default_joint_pos / the "
            f"config's reset_joint_pos, which are in the same order) to match PhysX."
        )
        assert list(robot.arm_names) == list(profile["arms"]), (
            f"[proprio layout] {robot.name}: profile arms {list(profile['arms'])} != definition arm_names "
            f"{list(robot.arm_names)}")
        for arm in profile["arms"]:
            assert list(robot.arm_joint_names[arm]) == list(profile["arm_joint_names"][arm]), (
                f"[proprio layout] {robot.name}/{arm}: arm joints {list(robot.arm_joint_names[arm])} != "
                f"profile {list(profile['arm_joint_names'][arm])}")
            assert set(robot.finger_joint_names[arm]) == set(profile["finger_joint_names"][arm]), (
                f"[proprio layout] {robot.name}/{arm}: finger joints {list(robot.finger_joint_names[arm])} != "
                f"profile {list(profile['finger_joint_names'][arm])}")
            assert profile["gripper_proxy_joints"][arm] in robot.finger_joint_names[arm]
        return
    n_arm = profile.get("arm_dof", DEFAULT_ARM_DOF)
    arm_joints = list(robot.arm_joint_names[robot.default_arm])
    assert dof_names[:n_arm] == arm_joints, (
        f"[proprio layout] {robot.name}: ROBOT_OBS_PROFILES says the first {n_arm} DOFs are the arm, "
        f"but the DOF order is {dof_names} and the arm joints are {arm_joints}. extract_from_obs would "
        f"feed the policy the wrong joints; fix arm_dof / the definition's arm_joint_names together."
    )
    g = profile["gripper_proprio_idx"]
    finger_joints = set(robot.finger_joint_names[robot.default_arm])
    assert g < len(dof_names) and dof_names[g] in finger_joints, (
        f"[proprio layout] {robot.name}: gripper_proprio_idx={g} is DOF "
        f"'{dof_names[g] if g < len(dof_names) else None}', not one of the finger joints {sorted(finger_joints)}. "
        f"DOF order: {dof_names}."
    )


def wrist_camera_obs_key(robot_name, arm=None):
    """Observation key of the wrist camera: the default arm's when `arm` is None."""
    profile = get_robot_obs_profile(robot_name)
    if arm is not None and "wrist_cameras" in profile:
        cam = profile["wrist_cameras"][arm]
        return f"{robot_name}:{cam['link']}:Camera:{cam['idx']}"
    return f"{robot_name}:{profile['wrist_camera_link']}:Camera:{profile['wrist_camera_idx']}"


def wrist_camera_obs_keys(robot_name):
    """Every wrist camera key, in arm order (one entry for single-arm robots)."""
    return [wrist_camera_obs_key(robot_name, arm) for arm in arm_names(robot_name)]


def _wrist_camera_prim(profile, arm):
    if arm is not None and "wrist_cameras" in profile:
        return profile["wrist_cameras"][arm].get("prim")
    return profile.get("wrist_camera_prim")


def assert_wrist_camera(robot):
    """Check every wrist camera the profile names resolves to the intended prim; returns the default
    arm's key (None when the robot has no cameras)."""
    if robot.name not in ROBOT_OBS_PROFILES:
        return None

    profile = get_robot_obs_profile(robot.name)
    cameras = {k: s for k, s in robot.sensors.items() if ":Camera:" in k}
    if not cameras:
        want = [_wrist_camera_prim(profile, arm) for arm in arm_names(robot.name)]
        assert all(p is None for p in want), (
            f"[wrist camera] {robot.name} has NO cameras, but ROBOT_OBS_PROFILES expects "
            f"{want}. The include_sensor_names/exclude_sensor_names filter in "
            f"realm/config/robots/*.yaml has matched nothing -- note that the match is a SUBSTRING "
            f"test, so an exclude on a prefix of the wanted name removes the wanted camera as well."
        )
        return None

    keys = []
    for arm in arm_names(robot.name):
        key = wrist_camera_obs_key(robot.name, arm)
        sensor = cameras.get(key)
        assert sensor is not None, (
            f"[wrist camera] {robot.name} has no '{key}'; extract_from_obs would silently fall back to "
            f"'{sorted(cameras)[0]}'. Cameras on this robot: "
            f"{ {k: s.prim_path.rsplit('/', 1)[-1] for k, s in sorted(cameras.items())} }. "
            f"OmniGibson numbers a link's cameras by creation order and skips filtered-out ones, so "
            f"ROBOT_OBS_PROFILES['{robot.name}'].wrist_camera_idx and the include_sensor_names filter in "
            f"realm/config/robots/*.yaml have to be changed together."
        )
        want_prim = _wrist_camera_prim(profile, arm)
        if want_prim is not None:
            got_prim = sensor.prim_path.rsplit("/", 1)[-1]
            assert got_prim == want_prim, (
                f"[wrist camera] '{key}' resolves to prim '{got_prim}', but "
                f"ROBOT_OBS_PROFILES['{robot.name}'] says the wrist camera is '{want_prim}'. The index "
                f"is pointing at the wrong physical camera -- the policy would get a view it was never "
                f"trained on, with no error. Full path: {sensor.prim_path}"
            )
        keys.append(key)
    return keys[0]


def _wrist_image(obs, robot_name, wrist_cam_key):
    if robot_name in obs and wrist_cam_key in obs[robot_name]:
        return obs[robot_name][wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
    cam_keys = [k for k in obs.get(robot_name, {}) if ":Camera:" in k]
    if cam_keys:
        print(f"[extract_from_obs] WARNING: no '{wrist_cam_key}' in obs; "
              f"falling back to '{cam_keys[0]}'. Update ROBOT_OBS_PROFILES for '{robot_name}'.")
        return obs[robot_name][cam_keys[0]]['rgb'].cpu().numpy()[..., :3]
    return np.zeros((128, 128, 3), dtype=np.uint8)


def extract_wrist_images(obs, robot_name='DROID'):
    """Every wrist image in arm order -- [left, right] for the bimanual YAM, one entry otherwise."""
    return [_wrist_image(obs, robot_name, key) for key in wrist_camera_obs_keys(robot_name)]


def _normalised_gripper(proprio, idx, profile):
    # Policies expect gripper state 0=open, 1=closed.
    _open, _closed = profile["gripper_open_qpos"], profile["gripper_closed_qpos"]
    return float(np.clip((proprio[idx] - _open) / (_closed - _open), 0.0, 1.0))


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    """Split an OmniGibson observation into what the policy and the recorder consume.

    Returns (base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state,
    gripper_state). `wrist_im` is the DEFAULT arm's camera; the other arms' images come from
    `extract_wrist_images`. For a multi-arm profile `robot_state` is the arms' joint positions
    concatenated in arm order and `gripper_state` a 1-D array with one normalised value per arm;
    single-arm robots get the plain proprio[:arm_dof] slice and a float, as before.
    """
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
    wrist_im = _wrist_image(obs, robot_name, wrist_camera_obs_key(robot_name))

    proprio = obs[robot_name]['proprio'].cpu().numpy()
    if "arms" in profile:
        robot_state = np.concatenate([proprio[_arm_dof_indices(profile, arm)] for arm in profile["arms"]])
        gripper_state = np.array([_normalised_gripper(proprio, _gripper_proprio_idx(profile, arm), profile)
                                  for arm in profile["arms"]])
    else:
        robot_state = proprio[:profile.get("arm_dof", DEFAULT_ARM_DOF)]
        gripper_state = _normalised_gripper(proprio, profile["gripper_proprio_idx"], profile)

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
