"""Pin the YAM port's YAMLs (and USD, when pxr is importable) against realm/robots/yam.py. Host-side.

WHY THIS EXISTS. The YAM robot is described in FOUR places that OmniGibson and REALM read
independently and that nothing at runtime cross-checks: the RobotDefinition
(realm/robots/definitions/yam/yam.yaml -- joint/link names, default joint pose), the robot configs
(realm/config/robots/YAM*.yaml -- DOF count, gains, camera filter, control frequency), the
observation profile (realm/inference/utils.py ROBOT_OBS_PROFILES -- where the arm ends and which
DOF is the gripper) and the USD itself (link/joint/camera prim names, drive effort limits).
`realm.robots.yam.YamRobot` is the transcription of YAMLab's numbers that all four are supposed to
follow; this test is what makes that true. A gain typed into one YAML but not the other, or a
gripper index that stops matching the definition's joint order, would otherwise only show up as a
policy that quietly receives the wrong state.

The USD checks need `pxr` (pip install usd-core) and are skipped without it; the rest runs anywhere
`uv sync` runs. Nothing here boots Isaac.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from realm.robots.yam import YamRobot as Y  # noqa: E402

DEFINITION = PROJECT_ROOT / "realm" / "robots" / "definitions" / Y.MODEL / f"{Y.MODEL}.yaml"
USD = PROJECT_ROOT / "realm" / "robots" / "yam" / "yam.usd"
CONFIGS = {
    "high_pd": PROJECT_ROOT / "realm" / "config" / "robots" / "YAM.yaml",
    "base": PROJECT_ROOT / "realm" / "config" / "robots" / "YAM_base_pd_control.yaml",
}


def _load(path):
    return yaml.safe_load(path.read_text())


# --- the spec itself ----------------------------------------------------------------------------

def test_spec_is_internally_consistent():
    assert len(Y.ARM_JOINTS) == Y.ARM_DOF
    assert len(Y.ARM_JOINTS) + len(Y.FINGER_JOINTS) == Y.N_DOF == len(Y.DEFAULT_JOINT_POS)
    assert Y.ACTION_DIM == Y.ARM_DOF + 1
    assert Y.ARM_LINKS[0] == Y.BASE_LINK and Y.FLANGE_LINK in Y.ARM_LINKS
    # OmniGibson hides the eef link, so it must be the virtual frame and never a real arm link.
    assert Y.EEF_LINK in Y.VIRTUAL_LINKS and Y.EEF_LINK not in Y.ARM_LINKS
    assert set(Y.FINGERTIP_KEYPOINTS) == set(Y.FINGER_LINKS)
    assert Y.WRIST_CAMERA_LINK in Y.ARM_LINKS
    assert set(Y.EFFORT_LIMITS) == set(Y.ARM_JOINTS) | set(Y.FINGER_JOINTS)
    for gain_set in Y.GAIN_SETS:
        kp, kd = Y.arm_gains(gain_set)
        assert len(kp) == len(kd) == Y.ARM_DOF
    assert Y.DEFAULT_GAIN_SET in Y.GAIN_SETS
    # fingers fully open in the default pose; closed is the upper limit, like the Robotiq finger_joint
    assert Y.GRIPPER_OPEN_QPOS < Y.GRIPPER_CLOSED_QPOS
    assert all(q == Y.GRIPPER_OPEN_QPOS for q in Y.DEFAULT_JOINT_POS[Y.ARM_DOF:])
    assert Y.CONTROL_FREQ_HZ * 4 == Y.PHYSICS_FREQ_HZ
    assert 70.0 < Y.wrist_camera_hfov_deg() < 90.0


# --- RobotDefinition ----------------------------------------------------------------------------

def test_definition_is_discoverable_and_matches_spec():
    assert DEFINITION.is_file(), (
        f"{DEFINITION} is missing; install_robot_definitions.py needs <dir>/<dir>.yaml with equal stems")
    d = _load(DEFINITION)
    assert d["usd_path"] == Y.USD_PATH
    assert Path(d["usd_path"]).name == USD.name and USD.is_file()
    assert d["raw_controller_order"] == ["arm_0", "gripper_0"]
    assert d["default_controllers"] == {"arm_0": "JointController", "gripper_0": "MultiFingerGripperController"}
    assert d["self_collisions"] is False, "YAMLab loads the arm with enabled_self_collisions=False"
    assert d["default_joint_pos"] == list(Y.DEFAULT_JOINT_POS)
    m = d["manipulation"]
    assert m["arm_link_names"]["0"] == list(Y.ARM_LINKS)
    assert m["arm_joint_names"]["0"] == list(Y.ARM_JOINTS)
    assert m["eef_link_names"]["0"] == Y.EEF_LINK
    assert m["finger_link_names"]["0"] == list(Y.FINGER_LINKS)
    assert m["finger_joint_names"]["0"] == list(Y.FINGER_JOINTS)
    assert "assisted_grasp_start_points" not in m, "REALM runs grasping_mode physical"


# --- robot configs ------------------------------------------------------------------------------

@pytest.mark.parametrize("gain_set,path", sorted(CONFIGS.items()), ids=sorted(CONFIGS))
def test_robot_config_matches_spec(gain_set, path):
    cfg = _load(path)
    robots = cfg["robots"]
    assert len(robots) == 1
    r = robots[0]
    assert r["name"] == Y.NAME and r["model"] == Y.MODEL
    assert r["dof"] == Y.N_DOF
    assert r["has_base_column"] is False, "bare arm; test_robot_base_column pins this against the USD name"
    assert r["mount_height"] == Y.MOUNT_HEIGHT
    assert r["spawn_offset"] == Y.spawn_offset()
    assert r["reset_joint_pos"] == list(Y.DEFAULT_JOINT_POS)
    assert r["control_freq"] == Y.CONTROL_FREQ_HZ
    assert r["action_normalize"] is False
    # camera filter must select the one camera prim the USD carries (substring match in OmniGibson)
    assert any(s in Y.WRIST_CAMERA_PRIM for s in r["include_sensor_names"])
    assert "proprio" in r["obs_modalities"] and r["proprio_obs"] == ["joint_qpos"]

    arm = r["controller_config"]["arm_0"]
    assert arm["name"] == "JointController"
    assert arm["motor_type"] == "position"
    assert arm["use_delta_commands"] is False, "OmniGibson's JointController defaults to delta commands"
    assert arm["use_impedances"] is False, "gains go onto the PhysX drives, as YAMLab's ImplicitActuator does"
    assert arm["command_input_limits"] is None and arm["command_output_limits"] is None
    assert arm["control_freq"] == Y.CONTROL_FREQ_HZ
    kp, kd = Y.arm_gains(gain_set)
    assert arm["isaac_kp"] == kp, f"{path.name}: arm isaac_kp is not YAMLab's {gain_set} set"
    assert arm["isaac_kd"] == kd, f"{path.name}: arm isaac_kd is not YAMLab's {gain_set} set"

    grip = r["controller_config"]["gripper_0"]
    assert grip["name"] == "MultiFingerGripperController" and grip["mode"] == "binary"
    assert grip["motor_type"] == "position"
    # stock binary controller: open -> UPPER limit, close -> LOWER; the YAM fingers are the other way round
    assert grip["open_qpos"] == [Y.GRIPPER_OPEN_QPOS] * len(Y.FINGER_JOINTS)
    assert grip["closed_qpos"] == [Y.GRIPPER_CLOSED_QPOS] * len(Y.FINGER_JOINTS)
    assert (grip["isaac_kp"], grip["isaac_kd"]) == Y.gripper_gains(gain_set)

    cam = r["sensor_config"]["VisionSensor"]["sensor_kwargs"]
    assert (cam["image_width"], cam["image_height"]) == Y.RENDER_RESOLUTION
    assert cam["horizontal_aperture"] == Y.WRIST_CAMERA_HORIZONTAL_APERTURE
    assert cam["focal_length"] == Y.wrist_camera_focal_length(cam["horizontal_aperture"])
    assert tuple(cam["clipping_range"]) == Y.WRIST_CAMERA_CLIPPING_RANGE
    assert cam["clipping_range"][0] >= 0.05, "the camera origin is inside the D405 housing mesh; a near plane below ~5 cm renders it"


def test_configs_differ_only_in_gains():
    """The two gain-set configs must stay the same robot: everything but the gains is equal."""
    def strip(cfg):
        r = yaml.safe_load(yaml.safe_dump(cfg))["robots"][0]
        for group in ("arm_0", "gripper_0"):
            r["controller_config"][group].pop("isaac_kp")
            r["controller_config"][group].pop("isaac_kd")
        return r
    a, b = (strip(_load(p)) for p in CONFIGS.values())
    assert a == b


# --- observation profile ------------------------------------------------------------------------

def _inference_utils():
    """realm/inference/utils.py loaded by path: the package __init__ imports omnigibson via client.py,
    the module itself needs only numpy and realm.robots.yam."""
    path = PROJECT_ROOT / "realm" / "inference" / "utils.py"
    spec = importlib.util.spec_from_file_location("_realm_inference_utils_hostside", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_obs_profile_matches_definition():
    utils = _inference_utils()
    ROBOT_OBS_PROFILES, arm_dof, wrist_camera_obs_key = (
        utils.ROBOT_OBS_PROFILES, utils.arm_dof, utils.wrist_camera_obs_key)
    profile = ROBOT_OBS_PROFILES[Y.NAME]
    assert profile == Y.obs_profile()
    assert arm_dof(Y.NAME) == Y.ARM_DOF
    d = _load(DEFINITION)
    # gripper proxy is the first finger joint, right after the arm joints in DOF order
    assert profile["gripper_proprio_idx"] == len(d["manipulation"]["arm_joint_names"]["0"])
    assert d["default_joint_pos"][profile["gripper_proprio_idx"]] == profile["gripper_open_qpos"]
    assert profile["wrist_camera_link"] == Y.WRIST_CAMERA_LINK
    assert wrist_camera_obs_key(Y.NAME) == f"{Y.NAME}:{Y.WRIST_CAMERA_LINK}:Camera:0"
    # DROID profiles are untouched by the arm_dof generalisation
    assert arm_dof("DROID") == arm_dof("DROID_mounted") == 7
    assert "arm_dof" not in ROBOT_OBS_PROFILES["DROID"]


def test_sim_config_and_env_config_recognise_the_robot():
    """sim_config imports omnigibson, so read the source: the YAM branch must exist and run at 30 Hz."""
    src = (PROJECT_ROOT / "realm" / "sim_config.py").read_text()
    m = re.search(r'robot\.startswith\("YAM"\).*?DEFAULT_SIM_STEP_FREQ = (\d+).*?DEFAULT_RENDERING_FREQ = (\d+)',
                  src, re.S)
    assert m, "sim_config.set_sim_config has no YAM branch"
    assert int(m.group(1)) == int(m.group(2)) == Y.CONTROL_FREQ_HZ
    env_cfg = (PROJECT_ROOT / "realm" / "environments" / "env_config.py").read_text()
    assert 'pop("mount_height"' in env_cfg and '"reset_joint_pos"' in env_cfg
    # spawn_offset must be popped (OmniGibson rejects unknown robot kwargs) and applied to the pose that
    # the exterior cameras and EE transforms are composed from, i.e. BEFORE env.robot_pos is set.
    assert 'pop("spawn_offset"' in env_cfg
    assert env_cfg.index('pop("spawn_offset"') < env_cfg.index("env.robot_pos = np.array(robot_pos")


def test_spawn_offset_moves_the_robot_toward_the_workspace_in_its_own_frame():
    """The offset is (forward, left, up) in the robot frame, whatever the scene's yaw: a robot facing +y
    world with the YAM offset lands 0.30 m further along +y and nowhere else."""
    import math

    from realm.geometry import offset_spawn_pose

    d = Y.SPAWN_OFFSET_POS
    assert d[0] > 0, "forward, toward the table"
    assert d[1] == 0.0, "no lateral offset"
    assert d[2] == 0.0 and Y.SPAWN_OFFSET_YAW_DEG == 0.0, "height stays mount_height, no yaw"
    pos, rpy = offset_spawn_pose([1.0, 2.0, 0.5], [0.0, 0.0, math.radians(90)], d, 0.0)
    assert pos == pytest.approx([1.0 - d[1], 2.0 + d[0], 0.5])
    assert rpy == pytest.approx([0.0, 0.0, math.radians(90)])
    # identity offset is the identity map, which is what every DROID config (no key) gets
    pos, rpy = offset_spawn_pose([1.0, 2.0, 0.5], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0], 0.0)
    assert pos == pytest.approx([1.0, 2.0, 0.5]) and rpy == pytest.approx([0.0, 0.0, 0.3])
    for path in sorted((PROJECT_ROOT / "realm" / "config" / "robots").glob("DROID*.yaml")):
        assert "spawn_offset" not in _load(path)["robots"][0], f"{path.name}: DROID must keep the scene pose"


# --- the USD ------------------------------------------------------------------------------------

pxr_available = importlib.util.find_spec("pxr") is not None


@pytest.mark.skipif(not pxr_available, reason="pxr (usd-core) not installed on this host")
def test_usd_has_the_structure_omnigibson_needs():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import build_yam_usd
    problems, summary = build_yam_usd.verify(str(USD))
    assert not problems, "\n".join(problems)
    assert set(summary["joints"]) >= set(Y.ARM_JOINTS) | set(Y.FINGER_JOINTS)
    assert summary["wrist_camera"] == f"/{Y.MODEL}/{Y.WRIST_CAMERA_LINK}/{Y.WRIST_CAMERA_PRIM}"
    assert summary["eef_link"] == Y.EEF_LINK and summary["tcp_in_flange_frame_m"] is not None


def test_provenance_records_the_source():
    prov = (USD.parent / "PROVENANCE").read_text()
    assert "yamlab" in prov and "source sha256" in prov and "output sha256" in prov


# =================================================================================================
# YAM bimanual workstation: realm/robots/yam.py::YamBimanualRobot pins the definition, the config, the
# obs profile, env_config's exterior-camera hook and (with pxr) the composed USD.
# =================================================================================================

from realm.robots.yam import YamBimanualRobot as B  # noqa: E402

B_DEFINITION = PROJECT_ROOT / "realm" / "robots" / "definitions" / B.MODEL / f"{B.MODEL}.yaml"
B_USD = PROJECT_ROOT / "realm" / "robots" / "yam" / "yam_bimanual.usd"
B_CONFIG = PROJECT_ROOT / "realm" / "config" / "robots" / f"{B.NAME}.yaml"


def test_bimanual_spec_is_internally_consistent():
    assert B.ARMS == ("left", "right"), "YAMLab's action layout is left arm first"
    assert B.N_DOF == 2 * Y.N_DOF == len(B.dof_order()) == len(B.default_joint_pos())
    assert B.ACTION_DIM == 14 and B.GRIPPER_ACTION_IDX == (6, 13), "YamActionLayout: LEFT_GRIPPER=6, RIGHT_GRIPPER=13"
    assert B.raw_controller_order() == ("arm_left", "gripper_left", "arm_right", "gripper_right")
    # per-arm names are the single-arm names with the arm prefix, so link_name stays a one-liner
    for arm in B.ARMS:
        assert B.arm_joints(arm) == tuple(f"{arm}_{j}" for j in Y.ARM_JOINTS)
        assert B.finger_joints(arm) == tuple(f"{arm}_{j}" for j in Y.FINGER_JOINTS)
        assert B.eef_link(arm) == f"{arm}_{Y.EEF_LINK}" and B.flange_link(arm) == f"{arm}_{Y.FLANGE_LINK}"
        assert B.gripper_proxy_joint(arm) == f"{arm}_{Y.FINGER_JOINTS[0]}", "YAMLab reads the driven left_finger"
    # YAMLab configs/robot/yam.yaml: arms 0.61 m apart in y, identity orientation, midpoint = mount frame
    assert B.ARM_OFFSETS["left"][1] == -B.ARM_OFFSETS["right"][1] == 0.305
    assert B.ARM_OFFSETS["left"][0] == B.ARM_OFFSETS["left"][2] == 0.0
    # right wrist camera has its own calibration, ~1 mm off the left one
    assert B.WRIST_CAMERA_POSITIONS["left"] == Y.WRIST_CAMERA_POSITION
    assert B.WRIST_CAMERA_POSITIONS["right"] == (0.0, 0.069638, 0.072)
    # every within-arm pair is filtered, and NO cross-arm pair (the arms must collide with each other)
    pairs = B.disabled_collision_pairs()
    assert all(a.split("_", 1)[0] == b.split("_", 1)[0] for a, b in pairs)
    n = len(B.collision_links("left"))
    assert len(pairs) == 2 * n * (n - 1) // 2


def test_bimanual_top_camera_is_yamlab_top_camera():
    """cameras.top: position (0.0860, -0.0090, 1.7043) minus the arm-base midpoint (0.2525, 0, 0.76);
    quaternion_opengl (w,x,y,z) reordered to xyzw. The USD camera looks down -Z, so the view direction in
    the mount frame must come out forward (+x) and 60 degrees down, up-vector forward-up."""
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    pos = np.array(B.EXTERIOR_CAMERA_POSITION)
    assert np.allclose(pos, np.array([0.08600512, -0.009, 1.70432055]) - np.array([0.2525, 0.0, 0.76]), atol=1e-6)
    q = R.from_quat(B.EXTERIOR_CAMERA_QUAT_XYZW)
    view = q.apply([0.0, 0.0, -1.0])
    up = q.apply([0.0, 1.0, 0.0])
    assert np.allclose(view, [0.5, 0.0, -np.sqrt(3) / 2], atol=1e-4), view
    assert np.allclose(up, [np.sqrt(3) / 2, 0.0, 0.5], atol=1e-4), up
    # the camera sits behind and above the arms and looks at the table between them
    assert pos[0] < 0 and pos[2] > 0.9
    hfov = np.degrees(2 * np.arctan(B.EXTERIOR_CAMERA_CALIB_RESOLUTION[0] / (2 * B.EXTERIOR_CAMERA_INTRINSICS["fx"])))
    assert 78.0 < hfov < 79.0
    cam = B.exterior_camera()
    assert set(cam) == {"cam1", "focal_length"} and set(cam["cam1"]) == {"pos", "rot"}
    assert cam["focal_length"] == B.exterior_camera_focal_length()
    assert cam["focal_length"] == round(B.EXTERIOR_CAMERA_HORIZONTAL_APERTURE * B.EXTERIOR_CAMERA_INTRINSICS["fx"] / 640, 4)


def test_bimanual_definition_matches_spec():
    assert B_DEFINITION.is_file()
    d = _load(B_DEFINITION)
    assert d["usd_path"] == B.USD_PATH and Path(d["usd_path"]).name == B_USD.name and B_USD.is_file()
    assert d["raw_controller_order"] == list(B.raw_controller_order())
    assert d["default_controllers"] == {"arm_left": "JointController", "gripper_left": "MultiFingerGripperController",
                                        "arm_right": "JointController", "gripper_right": "MultiFingerGripperController"}
    assert d["self_collisions"] is True, "the two arms must collide with each other (YAMLab: separate articulations)"
    assert sorted(map(sorted, d["disabled_collision_pairs"])) == sorted(map(sorted, B.disabled_collision_pairs()))
    assert d["default_joint_pos"] == list(B.default_joint_pos())
    m = d["manipulation"]
    assert m["n_arms"] == 2 and m["arm_names"] == list(B.ARMS)
    for arm in B.ARMS:
        assert m["arm_link_names"][arm] == list(B.arm_links(arm))
        assert m["arm_joint_names"][arm] == list(B.arm_joints(arm))
        assert m["eef_link_names"][arm] == B.eef_link(arm)
        assert m["finger_link_names"][arm] == list(B.finger_links(arm))
        assert m["finger_joint_names"][arm] == list(B.finger_joints(arm))
    assert "assisted_grasp_start_points" not in m


def test_bimanual_config_matches_spec():
    cfg = _load(B_CONFIG)
    assert len(cfg["robots"]) == 1
    r = cfg["robots"][0]
    assert r["name"] == B.NAME and r["model"] == B.MODEL
    assert r["dof"] == B.N_DOF
    assert r["has_base_column"] is False and r["mount_height"] == B.MOUNT_HEIGHT
    assert r["spawn_offset"] == B.spawn_offset() == Y.spawn_offset(), "both YAM robots see the same scene"
    assert r["reset_joint_pos"] == list(B.default_joint_pos())
    assert r["control_freq"] == B.CONTROL_FREQ_HZ and r["action_normalize"] is False
    assert any(s in Y.WRIST_CAMERA_PRIM for s in r["include_sensor_names"]), "must match BOTH <arm>_link_6/wrist_camera prims"
    assert r["proprio_obs"] == ["joint_qpos"]
    # controller order is the definition's, and each arm is the single-arm YAM.yaml controller verbatim
    assert list(r["controller_config"]) == list(B.raw_controller_order())
    single = _load(CONFIGS["high_pd"])["robots"][0]["controller_config"]
    for arm in B.ARMS:
        assert r["controller_config"][f"arm_{arm}"] == single["arm_0"], f"arm_{arm} differs from YAM.yaml arm_0"
        assert r["controller_config"][f"gripper_{arm}"] == single["gripper_0"], f"gripper_{arm} differs from YAM.yaml gripper_0"
    assert r["sensor_config"] == _load(CONFIGS["high_pd"])["robots"][0]["sensor_config"]
    assert r["exterior_camera"] == B.exterior_camera()


def test_bimanual_obs_profile_is_registered_and_consistent():
    utils = _inference_utils()
    profile = utils.ROBOT_OBS_PROFILES[B.NAME]
    assert profile == B.obs_profile()
    assert utils.is_multi_arm(B.NAME) and not utils.is_multi_arm(Y.NAME) and not utils.is_multi_arm("DROID")
    assert utils.arm_names(B.NAME) == B.ARMS and utils.arm_names("DROID") == (None,)
    assert utils.arm_dof(B.NAME) == Y.ARM_DOF and utils.n_arm_joints(B.NAME) == 12
    assert utils.n_arm_joints("DROID") == 7 and utils.n_arm_joints(Y.NAME) == 6
    assert utils.gripper_action_idx(B.NAME) == B.GRIPPER_ACTION_IDX
    assert utils.gripper_action_idx("DROID") == utils.gripper_action_idx(Y.NAME) == (-1,)
    # every index is looked up by name in dof_order, which is the definition's default_joint_pos order
    d = _load(B_DEFINITION)
    order = list(profile["dof_order"])
    assert len(order) == len(d["default_joint_pos"])
    for arm in B.ARMS:
        assert utils.finger_proprio_indices(B.NAME, arm) == [order.index(j) for j in d["manipulation"]["finger_joint_names"][arm]]
        for j in d["manipulation"]["arm_joint_names"][arm]:
            assert j in order
    assert utils.finger_proprio_indices("DROID") == [7, 8] and utils.finger_proprio_indices(Y.NAME) == [6, 7]
    # the wrist keys name each arm's flange camera; the default (no arm) key is the left one
    assert utils.wrist_camera_obs_keys(B.NAME) == [f"{B.NAME}:{B.flange_link(a)}:Camera:0" for a in B.ARMS]
    assert utils.wrist_camera_obs_key(B.NAME) == utils.wrist_camera_obs_key(B.NAME, "left")
    assert utils.wrist_camera_obs_keys("DROID") == [utils.wrist_camera_obs_key("DROID")]
    # legacy single-arm keys describe the first arm
    assert profile["wrist_camera_link"] == B.flange_link("left")
    assert order[profile["gripper_proprio_idx"]] == B.gripper_proxy_joint("left")


def test_bimanual_extract_from_obs_layout():
    """A synthetic observation in dof_order: the policy state is [left(6), right(6)], gripper state one
    normalised value per arm, wrist_im the left camera and extract_wrist_images [left, right]."""
    import numpy as np

    class _T:  # stands in for a torch tensor: extract_from_obs only calls .cpu().numpy()
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    utils = _inference_utils()
    order = list(B.dof_order())
    proprio = np.arange(len(order), dtype=np.float32) * 0.01
    for arm, g in (("left", B.GRIPPER_OPEN_QPOS), ("right", B.GRIPPER_CLOSED_QPOS)):
        proprio[order.index(B.gripper_proxy_joint(arm))] = g
    left_im = np.full((4, 4, 4), 10, dtype=np.uint8)
    right_im = np.full((4, 4, 4), 20, dtype=np.uint8)
    obs = {B.NAME: {"proprio": _T(proprio),
                    utils.wrist_camera_obs_key(B.NAME, "left"): {"rgb": _T(left_im)},
                    utils.wrist_camera_obs_key(B.NAME, "right"): {"rgb": _T(right_im)}}}
    base_im, _, second, _, wrist_im, state, grip = utils.extract_from_obs(obs, robot_name=B.NAME)
    assert base_im.shape == (128, 128, 3) and second is None
    assert wrist_im.shape == (4, 4, 3) and (wrist_im == 10).all()
    expected_state = np.concatenate([proprio[[order.index(j) for j in B.arm_joints(a)]] for a in B.ARMS])
    assert state.shape == (12,) and np.array_equal(state, expected_state)
    assert grip.shape == (2,) and grip.tolist() == [0.0, 1.0]
    ims = utils.extract_wrist_images(obs, B.NAME)
    assert len(ims) == 2 and (ims[0] == 10).all() and (ims[1] == 20).all()


def test_bimanual_env_config_places_the_top_camera():
    """env_config must pop the REALM-only exterior_camera key (OmniGibson would reject it as a robot kwarg)
    and use it for external_sensor0's pose and focal length."""
    env_cfg = (PROJECT_ROOT / "realm" / "environments" / "env_config.py").read_text()
    assert 'pop("exterior_camera"' in env_cfg
    assert 'robot_cam["cam1"]' in env_cfg and 'robot_cam["focal_length"]' in env_cfg
    # the mixed-rotation composition adds base_height, so the config's pose must be in the mount frame
    assert B.MOUNT_HEIGHT == Y.MOUNT_HEIGHT


@pytest.mark.skipif(not pxr_available, reason="pxr (usd-core) not installed on this host")
def test_bimanual_usd_has_the_structure_omnigibson_needs():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import build_yam_bimanual_usd
    problems, summary = build_yam_bimanual_usd.verify(str(B_USD))
    assert not problems, "\n".join(problems)


def test_bimanual_provenance_records_the_source():
    prov = (B_USD.parent / "PROVENANCE").read_text()
    assert "yam_bimanual.usd" in prov and "build_yam_bimanual_usd.py" in prov


# --- the YAMLab / LeRobot contract (realm/inference/yamlab.py, host-importable) --------------------

def test_yamlab_contract_round_trip():
    import numpy as np
    import importlib.util
    spec = importlib.util.spec_from_file_location("_realm_inference_yamlab", PROJECT_ROOT / "realm" / "inference" / "yamlab.py")
    yl = importlib.util.module_from_spec(spec); spec.loader.exec_module(yl)
    assert yl.STATE_DIM == 14 and yl.FINGER_IDX == (6, 13)
    assert yl.IMAGE_KEYS == ("observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"), \
        "YAMLab configs/robot/yam.yaml lerobot_key: top_rgb / left_rgb / right_rgb"
    robot_state = np.arange(12) * 0.1          # [left(6), right(6)]
    gripper = np.array([0.0, 1.0])             # left open, right closed (REALM normalised)
    st = yl.yamlab_state(robot_state, gripper)
    assert st.dtype == np.float32 and st.shape == (14,)
    # LeRobot layout: left joints, left finger (metres), right joints, right finger
    assert np.allclose(st[0:6], robot_state[:6]) and np.allclose(st[7:13], robot_state[6:])
    assert st[6] == B.GRIPPER_OPEN_QPOS and st[13] == B.GRIPPER_CLOSED_QPOS
    obs = yl.yamlab_observation("put it", np.zeros((720, 1280, 3), np.uint8), np.zeros((4, 4, 3), np.uint8),
                                np.ones((4, 4, 3), np.uint8), robot_state, gripper)
    assert set(obs) == {"prompt", yl.STATE_KEY, *yl.IMAGE_KEYS}
    assert obs["observation.images.top_rgb"].shape == (720, 1280, 3) and obs["observation.images.right_rgb"].max() == 1
    # actions: finger targets in metres -> open fraction, arm columns untouched; 1-D input becomes (1, 14)
    act = np.zeros(14); act[6] = B.GRIPPER_OPEN_QPOS; act[13] = B.GRIPPER_CLOSED_QPOS; act[0] = 0.7
    out = yl.yamlab_actions_to_realm(act)
    assert out.shape == (1, 14) and out[0, 0] == 0.7 and out[0, 6] == 1.0 and out[0, 13] == 0.0
    half = yl.open_fraction_from_finger_qpos(-0.0475 / 2)
    assert abs(half - 0.5) < 1e-9
    # the sweep server and the adapter agree on the registered model_type and gripper convention
    shared = (PROJECT_ROOT / "realm" / "config" / "shared.py").read_text()
    assert '"yamlab"' in shared.split("GRIPPER_OPEN_ABOVE_HALF")[1].split("\n")[0]
    client = (PROJECT_ROOT / "realm" / "inference" / "client.py").read_text()
    assert '"yamlab": _YamLabAdapter' in client


def test_link_com_guard_runs_after_rebase_and_usd_build_flattens_collisions():
    """OmniGibson's loader overwrites each link's CoM from its collision meshes composed only one level up
    (metres off on the YAM export, Slurm 204612). Two defences, both pinned here: the build script puts every
    collision Mesh directly under its link and verify() rejects nesting; the runtime restore runs in
    finalize_setup, AFTER rebase_initial_file, because an earlier call is undone (Slurm 204613/204615)."""
    build = (PROJECT_ROOT / "scripts" / "build_yam_usd.py").read_text()
    assert "def flatten_collision_xforms" in build
    assert "is not a direct child of its link" in build, "verify() must reject nested collision prims"
    assert 'assert n_flat > 0' in build
    env_dyn = (PROJECT_ROOT / "realm" / "environments" / "env_dynamic.py").read_text()
    finalize = env_dyn.split("def finalize_setup(self):")[1].split("\n    def ")[0]
    assert "self.restore_authored_link_coms()" in finalize
    bind = env_dyn.split("def bind_scene_handles(self):")[1].split("\n    def ")[0]
    assert "restore_authored_link_coms" not in bind, "a restore before rebase_initial_file is undone"
    setup = (PROJECT_ROOT / "realm" / "environments" / "scene_setup.py").read_text()
    assert 'attr.HasAuthoredValue()' in setup, "only links that author physics:centerOfMass may be touched (DROID authors none)"
    prov = (PROJECT_ROOT / "realm" / "robots" / "yam" / "PROVENANCE").read_text()
    assert "collision Mesh moved up under its link" in prov


def test_yam_mount_height_is_the_bare_droid_offset():
    """Martin (2026-09-05): the YAM arms sit at the same offset env_config adds for the bare (non-base) DROID,
    DROID_BASE_HEIGHT, so the exterior cameras and the workspace match the DROID setup. Pinned to the constant."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_realm_shared", PROJECT_ROOT / "realm" / "config" / "shared.py")
    shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
    assert Y.MOUNT_HEIGHT == B.MOUNT_HEIGHT == shared.DROID_BASE_HEIGHT
    for path in (*CONFIGS.values(), B_CONFIG):
        assert _load(path)["robots"][0]["mount_height"] == shared.DROID_BASE_HEIGHT, path
