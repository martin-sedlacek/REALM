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
