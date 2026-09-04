"""The YAM arm as REALM knows it: one host-importable spec, ported from YAMLab.

YAMLab (https://github.com/ARISE-Initiative/yamlab, commit ec0455d, 2026-06-28) is an IsaacLab
framework for the bimanual YAM workstation. Its robot facts live in ``configs/robot/yam.yaml``
(gripper limits, camera poses + intrinsics, PD gain sets) and ``robot/yam/yam.py`` (the IsaacLab
``ArticulationCfg``: joint groups, effort limits, initial joint state). REALM has no IsaacLab and
OmniGibson 3.9.1 has no robot classes -- a robot is a RobotDefinition YAML plus a robot config
YAML -- so this module is the single place the YAMLab numbers are transcribed to, and the YAMLs
are pinned against it by ``tests/test_yam_robot.py``. Edit the numbers HERE, then re-run the test to
find every YAML that has to follow.

Imports nothing from omnigibson/torch on purpose: the pin test, ``realm.inference.utils`` and
``scripts/build_yam_usd.py`` all run on the host.

What was ported, and how it maps onto OmniGibson:

* **Actuation.** YAMLab drives every joint with an IsaacLab ``ImplicitActuatorCfg``: a PhysX
  position drive whose stiffness/damping come from one of two gain sets (``base`` or ``high_pd``,
  default ``high_pd``). The OmniGibson equivalent is the stock ``JointController`` in
  ``motor_type: position`` with ``use_impedances: False`` and per-DOF ``isaac_kp``/``isaac_kd``,
  which OmniGibson writes onto the PhysX drives at controller load. Effort limits are the drive
  ``maxForce`` values authored in the USD (28 N m shoulder, 10 N m elbow/wrist, 100 N fingers).
* **Action layout.** YAMLab's 14-D bimanual action is ``[left_arm(6), left_gripper(1),
  right_arm(6), right_gripper(1)]`` of absolute joint targets; REALM is single-arm, so the YAM
  action is ``[arm(6), gripper(1)]``, the same ``[arm, gripper]`` shape as DROID with 6 instead of 7
  arm joints. The gripper command drives BOTH finger joints (YAMLab drives ``left_finger`` and
  mirrors ``right_finger`` in ``step()``; OmniGibson's binary gripper controller sends both to the
  same limit, which is the same thing).
* **Gripper convention.** Finger joints are prismatic, ``-0.0475`` m fully open, ``0.0`` fully
  closed -- closed is the UPPER limit, exactly the polarity of the DROID Robotiq ``finger_joint``
  (0 open, 0.785 closed), so the REALM/OmniGibson command mapping carries over unchanged.
* **Wrist camera.** YAMLab spawns its wrist cameras at runtime under ``<arm>/link_6/wrist_camera``
  from ``yam.yaml``; the USD has none. ``scripts/build_yam_usd.py`` authors the LEFT arm's camera
  into the REALM copy of the asset at the same offset (OmniGibson only discovers Camera prims that
  are direct children of a link). Intrinsics were calibrated at 640x480; REALM renders 1280x720 and
  keeps the horizontal FOV.
* **Gravity.** YAMLab loads the arm with ``disable_gravity=True``; OmniGibson disables gravity on
  every non-fixed robot link itself (``Robot.load``), so nothing is authored for it.
* **Not ported.** The workstation table/gate, the top camera, the second arm, contact-sensor
  grasp detection and MimicGen. The workstation USDs are copied verbatim under
  ``realm/robots/yam/workstation/`` for reference but no REALM config loads them.
"""

import math


class YamRobot:
    """YAM single-arm spec. Class attributes only; nothing here touches a simulator."""

    #: `model` in realm/config/robots/YAM*.yaml and the RobotDefinition directory/stem.
    MODEL = "yam"
    #: `name` in realm/config/robots/YAM*.yaml; also the observation-dict key and the
    #: ROBOT_OBS_PROFILES key. `--robot` values must start with this so sim_config and env_config
    #: recognise the robot.
    NAME = "YAM"
    #: Path OmniGibson loads inside the container (definition `usd_path`).
    USD_PATH = "/app/realm/robots/yam/yam.usd"

    # --- kinematic structure (names as they appear in realm/robots/yam/yam.usd) ---------------
    ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    FINGER_JOINTS = ("left_finger", "right_finger")
    #: `arm` in the YAMLab USD; renamed by the build script because OmniGibson's root-link
    #: fallback looks for "base_link" and "arm" collides with the articulation prim's name.
    BASE_LINK = "base_link"
    ARM_LINKS = ("base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6")
    #: Flange link: the gripper housing the fingers and camera bodies hang off.
    FLANGE_LINK = "link_6"
    #: Virtual tool-centre-point frame, authored by scripts/build_yam_usd.py as a massless,
    #: geometry-free link fixed to the flange at the midpoint of the two fingertip keypoints.
    #: OmniGibson makes every manipulation robot's eef link INVISIBLE at initialisation (its
    #: convention is that the eef link is such a virtual frame, cf. DROID's panda_link8); pointing
    #: eef_link_names at link_6 made the gripper housing disappear from every render (container run,
    #: 2026-09-04). YAMLab has no EE frame in its USD -- its IK is Jacobian-based.
    EEF_LINK = "eef_link"
    FINGER_LINKS = ("left_finger", "right_finger")
    #: YAMLab configs/robot/yam.yaml fingers.<lf|rf>.keypoints[0]: the exact fingertip, in the
    #: finger-link frame (metres). The TCP is their midpoint.
    FINGERTIP_KEYPOINTS = {
        "left_finger": (-0.088, 0.025, -0.045),
        "right_finger": (-0.025, -0.088, -0.045),
    }
    #: Non-actuated bodies fixed to link_6 (camera mount + Intel D405 housing + optical frame).
    FIXED_CAMERA_LINKS = ("camera_mount", "camera_d405", "camera_frame")
    #: Every geometry-free link fixed to the flange (the eef frame plus the D405 optical frame).
    VIRTUAL_LINKS = ("eef_link", "camera_frame")

    ARM_DOF = 6
    N_DOF = 8
    #: [arm(6), gripper(1)] -- the vector REALM's Rollout sends to env.step.
    ACTION_DIM = 7

    # --- gripper (YAMLab configs/robot/yam.yaml: gripper) ------------------------------------
    GRIPPER_OPEN_QPOS = -0.0475
    GRIPPER_CLOSED_QPOS = 0.0

    #: Reset/default joint state, in OmniGibson DOF order (arm joints then fingers): YAMLab's
    #: ArticulationCfg.InitialStateCfg -- every arm joint at 0, fingers fully open.
    DEFAULT_JOINT_POS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0475, -0.0475)

    # --- actuation (YAMLab robot/yam/yam.py actuators + configs/robot/yam.yaml controller) ----
    #: IsaacLab actuator groups -> joints. Order matters: it is the order gains are expanded in.
    ARM_GAIN_GROUPS = (
        ("yam_shoulder", ("joint1", "joint2", "joint3")),   # DM4340 motors
        ("yam_elbow", ("joint4",)),                          # DM4310
        ("yam_wrist", ("joint5", "joint6")),                 # DM4310
    )
    GRIPPER_GAIN_GROUP = "yam_gripper"
    #: stiffness (kp) / damping (kd) per actuator group, per gain set.
    GAIN_SETS = {
        "base": {
            "yam_shoulder": (40.0, 2.5),
            "yam_elbow": (10.0, 1.0),
            "yam_wrist": (10.0, 1.0),
            "yam_gripper": (100.0, 10.0),
        },
        "high_pd": {
            "yam_shoulder": (800.0, 50.0),
            "yam_elbow": (800.0, 50.0),
            "yam_wrist": (30.0, 5.0),
            "yam_gripper": (2000.0, 100.0),
        },
    }
    #: YAMLab `controller.default`.
    DEFAULT_GAIN_SET = "high_pd"
    #: Drive maxForce per joint (IsaacLab effort_limit_sim). The arm values are already authored in
    #: the YAMLab USD; the finger value is NOT (the USD says 0.0, which would leave the gripper
    #: unable to move under OmniGibson) and is authored by scripts/build_yam_usd.py.
    EFFORT_LIMITS = {
        "joint1": 28.0, "joint2": 28.0, "joint3": 28.0,
        "joint4": 10.0, "joint5": 10.0, "joint6": 10.0,
        "left_finger": 100.0, "right_finger": 100.0,
    }

    #: YAMLab steps physics at 120 Hz with decimation 4.
    CONTROL_FREQ_HZ = 30
    PHYSICS_FREQ_HZ = 120

    # --- wrist camera (YAMLab configs/robot/yam.yaml: cameras.left_wrist) ---------------------
    WRIST_CAMERA_PRIM = "wrist_camera"
    WRIST_CAMERA_LINK = "link_6"
    #: Offset from link_6, metres.
    WRIST_CAMERA_POSITION = (-0.0004, 0.069638, 0.073063)
    #: (w, x, y, z), OpenGL camera convention (looks down -Z, +Y up) -- which is also the USD
    #: Camera prim convention, so it is authored as xformOp:orient verbatim.
    WRIST_CAMERA_QUAT_WXYZ = (-0.003227, 0.002817, 0.975619, 0.219430)
    WRIST_CAMERA_INTRINSICS = {"fx": 390.666, "fy": 390.162, "cx": 317.526, "cy": 236.146}
    WRIST_CAMERA_CALIB_RESOLUTION = (640, 480)
    #: REALM renders every robot camera at this resolution (realm/config/robots/*.yaml).
    RENDER_RESOLUTION = (1280, 720)
    #: OmniGibson's VisionSensor default horizontal aperture; kept, and the focal length is derived
    #: so that the horizontal FOV equals the calibrated one.
    WRIST_CAMERA_HORIZONTAL_APERTURE = 20.955
    #: (near, far) in metres. The camera origin lies ~1.5 cm INSIDE the D405 housing's visual mesh
    #: (camera_d405 link), so with OmniGibson's default 1 mm near plane the housing walls fill the
    #: frame and the wrist image is black -- observed on the first container run, 2026-09-04.
    #: IsaacLab's PinholeCameraCfg default near plane is 0.1 m, which is what YAMLab rendered with.
    WRIST_CAMERA_CLIPPING_RANGE = (0.1, 10000000.0)

    # --- placement in REALM scenes ----------------------------------------------------------
    #: Height of the arm base above the scene's robot spawn point. REALM's scene spawn poses put the
    #: DROID column foot on the floor and the Franka base ends up DROID_BASE_HEIGHT (0.863891 m)
    #: up; the YAM asset is a bare arm, so the same height is applied as a mount offset. This keeps
    #: the exterior camera extrinsics (which are expressed relative to the arm base) identical
    #: across robots. YAMLab mounts the arm 0.76 m up on its own 0.7517 m table; REALM's tables are
    #: 0.80-1.05 m, so YAMLab's value would bury the arm. Tune per scene if needed via the
    #: `mount_height` key in realm/config/robots/YAM*.yaml.
    MOUNT_HEIGHT = 0.863891

    # ------------------------------------------------------------------------------------------

    @classmethod
    def arm_gains(cls, gain_set=None):
        """(kp[6], kd[6]) per arm joint, in ARM_JOINTS order, for `isaac_kp` / `isaac_kd`."""
        gains = cls.GAIN_SETS[cls.DEFAULT_GAIN_SET if gain_set is None else gain_set]
        by_joint = {}
        for group, joints in cls.ARM_GAIN_GROUPS:
            for joint in joints:
                by_joint[joint] = gains[group]
        assert set(by_joint) == set(cls.ARM_JOINTS)
        kp = [by_joint[j][0] for j in cls.ARM_JOINTS]
        kd = [by_joint[j][1] for j in cls.ARM_JOINTS]
        return kp, kd

    @classmethod
    def gripper_gains(cls, gain_set=None):
        """(kp, kd) shared by both finger joints."""
        gains = cls.GAIN_SETS[cls.DEFAULT_GAIN_SET if gain_set is None else gain_set]
        return gains[cls.GRIPPER_GAIN_GROUP]

    @classmethod
    def wrist_camera_focal_length(cls, horizontal_aperture=None):
        """Focal length giving the calibrated horizontal FOV at `horizontal_aperture`.

        HFOV = 2 atan(W / 2 fx) = 2 atan(aperture / 2 f)  =>  f = aperture * fx / W.
        """
        aperture = cls.WRIST_CAMERA_HORIZONTAL_APERTURE if horizontal_aperture is None else horizontal_aperture
        width = cls.WRIST_CAMERA_CALIB_RESOLUTION[0]
        return round(aperture * cls.WRIST_CAMERA_INTRINSICS["fx"] / width, 4)

    @classmethod
    def wrist_camera_hfov_deg(cls):
        width = cls.WRIST_CAMERA_CALIB_RESOLUTION[0]
        return math.degrees(2.0 * math.atan(width / (2.0 * cls.WRIST_CAMERA_INTRINSICS["fx"])))

    @classmethod
    def obs_profile(cls):
        """The ROBOT_OBS_PROFILES entry (see realm/inference/utils.py for the field contract).

        `proprio` is OmniGibson's joint_qpos in DOF order, so the arm is the first ARM_DOF entries
        and the gripper proxy joint (left_finger) is at index ARM_DOF -- the analogue of DROID's
        `proprio[:7]` / index 7.
        """
        return dict(
            wrist_camera_link=cls.WRIST_CAMERA_LINK,
            wrist_camera_idx=0,
            wrist_camera_prim=cls.WRIST_CAMERA_PRIM,
            arm_dof=cls.ARM_DOF,
            gripper_proprio_idx=cls.ARM_DOF,
            gripper_open_qpos=cls.GRIPPER_OPEN_QPOS,
            gripper_closed_qpos=cls.GRIPPER_CLOSED_QPOS,
        )
