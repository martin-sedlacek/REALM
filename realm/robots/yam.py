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
  closed -- closed is the UPPER limit. That is the OPPOSITE of what OmniGibson's stock binary
  ``MultiFingerGripperController`` assumes (open = upper limit, close = lower limit; DROID's Robotiq
  goes through REALM's own ``CustomGripperController``), so every YAM gripper block names
  ``open_qpos`` / ``closed_qpos`` explicitly. Without them a close command opened the fingers
  (measured in-container 2026-09-05, job 204583).
* **Wrist camera.** YAMLab spawns its wrist cameras at runtime under ``<arm>/link_6/wrist_camera``
  from ``yam.yaml``; the USD has none. ``scripts/build_yam_usd.py`` authors the LEFT arm's camera
  into the REALM copy of the asset at the same offset (OmniGibson only discovers Camera prims that
  are direct children of a link). Intrinsics were calibrated at 640x480; REALM renders 960x720, the same
  4:3 aspect, so both FOVs match the calibration.
* **Gravity.** YAMLab loads the arm with ``disable_gravity=True``; OmniGibson disables gravity on
  every non-fixed robot link itself (``Robot.load``), so nothing is authored for it.
* **Bimanual workstation.** :class:`YamBimanualRobot` composes two arms and YAMLab's fixed top
  camera into the second robot ``yam_bimanual`` (asset built from the single-arm USD by
  ``scripts/build_yam_bimanual_usd.py``); it reuses every number here and adds only YAMLab's arm
  placement, the right-wrist camera offset and the top-camera calibration.
* **Not ported.** The workstation table/gate, contact-sensor grasp detection and MimicGen. The
  workstation USDs are copied verbatim under ``realm/robots/yam/workstation/`` for reference but no
  REALM config loads them.
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
    #: Tool-centre point in the flange (link_6) frame, metres. None = derive it from
    #: FINGERTIP_KEYPOINTS (the build script does); a variant with a known TCP sets it directly.
    TCP_IN_FLANGE = None
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
    #: REALM renders the wrist cameras at this resolution (realm/config/robots/YAM*.yaml). 4:3, the aspect
    #: the D405 intrinsics were calibrated at and the aspect YAMLab (640x480) and ABC (224x168) render, so
    #: the vertical FOV is the calibrated one too. Rendering 16:9 kept the horizontal FOV but cut the
    #: vertical one to 49 deg, and a 16:9 frame letterboxed into a policy's square input comes out ~25%
    #: smaller than the 4:3 training frames -- the "zoomed out" wrist view seen next to ABC's renders.
    RENDER_RESOLUTION = (960, 720)
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
    #: Rigid offset of the robot from the DROID spawn pose, in the robot's own frame (forward, left, up)
    #: metres and yaw degrees: the YAM's reach (~0.6 m) is well short of the Franka's, so it is moved
    #: 0.30 m straight toward the workspace. Chosen by eye in the Isaac GUI on 2026-09-05
    #: (the bimanual robot, whose frame is the midpoint of the two arm bases); the same offset is applied
    #: to the single arm so both YAM robots see the same scene. `spawn_offset` key in every YAM config;
    #: env_config moves the spawn, the robot-frame cameras and the EE transforms together.
    SPAWN_OFFSET_POS = (0.30, 0.0, 0.0)
    SPAWN_OFFSET_YAW_DEG = 0.0

    #: YAMLab world position of the bimanual mount frame (midpoint of ``arms.left/right.position``), also
    #: the point of the gate frame the assets are centred on.
    YAMLAB_MOUNT_IN_WORLD = (0.2525, 0.0, 0.76)

    # --- workstation frame (YAMLab robot/yam/workstation/workstation.usd: visuals/gate_visual) -------
    #: Visual-only link carrying YAMLab's aluminium-extrusion "gate": the frame the two arms bolt onto,
    #: on every YAM asset (the single arm stands centred on its front cross bar, the bimanual pair on its
    #: two mount plates)
    #: (0.6 m deep, 1.3 m wide, 1.68 m tall in YAMLab's world; arm mount plates at z = 0.76, two posts
    #: and a top bar behind the arms where the top camera sits). Without it the arms float in mid-air at
    #: ``mount_height``. No collision geometry: REALM's scenes bring their own table, and a colliding
    #: frame link would be counted as an environment collision on every step by ``check_collisions``.
    FRAME_LINK = "frame"
    FRAME_SOURCE_USD = "realm/robots/yam/workstation/workstation.usd"
    FRAME_SOURCE_PRIM = "/workstation_only/workstation/visuals/gate_visual"
    #: YAMLab's arm mount plates sit 0.76 m up the frame, but REALM mounts the arms ``MOUNT_HEIGHT``
    #: (0.864 m) above the floor. The part of the frame BELOW the mount plane is stretched in z by this
    #: factor so the frame stands on the floor with its plates under the arms; the extrusion cross
    #: sections are unchanged and the posts/top bar above the plates keep YAMLab's dimensions (so the
    #: top camera still sits just above the top bar). Set to 1.0 for the verbatim mesh (floats 10 cm).
    FRAME_STRETCH_BELOW_MOUNT = MOUNT_HEIGHT / YAMLAB_MOUNT_IN_WORLD[2]
    #: YAMLab binds OmniPBR "Aluminum_Scratched_Light_Bumpy" with these constants (plus an emissive
    #: term tuned for their dim dome light, dropped here); authored as a UsdPreviewSurface.
    FRAME_MATERIAL = {"diffuse": (0.92, 0.92, 0.94), "metallic": 1.0, "roughness": 0.22}


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
    def spawn_offset(cls):
        """The robot config's `spawn_offset` entry (REALM-only key, read by env_config)."""
        return {"pos": list(cls.SPAWN_OFFSET_POS), "yaw_deg": cls.SPAWN_OFFSET_YAW_DEG}

    @classmethod
    def frame_origin_in_mount(cls):
        """Where YAMLab's world origin (the frame mesh's frame) sits in the mount frame, metres."""
        return tuple(-v for v in cls.YAMLAB_MOUNT_IN_WORLD)

    @classmethod
    def frame_z_in_mount(cls, z_yamlab):
        """Height of a frame-mesh point in the mount frame after the below-plate stretch: the mount
        plane (YAMLab z = 0.76) maps to 0, the frame's foot (z = 0) to -MOUNT_HEIGHT, i.e. the floor."""
        plate = cls.YAMLAB_MOUNT_IN_WORLD[2]
        if z_yamlab < plate:
            return -(plate - z_yamlab) * cls.FRAME_STRETCH_BELOW_MOUNT
        return z_yamlab - plate

    @classmethod
    def finger_open_qpos(cls):
        """Per-finger joint position at fully open, in FINGER_JOINTS order (the binary gripper
        controller's `open_qpos`). Both YAMLab fingers share the sign; the crank gripper does not."""
        return tuple(cls.GRIPPER_OPEN_QPOS for _ in cls.FINGER_JOINTS)

    @classmethod
    def finger_closed_qpos(cls):
        return tuple(cls.GRIPPER_CLOSED_QPOS for _ in cls.FINGER_JOINTS)

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


class YamCrankRobot(YamRobot):
    """The YAM arm with I2RT's earlier "crankshaft" gripper, from the ABC project's MuJoCo model
    (https://abc.bot, ``assets/put_bottles/assets/i2rt_yam/yam.xml``, model ``yam_v0``).

    The six arm links are the same parts as YAMLab's (identical ``model2*`` meshes and joint frames),
    so the asset is the built ``yam.usd`` with everything downstream of the ``link_6`` wrist motor
    replaced from the MJCF by ``scripts/build_yam_crank_usd.py``: the gripper housing (``link_6``
    visuals, capsule collisions, inertia), the two fingers (angled crank fingers with their MJCF
    capsule/box collision pads), the wrist D405 on ABC's steeper bracket, and the TCP.

    What differs from :class:`YamRobot` at runtime:

    * **Finger sign.** Both models put 0 at CLOSED, but ABC's fingers slide the other way: the left
      finger is fully open at ``+0.0475`` (right at ``-0.0475``), YAMLab's at ``-0.0475``. The binary
      gripper controller therefore gets per-finger ``open_qpos`` (:meth:`finger_open_qpos`), and the
      gripper proxy (left finger) normalises with ``GRIPPER_OPEN_QPOS = +0.0475``. ABC's own policies
      see the gripper as ``q / 0.0475`` (1 open, 0 closed).
    * **Wrist camera.** ABC mounts the D405 at ``(-0.0107, 0.095, 0.062)`` on ``link_6`` tilted 50
      degrees down (YAMLab: 25 degrees, 7 cm out); the composed camera pose is below. Same D405, so
      the intrinsics and the 0.1 m near plane are inherited.
    * **TCP.** ABC's ``grasp_site``: 13.47 cm along the flange axis (YAMLab's fingertip midpoint gave
      14.26 cm).
    * **Reset pose.** ABC's ``home`` keyframe, joints 2 and 3 at 60 degrees, fingers open.
    * **No camera mount link.** ABC hangs the camera body directly off ``link_6``.
    """

    MODEL = "yam_crank"
    NAME = "YAM_crank"
    USD_PATH = "/app/realm/robots/yam/yam_crank.usd"
    #: ABC MJCF: link_6 -> body camera_d405 (the housing mesh) -> body camera_frame (+9 mm) -> camera.
    FIXED_CAMERA_LINKS = ("camera_d405", "camera_frame")
    VIRTUAL_LINKS = ("eef_link", "camera_frame")
    #: ABC ``grasp_site`` on link_6.
    TCP_IN_FLANGE = (0.0, 0.0, 0.1347)
    FINGERTIP_KEYPOINTS = None

    #: Left finger: closed at 0, open at +0.0475 (right finger mirrored by the MJCF equality).
    GRIPPER_OPEN_QPOS = 0.0475
    GRIPPER_CLOSED_QPOS = 0.0
    #: (lower, upper) per finger joint, from the MJCF ranges.
    FINGER_LIMITS = {"left_finger": (-0.00205, 0.0475), "right_finger": (-0.0475, 0.00205)}
    #: ABC ``home`` keyframe arm pose (0, 60 deg, 60 deg, 0, 0, 0) with the fingers open.
    DEFAULT_JOINT_POS = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0475, -0.0475)

    #: ABC put_bottle.xml: body ``<arm>_camera_d405`` under link_6 (the D405 housing mesh ``d405.stl``
    #: plus a 2 cm collision sphere), then body ``<arm>_camera_frame`` and the MuJoCo camera.
    #: MuJoCo quaternions are (w, x, y, z), unnormalised as written.
    D405_BODY_POSITION = (-0.0107, 0.095, 0.062)
    D405_BODY_QUAT_WXYZ = (0.906, 0.423, 0.0, 0.0)
    D405_MESH = "d405.stl"
    D405_RGBA = (0.22, 0.22, 0.24, 1.0)
    D405_COLLISION_RADIUS = 0.02
    CAMERA_FRAME_POSITION = (0.009, 0.0, 0.0)
    CAMERA_FRAME_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)
    MUJOCO_CAMERA_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
    #: Camera pose in link_6, composed from that chain by scripts/build_yam_crank_usd.py (which asserts
    #: it lands here); USD/OpenGL convention, looks 50 degrees below the flange axis.
    WRIST_CAMERA_POSITION = (-0.0017, 0.095, 0.062)
    WRIST_CAMERA_QUAT_WXYZ = (0.0, 0.0, 0.906106, 0.42305)
    #: ABC's MuJoCo camera: ``fovy="58"`` rendered at 224x168 (abc_minimal/config.py camera_width/height),
    #: i.e. 72.9 deg horizontal. Expressed as the equivalent pinhole intrinsics at that resolution so the
    #: base class's focal-length formula applies unchanged: fx = (H/2) / tan(fovy/2).
    WRIST_CAMERA_FOVY_DEG = 58.0
    WRIST_CAMERA_CALIB_RESOLUTION = (224, 168)
    WRIST_CAMERA_INTRINSICS = {"fx": 151.54, "fy": 151.54, "cx": 112.0, "cy": 84.0}
    #: MJCF bodies whose contents become REALM links (link_6's children in yam.xml).
    MJCF_FINGER_BODIES = {"left_finger": "link_left_finger", "right_finger": "link_right_finger"}
    MJCF_FLANGE_BODY = "link_6"

    @classmethod
    def finger_open_qpos(cls):
        return tuple(cls.FINGER_LIMITS[j][1] if j == "left_finger" else cls.FINGER_LIMITS[j][0]
                     for j in cls.FINGER_JOINTS)

    @classmethod
    def finger_closed_qpos(cls):
        return tuple(0.0 for _ in cls.FINGER_JOINTS)


class YamBimanualRobot:
    """YAMLab's bimanual workstation as one OmniGibson robot: two YAM arms on a shared mount plus the
    fixed top camera. Every name is derived from :class:`YamRobot`; every number that is not a YAM arm
    number is from YAMLab ``configs/robot/yam.yaml`` (``arms``, ``cameras.top``, ``cameras.right_wrist``).

    Layout (YAMLab world frame, translated so the robot frame sits at the midpoint of the two arm
    bases): the left arm at ``+0.305`` m in y, the right arm at ``-0.305`` m, both facing +x with
    identity orientation; the top camera 0.166 m behind and 0.944 m above that midpoint, looking
    forward and 60 degrees down. In REALM the midpoint is the robot spawn pose raised by
    ``mount_height``, so the scenes' object placement (relative to the spawn) is centred between the
    arms and the camera extrinsics stay expressed in the arm-base frame like every DROID entry.

    OmniGibson vocabulary: ``arm_names = ["left", "right"]``; link and joint prims carry the arm as a
    prefix (``left_link_6``, ``right_joint1``), so the single-arm finger names become e.g.
    ``left_left_finger`` -- mechanical, but it keeps :meth:`link_name` a one-liner. Controllers are
    ``arm_left, gripper_left, arm_right, gripper_right`` in that order, which makes OmniGibson's
    action vector exactly YAMLab's 14-D ``[left_arm(6), left_gripper(1), right_arm(6),
    right_gripper(1)]``.

    The top camera is NOT authored into the USD: it is REALM's ``external_sensor0`` placed from
    :attr:`EXTERIOR_CAMERA_POSITION` / :attr:`EXTERIOR_CAMERA_QUAT_XYZW` through the robot config's
    ``exterior_camera`` key, so the V-VIEW perturbation and the video recorder treat it like every
    other exterior view. Its extrinsics are fixed to the arms because the robot is fixed-base.

    The workstation's aluminium frame IS in the USD, as the visual-only link :attr:`FRAME_LINK` fixed
    to ``base_link`` (see the FRAME_* constants): it is what the arms visibly stand on.
    """

    #: The single-arm spec both arms are built from; a variant subclass swaps it.
    ARM = YamRobot

    MODEL = "yam_bimanual"
    NAME = "YAM_bimanual"
    USD_PATH = "/app/realm/robots/yam/yam_bimanual.usd"

    #: OmniGibson arm names, in action order (YAMLab: left arm first).
    ARMS = ("left", "right")
    #: Virtual mount frame, the articulation root (geometry-free; OmniGibson fixes it to the world).
    BASE_LINK = "base_link"
    #: Arm-base offsets from the mount frame, metres (YAMLab ``arms.<side>.position`` minus the
    #: midpoint (0.2525, 0, 0.76)); both arms have identity orientation in YAMLab.
    ARM_OFFSETS = {"left": (0.0, 0.305, 0.0), "right": (0.0, -0.305, 0.0)}
    YAMLAB_MOUNT_IN_WORLD = YamRobot.YAMLAB_MOUNT_IN_WORLD

    #: The gate frame (see YamRobot.FRAME_*): the bimanual builder authors it centred on the mount frame,
    #: which is where YAMLab's arm plates are.
    FRAME_LINK = YamRobot.FRAME_LINK
    FRAME_SOURCE_USD = YamRobot.FRAME_SOURCE_USD
    FRAME_SOURCE_PRIM = YamRobot.FRAME_SOURCE_PRIM
    FRAME_STRETCH_BELOW_MOUNT = YamRobot.FRAME_STRETCH_BELOW_MOUNT
    FRAME_MATERIAL = YamRobot.FRAME_MATERIAL
    frame_origin_in_mount = YamRobot.frame_origin_in_mount
    frame_z_in_mount = YamRobot.frame_z_in_mount
    #: Per-arm wrist camera offset from ``<arm>_link_6`` (YAMLab ``cameras.left_wrist`` /
    #: ``cameras.right_wrist``; the two calibrations differ by ~1 mm).
    WRIST_CAMERA_POSITIONS = {
        "left": YamRobot.WRIST_CAMERA_POSITION,
        "right": (0.0, 0.069638, 0.072),
    }

    ARM_DOF = YamRobot.ARM_DOF
    N_DOF = 2 * YamRobot.N_DOF
    #: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] -- YAMLab's YamActionLayout.
    ACTION_DIM = 2 * YamRobot.ACTION_DIM
    #: Columns of the action vector that carry a gripper command (binarised by realm.rollout).
    GRIPPER_ACTION_IDX = (YamRobot.ARM_DOF, 2 * YamRobot.ARM_DOF + 1)

    GRIPPER_OPEN_QPOS = YamRobot.GRIPPER_OPEN_QPOS
    GRIPPER_CLOSED_QPOS = YamRobot.GRIPPER_CLOSED_QPOS
    CONTROL_FREQ_HZ = YamRobot.CONTROL_FREQ_HZ
    PHYSICS_FREQ_HZ = YamRobot.PHYSICS_FREQ_HZ
    MOUNT_HEIGHT = YamRobot.MOUNT_HEIGHT
    SPAWN_OFFSET_POS = YamRobot.SPAWN_OFFSET_POS
    SPAWN_OFFSET_YAW_DEG = YamRobot.SPAWN_OFFSET_YAW_DEG
    spawn_offset = YamRobot.spawn_offset
    RENDER_RESOLUTION = YamRobot.RENDER_RESOLUTION

    # --- top camera (YAMLab configs/robot/yam.yaml: cameras.top) -------------------------------
    #: Offset from the mount frame, metres: (0.0860, -0.0090, 1.7043) - (0.2525, 0, 0.76).
    EXTERIOR_CAMERA_POSITION = (-0.1664949, -0.009, 0.9443205)
    #: YAMLab ``quaternion_opengl`` (w, x, y, z) = (0.68301, 0.18301, -0.18301, -0.68301), reordered to
    #: the (x, y, z, w) REALM's camera_extrinsics use. OpenGL/USD camera convention (looks down -Z):
    #: view direction (0.5, 0, -0.866) in the mount frame -- forward and 60 degrees down.
    EXTERIOR_CAMERA_QUAT_XYZW = (0.18301, -0.18301, -0.68301, 0.68301)
    EXTERIOR_CAMERA_INTRINSICS = {"fx": 392.195617675781, "fy": 391.722351074219,
                                  "cx": 318.389434814453, "cy": 237.876312255859}
    EXTERIOR_CAMERA_CALIB_RESOLUTION = (640, 480)
    #: OmniGibson VisionSensor default aperture (realm/config/env/external_sensors/camera_config.yaml
    #: only sets focal_length).
    EXTERIOR_CAMERA_HORIZONTAL_APERTURE = 20.955

    # ------------------------------------------------------------------------------------------

    @classmethod
    def link_name(cls, arm, name):
        """Prim name of single-arm link (or joint) `name` on `arm`."""
        return f"{arm}_{name}"

    joint_name = link_name

    @classmethod
    def arm_links(cls, arm):
        return tuple(cls.link_name(arm, n) for n in cls.ARM.ARM_LINKS)

    @classmethod
    def arm_joints(cls, arm):
        return tuple(cls.joint_name(arm, n) for n in cls.ARM.ARM_JOINTS)

    @classmethod
    def finger_links(cls, arm):
        return tuple(cls.link_name(arm, n) for n in cls.ARM.FINGER_LINKS)

    @classmethod
    def finger_joints(cls, arm):
        return tuple(cls.joint_name(arm, n) for n in cls.ARM.FINGER_JOINTS)

    @classmethod
    def eef_link(cls, arm):
        return cls.link_name(arm, cls.ARM.EEF_LINK)

    @classmethod
    def flange_link(cls, arm):
        return cls.link_name(arm, cls.ARM.FLANGE_LINK)

    @classmethod
    def mount_joint(cls, arm):
        """Fixed joint base_link -> <arm>_base_link authored by scripts/build_yam_bimanual_usd.py."""
        return f"{arm}_mount"

    @classmethod
    def gripper_proxy_joint(cls, arm):
        """The finger joint read as the arm's gripper state (YAMLab reads the driven ``left_finger``)."""
        return cls.joint_name(arm, cls.ARM.FINGER_JOINTS[0])

    @classmethod
    def all_links(cls, arm):
        """Every link prim of one arm, including the geometry-free frames."""
        return tuple(cls.link_name(arm, n) for n in (*cls.ARM.ARM_LINKS, *cls.ARM.FINGER_LINKS,
                                                    *cls.ARM.FIXED_CAMERA_LINKS, *cls.ARM.VIRTUAL_LINKS))

    @classmethod
    def collision_links(cls, arm):
        """Links of one arm that carry collision geometry (everything but the virtual frames)."""
        return tuple(cls.link_name(arm, n) for n in (*cls.ARM.ARM_LINKS, *cls.ARM.FINGER_LINKS,
                                                    *cls.ARM.FIXED_CAMERA_LINKS))

    @classmethod
    def dof_order(cls):
        """Articulation DOF order OmniGibson reports (``dof_names_ordered``), the order of
        ``default_joint_pos`` / ``reset_joint_pos`` / ``proprio``.

        PhysX numbers an articulation's joints BREADTH-first from the root link, one tree level at a
        time with the links of a level visited in authoring order (left arm first): so the two
        ``joint1`` come first, then the two ``joint2``, ..., the two ``joint6``, and finally the fingers
        (children of the two ``link_6``: the left arm's pair, then the right arm's). Measured on the
        built robot in the container on 2026-09-05 (job 204581); the depth-first order assumed before
        that was wrong. ``assert_proprio_layout`` pins this list against the built robot at
        construction; the arm and gripper indices REALM uses are looked up in it by name, never
        assumed contiguous -- an arm's joints are NOT contiguous here.
        """
        order = []
        for joint in cls.ARM.ARM_JOINTS:
            for arm in cls.ARMS:
                order.append(cls.joint_name(arm, joint))
        for arm in cls.ARMS:
            order.extend(cls.finger_joints(arm))
        return tuple(order)

    @classmethod
    def default_joint_pos(cls):
        """YAMLab's initial state for both arms, in :meth:`dof_order`."""
        single = cls.ARM.DEFAULT_JOINT_POS
        by_joint = {}
        for arm in cls.ARMS:
            for j, value in zip((*cls.arm_joints(arm), *cls.finger_joints(arm)), single):
                by_joint[j] = value
        return tuple(by_joint[j] for j in cls.dof_order())

    @classmethod
    def raw_controller_order(cls):
        return tuple(f"{kind}_{arm}" for arm in cls.ARMS for kind in ("arm", "gripper"))

    @classmethod
    def disabled_collision_pairs(cls):
        """Every intra-arm pair of collision links, both arms.

        YAMLab loads each arm as its own articulation with ``enabled_self_collisions=False`` -- so an
        arm never collides with itself but the two arms DO collide with each other. In OmniGibson both
        arms are one articulation, so the same behaviour is ``self_collisions: true`` plus every
        within-arm pair filtered out. (PhysX already skips joint-adjacent pairs; listing them is
        harmless and keeps the rule "all pairs of one arm" mechanical.)
        """
        pairs = []
        for arm in cls.ARMS:
            links = cls.collision_links(arm)
            for i, a in enumerate(links):
                for b in links[i + 1:]:
                    pairs.append([a, b])
        return pairs

    @classmethod
    def exterior_camera_focal_length(cls, horizontal_aperture=None):
        """Focal length giving the top camera's calibrated horizontal FOV (78.4 deg); see
        :meth:`YamRobot.wrist_camera_focal_length` for the formula."""
        aperture = cls.EXTERIOR_CAMERA_HORIZONTAL_APERTURE if horizontal_aperture is None else horizontal_aperture
        width = cls.EXTERIOR_CAMERA_CALIB_RESOLUTION[0]
        return round(aperture * cls.EXTERIOR_CAMERA_INTRINSICS["fx"] / width, 4)

    @classmethod
    def exterior_camera(cls):
        """The robot config's ``exterior_camera`` entry (REALM-only key, read by env_config)."""
        return {
            "cam1": {"pos": list(cls.EXTERIOR_CAMERA_POSITION), "rot": list(cls.EXTERIOR_CAMERA_QUAT_XYZW)},
            "focal_length": cls.exterior_camera_focal_length(),
        }

    @classmethod
    def obs_profile(cls):
        """The multi-arm ROBOT_OBS_PROFILES entry (field contract in realm/inference/utils.py).

        ``arms`` marks the profile as multi-arm. The policy's joint state is the concatenation of the
        arms' joint positions in ``arms`` order (12), the gripper state one normalised value per arm
        (2), both looked up in ``proprio`` by joint name through ``dof_order``. The legacy single-arm
        keys (``wrist_camera_*``, ``gripper_proprio_idx``) describe the FIRST arm so every caller that
        only knows one wrist camera / one gripper keeps working on the default arm.
        """
        order = cls.dof_order()
        first = cls.ARMS[0]
        return dict(
            arms=cls.ARMS,
            arm_dof=cls.ARM_DOF,
            dof_order=order,
            arm_joint_names={arm: cls.arm_joints(arm) for arm in cls.ARMS},
            finger_joint_names={arm: cls.finger_joints(arm) for arm in cls.ARMS},
            gripper_proxy_joints={arm: cls.gripper_proxy_joint(arm) for arm in cls.ARMS},
            gripper_action_idx=cls.GRIPPER_ACTION_IDX,
            wrist_cameras={arm: dict(link=cls.flange_link(arm), idx=0, prim=cls.ARM.WRIST_CAMERA_PRIM)
                           for arm in cls.ARMS},
            # first-arm view for single-wrist / single-gripper callers
            wrist_camera_link=cls.flange_link(first),
            wrist_camera_idx=0,
            wrist_camera_prim=cls.ARM.WRIST_CAMERA_PRIM,
            gripper_proprio_idx=order.index(cls.gripper_proxy_joint(first)),
            gripper_open_qpos=cls.GRIPPER_OPEN_QPOS,
            gripper_closed_qpos=cls.GRIPPER_CLOSED_QPOS,
        )


class YamCrankBimanualRobot(YamBimanualRobot):
    """ABC's bimanual workstation: two :class:`YamCrankRobot` arms on the same gate frame.

    ABC's ``put_bottle.xml`` places the arms at ``(0.2525, +-0.31, 0.76)`` (YAMLab: +-0.305), mounts
    the top D405 at exactly YAMLab's ``cameras.top`` pose on the gate's top bar, and uses the same
    ``base_visual_gate`` mesh as YAMLab's ``gate_visual``, so everything but the arm spec, the arm
    spacing and the wrist-camera offsets is inherited. Both wrist cameras share ABC's single offset.
    """

    ARM = YamCrankRobot
    MODEL = "yam_crank_bimanual"
    NAME = "YAM_crank_bimanual"
    USD_PATH = "/app/realm/robots/yam/yam_crank_bimanual.usd"
    ARM_OFFSETS = {"left": (0.0, 0.31, 0.0), "right": (0.0, -0.31, 0.0)}
    WRIST_CAMERA_POSITIONS = {"left": YamCrankRobot.WRIST_CAMERA_POSITION,
                              "right": YamCrankRobot.WRIST_CAMERA_POSITION}
    GRIPPER_OPEN_QPOS = YamCrankRobot.GRIPPER_OPEN_QPOS
    GRIPPER_CLOSED_QPOS = YamCrankRobot.GRIPPER_CLOSED_QPOS
