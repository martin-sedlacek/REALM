"""Shared task rules and REALM environment switches."""

import os
import numpy as np


COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [],
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open_drawer": ["close_drawer"],
    "close_drawer": ["open_drawer"],
}

VERB_PHRASE = {
    "pick": "pick up",
    "put": "put",
    "rotate": "rotate",
    "stack": "stack",
    "push": "push",
    "open_drawer": "open",
    "close_drawer": "close",
}

UNSUPPORTED_BY_PERTURBATION = {
    "SB-VRB": {"open_drawer", "close_drawer"},
    "SB-NOUN": {"push"},
    "VB-MOBJ": {"open_drawer", "close_drawer"},
}


def env_flag(name: str, default: bool) -> bool:
    """Read a REALM boolean environment switch."""
    return os.environ.get(name, "1" if default else "0") == "1"


def env_value(name: str, default: str) -> str:
    """Read a REALM environment value."""
    return os.environ.get(name, default)


def env_is_set(name: str) -> bool:
    return name in os.environ


# Environment and rollout constants.
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
DROID_BASE_HEIGHT = 0.863891
DROID_DEFAULT_DOF = 13
DEFAULT_IMPULSE_THRESHOLD = 1e-3
WARMUP_STEPS = 30
INIT_OPENNESS_FRACTION = 1.0
JOINT_SETTLE_STEPS = 30
JOINT_HOLD_STEPS = 10
SETTLE_STEPS = 30
MASS_CLIP_KG = 2.0
RECEIVER_MAX_DIM = 0.185
DISTRACTOR_MAX_DIM = 0.12
SIGMA_RANGE = (0.0, 2.5)
ALPHA_RANGE = (0.25, 1.5)
LIGHT_INTENSITY_RANGE = (20000, 750000)
LIGHT_COLOR_MEAN = (255, 214, 170)
LIGHT_COLOR_STD = 15
MAX_POS_DEVIATION = 0.2
MAX_PITCH_DEVIATION = 0.2
MAX_YAW_DEVIATION = 0.2
RESCALE_RANGE = (0.5, 1.5)
MAX_VOLUME_FACTOR = 1.5
RESCALE_MAX_TRIES = 1000
DRAWER_BBOX_CLIP = (0.4, 0.75)
TABLETOP_BBOX_CLIP = (0.02, 0.175)
SWITCH_DZ_RANGE = 0.15
SWITCH_DXY_RANGE = 0.075
DRAWER_YAW_NOISE_STD = (0, 0, 0.12)
DRAWER_YAW_NOISE_MEAN = (0, 0, 0.25)
DRAWER_YAW_CLIP_MIN = [-3.14, -3.14, 0]
DRAWER_YAW_CLIP_MAX = [3.14, 3.14, 0.57]
TABLETOP_YAW_NOISE_STD = (0, 0, 3.14)
DRAWER_Z_OFFSET = 0.3
DEFAULT_BBOX_EXTENT = (0.08, 0.08, 0.08)
DROP_HEIGHT = 0.1
VIDEO_TARGET_HEIGHT = 480
CONTROL_HZ = 15.0
CONTROL_DT = 1.0 / CONTROL_HZ
TERMINAL_STEPS = 15
SHORT_TRAJECTORY_SAMPLES = 4
PLACEMENT_TASK_TYPES = ("put", "stack")
GRIPPER_OPEN_ABOVE_HALF = ("debug", "openpi", "GR00T", "GR00T_N16", "dreamzero", "yamlab", "openpi_yam")
GRIPPER_OPEN_BELOW_HALF = ("molmoact",)
