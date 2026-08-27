
import numpy as np

# Neutral DROID arm pose, used when neither the task nor the scene names a reset pose.
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])

# Measured panda_link0 height above the canonical mounted RoboLab v2 asset root. Camera and coordinate
# transforms add this to move between the bottom-of-column scene pose and the arm-base frame.
DROID_BASE_HEIGHT = 0.863891

# 7 arm joints + one actuated gripper joint and five mimic followers.
DROID_DEFAULT_DOF = 13
