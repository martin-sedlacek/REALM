"""Constants shared between the environment and its config builder."""
import numpy as np

# Neutral DROID arm pose, used when neither the task nor the scene names a reset pose.
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])

# Height of the DROID base column. Assets whose origin sits at the arm base rather than at the
# bottom of the column (has_base_column: false) must be raised by this much on base-mounted tasks,
# and it is added when converting between robot- and world-frame poses.
DROID_BASE_HEIGHT = 0.86244

# 7 arm joints + 4 gripper joints, for robot configs that do not declare their own `dof`.
DROID_DEFAULT_DOF = 11
