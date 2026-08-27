from realm.config.shared import (  # noqa: F401
    DEFAULT_RESET_JOINTPOS,
    DROID_BASE_HEIGHT,
    DROID_DEFAULT_DOF,
)

# Measured panda_link0 height above the canonical mounted RoboLab v2 asset root. Camera and coordinate
# transforms add this to move between the bottom-of-column scene pose and the arm-base frame.

# 7 arm joints + one actuated gripper joint and five mimic followers.
