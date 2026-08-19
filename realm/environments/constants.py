"""Constants shared between the environment and its config builder."""
import numpy as np

# Neutral DROID arm pose, used when neither the task nor the scene names a reset pose.
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])

# Height of the DROID base column. Assets whose origin sits at the arm base rather than at the
# bottom of the column (has_base_column: false) must be raised by this much on base-mounted tasks,
# and it is added when converting between robot- and world-frame poses.
#
# THIS CONSTANT IS NOT EXACT FOR THE ASSET IT WAS FITTED TO, and leaving it that way is a deliberate
# 2026-08-19 decision, not an oversight. Measured panda_link0 heights (logs/mountedasset/probe.log):
#
#     droid_robolab_v2_mounted.usd   0.863891   <- NOT loaded; the switch to it was reverted in bda06da
#     droid_mounted.usd (stock)      0.862880   <- what this constant was fitted to
#     this constant                  0.86244
#
# so it is 1.451 mm under the robolab mount and 0.440 mm under the stock one. env.robot_pos stays at
# the scene value (the bottom of the column) and _robot2world / _world2robot /
# construct_ext_cam_pose_by_name add this to reach the arm origin, so those three carry that error --
# they do NOT double-count the switch to a mounted asset, because only the SPAWN point was ever
# adjusted by has_base_column (env_config.py:111).
#
# Why it stays wrong: the constant is shared. Setting it to 0.863891 would make robolab_v2 exact and
# move the stock DROID path by 1.0 mm, i.e. trade a 1.45 mm error on the robot in use for a 1.0 mm
# error on one with recorded results. Correct fix is a per-asset base height rather than one global,
# which is a refactor of five call sites. Martin's call, taken explicitly: leave it, write it down.
# 1.45 mm is far below what any task rubric resolves (the drawer's tightest stage is 15 mm of travel).
DROID_BASE_HEIGHT = 0.86244

# 7 arm joints + 4 gripper joints, for robot configs that do not declare their own `dof`.
DROID_DEFAULT_DOF = 11
