"""The YAMLab / LeRobot observation and action contract for the bimanual YAM, as pure numpy.

A policy trained on YAMLab data (ARISE-Initiative/yamlab, ``yamlab/utils/recorders.py``
``LeRobotRecorder._get_default_bimanual_features``) sees

* ``observation.state``  float32 (14,): ``[left_joint1..6, left_finger, right_joint1..6, right_finger]``
  -- raw joint positions, fingers in metres (``-0.0475`` open, ``0.0`` closed);
* ``observation.images.top_rgb`` / ``left_rgb`` / ``right_rgb``  uint8 (H, W, 3);
* ``action``  float32 (14,): the same layout, absolute joint targets.

REALM's own vocabulary is ``robot_state = [left(6), right(6)]``, ``gripper_state = [left, right]``
normalised to (0 open, 1 closed), and an action whose gripper columns 6 and 13 are binarised by
``realm.rollout`` with the "open above 0.5" convention. This module converts between the two; the
websocket adapter in ``realm/inference/client.py`` (``model_type="yamlab"``) is a thin wrapper around it.
No omnigibson / torch imports so ``tests/test_yam_robot.py`` can pin it on the host.
"""

import numpy as np

from realm.robots.yam import YamBimanualRobot as _B

ARM_DOF = _B.ARM_DOF
STATE_DIM = _B.ACTION_DIM  # 14: the LeRobot state and action share the layout
FINGER_IDX = _B.GRIPPER_ACTION_IDX  # (6, 13)
IMAGE_KEYS = ("observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb")
STATE_KEY = "observation.state"


def finger_qpos_from_normalised(gripper_state, open_qpos=_B.GRIPPER_OPEN_QPOS, closed_qpos=_B.GRIPPER_CLOSED_QPOS):
    """REALM's (0 open, 1 closed) gripper state -> finger joint position in metres."""
    g = np.asarray(gripper_state, dtype=np.float64)
    return open_qpos + g * (closed_qpos - open_qpos)


def open_fraction_from_finger_qpos(finger_qpos, open_qpos=_B.GRIPPER_OPEN_QPOS, closed_qpos=_B.GRIPPER_CLOSED_QPOS):
    """Finger joint target in metres -> REALM's policy gripper value: 1 fully open, 0 fully closed
    (binarised at 0.5 by realm.rollout for model types in GRIPPER_OPEN_ABOVE_HALF)."""
    q = np.asarray(finger_qpos, dtype=np.float64)
    return np.clip((q - closed_qpos) / (open_qpos - closed_qpos), 0.0, 1.0)


def yamlab_state(robot_state, gripper_state):
    """``[left(6), right(6)]`` + normalised ``[left, right]`` grippers -> the 14-D LeRobot state."""
    robot_state = np.asarray(robot_state, dtype=np.float64)
    fingers = finger_qpos_from_normalised(np.atleast_1d(gripper_state))
    assert robot_state.shape == (2 * ARM_DOF,) and fingers.shape == (2,), (robot_state.shape, fingers.shape)
    state = np.concatenate([robot_state[:ARM_DOF], fingers[:1], robot_state[ARM_DOF:], fingers[1:]])
    return state.astype(np.float32)


def yamlab_observation(instruction, top_im, left_wrist_im, right_wrist_im, robot_state, gripper_state):
    """The dict sent to a YAMLab-policy server. Images are passed at REALM's native resolution
    (1280x720 uint8 RGB); the server owns its own resize/normalisation, as LeRobot policies do."""
    ims = (top_im, left_wrist_im, right_wrist_im)
    obs = {"prompt": instruction, STATE_KEY: yamlab_state(robot_state, gripper_state)}
    for key, im in zip(IMAGE_KEYS, ims):
        im = np.asarray(im)
        assert im.ndim == 3 and im.shape[2] == 3, f"{key}: expected (H, W, 3), got {im.shape}"
        obs[key] = im.astype(np.uint8, copy=False)
    return obs


def yamlab_actions_to_realm(actions):
    """(n, 14) LeRobot actions (finger targets in metres) -> REALM actions: same columns, the two
    finger columns rewritten as open fractions so realm.rollout's 0.5 threshold reads them."""
    a = np.array(actions, dtype=np.float64, copy=True)
    if a.ndim == 1:
        a = a[None, :]
    assert a.shape[1] == STATE_DIM, f"expected (n, {STATE_DIM}) actions, got {a.shape}"
    for i in FINGER_IDX:
        a[:, i] = open_fraction_from_finger_qpos(a[:, i])
    return a
