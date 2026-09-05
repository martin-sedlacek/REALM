"""Observation/action contract of the openpi `yam_pi05` policy (robocurve/pi05-yam-molmoact2), pure numpy.

The policy was fine-tuned on allenai/MolmoAct2-BimanualYAM-Dataset (LeRobot v3, `bi_yam_follower`) and its
openpi config (`yam_pi05`, openpi branch `yam-pi05`: ``openpi/policies/yam_policy.py``) takes

* ``state``  float32 (14,): ``[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]``, joints in
  radians, grippers in [0, 1] with **1 = open** (dataset: every episode starts at 0.999, dips to ~0.06 in a grasp);
* ``images``: ``{"top", "left", "right"}`` uint8 HWC, 360x640 (16:9) at training time;
* ``prompt``;

and returns ``actions`` (16, 14) in the same layout, absolute joint targets at 30 Hz.

REALM's vocabulary is ``robot_state = [left(6), right(6)]``, ``gripper_state = [left, right]`` normalised
(0 open, 1 closed), and an action whose gripper columns 6 and 13 realm.rollout binarises with "open above 0.5"
(``GRIPPER_OPEN_ABOVE_HALF``). The policy's gripper convention is already REALM's open fraction, so the action
columns pass through unchanged; the state gripper is flipped. Images are cropped to 16:9 (REALM's wrist renders
are 4:3, 960x720 -- the D405's 640x360 mode is a vertical crop of its 4:3 sensor, so a centre crop keeps the
training framing) and letterboxed to the model's 224x224 client-side, like ``_OpenPIAdapter`` does for DROID.
No omnigibson / torch imports so tests/test_yam_robot.py can pin it on the host.
"""

import numpy as np

from realm.robots.yam import YamBimanualRobot as _B

ARM_DOF = _B.ARM_DOF
STATE_DIM = _B.ACTION_DIM  # 14
GRIPPER_IDX = _B.GRIPPER_ACTION_IDX  # (6, 13)
IMAGE_KEYS = ("top", "left", "right")
POLICY_IMAGE_SIZE = (224, 224)
TRAIN_ASPECT = 16 / 9


def crop_to_aspect(im, aspect=TRAIN_ASPECT):
    """Centre-crop an (H, W, 3) image to width/height == aspect (no-op when it already is, or wider)."""
    im = np.asarray(im)
    h, w = im.shape[:2]
    target_h = int(round(w / aspect))
    if target_h >= h:
        return im
    top = (h - target_h) // 2
    return im[top:top + target_h]


def policy_state(robot_state, gripper_state):
    """``[left(6), right(6)]`` + REALM grippers (0 open, 1 closed) -> the policy's 14-D state (gripper 1 = open)."""
    robot_state = np.asarray(robot_state, dtype=np.float64)
    g = np.atleast_1d(np.asarray(gripper_state, dtype=np.float64))
    assert robot_state.shape == (2 * ARM_DOF,) and g.shape == (2,), (robot_state.shape, g.shape)
    open_fraction = np.clip(1.0 - g, 0.0, 1.0)
    state = np.concatenate([robot_state[:ARM_DOF], open_fraction[:1], robot_state[ARM_DOF:], open_fraction[1:]])
    return state.astype(np.float32)


def policy_observation(instruction, top_im, left_wrist_im, right_wrist_im, robot_state, gripper_state,
                       resize=None):
    """The dict sent to the `yam_pi05` server. `resize(im, h, w)` is openpi_client's resize_with_pad; images are
    cropped to 16:9 and letterboxed to 224x224 here when it is given, otherwise sent at REALM's resolution."""
    obs = {"prompt": instruction, "state": policy_state(robot_state, gripper_state), "images": {}}
    for key, im in zip(IMAGE_KEYS, (top_im, left_wrist_im, right_wrist_im)):
        im = np.asarray(im)
        assert im.ndim == 3 and im.shape[2] == 3, f"{key}: expected (H, W, 3), got {im.shape}"
        im = crop_to_aspect(im).astype(np.uint8, copy=False)
        if resize is not None:
            im = resize(im, *POLICY_IMAGE_SIZE)
        obs["images"][key] = im
    return obs


def policy_actions_to_realm(actions):
    """(n, 14) policy actions -> REALM actions: identical layout; the gripper columns are already open fractions
    (1 = open) for realm.rollout's 0.5 threshold, so this only validates and casts."""
    a = np.array(actions, dtype=np.float64, copy=True)
    if a.ndim == 1:
        a = a[None, :]
    assert a.shape[1] == STATE_DIM, f"expected (n, {STATE_DIM}) actions, got {a.shape}"
    for i in GRIPPER_IDX:
        a[:, i] = np.clip(a[:, i], 0.0, 1.0)
    return a
