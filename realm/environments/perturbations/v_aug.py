
import cv2
import numpy as np
import torch

from realm.inference.utils import wrist_camera_obs_key

SIGMA_RANGE = (0.0, 2.5)
ALPHA_RANGE = (0.25, 1.5)


def apply_blur_and_contrast(obs, sigma=None, alpha=None, robot_name='DROID'):
    if sigma is None:
        sigma = np.random.uniform(*SIGMA_RANGE)
    if alpha is None:
        alpha = np.random.uniform(*ALPHA_RANGE)

    def apply_random_image_augmentations(image_float):
        ksize_val = int(sigma * 4 + 1)
        if ksize_val % 2 == 0:
            ksize_val += 1

        ksize_val = max(1, ksize_val)
        blurred_image = cv2.GaussianBlur(image_float, (ksize_val, ksize_val), sigma)

        contrasted_image = np.clip(blurred_image * alpha, 0, 255)

        return contrasted_image.astype(np.uint8)

    for base_cam in list(obs['external'].keys()):
        base_im = obs['external'][base_cam]['rgb']
        obs['external'][base_cam]['rgb'][..., :3] = torch.tensor(
            apply_random_image_augmentations(
                base_im.cpu().numpy()[..., :3].astype(np.float32)
            )
        ).to(base_im.device)

    wrist_key = wrist_camera_obs_key(robot_name)
    robot_obs = obs.get(robot_name, {})
    if wrist_key not in robot_obs:
        cam_keys = [k for k in robot_obs if ":Camera:" in k]
        if not cam_keys:
            print(f"[V-AUG] WARNING: no camera on '{robot_name}' in obs; augmenting the external "
                  f"views only.")
            return obs
        print(f"[V-AUG] WARNING: no '{wrist_key}' in obs; augmenting '{cam_keys[0]}' instead. "
              f"Update ROBOT_OBS_PROFILES for '{robot_name}'.")
        wrist_key = cam_keys[0]

    wrist_im = robot_obs[wrist_key]['rgb']
    robot_obs[wrist_key]['rgb'][..., :3] = torch.tensor(
        apply_random_image_augmentations(
            wrist_im.cpu().numpy()[..., :3].astype(np.float32)
        )
    ).to(wrist_im.device)
    return obs
