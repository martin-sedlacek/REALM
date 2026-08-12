"""V-AUG: visual augmentation of rendered observations (blur + contrast).

Unlike the other perturbations V-AUG does not alter the scene, so it has no scene-setup
function -- RealmEnvironmentDynamic applies it to each observation instead.
"""
import cv2
import numpy as np
import torch


def apply_blur_and_contrast(obs, sigma=None, alpha=None, robot_name='DROID'):
    # 1. Random Gaussian Blur
    # Sigma for Gaussian blur: 0 (no blur) to 3.0 (moderate blur)
    if sigma is None:
        sigma = np.random.uniform(0.0, 3.0)

    # 2. Random Contrast Change
    # Contrast factor (alpha): 0.25 (lower contrast) to 1.5 (higher contrast)
    if alpha is None:
        alpha = np.random.uniform(0.25, 1.5)

    def apply_random_image_augmentations(image_float):
        # ksize (kernel size) should be positive and odd. If 0, it's computed from sigma.
        # Let's compute it from sigma for simplicity, ensuring it's odd and at least 1.
        ksize_val = int(sigma * 4 + 1)  # A common heuristic for ksize based on sigma
        if ksize_val % 2 == 0:
            ksize_val += 1

        # Ensure ksize is at least 1 if sigma is very small
        ksize_val = max(1, ksize_val)
        blurred_image = cv2.GaussianBlur(image_float, (ksize_val, ksize_val), sigma)

        # Apply contrast change: new_pixel = alpha * old_pixel
        # Clamp values to [0, 255] for uint8 output
        contrasted_image = np.clip(blurred_image * alpha, 0, 255)

        return contrasted_image.astype(np.uint8)

    for base_cam in list(obs['external'].keys()):
        base_im = obs['external'][base_cam]['rgb'] #obs['external']['external_sensor0']
        #obs['external']['external_sensor0']['rgb']
        obs['external'][base_cam]['rgb'][..., :3] = torch.tensor(
            apply_random_image_augmentations(
                base_im.cpu().numpy()[..., :3].astype(np.float32)
            )
        ).to(base_im.device)

    # TODO: this will only work for DORID dict structure right now:
    wrist_im = obs[robot_name][f'{robot_name}:gripper_link_camera:Camera:0']['rgb']
    obs[robot_name][f'{robot_name}:gripper_link_camera:Camera:0']['rgb'][..., :3] = torch.tensor(
        apply_random_image_augmentations(
            wrist_im.cpu().numpy()[..., :3].astype(np.float32)
        )
    ).to(wrist_im.device)
    return obs
