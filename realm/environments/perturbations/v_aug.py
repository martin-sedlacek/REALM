"""V-AUG: visual augmentation of rendered observations (blur + contrast).

Unlike the other perturbations V-AUG does not alter the scene, so it has no scene-setup
function -- RealmEnvironmentDynamic applies it to each observation instead.
"""
import cv2
import numpy as np
import torch

# Same import as realm/environments/env_base.py uses, and for the same reason: the module, not the
# package, so the environment side does not pull in the inference client's transport deps.
from realm.inference.utils import wrist_camera_obs_key

#: THE canonical V-AUG draw ranges -- sigma for the Gaussian blur (0 = no blur), alpha for the
#: contrast multiplier. Defined once here and imported by env_dynamic.py's per-reset draw.
#:
#: Canonicalised 2026-08-19 in the versioned number-moving batch (VERSION 1.0.0). Three sites used
#: to disagree: env_dynamic's construction-time draw said (0-3.0, 0.5-2.0), its per-reset draw said
#: (0-2.5, 0.25-1.5), and this module's None-fallback said (0-3.0, 0.25-1.5). The per-reset range
#: wins because it is the only one that ever reached a rendered observation: the construction-time
#: values were always overwritten by the per-reset draw before the first distortion, and the
#: fallback only fires when this function is called standalone. So the canonical range is exactly
#: what every recorded V-AUG rollout was actually produced with -- but the construction-time draw
#: itself is now REMOVED, which shifts the shared RNG stream for V-AUG-active runs: V-AUG numbers
#: recorded before 1.0.0 are not comparable. See CHANGE_LEDGER.md.
SIGMA_RANGE = (0.0, 2.5)
ALPHA_RANGE = (0.25, 1.5)


def apply_blur_and_contrast(obs, sigma=None, alpha=None, robot_name='DROID'):
    # Standalone-call fallback: env_dynamic always passes the values it drew at reset from the
    # same canonical ranges.
    if sigma is None:
        sigma = np.random.uniform(*SIGMA_RANGE)
    if alpha is None:
        alpha = np.random.uniform(*ALPHA_RANGE)

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

    # The wrist observation key is <robot>:<link>:Camera:<idx>, and BOTH halves depend on the robot:
    # obs is keyed by robot.name, while the profile resolves the RoboLab wrist link and camera index.
    # This used to hardcode robot_name='DROID' and that link, and all three call sites in
    # env_dynamic.py omitted robot_name -- so on the robolab assets (DROID_robolab /
    # DROID_mounted, the default robot for every eval since 2026-08-13) the lookup was
    # obs['DROID'], which does not exist, and V-AUG died with a KeyError inside reset(). Resolve
    # the key from the robot profile instead, exactly as inference/utils.extract_from_obs does, so
    # the two cannot disagree about which image the policy sees.
    wrist_key = wrist_camera_obs_key(robot_name)
    robot_obs = obs.get(robot_name, {})
    if wrist_key not in robot_obs:
        # Mirrors extract_from_obs's fallback: any camera on the robot beats no augmentation, and a
        # robot with no wrist camera at all (config/robots/DROID_no_wrist_cam.yaml) is a legitimate
        # configuration -- augmenting only the external views is correct there, not an error.
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
