import numpy as np


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    # Fallback to zeros if external sensors are missing (e.g. during no_render)
    if 'external' in obs and 'external_sensor0' in obs['external']:
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        # Dummy 128x128 image
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
        base_depth = np.zeros((128, 128), dtype=np.float32) if enable_depth else None

    if 'external' in obs and 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    # Handle wrist camera (name can be dynamic based on robot)
    wrist_cam_key = 'DROID:gripper_link_camera:Camera:0'
    if robot_name in obs and wrist_cam_key in obs[robot_name]:
        wrist_im = obs[robot_name][wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
    else:
        wrist_im = np.zeros((128, 128, 3), dtype=np.uint8)

    # Proprio is always present in DROID and other robots
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
