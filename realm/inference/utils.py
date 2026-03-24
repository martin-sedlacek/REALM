import numpy as np


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    if 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    wrist_im = obs[robot_name]['DROID:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
