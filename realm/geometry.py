"""Pure pose/rotation math shared across REALM.

Deliberately free of OmniGibson, OpenCV and YAML imports so the transform helpers can be
exercised without booting a simulator. Poses are (x, y, z, roll, pitch, yaw) unless a name
says otherwise; quaternions are xyzw to match scipy and OmniGibson.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw):
    """
    Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix.
    """
    # scipy's Rotation.from_quat expects [x, y, z, w]
    r = Rotation.from_quat(quaternion_xyzw)
    return r.as_matrix()


def rotation_matrix_to_quaternion_xyzw(rot_matrix):
    r = Rotation.from_matrix(rot_matrix)
    return r.as_quat().tolist() # Returns [x, y, z, w]


def rpy_radians_to_rotation_matrix(rpy_radians, order='xyz'):
    r = Rotation.from_euler(order, rpy_radians, degrees=False)
    return r.as_matrix()


def create_homogeneous_transform_from_quaternion(translation_xyz, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
    T[:3, 3] = translation_xyz
    return T


def create_homogeneous_transform_from_rpy(translation_xyz, rpy_radians, order='xyz'):
    T = np.eye(4)
    T[:3, :3] = rpy_radians_to_rotation_matrix(rpy_radians, order=order)
    T[:3, 3] = translation_xyz
    return T


def get_xyz_quaternion_from_homogeneous_transform(T_matrix):
    translation_xyz = T_matrix[:3, 3].tolist()
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(T_matrix[:3, :3])
    return translation_xyz, quaternion_xyzw


### Subtractions ###
def quat_diff(target, source):
    result = Rotation.from_quat(target) * Rotation.from_quat(source).inv()
    return result.as_quat()


def angle_diff(target, source, degrees=False):
    target_rot = Rotation.from_euler("xyz", target, degrees=degrees)
    source_rot = Rotation.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[:3]) - np.array(source[:3])
    rot_diff = angle_diff(target[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff])
    return result


### Additions ###
def add_quats(delta, source):
    result = Rotation.from_quat(delta) * Rotation.from_quat(source)
    return result.as_quat()


def add_angles(delta, source, degrees=False):
    delta_rot = Rotation.from_euler("xyz", delta, degrees=degrees)
    source_rot = Rotation.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


def add_poses(delta, source, degrees=False):
    lin_sum = np.array(delta[:3]) + np.array(source[:3])
    rot_sum = add_angles(delta[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_sum, rot_sum])
    return result


def robot_to_world(action, robot_pos, robot_yaw, base_height=0.0):
    """Convert a 7D EE action (xyz + RPY + gripper) from robot-local to world frame."""
    assert action.shape[-1] == 7
    action = action.copy()
    cos_y, sin_y = np.cos(robot_yaw), np.sin(robot_yaw)
    x_rel, y_rel = action[0], action[1]
    action[0] = cos_y * x_rel - sin_y * y_rel + robot_pos[0]
    action[1] = sin_y * x_rel + cos_y * y_rel + robot_pos[1]
    action[2] = action[2] + robot_pos[2] + base_height
    R_base = Rotation.from_euler('z', robot_yaw)
    R_pred = Rotation.from_euler('xyz', action[3:6])
    action[3:6] = (R_base * R_pred).as_euler('xyz')
    return action


def world_to_robot(action, robot_pos, robot_yaw, base_height=0.0):
    """Convert a 7D EE action (xyz + RPY + gripper) from world frame to robot-local frame."""
    action = action.copy()
    cos_y, sin_y = np.cos(robot_yaw), np.sin(robot_yaw)
    dx = action[0] - robot_pos[0]
    dy = action[1] - robot_pos[1]
    action[0] = cos_y * dx + sin_y * dy
    action[1] = -sin_y * dx + cos_y * dy
    action[2] = action[2] - robot_pos[2] - base_height
    R_base_inv = Rotation.from_euler('z', robot_yaw).inv()
    R_world = Rotation.from_euler('xyz', action[3:6])
    action[3:6] = (R_base_inv * R_world).as_euler('xyz')
    return action


def axisangle_to_rpy(action):
    """Convert rotation in an EE action from axis-angle to RPY (euler xyz).
    Works for a single action (..., 7) or a chunk (..., T, 7).
    """
    action = action.copy()
    action[..., 3:6] = Rotation.from_rotvec(action[..., 3:6]).as_euler('xyz')
    return action


def flip_pose_pointing_down(rpy_vec):
    r_old = Rotation.from_euler('xyz', rpy_vec)
    flip = Rotation.from_euler('xyz', [torch.pi, 0, 0])
    r_new = r_old * flip
    return r_new.as_euler('xyz')


def compute_rot_diff_magnitude(initial_quat,final_quat):
    r_initial = Rotation.from_quat(initial_quat)
    r_final = Rotation.from_quat(final_quat)
    r_diff = r_final * r_initial.inv()
    rotvec = r_diff.as_rotvec()
    return rotvec[2]


def add_rotation_noise(current_orientation_quat, noise_std_dev_rad_xyz, min_xyz=None, max_xyz=None, noise_mean=(0,0,0)):
    current_rot = Rotation.from_quat(current_orientation_quat)
    current_euler_xyz = current_rot.as_euler('xyz', degrees=False)
    noise_euler_xyz = np.random.normal(loc=noise_mean, scale=noise_std_dev_rad_xyz)
    new_euler_xyz = current_euler_xyz + noise_euler_xyz
    if min_xyz is not None and max_xyz is not None:
        new_euler_xyz = np.clip(new_euler_xyz, a_min=min_xyz, a_max=max_xyz)
    new_rot = Rotation.from_euler('xyz', new_euler_xyz, degrees=False)
    return new_rot.as_quat()


def calculate_new_camera_pose_mixed_rotations(
    camera_relative_to_base_xyz, camera_relative_to_base_quat_xyzw,
    new_base_pose_xyz, new_base_pose_rpy_rad):

    # 1. Create the camera's relative transformation matrix (T_base_camera) from XYZ and Quaternion
    T_base_camera = create_homogeneous_transform_from_quaternion(
        camera_relative_to_base_xyz,
        camera_relative_to_base_quat_xyzw
    )

    # 2. Create the new robot base's absolute transformation matrix (T_world_new_base) from XYZ and RPY
    T_world_new_base = create_homogeneous_transform_from_rpy(
        new_base_pose_xyz,
        new_base_pose_rpy_rad,
        order='xyz' # Assuming RPY rotation order for your robot base
    )

    # 3. Calculate the new absolute camera transformation matrix (T_world_new_camera)
    T_world_new_camera = T_world_new_base.dot(T_base_camera)

    # 4. Extract the new camera's XYZ and Quaternion
    new_camera_xyz, new_camera_quat_xyzw = get_xyz_quaternion_from_homogeneous_transform(
        T_world_new_camera
    )

    return new_camera_xyz, new_camera_quat_xyzw
