import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import omnigibson as og
from omnigibson import log
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_category_models
import yaml
import os

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


def add_rotation_noise(current_orientation_quat, noise_std_dev_rad_xyz, min_xyz=None, max_xyz=None, noise_mean=(0,0,0)):
    current_rot = Rotation.from_quat(current_orientation_quat)
    current_euler_xyz = current_rot.as_euler('xyz', degrees=False)
    noise_euler_xyz = np.random.normal(loc=noise_mean, scale=noise_std_dev_rad_xyz)
    new_euler_xyz = current_euler_xyz + noise_euler_xyz
    if min_xyz is not None and max_xyz is not None:
        new_euler_xyz = np.clip(new_euler_xyz, a_min=min_xyz, a_max=max_xyz)
    new_rot = Rotation.from_euler('xyz', new_euler_xyz, degrees=False)
    return new_rot.as_quat()


def apply_blur_and_contrast(obs, sigma=None, alpha=None):
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

    wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb']
    obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'][..., :3] = torch.tensor(
        apply_random_image_augmentations(
            wrist_im.cpu().numpy()[..., :3].astype(np.float32)
        )
    ).to(wrist_im.device)
    return obs


def compute_rot_diff_magnitude(initial_quat,final_quat):
    r_initial = Rotation.from_quat(initial_quat)
    r_final = Rotation.from_quat(final_quat)
    r_diff = r_final * r_initial.inv()
    rotvec = r_diff.as_rotvec()
    return rotvec[2]

def _load_categories_from_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "config/objects/categories.yaml")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

_CATEGORIES_DATA = None

def _get_categories_data():
    global _CATEGORIES_DATA
    if _CATEGORIES_DATA is None:
        _CATEGORIES_DATA = _load_categories_from_yaml()
    return _CATEGORIES_DATA

def get_non_droid_categories():
    return _get_categories_data()["non_droid_categories"]

def get_droid_categories_by_theme():
    return _get_categories_data()["droid_categories_by_theme"]


def get_objects_by_names(scene: InteractiveTraversableScene, names: list[str]) -> list[DatasetObject]:
    objects = []
    for obj in scene.objects:
        obj: DatasetObject
        if obj.name in names:
            objects.append(obj)
    return objects


def get_default_objects_cfg(scene: InteractiveTraversableScene, object_names: list[str]) -> dict[str, dict]:
    objects = get_objects_by_names(scene, object_names)
    cfgs = {}
    for obj in objects:
        this_cfg = {
            "category": obj.category,
            "pos": obj.aabb_center,
            "ori": obj.get_position_orientation()[1],
            "relative_prim_path": obj._relative_prim_path
        }

        far_pos = np.random.random((3,)) * 3 + np.array([0, 0, 20])
        obj.set_position(far_pos)
        obj.set_orientation([0, 0, 0, 1])
        og.sim.step()
        this_cfg["bounding_box"] = obj.aabb_extent

        obj.set_position_orientation(this_cfg["pos"], this_cfg["ori"])

        cfgs[obj.name] = this_cfg

    return cfgs


def find_and_remove_category(categories_dict, obj_category):
    for theme, sub_categories in categories_dict.items():
        for category, obj_list in sub_categories.items():
            if obj_category in obj_list:
                return theme
    return None


def process_droid_categories(original_dict, obj_category):
    processed_dict = original_dict.copy()

    theme_to_pop = find_and_remove_category(processed_dict, obj_category)

    if theme_to_pop:
        processed_dict.pop(theme_to_pop)

    flattened_list = []
    for sub_categories in processed_dict.values():
        for obj_list in sub_categories.values():
            flattened_list.extend(obj_list)

    return flattened_list


def get_non_colliding_positions_for_objects_v2(
        xmin, xmax, ymin, ymax, z, obj_cfg,
        main_object_names,
        min_separation=0.05,
        max_attempts_per_object=2500,
        seed=None,
        objects_to_skip=None,
        maximum_dim=0.12
):
    placed_objects_info = []
    objects_to_randomly_place = []
    if objects_to_skip is None:
        objects_to_skip = []

    # First pass: Identify main object, process skipped distractors, and collect other objects
    for i, cfg in enumerate(obj_cfg):
        if cfg["name"] in main_object_names:
            half_width_main = cfg["bounding_box"][0] / 2
            half_depth_main = cfg["bounding_box"][1] / 2
            x_center_main = cfg["position"][0]
            y_center_main = cfg["position"][1]
            placed_objects_info.append((x_center_main, y_center_main, half_width_main, half_depth_main))
            continue
        elif cfg["name"] in objects_to_skip:
            # These distractors are considered pre-placed at their existing positions
            if "bounding_box" not in cfg:
                # Assume a default bounding box if not specified
                cfg["bounding_box"] = [0.08, 0.08, 0.08]
            else:
                max_dim = np.max(np.array(cfg["bounding_box"]))
                new_scale_factor = maximum_dim / max_dim
                if new_scale_factor < 1.0:
                    #new_obj.scale = new_scale_factor  # TODO: explain method code in comments
                    cfg["bounding_box"] = np.array(cfg["bounding_box"]) * new_scale_factor

            # Ensure position exists for skipped distractors
            if "position" not in cfg or len(cfg["position"]) < 2:
                og.log.warn(f"Warning: Skipped distractor '{cfg['name']}' does not have a valid 'position' field. Skipping placement.")
                continue # Skip this distractor if position is invalid

            placed_objects_info.append((
                cfg["position"][0],
                cfg["position"][1],
                cfg["bounding_box"][0] / 2, # Corrected: Access width
                cfg["bounding_box"][1] / 2  # Corrected: Access depth
            ))
        else:
            # These objects will be placed randomly later
            objects_to_randomly_place.append((cfg, i))

    # --- Now, shuffle and place the remaining objects randomly ---
    # Shuffle the list of objects that need random placement
    np.random.shuffle(objects_to_randomly_place)

    for cfg, original_idx in objects_to_randomly_place:
        if "bounding_box" not in cfg:
            cfg["bounding_box"] = [0.08, 0.08, 0.08] # Default if not present

        bbox = cfg["bounding_box"]
        # Corrected: Access specific elements of bounding_box list
        half_width = bbox[0] / 2
        half_depth = bbox[1] / 2
        placed = False

        for _ in range(max_attempts_per_object):
            x_center = np.random.uniform(xmin + half_width, xmax - half_width)
            y_center = np.random.uniform(ymin + half_depth, ymax - half_depth)

            collision = False
            for px, py, phw, phd in placed_objects_info:
                dist_x = abs(x_center - px)
                dist_y = abs(y_center - py)

                # Check for collision with existing objects, considering min_separation
                if dist_x < (half_width + phw + min_separation) and \
                        dist_y < (half_depth + phd + min_separation):
                    collision = True
                    break

            if not collision:
                # If no collision, place the object
                placed_objects_info.append((x_center, y_center, half_width, half_depth))
                # Update the position in the original obj_cfg list using its original index
                obj_cfg[original_idx]["position"] = [x_center, y_center, z]
                placed = True
                break

        if not placed:
            og.log.error(f"Failed to place object '{cfg.get('name', 'Unnamed Object')}' after {max_attempts_per_object} attempts. Dropping it from the air.")
            x_center = np.random.uniform(xmin + half_width, xmax - half_width)
            y_center = np.random.uniform(ymin + half_depth, ymax - half_depth)
            obj_cfg[original_idx]["position"] = [x_center, y_center, z + 0.1]


    return obj_cfg
