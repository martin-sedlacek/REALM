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



def plot_state_comparisons(qpos, achieved_states, commanded_delta, num_timesteps, output_filename="comparison_plot.png"):
    # Validate input shapes
    if qpos.shape != (num_timesteps, 7) or achieved_states.shape != (num_timesteps, 7):
        raise ValueError(f"Expected qpos and achieved_states to have shape ({num_timesteps}, 7), "
                         f"but got {qpos.shape} and {achieved_states.shape} respectively.")

    time = np.arange(num_timesteps)  # Time axis from 0 to t-1

    # Create a figure with 7 subplots, arranged vertically
    # Adjust figsize as needed for better visualization
    fig, axes = plt.subplots(7, 1, figsize=(12, 22), sharex=True) # Increased figsize for better readability
    fig.suptitle('Comparison of qpos and achieved_states Over Time per Dimension', fontsize=18, y=0.995) # Adjusted y for suptitle

    for i in range(7):  # Iterate through each of the 7 dimensions
        ax = axes[i]

        # Plot qpos for the current dimension
        ax.plot(time, qpos[:, i] * 57.2958, label=f'real qpos', linestyle='-', marker='o', markersize=4, alpha=0.8)

        # Plot achieved_states for the current dimension
        ax.plot(time, achieved_states[:, i] * 57.2958, label=f'sim gpos', linestyle='--',  marker='x', markersize=4, alpha=0.8)

        # Plot achieved_states for the current dimension
        ax.plot(time, commanded_delta[:, i] * 57.2958, label=f'commanded qpos', linestyle='--',  marker='x', markersize=4, alpha=0.8)


        # Set title and labels
        ax.set_title(f'Dimension {i+1}', fontsize=12)
        ax.set_ylabel('Value', fontsize=10)
        #ax.set_ylim(-2.5, 2.5)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7) # Add a subtle grid for easier reading

    # Common X-axis label
    axes[-1].set_xlabel('Time Step', fontsize=12)

    # Adjust layout to prevent overlapping titles/labels
    #plt.axis('scaled')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to make space for suptitle and x-label

    # Save the plot to a file
    try:
        fig.savefig(output_filename, dpi=300, bbox_inches='tight') # Save with good resolution and tight bounding box
        log.info(f"Plot successfully saved to {output_filename}")
    except Exception as e:
        log.info(f"Error saving plot to {output_filename}: {e}")
    finally:
        plt.close(fig) # Close the figure to free up memory


def _apply_jet_colormap_numpy(normalized_values: np.ndarray) -> np.ndarray:
    # Create an empty array for the RGB image
    rgb_image = np.zeros((*normalized_values.shape, 3))

    # Define the key points of the jet colormap
    # The colormap transitions from blue -> cyan -> yellow -> red
    key_points = np.array([0, 0.33, 0.66, 1.0])
    colors = np.array([
        [0, 0, 1],  # Blue
        [0, 1, 1],  # Cyan
        [1, 1, 0],  # Yellow
        [1, 0, 0]  # Red
    ])

    # Apply the colormap by interpolating between the key points
    for i in range(len(key_points) - 1):
        # Create a mask for the current segment
        mask = (normalized_values >= key_points[i]) & (normalized_values < key_points[i + 1])

        # Linearly interpolate the values within the segment
        interp_vals = (normalized_values[mask] - key_points[i]) / (key_points[i + 1] - key_points[i])

        # Interpolate the colors for each channel
        for c in range(3):
            rgb_image[mask, c] = colors[i, c] + interp_vals * (colors[i + 1, c] - colors[i, c])

    # Handle the final key point separately to include the max value
    mask_last = normalized_values >= key_points[-1]
    rgb_image[mask_last] = colors[-1]

    return rgb_image


def create_heatmap_overlay_numpy(
    attention_grid: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    if attention_grid.ndim != 2:
        raise ValueError("attention_grid must be a 2D array.")
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("original_image must be an RGB image of shape (H, W, 3).")

    # 1. Upscale the attention grid to the size of the original image
    # We use the Kronecker product for a simple block-based upscaling (nearest neighbor).
    h_scale = original_image.shape[0] // attention_grid.shape[0]
    w_scale = original_image.shape[1] // attention_grid.shape[1]
    heatmap = np.kron(attention_grid, np.ones((h_scale, w_scale)))

    # Ensure the upscaled heatmap exactly matches the image dimensions
    if heatmap.shape != original_image.shape[:2]:
        # Simple resize if kron doesn't match perfectly due to non-integer scaling
        from PIL import Image
        heatmap_img = Image.fromarray(heatmap)
        heatmap_resized = heatmap_img.resize(original_image.shape[1::-1], Image.NEAREST)
        heatmap = np.array(heatmap_resized)

    # 2. Normalize heatmap to the range [0, 1] for colormapping
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    if max_val == min_val:
        normalized_heatmap = np.zeros_like(heatmap, dtype=np.float32)
    else:
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)

    # 3. Apply the colormap to the normalized heatmap
    colored_heatmap = _apply_jet_colormap_numpy(normalized_heatmap)

    # 4. Blend the colormap with the original image
    # Convert original image to float [0, 1] for blending
    original_image_float = original_image.astype(np.float32) / 255.0

    # Apply alpha blending
    blended_image_float = (alpha * colored_heatmap) + ((1 - alpha) * original_image_float)

    # Convert back to uint8 [0, 255]
    final_image = (np.clip(blended_image_float, 0, 1) * 255).astype(np.uint8)

    return final_image

#--------------------------------------------------------------------------------

def droid_matrix_to_sapien_matrix(mat: np.ndarray) -> np.ndarray:
    assert mat.shape == (3, 3), mat.shape
    res = np.zeros_like(mat)
    res[:, 0] = mat[:, 2]
    res[:, 1] = -mat[:, 0]
    res[:, 2] = -mat[:, 1]

    return res

#--------------------------------------------------------------------------------


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


def rotation_matrix_to_rpy_radians(rot_matrix, order='xyz'):
    r = Rotation.from_matrix(rot_matrix)
    return r.as_euler(order, degrees=False).tolist()


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
    print("Camera Relative to Base (T_base_camera):\n", T_base_camera)

    # 2. Create the new robot base's absolute transformation matrix (T_world_new_base) from XYZ and RPY
    T_world_new_base = create_homogeneous_transform_from_rpy(
        new_base_pose_xyz,
        new_base_pose_rpy_rad,
        order='xyz' # Assuming RPY rotation order for your robot base
    )
    print("\nNew Robot Base Pose (T_world_new_base):\n", T_world_new_base)

    # 3. Calculate the new absolute camera transformation matrix (T_world_new_camera)
    T_world_new_camera = T_world_new_base.dot(T_base_camera)
    print("\nNew Absolute Camera Pose (T_world_new_camera):\n", T_world_new_camera)

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


def get_non_colliding_positions_for_objects(xmin, xmax, ymin, ymax, z, obj_cfg,
                                            main_object_names,
                                            min_separation=0.05, max_attempts_per_object=10000, seed=None,
                                            objects_to_skip=None):
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
            # Ensure position exists for skipped distractors
            if "position" not in cfg or len(cfg["position"]) < 2:
                print(f"Warning: Skipped distractor '{cfg['name']}' does not have a valid 'position' field. Skipping placement.")
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
            # If an object cannot be placed after max_attempts, return an empty list
            print(f"Failed to place object '{cfg.get('name', 'Unnamed Object')}' after {max_attempts_per_object} attempts.")
            return None

    return obj_cfg


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
    # print(initial_quat)
    # print(final_quat)
    # r_initial = Rotation.from_quat(initial_quat)
    # r_final = Rotation.from_quat(final_quat)
    # r_diff = r_final * r_initial.inv()
    # rotvec = r_diff.as_rotvec()
    # overall_angle_rad = np.linalg.norm(rotvec)
    # return overall_angle_rad
    r_initial = Rotation.from_quat(initial_quat)
    r_final = Rotation.from_quat(final_quat)
    r_diff = r_final * r_initial.inv()
    rotvec = r_diff.as_rotvec()
    return rotvec[2]

def get_non_droid_categories():
    return [
        "acorn",
        "address",
        "agave",
        "air_filter",
        "alarm_clock",
        "allspice_shaker",
        "aluminum_foil",
        "apple_pie",
        "apricot",
        "apron",
        "arbor",
        "architectural_plan",
        "arepa",
        "artichoke",
        "arugula",
        "ashtray",
        "auricularia",
        "baby_bottle",
        "baby_shoe",
        "backpack",
        "bacon",
        "bag_of_auricularia",
        "bag_of_breadcrumbs",
        "bag_of_brown_rice",
        "bag_of_cookies",
        "bag_of_cream_cheese",
        "bag_of_fertilizer",
        "bag_of_flour",
        "bag_of_ice_cream",
        "bag_of_jerky",
        "bag_of_mulch",
        "bag_of_oranges",
        "bag_of_popcorn",
        "bag_of_rice",
        "bag_of_rubbish",
        "bag_of_shiitake",
        "bag_of_snacks",
        "bag_of_starch",
        "bag_of_tea",
        "bag_of_yeast",
        "bagel",
        "baguette",
        "baking_powder_jar",
        "baking_sheet",
        "balloon",
        "banana_bread",
        "bandage",
        "bandana",
        "bar_soap",
        "basil",
        "basil_jar",
        "bath_rug",
        "bath_towel",
        "battery",
        "batting_gloves",
        "bay_leaf",
        "beach_toy",
        "bean_curd",
        "beef_broth_carton",
        "beer_glass",
        "beer_keg",
        "beeswax_candle",
        "beet",
        "bell",
        "belt",
        "bicycle_chain",
        "bidet",
        "bikini",
        "binder",
        "bird_feed_bag",
        "biscuit",
        "biscuit_dough",
        "blackberry",
        "bleu",
        "board_game",
        "bobby_pin",
        "bok_choy",
        "bouillon_cube",
        "bouquet",
        "bow",
        "bowling_ball",
        "box_of_almond_milk",
        "box_of_aluminium_foil",
        "box_of_apple_juice",
        "box_of_baking_mix",
        "box_of_baking_powder",
        "box_of_baking_soda",
        "box_of_barley",
        "box_of_beer",
        "box_of_brown_sugar",
        "box_of_butter",
        "box_of_candy",
        "box_of_cane_sugar",
        "box_of_champagne",
        "box_of_chocolates",
        "box_of_coconut_milk",
        "box_of_coffee",
        "box_of_cookies",
        "box_of_corn_flakes",
        "box_of_crackers",
        "box_of_cream",
        "box_of_flour",
        "box_of_fruit",
        "box_of_granola_bars",
        "box_of_ice_cream",
        "box_of_lasagna",
        "box_of_lemons",
        "box_of_milk",
        "box_of_oatmeal",
        "box_of_raspberries",
        "box_of_rice",
        "box_of_rum",
        "box_of_sake",
        "box_of_salt",
        "box_of_sanitary_napkins",
        "box_of_shampoo",
        "box_of_takeout",
        "box_of_tissues",
        "box_of_tomato_juice",
        "box_of_vegetable_juice",
        "box_of_whiskey",
        "box_of_yogurt",
        "boxed_cake",
        "boxed_cpu_board",
        "boxed_ink_cartridge",
        "boxed_router",
        "boxers",
        "boxing_gloves",
        "bra",
        "bracelet",
        "branch",
        "bratwurst",
        "bread_slice",
        "briefcase",
        "brisket",
        "broccoli_rabe",
        "broccolini",
        "broken_glass",
        "broken_light_bulb",
        "brown_rice_sack",
        "brown_sugar_sack",
        "brownie",
        "brussels_sprouts",
        "bucket_of_paint",
        "bulldog_clip",
        "bunch_of_bananas",
        "buret",
        "buret_clamp",
        "burlap_bag",
        "burner",
        "burrito",
        "butter",
        "butter_cookie",
        "butter_package",
        "buttermilk_pancake",
        "butternut_squash",
        "calculator",
        "caliper",
        "camera_tripod",
        "candied_yam",
        "candle_holder",
        "candy_cane",
        "canister",
        "cantaloup",
        "canvas",
        "cap",
        "car",
        "cardigan",
        "cardstock",
        "carton",
        "carton_of_eggs",
        "carton_of_milk",
        "carton_of_orange_juice",
        "carton_of_pineapple_juice",
        "carton_of_soy_milk",
        "case_of_eyeshadow",
        "cast",
        "cat_food_tin",
        "cauliflower",
        "cayenne_shaker",
        "cd",
        "celery",
        "cell_phone",
        "cellulose_tape",
        "centerpiece",
        "ceramic_tile",
        "chalice",
        "chard",
        "charger",
        "cheddar",
        "cheese_danish",
        "cheese_tart",
        "cheesecake",
        "cherry",
        "chess_set",
        "chia_seed_bag",
        "chicken",
        "chicken_breast",
        "chicken_broth_carton",
        "chicken_coop",
        "chicken_leg",
        "chicken_soup_carton",
        "chicken_tender",
        "chicken_thigh",
        "chicken_wing",
        "chicken_wire",
        "chickpea_can",
        "chili",
        "china",
        "chip",
        "chisel",
        "chives",
        "chlorine_bottle",
        "chocolate_bar",
        "chocolate_biscuit",
        "chocolate_cake",
        "chocolate_chip_cookie",
        "chocolate_cookie_dough",
        "chopped_lettuce",
        "chorizo",
        "cigar",
        "cigarette",
        "cilantro",
        "cinnamon_roll",
        "cinnamon_shaker",
        "cinnamon_stick",
        "clam",
        "clamp",
        "cleansing_bottle",
        "cleaver",
        "clipper",
        "cloche",
        "clout_nail",
        "clove_jar",
        "club_sandwich",
        "coaster",
        "cocktail_glass",
        "cocoa_box",
        "cocoa_powder_box",
        "coconut_fruit",
        "coffee_bean_jar",
        "cola_bottle",
        "colander",
        "cold_cuts",
        "colored_pencil",
        "comb",
        "comic_book",
        "conch",
        "cookie_cutter",
        "cookie_dough",
        "cookie_stick",
        "copper_pot",
        "copper_wire",
        "coriander_shaker",
        "cork",
        "corkscrew",
        "cornstarch_jar",
        "cotton_ball",
        "cotton_thread",
        "crab",
        "crawfish",
        "crayon",
        "cream_carton",
        "cream_cheese_box",
        "cream_of_tartar_shaker",
        "credit_card",
        "crock_pot",
        "croissant",
        "cruet",
        "crystal",
        "cubicle",
        "cucumber",
        "cumin_shaker",
        "cup_holder",
        "cup_of_ranch",
        "cup_of_yogurt",
        "cupcake",
        "curry_powder_shaker",
        "cymbal",
        "daffodil_bulb",
        "dahlia_flower",
        "dart",
        "date",
        "decanter",
        "dental_floss",
        "denture",
        "deodorant_stick",
        "desk_phone",
        "diamond",
        "diaper",
        "dice",
        "digital_camera",
        "digital_scale",
        "digital_thermometer",
        "dinner_napkin",
        "dip_candle",
        "dipper",
        "dishtowel",
        "dog_collar",
        "dog_food_can",
        "doily",
        "donut",
        "dowel",
        "dreidel",
        "dress",
        "dress_shirt",
        "dried_apricot",
        "drip_pot",
        "drumstick",
        "duck",
        "dumpling",
        "durian",
        "dustpan",
        "dvd",
        "easel",
        "easter_egg",
        "edible_cookie_dough",
        "eggplant",
        "electric_hand_mixer",
        "emery_paper",
        "enchilada",
        "envelope",
        "eyeglasses",
        "face_mask",
        "facsimile",
        "fairy_light",
        "farm_stand",
        "feather",
        "feta",
        "feta_box",
        "fillet",
        "firewood",
        "first_aid_kit",
        "flashlight",
        "flask_clamp",
        "floor_lamp",
        "floor_wax_bottle",
        "flower",
        "flower_petal",
        "folder",
        "folderal",
        "frail",
        "frame",
        "french_fries",
        "french_fry_holder",
        "french_toast",
        "frosting_jar",
        "fruitcake",
        "fuel_can",
        "funnel",
        "fur_coat",
        "garlic",
        "garlic_bread",
        "garlic_clove",
        "gelatin",
        "gelatin_box",
        "geode",
        "gift_box",
        "ginger_root",
        "ginger_shaker",
        "gingerbread",
        "glass_lantern",
        "glaze_bottle",
        "globe",
        "goalkeeper_gloves",
        "goblet",
        "goggles",
        "gourd",
        "graduated_cylinder",
        "grandfather_clock",
        "granola_bar",
        "granola_box",
        "granulated_sugar_jar",
        "granulated_sugar_sack",
        "grapefruit",
        "grated_cheese_sack",
        "grater",
        "gravy_boat",
        "green_bean",
        "green_onion",
        "griddle",
        "ground_beef_package",
        "gym_shoe",
        "hall_tree",
        "hammam_bench",
        "hand_towel",
        "handset",
        "hard_boiled_egg",
        "hard_candy",
        "hard_drive",
        "hardback",
        "hat",
        "hazelnut",
        "head_cabbage",
        "headset",
        "heatgun",
        "herbicide_bottle",
        "high_heel",
        "hiking_boot",
        "hitch",
        "hockey_puck",
        "honey_jar",
        "hourglass",
        "hummus_box",
        "hutch",
        "ice",
        "ice_bucket",
        "ice_cream_carton",
        "ice_cream_cone",
        "ice_cube",
        "ice_lolly",
        "inhaler",
        "ink_bottle",
        "instant_coffee_jar",
        "instant_pot",
        "iron",
        "jade",
        "jade_roller",
        "jeans",
        "jelly_bean_jar",
        "jersey",
        "jigger",
        "jigsaw_puzzle",
        "jigsaw_puzzle_piece",
        "jimmies_jar",
        "joystick",
        "kabob",
        "kale",
        "kebab",
        "kettle",
        "key",
        "key_chain",
        "keyboard",
        "keys",
        "kid_glove",
        "kielbasa",
        "kitchen_analog_scale",
        "knife_block",
        "lace",
        "laptop",
        "lasagna",
        "leaf",
        "leek",
        "legal_document",
        "legging",
        "lemon_peel",
        "lemon_pepper_seasoning_shaker",
        "lens",
        "leotard",
        "letter",
        "lettuce",
        "license_plate",
        "lid",
        "light_bulb",
        "lighter",
        "lingerie",
        "lint_roller",
        "lint_screen",
        "lip_balm",
        "lipstick",
        "liquid_carton",
        "lock",
        "log",
        "loin",
        "lollipop",
        "loudspeaker",
        "lunch_box",
        "mac_and_cheese",
        "macaron",
        "magazine",
        "magnetic_stirrer",
        "magnifying_glass",
        "mail",
        "maillot",
        "mallet",
        "mango",
        "map",
        "margarine_box",
        "marigold",
        "marinara_jar",
        "marjoram_shaker",
        "masking_tape",
        "massage_bed",
        "mat",
        "match",
        "match_box",
        "mattress",
        "measuring_cup",
        "meat_loaf",
        "meat_thermometer",
        "meatball",
        "menu",
        "microfiber_cloth",
        "microphone",
        "microscope",
        "milk_carton",
        "mint",
        "mixing_bowl",
        "money",
        "mop",
        "mousepad",
        "mousetrap",
        "mozzarella",
        "muffin",
        "mulch_bag",
        "mushroom",
        "music_stool",
        "mussel",
        "mustard_leaf",
        "name_tag",
        "national_flag",
        "nativity_figurine",
        "necklace",
        "necktie",
        "newspaper",
        "nickel",
        "nightstand",
        "noodle_jar",
        "notebook",
        "notepad",
        "oat_box",
        "oden_cooker",
        "okra",
        "ottoman",
        "outlet",
        "oyster",
        "pack_of_bread",
        "pack_of_chocolate_bar",
        "pack_of_cigarettes",
        "pack_of_ground_beef",
        "pack_of_kielbasa",
        "pack_of_pasta",
        "pack_of_protein_powder",
        "pack_of_ramen",
        "package",
        "packing_box",
        "paddle",
        "paint_roller",
        "paintbrush",
        "pallet",
        "paper_bag",
        "paper_clip",
        "paper_coffee_filter",
        "paper_cup",
        "paper_lantern",
        "paper_liners",
        "paper_towel",
        "paperback_book",
        "paraffin_wax",
        "parallel_bars",
        "parmesan_shaker",
        "parsley",
        "parsnip",
        "pasta_box",
        "pasta_server",
        "pastry",
        "pastry_cutter",
        "patty",
        "paving_stone",
        "pay_phone",
        "pea_pod",
        "peanut_nut",
        "pearl",
        "pebble",
        "pegboard",
        "pellet",
        "pencil_box",
        "pencil_case",
        "pencil_holder",
        "pennant",
        "penny",
        "pepper_grinder",
        "pepper_shaker",
        "peppermint",
        "peppermint_candy",
        "pepperoni",
        "periodic_table",
        "pestle",
        "petri_dish",
        "pewter_teapot",
        "pickle",
        "picture",
        "picture_frame",
        "pine_cone",
        "pita",
        "pitcher",
        "pizza",
        "pizza_box",
        "pizza_dough",
        "place_mat",
        "plant_pot",
        "plant_stem",
        "plastic_bag",
        "plastic_wrap",
        "plate",
        "platter",
        "plug",
        "plum",
        "plywood",
        "poinsettia",
        "pool_ball",
        "pop_case",
        "popcorn_bag",
        "pork",
        "pork_chop",
        "pork_rib",
        "portafilter",
        "post_it",
        "postage_stamp",
        "postcard",
        "poster",
        "poster_roll",
        "power_strip",
        "pretzel",
        "price_tag",
        "prosciutto",
        "pumpkin",
        "pumpkin_seed_bag",
        "quail_breast",
        "quail_breast_raw",
        "quail_leg",
        "quarter",
        "quartz",
        "quiche",
        "quilt",
        "radish",
        "rag",
        "raisin_box",
        "rake",
        "ramen",
        "raspberry",
        "razor",
        "receipt",
        "refried_beans_can",
        "retainer",
        "rhubarb",
        "rib",
        "ribbon",
        "ring",
        "roasting_pan",
        "roll_dough",
        "rolling_pin",
        "rope",
        "rose",
        "rosehip",
        "rosemary_shaker",
        "router",
        "rubber_boot",
        "rubber_glove",
        "ruby",
        "ruler",
        "rutabaga",
        "saddle_soap_bottle",
        "saffron_shaker",
        "sage_shaker",
        "salad",
        "salmon",
        "salt_bottle",
        "salt_shaker",
        "sandal",
        "sanitary_napkin",
        "saucepan",
        "saxophone",
        "scallop",
        "scarf",
        "schnitzel",
        "scoop",
        "scoop_of_ice_cream",
        "scoreboard",
        "scraper",
        "screw",
        "scrub_brush",
        "seashell",
        "security_camera",
        "serving_cart",
        "set_triangle",
        "shears",
        "shoulder_bag",
        "shrimp",
        "sieve",
        "silver_coins",
        "skateboard_wheel",
        "skiff",
        "skirt",
        "sliced_brisket",
        "sliced_chocolate_cake",
        "sliced_cucumber",
        "sliced_eggplant",
        "sliced_lemon",
        "sliced_lime",
        "sliced_melon",
        "sliced_onion",
        "sliced_papaya",
        "sliced_roast_beef",
        "sliced_tomato",
        "slingback",
        "snapper",
        "snow_globe",
        "soap_bottle",
        "soap_dish",
        "sock",
        "sod",
        "soda_can",
        "soda_water_bottle",
        "sodium_carbonate_jar",
        "soft_roll",
        "solvent_bottle",
        "soybean",
        "sparkler",
        "spice_cookie",
        "spice_cookie_dough",
        "spinach",
        "spray_bottle",
        "spray_can",
        "spray_paint_can",
        "square_light",
        "squeegee",
        "squeeze_bottle",
        "staple",
        "star_anise",
        "steak",
        "steel_wool",
        "sticker",
        "sticky_note",
        "stocking",
        "straw",
        "sugar_cookie",
        "sugar_cookie_dough",
        "sugar_cube",
        "sugar_sack",
        "sugar_syrup_bottle",
        "sunflower",
        "sunflower_seed_bag",
        "sunglasses",
        "swiss_cheese",
        "syringe",
        "t_shirt",
        "table_runner",
        "tablet",
        "tackle_box",
        "taco",
        "tag",
        "tank",
        "tank_top",
        "tarp",
        "tartlet",
        "tassel",
        "teacup",
        "teddy_bear",
        "tenderloin",
        "tennis_racket",
        "test_tube",
        "test_tube_clamp",
        "test_tube_holder",
        "test_tube_rack",
        "textbook",
        "thermostat",
        "thumbtack",
        "thyme_shaker",
        "ticket",
        "tights",
        "tile",
        "tinsel",
        "tiramisu",
        "toast",
        "tobacco_pipe",
        "toilet_paper",
        "toilet_soap_bottle",
        "tomato_paste_can",
        "tomato_sauce_jar",
        "toothbrush",
        "toothpick",
        "tortilla",
        "tortilla_chips",
        "tote",
        "toy_box",
        "toy_car",
        "toy_dice",
        "toy_train",
        "trombone",
        "trophy",
        "trout",
        "trumpet",
        "tube_of_lotion",
        "tube_of_toothpaste",
        "tulip",
        "tuna",
        "tupperware",
        "turkey",
        "turkey_leg",
        "twine",
        "umbrella",
        "utility_knife",
        "valentine_wreath",
        "vanilla_bottle",
        "vanilla_flower",
        "vase",
        "veal",
        "vegetable_peeler",
        "vending_machine",
        "venison",
        "vest",
        "video_game",
        "violin",
        "violin_case",
        "virginia_ham",
        "wafer",
        "waffle",
        "wallet",
        "walnut",
        "watch",
        "water_filter",
        "water_glass",
        "watering_can",
        "wax_paper",
        "webcam",
        "weed",
        "weight_bar",
        "whisk",
        "whiskey_stone",
        "whistle",
        "white_chocolate",
        "white_rice_sack",
        "white_sauce_bottle",
        "white_turnip",
        "wind_chime",
        "wineglass",
        "wok",
        "wrapped_hamburger",
        "wrapping_paper",
        "wreath",
        "yeast_jar",
        "yeast_shaker"
    ]

def get_droid_categories_by_theme():
    return {
        "Stationary": {
            "Marker": ["marker"],
            "Pen": ["pen"],
            "Tape": ["duct_tape"],
            "Paper": ["paper_sheet"],
            "Eraser": ["rubber_eraser", "blackboard_eraser"],
            "Pencil": ["pencil"],
            "Glue": ["glue_stick", "bottle_of_glue"],
            "Stapler": ["stapler"],
        },
        "Containers": {
            "Cup": [
                "coffee_cup",
                "teacup_cup",
                "beaker_cup",
                "soda_cup",
                "mug"
            ],
            "Bottle": [
                "bottle_of_alcohol",
                "bottle_of_alfredo_sauce",
                "bottle_of_almond_oil",
                "bottle_of_ammonia",
                "bottle_of_antihistamines",
                "bottle_of_apple_cider",
                "bottle_of_apple_juice",
                "bottle_of_aspirin",
                "bottle_of_baby_oil",
                "bottle_of_barbecue_sauce",
                "bottle_of_beer",
                "bottle_of_black_pepper",
                "bottle_of_bleach_agent",
                "bottle_of_bug_repellent",
                "bottle_of_carrot_juice",
                "bottle_of_catsup",
                "bottle_of_caulk",
                "bottle_of_champagne",
                "bottle_of_chili_pepper",
                "bottle_of_chocolate_sauce",
                "bottle_of_cleaner",
                "bottle_of_cocoa",
                "bottle_of_coconut_milk",
                "bottle_of_coconut_oil",
                "bottle_of_coconut_water",
                "bottle_of_coffee",
                "bottle_of_coke",
                "bottle_of_cold_cream",
                "bottle_of_cologne",
                "bottle_of_conditioner",
                "bottle_of_cooking_oil",
                "bottle_of_cranberry_juice",
                "bottle_of_deicer",
                "bottle_of_detergent",
                "bottle_of_dish_soap",
                "bottle_of_disinfectant",
                "bottle_of_essential_oil",
                "bottle_of_fabric_softener",
                "bottle_of_face_cream",
                "bottle_of_fennel",
                "bottle_of_frosting",
                "bottle_of_fruit_punch",
                "bottle_of_garlic_sauce",
                "bottle_of_gin",
                "bottle_of_ginger",
                "bottle_of_ginger_beer",
                "bottle_of_glass_cleaner",
                "bottle_of_glue",
                "bottle_of_ground_cloves",
                "bottle_of_ground_mace",
                "bottle_of_ground_nutmeg",
                "bottle_of_hot_sauce",
                "bottle_of_lacquer",
                "bottle_of_lavender_oil",
                "bottle_of_lemon_juice",
                "bottle_of_lemon_sauce",
                "bottle_of_lemonade",
                "bottle_of_lighter_fluid",
                "bottle_of_lime_juice",
                "bottle_of_liquid_soap",
                "bottle_of_lotion",
                "bottle_of_lubricant",
                "bottle_of_maple_syrup",
                "bottle_of_mayonnaise",
                "bottle_of_medicine",
                "bottle_of_milk",
                "bottle_of_milkshake",
                "bottle_of_molasses",
                "bottle_of_mushroom_sauce",
                "bottle_of_mustard",
                "bottle_of_mustard_seeds",
                "bottle_of_oil",
                "bottle_of_olive_oil",
                "bottle_of_onion_powder",
                "bottle_of_orange_juice",
                "bottle_of_paint",
                "bottle_of_paint_remover",
                "bottle_of_papaya_juice",
                "bottle_of_paprika",
                "bottle_of_peanut_butter",
                "bottle_of_perfume",
                "bottle_of_pesticide",
                "bottle_of_pesto",
                "bottle_of_pizza_sauce",
                "bottle_of_pop",
                "bottle_of_poppy_seeds",
                "bottle_of_powder",
                "bottle_of_protein_powder",
                "bottle_of_pumpkin_pie_spice",
                "bottle_of_rum",
                "bottle_of_sage",
                "bottle_of_sake",
                "bottle_of_salsa",
                "bottle_of_sealant",
                "bottle_of_seasoning",
                "bottle_of_sesame_oil",
                "bottle_of_sesame_seeds",
                "bottle_of_shampoo",
                "bottle_of_skin_cream",
                "bottle_of_soda",
                "bottle_of_solvent",
                "bottle_of_soup",
                "bottle_of_sour_cream",
                "bottle_of_soy_milk",
                "bottle_of_soy_sauce",
                "bottle_of_sriracha",
                "bottle_of_strawberry_juice",
                "bottle_of_sunscreen",
                "bottle_of_supplements",
                "bottle_of_tea",
                "bottle_of_tea_leaves",
                "bottle_of_tequila",
                "bottle_of_tomato_paste",
                "bottle_of_tonic",
                "bottle_of_vinegar",
                "bottle_of_vodka",
                "bottle_of_water",
                "bottle_of_whiskey",
                "bottle_of_wine"
                "carafe",
                "wine_bottle",
                "beer_bottle",
                "water_bottle",
                "erlenmeyer_flask",
                "pill_bottle",
                "round_bottom_flask",
                "reagent_bottle"
            ],
            "Bowl": ["bowl"],
            "Pot": ["stockpot", "saucepot"],
            "Bin": ["trash_can"],
            "Can": [
                "canned_food",
                "can",
                "can_of_baking_mix",
                "can_of_bay_leaves",
                "can_of_beans",
                "can_of_cat_food",
                "can_of_coffee",
                "can_of_corn",
                "can_of_dog_food",
                "can_of_icetea",
                "can_of_oatmeal",
                "can_of_sardines",
                "can_of_soda",
                "can_of_tomato_paste",
                "can_of_tomatoes"
            ],
            "Basket": [
                "hamper",
                "wicker_basket",
                "shopping_basket",
            ],
            "Jar": [
                "jar",
                "jar_of_bath_salt",
                "jar_of_chilli_powder",
                "jar_of_clove",
                "jar_of_cocoa",
                "jar_of_coffee",
                "jar_of_cumin",
                "jar_of_curry_powder",
                "jar_of_dill_seed",
                "jar_of_grains",
                "jar_of_honey",
                "jar_of_jam",
                "jar_of_jelly",
                "jar_of_kidney_beans",
                "jar_of_mayonnaise",
                "jar_of_orange_jam",
                "jar_of_orange_sauce",
                "jar_of_pepper",
                "jar_of_pepper_seasoning",
                "jar_of_peppercorns",
                "jar_of_puree",
                "jar_of_sesame_seed",
                "jar_of_spaghetti_sauce",
                "jar_of_strawberry_jam",
                "jar_of_sugar",
                "jar_of_tumeric",
            ],
            "Jug": [
                "jug",
                "jug_of_milk",
            ],
            "Tray": ["tray"],
            "Teapot": ["teapot"],
            "Bucket": ["bucket"],
        },
        "Furniture": {
            # TODO
        },
        "Utensils": {
            "Spoon": [
                "wooden_spoon",
                "tablespoon"
            ],
            "Fork": [
                "tablefork",
                "toasting_fork"
            ],
            "Knife": [
                "carving_knife",
                "table_knife",
                "parer"
            ],
            "Chopstick": ["chopstick"],
            "Teaspoon": ["teaspoon"]
        },
        "Hardware": {
            "Tool": [
                "drill",
                "power_drill",
                "trowel",
                "tweezers",
                "pruner",
                "pocketknife",
                "putty_knife",
                "wire_cutter",
                "plier",
                "hammer"
            ],
            "Screwdriver": ["screwdriver"],
            "Wrench": [
                "wrench",
                "allen_wrench",
                "open_end_wrench"
            ],
            "Mouse": ["mouse"]
        },
        "Toys": {
            "Doll": [
                "doll",
                "toy_figure"
            ]
        },
        "Sports": {
            "Ball": [
                "softball",
                "soccer_ball",
                "volleyball",
                "baseball",
                "tennis_ball"
            ]
        },
        "Kitchen Tools": {
            "Spatula": ["spatula"],
            "Pan": ["frying_pan"],
            "Sponge": ["sponge"],
            "Shaker": ["shaker"],
            "Ladle": ["soup_ladle"],
            "Tongs": ["tongs"]
        },
        "Food": {
            "bread": [
                "hotdog_bun",
                "scone",
                "half_bagel",
                "sourdough"
            ],
            "egg": ["egg"],
            "fruit": [
                "papaya",
                "gooseberry",
                "pear",
                "pomelo",
                "chestnut",
                "nectarine",
                "lime"
            ],
            "lemon": ["lemon"],
            "orange": ["orange"],
            "banana": ["banana"],
            "pineapple": ["pineapple"],
            "grape": ["grape"],
            "peach": ["peach"],
            "apple": ["apple"],
            "avocado": ["avocado"],
            "strawberry": ["strawberry"],
            "vegetable": [
                "chilli"
                "vidalia_onion"
                "bell_pepper"
            ],
            "tomato": [
                "cherry_tomato",
                "beefsteak_tomato"
            ],
            "potato": ["potato"],
            "carrot": ["carrot"],
            "broccoli": ["broccoli"],
            "sushi": ["sushi"],
            "yogurt": [
                "yogurt",
                "yogurt_carton",
            ],
            "zucchini": ["zucchini"],
            "snack": ["bag_of_chips"],
            "corn": ["sweet_corn"],
            "soda": ["can_of_soda"],
            "teabag": ["tea_bag"],
            "cereal": ["box_of_cereal"],
            "burger": [
                "hamburger",
                "hamburger_bun",
                "half_hamburger_bun"
            ]
        }
    }


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
            "ori": obj.get_orientation(),
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
    print("DEBUG: Placing objects...")
    placed_objects_info = []
    objects_to_randomly_place = []
    if objects_to_skip is None:
        objects_to_skip = []

    # First pass: Identify main object, process skipped distractors, and collect other objects
    for i, cfg in enumerate(obj_cfg):
        print(f"DEBUG: Processing object '{cfg['name']}'...")
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
                print(f"Warning: Skipped distractor '{cfg['name']}' does not have a valid 'position' field. Skipping placement.")
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
            print(f"Failed to place object '{cfg.get('name', 'Unnamed Object')}' after {max_attempts_per_object} attempts. Dropping it from the air.")
            x_center = np.random.uniform(xmin + half_width, xmax - half_width)
            y_center = np.random.uniform(ymin + half_depth, ymax - half_depth)
            obj_cfg[original_idx]["position"] = [x_center, y_center, z + 0.1]


    return obj_cfg