"""Turning object configs into collision-free scene placements."""
import numpy as np

import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene


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
        # The step here only exists to flush the pose change before reading the AABB -- it is not a
        # physics settle. OG 3.9.1 asserts `is_playing()` inside step(), and callers such as the V-SC
        # perturbation run this while the simulator is stopped (it has to be stopped to add/remove
        # objects), which used to be tolerated. Render instead when stopped: it propagates the
        # transform without advancing physics, and 3.9.1 computes aabb from live collision points.
        if og.sim.is_playing():
            og.sim.step()
        else:
            og.sim.render()
        this_cfg["bounding_box"] = obj.aabb_extent

        obj.set_position_orientation(this_cfg["pos"], this_cfg["ori"])

        cfgs[obj.name] = this_cfg

    return cfgs


def get_non_colliding_positions_for_objects(
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
