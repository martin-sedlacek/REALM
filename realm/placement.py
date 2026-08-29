from realm.config.shared import DEFAULT_BBOX_EXTENT, DROP_HEIGHT

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

        # Park in the object's scene frame so vectorized scenes remain isolated.
        far_pos = np.random.random((3,)) * 3 + np.array([0, 0, 20])
        obj.set_position_orientation(position=far_pos, orientation=[0, 0, 0, 1], frame="scene")
        # Flush transforms without stepping a stopped simulator.
        if og.sim.is_playing():
            og.sim.step()
        else:
            og.sim.render()
        this_cfg["bounding_box"] = obj.aabb_extent

        obj.set_position_orientation(this_cfg["pos"], this_cfg["ori"])

        cfgs[obj.name] = this_cfg

    return cfgs


def _half_footprint(cfg):

    return cfg["bounding_box"][0] / 2, cfg["bounding_box"][1] / 2


def _partition_configs(obj_cfg, main_object_names, objects_to_skip, maximum_dim):

    placed_objects_info = []
    objects_to_randomly_place = []

    for i, cfg in enumerate(obj_cfg):
        if cfg["name"] in main_object_names:
            if "bounding_box" not in cfg or "position" not in cfg:
                raise KeyError(
                    f"main object '{cfg['name']}' needs 'position' and 'bounding_box' before "
                    f"placement -- backfill them from the live object first "
                    f"(perturbations/_helpers.backfill_object_cfgs)")
            half_width_main, half_depth_main = _half_footprint(cfg)
            placed_objects_info.append(
                (cfg["position"][0], cfg["position"][1], half_width_main, half_depth_main))
        elif cfg["name"] in objects_to_skip:
            if "bounding_box" not in cfg:
                cfg["bounding_box"] = list(DEFAULT_BBOX_EXTENT)
            else:
                max_dim = np.max(np.array(cfg["bounding_box"]))
                new_scale_factor = maximum_dim / max_dim
                if new_scale_factor < 1.0:
                    # Preserve the historical footprint-only scaling.
                    cfg["bounding_box"] = np.array(cfg["bounding_box"]) * new_scale_factor

            if "position" not in cfg or len(cfg["position"]) < 2:
                og.log.warn(f"Warning: Skipped distractor '{cfg['name']}' does not have a valid "
                            f"'position' field. Skipping placement.")
                continue

            placed_objects_info.append(
                (cfg["position"][0], cfg["position"][1], *_half_footprint(cfg)))
        else:
            objects_to_randomly_place.append((cfg, i))

    return placed_objects_info, objects_to_randomly_place


def _sample_free_position(placed, half_width, half_depth,
                          xmin, xmax, ymin, ymax, min_separation, max_attempts):

    for _ in range(max_attempts):
        x_center = np.random.uniform(xmin + half_width, xmax - half_width)
        y_center = np.random.uniform(ymin + half_depth, ymax - half_depth)

        collision = False
        for px, py, phw, phd in placed:
            if (abs(x_center - px) < (half_width + phw + min_separation)
                    and abs(y_center - py) < (half_depth + phd + min_separation)):
                collision = True
                break

        if not collision:
            return x_center, y_center
    return None


def get_non_colliding_positions_for_objects(
        xmin, xmax, ymin, ymax, z, obj_cfg,
        main_object_names,
        min_separation=0.05,
        max_attempts_per_object=2500,
        objects_to_skip=None,
        maximum_dim=0.12
):

    if objects_to_skip is None:
        objects_to_skip = []

    placed_objects_info, objects_to_randomly_place = _partition_configs(
        obj_cfg, main_object_names, objects_to_skip, maximum_dim)

    np.random.shuffle(objects_to_randomly_place)

    for cfg, original_idx in objects_to_randomly_place:
        if "bounding_box" not in cfg:
            cfg["bounding_box"] = list(DEFAULT_BBOX_EXTENT)
        half_width, half_depth = _half_footprint(cfg)

        position = _sample_free_position(placed_objects_info, half_width, half_depth,
                                         xmin, xmax, ymin, ymax, min_separation,
                                         max_attempts_per_object)
        if position is not None:
            x_center, y_center = position
            placed_objects_info.append((x_center, y_center, half_width, half_depth))
            obj_cfg[original_idx]["position"] = [x_center, y_center, z]
        else:
            og.log.error(f"Failed to place object '{cfg.get('name', 'Unnamed Object')}' after "
                         f"{max_attempts_per_object} attempts. Dropping it from the air.")
            x_center = np.random.uniform(xmin + half_width, xmax - half_width)
            y_center = np.random.uniform(ymin + half_depth, ymax - half_depth)
            obj_cfg[original_idx]["position"] = [x_center, y_center, z + DROP_HEIGHT]

    return obj_cfg


def place_within(spawn_bbox, obj_cfg, **kwargs):

    xmin, xmax, ymin, ymax, z = spawn_bbox
    return get_non_colliding_positions_for_objects(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, z=z, obj_cfg=obj_cfg, **kwargs)
