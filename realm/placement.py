"""Turning object configs into collision-free scene placements.

Two cfg dict shapes flow through this module, and they are NOT interchangeable:

* task/sampled configs (what the placement pass consumes):
  ``{"name", "category", "model", "position", "orientation", "bounding_box", ...}`` --
  "bounding_box" is an EXTENT (a size, like the ``[0.20, 0.20, 0.07]`` the task YAMLs write) and
  "position" is SCENE-relative;
* the read-back configs `get_default_objects_cfg` returns:
  ``{"category", "pos", "ori", "bounding_box", "relative_prim_path"}``, keyed by object name.
"""
import numpy as np

import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

#: Extent assumed for an object whose cfg carries no "bounding_box" (metres).
DEFAULT_BBOX_EXTENT = (0.08, 0.08, 0.08)

#: How far above the surface an unplaceable object is dropped from (metres).
DROP_HEIGHT = 0.1


def get_objects_by_names(scene: InteractiveTraversableScene, names: list[str]) -> list[DatasetObject]:
    """The scene objects whose names appear in @names, in scene iteration order."""
    objects = []
    for obj in scene.objects:
        obj: DatasetObject
        if obj.name in names:
            objects.append(obj)
    return objects


def get_default_objects_cfg(scene: InteractiveTraversableScene, object_names: list[str]) -> dict[str, dict]:
    """Read category/pose/AABB off the live objects, parking each ~20 m up to measure it.

    SIDE EFFECT: every object is teleported ~20 m above its own scene, its AABB read with
    neighbours out of contact, and then restored to where it was. One og.sim.step() (or render()
    when the sim is stopped) runs per object to flush the pose before the read.

    Returns ``{object name: {"category", "pos", "ori", "bounding_box", "relative_prim_path"}}`` --
    note the "pos"/"ori" keys: this is NOT the task-config dict shape (see the module docstring).
    """
    objects = get_objects_by_names(scene, object_names)
    cfgs = {}
    for obj in objects:
        this_cfg = {
            "category": obj.category,
            "pos": obj.aabb_center,
            "ori": obj.get_position_orientation()[1],
            "relative_prim_path": obj._relative_prim_path
        }

        # frame="scene", so "clear of everything" means clear of the object's OWN scene. In world
        # frame these coordinates land in scene 0's airspace no matter which scene the object
        # belongs to -- vector-env scenes are tiled ~25 m apart along +x, so every member's object
        # would be parked in the same column above member 0, measuring a spot other members are
        # also using.
        far_pos = np.random.random((3,)) * 3 + np.array([0, 0, 20])
        obj.set_position_orientation(position=far_pos, orientation=[0, 0, 0, 1], frame="scene")
        # Flush the pose change before reading the AABB -- this is not a physics settle. OG 3.9.1
        # asserts is_playing() inside step(), and callers such as V-SC run this while the simulator
        # is stopped (it has to be, to add/remove objects). Render instead when stopped: it
        # propagates the transform without advancing physics, and 3.9.1 computes the AABB from live
        # collision points.
        if og.sim.is_playing():
            og.sim.step()
        else:
            og.sim.render()
        this_cfg["bounding_box"] = obj.aabb_extent

        obj.set_position_orientation(this_cfg["pos"], this_cfg["ori"])

        cfgs[obj.name] = this_cfg

    return cfgs


def _half_footprint(cfg):
    """(half_width, half_depth) of a cfg's bounding_box extent, for XY overlap tests."""
    return cfg["bounding_box"][0] / 2, cfg["bounding_box"][1] / 2


def _partition_configs(obj_cfg, main_object_names, objects_to_skip, maximum_dim):
    """Split @obj_cfg into pre-placed footprints and (cfg, index) pairs still needing a position.

    Main objects keep their authored position and full-size footprint, and must already carry
    "position" and "bounding_box" (backfill them from the live objects first -- see
    perturbations/_helpers.backfill_object_cfgs). Skipped objects also stay where they are, but
    their footprint is shrunk to @maximum_dim and defaulted to DEFAULT_BBOX_EXTENT when absent.
    """
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
                    # Footprint only: the LIVE object is deliberately not rescaled here (unlike
                    # object_sampling.rescale_to_max_dim), so an oversized skipped object keeps its
                    # size but reserves a smaller footprint. Pre-existing behaviour, kept because
                    # changing it would move every placement drawn since.
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
    """A uniformly drawn (x, y) whose @min_separation-padded footprint clears everything in @placed.

    Returns None after @max_attempts rejections. Draws exactly two np.random.uniform values per
    attempt -- callers rely on this draw order for reproducibility.
    """
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
    """Give every cfg in @obj_cfg a collision-free "position" inside [xmin..xmax] x [ymin..ymax].

    MUTATES @obj_cfg's dicts in place AND returns the same list -- the entries are aliased, not
    copied, so a caller that must not see its configs rewritten has to deepcopy first (v_sc does).
    Placement is rejection sampling in the XY plane over axis-aligned footprints; "bounding_box"
    is an EXTENT and "position" is SCENE-relative (see the module docstring).

    @main_object_names keep their authored positions and full footprints. @objects_to_skip stay
    put too, but get their footprint shrunk to @maximum_dim. Everything else is shuffled and
    placed at height @z. An object that cannot be placed within @max_attempts_per_object attempts
    is dropped from the air: it gets a random position at z + DROP_HEIGHT regardless of overlap,
    and an error is logged.
    """
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
    """get_non_colliding_positions_for_objects over a spawn_bbox of [xmin, xmax, ymin, ymax, z].

    @spawn_bbox is the 5-element array env_config builds from scenes.yaml (env.spawn_bbox); every
    other argument passes through unchanged.
    """
    xmin, xmax, ymin, ymax, z = spawn_bbox
    return get_non_colliding_positions_for_objects(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, z=z, obj_cfg=obj_cfg, **kwargs)
