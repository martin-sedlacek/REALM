import tempfile
import random
import struct
import unittest
from pathlib import Path

from tooling.task_authoring.authoring import (
    asset_from_usd,
    discover_assets,
    discover_existing_task_names,
    discover_task_types,
    load_asset_bbox,
    load_drawer_cabinet_models,
    load_droid_categories,
    load_camera_extrinsics,
    load_panda_preview_meshes,
    load_scene_regions,
    sample_opposite_camera_pair,
    suggested_bbox,
)
from tooling.task_authoring.save_server import save_task_config
from tooling.task_authoring.generate_droid100_tabletop import (
    concepts,
    ensure_receiver_capacity,
    fit_bbox,
    initial_relation_type,
    place,
    place_initial_relation,
    primitive,
)


class AuthoringTest(unittest.TestCase):
    def test_batch_authoring_resolves_plural_stack_and_preserves_color_order(self):
        self.assertEqual(concepts("Stack the cups", "stack"), ["cup", "cup"])
        orange = primitive("block", "Put the orange block on the green block", 0)
        green = primitive("block", "Put the orange block on the green block", 1)
        self.assertEqual(orange["name"], "orange_block")
        self.assertEqual(green["name"], "green_block")

    def test_pick_from_instruction_keeps_source_object(self):
        instruction = "Remove the lid from the pot"
        self.assertEqual(concepts(instruction, "pick"), ["lid", "pot"])
        self.assertEqual(initial_relation_type(instruction, "pick"), "on_top")

    def test_initial_on_top_relation_has_nonintersecting_clearance(self):
        source = {
            "name": "pot",
            "bounding_box": [0.20, 0.20, 0.10],
            "relative_bbox_position": [0.25, 0.30, 0.10],
        }
        main = {"name": "lid", "bounding_box": [0.15, 0.15, 0.02]}
        audit = place_initial_relation(main, source, "on_top")
        source_top = source["relative_bbox_position"][2] + source["bounding_box"][2] / 2
        main_bottom = main["relative_bbox_position"][2] - main["bounding_box"][2] / 2
        self.assertEqual(main["relative_bbox_position"][:2], source["relative_bbox_position"][:2])
        self.assertGreater(main_bottom, source_top)
        self.assertEqual(audit["source"], "pot")

    def test_batch_authoring_uniformly_fits_oversized_bboxes(self):
        fitted, scale = fit_bbox([0.4, 0.3, 0.2], (0.2, 0.2))
        self.assertEqual(scale, 0.5)
        self.assertEqual(fitted, [0.2, 0.15, 0.1])

    def test_batch_placement_keeps_bbox_bottom_above_support(self):
        config = {"name": "object", "bounding_box": [0.05, 0.05, 0.10]}
        place(config, [], 0.5, 0.5)
        clearance = config["relative_bbox_position"][2] - 0.05
        self.assertGreaterEqual(clearance, 0.05)
        self.assertLess(clearance, 0.050001)

    def test_receiver_capacity_uses_yaw_then_uniform_scaling(self):
        main = {"bounding_box": [0.2, 0.1, 0.05]}
        target = {"bounding_box": [0.12, 0.24, 0.08]}
        audit = ensure_receiver_capacity(main, target, "put")
        self.assertEqual(audit["yaw_degrees"], 90)
        self.assertAlmostEqual(main["bounding_box"][0] / main["bounding_box"][1], 2.0)
        self.assertEqual(main["orientation"], [0.0, 0.0, 0.7071068, 0.7071068])

    def test_camera_extrinsics_are_loaded_as_numeric_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.yaml"
            path.write_text("default:\n  pos: [0, 1, 2]\n  rot: [0, 0, 0, 1]\n")
            poses = load_camera_extrinsics(path)
        self.assertEqual(poses["default"]["pos"], [0.0, 1.0, 2.0])

    def test_camera_pair_is_sampled_from_opposite_robot_sides(self):
        poses = {
            "negative": {"pos": [0.0, -0.2, 0.3], "rot": [0.0, 0.0, 0.0, 1.0]},
            "positive": {"pos": [0.0, 0.2, 0.3], "rot": [0.0, 0.0, 0.0, 1.0]},
        }
        pair = sample_opposite_camera_pair(poses, random.Random(7))
        sides = [pose["pos"][1] for pose in pair.values()]
        self.assertLess(sides[0] * sides[1], 0)

    def test_full_droid_camera_sampling_keeps_episode_pair_and_converts_axes(self):
        poses = {
            "droid_v2_ep_000001_cam1": {
                "pos": [0.4, -0.3, 0.4], "rot": [0.0, 1.0, 0.0, 0.0],
            },
            "droid_v2_ep_000001_cam2": {
                "pos": [0.4, 0.3, 0.4], "rot": [0.0, 1.0, 0.0, 0.0],
            },
        }
        pair = sample_opposite_camera_pair(poses, random.Random(7))
        self.assertEqual(set(pair), set(poses))
        positions = [pose["pos"] for pose in pair.values()]
        self.assertLess(positions[0][1] * positions[1][1], 0)
        self.assertLess((positions[0][0] + positions[1][0]) / 2, 0)

    def test_curated_camera_sampling_keeps_recorded_episode_pair(self):
        def pose(y):
            return {"pos": [0.2, y, 0.4], "rot": [0.0, 0.0, 0.0, 1.0]}

        poses = {
            "ep_000001_cam1": pose(-0.3), "ep_000001_cam2": pose(0.3),
            "ep_000002_cam1": pose(-0.4), "ep_000002_cam2": pose(0.4),
        }
        pair = sample_opposite_camera_pair(poses, random.Random(3))
        episode_names = {name.rsplit("_", 1)[0] for name in pair}
        self.assertEqual(len(episode_names), 1)
        self.assertEqual({name.rsplit("_", 1)[1] for name in pair}, {"cam1", "cam2"})

    def test_task_config_save_is_scoped_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_task_config(root, "pick_apple", 'task_type: "pick"\nmain_objects: []\n')
            self.assertEqual(
                path.relative_to(root),
                Path("realm/config/tasks/REALM_DROID10/pick_apple/default.yaml"),
            )
            with self.assertRaises(FileExistsError):
                save_task_config(root, "pick_apple", 'task_type: "pick"\n')

    def test_task_config_save_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_task_config(Path(directory), "../escape", 'task_type: "pick"\n')

    def test_droid_categories_are_flattened_from_themes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "categories.yaml"
            path.write_text("droid_categories_by_theme:\n  Food:\n    fruit: [apple, banana]\n    bowl: [bowl]\n")
            categories = load_droid_categories(path)
        self.assertEqual(categories, ["apple", "banana", "bowl"])

    def test_existing_task_names_come_from_default_config_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "suite" / "pick_spoon").mkdir(parents=True)
            (root / "suite" / "pick_spoon" / "default.yaml").touch()
            (root / "suite" / "ignored" / "variant.yaml").parent.mkdir(parents=True)
            (root / "suite" / "ignored" / "variant.yaml").touch()
            names = discover_existing_task_names(root)
        self.assertEqual(names, ["pick_spoon"])

    def test_drawer_model_allowlist_is_read_without_importing_omnigibson(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling.py"
            path.write_text('DRAWER_CABINET_MODELS = ["abc", "def"]\n')
            models = load_drawer_cabinet_models(path)
        self.assertEqual(models, ["abc", "def"])

    def test_task_types_are_discovered_in_supported_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pick.yaml").write_text("task_type: pick\n")
            (root / "put.yaml").write_text("task_type: put\n")
            (root / "duplicate.yaml").write_text("task_type: pick\n")
            task_types = discover_task_types(root)
        self.assertEqual(task_types, ["put", "pick"])

    def test_panda_preview_obj_is_triangulated_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            triangle = struct.pack("<12fH", *(0.0,) * 12, 0)
            (root / "link0.stl").write_bytes(bytes(80) + struct.pack("<I", 1) + triangle)
            meshes = load_panda_preview_meshes(root, triangles_per_link=1)
        self.assertEqual(len(meshes), 1)
        self.assertEqual(len(meshes[0]["indices"]), 3)

    def test_omnigibson_object_path(self):
        root = Path("/dataset")
        asset = asset_from_usd(root / "objects" / "apple" / "abc123" / "usd" / "abc123.usd", root)
        self.assertEqual(asset["category"], "apple")
        self.assertEqual(asset["model"], "abc123")

    def test_discovery_ignores_non_usd_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usd = root / "objects" / "bowl" / "model1" / "usd" / "model1.usd"
            usd.parent.mkdir(parents=True)
            usd.touch()
            (usd.parent / "notes.txt").touch()
            assets = discover_assets(root)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["category"], "bowl")

    def test_category_bbox_hint(self):
        self.assertEqual(suggested_bbox("apple"), [0.09, 0.09, 0.09])

    def test_model_metadata_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            metadata.write_text('{"bbox_size": [0.7, 1.3, 1.0]}', encoding="utf-8")
            bbox, source = load_asset_bbox(metadata, "table")
        self.assertEqual(bbox, [0.7, 1.3, 1.0])
        self.assertEqual(source, "model metadata")

    def test_scene_spawn_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "scenes.yaml"
            config.write_text(
                "Room:\n  Table:\n    x_min: -0.2\n    x_max: 0.2\n"
                "    y_min: 1.0\n    y_max: 1.6\n    z: 0.8\n"
                "    pos: [0.5, 0.2, 0.0]\n    rot: [0, 0, 270]\n",
                encoding="utf-8",
            )
            regions = load_scene_regions(config)
        self.assertEqual(regions[0]["width"], 0.4)
        self.assertAlmostEqual(regions[0]["depth"], 0.6)
        self.assertEqual(regions[0]["robot_rot"], [0.0, 0.0, 270.0])


if __name__ == "__main__":
    unittest.main()
