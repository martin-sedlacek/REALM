

from __future__ import annotations

import os
import json
import ast
import math
import random
import re
import struct
from pathlib import Path
from typing import Optional

import yaml


ROLE_KEYS = ("main_objects", "target_objects", "distractors", "immutables")
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
_DROID_PAIR_CACHE: dict[int, tuple[int, list[list[tuple[str, dict[str, list[float]]]]]]] = {}
_REALM_PAIR_CACHE: dict[int, tuple[int, list[list[tuple[str, dict[str, list[float]]]]]]] = {}


def load_panda_preview_meshes(mesh_root: Path, triangles_per_link: int = 1200) -> list[dict[str, object]]:

    meshes = []
    for link_index in range(8):
        path = mesh_root / f"link{link_index}.stl"
        if not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) < 84:
            continue
        triangle_count = min(struct.unpack_from("<I", data, 80)[0], triangles_per_link)
        positions = []
        for triangle_index in range(triangle_count):
            offset = 84 + triangle_index * 50 + 12
            positions.extend(struct.unpack_from("<9f", data, offset))
        meshes.append({
            "link": link_index,
            "positions": positions,
            "indices": list(range(triangle_count * 3)),
        })
    return meshes


def default_dataset_roots(repo_root: Path) -> list[Path]:

    candidates = [
        os.environ.get("OMNIGIBSON_DATASET_PATH"),
        os.environ.get("OG_DATASET_PATH"),
        repo_root / "datasets",
        repo_root / "data",
    ]
    roots = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return roots


def asset_from_usd(path: Path, root: Path) -> dict[str, object]:

    relative = path.relative_to(root)
    parts = relative.parts
    model = path.stem
    category = path.parent.name

    if "objects" in parts:
        index = parts.index("objects")
        if len(parts) > index + 1:
            category = parts[index + 1]
        if len(parts) > index + 2:
            model = parts[index + 2]
    elif path.parent.name.lower() == "usd" and len(path.parents) >= 3:
        model = path.parent.parent.name
        category = path.parent.parent.parent.name

    metadata_path = path.parent.parent / "misc" / "metadata.json"
    bbox, bbox_source = load_asset_bbox(metadata_path, category)
    return {
        "id": str(relative),
        "name": category,
        "category": category,
        "model": model,
        "usd_path": str(path),
        "bbox": bbox,
        "bbox_source": bbox_source,
    }


def load_asset_bbox(metadata_path: Path, category: str) -> tuple[list[float], str]:

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bbox = metadata.get("bbox_size")
        if isinstance(bbox, list) and len(bbox) == 3 and all(float(value) > 0 for value in bbox):
            return [float(value) for value in bbox], "model metadata"
    except (OSError, ValueError, TypeError):
        pass
    return suggested_bbox(category), "category estimate"


def load_scene_regions(path: Path) -> list[dict[str, object]]:

    if not path.is_file():
        return []
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regions = []
    for scene, supports in document.items():
        if not isinstance(supports, dict):
            continue
        for support, config in supports.items():
            if not isinstance(config, dict):
                continue
            keys = ("x_min", "x_max", "y_min", "y_max")
            if not all(key in config for key in keys):
                continue
            x_min, x_max, y_min, y_max = (float(config[key]) for key in keys)
            regions.append({
                "id": f"{scene} / {support}",
                "scene": str(scene),
                "support": str(support),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "width": x_max - x_min,
                "depth": y_max - y_min,
                "z": float(config.get("z", 0.0)),
                "robot_pos": [float(value) for value in config.get("pos", [0.0, 0.0, 0.0])],
                "robot_rot": [float(value) for value in config.get("rot", [0.0, 0.0, 0.0])],
            })
    return regions


def discover_task_types(config_root: Path) -> list[str]:

    task_types = set()
    if config_root.is_dir():
        for path in config_root.rglob("*.yaml"):
            try:
                document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            task_type = document.get("task_type") if isinstance(document, dict) else None
            if isinstance(task_type, str) and task_type:
                task_types.add(task_type)
    preferred = ("put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer")
    return [value for value in preferred if value in task_types] + sorted(task_types - set(preferred))


def discover_existing_task_names(config_root: Path) -> list[str]:

    if not config_root.is_dir():
        return []
    return sorted({path.parent.name for path in config_root.rglob("default.yaml")})


def load_drawer_cabinet_models(path: Path) -> list[str]:

    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DRAWER_CABINET_MODELS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return [str(model) for model in value]
    return []


def load_droid_categories(path: Path) -> list[str]:

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    themes = document.get("droid_categories_by_theme", {})
    categories = set()
    for subcategories in themes.values():
        if not isinstance(subcategories, dict):
            continue
        for values in subcategories.values():
            if isinstance(values, list):
                categories.update(str(value) for value in values)
    return sorted(categories)


def load_camera_extrinsics(path: Path) -> dict[str, dict[str, list[float]]]:

    poses = {}
    current_name = None
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line and not line[0].isspace() and not line.startswith("#"):
                    current_name = line.split(":", 1)[0].strip()
                    poses[current_name] = {}
                    continue
                match = re.match(r"\s+(pos|rot):\s*(\[[^]]+\])", line)
                if current_name and match:
                    poses[current_name][match.group(1)] = [float(value) for value in json.loads(match.group(2))]
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        name: pose for name, pose in poses.items()
        if len(pose.get("pos", [])) == 3 and len(pose.get("rot", [])) == 4
    }


def sample_opposite_camera_pair(
    poses: dict[str, dict[str, list[float]]], rng: random.Random
) -> dict[str, dict[str, list[float]]]:

    cached = _DROID_PAIR_CACHE.get(id(poses))
    if cached and cached[0] == len(poses):
        valid_pairs = cached[1]
    else:
        episode_pattern = re.compile(r"^(droid_v2_ep_\d+)_(cam[12])$")
        episodes = {}
        for name, pose in poses.items():
            match = episode_pattern.fullmatch(name)
            if match:
                episodes.setdefault(match.group(1), {})[match.group(2)] = (name, pose)
        valid_pairs = []
        for cameras in episodes.values():
            if set(cameras) != {"cam1", "cam2"}:
                continue
            pair = [cameras["cam1"], cameras["cam2"]]
            if all(_plausible_droid_camera(pose) for _, pose in pair):
                canonical = _canonicalize_droid_pair(pair)
                canonical_poses = [pose for _, pose in canonical]
                if canonical_poses[0]["pos"][1] * canonical_poses[1]["pos"][1] < 0:
                    valid_pairs.append(canonical)
        _DROID_PAIR_CACHE[id(poses)] = (len(poses), valid_pairs)
    if valid_pairs:
        pair = rng.choice(valid_pairs)
        return dict(pair)

    curated_pattern = re.compile(r"^(ep_\d+)_(cam[12])$")
    curated_episodes = {}
    for name, pose in poses.items():
        match = curated_pattern.fullmatch(name)
        if match:
            curated_episodes.setdefault(match.group(1), {})[match.group(2)] = (name, pose)
    curated_pairs = []
    for cameras in curated_episodes.values():
        if set(cameras) == {"cam1", "cam2"}:
            pair = [cameras["cam1"], cameras["cam2"]]
            if pair[0][1]["pos"][1] * pair[1][1]["pos"][1] < 0:
                curated_pairs.append(pair)
    if curated_pairs:
        return dict(rng.choice(curated_pairs))

    realm_cached = _REALM_PAIR_CACHE.get(id(poses))
    if realm_cached and realm_cached[0] == len(poses):
        realm_pairs = realm_cached[1]
    else:
        realm_pattern = re.compile(r"^(droid_realm_ep_\d+)_(cam[12])$")
        realm_episodes = {}
        for name, pose in poses.items():
            match = realm_pattern.fullmatch(name)
            if match:
                realm_episodes.setdefault(match.group(1), {})[match.group(2)] = (name, pose)
        realm_pairs = [
            [cameras["cam1"], cameras["cam2"]]
            for cameras in realm_episodes.values()
            if set(cameras) == {"cam1", "cam2"}
        ]
        _REALM_PAIR_CACHE[id(poses)] = (len(poses), realm_pairs)
    if realm_pairs:
        return dict(rng.choice(realm_pairs))

    # Small synthetic or curated pools do not necessarily carry episode names.
    negative = sorted(name for name, pose in poses.items() if pose["pos"][1] < 0)
    positive = sorted(name for name, pose in poses.items() if pose["pos"][1] > 0)
    if not negative or not positive:
        raise ValueError("camera pool must contain poses on both sides of the robot")
    names = [rng.choice(negative), rng.choice(positive)]
    if rng.random() < 0.5:
        names.reverse()
    return {name: poses[name] for name in names}


def _plausible_droid_camera(pose: dict[str, list[float]]) -> bool:

    x, y, z = pose["pos"]
    radial = (x * x + y * y) ** 0.5
    if not 0.15 <= radial <= 1.5 or not 0.15 <= z <= 1.5:
        return False
    qx, qy, qz, qw = pose["rot"]
    # Raw DROID calibration uses CV +Z as camera forward. Rotate that axis
    # into the base frame and require it to point substantially toward the base.
    forward = [
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy),
    ]
    distance = (x * x + y * y + z * z) ** 0.5
    alignment = -(forward[0] * x + forward[1] * y + forward[2] * z) / distance
    return alignment > 0.2


def _droid_cv_pose_to_omnigibson(
    pose: dict[str, list[float]],
) -> dict[str, list[float]]:

    qx, qy, qz, qw = pose["rot"]
    # Right-multiply by RotX(pi), quaternion [1, 0, 0, 0] in xyzw order.
    return {
        "pos": list(pose["pos"]),
        "rot": [qw, qz, -qy, -qx],
    }


def _canonicalize_droid_pair(
    pair: list[tuple[str, dict[str, list[float]]]],
) -> list[tuple[str, dict[str, list[float]]]]:

    converted = [(name, _droid_cv_pose_to_omnigibson(pose)) for name, pose in pair]
    first, second = converted[0][1]["pos"], converted[1][1]["pos"]
    dx, dy = first[0] - second[0], first[1] - second[1]
    yaw = math.pi / 2 - math.atan2(dy, dx)
    midpoint_x = (first[0] + second[0]) / 2
    midpoint_y = (first[1] + second[1]) / 2
    if math.cos(yaw) * midpoint_x - math.sin(yaw) * midpoint_y > 0:
        yaw += math.pi
    half = yaw / 2
    yaw_quat = [0.0, 0.0, math.sin(half), math.cos(half)]
    result = []
    for name, pose in converted:
        x, y, z = pose["pos"]
        rotated_pos = [
            math.cos(yaw) * x - math.sin(yaw) * y,
            math.sin(yaw) * x + math.cos(yaw) * y,
            z,
        ]
        rotated_quat = _multiply_quaternions_xyzw(yaw_quat, pose["rot"])
        result.append((name, {"pos": rotated_pos, "rot": rotated_quat}))
    return result


def _multiply_quaternions_xyzw(left: list[float], right: list[float]) -> list[float]:

    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def suggested_bbox(category: str) -> list[float]:

    name = category.lower()
    if "apple" in name or "orange" in name:
        return [0.09, 0.09, 0.09]
    if "bowl" in name:
        return [0.18, 0.18, 0.08]
    if "plate" in name:
        return [0.24, 0.24, 0.03]
    if "spoon" in name or "fork" in name or "knife" in name:
        return [0.19, 0.04, 0.02]
    if "bottle" in name:
        return [0.08, 0.08, 0.20]
    if "box" in name:
        return [0.25, 0.20, 0.12]
    return [0.12, 0.12, 0.12]


def discover_assets(root: Path, limit: Optional[int] = None) -> list[dict[str, object]]:

    if not root.is_dir():
        return []
    assets = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES:
            assets.append(asset_from_usd(path, root))
            if limit is not None and len(assets) >= limit:
                break
    return sorted(assets, key=lambda item: (str(item["category"]), str(item["model"])))


def demo_assets() -> list[dict[str, object]]:

    return [
        {"id": "demo/apple", "name": "apple", "category": "apple", "model": "", "usd_path": "", "bbox": [0.09, 0.09, 0.09], "bbox_source": "demo estimate"},
        {"id": "demo/bowl", "name": "bowl", "category": "bowl", "model": "", "usd_path": "", "bbox": [0.18, 0.18, 0.08], "bbox_source": "demo estimate"},
        {"id": "demo/plate", "name": "plate", "category": "plate", "model": "", "usd_path": "", "bbox": [0.24, 0.24, 0.03], "bbox_source": "demo estimate"},
        {"id": "demo/bottle", "name": "bottle_of_water", "category": "bottle_of_water", "model": "", "usd_path": "", "bbox": [0.08, 0.08, 0.20], "bbox_source": "demo estimate"},
    ]
