

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tooling.task_authoring.authoring import (
    discover_assets,
    load_camera_extrinsics,
    load_droid_categories,
    load_scene_regions,
    sample_opposite_camera_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "data" / "droid" / "DROID100_tabletop.json"
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets_og391" / "behavior-1k-assets"
DEFAULT_OUTPUT = REPO_ROOT / "realm" / "config" / "tasks" / "REALM_DROID100"
DEFAULT_CAMERA_EXTRINSICS = (
    REPO_ROOT / "realm" / "config" / "env" / "external_sensors" / "camera_extrinsics_droid_realm.yaml"
)
DEFAULT_SEED = 100
SUPPORT_CLEARANCE = 0.05
RELATION_CLEARANCE = 0.01
SUPPORT_EDGE_CLEARANCE = 0.025
UNSAFE_SCENE_REGIONS = {
    ("Pomaria_0_int", "Coffee_Table"),
    ("Pomaria_1_int", "Drawers_Near_Table"),
    ("office_cubicles_left", "Circular_Table"),
}
ELLIPTICAL_SUPPORTS = {"Coffee_Table", "Circular_Table"}
REVIEWED_CAMERA_SOURCES = {
    82: ("droid_realm_ep_060817_cam1", "droid_realm_ep_060817_cam2"),
    98: ("droid_realm_ep_044890_cam1", "droid_realm_ep_044890_cam2"),
}
REVIEWED_MODEL_OVERRIDES = {13: {"cup": "jgethp"}}
REVIEWED_POSITION_OVERRIDES = {
    47: {
        "lid": [0.28, 0.15],
        "pot": [0.28, 0.15],
        "distractor_beefsteak_tomato": [0.12, 0.15],
    },
}
REVIEWED_TASK_OVERRIDES = {
    57: {
        "instruction": "Remove the can from the bowl",
        "task_type": "pick",
        "reason": "No movable sink asset exists; the authored source is a bowl and the instruction must name it honestly.",
    },
    71: {
        "instruction": "Put the screwdriver in the bowl",
        "task_type": "put",
        "reason": "The DROID phrase describes an orange-handled tool; a screwdriver is the grounded asset.",
    },
    76: {
        "instruction": "Put the white object on the plate",
        "task_type": "stack",
        "reason": "The original multi-object arrangement is not representable by REALM's single-main contract.",
    },
    78: {
        "instruction": "Remove the white cloth from the plate",
        "task_type": "pick",
        "reason": "No stand asset exists; the authored horizontal support is a plate.",
    },
    79: {
        "instruction": "Put the block in the bowl",
        "task_type": "put",
        "reason": "The config contains one manipulated block, so the instruction must be singular.",
    },
    89: {
        "instruction": "Pick up the screwdriver and put it in the bowl",
        "task_type": "put",
        "reason": "The orange object tool is one tool, not a generic object plus a screwdriver receiver.",
    },
}

CATEGORY_BY_CONCEPT = {
    "marker": "marker",
    "pen": "pen",
    "cup": "mug",
    "mug": "mug",
    "bowl": "bowl",
    "lid": "lid",
    "pot": "saucepot",
    "pan": "frying_pan",
    "towel": "dishtowel",
    "box": "storage_box",
    "tape": "masking_tape",
    "plate": "plate",
    "can": "can",
    "cloth": "microfiber_cloth",
    "spoon": "teaspoon",
    "tool": "screwdriver",
    "screwdriver": "screwdriver",
    # Literal fixtures absent from the movable tabletop catalog use conservative proxies.
    "sink": "bowl",
    "stand": "plate",
}
CONCEPT_PATTERN = re.compile(
    r"\b(mug cup|glass lid|silver lid|masking tape|marker|pen|cup|mug|bowl|lid|pot|pan|"
    r"block|blocks|cube|object|objects|cups|towel|box|tape|plate|can|cloth|spoon|tool|screwdriver|sink|stand)\b",
    re.IGNORECASE,
)
COLORS = {
    "blue": [0.1, 0.25, 0.9, 1.0],
    "green": [0.1, 0.7, 0.25, 1.0],
    "yellow": [0.9, 0.8, 0.1, 1.0],
    "orange": [0.95, 0.4, 0.05, 1.0],
    "red": [0.85, 0.1, 0.1, 1.0],
    "white": [0.9, 0.9, 0.9, 1.0],
    "black": [0.08, 0.08, 0.08, 1.0],
    "silver": [0.65, 0.68, 0.72, 1.0],
}


def slug(value: str) -> str:

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def reviewed_task(task: dict[str, object]) -> tuple[str, str, dict[str, str] | None]:

    explicit = REVIEWED_TASK_OVERRIDES.get(int(task["rank"]))
    if explicit:
        return explicit["instruction"], explicit["task_type"], explicit
    instruction = str(task["instruction"])
    task_type = str(task["task_type"])
    if task_type == "pick":
        shortened = re.sub(
            r"\s+and\s+(?:put|place)\s+it\s+on\s+the\s+(?:table|counter)\.?\s*$",
            "",
            instruction,
            flags=re.IGNORECASE,
        )
        if shortened != instruction:
            override = {
                "instruction": shortened,
                "task_type": task_type,
                "reason": "REALM pick evaluates removal/lifting, not a subsequent placement on the scene support.",
            }
            return shortened, task_type, override
    return instruction, task_type, None


def fit_bbox(values: list[float], max_xy: tuple[float, float]) -> tuple[list[float], float]:

    scale = min(1.0, max_xy[0] / values[0], max_xy[1] / values[1])
    return [round(float(value) * scale, 7) for value in values], scale


def sample_camera_pair(
    poses: dict[str, dict[str, list[float]]], rng: random.Random
) -> dict[str, dict[str, list[float]]]:

    sampled = sample_opposite_camera_pair(poses, rng)
    pair = list(sampled.items())
    return {
        "cam1": {**pair[0][1], "source": pair[0][0]},
        "cam2": {**pair[1][1], "source": pair[1][0]},
    }


def instruction_terms(instruction: str, task_type: str) -> list[str]:

    found = [match.group(1).lower() for match in CONCEPT_PATTERN.finditer(instruction)]
    required = 2 if task_type in {"put", "stack"} or initial_relation_type(instruction, task_type) else 1
    if task_type == "stack" and len(found) == 1 and found[0] == "cups":
        found.append("cups")
    if len(found) < required:
        raise ValueError(f"could not resolve {required} object terms from {instruction!r}")
    return found[:required]


def initial_relation_type(instruction: str, task_type: str) -> str | None:

    if task_type != "pick":
        return None
    lowered = instruction.lower()
    if not re.search(r"\b(?:from|out of|off(?: of)?)\b", lowered):
        return None
    if (
        re.search(r"\boff(?: of)?\b", lowered)
        or re.search(r"\blid\b.*\b(?:pot|pan)\b", lowered)
        or re.search(r"\b(?:from|out of)\b[^,.]*\bstand\b", lowered)
    ):
        return "on_top"
    return "inside"


def concepts(instruction: str, task_type: str) -> list[str]:

    found = instruction_terms(instruction, task_type)
    normalized = []
    for value in found:
        if value in {"mug cup"}:
            value = "mug"
        elif value in {"glass lid", "silver lid"}:
            value = "lid"
        elif value == "masking tape":
            value = "tape"
        elif value == "cups":
            value = "cup"
        elif value in {"blocks", "cube", "objects"}:
            value = "block"
        normalized.append(value)
    return normalized


def primitive(concept: str, instruction: str, occurrence: int) -> dict[str, object]:

    color_words = re.findall(r"\b(?:blue|green|yellow|orange|red|white|black|silver)\b", instruction.lower())
    color = color_words[min(occurrence, len(color_words) - 1)] if color_words else "blue"
    name = f"{color}_{'block' if concept == 'block' else 'object'}"
    return {
        "type": "PrimitiveObject",
        "name": name,
        "primitive_type": "Cube",
        "rgba": COLORS[color],
        "bounding_box": [0.05, 0.05, 0.05],
        "scale": [0.05, 0.05, 0.05],
    }


def dataset_object(
    concept: str,
    assets_by_category: dict[str, list[dict[str, object]]],
    *,
    prefer_large: bool = False,
) -> dict[str, object]:

    category = CATEGORY_BY_CONCEPT.get(concept, concept)
    candidates = assets_by_category.get(category, [])
    if not candidates:
        raise ValueError(f"no indexed model for category {category!r}")
    chooser = max if prefer_large else min
    asset = chooser(candidates, key=lambda item: math.prod(item["bbox"][:2]))
    return {
        "type": "DatasetObject",
        "name": concept,
        "category": category,
        "model": asset["model"],
        "bounding_box": [round(float(value), 7) for value in asset["bbox"]],
    }


def apply_model_override(
    config: dict[str, object],
    model: str,
    assets_by_category: dict[str, list[dict[str, object]]],
    max_xy: tuple[float, float],
) -> dict[str, object]:

    category = str(config["category"])
    asset = next((item for item in assets_by_category[category] if item["model"] == model), None)
    if asset is None:
        raise ValueError(f"reviewed model {category}/{model} is not indexed")
    original = [round(float(value), 7) for value in asset["bbox"]]
    fitted, scale = fit_bbox(original, max_xy)
    config["model"] = model
    config["bounding_box"] = fitted
    return {
        "name": config["name"],
        "reason": "render_review_model_override",
        "original_bbox": original,
        "authored_bbox": fitted,
        "scale": round(scale, 5),
    }


def overlaps(candidate: dict[str, object], placed: list[dict[str, object]], margin: float = 0.012) -> bool:

    x, y, _ = candidate["relative_bbox_position"]
    bx, by, _ = candidate["bounding_box"]
    return any(
        abs(x - other["relative_bbox_position"][0]) < (bx + other["bounding_box"][0]) / 2 + margin
        and abs(y - other["relative_bbox_position"][1]) < (by + other["bounding_box"][1]) / 2 + margin
        for other in placed
    )


def bbox_fits_support(
    x: float,
    y: float,
    bbox: list[float],
    width: float,
    depth: float,
    elliptical: bool = False,
) -> bool:

    half_x, half_y = float(bbox[0]) / 2, float(bbox[1]) / 2
    if elliptical:
        radius_x = width / 2 - SUPPORT_EDGE_CLEARANCE
        radius_y = depth / 2 - SUPPORT_EDGE_CLEARANCE
        if radius_x <= half_x or radius_y <= half_y:
            return False
        return (
            ((abs(x - width / 2) + half_x) / radius_x) ** 2
            + ((abs(y - depth / 2) + half_y) / radius_y) ** 2
            <= 1.0
        )
    return (
        half_x + SUPPORT_EDGE_CLEARANCE <= x <= width - half_x - SUPPORT_EDGE_CLEARANCE
        and half_y + SUPPORT_EDGE_CLEARANCE <= y <= depth - half_y - SUPPORT_EDGE_CLEARANCE
    )


def place(
    config: dict[str, object],
    placed: list[dict[str, object]],
    width: float,
    depth: float,
    *,
    elliptical: bool = False,
) -> None:

    candidates = (
        (0.30, 0.30), (0.70, 0.70), (0.30, 0.70), (0.70, 0.30),
        (0.50, 0.50), (0.25, 0.50), (0.75, 0.50), (0.50, 0.25),
        (0.50, 0.75), (0.15, 0.50), (0.85, 0.50), (0.50, 0.85),
    )
    bx, by, _ = config["bounding_box"]
    for ux, uy in candidates:
        x = max(bx / 2, min(width - bx / 2, ux * width))
        y = max(by / 2, min(depth - by / 2, uy * depth))
        z = float(config["bounding_box"][2]) / 2 + SUPPORT_CLEARANCE
        authored_z = math.ceil(z * 10_000_000) / 10_000_000
        config["relative_bbox_position"] = [round(x, 5), round(y, 5), authored_z]
        if bbox_fits_support(x, y, config["bounding_box"], width, depth, elliptical) and not overlaps(config, placed):
            placed.append(config)
            return
    raise ValueError(f"no collision-free placement for {config['name']!r}")


def ensure_receiver_capacity(
    main: dict[str, object],
    target: dict[str, object],
    task_type: str,
) -> dict[str, object]:

    margin = 1.15 if task_type == "put" else 0.65
    main_x, main_y = (float(value) for value in main["bounding_box"][:2])
    target_x, target_y = (float(value) for value in target["bounding_box"][:2])
    scale_at_zero = min(target_x / (main_x * margin), target_y / (main_y * margin))
    scale_at_ninety = min(target_x / (main_y * margin), target_y / (main_x * margin))
    yaw = 90 if scale_at_ninety > scale_at_zero else 0
    scale = min(1.0, max(scale_at_zero, scale_at_ninety))
    original = list(main["bounding_box"])
    if scale < 1:
        main["bounding_box"] = [round(float(value) * scale, 7) for value in original]
        if "scale" in main:
            main["scale"] = list(main["bounding_box"])
    main["orientation"] = [0.0, 0.0, 0.7071068, 0.7071068] if yaw else [0.0, 0.0, 0.0, 1.0]
    return {
        "task_type": task_type,
        "margin": margin,
        "yaw_degrees": yaw,
        "main_bbox_before_capacity_fit": original,
        "main_bbox_after_capacity_fit": list(main["bounding_box"]),
        "target_bbox": list(target["bounding_box"]),
        "uniform_scale": round(scale, 5),
    }


def place_initial_relation(
    main: dict[str, object],
    source: dict[str, object],
    relation: str,
) -> dict[str, object]:

    source_x, source_y, source_z = (float(value) for value in source["relative_bbox_position"])
    main_height = float(main["bounding_box"][2])
    source_height = float(source["bounding_box"][2])
    if relation == "on_top":
        z = source_z + source_height / 2 + main_height / 2 + RELATION_CLEARANCE
    elif relation == "inside":
        dimensions = [float(value) for value in main["bounding_box"]]
        longest_axis = max(range(3), key=dimensions.__getitem__)
        if longest_axis == 0 and dimensions[0] > 2 * max(dimensions[1:]):
            # Pens and markers must be inserted lengthwise; horizontal placement at the rim is not containment.
            main["orientation"] = [0.0, 0.7071068, 0.0, 0.7071068]
            vertical_extent = dimensions[0]
        elif longest_axis == 1 and dimensions[1] > 2 * max(dimensions[0], dimensions[2]):
            main["orientation"] = [0.7071068, 0.0, 0.0, 0.7071068]
            vertical_extent = dimensions[1]
        else:
            vertical_extent = main_height
        # Put the lower quarter inside while leaving a graspable portion above the opening.
        source_top = source_z + source_height / 2
        z = max(vertical_extent / 2 + SUPPORT_CLEARANCE, source_top + vertical_extent / 4)
    else:
        raise ValueError(f"unsupported initial relation {relation!r}")
    main["relative_bbox_position"] = [source_x, source_y, round(z, 7)]
    return {
        "type": relation,
        "main": main["name"],
        "source": source["name"],
        "clearance": RELATION_CLEARANCE if relation == "on_top" else None,
    }


def object_for(
    concept: str,
    instruction: str,
    occurrence: int,
    assets_by_category: dict[str, list[dict[str, object]]],
    max_xy: tuple[float, float],
) -> tuple[dict[str, object], dict[str, object] | None]:

    if concept in {"block", "object"}:
        config = primitive(concept, instruction, occurrence)
        config["orientation"] = [0.0, 0.0, 0.0, 1.0]
        return config, None
    config = dataset_object(concept, assets_by_category, prefer_large=occurrence > 0)
    original = list(config["bounding_box"])
    fitted, scale = fit_bbox(original, max_xy)
    config["bounding_box"] = fitted
    config["orientation"] = [0.0, 0.0, 0.0, 1.0]
    audit = None if scale == 1 else {"name": config["name"], "original_bbox": original, "authored_bbox": fitted, "scale": round(scale, 5)}
    return config, audit


def distractor_family(category: str) -> str:

    for prefix in ("bottle_of_", "jar_of_", "can_of_", "box_of_", "bag_of_"):
        if category.startswith(prefix):
            return prefix.removesuffix("_of_")
    return category


def sample_distractors(
    assets_by_category: dict[str, list[dict[str, object]]],
    eligible_categories: list[str],
    excluded_categories: set[str],
    category_usage: Counter[str],
    family_usage: Counter[str],
    rng: random.Random,
) -> list[dict[str, object]]:

    ties = {category: rng.random() for category in eligible_categories}
    ranked = sorted(
        (category for category in eligible_categories if category not in excluded_categories),
        key=lambda category: (
            family_usage[distractor_family(category)], category_usage[category], ties[category]
        ),
    )
    chosen = []
    chosen_families = set()
    for category in ranked:
        family = distractor_family(category)
        if family in chosen_families:
            continue
        config = dataset_object(category, assets_by_category)
        config["name"] = f"distractor_{category}"
        config["bounding_box"], _ = fit_bbox(config["bounding_box"], (0.10, 0.10))
        config["orientation"] = [0.0, 0.0, 0.0, 1.0]
        chosen.append(config)
        chosen_families.add(family)
    if len(chosen) < 3:
        raise ValueError("fewer than three distinct distractor families are available")
    return chosen


def generate(
    source: Path,
    dataset: Path,
    output: Path,
    camera_extrinsics: Path = DEFAULT_CAMERA_EXTRINSICS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:

    scene_rng = random.Random(seed)
    camera_rng = random.Random(seed + 1)
    distractor_rng = random.Random(seed + 2)
    selection = json.loads(source.read_text(encoding="utf-8"))
    indexed = discover_assets(dataset)
    assets_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for asset in indexed:
        assets_by_category[str(asset["category"])].append(asset)
    regions = [
        item
        for item in load_scene_regions(REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml")
        if item["width"] >= 0.4 and item["depth"] >= 0.4 and item["z"] > 0
        and (item["scene"], item["support"]) not in UNSAFE_SCENE_REGIONS
    ]
    if not regions:
        raise ValueError("no tabletop scene region is large enough for DROID100 layouts")
    camera_poses = load_camera_extrinsics(camera_extrinsics)
    droid_categories = load_droid_categories(REPO_ROOT / "realm" / "config" / "objects" / "categories.yaml")
    eligible_distractors = []
    for category in droid_categories:
        candidates = assets_by_category.get(category, [])
        if not candidates:
            continue
        smallest = min(candidates, key=lambda item: math.prod(item["bbox"][:2]))
        if max(float(value) for value in smallest["bbox"][:2]) <= 0.24 and float(smallest["bbox"][2]) <= 0.35:
            eligible_distractors.append(category)
    if len({distractor_family(category) for category in eligible_distractors}) < 3:
        raise ValueError("DROID whitelist has fewer than three usable distractor families")
    scene_order = list(regions)
    scene_rng.shuffle(scene_order)
    category_usage: Counter[str] = Counter()
    family_usage: Counter[str] = Counter()
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for task_index, task in enumerate(selection["tasks"]):
        if task_index and task_index % len(scene_order) == 0:
            scene_rng.shuffle(scene_order)
        region = scene_order[task_index % len(scene_order)]
        elliptical = region["support"] in ELLIPTICAL_SUPPORTS
        rank = int(task["rank"])
        sampled_cameras = sample_camera_pair(camera_poses, camera_rng)
        reviewed_camera_sources = REVIEWED_CAMERA_SOURCES.get(rank)
        if reviewed_camera_sources:
            sampled_cameras = {
                f"cam{index}": {**camera_poses[source_name], "source": source_name}
                for index, source_name in enumerate(reviewed_camera_sources, start=1)
            }
        camera_sources = {key: value["source"] for key, value in sampled_cameras.items()}
        cameras = {
            key: {pose_key: pose_value for pose_key, pose_value in value.items() if pose_key != "source"}
            for key, value in sampled_cameras.items()
        }
        original_instruction = task["instruction"]
        instruction, task_type, reviewed_override = reviewed_task(task)
        initial_relation = initial_relation_type(instruction, task_type)
        terms = instruction_terms(instruction, task_type)
        resolved = concepts(instruction, task_type)
        configs, resize_audit, placed = [], [], []
        for index, concept in enumerate(resolved):
            max_xy = (0.14, 0.16) if index == 0 else (0.17, 0.17)
            config, audit = object_for(
                concept, instruction, index, assets_by_category,
                max_xy,
            )
            config["name"] = f"{config['name']}_{index + 1}" if resolved.count(concept) > 1 else config["name"]
            model_override = REVIEWED_MODEL_OVERRIDES.get(rank, {}).get(str(config["name"]))
            if model_override:
                audit = apply_model_override(config, model_override, assets_by_category, max_xy)
            configs.append(config)
            if audit:
                resize_audit.append(audit)
        receiver_capacity = None
        if len(configs) > 1:
            capacity_type = "stack" if initial_relation == "on_top" else "put" if initial_relation else task_type
            receiver_capacity = ensure_receiver_capacity(configs[0], configs[1], capacity_type)
            if receiver_capacity["uniform_scale"] < 1:
                resize_audit.append({
                    "name": configs[0]["name"],
                    "reason": "receiver_capacity",
                    "original_bbox": receiver_capacity["main_bbox_before_capacity_fit"],
                    "authored_bbox": receiver_capacity["main_bbox_after_capacity_fit"],
                    "scale": receiver_capacity["uniform_scale"],
                })
        relation_audit = None
        if initial_relation:
            place(configs[1], placed, region["width"], region["depth"], elliptical=elliptical)
            relation_audit = place_initial_relation(configs[0], configs[1], initial_relation)
            placed.append(configs[0])
        else:
            # Pack the receiver/support first; placing a small main object centrally can
            # otherwise strand a large bowl or pot despite ample free support area.
            for config in sorted(configs, key=lambda item: math.prod(item["bounding_box"][:2]), reverse=True):
                place(config, placed, region["width"], region["depth"], elliptical=elliptical)
        excluded = {str(config.get("category")) for config in configs if config.get("category")}
        distractor_candidates = sample_distractors(
            assets_by_category, eligible_distractors, excluded,
            category_usage, family_usage, distractor_rng,
        )
        distractors = []
        for distractor in distractor_candidates:
            try:
                place(distractor, placed, region["width"], region["depth"], elliptical=elliptical)
            except ValueError:
                continue
            category = str(distractor["category"])
            category_usage[category] += 1
            family_usage[distractor_family(category)] += 1
            distractors.append(distractor)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            raise ValueError(f"could not place three distractors for rank {task['rank']}")
        position_overrides = REVIEWED_POSITION_OVERRIDES.get(rank, {})
        if position_overrides:
            by_name = {str(item["name"]): item for item in configs + distractors}
            for name, xy in position_overrides.items():
                config = by_name[name]
                config["relative_bbox_position"][:2] = xy
        verb = {"put": "put", "pick": "pick", "stack": "stack", "rotate": "rotate"}[task_type]
        document = {
            "task": {"type": "DummyTask", "termination_config": {}, "reward_config": {}},
            "task_type": task_type,
            "instruction": instruction,
            "instruction_obj_to_replace": terms[0],
            "instruction_target_to_replace": terms[1] if len(terms) > 1 else "",
            "instruction_verb_to_replace": verb,
            "supported_scenes": {region["scene"]: [region["support"]]},
            "camera_extrinsics": cameras,
            "main_objects": [configs[0]],
            "target_objects": [configs[1]] if len(configs) > 1 and not initial_relation else [],
            "distractors": distractors,
            "immutables": [configs[1]] if initial_relation else [],
        }
        # Keep reviewed tasks at their established paths so regeneration cannot leave stale duplicates.
        directory_name = f"{task['rank']:03d}_{slug(original_instruction)[:72]}"
        task_directory = output / directory_name
        task_directory.mkdir(exist_ok=True)
        (task_directory / "default.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False, width=120), encoding="utf-8"
        )
        generated.append({
            "rank": task["rank"],
            "directory": directory_name,
            "task_type": task_type,
            "instruction": instruction,
            "original_instruction": original_instruction if reviewed_override else None,
            "reviewed_override": reviewed_override,
            "concepts": resolved,
            "scene": {
                "name": region["scene"],
                "support": region["support"],
                "width": region["width"],
                "depth": region["depth"],
                "footprint": "ellipse" if elliptical else "rectangle",
            },
            "distractor_categories": [item["category"] for item in distractors],
            "camera_extrinsic_sources": camera_sources,
            "camera_extrinsics": cameras,
            "resized_assets": resize_audit,
            "receiver_capacity": receiver_capacity,
            "initial_relation": relation_audit,
            "render_review_overrides": {
                "camera_sources": list(reviewed_camera_sources) if reviewed_camera_sources else None,
                "model": REVIEWED_MODEL_OVERRIDES.get(rank),
                "positions": REVIEWED_POSITION_OVERRIDES.get(rank),
            },
        })
    audit = {
        "family": "REALM_DROID100",
        "source": str(source),
        "dataset": str(dataset),
        "seed": seed,
        "scene_pool_size": len(regions),
        "excluded_scene_regions": [list(item) for item in sorted(UNSAFE_SCENE_REGIONS)],
        "distractor_pool_size": len(eligible_distractors),
        "distractor_category_usage": dict(sorted(category_usage.items())),
        "distractor_family_usage": dict(sorted(family_usage.items())),
        "camera_pose_source": str(camera_extrinsics),
        "camera_pose_pool_size": len(camera_poses),
        "generated_count": len(generated),
        "semantic_review": {
            "reviewed_count": len(generated),
            "reviewed_on": "2026-08-24",
            "checks": [
                "instruction-object closure",
                "single-main and task-type contract",
                "compound noun grounding",
                "initial spatial predicates",
                "receiver/support capacity",
                "orientation and support clearance",
            ],
            "remaining_runtime_checks": [
                "mesh-level containment and contact stability",
                "robot reachability",
                "texture-dependent color and material descriptions",
            ],
        },
        "tasks": generated,
    }
    (output / "generation_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-extrinsics", type=Path, default=DEFAULT_CAMERA_EXTRINSICS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate(
        args.source, args.dataset, args.output,
        camera_extrinsics=args.camera_extrinsics, seed=args.seed,
    )
    print(f"Generated {result['generated_count']} configs in {args.output}")


if __name__ == "__main__":
    main()
