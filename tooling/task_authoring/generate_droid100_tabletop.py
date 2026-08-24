"""Generate grounded REALM configs from the filtered DROID100 tabletop manifest."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from tooling.task_authoring.authoring import (
    discover_assets,
    load_camera_extrinsics,
    load_scene_regions,
    sample_opposite_camera_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "data" / "droid" / "DROID100_tabletop.json"
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets_og391" / "behavior-1k-assets"
DEFAULT_OUTPUT = REPO_ROOT / "realm" / "config" / "tasks" / "DROID100_tabletop"
DEFAULT_CAMERA_EXTRINSICS = (
    REPO_ROOT / "realm" / "config" / "env" / "external_sensors" / "camera_extrinsics_droid_realm.yaml"
)
DEFAULT_SEED = 100
DISTRACTOR_CATEGORIES = ("teaspoon", "masking_tape", "marker", "pen", "can_of_soda", "half_apple")
SUPPORT_CLEARANCE = 0.05
RELATION_CLEARANCE = 0.01
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
    """Return a stable config-directory component."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def reviewed_task(task: dict[str, object]) -> tuple[str, str, dict[str, str] | None]:
    """Apply human-reviewed semantic corrections while retaining their provenance."""
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
    """Uniformly shrink an asset to the authored tabletop footprint when necessary."""
    scale = min(1.0, max_xy[0] / values[0], max_xy[1] / values[1])
    return [round(float(value) * scale, 7) for value in values], scale


def sample_camera_pair(
    poses: dict[str, dict[str, list[float]]], rng: random.Random
) -> dict[str, dict[str, list[float]]]:
    """Sample robot-relative DROID cameras on opposite sides of the robot."""
    sampled = sample_opposite_camera_pair(poses, rng)
    pair = list(sampled.items())
    return {
        "cam1": {**pair[0][1], "source": pair[0][0]},
        "cam2": {**pair[1][1], "source": pair[1][0]},
    }


def instruction_terms(instruction: str, task_type: str) -> list[str]:
    """Extract literal ordered object phrases for semantic substitution fields."""
    found = [match.group(1).lower() for match in CONCEPT_PATTERN.finditer(instruction)]
    required = 2 if task_type in {"put", "stack"} or initial_relation_type(instruction, task_type) else 1
    if task_type == "stack" and len(found) == 1 and found[0] == "cups":
        found.append("cups")
    if len(found) < required:
        raise ValueError(f"could not resolve {required} object terms from {instruction!r}")
    return found[:required]


def initial_relation_type(instruction: str, task_type: str) -> str | None:
    """Return the initial spatial predicate implied by a pick instruction."""
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
    """Normalize literal instruction terms into asset-grounding concepts."""
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
    """Create a colored block for generic block/object language."""
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
    """Choose a compact movable model or a roomy receiving/support model."""
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


def overlaps(candidate: dict[str, object], placed: list[dict[str, object]], margin: float = 0.012) -> bool:
    """Return whether candidate overlaps an already placed XY bounding box."""
    x, y, _ = candidate["relative_bbox_position"]
    bx, by, _ = candidate["bounding_box"]
    return any(
        abs(x - other["relative_bbox_position"][0]) < (bx + other["bounding_box"][0]) / 2 + margin
        and abs(y - other["relative_bbox_position"][1]) < (by + other["bounding_box"][1]) / 2 + margin
        for other in placed
    )


def place(config: dict[str, object], placed: list[dict[str, object]], width: float, depth: float) -> None:
    """Place a resized bbox at the first collision-free normalized candidate."""
    candidates = (
        (0.22, 0.22), (0.72, 0.68), (0.20, 0.72), (0.76, 0.22),
        (0.48, 0.88), (0.48, 0.43), (0.18, 0.46), (0.82, 0.47),
        (0.35, 0.65), (0.64, 0.84), (0.35, 0.10), (0.65, 0.10),
    )
    bx, by, _ = config["bounding_box"]
    for ux, uy in candidates:
        x = max(bx / 2, min(width - bx / 2, ux * width))
        y = max(by / 2, min(depth - by / 2, uy * depth))
        z = float(config["bounding_box"][2]) / 2 + SUPPORT_CLEARANCE
        authored_z = math.ceil(z * 10_000_000) / 10_000_000
        config["relative_bbox_position"] = [round(x, 5), round(y, 5), authored_z]
        if not overlaps(config, placed):
            placed.append(config)
            return
    raise ValueError(f"no collision-free placement for {config['name']!r}")


def ensure_receiver_capacity(
    main: dict[str, object],
    target: dict[str, object],
    task_type: str,
) -> dict[str, object]:
    """Make the main footprint compatible with its receiver/support using uniform scaling and yaw."""
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
    """Place a pick object in the source/support state required by its instruction."""
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
    """Ground one concept and return its config plus optional resize audit."""
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


def generate(
    source: Path,
    dataset: Path,
    output: Path,
    camera_extrinsics: Path = DEFAULT_CAMERA_EXTRINSICS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Generate the full task family and return its audit manifest."""
    rng = random.Random(seed)
    selection = json.loads(source.read_text(encoding="utf-8"))
    indexed = discover_assets(dataset)
    assets_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for asset in indexed:
        assets_by_category[str(asset["category"])].append(asset)
    regions = [
        item
        for item in load_scene_regions(REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml")
        if item["width"] >= 0.4 and item["depth"] >= 0.4 and item["z"] > 0
    ]
    if not regions:
        raise ValueError("no tabletop scene region is large enough for DROID100 layouts")
    camera_poses = load_camera_extrinsics(camera_extrinsics)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for task in selection["tasks"]:
        region = rng.choice(regions)
        sampled_cameras = sample_camera_pair(camera_poses, rng)
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
            config, audit = object_for(
                concept, instruction, index, assets_by_category,
                (0.14, 0.16) if index == 0 else (0.22, 0.22),
            )
            config["name"] = f"{config['name']}_{index + 1}" if resolved.count(concept) > 1 else config["name"]
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
            place(configs[1], placed, region["width"], region["depth"])
            relation_audit = place_initial_relation(configs[0], configs[1], initial_relation)
            placed.append(configs[0])
        else:
            for config in configs:
                place(config, placed, region["width"], region["depth"])
        distractors = []
        for category in DISTRACTOR_CATEGORIES:
            if category in {config.get("category") for config in configs}:
                continue
            try:
                distractor = dataset_object(category, assets_by_category)
                distractor["name"] = f"distractor_{category}"
                distractor["bounding_box"], _ = fit_bbox(distractor["bounding_box"], (0.10, 0.10))
                distractor["orientation"] = [0.0, 0.0, 0.0, 1.0]
                place(distractor, placed, region["width"], region["depth"])
            except ValueError:
                continue
            distractors.append(distractor)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            raise ValueError(f"could not place three distractors for rank {task['rank']}")
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
            },
            "camera_extrinsic_sources": camera_sources,
            "camera_extrinsics": cameras,
            "resized_assets": resize_audit,
            "receiver_capacity": receiver_capacity,
            "initial_relation": relation_audit,
        })
    audit = {
        "family": "DROID100_tabletop",
        "source": str(source),
        "dataset": str(dataset),
        "seed": seed,
        "scene_pool_size": len(regions),
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
