"""Convert raw DROID world-to-camera solves into filtered REALM camera poses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tooling.task_authoring.authoring import load_camera_extrinsics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "realm/config/env/external_sensors/camera_extrinsics_droid_realm.yaml"
)
DEFAULT_REJECTIONS = (
    REPO_ROOT / "realm/config/env/external_sensors/camera_extrinsics_droid_realm_rejected.json"
)
NAME_PATTERN = re.compile(r"^droid_v2_(ep_\d+)_(cam[12])$")
CV_TO_OG = np.diag([1.0, -1.0, -1.0])
WORKSPACE_PROBE = np.array([0.55, 0.0, 0.10])


def convert_pose(raw: dict[str, list[float]]) -> dict[str, list[float]]:
    """Invert raw T_cam_base and convert its local CV axes to OmniGibson axes."""
    rotation_cam_base = Rotation.from_quat(raw["rot"]).as_matrix()
    translation_cam_base = np.asarray(raw["pos"], dtype=float)
    rotation_base_cam = rotation_cam_base.T
    position_base_cam = -(rotation_base_cam @ translation_cam_base)
    quaternion = Rotation.from_matrix(rotation_base_cam @ CV_TO_OG).as_quat()
    if quaternion[3] < 0:
        quaternion *= -1
    return {
        "pos": [round(float(value), 7) for value in position_base_cam],
        "rot": [round(float(value), 7) for value in quaternion],
    }


def rejection_reasons(pair: dict[str, dict[str, list[float]]]) -> list[str]:
    """Return conservative geometry failures for one converted physical pair."""
    reasons = []
    positions = []
    for camera_name, pose in pair.items():
        position = np.asarray(pose["pos"], dtype=float)
        rotation = Rotation.from_quat(pose["rot"])
        distance = float(np.linalg.norm(position))
        target = WORKSPACE_PROBE - position
        forward = rotation.apply([0.0, 0.0, -1.0])
        up = rotation.apply([0.0, 1.0, 0.0])
        facing_cos = float(np.dot(forward, target) / np.linalg.norm(target))
        if not 0.2 <= distance <= 3.0:
            reasons.append(f"{camera_name}:distance={distance:.6f}")
        if not -0.2 <= position[2] <= 2.5:
            reasons.append(f"{camera_name}:height={position[2]:.6f}")
        if facing_cos <= 0:
            reasons.append(f"{camera_name}:facing_cos={facing_cos:.6f}")
        if up[2] < -0.30:
            reasons.append(f"{camera_name}:upright_cos={up[2]:.6f}")
        positions.append(position)
    separation = float(np.linalg.norm(positions[0] - positions[1]))
    if not 0.05 <= separation <= 4.0:
        reasons.append(f"pair_separation={separation:.6f}")
    return reasons


def convert(source: Path, output: Path, rejection_output: Path) -> dict[str, int]:
    """Convert, filter, and write a REALM-ready paired pose catalogue."""
    raw_poses = load_camera_extrinsics(source)
    episodes: dict[str, dict[str, dict[str, list[float]]]] = {}
    for name, pose in raw_poses.items():
        match = NAME_PATTERN.fullmatch(name)
        if match:
            episodes.setdefault(match.group(1), {})[match.group(2)] = convert_pose(pose)

    accepted = {}
    rejected = []
    for episode, pair in sorted(episodes.items()):
        if set(pair) != {"cam1", "cam2"}:
            rejected.append({"episode": episode, "reasons": ["incomplete_pair"]})
            continue
        reasons = rejection_reasons(pair)
        if reasons:
            rejected.append({"episode": episode, "reasons": reasons})
            continue
        for camera_name in ("cam1", "cam2"):
            accepted[f"droid_realm_{episode}_{camera_name}"] = pair[camera_name]

    lines = [
        "# REALM-ready DROID external-camera poses.",
        f"# Generated from {source.name}; source stores T_cam_base and remains unchanged.",
        "# Values here are T_base_camera with OmniGibson -Z-forward, +Y-up local axes.",
        "# Poses are robot-base-relative XYZW and must still be composed with the scene robot pose.",
        "",
    ]
    for name, pose in accepted.items():
        pos = ", ".join(f"{value:.7f}" for value in pose["pos"])
        rot = ", ".join(f"{value:.7f}" for value in pose["rot"])
        lines.extend((f"{name}:", f"  pos: [{pos}]", f"  rot: [{rot}]", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    rejection_output.write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")
    return {
        "source_poses": len(raw_poses),
        "source_episodes": len(episodes),
        "accepted_episodes": len(accepted) // 2,
        "accepted_poses": len(accepted),
        "rejected_episodes": len(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the archived raw T_cam_base YAML; it is intentionally not kept in REALM.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    args = parser.parse_args()
    result = convert(args.source, args.output, args.rejections)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
