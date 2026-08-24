"""Filesystem and review helpers for the DROID100 frame browser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PANEL_NAMES = ("panel.jpg", "panel.jpeg")
REVIEW_FILE = "frame_review.json"


@dataclass(frozen=True)
class Frame:
    """One rendered task panel within a log run."""

    run: str
    task: str
    rank: int | None
    path: Path


def task_rank(task: str) -> int | None:
    """Extract a numeric DROID100 rank from a task-directory name."""
    prefix = task.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def discover_frames(root: Path) -> list[Frame]:
    """Index JPG and JPEG panels from every first-frame run."""
    if not root.is_dir():
        return []
    frames = []
    for panel_name in PANEL_NAMES:
        for path in root.glob(f"*/frames/*/{panel_name}"):
            task_dir = path.parent
            run_dir = task_dir.parent.parent
            frames.append(Frame(run_dir.name, task_dir.name, task_rank(task_dir.name), path))
    return sorted(frames, key=lambda item: (item.run, item.rank is None, item.rank or 0, item.task))


def load_reviews(run_dir: Path) -> dict[str, dict[str, str]]:
    """Read optional local review annotations for a run."""
    path = run_dir / REVIEW_FILE
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def save_reviews(run_dir: Path, reviews: dict[str, dict[str, str]]) -> None:
    """Atomically save local review annotations without changing render artifacts."""
    path = run_dir / REVIEW_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(reviews, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def display_name(task: str) -> str:
    """Format a task-directory name for display while preserving its rank."""
    return task.replace("_", " ")
