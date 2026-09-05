"""Per-task mapping of Robometer's raw progress score to REALM's 0-1 task progression.

Pure and host-importable (no omnigibson, no torch), like realm/geometry.py, so the calibration file
and the arithmetic are checked in tier 1. realm/progress_scorer.py applies it inside the container.

    table = load_calibration()                       # realm/config/robometer_calibration.yaml
    cal = calibration_for("put_banana_into_box_default_cola", table)   # -> the banana entry
    cal.apply(0.35)                                  # -> 0.5 with floor 0.0, ceiling 0.7

The file's contract is in its header comment. In one line: calibrated = clip((raw - floor) /
(ceiling - floor), 0, 1), so 1.0 means "raw reached the task's ceiling", which is the success
condition. A task without an entry gets the identity (floor 0, ceiling 1) and `calibrated=False`.
"""
import os
from dataclasses import dataclass

import yaml

DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "config", "robometer_calibration.yaml")


@dataclass(frozen=True)
class TaskCalibration:
    task: str          # the entry that matched, or the queried task name when nothing did
    floor: float
    ceiling: float
    calibrated: bool   # False = identity fallback; the task cannot reach success

    def apply(self, raw):
        if self.ceiling <= self.floor:
            raise ValueError(f"calibration for {self.task!r} has ceiling {self.ceiling} <= floor {self.floor}")
        return float(min(1.0, max(0.0, (float(raw) - self.floor) / (self.ceiling - self.floor))))


IDENTITY = dict(floor=0.0, ceiling=1.0)


def load_calibration(path=DEFAULT_CALIBRATION_PATH):
    """{task_name: {"floor": f, "ceiling": c}} from the YAML; validated so a typo fails at load,
    before the simulator boots, rather than as a silent identity at the first query."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    tasks = data.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise ValueError(f"{path}: `tasks` must be a mapping of task name -> {{ceiling, floor}}")
    table = {}
    for name, entry in tasks.items():
        entry = entry or {}
        if "ceiling" not in entry:
            raise ValueError(f"{path}: task {name!r} has no `ceiling`")
        floor = float(entry.get("floor", 0.0))
        ceiling = float(entry["ceiling"])
        if not (0.0 <= floor < ceiling <= 1.0):
            raise ValueError(f"{path}: task {name!r} needs 0 <= floor < ceiling <= 1, "
                             f"got floor={floor} ceiling={ceiling}")
        table[str(name)] = dict(floor=floor, ceiling=ceiling)
    return table


def calibration_for(task, table):
    """The entry for `task`: an exact match, else the LONGEST entry name that is a prefix of `task`
    (so `put_banana_into_box_default_cola`, a non-default config of the banana task, uses the banana
    entry), else the identity with calibrated=False."""
    task = str(task)
    if task in table:
        return TaskCalibration(task, table[task]["floor"], table[task]["ceiling"], True)
    prefixes = [name for name in table if task.startswith(name + "_")]
    if prefixes:
        name = max(prefixes, key=len)
        return TaskCalibration(name, table[name]["floor"], table[name]["ceiling"], True)
    return TaskCalibration(task, IDENTITY["floor"], IDENTITY["ceiling"], False)
