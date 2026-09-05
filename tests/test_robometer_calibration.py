"""realm/config/robometer_calibration.yaml names real tasks with sane numbers, and the calibration
arithmetic maps raw Robometer scores onto 0-1 the way the docs say.

WHY THIS EXISTS
---------------
Under --robometer, task_progression = clip((raw - floor) / (ceiling - floor), 0, 1) and success is
calibrated 1.0. The ceiling per task is a hand-entered number in a YAML file, and a typo there --
a task name that matches no config, a ceiling of 8.0, a floor above the ceiling -- would either make
a task silently unreachable (identity fallback) or blow up at the first query, after an Isaac boot.
This pins the file against the task configs and the arithmetic against worked examples, on the host,
in well under a second.

    python3 tests/test_robometer_calibration.py
"""
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.robometer_calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    TaskCalibration,
    calibration_for,
    load_calibration,
)


def declared_tasks():
    """Task names REALM knows: the directory of every task config (that is what resolve_task and
    the report's `task` column use; the config's own `task:` key is OmniGibson's task block)."""
    names = set()
    for path in PROJECT_ROOT.glob("realm/config/tasks/**/*.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("task_type"), str):
            names.add(path.parent.name)
    return names


def main():
    failures = []

    def check(cell, cond, detail):
        print(f"[{cell}] {detail}: {'ok' if cond else 'FAIL'}")
        if not cond:
            failures.append(f"[{cell}] {detail}")

    # [1] the shipped file ------------------------------------------------------------------------
    table = load_calibration(DEFAULT_CALIBRATION_PATH)
    known = declared_tasks()
    unknown = sorted(t for t in table if t not in known)
    check(1, len(table) >= 1, f"the file has entries ({sorted(table)})")
    check(1, not unknown, f"every entry names a task some config declares (unknown: {unknown})")
    check(1, all(0.0 <= e["floor"] < e["ceiling"] <= 1.0 for e in table.values()),
          "every entry has 0 <= floor < ceiling <= 1")

    # [2] arithmetic -------------------------------------------------------------------------------
    cal = TaskCalibration("t", floor=0.0, ceiling=0.7, calibrated=True)
    check(2, abs(cal.apply(0.35) - 0.5) < 1e-9 and cal.apply(0.7) == 1.0 and cal.apply(0.95) == 1.0
          and cal.apply(0.0) == 0.0 and cal.apply(-0.2) == 0.0,
          "floor 0 / ceiling 0.7: 0.35 -> 0.5, ceiling and above -> 1.0, below floor -> 0.0")
    cal = TaskCalibration("t", floor=0.2, ceiling=0.7, calibrated=True)
    check(2, abs(cal.apply(0.45) - 0.5) < 1e-9 and cal.apply(0.2) == 0.0 and cal.apply(0.1) == 0.0,
          "floor 0.2 / ceiling 0.7: 0.45 -> 0.5, floor and below -> 0.0")
    ident = TaskCalibration("t", 0.0, 1.0, False)
    check(2, ident.apply(0.83) == 0.83 and ident.apply(1.0) == 1.0, "identity leaves raw unchanged")
    try:
        TaskCalibration("t", 0.7, 0.7, True).apply(0.5)
        check(2, False, "ceiling <= floor raises")
    except ValueError:
        check(2, True, "ceiling <= floor raises")

    # [3] lookup -----------------------------------------------------------------------------------
    t = {"put_banana_into_box": dict(floor=0.0, ceiling=0.7), "put": dict(floor=0.0, ceiling=0.5)}
    exact = calibration_for("put_banana_into_box", t)
    check(3, exact.calibrated and exact.ceiling == 0.7 and exact.task == "put_banana_into_box",
          "exact task name matches its entry")
    variant = calibration_for("put_banana_into_box_default_cola", t)
    check(3, variant.calibrated and variant.task == "put_banana_into_box",
          "a non-default config variant falls back to the LONGEST prefix entry")
    check(3, calibration_for("put_green_block_into_bowl", t).task == "put",
          "prefix matching needs the separator underscore after the entry name")
    check(3, calibration_for("putt", t).calibrated is False, "a bare prefix without separator is not a match")
    missing = calibration_for("rotate_mug", t)
    check(3, missing.calibrated is False and missing.floor == 0.0 and missing.ceiling == 1.0
          and missing.apply(0.9) == 0.9, "an unknown task gets the identity and calibrated=False")

    # [4] validation on load -----------------------------------------------------------------------
    import tempfile
    for text, why in (("tasks:\n  pick_spoon: {floor: 0.9, ceiling: 0.7}\n", "floor above ceiling"),
                      ("tasks:\n  pick_spoon: {ceiling: 1.5}\n", "ceiling above 1"),
                      ("tasks:\n  pick_spoon: {floor: 0.1}\n", "missing ceiling"),
                      ("tasks: [pick_spoon]\n", "tasks not a mapping")):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(text)
        try:
            load_calibration(fh.name)
            check(4, False, f"{why} is rejected at load")
        except ValueError:
            check(4, True, f"{why} is rejected at load")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("tasks:\n  pick_spoon: {ceiling: 0.75}\n")
    check(4, load_calibration(fh.name) == {"pick_spoon": dict(floor=0.0, ceiling=0.75)},
          "floor defaults to 0.0")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- robometer_calibration.yaml names real tasks and the raw->0-1 mapping is as documented")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
