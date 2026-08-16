"""Which exterior camera a rollout hands the policy, and when.

`realm/rollout.py::wants_base_im_second` decides whether a control step sends the policy
`external_sensor1` (the second exterior view) instead of `external_sensor0`. The DROID policies
take exactly one exterior image; REALM frames the two drawer tasks better on the second camera, so
those two task types -- and only those -- select it.

WHAT THIS PINS, and why each half matters
-----------------------------------------
1. THE TASK TYPES ARE `open_drawer` / `close_drawer`. Until 2026-08-16 the call site compared
   `task_type == "open_close_drawer"`, a value no task config declares in this checkout or in the
   pre-port 1.1.1 one, so the second camera was never selected for anything. The tuple is
   cross-checked against the task configs here so the code and the configs cannot drift apart
   silently the way they already did once.

2. A None SECOND IMAGE MUST NOT BE SELECTED. `extract_from_obs` returns None for `base_im_second`
   whenever the observation carries no `external_sensor1`, which is every run without
   `--multi-view`. `InferenceClient.infer`'s openpi path does
   `img_to_use = base_im_second if use_base_im_second else base_im` and then
   `image_tools.resize_with_pad(img_to_use, 224, 224)`, so selecting None is a crash, not a
   fallback. Fixing only the string -- without this guard -- would have converted a silent no-op
   into a TypeError on exactly the two tasks the fix was for.

Needs the container (importing `realm.rollout` pulls in omnigibson) but NO GPU and no simulator:
nothing here builds an environment. The static counterpart, which needs neither, is
tests/test_task_type_literals.py.

    ./scripts/clara/interactive/rr python -u tests/test_rollout_camera_selection.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.rollout import DRAWER_TASK_TYPES, wants_base_im_second

IMAGE = np.zeros((128, 128, 3), dtype=np.uint8)   # stands in for a rendered frame


def declared_task_types():
    """Every `task_type` the task configs declare, so the tuple above cannot drift from them."""
    declared = {}
    for path in sorted(PROJECT_ROOT.glob("realm/config/tasks/**/*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("task_type"), str):
            declared.setdefault(cfg["task_type"], []).append(path.name)
    return declared


def main():
    declared = declared_task_types()
    failures = []

    print(f"DRAWER_TASK_TYPES = {DRAWER_TASK_TYPES}")
    print(f"declared task types = {sorted(declared)}\n")

    # ---- 1: the tuple names real, declared task types, and exactly the joint-articulating ones --
    undeclared = [t for t in DRAWER_TASK_TYPES if t not in declared]
    print(f"[1] DRAWER_TASK_TYPES entries no task config declares: "
          f"{undeclared if undeclared else 'none'}")
    for t in undeclared:
        failures.append(f"[1] DRAWER_TASK_TYPES names {t!r}, which no task config declares. "
                        f"Declared: {sorted(declared)}")
    missing = [t for t in declared if "drawer" in t and t not in DRAWER_TASK_TYPES]
    for t in missing:
        failures.append(f"[1] task type {t!r} looks like a drawer task but is not in "
                        f"DRAWER_TASK_TYPES, so it will not get the second camera")

    # ---- 2: the truth table -------------------------------------------------------------------
    # (task_type, base_im_second, expected). Every declared task type appears, so a future task
    # type added to the configs but not considered here shows up as an uncovered case in [3].
    cases = [(t, IMAGE, t in DRAWER_TASK_TYPES) for t in sorted(declared)]
    cases += [
        # The guard: a drawer task WITHOUT a second camera must not select the missing image.
        ("open_drawer", None, False),
        ("close_drawer", None, False),
        # getattr(env, "task_type", None) returns None for an env that has no task_type at all.
        (None, IMAGE, False),
        (None, None, False),
        # The string that was compared against until 2026-08-16 is not a task type and must not
        # select anything, so a half-revert cannot pass this test.
        ("open_close_drawer", IMAGE, False),
    ]

    print("[2] truth table:")
    for task_type, second, expected in cases:
        got = wants_base_im_second(task_type, second)
        ok = bool(got) == expected
        print(f"    task_type={str(task_type):<18} second={'image' if second is not None else 'None':<5} "
              f"-> {str(bool(got)):<5} expected {str(expected):<5} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"[2] wants_base_im_second({task_type!r}, "
                            f"{'<image>' if second is not None else None}) returned {got!r}, "
                            f"expected {expected}")

    # ---- 3: every declared task type is covered ------------------------------------------------
    covered = {c[0] for c in cases}
    uncovered = sorted(t for t in declared if t not in covered)
    print(f"\n[3] declared task types not covered by the truth table: "
          f"{uncovered if uncovered else 'none'}")
    for t in uncovered:
        failures.append(f"[3] declared task type {t!r} is not exercised by the truth table")

    # ---- 4: negative control -- the table must REJECT the predicate this replaced ---------------
    # Without this, [2] proves only "the current code agrees with the current expectations". The
    # two predicates that were live or plausible before 2026-08-16 are evaluated inline here and
    # must both disagree with the table, so a revert or a half-fix cannot pass this file.
    #
    #   old      -- what realm/rollout.py:314 actually did: a string no config declares.
    #   unguarded -- the string fixed but base_im_second not checked for None. This one is worse
    #                than the bug it replaces: it hands resize_with_pad(None) to the openpi path.
    def old(task_type, second):
        return task_type == "open_close_drawer"

    def unguarded(task_type, second):
        return task_type in DRAWER_TASK_TYPES

    print("\n[4] negative control -- rejected predicates:")
    for label, pred, must_differ_on in (
            ("pre-fix (== 'open_close_drawer')", old, ("open_drawer", IMAGE)),
            ("string-only fix (no None guard)", unguarded, ("open_drawer", None))):
        differs = [(t, s) for t, s, expected in cases if bool(pred(t, s)) != expected]
        agrees_where_it_must_not = bool(pred(*must_differ_on)) == wants_base_im_second(*must_differ_on)
        print(f"    {label:<36} disagrees with the table on {len(differs)} case(s)")
        if not differs:
            failures.append(
                f"[4] the rejected predicate {label!r} satisfies the whole truth table, so [2] "
                f"cannot tell the fix from the defect. Strengthen the table.")
        if agrees_where_it_must_not:
            failures.append(
                f"[4] the rejected predicate {label!r} agrees with the fixed one on "
                f"{must_differ_on[0]!r} / "
                f"{'image' if must_differ_on[1] is not None else None}, which is the exact case "
                f"it was supposed to get wrong.")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- the drawer task types, and only those, select the second exterior camera, "
              "and never when it is absent")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
