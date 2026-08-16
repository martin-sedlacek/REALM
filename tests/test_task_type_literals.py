"""Every string REALM compares a `task_type` against must be a `task_type` some config declares.

WHY THIS EXISTS
---------------
`realm/config/tasks/REALM_DROID10/*/default.yaml` is the only producer of `task_type`. Ten configs
declare seven distinct values. A comparison against anything else is silently constant False --
Python does not warn, the branch simply never runs, and nothing downstream can tell "this condition
is never true" apart from "this condition happens not to hold right now".

That is not hypothetical. `"open_close_drawer"` was compared against at two live sites and is not a
value any config has ever produced, in this checkout OR in the pre-port 1.1.1 tree
(`~/projects/REALM`, `realm/environments/env_base.py:234` and `realm/eval.py:422`):

  * `realm/rollout.py` -- selects the SECOND exterior camera for the drawer tasks. Constant False
    means `--multi-view` drawer runs fed the policy the first camera, for every run ever recorded.
  * `realm/environments/task_progression.py::check_reach_condition` -- the touch-only REACH branch
    for cabinets. Constant False means drawer REACH was scored by centre-to-centre distance to the
    cabinet origin, which is not a point the robot can reach for.

The second one also compared `self.task_progression` -- an `OrderedDict` of stage -> bool -- against
a list of strings, which cannot be True for a reason unrelated to the string being wrong. Both
failure modes are checked below.

STATIC, and deliberately so: no container, no simulator, no GPU, ~0.05 s on the login python. The
run-time behaviour of the fixed predicate is covered by tests/test_rollout_camera_selection.py,
which needs the container because `realm.rollout` imports omnigibson.

    python3 tests/test_task_type_literals.py
"""
import ast
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
TASK_CFG_GLOB = "realm/config/tasks/**/*.yaml"
# Scanned trees. tests/ is excluded on purpose: a test is allowed to name a task type that no
# config declares, in order to assert what happens to it.
SCAN_ROOTS = ("realm", "examples")

# Attribute/subscript names that hold a task type at runtime.
TASK_TYPE_NAMES = {"task_type"}
# Attributes that are NOT task types but have been compared against task-type strings anyway.
# self.task_progression is the rubric OrderedDict; see the module docstring.
NEVER_A_STRING = {"task_progression"}


def _declared_task_types():
    """Every value of `task_type` any task config declares."""
    declared = {}
    for path in sorted(PROJECT_ROOT.glob(TASK_CFG_GLOB)):
        try:
            cfg = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("task_type"), str):
            declared.setdefault(cfg["task_type"], []).append(
                str(path.relative_to(PROJECT_ROOT)))
    return declared


def _target_name(node):
    """The name a comparison operand refers to: `self.task_type`, `task_type`, `cfg["task_type"]`."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value if isinstance(node.slice.value, str) else None
    # getattr(env, "task_type", None) == "..."
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)):
        return node.args[1].value
    return None


def _string_literals(node):
    """String constants in a comparator: `"put"`, `["a", "b"]`, `("a", "b")`, `{"a"}`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _comparisons(path):
    """(lineno, name, [string literals]) for every `<name> ==/!=/in/not in <strings>` in a file."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not all(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            continue
        # Either side may hold the name; the other side holds the literals.
        for lhs, rhs in ((node.left, node.comparators[0]),
                         (node.comparators[0], node.left)):
            name = _target_name(lhs)
            literals = _string_literals(rhs)
            if name and literals:
                found.append((node.lineno, name, literals))
                break
    return found


def main():
    declared = _declared_task_types()
    known = set(declared)

    files = sorted(p for root in SCAN_ROOTS
                   for p in (PROJECT_ROOT / root).rglob("*.py"))

    print(f"{len(declared)} distinct task_type values declared by "
          f"{sum(len(v) for v in declared.values())} task configs:")
    for value in sorted(declared):
        print(f"  {value:<14} {len(declared[value])} config(s)")
    print(f"scanned {len(files)} python files under {', '.join(SCAN_ROOTS)}/\n")

    failures = []
    unknown_hits, confusion_hits = [], []

    for path in files:
        rel = path.relative_to(PROJECT_ROOT)
        for lineno, name, literals in _comparisons(path):
            if name in NEVER_A_STRING:
                confusion_hits.append((rel, lineno, name, literals))
                failures.append(
                    f"[2] {rel}:{lineno} compares self.{name} against string literal(s) "
                    f"{literals}. {name} is not a string -- the comparison is constant False "
                    f"regardless of which strings are on the right.")
                continue
            if name not in TASK_TYPE_NAMES:
                continue
            bad = [s for s in literals if s not in known]
            if bad:
                unknown_hits.append((rel, lineno, bad))
                failures.append(
                    f"[1] {rel}:{lineno} compares {name} against {bad}, which no task config "
                    f"declares, so the branch is constant False. Declared: {sorted(known)}")

    print(f"[1] task_type compared against a value no config declares: "
          f"{[(str(r), n, b) for r, n, b in unknown_hits] if unknown_hits else 'none'}")
    print(f"[2] non-string attributes compared against string literals: "
          f"{[(str(r), n, a) for r, n, a, _ in confusion_hits] if confusion_hits else 'none'}")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- every task_type literal in realm/ and examples/ is a declared task_type")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
