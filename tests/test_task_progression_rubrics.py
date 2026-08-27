"""Does every stage a task rubric names actually have a checker that can be called?

`TaskProgressionMixin.get_task_progression()` walks the rubric and does, per stage:

    checker_function = self.success_conditions.get(stage)
    if is_completed_flag or checker_function(obs):

`.get()` returns None for an unknown stage and the result is called ANYWAY, so a stage name in
`realm/config/tasks/task_progressions.yaml` that is not a key of `success_conditions` is not a
warning or a skipped stage -- it is `TypeError: 'NoneType' object is not callable`, thrown mid
rollout. And a checker whose signature does not accept `obs` is the same crash one line later.

Neither can be caught by the rest of tests/: every other test runs `--model_type debug`, which
holds the arm still, so no rollout ever advances far enough to evaluate a late stage.

THIS TEST NEEDS NO CONTAINER, NO SIMULATOR AND NO GPU -- it is the only one in tests/ that does
not. It reads the rubric YAML and parses `realm/environments/task_progression.py` with `ast`
rather than importing it, because importing anything under `realm.environments` pulls in
omnigibson (~50 s) for no benefit here.

    python3 tests/test_task_progression_rubrics.py
"""
import ast
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
RUBRICS = PROJECT_ROOT / "realm/config/tasks/task_progressions.yaml"
SOURCE = PROJECT_ROOT / "realm/environments/task_progression.py"


def registry_and_checkers():
    """(stage -> method name, method name -> [arg names]) read out of the module's source."""
    tree = ast.parse(SOURCE.read_text())

    registry, checkers = {}, {}
    for node in ast.walk(tree):
        # self.success_conditions = { "REACH": self.check_reach_condition, ... }
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and target.attr == "success_conditions"
                        and isinstance(node.value, ast.Dict)):
                    for key, value in zip(node.value.keys, node.value.values):
                        if isinstance(key, ast.Constant) and isinstance(value, ast.Attribute):
                            registry[key.value] = value.attr
        if isinstance(node, ast.FunctionDef):
            # REQUIRED positional parameters only. `checker(obs)` is a perfectly good call for
            # check_rotated(self, obs, rot_threshold=1.1) -- the extra parameter has a default, so
            # it is not part of the contract. An earlier version of this test compared the full
            # argument list and reported check_rotated and check_touching_and_moved_mo_joint as
            # broken; both are fine, and that was two false positives out of four findings.
            args = [a.arg for a in node.args.args]
            n_default = len(node.args.defaults)
            checkers[node.name] = args[:len(args) - n_default] if n_default else args
    return registry, checkers


def delegations():
    """method name -> the set of sibling `self.check_*` methods its body calls."""
    tree = ast.parse(SOURCE.read_text())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = set()
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "self"
                    and sub.func.attr.startswith("check_")):
                called.add(sub.func.attr)
        out[node.name] = called
    return out


#: What each MOVE_JOINT_* checker delegates to. PINNED, NOT DERIVED -- the point is that a change
#: has to be made here as well, deliberately, instead of slipping through as a "cleanup".
#:
#: check_moved_mo_joint_full calls the _LARGE pair, not the _FULL pair. That reads like a
#: copy-paste slip and IS NOT ONE TO FIX: it is identical to the pre-port 1.1.1 implementation
#: (the original environment progression implementation), so it is the behaviour every REALM
#: number was ever scored against, and under the standing rule that pre-port behaviour is
#: presumed intentional it stays. Changing it would tighten MOVE_JOINT_FULL from openness
#: >0.65/<0.35 to >0.95/<0.05. Nothing reaches it today -- only the `turn_faucet` rubric names
#: MOVE_JOINT_FULL and no task config declares that task_type -- so the change would move no
#: number now while silently redefining the stage for whoever adds that task.
EXPECTED_MOVE_JOINT_DELEGATION = {
    "check_moved_mo_joint_small": {"check_closed_mo_joint_small", "check_opened_mo_joint_small"},
    "check_moved_mo_joint_large": {"check_closed_mo_joint_large", "check_opened_mo_joint_large"},
    "check_moved_mo_joint_full":  {"check_closed_mo_joint_large", "check_opened_mo_joint_large"},
}


def main():
    rubrics = yaml.safe_load(RUBRICS.read_text())
    registry, checkers = registry_and_checkers()

    stages = {}          # stage -> [task types naming it]
    for task_type, stage_list in rubrics.items():
        for stage in stage_list:
            stages.setdefault(stage, []).append(task_type)

    print(f"{len(rubrics)} task types, {len(stages)} distinct stages, "
          f"{len(registry)} registered checkers")
    print(f"rubrics: {RUBRICS.relative_to(PROJECT_ROOT)}")
    print(f"source:  {SOURCE.relative_to(PROJECT_ROOT)}\n")

    failures = []

    # ---- 1: every stage a rubric names is registered ------------------------------------------
    unknown = sorted(s for s in stages if s not in registry)
    print(f"[1] stages named by a rubric but MISSING from success_conditions: "
          f"{unknown if unknown else 'none'}")
    for stage in unknown:
        near = [k for k in registry if k.startswith(stage) or stage.startswith(k)]
        failures.append(
            f"[1] rubric stage {stage!r} (used by {', '.join(stages[stage])}) has no entry in "
            f"success_conditions, so get_task_progression() calls None(obs) -> TypeError"
            + (f". Closest registered name(s): {near}" if near else ""))

    # ---- 2: every registered checker exists and takes (self, obs) ------------------------------
    bad_sig = []
    for stage, method in sorted(registry.items()):
        args = checkers.get(method)
        if args is None:
            failures.append(f"[2] success_conditions[{stage!r}] names self.{method}, which is not "
                            f"defined in {SOURCE.name}")
            continue
        if args != ["self", "obs"]:   # args = REQUIRED parameters; see registry_and_checkers()
            bad_sig.append((stage, method, args))
            reachable = [t for t in rubrics if stage in rubrics[t]]
            failures.append(
                f"[2] success_conditions[{stage!r}] -> {method}{tuple(args)} does not accept "
                f"(self, obs); it is invoked as checker(obs)"
                + (f". Reachable from rubric(s): {', '.join(reachable)}"
                   if reachable else ". Not named by any rubric today, so latent"))
    print(f"[2] registered checkers whose REQUIRED parameters are not (self, obs): "
          f"{[(s, m, a) for s, m, a in bad_sig] if bad_sig else 'none'}")

    # ---- 3: informational -- registered but no rubric names it ---------------------------------
    unused = sorted(k for k in registry if k not in stages)
    print(f"[3] registered but named by no rubric (informational, not a failure): {unused}")

    # ---- 4: the MOVE_JOINT_* family delegates where it is pinned to ----------------------------
    # A CHARACTERIZATION check, not a correctness one: it locks current behaviour, including the
    # _full -> _large delegation that looks wrong and is pre-port. See
    # EXPECTED_MOVE_JOINT_DELEGATION for why that one is deliberately not "fixed".
    calls = delegations()
    print("[4] MOVE_JOINT_* delegation (pinned):")
    for method, expected in EXPECTED_MOVE_JOINT_DELEGATION.items():
        got = calls.get(method)
        note = ("  <- deliberately the _large pair; pre-port, see the constant"
                if method.endswith("_full") else "")
        print(f"    {method} -> {sorted(got) if got is not None else '<missing>'}{note}")
        if got is None:
            failures.append(f"[4] {method} is not defined in {SOURCE.name}")
        elif got != expected:
            failures.append(
                f"[4] {method} now delegates to {sorted(got)}, pinned as {sorted(expected)}. If "
                f"this change is intended, update EXPECTED_MOVE_JOINT_DELEGATION and say why in "
                f"its comment -- for _full that means arguing the case for moving MOVE_JOINT_FULL "
                f"off pre-port thresholds, which is a scoring change, not a cleanup.")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- every rubric stage resolves to a checker that accepts (self, obs)")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
