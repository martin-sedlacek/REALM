"""Host tests of the stage hold (realm/stage_hold.py and its use in TaskProgressionMixin). No simulator.

Run directly: `uv run python -m pytest -q tests/test_stage_hold.py` (tier 1).
"""
import ast
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from realm.stage_hold import StageHold  # noqa: E402


def latch_steps(rows, holds):
    """Feed StageHold one row of conditions per step; return each stage's latch step (or None)."""
    hold = StageHold([str(i) for i in range(len(holds))], holds)
    flags, latch = [False] * len(holds), [None] * len(holds)
    for step, row in enumerate(rows):
        new = hold.step(flags, lambda i, row=row: row[i])
        for i, (a, b) in enumerate(zip(flags, new)):
            if b and not a:
                latch[i] = step
        flags = new
    return latch


T, F = True, False


def test_blip_shorter_than_n_does_not_latch_and_n_steps_do():
    assert latch_steps([[T]] * 4 + [[F]], [5]) == [None]
    assert latch_steps([[T]] * 5, [5]) == [4]


def test_broken_streak_restarts():
    rows = [[T], [T], [T], [F], [T], [T], [T], [T], [T]]
    assert latch_steps(rows, [5]) == [8]


def test_order_is_kept():
    # stage 1's condition is true from the start, but its streak only counts once stage 0 has latched
    # (that step included)
    rows = [[T, T], [T, T], [T, T], [F, T], [F, T]]
    assert latch_steps(rows, [3, 3]) == [2, 4]


def test_several_stages_latch_in_one_step_at_hold_1():
    assert latch_steps([[F, T, T], [T, T, T]], [1, 1, 1]) == [1, 1, 1]


def test_short_stage_latches_at_once_under_hold_5():
    rows = [[T, T]] * 5
    assert latch_steps(rows, [1, 5]) == [0, 4]


def test_latched_stage_stays_latched():
    rows = [[T, F], [F, T], [F, F]]
    assert latch_steps(rows, [1, 1]) == [0, 1]


def test_validation():
    with pytest.raises(ValueError):
        StageHold(["A"], [0])
    with pytest.raises(ValueError):
        StageHold(["A", "B"], [1])
    with pytest.raises(ValueError):
        StageHold(["A"], [1]).step([F, F], lambda i: T)


def _original_loop(task_progression, checkers):
    """A copy of recompute_task_progression before the hold existed (the reference for hold 1)."""
    reward = 0.0
    for stage, is_completed_flag in task_progression.items():
        if is_completed_flag or checkers[stage]():
            if not is_completed_flag:
                task_progression[stage] = True
            reward += 1 / len(task_progression.keys())
        else:
            break
    return reward


@pytest.mark.parametrize("seed", range(30))
def test_hold_1_makes_the_original_checker_calls(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 5)
    rows = [[rng.random() < 0.6 for _ in range(n)] for _ in range(rng.randint(1, 60))]
    calls_orig, flags = [], {str(i): False for i in range(n)}
    for step, row in enumerate(rows):
        _original_loop(flags, {str(i): (lambda i=i, step=step, row=row: calls_orig.append((step, i)) or row[i])
                               for i in range(n)})
    calls, hold, hflags = [], StageHold([str(i) for i in range(n)], [1] * n), [False] * n
    for step, row in enumerate(rows):
        hflags = hold.step(hflags, lambda i, step=step, row=row: calls.append((step, i)) or row[i])
    assert calls == calls_orig
    assert hflags == [flags[str(i)] for i in range(n)]


def _module_literal(name):
    tree = ast.parse((REPO / "realm/environments/task_progression.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _checker_stages():
    tree = ast.parse((REPO / "realm/environments/task_progression.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "attr", None) == "success_conditions" for t in node.targets)):
            return {k.value for k in node.value.keys}
    raise AssertionError("success_conditions not found")


def test_short_stages_are_real_stages():
    rubric_stages = {s for stages in yaml.safe_load(
        (REPO / "realm/config/tasks/task_progressions.yaml").read_text()).values() for s in stages}
    short = _module_literal("SHORT_STAGES")
    assert short == ("REACH", "TOUCH", "TOUCH_AND_MOVE_JOINT")
    assert set(short) <= _checker_stages() and set(short) <= rubric_stages
    assert _module_literal("DEFAULT_STAGE_HOLD_STEPS") == 5


# The mixin imports omnigibson; run it in a subprocess with omnigibson stubbed out.
_MIXIN = r'''
import json, sys
from unittest import mock
for name in ["omnigibson", "omnigibson.object_states", "omnigibson.object_states.open_state",
             "omnigibson.prims", "omnigibson.prims.joint_prim", "omnigibson.prims.rigid_prim",
             "omnigibson.objects", "omnigibson.objects.dataset_object"]:
    sys.modules[name] = mock.MagicMock()
sys.path.insert(0, sys.argv[1])
from realm.environments.task_progression import TaskProgressionMixin

class Env(TaskProgressionMixin):
    def __init__(self, stages):
        self.row, self.calls = None, 0
        self.task_type = None
        self._init_task_progression("none")
        self.task_progression = {s: False for s in stages}
        self.success_conditions = {s: self._checker(i) for i, s in enumerate(stages)}
        self.reset_stage_hold()
    def _checker(self, i):
        def f(obs):
            self.calls += 1
            return self.row[i]
        return f
    def run(self, rows):
        out = []
        for row in rows:
            self.row = row
            out.append(self.recompute_task_progression(None))
        return out

out = {}
stages = ["REACH", "GRASP", "PLACE_ONTO"]
env = Env(stages)
out["default_holds"] = env._stage_hold.holds
out["default_rewards"] = env.run([[True, True, True]] * 6)
env.configure_stage_hold(1)
out["hold_1_tracker"] = env._stage_hold is None
env = Env(stages); env.configure_stage_hold(2)
env.run([[True, True, False]])
env.reset_stage_hold()
out["reset_restarts"] = env.run([[True, True, True], [True, True, True]])
env.task_progression = {"REACH": False, "GRASP": False}
try:
    env.recompute_task_progression(None); out["swap_without_reset"] = "no error"
except RuntimeError:
    out["swap_without_reset"] = "raised"
try:
    env.configure_stage_hold(0); out["hold_0"] = "no error"
except ValueError:
    out["hold_0"] = "raised"
print("MIXIN " + json.dumps(out))
'''


def test_mixin(tmp_path):
    script = tmp_path / "mixin.py"
    script.write_text(_MIXIN)
    proc = subprocess.run([sys.executable, str(script), str(REPO)], capture_output=True, text=True, timeout=120)
    line = [x for x in proc.stdout.splitlines() if x.startswith("MIXIN ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[0][len("MIXIN "):])
    third = 1 / 3
    assert out["default_holds"] == [1, 5, 5]                       # REACH short, the rest 5
    # REACH latches at step 0; GRASP's streak counts from that step and completes at step 4; PLACE_ONTO's
    # streak only starts there
    assert out["default_rewards"] == [third, third, third, third, third + third, third + third]
    assert out["hold_1_tracker"] is True                           # hold 1: the original loop
    assert out["reset_restarts"] == [third, third + third]         # GRASP's earlier step did not count
    assert out["swap_without_reset"] == "raised"
    assert out["hold_0"] == "raised"
