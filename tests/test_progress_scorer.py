"""The `--robometer` scorer records what the server says, when it should, and leaves the default
path bit-for-bit alone.

WHY THIS EXISTS
---------------
realm/progress_scorer.py is the seam through which `--robometer` replaces the rubric's 0-1 task
progression. It sits on the hot path of BOTH evaluators, so two things have to hold and neither is
checked by any GPU test: (1) the default RubricScorer and the `success_threshold=1.0` default on
RolloutMetrics reproduce the pre-seam behaviour exactly -- same recorded values, same termination,
same binary_SR -- and (2) under Robometer the query cadence, frame choice, running max, threshold
semantics and extra result columns are what the docs promise.

Both are pinned here against a fake Robometer client and a fake observation, so no server and no
GPU are involved. Needs the container only because importing realm.rollout pulls in omnigibson
(same footing as tests/test_rollout_camera_selection.py):

    ./scripts/run_apptainer.sh python -u tests/test_progress_scorer.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from realm.progress_scorer import (  # noqa: E402
    RobometerScorer,
    RubricScorer,
    downscale_frame,
    exterior_frame,
)
from realm.rollout import TERMINAL_STEPS, Rollout, RolloutMetrics  # noqa: E402


class _Robot:
    name = "DROID"


class _Env:
    def __init__(self, task_type, instruction="do the thing"):
        self.robot = _Robot()
        self.task_type = task_type
        self.instruction = instruction


class _Result:
    def __init__(self, reward, success_prob=None):
        self.reward, self.success_prob = reward, success_prob


class _FakeClient:
    """Answers each clip with the next canned reward; records every call."""

    def __init__(self, rewards, success_probs=None):
        self.rewards = list(rewards)
        self.success_probs = list(success_probs) if success_probs is not None else None
        self.calls = []
        self.waited = False

    def wait_until_healthy(self, timeout_s=None):
        self.waited = True
        return {"status": "healthy"}

    def progress_batch(self, clips, tasks):
        self.calls.append(([c.shape for c in clips], list(tasks)))
        out = []
        for _ in clips:
            r = self.rewards.pop(0)
            s = self.success_probs.pop(0) if self.success_probs is not None else None
            out.append(_Result(r, s))
        return out


def _obs(first_value, second_value=None, h=36, w=64):
    """An observation dict shaped like OmniGibson's, with distinguishable exterior frames."""
    def rgba(v):
        return torch.full((h, w, 4), v, dtype=torch.uint8)
    external = {"external_sensor0": {"rgb": rgba(first_value)}}
    if second_value is not None:
        external["external_sensor1"] = {"rgb": rgba(second_value)}
    return {"external": external,
            "DROID": {"proprio": torch.zeros(13), "DROID:base_link:Camera:0": {"rgb": rgba(0)}}}


def main():
    failures = []

    def check(cell, cond, detail):
        print(f"[{cell}] {detail}: {'ok' if cond else 'FAIL'}")
        if not cond:
            failures.append(f"[{cell}] {detail}")

    # [1] default path: RubricScorer + threshold 1.0 reproduce the old behaviour exactly -----------
    env = _Env("put")
    rollout = Rollout(env, 0)
    rubric = RubricScorer()
    check(1, rubric.score([(rollout, 0.25, None, True), (rollout, 0.5, None, False)]) == [0.25, 0.5],
          "RubricScorer returns the rubric values untouched, fresh or not")
    check(1, rubric.result_columns(rollout) == {} and rubric.success_threshold == 1.0,
          "RubricScorer appends no columns and uses threshold 1.0")
    m = RolloutMetrics()
    for step, v in enumerate([0.0, 0.5, 0.5, 0.75, 0.5]):
        m.record_progression(v, step)
    check(1, m.task_progression == 0.75 and m.progression_timestamps == [1, 3] and not m.is_success
          and m.terminal_steps == TERMINAL_STEPS,
          "below 1.0 the rubric path never counts as success and the countdown does not start")
    m.record_progression(1.0, 5)
    check(1, m.is_success and m.terminal_steps == TERMINAL_STEPS - 1,
          "1.0 is success and the countdown starts on that step -- the old `>= 1.0`")
    check(1, not RolloutMetrics().is_success, "a fresh metrics object is not a success")

    # [2] threshold semantics for a learned scorer ------------------------------------------------
    m = RolloutMetrics(success_threshold=0.9)
    m.record_progression(0.89, 0)
    check(2, not m.is_success and m.terminal_steps == TERMINAL_STEPS, "0.89 < 0.9 is not success")
    m.record_progression(0.92, 1)
    check(2, m.is_success and m.terminal_steps == TERMINAL_STEPS - 1, "0.92 >= 0.9 is success")
    m.record_progression(0.3, 2)
    check(2, m.task_progression == 0.92 and m.terminal_steps == TERMINAL_STEPS - 2,
          "a later lower estimate does not undo success -- progression is a running max")

    # [3] frame helpers ----------------------------------------------------------------------------
    big = np.full((720, 1280, 3), 7, dtype=np.uint8)
    small = downscale_frame(big, 256)
    check(3, small.shape == (144, 256, 3) and small.dtype == np.uint8 and int(small[0, 0, 0]) == 7,
          "downscale_frame shrinks 1280x720 to 256x144 uint8")
    check(3, downscale_frame(np.zeros((100, 50, 3), np.uint8), 256).shape == (100, 50, 3),
          "downscale_frame never upscales")
    check(3, downscale_frame(np.zeros((8, 8, 4), np.uint8), 256).shape == (8, 8, 3),
          "downscale_frame drops an alpha channel")
    check(3, int(exterior_frame(_Env("put"), _obs(10, 20))[0, 0, 0]) == 10,
          "non-drawer tasks are scored on the first exterior camera")
    check(3, int(exterior_frame(_Env("open_drawer"), _obs(10, 20))[0, 0, 0]) == 20,
          "drawer tasks are scored on the second exterior camera when it exists")
    check(3, int(exterior_frame(_Env("open_drawer"), _obs(10))[0, 0, 0]) == 10,
          "drawer tasks fall back to the first camera without --multi-view")

    # [4] RobometerScorer: cadence, clip growth, recording ---------------------------------------
    client = _FakeClient(rewards=[0.2, 0.6, 0.95, 0.4], success_probs=[0.1, 0.5, 0.97, 0.3])
    scorer = RobometerScorer(client=client, success_threshold=0.9, frame_size=32)
    check(4, client.waited and scorer.name == "robometer" and scorer.success_threshold == 0.9,
          "constructor preflights the server and keeps the threshold")
    env = _Env("put", instruction="put the green block in the bowl")
    r = Rollout(env, 3, success_threshold=scorer.success_threshold)

    out = scorer.score([(r, 0.25, _obs(1), True)])
    check(4, out == [0.2] and len(client.calls) == 1 and client.calls[0] == ([(1, 18, 32, 3)], [env.instruction])
          and len(r.clip) == 1,
          "a fresh frame at a chunk boundary is queried with a 1-frame downscaled clip and the instruction")
    check(4, r.metrics.rubric_task_progression == 0.25 and r.metrics.scorer_success_prob == 0.1
          and r.metrics.scorer_queries == 1, "rubric max, success prob and query count are recorded")

    out = scorer.score([(r, 0.5, _obs(2), False)])
    check(4, out == [0.2] and len(client.calls) == 1 and len(r.clip) == 1,
          "a stale frame is neither appended nor queried; the latest estimate is returned")
    check(4, r.metrics.rubric_task_progression == 0.5, "the rubric max still advances on stale steps")

    r.action_buffer.put(np.zeros(8))   # mid-chunk: the buffer is not empty
    out = scorer.score([(r, 0.5, _obs(3), True)])
    check(4, out == [0.2] and len(client.calls) == 1, "a fresh frame mid-chunk is not queried")
    r.action_buffer.get()

    out = scorer.score([(r, 0.5, _obs(4), True)])
    check(4, out == [0.6] and client.calls[-1][0] == [(2, 18, 32, 3)] and len(r.clip) == 2,
          "the next boundary sends the whole 2-frame clip")
    r.record_progression(out[0], 10)
    check(4, r.metrics.task_progression == 0.6 and not r.metrics.is_success and r.active,
          "0.6 is recorded and is not a success at threshold 0.9")

    out = scorer.score([(r, 0.75, _obs(5), True)])
    r.record_progression(out[0], 11)
    check(4, out == [0.95] and r.metrics.is_success and r.metrics.terminal_steps == TERMINAL_STEPS - 1,
          "0.95 crosses the threshold and starts the terminal countdown")
    out = scorer.score([(r, 0.75, _obs(6), True)])
    r.record_progression(out[0], 12)
    check(4, out == [0.4] and r.metrics.task_progression == 0.95,
          "a later lower estimate is returned raw but the recorded progression keeps its max")
    cols = scorer.result_columns(r)
    check(4, cols == {"scorer": "robometer", "success_threshold": 0.9, "rubric_task_progression": 0.75,
                      "robometer_success_prob": 0.3, "robometer_queries": 4,
                      "robometer_query_steps": [0, 0, 1, 2],
                      "robometer_progress_trace": [0.2, 0.6, 0.95, 0.4],
                      "robometer_success_trace": [0.1, 0.5, 0.97, 0.3]},
          "result_columns carry scorer, threshold, rubric max, last success prob, query count and "
          "the raw per-query traces with their step indices")
    r2 = Rollout(env, 4)
    check(4, RobometerScorer(client=_FakeClient([]), success_threshold=0.9).result_columns(r2)[
        "robometer_success_prob"] == "", "a checkpoint without a success head writes an empty cell")

    # [5] batch: one request for a whole wave, results in member order ----------------------------
    client = _FakeClient(rewards=[0.3, 0.7])
    scorer = RobometerScorer(client=client)
    a, b, c = (Rollout(_Env("put", "task a"), 0), Rollout(_Env("pick", "task b"), 1),
               Rollout(_Env("put", "task c"), 2))
    c.action_buffer.put(np.zeros(8))   # c is mid-chunk: not queried
    out = scorer.score([(a, 0.0, _obs(1), True), (b, 0.0, _obs(2), True), (c, 0.0, _obs(3), True)])
    check(5, out == [0.3, 0.7, 0.0] and len(client.calls) == 1 and client.calls[0][1] == ["task a", "task b"],
          "one round trip covers every member due a query; others return their latest estimate")
    check(5, [len(a.clip), len(b.clip), len(c.clip)] == [1, 1, 0], "only queried members grow a clip")
    check(5, scorer.score([]) == [] and len(client.calls) == 1, "an empty wave makes no request")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED -- default scoring is unchanged and the Robometer scorer records what the "
              "server says, once per chunk boundary")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
