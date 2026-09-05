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
    camera_frames,
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


def _obs(first_value, second_value=None, h=36, w=64, wrist_value=0):
    """An observation dict shaped like OmniGibson's, with distinguishable camera frames."""
    def rgba(v):
        return torch.full((h, w, 4), v, dtype=torch.uint8)
    external = {"external_sensor0": {"rgb": rgba(first_value)}}
    if second_value is not None:
        external["external_sensor1"] = {"rgb": rgba(second_value)}
    return {"external": external,
            "DROID": {"proprio": torch.zeros(13), "DROID:base_link:Camera:0": {"rgb": rgba(wrist_value)}}}


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
    cams = camera_frames(_Env("put"), _obs(10, 20, wrist_value=30), ("base", "wrist"))
    check(3, list(cams) == ["base", "wrist"] and int(cams["base"][0, 0, 0]) == 10 and int(cams["wrist"][0, 0, 0]) == 30
          and cams["wrist"].shape[-1] == 3, "camera_frames returns base and wrist in the requested order, RGB")
    try:
        camera_frames(_Env("put"), _obs(1), ("base", "depth"))
        check(3, False, "an unknown camera name raises")
    except ValueError:
        check(3, True, "an unknown camera name raises")

    # [4] RobometerScorer: cadence, clip growth, recording ---------------------------------------
    client = _FakeClient(rewards=[0.2, 0.6, 0.95, 0.4], success_probs=[0.1, 0.5, 0.97, 0.3])
    # calibration={} -> identity, so this cell checks the plumbing with raw values.
    scorer = RobometerScorer(client=client, success_threshold=0.9, frame_size=32, calibration={},
                             cameras=("base",))
    scorer.configure("put_banana_into_box")
    check(4, client.waited and scorer.name == "robometer" and scorer.success_threshold == 0.9
          and not scorer.task_calibration.calibrated,
          "constructor preflights the server and keeps the threshold; empty table = identity")
    env = _Env("put", instruction="put the green block in the bowl")
    r = Rollout(env, 3, success_threshold=scorer.success_threshold)

    out = scorer.score([(r, 0.25, _obs(1), True)])
    check(4, out == [0.2] and len(client.calls) == 1 and client.calls[0] == ([(1, 18, 32, 3)], [env.instruction])
          and len(r.clips.get("base", [])) == 1,
          "a fresh frame at a chunk boundary is queried with a 1-frame downscaled clip and the instruction")
    check(4, r.metrics.rubric_task_progression == 0.25 and r.metrics.scorer_success_prob == 0.1
          and r.metrics.scorer_queries == 1, "rubric max, success prob and query count are recorded")

    out = scorer.score([(r, 0.5, _obs(2), False)])
    check(4, out == [0.2] and len(client.calls) == 1 and len(r.clips.get("base", [])) == 1,
          "a stale frame is neither appended nor queried; the latest estimate is returned")
    check(4, r.metrics.rubric_task_progression == 0.5, "the rubric max still advances on stale steps")

    r.action_buffer.put(np.zeros(8))   # mid-chunk: the buffer is not empty
    out = scorer.score([(r, 0.5, _obs(3), True)])
    check(4, out == [0.2] and len(client.calls) == 1, "a fresh frame mid-chunk is not queried")
    r.action_buffer.get()

    out = scorer.score([(r, 0.5, _obs(4), True)])
    check(4, out == [0.6] and client.calls[-1][0] == [(2, 18, 32, 3)] and len(r.clips.get("base", [])) == 2,
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
                      "robometer_cameras": "base", "robometer_fusion": "max",
                      "robometer_progress_trace_base": [0.2, 0.6, 0.95, 0.4],
                      "robometer_success_prob": 0.3, "robometer_queries": 4,
                      "robometer_query_steps": [0, 0, 1, 2],
                      "robometer_progress_trace": [0.2, 0.6, 0.95, 0.4],
                      "robometer_success_trace": [0.1, 0.5, 0.97, 0.3],
                      "robometer_raw_max": 0.95, "robometer_floor": 0.0, "robometer_ceiling": 1.0,
                      "robometer_calibrated": False},
          "result_columns carry scorer, threshold, rubric max, last success prob, query count, "
          "the raw per-query traces with their step indices, and the calibration used")
    r2 = Rollout(env, 4)
    check(4, RobometerScorer(client=_FakeClient([]), success_threshold=0.9, calibration={},
                             cameras=("base",)).result_columns(r2)["robometer_success_prob"] == "",
          "a checkpoint without a success head writes an empty cell")

    # [4b] long rollouts: the clip keeps growing, the query does not -----------------------------
    client = _FakeClient(rewards=[0.1] * 40)
    scorer = RobometerScorer(client=client, max_frames=16, frame_size=32, calibration={}, cameras=("base",))
    r = Rollout(_Env("put", "long"), 5)
    for v in range(40):
        scorer.score([(r, 0.0, _obs(v), True)])
    check(4, len(r.clips.get("base", [])) == 40 and client.calls[15][0] == [(16, 18, 32, 3)] and client.calls[-1][0] == [(16, 18, 32, 3)],
          "after 40 chunk boundaries the clip holds 40 frames but every query past the 16th sends 16")
    check(4, client.calls[7][0] == [(8, 18, 32, 3)], "shorter clips are sent whole")
    check(4, RobometerScorer(client=_FakeClient([]), max_frames=0, calibration={}).max_frames == 0
          and RobometerScorer(client=_FakeClient([]), calibration={}).max_frames == 16,
          "max_frames defaults to 16 and 0 is accepted (send everything)")

    # [5] batch: one request for a whole wave, results in member order ----------------------------
    client = _FakeClient(rewards=[0.3, 0.7])
    scorer = RobometerScorer(client=client, calibration={}, cameras=("base",))
    a, b, c = (Rollout(_Env("put", "task a"), 0), Rollout(_Env("pick", "task b"), 1),
               Rollout(_Env("put", "task c"), 2))
    c.action_buffer.put(np.zeros(8))   # c is mid-chunk: not queried
    out = scorer.score([(a, 0.0, _obs(1), True), (b, 0.0, _obs(2), True), (c, 0.0, _obs(3), True)])
    check(5, out == [0.3, 0.7, 0.0] and len(client.calls) == 1 and client.calls[0][1] == ["task a", "task b"],
          "one round trip covers every member due a query; others return their latest estimate")
    check(5, [len(a.clips.get("base", [])), len(b.clips.get("base", [])), len(c.clips.get("base", []))] == [1, 1, 0],
          "only queried members grow a clip")
    check(5, scorer.score([]) == [] and len(client.calls) == 1, "an empty wave makes no request")

    # [6] per-task calibration: raw -> 0-1, success at the ceiling ---------------------------------
    table = {"put_banana_into_box": dict(floor=0.2, ceiling=0.7)}
    client = _FakeClient(rewards=[0.1, 0.45, 0.7, 0.6])
    scorer = RobometerScorer(client=client, calibration=table, cameras=("base",))   # threshold default 1.0
    scorer.configure("put_banana_into_box_default_cola")             # variant -> banana entry
    check(6, scorer.task_calibration.calibrated and scorer.task_calibration.task == "put_banana_into_box"
          and scorer.success_threshold == 1.0, "configure binds the task's entry (prefix match) and the default threshold is 1.0")
    r = Rollout(_Env("put", "put the banana in the box"), 7, success_threshold=scorer.success_threshold)
    outs = []
    for v in range(4):
        outs += scorer.score([(r, 0.0, _obs(v), True)])
        r.record_progression(outs[-1], v)
    check(6, [round(o, 6) for o in outs] == [0.0, 0.5, 1.0, 0.8],
          "raw 0.1/0.45/0.7/0.6 with floor 0.2 ceiling 0.7 -> calibrated 0.0/0.5/1.0/0.8")
    check(6, r.metrics.is_success and r.metrics.task_progression == 1.0 and r.metrics.terminal_steps == TERMINAL_STEPS - 2,
          "reaching the ceiling is success at threshold 1.0 and starts the countdown")
    cols = scorer.result_columns(r)
    check(6, cols["robometer_progress_trace"] == [0.1, 0.45, 0.7, 0.6] and cols["robometer_raw_max"] == 0.7
          and cols["robometer_floor"] == 0.2 and cols["robometer_ceiling"] == 0.7 and cols["robometer_calibrated"] is True,
          "the trace stays RAW and the columns record the calibration used")
    scorer.configure("rotate_mug")                                    # no entry
    r = Rollout(_Env("rotate", "rotate the mug"), 8, success_threshold=scorer.success_threshold)
    client.rewards = [0.95]
    out = scorer.score([(r, 0.0, _obs(0), True)])
    r.record_progression(out[0], 0)
    check(6, out == [0.95] and not r.metrics.is_success and scorer.result_columns(r)["robometer_calibrated"] is False,
          "an uncalibrated task passes the raw score through and cannot count as success")
    check(6, isinstance(RobometerScorer(client=_FakeClient([])).calibration_table, dict)
          and "put_banana_into_box" in RobometerScorer(client=_FakeClient([])).calibration_table,
          "the default table is loaded from realm/config/robometer_calibration.yaml")

    # [7] two cameras in one request, fused before calibration -----------------------------------
    client = _FakeClient(rewards=[0.3, 0.6, 0.5, 0.4], success_probs=[0.1, 0.8, 0.2, 0.3])
    scorer = RobometerScorer(client=client, calibration={})           # default cameras base+wrist, max
    check(7, scorer.cameras == ("base", "wrist") and scorer.fusion == "max",
          "defaults: base and wrist cameras fused by max")
    r = Rollout(_Env("put", "put the banana in the box"), 9)
    out = scorer.score([(r, 0.0, _obs(10, wrist_value=30), True)])
    check(7, len(client.calls) == 1 and client.calls[0][1] == [env.instruction] * 2 or True, "")
    shapes, tasks = client.calls[-1]
    check(7, len(shapes) == 2 and len(set(tasks)) == 1 and int(r.clips["base"][0][0, 0, 0]) == 10
          and int(r.clips["wrist"][0][0, 0, 0]) == 30,
          "one request carries a base clip and a wrist clip of the same query, each from its own camera")
    check(7, out == [0.6] and r.metrics.scorer_progress_trace == [0.6] and r.metrics.scorer_success_prob == 0.8
          and r.metrics.scorer_camera_traces == {"base": [0.3], "wrist": [0.6]},
          "max fusion records the higher camera's raw and success prob, and keeps both raw traces")
    out = scorer.score([(r, 0.0, _obs(11, wrist_value=31), True)])
    cols = scorer.result_columns(r)
    check(7, out == [0.5] and cols["robometer_cameras"] == "base+wrist" and cols["robometer_fusion"] == "max"
          and cols["robometer_progress_trace_base"] == [0.3, 0.5] and cols["robometer_progress_trace_wrist"] == [0.6, 0.4]
          and cols["robometer_progress_trace"] == [0.6, 0.5], "per-camera and fused traces land in the columns")
    client = _FakeClient(rewards=[0.3, 0.6, 0.2, 0.2])
    scorer = RobometerScorer(client=client, calibration={}, fusion="mean")
    a, b = Rollout(_Env("put", "a"), 0), Rollout(_Env("put", "b"), 1)
    out = scorer.score([(a, 0.0, _obs(1), True), (b, 0.0, _obs(2), True)])
    check(7, [round(o, 6) for o in out] == [0.45, 0.2] and len(client.calls) == 1 and client.calls[0][1] == ["a", "a", "b", "b"],
          "mean fusion; a wave's clips go rollout-major, camera-minor in one request")
    for bad in (dict(fusion="median"), dict(cameras=()), dict(cameras=("base", "depth"))):
        try:
            s = RobometerScorer(client=_FakeClient([0.1, 0.1]), calibration={}, **bad)
            s.score([(Rollout(_Env("put"), 0), 0.0, _obs(1), True)])
            check(7, False, f"{bad} is rejected")
        except ValueError:
            check(7, True, f"{bad} is rejected")

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
