"""GPU check: the policy is given the current camera frame, and the video holds every step once.

Runs `realm.eval.evaluate` (or, with --num_envs N, `realm.vector_eval.evaluate_vectorized`) with the
debug policy (returning `--horizon`-step chunks), instrumented only from outside. It sets no render
setting itself.

Which scene a frame shows is read from its frame number, not its pixels: a Replicator
`ReferenceTime` annotator on every camera names the Kit update that rendered the frame the cameras
deliver now (`FRAME_NUMBER_OFFSET`). The rendered scene is the latest observation whose control
step had started before that update (Kit's update counter, read before every step). Per rollout:

    policy      at every inference, the frame given to the policy shows that control step
    video       render-on-demand: every recorded (fresh) frame shows its own control step;
                every step: the video frames (the observations after the first, plus `finish`'s)
                show control steps 0..n-1, each once, in order: no repeat, no skip
    recorder    with recording on, the recorder got exactly the frames above
    finish      `Rollout.finish` ran exactly once
    re-read     (every step) across each waited pass and observation re-read: numpy, random and
                torch RNG states and V-AUG's sigma/alpha unchanged, the step's observation images
                unchanged and not sharing memory with the re-read ones

`--control_kit_defaults` is the negative control (Kit's defaults, as in 1.0.1): the check must FAIL.
`--end_run_at RUN:STEP` ends rollout RUN after STEP, as a finished task would (a vector member
ending mid-wave). `--direct` builds the environment without an evaluator and only measures
(`main_direct`). One process runs one render-flag setting.
Prints a last line `CURRENT_FRAMES_CHECK: PASS|FAIL ...` and writes <log_dir>/check.json. Exit
codes are not trusted (Isaac); read the verdict line. Inside the container:

    python -u /app/tests/check_current_frames.py --log_dir /app/logs/current_frames/od_single \\
        --task_id 0 --repeats 2 --max_steps 60 [--no-render_on_demand] [--no_record] \\
        [--num_envs 2] [--perturbation_id 1] [--end_run_at 1:25]
"""
import argparse
import bisect
import functools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The frame Kit renders in its app update number u carries the reference-time numerator
#: u - FRAME_NUMBER_OFFSET (Isaac Sim 5.1, measured on branch waitidle-fix).
FRAME_NUMBER_OFFSET = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=0,
                        help="0: single env (realm.eval; with --direct RealmEnvironmentDynamic)")
    parser.add_argument("--render_on_demand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_record", action="store_true")
    parser.add_argument("--n_pre_obs_renders", type=int, default=None,
                        help="default: the evaluator's own default")
    parser.add_argument("--end_run_at", default=None, help="RUN:STEP")
    parser.add_argument("--control_kit_defaults", action="store_true",
                        help="negative control: launch with Kit's default check order and "
                             "waitIdle off, as REALM 1.0.1 did; the check is expected to FAIL")
    parser.add_argument("--repo", default=str(PROJECT_ROOT),
                        help="REALM checkout to import (default: this script's); with --direct it "
                             "may be another version, e.g. /app in a worktree of origin/main")
    parser.add_argument("--direct", action="store_true",
                        help="build the environment directly (RealmVectorEnvironment with --num_envs), no evaluator (see "
                             "main_direct); the evaluator options above do not apply")
    parser.add_argument("--render_every", type=int, default=1,
                        help="--direct: render every Nth control step (1: every step)")
    parser.add_argument("--steps", type=int, default=64, help="--direct: control steps")
    parser.add_argument("--wait_idle", action=argparse.BooleanOptionalAction, default=None,
                        help="--direct: pass wait_idle=True/False to the constructor (default: not "
                             "passed, so the constructor's default; a checkout without the keyword "
                             "runs unchanged)")
    return parser.parse_args()


class ReferenceTimes:
    """One `ReferenceTime` annotator per camera of an environment (exterior and robot cameras)."""

    def __init__(self, env):
        import omnigibson as og
        import omnigibson.lazy as lazy

        sensors = dict(getattr(env.omnigibson_env, "external_sensors", None) or {})
        sensors.update({name: s for name, s in env.robot.sensors.items() if ":Camera:" in name})
        self.annotators = {}
        for name, sensor in sensors.items():
            with og.sim.editing_usd():
                annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
                    "ReferenceTime")
                annotator.attach([sensor.render_product])
            self.annotators[name] = annotator

    def frame_number(self):
        """The frame the cameras deliver now (the oldest if they differ); None if unnamed."""
        numbers = []
        for annotator in self.annotators.values():
            try:
                numbers.append(int(annotator.get_data()["referenceTimeNumerator"]))
            except (KeyError, IndexError, TypeError, ValueError):
                return None
        return min(numbers) if numbers else None


def update_number():
    import omni.kit.app
    return int(omni.kit.app.get_app().get_update_number())


def rgb_tensors(obs, robot_name):
    """Every camera image tensor of an observation, by a stable key."""
    found = {}
    for cam, data in (obs.get("external") or {}).items():
        if "rgb" in data:
            found[f"external/{cam}"] = data["rgb"]
    for key, data in (obs.get(robot_name) or {}).items():
        if ":Camera:" in key and isinstance(data, dict) and "rgb" in data:
            found[f"{robot_name}/{key}"] = data["rgb"]
    return found


def rng_state():
    import torch
    return (np.random.get_state()[1].tobytes(), np.random.get_state()[2], random.getstate(),
            torch.random.get_rng_state().numpy().tobytes())


class Audit:
    """One rollout's observations: Kit update before its step, frame delivered, flags."""

    def __init__(self, recorded):
        self.recorded = recorded
        self.before = []      # Kit update counter before the step that produced observation i
        self.frames = []      # frame number the cameras delivered with observation i
        self.fresh = []
        self.policy = {}      # observation index -> frame number given to the policy
        self.finish_calls = 0
        self.recorder_frames = 0
        self.unread = 0

    def scene(self, frame_number):
        rendered = frame_number + FRAME_NUMBER_OFFSET
        return bisect.bisect_left(self.before, rendered) - 1

    def add(self, record, fresh):
        before, frame = record
        if frame is None:
            self.unread += 1
        self.before.append(before)
        self.frames.append(frame)
        self.fresh.append(fresh)

    def result(self, render_on_demand):
        n = len(self.before) - (1 if self.finish_calls else 0)   # observed control steps
        problems = []
        if self.unread:
            problems.append(f"{self.unread} reads named no frame")
            return {"steps": n, "problems": problems}
        scenes = [self.scene(f) for f in self.frames]
        policy = {i: self.scene(f) for i, f in self.policy.items()}
        stale = {i: s for i, s in policy.items() if s != i}
        if stale:
            problems.append(f"policy frames not current at {len(stale)} of {len(policy)} "
                            f"inferences: {dict(list(stale.items())[:5])}")
        if render_on_demand:
            video = [(i, scenes[i]) for i in range(n) if self.fresh[i]]
            wrong = [(i, s) for i, s in video if s != i]
            if wrong:
                problems.append(f"{len(wrong)} of {len(video)} fresh frames not current: "
                                f"{wrong[:5]}")
            expected_frames = len(video)
        else:
            video = scenes[1:n + 1]
            if video != list(range(n)):
                repeated = sum(b <= a for a, b in zip(video, video[1:]))
                skipped = n - len({s for s in video if 0 <= s < n})
                problems.append(f"video scenes != 0..{n - 1}: {repeated} repeated, {skipped} "
                                f"skipped, first differences "
                                f"{[(k, s) for k, s in enumerate(video) if s != k][:5]}")
            expected_frames = n
        if self.finish_calls != 1:
            problems.append(f"finish ran {self.finish_calls} times")
        if self.recorded and self.recorder_frames != expected_frames:
            problems.append(f"recorder got {self.recorder_frames} frames, expected "
                            f"{expected_frames}")
        return {"steps": n, "inferences": len(policy), "video_frames": expected_frames,
                "recorder_frames": self.recorder_frames if self.recorded else None,
                "frame_ages_at_inference": sorted({i - s for i, s in policy.items()}),
                "problems": problems}


def instrument(args, state):
    """Patch the classes the evaluators use; everything recorded goes into `state`."""
    from realm import frame_mode
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.environments.env_vector import RealmVectorEnvironment
    from realm.inference import client as client_module
    from realm.realm_logging import VideoRecorder
    from realm.rollout import Rollout

    envs = state["envs"]

    def wrap(cls, name):
        def deco(fn):
            original = getattr(cls, name)
            setattr(cls, name, functools.wraps(original)(lambda *a, **k: fn(original, *a, **k)))
        return deco

    @wrap(RealmEnvironmentDynamic, "__init__")
    def _(orig, self, *a, **k):
        orig(self, *a, **k)
        envs.append(self)

    original_apply = frame_mode.apply
    if args.control_kit_defaults:
        import contextlib
        import omnigibson.lazy as lazy
        frame_mode.launch_settings = contextlib.nullcontext

        def original_apply(*a, **k):
            settings = lazy.carb.settings.get_settings()
            settings.set_bool(frame_mode.WAIT_IDLE_SETTING, False)
            return (settings.get(frame_mode.RENDER_CHECK_ORDER_SETTING),
                    settings.get(frame_mode.WAIT_IDLE_SETTING))

    def apply(*a, **k):
        result = original_apply(*a, **k)
        state["settings"] = [str(v) for v in result]
        for env in envs:          # apply runs in the constructor and again in the evaluator
            if not hasattr(env, "_check_rt"):
                env._check_rt = ReferenceTimes(env)
        return result

    frame_mode.apply = apply

    def before_step(env):
        if hasattr(env, "_check_rt"):
            env._check_before = update_number()

    def after_step(env, result):
        if hasattr(env, "_check_rt"):
            env._check_last = (env._check_before, env._check_rt.frame_number())
            env._check_obs = result[0]
        return result

    @wrap(RealmEnvironmentDynamic, "step")
    def _(orig, self, *a, **k):
        before_step(self)
        return after_step(self, orig(self, *a, **k))

    @wrap(RealmEnvironmentDynamic, "pre_step")
    def _(orig, self, *a, **k):
        before_step(self)
        return orig(self, *a, **k)

    @wrap(RealmEnvironmentDynamic, "post_step")
    def _(orig, self, *a, **k):
        return after_step(self, orig(self, *a, **k))

    @wrap(RealmEnvironmentDynamic, "read_obs")
    def _(orig, self, *a, **k):
        obs = orig(self, *a, **k)
        self._check_policy = self._check_rt.frame_number()
        self._check_reread = obs
        return obs

    def check_reread(members, call):
        """Around one waited pass + re-read: RNG, V-AUG parameters, the step obs untouched."""
        snapshot = {}
        for env in members:
            obs = getattr(env, "_check_obs", None)
            if obs is not None:
                snapshot[id(env)] = {k: t.clone() for k, t in rgb_tensors(obs, env.robot.name).items()}
        aug = [(env.v_aug_sigma, env.v_aug_alpha) for env in members]
        rng = rng_state()
        result = call()
        if rng_state() != rng:
            state["reread_problems"].append("an RNG state changed across the re-read")
        if [(env.v_aug_sigma, env.v_aug_alpha) for env in members] != aug:
            state["reread_problems"].append("V-AUG sigma/alpha changed across the re-read")
        for env in members:
            if id(env) not in snapshot:
                continue
            now = rgb_tensors(env._check_obs, env.robot.name)
            reread = rgb_tensors(env._check_reread, env.robot.name)
            for key, before in snapshot[id(env)].items():
                if not bool((now[key] == before).all()):
                    state["reread_problems"].append(f"step obs {key} changed by the re-read")
                if key in reread and reread[key].data_ptr() == now[key].data_ptr():
                    state["reread_problems"].append(f"{key}: step obs and re-read share memory")
        state["rereads"] += 1
        return result

    @wrap(RealmEnvironmentDynamic, "current_obs")
    def _(orig, self):
        return check_reread([self], lambda: orig(self))

    @wrap(RealmVectorEnvironment, "current_obs")
    def _(orig, self):
        return check_reread(self.envs, lambda: orig(self))

    @wrap(Rollout, "__init__")
    def _(orig, self, *a, **k):
        orig(self, *a, **k)
        self._audit = Audit(recorded=self.recorder is not None)
        if self.recorder is not None:
            self.recorder._check_audit = self._audit
        state["rollouts"].append(self)

    @wrap(Rollout, "observe")
    def _(orig, self, obs, obs_is_fresh, current_obs=None):
        env = self.env
        assert env._check_obs is obs, "observation not from the hooked step"
        index = len(self._audit.before)
        self._audit.add(env._check_last, obs_is_fresh)
        if self.needs_fresh_obs():      # this iteration infers
            self._audit.policy[index] = (env._check_policy if current_obs is not None
                                         else env._check_last[1])
        return orig(self, obs, obs_is_fresh, current_obs)

    @wrap(Rollout, "finish")
    def _(orig, self, obs):
        assert self.env._check_obs is obs, "finish not given the last step's observation"
        self._audit.finish_calls += 1
        if self._audit.finish_calls == 1:
            self._audit.add(self.env._check_last, True)
        return orig(self, obs)

    if args.end_run_at:
        end_run, end_step = (int(v) for v in args.end_run_at.split(":"))

        @wrap(Rollout, "record_progression")
        def _(orig, self, task_progression, step):
            active = orig(self, task_progression, step)
            if self.run_id == end_run and step >= end_step:
                self.active = active = False
            return active

    @wrap(VideoRecorder, "add_frame")
    def _(orig, self, *a, **k):
        if hasattr(self, "_check_audit"):
            self._check_audit.recorder_frames += 1
        return orig(self, *a, **k)

    horizon = args.horizon

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
              use_base_im_second=False, ee_control=False, cartesian_position=None):
        joints = np.asarray(robot_state, dtype=np.float64).reshape(-1)[:7]
        offsets = 0.02 * np.sin(np.arange(1, horizon + 1))[:, None]
        chunk = np.zeros((horizon, 8))
        chunk[:, :7] = joints[None, :] + offsets
        return chunk

    client_module._DebugAdapter.infer = infer


def main_direct(args):
    """Build the environment directly, step it with scripted actions, record every control step.

    Uses only API origin/main also has, so `--repo` can point at an older checkout (`--wait_idle`
    needs this branch). Seeded as the evaluators seed. Step k renders when render_every <= 1 or
    (k + 1) % render_every == 0. The action is the warmup's hold command plus 0.05 rad *
    sin(2 pi k / 32) on the arm joints, the same for every member. Writes direct_steps.csv (per
    step and member: frame age, ms, joints, main-object pose) and direct.json (settings and RNG
    digests after construction, medians). A measurement, not a pass/fail check.
    """
    import csv
    import hashlib
    import pickle
    import statistics
    import time

    import omnigibson as og
    import omnigibson.lazy as lazy
    import torch
    from realm.eval import CONFIG_ROOT, SUPPORTED_TASKS
    from realm.rollout import resolve_task
    from realm.sim_config import set_sim_config
    from realm.task_seed import task_seed

    _, task_cfg_path = resolve_task(args.task_id, None, SUPPORTED_TASKS, name_includes_config=True)
    set_sim_config(robot="DROID", seed=task_seed(task_cfg_path, CONFIG_ROOT))
    kwargs = dict(task_cfg_path=task_cfg_path, perturbations=["Default"])
    if args.wait_idle is not None:
        kwargs["wait_idle"] = args.wait_idle
    if args.num_envs:
        from realm.environments.env_vector import RealmVectorEnvironment
        vec = RealmVectorEnvironment(args.num_envs, **kwargs)
        envs = vec.envs
    else:
        from realm.environments.env_dynamic import RealmEnvironmentDynamic
        vec = None
        envs = [RealmEnvironmentDynamic(config_path=CONFIG_ROOT, **kwargs)]

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    rng = {"python": digest(pickle.dumps(random.getstate())),
           "numpy": digest(pickle.dumps(np.random.get_state())),
           "torch": digest(torch.random.get_rng_state().numpy().tobytes())}
    carb = lazy.carb.settings.get_settings()
    settings = {key: carb.get(key) for key in ("/app/updateOrder/checkForHydraRenderComplete",
                                                "/app/hydraEngine/waitIdle")}
    print(f"[direct] after construction: settings {settings}, rng {rng}", flush=True)
    sources = [ReferenceTimes(env) for env in envs]

    if vec is not None:
        vec.warmup()
    else:
        obs, _ = envs[0].reset()
        envs[0].warmup(obs)
    holds = [np.asarray(env.warmup_action(10**6, env.warmup_ee_cmd()), dtype=np.float64)
             for env in envs]

    before, frames, rendered, ms, joints, poses = [], [], [], [], [], []
    for k in range(args.steps):
        render = args.render_every <= 1 or (k + 1) % args.render_every == 0
        actions = []
        for hold in holds:
            action = hold.copy()
            action[:7] += 0.05 * np.sin(2 * np.pi * k / 32)
            actions.append(action)
        before.append(update_number())
        start = time.perf_counter()
        with og.sim.render_on_step(render):
            if vec is not None:
                vec.step(actions, n_render_iterations=1)
            else:
                envs[0].step(actions[0], n_render_iterations=1)
        ms.append(1000 * (time.perf_counter() - start))
        rendered.append(render)
        frames.append([source.frame_number() for source in sources])
        joints.append([env.robot.get_joint_positions().cpu().numpy().tolist() for env in envs])
        step_poses = []
        for env in envs:
            pos, quat = env.main_objects[0].get_position_orientation()
            step_poses.append(pos.cpu().numpy().tolist() + quat.cpu().numpy().tolist())
        poses.append(step_poses)

    def scene(frame_number):
        return bisect.bisect_left(before, frame_number + FRAME_NUMBER_OFFSET) - 1

    n_joints = len(joints[0][0])
    path = os.path.join(args.log_dir, "direct_steps.csv")
    ages_rendering = []
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "env", "rendered", "update_before", "frame_number", "scene",
                         "frame_age", "ms"] + [f"joint_{j}" for j in range(n_joints)]
                        + ["mo_x", "mo_y", "mo_z", "mo_qx", "mo_qy", "mo_qz", "mo_qw"])
        for k in range(args.steps):
            for i in range(len(envs)):
                number = frames[k][i]
                shown = None if number is None else scene(number)
                age = None if shown is None else k - shown
                if rendered[k]:
                    ages_rendering.append(age)
                writer.writerow([k, i, int(rendered[k]), before[k], number, shown, age,
                                 round(ms[k], 3)] + joints[k][i] + poses[k][i])
    other = [m for m, r in zip(ms, rendered) if not r]
    report = {"args": vars(args), "settings": settings, "rng_after_construction": rng,
              "ms_median": statistics.median(ms),
              "ms_median_rendering": statistics.median(m for m, r in zip(ms, rendered) if r),
              "ms_median_other": statistics.median(other) if other else None,
              "age_histogram_rendering": {str(a): ages_rendering.count(a)
                                          for a in sorted(set(ages_rendering), key=str)},
              "steps_csv": path}
    with open(os.path.join(args.log_dir, "direct.json"), "w") as f:
        json.dump(report, f, indent=1)
    current = sum(a == 0 for a in ages_rendering)
    print(f"DIRECT_FRAMES: repo={args.repo} envs={len(envs)} render_every={args.render_every} "
          f"steps={args.steps} current_at_rendering_steps={current}/{len(ages_rendering)} "
          f"ages_rendering={report['age_histogram_rendering']} "
          f"ms_median={report['ms_median']:.1f} rendering={report['ms_median_rendering']:.1f} "
          f"other={report['ms_median_other']} settings={settings}", flush=True)
    og.shutdown()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    if args.direct:
        os.makedirs(args.log_dir, exist_ok=True)
        return main_direct(args)
    os.makedirs(args.log_dir, exist_ok=True)
    state = {"envs": [], "rollouts": [], "reread_problems": [], "rereads": 0, "settings": None}
    instrument(args, state)

    kwargs = dict(task_id=args.task_id, perturbation_id=args.perturbation_id,
                  repeats=args.repeats, max_steps=args.max_steps, horizon=args.horizon,
                  model_type="debug", port=0, log_dir=args.log_dir, no_record=args.no_record,
                  render_on_demand=args.render_on_demand)
    if args.n_pre_obs_renders is not None:
        kwargs["n_pre_obs_renders"] = args.n_pre_obs_renders
    error = None
    try:
        if args.num_envs:
            from realm.vector_eval import evaluate_vectorized
            evaluate_vectorized(num_envs=args.num_envs, **kwargs)
        else:
            from realm.eval import evaluate
            evaluate(**kwargs)
    except Exception as exc:   # reported in the verdict; Isaac's exit code is not trusted
        import traceback
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"

    rollouts = [{"run_id": r.run_id, **r._audit.result(args.render_on_demand)}
                for r in state["rollouts"]]
    problems = [f"run {r['run_id']}: {p}" for r in rollouts for p in r["problems"]]
    problems += sorted(set(state["reread_problems"]))
    if error:
        problems.append(f"evaluator raised {error}")
    if not rollouts:
        problems.append("no rollout ran")
    if not args.render_on_demand and state["rereads"] == 0:
        problems.append("no waited pass was made")
    if args.render_on_demand and state["rereads"]:
        problems.append(f"{state['rereads']} waited passes with render-on-demand")
    report = {"args": vars(args), "settings": state["settings"], "rereads": state["rereads"],
              "rollouts": rollouts, "problems": problems}
    with open(os.path.join(args.log_dir, "check.json"), "w") as f:
        json.dump(report, f, indent=1)
    for r in rollouts:
        print(f"[check] run {r['run_id']}: steps={r['steps']} inferences={r.get('inferences')} "
              f"video={r.get('video_frames')} recorder={r.get('recorder_frames')} "
              f"ages={r.get('frame_ages_at_inference')} problems={len(r['problems'])}",
              flush=True)
    mode = "render_on_demand" if args.render_on_demand else "every_step"
    where = f"vector{args.num_envs}" if args.num_envs else "single"
    record = "no_record" if args.no_record else "record"
    verdict = "PASS" if not problems else "FAIL"
    if args.control_kit_defaults:
        record += " control_kit_defaults"
    print(f"CURRENT_FRAMES_CHECK: {verdict} {mode} {where} {record} "
          f"perturbation={args.perturbation_id} rollouts={len(rollouts)} "
          f"rereads={state['rereads']} problems={len(problems)}"
          + (f" first: {problems[0]}" if problems else ""), flush=True)

    import omnigibson as og
    og.shutdown()


if __name__ == "__main__":
    main()
