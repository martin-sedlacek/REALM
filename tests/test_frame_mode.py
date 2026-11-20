"""Host tests for `realm/frame_mode.py` and where the loops and constructors use it.

Pins the launch setting, waitIdle and the read-back, the waited pass, `Rollout`'s late-frame video,
where both loops make the pass and call `finish`, and the constructors' `wait_idle` (AST). The
simulator modules are stubbed; the frames themselves are checked by tests/check_current_frames.py.

    uv run python -m pytest -q tests/test_frame_mode.py
"""
import contextlib
import importlib
import sys
import types

import numpy as np
import pytest

from realm import frame_mode

ORDER = frame_mode.RENDER_CHECK_ORDER_SETTING
WAIT = frame_mode.WAIT_IDLE_SETTING


class Settings(dict):
    """carb settings stand-in: get() and set_bool(), with a log of every write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = []

    def set_bool(self, key, value):
        self.writes.append((key, value))
        self[key] = value


# ---------------------------------------------------------------------------------------- launch


@pytest.fixture
def isaacsim(monkeypatch):
    """A stand-in `isaacsim` module whose SimulationApp records the launch config it gets."""
    launches = []

    def simulation_app(launch_config=None, *args, **kwargs):
        launches.append(launch_config)
        return "app"

    module = types.ModuleType("isaacsim")
    module.SimulationApp = simulation_app
    module.launches = launches
    monkeypatch.setitem(sys.modules, "isaacsim", module)
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    return module


def test_launch_inside_the_block_gets_the_check_order(isaacsim):
    with frame_mode.launch_settings():
        assert isaacsim.SimulationApp({"headless": True}) == "app"
    assert isaacsim.launches == [{"headless": True, "extra_args": [f"--{ORDER}=1000"]}]


def test_launch_keeps_the_callers_extra_args(isaacsim):
    with frame_mode.launch_settings():
        isaacsim.SimulationApp({"extra_args": ["--a=1"]})
    assert isaacsim.launches[0]["extra_args"] == ["--a=1", f"--{ORDER}=1000"]


def test_launch_refuses_extra_args_simulation_app_would_drop(isaacsim):
    with frame_mode.launch_settings(), pytest.raises(TypeError):
        isaacsim.SimulationApp({"extra_args": "--a=1"})


def test_simulation_app_is_restored_after_the_block(isaacsim):
    original = isaacsim.SimulationApp
    with pytest.raises(RuntimeError):
        with frame_mode.launch_settings():
            assert isaacsim.SimulationApp is not original
            raise RuntimeError("launch failed")
    assert isaacsim.SimulationApp is original


def test_launch_accepts_the_eula_like_omnigibson(isaacsim):
    import os
    with frame_mode.launch_settings():
        assert os.environ["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_setting_names():
    assert ORDER == "/app/updateOrder/checkForHydraRenderComplete"
    assert WAIT == "/app/hydraEngine/waitIdle"
    assert frame_mode.RENDER_CHECK_ORDER == 1000


@pytest.mark.parametrize("render_on_demand", [True, False])
def test_apply_sets_wait_idle_and_reports_what_it_read(render_on_demand, capsys):
    settings = Settings({ORDER: 1000, WAIT: not render_on_demand})
    assert frame_mode.apply(render_on_demand, settings) == (1000, render_on_demand)
    assert settings.writes == [(WAIT, render_on_demand)]
    line = capsys.readouterr().out.strip()
    assert line == f"[REALM_RENDER_SYNC] {ORDER}=1000 {WAIT}={render_on_demand}"


def test_apply_defaults_to_render_on_demand():
    settings = Settings({ORDER: 1000})
    frame_mode.apply(settings=settings)
    assert settings[WAIT] is True


@pytest.mark.parametrize("render_on_demand", [True, False])
@pytest.mark.parametrize("order", [-100, None])
def test_apply_raises_when_kit_started_outside_launch_settings(render_on_demand, order, capsys):
    with pytest.raises(RuntimeError, match="outside realm.frame_mode.launch_settings"):
        frame_mode.apply(render_on_demand, Settings({ORDER: order}))
    assert capsys.readouterr().out.startswith("[REALM_RENDER_SYNC] ")


class IgnoringSettings(Settings):
    """A simulator that does not take the write."""

    def set_bool(self, key, value):
        self.writes.append((key, value))


def test_apply_raises_when_wait_idle_does_not_take():
    with pytest.raises(RuntimeError, match="waitIdle on"):
        frame_mode.apply(True, IgnoringSettings({ORDER: 1000, WAIT: False}))


def test_render_current_waits_for_exactly_one_render_pass():
    settings = Settings({WAIT: False})
    seen = []
    frame_mode.render_current(settings, render=lambda: seen.append(settings[WAIT]))
    assert seen == [True]
    assert settings.writes == [(WAIT, True), (WAIT, False)]


def test_render_current_turns_wait_idle_off_when_the_render_raises():
    settings = Settings({WAIT: False})

    def render():
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError):
        frame_mode.render_current(settings, render=render)
    assert settings[WAIT] is False


# ------------------------------------------------------------------------------- stubbed modules


class Sim:
    """`og.sim` stand-in: logs render_on_step blocks."""

    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def render_on_step(self, render):
        self.log.append(("render_on_step", render))
        yield


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def realm_modules(monkeypatch):
    """Import realm.rollout, realm.eval and realm.vector_eval against simulator stubs."""
    sim = Sim()
    og = _module("omnigibson", sim=sim,
                 log=types.SimpleNamespace(info=lambda *a, **k: None))
    stubs = {
        "omnigibson": og,
        "realm.environments.env_dynamic": _module("realm.environments.env_dynamic",
                                                  RealmEnvironmentDynamic=object),
        "realm.environments.env_vector": _module("realm.environments.env_vector",
                                                 RealmVectorEnvironment=object),
        "realm.environments.task_progression": _module(
            "realm.environments.task_progression", DRAWER_TASK_TYPES=("open_drawer", "close_drawer")),
        "realm.inference": _module("realm.inference", InferenceClient=object,
                                   extract_from_obs=fake_extract),
        "realm.realm_logging": _module("realm.realm_logging", VideoRecorder=object,
                                       save_results=None, append_trajectory=None,
                                       append_video=None),
        "realm.sim_config": _module("realm.sim_config", set_sim_config=None),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("realm.rollout", "realm.eval", "realm.vector_eval"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    modules = types.SimpleNamespace(
        rollout=importlib.import_module("realm.rollout"),
        eval=importlib.import_module("realm.eval"),
        vector_eval=importlib.import_module("realm.vector_eval"),
        sim=sim,
    )
    yield modules
    for name in ("realm.rollout", "realm.eval", "realm.vector_eval"):
        sys.modules.pop(name, None)


def fake_extract(obs, robot_name="DROID"):
    """extract_from_obs stand-in: an observation is a dict naming its frame and state."""
    frame = obs["frame"]
    return (f"base{frame}", None, None, None, f"wrist{frame}", obs["state"], 0.0)


# -------------------------------------------------------------------------------------- Rollout


class Recorder:
    def __init__(self):
        self.frames = []

    def add_frame(self, base_im, wrist_im, base_im_second):
        self.frames.append(base_im)


class Metrics:
    def __init__(self):
        self.states = []

    def record_step(self, env, obs, robot_state, gripper_state):
        self.states.append(robot_state)
        return None, None


def _rollout(realm_modules, late_frames):
    env = types.SimpleNamespace(robot=types.SimpleNamespace(name="DROID"))
    rollout = realm_modules.rollout.Rollout(env, 0, recorder=Recorder(), late_frames=late_frames)
    rollout.metrics = Metrics()
    return rollout


def test_late_frames_video_holds_every_step_once(realm_modules):
    """Observation i carries the frame of step i-1: the video must be steps 0..n-1, once each."""
    rollout = _rollout(realm_modules, late_frames=True)
    steps = 5
    for i in range(steps):
        rollout.observe({"frame": i - 1, "state": i}, obs_is_fresh=True)
    rollout.finish({"frame": steps - 1, "state": steps})
    assert rollout.recorder.frames == [f"base{k}" for k in range(steps)]
    assert rollout.metrics.states == list(range(steps))


def test_on_time_frames_video_unchanged(realm_modules):
    """Render-on-demand: every fresh observation's frame, as on main; finish adds nothing."""
    rollout = _rollout(realm_modules, late_frames=False)
    rollout.observe({"frame": 0, "state": 0}, obs_is_fresh=True)
    rollout.observe({"frame": 0, "state": 1}, obs_is_fresh=False)
    rollout.observe({"frame": 2, "state": 2}, obs_is_fresh=True)
    rollout.finish({"frame": 3, "state": 3})
    assert rollout.recorder.frames == ["base0", "base2"]


def test_policy_gets_the_reread_frame_and_the_steps_state(realm_modules):
    rollout = _rollout(realm_modules, late_frames=True)
    rollout.observe({"frame": -1, "state": 0}, obs_is_fresh=True)
    observation = rollout.observe({"frame": 0, "state": 1}, obs_is_fresh=True,
                                  current_obs={"frame": 1, "state": 1})
    assert (observation.base_im, observation.wrist_im) == ("base1", "wrist1")
    assert observation.robot_state == 1
    assert rollout.recorder.frames == ["base0"]          # the video keeps the step's own frame
    assert rollout.metrics.states == [0, 1]


def test_finish_without_recorder(realm_modules):
    rollout = _rollout(realm_modules, late_frames=True)
    rollout.recorder = None
    rollout.observe({"frame": -1, "state": 0}, obs_is_fresh=True)
    rollout.finish({"frame": 0, "state": 1})              # no error, nothing to record


# --------------------------------------------------------------------------------- the two loops


class FakeRollout:
    """What the loops call on a Rollout, logging it; one inference per `horizon` steps."""

    def __init__(self, log, env, horizon, finish_after=None):
        self.log, self.env, self.horizon = log, env, horizon
        self.buffer = 0
        self.active = True
        self.finish_after = finish_after
        self.last_command = None
        self.finished = []
        self.run_id = env.name
        self.metrics = types.SimpleNamespace(task_progression=1.0)

    def needs_fresh_obs(self):
        return self.buffer == 0

    def observe(self, obs, obs_is_fresh, current_obs=None):
        self.log.append(("observe", self.env.name, obs["state"], obs_is_fresh,
                         None if current_obs is None else current_obs["state"]))
        return obs

    def act(self, observation, client, horizon):
        if self.buffer == 0:
            self.log.append(("infer", self.env.name))
            self.buffer = self.horizon
        self.buffer -= 1
        self.last_command = np.zeros(8)
        return self.last_command

    def record_progression(self, task_progression, step):
        if self.finish_after is not None and step + 1 >= self.finish_after:
            self.active = False
        return self.active

    def finish(self, obs):
        self.log.append(("finish", self.env.name, obs["state"]))
        self.finished.append(obs["state"])


class FakeEnv:
    """Single env: state counts control steps; current_obs re-reads it."""

    def __init__(self, log, name="env0"):
        self.log, self.name = log, name
        self.state = 0

    def reset(self):
        return {"frame": 0, "state": 0}, {}

    def warmup(self, obs):
        return {"frame": 0, "state": 0}, None, None, None, None

    def step(self, command, n_render_iterations=1):
        self.state += 1
        self.log.append(("step", self.name, n_render_iterations))
        return {"frame": self.state - 1, "state": self.state}, 0.0, False, False, {}

    def current_obs(self):
        self.log.append(("current_obs",))
        return {"frame": self.state, "state": self.state}


@pytest.mark.parametrize("n_pre_obs_renders", [1, 3])
def test_single_every_step_makes_one_waited_pass_before_each_inference(realm_modules,
                                                                       n_pre_obs_renders):
    log = []
    env = FakeEnv(log)
    rollout = FakeRollout(log, env, horizon=4)
    realm_modules.eval._run_rollout(rollout, None, 10, 4, False, n_pre_obs_renders, 8)

    # Every step: env.step(command) with one in-step render, whatever n_pre_obs_renders says.
    assert [e[2] for e in log if e[0] == "step"] == [1] * 10
    assert realm_modules.sim.log == []                         # no render_on_step toggling
    inferences = [i for i, e in enumerate(log) if e[0] == "infer"]
    assert len(inferences) == 3                                # steps 0, 4, 8
    for i in inferences:
        assert log[i - 2] == ("current_obs",)                  # pass, then observe, then infer
        assert log[i - 1][0] == "observe" and log[i - 1][4] == log[i - 1][2]
    assert sum(e == ("current_obs",) for e in log) == 3
    observes = [e for e in log if e[0] == "observe"]
    assert [e[4] is not None for e in observes] == [s % 4 == 0 for s in range(10)]
    assert rollout.finished == [10]                            # the last step's obs, once
    assert log[-1] == ("finish", "env0", 10)


def test_single_render_on_demand_has_no_extra_pass(realm_modules):
    log = []
    env = FakeEnv(log)
    rollout = FakeRollout(log, env, horizon=4)
    realm_modules.eval._run_rollout(rollout, None, 10, 4, True, 1, 8)

    assert ("current_obs",) not in log
    assert all(e[4] is None for e in log if e[0] == "observe")
    # Steps 3 and 7 feed an inference: they render with n_pre_obs_renders = 1 pass.
    assert realm_modules.sim.log == [("render_on_step", s % 4 == 3) for s in range(10)]
    assert rollout.finished == [10]   # Rollout.finish is a no-op without late frames


def test_single_render_on_demand_passes_n_pre_obs_renders(realm_modules):
    log = []
    env = FakeEnv(log)
    rollout = FakeRollout(log, env, horizon=4)
    realm_modules.eval._run_rollout(rollout, None, 8, 4, True, 3, 8)
    assert [e[2] for e in log if e[0] == "step"] == [1, 1, 1, 3, 1, 1, 1, 3]


def test_single_defaults(realm_modules):
    import inspect
    params = inspect.signature(realm_modules.eval.evaluate).parameters
    assert params["n_pre_obs_renders"].default == 1
    assert params["render_on_demand"].default is True
    params = inspect.signature(realm_modules.vector_eval.evaluate_vectorized).parameters
    assert params["n_pre_obs_renders"].default == 1
    assert params["render_on_demand"].default is True


class FakeVecEnv:
    def __init__(self, log, n):
        self.log = log
        self.envs = [FakeEnv(log, f"env{i}") for i in range(n)]

    def step(self, commands, n_render_iterations=1):
        self.log.append(("vec_step", n_render_iterations))
        return [(env.step(c)[0], 0.0) for env, c in zip(self.envs, commands)]

    def current_obs(self):
        self.log.append(("current_obs",))
        return [{"frame": env.state, "state": env.state} for env in self.envs]


def _wave(realm_modules, render_on_demand, finish_after=(None, 6, None), n=3):
    log = []
    vec_env = FakeVecEnv(log, n)
    members = [FakeRollout(log, env, horizon=4, finish_after=f)
               for env, f in zip(vec_env.envs, finish_after)]
    step_results = [({"frame": 0, "state": 0}, 0.0) for _ in range(n)]
    steps = realm_modules.vector_eval._step_wave(vec_env, [None] * n, members, step_results,
                                                 10, 4, render_on_demand, 1, 8)
    return log, members, steps


def test_vector_every_step_makes_one_pass_per_batch(realm_modules):
    log, members, steps = _wave(realm_modules, render_on_demand=False)
    assert steps == 10
    # One pass per inference boundary for the whole batch (steps 0, 4, 8), not one per member.
    assert sum(e == ("current_obs",) for e in log) == 3
    # The pass comes before the batch's observes of that iteration.
    for i, e in enumerate(log):
        if e == ("current_obs",):
            assert [x[0] for x in log[i + 1:i + 4]] == ["observe", "infer", "observe"]
    # Each inferring member's policy gets the re-read state; the others nothing.
    for e in log:
        if e[0] == "observe":
            assert (e[4] is not None) == (e[2] % 4 == 0)


def test_vector_finish_once_per_member(realm_modules):
    """Member 1 finishes at step 5 (mid-wave): finish once, with that step's observation."""
    log, members, _ = _wave(realm_modules, render_on_demand=False)
    assert members[1].finished == [6]
    assert members[0].finished == [10] and members[2].finished == [10]
    assert sum(1 for e in log if e[0] == "finish") == 3


def test_vector_render_on_demand_has_no_extra_pass(realm_modules):
    log, members, _ = _wave(realm_modules, render_on_demand=True)
    assert ("current_obs",) not in log
    assert [m.finished for m in members] == [[10], [6], [10]]


def test_vector_skips_a_member_that_never_started(realm_modules):
    log = []
    vec_env = FakeVecEnv(log, 2)
    members = [FakeRollout(log, vec_env.envs[0], horizon=4), None]
    vec_env.envs[1].warmup_action = lambda t, cmd: np.zeros(8)
    vec_env.envs[1].warmup_ee_cmd = lambda: None
    step_results = [({"frame": 0, "state": 0}, 0.0)] * 2
    realm_modules.vector_eval._step_wave(vec_env, [None] * 2, members, step_results,
                                         5, 4, False, 1, 8)
    assert members[0].finished == [5]
    assert sum(e == ("current_obs",) for e in log) == 2


# ------------------------------------------------------------------- the environment constructors


def _source_tree(relpath):
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / relpath
    return ast.parse(path.read_text())


def _init_of(tree, class_name):
    import ast
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")


def _calls(node, dotted):
    import ast
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]


def _keyword_default(init, name):
    import ast
    args = init.args
    positional = args.args[len(args.args) - len(args.defaults):]
    for arg, default in [*zip(positional, args.defaults), *zip(args.kwonlyargs, args.kw_defaults)]:
        if arg.arg == name:
            return ast.literal_eval(default)
    raise AssertionError(f"__init__ has no keyword {name!r} with a default")


@pytest.mark.parametrize("relpath, class_name", [
    ("realm/environments/env_dynamic.py", "RealmEnvironmentDynamic"),
    ("realm/environments/env_vector.py", "RealmVectorEnvironment"),
])
def test_constructor_wait_idle_keyword_defaults_to_off(relpath, class_name):
    assert _keyword_default(_init_of(_source_tree(relpath), class_name), "wait_idle") is False


def test_vector_env_does_not_pass_wait_idle_to_its_members():
    import ast
    init = _init_of(_source_tree("realm/environments/env_vector.py"), "RealmVectorEnvironment")
    for call in _calls(init, "RealmEnvironmentDynamic"):
        assert "wait_idle" not in ast.unparse(call)


def test_single_env_constructor_launches_inside_launch_settings_and_applies():
    import ast
    init = _init_of(_source_tree("realm/environments/env_dynamic.py"), "RealmEnvironmentDynamic")
    withs = [n for n in ast.walk(init) if isinstance(n, ast.With)]
    launch = [w for w in withs if _calls(w, "og.Environment")]
    assert len(launch) == 1, "og.Environment must be built inside one `with` block"
    # The block's context is launch_settings() unless the env is a vector member.
    assert "frame_mode.launch_settings()" in ast.unparse(init)
    assert [ast.unparse(c) for c in _calls(init, "frame_mode.apply")] == ["frame_mode.apply(wait_idle)"]


def test_vector_env_constructor_builds_its_members_inside_launch_settings_and_applies():
    import ast
    init = _init_of(_source_tree("realm/environments/env_vector.py"), "RealmVectorEnvironment")
    withs = [n for n in ast.walk(init) if isinstance(n, ast.With)
             and ast.unparse(n.items[0].context_expr) == "frame_mode.launch_settings()"]
    assert len(withs) == 1
    assert _calls(withs[0], "RealmEnvironmentDynamic"), "members must be built inside the block"
    assert [ast.unparse(c) for c in _calls(init, "frame_mode.apply")] == ["frame_mode.apply(wait_idle)"]
    # apply runs after the members' setup, as the evaluators' call did before
    assert init.body.index(next(s for s in init.body if _calls(s, "frame_mode.apply"))) > \
        init.body.index(withs[0])


@pytest.mark.parametrize("module", ["realm/eval.py", "realm/vector_eval.py"])
def test_evaluators_switch_wait_idle_after_construction(module):
    import ast
    source = ast.unparse(_source_tree(module))
    assert "frame_mode.apply(render_on_demand)" in source
    assert "launch_settings" not in source   # the constructors launch
    assert "wait_idle" not in source   # the evaluators set waitIdle by render mode, not the keyword
