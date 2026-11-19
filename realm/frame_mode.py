"""Kit render settings that make an observation's camera frame the current one.

    /app/updateOrder/checkForHydraRenderComplete = 1000   frames handed out after the update's
                                                         render (Kit's default -100: before it)
    /app/hydraEngine/waitIdle                           a render waits for the GPU

The environment constructors build inside `launch_settings()` (the check order is read only at
launch) and end with `apply(wait_idle)`. `wait_idle` defaults to False: a frame is then one step
old when every step renders and as old as the last render otherwise. For a current frame pass
`wait_idle=True`, or make one waited pass with `env.current_obs()` / `render_current()`. The
evaluators then call `apply(render_on_demand)`: on demand every render waits; every step, only
the pass before each inference. Details: wiki/Running-Evaluations.md, "Current camera frames".
"""
import contextlib
import os

#: Kit setting: where in an app update finished frames are handed to their readers.
RENDER_CHECK_ORDER_SETTING = "/app/updateOrder/checkForHydraRenderComplete"
#: After the update's own render (Kit's default is -100, before it).
RENDER_CHECK_ORDER = 1000

#: Kit setting: a render call returns only once the GPU has finished the frame.
WAIT_IDLE_SETTING = "/app/hydraEngine/waitIdle"


@contextlib.contextmanager
def launch_settings():
    """Within the block, `isaacsim.SimulationApp` adds the check order to `extra_args`.

    OmniGibson launches the app with the first environment and resets `sys.argv` for it, so the
    launch config is the only way in. No effect if the app already runs (`apply` then raises).
    """
    # As OmniGibson's _launch_app does before this import.
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    import isaacsim

    original = isaacsim.SimulationApp
    kit_arg = f"--{RENDER_CHECK_ORDER_SETTING}={RENDER_CHECK_ORDER}"

    def simulation_app(launch_config=None, *args, **kwargs):
        launch_config = dict(launch_config or {})
        extra_args = launch_config.get("extra_args")
        if extra_args is None:
            extra_args = []
        elif not isinstance(extra_args, (list, tuple)):
            # SimulationApp ignores a non-list: the setting would be lost.
            raise TypeError("launch_config['extra_args'] must be a list, got "
                            f"{type(extra_args).__name__}")
        launch_config["extra_args"] = [*extra_args, kit_arg]
        return original(launch_config, *args, **kwargs)

    isaacsim.SimulationApp = simulation_app
    try:
        yield
    finally:
        isaacsim.SimulationApp = original


def apply(render_on_demand=True, settings=None):
    """Set waitIdle, read both settings back, print them; raise unless they are as expected.

    Returns (check order, waitIdle) as read. `settings` defaults to the simulator's carb settings.
    """
    if settings is None:
        import omnigibson.lazy as lazy
        settings = lazy.carb.settings.get_settings()

    settings.set_bool(WAIT_IDLE_SETTING, bool(render_on_demand))
    order = settings.get(RENDER_CHECK_ORDER_SETTING)
    wait_idle = settings.get(WAIT_IDLE_SETTING)
    # print: OmniGibson's logger drops INFO once the simulator is up.
    print(f"[REALM_RENDER_SYNC] {RENDER_CHECK_ORDER_SETTING}={order} {WAIT_IDLE_SETTING}={wait_idle}",
          flush=True)

    problems = []
    if order != RENDER_CHECK_ORDER:
        problems.append(f"{RENDER_CHECK_ORDER_SETTING} = {RENDER_CHECK_ORDER}, got {order!r}")
    if bool(wait_idle) != bool(render_on_demand):
        problems.append(f"{WAIT_IDLE_SETTING} {'on' if render_on_demand else 'off'}, "
                        f"got {wait_idle!r}")
    if problems:
        raise RuntimeError(
            "Render settings: expected " + "; ".join(problems) + ". Kit reads the check order "
            "only while Isaac Sim starts, so this process's simulator was started outside "
            "realm.frame_mode.launch_settings(): build the first REALM environment before "
            "anything else launches Isaac Sim (og.launch(), another og.Environment), or launch "
            "it inside realm.frame_mode.launch_settings().")
    return order, wait_idle


def render_current(settings=None, render=None):
    """One render pass with waitIdle on (no physics), so the next observation read is current."""
    if settings is None:
        import omnigibson.lazy as lazy
        settings = lazy.carb.settings.get_settings()
    if render is None:
        import omnigibson as og
        render = og.sim.render

    settings.set_bool(WAIT_IDLE_SETTING, True)
    try:
        render()
    finally:
        settings.set_bool(WAIT_IDLE_SETTING, False)
