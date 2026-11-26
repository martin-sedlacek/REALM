"""Scoring on current geometry in a vector environment: every member, including scenes tiled away from the origin.

Needs the container and a GPU (it builds `stack_cubes` twice in one simulator), no policy server:

    ./scripts/run_apptainer.sh python -u tests/test_scoring_geometry_vector.py [--num_envs 2]

In every member the green cube is put 12 cm above the yellow one and falls onto it while the batch steps
without rendering, so Fabric (synced only on a rendered step) shows it floating while PhysX has it stacked.

1. `env.last_step_rendered` is what the batch step did: False after `vec_env.step` inside
   `og.sim.render_on_step(False)`, True after a rendered step -- recorded inside the open render block.
2. Inside `current_geometry(env)`, for every member and both cubes, the AABB centre equals the PhysX
   world position to 1e-5 m (members 1.. sit in scenes tiled away from the origin, so a frame mix-up
   would show here as an error of the tile offset). Not vacuous: Fabric lags by > 1 mm somewhere.
3. OnTop(green, yellow) read inside the context on a non-rendered step equals OnTop read without the
   context (caches emptied) after a forced render of that same state.
4. The context touches nothing outside it: the pose function is the original again.

Prints `PASSED -- ...` or `FAILED -- N problem(s)` (exit codes are not trusted).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

TOL = 1e-5
STEPS = 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2)
    args = p.parse_args()

    import numpy as np
    import omnigibson as og
    from omnigibson.object_states import OnTop
    from omnigibson.utils import usd_utils

    from realm import frame_mode
    from realm.environments.env_vector import RealmVectorEnvironment
    from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
    from realm.rollout import resolve_task
    from realm.scoring_geometry import _empty_caches, current_geometry
    from realm.sim_config import set_sim_config

    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)
            print(f"  problem: {message}", flush=True)

    _, cfg = resolve_task(6, None, SUPPORTED_TASKS, name_includes_config=False)
    set_sim_config(robot="DROID_mounted", seed=1234)
    vec = RealmVectorEnvironment(args.num_envs, task_cfg_path=cfg, perturbations=[SUPPORTED_PERTURBATIONS[0]],
                                 robot="DROID_mounted", rendering_mode="rt", multi_view=False)
    frame_mode.apply(True)
    vec.warmup()
    original = usd_utils.get_world_pose_with_scale

    def centre(lo_hi):
        lo, hi = lo_hi
        return (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2

    def physx(obj):
        return np.asarray(obj.get_position_orientation()[0], dtype=np.float64)

    holds = [np.concatenate([env.robot.get_joint_positions()[:7].cpu().numpy(), [-1.0]]) for env in vec.envs]
    pairs = [(env.main_objects[0], env.target_objects[0]) for env in vec.envs]
    for i, (green, yellow) in enumerate(pairs):
        print(f"member {i}: green at {np.round(physx(green), 3).tolist()}, yellow at "
              f"{np.round(physx(yellow), 3).tolist()}", flush=True)
        ypos = physx(yellow)
        _, quat = green.get_position_orientation()
        green.set_position_orientation(position=ypos + np.array([0.0, 0.0, 0.03 + 0.12]), orientation=quat)

    lag = [0.0] * len(vec.envs)
    inside_err = [0.0] * len(vec.envs)
    ontop_checked = [0] * len(vec.envs)
    for k in range(STEPS):
        with og.sim.render_on_step(False):
            vec.step(holds)
        for i, env in enumerate(vec.envs):
            check(env.last_step_rendered is False, f"step {k} member {i}: last_step_rendered is "
                                                   f"{env.last_step_rendered} after a non-rendered batch step")
        current_ontop = []
        for i, env in enumerate(vec.envs):
            green, yellow = pairs[i]
            lag[i] = max(lag[i], float(np.abs(centre(green.aabb) - physx(green)).max()))
            with current_geometry(env):
                for obj in (green, yellow):
                    err = float(np.abs(centre(obj.aabb) - physx(obj)).max())
                    inside_err[i] = max(inside_err[i], err)
                    check(err <= TOL, f"step {k} member {i}: {obj.name} AABB centre inside the context is "
                                      f"{err:.2e} m from PhysX (> {TOL})")
                current_ontop.append(bool(green.states[OnTop].get_value(yellow)))
            check(usd_utils.get_world_pose_with_scale is original, f"step {k} member {i}: pose function not restored")
        if k in (2, 5, 10, 29):          # the same state rendered: Fabric synced, read without the context
            og.sim.render()
            for i, env in enumerate(vec.envs):
                green, yellow = pairs[i]
                with _empty_caches(env):
                    rendered_ontop = bool(green.states[OnTop].get_value(yellow))
                check(rendered_ontop == current_ontop[i], f"step {k} member {i}: OnTop inside the context "
                                                          f"{current_ontop[i]} != after a forced render {rendered_ontop}")
                ontop_checked[i] += 1
    with og.sim.render_on_step(True):
        vec.step(holds)
    for i, env in enumerate(vec.envs):
        check(env.last_step_rendered is True, f"member {i}: last_step_rendered is {env.last_step_rendered} "
                                              "after a rendered batch step")
    for i in range(len(vec.envs)):
        print(f"member {i}: Fabric lag max {lag[i]:.4f} m, inside-context error max {inside_err[i]:.2e} m, "
              f"OnTop compared {ontop_checked[i]}x, final OnTop(current) {current_ontop[i]}", flush=True)
    check(max(lag) > 1e-3, f"Fabric never lagged PhysX by > 1 mm (max {max(lag):.2e} m): the check proves nothing")
    check(any(current_ontop), "OnTop never true: the stacked state was not reached, the OnTop check proves little")

    if problems:
        print(f"FAILED -- {len(problems)} problem(s)", flush=True)
    else:
        print(f"PASSED -- {len(vec.envs)} members: current geometry within {TOL} m, OnTop equal to a forced "
              f"render, last_step_rendered as stepped", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
