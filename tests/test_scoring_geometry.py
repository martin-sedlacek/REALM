"""Scoring on current geometry (`realm/scoring_geometry.py`): what the context reads, and what it leaves behind.

Needs the container and a GPU (it builds `stack_cubes`), no policy server:

    ./scripts/run_apptainer.sh python -u tests/test_scoring_geometry.py

The cube is lifted 12 cm above its spawn and the simulator steps without rendering while it falls and settles, so
Fabric (synced only on a rendered step) lags its PhysX pose. On every such step:

1. INSIDE `current_geometry(env)`: the cube's AABB centre equals its PhysX position to 1e-5 m (the cube's origin is
   its centre), and so does the second cube's.
2. OUTSIDE, after the context: `usd_utils.get_world_pose_with_scale` and `xform_prim`'s name for it are the
   original function again, every object state's `_cache` / `_changed` dict is the same object with the same
   keys and the same values as before the context (nothing computed inside leaks out), and the cube's Fabric
   AABB is what it was before.
3. The check is not vacuous: on at least one of those steps Fabric's AABB centre is more than 1 mm from PhysX.
4. After a rendered step the context is a no-op (AABB inside == Fabric's).
5. A changed patch target fails loudly: with `get_world_pose_with_scale` replaced by a function of another
   signature, entering `physx_link_poses` raises RuntimeError.

Prints `PASSED -- ...` or `FAILED -- N problem(s)` (exit codes are not trusted: Isaac exits 0 on exceptions).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

TOL = 1e-5
STEPS = 40


def main():
    import numpy as np
    import omnigibson as og
    import omnigibson.prims.xform_prim as xform_prim
    from omnigibson.utils import usd_utils

    from realm import frame_mode
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.eval import CONFIG_ROOT, SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
    from realm.rollout import resolve_task
    from realm.scoring_geometry import _scene_states as scene_states
    from realm.scoring_geometry import current_geometry, physx_link_poses
    from realm.sim_config import set_sim_config

    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)
            print(f"  problem: {message}", flush=True)

    task, cfg = resolve_task(6, None, SUPPORTED_TASKS, name_includes_config=True)
    set_sim_config(robot="DROID_mounted", seed=1234)
    env = RealmEnvironmentDynamic(config_path=CONFIG_ROOT, task_cfg_path=cfg,
                                  perturbations=[SUPPORTED_PERTURBATIONS[0]], multi_view=False,
                                  no_rendering=False, rendering_mode="rt", robot="DROID_mounted")
    frame_mode.apply(True)
    env.reset()
    env.warmup()
    cubes = list(env.main_objects) + list(env.target_objects)
    cube = cubes[0]

    def centre(lo_hi):
        lo, hi = lo_hi
        return ((np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2)

    def physx(obj):
        return np.asarray(obj.get_position_orientation()[0], dtype=np.float64)

    def snapshot():
        out = []
        for state in scene_states(env):
            cache, changed = getattr(state, "_cache", None), getattr(state, "_changed", None)
            out.append((state, cache, None if cache is None else dict(cache),
                        changed, None if changed is None else dict(changed)))
        return out

    def same_values(a, b):
        if a.keys() != b.keys():
            return False
        return all(a[k] is b[k] for k in a)

    hold_action = np.concatenate([env.robot.get_joint_positions()[:7].cpu().numpy(), [-1.0]])
    pos, quat = cube.get_position_orientation()
    cube.set_position_orientation(position=pos + np.array([0.0, 0.0, 0.12]), orientation=quat)
    original = usd_utils.get_world_pose_with_scale

    lag_max, inside_max = 0.0, 0.0
    for k in range(STEPS):
        with og.sim.render_on_step(False):
            env.step(hold_action)
        before = snapshot()
        fabric_before = centre(cube.aabb)
        lag_max = max(lag_max, float(np.abs(fabric_before - physx(cube)).max()))
        with current_geometry(env):
            for obj in cubes:
                err = float(np.abs(centre(obj.aabb) - physx(obj)).max())
                inside_max = max(inside_max, err)
                check(err <= TOL, f"step {k}: {obj.name} AABB centre inside the context is {err:.2e} m "
                                  f"from PhysX (> {TOL})")
        check(usd_utils.get_world_pose_with_scale is original
              and xform_prim.get_world_pose_with_scale is original,
              f"step {k}: get_world_pose_with_scale not restored")
        for state, cache, cache_copy, changed, changed_copy in before:
            check(getattr(state, "_cache", None) is cache and (cache is None or same_values(cache, cache_copy))
                  and getattr(state, "_changed", None) is changed
                  and (changed is None or same_values(changed, changed_copy)),
                  f"step {k}: {type(state).__name__} of {getattr(state, 'obj', None)} cache changed by the context")
        check(np.array_equal(centre(cube.aabb), fabric_before), f"step {k}: Fabric AABB changed by the context")
    print(f"non-rendered steps {STEPS}: Fabric lag max {lag_max:.4f} m, inside-context error max "
          f"{inside_max:.2e} m", flush=True)
    check(lag_max > 1e-3, f"Fabric never lagged PhysX by > 1 mm (max {lag_max:.2e} m): the check proves nothing")

    # 4. no-op on a rendered step
    with og.sim.render_on_step(True):
        env.step(hold_action)
    fabric = centre(cube.aabb)
    with current_geometry(env):
        check(np.array_equal(centre(cube.aabb), fabric), "context on a rendered step changed the AABB read")

    # 5. a changed patch target fails loudly
    def other_signature(prim_path, extra):
        return original(prim_path)

    usd_utils.get_world_pose_with_scale = other_signature
    try:
        with physx_link_poses(env):
            pass
        check(False, "physx_link_poses accepted a patch target with another signature")
    except RuntimeError as error:
        print(f"changed target refused: {error}", flush=True)
    finally:
        usd_utils.get_world_pose_with_scale = original

    if problems:
        print(f"FAILED -- {len(problems)} problem(s)", flush=True)
    else:
        print(f"PASSED -- current geometry within {TOL} m inside, nothing changed outside, over {STEPS} "
              f"non-rendered steps", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
