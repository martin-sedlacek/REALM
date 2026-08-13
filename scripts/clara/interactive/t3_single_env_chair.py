"""Does `reset()` undo `apply_scene_fixes_from_cfg`'s object removal in the SINGLE-env path?

Found while debugging the vector env: after warmup, `n_objects` went 127 -> 128 and
`straight_chair_pmpwwi_0` was back to `active=True` in **every** scene, including scene 0.
`Scene.reset(hard=True)` restores `self._initial_file`, which `Scene.initialize()` captures before
REALM's scene fixes ever run.

Scene 0 being affected means this is probably not a vector-env bug at all -- the single-env path
calls `reset()` once per repeat, so every production REALM eval may be running with a chair the task
config asked to delete. That is what this checks, on the plain single-env construction path with no
vector machinery involved.

    MODE=stock ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t3_single_env_chair.py --task_id 0

Reports PASS if the removal survives reset, FAIL if the object comes back.
"""
import argparse

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
try:
    # 3.9.1 port split this out of eval.py
    from realm.sim_config import set_sim_config
except ImportError:
    # OmniGibson 1.1.1 REALM keeps it in eval.py
    from realm.eval import set_sim_config

WATCH = "straight_chair_pmpwwi_0"


def report(tag, env):
    scene = env.omnigibson_env.scene
    names = {o.name for o in scene.objects}
    in_registry = WATCH in names
    # og.sim.stage is 3.9.1 spelling; 1.1.1 may expose it differently, and this probe is meant to
    # run on both. Registry membership is the load-bearing signal either way.
    try:
        stage = getattr(og.sim, "stage", None) or og.sim.get_stage()
        prim = stage.GetPrimAtPath(f"{scene.prim_path}/{WATCH}")
        valid = prim.IsValid()
        active = prim.IsActive() if valid else False
    except Exception as e:
        print(f"    (stage check unavailable on this stack: {type(e).__name__}: {e})")
        valid = active = in_registry
    print(f"[{tag}] n_objects={len(names)}  {WATCH}: in_registry={in_registry} "
          f"stage_valid={valid} stage_active={active} -> would_render={valid and active}",
          flush=True)
    return in_registry, (valid and active)


def main(task_id, robot, repeats):
    set_sim_config(robot=robot)
    env = RealmEnvironmentDynamic(
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[SUPPORTED_PERTURBATIONS[0]],
        robot=robot,
    )

    print("\n########## single-env chair check ##########", flush=True)
    reg0, ren0 = report("after construction", env)

    results = []
    for i in range(repeats):
        env.reset()
        results.append(report(f"after reset #{i + 1}", env))

    print("\n########## VERDICT ##########")
    print(f"  after construction : registry={reg0} renders={ren0}   (want False/False)")
    back = [r for r in results if r[0] or r[1]]
    if not reg0 and not ren0 and not back:
        print(f"  PASS -- the removal survives {repeats} reset(s); the vector-env observation does "
              f"NOT extend to the single-env path.")
    elif not reg0 and not ren0 and back:
        print(f"  FAIL -- removal is correct at construction but {len(back)}/{repeats} reset(s) "
              f"bring '{WATCH}' back. Every production repeat after the first runs with an object "
              f"the task config asked to delete.")
    else:
        print(f"  UNEXPECTED -- the removal did not take at construction either; re-read the probe.")
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID_robolab_v2")
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    main(a.task_id, a.robot, a.repeats)
