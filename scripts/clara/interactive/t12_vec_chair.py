"""Which members keep `apply_scene_fixes_from_cfg`'s removal, and which get the chair back?

The vector-env sibling of `t3_single_env_chair.py`. `apply_scene_fixes_from_cfg` removes every
object the scene config lists under `to_remove`, but `Scene.reset(hard=True)` restores
`self._initial_file`, which `Scene.initialize()` captured BEFORE those removals ran -- so the first
reset re-adds the object. docs/vector_env/README.md measured the survivor as the LAST member only
(env2 of 3), which is the same signature as the init-queue eviction bug, so this probe reports the
state PER MEMBER at each phase rather than a single verdict:

    ./run python -u scripts/clara/interactive/t12_vec_chair.py --num_envs 3 --resets 2

For each member it prints whether the watched object is in the scene registry, whether its prim is
on the stage and active (i.e. whether it RENDERS -- the registry and the stage can disagree), and
whether the member's own `_initial_file` still describes it. That last column is the one that says
which of the two candidate causes is in play: if the initial file lists the object in every member,
the restore is doing it and the fix belongs at `update_initial_file()`; if the members differ, the
removal itself did not take everywhere.
"""
import argparse

import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

WATCH = "straight_chair_pmpwwi_0"


def probe(env):
    """(in_registry, renders, in_initial_file, n_objects, world_xyz) for one member."""
    scene = env.omnigibson_env.scene
    names = {o.name for o in scene.objects}
    in_registry = WATCH in names

    stage = getattr(og.sim, "stage", None) or og.sim.get_stage()
    prim = stage.GetPrimAtPath(f"{scene.prim_path}/{WATCH}")
    renders = prim.IsValid() and prim.IsActive()

    init_file = scene._initial_file
    in_init = init_file is not None and WATCH in init_file["objects_info"]["init_info"]

    # Where it actually is matters as much as whether it exists: the earlier measurement of this
    # bug was read off rendered tiles, and an object restored to the wrong tile (or 100 m up, the
    # failure OG-lite ef7442b fixed) is invisible in the frame while being fully present here.
    pos = None
    if in_registry:
        p = scene.object_registry("name", WATCH).get_position_orientation()[0]
        pos = tuple(round(float(v), 3) for v in (p.cpu() if hasattr(p, "cpu") else p))
    return in_registry, renders, in_init, len(names), pos


def report(tag, vec_env):
    rows = []
    print(f"\n[{tag}]  sim init queue: {len(og.sim._objects_to_initialize)} pending", flush=True)
    for i, env in enumerate(vec_env.envs):
        in_registry, renders, in_init, n, pos = probe(env)
        rows.append((in_registry, renders, in_init))
        print(f"    member {i} (scene {env.omnigibson_env.scene.idx}): n_objects={n:4d}  "
              f"{WATCH}: registry={in_registry!s:5s} renders={renders!s:5s} "
              f"in_initial_file={in_init!s:5s} world_xyz={pos}", flush=True)
    return rows


def main(num_envs, resets, task_id, robot, perturbation):
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[perturbation],
        robot=robot,
    )

    print("\n########## vector chair check ##########", flush=True)
    base = report("after construction (scene fixes applied)", vec_env)

    after = []
    for r in range(resets):
        vec_env.reset()
        after.append(report(f"after reset #{r + 1}", vec_env))

    print("\n########## VERDICT ##########", flush=True)
    bad_build = [i for i, (reg, ren, _) in enumerate(base) if reg or ren]
    if bad_build:
        print(f"  UNEXPECTED: the removal did not take at construction for member(s) {bad_build}; "
              f"the bug is in apply_scene_fixes_from_cfg, not in reset().", flush=True)
    survivors = sorted({i for rows in after for i, (reg, ren, _) in enumerate(rows) if reg or ren})
    if not bad_build and not survivors:
        print(f"  PASS -- the removal of {WATCH!r} survives {resets} reset(s) in all {num_envs} "
              f"member(s).", flush=True)
    else:
        print(f"  FAIL -- {WATCH!r} is back after reset in member(s) {survivors} "
              f"(of {num_envs}).", flush=True)
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--resets", type=int, default=2)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    a = p.parse_args()
    main(a.num_envs, a.resets, a.task_id, a.robot, a.perturbation)
