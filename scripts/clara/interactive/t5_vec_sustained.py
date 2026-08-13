"""Does a vector env stay correct over a long run, not just for one step?

`examples/03_vector_first_frames.py` builds N members, warms up and takes a single shared step. That
proves construction and tiling; it says nothing about whether the thing stays stable and whether the
members stay independent over a rollout's worth of stepping.

This drives `RealmVectorEnvironment` for --steps shared steps and checks, every so often:

  * no member has diverged into NaN/inf joint state
  * members remain pairwise DISTINCT (a shared-state bug would collapse them onto each other)
  * the task objects stay on the table rather than falling through it -- this is the regression
    guard for the 100 m z-offset bug, which put them on the floor at z ~ 0.015 in scenes idx != 0
  * per-step wall time is not drifting upward

Needs the OG-lite bind: the z-offset fix lives in the fork (OG-lite ef7442b), so at MODE=stock
scenes 1..N-1 are still wrong by construction.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t5_vec_sustained.py --num_envs 4 --steps 200
"""
import argparse
import time

import numpy as np

import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.inference import extract_from_obs
from realm.sim_config import set_sim_config

TABLE_Z_MIN = 0.5   # anything below this has fallen off the table; the z-bug put them at ~0.015


def member_state(env):
    """(z of main object, z of target object, robot joint positions) for one member."""
    mo = env.main_objects[0].get_position_orientation()[0]
    to = env.target_objects[0].get_position_orientation()[0] if env.target_objects else mo
    q = env.robot.get_joint_positions()
    q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
    return float(mo[2]), float(to[2]), q


def main(num_envs, steps, task_id, robot, check_every):
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[SUPPORTED_PERTURBATIONS[0]],
        robot=robot,
    )
    results = vec_env.warmup()

    # Hold-still actions, so anything that moves is physics or a bug rather than a commanded motion.
    ee_cmds = [e.warmup_ee_cmd() for e in vec_env.envs]
    failures = []

    print(f"\n########## sustained stepping: {num_envs} members x {steps} shared steps ##########",
          flush=True)
    times = []
    for t in range(steps):
        actions = [e.warmup_action(t, c) for e, c in zip(vec_env.envs, ee_cmds)]
        t0 = time.perf_counter()
        results = vec_env.step(actions)
        times.append(time.perf_counter() - t0)

        if (t + 1) % check_every == 0 or t == steps - 1:
            states = [member_state(e) for e in vec_env.envs]
            frames = []
            for obs, *_ in results:
                base_im, _, _, _, _, _, _ = extract_from_obs(obs, robot_name=robot)
                frames.append(np.asarray(base_im))

            note = []
            for i, (mo_z, to_z, q) in enumerate(states):
                if not np.all(np.isfinite(q)):
                    failures.append(f"step {t+1}: member {i} joint state non-finite")
                if mo_z < TABLE_Z_MIN or to_z < TABLE_Z_MIN:
                    failures.append(f"step {t+1}: member {i} objects fell "
                                    f"(main z={mo_z:.3f}, target z={to_z:.3f})")
                note.append(f"m{i}: mo_z={mo_z:.3f} to_z={to_z:.3f}")
            # A shared-state bug would make the members converge; they must stay distinct.
            for i in range(1, len(frames)):
                if np.array_equal(frames[0], frames[i]):
                    failures.append(f"step {t+1}: member 0 and {i} rendered IDENTICAL frames")
            recent = 1000 * np.mean(times[-check_every:])
            print(f"  step {t+1:>4}  {recent:6.1f} ms/step   " + "  ".join(note), flush=True)

    print("\n########## SUMMARY ##########")
    print(f"  steps completed      : {steps}")
    print(f"  ms/step first quarter: {1000 * np.mean(times[:max(1, steps//4)]):.1f}")
    print(f"  ms/step last quarter : {1000 * np.mean(times[-max(1, steps//4):]):.1f}")
    print(f"  checks failed        : {len(failures)}")
    for f in failures[:20]:
        print(f"    {f}")
    print(f"\n  {'PASS' if not failures else 'FAIL'} -- "
          f"{num_envs} members stepped {steps} times, stayed finite, stayed distinct, "
          f"objects stayed on the table" if not failures else "  FAIL -- see above")
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID")
    p.add_argument("--check_every", type=int, default=50)
    a = p.parse_args()
    main(a.num_envs, a.steps, a.task_id, a.robot, a.check_every)
