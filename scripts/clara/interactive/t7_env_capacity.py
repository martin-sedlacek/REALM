"""How many REALM environments fit on one GPU?

Builds members one at a time and reports GPU memory and host RSS after each, so a single run yields
the per-scene cost and the ceiling instead of one run per candidate `num_envs`. Stops early if free
VRAM drops under --reserve, and reports what it reached rather than dying on an OOM.

Everything is printed and flushed as it goes: if the process is killed anyway, the trace up to that
point is still the answer.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t7_env_capacity.py --max_envs 12 --reserve 3000

NOTE ON WHAT IS BEING MEASURED: `nvidia-smi` reports the whole card, so if a policy server is
resident its footprint is included in `used` -- which is the operationally relevant number, because
that is how REALM actually runs. The per-scene *increment* is reported separately so either
configuration can be computed from it.

This deliberately does NOT play the simulator or step. Stepping throughput at a given N is
t5_vec_sustained.py; conflating the two would mean paying the build cost twice.
"""
import argparse
import subprocess
import time

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config


def gpu_mem():
    """(used_MiB, total_MiB) for the whole card."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return used, total
    except Exception as e:
        return -1, -1


def host_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def main(max_envs, task_id, robot, reserve):
    set_sim_config(robot=robot)
    task_cfg_path = f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml"
    perturbation = SUPPORTED_PERTURBATIONS[0]

    base_used, total = gpu_mem()
    print(f"\n########## env capacity probe ##########", flush=True)
    print(f"  card: {total} MiB total, {base_used} MiB already used before any scene "
          f"(policy server + Isaac boot)", flush=True)
    print(f"  stopping when free VRAM < {reserve} MiB\n", flush=True)
    print(f"{'n_envs':>7}{'gpu_used':>10}{'gpu_free':>10}{'d_gpu':>8}{'host_rss':>10}"
          f"{'build_s':>9}{'cum_s':>8}", flush=True)

    envs = []
    prev_used = base_used
    t_start = time.perf_counter()
    reached = 0
    for i in range(max_envs):
        used, _ = gpu_mem()
        if total > 0 and (total - used) < reserve:
            print(f"  stopping before member {i + 1}: only {total - used} MiB free "
                  f"(< reserve {reserve})", flush=True)
            break
        t0 = time.perf_counter()
        try:
            envs.append(RealmEnvironmentDynamic(
                in_vec_env=True, task_cfg_path=task_cfg_path,
                perturbations=[perturbation], robot=robot))
        except Exception as e:
            print(f"  member {i + 1} FAILED to build: {type(e).__name__}: {e}", flush=True)
            break
        build = time.perf_counter() - t0
        used, _ = gpu_mem()
        reached = i + 1
        print(f"{reached:>7}{used:>10}{total - used:>10}{used - prev_used:>8}"
              f"{host_rss_mb():>10}{build:>9.1f}{time.perf_counter() - t_start:>8.1f}", flush=True)
        prev_used = used

    used, _ = gpu_mem()
    print(f"\n########## SUMMARY ##########")
    print(f"  scenes built (loaded, not played): {reached}")
    print(f"  GPU: {used} / {total} MiB used, {total - used} MiB free")
    if reached >= 2:
        per_scene = (used - base_used) / reached
        print(f"  mean per-scene GPU cost: {per_scene:.0f} MiB")
        if per_scene > 0:
            print(f"  extrapolated ceiling from this baseline: "
                  f"{int((total - base_used - reserve) / per_scene)} scenes "
                  f"(keeping {reserve} MiB free)")
    print(f"  host RSS: {host_rss_mb()} MiB")
    print(f"  NOTE: scenes are loaded but NOT played. play() builds the physics/contact views and\n"
          f"        costs more, so treat this as an upper bound on how many will actually run.",
          flush=True)
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_envs", type=int, default=12)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID_robolab_v2")
    p.add_argument("--reserve", type=int, default=3000, help="MiB of VRAM to keep free")
    a = p.parse_args()
    main(a.max_envs, a.task_id, a.robot, a.reserve)
