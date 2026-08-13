"""Profiled throughput scaling for vector envs, in the MAX-JUICE configuration only.

Max juice = MODE=oglite + gm.INCREMENTAL_CONTACT_CACHE + gm.PROXIMITY_GATE_ENABLED (both default ON
as of 2026-08-13) + ENABLE_VISUAL_UPDATES=False + OBJECT_STATE_UPDATE_WHITELIST=["ToggledOn"] +
**render-on-demand**. That is what a production eval runs, so it is the only configuration whose
scaling numbers mean anything. Rendering every step is not measured here on purpose.

ROD cadence without a policy in the loop: render every `--horizon`-th step. That is exactly what
realm/vector_eval.py does once members are in phase, and they are, because every active member pops
one action per step and refills on the same chunk boundary.

Unlike t5_vec_sustained.py this also PROFILES, decomposing each shared step into:

    vec.pre_step   -- applying N actions to N robots
    og.sim.step    -- physics substeps + in-step render, and inside it _non_physics_step
    og.sim.render  -- the explicit pre-obs flush passes
    vec.post_step  -- reading N members' observations back

That decomposition is the point: in a vector env `pre_step` and `post_step` are O(N) Python/readback
work while `og.sim.step` is one batched call, so it says which of them stops scaling.

    MODE=oglite ./scripts/clara/interactive/rr \
        python -u scripts/clara/interactive/t8_vec_scaling.py --num_envs 16 --steps 96
"""
import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_rendering_mode, set_sim_config

T = defaultdict(list)
TABLE_Z_MIN = 0.5


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        return tuple(int(x.strip()) for x in out.split(","))
    except Exception:
        return (-1, -1)


def wrap(obj, attr, key):
    fn = getattr(obj, attr, None)
    if fn is None:
        print(f"[prof] NOTE: {attr} absent; not timed", flush=True)
        return

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            T[key].append(time.perf_counter() - t0)

    setattr(obj, attr, timed)
    print(f"[prof] patched {key}", flush=True)


def install_sim_probes():
    import omnigibson.utils.usd_utils as uu
    wrap(og.sim, "_non_physics_step", "og.sim._non_physics_step")
    wrap(og.sim, "render", "og.sim.render (explicit flush)")
    wrap(uu.RigidContactAPI, "update_contact_cache", "RigidContactAPI.update_contact_cache")
    wrap(uu.RigidContactAPI, "add_contacts_from_physics_step", "RigidContactAPI.add_from_substep")
    sc = getattr(og.sim, "_sim_context", None)
    if sc is not None:
        inner = sc.step

        def timed(*a, **kw):
            rendered = kw.get("render", a[0] if a else None)
            t0 = time.perf_counter()
            try:
                return inner(*a, **kw)
            finally:
                T[f"_sim_context.step(render={rendered})"].append(time.perf_counter() - t0)

        sc.step = timed
        print("[prof] patched _sim_context.step", flush=True)


def stats(v):
    s = sorted(v)
    n = len(s)
    return n, sum(s), 1000 * sum(s) / n, 1000 * s[n // 2], 1000 * s[int(0.9 * (n - 1))]


def main(num_envs, steps, horizon, task_id, robot, out, pre_render_mode=None):
    set_sim_config(robot=robot)
    from omnigibson.macros import gm
    t_build = time.perf_counter()
    def _early_renderer():
        # Applied after member 0 (Isaac is up) but before the other N-1 scenes load, which is the
        # only window where it can affect how the RTX pools are sized.
        if pre_render_mode:
            print(f"[cfg ] applying rendering_mode={pre_render_mode!r} before the remaining scenes",
                  flush=True)
            set_rendering_mode(pre_render_mode)

    vec_env = RealmVectorEnvironment(
        num_envs, task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[SUPPORTED_PERTURBATIONS[0]], robot=robot,
        on_first_env_built=_early_renderer)
    build_s = time.perf_counter() - t_build
    used, total = gpu_mem()
    print(f"\n[cfg ] INCREMENTAL_CONTACT_CACHE={getattr(gm,'INCREMENTAL_CONTACT_CACHE',None)} "
          f"PROXIMITY_GATE={getattr(gm,'PROXIMITY_GATE_ENABLED',None)} "
          f"VISUAL_UPDATES={getattr(gm,'ENABLE_VISUAL_UPDATES',None)}", flush=True)
    print(f"[mem ] {num_envs} envs built+played in {build_s:.0f}s: {used}/{total} MiB, "
          f"{total - used} free", flush=True)

    vec_env.warmup()
    install_sim_probes()
    ee = [e.warmup_ee_cmd() for e in vec_env.envs]

    print(f"\n##### MAX JUICE scaling: {num_envs} members x {steps} steps, ROD every {horizon} #####",
          flush=True)
    step_t, pre_t, post_t, sim_t = [], [], [], []
    for t in range(steps):
        actions = [e.warmup_action(t, c) for e, c in zip(vec_env.envs, ee)]
        need_render = (t % horizon) == 0
        t0 = time.perf_counter()
        with og.sim.render_on_step(need_render):
            t1 = time.perf_counter()
            for env, a in zip(vec_env.envs, actions):
                env.pre_step(a)
            t2 = time.perf_counter()
            og.sim.step()
            for _ in range(1 if need_render else 0):
                og.sim.render()
            t3 = time.perf_counter()
            results = [env.post_step(a) for env, a in zip(vec_env.envs, actions)]
            t4 = time.perf_counter()
        step_t.append(t4 - t0); pre_t.append(t2 - t1); sim_t.append(t3 - t2); post_t.append(t4 - t3)

    mo_z = [float(e.main_objects[0].get_position_orientation()[0][2]) for e in vec_env.envs]
    fell = [i for i, z in enumerate(mo_z) if z < TABLE_Z_MIN]

    per_step = 1000 * float(np.mean(step_t))
    print(f"\n########## RESULT: {num_envs} members ##########")
    print(f"  ms per shared step   : {per_step:.1f}   (median {1000*np.median(step_t):.1f})")
    print(f"  ms per MEMBER-step   : {per_step / num_envs:.2f}")
    print(f"  member-steps / s     : {1000 * num_envs / per_step:.1f}")
    print(f"  first/last quarter   : {1000*np.mean(step_t[:steps//4]):.1f} / "
          f"{1000*np.mean(step_t[-steps//4:]):.1f} ms")
    print(f"  objects fell off table: {fell if fell else 'none'}")
    print(f"\n  {'phase':<24}{'ms/step':>10}{'share':>8}")
    for name, arr in (("vec.pre_step (N robots)", pre_t), ("og.sim.step + flush", sim_t),
                      ("vec.post_step (N obs)", post_t)):
        m = 1000 * float(np.mean(arr))
        print(f"  {name:<24}{m:>10.1f}{100*m/per_step:>7.0f}%")
    print(f"\n  {'probe':<40}{'n':>7}{'total_s':>10}{'mean_ms':>10}{'median_ms':>11}{'p90_ms':>10}")
    for k in sorted(T):
        n, tot, mean, med, p90 = stats(T[k])
        print(f"  {k:<40}{n:>7}{tot:>10.2f}{mean:>10.2f}{med:>11.2f}{p90:>10.2f}")

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"num_envs": num_envs, "steps": steps, "horizon": horizon,
                       "build_s": build_s, "gpu_used": used, "gpu_total": total,
                       "ms_per_shared_step": per_step,
                       "ms_per_member_step": per_step / num_envs,
                       "member_steps_per_s": 1000 * num_envs / per_step,
                       "phases_ms": {"pre_step": 1000*float(np.mean(pre_t)),
                                     "sim_step": 1000*float(np.mean(sim_t)),
                                     "post_step": 1000*float(np.mean(post_t))},
                       "probes": {k: list(v) for k, v in T.items()}}, f)
        print(f"\n  raw -> {out}")
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--robot", type=str, default="DROID_robolab_v2")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--pre_render_mode", type=str, default=None,
                   help="renderer profile to apply before the bulk of scenes load "
                        "('r' = performance RTX: no reflections/AO/shadows/denoiser)")
    a = p.parse_args()
    main(a.num_envs, a.steps, a.horizon, a.task_id, a.robot, a.out, a.pre_render_mode)
