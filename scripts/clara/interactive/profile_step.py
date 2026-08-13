"""Time the contact cache and _non_physics_step around a real REALM eval.

Recreates the measurement harness described in docs/perf/og391_step_profile.md (the original
tmp/fork_ab_profile.py was gitignored and did not survive the machine it was written on). Used to
A/B gm.INCREMENTAL_CONTACT_CACHE on vs off, which is the lever that doc ranks first: the contact
cache is ~50% of stepping and ~98% of _non_physics_step.

Usage -- everything after `--` is passed verbatim to examples/02_evaluate.py:

    python -u scripts/clara/interactive/profile_step.py --out tmp/interactive/prof/inc_on.json -- \
        --task_id 0 --perturbation_id 0 --repeats 3 --max_steps 100 \
        --model_name debug --model_type debug --port 8000 ...

Instrumentation gotchas this respects (all learned the hard way, see the perf doc):
  1. og.sim is None until og.Environment.__init__ runs, and Simulator is defined inside a factory
     function so it has no importable class -- patch the singleton's BOUND methods from a wrapper
     around Environment.__init__.
  2. Simulator.step() renders internally via _sim_context.step(render=True). Patching og.sim.render
     catches only explicit render calls, so physics-vs-render needs _sim_context.step keyed on its
     render= argument.
  3. Raw per-sample data is written to JSON BEFORE any summary is printed. A formatting bug must
     not destroy a multi-minute measurement.
  4. Compare stepping time, never wall clock -- startup is ~64% of wall and swamps everything.
"""
import argparse
import atexit
import json
import os
import runpy
import sys
import time

SAMPLES = {}
META = {}
OUT_PATH = None
_since_dump = 0
DUMP_EVERY = 400
_finished = False


def record(key, dt):
    global _since_dump
    SAMPLES.setdefault(key, []).append(dt)
    _since_dump += 1
    if OUT_PATH and _since_dump >= DUMP_EVERY:
        _since_dump = 0
        try:
            dump(OUT_PATH)
        except Exception:
            pass


def finish():
    """Write results exactly once, from whichever exit path fires first."""
    global _finished
    if _finished:
        return
    _finished = True
    try:
        dump(OUT_PATH)
        summarize()
    except Exception as e:
        print(f"[prof] ERROR while writing results: {type(e).__name__}: {e}")


def wrap(obj, attr, key):
    """Patch a bound method in place, timing every call. Returns False if the attr is absent."""
    fn = getattr(obj, attr, None)
    if fn is None:
        print(f"[prof] WARNING: {attr} not found on {obj!r}; not timed")
        return False

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            record(key, time.perf_counter() - t0)

    setattr(obj, attr, timed)
    print(f"[prof] patched {key}")
    return True


def dump(out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"meta": META, "samples": SAMPLES}
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[prof] raw samples -> {out_path}")


def summarize():
    def stats(v):
        if not v:
            return None
        s = sorted(v)
        n = len(s)
        return {
            "n": n,
            "total_s": round(sum(s), 4),
            "mean_ms": round(1000 * sum(s) / n, 3),
            "median_ms": round(1000 * s[n // 2], 3),
            "p90_ms": round(1000 * s[int(0.9 * (n - 1))], 3),
            "max_ms": round(1000 * s[-1], 3),
        }

    print("\n" + "=" * 78)
    print(f"PROFILE SUMMARY  (INCREMENTAL_CONTACT_CACHE={META.get('incremental')}, "
          f"PROXIMITY_GATE={META.get('gate')})")
    print("=" * 78)
    print(f"{'key':<34}{'n':>6}{'total_s':>10}{'mean_ms':>10}{'median_ms':>11}{'p90_ms':>9}")
    for key in sorted(SAMPLES):
        st = stats(SAMPLES[key])
        print(f"{key:<34}{st['n']:>6}{st['total_s']:>10.3f}{st['mean_ms']:>10.3f}"
              f"{st['median_ms']:>11.3f}{st['p90_ms']:>9.3f}")
    print("=" * 78)


def install():
    """Patch everything that needs a live og.sim, from a wrapper around Environment.__init__."""
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.utils.usd_utils as uu

    # The contact API is a module-level singleton instance, so its bound methods can be patched
    # directly and before any simulator exists.
    api = uu.RigidContactAPI
    wrap(api, "update_contact_cache", "RigidContactAPI.update_contact_cache")
    wrap(api, "add_contacts_from_physics_step", "RigidContactAPI.add_contacts_from_physics_step")
    if hasattr(api, "_flush_incremental_accumulators"):
        wrap(api, "_flush_incremental_accumulators", "RigidContactAPI._flush_incremental_accum")

    orig_init = og.Environment.__init__

    def patched_init(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_init(self, *a, **kw)
        finally:
            record("Environment.__init__", time.perf_counter() - t0)
            if og.sim is not None and not META.get("sim_patched"):
                META["sim_patched"] = True
                # Bound methods on the singleton: Simulator has no importable class.
                wrap(og.sim, "_non_physics_step", "Simulator._non_physics_step")
                wrap(og.sim, "step", "Simulator.step")
                wrap(og.sim, "render", "Simulator.render_explicit")
                sc = getattr(og.sim, "_sim_context", None)
                if sc is not None:
                    inner = sc.step

                    def timed_step(*sa, **skw):
                        rendered = skw.get("render", sa[0] if sa else None)
                        t = time.perf_counter()
                        try:
                            return inner(*sa, **skw)
                        finally:
                            record(f"_sim_context.step(render={rendered})",
                                   time.perf_counter() - t)

                    sc.step = timed_step
                    print("[prof] patched _sim_context.step")
                META["incremental"] = bool(getattr(gm, "INCREMENTAL_CONTACT_CACHE", False))
                META["gate"] = getattr(gm, "PROXIMITY_GATE_ENABLED", "<undef>")
                META["usd_utils"] = uu.__file__
                META["oglite"] = "PROXIMITY_GATE" in open(uu.__file__).read()

    og.Environment.__init__ = patched_init

    # examples/02_evaluate.py ends with og.shutdown(), and Isaac's SimulationApp.close() takes the
    # process down hard: atexit handlers and `finally` blocks do NOT run. Registering the dump with
    # atexit alone silently produced no output at all for four completed A/B runs. Write here.
    orig_shutdown = getattr(og, "shutdown", None)
    if orig_shutdown is not None:
        def shutdown(*a, **kw):
            finish()
            return orig_shutdown(*a, **kw)
        og.shutdown = shutdown
        print("[prof] patched og.shutdown (results are written from here)")
    else:
        print("[prof] WARNING: og.shutdown not found; relying on periodic dumps only")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="examples/02_evaluate.py")
    a, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    META["argv"] = rest
    META["env"] = {k: os.environ.get(k) for k in
                   ("REALM_INCREMENTAL_CONTACT_CACHE", "REALM_PROXIMITY_GATE")}

    OUT_PATH = a.out
    install()
    # Belt and braces: og.shutdown is the reliable hook; atexit only catches paths that unwind.
    atexit.register(finish)

    sys.argv = [a.target] + rest
    t0 = time.perf_counter()
    try:
        runpy.run_path(a.target, run_name="__main__")
    finally:
        META["wall_s"] = round(time.perf_counter() - t0, 3)
