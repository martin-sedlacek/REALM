"""Phase timing for a REALM eval -- cold start, reset, warmup, per-step.

Deliberately portable across **both** stacks so the two REALM checkouts can be compared directly:

  * OmniGibson 1.1.1  (~/projects/REALM, branch dev, realm-dm.sif, OG-lite bound at /omnigibson-src)
  * OmniGibson 3.9.1  (~/projects/REALM_og391, realm_og391.sif)

Everything it patches exists in both: `omnigibson.Environment.__init__/reset/step`,
`realm.environments.env_dynamic.RealmEnvironmentDynamic.__init__/reset/step`, and the `og.sim`
singleton's bound `step`. Anything missing is reported and skipped rather than raising, so a version
difference degrades the report instead of killing the run.

Usage -- everything after `--` goes verbatim to examples/02_evaluate.py:

    python -u tmp/profile_phases.py --out /logs/prof/og111.json --label og111 -- \
        --task_id 0 --perturbation_id 0 --repeats 3 --max_steps 100 \
        --model_name debug --model_type debug --port 8000 ...

Reporting rules taken from docs/perf/og391_step_profile.md, which were learned the hard way:
  * raw per-sample data is written to JSON BEFORE any summary is printed, so a formatting bug
    cannot destroy a multi-minute measurement;
  * compare **stepping** time, never wall clock -- startup is ~64% of wall and swamps everything;
  * quote the median as well as the mean: the per-step distribution is bimodal (contact-cache
    spikes), so the mean is dragged around by the tail.
"""
import argparse
import atexit
import json
import os
import runpy
import sys
import time

T_PROCESS_START = time.perf_counter()
SAMPLES = {}
META = {"events": {}}
OUT_PATH = None
_since_dump = 0
DUMP_EVERY = 400


def record(key, dt):
    global _since_dump
    SAMPLES.setdefault(key, []).append(dt)
    # Periodic checkpointing, because the process can die without unwinding (see below).
    _since_dump += 1
    if OUT_PATH and _since_dump >= DUMP_EVERY:
        _since_dump = 0
        try:
            dump(OUT_PATH)
        except Exception:
            pass


def mark(name):
    META["events"][name] = round(time.perf_counter() - T_PROCESS_START, 4)


def wrap_attr(owner, attr, key):
    """Time a function attribute in place. Returns False (with a note) if it is absent."""
    fn = getattr(owner, attr, None)
    if fn is None:
        META.setdefault("missing", []).append(key)
        print(f"[phases] NOTE: {key} not present on this stack; skipped")
        return False

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            record(key, time.perf_counter() - t0)

    setattr(owner, attr, timed)
    print(f"[phases] patched {key}")
    return True


def dump(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": META, "samples": SAMPLES}, f)
    print(f"[phases] raw samples -> {path}")


def stats(v):
    s = sorted(v)
    n = len(s)
    return {
        "n": n,
        "total_s": sum(s),
        "mean_ms": 1000 * sum(s) / n,
        "median_ms": 1000 * s[n // 2],
        "p90_ms": 1000 * s[int(0.9 * (n - 1))],
        "max_ms": 1000 * s[-1],
    }


def summarize():
    print("\n" + "=" * 96)
    print(f"PHASE PROFILE  label={META.get('label')}  stack={META.get('og_version')}  "
          f"oglite={META.get('oglite')}")
    print("=" * 96)
    ev = META["events"]
    print("Cold start (seconds from process start):")
    for k in ("omnigibson_imported", "first_env_init_done", "eval_done"):
        if k in ev:
            print(f"  {k:<28} {ev[k]:>10.2f} s")
    if "omnigibson_imported" in ev and "first_env_init_done" in ev:
        print(f"  {'-> isaac import':<28} {ev['omnigibson_imported']:>10.2f} s")
        print(f"  {'-> env creation (scene+robot)':<28} "
              f"{ev['first_env_init_done'] - ev['omnigibson_imported']:>10.2f} s")
    print()
    print(f"{'phase':<42}{'n':>6}{'total_s':>10}{'mean_ms':>11}{'median_ms':>12}{'p90_ms':>10}")
    for key in sorted(SAMPLES):
        st = stats(SAMPLES[key])
        print(f"{key:<42}{st['n']:>6}{st['total_s']:>10.2f}{st['mean_ms']:>11.2f}"
              f"{st['median_ms']:>12.2f}{st['p90_ms']:>10.2f}")
    print("=" * 96)
    print("Compare stepping time (RealmEnv.step / og.sim.step), never wall clock: startup is ~64% "
          "of wall.\nQuote the median too -- per-step time is bimodal because of contact-cache "
          "spikes.")


_finished = False


def finish():
    """Write results exactly once, from whichever exit path fires first."""
    global _finished
    if _finished:
        return
    _finished = True
    mark("eval_done")
    try:
        dump(OUT_PATH)
        summarize()
    except Exception as e:
        print(f"[phases] ERROR while writing results: {type(e).__name__}: {e}")


def install():
    import omnigibson as og
    mark("omnigibson_imported")
    META["og_version"] = getattr(og, "__version__", "?")
    try:
        import omnigibson.utils.usd_utils as uu
        META["og_source"] = uu.__file__
        src = open(uu.__file__).read()
        # The 1.1.1 fork and the 3.9.1 fork carry different markers; record whichever is present.
        META["oglite"] = ("PROXIMITY_GATE" in src) or ("INCREMENTAL_CONTACT_CACHE" in src)
    except Exception as e:
        META["og_source"] = f"<{type(e).__name__}>"

    # REALM level: the phases a user actually cares about.
    try:
        import realm.environments.env_dynamic as ed
        cls = ed.RealmEnvironmentDynamic
        wrap_attr(cls, "__init__", "RealmEnv.__init__")
        wrap_attr(cls, "reset", "RealmEnv.reset")
        wrap_attr(cls, "step", "RealmEnv.step")
    except Exception as e:
        print(f"[phases] NOTE: could not patch RealmEnvironmentDynamic: {type(e).__name__}: {e}")

    # OmniGibson level.
    orig_init = og.Environment.__init__

    def patched_init(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_init(self, *a, **kw)
        finally:
            record("og.Environment.__init__", time.perf_counter() - t0)
            if "first_env_init_done" not in META["events"]:
                mark("first_env_init_done")
            if og.sim is not None and not META.get("sim_patched"):
                META["sim_patched"] = True
                # Simulator is built by a factory in 3.9.1 and has no importable class, so patch
                # the singleton's bound methods rather than the type.
                wrap_attr(og.sim, "step", "og.sim.step")
                wrap_attr(og.sim, "render", "og.sim.render")
                wrap_attr(og.sim, "_non_physics_step", "og.sim._non_physics_step")

    og.Environment.__init__ = patched_init
    wrap_attr(og.Environment, "reset", "og.Environment.reset")
    wrap_attr(og.Environment, "step", "og.Environment.step")

    # examples/02_evaluate.py ends with og.shutdown(), and Isaac's SimulationApp.close() takes the
    # process down hard -- neither atexit handlers nor `finally` blocks run. An earlier version of
    # this profiler registered its dump with atexit and silently produced NO output at all for three
    # completed jobs. Write the results just before handing over to it.
    orig_shutdown = getattr(og, "shutdown", None)
    if orig_shutdown is not None:
        def shutdown(*a, **kw):
            finish()
            return orig_shutdown(*a, **kw)
        og.shutdown = shutdown
        print("[phases] patched og.shutdown (results are written from here)")
    else:
        print("[phases] WARNING: og.shutdown not found; relying on periodic dumps only")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="unlabelled")
    p.add_argument("--target", default="examples/02_evaluate.py")
    a, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    META["label"] = a.label
    META["argv"] = rest

    OUT_PATH = a.out
    install()
    # Belt and braces: og.shutdown is the reliable hook, atexit only catches the paths that unwind.
    atexit.register(finish)

    sys.argv = [a.target] + rest
    runpy.run_path(a.target, run_name="__main__")
