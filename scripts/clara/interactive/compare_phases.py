"""Side-by-side phase comparison of the pre-port (1.1.1) and ported (3.9.1) REALM stacks.

    python scripts/clara/interactive/compare_phases.py /mnt/home_lustre/sedlam56/projects/REALM/logs/phase_ref

Reads every *.json written by profile_phases.py in that directory and prints one table per phase,
plus the cold-start breakdown. Deliberately reports medians alongside means: per-step time is
bimodal (contact-cache spikes), so the mean is dragged by the tail.

Caveat this cannot detect for you: these runs used `--model_type debug`, whose client returns a
CONSTANT action, so the gripper never contacts anything. That keeps the workload identical across
stacks -- which is what makes the comparison fair -- but it exercises only the cheap branch of the
contact cache, so absolute per-step numbers are a floor, not what a pi0.5 rollout costs.
"""
import glob
import json
import os
import sys


def load(d):
    runs = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(path) as f:
                p = json.load(f)
        except Exception as e:
            print(f"  skipping {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue
        if "samples" not in p:
            continue
        runs.append((os.path.basename(path)[:-5], p))
    return runs


def stats(v):
    s = sorted(v)
    n = len(s)
    return {"n": n, "total": sum(s), "mean": sum(s) / n,
            "median": s[n // 2], "p90": s[int(0.9 * (n - 1))]}


PHASES = [
    ("RealmEnv.__init__", "REALM env construction"),
    ("og.Environment.__init__", "  og.Environment.__init__"),
    ("RealmEnv.reset", "reset (per repeat)"),
    ("og.Environment.reset", "  og.Environment.reset"),
    ("RealmEnv.step", "step (per control step)"),
    ("og.sim.step", "  og.sim.step"),
    ("og.sim._non_physics_step", "  og.sim._non_physics_step"),
    ("og.sim.render", "  explicit og.sim.render"),
]


def main(d):
    runs = load(d)
    if not runs:
        print(f"no profile JSONs in {d}")
        return
    print(f"\n{len(runs)} run(s) from {d}\n")
    print(f"{'run':<34}{'stack':>8}{'oglite':>8}{'wall_s':>10}{'argv'}")
    for tag, p in runs:
        m = p["meta"]
        ev = m.get("events", {})
        argv = " ".join(m.get("argv", []))
        argv = (argv[:60] + "...") if len(argv) > 63 else argv
        print(f"{tag:<34}{str(m.get('og_version')):>8}{str(m.get('oglite')):>8}"
              f"{ev.get('eval_done', float('nan')):>10.1f}  {argv}")

    print("\n" + "=" * 100)
    print("COLD START (seconds from process start)")
    print("=" * 100)
    print(f"{'run':<34}{'isaac import':>15}{'env creation':>15}{'total to 1st env':>19}")
    for tag, p in runs:
        ev = p["meta"].get("events", {})
        imp = ev.get("omnigibson_imported")
        fin = ev.get("first_env_init_done")
        if imp is None or fin is None:
            print(f"{tag:<34}{'--':>15}{'--':>15}{'--':>19}")
            continue
        print(f"{tag:<34}{imp:>15.1f}{fin - imp:>15.1f}{fin:>19.1f}")

    for key, label in PHASES:
        rows = [(tag, stats(p["samples"][key])) for tag, p in runs
                if p["samples"].get(key)]
        if not rows:
            continue
        print("\n" + "=" * 100)
        print(f"{label}   [{key}]")
        print("=" * 100)
        print(f"{'run':<34}{'n':>7}{'total_s':>11}{'mean_ms':>11}{'median_ms':>12}{'p90_ms':>11}")
        for tag, st in rows:
            print(f"{tag:<34}{st['n']:>7}{st['total']:>11.2f}{1000 * st['mean']:>11.2f}"
                  f"{1000 * st['median']:>12.2f}{1000 * st['p90']:>11.2f}")
        if len(rows) > 1:
            base_tag, base = rows[0]
            for tag, st in rows[1:]:
                dm = (st["median"] - base["median"]) / base["median"] * 100 if base["median"] else 0
                print(f"    {tag} vs {base_tag}: median {dm:+.1f}%")

    print("\nCompare stepping time, never wall clock -- startup dominates wall.")
    print("These are --model_type debug runs: constant action, no gripper contact, so per-step")
    print("numbers are a floor rather than a pi0.5 rollout cost.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/mnt/home_lustre/sedlam56/projects/REALM/logs/phase_ref")
