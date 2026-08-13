"""Compare the incremental-contact-cache A/B runs written by profile_step.py.

Statistics deliberately conservative, because the previous fork-level comparison in
docs/perf/og391_step_profile.md died of exactly the mistakes this avoids:

  - The RUN is the unit of replication, not the call. Pooling every per-call sample across runs and
    running a test on the pool is pseudo-replication: it treats 600 correlated calls from one
    process as 600 independent observations and will report p < 1e-9 for a difference that is
    really just machine-state drift between two processes. Per-run medians are the primary
    statistic; the pooled view is printed but labelled.
  - Compare stepping-side quantities, never wall clock -- startup is ~64% of wall.
  - Report the spread, and refuse to call a winner when the within-condition spread is comparable
    to the between-condition gap.

    python scripts/clara/interactive/analyze_ab.py tmp/interactive/prof
"""
import glob
import json
import os
import statistics as st
import sys

KEYS = [
    "RigidContactAPI.update_contact_cache",
    "RigidContactAPI._flush_incremental_accum",
    "RigidContactAPI.add_contacts_from_physics_step",
    "Simulator._non_physics_step",
    "Simulator.step",
    "_sim_context.step(render=True)",
    "_sim_context.step(render=False)",
]


def load(d):
    runs = {"0": [], "1": []}
    for path in sorted(glob.glob(os.path.join(d, "inc*_r*.json"))):
        tag = os.path.basename(path)[:-5]
        inc = tag[3]
        with open(path) as f:
            payload = json.load(f)
        runs.setdefault(inc, []).append((tag, payload))
    return runs


def summarize_run(payload, key):
    v = payload["samples"].get(key)
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "total": sum(s), "median": s[len(s) // 2],
            "mean": sum(s) / len(s), "p90": s[int(0.9 * (len(s) - 1))]}


def main(d):
    runs = load(d)
    for inc in ("0", "1"):
        if not runs.get(inc):
            print(f"no runs found for INCREMENTAL_CONTACT_CACHE={inc} in {d}")
    print(f"\nloaded: " + ", ".join(f"inc{k}={len(v)}" for k, v in runs.items()))

    # Sanity: the flag must actually have differed, or this is a null comparison.
    for inc, rs in runs.items():
        for tag, p in rs:
            meta = p.get("meta", {})
            got = meta.get("incremental")
            flag = "OK " if str(bool(int(inc))) == str(got) else "MISMATCH"
            print(f"  [{flag}] {tag}: gm.INCREMENTAL_CONTACT_CACHE={got} "
                  f"oglite={meta.get('oglite')} wall={meta.get('wall_s')}s")

    for key in KEYS:
        per_run = {}
        for inc in ("0", "1"):
            rows = [(tag, summarize_run(p, key)) for tag, p in runs.get(inc, [])]
            rows = [(t, r) for t, r in rows if r]
            if rows:
                per_run[inc] = rows
        if len(per_run) < 2:
            continue

        print("\n" + "=" * 92)
        print(f"{key}")
        print("=" * 92)
        print(f"{'run':<14}{'calls':>8}{'total_s':>11}{'median_ms':>12}{'mean_ms':>11}{'p90_ms':>10}")
        for inc in ("0", "1"):
            for tag, r in per_run[inc]:
                print(f"{tag:<14}{r['n']:>8}{r['total']:>11.3f}{1000 * r['median']:>12.3f}"
                      f"{1000 * r['mean']:>11.3f}{1000 * r['p90']:>10.3f}")

        # Run-level comparison: the honest one.
        for stat in ("median", "total"):
            a = [r[stat] for _, r in per_run["0"]]
            b = [r[stat] for _, r in per_run["1"]]
            ma, mb = st.median(a), st.median(b)
            unit = 1000 if stat == "median" else 1
            label = "median_ms" if stat == "median" else "total_s"
            spread_a = (max(a) - min(a)) / ma * 100 if ma else 0
            spread_b = (max(b) - min(b)) / mb * 100 if mb else 0
            delta = (mb - ma) / ma * 100 if ma else 0
            print(f"\n  run-level {label}:  off={ma * unit:.3f}  on={mb * unit:.3f}  "
                  f"delta={delta:+.1f}%")
            print(f"    within-condition spread: off={spread_a:.1f}%  on={spread_b:.1f}%  "
                  f"(n={len(a)},{len(b)})")
            # Spread-vs-gap is a crude screen. The question that actually killed the earlier
            # fork comparison was overlap: there, the FASTEST stock run beat every OG-lite run, so
            # the ordering did not survive run selection. Test that directly.
            if max(b) < min(a):
                print(f"    -> SEPARATED: worst 'on' ({max(b) * unit:.3f}) beats best 'off' "
                      f"({min(a) * unit:.3f}); ordering holds for every pairing at n={len(a)},{len(b)}.")
            elif min(b) > max(a):
                print(f"    -> SEPARATED the other way: best 'on' ({min(b) * unit:.3f}) is worse "
                      f"than worst 'off' ({max(a) * unit:.3f}).")
            elif max(spread_a, spread_b) >= abs(delta):
                print(f"    -> NOT RESOLVED: distributions overlap AND within-condition spread "
                      f">= the gap. Need more runs or a lower-variance protocol.")
            else:
                print(f"    -> OVERLAPPING: gap exceeds within-condition spread, but the ranges "
                      f"still cross. Treat as suggestive only.")

        # Pooled per-call view, explicitly labelled so it is never quoted as the result.
        pa = [x for _, p in runs["0"] for x in p["samples"].get(key, [])]
        pb = [x for _, p in runs["1"] for x in p["samples"].get(key, [])]
        if pa and pb:
            print(f"\n  pooled per-call medians (PSEUDO-REPLICATED -- descriptive only): "
                  f"off={1000 * st.median(pa):.3f} ms (n={len(pa)}), "
                  f"on={1000 * st.median(pb):.3f} ms (n={len(pb)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tmp/interactive/prof")
