#!/usr/bin/env python3
"""Read rbm_replay.py's JSON and answer: does a rubric-specific prompt help, and is the wrist any use?

    ./rbm_replay_analyze.py /path/to/rbm_replay.json

Every condition scored the SAME pixels of the SAME rollouts, so differences between conditions are
attributable to the condition -- unlike a re-run, where pi0.5's non-determinism moves the rollouts
underneath the comparison.

The figure of merit is deliberately NOT agreement at the shipped 0.9: nothing clears 0.9, so that
number is 'always failure' for every condition and cannot rank them. Instead:
  separation  = best achievable accuracy over ALL thresholds (how well the ordering could ever do)
  spearman    = rank correlation with the rubric
  AUC         = P(random success scores above random failure); 0.5 is chance, threshold-free
AUC is the honest headline: it needs no cut, so it cannot be flattered by fitting one at n=8.

Stdlib only.
"""
import json
import sys
from itertools import product


def spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def auc(scores, labels):
    """Mann-Whitney AUC with ties at 0.5. None when one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    tot = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a, b in product(pos, neg))
    return tot / (len(pos) * len(neg))


def best_acc(scores, labels):
    cand = sorted({0.0, 1.01} | set(scores)
                  | {(a + b) / 2 for a, b in zip(sorted(scores), sorted(scores)[1:])})
    n, t = max((sum((s >= k) == y for s, y in zip(scores, labels)), -k) for k in cand)
    return n, -t


# Filled per dataset: a 2-tile run has base/wrist, a --multi-view run adds base2.
CONDS = []


def build_conds(rows):
    cams = sorted({c for r in rows for c in r.get("cameras", ["base", "wrist"])})
    conds = [f"{c}_{p}" for c in cams for p in ("orig", "enh")]
    conds += ["pool_max"]                       # max over ALL cameras, enhanced prompt
    if "base2" in cams:
        # The occlusion question, isolated: do the two EXTERIORS together beat either alone, and
        # does adding the wrist on top add anything the exteriors did not already have?
        conds += ["ext_max", "ext_mean", "all_max"]
    return cams, conds


def enrich(rows):
    global CONDS
    cams, CONDS = build_conds(rows)
    for r in rows:
        r["pool_max"] = max(r[f"{c}_enh"] for c in cams)
        r["pool_mean"] = sum(r[f"{c}_enh"] for c in cams) / len(cams)
        if "base2" in cams:
            r["ext_max"] = max(r["base_enh"], r["base2_enh"])
            r["ext_mean"] = (r["base_enh"] + r["base2_enh"]) / 2
            r["all_max"] = max(r[f"{c}_enh"] for c in cams)
        r["label"] = r["rubric"] >= 1.0
    return rows


def report(rows, title):
    labels = [r["label"] for r in rows]
    n, npos = len(rows), sum(labels)
    print(f"\n=== {title}  (n={n}, rubric successes={npos})")
    if npos in (0, n):
        print(f"    only one class present -- AUC/separation undefined, skipping")
        return
    print(f"    {'condition':<12}{'AUC':>7}{'spearman':>10}{'best acc':>10}{'@thr':>8}{'range':>18}")
    for c in CONDS:
        s = [r[c] for r in rows]
        a = auc(s, labels)
        rho = spearman(s, [float(r["rubric"]) for r in rows])
        acc, t = best_acc(s, labels)
        print(f"    {c:<12}{(f'{a:.3f}' if a is not None else 'n/a'):>7}"
              f"{(f'{rho:+.3f}' if rho is not None else 'n/a'):>10}"
              f"{acc:>7}/{n}{t:>8.3f}   [{min(s):.3f}, {max(s):.3f}]")


def main(path):
    rows = enrich(json.load(open(path)))

    # Control: does replaying compressed video reproduce the live scorer's LAST trace entry?
    ok = [(r["base_orig"], r["live_trace_last"]) for r in rows if r.get("live_trace_last") is not None]
    if ok:
        d = [abs(a - b) for a, b in ok]
        rho = spearman([a for a, _ in ok], [b for _, b in ok])
        print(f"CONTROL  replay(base,orig) vs live last trace entry, n={len(ok)}")
        print(f"    mean|delta| {sum(d)/len(d):.4f}   max|delta| {max(d):.4f}   "
              f"spearman {(f'{rho:+.3f}' if rho is not None else 'n/a')}")
        print("    (large deltas => H.264/resize is interfering and the conditions below are suspect)")

    for rid in dict.fromkeys(r["run_id"] for r in rows):
        sub = [r for r in rows if r["run_id"] == rid]
        report(sub, f"{sub[0]['task']}")

    report(rows, "POOLED over all tasks")

    # Wrist-vs-base agreement is its own question: do the two cameras rank rollouts the same way?
    cams = sorted({c for r in rows for c in r.get("cameras", ["base", "wrist"])})
    pairs = [(a, b) for i, a in enumerate(cams) for b in cams[i + 1:]]
    print("\n=== do the cameras agree with each other? (rank corr of *_enh)")
    print("    " + f"{'task':<28}" + "".join(f"{a}~{b}".ljust(16) for a, b in pairs))
    for rid in dict.fromkeys(r["run_id"] for r in rows):
        sub = [r for r in rows if r["run_id"] == rid]
        cells = []
        for a, b in pairs:
            rho = spearman([r[f"{a}_enh"] for r in sub], [r[f"{b}_enh"] for r in sub])
            cells.append(f"{rho:+.3f}" if rho is not None else "n/a")
        print(f"    {sub[0]['task']:<28}" + "".join(f"{c:<16}" for c in cells))
    cells = []
    for a, b in pairs:
        rho = spearman([r[f"{a}_enh"] for r in rows], [r[f"{b}_enh"] for r in rows])
        cells.append(f"{rho:+.3f}" if rho is not None else "n/a")
    print(f"    {'POOLED':<28}" + "".join(f"{c:<16}" for c in cells))

    # Occlusion test: if low exterior scores are occlusion, the SECOND exterior should score HIGH on
    # exactly the rollouts where the first scored LOW -- i.e. max(base, base2) should beat both, and
    # per-rollout disagreement should be large. If instead the model simply has no completion
    # concept on the task, the second angle reshuffles the ranking without lifting the values.
    if "base2" in cams:
        print("\n=== occlusion probe: does a second EXTERIOR angle lift the values?")
        print(f"    {'task':<28}{'base max':>10}{'base2 max':>11}{'ext_max':>9}{'lift':>8}")
        for rid in dict.fromkeys(r["run_id"] for r in rows):
            sub = [r for r in rows if r["run_id"] == rid]
            b = max(r["base_enh"] for r in sub)
            b2 = max(r["base2_enh"] for r in sub)
            em = max(r["ext_max"] for r in sub)
            print(f"    {sub[0]['task']:<28}{b:>10.3f}{b2:>11.3f}{em:>9.3f}{em - b:>+8.3f}")

    print("\n=== prompt effect, per condition (enhanced minus original)")
    for cam in cams:
        d = [r[f"{cam}_enh"] - r[f"{cam}_orig"] for r in rows]
        ds = [r[f"{cam}_enh"] - r[f"{cam}_orig"] for r in rows if r["label"]]
        df = [r[f"{cam}_enh"] - r[f"{cam}_orig"] for r in rows if not r["label"]]
        print(f"    {cam:<6} mean {sum(d)/len(d):+.4f}   on successes {sum(ds)/max(1,len(ds)):+.4f}"
              f"   on failures {sum(df)/max(1,len(df)):+.4f}"
              f"   (want: up on successes, down on failures)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/mnt/home_lustre/sedlam56/projects/REALM/logs/rbm_replay.json")
