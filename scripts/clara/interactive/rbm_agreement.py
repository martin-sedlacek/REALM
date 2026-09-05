#!/usr/bin/env python3
"""How well does the Robometer scorer's verdict track the rubric's, on the same rollouts?

    ./rbm_agreement.py <run_dir> [<run_dir> ...]

<run_dir> is a run_log_dir() directory -- the thing with reports/ under it -- produced by an eval
launched with --robometer. Every rubric-scored row is refused: under --robometer the report carries
BOTH scores for the same rollout (task_progression = Robometer's running max,
rubric_task_progression = the rubric's own max over that same rollout), and that pairing is the
only honest comparison available. A rubric-only run has no Robometer column to compare against, and
a separate rubric run is a DIFFERENT rollout -- see the caveat printed at the bottom.

Reports the 2x2 agreement on the success call, the rank correlation between the two continuous
progress values, and the per-rollout table, so a disagreement can be traced to a rollout and its
video rather than being averaged away.

Stdlib only: it runs on the login node with no env.
"""
import csv
import glob
import os
import sys


def _f(row, key):
    v = (row.get(key) or "").strip()
    if v in ("", "None", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def spearman(xs, ys):
    """Rank correlation, average ranks for ties. Returns None if either side is constant."""
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
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
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def load(run_dirs):
    rows = []
    for d in run_dirs:
        paths = sorted(glob.glob(os.path.join(d, "reports", "*.csv")))
        if not paths:
            print(f"!! no reports/*.csv under {d}", file=sys.stderr)
        for p in paths:
            with open(p, newline="") as fh:
                for row in csv.DictReader(fh):
                    row["_src"] = os.path.basename(p)
                    rows.append(row)
    return rows


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = load(argv[1:])
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    scorers = {(r.get("scorer") or "<absent>") for r in rows}
    if scorers != {"robometer"}:
        print(f"!! expected every row to carry scorer=robometer, found {sorted(scorers)}", file=sys.stderr)
        print("   A rubric-only report has nothing to compare against -- rerun with --robometer.", file=sys.stderr)
        return 1

    thresholds = {(r.get("success_threshold") or "?") for r in rows}
    thr = float(sorted(thresholds)[0]) if len(thresholds) == 1 else None
    if thr is None:
        print(f"!! rows mix success thresholds {sorted(thresholds)}; not poolable", file=sys.stderr)
        return 1

    print(f"{len(rows)} rollouts | success_threshold={thr}")
    print()
    hdr = f"{'#':>3}  {'task':<24} {'pert':<9} {'rbm_TP':>7} {'rubric_TP':>9} {'succ_p':>7} " \
          f"{'rbm_ok':>6} {'rub_ok':>6} {'agree':>6} {'queries':>7} {'stage':<12}"
    print(hdr)
    print("-" * len(hdr))

    rbm_vals, rub_vals = [], []
    tt = tf = ft = ff = 0
    disagreements = []
    for i, r in enumerate(rows):
        rbm = _f(r, "task_progression")
        rub = _f(r, "rubric_task_progression")
        sp = _f(r, "robometer_success_prob")
        q = (r.get("robometer_queries") or "").strip()
        stage = (r.get("stage") or "").strip()
        if rbm is None or rub is None:
            print(f"{i:>3}  {r.get('task','?'):<24} !! missing score column (rbm={rbm} rubric={rub})")
            continue
        rbm_ok = rbm >= thr
        # The rubric's own success rule is "every stage done" == 1.0, NOT the robometer threshold.
        rub_ok = rub >= 1.0
        agree = rbm_ok == rub_ok
        rbm_vals.append(rbm)
        rub_vals.append(rub)
        if rbm_ok and rub_ok:
            tt += 1
        elif rbm_ok and not rub_ok:
            ft += 1          # robometer says success, rubric says no -> false positive
        elif not rbm_ok and rub_ok:
            tf += 1          # robometer misses a real success -> false negative
        else:
            ff += 1
        if not agree:
            disagreements.append((i, r, rbm, rub))
        print(f"{i:>3}  {r.get('task','?'):<24} {r.get('perturbation','?'):<9} "
              f"{rbm:>7.3f} {rub:>9.3f} {('' if sp is None else f'{sp:.3f}'):>7} "
              f"{str(rbm_ok):>6} {str(rub_ok):>6} {('OK' if agree else 'DIFF'):>6} {q:>7} {stage:<12}")

    n = tt + tf + ft + ff
    print()
    print("Success call, Robometer vs rubric (rubric is the benchmark's definition):")
    print(f"    both success        {tt}")
    print(f"    both failure        {ff}")
    print(f"    robometer FP        {ft}   (robometer >= {thr}, rubric incomplete)")
    print(f"    robometer FN        {tf}   (rubric complete, robometer < {thr})")
    if n:
        print(f"    agreement           {(tt + ff)}/{n} = {(tt + ff) / n:.3f}")
    rho = spearman(rbm_vals, rub_vals)
    print(f"    spearman(rbm, rubric progress) = {'n/a (constant or n<3)' if rho is None else f'{rho:+.3f}'}")

    # Does the DEFAULT threshold explain the disagreements, or does the ranking itself fail to
    # separate? Sweeping every threshold that could change a call (midpoints between adjacent
    # observed scores, plus the extremes) answers that. If the best achievable agreement is still
    # poor, no amount of threshold tuning saves it and the problem is the estimate's ordering.
    if rbm_vals:
        truth = [rub >= 1.0 for rub in rub_vals]
        cand = sorted({0.0, 1.0001} | {v for v in rbm_vals}
                      | {(a + b) / 2 for a, b in zip(sorted(rbm_vals), sorted(rbm_vals)[1:])})
        best = max((sum((v >= t) == y for v, y in zip(rbm_vals, truth)), -t) for t in cand)
        best_n, best_t = best[0], -best[1]
        print()
        print(f"    best achievable over ALL thresholds: {best_n}/{n} = {best_n / n:.3f} at "
              f"threshold {best_t:.3f}")
        if best_n == n:
            print(f"    -> the ranking separates perfectly; only the default {thr} is miscalibrated.")
        elif best_n <= (tt + ff):
            print(f"    -> no threshold beats the default. The ordering itself does not separate;")
            print("       recalibrating the threshold cannot fix this.")
        else:
            print(f"    -> a better threshold helps but does not fully separate "
                  f"({n - best_n} rollout(s) still misclassified at the best cut).")
        print(f"    robometer progress range: [{min(rbm_vals):.3f}, {max(rbm_vals):.3f}]"
              f"  (default threshold {thr})")
        if max(rbm_vals) < thr:
            print(f"    !! EVERY rollout scored below the default threshold -- at {thr} this scorer")
            print("       calls every rollout a failure, and the TERMINAL_STEPS countdown never")
            print("       fires, so every rollout also runs to max_steps.")

    if disagreements:
        print()
        print("Disagreements worth watching the video for:")
        for i, r, rbm, rub in disagreements:
            print(f"    row {i}: {r.get('task')} / {r.get('perturbation')} "
                  f"robometer={rbm:.3f} rubric={rub:.3f} stage={r.get('stage')} "
                  f"timestamps={r.get('task_progression_timestamps')}")

    print()
    print("CAVEAT. The threshold drives the TERMINAL_STEPS countdown, so a Robometer-scored rollout")
    print("does not necessarily stop where a rubric-scored one would. rubric_task_progression is the")
    print("rubric's max over the rollout AS ACTUALLY RUN. These counts are honest; a bit-for-bit")
    print("comparison against a separately launched rubric run is not available.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
