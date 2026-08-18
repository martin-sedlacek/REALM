#!/usr/bin/env python3
"""Separability of two REALM experiments, cell by cell, on SR / TP / RF.

    python3 tmp/separability_test.py <expA> <expB> [--reps 25] [--alpha 0.05] [--csv out.csv]

`expA`/`expB` are experiment folders under the log root (a bare name or a path). Checkpoint
subdirectories are paired BY NAME; anything unpaired is reported and skipped.

Purpose: decide whether the benchmark can actually tell two stacks apart (1.1.1 vs og391), and
whether a cell that looks different is resolvable at n=25.

WHY THE TESTS ARE THE ONES THEY ARE (see ~/runbook/references/eval_statistics.md)

  * REALM reuses one RNG stream per `run_id` within a (task, perturbation), so rollout i of A and
    rollout i of B face the same condition. That makes the comparison PAIRED, which roughly doubles
    power for free -- on one real pair the pooled test said p=0.104 and the paired test p=0.049 on
    the same rollouts. Pairing is on (task, perturbation, run_id).
  * SR and RF are per-rollout binary -> exact McNemar (two-sided binomial on the discordant pairs).
  * TP is continuous on [0,1] and not normal -> Wilcoxon signed-rank on the paired differences.
  * Pooled SR is task-STRATIFIED, never a pooled binomial ignoring task strata.
  * Every cell tested counts toward multiplicity: Holm-corrected across cells, since scanning cells
    and reporting the nominal hits is exactly how this project has fooled itself before.

Noise floor to keep in mind: at n=25 per cell the Clopper-Pearson half-width is ~20pp, so a single
cell resolves only large differences. The pooled figure is the one with power.

No scipy: the exact binomial and normal tails are hand-rolled, matching
REALM/scripts/compare_two_runs.py so numbers are comparable with it.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# Bare experiment names resolve under this. Override with REALM_LOGS when running off-cluster:
#   REALM_LOGS=~/Downloads/logs python3 scripts/separability_test.py expA expB
# Absolute or relative paths are used as-is, so REALM_LOGS is optional.
# Task indices as SUPPORTED_TASKS orders them, so --exclude-tasks accepts either form.
TASK_NAMES = ["put_green_block_into_bowl", "put_banana_into_box", "rotate_marker", "rotate_mug",
              "pick_spoon", "pick_water_bottle", "stack_cubes", "push_switch",
              "open_drawer", "close_drawer"]

LOG_ROOT = os.path.expanduser(os.environ.get("REALM_LOGS", "logs"))


# ---- distributions, no scipy -------------------------------------------------------------------

HOW_TO_READ = """
HOW TO READ THIS
----------------
dSR / dRF are percentage points, B minus A. dTP is on TP's own 0-1 scale. Negative means B did
worse. p-values are two-sided. CIs are Clopper-Pearson (exact) for SR/RF and normal-approximation
for TP, which is bounded and skewed, so treat TP's interval as indicative.

The per-cell tables are underpowered by construction. At n=25 the Clopper-Pearson half-width is
about 20pp, so a cell needs roughly a 30pp difference before one cell alone can show it. A "no" in
the separable column means "not resolvable at n=25", NOT "these are the same". Absence of
separability is absence of evidence.

The POOLED task-stratified rows are the ones with power. Read them first. They average per-task
differences instead of pooling rollouts across strata, which stops one task with an extreme rate
from dominating.

  pooled p < alpha   -> the benchmark distinguishes these two arms; the size that matters is d.
  pooled p > alpha   -> it CANNOT distinguish them at this sample size. A statement about power,
                        not a finding of equivalence.

Three failure modes this script exists to stop:
  * a cell counted when one arm is short or its run_ids do not line up -- excluded and listed,
    never silently truncated, because a half-populated cell reads as a real result;
  * multiplicity -- scanning cells and quoting the nominal hits is how this project has previously
    "found" effects that were chance, so every reported cell is Holm-corrected;
  * unpaired comparison -- REALM reuses one RNG stream per run_id, so pairing is free power and the
    unpaired version can miss a real effect (one real pair: pooled p=0.104, paired p=0.049).

RF is a DISPROVEN selection surrogate (rho = -0.554 with SR). Report it, never select on it.
"""


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial. Used for McNemar on the discordant pairs."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * p**i * (1 - p)**(n - i)
    obs = pmf(k)
    # Sum every outcome no more likely than the observed one; 1e-12 absorbs float wobble on ties.
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-12)))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))


def cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact interval, found by bisecting the binomial CDF directly.

    Lower = smallest p with P(X >= k) >= alpha/2; upper = largest p with P(X <= k) >= alpha/2.
    Avoids needing an incomplete-beta implementation.
    """
    if n == 0:
        return 0.0, 1.0
    lo = 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            if 1.0 - binom_cdf(k - 1, n, m) >= alpha / 2:
                b = m
            else:
                a = m
        lo = (a + b) / 2
    hi = 1.0
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            if binom_cdf(k, n, m) >= alpha / 2:
                a = m
            else:
                b = m
        hi = (a + b) / 2
    return lo, hi


def mean_ci(xs, alpha=0.05):
    """Normal-approximation interval for a continuous metric (TP). Not exact -- TP is bounded and
    skewed -- but it is the right order and is labelled as such in the output."""
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), (xs[0] if xs else 0.0)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m)**2 for x in xs) / (n - 1))
    h = 1.959964 * sd / math.sqrt(n)
    return max(0.0, m - h), min(1.0, m + h)


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie and continuity correction.

    Returns (p, n_nonzero). Zero differences are dropped, which is the standard Pratt-free
    treatment; with n<10 non-zero pairs the normal approximation is weak, so the caller should
    report n_nonzero alongside p rather than p alone.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t**3 - t
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0, n
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return min(1.0, 2 * norm_sf(max(z, 0.0))), n


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ---- loading -----------------------------------------------------------------------------------

def resolve(exp):
    """A path if it is one, else a name under LOG_ROOT. Fails loudly with both tried."""
    cand = os.path.expanduser(exp)
    if os.path.isdir(cand):
        return cand
    under = os.path.join(LOG_ROOT, exp)
    if os.path.isdir(under):
        return under
    sys.exit(f"not found as a path or under LOG_ROOT={LOG_ROOT}:\n  {cand}\n  {under}")


def load_experiment(root):
    """-> {ckpt_name: {(task, pert, run_id): (sr, tp, rf)}}

    RF is 1 when the rollout's FINAL stage is REACH, i.e. it never progressed past reaching.
    REALM writes no reachFail column, so it is derived here.
    """
    out = defaultdict(dict)
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    for ckpt in sorted(os.listdir(root)):
        cdir = os.path.join(root, ckpt)
        if not os.path.isdir(cdir):
            continue
        for run in sorted(os.listdir(cdir)):
            rep = os.path.join(cdir, run, "reports")
            if not os.path.isdir(rep):
                continue
            for fn in sorted(os.listdir(rep)):
                if not fn.endswith(".csv"):
                    continue
                for r in csv.DictReader(open(os.path.join(rep, fn))):
                    try:
                        key = (r["task"], r["perturbation"], int(r["run_id"]))
                        out[ckpt][key] = (float(r["binary_SR"]),
                                          float(r["task_progression"]),
                                          1.0 if r["stage"] == "REACH" else 0.0)
                    except (KeyError, ValueError):
                        continue
    return out


# ---- reporting ---------------------------------------------------------------------------------

def mcnemar(a, b):
    """(p, n01, n10, dSR) for paired binary vectors."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    p = binom_two_sided(min(n01, n10), n01 + n10)
    return p, n01, n10, (sum(b) - sum(a)) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expA")
    ap.add_argument("expB")
    ap.add_argument("--reps", type=int, default=25,
                    help="required rollouts per cell in BOTH arms; a cell short of this is excluded "
                         "and listed, never silently truncated")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--exclude-tasks", default="",
                    help="comma-separated task names or indices to drop, e.g. '2,6' or "
                         "'rotate_marker,stack_cubes'. Tasks 2 and 6 spawn their objects off the "
                         "work surface, so their cells measure scene layout rather than policy or "
                         "tone -- excluding them is usually right, but it is never the default: an "
                         "aggregate that silently drops tasks is how a benchmark starts lying.")
    args = ap.parse_args()

    excluded = set()
    for tok in (t.strip() for t in args.exclude_tasks.split(",") if t.strip()):
        if tok.isdigit() and int(tok) < len(TASK_NAMES):
            excluded.add(TASK_NAMES[int(tok)])
        else:
            excluded.add(tok)
    if excluded:
        print(f"EXCLUDED tasks (by request): {', '.join(sorted(excluded))}\n")

    print(HOW_TO_READ)
    A, B = load_experiment(resolve(args.expA)), load_experiment(resolve(args.expB))
    print(f"A = {resolve(args.expA)}\nB = {resolve(args.expB)}\n")

    shared = sorted(set(A) & set(B))
    for lbl, only in (("A only", sorted(set(A) - set(B))), ("B only", sorted(set(B) - set(A)))):
        if only:
            print(f"UNPAIRED checkpoints, skipped ({lbl}): {', '.join(only)}")
    if not shared:
        sys.exit("no checkpoint name appears in both experiments -- nothing to pair")
    print(f"PAIRED checkpoints: {', '.join(shared)}\n")

    rows = []
    for ckpt in shared:
        cellsA = defaultdict(list)
        cellsB = defaultdict(list)
        for (t, p, rid), v in A[ckpt].items():
            cellsA[(t, p)].append((rid, v))
        for (t, p, rid), v in B[ckpt].items():
            cellsB[(t, p)].append((rid, v))

        print(f"===== {ckpt}")
        print(f"{'task':28s} {'pert':10s} {'nA':>4} {'nB':>4} {'paired':>6}  status")
        eligible = []
        for cell in sorted(set(cellsA) | set(cellsB)):
            if cell[0] in excluded:
                continue
            ra = {rid: v for rid, v in cellsA.get(cell, [])}
            rb = {rid: v for rid, v in cellsB.get(cell, [])}
            shared_ids = sorted(set(ra) & set(rb))
            if not ra or not rb:
                status = "MISSING in " + ("B" if ra else "A")
            elif len(ra) < args.reps or len(rb) < args.reps:
                status = f"SHORT (<{args.reps})"
            elif len(shared_ids) < args.reps:
                status = f"run_id mismatch ({len(shared_ids)} paired)"
            else:
                status = "OK"
                eligible.append((cell, [ra[i] for i in shared_ids], [rb[i] for i in shared_ids]))
            print(f"{cell[0]:28s} {cell[1]:10s} {len(ra):>4} {len(rb):>4} {len(shared_ids):>6}  {status}")

        if not eligible:
            print("  no complete paired cell for this checkpoint\n")
            continue

        # ---- per-cell tests ----
        print(f"\n{'task':28s} {'pert':10s} {'dSR':>8} {'pSR':>8} {'dTP':>8} {'pTP':>8} {'dRF':>8} {'pRF':>8}")
        cell_rows = []
        for cell, va, vb in eligible:
            sa, sb = [v[0] for v in va], [v[0] for v in vb]
            ta, tb = [v[1] for v in va], [v[1] for v in vb]
            fa, fb = [v[2] for v in va], [v[2] for v in vb]
            p_sr, n01, n10, d_sr = mcnemar(sa, sb)
            p_tp, n_tp = wilcoxon_signed_rank([y - x for x, y in zip(ta, tb)])
            p_rf, _, _, d_rf = mcnemar(fa, fb)
            d_tp = sum(tb) / len(tb) - sum(ta) / len(ta)
            cell_rows.append(dict(ckpt=ckpt, task=cell[0], pert=cell[1], n=len(va),
                                  dSR=d_sr, pSR=p_sr, n01=n01, n10=n10,
                                  dTP=d_tp, pTP=p_tp, nTP=n_tp, dRF=d_rf, pRF=p_rf))
            print(f"{cell[0]:28s} {cell[1]:10s} {100*d_sr:+7.1f}p {p_sr:8.4f} "
                  f"{d_tp:+8.3f} {p_tp:8.4f} {100*d_rf:+7.1f}p {p_rf:8.4f}")

        # ---- multiplicity: every cell scanned counts ----
        for metric in ("SR", "TP", "RF"):
            adj = holm([r["p" + metric] for r in cell_rows])
            for r, a in zip(cell_rows, adj):
                r["p" + metric + "_holm"] = a
        hits = [(r, m) for r in cell_rows for m in ("SR", "TP", "RF")
                if r["p" + m + "_holm"] < args.alpha]
        print(f"\n  Holm-corrected across {len(cell_rows)} cells x 3 metrics: "
              f"{len(hits)} significant at alpha={args.alpha}")
        for r, m in sorted(hits, key=lambda x: x[0]["p" + x[1] + "_holm"]):
            print(f"    {r['task']:26s} {r['pert']:9s} {m}  d={r['d'+m]:+.3f}  "
                  f"p_holm={r['p'+m+'_holm']:.4f}")

        # ---- pooled, task-stratified: the figure with actual power ----
        print()
        for metric, i in (("SR", 0), ("TP", 1), ("RF", 2)):
            per_task = defaultdict(lambda: ([], []))
            for cell, va, vb in eligible:
                per_task[cell[0]][0].extend(v[i] for v in va)
                per_task[cell[0]][1].extend(v[i] for v in vb)
            diffs, ses = [], []
            for t, (xa, xb) in per_task.items():
                n = len(xa)
                ma, mb = sum(xa) / n, sum(xb) / n
                diffs.append(mb - ma)
                if metric == "TP":
                    va_ = sum((x - ma)**2 for x in xa) / max(n - 1, 1)
                    vb_ = sum((x - mb)**2 for x in xb) / max(n - 1, 1)
                else:
                    va_, vb_ = ma * (1 - ma), mb * (1 - mb)
                ses.append(math.sqrt((va_ + vb_) / n))
            d = sum(diffs) / len(diffs)
            se = math.sqrt(sum(s**2 for s in ses)) / len(ses)
            z = d / se if se > 0 else 0.0
            print(f"  POOLED task-stratified {metric}: A={sum(sum(v[i] for v in va) for _, va, _ in eligible)/sum(len(va) for _, va, _ in eligible):.3f} "
                  f"B={sum(sum(v[i] for v in vb) for _, _, vb in eligible)/sum(len(vb) for _, _, vb in eligible):.3f} "
                  f"d={d:+.4f} SE={se:.4f} z={z:+.2f} p={2*norm_sf(abs(z)):.5f}  ({len(per_task)} tasks)")
        print()
        # ---- rates with intervals, so a difference can be judged against its own spread ----
        print(f"  {'task':28s} {'pert':10s} {'SR_A':>6} {'95% CI':>14}  {'SR_B':>6} {'95% CI':>14}   "
              f"{'TP_A':>6} {'95% CI':>14}  {'TP_B':>6} {'95% CI':>14}")
        for cell, va, vb in eligible:
            n = len(va)
            ka, kb = round(sum(v[0] for v in va)), round(sum(v[0] for v in vb))
            la, ha = cp_ci(ka, n, args.alpha)
            lb, hb = cp_ci(kb, n, args.alpha)
            ta, tb = [v[1] for v in va], [v[1] for v in vb]
            lta, hta = mean_ci(ta, args.alpha)
            ltb, htb = mean_ci(tb, args.alpha)
            print(f"  {cell[0]:28s} {cell[1]:10s} {ka/n:6.3f} [{la:.3f},{ha:.3f}]  {kb/n:6.3f} "
                  f"[{lb:.3f},{hb:.3f}]   {sum(ta)/n:6.3f} [{lta:.3f},{hta:.3f}]  "
                  f"{sum(tb)/n:6.3f} [{ltb:.3f},{htb:.3f}]")
        print()

        # ---- the plain answer: is this cell separable at all ----
        w = max(len(ckpt), 5)
        print(f"  {'model':{w}s} {'task':28s} {'perturbation':12s} separable   on")
        print(f"  {'-'*w} {'-'*28} {'-'*12} ---------   --")
        for r in cell_rows:
            sig = [m for m in ("SR", "TP", "RF") if r["p" + m + "_holm"] < args.alpha]
            print(f"  {ckpt:{w}s} {r['task']:28s} {r['pert']:12s} "
                  f"{'YES' if sig else 'no':9s}   {','.join(sig)}")
        n_sep = sum(1 for r in cell_rows
                    if any(r["p" + m + "_holm"] < args.alpha for m in ("SR", "TP", "RF")))
        print(f"  -> {n_sep} of {len(cell_rows)} cells separable at alpha={args.alpha} "
              f"(Holm-corrected over {3*len(cell_rows)} tests)\n")

        rows.extend(cell_rows)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")



if __name__ == "__main__":
    main()
