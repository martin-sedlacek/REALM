#!/usr/bin/env python3
"""Can a REWEIGHTED pool + refitted threshold make Robometer track the privileged-state rubric?

    ./rbm_fit_pool.py /path/to/rbm_replay.json

Fits a logistic probe on the 8 replay features (4 progress + 4 success-head, base/wrist x
orig/enhanced prompt) against the rubric's binary success, and reports BOTH:

  IN-SAMPLE     fitted and scored on the same 40 rollouts. This is the number that looks good and
                means nothing on its own -- 8 free weights + an intercept on n=40 with 22 positives.
  LOTO          leave-one-TASK-out: fit on 4 tasks, predict the held-out 5th. This is the question
                actually being asked, because a scorer is deployed on tasks it was not tuned on.
  LORO          leave-one-ROLLOUT-out within the pooled set. Optimistic relative to LOTO because
                the held-out rollout's own task is still in the training set, i.e. the fit has seen
                that task's score scale. The LORO-vs-LOTO gap IS the cross-task generalisation gap.

Also tests per-task standardisation (z-score / rank within the task's own batch of repeats). That
needs NO privileged labels -- only the other rollouts of the same cell, which REALM always has --
so it is a legitimate deployable transform, unlike a per-task threshold fitted on rubric labels.

The headline metric is not accuracy but **SR error**: REALM reports a success rate, so what matters
is |predicted SR - rubric SR| per task.

numpy + scipy only.
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize

FEATS = ["base_orig", "base_enh", "wrist_orig", "wrist_enh",
         "base_orig_succ", "base_enh_succ", "wrist_orig_succ", "wrist_enh_succ"]


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def fit_logistic(X, y, l2=1.0):
    """L2-regularised logistic regression. l2 is deliberately strong: n=40, 8 features."""
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def nll(w):
        z = Xb @ w
        # log(1+exp(z)) computed stably
        ll = np.sum(np.logaddexp(0, z) - y * z)
        return ll + l2 * np.sum(w[:-1] ** 2)

    r = minimize(nll, np.zeros(Xb.shape[1]), method="L-BFGS-B")
    return r.x


def predict(w, X):
    return 1.0 / (1.0 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def zscore_by_task(X, tasks):
    """Standardise each feature within each task's own batch. No labels used."""
    Z = X.copy()
    for t in set(tasks):
        m = tasks == t
        mu, sd = X[m].mean(0), X[m].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z[m] = (X[m] - mu) / sd
    return Z


def best_acc(s, y):
    cand = np.unique(np.concatenate([[-1e9, 1e9], s, (np.sort(s)[:-1] + np.sort(s)[1:]) / 2]))
    accs = [(np.mean((s >= c) == y), c) for c in cand]
    a, c = max(accs)
    return a, c


def main(path):
    rows = json.load(open(path))
    tasks = np.array([r["task"] for r in rows])
    y = np.array([1 if r["rubric"] >= 1.0 else 0 for r in rows])
    X = np.array([[float(r[f] if r[f] is not None else 0.0) for f in FEATS] for r in rows])
    uniq = list(dict.fromkeys(tasks))

    print(f"n={len(y)} rollouts, {y.sum()} rubric successes, {len(uniq)} tasks, {X.shape[1]} features")
    print(f"true SR per task: " + "  ".join(f"{t.split('_')[0]}={y[tasks==t].mean():.3f}" for t in uniq))

    def evaluate(name, scores):
        a = auc(scores, y)
        acc, thr = best_acc(scores, y)
        # SR error at the globally best threshold
        pred = scores >= thr
        errs = [abs(pred[tasks == t].mean() - y[tasks == t].mean()) for t in uniq]
        print(f"  {name:<34}{a:>7.3f}{acc:>9.3f}{np.mean(errs):>11.3f}{max(errs):>10.3f}")
        return a

    print("\n" + "=" * 74)
    print(f"  {'model':<34}{'AUC':>7}{'best acc':>9}{'mean|dSR|':>11}{'max|dSR|':>10}")
    print("-" * 74)
    print("BASELINES (no fitting)")
    evaluate("base_orig  (what --robometer uses)", X[:, 0])
    evaluate("max(base_enh, wrist_enh)", np.maximum(X[:, 1], X[:, 3]))

    print("\nFITTED, IN-SAMPLE (upper bound, not a result)")
    w = fit_logistic(X, y)
    evaluate("logistic, 8 feats, raw", predict(w, X))
    Z = zscore_by_task(X, tasks)
    wz = fit_logistic(Z, y)
    evaluate("logistic, 8 feats, per-task z", predict(wz, Z))

    print("\nLORO  (leave-one-ROLLOUT-out; held-out task still seen in training)")
    for nm, XX in (("raw", X), ("per-task z", Z)):
        p = np.zeros(len(y))
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            p[i] = predict(fit_logistic(XX[m], y[m]), XX[i:i + 1])[0]
        evaluate(f"logistic, {nm}", p)

    print("\nLOTO  (leave-one-TASK-out; the question actually asked)")
    for nm, XX in (("raw", X), ("per-task z", Z)):
        p = np.zeros(len(y))
        for t in uniq:
            m = tasks != t
            p[~m] = predict(fit_logistic(XX[m], y[m]), XX[~m])
        evaluate(f"logistic, {nm}", p)
        for t in uniq:
            sub = p[tasks == t]
            print(f"        held out {t:<28} AUC={auc(sub, y[tasks==t]):>6.3f} "
                  f"(true SR {y[tasks==t].mean():.3f})")

    print("\n" + "=" * 74)
    print("PER-TASK THRESHOLD, fitted on that task's OWN rubric labels (the cheat)")
    accs = []
    for t in uniq:
        m = tasks == t
        a, c = best_acc(np.maximum(X[m, 1], X[m, 3]), y[m])
        accs.append(a)
        print(f"  {t:<34}best acc {a:.3f} @ {c:.3f}")
    print(f"  {'mean':<34}{np.mean(accs):.3f}")
    print("  ^ requires rubric labels for the task you are scoring. If you have those, you do not")
    print("    need a reward model. Reported only as the ceiling any per-task calibration could hit.")

    print("\nFitted weights (raw features, in-sample):")
    for f, wi in sorted(zip(FEATS, w[:-1]), key=lambda kv: -abs(kv[1])):
        print(f"    {f:<20}{wi:+.3f}")
    print(f"    {'(intercept)':<20}{w[-1]:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/mnt/home_lustre/sedlam56/projects/REALM/logs/rbm_replay.json")
