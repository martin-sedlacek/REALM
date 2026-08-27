"""Statistical findings for REALM experiment data."""

import numpy as np
import pandas as pd
from scipy import stats


def _clean_perturbation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pert_clean"] = df["perturbation"].apply(
        lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x)
    )
    return df


def _fmt_task(t: str) -> str:
    return t.replace("_", " ")


def _fmt_pert(p: str) -> str:
    return p


def _bh_adjusted_pvalues(p_values: list) -> list:
    n = len(p_values)
    if n == 0:
        return []
    arr = np.array(p_values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, arr * n / ranks)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj.tolist()


def _bayesian_sr_p_equiv(
    s_cell: int, n_cell: int, s_rest: int, n_rest: int, n_samples: int = 8000
) -> float:
    """Compare Beta posteriors with a fixed seed for stable dashboard results."""
    rng = np.random.default_rng(42)
    theta_cell = rng.beta(1 + s_cell, 1 + (n_cell - s_cell), n_samples)
    theta_rest = rng.beta(1 + s_rest, 1 + (n_rest - s_rest), n_samples)
    p_cell_greater = float(np.mean(theta_cell > theta_rest))
    evidence = max(p_cell_greater, 1.0 - p_cell_greater)
    return 2.0 * (1.0 - evidence)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _sf(p: float) -> float:
    return min(1.0, -np.log10(max(p, 1e-12)) / 4.0)


def _sw(n: int, min_cell: int) -> float:
    return 0.6 + 0.4 * min(1.0, (n - min_cell) / 15.0)


def _stage_deviation_findings(df: pd.DataFrame, min_cell_size: int) -> list:
    findings = []

    df = df.copy()
    df["_stage"] = df["stage"].astype(str).str.upper()
    tasks = df["task"].unique()

    candidate_p = []
    candidate_data = []

    for task in tasks:
        task_df = df[df["task"] == task]
        task_stages = task_df["_stage"].value_counts()
        task_total = len(task_df)

        for pert in task_df["pert_clean"].unique():
            cell_df = task_df[task_df["pert_clean"] == pert]
            n_cell = len(cell_df)
            if n_cell < min_cell_size:
                continue

            cell_stages = cell_df["_stage"].value_counts()
            all_stages = task_stages.index.union(cell_stages.index)

            observed = np.array([cell_stages.get(s, 0) for s in all_stages])
            task_counts = np.array([task_stages.get(s, 0) for s in all_stages])
            task_props = task_counts / task_total
            expected = task_props * n_cell

            valid = expected > 0
            if valid.sum() < 2:
                continue

            try:
                _, p, _, _ = stats.chi2_contingency(
                    np.vstack([observed[valid], (task_counts - observed)[valid]])
                )
            except Exception:
                continue

            cell_pcts = observed / n_cell
            deltas = cell_pcts - task_props
            max_idx = np.argmax(np.abs(deltas))
            max_delta = deltas[max_idx]

            if abs(max_delta) < 0.10:  # pre-filter tiny effects
                continue

            candidate_p.append(p)
            candidate_data.append({
                "task": task,
                "pert": pert,
                "stage": list(all_stages)[max_idx],
                "delta_pp": max_delta,
                "cell_pct": cell_pcts[max_idx],
                "task_pct": task_props[max_idx],
                "p_raw": p,
                "n_cell": n_cell,
            })

    if not candidate_data:
        return findings

    adj_p_values = _bh_adjusted_pvalues(candidate_p)

    for d, p_adj in zip(candidate_data, adj_p_values):
        if p_adj >= 0.15:
            continue

        direction = "spike" if d["delta_pp"] > 0 else "drop"
        stage_label = d["stage"].replace("_", " ").title()
        p_str = f"p={d['p_raw']:.3f}" if d["p_raw"] >= 0.001 else "p<0.001"

        headline = (
            f"**{_fmt_pert(d['pert'])}** on *{_fmt_task(d['task'])}* "
            f"→ {stage_label} failures {direction} to "
            f"**{d['cell_pct']:.0%}** (task avg {d['task_pct']:.0%}, "
            f"Δ{d['delta_pp']:+.0%}, {p_str})"
        )

        ES = abs(d["delta_pp"])
        SF = _sf(p_adj)
        SW = _sw(d["n_cell"], min_cell_size)
        merit = ES * SF * SW * 1.0  # CW = 1.0

        findings.append({
            "headline": headline,
            "merit_score": merit,
            "category": "stage_deviation",
        })

    return findings


def _beta_posterior_mean(successes: int, n: int) -> float:
    return (1 + successes) / (2 + n)


def _success_rate_findings(df: pd.DataFrame, min_cell_size: int) -> list:
    findings = []
    THRESHOLD = 0.15

    tasks = df["task"].unique()
    perts = df["pert_clean"].unique()

    def _task_posterior_mean(task):
        s = int(df[df["task"] == task]["binary_SR"].sum())
        n = len(df[df["task"] == task])
        return _beta_posterior_mean(s, n)

    def _pert_posterior_mean(pert):
        s = int(df[df["pert_clean"] == pert]["binary_SR"].sum())
        n = len(df[df["pert_clean"] == pert])
        return _beta_posterior_mean(s, n)

    task_mean_sr = {t: _task_posterior_mean(t) for t in tasks}
    pert_mean_sr = {p: _pert_posterior_mean(p) for p in perts}

    for task in tasks:
        task_df = df[df["task"] == task]
        for pert in perts:
            cell = task_df[task_df["pert_clean"] == pert]
            n = len(cell)
            if n < min_cell_size:
                continue

            cell_succ = int(cell["binary_SR"].sum())
            sr_bayes = _beta_posterior_mean(cell_succ, n)

            delta_task = sr_bayes - task_mean_sr[task]
            delta_pert = sr_bayes - pert_mean_sr[pert]

            if abs(delta_task) < THRESHOLD or abs(delta_pert) < THRESHOLD:
                continue
            if np.sign(delta_task) != np.sign(delta_pert):
                continue

            rest = task_df[task_df["pert_clean"] != pert]
            n_rest = len(rest)
            if n_rest == 0:
                continue
            rest_succ = int(rest["binary_SR"].sum())
            p_bayes = _bayesian_sr_p_equiv(cell_succ, n, rest_succ, n_rest)

            h = 2 * np.arcsin(np.sqrt(sr_bayes)) - 2 * np.arcsin(np.sqrt(task_mean_sr[task]))
            ES = min(1.0, abs(h) / np.pi)

            direction = "above" if delta_task > 0 else "below"
            lo, hi = _wilson_ci(cell_succ, n)
            ci_str = f"[{lo:.0%}–{hi:.0%}]"
            sr_mle = cell["binary_SR"].mean()

            headline = (
                f"**{_fmt_pert(pert)}** × *{_fmt_task(task)}*: "
                f"**{sr_mle:.0%}** SR {ci_str} — "
                f"{abs(delta_task):.0%} {direction} task avg ({task_mean_sr[task]:.0%}) "
                f"and {abs(delta_pert):.0%} {direction} {pert} avg ({pert_mean_sr[pert]:.0%})"
            )

            SF = _sf(p_bayes)
            SW = _sw(n, min_cell_size)
            merit = ES * SF * SW * 0.9  # CW = 0.9

            findings.append({
                "headline": headline,
                "merit_score": merit,
                "category": "success_rate",
            })

    return findings


def _metric_anomaly_findings(df: pd.DataFrame, min_cell_size: int) -> list:
    findings = []
    METRICS = {
        "collisions_env": "env collisions",
        "object_drops": "object drops",
        "joint_vel_var": "joint velocity variance",
        "cart_jerk": "Cartesian jerk",
    }
    SIGMA_THRESH = 1.5

    failures = df[df["binary_SR"] == 0].copy()
    if failures.empty:
        return findings

    global_stats = {}
    for col in METRICS:
        if col not in failures.columns:
            continue
        global_stats[col] = (failures[col].mean(), failures[col].std())

    for task in failures["task"].unique():
        for pert in failures["pert_clean"].unique():
            cell = failures[(failures["task"] == task) & (failures["pert_clean"] == pert)]
            n_cell = len(cell)
            if n_cell < min_cell_size:
                continue

            for col, label in METRICS.items():
                if col not in global_stats:
                    continue
                g_mean, g_std = global_stats[col]
                if g_std == 0:
                    continue

                cell_mean = cell[col].mean()
                sigma = (cell_mean - g_mean) / g_std
                if sigma < SIGMA_THRESH:
                    continue

                p_normal = float(stats.norm.sf(sigma))

                headline = (
                    f"**{_fmt_pert(pert)}** × *{_fmt_task(task)}* (failures): "
                    f"**{cell_mean:.1f}** {label} — "
                    f"{sigma:.1f}σ above global failure avg ({g_mean:.1f})"
                )

                ES = min(1.0, sigma / 4.0)
                SF = _sf(p_normal)
                SW = _sw(n_cell, min_cell_size)
                merit = ES * SF * SW * 0.7  # CW = 0.7

                findings.append({
                    "headline": headline,
                    "merit_score": merit,
                    "category": "metric_anomaly",
                })

    return findings


def _ranking_findings(df: pd.DataFrame) -> list:
    findings = []

    cell_sr = df.groupby(["task", "pert_clean"])["binary_SR"].agg(["mean", "count"])
    cell_sr = cell_sr[cell_sr["count"] >= 5]
    if cell_sr.empty:
        return findings

    best = cell_sr["mean"].idxmax()
    worst = cell_sr["mean"].idxmin()
    findings.append({
        "headline": (
            f"Best combo: **{_fmt_pert(best[1])}** × *{_fmt_task(best[0])}* "
            f"({cell_sr.loc[best, 'mean']:.0%} SR)"
        ),
        "merit_score": 0.10,
        "category": "ranking",
    })
    findings.append({
        "headline": (
            f"Worst combo: **{_fmt_pert(worst[1])}** × *{_fmt_task(worst[0])}* "
            f"({cell_sr.loc[worst, 'mean']:.0%} SR)"
        ),
        "merit_score": 0.09,
        "category": "ranking",
    })

    pert_sr = cell_sr.groupby("pert_clean")["mean"]
    pert_std = pert_sr.std().dropna()
    pert_avg = pert_sr.mean()
    if not pert_std.empty:
        most_consistent = pert_std.idxmin()
        least_consistent = pert_std.idxmax()
        findings.append({
            "headline": (
                f"Most consistent perturbation: **{_fmt_pert(most_consistent)}** "
                f"(avg {pert_avg[most_consistent]:.0%} SR, "
                f"±{pert_std[most_consistent]:.0%} std across tasks)"
            ),
            "merit_score": 0.08,
            "category": "ranking",
        })
        findings.append({
            "headline": (
                f"Most variable perturbation: **{_fmt_pert(least_consistent)}** "
                f"(avg {pert_avg[least_consistent]:.0%} SR, "
                f"±{pert_std[least_consistent]:.0%} std across tasks)"
            ),
            "merit_score": 0.07,
            "category": "ranking",
        })

    task_sr_range = cell_sr.groupby("task")["mean"].apply(lambda x: x.max() - x.min())
    if not task_sr_range.empty:
        most_sensitive = task_sr_range.idxmax()
        findings.append({
            "headline": (
                f"Perturbation matters most for *{_fmt_task(most_sensitive)}* "
                f"(SR range {task_sr_range[most_sensitive]:.0%} across perturbations)"
            ),
            "merit_score": 0.06,
            "category": "ranking",
        })

    task_mean_sr = df.groupby("task")["binary_SR"].mean()
    hardest = task_mean_sr.idxmin()
    easiest = task_mean_sr.idxmax()
    findings.append({
        "headline": (
            f"Hardest task: *{_fmt_task(hardest)}* "
            f"({task_mean_sr[hardest]:.0%} mean SR across all perturbations)"
        ),
        "merit_score": 0.05,
        "category": "ranking",
    })
    findings.append({
        "headline": (
            f"Easiest task: *{_fmt_task(easiest)}* "
            f"({task_mean_sr[easiest]:.0%} mean SR across all perturbations)"
        ),
        "merit_score": 0.04,
        "category": "ranking",
    })

    return findings


def run_all_analytics(df: pd.DataFrame, min_cell_size: int = 10) -> list:
    if df is None or df.empty:
        return []

    required = {"task", "perturbation", "binary_SR", "stage"}
    if not required.issubset(df.columns):
        return []

    df = _clean_perturbation(df)

    findings = []
    findings += _stage_deviation_findings(df, min_cell_size)
    findings += _success_rate_findings(df, min_cell_size)
    findings += _metric_anomaly_findings(df, min_cell_size)
    findings += _ranking_findings(df)

    findings.sort(key=lambda x: x["merit_score"], reverse=True)
    return findings
