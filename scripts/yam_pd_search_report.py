"""Host-side report for scripts/yam_pd_search.py: merge shard grids, rank cells, plot the winner.

    uv run python scripts/yam_pd_search_report.py tmp/yam_pd_search/coarse_s* --data ~/abc_preview --out tmp/yam_pd_search/report

Writes <out>/grid_merged.csv (sorted by rmse_all), <out>/heatmap.png (rmse over kp x kd, unified cells only) and
<out>/overlay_<cell>.png per requested cell (recorded vs simulated joints for every episode, plus the reference
sets), and prints the ranking. No simulator needed.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from yam_pd_search import load_episodes  # noqa: E402  (same directory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="output dirs of yam_pd_search.py (one per shard)")
    ap.add_argument("--data", default=str(Path.home() / "abc_preview"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", nargs="*", default=None,
                    help="cells to overlay (default: the best unified cell + high_pd + base if present)")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in args.dirs]
    grid = pd.concat([pd.read_csv(d / "grid.csv").assign(shard=d.name) for d in dirs if (d / "grid.csv").exists()],
                     ignore_index=True)
    grid = grid.drop_duplicates(subset=["cell", "mode", "stepper"], keep="last").sort_values("rmse_all").reset_index(drop=True)
    grid.to_csv(out / "grid_merged.csv", index=False)
    cols = ["cell", "kp", "kd", "rmse_all"] + [f"rmse_j{j}" for j in range(1, 7)] + [f"tau_sim_j{j}_ms" for j in range(1, 7)]
    pd.set_option("display.width", 250)
    print(grid[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    real_tau = grid.iloc[0][[f"tau_real_j{j}_ms" for j in range(1, 7)]].to_numpy(dtype=float)
    print("real tau (ms):", " ".join(f"{v:.0f}" for v in real_tau))

    unified = grid[grid["kp"] != "group"].copy()
    unified["kp"] = unified["kp"].astype(float)
    unified["kd"] = unified["kd"].astype(float)
    if len(unified):
        piv = unified.pivot(index="kd", columns="kp", values="rmse_all")
        fig, ax = plt.subplots(figsize=(1.1 * len(piv.columns) + 2, 0.7 * len(piv.index) + 2))
        im = ax.imshow(np.log10(piv.to_numpy()), cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
        ax.set_xlabel("kp (all 12 arm joints)")
        ax.set_ylabel("kd")
        for i, kd in enumerate(piv.index):
            for j, kp in enumerate(piv.columns):
                v = piv.loc[kd, kp]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7,
                            color="w" if v > np.nanmedian(piv.to_numpy()) else "k")
        best = unified.iloc[0]
        ax.set_title(f"open-loop RMSE (rad) vs ABC real episodes -- best {best['cell']} = {best['rmse_all']:.4f}")
        fig.colorbar(im, ax=ax, label="log10 RMSE")
        fig.tight_layout()
        fig.savefig(out / "heatmap.png", dpi=130)
        print("wrote", out / "heatmap.png")

    cells = args.plot
    if cells is None:
        cells = ([unified.iloc[0]["cell"]] if len(unified) else []) + \
                [c for c in ("high_pd", "base") if (grid["cell"] == c).any()]
    episodes = load_episodes(args.data, max_steps=args.max_steps)
    data_arm_cols = [c for c in range(14) if c not in (6, 13)]
    for cell in cells:
        traj = None
        for d in dirs:
            p = d / f"traj_{cell}.npz"
            if p.exists():
                traj = np.load(p)
                break
        if traj is None:
            print(f"no trajectories for {cell}")
            continue
        fig, axes = plt.subplots(6, len(episodes), figsize=(4 * len(episodes), 13), sharex="col")
        for e, (name, s, a) in enumerate(episodes):
            sim = traj[name]
            t = np.arange(len(s)) / 30.0
            for j in range(6):
                ax = axes[j, e]
                for arm, off, ls in (("L", 0, "-"), ("R", 7, "--")):
                    ax.plot(t, s[:, off + j], "k", ls=ls, lw=0.8, label=f"real {arm}" if e == 0 and j == 0 else None)
                    ax.plot(t, a[:, off + j], "0.6", ls=ls, lw=0.5, label=f"cmd {arm}" if e == 0 and j == 0 else None)
                    ax.plot(t[1:], sim[:-1, (0 if arm == "L" else 6) + j], "C3", ls=ls, lw=0.8,
                            label=f"sim {arm}" if e == 0 and j == 0 else None)
                if e == 0:
                    ax.set_ylabel(f"j{j + 1} (rad)")
                if j == 0:
                    ax.set_title(name, fontsize=9)
            axes[-1, e].set_xlabel("s")
        axes[0, 0].legend(fontsize=7, ncol=3)
        fig.suptitle(f"{cell}: recorded (black) vs commanded (grey) vs REALM sim (red); solid L, dashed R")
        fig.tight_layout()
        fig.savefig(out / f"overlay_{cell}.png", dpi=110)
        print("wrote", out / f"overlay_{cell}.png")


if __name__ == "__main__":
    main()
