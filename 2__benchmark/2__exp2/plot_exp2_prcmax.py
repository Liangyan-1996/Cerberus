#!/usr/bin/env python3
"""EXP2 region peak-restricted max：AUPRC 柱 + PR 曲线（mean±SEM，5 split）。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from final_style import COLORS, MODELS  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["legend.frameon"] = False
FIG_W, FIG_H = 88 / 25.4, 78 / 25.4


def interp_pr(rec, prec, grid):
    keep = ~((rec == 0) & (prec == 0))
    rec, prec = rec[keep], prec[keep]
    o = np.argsort(rec, kind="stable")
    rec, prec = rec[o], prec[o]
    u, i = np.unique(rec, return_index=True)
    return np.interp(grid, u, np.maximum.reduceat(prec, i))


def main():
    dfs = []
    curves = {m: [] for m in MODELS}
    grid = np.linspace(0, 1, 200)
    for s in range(5):
        d = os.path.join(BASE, "exp2", f"eval_split{s}")
        dfs.append(pd.read_csv(os.path.join(d, "EXP2_prcmax_region_metrics.csv")))
        z = np.load(os.path.join(d, "EXP2_prcmax_region_pr_curves.npz"))
        for m in MODELS:
            curves[m].append(interp_pr(z[f"{m}_recall"], z[f"{m}_precision"], grid))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(BASE, "exp2", "summary",
                           "EXP2_prcmax_region_metrics_all.csv"), index=False)
    fig_dir = os.path.join(BASE, "exp2", "fig")
    os.makedirs(fig_dir, exist_ok=True)
    # AUPRC bar
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, m in enumerate(MODELS):
        v = df[df.source == m]["auprc"].to_numpy(float)
        mean, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.bar(i, mean, width=0.58, color=COLORS[m], edgecolor="white", linewidth=0.3)
        ax.scatter(i + np.random.default_rng(0).uniform(-0.1, 0.1, len(v)), v,
                   s=11, color="#1a1a1a", zorder=4, linewidths=0)
        ax.errorbar(i, mean, yerr=sem, fmt="none", ecolor="#1a1a1a",
                    elinewidth=0.7, capsize=2.0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m.replace("_official", "") if m != "Cerberus_union"
                        else "Cerberus" for m in MODELS], fontsize=6.5)
    ax.set_ylabel("Region AUPRC (peak-restricted max)")
    ax.set_title("EXP2-prcmax · Union region AUPRC (5-split)", fontsize=7, pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "EXP2_prcmax_region_auprc_bar.pdf"))
    plt.close(fig)
    print("[fig] EXP2_prcmax_region_auprc_bar.pdf")
    # PR curves
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for m in MODELS:
        a = np.asarray(curves[m]); mean = a.mean(0); sem = a.std(0, ddof=1) / np.sqrt(5)
        lab = m.replace("_official", "") if m != "Cerberus_union" else "Cerberus"
        ax.plot(grid, mean, color=COLORS[m], lw=1.0, label=lab)
        ax.fill_between(grid, mean - sem, mean + sem, color=COLORS[m], alpha=0.18,
                        linewidth=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=5.8, loc="upper right")
    ax.set_title("EXP2-prcmax · Region PR (mean ± SEM)", fontsize=7, pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "EXP2_prcmax_region_pr.pdf"))
    plt.close(fig)
    print("[fig] EXP2_prcmax_region_pr.pdf")
    # significance
    sig = []
    a = df[df.source == "Cerberus_union"]["auprc"].to_numpy(float)
    for m in MODELS[1:]:
        b = df[df.source == m]["auprc"].to_numpy(float)
        t, p = stats.ttest_rel(a, b)
        sig.append({"metric": "region_auprc", "model_b": m,
                    "mean_cerberus": round(float(a.mean()), 4),
                    "mean_base": round(float(b.mean()), 4),
                    "t": round(float(t), 3), "p": round(float(p), 5)})
    pd.DataFrame(sig).to_csv(os.path.join(
        BASE, "exp2", "summary", "EXP2_prcmax_significance_tests.csv"), index=False)
    print("[Done] EXP2 prcmax figures")


if __name__ == "__main__":
    main()
