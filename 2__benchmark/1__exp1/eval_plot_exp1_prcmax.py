#!/usr/bin/env python3
"""EXP1 peak-restricted+max：指标（AUROC/AUPRC/F1_vt）+ 显著性 + bar/ROC/PR。
输出 exp1/fig/EXP1_prcmax_*.pdf 与 summary/EXP1_prcmax_*.csv。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

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
THR = np.arange(0.01, 0.995, 0.005)


def f1_at(y, p, t):
    pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def path(split, name):
    return os.path.join(BASE, "exp1", f"split{split}", name)


def interp_pr(rec, prec, grid):
    keep = ~((rec == 0) & (prec == 0))
    rec, prec = rec[keep], prec[keep]
    o = np.argsort(rec, kind="stable")
    rec, prec = rec[o], prec[o]
    u, i = np.unique(rec, return_index=True)
    return np.interp(grid, u, np.maximum.reduceat(prec, i))


def main():
    os.makedirs(os.path.join(BASE, "exp1", "fig"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "summary"), exist_ok=True)
    rows = []
    curves = {m: [] for m in MODELS}
    grid = np.linspace(0, 1, 200)
    for split in range(5):
        for ttag in ("val", "test"):
            pass
        val = pd.read_csv(path(split, "EXP1_pool_val.csv"))
        test = pd.read_csv(path(split, "EXP1_pool_test.csv"))
        yv = (val["kind"] == "SE").to_numpy().astype(int)
        yt = (test["kind"] == "SE").to_numpy().astype(int)
        for m in MODELS:
            pv = np.load(path(split, f"EXP1_prcmax_cerberus_val.npy"
                           if m == "Cerberus_union" else
                           f"EXP1_prcmax_baseline_{m}_val.npy"))
            pt = np.load(path(split, f"EXP1_prcmax_cerberus_test.npy"
                           if m == "Cerberus_union" else
                           f"EXP1_prcmax_baseline_{m}_test.npy"))
            okv, okt = np.isfinite(pv), np.isfinite(pt)
            best_t, best_f1 = 0.5, 0.0
            for t in THR:
                f = f1_at(yv[okv], pv[okv], t)
                if f > best_f1:
                    best_f1, best_t = f, t
            rows.append({"split": split, "model": m, "n_test": int(okt.sum()),
                         "n_pos_test": int(yt[okt].sum()),
                         "auroc": round(float(roc_auc_score(yt[okt], pt[okt])), 4),
                         "auprc": round(float(average_precision_score(
                             yt[okt], pt[okt])), 4),
                         "val_thr": round(float(best_t), 4),
                         "test_f1": round(float(f1_at(yt[okt], pt[okt], best_t)), 4)})
            if split < 5:
                fpr, tpr, _ = roc_curve(yt[okt], pt[okt])
                prec, rec, _ = precision_recall_curve(yt[okt], pt[okt])
                curves[m].append({
                    "roc": np.interp(grid, fpr, tpr),
                    "pr": interp_pr(rec, prec, grid)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "summary",
                           "EXP1_prcmax_cluster_metrics_all.csv"), index=False)
    sig = []
    for metric in ("auroc", "auprc", "test_f1"):
        a = df[df.model == "Cerberus_union"][metric].to_numpy(float)
        for m in MODELS[1:]:
            b = df[df.model == m][metric].to_numpy(float)
            t, p = stats.ttest_rel(a, b)
            sig.append({"metric": metric, "model_b": m,
                        "mean_a": round(float(a.mean()), 4),
                        "mean_b": round(float(b.mean()), 4),
                        "t": round(float(t), 3), "df": 4, "p": round(float(p), 5)})
    pd.DataFrame(sig).to_csv(os.path.join(
        BASE, "summary", "EXP1_prcmax_significance_tests.csv"), index=False)
    fig_dir = os.path.join(BASE, "exp1", "fig")
    # bar
    for metric, lab, name in (("auprc", "Cluster-level AUPRC", "EXP1_prcmax_auprc_bar.pdf"),
                              ("auroc", "Cluster-level AUROC", "EXP1_prcmax_auroc_bar.pdf"),
                              ("test_f1", "Cluster-level F1 (val-tuned)",
                               "EXP1_prcmax_f1vt_bar.pdf")):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for i, m in enumerate(MODELS):
            v = df[df.model == m][metric].to_numpy(float)
            mean, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(i, mean, width=0.58, color=COLORS[m], edgecolor="white",
                   linewidth=0.3)
            ax.scatter(i + np.random.default_rng(0).uniform(-0.1, 0.1, len(v)),
                       v, s=11, color="#1a1a1a", zorder=4, linewidths=0)
            ax.errorbar(i, mean, yerr=sem, fmt="none", ecolor="#1a1a1a",
                        elinewidth=0.7, capsize=2.0, zorder=5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m.replace("_official", "") if m != "Cerberus_union"
                            else "Cerberus" for m in MODELS], fontsize=6.5)
        ax.set_ylabel(lab)
        ax.set_title(f"EXP1 · peak-restricted max {lab.split(' ', 1)[-1]}",
                     fontsize=7, pad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name))
        plt.close(fig)
        print("[fig]", name)
    # curves
    for kind, name in (("roc", "EXP1_prcmax_roc.pdf"), ("pr", "EXP1_prcmax_pr.pdf")):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for m in MODELS:
            a = np.asarray([c[kind] for c in curves[m]])
            mean = a.mean(0); sem = a.std(0, ddof=1) / np.sqrt(a.shape[0])
            lab = m.replace("_official", "") if m != "Cerberus_union" else "Cerberus"
            ax.plot(grid, mean, color=COLORS[m], lw=1.0, label=lab)
            ax.fill_between(grid, mean - sem, mean + sem, color=COLORS[m],
                            alpha=0.18, linewidth=0)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="#bbbbbb", lw=0.6)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        else:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=5.8, loc="lower right" if kind == "roc" else "upper right")
        ax.set_title(f"EXP1-prcmax · {kind.upper()} (mean ± SEM)", fontsize=7, pad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name))
        plt.close(fig)
        print("[fig]", name)
    print("[Done] EXP1 peak-restricted max metrics + figures")


if __name__ == "__main__":
    main()
