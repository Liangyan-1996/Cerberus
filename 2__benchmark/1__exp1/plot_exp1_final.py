#!/usr/bin/env python3
"""EXP1 S5：最终图（Mean 柱 + 5-Fold 散点 + SEM；Mean 线 + SEM 带）。

数据：summary/EXP1_cluster_metrics_all.csv + exp1/curves/EXP1_curves_split{s}.npz
输出：exp1/fig/EXP1_cluster_{auroc,auprc,f1vt}_bar.pdf
      exp1/fig/EXP1_cluster_{roc,pr}.pdf
样式：引用 benchmark_style.py；TrueType、PDF only、无 Panel Label。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2, width=0.6)


def bar_metric(df, metric, label, out_path, title, log=False):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        v = df[df.model == m][metric].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        mean, sem = v.mean(), (v.std(ddof=1) / np.sqrt(len(v))
                               if len(v) > 1 else 0.0)
        ax.bar(i, mean, width=0.58, color=COLORS[m], alpha=1.0,
               edgecolor="white", linewidth=0.3)
        rng = np.random.default_rng(0)
        jit = (rng.uniform(-0.10, 0.10, len(v)))
        ax.scatter(i + jit, v, s=11, color="#1a1a1a", zorder=4, linewidths=0)
        ax.errorbar(i, mean, yerr=sem, fmt="none", ecolor="#1a1a1a",
                    elinewidth=0.7, capsize=2.0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_official", "") if m != "Cerberus_union"
                        else "Cerberus" for m in MODELS], rotation=0, fontsize=6.5)
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=7, pad=4)
    if log:
        ax.set_yscale("log")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("[fig]", out_path)


def interp_roc_arrays(curves_by_split, grid):
    out = []
    for d in curves_by_split:
        fpr = np.asarray(d["fpr"])
        tpr = np.asarray(d["tpr"])
        if fpr[0] > 0:
            fpr = np.concatenate([[0.0], fpr])
            tpr = np.concatenate([[tpr[0]], tpr])
        out.append(np.interp(grid, fpr, tpr))
    return np.asarray(out)


def interp_pr_arrays(curves_by_split, grid):
    """标准 PR 插值：剔除无正例预测点 (recall=0, precision=0)，保留 (0,1)
    哨兵；相同 recall 取最大 precision（interpolated precision 约定）。"""
    out = []
    for d in curves_by_split:
        rec = np.asarray(d["recall"], dtype=float)
        prec = np.asarray(d["precision"], dtype=float)
        keep = ~((rec == 0) & (prec == 0))
        rec, prec = rec[keep], prec[keep]
        order = np.argsort(rec, kind="stable")
        rec, prec = rec[order], prec[order]
        u, idx = np.unique(rec, return_index=True)
        pu = np.maximum.reduceat(prec, idx)
        out.append(np.interp(grid, u, pu))
    return np.asarray(out)


def curve_fig(curves, kind, ax, title):
    grid = np.linspace(0, 1, 200)
    for m in MODELS:
        if m not in curves:
            continue
        arr = (interp_roc_arrays(curves[m], grid) if kind == "roc"
               else interp_pr_arrays(curves[m], grid))
        mean, sem = arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        lab = f"{m.replace('_official','') if m!='Cerberus_union' else 'Cerberus'}"
        ax.plot(grid, mean, color=COLORS[m], lw=1.0,
                label=lab, zorder=3)
        ax.fill_between(grid, mean - sem, mean + sem, color=COLORS[m],
                        alpha=0.18, linewidth=0)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], ls="--", color="#bbbbbb", lw=0.6, zorder=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.legend(fontsize=5.8, loc="lower right" if kind == "roc" else "upper right")
    ax.set_title(title, fontsize=7, pad=4)
    style_ax(ax)


def main():
    metrics = pd.read_csv(os.path.join(BASE, "summary",
                                       "EXP1_cluster_metrics_all.csv"))
    fig_dir = os.path.join(BASE, "exp1", "fig")
    os.makedirs(fig_dir, exist_ok=True)
    bar_metric(metrics, "auroc", "Cluster-level AUROC",
               os.path.join(fig_dir, "EXP1_cluster_auroc_bar.pdf"),
               "EXP1 · ROSE cluster-level AUROC (5-split)")
    bar_metric(metrics, "auprc", "Cluster-level AUPRC",
               os.path.join(fig_dir, "EXP1_cluster_auprc_bar.pdf"),
               "EXP1 · ROSE cluster-level AUPRC (5-split)")
    bar_metric(metrics, "test_f1", "Cluster-level F1 (val-tuned)",
               os.path.join(fig_dir, "EXP1_cluster_f1vt_bar.pdf"),
               "EXP1 · ROSE cluster-level F1 (val-tuned, 5-split)")

    curves = {m: [] for m in MODELS}
    for s in range(5):
        p = os.path.join(BASE, "exp1", "curves", f"EXP1_curves_split{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for m in MODELS:
            curves[m].append({k: z[f"{m}_{k}"] for k in
                              ("fpr", "tpr", "precision", "recall")})
    for kind, name in (("roc", "EXP1_cluster_roc.pdf"), ("pr", "EXP1_cluster_pr.pdf")):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        curve_fig(curves, kind, ax,
                  f"EXP1 · Cluster-level {kind.upper()} (mean ± SEM)")
        fig.tight_layout()
        out = os.path.join(fig_dir, name)
        fig.savefig(out)
        plt.close(fig)
        print("[fig]", out)
    print("[Done] EXP1 figures")


if __name__ == "__main__":
    main()
