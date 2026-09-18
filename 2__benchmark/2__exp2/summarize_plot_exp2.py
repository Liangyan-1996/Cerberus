#!/usr/bin/env python3
"""EXP2 S6d：汇总 + 最终图（Mean 柱 + 5 散点 + SEM；Mean 线 + SEM 带）。

输入：exp2/eval_split{s}/EXP2_*；assets/eval_split{s}/union_f1.csv（region F1）
输出：exp2/summary/EXP2_*_all.csv；exp2/fig/EXP2_*.pdf
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


def bar(df, metric, label, out, title, log=False):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # log 轴：先定有限基线/上限，柱底=ymin，避免 bar 路径延伸到画板外
    if log:
        allv = df[metric].to_numpy(float)
        allv = allv[np.isfinite(allv)]
        ymin = float(allv.min()) / 3.0
        ymax = float(allv.max()) * 1.5
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
    for i, m in enumerate(MODELS):
        v = df[df.source == m][metric].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        mean, sem = v.mean(), (v.std(ddof=1) / np.sqrt(len(v))
                               if len(v) > 1 else 0.0)
        if log:
            ax.bar(i, mean - ymin, bottom=ymin, width=0.58, color=COLORS[m],
                   alpha=1.0, edgecolor="white", linewidth=0.3)
        else:
            ax.bar(i, mean, width=0.58, color=COLORS[m], alpha=1.0,
                   edgecolor="white", linewidth=0.3)
        ax.scatter(i + np.random.default_rng(0).uniform(-0.1, 0.1, len(v)),
                   v, s=11, color="#1a1a1a", zorder=4, linewidths=0)
        ax.errorbar(i, mean, yerr=sem, fmt="none", ecolor="#1a1a1a",
                    elinewidth=0.7, capsize=2.0, zorder=5)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m.replace("_official", "") if m != "Cerberus_union"
                        else "Cerberus" for m in MODELS], fontsize=6.5)
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=7, pad=4)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("[fig]", out)


def interp_arrays(arrs, grid, kind):
    out = []
    for d in arrs:
        if kind == "roc":
            fpr, tpr = np.asarray(d["fpr"]), np.asarray(d["tpr"])
            if fpr[0] > 0:
                fpr = np.concatenate([[0], fpr])
                tpr = np.concatenate([[tpr[0]], tpr])
            out.append(np.interp(grid, fpr, tpr))
        else:
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


def curve_figs(split_dirs, kind, name, title):
    grid = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for m in MODELS:
        arrs = []
        for d in split_dirs:
            z = np.load(os.path.join(d, "EXP2_bin_curves.npz"))
            arrs.append({k: z[f"{m}_{k}"] for k in
                         (("fpr", "tpr") if kind == "roc"
                          else ("recall", "precision"))})
        a = interp_arrays(arrs, grid, kind)
        mean, sem = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
        lab = m.replace("_official", "") if m != "Cerberus_union" else "Cerberus"
        ax.plot(grid, mean, color=COLORS[m], lw=1.0, label=lab)
        ax.fill_between(grid, mean - sem, mean + sem, color=COLORS[m],
                        alpha=0.18, linewidth=0)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], ls="--", color="#bbbbbb", lw=0.6)
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
    fig.tight_layout()
    out = os.path.join(BASE, "exp2", "fig", name)
    fig.savefig(out)
    plt.close(fig)
    print("[fig]", out)


def region_pr_curves():
    grid = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for m in MODELS:
        arrs = []
        for s in range(5):
            z = np.load(os.path.join(BASE, "exp2", f"eval_split{s}",
                                     "EXP2_prcmax_region_pr_curves.npz"))
            arrs.append({"recall": z[f"{m}_recall"],
                         "precision": z[f"{m}_precision"]})
        a = interp_arrays(arrs, grid, "pr")
        mean, sem = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
        lab = m.replace("_official", "") if m != "Cerberus_union" else "Cerberus"
        ax.plot(grid, mean, color=COLORS[m], lw=1.0, label=lab)
        ax.fill_between(grid, mean - sem, mean + sem, color=COLORS[m],
                        alpha=0.18, linewidth=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=5.8, loc="upper right")
    ax.set_title("EXP2-prcmax · Region PR (mean ± SEM)", fontsize=7, pad=4)
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(BASE, "exp2", "fig", "EXP2_prcmax_region_pr.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("[fig]", out)


def main():
    eval_dirs = [os.path.join(BASE, "exp2", f"eval_split{s}") for s in range(5)]
    for d in eval_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"缺 {d}（先跑 S6b/S6c）")
    os.makedirs(os.path.join(BASE, "exp2", "summary"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "exp2", "fig"), exist_ok=True)

    mbin = pd.concat([pd.read_csv(os.path.join(d, "EXP2_metrics_bin.csv"))
                      for d in eval_dirs], ignore_index=True)
    dfp = pd.concat([pd.read_csv(os.path.join(d, "EXP2_desert_fp_at_recall.csv"))
                     for d in eval_dirs], ignore_index=True)
    mreg = pd.concat([pd.read_csv(os.path.join(d, "EXP2_prcmax_region_metrics.csv"))
                      for d in eval_dirs], ignore_index=True)
    iou = pd.concat([pd.read_csv(os.path.join(d, "EXP2_region_iou_recall08.csv"))
                     for d in eval_dirs], ignore_index=True)
    f1s = pd.concat([pd.read_csv(os.path.join(
        BASE, f"assets/eval_split{s}/eval/union_f1.csv")).assign(split=s)
        for s in range(5)], ignore_index=True)
    f1s = f1s.rename(columns={"model": "source"})
    mbin.to_csv(os.path.join(BASE, "exp2", "summary", "EXP2_metrics_bin_all.csv"),
                index=False)
    dfp.to_csv(os.path.join(BASE, "exp2", "summary",
                            "EXP2_desert_fp_all.csv"), index=False)
    mreg.to_csv(os.path.join(BASE, "exp2", "summary",
                             "EXP2_prcmax_region_metrics_all.csv"), index=False)
    iou.to_csv(os.path.join(BASE, "exp2", "summary",
                            "EXP2_region_iou_all.csv"), index=False)
    f1s.to_csv(os.path.join(BASE, "exp2", "summary", "EXP2_union_f1_all.csv"),
               index=False)
    fig_dir = os.path.join(BASE, "exp2", "fig")
    bar(mbin, "auroc", "Bin-level AUROC",
        os.path.join(fig_dir, "EXP2_bin_auroc_bar.pdf"),
        "EXP2 · Bin-level AUROC (5-split)")
    bar(mbin, "auprc", "Bin-level AUPRC",
        os.path.join(fig_dir, "EXP2_bin_auprc_bar.pdf"),
        "EXP2 · Bin-level AUPRC (5-split)")
    bar(dfp[dfp.se_recall == 0.8], "fp_per_mb", "Desert FP / Mb @recall=0.8",
        os.path.join(fig_dir, "EXP2_desert_fp_recall08_bar.pdf"),
        "EXP2 · Desert FP/Mb @ SE recall 0.8 (5-split)", log=True)
    bar(mreg, "auprc", "Region-level AUPRC",
        os.path.join(fig_dir, "EXP2_prcmax_region_auprc_bar.pdf"),
        "EXP2-prcmax · Union region AUPRC (5-split)")
    bar(iou, "median_iou_tp", "TP median IoU @recall=0.8",
        os.path.join(fig_dir, "EXP2_region_iou_median_bar.pdf"),
        "EXP2 · TP median IoU @ SE recall 0.8 (5-split)")
    bar(f1s, "f1", "Union region F1 (val-tuned)",
        os.path.join(fig_dir, "EXP2_region_f1_bar.pdf"),
        "EXP2 · Union region F1 (val-tuned, 5-split)")
    curve_figs(eval_dirs, "roc", "EXP2_bin_roc.pdf",
               "EXP2 · Bin-level ROC (mean ± SEM)")
    curve_figs(eval_dirs, "pr", "EXP2_bin_pr.pdf",
               "EXP2 · Bin-level PR (mean ± SEM)")
    region_pr_curves()
    print("[Done] EXP2 summary + figures")


if __name__ == "__main__":
    main()
