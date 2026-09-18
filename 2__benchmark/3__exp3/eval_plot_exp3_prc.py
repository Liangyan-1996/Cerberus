#!/usr/bin/env python3
"""EXP3 peak-restricted cluster 指标/图：
簇分数 = 簇内构成峰中心 3 kb 窗（基线）或构成峰 bins 通道均值（Cerberus）的平均。
输出 EXP3_prc_{auprc,auroc,test_f1}_bar.pdf、EXP3_prc_cross_heatmap_<model>.pdf、
EXP3_prc_cluster_metrics_all.csv、EXP3_prc_significance_tests.csv。
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from final_style import AUPRC_VMAX, COLORS  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["legend.frameon"] = False

CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
MODELS = ["Cerberus", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]
DISPLAY = {"Cerberus": "Cerberus", "DeepSE_official": "DeepSE",
           "SENet": "SENet", "MuSE_official": "MuSE", "DcSE": "DcSE"}
THR = np.arange(0.01, 0.995, 0.005)


def f1_at(y, p, t):
    pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def path(split, cell, name):
    return os.path.join(BASE, "exp3", f"pool_split{split}", cell, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="EXP3_prc")
    ap.add_argument("--title", default="EXP3-prc")
    args = ap.parse_args()
    PREFIX, TITLE = args.prefix, args.title
    os.makedirs(os.path.join(BASE, "exp3", "summary"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "exp3", "fig"), exist_ok=True)
    rows = []
    labels = {}
    for split in range(5):
        for cell in CELLS:
            d = pd.read_csv(path(split, cell, "EXP3_pool_test.csv"))
            labels[(split, cell)] = (d.kind == "SE").to_numpy().astype(int)
            dv = pd.read_csv(path(split, cell, "EXP3_pool_val.csv"))
            yv = (dv.kind == "SE").to_numpy().astype(int)
            for m in MODELS:
                pv = np.load(path(split, cell, f"{PREFIX}_cerberus_val.npy")
                             if m == "Cerberus" else
                             path(split, cell, f"{PREFIX}_baseline_{m}_val.npy"))
                pt = np.load(path(split, cell, f"{PREFIX}_cerberus_test.npy")
                             if m == "Cerberus" else
                             path(split, cell, f"{PREFIX}_baseline_{m}_test.npy"))
                okv, okt = np.isfinite(pv), np.isfinite(pt)
                best_t, best_f1 = 0.5, 0.0
                for t in THR:
                    f1v = f1_at(yv[okv], pv[okv], t)
                    if f1v > best_f1:
                        best_f1, best_t = f1v, t
                rows.append({"cell": cell, "split": split, "model": m,
                             "n_test": int(okt.sum()),
                             "n_pos_test": int(labels[(split, cell)][okt].sum()),
                             "auroc": round(float(roc_auc_score(
                                 labels[(split, cell)][okt], pt[okt])), 4),
                             "auprc": round(float(average_precision_score(
                                 labels[(split, cell)][okt], pt[okt])), 4),
                             "val_thr": round(float(best_t), 4),
                             "test_f1": round(float(f1_at(
                                 labels[(split, cell)][okt], pt[okt], best_t)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "exp3", "summary",
                           f"{PREFIX}_cluster_metrics_all.csv"), index=False)
    sig = []
    for cell in CELLS:
        for metric in ("auroc", "auprc", "test_f1"):
            a = df[(df.cell == cell) & (df.model == "Cerberus")][metric].to_numpy(float)
            for m in MODELS[1:]:
                b = df[(df.cell == cell) & (df.model == m)][metric].to_numpy(float)
                t, p = stats.ttest_rel(a, b)
                sig.append({"cell": cell, "metric": metric, "model_b": m,
                            "mean_Cerberus": round(float(a.mean()), 4),
                            "mean_base": round(float(b.mean()), 4),
                            "t": round(float(t), 3), "p": round(float(p), 5)})
    pd.DataFrame(sig).to_csv(os.path.join(
        BASE, "exp3", "summary", f"{PREFIX}_significance_tests.csv"), index=False)
    # 分组柱
    fig_dir = os.path.join(BASE, "exp3", "fig")
    for metric, lab in (("auprc", "AUPRC"), ("auroc", "AUROC"),
                        ("test_f1", "F1 (val-tuned)")):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 78 / 25.4))
        x = np.arange(len(CELLS)); w = 0.16
        rng = np.random.default_rng(0)
        for i, m in enumerate(MODELS):
            vals = np.asarray([df[(df.cell == c) & (df.model == m)][metric]
                               for c in CELLS], dtype=float)
            mean = vals.mean(1); sem = vals.std(1, ddof=1) / np.sqrt(5)
            ax.bar(x + (i - 2) * w, mean, w, color=COLORS[m], label=DISPLAY[m])
            for ci in range(len(CELLS)):
                ax.scatter(np.full(5, x[ci] + (i - 2) * w)
                           + rng.uniform(-0.02, 0.02, 5), vals[ci],
                           s=4, color="#1a1a1a", zorder=5, linewidths=0)
            ax.errorbar(x + (i - 2) * w, mean, yerr=sem, fmt="none",
                        ecolor="#1a1a1a", elinewidth=0.5, capsize=1.2)
        ax.set_xticks(x); ax.set_xticklabels(CELLS)
        ax.set_ylabel(lab)
        ax.set_title(f"{TITLE} · Lineage-specific {lab} (5-split)",
                     fontsize=7, pad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=5.5, ncol=3)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"{PREFIX}_{metric}_bar.pdf")
        fig.savefig(out); plt.close(fig)
        print("[fig]", out)
    # 4×4 交叉热图（seaborn、统一色标、无对角框）
    auprc = {}
    for m in MODELS:
        M = np.full((4, 4, 5), np.nan)
        for i, pred in enumerate(CELLS):
            for j, pool in enumerate(CELLS):
                for split in range(5):
                    f = (path(split, pool, f"{PREFIX}_cerberus_x_{pred}_test.npy")
                         if m == "Cerberus" else
                         path(split, pool, f"{PREFIX}_baseline_x_{m}_{pred}"
                                           f"_on_{pool}_test.npy"))
                    if not os.path.exists(f):
                        continue
                    sc = np.load(f); y = labels[(split, pool)]; ok = np.isfinite(sc)
                    if ok.sum() and 0 < y[ok].sum() < ok.sum():
                        M[i, j, split] = average_precision_score(y[ok], sc[ok])
        auprc[m] = M
    vmax = AUPRC_VMAX
    for m in MODELS:
        mean = np.nanmean(auprc[m], axis=2)
        sem = np.nanstd(auprc[m], axis=2, ddof=1) / np.sqrt(5)
        fig, ax = plt.subplots(figsize=(92 / 25.4, 76 / 25.4))
        sns.heatmap(mean, ax=ax, cmap="viridis", vmin=0, vmax=vmax,
                    square=True, linewidths=0.8, linecolor="white",
                    xticklabels=CELLS, yticklabels=CELLS, cbar=True,
                    cbar_kws={"label": "AUPRC (5-split mean)",
                              "ticks": np.arange(0, vmax + 1e-9, 0.1)})
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        ax.set_xlabel("GT lineage (cluster pool) / peak-restricted score")
        ax.set_ylabel("Pred lineage (model / channel)")
        ax.collections[0].colorbar.ax.tick_params(labelsize=6)
        ax.collections[0].colorbar.set_label("AUPRC (5-split mean)", fontsize=6)
        for i in range(4):
            for j in range(4):
                v = mean[i, j]
                txt = "NA" if not np.isfinite(v) else f"{v:.3f}"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=5.2, color="white")
        ax.set_title(f"{TITLE} · {DISPLAY[m]} · cross-lineage AUPRC",
                     fontsize=7, pad=5)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"{PREFIX}_cross_heatmap_{m}.pdf")
        fig.savefig(out); plt.close(fig)
        print("[fig]", out)
        print(f"  mean matrix ({m}):\n{pd.DataFrame(mean.round(3), index=CELLS, columns=CELLS).to_string()}")
    print("[Done] EXP3 peak-restricted cluster metrics + figures")


if __name__ == "__main__":
    main()
