#!/usr/bin/env python3
"""EXP3 S7c3：簇级指标 + 配对 t 检验 + 分组柱状图（AUPRC 主）。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from final_style import COLORS  # noqa: E402

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
THR = np.arange(0.01, 0.995, 0.005)


def f1_at(y, p, t):
    pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum())
    fp = int(((pr == 1) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def load_scores(split, cell, tag, vt):
    prefix = "cerberus" if tag == "Cerberus" else f"baseline_{tag}"
    p = os.path.join(BASE, "exp3", f"pool_split{split}", cell,
                     f"EXP3_{prefix}_{vt}.npy")
    return np.load(p)


def main():
    os.makedirs(os.path.join(BASE, "exp3", "summary"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "exp3", "fig"), exist_ok=True)
    rows = []
    for cell in CELLS:
        for split in range(5):
            pd_ = os.path.join(BASE, "exp3", f"pool_split{split}", cell)
            val = pd.read_csv(os.path.join(pd_, "EXP3_pool_val.csv"))
            test = pd.read_csv(os.path.join(pd_, "EXP3_pool_test.csv"))
            yv = (val.kind == "SE").to_numpy().astype(int)
            yt = (test.kind == "SE").to_numpy().astype(int)
            for m in MODELS:
                pv = load_scores(split, cell, m, "val")
                pt = load_scores(split, cell, m, "test")
                okv, okt = np.isfinite(pv), np.isfinite(pt)
                best_t, best_f1 = 0.5, 0.0
                for t in THR:
                    f1v = f1_at(yv[okv], pv[okv], t)
                    if f1v > best_f1:
                        best_f1, best_t = f1v, t
                auc = roc_auc_score(yt[okt], pt[okt]) if okt.sum() else np.nan
                aupr = average_precision_score(yt[okt], pt[okt])
                rows.append({"cell": cell, "split": split, "model": m,
                             "n_test": int(okt.sum()),
                             "auroc": round(float(auc), 4),
                             "auprc": round(float(aupr), 4),
                             "val_thr": round(float(best_t), 4),
                             "val_f1": round(float(best_f1), 4),
                             "test_f1": round(float(f1_at(yt[okt], pt[okt],
                                                          best_t)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "exp3", "summary",
                           "EXP3_cluster_metrics_all.csv"), index=False)

    sig = []
    for cell in CELLS:
        for metric in ("auroc", "auprc", "test_f1"):
            a = df[(df.cell == cell) & (df.model == "Cerberus")]
            for m in MODELS[1:]:
                b = df[(df.cell == cell) & (df.model == m)]
                x = a[metric].to_numpy(float)
                y = b[metric].to_numpy(float)
                t, p = stats.ttest_rel(x, y)
                sig.append({"cell": cell, "metric": metric,
                            "model_a": "Cerberus", "model_b": m,
                            "mean_a": round(float(x.mean()), 4),
                            "mean_b": round(float(y.mean()), 4),
                            "t": round(float(t), 3), "df": 4,
                            "p": round(float(p), 5)})
    pd.DataFrame(sig).to_csv(os.path.join(
        BASE, "exp3", "summary", "EXP3_benchmark_significance_tests.csv"),
        index=False)

    fig_dir = os.path.join(BASE, "exp3", "fig")
    for metric, lab in (("auprc", "AUPRC"), ("test_f1", "F1 (val-tuned)"),
                        ("auroc", "AUROC")):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 78 / 25.4))
        x = np.arange(len(CELLS))
        w = 0.16
        rng = np.random.default_rng(0)
        for i, m in enumerate(MODELS):
            vals = np.asarray([df[(df.cell == c) & (df.model == m)][metric]
                               for c in CELLS], dtype=float)
            mean = vals.mean(1)
            sem = vals.std(1, ddof=1) / np.sqrt(5)
            ax.bar(x + (i - 2) * w, mean, w, color=COLORS[m], alpha=1.0,
                   label=m.replace("_official", "") if m != "Cerberus" else "Cerberus")
            for c_idx in range(len(CELLS)):
                jit = rng.uniform(-0.02, 0.02, 5)
                ax.scatter(np.full(5, x[c_idx] + (i - 2) * w) + jit,
                           vals[c_idx], s=4, color="#1a1a1a", zorder=5,
                           linewidths=0)
            ax.errorbar(x + (i - 2) * w, mean, yerr=sem, fmt="none",
                        ecolor="#1a1a1a", elinewidth=0.5, capsize=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(CELLS)
        ax.set_ylabel(lab)
        ax.set_title(f"EXP3 · Lineage-specific {lab} (5-split)", fontsize=7, pad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=5.5, ncol=3)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"EXP3_{metric}_bar.pdf")
        fig.savefig(out)
        plt.close(fig)
        print("[fig]", out)
    print("[Done] EXP3 metrics + significance + figures")


if __name__ == "__main__":
    main()
