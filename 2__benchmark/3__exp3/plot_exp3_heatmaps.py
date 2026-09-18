#!/usr/bin/env python3
"""EXP3 热图：每模型一张（4 细胞 × 3 指标，5-split mean±SEM），
外加每模型一张 AUPRC 分 split 透明版（4 细胞 × 5 split + mean±SEM 汇总列）。

输出：exp3/fig/EXP3_heatmap_<model>.pdf
      exp3/fig/EXP3_heatmap_<model>_auprc_splits.pdf
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
from final_style import COLORS  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 7

CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
MODELS = ["Cerberus", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]
DISPLAY = {"Cerberus": "Cerberus", "DeepSE_official": "DeepSE",
           "SENet": "SENet", "MuSE_official": "MuSE", "DcSE": "DcSE"}
METRICS = [("auprc", "AUPRC"), ("auroc", "AUROC"), ("test_f1", "F1_vt")]


def cell_text(mean, sem, fmt="{:.3f}±{:.3f}"):
    return fmt.format(mean, sem)


def text_color(v):
    return "white" if v > 0.6 else "#1a1a1a"


def main():
    df = pd.read_csv(os.path.join(BASE, "exp3", "summary",
                                  "EXP3_cluster_metrics_all.csv"))
    fig_dir = os.path.join(BASE, "exp3", "fig")
    os.makedirs(fig_dir, exist_ok=True)

    # 每指标全局 min–max（跨 5 模型 × 4 细胞 的 5-split 均值）
    means, sems = {}, {}
    norm = {}
    for metric, _ in METRICS:
        m = np.zeros((len(MODELS), len(CELLS)))
        s = np.zeros_like(m)
        for i, model in enumerate(MODELS):
            for j, c in enumerate(CELLS):
                v = df[(df.cell == c) & (df.model == model)][metric].to_numpy(float)
                m[i, j] = v.mean()
                s[i, j] = v.std(ddof=1) / np.sqrt(len(v))
        means[metric], sems[metric] = m, s
        lo, hi = m.min(), m.max()
        norm[metric] = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)

    # ---- 主热图：4 细胞 × 3 指标（颜色=每指标 min–max 归一化）----
    for i, model in enumerate(MODELS):
        mat = np.stack([norm[k][i] for k, _ in METRICS], axis=1)  # (4,3)
        fig, ax = plt.subplots(figsize=(88 / 25.4, 62 / 25.4))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels([lab for _, lab in METRICS])
        ax.set_yticks(range(len(CELLS)))
        ax.set_yticklabels(CELLS)
        ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CELLS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for j, (metric, _) in enumerate(METRICS):
            for k, c in enumerate(CELLS):
                v = norm[metric][i, k]
                ax.text(j, k, cell_text(means[metric][i, k],
                                        sems[metric][i, k]),
                        ha="center", va="center", fontsize=5.6,
                        color=text_color(v))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("per-metric min–max norm.", fontsize=6)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(f"EXP3 · {DISPLAY[model]} (5-split mean ± SEM)",
                     fontsize=7, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"EXP3_heatmap_{model}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print("[fig]", out)

    # ---- 附图：AUPRC 分 split（4 细胞 × 5 split + mean±SEM 汇总列）----
    auprc = np.zeros((len(MODELS), len(CELLS), 5))
    for i, model in enumerate(MODELS):
        for j, c in enumerate(CELLS):
            v = df[(df.cell == c) & (df.model == model)].sort_values("split")
            auprc[i, j, :] = v["auprc"].to_numpy(float)
    lo, hi = auprc.min(), auprc.max()
    for i, model in enumerate(MODELS):
        fig, ax = plt.subplots(figsize=(110 / 25.4, 58 / 25.4))
        mat = auprc[i]
        norm_i = (mat - lo) / (hi - lo)
        im = ax.imshow(norm_i, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlim(-0.5, 5.6)
        ax.set_xticks(list(range(5)) + [5.0])
        ax.set_xticklabels([f"s{k}" for k in range(5)] + ["mean±SEM"])
        ax.set_yticks(range(len(CELLS)))
        ax.set_yticklabels(CELLS)
        ax.axvline(4.5, color="white", lw=1.4)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CELLS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for j, c in enumerate(CELLS):
            for k in range(5):
                ax.text(k, j, f"{mat[j, k]:.3f}", ha="center", va="center",
                        fontsize=5.4, color=text_color(norm_i[j, k]))
            ax.text(5.0, j, cell_text(mat[j].mean(),
                                      mat[j].std(ddof=1) / np.sqrt(5)),
                    ha="center", va="center", fontsize=5.4, color="#1a1a1a")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("AUPRC (global min–max)", fontsize=6)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(f"EXP3 · {DISPLAY[model]} · AUPRC per split",
                     fontsize=7, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        out = os.path.join(fig_dir,
                           f"EXP3_heatmap_{model}_auprc_splits.pdf")
        fig.savefig(out)
        plt.close(fig)
        print("[fig]", out)
    print("[Done] EXP3 heatmaps")


if __name__ == "__main__":
    main()
