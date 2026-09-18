#!/usr/bin/env python3
"""EXP3 交叉验证热图：每模型一张 4×4。

行 = pred 谱系（在该细胞训练的模型 / Cerberus 该细胞通道）；
列 = GT 谱系池（该细胞 ROSE SE 行 正 / TE 行 负）；
颜色 = AUPRC（test，5 split 取均值；格内标注 mean±SEM）。
注意：Cerberus 行来自既有 90 通道全染色体稠密轨道（multi-anchor max 池化，
零新推理）；其对角格与 Exp3 主表的逐簇 524 kb 前向值可差 ~0.02（相关 ~0.95），
图注需说明口径。
全局共用色标 0–vmax，便于 5 个模型横向比较。
输出：exp3/fig/EXP3_cross_heatmap_{model}.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from final_style import AUPRC_VMAX  # noqa: E402

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


def score_path(model, split, pred, pool):
    pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
    if model == "Cerberus":
        return os.path.join(pdir, f"EXP3_cerberus_x_{pred}_test.npy")
    return os.path.join(
        pdir, f"EXP3_baseline_x_{model}_{pred}_on_{pool}_test.npy")


def main():
    fig_dir = os.path.join(BASE, "exp3", "fig")
    os.makedirs(fig_dir, exist_ok=True)
    # labels（每个 split × pool 只读一次）
    labels = {}
    for split in range(5):
        for pool in CELLS:
            d = pd.read_csv(os.path.join(BASE, "exp3", f"pool_split{split}",
                                         pool, "EXP3_pool_test.csv"))
            labels[(split, pool)] = (d["kind"] == "SE").to_numpy().astype(int)
    # AUPRC: [model][pred][pool][split]
    auprc = {}
    for model in MODELS:
        M = np.full((4, 4, 5), np.nan)
        for i, pred in enumerate(CELLS):
            for j, pool in enumerate(CELLS):
                for split in range(5):
                    p = score_path(model, split, pred, pool)
                    if not os.path.exists(p):
                        continue
                    sc = np.load(p)
                    y = labels[(split, pool)]
                    ok = np.isfinite(sc)
                    if ok.sum() and 0 < y[ok].sum() < ok.sum():
                        M[i, j, split] = average_precision_score(y[ok], sc[ok])
        auprc[model] = M
    vmax = AUPRC_VMAX  # 所有 AUPRC 热图统一色标
    print(f"[info] unified vmax = {vmax:.2f}")
    for model in MODELS:
        M = auprc[model]
        mean = np.nanmean(M, axis=2)
        sem = np.nanstd(M, axis=2, ddof=1) / np.sqrt(5)
        fig, ax = plt.subplots(figsize=(92 / 25.4, 76 / 25.4))
        sns.heatmap(mean, ax=ax, cmap="viridis", vmin=0, vmax=vmax,
                    square=True, linewidths=0.8, linecolor="white",
                    xticklabels=CELLS, yticklabels=CELLS, cbar=True,
                    cbar_kws={"label": "AUPRC (5-split mean)",
                              "ticks": np.arange(0, vmax + 1e-9, 0.1)})
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        ax.set_xlabel("GT lineage (evaluation pool)")
        ax.set_ylabel("Pred lineage (model / channel)")
        ax.collections[0].colorbar.ax.tick_params(labelsize=6)
        ax.collections[0].colorbar.set_label("AUPRC (5-split mean)", fontsize=6)
        for i in range(4):
            for j in range(4):
                v = mean[i, j]
                txt = "NA" if not np.isfinite(v) else f"{v:.3f}"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=5.2, color="white")
        ax.set_title(f"EXP3 · {DISPLAY[model]} · cross-lineage AUPRC",
                     fontsize=7, pad=5)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"EXP3_cross_heatmap_{model}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print("[fig]", out)
        print(f"  mean matrix ({model}):\n{pd.DataFrame(mean.round(3), index=CELLS, columns=CELLS).to_string()}")
    print("[Done] EXP3 cross heatmaps")


if __name__ == "__main__":
    main()
