#!/usr/bin/env python3
"""基于保存的逐样本 probs 画 PR/ROC 曲线与 F1 barplot（nature 风格，仅 PDF）。

用法：
  python plot_benchmark_curves.py --metrics test_eval/test_metrics.csv \
      --data_dir test_eval --out_dir test_eval --prefix dcse

产出（每 prefix）：
  {prefix}_roc_test_balanced.pdf / {prefix}_roc_test_all.pdf
  {prefix}_pr_test_balanced.pdf  / {prefix}_pr_test_all.pdf
  {prefix}_f1_bar.pdf            （F1 用 val 最优阈值 f1_vt）
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Arial", "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

COLORS = {
    "Cerberus_ctx": "#C1272D",
    "Cerberus": "#C1272D",
    "DcSE": "#984EA3",
    "DeepSE_official": "#FF7F00",
    "DeepSE_orig": "#FDBF6F",
    "MuSE_official": "#4DAF4A",
    "MuSE_orig": "#A6CEE3",
    "SENet": "#377EB8",
}
ORIG_TAGS = {"DeepSE_orig", "MuSE_orig"}
TS_COLORS = {"test_balanced": "#377EB8", "test_all": "#FF7F00"}


def load_metrics(args):
    df = pd.read_csv(args.metrics)
    df = df[df["testset"].isin(args.testsets)].copy()
    # 只保留有逐样本 probs 的行（排除 N-pad 对照若 tag==Cerberus 且无 probs 也可跳过）
    return df


def curve_figs(metrics, data_dir, prefix):
    for ts in args.testsets:
        label_path = os.path.join(data_dir, f"labels_{ts}.npy")
        if not os.path.exists(label_path):
            continue
        y = np.load(label_path)
        tags = []
        for _, row in metrics[metrics["testset"] == ts].iterrows():
            tag = row["tag"]
            p = os.path.join(data_dir, f"{tag}.{ts}.probs.npy")
            if os.path.exists(p) and (tag != "Cerberus" or args.include_npad):
                tags.append((tag, np.load(p)))
        if not tags:
            continue
        pos_rate = float(y.mean())

        # ROC
        fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="0.6")
        for tag, p in tags:
            fpr, tpr, _ = roc_curve(y, p)
            a = roc_auc_score(y, p)
            ax.plot(fpr, tpr, color=COLORS.get(tag, "#333333"),
                    lw=1.4, ls="--" if tag in ORIG_TAGS else "-",
                    label=f"{tag} (AUC={a:.3f})")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{prefix}_roc_{ts}.pdf"))
        plt.close(fig)

        # PR
        fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
        ax.axhline(pos_rate, ls="--", lw=0.7, color="0.6")
        for tag, p in tags:
            prec, rec, _ = precision_recall_curve(y, p)
            a = average_precision_score(y, p)
            ax.plot(rec, prec, color=COLORS.get(tag, "#333333"),
                    lw=1.4, ls="--" if tag in ORIG_TAGS else "-",
                    label=f"{tag} (AUPRC={a:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{prefix}_pr_{ts}.pdf"))
        plt.close(fig)
        print(f"[curve] {prefix} {ts}: {len(tags)} models")


def f1_bar(metrics, prefix):
    m = metrics[["tag", "testset", "f1_vt"]].dropna().copy()
    if m.empty:
        print("[skip] 无 f1_vt 列，跳过 F1 barplot")
        return
    order = sorted(m["tag"].unique(),
                   key=lambda t: (t not in ["Cerberus_ctx"], t))
    testsets = [t for t in ["test_balanced", "test_all"] if t in m["testset"].unique()]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=300)
    for i, ts in enumerate(testsets):
        vals = []
        for t in order:
            sub = m[(m["tag"] == t) & (m["testset"] == ts)]
            vals.append(float(sub["f1_vt"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - 0.5) * w, vals, width=w,
               color=TS_COLORS.get(ts, "#377EB8"),
               edgecolor="none", label=ts)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("F1 (val-optimal threshold)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{prefix}_f1_bar.pdf"))
    plt.close(fig)
    print(f"[bar] {prefix}_f1_bar.pdf")


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--testsets", nargs="+", default=["test_balanced", "test_all"])
    ap.add_argument("--include_npad", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = load_metrics(args)
    curve_figs(metrics, args.data_dir, args.prefix)
    f1_bar(metrics, args.prefix)


if __name__ == "__main__":
    main()
