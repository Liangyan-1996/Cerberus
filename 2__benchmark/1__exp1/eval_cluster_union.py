#!/usr/bin/env python3
"""EXP1 S4：簇级指标 + val 阈值 + ROC/PR 曲线缓存。

对每 split 每模型：
  - val 池：阈值网格 [0.01,0.99] 最大化 F1 -> T*_val（簇级二分类直接判定）；
  - test 池：AUROC / AUPRC（自然分布）+ F1@T*_val（P/R 同报）；
  - 缓存 test 池 ROC/PR 轨迹（供 5-split SEM ribbon）。
输出：exp1/split{s}/EXP1_cluster_metrics.csv + exp1/curves/EXP1_curves_*.npz
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    roc_auc_score, roc_curve

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
MODELS = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]
THR_GRID = np.arange(0.01, 0.995, 0.005)


def f1_at(y, p, thr):
    if len(y) == 0:
        return 0.0
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p_, r = (tp / (tp + fp)) if (tp + fp) else 0.0, \
            (tp / (tp + fn)) if (tp + fn) else 0.0
    return (2 * p_ * r / (p_ + r)) if (p_ + r) else 0.0, p_, r


def score_of(split, tag, val_test):
    p = os.path.join(BASE, "exp1", f"split{split}", f"EXP1_{tag}_{val_test}.npy")
    arr = np.load(p)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    os.makedirs(os.path.join(BASE, "exp1", "curves"), exist_ok=True)
    all_rows = []
    for split in args.splits:
        split_dir = os.path.join(BASE, "exp1", f"split{split}")
        val = pd.read_csv(os.path.join(split_dir, "EXP1_pool_val.csv"))
        test = pd.read_csv(os.path.join(split_dir, "EXP1_pool_test.csv"))
        yv = (val["kind"] == "SE").to_numpy().astype(int)
        yt = (test["kind"] == "SE").to_numpy().astype(int)
        srcs = {m: ("cerberus_union" if m == "Cerberus_union"
                    else f"baseline_{m}") for m in MODELS}
        curves = {}
        for m in MODELS:
            pv = score_of(split, srcs[m], "val")
            pt = score_of(split, srcs[m], "test")
            assert len(pv) == len(val) and len(pt) == len(test), (m, len(pv), len(pt))
            okv = np.isfinite(pv)
            okt = np.isfinite(pt)
            yv2, pv2 = yv[okv], pv[okv]
            yt2, pt2 = yt[okt], pt[okt]
            if yv2.sum() == 0 or yv2.sum() == len(yv2) or len(yv2) < 20:
                print(f"[warn] split{split} {m} val 有效样本不足/单类，跳过")
                continue
            best_f1, best_thr = 0.0, 0.5
            for t in THR_GRID:
                f1v, _, _ = f1_at(yv2, pv2, t)
                if f1v > best_f1:
                    best_f1, best_thr = f1v, float(t)
            test_f1, pr_, rc_ = f1_at(yt2, pt2, best_thr)
            if len(yt2) == 0 or yt2.sum() == 0 or yt2.sum() == len(yt2):
                print(f"[warn] split{split} {m} test 单类，指标置 NaN")
                auc = aupr = float("nan")
            else:
                auc = float(roc_auc_score(yt2, pt2))
                aupr = float(average_precision_score(yt2, pt2))
            all_rows.append({
                "split": split, "model": m, "n_val": int(len(yv2)),
                "n_pos_val": int(yv2.sum()), "n_test": int(len(yt2)),
                "n_pos_test": int(yt2.sum()),
                "val_thr": round(best_thr, 4), "val_f1": round(best_f1, 4),
                "auroc": round(auc, 4), "auprc": round(aupr, 4),
                "test_f1": round(test_f1, 4), "precision": round(pr_, 4),
                "recall": round(rc_, 4)})
            fpr, tpr, _ = roc_curve(yt2, pt2)
            prec, rec, _ = precision_recall_curve(yt2, pt2)
            curves[m] = {"fpr": fpr, "tpr": tpr, "precision": prec,
                         "recall": rec, "auroc": auc, "auprc": aupr,
                         "val_thr": best_thr}
        df = pd.DataFrame(all_rows)
        if df[df.split == split].empty:
            continue
        p_metrics = os.path.join(split_dir, "EXP1_cluster_metrics.csv")
        df[df.split == split].to_csv(p_metrics, index=False)
        np.savez(os.path.join(BASE, "exp1", "curves",
                              f"EXP1_curves_split{split}.npz"),
                 **{f"{m}_{k}": np.asarray(v) for m, d in curves.items()
                    for k, v in d.items()})
        print(f"[save] split{split} metrics -> {p_metrics}")
    pd.DataFrame(all_rows).to_csv(
        os.path.join(BASE, "summary", "EXP1_cluster_metrics_all.csv"),
        index=False) if all_rows else None
    print("[Done] EXP1 cluster metrics")


if __name__ == "__main__":
    main()
