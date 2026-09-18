#!/usr/bin/env python3
"""own-SE benchmark 汇总：四基线 val 阈值筛选 + 回填 test 指标 + Cerberus 合并。

前置（依赖链保证）：
  1) 四基线训练完成（319982-319985）;
  2) ownSE Cerberus val/test 打分完成（319986，产出 Cerberus_ctx.*.probs.npy）;
  3) eval_baselines_test.py 已写出基线 test 行与 labels_*.npy。

产出（ADIR/test_eval/）：
  val.<tag>.probs.npy（基线 val 概率）
  val_thresholds_ownSE.json
  test_metrics.csv（追加 val_auc/val_aupr/val_thr/acc_vt/f1_vt，只留 test 行）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BDIR = os.path.join(REPO, "test10_eval_new/20260905.model_benchmark/benchmark")
ADIR = os.path.join(REPO, "test10_eval_new/20260906.ownSE_annotation__benchmark")
sys.path.insert(0, BDIR)

from features import load_kmer_matrices  # noqa: E402
from train_baseline import SeqDataset, _collate, get_model  # noqa: E402

DATA = os.path.join(ADIR, "data")
OUT = os.path.join(ADIR, "test_eval")
CKPT = os.path.join(ADIR, "ckpt")
KMER = os.path.join(BDIR, "kmer_deepse_hg19.npz")
CONFIGS = [
    ("DeepSE_orig", "DeepSE", KMER),
    ("DeepSE_official", "DeepSE", os.path.join(BDIR, "kmer_official_k3to8.npz")),
    ("SENet", "SENet", None),
    ("MuSE_orig", "MuSE", KMER),
    ("MuSE_official", "MuSE", os.path.join(BDIR, "kmer_official_k3to8.npz")),
    ("DcSE", "DcSE", None),
]
TESTSETS = ["test_balanced", "test_all"]


def predict(model_name, ckpt_path, csv_path, kmer_mats, scaler, device):
    ds = SeqDataset(csv_path, model_name, kmer_mats=kmer_mats, x2_scaler=scaler)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        collate_fn=_collate)
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                     weights_only=True))
    model.to(device)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if model_name == "MuSE":
                x6, x2, x3, y = batch
                out = model(x6.to(device), x2.to(device), x3.to(device))
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            else:
                x, y = batch
                p = model(x.to(device)).cpu().numpy()
            ys.append(y.numpy())
            ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


def best_threshold_f1(y, p):
    grid = np.unique(np.concatenate([np.linspace(p.min(), p.max(), 2001), [0.5]]))
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), float(best_f1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")
    os.makedirs(OUT, exist_ok=True)
    train_csv = os.path.join(DATA, "train.csv")
    val_csv = os.path.join(DATA, "val.csv")
    val_df = pd.read_csv(val_csv)
    val_y = val_df["label"].to_numpy()

    thr = {}
    # ---- baselines: val 推理 + F1 最优阈值 ----
    for tag, model_name, km in CONFIGS:
        ckpt = os.path.join(CKPT, tag, f"{model_name}_best_chrsplit.pth")
        assert os.path.exists(ckpt), f"{ckpt} 缺失"
        mats = load_kmer_matrices(km) if km else None
        scaler = None
        if model_name == "DeepSE":
            ds0 = SeqDataset(train_csv, "DeepSE", kmer_mats=mats)
            scaler = StandardScaler().fit(np.stack(ds0._x2))
        y, p = predict(model_name, ckpt, val_csv, mats, scaler, device)
        np.save(os.path.join(OUT, f"val.{tag}.probs.npy"), p)
        t, f1v = best_threshold_f1(y, p)
        thr[tag] = {"thr": t, "val_f1": f1v,
                    "val_auc": float(roc_auc_score(y, p)),
                    "val_aupr": float(average_precision_score(y, p))}
        print(f"[Val] {tag:14s} thr={t:.4f} F1={f1v:.4f} "
              f"AUC={thr[tag]['val_auc']:.4f} AUPR={thr[tag]['val_aupr']:.4f}")

    # ---- Cerberus: 直接用 319986 保存的 val 概率 ----
    cp = os.path.join(OUT, "Cerberus_ctx.val.probs.npy")
    assert os.path.exists(cp), f"{cp} 缺失：需先跑 ownSE Cerberus（319986）"
    pv = np.load(cp)
    t, f1v = best_threshold_f1(val_y, pv)
    thr["Cerberus_ctx"] = {"thr": t, "val_f1": f1v,
                           "val_auc": float(roc_auc_score(val_y, pv)),
                           "val_aupr": float(average_precision_score(val_y, pv))}
    print(f"[Val] Cerberus_ctx    thr={t:.4f} F1={f1v:.4f} "
          f"AUC={thr['Cerberus_ctx']['val_auc']:.4f} "
          f"AUPR={thr['Cerberus_ctx']['val_aupr']:.4f}")
    with open(os.path.join(OUT, "val_thresholds_ownSE.json"), "w") as f:
        json.dump(thr, f, indent=2, default=float)

    # ---- 回填 test 指标 ----
    csv_path = os.path.join(OUT, "test_metrics.csv")
    df = pd.read_csv(csv_path)
    df = df[df["testset"].isin(TESTSETS)].copy()
    new_rows = []
    for _, row in df.iterrows():
        tag = row["tag"]
        if tag not in thr:
            new_rows.append(row.to_dict())
            continue
        p = np.load(os.path.join(OUT, f"{tag}.{row['testset']}.probs.npy"))
        y = np.load(os.path.join(OUT, f"labels_{row['testset']}.npy"))
        pred = (p >= thr[tag]["thr"]).astype(int)
        r = row.to_dict()
        r["val_auc"] = round(thr[tag]["val_auc"], 4)
        r["val_aupr"] = round(thr[tag]["val_aupr"], 4)
        r["val_thr"] = round(thr[tag]["thr"], 4)
        r["acc_vt"] = round(accuracy_score(y, pred), 4)
        r["f1_vt"] = round(f1_score(y, pred), 4)
        new_rows.append(r)
    out = pd.DataFrame(new_rows)
    out.to_csv(csv_path, index=False)
    print("\n[Final] own-SE test metrics:")
    cols = ["tag", "testset", "auc", "aupr", "f1", "acc_vt", "f1_vt", "val_thr"]
    show = out[[c for c in cols if c in out.columns]]
    print(show.to_string(index=False))
    print(f"[Done] {csv_path}")


if __name__ == "__main__":
    main()
