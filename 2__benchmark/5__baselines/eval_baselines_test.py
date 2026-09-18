#!/usr/bin/env python3
"""chr3/11 盲测推理：对 6 个 chr-split best checkpoint 输出测试指标。

配置（TAG 与 BENCHMARK_RUN_LOG.md 命名一致）：
  DeepSE_orig / DeepSE_official / SENet / MuSE_orig / MuSE_official / DcSE
测试集：test_balanced.csv（1:1）与 test_all.csv（真实分布），均只含 chr3/chr11。
产出：benchmark/test_eval/
  test_metrics.csv        : tag, embedding, testset, n, auc, aupr, acc, f1
  labels_{testset}.npy    : 真实标签
  {tag}.{testset}.probs.npy : 逐样本 P(SE)

用法：
  python eval_baselines_test.py [--device cuda] [--test_dir ../alignment]
"""
import argparse
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

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from features import load_kmer_matrices  # noqa: E402
from train_baseline import SeqDataset, _collate, get_model  # noqa: E402

KMER_ORIG = os.path.join(_DIR, "kmer_deepse_hg19.npz")
KMER_OFFICIAL = os.path.join(_DIR, "kmer_official_k3to8.npz")
CKPT = os.path.join(_DIR, "ckpt")

CONFIGS = [
    # (tag, model, kmer_npz_or_None)
    ("DeepSE_orig", "DeepSE", KMER_ORIG),
    ("DeepSE_official", "DeepSE", KMER_OFFICIAL),
    ("SENet", "SENet", None),
    ("MuSE_orig", "MuSE", KMER_ORIG),
    ("MuSE_official", "MuSE", KMER_OFFICIAL),
    ("DcSE", "DcSE", None),
]


def fit_deepse_scaler(train_csv, kmer_mats):
    ds = SeqDataset(train_csv, "DeepSE", kmer_mats=kmer_mats)
    scaler = StandardScaler().fit(np.stack(ds._x2))
    return scaler, ds


def predict_set(model, loader, device, is_muse):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            if is_muse:
                x6, x2, x3, y = batch
                out = model(x6.to(device), x2.to(device), x3.to(device))
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            else:
                x, y = batch
                out = model(x.to(device))
                p = out.cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_csv",
                    default=os.path.normpath(os.path.join(_DIR, "../alignment/train.csv")))
    ap.add_argument("--test_dir",
                    default=os.path.normpath(os.path.join(_DIR, "../alignment")))
    ap.add_argument("--testsets", nargs="+", default=["test_balanced", "test_all"])
    ap.add_argument("--out_dir", default=os.path.join(_DIR, "test_eval"))
    ap.add_argument("--ckpt_root", default=os.path.join(_DIR, "ckpt"))
    ap.add_argument("--tags", nargs="+", default=None,
                    help="只跑指定 tag（默认全部 6 个）")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Info] device={device} testsets={args.testsets} out={args.out_dir}")

    # DeepSE scaler 只需拟合一次（train 上，与训练一致）
    deepse_scaler = {}
    deepse_mats = {}
    for tag, model, km in CONFIGS:
        if args.tags and tag not in args.tags:
            continue
        if model == "DeepSE":
            mats = load_kmer_matrices(km)
            if tag not in deepse_scaler:
                scaler, _ = fit_deepse_scaler(args.train_csv, mats)
                deepse_scaler[tag] = scaler
                deepse_mats[tag] = mats

    rows = []
    for testset in args.testsets:
        csv_path = os.path.join(args.test_dir, f"{testset}.csv")
        df = pd.read_csv(csv_path)
        labels = df["label"].to_numpy()
        np.save(os.path.join(args.out_dir, f"labels_{testset}.npy"), labels)
        print(f"[Data] {testset}: {len(df)} rows, SE={int(labels.sum())}")

        for tag, model_name, km in CONFIGS:
            if args.tags and tag not in args.tags:
                continue
            ckpt_path = os.path.join(args.ckpt_root, tag,
                                     f"{model_name}_best_chrsplit.pth")
            if not os.path.exists(ckpt_path):
                print(f"[Skip] {tag}: ckpt 不存在 {ckpt_path}")
                continue
            kmer_mats = load_kmer_matrices(km) if km else None
            scaler = deepse_scaler.get(tag)
            ds = SeqDataset(csv_path, model_name, kmer_mats=kmer_mats,
                            x2_scaler=scaler)
            loader = DataLoader(ds, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=_collate)
            model = get_model(model_name)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            model.to(device)
            probs, y = predict_set(model, loader, device,
                                   is_muse=(model_name == "MuSE"))
            assert np.array_equal(y, labels), f"{tag}: label 顺序不一致"
            np.save(os.path.join(args.out_dir, f"{tag}.{testset}.probs.npy"), probs)

            pred = (probs >= 0.5).astype(int)
            row = {
                "tag": tag, "embedding": "orig" if "orig" in tag
                else ("official" if "official" in tag else "official6mer"),
                "testset": testset, "n": len(y),
                "auc": round(roc_auc_score(y, probs), 4),
                "aupr": round(average_precision_score(y, probs), 4),
                "acc": round(accuracy_score(y, pred), 4),
                "f1": round(f1_score(y, pred), 4),
            }
            rows.append(row)
            print(f"[Eval] {tag:18s} {testset:13s} "
                  f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} "
                  f"ACC={row['acc']:.4f} F1={row['f1']:.4f}")

    out_csv = os.path.join(args.out_dir, "test_metrics.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        if new_df.empty:
            new_df = old
        else:
            keep_tags = set(old["tag"]) - set(new_df["tag"])
            old = old[old["tag"].isin(keep_tags)]
            new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(out_csv, index=False)
    print(f"[Done] {out_csv} (rows={len(new_df)})")


if __name__ == "__main__":
    main()
