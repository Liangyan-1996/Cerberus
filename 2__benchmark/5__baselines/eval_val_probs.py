#!/usr/bin/env python3
"""在 chr9/14 validation 上为每个模型选分类阈值（F1 最优），并回填 test 指标。

协议（2026-09-06 与用户确认）：
  - AUROC/AUPRC 阈值无关，直接报告，不涉及 validation 阈值；
  - ACC/F1 阈值相关：每个模型（含 Cerberus_ctx）用各自 val 概率选 F1 最优
    阈值（网格搜索），再计算 test 两套集上的 ACC/F1。

产出：benchmark/test_eval/
  val.{tag}.probs.npy / labels_val.npy / Cerberus_ctx.val.probs.npy
  val_thresholds.json
  test_metrics.csv（新增列 val_auc/val_aupr/val_thr/acc_vt/f1_vt）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyfaidx
import torch
from scipy.ndimage import gaussian_filter1d
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
    ("DeepSE_orig", "DeepSE", KMER_ORIG),
    ("DeepSE_official", "DeepSE", KMER_OFFICIAL),
    ("SENet", "SENet", None),
    ("MuSE_orig", "MuSE", KMER_ORIG),
    ("MuSE_official", "MuSE", KMER_OFFICIAL),
    ("DcSE", "DcSE", None),
]


def predict_baseline(model_name, ckpt_path, csv_path, kmer_mats, scaler, device):
    ds = SeqDataset(csv_path, model_name, kmer_mats=kmer_mats, x2_scaler=scaler)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        collate_fn=_collate)
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
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
    """网格搜索使 val F1 最大的阈值。"""
    grid = np.unique(np.concatenate([np.linspace(p.min(), p.max(), 2001),
                                     [0.5]]))
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), float(best_f1)

def cerberus_ctx_scores(csv_path, coords, chrom_len, fa, model, device,
                        batch_size=4, use_bf16=False):
    from eval_cerberus_test_ctx import (  # noqa: PLC0415
        HALF, INPUT_LEN, safe_center, window_bins)
    from se_data_utils import dna_one_hot  # noqa: PLC0415
    df = pd.read_csv(csv_path)
    encs, binss, kept = [], [], []
    for i, key in enumerate(df["row"].astype(int)):
        key = int(key)
        if key not in coords:
            continue
        c, fs, fe = coords[key]
        center = (fs + fe) // 2
        ln = chrom_len[c]
        sc = safe_center(center, ln)
        ws, we = sc - HALF, sc + HALF          # 窗口必在染色体内部
        seq = fa[c][ws:we].upper()
        assert len(seq) == INPUT_LEN
        encs.append(dna_one_hot(np.frombuffer(seq.encode(), dtype=np.int8)))
        binss.append(window_bins(fs, fe, sc))
        kept.append(i)
    labels = df["label"].to_numpy()[np.asarray(kept, dtype=int)]
    scores = []
    with torch.no_grad():
        for i0 in range(0, len(encs), batch_size):
            chunk = torch.stack(encs[i0:i0 + batch_size]).to(device)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    cl, sl = model(chunk)
            else:
                cl, sl = model(chunk)
            joint = (torch.sigmoid(cl.float()) * torch.sigmoid(sl.float())).cpu().numpy()
            joint = gaussian_filter1d(joint, sigma=1.0, axis=-1)
            mx = joint.max(axis=1)
            for j, (b0, b1) in enumerate(binss[i0:i0 + batch_size]):
                scores.append(float(mx[j, b0:b1].mean()))
    return labels, np.asarray(scores, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_csv", default=os.path.normpath(os.path.join(
        _DIR, "../alignment/train.csv")))
    ap.add_argument("--val_csv", default=os.path.normpath(os.path.join(
        _DIR, "../alignment/val.csv")))
    ap.add_argument("--test_dir", default=os.path.normpath(os.path.join(
        _DIR, "../alignment")))
    ap.add_argument("--out_dir", default=os.path.join(_DIR, "test_eval"))
    ap.add_argument("--testsets", nargs="+", default=["test_balanced", "test_all"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    use_bf16 = device.type == "cuda" and cap[0] >= 8
    os.makedirs(args.out_dir, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    val_labels = val_df["label"].to_numpy()
    np.save(os.path.join(args.out_dir, "labels_val.npy"), val_labels)

    # ---------- baselines ----------
    base_thr = {}
    for tag, model_name, km in CONFIGS:
        ckpt = os.path.join(CKPT, tag, f"{model_name}_best_chrsplit.pth")
        if not os.path.exists(ckpt):
            print(f"[Skip] {tag} ckpt missing")
            continue
        mats = load_kmer_matrices(km) if km else None
        scaler = None
        if model_name == "DeepSE":
            ds0 = SeqDataset(args.train_csv, "DeepSE", kmer_mats=mats)
            scaler = StandardScaler().fit(np.stack(ds0._x2))
        y, p = predict_baseline(model_name, ckpt, args.val_csv, mats, scaler, device)
        np.save(os.path.join(args.out_dir, f"val.{tag}.probs.npy"), p)
        thr, f1v = best_threshold_f1(y, p)
        base_thr[tag] = {
            "thr": thr, "val_f1": f1v,
            "val_auc": roc_auc_score(y, p),
            "val_aupr": average_precision_score(y, p),
        }
        print(f"[Val] {tag:18s} thr={thr:.4f} F1={f1v:.4f} "
              f"AUC={base_thr[tag]['val_auc']:.4f} AUPR={base_thr[tag]['val_aupr']:.4f}")

    # ---------- Cerberus (真实上下文) ----------
    from eval_cerberus_test_ctx import load_model, parse_sam_coords  # noqa: PLC0415
    SAM = os.path.normpath(os.path.join(_DIR, "../alignment/dcse_human.sam"))
    FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
    coords, chrom_len = parse_sam_coords(SAM)
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    model = load_model(device)
    yv, pv = cerberus_ctx_scores(args.val_csv, coords, chrom_len, fa, model,
                                 device, use_bf16=use_bf16)
    np.save(os.path.join(args.out_dir, "val.Cerberus_ctx.probs.npy"), pv)
    thr, f1v = best_threshold_f1(yv, pv)
    cer_thr = {"thr": thr, "val_f1": f1v,
               "val_auc": roc_auc_score(yv, pv),
               "val_aupr": average_precision_score(yv, pv)}
    print(f"[Val] Cerberus_ctx       thr={thr:.4f} F1={f1v:.4f} "
          f"AUC={cer_thr['val_auc']:.4f} AUPR={cer_thr['val_aupr']:.4f}")

    thr_all = {**base_thr, "Cerberus_ctx": cer_thr}
    with open(os.path.join(args.out_dir, "val_thresholds.json"), "w") as f:
        json.dump(thr_all, f, indent=2, default=float)

    # ---------- 回填 test 指标 ----------
    df = pd.read_csv(os.path.join(args.out_dir, "test_metrics.csv"))
    new_cols = []
    for _, row in df.iterrows():
        tag = row["tag"]
        if tag not in thr_all:
            new_cols.append(row.to_dict())
            continue
        p = np.load(os.path.join(args.out_dir,
                                 f"{tag}.{row['testset']}.probs.npy"))
        y = np.load(os.path.join(args.out_dir, f"labels_{row['testset']}.npy"))
        pred = (p >= thr_all[tag]["thr"]).astype(int)
        r = row.to_dict()
        r["val_auc"] = round(thr_all[tag]["val_auc"], 4)
        r["val_aupr"] = round(thr_all[tag]["val_aupr"], 4)
        r["val_thr"] = round(thr_all[tag]["thr"], 4)
        r["acc_vt"] = round(accuracy_score(y, pred), 4)
        r["f1_vt"] = round(f1_score(y, pred), 4)
        new_cols.append(r)
    out = pd.DataFrame(new_cols)
    out.to_csv(os.path.join(args.out_dir, "test_metrics.csv"), index=False)
    print("[Done] test_metrics.csv 已追加 val 阈值列 (val_auc/val_aupr/val_thr/acc_vt/f1_vt)")


if __name__ == "__main__":
    main()
