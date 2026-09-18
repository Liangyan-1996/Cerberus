#!/usr/bin/env python3
"""EXP3 peak-restricted cluster 基线打分：簇分数 = 该簇构成峰中心 3 kb 窗分数均值。

输出（pool_split{s}/{cell}/）：
  EXP3_prc_baseline_x_{tag}_{pred}_on_{pool}_test.npy
  pool==pred 时另存 EXP3_prc_baseline_{tag}_{val,test}.npy
用法：python score_exp3_baseline_prc.py --split 0 --pred K562 --tag DcSE
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.preprocessing import StandardScaler

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
BDIR = os.path.join(REPO, "test10_eval_new/20260905.model_benchmark/benchmark")
sys.path.insert(0, BDIR)
from features import (load_kmer_matrices, seq_to_6mer_tokens,  # noqa: E402
                      seq_to_deepse_vec, seq_to_onehot)
from train_baseline import get_model  # noqa: E402

FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
KMER = os.path.join(BDIR, "kmer_official_k3to8.npz")
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
TAGS = ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]
BASE_MODEL = {"DeepSE_official": "DeepSE", "SENet": "SENet",
              "MuSE_official": "MuSE", "DcSE": "DcSE"}
W = 3000


def infer(model, device, base, mats, scaler, seqs, batch=64):
    out = []
    with torch.no_grad():
        for i0 in range(0, len(seqs), batch):
            good = seqs[i0:i0 + batch]
            if base in ("DcSE", "SENet"):
                x = np.stack([seq_to_6mer_tokens(s) for s in good])
                p = model(torch.from_numpy(x).to(device)).cpu().numpy()
            elif base == "DeepSE":
                x2 = scaler.transform(np.stack(
                    [seq_to_deepse_vec(s, mats) for s in good]))
                p = model(torch.from_numpy(x2.astype(np.float32)).to(device)).cpu().numpy()
            else:
                x6 = np.stack([seq_to_6mer_tokens(s) for s in good])
                x2 = np.stack([seq_to_deepse_vec(s, mats).reshape(1, 300)
                               for s in good])
                x3 = np.stack([seq_to_onehot(s, alphabet="atcg") for s in good])
                lg = model(torch.from_numpy(x6).to(device),
                           torch.from_numpy(x2.astype(np.float32)).to(device),
                           torch.from_numpy(x3).to(device))
                p = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def score_pool(pool_dir, tag, fa, model, device, base, mats, scaler, df, agg):
    npz = np.load(os.path.join(pool_dir, f"EXP3_prc_peaks_{tag}.npz"))
    cidx, pc = npz["cidx"], npz["p_center"]
    scores = np.full(len(df), np.nan, dtype=np.float32)
    if len(cidx) == 0:
        return scores
    # 生成每个峰中心窗口
    rows = df.iloc[cidx]
    chroms = rows["chr"].to_numpy()
    keep, seqs, kidx = [], [], []
    for k, (c, cc) in enumerate(zip(chroms, pc)):
        s, e = int(cc) - W // 2, int(cc) + W // 2
        if s < 0 or e > len(fa[c]):
            continue
        seq = str(fa[c][s:e]).upper()
        if seq.count("N") > 0.5 * W:
            continue
        kidx.append(k); seqs.append(seq)
    if not seqs:
        return scores
    p = infer(model, device, base, mats, scaler, seqs)
    sums = np.zeros(len(df)); cnt = np.zeros(len(df), np.int32)
    mx = np.full(len(df), -np.inf)
    for k, val in zip(kidx, p):
        j = cidx[k]
        sums[j] += float(val); cnt[j] += 1
        mx[j] = max(mx[j], float(val))
    ok = cnt > 0
    scores[ok] = ((sums[ok] / cnt[ok]) if agg == "mean" else mx[ok]).astype(np.float32)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--pred", required=True, choices=CELLS)
    ap.add_argument("--tag", required=True, choices=TAGS)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    PRE = "EXP3_prc" if args.agg == "mean" else "EXP3_prcmax"
    split, pred, tag = args.split, args.pred, args.tag
    base = BASE_MODEL[tag]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mats = load_kmer_matrices(KMER) if base in ("DeepSE", "MuSE") else None
    scaler = None
    if base == "DeepSE":
        tr = pd.read_csv(os.path.join(BASE, "exp3", f"data_split{split}",
                                      pred, "train.csv"), usecols=["sequence"])
        x2 = np.stack([seq_to_deepse_vec(str(s), mats) for s in tr["sequence"]])
        scaler = StandardScaler().fit(x2)
    model = get_model(base)
    ck = os.path.join(BASE, "exp3", f"ckpt_split{split}", pred, tag,
                      f"{base}_best_chrsplit.pth")
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model.to(device); model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    for pool in CELLS:
        pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        for ttag in ("val", "test"):
            if ttag == "val" and pool != pred:
                continue
            df = pd.read_csv(os.path.join(pdir, f"EXP3_pool_{ttag}.csv"))
            sc = score_pool(pdir, ttag, fa, model, device, base, mats, scaler, df,
                            args.agg)
            if pool == pred:
                np.save(os.path.join(pdir, f"{PRE}_baseline_{tag}_{ttag}.npy"), sc)
            if ttag == "test":
                np.save(os.path.join(pdir, f"{PRE}_baseline_x_{tag}_{pred}"
                                           f"_on_{pool}_test.npy"), sc)
            print(f"[prc {pred}->{pool} {ttag}] scored="
                  f"{int(np.isfinite(sc).sum())}/{len(sc)}")
    print("[Done] EXP3 baseline peak-restricted cluster scores")


if __name__ == "__main__":
    main()
