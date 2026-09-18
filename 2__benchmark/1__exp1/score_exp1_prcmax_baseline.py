#!/usr/bin/env python3
"""EXP1 peak-restricted+max 基线打分：块分数 = 块内构成峰中心 3 kb 窗分数最大值。
输出：exp1/split{s}/EXP1_prcmax_baseline_{tag}_{val,test}.npy
用法：python score_exp1_prcmax_baseline.py --split 0 --tag DcSE
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--tag", required=True, choices=TAGS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split, tag = args.split, args.tag
    base = BASE_MODEL[tag]
    out_dir = os.path.join(BASE, "exp1", f"split{split}")
    if split == 0:
        ckpt_dir = os.path.join(REPO, "test10_eval_new/"
                                "20260906.ownSE_annotation__benchmark/ckpt")
        train_csv = os.path.join(REPO, "test10_eval_new/"
                                 "20260906.ownSE_annotation__benchmark/data/train.csv")
    else:
        ckpt_dir = os.path.join(REPO, "test10_eval_new/"
                                f"20260907.multi_chr_split__benchmark/ckpt_split{split}")
        train_csv = os.path.join(REPO, "test10_eval_new/"
                                 f"20260907.multi_chr_split__benchmark/data_split{split}/train.csv")
    ckpt = os.path.join(ckpt_dir, tag, f"{base}_best_chrsplit.pth")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mats = load_kmer_matrices(KMER) if base in ("DeepSE", "MuSE") else None
    scaler = None
    if base == "DeepSE":
        tr = pd.read_csv(train_csv, usecols=["sequence"])
        x2 = np.stack([seq_to_deepse_vec(str(s), mats) for s in tr["sequence"]])
        scaler = StandardScaler().fit(x2)
    model = get_model(base)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device); model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    for ttag in ("val", "test"):
        npz = np.load(os.path.join(out_dir, f"EXP1_prcmax_peaks_{ttag}.npz"))
        cidx, pc = npz["cidx"], npz["p_center"]
        df = pd.read_csv(os.path.join(out_dir, f"EXP1_pool_{ttag}.csv"))
        chroms = df.iloc[cidx]["chr"].to_numpy()
        keep, seqs = [], []
        for k, (c, cc) in enumerate(zip(chroms, pc)):
            s, e = int(cc) - W // 2, int(cc) + W // 2
            if s < 0 or e > len(fa[c]):
                continue
            seq = str(fa[c][s:e]).upper()
            if seq.count("N") > 0.5 * W:
                continue
            keep.append(k); seqs.append(seq)
        scores = np.full(len(df), np.nan, dtype=np.float32)
        if seqs:
            p = infer(model, device, base, mats, scaler, seqs)
            mx = np.full(len(df), -np.inf)
            for k, val in zip(keep, p):
                j = cidx[k]
                mx[j] = max(mx[j], float(val))
            ok = mx > -np.inf
            scores[ok] = mx[ok].astype(np.float32)
        out = os.path.join(out_dir, f"EXP1_prcmax_baseline_{tag}_{ttag}.npy")
        np.save(out, scores)
        print(f"[{ttag}] scored={int(np.isfinite(scores).sum())}/{len(df)} -> {out}")
    print("[Done] EXP1 peak-restricted max baseline scores")


if __name__ == "__main__":
    main()
