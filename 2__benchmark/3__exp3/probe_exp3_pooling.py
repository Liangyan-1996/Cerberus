#!/usr/bin/env python3
"""对照实验：簇内 3 kb 窗分数用 mean 池化 vs max 池化，对比簇级判别力。

用法：python probe_exp3_pooling.py --split 0 --pred K562 --tag DeepSE_official --pool K562
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
BDIR = os.path.join(REPO, "test10_eval_new/20260905.model_benchmark/benchmark")
sys.path.insert(0, BDIR)
from features import (load_kmer_matrices, seq_to_6mer_tokens,  # noqa: E402
                      seq_to_deepse_vec, seq_to_onehot)
from train_baseline import get_model  # noqa: E402

FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
BLACKLIST = "/public/home/liunangroup/liangyan/Genome/blacklist/hg38.blacklist.bed"
KMER = os.path.join(BDIR, "kmer_official_k3to8.npz")
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
TAGS = ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]
BASE_MODEL = {"DeepSE_official": "DeepSE", "SENet": "SENet",
              "MuSE_official": "MuSE", "DcSE": "DcSE"}
W, STEP = 3000, 500


def blacklist(chrom):
    rows = sorted((int(p[1]), int(p[2]))
                  for p in (l.rstrip().split("\t") for l in open(BLACKLIST))
                  if len(p) >= 3 and p[0] == chrom)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ms, me = [], []
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            ms.append(cs); me.append(ce); cs, ce = s, e
    ms.append(cs); me.append(ce)
    return np.asarray(ms, np.int64), np.asarray(me, np.int64)


def ov(ms, me, s, e):
    if len(ms) == 0:
        return np.zeros(len(s), bool)
    i = np.searchsorted(ms, s, side="left")
    o = np.zeros(len(s), bool)
    ok = i < len(ms); o[ok] |= ms[i[ok]] < e[ok]
    ok2 = i > 0; o[ok2] |= me[i[ok2] - 1] > s[ok2]
    return o


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
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--split", type=int, default=0)
    ap_.add_argument("--pred", default="K562", choices=CELLS)
    ap_.add_argument("--tag", default="DeepSE_official", choices=TAGS)
    ap_.add_argument("--pool", default="K562", choices=CELLS)
    args = ap_.parse_args()
    base = BASE_MODEL[args.tag]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mats = load_kmer_matrices(KMER) if base in ("DeepSE", "MuSE") else None
    scaler = None
    if base == "DeepSE":
        tr = pd.read_csv(os.path.join(BASE, "exp3", f"data_split{args.split}",
                                      args.pred, "train.csv"), usecols=["sequence"])
        x2 = np.stack([seq_to_deepse_vec(str(s), mats) for s in tr["sequence"]])
        scaler = StandardScaler().fit(x2)
    model = get_model(base)
    ck = os.path.join(BASE, "exp3", f"ckpt_split{args.split}", args.pred,
                      args.tag, f"{base}_best_chrsplit.pth")
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model.to(device); model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    pool_dir = os.path.join(BASE, "exp3", f"pool_split{args.split}", args.pool)
    df = pd.read_csv(os.path.join(pool_dir, "EXP3_pool_test.csv"))
    y = (df.kind == "SE").to_numpy().astype(int)
    means = np.full(len(df), np.nan); maxs = np.full(len(df), np.nan)
    for j in range(len(df)):
        c = str(df.iloc[j]["chr"]); s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
        clen = len(fa[c])
        st = (np.asarray([(s + e) // 2 - W // 2]) if e - s <= W else
              np.arange(s, e - W + 1, STEP))
        if e - s > W and len(st) and st[-1] + W < e:
            st = np.append(st, e - W)
        st = st[(st >= 0) & (st + W <= clen)]
        bs, be = blacklist(c)
        st = st[~ov(bs, be, st, st + W)]
        if len(st) == 0:
            continue
        seqs = [str(fa[c][int(x):int(x + W)]).upper() for x in st]
        keep = [i for i, sq in enumerate(seqs) if sq.count("N") <= 0.5 * W]
        if not keep:
            continue
        p = infer(model, device, base, mats, scaler, [seqs[i] for i in keep])
        means[j] = p.mean(); maxs[j] = p.max()
    ok = np.isfinite(means)
    for name, sc in (("mean", means), ("max", maxs)):
        print(f'[{args.pred}->{args.pool} {args.tag} {name}] '
              f'n={int(ok.sum())} AUC={roc_auc_score(y[ok], sc[ok]):.4f} '
              f'AP={average_precision_score(y[ok], sc[ok]):.4f} '
              f'SEmean={sc[ok & (y == 1)].mean():.4f} '
              f'TEmean={sc[ok & (y == 0)].mean():.4f}')


if __name__ == "__main__":
    main()
