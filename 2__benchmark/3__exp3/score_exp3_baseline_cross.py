#!/usr/bin/env python3
"""EXP3 cross 打分（基线）：pred 细胞上训练的模型，在 4 个 GT 池的 test 集上
评测（3 kb 窗/步长 500 bp 局部滑窗 + 均值池化）。

输出（每个 GT pool 目录内）：
  EXP3_baseline_x_{tag}_{pred}_on_{pool}_test.npy
用法：python score_exp3_baseline_cross.py --split 0 --pred K562 --tag DcSE
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
BLACKLIST = "/public/home/liunangroup/liangyan/Genome/blacklist/hg38.blacklist.bed"
KMER_OFFICIAL = os.path.join(BDIR, "kmer_official_k3to8.npz")
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
TAGS = ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]
BASE_MODEL = {"DeepSE_official": "DeepSE", "SENet": "SENet",
              "MuSE_official": "MuSE", "DcSE": "DcSE"}
W = 3000
STEP = 500


def load_blacklist(chrom):
    rows = []
    for line in open(BLACKLIST):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] == chrom:
            rows.append((int(p[1]), int(p[2])))
    rows = sorted(rows)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ms, me = [], []
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            ms.append(cs)
            me.append(ce)
            cs, ce = s, e
    ms.append(cs)
    me.append(ce)
    return np.asarray(ms, np.int64), np.asarray(me, np.int64)


def overlaps(ms, me, s, e):
    if len(ms) == 0:
        return np.zeros(len(s), dtype=bool)
    idx = np.searchsorted(ms, s, side="left")
    out = np.zeros(len(s), dtype=bool)
    ok = idx < len(ms)
    out[ok] |= ms[idx[ok]] < e[ok]
    ok2 = idx > 0
    out[ok2] |= me[idx[ok2] - 1] > s[ok2]
    return out


def score_pool(df, fa, model, device, base, mats, scaler, batch_size=64):
    scores = np.full(len(df), np.nan, dtype=np.float32)
    starts_all, idx_all = [], []
    blc = {}
    for j in range(len(df)):
        c = str(df.iloc[j]["chr"])
        s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
        if c not in blc:
            blc[c] = load_blacklist(c)
        clen = len(fa[c])
        if e - s <= W:
            st = np.asarray([(s + e) // 2 - W // 2], dtype=np.int64)
        else:
            st = np.arange(s, e - W + 1, STEP, dtype=np.int64)
            if len(st) and st[-1] + W < e:
                st = np.append(st, e - W)
        st = st[(st >= 0) & (st + W <= clen)]
        bs, be = blc[c]
        st = st[~overlaps(bs, be, st, st + W)]
        if len(st) == 0:
            continue
        starts_all.append(st)
        idx_all.append(np.full(len(st), j, dtype=np.int32))
    if not starts_all:
        return scores
    starts = np.concatenate(starts_all)
    cidx = np.concatenate(idx_all)
    sums = np.zeros(len(df), dtype=np.float64)
    cnt = np.zeros(len(df), dtype=np.int32)
    with torch.no_grad():
        for i0 in range(0, len(starts), batch_size):
            seg_c = cidx[i0:i0 + batch_size]
            seqs = []
            for st, j in zip(starts[i0:i0 + batch_size], seg_c):
                c = str(df.iloc[j]["chr"])
                seqs.append(str(fa[c][int(st):int(st + W)]).upper())
            keep = [i for i, s in enumerate(seqs) if s.count("N") <= 0.5 * W]
            if not keep:
                continue
            good = [seqs[i] for i in keep]
            if base in ("DcSE", "SENet"):
                x = np.stack([seq_to_6mer_tokens(s) for s in good])
                out = model(torch.from_numpy(x).to(device))
            elif base == "DeepSE":
                x2 = scaler.transform(np.stack(
                    [seq_to_deepse_vec(s, mats) for s in good]))
                out = model(torch.from_numpy(x2.astype(np.float32)).to(device))
            else:
                x6 = np.stack([seq_to_6mer_tokens(s) for s in good])
                x2 = np.stack([seq_to_deepse_vec(s, mats).reshape(1, 300)
                               for s in good])
                x3 = np.stack([seq_to_onehot(s, alphabet="atcg") for s in good])
                logits = model(torch.from_numpy(x6).to(device),
                               torch.from_numpy(x2.astype(np.float32)).to(device),
                               torch.from_numpy(x3).to(device))
                out = torch.softmax(logits, dim=1)[:, 1]
            p = out.cpu().numpy()
            for k, oi in enumerate(keep):
                sums[seg_c[oi]] += float(p[k])
                cnt[seg_c[oi]] += 1
    ok = cnt > 0
    scores[ok] = (sums[ok] / cnt[ok]).astype(np.float32)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--pred", required=True, choices=CELLS)
    ap.add_argument("--tag", required=True, choices=TAGS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split, pred, tag = args.split, args.pred, args.tag
    base = BASE_MODEL[tag]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(BASE, "exp3", f"data_split{split}", pred)
    ckpt = os.path.join(BASE, "exp3", f"ckpt_split{split}", pred, tag,
                        f"{base}_best_chrsplit.pth")
    mats = load_kmer_matrices(KMER_OFFICIAL) if base in ("DeepSE", "MuSE") else None
    scaler = None
    if base == "DeepSE":
        tr = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["sequence"])
        x2 = np.stack([seq_to_deepse_vec(str(s), mats) for s in tr["sequence"]])
        scaler = StandardScaler().fit(x2)
    model = get_model(base)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    for pool in CELLS:
        pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        df = pd.read_csv(os.path.join(pdir, "EXP3_pool_test.csv"))
        sc = score_pool(df, fa, model, device, base, mats, scaler)
        out = os.path.join(pdir, f"EXP3_baseline_x_{tag}_{pred}_on_{pool}_test.npy")
        np.save(out, sc)
        print(f"[save] {out} mean={np.nanmean(sc):.4f} "
              f"scored={int(np.isfinite(sc).sum())}/{len(sc)}")


if __name__ == "__main__":
    main()
