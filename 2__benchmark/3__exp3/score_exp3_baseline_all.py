#!/usr/bin/env python3
"""EXP3 基线 cluster 级打分（修正标签后）：
  own pool val+test（3 kb/500 bp 滑窗均值池化）+ 4 池 test（cross）。
模型只载入一次。输出沿用既有命名，供指标/热图脚本直接读取。
用法：python score_exp3_baseline_all.py --split 0 --pred K562 --tag DcSE
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
KMER = os.path.join(BDIR, "kmer_official_k3to8.npz")
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
TAGS = ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]
BASE_MODEL = {"DeepSE_official": "DeepSE", "SENet": "SENet",
              "MuSE_official": "MuSE", "DcSE": "DcSE"}
W, STEP = 3000, 500


def load_blacklist(chrom):
    rows = sorted((int(p[1]), int(p[2]))
                  for p in (l.rstrip().split("\t") for l in open(BLACKLIST))
                  if len(p) >= 3 and p[0] == chrom)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ms, me = [], []; cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            ms.append(cs); me.append(ce); cs, ce = s, e
    ms.append(cs); me.append(ce)
    return np.asarray(ms, np.int64), np.asarray(me, np.int64)


def overlaps(ms, me, s, e):
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


def score_cluster(df, fa, model, device, base, mats, scaler):
    """3 kb/500 bp 滑窗 -> 均值池化。"""
    scores = np.full(len(df), np.nan, dtype=np.float32)
    st_all, ci_all, blc = [], [], {}
    for j in range(len(df)):
        c = str(df.iloc[j]["chr"])
        s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
        if c not in blc:
            blc[c] = load_blacklist(c)
        clen = len(fa[c])
        st = (np.asarray([(s + e) // 2 - W // 2], np.int64) if e - s <= W else
              np.arange(s, e - W + 1, STEP, dtype=np.int64))
        if e - s > W and len(st) and st[-1] + W < e:
            st = np.append(st, e - W)
        st = st[(st >= 0) & (st + W <= clen)]
        bs, be = blc[c]
        st = st[~overlaps(bs, be, st, st + W)]
        if len(st):
            st_all.append(st); ci_all.append(np.full(len(st), j, np.int32))
    if not st_all:
        return scores
    st = np.concatenate(st_all); ci = np.concatenate(ci_all)
    sums = np.zeros(len(df)); cnt = np.zeros(len(df), np.int32)
    for i0 in range(0, len(st), 256):
        seg = ci[i0:i0 + 256]
        seqs = []
        for x, j in zip(st[i0:i0 + 256], seg):
            c = str(df.iloc[j]["chr"])
            seqs.append(str(fa[c][int(x):int(x + W)]).upper())
        keep = [i for i, sq in enumerate(seqs) if sq.count("N") <= 0.5 * W]
        if not keep:
            continue
        p = infer(model, device, base, mats, scaler, [seqs[i] for i in keep])
        for k, oi in enumerate(keep):
            sums[seg[oi]] += float(p[k]); cnt[seg[oi]] += 1
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
        cdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        for ttag in ("val", "test"):
            if ttag == "val" and pool != pred:
                continue  # val 只需 own pool（F1 阈值）
            df = pd.read_csv(os.path.join(cdir, f"EXP3_pool_{ttag}.csv"))
            sc = score_cluster(df, fa, model, device, base, mats, scaler)
            if pool == pred:  # own pool 命名（主表用）
                np.save(os.path.join(cdir, f"EXP3_baseline_{tag}_{ttag}.npy"), sc)
            if ttag == "test":
                np.save(os.path.join(cdir, f"EXP3_baseline_x_{tag}_{pred}"
                                           f"_on_{pool}_test.npy"), sc)
            print(f"[cluster {pred}->{pool} {ttag}] scored="
                  f"{int(np.isfinite(sc).sum())}/{len(sc)}")
    print("[Done] EXP3 baseline cluster scores")


if __name__ == "__main__":
    main()
