#!/usr/bin/env python3
"""EXP1 S3：基线簇级打分（局部滑窗 3 kb/1 kb + 均值池化；D2/D9，零重训）。

对 pool 中每条簇：
  - 簇长 <= 3 kb：center ± 1500 单窗；
  - 簇长 > 3 kb：窗起点 = 簇起点 + k*1000（末窗补 cluster_end-3000，保证覆盖）；
  - 黑名单重叠或 >50% N 的窗剔除；
  - P_cluster = 有效窗分数均值。

用法：python score_baseline_clusters.py --split 0 --tag DeepSE_official
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.preprocessing import StandardScaler

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
BDIR = os.path.join(REPO, "test10_eval_new/20260905.model_benchmark/benchmark")
sys.path.insert(0, BDIR)

from features import (  # noqa: E402
    load_kmer_matrices,
    seq_to_6mer_tokens,
    seq_to_deepse_vec,
    seq_to_onehot,
)
from train_baseline import get_model  # noqa: E402

FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
BLACKLIST = "/public/home/liunangroup/liangyan/Genome/blacklist/hg38.blacklist.bed"
KMER_OFFICIAL = os.path.join(BDIR, "kmer_official_k3to8.npz")
BASE_MODEL = {
    "DeepSE_official": "DeepSE",
    "SENet": "SENet",
    "MuSE_official": "MuSE",
    "DcSE": "DcSE",
}
W = 3000
STEP = 1000
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}
FRAG_DATA = {
    0: os.path.join(BASE, "assets/fragment_data_split0/data"),
}
for s in (1, 2, 3, 4):
    FRAG_DATA[s] = os.path.join(
        BASE, f"assets/fragment_data_split{s}/data")


def load_blacklist(chrom):
    rows = []
    for line in open(BLACKLIST):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] == chrom:
            rows.append((int(p[1]), int(p[2])))
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rows = sorted(rows)
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


def cluster_windows(s, e, clen):
    """簇内 3 kb 窗起点（覆盖整簇；不足 3 kb 取中心单窗）。"""
    if e - s <= W:
        c = (s + e) // 2
        st = c - W // 2
        if 0 <= st and st + W <= clen:
            return np.asarray([st], dtype=np.int64)
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(s, e - W + 1, STEP, dtype=np.int64)
    if len(starts) and starts[-1] + W < e:
        starts = np.append(starts, e - W)
    if len(starts) and starts[-1] + W > clen:
        starts = starts[starts + W <= clen]
    return starts


def fit_deepse_scaler(train_csv, mats):
    df = pd.read_csv(train_csv, usecols=["sequence"])
    x2 = np.stack([seq_to_deepse_vec(str(s), mats) for s in df["sequence"]])
    return StandardScaler().fit(x2)


def score_pool(df, fa, model, device, batch_size, base, mats, scaler, tag):
    t0 = time.time()
    scores = np.full(len(df), np.nan, dtype=np.float32)
    # 预取所有有效窗
    all_st, all_cidx = [], []
    bls_cache = {}
    for j in range(len(df)):
        c = str(df.iloc[j]["chr"])
        s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
        if c not in bls_cache:
            bls_cache[c] = load_blacklist(c)
        clen = len(fa[c])
        starts = cluster_windows(s, e, clen)
        if len(starts) == 0:
            continue
        bs, be = bls_cache[c]
        bad = overlaps(bs, be, starts, starts + W)
        starts = starts[~bad]
        if len(starts) == 0:
            continue
        all_cidx.append(np.full(len(starts), j, dtype=np.int32))
        all_st.append(starts)
    if not all_st:
        return scores
    starts = np.concatenate(all_st)
    cidx = np.concatenate(all_cidx)
    n_win = len(starts)
    sums = np.zeros(len(df), dtype=np.float64)
    cnt = np.zeros(len(df), dtype=np.int32)
    t1 = time.time()
    print(f"[{tag}] total windows={n_win} (collect {t1-t0:.0f}s)")
    with torch.no_grad():
        for i0 in range(0, n_win, batch_size):
            seg_st = starts[i0:i0 + batch_size]
            seg_cidx = cidx[i0:i0 + batch_size]
            seqs = []
            for st, j in zip(seg_st, seg_cidx):
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
                x2 = scaler.transform(
                    np.stack([seq_to_deepse_vec(s, mats) for s in good]))
                out = model(torch.from_numpy(x2.astype(np.float32)).to(device))
            else:  # MuSE
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
                j = seg_cidx[oi]
                sums[j] += float(p[k])
                cnt[j] += 1
            if ((i0 // batch_size) % 100) == 0:
                print(f"  ... {i0}/{n_win} ({time.time()-t1:.0f}s)")
    ok = cnt > 0
    scores[ok] = (sums[ok] / cnt[ok]).astype(np.float32)
    print(f"[{tag}] scored {int(ok.sum())}/{len(df)} clusters "
          f"({time.time()-t0:.0f}s)")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--tag", required=True, choices=list(BASE_MODEL))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split, tag = args.split, args.tag
    val_chrs, test_chrs = SPLIT_CHRS[split]
    base = BASE_MODEL[tag]
    split_dir = os.path.join(BASE, "exp1", f"split{split}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] split{split} {tag} ({base}) device={device}")

    mats = None
    scaler = None
    if base in ("DeepSE", "MuSE"):
        mats = load_kmer_matrices(KMER_OFFICIAL)
    if base == "DeepSE":
        train_csv = os.path.join(FRAG_DATA[split], "train.csv")
        t0 = time.time()
        scaler = fit_deepse_scaler(train_csv, mats)
        print(f"[DeepSE] scaler fit ({time.time()-t0:.0f}s)")
    ckpt_dir = os.path.join(BASE, f"assets/baseline_ckpt_split{split}/ckpt")
    ckpt = os.path.join(ckpt_dir, tag, f"{base}_best_chrsplit.pth")
    model = get_model(base)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)

    for ttag, chrs_ in (("val", val_chrs), ("test", test_chrs)):
        pool_path = os.path.join(split_dir, f"EXP1_pool_{ttag}.csv")
        df = pd.read_csv(pool_path)
        df = df[df["chr"].isin(chrs_)].reset_index(drop=True)
        scores = score_pool(df, fa, model, device, args.batch_size,
                            base, mats, scaler, f"{tag}.{ttag}")
        out_p = os.path.join(split_dir, f"EXP1_baseline_{tag}_{ttag}.npy")
        np.save(out_p, scores)
        print(f"[save] {ttag}: n={len(df)} scored={int(np.isfinite(scores).sum())} "
              f"-> {out_p}")
    print("[Done] EXP1 baseline cluster scores")


if __name__ == "__main__":
    main()
