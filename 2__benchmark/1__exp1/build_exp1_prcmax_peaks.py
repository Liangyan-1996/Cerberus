#!/usr/bin/env python3
"""EXP1 peak-restricted+max：为每个 union 簇块（SE 块 / TE-only 块）找出
块内构成峰（90 cell 的 MACS2 peaks，去重）。

输出：exp1/split{s}/EXP1_prcmax_peaks_{val,test}.npz
  cidx(块行号), p_start/p_end/p_center
用法：python build_exp1_prcmax_peaks.py --split 0
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
PEAKS_ROOT = os.path.join(BASE, "assets/se_pipeline/peaks")
META = os.path.join(BASE, "assets/se_pipeline/metadata.pkl")
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    val_chrs, test_chrs = SPLIT_CHRS[split]
    chroms = val_chrs + test_chrs
    with open(META, "rb") as f:
        meta = pickle.load(f)
    names = sorted(meta)
    # 收集 90 cell 在 val/test 染色体的所有峰
    starts, ends = {c: [] for c in chroms}, {c: [] for c in chroms}
    for name in names:
        p = os.path.join(PEAKS_ROOT, f"{name}_peaks.bed")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"])
        df = df[df["chrom"].isin(chroms)]
        for c, g in df.groupby("chrom"):
            starts[c].append(g["start"].to_numpy(np.int64))
            ends[c].append(g["end"].to_numpy(np.int64))
    pk = {}
    for c in chroms:
        if starts[c]:
            s = np.concatenate(starts[c]); e = np.concatenate(ends[c])
            order = np.argsort(s, kind="stable")
            pk[c] = (s[order], e[order])
        else:
            pk[c] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        print(f"[peaks] {c}: {len(pk[c][0])}")
    for tag, chrs in (("val", val_chrs), ("test", test_chrs)):
        d = pd.read_csv(os.path.join(BASE, "exp1", f"split{split}",
                                     f"EXP1_pool_{tag}.csv"))
        cidx, ps, pe = [], [], []
        n_zero = 0
        for i in range(len(d)):
            c = str(d.iloc[i]["chr"]); s, e = int(d.iloc[i]["start"]), int(d.iloc[i]["end"])
            ms, me = pk[c]
            if len(ms) == 0:
                n_zero += 1; continue
            lo = int(np.searchsorted(ms, s, side="right") - 1); lo = max(0, lo)
            hit = []
            for j in range(lo, len(ms)):
                if ms[j] >= e:
                    break
                if me[j] > s:
                    hit.append(j)
            if not hit:
                n_zero += 1; continue
            sub = np.stack([ms[hit], me[hit]], axis=1)
            uniq = np.unique(sub, axis=0)   # 去重（不同 cell 的同坐标峰）
            for a, b in uniq:
                cidx.append(i); ps.append(int(a)); pe.append(int(b))
        out = os.path.join(BASE, "exp1", f"split{split}",
                           f"EXP1_prcmax_peaks_{tag}.npz")
        np.savez(out, cidx=np.asarray(cidx, np.int32),
                 p_start=np.asarray(ps, np.int64), p_end=np.asarray(pe, np.int64),
                 p_center=((np.asarray(ps, np.int64) + np.asarray(pe, np.int64)) // 2))
        print(f"[{tag}] blocks={len(d)} peak-links={len(cidx)} "
              f"blocks_without_peak={n_zero} -> {out}")


if __name__ == "__main__":
    main()
