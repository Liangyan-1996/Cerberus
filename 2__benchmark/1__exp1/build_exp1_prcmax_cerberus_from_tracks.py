#!/usr/bin/env python3
"""EXP1 peak-restricted+max Cerberus：块分数 = 块内构成峰的
max_c(通道 c 在峰 bins 的均值) 再对峰取 max；读 Exp2 稠密轨道（零推理）。
输出：exp1/split{s}/EXP1_prcmax_cerberus_{val,test}.npy
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
RES = 256
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
    tdir = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    vdir = tdir if split == 0 else os.path.join(
        BASE, f"assets/tracks_val_split{split}/tracks")
    cache = {}
    for tag, chrs, d in (("test", test_chrs, tdir), ("val", val_chrs, vdir)):
        cache[tag] = {c: np.load(os.path.join(d, f"cerberus.{c}.joint.npy"))
                      for c in chrs}
        print(f"[load {tag}] {list(cache[tag])}")
    for tag in ("val", "test"):
        npz = np.load(os.path.join(BASE, "exp1", f"split{split}",
                                   f"EXP1_prcmax_peaks_{tag}.npz"))
        cidx, ps, pe = npz["cidx"], npz["p_start"], npz["p_end"]
        df = pd.read_csv(os.path.join(BASE, "exp1", f"split{split}",
                                      f"EXP1_pool_{tag}.csv"))
        mx = np.full(len(df), -np.inf, dtype=np.float64)
        for k in range(len(cidx)):
            c = str(df.iloc[int(cidx[k])]["chr"])
            if c not in cache[tag]:
                continue
            b0 = int(ps[k]) // RES
            b1 = int(np.ceil(int(pe[k]) / RES))
            if b1 <= b0:
                continue
            seg = cache[tag][c][:, b0:b1]
            peak_score = float(seg.mean(axis=1).max())   # 90 通道取 max
            mx[cidx[k]] = max(mx[cidx[k]], peak_score)
        scores = np.full(len(df), np.nan, dtype=np.float32)
        ok = mx > -np.inf
        scores[ok] = mx[ok].astype(np.float32)
        out = os.path.join(BASE, "exp1", f"split{split}",
                           f"EXP1_prcmax_cerberus_{tag}.npy")
        np.save(out, scores)
        print(f"[{tag}] scored={int(ok.sum())}/{len(df)} -> {out}")
    print("[Done] EXP1 peak-restricted max Cerberus scores")


if __name__ == "__main__":
    main()
