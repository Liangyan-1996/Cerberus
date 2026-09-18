#!/usr/bin/env python3
"""EXP3 peak-restricted cluster：Cerberus 簇分数 = 构成峰的通道 bins 均值（峰间再平均）。
读 Exp2 全染色体稠密轨道（零新推理）。
输出：pool_split{s}/{cell}/EXP3_prc_cerberus_{val,test}.npy
      pool_split{s}/{cell}/EXP3_prc_cerberus_x_{pred}_test.npy
用法：python build_exp3_cerberus_prc_from_tracks.py --split 0
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
CELL_CH = {"K562": 31, "HepG2": 35, "GM12878": 30, "WTC11": 2}
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
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    args = ap.parse_args()
    split = args.split
    PRE = "EXP3_prc" if args.agg == "mean" else "EXP3_prcmax"
    val_chrs, test_chrs = SPLIT_CHRS[split]
    tdir = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    vdir = tdir if split == 0 else os.path.join(
        BASE, f"assets/tracks_val_split{split}/tracks")
    cache = {}
    for tag, chrs, d in (("test", test_chrs, tdir), ("val", val_chrs, vdir)):
        cache[tag] = {c: np.load(os.path.join(d, f"cerberus.{c}.joint.npy"))
                      for c in chrs}
        print(f"[load {tag}] {list(cache[tag])}")
    for pool in CELLS:
        pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        for tag in ("val", "test"):
            df = pd.read_csv(os.path.join(pdir, f"EXP3_pool_{tag}.csv"))
            npz = np.load(os.path.join(pdir, f"EXP3_prc_peaks_{tag}.npz"))
            cidx, ps, pe = npz["cidx"], npz["p_start"], npz["p_end"]
            sums = {c: np.zeros(len(df)) for c in CELLS}
            mx = {c: np.full(len(df), -np.inf) for c in CELLS}
            cnt = np.zeros(len(df), np.int32)
            for k in range(len(cidx)):
                c = str(df.iloc[int(cidx[k])]["chr"])
                if c not in cache[tag]:
                    continue
                b0 = int(ps[k]) // RES
                b1 = int(np.ceil(int(pe[k]) / RES))
                seg = cache[tag][c][:, b0:b1]
                for pred in CELLS:
                    v = float(seg[CELL_CH[pred]].mean())
                    sums[pred][cidx[k]] += v
                    mx[pred][cidx[k]] = max(mx[pred][cidx[k]], v)
                cnt[cidx[k]] += 1
            ok = cnt > 0
            sc = {c: np.full(len(df), np.nan, np.float32) for c in CELLS}
            for c in CELLS:
                sc[c][ok] = ((sums[c][ok] / cnt[ok]) if args.agg == "mean"
                             else mx[c][ok]).astype(np.float32)
            np.save(os.path.join(pdir, f"{PRE}_cerberus_{tag}.npy"), sc[pool])
            if tag == "test":
                for pred in CELLS:
                    np.save(os.path.join(
                        pdir, f"{PRE}_cerberus_x_{pred}_test.npy"), sc[pred])
            print(f"[prc {pool} {tag}] own mean={np.nanmean(sc[pool]):.4f} "
                  f"scored={int(ok.sum())}/{len(df)}")
    print("[Done] Cerberus peak-restricted cluster scores")


if __name__ == "__main__":
    main()
