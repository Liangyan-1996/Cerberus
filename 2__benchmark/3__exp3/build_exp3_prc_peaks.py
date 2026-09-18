#!/usr/bin/env python3
"""EXP3 peak-restricted cluster：为每个 ROSE 簇行找出其构成 MACS2 峰。

输出（每 split × cell）：exp3/pool_split{s}/{cell}/EXP3_prc_peaks_{val,test}.npz
  cidx( int32, 簇在 pool csv 中的行号 ), p_start/p_end/p_center(int64)
用法：python build_exp3_prc_peaks.py --split 0 --cell K562
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
PEAKS_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/peaks"
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
CELL_NAMES = {
    "K562": "hg38.SE.ENCSR000AKP.K562",
    "HepG2": "hg38.SE.ENCSR000AMO.HepG2",
    "GM12878": "hg38.SE.ENCSR000AKC.GM12878",
    "WTC11": "hg38.PE.ENCSR146DPQ.WTC11",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--cell", required=True, choices=CELLS)
    args = ap.parse_args()
    split, cell = args.split, args.cell
    pk = pd.read_csv(os.path.join(PEAKS_ROOT, f"{CELL_NAMES[cell]}_peaks.bed"),
                     sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    pk_by_chr = {c: g.sort_values("start").reset_index(drop=True)
                 for c, g in pk.groupby("chrom")}
    pool_dir = os.path.join(BASE, "exp3", f"pool_split{split}", cell)
    for tag in ("val", "test"):
        d = pd.read_csv(os.path.join(pool_dir, f"EXP3_pool_{tag}.csv"))
        cidx, ps, pe = [], [], []
        n_zero = 0
        for i in range(len(d)):
            c = str(d.iloc[i]["chr"]); s, e = int(d.iloc[i]["start"]), int(d.iloc[i]["end"])
            g = pk_by_chr.get(c)
            if g is None or len(g) == 0:
                n_zero += 1
                continue
            starts = g["start"].to_numpy(np.int64)
            ends = g["end"].to_numpy(np.int64)
            lo = int(np.searchsorted(starts, s, side="right") - 1)
            lo = max(0, lo)
            hit = []
            for j in range(lo, len(g)):
                if starts[j] >= e:
                    break
                if ends[j] > s:
                    hit.append(j)
            if not hit:
                n_zero += 1
                continue
            for j in hit:
                cidx.append(i); ps.append(int(starts[j])); pe.append(int(ends[j]))
        out = os.path.join(pool_dir, f"EXP3_prc_peaks_{tag}.npz")
        np.savez(out, cidx=np.asarray(cidx, np.int32),
                 p_start=np.asarray(ps, np.int64),
                 p_end=np.asarray(pe, np.int64),
                 p_center=((np.asarray(ps, np.int64) + np.asarray(pe, np.int64)) // 2))
        n_cl = len(d)
        print(f"[{tag}] clusters={n_cl} peak-links={len(cidx)} "
              f"clusters_without_peak={n_zero} -> {out}")


if __name__ == "__main__":
    main()
