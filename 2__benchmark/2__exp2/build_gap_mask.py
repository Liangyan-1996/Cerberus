#!/usr/bin/env python3
"""EXP2 S6a：从 GRCh38 primary assembly 的连续 N 段生成 256-bp gap mask。

输出：
  exp2/masks/gap_mask.{chr}.npy        (int8, n_bins=ceil(len/256))
  exp2/masks/gap_runs.tsv              (chr,start,end)
  exp2/masks/gap_stats.json
染色体：chr1-22（5-split 用到的全部 autosome）。
"""
import json
import os

import numpy as np
import pyfaidx

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
RES = 256
CHROMS = [f"chr{i}" for i in range(1, 23)]


def runs_of_n(seq):
    """连续 N 段（0-based half-open）。"""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    nidx = np.flatnonzero(arr == ord("N"))
    if len(nidx) == 0:
        return []
    out = []
    s0 = p0 = int(nidx[0])
    for x in nidx[1:]:
        if x == p0 + 1:
            p0 = int(x)
        else:
            out.append((s0, p0 + 1))
            s0 = p0 = int(x)
    out.append((s0, p0 + 1))
    return out


def main():
    out_dir = os.path.join(BASE, "exp2", "masks")
    os.makedirs(out_dir, exist_ok=True)
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    stats = {}
    rows = []
    for c in CHROMS:
        seq = fa[c][:].upper()
        runs = runs_of_n(seq)
        nbins = int(np.ceil(len(seq) / RES))
        mask = np.zeros(nbins, dtype=np.int8)
        n_bp = 0
        for a, b in runs:
            n_bp += b - a
            mask[a // RES: (b + RES - 1) // RES] = 1
            rows.append((c, int(a), int(b)))
        np.save(os.path.join(out_dir, f"gap_mask.{c}.npy"), mask)
        stats[c] = {"len_bp": int(len(seq)), "gap_bp": int(n_bp),
                    "frac": round(n_bp / len(seq), 5),
                    "n_runs": len(runs),
                    "gap_mb": round(n_bp / 1e6, 2)}
        print(f"[{c}] len={len(seq)/1e6:.1f}Mb gap={n_bp/1e6:.2f}Mb "
              f"({stats[c]['frac']*100:.2f}%) runs={len(runs)}")
    with open(os.path.join(out_dir, "gap_runs.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(out_dir, "gap_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Done] gap masks -> {out_dir}")


if __name__ == "__main__":
    main()
