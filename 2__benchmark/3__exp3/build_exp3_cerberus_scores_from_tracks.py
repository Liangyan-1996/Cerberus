#!/usr/bin/env python3
"""EXP3 Cerberus cluster 级打分（零推理，读 Exp2 全染色体稠密轨道）：
  own 通道 val+test；test 另存 4 pred 通道（交叉热图）。
用法：python build_exp3_cerberus_scores_from_tracks.py --split 0
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


def score_units(df, joint):
    """返回 dict: pred cell -> 每个 unit 的通道均值（簇/peak 覆盖 bins）。"""
    out = {c: np.full(len(df), np.nan, dtype=np.float32) for c in CELLS}
    for i in range(len(df)):
        c = str(df.iloc[i]["chr"])
        if c not in joint:
            continue
        b0 = int(df.iloc[i]["start"]) // RES
        b1 = int(np.ceil(int(df.iloc[i]["end"]) / RES))
        seg = joint[c][:, b0:b1]
        for pred in CELLS:
            out[pred][i] = float(seg[CELL_CH[pred]].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    val_chrs, test_chrs = SPLIT_CHRS[split]
    tdir = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    vdir = tdir if split == 0 else os.path.join(
        BASE, f"assets/tracks_val_split{split}/tracks")
    # 分 tag 载入（val/test 染色体不同）
    cache = {}
    for tag, chrs, d in (("test", test_chrs, tdir), ("val", val_chrs, vdir)):
        cache[tag] = {}
        for c in chrs:
            p = os.path.join(d, f"cerberus.{c}.joint.npy")
            cache[tag][c] = np.load(p)
            print(f"[load {tag}] {p} {cache[tag][c].shape}")
    for pool in CELLS:
        pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        for tag in ("val", "test"):
            df = pd.read_csv(os.path.join(pdir, f"EXP3_pool_{tag}.csv"))
            sc = score_units(df, cache[tag])
            own = sc[pool]
            np.save(os.path.join(pdir, f"EXP3_cerberus_{tag}.npy"), own)
            if tag == "test":
                for pred in CELLS:
                    np.save(os.path.join(
                        pdir, f"EXP3_cerberus_x_{pred}_test.npy"), sc[pred])
            print(f"[cluster {pool} {tag}] own mean={np.nanmean(own):.4f}")
    print("[Done] Cerberus cluster scores from tracks (val+test)")


if __name__ == "__main__":
    main()
