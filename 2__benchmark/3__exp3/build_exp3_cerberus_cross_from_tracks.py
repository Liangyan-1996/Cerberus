#!/usr/bin/env python3
"""EXP3 Cerberus cross（零新推理）：从既有 90 通道全染色体稠密轨道读
通道 c 在 4 个 GT 池 test 簇上的空间均值。

输出：exp3/pool_split{s}/{pool}/EXP3_cerberus_x_{pred_cell}_test.npy
（每个 split 16 个文件 = 4 pred 通道 × 4 GT 池；与本目录旧 forward 版命名一致）
用法：python build_exp3_cerberus_cross_from_tracks.py --split 0
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
TEST_CHRS = {
    0: ["chr3", "chr11"],
    1: ["chr2", "chr12"],
    2: ["chr5", "chr17"],
    3: ["chr7", "chr20"],
    4: ["chr10", "chr13"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    track_dir = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    chrs = TEST_CHRS[split]
    joint = {}
    for c in chrs:
        p = os.path.join(track_dir, f"cerberus.{c}.joint.npy")
        joint[c] = np.load(p)
        print(f"[load] {p} shape={joint[c].shape}")
    for pool in CELLS:
        pdir = os.path.join(BASE, "exp3", f"pool_split{split}", pool)
        df = pd.read_csv(os.path.join(pdir, "EXP3_pool_test.csv"))
        scores = {c: np.zeros(len(df), dtype=np.float32) for c in CELLS}
        for i in range(len(df)):
            c = str(df.iloc[i]["chr"])
            if c not in joint:
                continue
            b0 = int(df.iloc[i]["start"]) // RES
            b1 = int(np.ceil(int(df.iloc[i]["end"]) / RES))
            seg = joint[c][:, b0:b1]
            for pred in CELLS:
                scores[pred][i] = float(seg[CELL_CH[pred]].mean())
        for pred in CELLS:
            p = os.path.join(pdir, f"EXP3_cerberus_x_{pred}_test.npy")
            np.save(p, scores[pred])
            print(f"[save] {p} mean={scores[pred].mean():.4f}")
    print("[Done] Cerberus cross from dense tracks")


if __name__ == "__main__":
    main()
