#!/usr/bin/env python3
"""EXP3 S7c1：Cerberus 谱系单通道簇级打分。

每簇：簇中心 524 kb 真实上下文 forward，定位簇覆盖 bins，
取 channel c 在该区间上的空间均值（不跨通道 max）。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyfaidx
import torch
from scipy.ndimage import gaussian_filter1d

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
CER_DIR = os.path.join(REPO, "Cerberus")
sys.path.insert(0, CER_DIR)
from se_data_utils import dna_one_hot  # noqa: E402
from se_model_arch import BorzoiSEAdapter  # noqa: E402

INPUT_LEN = 524288
HALF = INPUT_LEN // 2
OUT_BINS = 1600
RES = 256
BIN0_BP = 57344
FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
HF_MODEL = "/public/home/liunangroup/liangyan/project/SE_model/pretrained_models/borzoi-replicate-0"
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
CELL_CH = {"K562": 31, "HepG2": 35, "GM12878": 30, "WTC11": 2}
CKPT = {0: os.path.join(REPO, "test10/Cerberus_human_best.pth")}
for s in (1, 2, 3, 4):
    CKPT[s] = os.path.join(
        REPO, f"test10_eval_new/20260907.multi_chr_split__benchmark/"
              f"cerberus_split{s}/finetune/Cerberus_human_best.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split, cell = args.split, args.cell
    ch = CELL_CH[cell]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    use_bf16 = device.type == "cuda" and cap[0] >= 8
    model = BorzoiSEAdapter(HF_MODEL, 90)
    model.load_state_dict(torch.load(CKPT[split], map_location="cpu",
                                     weights_only=True))
    model.to(device)
    model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    pool_dir = os.path.join(BASE, "exp3", f"pool_split{split}", cell)
    meta = {"split": split, "cell": cell, "channel_idx": ch,
            "ckpt": CKPT[split]}
    for tag in ("val", "test"):
        df = pd.read_csv(os.path.join(pool_dir, f"EXP3_pool_{tag}.csv"))
        scores = np.zeros(len(df), dtype=np.float32)
        with torch.no_grad():
            for i0 in range(0, len(df), args.batch_size):
                encs, binss = [], []
                for j in range(i0, min(i0 + args.batch_size, len(df))):
                    c = str(df.iloc[j]["chr"])
                    s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
                    ln = len(fa[c])
                    lo, hi = HALF, ln - HALF
                    sc = int(min(max((s + e) // 2, lo), hi))
                    ws, we = sc - HALF, sc + HALF
                    seq = fa[c][ws:we].upper()
                    encs.append(dna_one_hot(np.frombuffer(
                        seq.encode("utf-8"), dtype=np.int8)))
                    g0 = sc - HALF + BIN0_BP
                    b0 = max(0, min(int(np.floor((s - g0) / RES)), OUT_BINS - 1))
                    b1 = min(OUT_BINS, max(b0 + 1,
                                           int(np.ceil((e - g0) / RES))))
                    binss.append((b0, b1))
                chunk = torch.stack(encs).to(device)
                if use_bf16:
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16):
                        cl, sl = model(chunk)
                else:
                    cl, sl = model(chunk)
                joint = (torch.sigmoid(cl.float()) *
                         torch.sigmoid(sl.float())).cpu().numpy()
                joint = gaussian_filter1d(joint, sigma=1.0, axis=-1)
                for k, (b0, b1) in enumerate(binss):
                    scores[i0 + k] = float(joint[k, ch, b0:b1].mean())
        p = os.path.join(pool_dir, f"EXP3_cerberus_{tag}.npy")
        np.save(p, scores)
        meta[tag] = {"n": len(df), "mean": float(scores.mean())}
        print(f"[{tag}] n={len(df)} mean={scores.mean():.4f} -> {p}")
    with open(os.path.join(pool_dir, "EXP3_cerberus_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
