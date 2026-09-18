#!/usr/bin/env python3
"""EXP1 S2：Cerberus 簇级打分（D2 规范）。

对 pool 中每条 ROSE 缝合簇：
  1) center=(start+end)/2，safety shift clamp 到 [262144, len-262144]；
  2) 取真实 524 kb 上下文 forward，joint=sigmoid(cluster)*sigmoid(se)，
     输出网格起点 = 窗口起点 + 57,344 bp（224 bins*256），高斯平滑 sigma=1；
  3) 定位簇覆盖的全部 256-bp bins 与输出网格交集；
  4) 逐 channel 对簇覆盖 bins 求空间均值，再跨 90 channel 取 max
     （= mean-then-max，勿与旧 fragment 的 max-then-mean 混用）。

用法：python score_cerberus_clusters.py --split 0
"""
import argparse
import json
import os
import sys
import time

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
BIN0_BP = (INPUT_LEN - OUT_BINS * RES) // 2  # 57344
FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
HF_MODEL = "/public/home/liunangroup/liangyan/project/SE_model/pretrained_models/borzoi-replicate-0"
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}
CKPT = {
    0: os.path.join(REPO, "test10/Cerberus_human_best.pth"),
}
for s in (1, 2, 3, 4):
    CKPT[s] = os.path.join(
        REPO, f"test10_eval_new/20260907.multi_chr_split__benchmark/"
              f"cerberus_split{s}/finetune/Cerberus_human_best.pth")


def safe_center(center, chrom_len):
    lo, hi = HALF, chrom_len - HALF
    if hi < lo:
        lo = hi = max(0, min(center, chrom_len))
    return int(min(max(center, lo), hi))


def cluster_bins(s, e, sc):
    """簇在输出 1600-bin 网格内的 bin 区间（网格起点 = sc-HALF+BIN0_BP）。"""
    g0 = sc - HALF + BIN0_BP
    b0 = int(np.floor((s - g0) / RES))
    b1 = int(np.ceil((e - g0) / RES))
    b0 = max(0, min(b0, OUT_BINS - 1))
    b1 = min(OUT_BINS, max(b1, b0 + 1))
    return b0, b1


def score_pool(df, fa, model, device, batch_size, use_bf16, tag):
    n = len(df)
    scores = np.full(n, np.nan, dtype=np.float32)
    cov = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            encs, binss, rows = [], [], []
            for j in range(i0, min(i0 + batch_size, n)):
                c = str(df.iloc[j]["chr"])
                s, e = int(df.iloc[j]["start"]), int(df.iloc[j]["end"])
                ln = len(fa[c])
                sc = safe_center((s + e) // 2, ln)
                ws, we = sc - HALF, sc + HALF
                seq = fa[c][ws:we].upper()
                encs.append(dna_one_hot(
                    np.frombuffer(seq.encode("utf-8"), dtype=np.int8)))
                b0, b1 = cluster_bins(s, e, sc)
                g0 = sc - HALF + BIN0_BP
                covered = (min(e, g0 + b1 * RES) - max(s, g0 + b0 * RES))
                cov[j] = max(0.0, covered) / max(1, e - s)
                binss.append((b0, b1))
                rows.append(j)
            chunk = torch.stack(encs).to(device)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    cl, sl = model(chunk)
            else:
                cl, sl = model(chunk)
            joint = (torch.sigmoid(cl.float()) *
                     torch.sigmoid(sl.float())).cpu().numpy()
            joint = gaussian_filter1d(joint, sigma=1.0, axis=-1)
            for k, j in enumerate(rows):
                b0, b1 = binss[k]
                seg = joint[k, :, b0:b1]          # (90, T)
                if seg.shape[1] > 0:
                    scores[j] = float(seg.mean(axis=1).max())  # mean-then-max
            if (i0 // batch_size) % 20 == 0:
                print(f"  [{tag}] {i0}/{n} ({time.time()-t0:.0f}s)")
    return scores, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split = args.split
    val_chrs, test_chrs = SPLIT_CHRS[split]
    split_dir = os.path.join(BASE, "exp1", f"split{split}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    use_bf16 = device.type == "cuda" and cap[0] >= 8
    print(f"[Info] split{split} device={device} bf16={use_bf16}")

    model = BorzoiSEAdapter(HF_MODEL, 90)
    sd = torch.load(CKPT[split], map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    fa = pyfaidx.Fasta(FASTA, as_raw=True)

    meta = {"split": split, "ckpt": CKPT[split]}
    for tag, chrs_ in (("val", val_chrs), ("test", test_chrs)):
        pool_path = os.path.join(split_dir, f"EXP1_pool_{tag}.csv")
        df = pd.read_csv(pool_path)
        df = df[df["chr"].isin(chrs_)].reset_index(drop=True)
        print(f"[Data] {tag}: n={len(df)} SE={int((df.kind=='SE').sum())}")
        scores, cov = score_pool(df, fa, model, device, args.batch_size,
                                 use_bf16, tag)
        out_p = os.path.join(split_dir, f"EXP1_cerberus_union_{tag}.npy")
        np.save(out_p, scores)
        meta[tag] = {
            "n": int(len(df)), "n_scored": int(np.isfinite(scores).sum()),
            "mean_score": float(np.nanmean(scores)),
            "median_cov": float(np.median(cov)),
            "min_cov": float(cov.min()) if len(cov) else None,
            "pool": pool_path,
        }
        print(f"[save] {tag}: mean={meta[tag]['mean_score']:.4f} "
              f"median_cov={meta[tag]['median_cov']:.4f} -> {out_p}")
    with open(os.path.join(split_dir, "EXP1_cerberus_union_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[Done] EXP1 Cerberus cluster scores")


if __name__ == "__main__":
    main()
