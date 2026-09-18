#!/usr/bin/env python3
"""生成 k-mer embedding 缓存矩阵并验证映射规则（2026-09-06 实证）。

产出（benchmark/ 下）：
  kmer_deepse_hg19.npz   : DeepSE 自带 hg19 w2v (4/5/6-mer) -> {k:(4^k+1,100)}
  kmer_official_k3to8.npz: 官方 dna2vec k3to8 (4/5/6-mer) -> {k:(4^k+1,100)}

验证：
  1) 官方 w2v 的 k6 矩阵 vs DcSE/SENet/MuSE 共用 embedding_matrix.npy（应全等）；
  2) MuSE 官方预计算 test_human_{k}mer100_1.npz 逐行 vs 本脚本用 DeepSE w2v +
     非重叠求和生成的 (300,) 向量（应 cos=1 / ratio=1，证明 MuSE X2 == DeepSE 映射）。

用法：
  python prep_kmer_embeddings.py [--full] [--out_dir .]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from features import (  # noqa: E402
    build_kmer_matrices,
    load_kmer_matrices,
    save_kmer_matrices,
    seq_to_deepse_vec,
)

ROOT = os.path.normpath(os.path.join(_DIR, ".."))
W2V_OFFICIAL = os.path.join(
    ROOT, "dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v")
W2V_DEEPSE = os.path.join(
    ROOT, "models/original_sources/DeepSE-master/data/hg19_embedding.w2v")
REF_6MER_MATRIX = os.path.join(_DIR, "embedding_matrix.npy")
MUSE_NPZ_DIR = os.path.join(
    ROOT, "models/original_sources/MuSE-main/MuSE/datasets/test_human")
MUSE_TEST_CSV = os.path.join(
    ROOT, "models/DcSE-main/src/data/datasets/test_human.csv")


def verify_k6_official(mats):
    ref = np.load(REF_6MER_MATRIX)
    d = np.abs(mats[6] - ref).max()
    ok = d < 1e-6
    print(f"[Verify] 官方 w2v k6 矩阵 vs embedding_matrix.npy: "
          f"max_abs_diff={d:.3e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def verify_muse_x2(mats_deepse, n_check):
    """MuSE test_human npz (2068 行) 与 DeepSE w2v 求和的逐行比对。"""
    refs = []
    for k in (4, 5, 6):
        with np.load(os.path.join(MUSE_NPZ_DIR, f"test_human_{k}mer100_1.npz")) as z:
            refs.append(z["seq_vec"].astype(np.float64))
    labels = None
    df = pd.read_csv(MUSE_TEST_CSV)
    with np.load(os.path.join(MUSE_NPZ_DIR, "test_human_4mer100_1.npz")) as z:
        labels = z["label"]
    if len(df) != len(labels):
        print(f"[Warn] csv 行数 {len(df)} != npz 行数 {len(labels)}")
    assert len(refs[0]) == len(labels)
    worst = 0.0
    worst_per_k = {k: 0.0 for k in (4, 5, 6)}
    n = min(n_check, len(df))
    for i in range(n):
        seq = str(df.iloc[i]["sequence"])
        got_full = seq_to_deepse_vec(seq, mats_deepse)
        tgt_full = np.concatenate([r[i] for r in refs])
        cos = float(np.dot(got_full, tgt_full) /
                    (np.linalg.norm(got_full) * np.linalg.norm(tgt_full) + 1e-15))
        worst = max(worst, 1.0 - cos)
        off = 0
        for j, k in enumerate((4, 5, 6)):
            g, t = got_full[off:off + 100], refs[j][i]
            c = float(np.dot(g, t) / (np.linalg.norm(g) * np.linalg.norm(t) + 1e-15))
            worst_per_k[k] = max(worst_per_k[k], 1.0 - c)
            off += 100
        if i < 5:
            print(f"  row {i}: label={int(labels[i])} "
                  f"cos={cos:.9f} |got|-|tgt|="
                  f"{abs(np.linalg.norm(got_full)-np.linalg.norm(tgt_full)):.3e}")
    ok = worst < 1e-5  # float32 数值精度级别（~1e-7）视为一致
    print(f"[Verify] MuSE X2 逐行比对 {n}/{len(df)} 行: "
          f"worst(1-cos)={worst:.3e} per_k={ {k: f'{v:.2e}' for k, v in worst_per_k.items()} } "
          f"-> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=_DIR)
    ap.add_argument("--n_check", type=int, default=500,
                    help="MuSE X2 比对行数；<=0 表示全量")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_deepse = os.path.join(args.out_dir, "kmer_deepse_hg19.npz")
    out_official = os.path.join(args.out_dir, "kmer_official_k3to8.npz")

    print("[1/2] 构建 DeepSE hg19 w2v 矩阵 ...")
    mats_deepse = save_kmer_matrices(W2V_DEEPSE, out_deepse)
    print("      ->", out_deepse,
          {k: mats_deepse[k].shape for k in mats_deepse})

    print("[2/2] 构建官方 dna2vec k3to8 矩阵 ...")
    mats_official = save_kmer_matrices(W2V_OFFICIAL, out_official)
    print("      ->", out_official,
          {k: mats_official[k].shape for k in mats_official})

    ok1 = verify_k6_official(mats_official)
    n_check = args.n_check if args.n_check > 0 else None
    ok2 = verify_muse_x2(mats_deepse, n_check if n_check else 2068)

    # 顺带报告两套 w2v 数值差异（证明 DeepSE 与官方不是同一模型）
    m1 = load_kmer_matrices(out_deepse)
    m2 = load_kmer_matrices(out_official)
    diffs = {k: float(np.abs(m1[k] - m2[k]).max()) for k in (4, 5, 6)}
    print(f"[Info] DeepSE w2v vs 官方 w2v 同词向量 max_abs_diff: {diffs}")

    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
