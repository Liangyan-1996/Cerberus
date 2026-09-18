#!/usr/bin/env python3
"""EXP2 S6b：按新 gap 掩码重算 desert_mask 与 bin 级指标（D3/D10）。

对 split s 的 test 染色体：
  desert_new = cluster_mask==0 且 blacklist_mask==0 且 gap_mask==0；
  bin AUROC/AUPRC：valid = ~(blacklist|gap)，GT = SE mask；
  desert FP/Mb@recall 0.5/0.8 = 荒漠中阳性 bin 长度(Mb) ÷ desert_new 长度(Mb)。
输出：exp2/masks_split{s}/desert_mask.{chr}.npy
      exp2/eval_split{s}/EXP2_metrics_bin.csv、EXP2_desert_fp_at_recall.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    roc_auc_score, roc_curve

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
RES = 256
SOURCES = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]
RECALLS = [0.5, 0.8]
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}


def baseline_bin_track(tracks, masks_dir, chr_):
    st = np.load(os.path.join(tracks, f"baseline_windows.{chr_}.start.npy"))
    nbins = len(np.load(os.path.join(masks_dir, f"se_mask.{chr_}.npy")))
    out = {}
    for tag in ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]:
        sc = np.load(os.path.join(tracks, f"baseline_{tag}.{chr_}.win_scores.npy"))
        track = np.zeros(nbins, dtype=np.float32)
        ok = sc >= 0
        for a, b, x in zip(st[ok] // RES,
                           (st[ok] + 3000 + RES - 1) // RES, sc[ok]):
            track[a:min(b, nbins)] = np.maximum(track[a:min(b, nbins)], x)
        out[tag] = track
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    _, test_chrs = SPLIT_CHRS[split]
    gt_root = os.path.join(BASE, f"assets/gt_split{split}/gt")
    tracks = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    gap_root = os.path.join(BASE, "exp2", "masks")
    mask_out = os.path.join(BASE, "exp2", f"masks_split{split}")
    eval_out = os.path.join(BASE, "exp2", f"eval_split{split}")
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(eval_out, exist_ok=True)

    y_all = {s: [] for s in SOURCES}
    p_all = {s: [] for s in SOURCES}
    desert_p = {s: [] for s in SOURCES}
    desert_stats = {}
    for chr_ in test_chrs:
        se = np.load(os.path.join(gt_root, "masks", f"se_mask.{chr_}.npy")).astype(bool)
        cl = np.load(os.path.join(gt_root, "masks",
                                  f"cluster_mask.{chr_}.npy")).astype(bool)
        bl = np.load(os.path.join(gt_root, "masks",
                                  f"blacklist_mask.{chr_}.npy")).astype(bool)
        gap = np.load(os.path.join(gap_root, f"gap_mask.{chr_}.npy")).astype(bool)
        assert len(gap) == len(se), (chr_, len(gap), len(se))
        desert = ((~cl) & (~bl) & (~gap)).astype(np.int8)
        np.save(os.path.join(mask_out, f"desert_mask.{chr_}.npy"), desert)
        desert_stats[chr_] = {
            "n_bins": int(len(desert)),
            "desert_bp_frac": round(float(desert.mean()), 5),
            "desert_mb": round(float(desert.sum() * RES / 1e6), 2),
            "old_desert_mb": round(float(((~cl) & (~bl)).sum() * RES / 1e6), 2),
        }
        valid = (~bl) & (~gap)
        bt = baseline_bin_track(tracks, os.path.join(gt_root, "masks"), chr_)
        for src in SOURCES:
            if src == "Cerberus_union":
                track = np.load(os.path.join(tracks, f"cerberus.{chr_}.union.npy"))
            else:
                track = bt[src]
            y_all[src].append(se[valid])
            p_all[src].append(track[valid])
            desert_p[src].append(track[desert.astype(bool)])
        print(f"[{chr_}] desert_frac={desert.mean():.5f} "
              f"old={desert_stats[chr_]['old_desert_mb']}Mb "
              f"new={desert_stats[chr_]['desert_mb']}Mb")

    rows_m, rows_f = [], []
    curves = {}
    for src in SOURCES:
        y = np.concatenate(y_all[src])
        p = np.concatenate(p_all[src])
        pd_ = np.concatenate(desert_p[src])
        auc = roc_auc_score(y, p)
        aupr = average_precision_score(y, p)
        rows_m.append({"split": split, "source": src, "n_bins": int(len(y)),
                       "n_se_bins": int(y.sum()), "auroc": round(float(auc), 4),
                       "auprc": round(float(aupr), 4)})
        pos_p = p[y]
        desert_mb = float(pd_.size * RES / 1e6)
        for rec in RECALLS:
            thr = float(np.quantile(pos_p, 1.0 - rec))
            fp_mb = float((pd_ >= thr).sum() * RES / 1e6)
            rows_f.append({"split": split, "source": src, "se_recall": rec,
                           "thr": round(thr, 5), "desert_mb": round(desert_mb, 3),
                           "fp_per_mb": round(fp_mb / desert_mb, 4)})
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        curves[src] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec,
                       "auroc": float(auc), "auprc": float(aupr)}
    np.savez(os.path.join(eval_out, "EXP2_bin_curves.npz"),
             **{f"{k}_{kk}": np.asarray(vv) for k, d in curves.items()
                for kk, vv in d.items()})
    pd.DataFrame(rows_m).to_csv(os.path.join(eval_out, "EXP2_metrics_bin.csv"),
                                index=False)
    pd.DataFrame(rows_f).to_csv(
        os.path.join(eval_out, "EXP2_desert_fp_at_recall.csv"), index=False)
    with open(os.path.join(mask_out, "desert_stats.json"), "w") as f:
        json.dump(desert_stats, f, indent=2)
    print(f"[save] exp2/eval_split{split} metrics + desert masks")


if __name__ == "__main__":
    main()
