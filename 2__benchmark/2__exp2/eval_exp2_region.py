#!/usr/bin/env python3
"""EXP2 S6c：region 级指标（正=union SE；负=typical ∪ desert 瓦片；D4 无 ROC）。

分数 = 区间内 max(bin prob)（沿用 2A region 口径）；产出：
  exp2/eval_split{s}/EXP2_region_metrics.csv      (n_pos/n_neg/auprc)
  exp2/eval_split{s}/EXP2_region_iou_recall08.csv  (IoU>=0.3 median @rec0.8)
  exp2/eval_split{s}/EXP2_region_pr_curves.npz     (per-split PR 轨迹)
desert 瓦片读取 exp2/masks_split{s}/desert_mask.*（含 gap 排除）。
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
RES = 256
TILE_BINS = 80
GAP_BINS = 49
IOU_THR = 0.3
SOURCES = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}


def baseline_bin_track(tracks, masks_gt, chr_, tag):
    sc = np.load(os.path.join(tracks, f"baseline_{tag}.{chr_}.win_scores.npy"))
    st = np.load(os.path.join(tracks, f"baseline_windows.{chr_}.start.npy"))
    nbins = len(np.load(os.path.join(masks_gt, f"se_mask.{chr_}.npy")))
    track = np.zeros(nbins, dtype=np.float32)
    ok = sc >= 0
    for a, b, x in zip(st[ok] // RES,
                       (st[ok] + 3000 + RES - 1) // RES, sc[ok]):
        track[a:min(b, nbins)] = np.maximum(track[a:min(b, nbins)], x)
    return track


def typical_union_regions(gt_union, chr_):
    cu = np.load(os.path.join(gt_union, f"cluster_union.{chr_}.npy"))
    su = np.load(os.path.join(gt_union, f"se_union.{chr_}.npy"))
    cs, ce = cu[0], cu[1]
    ss, se_ = su[0], su[1]
    out_s, out_e = [], []
    for a, b in zip(cs, ce):
        i0 = np.searchsorted(ss, a, side="left")
        i1 = np.searchsorted(ss, b, side="right")
        cur = a
        for i in range(i0, i1):
            if se_[i] <= cur:
                continue
            if ss[i] > cur:
                out_s.append(cur)
                out_e.append(min(ss[i], b))
            cur = max(cur, se_[i])
            if cur >= b:
                break
        if cur < b:
            out_s.append(cur)
            out_e.append(b)
    return np.asarray(out_s, np.int64), np.asarray(out_e, np.int64)


def desert_tiles(desert_mask_dir, chr_):
    desert = np.load(os.path.join(desert_mask_dir,
                                  f"desert_mask.{chr_}.npy")).astype(bool)
    tiles = []
    i, n = 0, len(desert)
    while i < n:
        if not desert[i]:
            i += 1
            continue
        j = i
        while j < n and desert[j]:
            j += 1
        for k in range(i, j, TILE_BINS):
            a, b = k, min(k + TILE_BINS, j)
            tiles.append((a * RES, b * RES))
        i = j
    return tiles


def interval_iou(a0, a1, b0, b1):
    ov = min(a1, b1) - max(a0, b0)
    if ov <= 0:
        return 0.0
    return ov / ((a1 - a0) + (b1 - b0) - ov)


def merge_pred_bins(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out = []
    s0 = e0 = idx[0]
    for x in idx[1:]:
        if x - e0 <= GAP_BINS + 1:
            e0 = x
        else:
            out.append((s0, e0))
            s0 = e0 = x
    out.append((s0, e0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    _, test_chrs = SPLIT_CHRS[split]
    gt_root = os.path.join(BASE, f"assets/gt_split{split}/gt")
    gt_union = os.path.join(gt_root, "union")
    masks_gt = os.path.join(gt_root, "masks")
    desert_dir = os.path.join(BASE, "exp2", f"masks_split{split}")
    tracks = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    out_dir = os.path.join(BASE, "exp2", f"eval_split{split}")
    os.makedirs(out_dir, exist_ok=True)

    mats = {}
    for chr_ in test_chrs:
        us = np.load(os.path.join(gt_union, f"se_union.{chr_}.npy"))
        ms, me = us[0], us[1]
        ts, te = typical_union_regions(gt_union, chr_)
        tiles = desert_tiles(desert_dir, chr_)
        ns = np.concatenate([ts] + ([np.asarray([t[0] for t in tiles])] if tiles else []))
        ne = np.concatenate([te] + ([np.asarray([t[1] for t in tiles])] if tiles else []))
        mats[chr_] = (ms, me, ns, ne)
        print(f"[{chr_}] pos={len(ms)} neg_typical={len(ts)} "
              f"neg_desert_tiles={len(tiles)}")

    rows_m, rows_i = [], []
    curves = {}
    for src in SOURCES:
        pos_all, neg_all = [], []
        chr_preds = {}
        for chr_ in test_chrs:
            if src == "Cerberus_union":
                track = np.load(os.path.join(tracks, f"cerberus.{chr_}.union.npy"))
            else:
                track = baseline_bin_track(tracks, masks_gt, chr_, src)
            ms, me, ns, ne = mats[chr_]
            pos = np.asarray([track[int(a // RES):int((b + RES - 1) // RES)].max()
                              for a, b in zip(ms, me)])
            neg = np.asarray([track[int(a // RES):int((b + RES - 1) // RES)].max()
                              for a, b in zip(ns, ne)])
            pos_all.append(pos)
            neg_all.append(neg)
            chr_preds[chr_] = (track, ms, me)
        pos = np.concatenate(pos_all)
        neg = np.concatenate(neg_all)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        sc = np.concatenate([pos, neg])
        aupr = float(average_precision_score(y, sc))
        rows_m.append({"split": split, "source": src, "n_pos": int(len(pos)),
                       "n_neg": int(len(neg)), "auprc": round(aupr, 4)})
        prec, rec, _ = precision_recall_curve(y, sc)
        curves[src] = {"precision": prec, "recall": rec}
        thr = float(np.quantile(pos, 0.2))
        n_pred = tp = 0
        ious = []
        for chr_ in test_chrs:
            track, ms, me = chr_preds[chr_]
            preds = merge_pred_bins(track > thr)
            n_pred += len(preds)
            for s0, s1 in preds:
                a0, a1 = s0 * RES, (s1 + 1) * RES
                best = 0.0
                for b0, b1 in zip(ms, me):
                    best = max(best, interval_iou(a0, a1, b0, b1))
                if best >= IOU_THR:
                    tp += 1
                    ious.append(best)
        rows_i.append({"split": split, "source": src, "recall_target": 0.8,
                       "thr": round(thr, 5), "n_pred": n_pred,
                       "n_tp_iou0.3": tp,
                       "median_iou_tp": round(float(np.median(ious)), 4)
                       if ious else float("nan")})
        print(f"[{src}] auprc={aupr:.4f} pos={len(pos)} neg={len(neg)}")
    pd.DataFrame(rows_m).to_csv(os.path.join(out_dir, "EXP2_region_metrics.csv"),
                                index=False)
    pd.DataFrame(rows_i).to_csv(
        os.path.join(out_dir, "EXP2_region_iou_recall08.csv"), index=False)
    np.savez(os.path.join(out_dir, "EXP2_region_pr_curves.npz"),
             **{f"{k}_{kk}": np.asarray(vv) for k, d in curves.items()
                for kk, vv in d.items()})
    print("[Done] EXP2 region metrics")


if __name__ == "__main__":
    main()
