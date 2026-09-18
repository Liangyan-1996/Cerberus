#!/usr/bin/env python3
"""EXP2 region 级 peak-restricted + max（零新推理）：
正 = union SE 区间；负 = typical（cluster_union−se_union）∪ desert 瓦片；
每个区间只取"构成峰"（90 cell MACS2 峰中落在区间内的峰，去重）：
  - 基线：峰分数 = 峰覆盖 bins 在 bin 轨道上的 max；区间分数 = 峰分数 max
  - Cerberus：峰分数 = 峰覆盖 bins 在 Union 轨道上的 max；区间分数 = 峰分数 max
无构成峰的区间（如部分 desert 瓦片）回退为区间内 bin max（记录 fallback 数）。
输出：exp2/eval_split{s}/EXP2_prcmax_region_metrics.csv
      exp2/eval_split{s}/EXP2_prcmax_region_pr_curves.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
PEAKS_ROOT = os.path.join(BASE, "assets/se_pipeline/peaks")
META = os.path.join(BASE, "assets/se_pipeline/metadata.pkl")
RES = 256
TILE_BINS = 80
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}
SOURCES = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]


def baseline_bin_track(tracks, masks_gt, chr_, tag):
    sc = np.load(os.path.join(tracks, f"baseline_{tag}.{chr_}.win_scores.npy"))
    st = np.load(os.path.join(tracks, f"baseline_windows.{chr_}.start.npy"))
    nbins = len(np.load(os.path.join(masks_gt, f"se_mask.{chr_}.npy")))
    track = np.zeros(nbins, dtype=np.float32)
    ok = sc >= 0
    for a, b, x in zip(st[ok] // RES, (st[ok] + 3000 + RES - 1) // RES, sc[ok]):
        track[a:min(b, nbins)] = np.maximum(track[a:min(b, nbins)], x)
    return track


def typical_union_regions(gt_union, chr_):
    cu = np.load(os.path.join(gt_union, f"cluster_union.{chr_}.npy"))
    su = np.load(os.path.join(gt_union, f"se_union.{chr_}.npy"))
    cs, ce = cu[0], cu[1]
    ss, se_ = su[0], su[1]
    out_s, out_e = [], []
    for a, b in zip(cs, ce):
        i0 = np.searchsorted(ss, a, side="left"); i1 = np.searchsorted(ss, b, side="right")
        cur = a
        for i in range(i0, i1):
            if se_[i] <= cur:
                continue
            if ss[i] > cur:
                out_s.append(cur); out_e.append(min(ss[i], b))
            cur = max(cur, se_[i])
            if cur >= b:
                break
        if cur < b:
            out_s.append(cur); out_e.append(b)
    return np.asarray(out_s, np.int64), np.asarray(out_e, np.int64)


def desert_tiles(desert_dir, chr_):
    desert = np.load(os.path.join(desert_dir, f"desert_mask.{chr_}.npy")).astype(bool)
    tiles = []
    i, n = 0, len(desert)
    while i < n:
        if not desert[i]:
            i += 1; continue
        j = i
        while j < n and desert[j]:
            j += 1
        for k in range(i, j, TILE_BINS):
            tiles.append((k * RES, min((k + TILE_BINS) * RES, j * RES)))
        i = j
    return tiles


def load_peaks(chroms):
    import pickle
    with open(META, "rb") as f:
        meta = pickle.load(f)
    starts, ends = {c: [] for c in chroms}, {c: [] for c in chroms}
    for name in sorted(meta):
        p = os.path.join(PEAKS_ROOT, f"{name}_peaks.bed")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"])
        df = df[df["chrom"].isin(chroms)]
        for c, g in df.groupby("chrom"):
            starts[c].append(g["start"].to_numpy(np.int64))
            ends[c].append(g["end"].to_numpy(np.int64))
    out = {}
    for c in chroms:
        if starts[c]:
            s = np.concatenate(starts[c]); e = np.concatenate(ends[c])
            o = np.argsort(s, kind="stable")
            out[c] = (s[o], e[o])
        else:
            out[c] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return out


def region_peak_scores(reg_chrom, reg_s, reg_e, pk, track_by_chr):
    """每个区间的 peak-restricted max 分数 + fallback 计数。"""
    scores = np.full(len(reg_s), np.nan, dtype=np.float32)
    n_fallback = 0
    for k in range(len(reg_s)):
        c = reg_chrom[k]
        if c not in pk:
            n_fallback += 1; continue
        ms, me = pk[c]
        s, e = int(reg_s[k]), int(reg_e[k])
        if len(ms) == 0:
            n_fallback += 1
            t = track_by_chr[c]
            scores[k] = float(t[s // RES:int(np.ceil(e / RES))].max())
            continue
        lo = int(np.searchsorted(ms, s, side="right") - 1); lo = max(0, lo)
        hits = []
        for j in range(lo, len(ms)):
            if ms[j] >= e:
                break
            if me[j] > s:
                hits.append(j)
        if not hits:
            n_fallback += 1
            t = track_by_chr[c]
            scores[k] = float(t[s // RES:int(np.ceil(e / RES))].max())
            continue
        t = track_by_chr[c]
        best = -np.inf
        for j in hits:
            b0 = int(ms[j]) // RES
            b1 = int(np.ceil(int(me[j]) / RES))
            best = max(best, float(t[b0:b1].max()))
        scores[k] = best
    return scores, n_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    args = ap.parse_args()
    split = args.split
    _, test_chrs = SPLIT_CHRS[split]
    gt_root = os.path.join(BASE, f"assets/gt_split{split}/gt")
    masks_gt = os.path.join(gt_root, "masks")
    gt_union = os.path.join(gt_root, "union")
    desert_dir = os.path.join(BASE, "exp2", f"masks_split{split}")
    tracks = os.path.join(BASE, f"assets/tracks_split{split}/tracks")
    out_dir = os.path.join(BASE, "exp2", f"eval_split{split}")
    os.makedirs(out_dir, exist_ok=True)
    pk = load_peaks(test_chrs)
    pos_chrom, pos_s, pos_e = [], [], []
    neg_chrom, neg_s, neg_e = [], [], []
    for c in test_chrs:
        su = np.load(os.path.join(gt_union, f"se_union.{c}.npy"))
        pos_chrom += [c] * len(su[0]); pos_s += list(su[0]); pos_e += list(su[1])
        ts, te = typical_union_regions(gt_union, c)
        neg_chrom += [c] * len(ts); neg_s += list(ts); neg_e += list(te)
        for (a, b) in desert_tiles(desert_dir, c):
            neg_chrom.append(c); neg_s.append(a); neg_e.append(b)
    pos_chrom = np.asarray(pos_chrom); neg_chrom = np.asarray(neg_chrom)
    pos_s, pos_e = np.asarray(pos_s, np.int64), np.asarray(pos_e, np.int64)
    neg_s, neg_e = np.asarray(neg_s, np.int64), np.asarray(neg_e, np.int64)
    rows, curves = [], {}
    for src in SOURCES:
        if src == "Cerberus_union":
            tb = {c: np.load(os.path.join(tracks, f"cerberus.{c}.union.npy"))
                  for c in test_chrs}
        else:
            tb = {c: baseline_bin_track(tracks, masks_gt, c, src) for c in test_chrs}
        ps, nf_p = region_peak_scores(pos_chrom, pos_s, pos_e, pk, tb)
        ns, nf_n = region_peak_scores(neg_chrom, neg_s, neg_e, pk, tb)
        y = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
        sc = np.r_[ps, ns]
        aupr = float(average_precision_score(y, sc))
        rows.append({"split": split, "source": src, "n_pos": len(ps),
                     "n_neg": len(ns), "auprc": round(aupr, 4),
                     "fallback_pos": nf_p, "fallback_neg": nf_n})
        prec, rec, _ = precision_recall_curve(y, sc)
        curves[src] = {"precision": prec, "recall": rec}
        print(f"[{src}] AUPRC={aupr:.4f} pos={len(ps)} neg={len(ns)} "
              f"fallback={nf_p}+{nf_n}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "EXP2_prcmax_region_metrics.csv"),
                              index=False)
    np.savez(os.path.join(out_dir, "EXP2_prcmax_region_pr_curves.npz"),
             **{f"{k}_{kk}": np.asarray(vv) for k, d in curves.items()
                for kk, vv in d.items()})
    print("[Done] EXP2 region peak-restricted max")


if __name__ == "__main__":
    main()
