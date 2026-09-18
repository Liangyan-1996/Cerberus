#!/usr/bin/env python3
"""用本项目自己的 SE annotation（90 channel ROSE + H3K27ac peaks）构建
chr-split 训练数据，供四个基线模型重训。

样本定义（与用户确认，2026-09-06）：
  阳性 = H3K27ac peak 落入任意 channel 的 ROSE SE（AllStitched isSuper==1）
         区间（SE 常含多个 constituent peak，因此以 peak 为单位，不取 SE 中心）；
  阴性 = H3K27ac peak 未与**任何** channel 的 SE 并集重叠的纯 typical enhancer
         峰（在 90 个谱系中从未达到 SE 阈值）。
  去重 = 所有 channel 汇总后按 `merge -d 3000` 合并过近/重复的峰。
输入 = 合并单元的中心 3,000 bp 窗口（基线模型输入规格）。
划分 = train: 非 3/9/11/14；val: 9/14（阈值/选型）；test: 3/11。
       train/val/test_balanced 均正负 1:1；test_all = chr3/11 全量（真实分布）。

用法：
  python build_own_se_dataset.py --out_dir data [--cap_per_class 30000]
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import pyfaidx

SE_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/SuperEnhancer"
PEAKS_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/peaks"
META = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/AllSample.hg38.SE.label_cache.metadata.pkl"
FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"

FRAG = 3000
PEAK_MERGE_D = 3000
STD_CHRS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
TRAIN_CHRS = [f"chr{i}" for i in [1, 2, 4, 5, 6, 7, 8, 10, 12, 13,
                                  15, 16, 17, 18, 19, 20, 21, 22]] + ["chrX"]
VAL_CHRS = ["chr9", "chr14"]
TEST_CHRS = ["chr3", "chr11"]


def merge_intervals(starts, ends, dist=0):
    """starts/ends int arrays -> (m_start, m_end)，gap<=dist 合并。"""
    if len(starts) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    s = starts[order]
    e = ends[order]
    ms, me = [], []
    cs, ce = s[0], e[0]
    for i in range(1, len(s)):
        if s[i] <= ce + dist:
            ce = max(ce, e[i])
        else:
            ms.append(cs)
            me.append(ce)
            cs, ce = s[i], e[i]
    ms.append(cs)
    me.append(ce)
    return np.asarray(ms, dtype=np.int64), np.asarray(me, dtype=np.int64)


def overlaps(ms, me, s, e):
    """merged 不相交区间 (ms,me) 与批量 (s,e) 是否重叠（向量化）。"""
    idx = np.searchsorted(ms, s, side="left")
    out = np.zeros(len(s), dtype=bool)
    m = len(ms)
    ok = idx < m
    out[ok] |= ms[idx[ok]] < e[ok]
    ok2 = idx > 0
    out[ok2] |= me[idx[ok2] - 1] > s[ok2]
    return out


def read_se_table(name):
    """该 channel ROSE AllStitched -> SE intervals {chrom: (starts,ends)}。"""
    path = os.path.join(SE_ROOT, name, "hg38_AllStitched.table.txt")
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df["isSuper"] == 1]
    out = {}
    for chrom, g in df.groupby("CHROM"):
        if chrom not in STD_CHRS:
            continue
        s = g["START"].to_numpy(dtype=np.int64)
        e = g["STOP"].to_numpy(dtype=np.int64)
        ms, me = merge_intervals(s, e, dist=0)
        out[chrom] = (ms, me)
    return out


def read_peaks(name):
    """narrowPeak 前 3 列 -> {chrom: (starts, ends)}。"""
    path = os.path.join(PEAKS_ROOT, f"{name}_peaks.bed")
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    out = {}
    for chrom, g in df.groupby("chrom"):
        if chrom not in STD_CHRS:
            continue
        out[chrom] = (g["start"].to_numpy(dtype=np.int64),
                      g["end"].to_numpy(dtype=np.int64))
    return out


def fragment_center(ms, me, chrom_len, half=FRAG // 2):
    """合并单元中心 -> 3000bp 窗口 (win_s, win_e)；越界丢弃。"""
    c = (ms + me) // 2
    ws = c - half
    we = c + half
    keep = (ws >= 0) & (we <= chrom_len)
    return ws[keep], we[keep], c[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cap_per_class", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_chrs", nargs="+", default=["chr9", "chr14"])
    ap.add_argument("--test_chrs", nargs="+", default=["chr3", "chr11"])
    args = ap.parse_args()
    global TRAIN_CHRS, VAL_CHRS, TEST_CHRS
    VAL_CHRS = list(args.val_chrs)
    TEST_CHRS = list(args.test_chrs)
    held = set(VAL_CHRS) | set(TEST_CHRS)
    TRAIN_CHRS = [c for c in STD_CHRS if c not in held]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(META, "rb") as f:
        meta = pickle.load(f)
    names = sorted(meta)
    print(f"[Meta] {len(names)} channels")

    chrom_len = {}
    with pyfaidx.Fasta(FASTA, as_raw=True) as fa:
        for c in STD_CHRS:
            chrom_len[c] = len(fa[c])

    # ---- 1) 每个 channel 的 SE 区间（合并，dist=0）----
    per_channel_se = {}
    all_se = {c: ([], []) for c in STD_CHRS}   # 全基因组 SE 并集原料
    for name in names:
        se = read_se_table(name)
        per_channel_se[name] = se
        for c in se:
            s, e = all_se[c]
            s.append(se[c][0])
            e.append(se[c][1])
    print("[SE] per-channel SE loaded")

    # ---- 2) 全基因组 SE 并集 ----
    se_union = {}
    n_se_chr = {}
    for c in STD_CHRS:
        s = np.concatenate(all_se[c][0]) if all_se[c][0] else np.zeros(0, np.int64)
        e = np.concatenate(all_se[c][1]) if all_se[c][1] else np.zeros(0, np.int64)
        ms, me = merge_intervals(s, e, dist=0)
        se_union[c] = (ms, me)
        n_se_chr[c] = len(ms)
    print("[SE] genome-wide union per chr:",
          {c: n_se_chr[c] for c in ["chr1", "chr3", "chr9", "chr11", "chrX"]},
          "...")

    # ---- 3) peaks: 正=落入本 channel SE；负=不落任何 SE 并集 ----
    pos_pool = {c: ([], []) for c in STD_CHRS}
    neg_pool = {c: ([], []) for c in STD_CHRS}
    n_pos_peak = n_neg_peak = 0
    for i, name in enumerate(names):
        peaks = read_peaks(name)
        se = per_channel_se[name]
        for c in STD_CHRS:
            if c not in peaks:
                continue
            ps, pe = peaks[c]
            # 正：与同一 channel 的 SE 重叠
            if c in se and len(se[c][0]):
                ov = overlaps(se[c][0], se[c][1], ps, pe)
                if ov.any():
                    pos_pool[c][0].append(ps[ov])
                    pos_pool[c][1].append(pe[ov])
            # 负：与全基因组 SE 并集不重叠
            us, ue = se_union[c]
            if len(us):
                ov_se = overlaps(us, ue, ps, pe)
                nv = ~ov_se
                if nv.any():
                    neg_pool[c][0].append(ps[nv])
                    neg_pool[c][1].append(pe[nv])
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(names)} channels")
    print("[Peaks] pooled")

    # ---- 4) merge -d 3000 -> 正/负单元 + n_cells ----
    pos_units = {}
    neg_units = {}
    for c in STD_CHRS:
        ps = np.concatenate(pos_pool[c][0]) if pos_pool[c][0] else np.zeros(0, np.int64)
        pe = np.concatenate(pos_pool[c][1]) if pos_pool[c][1] else np.zeros(0, np.int64)
        ms, me = merge_intervals(ps, pe, dist=PEAK_MERGE_D)
        pos_units[c] = (ms, me)
        ns = np.concatenate(neg_pool[c][0]) if neg_pool[c][0] else np.zeros(0, np.int64)
        ne = np.concatenate(neg_pool[c][1]) if neg_pool[c][1] else np.zeros(0, np.int64)
        ms2, me2 = merge_intervals(ns, ne, dist=PEAK_MERGE_D)
        neg_units[c] = (ms2, me2)
    n_pos = sum(len(v[0]) for v in pos_units.values())
    n_neg = sum(len(v[0]) for v in neg_units.values())
    print(f"[Units] positive(SEC-peak, merge-d3000)={n_pos}, "
          f"negative(pure-TE-peak)={n_neg}")

    # 每个正单元的 n_cells（多少 channel 的 SE 覆盖它）
    def count_cells(ms, me, c):
        n = np.zeros(len(ms), dtype=np.int16)
        for name in names:
            se = per_channel_se[name]
            if c not in se or not len(se[c][0]):
                continue
            idx = np.searchsorted(se[c][0], ms, side="left")
            m = len(se[c][0])
            ov = np.zeros(len(ms), dtype=bool)
            ok = idx < m
            ov[ok] |= se[c][0][idx[ok]] < me[ok]
            ok2 = idx > 0
            ov[ok2] |= se[c][1][idx[ok2] - 1] > ms[ok2]
            n += ov
        return n

    # ---- 5) 中心 3kb 窗口 + 逐 split 平衡 ----
    rows = []
    for c in STD_CHRS:
        for label, units in ((1, pos_units[c]), (0, neg_units[c])):
            ms, me = units
            if not len(ms):
                continue
            half = FRAG // 2
            c_all = (ms + me) // 2
            keep = (c_all - half >= 0) & (c_all + half <= chrom_len[c])
            if not keep.any():
                continue
            ms_k, me_k = ms[keep], me[keep]
            ctr = c_all[keep]
            ws, we = ctr - half, ctr + half
            ncell = (count_cells(ms_k, me_k, c) if label == 1
                     else np.zeros(len(ms_k), np.int16))
            for j in range(len(ws)):
                rows.append((f"{c}:{ws[j]}-{we[j]}", c, int(ms_k[j]), int(me_k[j]),
                             int(ctr[j]), label, int(ncell[j])))
    df = pd.DataFrame(rows, columns=["id", "chr", "unit_start", "unit_end",
                                     "center", "label", "n_cells"])
    df["split"] = df["chr"].map(lambda ch: ("train" if ch in TRAIN_CHRS
                                            else ("val" if ch in VAL_CHRS
                                                  else ("test" if ch in TEST_CHRS
                                                        else None))))
    df = df[df["split"].notna()]
    print(f"[Fragments] total={len(df)} "
          f"pos={int((df.label == 1).sum())} neg={int((df.label == 0).sum())}")

    def balanced_sample(sub, cap, rng_):
        pos = sub[sub.label == 1]
        neg = sub[sub.label == 0]
        n = int(min(len(pos), len(neg), cap))
        pos = pos.sample(n=n, random_state=int(rng_.integers(1, 2**31 - 1)))
        neg = neg.sample(n=n, random_state=int(rng_.integers(1, 2**31 - 1)))
        return pd.concat([pos, neg], ignore_index=True)

    out_paths = {}
    for sp, chrs in (("train", TRAIN_CHRS), ("val", VAL_CHRS),
                     ("test_balanced", TEST_CHRS)):
        sub = df[df["split"] == sp.split("_")[0]] if sp != "test_balanced" else df[df.split == "test"]
        if sp == "test_all":
            continue
        bal = balanced_sample(sub, args.cap_per_class, rng)
        p = os.path.join(args.out_dir, f"{sp}.csv")
        bal.to_csv(p, index=False)
        out_paths[sp] = len(bal)
        print(f"[Split] {sp}: {len(bal)} (pos={int((bal.label == 1).sum())}) -> {p}")
    # test_all：真实分布（chr3/11 全部）
    ta = df[df.split == "test"]
    p = os.path.join(args.out_dir, "test_all.csv")
    ta.to_csv(p, index=False)
    out_paths["test_all"] = len(ta)
    print(f"[Split] test_all: {len(ta)} (pos={int((ta.label == 1).sum())}) -> {p}")

    # ---- 6) 取序列 ----
    with pyfaidx.Fasta(FASTA, as_raw=True) as fa:
        for csv_name in ["train.csv", "val.csv", "test_balanced.csv", "test_all.csv"]:
            path = os.path.join(args.out_dir, csv_name)
            d = pd.read_csv(path)
            seqs = []
            for ch, ws, we in zip(d["chr"], d["center"] - FRAG // 2,
                                  d["center"] + FRAG // 2):
                seqs.append(str(fa[ch][int(ws):int(we)]).upper())
            d["sequence"] = seqs
            d.to_csv(path, index=False)
            print(f"[Seq] {csv_name}: {len(d)} sequences written")

    stats = {
        "channels": len(names),
        "pos_units": int(n_pos), "neg_units": int(n_neg),
        "fragments_total": int(len(df)),
        "frag_pos": int((df.label == 1).sum()),
        "frag_neg": int((df.label == 0).sum()),
        "cap_per_class": args.cap_per_class,
        "splits": out_paths,
    }
    with open(os.path.join(args.out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("[Done]")


if __name__ == "__main__":
    main()
