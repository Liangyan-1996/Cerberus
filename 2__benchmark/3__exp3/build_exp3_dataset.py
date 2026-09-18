#!/usr/bin/env python3
"""EXP3 S7a：谱系特异训练数据（纯单细胞标签，1:1 cap 30k）。

对 (cell, split) 生成 train.csv / val.csv：
  正 = 该细胞 ROSE SE（isSuper==1）内的 MACS2 peak（合并单元 merge -d 3000，
       中心 3 kb 切片）；
  负 = 该细胞其余（非 SE）peaks（同上建单元）∪ 随机 3 kb 背景区
       （排除 blacklist / >50% N / 与该细胞 SE 或 peaks 重叠）；
  平衡：正负 1:1（cap 30k，seed=42+cell*5+split）；负样本内 typical:random≈1:1。
用法：python build_exp3_dataset.py --split 0 --cell K562
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyfaidx

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
SE_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/SuperEnhancer"
PEAKS_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/peaks"
FASTA = "/public/home/liunangroup/liangyan/Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa"
BLACKLIST = "/public/home/liunangroup/liangyan/Genome/blacklist/hg38.blacklist.bed"

CELLS = {
    "K562": "hg38.SE.ENCSR000AKP.K562",
    "HepG2": "hg38.SE.ENCSR000AMO.HepG2",
    "GM12878": "hg38.SE.ENCSR000AKC.GM12878",
    "WTC11": "hg38.PE.ENCSR146DPQ.WTC11",
}
SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}
STD_CHRS = [f"chr{i}" for i in range(1, 23)]
FRAG = 3000
PEAK_MERGE_D = 3000
CAP = 30000


def load_blacklist(chroms):
    rows = {c: [] for c in chroms}
    for line in open(BLACKLIST):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] in chroms:
            rows[p[0]].append((int(p[1]), int(p[2])))
    out = {}
    for c in chroms:
        if rows[c]:
            arr = np.asarray(sorted(rows[c]), dtype=np.int64)
            out[c] = (arr[:, 0], arr[:, 1])
        else:
            out[c] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return out


def merge_intervals(starts, ends, dist=0):
    if len(starts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    ms, me = [], []
    cs, ce = int(s[0]), int(e[0])
    for i in range(1, len(s)):
        si, ei = int(s[i]), int(e[i])
        if si <= ce + dist:
            ce = max(ce, ei)
        else:
            ms.append(cs)
            me.append(ce)
            cs, ce = si, ei
    ms.append(cs)
    me.append(ce)
    return np.asarray(ms, np.int64), np.asarray(me, np.int64)


def overlaps(ms, me, s, e):
    if len(ms) == 0:
        return np.zeros(len(s), dtype=bool)
    idx = np.searchsorted(ms, s, side="left")
    out = np.zeros(len(s), dtype=bool)
    m = len(ms)
    ok = idx < m
    out[ok] |= ms[idx[ok]] < e[ok]
    ok2 = idx > 0
    out[ok2] |= me[idx[ok2] - 1] > s[ok2]
    return out


def read_se(name, chroms):
    df = pd.read_csv(os.path.join(SE_ROOT, name, "hg38_AllStitched.table.txt"),
                     sep="\t", comment="#")
    df = df[(df["CHROM"].isin(chroms)) & (df["isSuper"] == 1)]
    out = {}
    for c, g in df.groupby("CHROM"):
        s, e = g["START"].to_numpy(np.int64), g["STOP"].to_numpy(np.int64)
        ms, me = merge_intervals(s, e, dist=0)
        out[c] = (ms, me)
    return out


def read_peaks(name, chroms):
    df = pd.read_csv(os.path.join(PEAKS_ROOT, f"{name}_peaks.bed"),
                     sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    df = df[df["chrom"].isin(chroms)]
    return df


def make_units(peaks_df, chroms, excl_s=None, excl_e=None, excl_chr=None):
    """peaks -> merge -d3000 单元（中心 3 kb 可切片的）。"""
    units = {}
    for c in chroms:
        sub = peaks_df[peaks_df["chrom"] == c]
        if len(sub) == 0:
            units[c] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
            continue
        s = sub["start"].to_numpy(np.int64)
        e = sub["end"].to_numpy(np.int64)
        if excl_s is not None:
            keep = ~overlaps(excl_s[c], excl_e[c], s, e)
            s, e = s[keep], e[keep]
        ms, me = merge_intervals(s, e, dist=PEAK_MERGE_D)
        units[c] = (ms, me)
    return units


def sample_random(fa, bl, se_union, all_peaks, chroms, chrom_len, n_need,
                  rng, max_attempts=800000):
    """随机 3 kb 背景中心：不与 SE 并集/任何 peaks/blacklist 重叠，N<=50%。"""
    half = FRAG // 2
    centers = []
    total = 0
    for c in chroms:
        if total >= n_need:
            break
        ln = chrom_len[c]
        bs, be = bl[c]
        ss, se_ = se_union[c]
        pk = all_peaks[c]
        ps = pk["start"].to_numpy(np.int64) if len(pk) else np.zeros(0, np.int64)
        pe = pk["end"].to_numpy(np.int64) if len(pk) else np.zeros(0, np.int64)
        attempts = 0
        while total < n_need and attempts < max_attempts:
            attempts += 1
            cc = int(rng.integers(half, ln - half))
            ws, we = cc - half, cc + half
            if overlaps(bs, be, np.asarray([ws]), np.asarray([we]))[0]:
                continue
            if overlaps(ss, se_, np.asarray([ws]), np.asarray([we]))[0]:
                continue
            if len(ps) and overlaps(ps, pe, np.asarray([ws]), np.asarray([we]))[0]:
                continue
            seq = fa[c][ws:we].upper()
            if seq.count("N") > 0.5 * FRAG:
                continue
            centers.append((c, ws, we, cc))
            total += 1
    return centers


def center_units(units, chrom_len):
    """unit -> (chr, unit_s, unit_e, center) 可切片列表。"""
    rows = []
    half = FRAG // 2
    for c, (ms, me) in units.items():
        if len(ms) == 0:
            continue
        ctr = (ms + me) // 2
        keep = (ctr - half >= 0) & (ctr + half <= chrom_len[c])
        for a, b, cc in zip(ms[keep], me[keep], ctr[keep]):
            rows.append((c, int(a), int(b), int(cc)))
    return rows


def write_split(rows_pos, rows_neg_peak, rand_rows, out_csv, fa, tag,
                rng, cap=CAP):
    n = min(len(rows_pos), cap)
    if n == 0 or (len(rows_neg_peak) + len(rand_rows)) == 0:
        return 0
    n_rand = min(len(rand_rows), n // 2)
    n_peak = n - n_rand
    if n_peak > len(rows_neg_peak):
        n_peak = len(rows_neg_peak)
        n_rand = n - n_peak
    if n_rand > len(rand_rows):
        n_rand = len(rand_rows)
    n = n_peak + n_rand
    pos = rng.choice(len(rows_pos), size=n, replace=False)
    pk = rng.choice(len(rows_neg_peak), size=n_peak, replace=False)
    rd = rng.choice(len(rand_rows), size=n_rand, replace=False)
    recs = []
    for kind, idx in (("pos", pos), ("pk", pk), ("rd", rd)):
        for i in idx:
            if kind == "pos":
                c, us, ue, cc = rows_pos[i]
                label = 1
            elif kind == "pk":
                c, us, ue, cc = rows_neg_peak[i]
                label = 0
            else:
                c, ws, we, cc = rand_rows[i]
                us, ue, label = ws, we, 0
            recs.append((f"{kind}|{c}:{cc - FRAG // 2}-{cc + FRAG // 2}",
                         c, us, ue, cc, label, kind))
    d = pd.DataFrame(recs, columns=["id", "chr", "unit_start", "unit_end",
                                    "center", "label", "src"])
    seqs = []
    for ch, cc in zip(d["chr"], d["center"]):
        seqs.append(str(fa[ch][int(cc) - FRAG // 2:int(cc) + FRAG // 2]).upper())
    d["sequence"] = seqs
    d.to_csv(out_csv, index=False)
    n_pos = int((d.label == 1).sum())
    print(f"[{tag}] {out_csv}: n={len(d)} pos={n_pos} "
          f"neg_peak={int((d.src=='pk').sum())} random={int((d.src=='rd').sum())}")
    return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    args = ap.parse_args()
    split, cell = args.split, args.cell
    name = CELLS[cell]
    val_chrs, test_chrs = SPLIT_CHRS[split]
    held = set(val_chrs) | set(test_chrs)
    train_chrs = [c for c in STD_CHRS if c not in held]
    out_dir = os.path.join(BASE, "exp3", f"data_split{split}", cell)
    os.makedirs(out_dir, exist_ok=True)
    fa = pyfaidx.Fasta(FASTA, as_raw=True)
    chrom_len = {c: len(fa[c]) for c in STD_CHRS}
    bl = load_blacklist(STD_CHRS)
    se = read_se(name, STD_CHRS)
    peaks = read_peaks(name, STD_CHRS)
    # 正向原始 peaks = 与自身 SE 行重叠；负向 = 不与自身 SE 并集重叠
    peak_pos_df = []
    peak_neg_df = []
    se_union = {}
    for c in STD_CHRS:
        ss, se_ = se.get(c, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        se_union[c] = (ss, se_)
        sub = peaks[peaks["chrom"] == c]
        if len(sub) == 0 or len(ss) == 0:
            peak_neg_df.append(sub)
            continue
        s = sub["start"].to_numpy(np.int64)
        e = sub["end"].to_numpy(np.int64)
        hit = overlaps(ss, se_, s, e)
        peak_pos_df.append(sub[hit])
        peak_neg_df.append(sub[~hit])
    peak_pos = pd.concat(peak_pos_df, ignore_index=True)
    peak_neg = pd.concat(peak_neg_df, ignore_index=True)
    print(f"[Peaks] positive(SE-peak)={len(peak_pos)} "
          f"negative(non-SE-peak)={len(peak_neg)}")
    pos_units = make_units(peak_pos, STD_CHRS)
    neg_units = make_units(peak_neg, STD_CHRS)
    all_peaks = {c: peaks[peaks["chrom"] == c] for c in STD_CHRS}
    rng = np.random.default_rng(42 + list(CELLS).index(cell) * 5 + split)

    stats = {}
    for tag, chrs_ in (("train", train_chrs), ("val", val_chrs)):
        pos_rows = [r for r in center_units(
            {c: pos_units[c] for c in chrs_}, chrom_len)
            if r[0] in chrs_]
        pk_rows = [r for r in center_units(
            {c: neg_units[c] for c in chrs_}, chrom_len)
            if r[0] in chrs_]
        n_need = min(len(pos_rows), CAP) // 2 + 10
        rand_rows = sample_random(fa, bl, se_union, all_peaks, chrs_,
                                  chrom_len, n_need, rng)
        n = write_split(pos_rows, pk_rows, rand_rows,
                        os.path.join(out_dir, f"{tag}.csv"), fa, tag, rng)
        stats[tag] = {"n": n, "pos": len(pos_rows), "neg_peak": len(pk_rows),
                      "random_available": len(rand_rows)}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("[Done] EXP3 dataset")


if __name__ == "__main__":
    main()
