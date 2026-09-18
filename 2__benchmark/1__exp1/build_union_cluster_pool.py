#!/usr/bin/env python3
"""EXP1 S1：构建 ROSE 缝合簇级 union 样本池（90-cell 不分 cell 并集口径）。

每 split 输出 val/test 两个簇池（Exp1 只评估，不用训练集，D9 零重训）：
  正样本 = 90 cell isSuper==1 缝合行（黑名单重叠剔除）按染色体检合并（gap=0）
           的 SE 并集块；
  负样本 = 90 cell isSuper==0 缝合行（黑名单重叠剔除）并集 TE 块，再按碱基
           层次减去 SE 并集 -> TE-only 并集块（与正样本零重叠）。

用法：
  python build_union_cluster_pool.py --split 0
  python build_union_cluster_pool.py --split 0 --only_cells hg38.SE.ENCSR000AKP.K562  # 冒烟
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
SE_ROOT = os.path.join(BASE, "assets/se_pipeline/superenhancer")
META = os.path.join(BASE, "assets/se_pipeline/metadata.pkl")
CHROM_SIZES = os.path.join(BASE, "assets/se_pipeline/hg38.chrom.sizes")
BLACKLIST = os.path.join(BASE, "assets/genome/genome/blacklist/hg38.blacklist.bed")

SPLIT_CHRS = {
    0: (["chr9", "chr14"], ["chr3", "chr11"]),
    1: (["chr1", "chr15"], ["chr2", "chr12"]),
    2: (["chr4", "chr18"], ["chr5", "chr17"]),
    3: (["chr6", "chr19"], ["chr7", "chr20"]),
    4: (["chr8", "chr21"], ["chr10", "chr13"]),
}
RES = 256


def load_chrom_sizes(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out[p[0]] = int(p[1])
    return out


def load_blacklist(path, chroms):
    rows = {c: [] for c in chroms}
    for line in open(path):
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
    """批量区间是否与合并后区间集（ms,me 升序不相交）重叠。"""
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


def subtract_intervals(te_s, te_e, se_s, se_e):
    """TE 并集减去 SE 并集，返回剩余连续片段（碱基层次）。"""
    if len(se_s) == 0:
        return te_s.copy(), te_e.copy()
    out_s, out_e = [], []
    for a0, a1 in zip(te_s, te_e):
        cur = int(a0)
        lo = int(np.searchsorted(se_s, a0, side="left") - 1)
        lo = max(0, lo)
        for j in range(lo, len(se_s)):
            b0, b1 = int(se_s[j]), int(se_e[j])
            if b1 <= cur:
                continue
            if b0 >= a1:
                break
            if b0 > cur:
                out_s.append(cur)
                out_e.append(min(b0, a1))
            cur = max(cur, b1)
            if cur >= a1:
                break
        if cur < a1:
            out_s.append(cur)
            out_e.append(a1)
    if not out_s:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.asarray(out_s, np.int64), np.asarray(out_e, np.int64)


def read_stitched_rows(name, chroms):
    """返回 (se_rows, te_rows) 字典 {chrom: (starts, ends)}（黑名单过滤前）。"""
    path = os.path.join(SE_ROOT, name, "hg38_AllStitched.table.txt")
    df = pd.read_csv(path, sep="\t", comment="#")
    if not {"CHROM", "START", "STOP", "isSuper"}.issubset(df.columns):
        raise ValueError(f"{path} 缺列: {list(df.columns)[:12]}")
    df = df[df["CHROM"].isin(chroms)]
    se = {c: ([], []) for c in chroms}
    te = {c: ([], []) for c in chroms}
    for chrom, g in df.groupby("CHROM"):
        for dst, mask in ((se, g["isSuper"] == 1), (te, g["isSuper"] == 0)):
            sub = g[mask]
            if len(sub):
                dst[chrom][0].append(sub["START"].to_numpy(np.int64))
                dst[chrom][1].append(sub["STOP"].to_numpy(np.int64))
    for c in chroms:
        for dst in (se, te):
            dst[c] = ((np.concatenate(dst[c][0]) if dst[c][0]
                       else np.zeros(0, np.int64)),
                      (np.concatenate(dst[c][1]) if dst[c][1]
                       else np.zeros(0, np.int64)))
    return se, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="默认 {BASE}/exp1/split{split}")
    ap.add_argument("--only_cells", nargs="*", default=None,
                    help="冒烟测试：只读指定 cell")
    args = ap.parse_args()
    split = args.split
    val_chrs, test_chrs = SPLIT_CHRS[split]
    chroms = val_chrs + test_chrs
    out_dir = args.out or os.path.join(BASE, "exp1", f"split{split}")
    os.makedirs(out_dir, exist_ok=True)

    chrom_len = load_chrom_sizes(CHROM_SIZES)
    bl = load_blacklist(BLACKLIST, chroms)
    bl_merged = {c: merge_intervals(bl[c][0], bl[c][1], dist=0) for c in chroms}
    with open(META, "rb") as f:
        meta = pickle.load(f)
    names = sorted(meta) if not args.only_cells else sorted(args.only_cells)
    if args.only_cells and not set(names).issubset(meta):
        raise SystemExit("only_cells 中名字不在 90-channel metadata 内")
    print(f"[Info] split{split} chrs={chroms} cells={len(names)}")

    se_pool = {c: ([], []) for c in chroms}
    te_pool = {c: ([], []) for c in chroms}
    for i, name in enumerate(names):
        se, te = read_stitched_rows(name, chroms)
        for c in chroms:
            bs, be = bl_merged[c]
            for dst, d in ((se_pool, se), (te_pool, te)):
                s, e = d[c]
                keep = ~overlaps(bs, be, s, e)
                if keep.any():
                    dst[c][0].append(s[keep])
                    dst[c][1].append(e[keep])
        if (i + 1) % 20 == 0 or i + 1 == len(names):
            print(f"  ... {i + 1}/{len(names)} cells")

    se_union = {}
    te_union = {}
    for c in chroms:
        s = np.concatenate(se_pool[c][0]) if se_pool[c][0] else np.zeros(0, np.int64)
        e = np.concatenate(se_pool[c][1]) if se_pool[c][1] else np.zeros(0, np.int64)
        se_union[c] = merge_intervals(s, e, dist=0)
        s = np.concatenate(te_pool[c][0]) if te_pool[c][0] else np.zeros(0, np.int64)
        e = np.concatenate(te_pool[c][1]) if te_pool[c][1] else np.zeros(0, np.int64)
        te_union[c] = merge_intervals(s, e, dist=0)

    def link_dir(tag):
        return os.path.join(BASE, "assets",
                            f"gt_{tag}_split{split}" if tag == "val"
                            else f"gt_split{split}", "gt")
    for tag, chrs_ in (("val", val_chrs), ("test", test_chrs)):
        gd = os.path.join(link_dir(tag), "union")
        ok_all = True
        for c in chrs_:
            p = os.path.join(gd, f"se_union.{c}.npy")
            if os.path.exists(p):
                old = np.load(p)
                new = np.stack(se_union[c])
                if old.shape == new.shape and np.array_equal(old, new):
                    print(f"[check] {tag} {c}: SE union 与既有 GT 一致")
                else:
                    ok_all = False
                    print(f"[check] {tag} {c}: 不一致 old={old.shape} new={new.shape} "
                          f"n_old={old.shape[1] if old.ndim==2 else -1} "
                          f"n_new={new.shape[1] if new.ndim==2 else -1}")
            else:
                print(f"[check] {tag} {c}: 无既有 GT（跳过交叉校验）")
        if not ok_all:
            print("[warn] 至少一条染色体 SE union 与既有 GT 不一致，请核对黑名单/过滤口径")

    # 负样本：TE 并集 − SE 并集（TE-only 连续段）
    neg = {}
    for c in chroms:
        ns, ne = subtract_intervals(te_union[c][0], te_union[c][1],
                                    se_union[c][0], se_union[c][1])
        neg[c] = (ns, ne)
        assert not overlaps(se_union[c][0], se_union[c][1], ns, ne).any(), c

    rows = []
    stats = {"split": split, "val_chrs": val_chrs, "test_chrs": test_chrs}
    for tag, chrs_ in (("val", val_chrs), ("test", test_chrs)):
        n_se = n_te = 0
        mb_se = mb_te = 0.0
        med_se, med_te = [], []
        for c in chrs_:
            for kind, (ss, ee), med in (("SE", se_union[c], med_se),
                                        ("TE", neg[c], med_te)):
                for a, b in zip(ss, ee):
                    rows.append((f"{kind}|{c}:{a}-{b}", c, int(a), int(b),
                                 int((a + b) // 2), kind))
                    if kind == "SE":
                        n_se += 1
                        mb_se += (b - a) / 1e6
                    else:
                        n_te += 1
                        mb_te += (b - a) / 1e6
                med.extend((ee - ss).tolist())
        med_se = np.asarray(med_se, np.float64)
        med_te = np.asarray(med_te, np.float64)
        stats[tag] = {
            "n_se": n_se, "n_te": n_te,
            "se_mb": round(mb_se, 2), "te_mb": round(mb_te, 2),
            "se_median_kb": round(float(np.median(med_se)) / 1e3, 1) if len(med_se) else 0,
            "te_median_kb": round(float(np.median(med_te)) / 1e3, 1) if len(med_te) else 0,
            "ratio_te_over_se": round(n_te / n_se, 2) if n_se else None,
        }
    df = pd.DataFrame(rows, columns=["id", "chr", "start", "end", "center", "kind"])
    for tag in ("val", "test"):
        chrs_ = val_chrs if tag == "val" else test_chrs
        sub = df[df["chr"].isin(chrs_)].copy()
        p = os.path.join(out_dir, f"EXP1_pool_{tag}.csv")
        sub.to_csv(p, index=False)
        print(f"[save] {tag}: n={len(sub)} "
              f"pos={int((sub.kind=='SE').sum())} -> {p}")
    p = os.path.join(out_dir, "EXP1_pool_stats.json")
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("[Done] EXP1 pool built")


if __name__ == "__main__":
    main()
