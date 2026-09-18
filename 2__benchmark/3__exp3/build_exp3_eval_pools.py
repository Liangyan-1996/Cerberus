#!/usr/bin/env python3
"""EXP3 S7c0：每谱系簇级评估池（SE 行正 / TE 行负，黑名单过滤）。

输出：exp3/pool_split{s}/{cell}/EXP3_pool_{val,test}.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")
SE_ROOT = "/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/SuperEnhancer"
BLACKLIST = "/public/home/liunangroup/liangyan/Genome/blacklist/hg38.blacklist.bed"
CELLS = ["K562", "HepG2", "GM12878", "WTC11"]
CELL_NAMES = {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--cell", required=True, choices=CELLS)
    args = ap.parse_args()
    split, cell = args.split, args.cell
    name = CELL_NAMES[cell]
    val_chrs, test_chrs = SPLIT_CHRS[split]
    bl = {}
    for line in open(BLACKLIST):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] in val_chrs + test_chrs:
            bl.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    blm = {c: merge_intervals(
        np.asarray([x[0] for x in v], np.int64),
        np.asarray([x[1] for x in v], np.int64)) for c, v in bl.items()}
    df = pd.read_csv(os.path.join(SE_ROOT, name, "hg38_AllStitched.table.txt"),
                     sep="\t", comment="#")
    df = df[df["CHROM"].isin(val_chrs + test_chrs)]
    out_dir = os.path.join(BASE, "exp3", f"pool_split{split}", cell)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    stats = {}
    for tag, chrs_ in (("val", val_chrs), ("test", test_chrs)):
        n_se = n_te = 0
        for c in chrs_:
            sub = df[(df["CHROM"] == c)]
            if c not in blm:
                blm[c] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
            bs, be = blm[c]
            s = sub["START"].to_numpy(np.int64)
            e = sub["STOP"].to_numpy(np.int64)
            keep = ~overlaps(bs, be, s, e)
            sub = sub[keep]
            for _, r in sub.iterrows():
                kind = "SE" if int(r["isSuper"]) == 1 else "TE"
                rows.append((f"{kind}|{c}:{int(r['START'])}-{int(r['STOP'])}",
                             c, int(r["START"]), int(r["STOP"]),
                             int((r["START"] + r["STOP"]) // 2), kind))
                if kind == "SE":
                    n_se += 1
                else:
                    n_te += 1
        stats[tag] = {"n_se": n_se, "n_te": n_te}
    d = pd.DataFrame(rows, columns=["id", "chr", "start", "end", "center", "kind"])
    for tag in ("val", "test"):
        chrs_ = val_chrs if tag == "val" else test_chrs
        sub = d[d["chr"].isin(chrs_)]
        p = os.path.join(out_dir, f"EXP3_pool_{tag}.csv")
        sub.to_csv(p, index=False)
        print(f"[{tag}] {len(sub)} rows -> {p}")
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
