#!/usr/bin/env python3
"""最终报告生成（Exp1/Exp2 汇总表；在 S4/S5 后运行）。

输出：summary/BENCHMARK_REPORT.md（含 mean±SEM 表）
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1/" \
       "test10_eval_new/20260909.final_benchmark"
MODELS = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]


def ms_table(df, metric):
    rows = []
    for m in MODELS:
        v = df[df.model == m][metric].to_numpy(float) if "model" in df else \
            df[df.source == m][metric].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        mean = v.mean()
        sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append({"model": m, "mean": round(float(mean), 4),
                     "sem": round(float(sem), 4)})
    return pd.DataFrame(rows)


def fmt_md(df):
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main():
    lines = ["# 20260909.final_benchmark —— 结果报告（自动生成）", ""]
    p1 = os.path.join(BASE, "summary", "EXP1_prcmax_cluster_metrics_all.csv")
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        lines += ["## EXP1 Union 簇级（5-split mean ± SEM）", ""]
        for metric, name in (("auroc", "AUROC"), ("auprc", "AUPRC"),
                             ("test_f1", "F1_vt")):
            t = ms_table(df, metric)
            lines += [f"### {name}", "", fmt_md(t), ""]
    p2 = os.path.join(BASE, "exp2", "summary", "EXP2_metrics_bin_all.csv")
    if os.path.exists(p2):
        df = pd.read_csv(p2)
        lines += ["## EXP2 全染色体（bin 级）", ""]
        for metric in ("auroc", "auprc"):
            t = ms_table(df, metric)
            lines += [f"### bin {metric.upper()}", "", fmt_md(t), ""]
        dfp = pd.read_csv(os.path.join(BASE, "exp2", "summary",
                                       "EXP2_desert_fp_all.csv"))
        dfp = dfp[dfp.se_recall == 0.8]
        lines += ["### Desert FP/Mb @recall0.8", "",
                  fmt_md(ms_table(dfp, "fp_per_mb")), ""]
        pr = os.path.join(BASE, "exp2", "summary",
                          "EXP2_prcmax_region_metrics_all.csv")
        if os.path.exists(pr):
            dfr = pd.read_csv(pr)
            lines += ["## EXP2 全染色体（region 级）", "",
                      "### Region AUPRC（peak-restricted max，主口径）", "",
                      fmt_md(ms_table(dfr, "auprc")), ""]
        f1p = os.path.join(BASE, "exp2", "summary", "EXP2_union_f1_all.csv")
        if os.path.exists(f1p):
            lines += ["### Region F1（val-tuned，IoU≥0.3）", "",
                      fmt_md(ms_table(pd.read_csv(f1p), "f1")), ""]
        ioup = os.path.join(BASE, "exp2", "summary", "EXP2_region_iou_all.csv")
        if os.path.exists(ioup):
            lines += ["### TP median IoU @recall0.8", "",
                      fmt_md(ms_table(pd.read_csv(ioup), "median_iou_tp")), ""]
    p3 = os.path.join(BASE, "exp3", "summary", "EXP3_prcmax_cluster_metrics_all.csv")
    if os.path.exists(p3):
        df3 = pd.read_csv(p3)
        cells = ["K562", "HepG2", "GM12878", "WTC11"]
        models = ["Cerberus", "DeepSE_official", "SENet",
                  "MuSE_official", "DcSE"]
        lines += ["## EXP3 谱系特异本系对决（5-split mean ± SEM）", "",
                  "单元格 = mean ± SEM（n=5 splits）", ""]
        for metric, name in (("auprc", "AUPRC"), ("auroc", "AUROC"),
                             ("test_f1", "F1 (val-tuned)")):
            head = "| model | " + " | ".join(cells) + " |"
            sep = "|" + "|".join(["---"] * (len(cells) + 1)) + "|"
            body = []
            for m in models:
                row = [m]
                for c in cells:
                    v = df3[(df3.cell == c) & (df3.model == m)][metric].to_numpy(float)
                    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                    row.append(f"{v.mean():.3f} ± {sem:.3f}")
                body.append("| " + " | ".join(row) + " |")
            lines += [f"### {name}", "", head, sep] + body + [""]
        sig3 = os.path.join(BASE, "exp3", "summary",
                            "EXP3_benchmark_significance_tests.csv")
        if os.path.exists(sig3):
            s = pd.read_csv(sig3)
            lines += [f"配对 t 检验：{int((s.p < 0.05).sum())}/{len(s)} 项 "
                      f"p<0.05（最大 p={s.p.max():.2e}）", ""]
        # 交叉热图矩阵（pred × GT，AUPRC 5-split mean）
        cells = ["K562", "HepG2", "GM12878", "WTC11"]
        models = ["Cerberus", "DeepSE_official", "SENet",
                  "MuSE_official", "DcSE"]
        labels = {}
        for split in range(5):
            for pool in cells:
                d = pd.read_csv(os.path.join(BASE, "exp3", f"pool_split{split}",
                                             pool, "EXP3_pool_test.csv"))
                labels[(split, pool)] = (d["kind"] == "SE").to_numpy().astype(int)
        lines += ["### 交叉热图矩阵（pred × GT，AUPRC 5-split mean）", "",
                  "行 = pred 谱系（在该细胞训练的模型 / Cerberus 通道）；"
                  "列 = GT 谱系池；Cerberus 行读 Exp2 全染色体稠密轨道，"
                  "对角与主表可差 ~0.02。", ""]
        for model in models:
            M = np.full((4, 4), np.nan)
            for i, pred in enumerate(cells):
                for j, pool in enumerate(cells):
                    vals = []
                    for split in range(5):
                        pdir = os.path.join(BASE, "exp3",
                                            f"pool_split{split}", pool)
                        f = (os.path.join(pdir, f"EXP3_prcmax_cerberus_x_{pred}_test.npy")
                             if model == "Cerberus" else
                             os.path.join(pdir, f"EXP3_prcmax_baseline_x_{model}_{pred}"
                                                f"_on_{pool}_test.npy"))
                        if not os.path.exists(f):
                            continue
                        sc = np.load(f)
                        y = labels[(split, pool)]
                        ok = np.isfinite(sc)
                        if ok.sum() and 0 < y[ok].sum() < ok.sum():
                            vals.append(average_precision_score(y[ok], sc[ok]))
                    if vals:
                        M[i, j] = float(np.mean(vals))
            t = pd.DataFrame(np.round(M, 3), index=cells, columns=cells)
            t.index.name = f"pred ({model})"
            lines += [f"**{model}**", "", fmt_md(t.reset_index()), ""]
    out = os.path.join(BASE, "summary", "BENCHMARK_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("[save]", out)


if __name__ == "__main__":
    main()
