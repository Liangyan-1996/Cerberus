#!/usr/bin/env python3
"""汇总最终口径（prcmax）下 EXP1 / EXP2 的 PR-AUC(=AP) mean ± SD / SEM（5 splits）。

数据源（均为 5-split 逐 split 行）：
  EXP1 cluster (ROSE)  summary/EXP1_prcmax_cluster_metrics_all.csv      [auprc]
  EXP2 region  (ROSE)  exp2/summary/EXP2_prcmax_region_metrics_all.csv  [auprc]
  EXP2 bin     (256bp) exp2/summary/EXP2_metrics_bin_all.csv            [auprc]（参考口径）

输出：
  summary/PR_AUC_mean_sem.csv   机读表（experiment, level, metric, model, n, mean, sd, sem）
  summary/PR_AUC_mean_sem.md    论文/图注可直接引用的 mean ± SEM 表

注：EXP1 的 PR 图（exp1/fig/EXP1_prcmax_pr.pdf）与 EXP2 region PR 图均由逐 split
曲线插值后取均值绘制；表中 AP 为逐 split AP 的均值（mean），阴影为 SEM。
（exp1/curves/*.npz 是 2026-09-09 旧 full-mean 口径产物，已弃用，勿再引用。）
"""
import os

import numpy as np
import pandas as pd

REPO = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1"
BASE = os.path.join(REPO, "test10_eval_new/20260909.final_benchmark")

SOURCES = [
    ("EXP1", "cluster (ROSE)", "summary/EXP1_prcmax_cluster_metrics_all.csv", "model"),
    ("EXP2", "region (ROSE)", "exp2/summary/EXP2_prcmax_region_metrics_all.csv", "source"),
    ("EXP2", "bin (256bp)", "exp2/summary/EXP2_metrics_bin_all.csv", "source"),
]

MODEL_ORDER = ["Cerberus_union", "DeepSE_official", "SENet", "MuSE_official", "DcSE"]


def main():
    rows = []
    for exp, level, rel, model_col in SOURCES:
        df = pd.read_csv(os.path.join(BASE, rel))
        g = df.groupby(model_col)["auprc"]
        n = g.count()
        mean = g.mean()
        sd = g.std(ddof=1)
        sem = sd / np.sqrt(n)
        order = [m for m in MODEL_ORDER if m in mean.index]
        order += [m for m in mean.index if m not in order]
        for m in order:
            rows.append({
                "experiment": exp,
                "level": level,
                "metric": "PR-AUC (AP)",
                "model": m,
                "n": int(n.loc[m]),
                "mean": round(float(mean.loc[m]), 4),
                "sd": round(float(sd.loc[m]), 4),
                "sem": round(float(sem.loc[m]), 4),
                "mean_sem": f"{mean.loc[m]:.4f} ± {sem.loc[m]:.4f}",
                "mean_sd": f"{mean.loc[m]:.4f} ± {sd.loc[m]:.4f}",
            })
    out = pd.DataFrame(rows)
    csv_path = os.path.join(BASE, "summary", "PR_AUC_mean_sem.csv")
    out.to_csv(csv_path, index=False)

    lines = ["# EXP1 / EXP2 PR-AUC (AP) mean ± SEM（prcmax 最终口径，n = 5 splits）", ""]
    for (exp, level), sub in out.groupby(["experiment", "level"], sort=False):
        lines += [f"## {exp} — {level}", "",
                  "| model | mean ± SEM | n | SD |", "|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['model']} | {r['mean_sem']} | {r['n']} | {r['sd']:.4f} |")
        lines.append("")
    md_path = os.path.join(BASE, "summary", "PR_AUC_mean_sem.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(out.to_string(index=False))
    print(f"\n[write] {csv_path}\n[write] {md_path}")


if __name__ == "__main__":
    main()
