#!/usr/bin/env python3
"""EXP1 S4b：5-split 配对 t 检验（Cerberus_union vs 各基线）导出附表。

输入：summary/EXP1_cluster_metrics_all.csv
输出：summary/EXP1_benchmark_significance_tests.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = "/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1/" \
       "test10_eval_new/20260909.final_benchmark"
METRICS = ["auroc", "auprc", "test_f1"]
BASELINES = ["DeepSE_official", "SENet", "MuSE_official", "DcSE"]


def main():
    df = pd.read_csv(os.path.join(BASE, "summary", "EXP1_cluster_metrics_all.csv"))
    rows = []
    for metric in METRICS:
        a = df[df.model == "Cerberus_union"][["split", metric]].set_index("split")[metric]
        for b in BASELINES:
            bv = df[df.model == b][["split", metric]].set_index("split")[metric]
            common = a.index.intersection(bv.index).sort_values()
            x, y = a.loc[common].to_numpy(), bv.loc[common].to_numpy()
            if len(x) < 3:
                rows.append({"metric": metric, "model_a": "Cerberus_union",
                             "model_b": b, "n_folds": int(len(x)),
                             "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                             "t": float("nan"), "df": len(x) - 1,
                             "p": float("nan")})
                continue
            t, p = stats.ttest_rel(x, y)
            rows.append({"metric": metric, "model_a": "Cerberus_union",
                         "model_b": b, "n_folds": int(len(x)),
                         "mean_a": round(float(x.mean()), 4),
                         "mean_b": round(float(y.mean()), 4),
                         "delta": round(float(x.mean() - y.mean()), 4),
                         "t": round(float(t), 4), "df": len(x) - 1,
                         "p": round(float(p), 5)})
    out = pd.DataFrame(rows)
    p_vals = out["p"].astype(float)
    out["sig"] = p_vals.apply(lambda x: "***" if x < 0.001
                              else ("**" if x < 0.01 else ("*" if x < 0.05 else "")))
    path = os.path.join(BASE, "summary", "EXP1_benchmark_significance_tests.csv")
    out.to_csv(path, index=False)
    print(f"[save] {path}\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
