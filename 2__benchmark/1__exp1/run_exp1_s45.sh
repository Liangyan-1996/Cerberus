#!/usr/bin/env bash
# EXP1 S4+S5：打分完成后自动：指标/曲线缓存 -> 显著性 -> 最终图（CPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp1_s45 -p cu,fat -c 4 \
  --dependency=afterok:322109:322136 \
  -o "${LOGDIR}/EXP1_s45_%j.out" -e "${LOGDIR}/EXP1_s45_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_cluster_union.py && \
          ${PY} scripts/significance_tests.py && \
          ${PY} scripts/plot_exp1_final.py && \
          ${PY} scripts/write_final_report.py")
echo "${jobid}"
