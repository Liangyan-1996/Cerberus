#!/usr/bin/env bash
# EXP2 S6d：汇总 + 最终图（依赖 region 322154 与 Exp1 无依赖）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp2_summary -p cu,fat -c 4 \
  --dependency=afterok:322154 \
  -o "${LOGDIR}/EXP2_summary_%j.out" -e "${LOGDIR}/EXP2_summary_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/summarize_plot_exp2.py")
echo "${jobid}"
