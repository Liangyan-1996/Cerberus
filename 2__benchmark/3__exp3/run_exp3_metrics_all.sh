#!/usr/bin/env bash
# EXP3（修正后）：cluster 指标+热图（CPU；依赖打分完成）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_metricsAll -p cu,fat -c 4 "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_metricsAll_%j.out" -e "${LOGDIR}/EXP3_metricsAll_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_plot_exp3.py && \
          ${PY} scripts/plot_exp3_cross_heatmaps.py && \
          ${PY} scripts/write_final_report.py")
echo "${jobid}"
