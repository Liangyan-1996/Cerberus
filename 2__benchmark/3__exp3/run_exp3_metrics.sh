#!/usr/bin/env bash
# EXP3 S7c3：评分完成后汇总/显著性/图（CPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_metrics -p cu,fat -c 4 "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_metrics_%j.out" -e "${LOGDIR}/EXP3_metrics_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_plot_exp3.py")
echo "${jobid}"
