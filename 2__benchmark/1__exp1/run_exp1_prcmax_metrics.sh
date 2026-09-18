#!/usr/bin/env bash
# EXP1 peak-restricted+max 指标/图（CPU；依赖打分完成）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
if [ -n "${DEP}" ]; then EXTRA=(--dependency=afterok:${DEP}); fi
jobid=$(sbatch -J final_exp1_prcMet -p cu,fat -c 4 "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP1_prcMet_%j.out" -e "${LOGDIR}/EXP1_prcMet_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_plot_exp1_prcmax.py")
echo "${jobid}"
