#!/usr/bin/env bash
# EXP3 cross 热图（依赖 322853 Cerberus + 322854 基线 cross 打分）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_heatX -p cu,fat -c 4 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_heatX_%j.out" -e "${LOGDIR}/EXP3_heatX_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/plot_exp3_cross_heatmaps.py")
echo "${jobid}"
