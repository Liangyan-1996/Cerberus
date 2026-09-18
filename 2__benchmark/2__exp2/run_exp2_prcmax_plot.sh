#!/usr/bin/env bash
# EXP2 prcmax 出图（依赖 5 split 的 prcmax 打分）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
if [ -n "${DEP}" ]; then EXTRA=(--dependency=afterok:${DEP}); fi
jobid=$(sbatch -J final_exp2_prcPlot -p cu,fat -c 4 "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP2_prcPlot_%j.out" -e "${LOGDIR}/EXP2_prcPlot_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/plot_exp2_prcmax.py")
echo "${jobid}"
