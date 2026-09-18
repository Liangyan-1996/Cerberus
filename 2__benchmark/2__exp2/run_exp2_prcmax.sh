#!/usr/bin/env bash
# EXP2 region 级 peak-restricted+max（5 CPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp2_prcMax -p cu,fat -c 4 --mem=24G --array=0-4 \
  -o "${LOGDIR}/EXP2_prcMax_%A_%a.out" -e "${LOGDIR}/EXP2_prcMax_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_exp2_region_prcmax.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
