#!/usr/bin/env bash
# EXP1 peak-restricted+max：块→构成峰映射（5 split CPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp1_prcPk -p cu,fat -c 4 --mem=24G --array=0-4 \
  -o "${LOGDIR}/EXP1_prcPk_%A_%a.out" -e "${LOGDIR}/EXP1_prcPk_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_exp1_prcmax_peaks.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
