#!/usr/bin/env bash
# EXP2 S6b：5 split desert 重算 + bin 指标（依赖 gap 掩码 322128）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp2_desert -p cu,fat -c 8 --array=0-4 \
  \
  -o "${LOGDIR}/EXP2_desert_%A_%a.out" -e "${LOGDIR}/EXP2_desert_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/update_desert_metrics.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
