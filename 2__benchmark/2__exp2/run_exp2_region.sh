#!/usr/bin/env bash
# EXP2 S6c：region 级指标（5 splits，CPU；依赖新 desert mask）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp2_region -p cu,fat -c 8 --array=0-4 \
  --dependency=afterok:322147 \
  -o "${LOGDIR}/EXP2_region_%A_%a.out" -e "${LOGDIR}/EXP2_region_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/eval_exp2_region.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
