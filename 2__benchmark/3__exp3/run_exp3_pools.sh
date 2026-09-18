#!/usr/bin/env bash
# EXP3 S7c0：20 个评估池（CPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp3_pools -p cu,fat -c 4 --array=0-19 \
  -o "${LOGDIR}/EXP3_pools_%A_%a.out" -e "${LOGDIR}/EXP3_pools_%A_%a.err" \
  --wrap="CELLS=(K562 HepG2 GM12878 WTC11); cd ${BASE} && ${PY} scripts/build_exp3_eval_pools.py \
    --split \$((SLURM_ARRAY_TASK_ID % 5)) \
    --cell \${CELLS[\$((SLURM_ARRAY_TASK_ID / 5))]}")
echo "${jobid}"
