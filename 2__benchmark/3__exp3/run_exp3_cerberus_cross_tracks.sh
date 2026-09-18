#!/usr/bin/env bash
# EXP3 Cerberus cross（稠密轨道版，零推理）：5 split CPU 任务
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp3_cerT -p cu,fat -c 4 --mem=16G --array=0-4 \
  -o "${LOGDIR}/EXP3_cerT_%A_%a.out" -e "${LOGDIR}/EXP3_cerT_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_exp3_cerberus_cross_from_tracks.py \
    --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
