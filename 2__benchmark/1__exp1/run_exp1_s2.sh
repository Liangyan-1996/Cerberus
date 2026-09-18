#!/usr/bin/env bash
# EXP1 S2：Cerberus 簇级打分（5 splits，GPU）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp1_s2 -p gpu03,gpu -G 1 -c 8 --array=0-4 \
  -o "${LOGDIR}/EXP1_s2_%A_%a.out" -e "${LOGDIR}/EXP1_s2_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/score_cerberus_clusters.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
