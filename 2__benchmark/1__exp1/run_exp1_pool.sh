#!/usr/bin/env bash
# EXP1 S1：5-split union 簇级样本池（CPU，array 0-4）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp1_pool -p cu,fat -c 4 --array=0-4 \
  -o "${LOGDIR}/EXP1_pool_%A_%a.out" -e "${LOGDIR}/EXP1_pool_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_union_cluster_pool.py --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
