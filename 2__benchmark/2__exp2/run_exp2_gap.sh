#!/usr/bin/env bash
# EXP2 S6a：N/gap 掩码（CPU，单任务）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp2_gap -p cu,fat -c 4 \
  -o "${LOGDIR}/EXP2_gap_%j.out" -e "${LOGDIR}/EXP2_gap_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_gap_mask.py")
echo "${jobid}"
