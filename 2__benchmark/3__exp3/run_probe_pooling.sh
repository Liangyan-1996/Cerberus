#!/usr/bin/env bash
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J probe_pool -p gpu03,gpu,4090 -G 1 -c 4 \
  -o "${LOGDIR}/probe_pool_%j.out" -e "${LOGDIR}/probe_pool_%j.err" \
  --wrap="cd ${BASE} && ${PY} scripts/probe_exp3_pooling.py --split 0 --pred K562 --tag DeepSE_official --pool K562")
echo "${jobid}"
