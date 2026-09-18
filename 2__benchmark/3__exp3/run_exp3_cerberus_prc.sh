#!/usr/bin/env bash
# EXP3 peak-restricted cluster：Cerberus 读稠密轨道（5 CPU；依赖峰映射 323509）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}; EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_cerPrc -p cu,fat -c 4 --mem=20G --array=0-4 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_cerPrc_%A_%a.out" -e "${LOGDIR}/EXP3_cerPrc_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_exp3_cerberus_prc_from_tracks.py \
    --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
