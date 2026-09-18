#!/usr/bin/env bash
# EXP1 peak-restricted+max Cerberus（稠密轨道，5 CPU；依赖映射作业）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
if [ -n "${DEP}" ]; then EXTRA=(--dependency=afterok:${DEP}); fi
jobid=$(sbatch -J final_exp1_prcCer -p cu,fat -c 4 --mem=20G --array=0-4 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP1_prcCer_%A_%a.out" -e "${LOGDIR}/EXP1_prcCer_%A_%a.err" \
  --wrap="cd ${BASE} && ${PY} scripts/build_exp1_prcmax_cerberus_from_tracks.py \
    --split \${SLURM_ARRAY_TASK_ID}")
echo "${jobid}"
