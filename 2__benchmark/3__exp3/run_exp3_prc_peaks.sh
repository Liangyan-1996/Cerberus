#!/usr/bin/env bash
# EXP3 peak-restricted：簇→构成峰映射（20 CPU 任务）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp3_prcPk -p cu,fat -c 4 --array=0-19 \
  -o "${LOGDIR}/EXP3_prcPk_%A_%a.out" -e "${LOGDIR}/EXP3_prcPk_%A_%a.err" \
  --wrap="CELLS=(K562 HepG2 GM12878 WTC11); cd ${BASE} && ${PY} scripts/build_exp3_prc_peaks.py \
    --split \$((SLURM_ARRAY_TASK_ID % 5)) \
    --cell \${CELLS[\$((SLURM_ARRAY_TASK_ID / 5))]}")
echo "${jobid}"
