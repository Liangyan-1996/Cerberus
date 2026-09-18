#!/usr/bin/env bash
# EXP3 cross：Cerberus 4 通道 × 4 GT 池打分（5 split × 4 pool = 20 任务）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp3_cerX -p gpu03,gpu -G 1 -c 8 --array=0-19 \
  -o "${LOGDIR}/EXP3_cerX_%A_%a.out" -e "${LOGDIR}/EXP3_cerX_%A_%a.err" \
  --wrap="POOLS=(K562 HepG2 GM12878 WTC11); cd ${BASE} && ${PY} scripts/score_exp3_cerberus_cross.py \
    --split \$((SLURM_ARRAY_TASK_ID % 5)) \
    --pool \${POOLS[\$((SLURM_ARRAY_TASK_ID / 5))]}")
echo "${jobid}"
