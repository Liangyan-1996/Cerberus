#!/usr/bin/env bash
# EXP1 S3：基线簇级打分（5 splits × 4 models = 20，GPU，零重训）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
TAGS=(DeepSE_official SENet MuSE_official DcSE)
jobid=$(sbatch -J final_exp1_s3 -p gpu03,gpu -G 1 -c 8 --array=0-19 \
  -o "${LOGDIR}/EXP1_s3_%A_%a.out" -e "${LOGDIR}/EXP1_s3_%A_%a.err" \
  --wrap="TAGS=(DeepSE_official SENet MuSE_official DcSE); cd ${BASE} && ${PY} scripts/score_baseline_clusters.py \
    --split \$((SLURM_ARRAY_TASK_ID / 4)) \
    --tag \${TAGS[\$((SLURM_ARRAY_TASK_ID % 4))]}")
echo "${jobid}"
