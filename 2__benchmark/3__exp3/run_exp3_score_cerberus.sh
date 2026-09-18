#!/usr/bin/env bash
# EXP3 S7c1：Cerberus 谱系单通道打分（20 任务，依赖 pool 322328）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}
EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_cer -p gpu03,gpu -G 1 -c 8 --array=0-19 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_cer_%A_%a.out" -e "${LOGDIR}/EXP3_cer_%A_%a.err" \
  --wrap="CELLS=(K562 HepG2 GM12878 WTC11); cd ${BASE} && ${PY} scripts/score_exp3_cerberus.py \
    --split \$((SLURM_ARRAY_TASK_ID % 5)) \
    --cell \${CELLS[\$((SLURM_ARRAY_TASK_ID / 5))]}")
echo "${jobid}"
