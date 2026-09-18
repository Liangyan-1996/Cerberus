#!/usr/bin/env bash
# EXP1 peak-restricted+max 基线打分（4 模型 × 5 split = 20 GPU；依赖映射作业）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
DEP=${1:-}; EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp1_prcBsc -p gpu,gpu03,4090 -G 1 -c 8 --array=0-19 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP1_prcBsc_%A_%a.out" -e "${LOGDIR}/EXP1_prcBsc_%A_%a.err" \
  --wrap="TAGS=(DeepSE_official SENet MuSE_official DcSE); \
    IDX=\${SLURM_ARRAY_TASK_ID}; SP=\$((IDX % 5)); MI=\$((IDX / 5)); \
    cd ${BASE} && ${PY} scripts/score_exp1_prcmax_baseline.py \
      --split \${SP} --tag \${TAGS[\$MI]}")
echo "${jobid}"
