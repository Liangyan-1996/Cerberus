#!/usr/bin/env bash
# EXP3 cross：4 模型 × 4 pred 细胞 × 5 split = 80 任务（各自在 4 个 GT 池上评测）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
LOGDIR=${REPO}/logs/20260910
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
jobid=$(sbatch -J final_exp3_bscX -p gpu,gpu03,4090 -G 1 -c 8 --array=0-79 \
  -o "${LOGDIR}/EXP3_bscX_%A_%a.out" -e "${LOGDIR}/EXP3_bscX_%A_%a.err" \
  --wrap="TAGS=(DeepSE_official SENet MuSE_official DcSE); \
    CELLS=(K562 HepG2 GM12878 WTC11); \
    IDX=\${SLURM_ARRAY_TASK_ID}; SP=\$((IDX % 5)); \
    CI=\$(((IDX / 5) % 4)); MI=\$((IDX / 20)); \
    cd ${BASE} && ${PY} scripts/score_exp3_baseline_cross.py \
      --split \${SP} --pred \${CELLS[\$CI]} --tag \${TAGS[\$MI]}")
echo "${jobid}"
