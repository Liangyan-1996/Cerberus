#!/usr/bin/env bash
# EXP3 S7b：80 次谱系特异基线重训（4 模型 × 4 细胞 × 5 splits）
set -euo pipefail
REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BASE=${REPO}/test10_eval_new/20260909.final_benchmark
BDIR=${REPO}/test10_eval_new/20260905.model_benchmark/benchmark
LOGDIR=${REPO}/logs/20260909
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"
TAGS=(DeepSE_official SENet MuSE_official DcSE)
CELLS=(K562 HepG2 GM12878 WTC11)
MODELS=(DeepSE SENet MuSE DcSE)
DEP=${1:-}
EXTRA=()
[ -n "${DEP}" ] && EXTRA=(--dependency=afterok:${DEP})
jobid=$(sbatch -J final_exp3_train -p gpu,gpu03,4090 -G 1 -c 8 --array=0-79 \
  "${EXTRA[@]}" \
  -o "${LOGDIR}/EXP3_train_%A_%a.out" -e "${LOGDIR}/EXP3_train_%A_%a.err" \
  --wrap="TAGS=(DeepSE_official SENet MuSE_official DcSE); \
    CELLS=(K562 HepG2 GM12878 WTC11); MODELS=(DeepSE SENet MuSE DcSE); \
    LR=(1e-3 5e-4 1e-5 1e-4); EP=(80 50 80 30); \
    IDX=\${SLURM_ARRAY_TASK_ID}; SP=\$((IDX % 5)); \
    CI=\$(((IDX / 5) % 4)); MI=\$((IDX / 20)); \
    TAG=\${TAGS[\$MI]}; CELL=\${CELLS[\$CI]}; M=\${MODELS[\$MI]}; \
    mkdir -p ${BASE}/exp3/ckpt_split\${SP}/\${CELL}/\${TAG}; \
    cd ${BDIR} && ${PY} train_baseline.py --model \${M} \
      --train ${BASE}/exp3/data_split\${SP}/\${CELL}/train.csv \
      --val ${BASE}/exp3/data_split\${SP}/\${CELL}/val.csv \
      --out_dir ${BASE}/exp3/ckpt_split\${SP}/\${CELL}/\${TAG} \
      --epochs \${EP[\$MI]} --lr \${LR[\$MI]} --batch_size 64")
echo "${jobid}"
