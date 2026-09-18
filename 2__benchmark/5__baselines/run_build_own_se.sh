#!/usr/bin/env bash
# 用自己的 SE annotation（90 channel ROSE + H3K27ac peaks）构建 chr-split 数据集
set -euo pipefail

REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
ADIR=${REPO}/test10_eval_new/20260906.ownSE_annotation__benchmark
LOGDIR=${REPO}/logs/20260906
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"

jobid=$(sbatch -J ownSE_build -p cu,fat -c 8 \
    -o "${LOGDIR}/ownSE_build_%j.out" -e "${LOGDIR}/ownSE_build_%j.err" \
    --wrap="cd ${ADIR} && ${PY} build_own_se_dataset.py --out_dir data --cap_per_class 30000")
echo "[submitted] ownSE_build -> ${jobid##* }"
