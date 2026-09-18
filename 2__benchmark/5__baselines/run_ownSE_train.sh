#!/usr/bin/env bash
# 用自己的 SE annotation 数据重训四个基线（chr-split, val 早停）
set -euo pipefail

REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
BDIR=${REPO}/test10_eval_new/20260905.model_benchmark/benchmark
ADIR=${REPO}/test10_eval_new/20260906.ownSE_annotation__benchmark
LOGDIR=${REPO}/logs/20260906
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
mkdir -p "${LOGDIR}"

submit() {
    local tag=$1; shift
    local jobid
    jobid=$(sbatch -J "ownSE_${tag}" -p H100,gpu03,gpu -G 1 -c 8 \
        -o "${LOGDIR}/ownSE_train_${tag}_%j.out" -e "${LOGDIR}/ownSE_train_${tag}_%j.err" \
        --wrap="cd ${BDIR} && ${PY} train_baseline.py $*")
    jobid=${jobid##* }
    echo -e "${tag}\t${jobid}\t$(date '+%F %T')" >> "${ADIR}/submitted_jobs.tsv"
    echo "[submitted] ${tag} -> ${jobid}"
}

TRAIN=${ADIR}/data/train.csv
VAL=${ADIR}/data/val.csv

submit DeepSE_orig --model DeepSE --train "${TRAIN}" --val "${VAL}" --out_dir "${ADIR}/ckpt/DeepSE_orig" --kmer_npz kmer_deepse_hg19.npz --epochs 80 --lr 1e-3 --batch_size 64
submit SENet       --model SENet  --train "${TRAIN}" --val "${VAL}" --out_dir "${ADIR}/ckpt/SENet"       --epochs 50 --lr 5e-4 --batch_size 64
submit MuSE_orig   --model MuSE   --train "${TRAIN}" --val "${VAL}" --out_dir "${ADIR}/ckpt/MuSE_orig"   --kmer_npz kmer_deepse_hg19.npz --epochs 80 --lr 1e-5 --batch_size 64
submit DcSE        --model DcSE   --train "${TRAIN}" --val "${VAL}" --out_dir "${ADIR}/ckpt/DcSE"        --epochs 30 --lr 1e-4 --batch_size 64

echo "[done] 4 jobs submitted, see ${ADIR}/submitted_jobs.tsv"
