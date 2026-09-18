#!/usr/bin/env bash
# Stage 2 of the two-stage protocol: resume from the stage-1 checkpoint, unfreeze the last transformer
# layer of the Borzoi trunk plus the decoder blocks, and fine-tune with a much smaller learning rate.
#
# Hyper-parameters follow the protocol recorded in 2__benchmark/Benchmark_PLAN.md (stage B);
# the values actually used for the released model are listed in the manuscript Methods.
set -euo pipefail

REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
CODE=${REPO}/test10_eval_new/paper_write/code_release/1__training/1__code
STAGE1=${REPO}/test10_eval_new/paper_write/code_release/1__training/_output_stage1/Cerberus_human_best.pth
OUT=${REPO}/test10_eval_new/paper_write/code_release/1__training/_output_stage2
mkdir -p "${OUT}"

"${PY}" "${CODE}/train_se.py" \
  --species human \
  --save_dir "${OUT}" \
  --load_checkpoint "${STAGE1}" \
  --unfreeze_attn_layers 1 \
  --unfreeze_decoder \
  --lr_head 1e-6 \
  --lr_borzoi_ratio 0.2 \
  --lambda_cluster 0.25 \
  --head_dropout 0.5 \
  --block_dropout 0.2 \
  --estimated_stop_epoch 20 \
  --patience 8

# --unfreeze_attn_layers 1  -> last transformer layer of the trunk becomes trainable (+ decoder)
# --lr_borzoi_ratio 0.2     -> trunk learning rate = 0.2 x head learning rate
# Output: ${OUT}/Cerberus_human_best.pth
#
# For the 5 chromosome splits (benchmark) use the parameterised entry point instead:
#   ${PY} ${CODE}/train_se_split.py --species human --save_dir <out> \
#       --val_chroms chr9 chr14 --test_chroms chr3 chr11 --batch_size 12 ... (same stage flags)
