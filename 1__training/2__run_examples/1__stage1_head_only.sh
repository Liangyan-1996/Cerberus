#!/usr/bin/env bash
# Stage 1 of the two-stage protocol: the pretrained Borzoi trunk is frozen and only the randomly
# initialised adapter (downsampler + cluster head + fusion attention + SE head) is trained.
#
# Hyper-parameters follow the protocol recorded in 2__benchmark/Benchmark_PLAN.md (stage A);
# the values actually used for the released model are listed in the manuscript Methods.
set -euo pipefail

REPO=/public/home/liunangroup/liangyan/project/SE_model/20251227_experiment1
PY=/public/home/liunangroup/liangyan/miniforge3/envs/borzoi-pytorch/bin/python
CODE=${REPO}/test10_eval_new/paper_write/code_release/1__training/1__code
OUT=${REPO}/test10_eval_new/paper_write/code_release/1__training/_output_stage1
mkdir -p "${OUT}"

"${PY}" "${CODE}/train_se.py" \
  --species human \
  --save_dir "${OUT}" \
  --lr_head 1e-4 \
  --lr_borzoi_ratio 0.01 \
  --lambda_cluster 0.25 \
  --head_dropout 0.5 \
  --block_dropout 0.2 \
  --estimated_stop_epoch 6 \
  --patience 3

# Trunk frozen since --unfreeze_attn_layers is not given (default 0) and --unfreeze_decoder omitted.
# Output: ${OUT}/Cerberus_human_best.pth (= stage-1 weights used to initialise stage 2)
