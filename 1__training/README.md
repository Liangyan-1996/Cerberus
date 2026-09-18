# 1__training — Cerberus training code

## Files

| File (`1__code/`) | Role |
|---|---|
| `se_model_arch.py` | `BorzoiSEAdapter`: Borzoi trunk (frozen or partially unfrozen) + stride-8 downsampler + cluster head + channel-attention fusion + SE head; `ResBlock1d`, `ChannelAttention` |
| `se_data_utils.py` | `GenomicSampler` (anchor universe, per-epoch sampling), `GenomicDataset`, soft-label construction from the ROSE label cache, reverse-complement augmentation, genome memmap loading |
| `se_train_utils.py` | `hierarchical_loss` (two BCE terms), AUPRC utilities, threshold search, `evaluate_unified`, training-curve and PR-curve plotting, per-cell metric tables |
| `train_se.py` | training entry point used for the released human model (fixed chromosome split, two stages via CLI flags) |
| `train_se_split.py` | same loop with parameterised `--val_chroms/--test_chroms/--batch_size`, used for the 5 chromosome-split benchmark models |
| `test_se.py` | standalone evaluation on the held-out chromosomes (AUROC/AUPRC, best-F1 thresholds, per-cell table) |
| `predict_se.py` | inference on user-supplied intervals with a trained checkpoint |
| `borzoi_pytorch/` | vendored Borzoi model code (config, model, utils) required by `se_model_arch.py` |

Environment: see `environment_borzoi-pytorch.versions.txt` (pip freeze in
`environment_borzoi-pytorch.pip.txt`).

## Required inputs

Paths are set in the `CONFIGS` dictionary at the top of `train_se.py` / `train_se_split.py`.

| Input | Description |
|---|---|
| `fasta` | reference genome (`hg38.fa`, plus per-chromosome `.npy` memmaps built on first use) |
| `chrom_sizes` | chromosome sizes file (defines which chromosomes exist) |
| `pool_file` | `AllSample.hg38.AnchorPool.parquet`: every ROSE stitched region of every channel with its `isSuper` flag |
| `label_cache` | `AllSample.hg38.SE.label_cache.pkl` + `.metadata.pkl`: per-channel, per-chromosome interval annotations (`Cluster`, `SuperEnhancer`) and the 90-channel index |
| `blacklist` | hg38 blacklist BED (background anchors are filtered against it) |
| backbone | pretrained Borzoi (`borzoi-replicate-0`), path in `HF_MODEL_ID` |

The `.pkl`/`.parquet` annotation files are derived from the ROSE tables described in the manuscript
Methods (90 ENCODE H3K27ac datasets, MACS2 + ROSE with 12.5 kb stitching).

## Two-stage training protocol

Both stages use the same data pipeline, loss and chromosome split (train = 18 autosomes,
validation = chr9 + chr14, test = chr3 + chr11), and both keep the checkpoint with the best
validation SE AUPRC.

1. **Stage 1 — head-only.** The pretrained Borzoi trunk is loaded and frozen; only the randomly
   initialised adapter (downsampler, cluster head, fusion attention, SE head; 102 tensors) is trained,
   with a comparatively large learning rate. 54,000 anchors per epoch (up to 200 SE and 200 non-SE
   stitched-region anchors per channel, plus half as many blacklist-free background anchors).
2. **Stage 2 — low-learning-rate fine-tuning.** Training resumes from the stage-1 checkpoint and the
   **last transformer layer of the trunk plus the decoder blocks are unfrozen** (38 trunk tensors
   become trainable alongside the 102 adapter tensors); the learning rate is reduced by two orders of
   magnitude so that the pretrained representation is refined rather than overwritten.

`2__run_examples/1__stage1_head_only.sh` and `2__run_examples/2__stage2_finetune.sh` reproduce the two
stages. Learning-rate / patience values in those scripts follow the protocol recorded in
`2__benchmark/Benchmark_PLAN.md`; the values actually passed in the original jobs are listed in the
manuscript Methods.

## Outputs

`<save_dir>/Cerberus_human_best.pth` (best validation checkpoint), `anchors_epoch_*.csv` (the anchors
sampled in each epoch, useful for auditing the sampler), `training_curves.png`, `best_val_pr_curve.png`,
and — after `test_se.py` — the per-cell metric table and the global PR curve.
