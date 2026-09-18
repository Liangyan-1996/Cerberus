# 2__benchmark — final benchmark (Exp1 / Exp2 / Exp3)

Only the three experiments reported in the manuscript are archived here. Earlier protocol variants
(full-mean / prc-mean aggregation, peak-level scoring, the `former_benchmark` tree) are intentionally
left out; the frozen protocol of the final version is `Benchmark_PLAN.md`.

## The three experiments

| Exp | Question | Unit of evaluation | Primary metric |
|---|---|---|---|
| **Exp1** (`1__exp1/`) | Can the model discriminate SE-constituent peaks from typical-enhancer peaks? | 3 kb fragment, balanced test set | AUPRC (also AUROC, val-tuned F1) |
| **Exp2** (`2__exp2/`) | How well does the model detect SEs genome-wide? | 256 bp bins and ROSE GT regions, whole test chromosomes | region AUPRC; also bin AUROC/AUPRC, region-detection F1, Desert FP/Mb, median TP IoU |
| **Exp3** (`3__exp3/`) | Is the model lineage specific? | per-cell clusters for 4 cells (GM12878, HepG2, K562, WTC11) × 5 splits | AUPRC (also AUROC, val-tuned F1) |

Common conventions (details in `Benchmark_PLAN.md`):

- **5 chromosome splits**, one model trained per split; val→test chromosome pairs:
  s0 9/14→3/11, s1 1/15→2/12, s2 4/18→5/17, s3 6/19→7/20, s4 8/21→10/13.
- Each split's Cerberus is trained **from the pretrained Borzoi backbone with randomly initialised
  heads**, never from the released checkpoint, so no test chromosome is seen during training.
- **prcmax is the primary aggregation** (peak-restricted cluster aggregation by max); per-split
  prcmax tables are the numbers quoted in the manuscript.
- All model comparisons use a **two-tailed paired t-test across the five splits (df = 4)**.

## Directory contents

| Directory | Contents |
|---|---|
| `1__exp1/` | union/cluster pool construction, Cerberus and baseline scoring, metric assembly, Exp1 figures |
| `2__exp2/` | region-level evaluation (`eval_exp2_region*.py`), gap/desert masks, Desert FP, summarising and plotting |
| `3__exp3/` | Exp3 dataset construction, per-cell scoring for Cerberus and the four baselines, cross-cell heatmaps |
| `4__shared/` | `significance_tests.py`, `summary_pr_auc_mean_sem.py`, `write_final_report.py`, `final_style.py` (single source of figure style) |
| `5__baselines/` | `build_own_se_dataset.py` (ownSE data construction), `train_baseline.py`, model definitions (`models_baseline.py`, `models_dcse.py`, `CBAM.py`, `DenseNet.py`, `dynamic_tanh.py`, `features.py`), k-mer resources, and the Slurm wrappers `run_ownSE_train.sh` / `run_build_own_se.sh` |
| `6__results/` | aggregate result tables (metrics per experiment, Desert FP, IoU, F1), the paired-test tables, `PR_AUC_mean_sem.csv`, `BENCHMARK_REPORT.md`, and `EXP1-3_significance_tests.xlsx` (multi-sheet significance workbook) |

## Running the benchmark

The `run_*.sh` scripts are the original Slurm drivers; each one encodes the working directory, the
conda environment and the job dependencies of that step. Typical end-to-end order per experiment:

```bash
# Exp1: pool -> Cerberus / baseline scoring -> metrics -> figures
bash 1__exp1/run_exp1_pool.sh
bash 1__exp1/run_exp1_prcmax_cerberus.sh && bash 1__exp1/run_exp1_prcmax_baseline.sh
bash 1__exp1/run_exp1_prcmax_metrics.sh
# Exp2: masks -> region evaluation -> summary/plots
bash 2__exp2/run_exp2_gap.sh && bash 2__exp2/run_exp2_region.sh
bash 2__exp2/run_exp2_prcmax.sh && bash 2__exp2/run_exp2_prcmax_plot.sh
# Exp3: dataset -> training -> per-cell scoring -> heatmaps
bash 3__exp3/run_exp3_data.sh && bash 3__exp3/run_exp3_train.sh
bash 3__exp3/run_exp3_cerberus_prcmax.sh && bash 3__exp3/run_exp3_baseline_prcmax.sh
bash 3__exp3/run_exp3_metrics_all.sh && bash 3__exp3/run_exp3_cross_heatmaps.sh
```

Paths inside these scripts are absolute cluster paths; adjust `REPO` and `PY` before running.

## Baseline training

Four baselines are trained per split on the **ownSE dataset** built from this project's own annotation
(SE-constituent H3K27ac peaks as positives, pure typical-enhancer peaks as negatives, 3 kb windows):

```bash
PY=python                      # environment with PyTorch
python train_baseline.py --model DeepSE --train data/train.csv --val data/val.csv \
    --out_dir ckpt/DeepSE_official --kmer_npz kmer_official_k3to8.npz \
    --epochs 80 --lr 1e-3 --batch_size 64
# --model in {DeepSE, SENet, MuSE, DcSE}; see run_ownSE_train.sh for the exact per-model settings
```

Sequence k-mer/w2v embeddings follow the official `dna2vec k3to8` vectors
(`kmer_official_k3to8.npz`); `kmer_deepse_hg19.npz` is kept only for the legacy `*_orig` variants that
are not part of the final benchmark.

## Statistical testing

`4__shared/significance_tests.py` performs the paired comparison (Cerberus vs each baseline, two-tailed
paired t-test over the five splits, df = 4). The resulting tables and the consolidated workbook live in
`6__results/`:

- `EXP1_prcmax_significance_tests.csv` (12 tests: 3 metrics × 4 baselines)
- `EXP2_prcmax_significance_tests.csv` (4 tests, region AUPRC)
- `EXP3_prcmax_significance_tests.csv` (48 tests: 4 cells × 3 metrics × 4 baselines)
- `EXP1-3_significance_tests.xlsx`: README sheet, one sheet per experiment, combined long table,
  mean ± SEM table, and the legacy-protocol sheets kept for provenance
