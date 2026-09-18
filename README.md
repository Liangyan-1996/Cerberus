# Cerberus

**Clustered Enhancer Recognition via Borzoi-Embedded Representations of Underlying Sequences**

Cerberus predicts super-enhancers (SEs) directly from DNA sequence. It couples the pretrained Borzoi
backbone to a resolution-matching downsampler, a cluster (constituent-enhancer) head and a
cluster-conditioned SE head, and is trained jointly on 90 ENCODE H3K27ac channels annotated with
MACS2 + ROSE. This repository contains the model training code and the code for the three-experiment
benchmark against four SE-calling baselines (DeepSE, SENet, MuSE, DcSE).

**Model weights are not stored in git** — they are archived on Zenodo together with the full code
release: `[TODO: Zenodo DOI 10.5281/zenodo.XXXXXXX]`. See [`3__weights/README.md`](3__weights/README.md)
for the file list and `3__weights/weights_manifest.csv` for per-file sha256 digests.

## Layout

```
.
├── 1__training/                     # model training (Cerberus)
│   ├── 1__code/                     # model, data pipeline, training/eval entry points
│   ├── 2__run_examples/             # the two training stages as runnable example scripts
│   └── environment_borzoi-pytorch.* # python/package versions of the environment used
├── 2__benchmark/                    # benchmark code (final experiments 1-3 only)
│   ├── Benchmark_PLAN.md            # frozen protocol document for the final benchmark
│   ├── 1__exp1/  2__exp2/  3__exp3/ # per-experiment driver scripts
│   ├── 4__shared/                   # style, statistics, summary utilities
│   ├── 5__baselines/                # four baselines: training code + ownSE data builder
│   └── 6__results/                  # aggregate result tables and the significance workbook
└── 3__weights/                      # README + manifest (checkpoints are on Zenodo)
```

Sub-directories are numbered so that `ls` shows them in reading order.

## Model training

`1__training/1__code/` contains the model (`se_model_arch.py`), the anchor sampler / soft-label
pipeline (`se_data_utils.py`), the hierarchical loss and evaluation utilities (`se_train_utils.py`),
and the training entry points (`train_se.py` for the fixed split, `train_se_split.py` for arbitrary
chromosome splits).

Cerberus is trained in **two stages**: (i) the pretrained Borzoi trunk is frozen and only the randomly
initialised heads are trained with a comparatively large learning rate; (ii) training resumes from
that checkpoint with the **last transformer layer and the decoder unfrozen** and a learning rate two
orders of magnitude smaller. Both stages use the same data pipeline, loss and chromosome split, and
keep the checkpoint with the best validation SE AUPRC. Reproduce with:

```bash
bash 1__training/2__run_examples/1__stage1_head_only.sh
bash 1__training/2__run_examples/2__stage2_finetune.sh
```

Training requires the ROSE annotation pool (`AllSample.hg38.AnchorPool.parquet`), the label cache
(`AllSample.hg38.SE.label_cache.pkl` + `.metadata.pkl`), the reference genome and the blacklist; the
paths are set in `CONFIGS` at the top of the training scripts. See
[`1__training/README.md`](1__training/README.md) for details and for the pretrained Borzoi backbone
(`borzoi-replicate-0`, downloaded separately from Hugging Face).

## Benchmark

Three experiments are archived, exactly as reported in the manuscript (`2__benchmark/Benchmark_PLAN.md`
is the frozen protocol):

| Exp | Question | Unit | Primary metric |
|---|---|---|---|
| Exp1 | SE-constituent vs typical-enhancer peaks | 3 kb fragment, balanced test set | AUPRC (AUROC, val-tuned F1) |
| Exp2 | genome-wide SE detection | 256 bp bins / ROSE GT regions | region AUPRC (bin metrics, region F1, Desert FP/Mb, median IoU) |
| Exp3 | lineage specificity | per-cell clusters, 4 cells × 5 splits | AUPRC (AUROC, val-tuned F1) |

All comparisons use a **two-tailed paired t-test across five chromosome splits (df = 4)**; `prcmax` is
the primary aggregation. Aggregate tables, the paired-test results and the significance workbook are in
[`2__benchmark/6__results/`](2__benchmark/6__results). Baseline checkpoints (four methods × five
splits) are part of the Zenodo archive.

## Weights

Download the checkpoints from Zenodo (`[TODO: DOI]`) and verify them against
`3__weights/weights_manifest.csv`. Loading:

```python
import torch
from Cerberus.se_model_arch import BorzoiSEAdapter          # 1__training/1__code/

model = BorzoiSEAdapter("/path/to/borzoi-replicate-0", num_cell_types=90)
model.load_state_dict(torch.load("3__weights/1__cerberus/cerberus_human_best.pth",
                                 map_location="cpu", weights_only=True))
model.eval()
```

| File (Zenodo archive) | Role | Size |
|---|---|---|
| `3__weights/1__cerberus/cerberus_human_best.pth` | released human model (identical to split0) | 822 MB |
| `3__weights/1__cerberus/cerberus_split1..4.pth` | chromosome-split models of the 5-split benchmark | 822 MB each |
| `3__weights/2__baselines/ckpt_split{0..4}/…` | DeepSE / SENet / MuSE / DcSE checkpoints (+ training history) | 4-15 MB each |

The intermediate stage-1 (head-only) checkpoint is not part of the release.

## Environment

Python 3.13.7, PyTorch 2.5.1, transformers 4.57.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3 — full
package list in `1__training/environment_borzoi-pytorch.pip.txt`. Training and genome-wide scoring ran
on H100 / A100 nodes via Slurm; sequence-attribution analyses additionally used `captum`, `deeplift`
and `modisco-lite`.

## Citation

If you use this code or the released weights, please cite the manuscript `[TODO: title, journal, DOI]`
and the Zenodo archive (`[TODO: DOI]`). Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## Licence

MIT — see [`LICENSE`](LICENSE). The pretrained Borzoi backbone is third-party software and remains
under its own licence; cite Linder et al. and follow the model card when redistributing it.
