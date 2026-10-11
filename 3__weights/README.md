# 3__weights — model checkpoints (hosted on Zenodo)

The checkpoints are **not** stored in this git repository. They are archived on Zenodo together with
the full code release; the Zenodo DOI is [10.5281/zenodo.22840452](https://doi.org/10.5281/zenodo.22840452).

| File (in the Zenodo archive, under `3__weights/`) | Role |
|---|---|
| `1__cerberus/cerberus_human_best.pth` | released human model (identical to split0) |
| `1__cerberus/cerberus_split1..4.pth` | chromosome-split models of the 5-split benchmark |
| `2__baselines/ckpt_split{0..4}/…` | DeepSE / SENet / MuSE / DcSE checkpoints (+ training history) |

`weights_manifest.csv` lists byte size and sha256 for every file; verify a downloaded checkpoint with

```bash
grep cerberus_human_best weights_manifest.csv     # path,size,sha256
sha256sum /path/to/cerberus_human_best.pth        # compare the digest
```

Loading:

```python
import torch
from Cerberus.se_model_arch import BorzoiSEAdapter          # see 1__training/1__code/
model = BorzoiSEAdapter("/path/to/borzoi-replicate-0", num_cell_types=90)
model.load_state_dict(torch.load("3__weights/1__cerberus/cerberus_human_best.pth",
                                 map_location="cpu", weights_only=True))
```
