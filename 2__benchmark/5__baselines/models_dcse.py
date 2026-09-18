"""DcSE 模型（BMC Genomics 2026，按官方仓库实现，去除 argparse 副作用）。

架构：Embedding(6mer,100d,可微调) -> DenseNet1D(growth32,2x5) + 动态 CBAM
      -> BN -> FC(256->64) -> ReLU -> Dropout(0.2) -> FC(1) -> Sigmoid
注：官方 model.py 中 Transformer encoder 被注释，本实现保持一致（不使用）。
"""
import os
import sys

import numpy as np
import torch
from torch import nn

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from DenseNet import DenseNet1D  # noqa: E402
from CBAM import CBAMLayer  # noqa: E402


class DcSE(nn.Module):
    def __init__(self, emb_path=None):
        super().__init__()
        if emb_path is None:
            emb_path = os.path.join(_DIR, "embedding_matrix.npy")
        pretrain = torch.tensor(np.load(emb_path))
        self.embedding_en = nn.Embedding.from_pretrained(pretrain, freeze=False)
        self.dense = DenseNet1D(in_channels=100, growth_rate=32,
                                layers_per_block=5, dropout_rate=0.3, num_blocks=2)
        self.merger_dense = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(256)
        self.bn1 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x6):
        x6 = x6.to(torch.long)
        emb = self.embedding_en(x6).float()            # (B,3000,100)
        conv = self.dense(emb.transpose(1, 2)).squeeze(-1)  # (B,256)
        merge = self.bn(conv)
        merge_dense = torch.relu(self.merger_dense(merge))
        merge_dense = self.bn1(merge_dense)
        merge_dense = self.dropout(merge_dense)
        return torch.sigmoid(self.output_layer(merge_dense)).squeeze(1)
