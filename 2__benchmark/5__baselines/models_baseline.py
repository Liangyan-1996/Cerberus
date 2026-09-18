#!/usr/bin/env python3
"""DeepSE / SENet / MuSE 官方架构的 PyTorch 移植（chr-split 统一重训用）。

依据官方源码（original_sources/ 下三个仓库）：
  DeepSE (Genomics 2021, keras) :
    Input(300,1) -> Conv32(k3) -> MaxPool(3) -> Conv64(k3) -> MaxPool(4)
    -> Dropout(0.5) -> Dense640(L2=0.01,ReLU) -> Dropout(0.25) -> Dense1(sigmoid)
    输入 = 4/5/6-mer dna2vec(hg19 w2v) 非重叠求和拼接 (300,)，经 StandardScaler。
  SENet  (CBC 2023, sonnet, 官方 train.py: channels=64, layers=3, heads=8) :
    Embed(6mer 矩阵, 微调) -> Conv64(k40)+BN+ReLU -> SoftmaxPool(5)
    -> Conv64(k40)+BN+ReLU -> SoftmaxPool(4) -> 3x Transformer(pre-LN, dff=128)
    -> Flatten -> FC64+BN+ReLU+Dropout -> FC1(sigmoid)
  MuSE   (CBC 2024, torch 官方) :
    X1: 6mer token(3000) -> Embed(6mer 矩阵, frozen) -> Conv64x41/p5 -> Conv32x41/p2
    X2: k-mer 求和 (1,300) -> Conv(32x1)
    X3: one-hot (3000,4) -> 2x Conv32x5(p2/p5)+drop0.3 -> Conv32x2
    融合: stack->Conv2D32x2 x2 级联 -> MaxPool(3) -> FC640 -> FC2(CE/logits)

注：与 DcSE-main/src/models 下的简化版不同，本文件以官方仓库为准。
"""
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from features import load_6mer_embedding  # noqa: E402


# --------------------------------------------------------------------------
# DeepSE
# --------------------------------------------------------------------------
class DeepSE(nn.Module):
    """官方 keras 模型移植；forward(x: (B,300)) -> (B,) sigmoid 概率。
    dense1 权重 L2=0.01（官方 kernel_regularizer），其余无 weight decay。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(64 * 24, 640)  # 300->298->99->97->24
        self.drop2 = nn.Dropout(0.25)
        self.output = nn.Linear(640, 1)

    def forward(self, x):
        x = x.float().unsqueeze(1)          # (B,1,300)
        x = F.relu(self.conv1(x))           # (B,32,298)
        x = self.pool1(x)                   # (B,32,99)
        x = F.relu(self.conv2(x))           # (B,64,97)
        x = self.pool2(x)                   # (B,64,24)
        x = self.drop1(x)
        x = torch.flatten(x, 1)             # (B,1536)
        x = F.relu(self.dense1(x))
        x = self.drop2(x)
        return torch.sigmoid(self.output(x)).squeeze(-1)


# --------------------------------------------------------------------------
# SENet（官方 sonnet Enformer 移植）
# --------------------------------------------------------------------------
class SoftmaxPool1d(nn.Module):
    """官方 SoftmaxPooling1D 移植：logit = 通道线性求和(共享窗口权重)，
    窗口内 softmax 加权求和；w_init=Identity(0) -> 初始均匀（等价均值池化）。"""

    def __init__(self, channels, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.channel_weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):                           # (B,C,L)
        B, C, L = x.shape
        ps = self.pool_size
        T = L // ps
        x = x.view(B, C, T, ps)                     # (B,C,T,ps)
        logit = torch.einsum("bctp,c->btp", x, self.channel_weight)  # (B,T,ps)
        w = torch.softmax(logit, dim=-1)            # (B,T,ps)
        return (x * w.unsqueeze(1)).sum(dim=-1)     # (B,C,T)


class TransformerBlock(nn.Module):
    """官方 pre-LN residual block：LN->MHSA->drop + LN->MLP(2C,ReLU)->drop。"""

    def __init__(self, channels, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(channels * 2, channels),
            nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x),
                                     self.norm1(x))[0])
        x = x + self.ff(self.norm2(x))
        return x


class SENet(nn.Module):
    """forward(x6: (B,3000) token) -> (B,) sigmoid 概率。
    序列固定 pad 3000（keras pad_sequences post），pool 后 T=150。"""

    def __init__(self, emb_path=None, channels=64, num_transformer_layers=3,
                 num_heads=8, dropout=0.2):
        super().__init__()
        if emb_path is None:
            emb_path = os.path.join(_DIR, "embedding_matrix.npy")
        pretrain = torch.tensor(load_6mer_embedding(emb_path))
        self.emb = nn.Embedding.from_pretrained(pretrain, freeze=False)
        # TensorFlow SAME 非对称补边：k40 -> 左 19 / 右 20，长度保持不变
        self.pad = (19, 20)
        self.conv1 = nn.Conv1d(100, channels, kernel_size=40)
        self.bn1 = nn.BatchNorm1d(channels)
        self.sp1 = SoftmaxPool1d(channels, 5)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=40)
        self.bn2 = nn.BatchNorm1d(channels)
        self.sp2 = SoftmaxPool1d(channels, 4)
        self.transformer = nn.Sequential(*[
            TransformerBlock(channels, num_heads, dropout)
            for _ in range(num_transformer_layers)])
        self.head = nn.Sequential(
            nn.Linear(channels * 150, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x6):
        x6 = x6.to(torch.long)
        x = self.emb(x6).float()                 # (B,3000,100)
        x = x.transpose(1, 2)                    # (B,100,3000)
        x = F.relu(self.bn1(self.conv1(F.pad(x, self.pad))))
        x = self.sp1(x)                          # (B,64,600)
        x = F.relu(self.bn2(self.conv2(F.pad(x, self.pad))))
        x = self.sp2(x)                          # (B,64,150)
        x = self.transformer(x.transpose(1, 2))  # (B,150,64)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.head(x)).squeeze(-1)


# --------------------------------------------------------------------------
# MuSE（官方 torch 移植，输出 2 类 logits，训练用 CrossEntropyLoss）
# --------------------------------------------------------------------------
class MuSEOnehot(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(0.3))
        self.pool1 = nn.MaxPool1d(2)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(0.3))
        self.pool2 = nn.MaxPool1d(5)
        self.cnn3 = nn.Conv1d(32, 32, kernel_size=2)

    def forward(self, x):                        # (B,3000,4)
        x = x.permute(0, 2, 1)                   # (B,4,3000)
        x = self.cnn1(x)
        x = self.pool1(x)                        # 1500
        x = self.cnn2(x)
        x = self.pool2(x)                        # 300
        x = self.cnn3(x)                         # 299
        return x


class MuSE(nn.Module):
    """forward(x1:(B,3000) 6mer token, x2:(B,1,300), x3:(B,3000,4)) -> (B,2) logits。"""

    def __init__(self, emb_path=None):
        super().__init__()
        if emb_path is None:
            emb_path = os.path.join(_DIR, "embedding_matrix.npy")
        pretrain = torch.tensor(load_6mer_embedding(emb_path))
        self.emb = nn.Embedding.from_pretrained(pretrain, freeze=True)
        self.cnn_x2 = nn.Conv1d(1, 32, 1)
        self.cnn1 = nn.Conv1d(100, 64, 41, padding=20)
        self.pool1 = nn.MaxPool1d(5)
        self.cnn2 = nn.Conv1d(64, 32, 41, padding=20)
        self.pool2 = nn.MaxPool1d(2)
        self.fuse1 = nn.Conv2d(32, 32, 2)       # x1x2
        self.fuse2 = nn.Conv2d(32, 32, 2)       # x1x2x3
        self.pool3 = nn.MaxPool1d(3)
        self.onehot_cnn = MuSEOnehot()
        self.linear1 = nn.Linear(3168, 640)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(640, 2)

    def forward(self, x1, x2, x3):
        x = self.emb(x1.to(torch.long)).float()      # (B,3000,100)
        x = x.transpose(1, 2)                        # (B,100,3000)
        x = self.pool1(F.relu(self.cnn1(x)))         # (B,64,600)
        x = self.pool2(F.relu(self.cnn2(x)))         # (B,32,300)
        y2 = self.cnn_x2(x2.float())                 # (B,32,300)
        y3 = self.onehot_cnn(x3)                     # (B,32,299)
        z = torch.stack((x, y2), dim=-1)             # (B,32,300,2)
        z = self.fuse1(z).squeeze(-1)                # (B,32,299)
        z = torch.stack((z, y3), dim=-1)             # (B,32,299,2)
        z = self.fuse2(z).squeeze(-1)                # (B,32,298)
        z = self.pool3(z)                            # (B,32,99)
        z = torch.flatten(z, 1)                      # (B,3168)
        z = self.linear1(z)
        z = F.relu(self.dropout(z))
        return self.linear2(z)                       # (B,2) logits
