#!/usr/bin/env python3
"""统一 chr-split 重训脚本（DeepSE / SENet / MuSE / DcSE）。

协议：
  - train = 非 {3,9,11,14} 染色体（1:1）；val = chr9/14 早停（监控 AUROC）
  - 特征构造按各模型官方仓库（features.py 内注释）：
      DcSE/SENet : 6-mer token (3000,)（embedding 微调）
      DeepSE     : 4/5/6-mer hg19 w2v 非重叠求和 (300,)，StandardScaler(train)
      MuSE       : X1 6mer token(3000) + X2 同上(300, 官方未缩放) + X3 one-hot(3000,4)
  - 损失：DeepSE/SENet/DcSE = BCE；MuSE = CrossEntropy（官方 2 类 softmax 输出）

用法:
  python train_baseline.py --model DeepSE --train ../alignment/train.csv \
      --val ../alignment/val.csv --out_dir ./ckpt --epochs 80 --lr 1e-3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from features import (  # noqa: E402
    load_kmer_matrices,
    seq_to_6mer_tokens,
    seq_to_deepse_vec,
    seq_to_onehot,
)

KMER_NPZ = os.path.join(_DIR, "kmer_deepse_hg19.npz")


class SeqDataset(Dataset):
    """按模型返回特征：
    DcSE/SENet -> x6 (3000,)
    DeepSE     -> x2 (300,)（StandardScaler 由 train 拟合后传入）
    MuSE       -> (x6 (3000,), x2 (300,), x3 (3000,4))
    """

    def __init__(self, csv_path, model, kmer_mats=None, x2_scaler=None):
        df = pd.read_csv(csv_path)
        self.seqs = df["sequence"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.model = model
        self._x2 = None
        if model in ("DeepSE", "MuSE"):
            if kmer_mats is None:
                raise ValueError(f"{model} 需要 --kmer_npz（DeepSE hg19 4/5/6-mer）")
            self._x2 = np.stack([seq_to_deepse_vec(s, kmer_mats) for s in self.seqs])
            if model == "DeepSE" and x2_scaler is not None:
                self._x2 = x2_scaler.transform(self._x2).astype(np.float32)
            elif model == "DeepSE":
                self._x2 = self._x2.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        seq = self.seqs[i]
        label = self.labels[i]
        if self.model in ("DcSE", "SENet"):
            return seq_to_6mer_tokens(seq), label
        if self.model == "DeepSE":
            return self._x2[i], label
        if self.model == "MuSE":
            x6 = seq_to_6mer_tokens(seq)
            x3 = seq_to_onehot(seq, alphabet="atcg")  # MuSE 官方通道序 A,T,C,G
            return (x6, self._x2[i].reshape(1, 300), x3), label  # 官方 unsqueeze(1)
        raise ValueError(self.model)


def _collate(batch):
    """batch: [(sample, label)]；sample 为 ndarray 或 (ndarray, ndarray, ndarray)。"""
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    first = batch[0][0]
    if isinstance(first, tuple):
        parts = []
        for j in range(len(first)):
            arr = np.stack([np.asarray(b[0][j]) for b in batch])
            parts.append(torch.from_numpy(arr))
        return (*parts, y)
    arr = np.stack([np.asarray(b[0]) for b in batch])
    return torch.from_numpy(arr), y


def build_dataloader(csv_path, model, batch_size, shuffle, kmer_mats=None,
                     x2_scaler=None, num_workers=0):
    ds = SeqDataset(csv_path, model, kmer_mats=kmer_mats, x2_scaler=x2_scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=_collate), ds


def get_model(name):
    if name == "DcSE":
        from models_dcse import DcSE
        return DcSE()
    if name == "DeepSE":
        from models_baseline import DeepSE
        return DeepSE()
    if name == "SENet":
        from models_baseline import SENet
        return SENet()
    if name == "MuSE":
        from models_baseline import MuSE
        return MuSE()
    raise NotImplementedError(name)


def model_outputs(model, batch, name):
    """返回 (logits_or_prob, y, is_muse)。"""
    if name == "MuSE":
        x6, x2, x3, y = batch
        out = model(x6, x2, x3)          # (B,2) logits
        return out, y, True
    if name == "DeepSE":
        x2, y = batch
        return model(x2), y, False       # 已含 sigmoid
    x6, y = batch
    return model(x6), y, False


def probs_from_output(out, is_muse):
    if is_muse:
        return torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    return out.cpu().numpy()


def train_one_epoch(model, loader, opt, crit_ce, crit_bce, device, name):
    model.train()
    losses = []
    for batch in loader:
        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
        out, y, is_muse = model_outputs(model, batch, name)
        opt.zero_grad()
        if is_muse:
            loss = crit_ce(out, y.to(device).long())
        else:
            loss = crit_bce(out, y.to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(model, loader, device, name):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if name == "MuSE":
                x6, x2, x3, y = batch
                out = model(x6.to(device), x2.to(device), x3.to(device))
            elif name == "DeepSE":
                x2, y = batch
                out = model(x2.to(device))
            else:
                x6, y = batch
                out = model(x6.to(device))
            ys.append(y.numpy())
            ps.append(probs_from_output(out, name == "MuSE"))
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["DeepSE", "SENet", "MuSE", "DcSE"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=None,
                    help="默认按模型：DeepSE 1e-3 / SENet 5e-4 / MuSE 1e-5 / DcSE 1e-4")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmer_npz", default=KMER_NPZ)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lr_defaults = {"DeepSE": 1e-3, "SENet": 5e-4, "MuSE": 1e-5, "DcSE": 1e-4}
    lr = args.lr if args.lr is not None else lr_defaults[args.model]
    print(f"[Info] model={args.model} device={device} epochs={args.epochs} "
          f"lr={lr} bs={args.batch_size}")

    kmer_mats = None
    if args.model in ("DeepSE", "MuSE"):
        kmer_mats = load_kmer_matrices(args.kmer_npz)

    # DeepSE 需要 StandardScaler（官方 train_specific.py 在 train 上拟合）
    x2_scaler = None
    train_loader = None
    if args.model == "DeepSE":
        tr_ds = SeqDataset(args.train, args.model, kmer_mats=kmer_mats)
        x2_scaler = StandardScaler().fit(np.stack(tr_ds._x2))
        tr_ds._x2 = x2_scaler.transform(tr_ds._x2).astype(np.float32)
        train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=_collate)
    else:
        train_loader, _ = build_dataloader(args.train, args.model, args.batch_size,
                                           shuffle=True, kmer_mats=kmer_mats,
                                           x2_scaler=x2_scaler)
    val_loader, _ = build_dataloader(args.val, args.model, args.batch_size,
                                     shuffle=False, kmer_mats=kmer_mats,
                                     x2_scaler=x2_scaler)

    model = get_model(args.model).to(device)
    # DeepSE 官方仅 dense1 有 L2=0.01；其余参数无 weight decay
    if args.model == "DeepSE":
        dense1 = model.dense1
        other = [p for n, p in model.named_parameters() if not n.startswith("dense1.")]
        opt = torch.optim.Adam(
            [{"params": list(dense1.parameters()), "weight_decay": 0.01},
             {"params": other, "weight_decay": 0.0}], lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit_ce = torch.nn.CrossEntropyLoss()
    crit_bce = torch.nn.BCELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_auc, best_ep, best_state, wait = -1.0, -1, None, 0
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, crit_ce, crit_bce,
                               device, args.model)
        yv, pv = evaluate(model, val_loader, device, args.model)
        auc = roc_auc_score(yv, pv)
        aupr = average_precision_score(yv, pv)
        history.append({"epoch": ep, "loss": round(loss, 4),
                        "val_auc": round(auc, 4), "val_aupr": round(aupr, 4)})
        print(f"[Epoch {ep}/{args.epochs}] loss={loss:.4f} "
              f"val_AUC={auc:.4f} val_AUPR={aupr:.4f} ({time.time()-t0:.0f}s)")
        if auc > best_auc:
            best_auc, best_ep = auc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EarlyStop] {args.patience} epochs no improvement, stop at ep{ep}")
                break

    ckpt = os.path.join(args.out_dir, f"{args.model}_best_chrsplit.pth")
    torch.save(best_state, ckpt)
    hist_path = os.path.join(args.out_dir, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump({"model": args.model, "best_epoch": best_ep,
                   "best_val_auc": round(best_auc, 4), "history": history},
                  f, indent=2)
    print(f"[Done] {args.model}: best ep={best_ep} val_AUC={best_auc:.4f} "
          f"ckpt={ckpt} history={hist_path}")


if __name__ == "__main__":
    main()
