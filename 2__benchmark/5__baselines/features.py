#!/usr/bin/env python3
"""序列特征编码（k-mer 映射规则已实证，2026-09-06）。

模型输入规格（修订意见_v1.md + 官方源码核对）：
  DcSE  : X6  (3000,) 整数 6-mer token（stride=1，字典序 idx=product_idx+1, null=0）
  SENet : X6  (3000,) 同上（embedding 由官方 6-mer 矩阵初始化并微调）
  MuSE  : X1 6-mer (3000,) + X2 全局 k-mer 求和 (300,) + X3 one-hot (3000,4)
  DeepSE: X2 全局 k-mer 求和 (300,)（4/5/6-mer 各 100 维拼接）

映射规则（实证结论）：
  1) DcSE/SENet/MuSE 的 6-mer embedding_matrix.npy 四份 md5 相同
     (920d01ecc4cf6928f0919ab20519db80)，且逐行 == 官方 dna2vec
     (k3to8, 87,360x100) 中 6-mer 向量按 product('ACGT') 字典序排列（L2=0）。
  2) DeepSE 自带 hg19_embedding.w2v 只含 4/5/6-mer（5,376 词），词表与官方
     相同但向量数值不同（自训模型）；MuSE 官方预计算 X2 npz 与"DeepSE w2v +
     非重叠步长=k 求和"逐行完全一致（cos=1.0, ratio=1.0）。
  因此：X1/6-mer 类输入用 6-mer 官方矩阵；X2/DeepSE 用 hg19_embedding.w2v。
"""
import itertools
import os

import numpy as np

BASES = "acgt"     # 小写字母序 a<c<g<t（keras Tokenizer product 顺序）
BASES_U = "ACGT"
_VOCAB6 = None
_IDX6 = None


def _ensure_vocab():
    global _VOCAB6, _IDX6
    if _VOCAB6 is None:
        _VOCAB6 = ["".join(p) for p in itertools.product(BASES, repeat=6)]
        _IDX6 = {m: i + 1 for i, m in enumerate(_VOCAB6)}  # idx = product_idx + 1


def seq_to_6mer_tokens(seq: str, max_len: int = 3000):
    """stride=1 切 6-mer；含 N 记 null(0)；pad 到 max_len（post，右侧补 0）。"""
    _ensure_vocab()
    s = seq.lower()
    n = len(s) - 5
    if n <= 0:
        return np.zeros(max_len, dtype=np.int64)
    out = np.zeros(max_len, dtype=np.int64)
    for i in range(min(n, max_len)):
        m = s[i:i + 6]
        out[i] = 0 if "n" in m else _IDX6.get(m, 0)
    return out


def seq_to_onehot(seq: str, max_len: int = 3000, alphabet: str = "acgt"):
    """one-hot (max_len,4)，碱基顺序由 alphabet 决定（默认 A,C,G,T）；
    N/其他全零；不足右补零行。MuSE 官方 create_onehot 顺序为 A,T,C,G，
    复刻其 X3 时传 alphabet='atcg'。"""
    s = seq.lower()
    map_b = {b: j for j, b in enumerate(alphabet.lower())}
    out = np.zeros((max_len, 4), dtype=np.float32)
    n = min(len(s), max_len)
    for i in range(n):
        j = map_b.get(s[i])
        if j is not None:
            out[i, j] = 1.0
    return out


def mer_index(mer: str) -> int:
    """k-mer（<=6，acgt 任意大小写）-> 字典序 0-based 索引（product over acgt）。"""
    idx = 0
    for ch in mer.lower():
        idx = idx * 4 + BASES.index(ch)
    return idx


# ---------------- dna2vec w2v -> (4^k+1, 100) 矩阵 ----------------

def _matrix_from_kmer_dict(dmap, k):
    """{kmer: 100d} -> (4^k+1, 100)；行 0 = null(0 向量)；
    行 i (i=1..4^k) = 第 i 个字典序 k-mer。与官方 6-mer matrix 同一约定。"""
    dim = dmap[next(iter(dmap))].size
    M = np.zeros((4 ** k + 1, dim), dtype=np.float64)
    for tup in itertools.product(BASES_U, repeat=k):
        mer = "".join(tup)
        M[mer_index(mer) + 1] = dmap[mer]
    return M


def load_w2v_kmer_dict(w2v_path: str, ks=(4, 5, 6)):
    """读 gensim text 格式 w2v（首行 '<words> <dim>'）-> {k: {kmer: 100d}}。
    只解析需要的 k 长度，节省内存。"""
    out = {k: {} for k in ks}
    with open(w2v_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split(" ")
            k = len(parts[0])
            if k in out:
                out[k][parts[0]] = np.asarray(parts[1:], dtype=np.float64)
    return out


def build_kmer_matrices(w2v_path: str, ks=(4, 5, 6)):
    """w2v -> {k: (4^k+1,100) matrix}；校验词表完备 (4^k)。"""
    d = load_w2v_kmer_dict(w2v_path, ks)
    mats = {}
    for k in ks:
        if len(d[k]) != 4 ** k:
            raise ValueError(f"{w2v_path}: {k}-mer 词表 {len(d[k])} != {4**k}")
        mats[k] = _matrix_from_kmer_dict(d[k], k)
    return mats


def save_kmer_matrices(w2v_path: str, out_npz: str, ks=(4, 5, 6)):
    mats = build_kmer_matrices(w2v_path, ks)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(out_npz, **{f"k{k}": mats[k] for k in ks})
    return mats


def load_kmer_matrices(npz_path: str, ks=(4, 5, 6)):
    with np.load(npz_path) as z:
        return {k: z[f"k{k}"] for k in ks}


# ---------------- k-mer 全局求和向量（DeepSE 输入 / MuSE X2） ----------------

def kmer_sum_pool(seq: str, k: int, mat: np.ndarray):
    """非重叠步长=k 的 k-mer 查表求和 -> (100,)。含 N 的窗口贡献 0。
    与 DeepSE get_datavec.py / MuSE 官方 npz 生成规则一致（已逐行验证）。"""
    s = seq.upper()
    acc = np.zeros(mat.shape[1], dtype=np.float64)
    for i in range(0, len(s) - k + 1, k):
        m = s[i:i + k]
        if "N" not in m:
            acc += mat[mer_index(m) + 1]
    return acc


def seq_to_deepse_vec(seq: str, mats: dict):
    """DeepSE/MuSE X2：(300,)，= 4/5/6-mer 非重叠求和向量拼接（k 升序）。"""
    return np.concatenate(
        [kmer_sum_pool(seq, k, mats[k]).astype(np.float32) for k in (4, 5, 6)]
    )


def load_6mer_embedding(path="embedding_matrix.npy"):
    return np.load(path).astype(np.float32)  # (4097, 100)
