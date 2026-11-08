"""演示数据集：确定性生成的 2 维 3 类分类数据（螺旋/高斯混合）。

所有节点用同一种子生成同一数据集，按 offset/length 取分片——
避免在网络上传输训练数据本身（真实场景由数据分片描述符替代）。
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 2
N_CLASSES = 3
DATASET_SIZE = 6000
DATA_SEED = 20261108


def make_dataset(n: int = DATASET_SIZE, seed: int = DATA_SEED):
    rng = np.random.default_rng(seed)
    per = n // N_CLASSES
    Xs, ys = [], []
    for c in range(N_CLASSES):
        # 每个类别是旋转 120° 的香蕉形簇
        t = rng.uniform(0, np.pi, per)
        r = 1.0 + 0.3 * rng.standard_normal(per)
        base = np.stack([r * np.cos(t), r * np.sin(t) * 0.6 + 0.5 * t], axis=1)
        angle = c * 2 * np.pi / N_CLASSES
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        Xs.append((base @ rot.T).astype(np.float32))
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def shard(X: np.ndarray, y: np.ndarray, offset: int, length: int):
    return X[offset:offset + length], y[offset:offset + length]
