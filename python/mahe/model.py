"""NumPy MLP（CPU-only 节点的参考训练后端）。

模型：X → Linear(d,h) → tanh → Linear(h,c) → softmax，交叉熵损失。
参数以一维 float32 向量形式存取（便于 C++ 内核聚合与网络传输）。
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, d_in: int, d_hidden: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.shapes = [
            ("W1", (d_in, d_hidden)),
            ("b1", (d_hidden,)),
            ("W2", (d_hidden, n_classes)),
            ("b2", (n_classes,)),
        ]
        scale1 = np.sqrt(2.0 / d_in)
        scale2 = np.sqrt(2.0 / d_hidden)
        init = {
            "W1": rng.normal(0, scale1, (d_in, d_hidden)),
            "b1": np.zeros(d_hidden),
            "W2": rng.normal(0, scale2, (d_hidden, n_classes)),
            "b2": np.zeros(n_classes),
        }
        self.params = {k: v.astype(np.float32) for k, v in init.items()}

    # ---- 参数向量序列化 ----
    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.params[k].ravel() for k, _ in self.shapes]
        ).astype(np.float32)

    def load_vector(self, vec: np.ndarray) -> None:
        pos = 0
        for k, shape in self.shapes:
            size = int(np.prod(shape))
            self.params[k] = vec[pos:pos + size].reshape(shape).astype(np.float32)
            pos += size
        if pos != vec.size:
            raise ValueError(f"param vector size mismatch: {pos} != {vec.size}")

    def vector_size(self) -> int:
        return sum(int(np.prod(s)) for _, s in self.shapes)

    # ---- 前向/反向 ----
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1)
        logits = a1 @ self.params["W2"] + self.params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        cache = {"X": X, "a1": a1, "probs": probs}
        return probs, cache

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        probs, cache = self.forward(X)
        n = X.shape[0]
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads: dict[str, np.ndarray] = {}
        grads["W2"] = cache["a1"].T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)
        da1 = dlogits @ self.params["W2"].T
        dz1 = da1 * (1.0 - cache["a1"] ** 2)
        grads["W1"] = cache["X"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        flat = np.concatenate(
            [grads[k].ravel() for k, _ in self.shapes]
        ).astype(np.float32)
        return float(loss), flat

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        probs, _ = self.forward(X)
        return float((probs.argmax(axis=1) == y).mean())
