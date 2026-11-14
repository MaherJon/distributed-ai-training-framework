"""数据集注册表：内置可复现数据集 + 统一的数据集描述符。

所有节点用同一份描述符（spec）在本地生成同一数据集，再按 offset/length
取分片——避免在网络上传输训练数据本身。描述符是纯 JSON，可随任务下发，
因此子端无需提前下载数据文件。

描述符字段：
  id          唯一标识（内置用 ``builtin:`` 前缀，社区用 ``community:``）
  name        展示名称
  generator   生成器名（目前支持 spiral3 / gauss3 / rings4）
  n_features  特征维度
  n_classes   类别数
  size        样本总数
  seed        随机种子
  params      生成器参数（噪声、间距等）
  source      builtin / community
  description 一句话说明
  recommended 推荐训练参数（rounds / k_steps / batch）
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

# 兼容旧代码的默认值：默认数据集为三簇螺旋
N_FEATURES = 2
N_CLASSES = 3
DATASET_SIZE = 6000
DATA_SEED = 20261108

DEFAULT_DATASET_ID = "builtin:spiral3"


# ---------------- 生成器 ----------------
def _spiral3(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """三簇旋转螺旋（入门：非线性但可分）。"""
    n = int(spec["size"])
    seed = int(spec["seed"])
    noise = float(spec.get("params", {}).get("noise", 0.3))
    classes = int(spec["n_classes"])
    rng = np.random.default_rng(seed)
    per = max(1, n // classes)
    Xs, ys = [], []
    for c in range(classes):
        t = rng.uniform(0, np.pi, per)
        r = 1.0 + noise * rng.standard_normal(per)
        base = np.stack([r * np.cos(t), r * np.sin(t) * 0.6 + 0.5 * t], axis=1)
        angle = c * 2 * np.pi / max(1, classes)
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        Xs.append((base @ rot.T).astype(np.float32))
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _gauss3(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """多类高斯团（简单：适合确认链路是否打通）。"""
    n = int(spec["size"])
    seed = int(spec["seed"])
    spread = float(spec.get("params", {}).get("spread", 0.6))
    classes = int(spec["n_classes"])
    rng = np.random.default_rng(seed)
    per = max(1, n // classes)
    Xs, ys = [], []
    for c in range(classes):
        angle = 2 * np.pi * c / max(1, classes)
        center = np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)],
                          dtype=np.float32)
        Xs.append(center + spread * rng.standard_normal((per, 2)).astype(np.float32))
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _rings4(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """四类同心环（较难：类别边界非线性）。"""
    n = int(spec["size"])
    seed = int(spec["seed"])
    noise = float(spec.get("params", {}).get("noise", 0.12))
    classes = int(spec["n_classes"])
    rng = np.random.default_rng(seed)
    per = max(1, n // classes)
    Xs, ys = [], []
    for c in range(classes):
        radius = 1.0 + 0.9 * c
        theta = rng.uniform(0, 2 * np.pi, per)
        r = radius + noise * rng.standard_normal(per)
        Xs.append(np.stack([r * np.cos(theta), r * np.sin(theta)],
                           axis=1).astype(np.float32))
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


GENERATORS = {
    "spiral3": _spiral3,
    "gauss3": _gauss3,
    "rings4": _rings4,
}


# ---------------- 内置数据集 ----------------
BUILTIN_DATASETS: list[dict[str, Any]] = [
    {
        "id": "builtin:spiral3",
        "name": "三簇螺旋",
        "generator": "spiral3",
        "n_features": 2,
        "n_classes": 3,
        "size": DATASET_SIZE,
        "seed": DATA_SEED,
        "params": {"noise": 0.3},
        "source": "builtin",
        "description": "入门首选：非线性可分，观察 loss 下降最直观。",
        "recommended": {"rounds": 20, "k_steps": 20, "batch": 64},
    },
    {
        "id": "builtin:gauss3",
        "name": "高斯团（三类）",
        "generator": "gauss3",
        "n_features": 2,
        "n_classes": 3,
        "size": 6000,
        "seed": 20261109,
        "params": {"spread": 0.6},
        "source": "builtin",
        "description": "最简数据：先确认集群与训练链路是否正常。",
        "recommended": {"rounds": 10, "k_steps": 10, "batch": 64},
    },
    {
        "id": "builtin:rings4",
        "name": "同心环（四类）",
        "generator": "rings4",
        "n_features": 2,
        "n_classes": 4,
        "size": 8000,
        "seed": 20261110,
        "params": {"noise": 0.12},
        "source": "builtin",
        "description": "稍难：四类非线性边界，适合观察更多轮次的效果。",
        "recommended": {"rounds": 30, "k_steps": 20, "batch": 64},
    },
]

_BY_ID = {d["id"]: d for d in BUILTIN_DATASETS}


# ---------------- 描述符处理 ----------------
def list_builtin() -> list[dict[str, Any]]:
    return [dict(d) for d in BUILTIN_DATASETS]


def get(dataset_id: str) -> dict[str, Any] | None:
    spec = _BY_ID.get(dataset_id)
    return dict(spec) if spec else None


def default_spec() -> dict[str, Any]:
    return dict(_BY_ID[DEFAULT_DATASET_ID])


def normalize(spec: dict[str, Any] | str | None) -> dict[str, Any]:
    """接受 id 或描述符，返回补齐默认值的合法描述符。"""
    if spec is None or spec == "":
        return default_spec()
    if isinstance(spec, str):
        found = get(spec)
        if found is None:
            raise ValueError(f"未知数据集：{spec}")
        return found
    if not isinstance(spec, dict):
        raise ValueError("数据集描述符必须是对象或 id")

    base = get(str(spec.get("id") or "")) or default_spec()
    merged = {**base, **spec}
    merged["params"] = {**base.get("params", {}), **(spec.get("params") or {})}
    merged["recommended"] = {**base.get("recommended", {}),
                             **(spec.get("recommended") or {})}

    generator = str(merged.get("generator") or "")
    if generator not in GENERATORS:
        raise ValueError(f"不支持的数据集生成器：{generator or '未指定'}")
    merged["n_features"] = int(merged.get("n_features") or N_FEATURES)
    merged["n_classes"] = int(merged.get("n_classes") or N_CLASSES)
    merged["size"] = max(1, int(merged.get("size") or DATASET_SIZE))
    merged["seed"] = int(merged.get("seed") or DATA_SEED)
    if not merged.get("name"):
        merged["name"] = merged.get("id") or "未命名数据集"
    return merged


def make(spec: dict[str, Any] | str | None = None,
         n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """按描述符确定性生成数据集；n 仅用于临时缩放（评估/调试）。"""
    resolved = normalize(spec)
    if n is not None and n != resolved["size"]:
        resolved = {**resolved, "size": int(n)}
    generator = GENERATORS[resolved["generator"]]
    return generator(resolved)


def make_dataset(n: int = DATASET_SIZE, seed: int = DATA_SEED):
    """兼容旧接口：默认三簇螺旋。"""
    spec = {**default_spec(), "size": int(n), "seed": int(seed)}
    return GENERATORS[spec["generator"]](spec)


def shard(X: np.ndarray, y: np.ndarray, offset: int, length: int):
    return X[offset:offset + length], y[offset:offset + length]


def estimate(spec: dict[str, Any] | str | None, rounds: int, k_steps: int,
             batch: int, workers: int = 1) -> dict[str, Any]:
    """训练量估算：步数、样本吞吐与等效轮数（epoch）。

    等效轮数 = 全局处理样本数 / 数据集大小，是判断“欠训练 / 过训练”的依据：
      < 1    基本没学到东西
      1 ~ 50 正常
      > 200  可能过拟合（小数据集尤其明显）
    """
    resolved = normalize(spec)
    size = int(resolved["size"])
    workers = max(1, int(workers))
    rounds = max(1, int(rounds))
    k_steps = max(1, int(k_steps))
    batch = max(1, int(batch))

    steps = rounds * k_steps
    samples_per_worker = size / workers
    per_round_samples = min(size, k_steps * batch * workers)
    epochs = rounds * per_round_samples / size
    return {
        "dataset_size": size,
        "workers": workers,
        "steps_total": steps,
        "steps_per_worker": k_steps,
        "samples_per_worker": int(samples_per_worker),
        "samples_per_round": int(min(size, per_round_samples)),
        "epochs": round(epochs, 2),
        "verdict": _verdict(epochs),
    }


def _verdict(epochs: float) -> str:
    if epochs < 1:
        return "训练量偏少：模型可能还没开始收敛"
    if epochs > 200:
        return "训练量偏大：小数据集上容易过拟合"
    if epochs > 50:
        return "训练量充足：注意观察验证集指标是否继续提升"
    return "训练量适中"


def dumps(spec: dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
