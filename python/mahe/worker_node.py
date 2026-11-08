"""Worker 节点：入网、接收任务、本地多步训练、回传参数差值(Δ)。"""

from __future__ import annotations

import logging
import time

import numpy as np

from . import data as data_mod
from . import wire
from .model import MLP

log = logging.getLogger("mahe.worker")


def micro_benchmark() -> float:
    """简易算力基准：矩阵乘耗时 → 相对分数（越大越快）。"""
    a = np.random.standard_normal((192, 192)).astype(np.float32)
    b = np.random.standard_normal((192, 192)).astype(np.float32)
    a @ b  # warmup
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - t0
    return round(reps / max(dt, 1e-6), 3)


class WorkerNode:
    def __init__(self, core, hidden: int = 32, seed: int = 0):
        self.core = core  # mahe_core.CoreNode (已 start_worker)
        self.hidden = hidden
        self.seed = seed
        self.stop_flag = False
        core.set_message_handler(self._on_message)

    # ---- 本地多步训练（FedAvg 式，返回 Δ 与本地 loss）----
    def train_local(self, params_vec: np.ndarray, X, y, k_steps: int,
                    batch: int, lr: float) -> tuple[np.ndarray, float]:
        model = MLP(data_mod.N_FEATURES, self.hidden, data_mod.N_CLASSES,
                    seed=self.seed)
        model.load_vector(params_vec)
        before = model.to_vector()
        n = X.shape[0]
        rng = np.random.default_rng(self.seed + 1)
        loss = 0.0
        for step in range(k_steps):
            idx = rng.integers(0, n, size=min(batch, n))
            loss, grads = model.loss_and_grads(X[idx], y[idx])
            # 原地更新走 C++ 汇编内核
            import mahe

            if mahe.HAS_CORE:
                vec = model.to_vector()
                mahe.mahe_core.sgd_update_f32(vec, grads, np.float32(lr))
                model.load_vector(vec)
            else:
                model.load_vector(model.to_vector() - lr * grads)
        delta = model.to_vector() - before
        return delta.astype(np.float32), loss

    def _on_message(self, node_id: int, msg) -> None:
        if msg.type == self._mt("TASK_ASSIGN"):
            self._handle_task(msg)
        elif msg.type == self._mt("BYE"):
            log.info("收到 BYE，退出")
            self.stop_flag = True

    def _handle_task(self, msg) -> None:
        meta, blob = wire.unpack(msg.payload)
        round_id = meta["round"]
        task_id = meta["task_id"]
        offset, length = meta["shard_offset"], meta["shard_len"]
        params = wire.blob_to_vec(blob)

        X, y = data_mod.make_dataset()
        Xs, ys = data_mod.shard(X, y, offset, length)

        t0 = time.perf_counter()
        delta, loss = self.train_local(
            params, Xs, ys,
            k_steps=meta["k_steps"], batch=meta["batch"], lr=meta["lr"],
        )
        dt = time.perf_counter() - t0
        log.info(
            "round=%d task=%d shard=[%d,+%d) loss=%.4f 用时=%.2fs",
            round_id, task_id, offset, length, loss, dt,
        )
        reply = wire.pack(
            {
                "task_id": task_id,
                "round": round_id,
                "n_samples": length,
                "loss": loss,
                "train_seconds": dt,
            },
            wire.vec_to_blob(delta),
        )
        out = self._msg("GRAD_PUSH", reply)
        ok = self.core.send_to_master(out)
        if not ok:
            log.warning("GRAD_PUSH 发送失败（背压或断线）")
        ack = self._msg("TASK_ACK", wire.pack({"task_id": task_id}))
        self.core.send_to_master(ack)

    # ---- mahe_core 枚举包装（延迟导入错误提示）----
    @staticmethod
    def _mt(name: str):
        import mahe

        if not mahe.HAS_CORE:
            raise RuntimeError("mahe_core 扩展未构建，请先运行 build.py")
        return getattr(mahe.mahe_core.MsgType, name)

    @staticmethod
    def _msg(name: str, payload: bytes):
        import mahe

        m = mahe.mahe_core.Message()
        m.type = getattr(mahe.mahe_core.MsgType, name)
        m.payload = payload
        return m
