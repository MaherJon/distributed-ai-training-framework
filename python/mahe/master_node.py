"""Master / 参数服务器：加权分片派发、Δ 加权聚合（AVX2 内核）、评估与检查点。"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import wire
from .model import MLP

log = logging.getLogger("mahe.master")


class ParameterServer:
    """FedAvg 式加权聚合：new_params = params + Σ (n_i/N) · Δ_i"""

    def __init__(self, vec_size: int):
        self.params = np.zeros(vec_size, dtype=np.float32)
        self.round = 0

    def aggregate(self, deltas: list[tuple[int, np.ndarray]]) -> None:
        """deltas: [(n_samples, delta_vec)]，按样本数加权。"""
        import mahe

        total = sum(n for n, _ in deltas)
        if total == 0:
            return
        acc = np.zeros_like(self.params)
        for n, delta in deltas:
            w = np.float32(n / total)
            scaled = (delta * w).astype(np.float32)
            if mahe.HAS_CORE:
                mahe.mahe_core.reduce_add_f32(acc, scaled)  # AVX2: acc += scaled
            else:
                acc += scaled
        if mahe.HAS_CORE:
            mahe.mahe_core.reduce_add_f32(self.params, acc)
        else:
            self.params += acc
        self.round += 1

    def save_checkpoint(self, path: str | Path) -> None:
        np.savez(path, params=self.params, round=self.round)
        log.info("检查点已保存：%s (round=%d)", path, self.round)


class MasterNode:
    def __init__(self, core, hidden: int = 32, on_event=None):
        import mahe

        self.core = core
        self.mahe = mahe
        self.hidden = hidden
        self.on_event = on_event or (lambda ev: None)

        model = MLP(data_mod.N_FEATURES, hidden, data_mod.N_CLASSES, seed=0)
        self.ps = ParameterServer(model.vector_size())
        self.ps.params = model.to_vector()

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._round_deltas: dict[int, list[tuple[int, np.ndarray, dict]]] = {}
        self.stop_flag = False

        core.set_message_handler(self._on_message)
        core.set_on_node_join(lambda n: self._emit({
            "type": "node_join", "id": n.id, "name": n.name,
            "cores": n.cores, "score": n.score,
        }))
        core.set_on_node_dead(lambda n: self._emit({
            "type": "node_dead", "id": n.id, "name": n.name,
        }))

    def _emit(self, ev: dict) -> None:
        try:
            self.on_event(ev)
        except Exception:  # UI 事件不影响训练
            log.exception("event sink error")

    def _on_message(self, node_id: int, msg) -> None:
        mt = self.mahe.mahe_core.MsgType
        if msg.type == mt.GRAD_PUSH:
            meta, blob = wire.unpack(msg.payload)
            delta = wire.blob_to_vec(blob)
            with self._cv:
                self._round_deltas.setdefault(meta["round"], []).append(
                    (meta["n_samples"], delta, meta)
                )
                self.core.scheduler().task_done(meta["task_id"])
                self._cv.notify_all()

    # ---- 等待 ≥min_workers 个存活节点 ----
    def wait_workers(self, min_workers: int, timeout_s: float = 1800.0) -> bool:
        """等待节点入网。

        超时给得比较宽：Master 往往在子端编译部署完成后才等到第一个 Worker。
        """
        t0 = time.time()
        last_note = 0.0
        while time.time() - t0 < timeout_s:
            if self.core.scheduler().alive_count() >= min_workers:
                return True
            if self.stop_flag:
                return False
            if time.time() - last_note > 15:
                last_note = time.time()
                log.info("等待 Worker 入网（当前 %d / 需要 %d）…",
                         self.core.scheduler().alive_count(), min_workers)
            time.sleep(0.2)
        return False

    def train_round(self, k_steps: int, batch: int, lr: float,
                    quorum_ratio: float = 0.8,
                    round_timeout_s: float = 120.0) -> dict | None:
        sched = self.core.scheduler()
        nodes = [n for n in sched.nodes() if n.alive]
        if not nodes:
            return None
        round_id = self.ps.round + 1

        # 加权分片
        split = sched.weighted_split(data_mod.DATASET_SIZE)
        split = {nid: sz for nid, sz in split.items()
                 if any(n.id == nid for n in nodes)}
        if not split:
            return None

        mt = self.mahe.mahe_core.MsgType
        params_blob = wire.vec_to_blob(self.ps.params)
        assigned: dict[int, int] = {}  # node_id -> task_id
        for nid, size in split.items():
            # 计算 offset：按 node 顺序连续切分
            offset = sum(v for k, v in sorted(split.items()) if k < nid)
            task_id = sched.create_task()
            sched.assign_task(task_id, nid)
            payload = wire.pack(
                {
                    "task_id": task_id,
                    "round": round_id,
                    "shard_offset": offset,
                    "shard_len": int(size),
                    "k_steps": k_steps,
                    "batch": batch,
                    "lr": lr,
                },
                params_blob,
            )
            msg = self.mahe.mahe_core.Message()
            msg.type = mt.TASK_ASSIGN
            msg.payload = payload
            if not self.core.send_to(nid, msg):
                sched.task_failed(task_id)
                log.warning("节点 %d 派发失败，任务回收", nid)
                continue
            assigned[nid] = task_id

        if not assigned:
            return None

        # 等待 Δ：quorum = max(1, ceil(quorum_ratio * assigned))
        quorum = max(1, int(np.ceil(quorum_ratio * len(assigned))))
        deadline = time.time() + round_timeout_s
        with self._cv:
            self._round_deltas.setdefault(round_id, [])
            while True:
                got = len(self._round_deltas[round_id])
                if got >= quorum or got >= len(assigned):
                    break
                remaining = deadline - time.time()
                if remaining <= 0 or self.stop_flag:
                    log.warning("round=%d 超时：%d/%d 节点回报，按 quorum 聚合",
                                round_id, got, len(assigned))
                    break
                self._cv.wait(timeout=min(1.0, remaining))
            deltas = [(n, d) for n, d, _ in self._round_deltas.pop(round_id, [])]

        if not deltas:
            log.warning("round=%d 无任何回报，跳过", round_id)
            return None

        self.ps.aggregate(deltas)

        # 评估（固定留出集）
        X, y = data_mod.make_dataset()
        Xe, ye = X[:1000], y[:1000]
        model = MLP(data_mod.N_FEATURES, self.hidden, data_mod.N_CLASSES, seed=0)
        model.load_vector(self.ps.params)
        acc = model.accuracy(Xe, ye)
        loss, _ = model.loss_and_grads(Xe, ye)

        stat = {
            "type": "round",
            "round": self.ps.round,
            "workers": len(deltas),
            "eval_loss": round(loss, 4),
            "eval_acc": round(acc, 4),
            "alive": int(sched.alive_count()),
        }
        log.info(
            "round=%d workers=%d eval_loss=%.4f eval_acc=%.4f",
            stat["round"], stat["workers"], stat["eval_loss"], stat["eval_acc"],
        )
        self._emit(stat)
        return stat

    def train(self, rounds: int, k_steps: int, batch: int, lr: float,
              min_workers: int = 1, ckpt_path: str | None = None,
              ckpt_every: int = 5) -> None:
        log.info("等待 worker 入网（≥%d）…", min_workers)
        if not self.wait_workers(min_workers):
            log.error("无 worker 入网，退出")
            return
        time.sleep(1.0)  # 等 worker 装好消息处理器，避免首轮任务被丢弃
        self._emit({"type": "train_start", "rounds": rounds})
        for _ in range(rounds):
            if self.stop_flag:
                break
            stat = self.train_round(k_steps, batch, lr)
            if stat is None:
                time.sleep(0.5)
                continue
            if ckpt_path and stat["round"] % ckpt_every == 0:
                self.ps.save_checkpoint(ckpt_path)
        if ckpt_path:
            self.ps.save_checkpoint(ckpt_path)
        self._emit({"type": "train_end", "round": self.ps.round})
