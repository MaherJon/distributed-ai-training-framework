"""MAHE 演示入口。

用法：
  python -m mahe.demo_train --role master [--rounds 20 --min-workers 2]
  python -m mahe.demo_train --role worker [--master 127.0.0.1]
  python -m mahe.demo_train --role demo   # 单机：master + 2 个 worker 子进程
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

import mahe

log = logging.getLogger("mahe.demo")

TOKEN = "mahe-demo-token"
TCP_PORT = 47820
UDP_PORT = 47821


def _need_core():
    if not mahe.HAS_CORE:
        sys.exit("错误：mahe_core 扩展未构建。请先运行 python build.py")


def run_master(args) -> None:
    _need_core()
    from .master_node import MasterNode
    from .ui_bridge import UiBridge

    mc = mahe.mahe_core.MasterConfig()
    mc.cluster_name = "mahe-demo"
    mc.token = TOKEN
    mc.tcp_port = TCP_PORT
    mc.udp_port = UDP_PORT
    mc.enable_beacon = True

    core = mahe.mahe_core.CoreNode()
    if not core.start_master(mc):
        sys.exit("master 启动失败（端口被占用？）")
    log.info("master 已启动 tcp=%d udp=%d kernel=%s",
             TCP_PORT, UDP_PORT, mahe.mahe_core.kernel_impl())

    bridge = UiBridge()
    bridge.start(on_stop=lambda: setattr(master, "stop_flag", True))
    master = MasterNode(core, hidden=args.hidden, on_event=bridge.emit)
    try:
        master.train(
            rounds=args.rounds, k_steps=args.k_steps, batch=args.batch,
            lr=args.lr, min_workers=args.min_workers,
            ckpt_path=args.ckpt,
        )
    except KeyboardInterrupt:
        log.info("中断，停止…")
    finally:
        bridge.stop()
        core.stop()


def run_worker(args) -> None:
    _need_core()
    from .worker_node import WorkerNode, micro_benchmark

    score = micro_benchmark()
    log.info("算力基准分：%.2f", score)

    wc = mahe.mahe_core.WorkerConfig()
    wc.node_name = args.name or f"worker-{os.getpid()}"
    wc.token = TOKEN
    wc.master_host = args.master or ""  # 空 → beacon 自动发现
    wc.master_port = TCP_PORT
    wc.udp_port = UDP_PORT
    wc.score = score

    core = mahe.mahe_core.CoreNode()
    if not core.start_worker(wc):
        sys.exit("worker 入网失败（未发现 master 或 token 不匹配）")
    log.info("已入网，node_id 分配中…")
    for _ in range(100):
        if core.self_id() != 0:
            break
        time.sleep(0.05)
    log.info("node_id=%d，等待任务…", core.self_id())

    worker = WorkerNode(core, hidden=args.hidden)
    try:
        while not worker.stop_flag:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        core.stop()


def run_demo(args) -> None:
    """单机端到端：当前进程跑 master，拉起 2 个 worker 子进程。"""
    from pathlib import Path

    env = os.environ.copy()
    # PyInstaller 冻结后 sys.executable 即引擎本身，不能再加 "-m mahe.demo_train"
    frozen = getattr(sys, "frozen", False)
    if frozen:
        base_cmd = [sys.executable]
    else:
        base_cmd = [sys.executable, "-m", "mahe.demo_train"]
        # 源码运行时子进程需要能 import mahe 包
        pkg_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    workers = []
    for i in range(2):
        p = subprocess.Popen(
            [*base_cmd, "--role", "worker",
             "--master", "127.0.0.1", "--name", f"demo-w{i}",
             "--hidden", str(args.hidden)],
            env=env,
        )
        workers.append(p)
    try:
        args.min_workers = 2
        run_master(args)
    finally:
        for p in workers:
            p.terminate()
        for p in workers:
            p.wait(timeout=5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(prog="mahe.demo_train")
    ap.add_argument("--role", choices=["master", "worker", "demo"],
                    default="demo")
    ap.add_argument("--master", default="", help="master 地址（worker 用；空=自动发现）")
    ap.add_argument("--name", default="")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--k-steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--min-workers", type=int, default=1)
    # 冻结为 GUI 应用时工作目录不可预期（Windows 可能是 C:\Windows\System32），
    # 默认检查点落到用户目录，避免写系统路径失败
    if getattr(sys, "frozen", False):
        default_ckpt = os.path.join(os.path.expanduser("~"), ".mahe",
                                    "checkpoints", "mahe_ckpt.npz")
    else:
        default_ckpt = "checkpoints/mahe_ckpt.npz"
    ap.add_argument("--ckpt", default=default_ckpt)
    args = ap.parse_args()

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    if args.role == "master":
        run_master(args)
    elif args.role == "worker":
        run_worker(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
