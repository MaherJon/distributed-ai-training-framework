"""MAHE 主控入口。

启动顺序（Web 终端模式）：
  1. 只启动回环 HTTP 服务，提供静态页面与事件流；
  2. 浏览器播放启动字幕，播放完毕后 POST ``ui_ready``；
  3. Python 控制面收到信号后才初始化 C++ 核心、调度器与训练模块；
  4. 初始化日志通过左侧信息流输出，格式为 ``[信息类型] 内容``；
  5. 初始化完成后界面弹出子端部署选择面板。

默认行为是「干净环境」：不拉起演示 Worker，也不自动开始训练，等用户在
「训练」页选择数据集、模型保存位置并点击开始。开发者自测用 ``--test`` /
``-t``：会拉起 2 个本地 Worker 并立即开跑。

用法：
  python -m mahe.demo_train                    # 干净启动（仅 Master + Web UI）
  python -m mahe.demo_train --test             # 开发者自测（2 Worker + 自动训练）
  python -m mahe.demo_train --role worker --master IP   # 子端入网
  python -m mahe.demo_train --no-ui            # 无界面（服务器场景，直接训练）
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mahe

from . import agent
from . import community
from . import data
from . import deployer
from . import lan_scan
from . import platform_dialogs

log = logging.getLogger("mahe.demo")

TOKEN = "mahe-demo-token"
TCP_PORT = 47820
UDP_PORT = 47821
UI_PORT = 47830

# 训练参数的安全边界（前端也会限制，后端再兜一层）
LIMITS = {
    "rounds": (1, 2000),
    "k_steps": (1, 20000),
    "batch": (1, 8192),
    "lr": (1e-4, 10.0),
    "min_workers": (1, 128),
}


def _need_core() -> None:
    if not mahe.HAS_CORE:
        sys.exit(f"错误：mahe_core 扩展无法加载：{mahe.CORE_ERROR}\n"
                 "请先运行 python build.py；Windows 上若提示 DLL load failed，"
                 "见 docs/USER_GUIDE.md 的排错表")


class _UiLogHandler(logging.Handler):
    """把 mahe.* 的日志转发到界面信息流。"""

    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(record.levelname, record.getMessage())
        except Exception:  # 日志失败绝不影响训练
            pass


def _advertise_host(args) -> str:
    if args.advertise_host:
        return args.advertise_host
    ips = lan_scan.local_ipv4_addresses()
    return ips[0] if ips else "127.0.0.1"


def _spawn_demo_workers(args) -> list[subprocess.Popen]:
    """自测模式：拉起 2 个本机 Worker 子进程（在 Master 初始化完成后调用）。"""
    env = os.environ.copy()
    frozen = getattr(sys, "frozen", False)
    if frozen:
        base_cmd = [sys.executable]
    else:
        base_cmd = [sys.executable, "-m", "mahe.demo_train"]
        pkg_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")

    workers: list[subprocess.Popen] = []
    for i in range(2):
        workers.append(subprocess.Popen(
            [*base_cmd, "--role", "worker",
             "--master", "127.0.0.1",
             "--master-port", str(args.tcp_port),
             "--token", args.token,
             "--name", f"demo-w{i}",
             "--hidden", str(args.hidden)],
            env=env,
        ))
    return workers


def _make_deploy_config(args, master_host: str) -> deployer.DeployConfig:
    return deployer.DeployConfig(
        repo=deployer.repo_url(args.repo),
        branch=deployer.branch_name(args.branch),
        master_host=master_host,
        master_port=args.tcp_port,
        token=args.token,
        hidden=args.hidden,
    )


def _base_settings(args) -> dict:
    spec = data.normalize(args.dataset or None)
    return {
        "dataset": spec["id"],
        "dataset_name": spec["name"],
        "model_path": args.ckpt,
        "rounds": args.rounds,
        "k_steps": args.k_steps,
        "batch": args.batch,
        "lr": args.lr,
        "min_workers": args.min_workers,
    }


def _merge_settings(cmd: dict, current: dict):
    """合并前端提交的训练设置，返回 (settings, spec, error)。"""
    merged = dict(current)
    for key in ("rounds", "k_steps", "batch", "min_workers"):
        if key in cmd:
            try:
                merged[key] = int(cmd[key])
            except (TypeError, ValueError):
                return None, None, f"{key} 必须是整数"
    if "lr" in cmd:
        try:
            merged["lr"] = float(cmd["lr"])
        except (TypeError, ValueError):
            return None, None, "学习率必须是数字"
    for key, (lo, hi) in LIMITS.items():
        value = merged.get(key)
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            return None, None, f"{key} 需在 {lo} ~ {hi} 之间"

    try:
        spec = data.normalize(cmd.get("dataset") or merged.get("dataset"))
    except (ValueError, TypeError) as exc:
        return None, None, f"数据集无效：{exc}"

    raw_path = str(cmd.get("model_path") or merged.get("model_path") or "").strip()
    path = Path(raw_path).expanduser() if raw_path else Path(platform_dialogs.default_model_path())
    if not path.suffix:
        path = path.with_suffix(".npz")
    merged.update({
        "dataset": spec["id"],
        "dataset_name": spec["name"],
        "model_path": str(path),
    })
    return merged, spec, ""


def run_master(args, spawn_workers: bool = False,
               auto_start: bool = False) -> None:
    """Master：Web 终端 + 训练控制面。"""
    _need_core()
    if args.no_ui:
        _run_headless(args, spawn_workers)
        return

    from .master_node import MasterNode
    from .web_server import WebUiServer

    web = WebUiServer(port=args.ui_port)

    runtime: dict = {"core": None, "master": None, "workers": [],
                     "train_thread": None}
    stop_evt = threading.Event()

    handler = _UiLogHandler(web.emit_info)
    logging.getLogger("mahe").addHandler(handler)

    web.state.kernel = mahe.mahe_core.kernel_impl()
    web.state.master_host = _advertise_host(args)
    web.state.set_settings(_base_settings(args))
    web.set_agent(agent.status())

    def is_training() -> bool:
        thread = runtime.get("train_thread")
        return bool(thread and thread.is_alive())

    def train_loop(settings: dict, spec: dict) -> None:
        master = runtime.get("master")
        if master is None:
            return
        stopped = False
        try:
            master.train(
                rounds=settings["rounds"], k_steps=settings["k_steps"],
                batch=settings["batch"], lr=settings["lr"],
                min_workers=settings["min_workers"],
                ckpt_path=settings["model_path"], dataset=spec,
                wait_timeout_s=args.wait_timeout,
            )
            stopped = master.stop_flag
        except Exception as exc:
            log.exception("训练异常")
            web.emit_info("ERROR", f"训练异常：{exc}")
            web.set_status("error")
            web.publish_train_state(False)
            return
        finally:
            web.publish_train_state(False)
        if stopped:
            web.emit_info("WARN", "训练已停止")
            web.set_status("stopped")
        else:
            web.emit_info("TRAIN",
                          f"训练结束，模型已保存：{settings['model_path']}")
            web.set_status("done")

    def do_start_training(cmd: dict) -> dict:
        master = runtime.get("master")
        if master is None:
            return {"ok": False, "error": "训练模块尚未初始化"}
        if is_training():
            return {"ok": False, "error": "训练已在进行中"}
        settings, spec, err = _merge_settings(cmd, web.state.settings)
        if err:
            return {"ok": False, "error": err}
        try:
            Path(settings["model_path"]).parent.mkdir(parents=True,
                                                     exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"保存目录不可用：{exc}"}

        web.state.set_settings(settings)
        web.publish({"type": "settings", "settings": settings})
        est = data.estimate(spec, settings["rounds"], settings["k_steps"],
                            settings["batch"], settings["min_workers"])
        web.emit_info(
            "TRAIN",
            f"开始训练：{spec['name']} · {settings['rounds']} 轮 × "
            f"{settings['k_steps']} 步 · batch={settings['batch']} · "
            f"等效 {est['epochs']} epoch（{est['verdict']}）",
        )
        web.emit_info("TRAIN", f"模型保存位置：{settings['model_path']}")
        web.set_status("training")
        web.publish_train_state(True, settings["rounds"])

        thread = threading.Thread(target=train_loop, args=(settings, spec),
                                  daemon=True, name="mahe-train")
        runtime["train_thread"] = thread
        thread.start()
        return {"ok": True}

    def init_training() -> None:
        try:
            web.emit_info("INIT", "初始化 C++ 核心（传输 / 发现 / 调度 / SIMD）…")
            mc = mahe.mahe_core.MasterConfig()
            mc.cluster_name = args.cluster_name
            mc.token = args.token
            mc.tcp_port = args.tcp_port
            mc.udp_port = args.udp_port
            mc.enable_beacon = True

            core = mahe.mahe_core.CoreNode()
            if not core.start_master(mc):
                web.emit_info("ERROR",
                              f"训练核心启动失败：TCP {args.tcp_port} 可能被占用")
                web.set_status("error")
                return
            runtime["core"] = core
            web.emit_info("INIT",
                          f"传输层就绪：TCP {args.tcp_port} / UDP {args.udp_port}")
            web.emit_info("INIT",
                          f"计算内核：{mahe.mahe_core.kernel_impl()}")

            spec = data.normalize(web.state.settings.get("dataset"))
            master = MasterNode(core, hidden=args.hidden, dataset=spec,
                                on_event=web.publish_training_event)
            runtime["master"] = master
            web.emit_info("INIT", "参数服务器、评估器与检查点模块就绪")
            web.emit_info("INIT", f"默认数据集：{spec['name']}")

            if spawn_workers:
                runtime["workers"] = _spawn_demo_workers(args)
                web.emit_info("INIT",
                              f"自测模式：已启动 {len(runtime['workers'])} 个本地 Worker")

            web.set_status("ready")
            web.emit_info("INIT", "训练模块初始化完成")
            do_refresh_datasets()
            agent_info = agent.status()
            web.set_agent(agent_info)
            web.emit_info("INIT",
                          "环境已就绪：在「训练」页选择数据集与保存位置后开始")
            web.publish({
                "type": "app_ready",
                "master_host": web.state.master_host,
                "token": args.token,
                "repo": deployer.repo_url(args.repo),
                "branch": deployer.branch_name(args.branch),
                "min_workers": args.min_workers,
                "test_mode": bool(auto_start),
            })
            if auto_start:
                result = do_start_training({})
                if not result.get("ok"):
                    web.emit_info("ERROR",
                                  f"自测训练启动失败：{result.get('error')}")
        except Exception as exc:
            log.exception("训练模块初始化失败")
            web.emit_info("ERROR", f"训练模块初始化失败：{exc}")
            web.set_status("error")

    def do_refresh_datasets() -> dict:
        info = community.catalog(args.community or None)
        web.state.set_datasets(info["datasets"], info["community"])
        web.publish({"type": "datasets", **info})
        return {"ok": True, **info}

    def do_scan(cmd: dict) -> None:
        web.state.set_scanning(True)
        web.publish({"type": "scan", "phase": "start"})
        try:
            devices = lan_scan.scan_devices(
                cmd.get("subnet") or args.subnet,
                on_device=web.publish_scan_device,
                on_log=web.emit_info,
            )
            web.state.set_devices(devices)
            web.publish({"type": "scan", "phase": "done", "devices": devices})
        except Exception as exc:
            web.emit_info("ERROR", f"扫描失败：{exc}")
            web.publish({"type": "scan", "phase": "error", "text": str(exc)})
        finally:
            web.state.set_scanning(False)

    def do_deploy(cmd: dict) -> None:
        dev = cmd.get("device") or {}
        ip = str(dev.get("ip") or "")
        if not ip:
            web.publish_deploy("error", "缺少目标设备地址")
            return
        if dev.get("channel") != "ssh":
            web.publish_deploy(
                "error", "该设备未开放 SSH，请在“手动部署”中下载脚本", dev)
            return
        cfg = _make_deploy_config(args, web.state.master_host)
        web.publish_deploy("start",
                           f"开始部署 {dev.get('name')} ({ip})", dev)
        try:
            rc = deployer.run_ssh_deploy(
                ip, cfg,
                user=str(cmd.get("user") or ""),
                key_path=cmd.get("key") or None,
                on_log=lambda t: web.publish_deploy("log", t),
            )
            web.publish_deploy(
                "done" if rc == 0 else "error",
                "部署完成，等待 Worker 入网" if rc == 0 else
                f"部署失败（退出码 {rc}），可改用“手动部署”",
                dev,
            )
        except Exception as exc:
            log.exception("部署失败")
            web.publish_deploy("error", f"部署失败：{exc}", dev)

    def do_kick_node(cmd: dict) -> None:
        master = runtime.get("master")
        node_id = int(cmd.get("id") or 0)
        if master is None or node_id <= 0:
            return
        master.kick(node_id)
        web.emit_info("NODE", f"节点 {node_id} 已移出分组")

    def do_clear_nodes() -> None:
        master = runtime.get("master")
        if master is None:
            return
        ids = master.clear_nodes()
        web.emit_info("NODE", f"已清空 {len(ids)} 个节点")

    def do_pick_path(cmd: dict) -> str:
        current = str(cmd.get("current") or
                      web.state.settings.get("model_path") or "")
        path = platform_dialogs.pick_save_file(current)
        if path:
            settings = web.state.set_settings({"model_path": path})
            web.publish({"type": "settings", "settings": settings})
            web.emit_info("UI", f"模型保存位置：{path}")
        return path

    def do_agent_ask(cmd: dict) -> dict:
        try:
            reply = agent.ask(str(cmd.get("text") or ""), cmd.get("history"))
        except agent.AgentUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # 兜底：绝不影响训练进程
            log.exception("智能助手调用失败")
            return {"ok": False, "error": f"智能助手异常：{exc}"}
        return {"ok": True, "reply": reply}

    def do_stop() -> None:
        master = runtime.get("master")
        if master is not None:
            master.stop_flag = True
        web.emit_info("WARN", "收到停止训练指令")
        web.set_status("stopping")

    def do_shutdown() -> None:
        do_stop()
        stop_evt.set()

    def script_for(platform_name: str) -> str:
        cfg = _make_deploy_config(args, web.state.master_host)
        return deployer.script_for(cfg, platform_name)

    web.on_ui_ready = lambda: threading.Thread(
        target=init_training, daemon=True, name="mahe-init").start()
    web.on_scan = do_scan
    web.on_deploy = do_deploy
    web.on_stop = do_stop
    web.on_shutdown = do_shutdown
    web.on_start_training = do_start_training
    web.on_kick_node = do_kick_node
    web.on_clear_nodes = do_clear_nodes
    web.on_pick_path = do_pick_path
    web.on_refresh_datasets = do_refresh_datasets
    web.on_agent_ask = do_agent_ask
    web.set_script_provider(script_for)

    try:
        web.start(open_browser=not args.no_browser)
    except OSError as exc:
        logging.getLogger("mahe").removeHandler(handler)
        sys.exit(f"Web 终端启动失败（127.0.0.1:{args.ui_port}）：{exc}")

    web.emit_info("UI", f"控制终端已启动：http://127.0.0.1:{args.ui_port}/")
    web.emit_info("UI", "等待界面初始化完成…")

    if not web.wait_ui_ready(args.ui_timeout):
        web.emit_info("WARN",
                      f"{args.ui_timeout:.0f}s 内未收到界面就绪信号，按无界面模式继续")
        threading.Thread(target=init_training, daemon=True,
                         name="mahe-init").start()

    try:
        while not stop_evt.wait(0.5):
            pass
    except KeyboardInterrupt:
        web.emit_info("WARN", "收到中断，正在停止…")
        if runtime["master"] is not None:
            runtime["master"].stop_flag = True
    finally:
        for p in runtime["workers"]:
            try:
                p.terminate()
            except OSError:
                pass
        for p in runtime["workers"]:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        if runtime["core"] is not None:
            runtime["core"].stop()
        logging.getLogger("mahe").removeHandler(handler)
        web.stop()


def _run_headless(args, spawn_workers: bool) -> None:
    """服务器场景：跳过 Web UI，直接初始化并训练。"""
    from .master_node import MasterNode

    settings = _base_settings(args)
    spec = data.normalize(settings["dataset"])

    mc = mahe.mahe_core.MasterConfig()
    mc.cluster_name = args.cluster_name
    mc.token = args.token
    mc.tcp_port = args.tcp_port
    mc.udp_port = args.udp_port
    mc.enable_beacon = True

    core = mahe.mahe_core.CoreNode()
    if not core.start_master(mc):
        sys.exit("master 启动失败（端口被占用？）")
    log.info("master 已启动 tcp=%d udp=%d kernel=%s",
             args.tcp_port, args.udp_port, mahe.mahe_core.kernel_impl())

    workers = _spawn_demo_workers(args) if spawn_workers else []
    master = MasterNode(core, hidden=args.hidden, dataset=spec)
    try:
        master.train(
            rounds=settings["rounds"], k_steps=settings["k_steps"],
            batch=settings["batch"], lr=settings["lr"],
            min_workers=settings["min_workers"], ckpt_path=settings["model_path"],
            dataset=spec, wait_timeout_s=args.wait_timeout,
        )
    except KeyboardInterrupt:
        log.info("中断，停止…")
    finally:
        for p in workers:
            p.terminate()
        for p in workers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        core.stop()


def run_worker(args) -> None:
    _need_core()
    from .worker_node import WorkerNode, micro_benchmark

    score = micro_benchmark()
    log.info("算力基准分：%.2f", score)

    wc = mahe.mahe_core.WorkerConfig()
    wc.node_name = args.name or f"worker-{os.getpid()}"
    wc.token = args.token
    wc.master_host = args.master or ""  # 空 → beacon 自动发现
    wc.master_port = args.master_port
    wc.udp_port = args.udp_port
    wc.score = score

    core = mahe.mahe_core.CoreNode()
    if not core.start_worker(wc):
        sys.exit("worker 入网失败（未发现 master 或 token 不匹配）")
    log.info("已入网，等待 node_id 分配…")
    for _ in range(100):
        if core.self_id() != 0:
            break
        time.sleep(0.05)
    log.info("node_id=%d，等待任务…", core.self_id())

    worker = WorkerNode(core, hidden=args.hidden)
    try:
        # 被 Master 移出分组（BYE）或连接断开时，core 会停止，进程随之退出
        while not worker.stop_flag and core.is_running():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        core.stop()
        log.info("worker 已退出")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mahe.demo_train")
    # 安装器快捷方式不带参数，用环境变量切换角色（默认 master）
    default_role = os.environ.get("MAHE_ENGINE_ROLE", "master").lower()
    if default_role not in ("master", "worker", "demo"):
        default_role = "master"
    ap.add_argument("--role", choices=["master", "worker", "demo"],
                    default=default_role,
                    help="master=主控（默认）；worker=子端；demo=等同 --test")
    ap.add_argument("-t", "--test", action="store_true",
                    help="开发者自测：拉起 2 个本地 Worker 并自动开始训练")
    ap.add_argument("--master", default="", help="master 地址（worker 用；空=自动发现）")
    ap.add_argument("--master-port", type=int, default=TCP_PORT)
    ap.add_argument("--name", default="")
    ap.add_argument("--cluster-name", default="mahe-demo")
    ap.add_argument("--token", default=TOKEN,
                    help="集群令牌；真实部署务必替换")
    ap.add_argument("--tcp-port", type=int, default=TCP_PORT)
    ap.add_argument("--udp-port", type=int, default=UDP_PORT)
    ap.add_argument("--ui-port", type=int, default=UI_PORT,
                    help="Web 终端监听端口（仅回环）")
    ap.add_argument("--no-ui", action="store_true", help="不启动 Web 终端（服务器模式）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--ui-timeout", type=float, default=30.0,
                    help="等待界面初始化完成的秒数，超时后按无界面模式继续")
    ap.add_argument("--advertise-host", default="",
                    help="告知子端的 Master 地址；空=自动取本机局域网 IP")
    ap.add_argument("--subnet", default="",
                    help="扫描网段（CIDR）；空=按本机 IPv4 推导 /24")
    ap.add_argument("--repo", default="", help="子端部署源码仓库；空=默认仓库")
    ap.add_argument("--branch", default="", help="子端部署分支；空=main")
    ap.add_argument("--community", default="",
                    help="社区数据集目录地址（也可用 MAHE_COMMUNITY_URL）")
    ap.add_argument("--dataset", default="",
                    help=f"默认数据集 id，如 {data.DEFAULT_DATASET_ID}")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--k-steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--min-workers", type=int, default=1)
    ap.add_argument("--wait-timeout", type=float, default=300.0,
                    help="开始训练后等待节点入网的最长秒数")
    ap.add_argument("--ckpt", default=platform_dialogs.default_model_path(),
                    help="模型保存路径（UI 里可改）")
    return ap


def main() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if getattr(sys, "frozen", False):
        # 快捷方式启动时保留一份日志，便于排查
        try:
            log_dir = Path.home() / ".mahe"
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_dir / "engine.log",
                                                encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    args = build_parser().parse_args()

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    test_mode = bool(args.test or args.role == "demo")
    if test_mode:
        args.min_workers = max(2, args.min_workers)
        run_master(args, spawn_workers=True, auto_start=True)
    elif args.role == "worker":
        run_worker(args)
    else:
        run_master(args)


if __name__ == "__main__":
    main()
