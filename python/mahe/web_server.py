"""MAHE Web 控制终端后端。

进程拓扑：浏览器（独立进程） ⇄ localhost:47830 ⇄ 本模块 ⇄ 训练模块。

- 静态页面、SSE 事件流与 JSON 指令通道共用一个回环 HTTP 服务；
- 前端把启动字幕播放完毕后才 POST ``ui_ready``，服务端收到该信号前
  不会初始化传输层、调度器与训练模块；
- 训练侧日志通过 ``WebUiServer.emit_info`` 输出到界面左侧信息流，
  前端按 ``[信息类型] 内容`` 渲染；
- UI 断开或刷新不影响训练，重连后先收到完整快照。
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("mahe.web")

WEB_ROOT = Path(__file__).resolve().parent / "webui"
MAX_INFO = 400
MAX_EVENTS = 300


class UiState:
    """界面状态快照（所有访问都在锁内）。"""

    def __init__(self, ui_port: int) -> None:
        self._lock = threading.Lock()
        self.ui_port = ui_port
        self.status = "booting"
        self.kernel = ""
        self.master_host = ""
        self.round = 0
        self.info: list[dict[str, Any]] = []
        self.rounds: list[dict[str, Any]] = []
        self.nodes: dict[int, dict[str, Any]] = {}
        self.devices: list[dict[str, Any]] = []
        self.scanning = False
        self.settings: dict[str, Any] = {}
        self.datasets: list[dict[str, Any]] = []
        self.community: dict[str, Any] = {}
        self.train: dict[str, Any] = {"running": False, "rounds_total": 0}
        self.agent: dict[str, Any] = {}
        self.ui_ready = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "snapshot",
                "status": self.status,
                "kernel": self.kernel,
                "master_host": self.master_host,
                "round": self.round,
                "info": list(self.info[-120:]),
                "rounds": list(self.rounds[-MAX_EVENTS:]),
                "nodes": list(self.nodes.values()),
                "devices": list(self.devices),
                "scanning": self.scanning,
                "settings": dict(self.settings),
                "datasets": list(self.datasets),
                "community": dict(self.community),
                "train": dict(self.train),
                "agent": dict(self.agent),
                "ui_ready": self.ui_ready.is_set(),
                "ui_port": self.ui_port,
            }

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def add_info(self, kind: str, text: str) -> list[dict[str, Any]]:
        ev = {
            "type": "info",
            "kind": kind,
            "text": text,
            "ts": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            self.info.append(ev)
            if len(self.info) > MAX_INFO:
                del self.info[: len(self.info) - MAX_INFO]
            return [ev]

    def add_round(self, stat: dict[str, Any]) -> None:
        with self._lock:
            self.round = int(stat.get("round") or self.round)
            self.rounds.append(stat)
            if len(self.rounds) > MAX_EVENTS:
                del self.rounds[: len(self.rounds) - MAX_EVENTS]

    def upsert_node(self, node: dict[str, Any]) -> None:
        with self._lock:
            nid = int(node.get("id") or 0)
            if nid:
                self.nodes[nid] = {**self.nodes.get(nid, {}), **node}

    def mark_node_dead(self, node: dict[str, Any]) -> None:
        with self._lock:
            nid = int(node.get("id") or 0)
            if nid in self.nodes:
                self.nodes[nid]["alive"] = False

    def remove_node(self, node_id: int) -> None:
        with self._lock:
            self.nodes.pop(int(node_id), None)

    def set_settings(self, update: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.settings.update(update)
            return dict(self.settings)

    def set_datasets(self, datasets: list[dict[str, Any]],
                     community: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.datasets = list(datasets)
            if community is not None:
                self.community = dict(community)

    def set_train(self, running: bool, rounds_total: int = 0) -> None:
        with self._lock:
            self.train = {"running": bool(running),
                          "rounds_total": int(rounds_total)}

    def set_agent(self, info: dict[str, Any]) -> None:
        with self._lock:
            self.agent = dict(info)

    def set_devices(self, devices: list[dict[str, Any]]) -> None:
        with self._lock:
            self.devices = list(devices)

    def set_scanning(self, value: bool) -> None:
        with self._lock:
            self.scanning = value


class EventHub:
    """SSE 订阅者广播（队列满时丢弃最旧事件，避免拖慢训练线程）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: set[queue.Queue] = set()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=512)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    def close(self) -> None:
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for q in subs:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "MAHE-Web/1.0"
    protocol_version = "HTTP/1.1"

    app: "WebUiServer"  # 由 _make_handler 注入

    # ---- 基础响应 ----
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _text(self, text: str, ctype: str = "text/plain; charset=utf-8") -> None:
        self._send(200, text.encode("utf-8"), ctype)

    # ---- GET ----
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/events":
                self._serve_events()
            elif path == "/api/status":
                self._json(self.app.state.snapshot())
            elif path == "/api/deploy/script":
                qs = parse_qs(parsed.query)
                platform = (qs.get("platform") or ["posix"])[0].lower()
                self._text(self.app.deploy_script(platform))
            else:
                self._serve_static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        rel = path.lstrip("/")
        target = (WEB_ROOT / rel).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._json({"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in (
            "application/javascript", "application/json",
        ):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = self.app.hub.subscribe()
        try:
            self._sse(self.app.state.snapshot())
            while self.app.running:
                try:
                    ev = q.get(timeout=10.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if ev is None:
                    break
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.app.hub.unsubscribe(q)

    def _sse(self, event: dict[str, Any]) -> None:
        data = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ---- POST ----
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/command":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            cmd = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(cmd, dict):
                raise ValueError("command must be an object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._json({"error": f"bad request: {exc}"}, 400)
            return
        try:
            result = self.app.handle_command(cmd)
            self._json({"ok": True, **(result or {})})
        except Exception as exc:  # 指令失败不应打断服务
            log.exception("command failed: %s", cmd)
            self._json({"ok": False, "error": str(exc)}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        log.debug("%s - %s", self.address_string(), fmt % args)


def _make_handler(app: "WebUiServer") -> type[_Handler]:
    return type("BoundHandler", (_Handler,), {"app": app})


class WebUiServer:
    """回环 Web 服务：静态资源 + SSE + 指令通道。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 47830) -> None:
        self.host = host
        self.port = port
        self.state = UiState(port)
        self.hub = EventHub()
        self.running = False
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        # 由控制面注入
        self.on_ui_ready: Callable[[], None] | None = None
        self.on_scan: Callable[[dict[str, Any]], None] | None = None
        self.on_deploy: Callable[[dict[str, Any]], None] | None = None
        self.on_stop: Callable[[], None] | None = None
        self.on_shutdown: Callable[[], None] | None = None
        self.on_start_training: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self.on_kick_node: Callable[[dict[str, Any]], None] | None = None
        self.on_clear_nodes: Callable[[], None] | None = None
        self.on_pick_path: Callable[[dict[str, Any]], str] | None = None
        self.on_refresh_datasets: Callable[[], dict[str, Any]] | None = None
        self.on_agent_ask: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._script_provider: Callable[[str], str] | None = None

    # ---- 生命周期 ----
    def start(self, open_browser: bool = True) -> None:
        if not WEB_ROOT.is_dir():
            raise RuntimeError(f"Web UI 资源缺失：{WEB_ROOT}")
        self._httpd = ThreadingHTTPServer((self.host, self.port),
                                          _make_handler(self))
        self._httpd.daemon_threads = True
        self.running = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.4},
            daemon=True,
            name="mahe-web",
        )
        self._thread.start()
        log.info("Web UI：http://%s:%d/", self.host, self.port)
        if open_browser:
            threading.Thread(target=self._open_browser, daemon=True).start()

    def _open_browser(self) -> None:
        # 服务刚起时浏览器偶发连接过早，稍等一拍
        time.sleep(0.6)
        try:
            webbrowser.open(f"http://{self.host}:{self.port}/")
        except Exception:  # 无桌面环境时静默，用户可手动打开
            log.debug("无法自动打开浏览器", exc_info=True)

    def stop(self) -> None:
        self.running = False
        self.hub.close()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ---- 事件 ----
    def publish(self, event: dict[str, Any]) -> None:
        self.hub.publish(event)

    def emit_info(self, kind: str, text: str) -> None:
        events = self.state.add_info(kind, text)
        for ev in events:
            self.hub.publish(ev)

    def set_status(self, status: str) -> None:
        self.state.set_status(status)
        self.hub.publish({"type": "status", "status": status})

    def set_agent(self, info: dict[str, Any]) -> None:
        self.state.set_agent(info)
        self.hub.publish({"type": "agent", "agent": info})

    def publish_training_event(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "round":
            self.state.add_round(ev)
        elif t == "node_join":
            self.state.upsert_node({
                "id": ev.get("id"),
                "name": ev.get("name"),
                "cores": ev.get("cores", 0),
                "score": ev.get("score", 1.0),
                "alive": True,
            })
        elif t == "node_dead":
            self.state.mark_node_dead(ev)
        elif t == "node_removed":
            self.state.remove_node(int(ev.get("id") or 0))
        elif t == "train_start":
            self.state.set_train(True, int(ev.get("rounds") or 0))
        elif t == "train_end":
            self.state.set_train(False)
        self.hub.publish(ev)

    def publish_train_state(self, running: bool, rounds_total: int = 0) -> None:
        self.state.set_train(running, rounds_total)
        self.hub.publish({"type": "train_state", "running": bool(running),
                          "rounds_total": int(rounds_total)})

    def publish_scan_device(self, device: dict[str, Any]) -> None:
        devices = {d["ip"]: d for d in self.state.devices}
        devices[device["ip"]] = device
        self.state.set_devices(list(devices.values()))
        self.hub.publish({"type": "scan", "phase": "device", "device": device})

    def publish_deploy(self, phase: str, text: str = "",
                       device: dict[str, Any] | None = None) -> None:
        ev: dict[str, Any] = {"type": "deploy", "phase": phase, "text": text}
        if device:
            ev["device"] = device
        self.hub.publish(ev)

    def wait_ui_ready(self, timeout: float) -> bool:
        return self.state.ui_ready.wait(timeout=timeout)

    # ---- 指令 ----
    def deploy_script(self, platform: str) -> str:
        if self._script_provider is None:
            return "# 部署脚本生成器尚未就绪"
        return self._script_provider(platform)

    def set_script_provider(self, provider: Callable[[str], str]) -> None:
        self._script_provider = provider

    def handle_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        name = str(cmd.get("cmd") or "")
        if name == "ui_ready":
            if not self.state.ui_ready.is_set():
                self.state.ui_ready.set()
                self.set_status("initializing")
                self.emit_info("UI", "界面初始化完成，通知训练模块开始初始化")
                if self.on_ui_ready:
                    self.on_ui_ready()
            return {"status": self.state.status}
        if name == "status":
            return {"state": self.state.snapshot()}
        if name == "scan_start":
            if self.state.scanning:
                return {"status": "scanning"}
            if self.on_scan:
                threading.Thread(target=self.on_scan, args=(cmd,),
                                 daemon=True, name="mahe-scan").start()
            return {"status": "scanning"}
        if name == "deploy":
            if self.on_deploy:
                threading.Thread(target=self.on_deploy, args=(cmd,),
                                 daemon=True, name="mahe-deploy").start()
            return {"status": "deploying"}
        if name == "start_training":
            if self.on_start_training is None:
                return {"ok": False, "error": "训练模块尚未初始化"}
            return self.on_start_training(cmd)
        if name == "kick_node":
            node_id = int(cmd.get("id") or 0)
            if node_id <= 0:
                return {"ok": False, "error": "缺少节点编号"}
            if self.on_kick_node:
                self.on_kick_node(cmd)
            return {"status": "ok", "id": node_id}
        if name == "clear_nodes":
            if self.on_clear_nodes:
                self.on_clear_nodes()
            return {"status": "ok"}
        if name == "pick_model_path":
            path = self.on_pick_path(cmd) if self.on_pick_path else ""
            return {"path": path}
        if name == "refresh_datasets":
            if self.on_refresh_datasets is None:
                return {"ok": False, "error": "目录服务未就绪"}
            return self.on_refresh_datasets()
        if name == "agent_ask":
            if self.on_agent_ask is None:
                return {"ok": False, "error": "智能助手未启用"}
            return self.on_agent_ask(cmd)
        if name == "stop_training":
            if self.on_stop:
                self.on_stop()
            return {"status": "stopping"}
        if name == "shutdown":
            if self.on_shutdown:
                self.on_shutdown()
            return {"status": "stopping"}
        return {"error": f"unknown command: {name}"}
