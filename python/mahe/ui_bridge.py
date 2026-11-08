"""UI 控制信道：localhost TCP + JSON Lines。

- Master 侧启动 UiBridge，Kotlin UI 进程连接后：
  - 持续接收事件（node_join / node_dead / round / train_start / train_end / log）
  - 可发送命令：{"cmd":"status"} / {"cmd":"stop"}
UI 崩溃或被杀死不影响训练；断开后可重连，重连即收到最新快照。
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import threading

log = logging.getLogger("mahe.ui_bridge")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        bridge: "UiBridge" = self.server.bridge  # type: ignore[attr-defined]
        with bridge._clients_lock:
            bridge._clients.add(self.wfile)
        try:
            # 新客户端：先给快照
            self.wfile.write((json.dumps(bridge.snapshot()) + "\n").encode())
            for line in self.rfile:
                try:
                    cmd = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue
                bridge.handle_command(cmd, self.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with bridge._clients_lock:
                bridge._clients.discard(self.wfile)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class UiBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 47830):
        self.host = host
        self.port = port
        self._server = _Server((host, port), _Handler)
        self._server.bridge = self  # type: ignore[attr-defined]
        self._clients: set = set()
        self._clients_lock = threading.Lock()
        self._state: dict = {"type": "snapshot", "status": "idle", "rounds": []}
        self._on_stop = None
        self._thread: threading.Thread | None = None

    def start(self, on_stop=None) -> None:
        self._on_stop = on_stop
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.5},
            daemon=True, name="ui-bridge",
        )
        self._thread.start()
        log.info("UI 控制信道：%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # ---- 训练侧调用 ----
    def emit(self, ev: dict) -> None:
        if ev.get("type") == "round":
            self._state.setdefault("rounds", []).append(ev)
            self._state["status"] = "training"
        elif ev.get("type") == "train_end":
            self._state["status"] = "done"
        elif ev.get("type") == "train_start":
            self._state["status"] = "training"
        line = (json.dumps(ev) + "\n").encode()
        with self._clients_lock:
            dead = []
            for w in self._clients:
                try:
                    w.write(line)
                    w.flush()
                except (BrokenPipeError, ConnectionResetError):
                    dead.append(w)
            for w in dead:
                self._clients.discard(w)

    # ---- UI 侧消息 ----
    def snapshot(self) -> dict:
        return self._state

    def handle_command(self, cmd: dict, wfile) -> None:
        c = cmd.get("cmd")
        if c == "status":
            wfile.write((json.dumps(self.snapshot()) + "\n").encode())
        elif c == "stop":
            self._state["status"] = "stopping"
            if self._on_stop:
                threading.Thread(target=self._on_stop, daemon=True).start()
