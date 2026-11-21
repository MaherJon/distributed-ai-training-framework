"""Web 控制面后端测试：状态快照、ui_ready 幂等、异步对话框与指令分发。"""

from __future__ import annotations

import queue

from mahe.web_server import UiState, WebUiServer


def _app() -> WebUiServer:
    return WebUiServer(host="127.0.0.1", port=0)


def test_snapshot_contains_ui_state():
    snap = UiState(47830).snapshot()
    for key in ("status", "kernel", "master_host", "rounds", "nodes",
                "devices", "scanning", "settings", "datasets", "community",
                "train", "agent", "pick", "ui_ready", "ui_port"):
        assert key in snap, key
    assert snap["ui_ready"] is False
    assert snap["pick"] == {"status": "idle"}


def test_ui_ready_is_idempotent():
    app = _app()
    calls: list[int] = []
    app.on_ui_ready = lambda: calls.append(1)

    app.handle_command({"cmd": "ui_ready"})
    app.handle_command({"cmd": "ui_ready"})

    assert calls == [1]
    assert app.state.ui_ready.is_set()
    assert app.state.status == "initializing"


def test_pick_model_path_returns_pending_without_blocking():
    app = _app()
    seen: list[dict] = []

    def handler(cmd: dict) -> dict:
        seen.append(cmd)
        return {"status": "pending"}

    app.on_pick_path = handler
    result = app.handle_command(
        {"cmd": "pick_model_path", "current": "/tmp/old.npz"})

    assert result == {"status": "pending"}
    assert seen and seen[0]["current"] == "/tmp/old.npz"


def test_pick_model_path_without_handler_is_explicit():
    result = _app().handle_command({"cmd": "pick_model_path"})
    assert result["status"] == "unavailable"


def test_publish_model_path_updates_state_and_stream():
    app = _app()
    q = app.hub.subscribe()
    app.publish_model_path("ok", "/tmp/model.npz", "")

    event = q.get_nowait()
    assert event == {"type": "model_path", "status": "ok",
                     "path": "/tmp/model.npz", "detail": ""}
    assert app.state.snapshot()["pick"]["path"] == "/tmp/model.npz"


def test_train_state_roundtrip():
    app = _app()
    app.publish_train_state(True, 42)
    snap = app.state.snapshot()
    assert snap["train"] == {"running": True, "rounds_total": 42}

    app.publish_training_event({"type": "train_end"})
    assert app.state.snapshot()["train"]["running"] is False


def test_node_events_update_snapshot():
    app = _app()
    app.publish_training_event({
        "type": "node_join", "id": 7, "name": "worker-a",
        "cores": 8, "score": 1.5,
    })
    node = app.state.snapshot()["nodes"][0]
    assert node["id"] == 7 and node["alive"] is True

    app.publish_training_event({"type": "node_dead", "id": 7})
    assert app.state.snapshot()["nodes"][0]["alive"] is False

    app.publish_training_event({"type": "node_removed", "id": 7})
    assert app.state.snapshot()["nodes"] == []


def test_kick_node_requires_valid_id():
    app = _app()
    app.on_kick_node = lambda cmd: None
    assert app.handle_command({"cmd": "kick_node"})["ok"] is False
    assert app.handle_command({"cmd": "kick_node", "id": 3})["status"] == "ok"


def test_unknown_command_reports_error():
    result = _app().handle_command({"cmd": "does-not-exist"})
    assert "error" in result


def test_scan_start_is_accepted():
    app = _app()
    app.on_scan = lambda cmd: None
    assert app.handle_command({"cmd": "scan_start"}) == {"status": "scanning"}


def test_event_hub_drops_oldest_when_full():
    hub = WebUiServer(port=0).hub
    q: queue.Queue = hub.subscribe()
    for i in range(513):
        hub.publish({"type": "info", "n": i})
    first = q.get_nowait()
    assert first["n"] >= 1  # 最旧的事件被丢弃，但队列不会阻塞发布方
