"""原生对话框与静态 Web 资源的一致性测试（不需要桌面环境）。"""

from __future__ import annotations

from mahe import platform_dialogs as dialogs


def test_default_model_path_is_npz():
    path = dialogs.default_model_path()
    assert path.endswith(".npz")
    assert "mahe_model" in path


def test_ps_quote_escapes_single_quotes():
    assert dialogs._ps_quote("a'b") == "'a''b'"
    assert dialogs._ps_quote(r"C:\models\x.npz") == r"'C:\models\x.npz'"


def test_dialog_result_ok_flag():
    assert dialogs.DialogResult("ok", "/tmp/a.npz").ok is True
    assert dialogs.DialogResult("cancelled", "").ok is False
    assert dialogs.DialogResult("unavailable", "").ok is False


def test_missing_native_dialog_reports_unavailable(monkeypatch):
    monkeypatch.setattr(dialogs, "_IS_WIN", False)
    monkeypatch.setattr(dialogs, "_IS_MAC", False)
    monkeypatch.setattr(dialogs.shutil, "which", lambda name: None)

    result = dialogs.pick_save_file_result("")
    assert result.status == "unavailable"
    assert result.path == ""
    assert "zenity" in result.detail or "kdialog" in result.detail
    # 兼容旧接口：失败时返回空串而不是抛异常
    assert dialogs.pick_save_file("") == ""
