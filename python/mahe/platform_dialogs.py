"""跨平台原生文件对话框（在引擎所在主机上弹出）。

浏览器拿不到本地路径，所以「浏览…」按钮由后端进程调用系统对话框：
  - Windows：PowerShell + System.Windows.Forms.SaveFileDialog
  - macOS：osascript choose file name
  - Linux：zenity / kdialog（都不可用时返回空串，让用户手填）

所有函数都只返回路径字符串，取消或环境不支持时返回空串，不抛异常。
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("mahe.dialog")

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"
_CREATE_NO_WINDOW = 0x08000000 if _IS_WIN else 0

TITLE = "选择模型保存位置"
DEFAULT_NAME = "mahe_model.npz"


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _windows(initial_dir: str, default_name: str) -> str:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.SaveFileDialog;"
        f"$d.Title = {_ps_quote(TITLE)};"
        "$d.Filter = 'MAHE 模型 (*.npz)|*.npz|所有文件 (*.*)|*.*';"
        f"$d.FileName = {_ps_quote(default_name)};"
    )
    if initial_dir:
        script += f"$d.InitialDirectory = {_ps_quote(initial_dir)};"
    script += (
        "$d.OverwritePrompt = $true;"
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
        " { [Console]::Out.Write($d.FileName) }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, timeout=600,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("Windows 文件对话框不可用：%s", exc)
        return ""
    return (proc.stdout or "").strip()


def _macos(initial_dir: str, default_name: str) -> str:
    prompt = TITLE.replace('"', "")
    script = (
        f'POSIX path of (choose file name with prompt "{prompt}" '
        f'default name "{default_name}")'
    )
    try:
        proc = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("macOS 文件对话框不可用：%s", exc)
        return ""
    return (proc.stdout or "").strip()


def _linux(initial_dir: str, default_name: str) -> str:
    start = str(Path(initial_dir or "~").expanduser() / default_name)
    commands = [
        ["zenity", "--file-selection", "--save", "--confirm-overwrite",
         f"--title={TITLE}", f"--filename={start}"],
        ["kdialog", "--getsavefilename", start, "*.npz"],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            continue
        path = (proc.stdout or "").strip()
        if proc.returncode == 0 and path:
            return path
        return ""
    return ""


def pick_save_file(initial: str = "", default_name: str = DEFAULT_NAME) -> str:
    """弹出「另存为」对话框，返回绝对路径；取消时返回空串。"""
    start = str(Path(initial).expanduser()) if initial else ""
    initial_dir = str(Path(start).parent) if start else str(Path.home())
    name = Path(start).name if start else default_name

    if _IS_WIN:
        return _windows(initial_dir, name)
    if _IS_MAC:
        return _macos(initial_dir, name)
    return _linux(initial_dir, name)


def default_model_path() -> str:
    """默认模型保存位置：用户目录下，避免写系统路径失败。"""
    if getattr(sys, "frozen", False):
        return str(Path.home() / ".mahe" / "checkpoints" / DEFAULT_NAME)
    return os.path.join("checkpoints", DEFAULT_NAME)
