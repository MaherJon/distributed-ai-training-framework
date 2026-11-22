"""跨平台原生文件对话框（在引擎所在主机上弹出）。

浏览器拿不到本地路径，所以「浏览…」按钮由后端进程调用系统对话框：
  - Windows：PowerShell + System.Windows.Forms.SaveFileDialog
  - macOS：osascript choose file name
  - Linux：zenity / kdialog（都不可用时返回不可用状态）

设计要点（针对“点了没反应”）：
  - Windows 上先显示一个不可见的 TopMost 宿主窗体，再把对话框作为它的**所有窗**
    弹出，避免对话框被全屏浏览器压到后面；命令行用 ``-EncodedCommand`` 传入，
    不再受控制台代码页影响；
  - 结果通过临时文件回传并明确区分 ``ok / cancelled / unavailable / error``，
    调用方必须给用户可读反馈，而不是静默返回空串；
  - 不在无桌面会话里硬等 600s：Windows 先检查 ``UserInteractive``，非交互直接
    返回 ``unavailable``，让前端提示“请手填路径”。
"""

from __future__ import annotations

import base64
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mahe.dialog")

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"
_CREATE_NO_WINDOW = 0x08000000 if _IS_WIN else 0

TITLE = "选择模型保存位置"
DEFAULT_NAME = "mahe_model.npz"

# 对话框等待上限：超时后返回 unavailable，前端不会永久转圈
DIALOG_TIMEOUT = 600.0


@dataclass
class DialogResult:
    """原生对话框结果。status 取值：ok / cancelled / unavailable / error。"""

    status: str
    path: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.path)


def _result(status: str, path: str = "", detail: str = "") -> DialogResult:
    return DialogResult(status=status, path=path, detail=detail)


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _powershell_executable() -> str | None:
    """优先 Windows PowerShell 5.1，其次 PowerShell 7（pwsh）。"""
    for name in ("powershell", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _windows(initial_dir: str, default_name: str) -> DialogResult:
    ps = _powershell_executable()
    if not ps:
        return _result("unavailable", detail="未找到 PowerShell，无法打开系统文件对话框")

    # 结果写临时文件，规避中文路径在不同代码页/控制台下的乱码问题
    fd, result_file = tempfile.mkstemp(prefix="mahe-pick-", suffix=".txt")
    os.close(fd)
    try:
        lines = [
            "$ErrorActionPreference = 'Stop'",
            f"$resultFile = {_ps_quote(result_file)}",
            ("if (-not [Environment]::UserInteractive) { "
             "[Console]::Out.Write('NOGUI'); exit 0 }"),
            "try {",
            "  Add-Type -AssemblyName System.Windows.Forms",
            "  Add-Type -AssemblyName System.Drawing",
            "  $owner = New-Object System.Windows.Forms.Form",
            "  $owner.Text = 'MAHE'",
            "  $owner.TopMost = $true",
            "  $owner.ShowInTaskbar = $false",
            "  $owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None",
            "  $owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen",
            "  $owner.Size = New-Object System.Drawing.Size(1, 1)",
            "  $owner.Opacity = 0.01",
            "  $owner.Show()",
            "  [System.Windows.Forms.Application]::DoEvents()",
            "  $d = New-Object System.Windows.Forms.SaveFileDialog",
            f"  $d.Title = {_ps_quote(TITLE)}",
            "  $d.Filter = 'MAHE 模型 (*.npz)|*.npz|所有文件 (*.*)|*.*'",
            f"  $d.FileName = {_ps_quote(default_name)}",
        ]
        if initial_dir:
            lines.append(f"  $d.InitialDirectory = {_ps_quote(initial_dir)}")
        lines += [
            "  $d.OverwritePrompt = $true",
            "  $d.RestoreDirectory = $true",
            "  $d.AddExtension = $true",
            "  $d.DefaultExt = 'npz'",
            "  $shown = $d.ShowDialog($owner)",
            "  $picked = $d.FileName",
            "  $owner.Close()",
            "  if ($shown -eq [System.Windows.Forms.DialogResult]::OK) {",
            "    [System.IO.File]::WriteAllText($resultFile, $picked,",
            "      (New-Object System.Text.UTF8Encoding($false)))",
            "    [Console]::Out.Write('OK')",
            "  } elseif ($shown -eq [System.Windows.Forms.DialogResult]::Cancel) {",
            "    [Console]::Out.Write('CANCEL')",
            "  } else {",
            "    [Console]::Out.Write('ERROR')",
            "  }",
            "} catch {",
            "  [System.IO.File]::WriteAllText($resultFile, $_.Exception.Message,",
            "    (New-Object System.Text.UTF8Encoding($false)))",
            "  [Console]::Out.Write('ERROR')",
            "}",
        ]
        script = "\n".join(lines)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        try:
            proc = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-STA",
                 "-EncodedCommand", encoded],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=DIALOG_TIMEOUT,
                creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            return _result("unavailable", detail="系统文件对话框超时未响应")
        except OSError as exc:
            log.info("Windows 文件对话框不可用：%s", exc)
            return _result("unavailable", detail=f"无法启动 PowerShell：{exc}")

        token = (proc.stdout or "").strip().splitlines()
        token = token[-1].strip() if token else ""
        detail = _read_text(result_file)
        if token == "OK" and detail:
            return _result("ok", path=detail)
        if token == "CANCEL":
            return _result("cancelled", detail="已取消选择")
        if token == "NOGUI":
            return _result("unavailable",
                           detail="当前会话没有可见桌面，无法弹出对话框")
        return _result("error", detail=detail or
                       f"对话框失败（退出码 {proc.returncode}）")
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _macos(initial_dir: str, default_name: str) -> DialogResult:
    if not shutil.which("osascript"):
        return _result("unavailable", detail="未找到 osascript，无法打开系统对话框")
    prompt = TITLE.replace('"', "")
    script = (
        'tell application "System Events" to activate\n'
        "try\n"
        f'  set f to choose file name with prompt "{prompt}" '
        f'default name "{default_name}"\n'
        '  return "OK|" & POSIX path of f\n'
        "on error errMsg number errNum\n"
        "  if errNum = -128 then return \"CANCEL|\"\n"
        '  return "ERROR|" & errMsg\n'
        "end try\n"
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=DIALOG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return _result("unavailable", detail="系统文件对话框超时未响应")
    except OSError as exc:
        log.info("macOS 文件对话框不可用：%s", exc)
        return _result("unavailable", detail=str(exc))

    out = (proc.stdout or "").strip()
    status, _, payload = out.partition("|")
    if status == "OK" and payload:
        return _result("ok", path=payload)
    if status == "CANCEL":
        return _result("cancelled", detail="已取消选择")
    return _result("error", detail=payload or (proc.stderr or "").strip() or
                   "对话框失败")


def _linux(initial_dir: str, default_name: str) -> DialogResult:
    start = str(Path(initial_dir or "~").expanduser() / default_name)
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--save", "--confirm-overwrite",
               f"--title={TITLE}", f"--filename={start}"]
        return _run_linux_dialog(cmd, cancelled_codes={1})
    if shutil.which("kdialog"):
        cmd = ["kdialog", "--getsavefilename", start, "*.npz"]
        return _run_linux_dialog(cmd, cancelled_codes={1})
    return _result("unavailable",
                   detail="未安装 zenity / kdialog，无法打开系统对话框")


def _run_linux_dialog(cmd: list[str],
                      cancelled_codes: set[int]) -> DialogResult:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=DIALOG_TIMEOUT)
    except subprocess.TimeoutExpired:
        return _result("unavailable", detail="系统文件对话框超时未响应")
    except OSError as exc:
        log.info("Linux 文件对话框不可用：%s", exc)
        return _result("unavailable", detail=str(exc))
    if proc.returncode in cancelled_codes:
        return _result("cancelled", detail="已取消选择")
    path = (proc.stdout or "").strip()
    if proc.returncode == 0 and path:
        return _result("ok", path=path)
    detail = (proc.stderr or "").strip() or f"对话框失败（退出码 {proc.returncode}）"
    return _result("error", detail=detail)


def pick_save_file_result(initial: str = "",
                          default_name: str = DEFAULT_NAME) -> DialogResult:
    """弹出「另存为」对话框，返回结构化结果（不抛异常）。"""
    start = str(Path(initial).expanduser()) if initial else ""
    initial_dir = str(Path(start).parent) if start else str(Path.home())
    name = Path(start).name if start else default_name

    if _IS_WIN:
        return _windows(initial_dir, name)
    if _IS_MAC:
        return _macos(initial_dir, name)
    return _linux(initial_dir, name)


def pick_save_file(initial: str = "", default_name: str = DEFAULT_NAME) -> str:
    """兼容旧接口：只在成功时返回路径，其余情况返回空串。"""
    return pick_save_file_result(initial, default_name).path


def default_model_path() -> str:
    """默认模型保存位置：用户目录下，避免写系统路径失败。"""
    if getattr(sys, "frozen", False):
        return str(Path.home() / ".mahe" / "checkpoints" / DEFAULT_NAME)
    return os.path.join("checkpoints", DEFAULT_NAME)
