"""子端部署：从 GitHub 拉取源码并在目标机编译、启动 Worker。

实现方式是“生成引导脚本 + 通过 SSH 执行”：
- 目标机只需要 git、python3(3.10+)、C++ 编译器与 NASM（缺 NASM 时 build.py 自动标量回退）；
- 脚本会浅克隆指定分支到 ``~/.mahe/src``，执行 ``build.py``，随后后台启动 Worker；
- 不支持 SSH 的设备（例如只开 WinRM 的 Windows）返回手动脚本，由用户在目标机执行。

本模块不保存任何密码：自动部署依赖 SSH 密钥（BatchMode），失败时给出明确提示。
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_REPO = "https://github.com/MaherJon/distributed-ai-training-framework.git"
DEFAULT_BRANCH = "main"
DEFAULT_TOKEN = "mahe-demo-token"


def _source_root() -> Path | None:
    if getattr(sys, "frozen", False):
        return None
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str:
    root = _source_root()
    if root is None:
        return ""
    try:
        proc = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=2)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def repo_url(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MAHE_REPO_URL")
    if env:
        return env
    # 源码运行：沿用当前仓库的 origin，避免 fork 后仍拉上游
    remote = _git_output(["remote", "get-url", "origin"])
    return remote or DEFAULT_REPO


def branch_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MAHE_REPO_BRANCH")
    if env:
        return env
    # 源码运行：默认部署当前分支（frozen 安装器则退回 main）
    branch = _git_output(["branch", "--show-current"])
    return branch or DEFAULT_BRANCH


@dataclass
class DeployConfig:
    repo: str = DEFAULT_REPO
    branch: str = DEFAULT_BRANCH
    master_host: str = "127.0.0.1"
    master_port: int = 47820
    token: str = DEFAULT_TOKEN
    workdir: str = "~/.mahe/src"
    hidden: int = 32


def _posix_root(cfg: DeployConfig) -> str:
    if cfg.workdir and not cfg.workdir.startswith("~"):
        return shlex.quote(cfg.workdir)
    return '"$HOME/.mahe/src"'


def posix_script(cfg: DeployConfig) -> str:
    """生成 POSIX（Linux/macOS）Worker 部署脚本。"""
    repo = shlex.quote(cfg.repo)
    branch = shlex.quote(cfg.branch)
    master = shlex.quote(cfg.master_host)
    token = shlex.quote(cfg.token)
    root = _posix_root(cfg)
    return f"""#!/usr/bin/env sh
# MAHE Worker 自动部署脚本（由 Master 生成，可重复执行）
set -eu

REPO={repo}
BRANCH={branch}
ROOT={root}
MASTER={master}
PORT={cfg.master_port}
TOKEN={token}
HIDDEN={cfg.hidden}

echo "[MAHE] 目标机：$(uname -s) $(uname -m)"
if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 未找到 git，请先安装：apt install git / brew install git"
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python3（需要 3.10+）"
  exit 1
fi

if [ -d "$ROOT/.git" ]; then
  echo "[MAHE] 更新源码：$ROOT ($BRANCH)"
  git -C "$ROOT" fetch --depth 1 origin "$BRANCH"
  git -C "$ROOT" checkout -B "$BRANCH" FETCH_HEAD
else
  echo "[MAHE] 拉取源码：$REPO ($BRANCH)"
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$ROOT"
fi

cd "$ROOT"
PY=python3
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python3 -m venv "$ROOT/.venv" >/dev/null 2>&1 || true
fi
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
  echo "[MAHE] 使用虚拟环境：$ROOT/.venv"
else
  echo "[WARN] 无法创建虚拟环境，回退系统 Python"
fi
echo "[MAHE] 安装 Python 依赖…"
"$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
if ! "$PY" -m pip install -q numpy pybind11; then
  "$PY" -m pip install -q --user numpy pybind11 2>/dev/null || \
    "$PY" -m pip install -q --break-system-packages numpy pybind11 || \
    echo "[WARN] 依赖安装失败，继续尝试构建"
fi

echo "[MAHE] 编译 C++ 核心与扩展…"
"$PY" build.py

mkdir -p "$HOME/.mahe/logs"
if [ -f "$HOME/.mahe/logs/worker.pid" ]; then
  kill "$(cat "$HOME/.mahe/logs/worker.pid")" 2>/dev/null || true
fi
echo "[MAHE] 启动 Worker…"
PYTHONPATH="$ROOT/python" nohup "$PY" -m mahe.demo_train \\
  --role worker --master "$MASTER" --master-port "$PORT" --token "$TOKEN" \\
  --hidden "$HIDDEN" \\
  >"$HOME/.mahe/logs/worker.log" 2>&1 &
echo $! >"$HOME/.mahe/logs/worker.pid"
echo "[OK] Worker 已启动，pid=$(cat "$HOME/.mahe/logs/worker.pid")"
echo "[OK] 日志：$HOME/.mahe/logs/worker.log"
"""


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def windows_script(cfg: DeployConfig) -> str:
    """生成 Windows PowerShell Worker 部署脚本（手动执行）。"""
    repo = _ps_quote(cfg.repo)
    branch = _ps_quote(cfg.branch)
    master = _ps_quote(cfg.master_host)
    token = _ps_quote(cfg.token)
    return f"""# MAHE Worker 自动部署脚本（由 Master 生成，可重复执行）
$ErrorActionPreference = 'Stop'
$Repo = {repo}
$Branch = {branch}
$Master = {master}
$Port = {cfg.master_port}
$Token = {token}
$Hidden = {cfg.hidden}
$Root = Join-Path $env:USERPROFILE '.mahe\\src'

Write-Host "[MAHE] 目标机：Windows $([Environment]::OSVersion.VersionString)"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {{
    throw "[ERROR] 未找到 git，请先安装 Git for Windows"
}}
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {{ $pyExe = $py.Source; $pyArgs = @('-3') }}
else {{
    $p = Get-Command python -ErrorAction SilentlyContinue
    if (-not $p) {{ throw "[ERROR] 未找到 Python（需要 3.10+）" }}
    $pyExe = $p.Source; $pyArgs = @()
}}

if (Test-Path (Join-Path $Root '.git')) {{
    Write-Host "[MAHE] 更新源码：$Root ($Branch)"
    git -C $Root fetch --depth 1 origin $Branch
    git -C $Root checkout -B $Branch FETCH_HEAD
}} else {{
    Write-Host "[MAHE] 拉取源码：$Repo ($Branch)"
    New-Item -ItemType Directory -Force (Split-Path $Root) | Out-Null
    git clone --depth 1 --branch $Branch $Repo $Root
}}

Set-Location $Root
Write-Host "[MAHE] 安装 Python 依赖…"
& $pyExe @pyArgs -m pip install --user -q numpy pybind11

Write-Host "[MAHE] 编译 C++ 核心与扩展…"
& $pyExe @pyArgs build.py

$logDir = Join-Path $env:USERPROFILE '.mahe\\logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONPATH = Join-Path $Root 'python'
$workerArgs = @('-m', 'mahe.demo_train', '--role', 'worker', '--master', $Master,
               '--master-port', "$Port", '--token', $Token, '--hidden', "$Hidden")
Write-Host "[MAHE] 启动 Worker…"
Start-Process -WindowStyle Hidden -FilePath $pyExe -ArgumentList (@() + $pyArgs + $workerArgs) `
    -RedirectStandardOutput (Join-Path $logDir 'worker.out.log') `
    -RedirectStandardError (Join-Path $logDir 'worker.err.log')
Write-Host "[OK] Worker 已启动，日志目录：$logDir"
"""


def script_for(cfg: DeployConfig, platform_name: str) -> str:
    name = (platform_name or "posix").lower()
    if name in ("windows", "win", "powershell", "ps1"):
        return windows_script(cfg)
    return posix_script(cfg)


def run_ssh_deploy(
    host: str,
    cfg: DeployConfig,
    *,
    user: str = "",
    key_path: str | None = None,
    on_log: Callable[[str], None] | None = None,
    timeout: float = 1800.0,
) -> int:
    """通过 SSH 在目标机执行部署脚本，返回退出码。

    依赖 ssh 客户端与密钥登录（BatchMode）。密码登录请改用手动脚本。
    """
    def log(text: str) -> None:
        if on_log:
            on_log(text)

    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError("本机未找到 ssh 客户端，请改用手动部署脚本")

    target = f"{user}@{host}" if user else host
    args = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8"]
    if key_path:
        args += ["-i", os.path.expanduser(key_path)]
    args += [target, "sh -s"]

    log(f"$ ssh {target} sh -s")
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(posix_script(cfg))
        proc.stdin.close()
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip())
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        log("[ERROR] 部署超时，已终止")
        return -1
    finally:
        if proc.poll() is None:
            proc.kill()

    if rc == 255:
        log("[ERROR] SSH 连接或认证失败：请确认目标机已安装 SSH 服务，"
            "且 Master 已配置免密登录（ssh-copy-id）")
    elif rc != 0:
        log(f"[ERROR] 远端部署失败，退出码 {rc}")
    else:
        log("[OK] 远端部署完成，Worker 应在数秒内入网")
    return rc


def manual_hint(cfg: DeployConfig) -> str:
    return (
        "手动部署：在界面中下载 POSIX / Windows 脚本，复制到目标机执行；\n"
        f"目标仓库：{cfg.repo}（分支 {cfg.branch}）\n"
        "脚本会执行：git 浅克隆 → pip 安装依赖 → python build.py → 后台启动 Worker。"
    )
