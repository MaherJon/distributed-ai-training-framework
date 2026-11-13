"""MAHE 安装器入口（由 build.py --bundle 打成单文件安装器）。

安装行为（用户级，无需管理员权限）：
- Windows：复制到 %LOCALAPPDATA%\\Programs\\MAHE，创建桌面与开始菜单快捷方式，
  写入卸载注册表项，并生成 uninstall.ps1；
- Linux：复制到 ~/.local/share/mahe，写入应用菜单 .desktop 并在桌面创建入口；
- macOS：复制到 ~/Applications/MAHE，生成 MAHE.app 并在桌面创建入口。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "MAHE"
VERSION = "0.1.0"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _payload_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    payload = base / "app"
    if not payload.is_dir():
        raise SystemExit(f"安装器载荷缺失：{payload}")
    return payload


def _install_root() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Programs" / APP_NAME
    if system == "Darwin":
        return Path.home() / "Applications" / APP_NAME
    return Path.home() / ".local" / "share" / "mahe"


def _engine_exe(root: Path) -> Path:
    name = "mahe-engine.exe" if os.name == "nt" else "mahe-engine"
    return root / "mahe-engine" / name


def _copy_payload(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_powershell(script: str, env: dict[str, str]) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
        env=env,
        creationflags=CREATE_NO_WINDOW,
    )


# ---------------- Windows ----------------
def install_windows(root: Path, exe: Path) -> None:
    uninstall = root / "uninstall.ps1"
    uninstall.write_text(
        "$ErrorActionPreference = 'SilentlyContinue'\n"
        "Get-Process mahe-engine | Stop-Process -Force\n"
        f"Remove-Item -Recurse -Force {_ps_quote(str(root))}\n"
        "Remove-Item -Force (Join-Path ([Environment]::GetFolderPath('Desktop')) 'MAHE.lnk')\n"
        "Remove-Item -Force (Join-Path ([Environment]::GetFolderPath('Programs')) 'MAHE.lnk')\n"
        "Remove-Item -Recurse -Force 'HKCU:\\Software\\Microsoft\\Windows\\"
        "CurrentVersion\\Uninstall\\MAHE'\n",
        encoding="utf-8-sig",
    )

    env = os.environ.copy()
    env["MAHE_EXE"] = str(exe)
    env["MAHE_DIR"] = str(root)
    env["MAHE_UNINSTALL"] = str(uninstall)
    script = r"""
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'),
                   [Environment]::GetFolderPath('Programs'))) {
  $lnk = $ws.CreateShortcut((Join-Path $dir 'MAHE.lnk'))
  $lnk.TargetPath = $env:MAHE_EXE
  $lnk.WorkingDirectory = $env:MAHE_DIR
  $lnk.Description = 'MAHE 分布式 AI 训练终端'
  $lnk.Save()
}
$key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MAHE'
New-Item -Path $key -Force | Out-Null
Set-ItemProperty -Path $key -Name DisplayName -Value 'MAHE'
Set-ItemProperty -Path $key -Name DisplayVersion -Value '0.1.0'
Set-ItemProperty -Path $key -Name Publisher -Value 'MAHE'
Set-ItemProperty -Path $key -Name InstallLocation -Value $env:MAHE_DIR
Set-ItemProperty -Path $key -Name UninstallString `
  -Value ('powershell -ExecutionPolicy Bypass -File "' + $env:MAHE_UNINSTALL + '"')
"""
    _run_powershell(script, env)


def _desktop_dir() -> Path:
    """优先读取 XDG 配置（中文桌面可能是 ~/桌面），否则回退 ~/Desktop。"""
    cfg = Path.home() / ".config" / "user-dirs.dirs"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.startswith("XDG_DESKTOP_DIR="):
                value = line.split("=", 1)[1].strip().strip('"')
                value = value.replace("$HOME", str(Path.home()))
                if value:
                    return Path(value)
    except OSError:
        pass
    return Path.home() / "Desktop"


# ---------------- Linux ----------------
def install_linux(root: Path, exe: Path) -> None:
    exe.chmod(0o755)
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=MAHE\n"
        "Comment=MAHE 分布式 AI 训练终端\n"
        f'Exec="{exe}"\n'
        f'Path="{exe.parent}"\n'
        "Terminal=false\n"
        "Categories=Development;Education;\n"
        "StartupNotify=false\n"
    )
    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    menu_file = apps_dir / "mahe.desktop"
    menu_file.write_text(entry, encoding="utf-8")
    menu_file.chmod(0o755)

    desktop = _desktop_dir()
    desktop_file = desktop / "mahe.desktop"
    if desktop.is_dir():
        desktop_file.write_text(entry, encoding="utf-8")
        desktop_file.chmod(0o755)
        # GNOME 需要 trusted 标记，失败不影响使用
        subprocess.run(
            ["gio", "set", str(desktop_file), "metadata::trusted", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    (root / "uninstall.sh").write_text(
        "#!/bin/sh\n"
        f'rm -rf "{root}"\n'
        f'rm -f "{menu_file}"\n'
        f'rm -f "{desktop_file}"\n',
        encoding="utf-8",
    )
    (root / "uninstall.sh").chmod(0o755)


# ---------------- macOS ----------------
def install_macos(root: Path, exe: Path) -> None:
    exe.chmod(0o755)
    app = Path.home() / "Applications" / f"{APP_NAME}.app"
    macos_dir = app / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Info.plist").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MAHE</string>
  <key>CFBundleDisplayName</key><string>MAHE</string>
  <key>CFBundleIdentifier</key><string>org.mahe.terminal</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>MAHE</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    launcher = macos_dir / "MAHE"
    launcher.write_text(f'#!/bin/sh\nexec "{exe}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)

    desktop_link = Path.home() / "Desktop" / f"{APP_NAME}.app"
    if not desktop_link.exists():
        try:
            desktop_link.symlink_to(app)
        except OSError:
            pass

    (root / "uninstall.sh").write_text(
        "#!/bin/sh\n"
        f'rm -rf "{root}"\n'
        f'rm -rf "{app}"\n'
        f'rm -f "{desktop_link}"\n',
        encoding="utf-8",
    )
    (root / "uninstall.sh").chmod(0o755)


def main() -> None:
    src = _payload_dir()
    root = _install_root()
    print(f"MAHE 安装器 {VERSION}")
    print(f"安装目录：{root}")
    print("正在复制程序文件…")
    _copy_payload(src, root)

    exe = _engine_exe(root)
    if not exe.is_file():
        raise SystemExit(f"安装失败：未找到引擎 {exe}")

    print("正在创建快捷方式…")
    if os.name == "nt":
        install_windows(root, exe)
    elif sys.platform == "darwin":
        install_macos(root, exe)
    else:
        install_linux(root, exe)

    print("安装完成：已创建桌面快捷方式。")
    print("卸载：运行安装目录下的 uninstall 脚本。")
    if os.name == "nt":
        try:
            input("按 Enter 退出…")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
