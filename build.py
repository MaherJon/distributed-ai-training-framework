#!/usr/bin/env python3
"""MAHE 跨平台构建系统。

职责：
  1. 环境探测（OS / 编译器 / NASM / Python 头文件 / pybind11）
  2. 汇编内核构建（x86-64 → NASM；其他架构自动回退标量）
  3. C++ 核心静态库 + pybind11 扩展（输出到 python/mahe/mahe_core）
  4. C++ 单元测试构建与运行（--test）
  5. 集群部署包（--build-cluster）
  6. 一体化安装包（--bundle：PyInstaller 冻结引擎 + jpackage 打包 UI/JRE）

用法：
  python build.py                 # 构建全部（核心 + 扩展）
  python build.py --test          # 构建并运行 C++ 单测
  python build.py --bundle        # 生成当前平台的安装包（MSI/DMG/DEB）
  python build.py --compiler g++-13   # 指定编译器（Linux 多 gcc 并存）
  python build.py --clean

平台说明：
  Windows：两条路径均支持
            · MSVC（cl，需在 "x64 Native Tools" 命令行中运行）
            · MinGW-w64 / MSYS2（g++，推荐新手）
            汇编用 NASM（-f win64）；未安装 NASM 时自动回退标量内核。
  macOS  ：Apple Silicon 用 clang++ 汇编 reduce_neon.S（NEON 加速），Intel 走 NASM；
            --bundle 仅在 macOS 上才能产出 DMG。
  Linux  ：x86-64 用 NASM AVX2，aarch64 用 NEON；--bundle 产出 DEB。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SRC = CORE / "src"
ASM = CORE / "asm"
BINDINGS = CORE / "bindings"
TESTS = CORE / "tests"
BUILD = ROOT / "build"
EXT_OUT = ROOT / "python" / "mahe"

CORE_SOURCES = ["transport.cpp", "discovery.cpp", "scheduler.cpp",
                "kernel.cpp", "node.cpp"]

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_X86_64 = platform.machine().lower() in ("x86_64", "amd64")
IS_AARCH64 = platform.machine().lower() in ("aarch64", "arm64")
HOST_OS = platform.system().lower()  # windows / darwin / linux
HOST_ARCH = platform.machine().lower()  # x86_64 / aarch64 / arm64 / ...


def log(msg: str) -> None:
    print(f"[build] {msg}")


def run(cmd: list[str], **kw) -> None:
    log(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def find_compiler(prefer: str | None) -> str:
    candidates = [prefer] if prefer else []
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        candidates.append(env_cxx)
    if IS_WIN:
        candidates += ["cl", "g++", "clang++"]
    elif IS_MAC:
        candidates += ["clang++", "g++"]
    else:
        candidates += ["g++", "clang++"]
    for c in candidates:
        if c and shutil.which(c):
            return c
    sys.exit("未找到 C++ 编译器。Linux 可 apt install g++；"
             "Windows 请安装 MSVC 或 MSYS2，或用 --compiler 指定。")


def pybind11_includes() -> list[str]:
    try:
        import pybind11

        return [pybind11.get_include()]
    except ImportError:
        sys.exit(
            "未找到 pybind11。请先安装：\n"
            "  python -m pip install pybind11 numpy\n"
            "（项目不静默修改你的环境）"
        )


def build_asm(cxx: str, no_asm: bool, obj_dir: Path) -> Path | None:
    if no_asm:
        log("已通过 --no-asm 禁用汇编内核，使用标量回退")
        return None
    if IS_X86_64 and not IS_MAC:
        nasm = shutil.which("nasm")
        if not nasm:
            if IS_WIN:
                log("警告：未找到 NASM，回退标量内核（--no-asm 可静默此警告）")
                return None
            sys.exit("未找到 NASM。Linux: apt install nasm；"
                     "或用 --no-asm 回退标量内核。")
        obj = obj_dir / ("reduce_avx2.obj" if IS_WIN else "reduce_avx2.o")
        fmt = "win64" if IS_WIN else "elf64"
        defines = ["-DWIN64"] if IS_WIN else ["-DELF"]
        run([nasm, "-f", fmt, *defines, "-o", str(obj),
             str(ASM / "reduce_avx2.asm")])
        return obj
    if IS_AARCH64:
        # aarch64（Apple Silicon / AArch64 Linux）：NEON 汇编，C 编译器直接汇编 .S
        obj = obj_dir / "reduce_neon.o"
        run([cxx, "-c", "-o", str(obj), str(ASM / "reduce_neon.S")])
        return obj
    log("当前架构不支持汇编加速，使用标量回退")
    return None


def is_msvc(cxx: str) -> bool:
    return Path(cxx).name.lower() in ("cl", "cl.exe")


def cxx_flags(cxx: str) -> list[str]:
    if is_msvc(cxx):
        # /utf-8：源码含中文，避免 MSVC 按本地编码误读
        return ["/std:c++17", "/O2", "/EHsc", "/MD", "/utf-8"]
    return ["-std=c++17", "-O2", "-Wall", "-fPIC"]


def include_flags(cxx: str) -> list[str]:
    dirs = [str(CORE / "include"), sysconfig.get_paths()["include"],
            *pybind11_includes()]
    if is_msvc(cxx):
        return [f"/I{d}" for d in dirs]
    return [f"-I{d}" for d in dirs]


def python_link_flags(cxx: str) -> list[str]:
    """Windows 上扩展必须显式链接 Python 的导入库（无 lazy binding）。"""
    if not IS_WIN:
        return []
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    dirs = [Path(p) for p in
            {libdir, str(Path(sys.base_prefix) / "libs"),
             str(Path(sys.base_prefix) / "lib")} if p]
    if is_msvc(cxx):
        out: list[str] = []
        found = next((d / f"python{ver}.lib" for d in dirs
                      if (d / f"python{ver}.lib").exists()), None)
        if found:
            out.append(str(found))
        else:
            out += [f"/LIBPATH:{d}" for d in dirs]
            out.append(f"python{ver}.lib")
        return out
    # MinGW：GNU ld 能直接读取 MSVC 导入库 python312.lib，优先用全路径
    for d in dirs:
        lib = d / f"python{ver}.lib"
        if lib.exists():
            return [str(lib)]
    return [*[f"-L{d}" for d in dirs], f"-lpython{ver}"]


def win_system_libs(cxx: str) -> list[str]:
    """Windows 系统库。MSVC 由源码内 pragma 引入 ws2_32，MinGW 需显式链接。"""
    if not IS_WIN or is_msvc(cxx):
        return []
    return ["-lws2_32"]


def build_objects(cxx: str, obj_dir: Path, extra: list[str]) -> list[Path]:
    objs = []
    for src in CORE_SOURCES:
        ext = ".obj" if is_msvc(cxx) else (".o" if not IS_WIN else ".o")
        obj = obj_dir / (Path(src).stem + ext)
        if is_msvc(cxx):
            run([cxx, *cxx_flags(cxx), *include_flags(cxx), *extra,
                 "/c", str(SRC / src), f"/Fo{obj}"])
        else:
            run([cxx, *cxx_flags(cxx), *include_flags(cxx), *extra,
                 "-c", str(SRC / src), "-o", str(obj)])
        objs.append(obj)
    return objs


def build_extension(cxx: str, objs: list[Path], asm_obj: Path | None) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = EXT_OUT / f"mahe_core{suffix}"
    sources = [str(BINDINGS / "pybind.cpp"), *map(str, objs)]
    if asm_obj:
        sources.append(str(asm_obj))

    if is_msvc(cxx):
        run([cxx, *cxx_flags(cxx), *include_flags(cxx), "/LD", *sources,
             "/link", f"/OUT:{out}", *python_link_flags(cxx)])
    else:
        link = ["-shared"]
        if IS_MAC:
            link += ["-undefined", "dynamic_lookup"]
        elif IS_WIN:
            # -static 让 libgcc / libstdc++ / winpthread 全部静态链接。
            # 注意：Python 3.8+ 加载扩展时不搜索 PATH，若动态依赖 libwinpthread-1.dll
            # 会报 "DLL load failed ... 找不到指定的模块"。
            link += ["-static-libgcc", "-static-libstdc++", "-static"]
            link += python_link_flags(cxx)
            link += win_system_libs(cxx)
        else:
            link += ["-lpthread"]
        run([cxx, *cxx_flags(cxx), *include_flags(cxx), *sources, *link,
             "-o", str(out)])
    log(f"扩展输出：{out}")
    return out


def build_tests(cxx: str, objs: list[Path], asm_obj: Path | None) -> Path:
    out = BUILD / ("test_core.exe" if IS_WIN else "test_core")
    sources = [str(TESTS / "test_core.cpp"), *map(str, objs)]
    if asm_obj:
        sources.append(str(asm_obj))
    if is_msvc(cxx):
        run([cxx, *cxx_flags(cxx), *include_flags(cxx), *sources,
             "/link", f"/OUT:{out}"])
    else:
        # Windows 上 std::thread 经 libstdc++ 传递依赖 winpthreads，无需 -lpthread；
        # 但要显式链接 ws2_32（MinGW 忽略 MSVC 的 #pragma comment）
        link = win_system_libs(cxx) if IS_WIN else ["-lpthread"]
        run([cxx, *cxx_flags(cxx), *include_flags(cxx), *sources, *link,
             "-o", str(out)])
    return out


# =========================================================================
# 集群打包：按设备数量与目标系统，生成 master / worker 部署包
# =========================================================================

def normalize_system(spec: str) -> tuple[str, str]:
    s = spec.strip().lower()
    if "-" not in s:
        sys.exit(f"无法解析系统规格：{spec}（格式应为 os-arch，如 linux-x86_64）")
    osname, arch = s.split("-", 1)
    osname = {"macos": "darwin", "mac": "darwin", "osx": "darwin"}.get(osname, osname)
    # 兼容 arm64/amd64/x64/x86-64 等写法
    arch = {"arm64": "aarch64", "amd64": "x86_64", "x64": "x86_64",
            "x86-64": "x86_64"}.get(arch, arch)
    if osname not in ("linux", "darwin", "windows"):
        sys.exit(f"不支持的操作系统：{osname}（仅支持 linux/darwin/windows）")
    if arch not in ("x86_64", "aarch64"):
        sys.exit(f"不支持的架构：{arch}（仅支持 x86_64/aarch64）")
    return osname, arch


def parse_systems(spec: str) -> list[tuple[str, str]]:
    return [normalize_system(s) for s in spec.split(",") if s.strip()]


def target_matches_host(osname: str, arch: str) -> bool:
    return osname == HOST_OS and arch == HOST_ARCH


def copy_python_package(bundle: Path) -> None:
    """复制 mahe 包（排除已编译的扩展，扩展按需单独放置）。"""
    dst = bundle / "mahe"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(
        EXT_OUT, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                      "mahe_core*.so", "mahe_core*.pyd"),
    )


def write_cluster_launchers(bundle: Path, role: str) -> None:
    posix = f"""#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
if ! ls mahe/mahe_core*.so >/dev/null 2>&1; then
  echo "[mahe] 未找到预编译扩展，正在目标机上构建..."
  sh build_here.sh
fi
export PYTHONPATH="$(pwd)"
exec python3 -m mahe.demo_train --role {role} "$@"
"""
    win = f"""@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found in PATH & exit /b 1)
dir mahe\\mahe_core*.pyd >nul 2>nul
if errorlevel 1 (
  echo [mahe] 未找到预编译扩展，正在目标机上构建...
  call build_here.bat
  if errorlevel 1 exit /b 1
)
set PYTHONPATH=%CD%
python -m mahe.demo_train --role {role} %*
"""
    (bundle / f"run_{role}.sh").write_text(posix)
    (bundle / f"run_{role}.sh").chmod(0o755)
    (bundle / f"run_{role}.bat").write_text(win)

    (bundle / "build_here.sh").write_text(
        "#!/usr/bin/env sh\nset -e\ncd \"$(dirname \"$0\")\"\n"
        "exec python3 build.py \"$@\"\n")
    (bundle / "build_here.sh").chmod(0o755)
    (bundle / "build_here.bat").write_text(
        "@echo off\ncd /d \"%~dp0\"\npython build.py %*\n")


def build_cluster(systems: list[tuple[str, str]], master_index: int,
                  no_asm: bool, compiler: str | None) -> None:
    if not (0 <= master_index < len(systems)):
        sys.exit(f"--master-index 越界：{master_index}（设备数 {len(systems)}）")

    cluster_dir = BUILD / "cluster"
    shutil.rmtree(cluster_dir, ignore_errors=True)
    cluster_dir.mkdir(parents=True)

    # 若有目标与主机一致，先在主机构建一次扩展供这些包复用
    host_ext_path: Path | None = None
    if any(target_matches_host(o, a) for o, a in systems):
        cxx = find_compiler(compiler)
        log(f"主机 {HOST_OS}-{HOST_ARCH} 与部分目标一致，构建扩展...")
        obj_dir = BUILD / "obj"
        obj_dir.mkdir(parents=True, exist_ok=True)
        asm = build_asm(cxx, no_asm, obj_dir)
        objs = build_objects(cxx, obj_dir, [])
        host_ext_path = build_extension(cxx, objs, asm)

    for i, (osname, arch) in enumerate(systems):
        role = "master" if i == master_index else "worker"
        tag = f"{osname}-{arch}"
        bundle = cluster_dir / f"device_{i:02d}_{role}_{tag}"
        bundle.mkdir(parents=True)

        # C++ 核心源码 + 构建脚本 + Python 包
        shutil.copytree(
            CORE, bundle / "core",
            ignore=shutil.ignore_patterns("__pycache__", "build",
                                          "*.o", "*.obj", "*.so", "*.pyd"),
        )
        shutil.copy(ROOT / "build.py", bundle / "build.py")
        req = ROOT / "requirements.txt"
        if req.exists():
            shutil.copy(req, bundle / "requirements.txt")
        copy_python_package(bundle)

        built = False
        if host_ext_path and host_ext_path.exists() and \
                target_matches_host(osname, arch):
            shutil.copy(host_ext_path, bundle / "mahe" / host_ext_path.name)
            built = True
        else:
            (bundle / "BUILD_ON_TARGET.txt").write_text(
                f"目标 {osname}-{arch} 与当前主机 {HOST_OS}-{HOST_ARCH} 不一致，"
                "未预编译扩展。首次运行 run_*.sh / run_*.bat 时会自动在目标机构建；"
                "也可手动执行 build_here.sh（POSIX）或 build_here.bat（Windows）。\n")

        write_cluster_launchers(bundle, role)
        info = {
            "device_index": i,
            "role": role,
            "target_os": osname,
            "target_arch": arch,
            "extension_built": built,
            "run": f"run_{role}.sh (POSIX) / run_{role}.bat (Windows)",
        }
        (bundle / "device_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False))
        log(f"打包：{bundle.relative_to(ROOT)}  role={role}  预编译={built}")

    log(f"集群打包完成（{len(systems)} 台）→ {cluster_dir}")


# =========================================================================
# 一体化安装包：PyInstaller 冻结 Python 引擎 + jpackage 打包 UI 与引擎
# 说明：PyInstaller 与 jpackage 都只产出「本机平台」的产物，
#       要得到三平台安装包需分别在 Windows / macOS / Linux 上各跑一次 --bundle。
# =========================================================================

ENGINE_NAME = "mahe-engine"
ENGINE_ENTRY = ROOT / "packaging" / "engine_entry.py"
UI_DIR = ROOT / "ui"


def compose_res_subdir() -> str:
    """Compose 插件只打包 resources/common、resources/<os>、resources/<os>-<arch>
    三个子目录，引擎必须放进按主机 OS/架构命名的子目录。"""
    os_name = "windows" if IS_WIN else ("macos" if IS_MAC else "linux")
    arch = "arm64" if IS_AARCH64 else "x64"
    return f"{os_name}-{arch}"


APP_RES_DIR = UI_DIR / "resources" / compose_res_subdir() / "engine"
PYI_OUT = BUILD / "pyinstaller"


def build_engine() -> Path:
    """把 Python 控制面 + C++ 扩展冻结成 mahe-engine（onedir，启动快）。"""
    if not ENGINE_ENTRY.exists():
        sys.exit(f"缺少打包入口：{ENGINE_ENTRY}")
    if importlib.util.find_spec("PyInstaller") is None:
        sys.exit("未安装 PyInstaller：\n  python -m pip install pyinstaller")

    shutil.rmtree(PYI_OUT, ignore_errors=True)
    # 剔除 LD_LIBRARY_PATH：否则 PyInstaller 会把宿主环境里的旧 libstdc++ 打进包，
    # 与编译扩展时使用的系统库版本不匹配（表现为运行时 ImportError: mahe_core）
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--onedir", "--name", ENGINE_NAME,
         "--paths", str(ROOT / "python"),
         "--hidden-import", "mahe.mahe_core",
         "--collect-submodules", "mahe",
         "--distpath", str(PYI_OUT),
         "--workpath", str(PYI_OUT / "work"),
         "--specpath", str(PYI_OUT),
         str(ENGINE_ENTRY)], env=env)

    engine_dir = PYI_OUT / ENGINE_NAME
    exe = engine_dir / (ENGINE_NAME + (".exe" if IS_WIN else ""))
    if not exe.is_file():
        sys.exit(f"PyInstaller 未产出引擎可执行文件：{exe}")
    return engine_dir


def install_engine_into_ui(engine_dir: Path) -> None:
    """把引擎目录复制到 UI 的 appResourcesRootDir 下的 <os>-<arch>/engine，
    供 jpackage 按平台打包。"""
    if APP_RES_DIR.exists():
        for p in APP_RES_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
    APP_RES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(engine_dir, APP_RES_DIR, dirs_exist_ok=True)
    log(f"引擎已放入 UI 资源目录：{APP_RES_DIR.relative_to(ROOT)}")


def gradle_task() -> str:
    # 用非 release 任务：packageReleaseMsi 会先跑 ProGuard，
    # 而 Compose/Swing 有大量反射引用会导致 ProGuard 报 800+ unresolved
    # references 直接失败；本应用无需混淆裁剪。
    if IS_WIN:
        return "packageMsi"
    if IS_MAC:
        return "packageDmg"
    return "packageDeb"


def bundle_app(cxx: str, no_asm: bool) -> None:
    """构建扩展 → 冻结引擎 → 调用 Gradle/jpackage 产出安装包。"""
    obj_dir = BUILD / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    asm_obj = build_asm(cxx, no_asm, obj_dir)
    objs = build_objects(cxx, obj_dir, extra=[])
    build_extension(cxx, objs, asm_obj)

    engine_dir = build_engine()
    install_engine_into_ui(engine_dir)

    gradlew = UI_DIR / ("gradlew.bat" if IS_WIN else "gradlew")
    if not gradlew.exists():
        sys.exit(f"未找到 Gradle wrapper：{gradlew}")
    if not shutil.which("java") and not os.environ.get("JAVA_HOME"):
        sys.exit("未找到 JDK（jpackage 需要 JDK 17+）：\n"
                 "  Windows: winget install EclipseAdoptium.Temurin.17.JDK\n"
                 "  macOS:   brew install --cask temurin@17\n"
                 "  Linux:   sudo apt install openjdk-17-jdk")
    if not IS_WIN:
        try:
            gradlew.chmod(0o755)
        except OSError:
            pass
    task = gradle_task()
    log(f"调用 Gradle：{task}（需 JDK 17+，首次会联网下载依赖）")
    run([str(gradlew), task], cwd=str(UI_DIR))

    pattern = {"msi": "*.msi", "dmg": "*.dmg", "deb": "*.deb"}[
        "msi" if IS_WIN else ("dmg" if IS_MAC else "deb")]
    out_dir = UI_DIR / "build" / "compose" / "binaries" / "main" / \
        ("msi" if IS_WIN else ("dmg" if IS_MAC else "deb"))
    outs = sorted(out_dir.glob(pattern)) if out_dir.exists() else []
    if outs:
        for f in outs:
            log(f"安装包：{f}")
    else:
        log(f"未找到安装包产物，请检查 {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MAHE 构建系统")
    ap.add_argument("--compiler", help="指定 C++ 编译器（如 g++-13）")
    ap.add_argument("--no-asm", action="store_true", help="跳过汇编内核")
    ap.add_argument("--test", action="store_true", help="构建并运行 C++ 单测")
    ap.add_argument("--no-ext", action="store_true", help="不构建 Python 扩展")
    ap.add_argument("--clean", action="store_true")
    # 集群打包：按设备数与目标系统生成 master / worker 部署包
    ap.add_argument("--build-cluster", action="store_true",
                    help="按 --systems 列出的设备生成集群部署包")
    ap.add_argument("--systems",
                    help="设备系统列表，逗号分隔，格式 os-arch；"
                         "如 linux-x86_64,linux-aarch64,mac-arm64,windows-x86_64")
    ap.add_argument("--master-index", type=int, default=0,
                    help="哪台设备作为 master（默认第 0 台）")
    # 一体化安装包：Python 引擎 + UI + JRE 打成一个安装包
    ap.add_argument("--bundle", action="store_true",
                    help="生成当前平台的一体化安装包（MSI/DMG/DEB）")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(BUILD, ignore_errors=True)
        for f in EXT_OUT.glob("mahe_core*"):
            f.unlink()
        shutil.rmtree(APP_RES_DIR, ignore_errors=True)
        log("已清理")
        return

    if args.build_cluster:
        if not args.systems:
            sys.exit("--build-cluster 需要配合 --systems 指定设备列表")
        systems = parse_systems(args.systems)
        log(f"集群打包：{len(systems)} 台设备，master 索引 {args.master_index}")
        build_cluster(systems, args.master_index, args.no_asm, args.compiler)
        return

    BUILD.mkdir(exist_ok=True)
    obj_dir = BUILD / "obj"
    obj_dir.mkdir(exist_ok=True)

    cxx = find_compiler(args.compiler)
    log(f"平台：{platform.system()} {platform.machine()}，编译器：{cxx}")

    if args.bundle:
        bundle_app(cxx, args.no_asm)
        return

    asm_obj = build_asm(cxx, args.no_asm, obj_dir)
    objs = build_objects(cxx, obj_dir, extra=[])

    if args.test:
        test_bin = build_tests(cxx, objs, asm_obj)
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)  # 避免宿主 LD_LIBRARY_PATH 污染
        run([str(test_bin)], env=env)

    if not args.no_ext:
        if not (EXT_OUT / "__init__.py").exists():
            sys.exit(f"未找到 Python 包目录：{EXT_OUT}")
        try:
            import numpy  # noqa: F401
        except ImportError:
            log("提示：未安装 numpy，训练核心需要它：python -m pip install numpy")
        build_extension(cxx, objs, asm_obj)

    log("完成")


if __name__ == "__main__":
    main()
