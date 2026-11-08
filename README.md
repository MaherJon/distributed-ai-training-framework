# MAHE · 分布式 AI 训练框架

把局域网里的一堆普通电脑（最低 2 核 CPU + 网络）组织成一个训练集群。
Master 只做参数服务与调度，Worker 用自己的算力分摊数据，适合宿舍/办公室的闲置机器。

- 设计文档：[DESIGN.md](DESIGN.md)
- 状态：v0.1，C++ 核心 + AVX2 汇编内核 + Python 参数服务器训练 + Kotlin UI 脚手架
- 协议：GPLv3

## 架构

```text
Kotlin UI 进程 ──(localhost TCP / JSON Lines)── Python 控制面（Master）
                                                   │ pybind11
                                              C++ 核心
                                    ┌──────────┼──────────┐
                                transport   discovery  scheduler
                                  TCP         UDP 发现   加权分片/心跳
                                                   │
                                   ── TCP 帧协议 ── Worker 进程（Python + C++）
```

训练策略：参数服务器 + **算力加权数据并行** + **本地多步聚合（FedAvg 式）**。
Worker 在本地分片训练 K 步后只上传参数差值 Δ，Master 按样本数加权合并，
通信量比"每步传梯度"低 1~2 个数量级。心跳超时节点会被回收，其分片重新派发。

## 目录

```text
core/                C++ 核心
  include/mahe/      协议、传输、发现、调度、内核头文件
  src/               protocol/transport/discovery/scheduler/kernel/node 实现
  asm/reduce_avx2.asm  AVX2 梯度归约与 SGD 更新（NASM）
  bindings/pybind.cpp  Python 扩展
  tests/test_core.cpp  C++ 单元测试（含端到端握手）
python/mahe/         Python 训练核心与演示
  model.py            NumPy MLP（CPU-only 后端）
  worker_node.py     Worker：本地多步训练、Δ 回传
  master_node.py     Master/参数服务器：加权聚合、评估、检查点
  ui_bridge.py       给 Kotlin UI 的 JSON Lines 控制信道
  demo_train.py      演示入口（master / worker / 单机 demo）
ui/                  Kotlin Compose Desktop 界面
build.py             跨平台构建脚本
```

## 快速开始（Linux / macOS）

```bash
# 依赖
python3 -m pip install numpy pybind11
# 汇编内核（x86-64 Linux/macOS 可选装；不装则自动回退标量实现）
sudo apt install nasm          # Ubuntu/Debian；macOS: brew install nasm

# 构建 + 跑 C++ 单测
python3 build.py --test

# 构建 Python 扩展
python3 build.py

# 单机演示：master + 2 个 worker 子进程
PYTHONPATH=python python3 -m mahe.demo_train --role demo --rounds 20
```

预期输出：

```text
INFO mahe.master: round=1  workers=2 eval_loss=... eval_acc=...
...
INFO mahe.master: round=20 workers=2 eval_loss=0.0460 eval_acc=0.9850
```

## 在 Windows 上构建

需要 **64 位 Python**（3.10+）与 x64 编译器，架构必须一致（x64 Python ↔ x64 编译器）。

C++ 编译器二选一：**MinGW-w64（推荐，配置最省事）** 或 **MSVC**。

### 方案 A：MinGW-w64 / MSYS2（推荐）

1. 安装 Python 3.10+（python.org，勾选 *Add python.exe to PATH*）。
2. 安装 MSYS2（<https://www.msys2.org>），然后在 **MSYS2 MINGW64** 终端里：

   ```bash
   pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-nasm
   ```

3. 把 MSYS2 的编译器目录加进 PATH（在 Windows 的 **cmd** 或 **PowerShell** 里操作，
   不要用 MSYS2 的终端，避免路径转换问题）：

   ```bat
   :: cmd
   set PATH=C:\msys64\mingw64\bin;%PATH%
   g++ --version
   nasm -version
   ```

   ```powershell
   # PowerShell
   $env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
   ```

4. 安装 Python 依赖并构建：

   ```bat
   python -m pip install numpy pybind11
   python build.py --test
   python build.py
   ```

### 方案 B：MSVC

1. 安装 **Build Tools for Visual Studio 2022**，勾选 *使用 C++ 的桌面开发*
   （含 MSVC v143 与 Windows SDK）。
2. 安装 NASM 并加入 PATH：

   ```powershell
   winget install nasm
   # 或 choco install nasm
   ```

3. 安装 64 位 Python 3.10+（勾选 *Add python.exe to PATH*），然后：

   ```bat
   python -m pip install numpy pybind11
   ```

4. **必须从 "x64 Native Tools Command Prompt for VS 2022" 启动**（或先执行
   `vcvars64.bat`），否则找不到 `cl.exe`：

   ```bat
   cd /d D:\path\to\MAHE-distributed-ai-training-framework
   python build.py --test
   python build.py
   ```

   如果 `cl` 已被误判或需要显式指定：

   ```bat
   python build.py --compiler cl
   ```

构建产物：`python\mahe\mahe_core.cp3XX-win_amd64.pyd`。

### Windows 注意事项

- **没有 NASM**：`python build.py` 会打印警告并自动回退标量内核（功能完整、速度略低），
  也可显式 `--no-asm`。装了 NASM 才能启用 AVX2 汇编路径。
- **防火墙**：Master 监听 TCP `47820` 与 UDP `47821`，首次运行请允许
  "*专用网络*" 访问，否则 Worker 无法入网。
- 若 `python build.py --test` 报链接错误，通常是 Python 位数与编译器不一致
  （32 位 Python + x64 编译器）。`python -c "import struct;print(struct.calcsize('P')*8)"`
  应输出 `64`。
- 本仓库的 `/tmp`、`apt`、`brew` 等命令仅适用于 Linux/macOS；Windows 请使用上面对应的
  `winget` / `choco` / `pacman`。

### Windows 常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `undefined reference to '__imp_WSAStartup'` 等 | MinGW 忽略 MSVC 的 `#pragma comment(lib)` | 已由 `build.py` 自动加 `-lws2_32`；请用仓库最新脚本重新构建 |
| `cannot find -lpython312` / 无法解析 PyInit | 找不到 Python 导入库 | 确认装的是 64 位 Python，且 `python -c "import sys;print(sys.base_prefix)"` 下存在 `libs\python312.lib` |
| `'cl' 不是内部或外部命令` | 没在 VS x64 开发终端里 | 打开 *x64 Native Tools Command Prompt for VS 2022*，或先执行 `vcvars64.bat` |
| 链接期 `file in wrong format` | 32 位 Python 配 x64 编译器（或反之） | 统一为 64 位 |
| Worker 一直连不上 Master | 防火墙拦截 / 不同网段广播 | 放行 TCP 47820 + UDP 47821，或给 Worker 加 `--master <IP>` |

### 运行演示（Windows）

```bat
set PYTHONPATH=%CD%\python
python -m mahe.demo_train --role demo --rounds 20
```

PowerShell：

```powershell
$env:PYTHONPATH="$PWD\python"
python -m mahe.demo_train --role demo --rounds 20
```

## 局域网多机训练

在 Master 机器：

```bash
PYTHONPATH=python python -m mahe.demo_train --role master --min-workers 2 --rounds 30
```

在每台 Worker 机器（同网段，UDP 广播可发现 Master）：

```bash
PYTHONPATH=python python -m mahe.demo_train --role worker
```

自动发现失败（跨网段 / 广播被拦）时直连：

```bash
PYTHONPATH=python python -m mahe.demo_train --role worker --master 192.168.1.10
```

说明：演示脚本内置集群令牌 `mahe-demo-token`；真实使用请在
[demo_train.py](python/mahe/demo_train.py) 或你自己的入口中修改，令牌不匹配的连接会被拒绝。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--role` | `demo` | `master` / `worker` / `demo` |
| `--rounds` | 20 | 通信轮数 |
| `--k-steps` | 20 | 每轮 Worker 本地训练步数 |
| `--batch` | 64 | 本地 batch |
| `--lr` | 0.5 | 学习率 |
| `--hidden` | 32 | 隐藏层宽度 |
| `--min-workers` | 1 | Master 等待的最少节点数 |
| `--ckpt` | `checkpoints/mahe_ckpt.npz` | 检查点路径 |

端口：TCP `47820`（训练/心跳），UDP `47821`（发现广播），TCP `47830`（UI 控制信道）。

## 构建脚本

```bash
python build.py                     # 核心 + Python 扩展
python build.py --test              # 构建并运行 C++ 单测
python build.py --no-asm            # 强制标量内核
python build.py --no-ext            # 只构建核心与单测
python build.py --compiler g++-13   # 指定编译器
python build.py --clean             # 清理产物
```

`build.py` 会自动探测：OS/架构、编译器（`CXX` 环境变量优先）、NASM、Python 头文件、
pybind11；缺依赖时给出安装提示，不会静默改动你的环境。

## 测试

- `python build.py --test`：内核与标量实现逐元素比对、帧编解码、加权分片、
  Master/Worker 握手、令牌拒绝、双向消息。
- 端到端：`--role demo` 会真实起 2 个 Worker 子进程，检查点落在 `checkpoints/`。

## UI（Kotlin Compose Desktop）

UI 是独立进程：训练照常在 Python 侧跑，UI 通过 `127.0.0.1:47830` 订阅事件，
断开或崩溃都不影响训练，重连后自动补发快照。

只需要 **JDK 17+**，Gradle 由仓库自带的 wrapper 下载：

Windows（PowerShell）：

```powershell
winget install EclipseAdoptium.Temurin.17.JDK   # 或已装 JDK 可跳过
cd D:\MAHE-distributed-ai-training-framework\ui
.\gradlew.bat run
```

Linux / macOS：

```bash
cd ui
./gradlew run
```

首次运行会下载 Gradle 8.5 与 Compose 依赖（需联网，约数百 MB），之后可离线运行。
仓库已在 [settings.gradle.kts](ui/settings.gradle.kts) / [build.gradle.kts](ui/build.gradle.kts)
预置阿里云 Maven 镜像（官方仓库作为兜底），国内网络更稳；不需要可删掉对应 `maven(...)` 行。

**先启动训练，UI 才有数据**：

```powershell
# 终端 1：训练（demo 或 master 均可，都会开启 47830）
$env:PYTHONPATH="D:\MAHE-distributed-ai-training-framework\python"
python -m mahe.demo_train --role demo --rounds 20
# 终端 2：UI
cd D:\MAHE-distributed-ai-training-framework\ui
.\gradlew.bat run
```

界面内容：启动自检序列 → 集群节点表（节点数/算力分/存活）→ loss/acc 双折线 →
事件日志。右上角 `◉ LINKED` 表示已连上控制面，`◌ OFFLINE` 表示未运行 Master，会自动重连。

可选：打包成安装包。

```powershell
cd ui
.\gradlew.bat packageReleaseMsi    # Windows 安装包（也可 packageReleaseDeb / Dmg）
.\gradlew.bat runDistributable     # 免安装可执行目录
```

## 当前边界

- 训练后端是 NumPy MLP（CPU），尚未接 PyTorch；
- 梯度压缩（int8）已在设计中预留，尚未实现；
- 多机验证限于本机多进程，真实局域网需按上面步骤自行组网；
- UI 目前是可用脚手架（启动序列 + 三块面板），交互（建集群/起停训练按钮）尚未接控制指令。

路线图见 [DESIGN.md](DESIGN.md) §12。
