[English](README.md) | **简体中文**

# MAHE · 分布式 AI 训练框架

> 把宿舍、办公室里的闲置电脑，变成一台能训练的 AI 集群。

**MAHE** 用参数服务器架构把局域网内的普通电脑组织成训练集群：无需 GPU 预算，
2 核 CPU 起，Master 只做参数聚合与调度，Worker 用本地算力分摊数据。
核心用 C++17 + SIMD 汇编实现，Python 负责训练与编排，自带由 Python 托管的
Web 终端（HTML/CSS/JS）作为控制台。

*English: Turn idle LAN machines into a distributed AI training cluster — parameter server,
compute-weighted data parallelism, AVX2/NEON kernels, one-click packaging.*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-HTML%20%2F%20CSS%20%2F%20JS-58E0B6?logo=html5&logoColor=white)
![Status](https://img.shields.io/badge/status-v0.1-orange)

如果这个项目对你有帮助，欢迎点一个 Star。

![演示](./docs/demo.gif)

---

## 目录

- [为什么是 MAHE](#为什么是-mahe)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [架构](#架构)
- [工作原理](#工作原理)
- [SIMD 内核](#simd-内核)
- [跨平台构建](#跨平台构建)
- [局域网组网](#局域网组网)
- [Web 控制台](#web-控制台)
- [一键安装器](#一键安装器)
- [项目结构](#项目结构)
- [测试](#测试)
- [路线图](#路线图)
- [已知边界](#已知边界)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

用户手册：[docs/USER_GUIDE.md](docs/USER_GUIDE.md) · 设计文档：[DESIGN.md](DESIGN.md)

---

## 为什么是 MAHE

- **没有 GPU 也能开始**：训练瓶颈常常不是算力，而是“把一堆机器组织起来”的工程成本。
  MAHE 只要求 2 核 CPU + 局域网，办公室淘汰的台式机就是现成算力。
- **弱网友好**：Worker 本地训练 K 步后只回传参数差值 Δ，Master 按样本数加权合并，
  通信量比“每步传梯度”低 1~2 个数量级。
- **不挑平台、不挑编译器**：Windows（MinGW/MSVC）、Linux、macOS 同一套构建脚本；
  x86-64 走 AVX2，Apple Silicon / ARM64 走 NEON，缺汇编工具链自动回退标量实现。
- **一个安装器就能分发**：Python 引擎、C++ 内核与 Web 终端一起打进安装器，
  安装时自动创建桌面快捷方式；终端用户不需要装 Python、numpy 或 JDK。

## 核心特性

| 能力 | 说明 |
|---|---|
| 参数服务器训练 | Master 持有全局参数与优化器状态，Worker 只持副本与数据分片 |
| 算力加权数据并行 | 入网时上报微基准分，分片大小按 `score / Σscore` 分配，快节点不多等慢节点 |
| 本地多步聚合 | 每轮本地训练 K 步（FedAvg 式），只上传 Δ，显著压缩通信 |
| 自动发现与直连 | Master 周期性 UDP 广播；跨网段可 `--master <IP>` 直连 |
| 容错调度 | 心跳看门狗回收掉线节点，分片回到待派池；每轮聚合前等待 `--min-workers` |
| SIMD 内核 | AVX2（x86-64）/ NEON（aarch64）实现梯度归约与 SGD 更新，运行时自动分发 |
| 检查点 | 训练结束/中断落盘 `.npz`，可续训 |
| 可观测界面 | 浏览器访问 `127.0.0.1:47830`；启动字幕完成前不初始化训练；日志按 `[类型] 内容` 输出 |
| 局域网扫描部署 | 发现在线设备并按远程管理端口筛选，确认后从 GitHub 拉取源码在子端编译 |
| 一体化安装器 | `build.py --bundle` 生成单文件安装器，自动创建桌面快捷方式 |
| 集群打包 | `--build-cluster` 按设备数量与系统生成 master/worker 部署包 |

## 快速开始

三行命令跑起单机端到端演示（1 个 Master + 2 个 Worker 子进程）：

```bash
python3 -m pip install numpy pybind11

python3 build.py --test        # 构建并运行 C++ 单测（内核/协议/调度/握手）
python3 build.py               # 构建 Python 扩展

PYTHONPATH=python python3 -m mahe.demo_train --role demo --rounds 20
```

演示会自动在浏览器打开 `http://127.0.0.1:47830/`。启动字幕播放完成后，
前端才向 Python 控制面发送 `ui_ready`，随后训练模块开始初始化；初始化日志
在左侧信息区按 `[INIT] 内容` 输出。`--no-browser` 可关闭自动打开浏览器，
`--no-ui` 用于无界面的服务器场景。

终端实录：

```text
INFO mahe.master: round=1  workers=2 eval_loss=0.2816 eval_acc=0.9080
INFO mahe.master: round=2  workers=2 eval_loss=0.2356 eval_acc=0.9140
...
INFO mahe.master: 检查点已保存：checkpoints/mahe_ckpt.npz (round=20)
```

Windows 用 `set PYTHONPATH=%CD%\python` 代替 `PYTHONPATH=python`，其余相同。
完整依赖、编译器与排错说明见 [用户手册](docs/USER_GUIDE.md)。

## 架构

```text
┌──────────────────────────────┐
│ Web 终端（浏览器，独立进程）  │  崩溃/关闭不影响训练
└──────────────┬───────────────┘
               │ localhost:47830 · HTTP + SSE + JSON
┌──────────────▼───────────────┐
│ Python 控制面（Master）       │  参数聚合 / 评估 / 检查点 / UI 事件
└──────────────┬───────────────┘
               │ pybind11
┌──────────────▼───────────────┐
│ C++17 核心                    │
│  transport  discovery  scheduler  kernel │
│  TCP 帧协议  UDP beacon  加权分片   SIMD │
└──────────────┬───────────────┘
               │ TCP 47820 · 16 字节帧头（魔数 MAHE / v1）
     ┌─────────┴─────────┐
┌────▼─────┐        ┌────▼─────┐
│ Worker A │  ...   │ Worker N │   Python + C++ 扩展
└──────────┘        └────────────┘
```

## 工作原理

1. **入网**：Worker 监听 UDP 广播（47821）发现 Master，携带算力分与令牌发起 TCP 入网；
   Master 校验令牌并分配 `node_id`。
2. **分片**：调度器按算力分把本轮数据切成加权分片，派发给各 Worker。
3. **本地训练**：Worker 在分片上训练 K 步（`--k-steps`），只计算参数差值 Δ。
4. **聚合**：Master 收到 Δ 后按分片样本数加权平均，用 SIMD 内核更新全局参数。
5. **容错**：心跳超时节点被标记 dead，分片回收到待派池，下一轮重新分配。

通信量对比（同一模型、同一轮）：

| 策略 | 每轮上传 | 相对通信量 |
|---|---|---|
| 每步传梯度 | K × 参数量 | 基准 |
| 本地多步 + Δ（MAHE） | 1 × 参数量 | 约 1/K |

## SIMD 内核

| 内核 | 平台 | 实现 |
|---|---|---|
| `mahe_reduce_add_f32` | x86-64 | NASM / AVX2，8×f32 每指令 |
| `mahe_reduce_add_f32` | aarch64 | GAS / NEON，16×f32 每轮（4×q 寄存器） |
| `mahe_sgd_update_f32` | x86-64 | AVX2 FMA |
| `mahe_sgd_update_f32` | aarch64 | `dup + fneg + fmla`（`param -= lr*grad`） |
| 全部 | 任意 | 标量 C++ 回退，结果逐元素一致 |

运行时自动分发，`python -c "import mahe; print(mahe.mahe_core.kernel_impl())"`
输出 `avx2` / `neon` / `scalar`。`--no-asm` 可强制标量路径。

## 跨平台构建

| 平台 | 编译器 | 汇编 | 产物 |
|---|---|---|---|
| Windows x64 | MinGW-w64（推荐）/ MSVC | NASM `-f win64` | `mahe_core.cp3XX-win_amd64.pyd` |
| Linux x86-64 | g++ / clang++ | NASM `-f elf64` | `mahe_core.cpython-3XX-x86_64-linux-gnu.so` |
| Linux aarch64 | g++ / clang++ | NEON（`.S` 直接汇编） | `mahe_core.cpython-3XX-aarch64-linux-gnu.so` |
| macOS Apple Silicon | clang++ | NEON（`.S` 直接汇编） | `mahe_core.cpython-3XX-darwin.so` |
| macOS Intel | clang++ | NASM | 同上 |

```bash
python build.py                     # 核心 + Python 扩展
python build.py --test              # 构建并运行 C++ 单测
python build.py --bundle            # 一体化安装器（引擎 + Web 终端）
python build.py --no-asm            # 强制标量内核
python build.py --compiler g++-13   # 指定编译器
python build.py --clean             # 清理产物
```

Windows 需要 64 位 Python 与 64 位编译器保持一致；MinGW 安装见
[用户手册 · Windows](docs/USER_GUIDE.md#windows)。

## 局域网组网

Master（监听 TCP 47820 / UDP 47821）：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role master --min-workers 2 --rounds 30
```

Worker（同网段，UDP 广播自动发现）：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker
```

跨网段或广播被拦时直连：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker --master 192.168.1.10
```

> 演示令牌 `mahe-demo-token` 硬编码在 [demo_train.py](python/mahe/demo_train.py)，
> 真实部署务必替换；令牌不匹配的连接会被拒绝。

按设备数量与系统批量生成部署包：

```bash
python3 build.py --build-cluster \
  --systems "linux-x86_64,linux-aarch64,windows-x86_64,mac-arm64" \
  --master-index 0
```

输出在 `build/cluster/device_XX_<role>_<os>-<arch>/`，每个包含源码、构建脚本、
启动脚本与 `device_info.json`；与打包机同系统的设备会预编译扩展，其余首次运行时
在目标机自动构建。

## Web 控制台

Web 终端由 Python 控制面托管在 `127.0.0.1:47830`，训练启动后自动打开浏览器
（HTML/CSS/JS，无外部依赖）。

```bash
python3 -m mahe.demo_train --role demo      # 自动打开浏览器
python3 -m mahe.demo_train --role master    # 局域网 Master + 控制台
python3 -m mahe.demo_train --role master --no-browser --no-ui   # 无界面
```

初始化流程：

1. 浏览器播放启动字幕；
2. 播放完成后 POST `ui_ready`，后端这时才初始化 C++ 核心、调度器与训练模块；
3. 初始化信息在左侧信息区按 `[INIT] …`、`[WARN] …`、`[ERROR] …` 输出；
4. 初始化完成后弹出子端部署面板，可选择扫描局域网自动部署或手动脚本。

界面是收敛版的 eDEX 风格：深色底、细线网格、等宽字体与单一青绿强调色，
包含启动输出、节点表、loss/acc 曲线、事件日志与部署视图。关闭浏览器不影响训练，
重新打开会收到完整快照。

### 子端部署

- **扫描**：Master 探测本机私有 /24（也可手动指定 CIDR），解析主机名并检测
  SSH/SMB/RDP/WinRM/AFP/VNC 端口；列表显示设备名称、类型与是否可自动部署。
- **自动部署**：对开放 SSH 的设备，Master 通过 SSH 执行生成的引导脚本：
  浅克隆指定 GitHub 仓库/分支 → 安装 numpy/pybind11 → 运行 `build.py` →
  后台启动 Worker。
- **手动部署**：下载 POSIX 或 Windows 脚本，在目标机执行。自动部署使用 SSH
  密钥（`BatchMode`），不保存任何密码。

## 一键安装器

把冻结后的 Python 引擎与 Web 终端打成单文件安装器：

```bash
python -m pip install pyinstaller
python build.py --bundle
```

| 平台 | 产物 |
|---|---|
| Windows | `build/installer/MAHE-Setup.exe` |
| macOS | `build/installer/MAHE-Setup` |
| Linux | `build/installer/MAHE-Setup` |

安装器按用户级安装（无需管理员权限），并且**一定会创建桌面快捷方式**；
Windows 还会创建开始菜单项与卸载注册表项。双击快捷方式即启动 `mahe-engine`
（默认 `--role demo`，可用 `MAHE_ENGINE_ROLE=master|worker` 覆盖）并打开 Web 终端。

PyInstaller 只产出本机平台产物，三平台需分别在对应系统上构建；体积约 100–130 MB。
macOS 对外分发还需签名与公证。

## 项目结构

```text
core/
  include/mahe/         协议、传输、发现、调度、内核头文件
  src/                  protocol / transport / discovery / scheduler / kernel / node
  asm/reduce_avx2.asm   AVX2 梯度归约与 SGD 更新（NASM，x86-64）
  asm/reduce_neon.S     NEON 梯度归约与 SGD 更新（GAS，aarch64）
  bindings/pybind.cpp   Python 扩展（NumPy 零拷贝）
  tests/test_core.cpp   C++ 单元测试（含端到端握手）
python/mahe/
  model.py              NumPy MLP（CPU-only 参考后端）
  worker_node.py        本地多步训练、Δ 回传
  master_node.py        参数服务器：加权聚合、评估、检查点
  web_server.py         HTTP + SSE + 指令通道
  webui/                静态 Web 终端（HTML/CSS/JS）
  lan_scan.py           局域网设备发现与可部署性筛选
  deployer.py           GitHub 源码拉取与远程编译部署脚本
  demo_train.py         demo / master / worker 入口
packaging/
  engine_entry.py       mahe-engine 的 PyInstaller 入口
  installer_entry.py    安装器：复制文件 + 创建桌面快捷方式
docs/USER_GUIDE.md      用户手册
build.py                跨平台构建 / 测试 / 组网打包 / 安装器
```

## 测试

```bash
python build.py --test
```

覆盖：

- SIMD 内核与标量实现逐元素比对（容差 1e-6）
- 帧编解码：长度、魔数、头部解析
- 加权分片 3:1 → 750/250
- 任务派发、完成、节点移除
- Master/Worker 握手、令牌拒绝、双向消息、优雅停止
- Web 终端：静态资源、`ui_ready` 握手与 `/api/status` 快照（冒烟）

端到端：`--role demo` 真实启动 2 个 Worker 子进程并落盘检查点。

## 路线图

- **v0.1（当前）**：C++ 核心 + AVX2/NEON 内核 + Python 参数服务器训练 + 本地演示 +
  Web 终端 + 局域网子端部署流程
- **v0.2**：真实局域网多机验证、检查点恢复、int8 梯度压缩
- **v0.3**：PyTorch 后端、更完整的 Web 视图与控制指令
- **v0.4**：TLS、Web 监控端

## 已知边界

- 训练后端是 NumPy MLP（CPU），尚未接 PyTorch；
- int8 梯度压缩在设计文档中预留，尚未实现；
- 多机验证以本机多进程为主，真实局域网需按手册组网；
- 自动子端部署依赖 SSH 密钥认证；仅开放 WinRM 的 Windows 主机需使用手动脚本。

## 参与贡献

欢迎 Issue 与 PR。提交前请：

1. 跑通 `python build.py --test`；
2. 保持 C++ 与 Python 两侧接口一致，新增内核需同时提供标量回退；
3. 汇编改动请在目标架构上实测，并在 PR 描述中写明 `kernel_impl()` 输出。

如果这个项目对你有帮助，点一个 Star 就是最好的支持。

## 许可证

[GPL-3.0](LICENSE) · pybind11（BSD）与 NASM（BSD-2）兼容。
