# MAHE 分布式 AI 训练框架 · 设计文档（修订版）

> 本文档在原《设计草案》基础上修订。主要改动：明确训练并行策略、重新定位汇编的用途、
> 放弃"自研传输层协议"、补充异构调度/容错/安全设计、解耦 UI 与训练核心。

## 1. 设计动机

GPU 集群与租用显卡成本高昂，个人爱好者难以负担。MAHE 的目标：**以一台主机为核心，
在局域网内把多台普通设备（最低配置 a = 2 核 CPU + 网络连接）组织成训练集群**，
用数据并行方式分摊训练负载。

关键约束（决定技术选型）：

- 节点**异构**（算力差异大，可能有 2 核老机器，也可能有主力机）→ 不能平均分数据，
  必须按算力加权分片；
- 节点**可能纯 CPU** → 训练核心必须支持 CPU-only，模型规模可控，支持 micro-batch；
- 局域网带宽**不确定**（WiFi 设备可能只有几十 Mbps）→ 需要梯度压缩与本地多步
  （减少通信轮次）；
- 节点**可能掉线**（笔记本合盖、网络抖动）→ 心跳 + 任务重派 + 检查点是必需品，
  不是可选项。

## 2. 对原草案的主要修订

| 原草案 | 问题 | 修订 |
|---|---|---|
| 自研网络协议 + 汇编介入网络层 | 传输层瓶颈在 syscall/内核，自研协议相对 TCP 无收益，反而引入可靠性与安全问题 | 传输用 **TCP**，应用层定义**轻量二进制帧协议**（魔数+类型+长度+负载，~16B 头）；汇编不碰网络 |
| 汇编"暴力加速"性能拉胯的设备 | 方向正确但落点不清 | 汇编只用于**计算热点内核**：梯度归约（reduce-add）、权重更新、（可选）int8 量化。这些是可证明的热点，AVX2 相对标量有 4~8 倍收益。运行时 CPUID 探测，无 AVX2 自动回退 C 实现 |
| 未定义并行策略 | 分布式训练的核心缺失 | **参数服务器（PS）架构 + 算力加权数据并行**；同步模式采用 **FedAvg 式本地多步聚合**（降低通信频率），支持 stale 容忍的半异步 |
| UI 逻辑写在 Python 层 | Kotlin UI 与 Python 紧耦合、跨进程边界混乱 | UI 是**独立进程**：浏览器加载 Python 托管的静态 Web 终端，通过 localhost（HTTP + SSE + JSON 指令）与控制面通信；Python 层暴露 ClusterController API |
| 无容错设计 | 弱设备集群掉线是常态 | 心跳（默认 3s 周期 / 10s 判死）+ 任务重派 + 周期性检查点 |
| 无安全设计 | 局域网不等于可信网络（宿舍/公司网）| 集群令牌（cluster token）认证握手；发现广播不含敏感信息 |

## 3. 总体架构

```text
高层 ┌──────────────────────────────────────────────┐
     │  Web 终端（浏览器，eDEX 风格收敛版）            │
     └───────────────┬──────────────────────────────┘
                     │ 控制信道：localhost HTTP
                     │ 事件：SSE；指令：JSON POST
     ┌───────────────┴──────────────────────────────┐
     │  Python 控制面 + 训练核心（mahe 包）            │
     │  · WebUiServer：静态页面/事件流/指令分发        │
     │  · ClusterController：建集群/节点管理/任务编排   │
     │  · Trainer：模型定义、本地多步训练、梯度产出      │
     │  · ParameterServer：加权聚合、优化器、检查点     │
     │  · data / community：数据集描述符与社区目录      │
     │  · lan_scan / deployer：局域网扫描与子端部署     │
     │  · platform_dialogs / agent：原生对话框与助手    │
     │  · 通过 pybind11 扩展 mahe_core 调用 C++ 核心    │
     └───────────────┬──────────────────────────────┘
                     │ pybind11（同进程，零拷贝传 NumPy buffer）
     ┌───────────────┴──────────────────────────────┐
     │  C++ 核心（libmahe_core）                       │
     │  · transport：TCP 帧收发、连接池、背压           │
     │  · discovery：UDP 广播发现 + 入网握手            │
     │  · scheduler：加权分片、任务派发、心跳、重派      │
     │  · kernel：AVX2 汇编归约内核（CPUID 分发）       │
     └──────────────────────────────────────────────┘
低层        │  节点间：TCP（控制+梯度）  节点发现：UDP 广播
```

**进程拓扑**：Master 节点跑完整栈（浏览器 UI + Python 控制面 + C++ 核心 + PS）；
Worker 节点跑精简栈（Python worker + C++ 核心，无 UI）。Worker 只被动接任务。

**初始化握手**：Web 服务先启动，但 C++ 核心/调度器/训练模块保持未初始化；
前端播放完启动字幕后才 POST `ui_ready`，控制面收到信号再初始化训练链路。
初始化日志通过 SSE 输出到界面左侧信息区，格式 `[信息类型] 内容`。
这样避免了“界面还没准备好，后端已经广播/占用端口/开始训练”的竞态。

**干净启动与训练入口**：初始化完成后只到 `ready`，不拉起演示节点、不自动训练；
训练参数（数据集、轮数、步数、批大小、学习率、最少节点、模型保存位置）由
「训练」页提交 `start_training` 指令后生效，训练线程可重复启动与停止。
开发者自测用 `--test` / `-t`：初始化后自动拉起 2 个本机 Worker 并立即开跑。
`--no-ui`（服务器场景）没有可点击的界面，直接按命令行参数训练。

**节点移出**：`Scheduler::remove_node` 已暴露给 Python。Master 移出节点时先发
`BYE` 再摘除调度记录；Worker 收到 `BYE` 后 `CoreNode` 停止，进程随
`is_running()` 变为 false 而退出。这样分组不再是“只能加不能减”。

## 4. 应用层协议（帧格式）

传输层用 TCP，不重复造轮子。应用层帧：

```text
 0        4        6     7          11              15        N
┌──────────┬────────┬────┬──────────┬───────────────┬──────────┐
│ "MAHE"   │ ver=1  │type│  flags   │ payload_len   │ payload  │
│ 魔数4B   │  2B    │ 1B │   4B     │  4B (LE)      │ ≤64MB    │
└──────────┴────────┴────┴──────────┴───────────────┴──────────┘
头部 16B，全小端。flags bit0=压缩(zstd, 预留) bit1=分片续传 bit2=请求-响应配对
```

消息类型（v1）：

| type | 名称 | 方向 | 负载 |
|---|---|---|---|
| 0x01 | HELLO | W→M | token + 节点能力（核数/内存/AVX2/基准分） |
| 0x02 | HELLO_ACK | M→W | 节点 ID + 心跳周期 |
| 0x03 | HEARTBEAT | 双向 | 节点 ID + 当前进度 |
| 0x10 | TASK_ASSIGN | M→W | epoch、数据分片描述、超参、模型参数 |
| 0x11 | TASK_PROGRESS | W→M | task_id、完成比例、吞吐 |
| 0x12 | GRAD_PUSH | W→M | task_id、步数、梯度 blob（可 int8 量化） |
| 0x13 | PARAM_SYNC | M→W | 全局步数、新参数 blob |
| 0x14 | TASK_ACK / 0x15 TASK_FAIL | W→M | 任务接收/失败原因 |
| 0x20 | CHECKPOINT_NOTICE | M→W | 检查点 ID |
| 0x7F | BYE | 双向 | 优雅退出 |

张量 blob 编码：`[dtype:1B][ndim:1B][shape:ndim×8B][raw data]`，
Python 侧零拷贝映射 NumPy。

## 5. 训练策略（核心修订点）

**参数服务器架构**：Master 持有全局参数与优化器状态；Worker 只持有参数副本与本地数据分片。

**加权数据并行**：入网握手时 Worker 上报基准分（一次固定微基准测得），
调度器按 `shard_size ∝ score / Σscores` 分配数据，避免快节点等慢节点。

**本地多步聚合（FedAvg 式）**：每个通信轮次 Worker 在本地分片上训练 K 步
（K 可配，默认几十步），上传**参数差值（Δ）而非每步梯度**；Master 按分片样本数
加权平均 Δ 并应用。通信量下降 1~2 个数量级——这是弱局域网能跑起来的关键。

**梯度压缩（可选）**：Δ 上传前做 int8 逐通道量化（kernel 提供汇编量化例程），
弱网节点自动开启。

**同步围栏**：一轮内等待 `min(wait_all, ⌈0.8×N⌉)` 个节点回报即可聚合，
掉队节点下一轮用新参数重算（stale 丢弃），防止单点拖死。

**容错**：心跳超时 → 节点标记 dead，其分片回收到待派池；检查点每 R 轮落盘，
重启可续训。

## 6. 汇编加速内核（kernel 模块）

定位：**只做 SIMD 计算，不碰网络/系统逻辑**。

| 内核 | 用途 | 加速 |
|---|---|---|
| `mahe_reduce_add_f32(dst, src, n)` | 多节点 Δ 累加、梯度归约 | AVX2 8×f32/指令 |
| `mahe_sgd_update_f32(param, grad, lr, n)` | 参数原地更新（fused multiply-add） | 减少内存往返 |
| `mahe_quantize_int8`（后续） | 梯度压缩 | — |

实现方式：NASM 编写 AVX2 例程（x86-64 System V / Win64 双调用约定）；
C++ 侧 `cpuid` 运行时探测，函数指针分发到 AVX2 或标量 C 实现。
ARM（Mac）走 NEON intrinsics 回退路径。

## 7. C++ 核心模块

- **transport**：`TcpServer/TcpClient`，每连接一个读线程 + 写队列（写线程），
  帧重组、最大帧限制、背压（写队列水位）；
- **discovery**：UDP 广播 beacon（Master 周期性宣告 `cluster_name/port/token_hash`），
  Worker 监听后主动 TCP 入网；Master 校验 token；
- **scheduler**：任务队列、加权分片计算、派发、进度表、心跳看门狗、重派；
- **kernel**：如上；
- **pybind11 绑定**：向 Python 暴露 `CoreNode`（start_master/start_worker）、
  回调（on_task/on_progress）、`send_blob`，NumPy buffer 零拷贝。

## 8. Python 层

- `mahe.web_server.WebUiServer`：回环 HTTP 服务，负责静态页面、SSE 事件流、
  JSON 指令分发与状态快照；指令覆盖 `ui_ready`、扫描/部署、训练启停、
  节点移出、模型路径选择、目录刷新与助手对话；
- `mahe.master_node`：参数服务器、加权聚合（调用汇编内核）、评估与检查点；
  支持按任务切换数据集描述符、移出单个节点或清空节点；
- `mahe.data`：数据集注册表。每条描述符包含 `id/name/generator/n_features/
  n_classes/size/seed/params/recommended`，各节点本地确定性重建数据，任务负载
  里只带描述符；同模块提供 `estimate()` 计算总步数、每轮样本与等效 epoch；
- `mahe.community`：可配置的社区目录。读取 `--community` / `MAHE_COMMUNITY_URL`，
  拉取 `<base>/catalog.json` 并与内置目录合并；网络失败静默回退，未知生成器跳过；
- `mahe.worker_node`：本地多步训练与 Δ 回传，按任务下发的描述符构建数据集与模型；
- `mahe.lan_scan`：私有网段扫描、主机名解析、远程管理端口探测与可部署性筛选；
- `mahe.deployer`：生成 POSIX/PowerShell 部署脚本，支持 SSH 远程执行
  （浅克隆指定仓库/分支 → 安装依赖 → `build.py` → 后台启动 Worker）；
- `mahe.platform_dialogs`：跨平台原生「另存为」对话框，Windows 用 PowerShell +
  WinForms，macOS 用 osascript，Linux 用 zenity/kdialog；失败时返回空串由前端兜底；
- `mahe.agent`：DeepSeek 助手预留接口。统一的 `status()/ask()`，密钥只从环境变量
  读取；未配置时抛出 `AgentUnavailable`，不发起任何网络请求；
- `mahe.demo_train`：默认（干净）/ `--test` / worker 入口，负责 UI 就绪握手、
  训练线程生命周期与 `--no-ui` 模式。

## 9. Web UI（HTML/CSS/JS）

- 静态资源位于 `python/mahe/webui/`，由 Python 直接托管，无 Node/前端构建链；
- 设计上参考 eDEX 的全屏终端气质，但做了收敛：深色底、细线网格、等宽字体、
  单一青绿强调色，不使用大面积辉光与高频动效；
- 视图：总览（指标/loss 曲线/节点/事件）、训练（数据集/保存位置/预设/估算）、
  集群（节点移出与清空）、部署（设备表/部署日志）、助手（预留）、日志；
- 训练页把“给新手的必要决策”收敛成三步：选数据 → 选模型保存位置 → 确认训练量；
  `rounds / k_steps / batch` 由预设填充也可手改，前端与服务端各自做一次边界校验；
- 数据集与参数保存位置属于“非必要后端初始化”，一律在 `ui_ready` 之后再加载；
- 左上角菜单栏下方固定信息输出区，按 `[信息类型] 内容` 追加训练侧日志；
- 初始化流程：启动字幕 → POST `ui_ready` → 后端初始化训练模块 →
  初始化完成事件 `app_ready` → 弹出部署方式选择面板；
- SSE 断线自动重连，服务端先补发快照；浏览器关闭不影响训练。

## 10. 构建与分发（build.py）

Python 驱动（跨平台），职责：

1. 环境探测：OS、编译器（g++/clang++/MSVC）、NASM、Python 头文件；
2. 平台处理：
   - Windows：支持 MSVC（`cl`，需 VS x64 开发终端）与 MinGW-w64/MSYS2 两条路径，
     编译/链接参数按编译器分支生成，扩展显式链接 Python 导入库（Windows 无 lazy binding）；
     NASM 缺失时回退标量内核；
   - Linux：处理多 gcc 并存（`CC/CXX` 环境变量优先，或 `--compiler` 指定）；
   - macOS：clang，跳过 NASM（标量/NEON 路径）；
3. 构建：C++ 静态库 → 汇编内核 → pybind11 扩展；
4. 测试：`build.py --test` 跑 C++ 单元测试；
5. 依赖检查：pybind11/numpy 缺失时给出明确指引（不静默全局安装）；
6. 安装器：PyInstaller onedir 冻结引擎（含 Web UI 静态资源）→ onefile 安装器；
   安装器按用户级目录复制文件并创建桌面快捷方式，Windows 额外写开始菜单与卸载项。
   该流程不再依赖 JDK / jpackage / WiX / Inno Setup。

## 11. 开源协议

GPLv3。注意 pybind11（BSD）与 NASM（BSD-2）兼容。

## 12. 路线图

- v0.1（本次实现）：C++ 核心 + 汇编内核 + Python PS 训练 + Web 终端 +
  局域网扫描与子端部署流程 + 新手训练页/数据集目录/节点分组管理
- v0.2：真实局域网多机验证、检查点恢复、梯度压缩
- v0.3：DeepSeek 智能助手正式接通、PyTorch 后端、更完整的 Web 视图
- v0.4：TLS、Web 监控端

## 13. 智能助手接口（预留）

- 协议：前端 `POST /api/command {"cmd":"agent_ask","text":…,"history":[…]}`，
  服务端返回 `{"ok":true,"reply":…}` 或 `{"ok":false,"error":…}`；
- 配置：`DEEPSEEK_API_KEY` 为唯一启用条件，`MAHE_DEEPSEEK_MODEL` /
  `MAHE_DEEPSEEK_BASE` 可选覆盖；密钥不经前端、不写入日志；
- 行为：未配置密钥时只返回提示，不建立任何外部连接；已配置时走
  `/chat/completions`（非流式），超时 60s，失败回退为可读错误；
- 系统提示词把“轮数 / 每轮步数 / 等效 epoch / 模型路径”的口径固定下来，
  后续接入训练指标时可在同一条命令通道里扩展上下文。
