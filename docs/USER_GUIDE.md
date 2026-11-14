# MAHE 用户手册

本手册面向第一次使用 MAHE 的用户，覆盖安装、构建、单机演示、局域网组网、
控制台界面、安装包分发与排错。设计细节见 [DESIGN.md](../DESIGN.md)。

---

## 1. 适用场景

MAHE 适合“有机器、没 GPU 集群”的场景：

- 宿舍/办公室的闲置台式机、笔记本，同网段、能互相访问；
- 想验证分布式训练（参数服务器、数据并行、通信压缩）的教学与实验；
- 需要一个可以打包分发的桌面端训练演示程序。

当前版本（v0.1）的训练后端是 **NumPy 实现的 MLP**，内置三套确定性生成的 2 维
合成分类集（三簇螺旋 / 三类高斯 / 四类同心环），可在「训练」页切换。它不是生产级
大模型训练器，PyTorch 后端在 [v0.3 路线图](../DESIGN.md#12-路线图)。

## 2. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11 x64、Linux x64/aarch64、macOS（Intel / Apple Silicon） |
| Python | 3.10+，**必须 64 位**，且与编译器位数一致 |
| C++ 编译器 | Windows: MinGW-w64（推荐）或 MSVC v143；Linux: g++/clang++；macOS: clang++（Xcode CLT） |
| 汇编工具 | Windows/Linux x86-64: NASM；ARM64/macOS 用编译器直接汇编 `.S`，无需 NASM |
| Python 包 | `numpy>=1.21`、`pybind11>=2.10` |
| 浏览器 | Chrome / Edge / Firefox / Safari 等现代浏览器（Web 控制台） |
| 打包额外依赖 | `pyinstaller`（仅 `--bundle`） |

没有 NASM 不影响功能：`build.py` 会自动回退标量内核，只是慢一些。

## 3. 安装依赖

### Linux（Debian/Ubuntu）

```bash
sudo apt update
sudo apt install -y build-essential nasm python3-dev python3-pip
python3 -m pip install numpy pybind11
```

Fedora：

```bash
sudo dnf install -y gcc-c++ nasm python3-devel
python3 -m pip install numpy pybind11
```

### macOS

```bash
xcode-select --install          # 提供 clang++
brew install nasm               # Intel Mac 需要；Apple Silicon 可跳过
python3 -m pip install numpy pybind11
```

### Windows

Windows 需要 64 位 Python（python.org 安装时勾选 *Add python.exe to PATH*）。

**方案 A：MinGW-w64 / MSYS2（推荐）**

1. 安装 [MSYS2](https://www.msys2.org)，在 **MSYS2 MINGW64** 终端执行：

   ```bash
   pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-nasm
   ```

2. 在 **cmd** 或 **PowerShell**（不要用 MSYS2 终端，避免路径转换）里加入 PATH：

   ```bat
   set PATH=C:\msys64\mingw64\bin;%PATH%
   g++ --version
   nasm -version
   ```

   PowerShell：

   ```powershell
   $env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
   ```

3. 安装 Python 依赖：

   ```bat
   python -m pip install numpy pybind11
   ```

**方案 B：MSVC**

1. 安装 *Build Tools for Visual Studio 2022*，勾选“使用 C++ 的桌面开发”。
2. `winget install nasm`（或 `choco install nasm`）。
3. 必须从 **x64 Native Tools Command Prompt for VS 2022** 启动，或先执行 `vcvars64.bat`。

## 4. 构建与自检

```bash
python build.py --test    # 构建核心 + 单测并运行
python build.py           # 构建核心 + Python 扩展
```

检查内核分发是否生效：

```bash
# Linux / macOS
PYTHONPATH=python python -c "import mahe; print(mahe.mahe_core.kernel_impl())"
```

```bat
:: Windows cmd
set PYTHONPATH=%CD%\python
python -c "import mahe; print(mahe.mahe_core.kernel_impl())"
```

预期输出 `avx2`（x86-64）、`neon`（aarch64/Apple Silicon）或 `scalar`（回退）。

构建产物：

| 平台 | 扩展文件 |
|---|---|
| Windows | `python\mahe\mahe_core.cp3XX-win_amd64.pyd` |
| Linux | `python/mahe/mahe_core.cpython-3XX-x86_64-linux-gnu.so` |
| macOS | `python/mahe/mahe_core.cpython-3XX-darwin.so` |

`python build.py --clean` 可清理 `build/`、扩展与打包中间产物。

## 5. 五分钟单机演示

默认启动是**干净环境**：只跑 Master 与 Web 控制台，不会自动拉起节点，也不会
自动训练。打开控制台后：

1. 进入「训练」页；
2. 选择一个数据集（内置三个可选）；
3. 确认或修改模型保存位置；
4. 点「快速体验」或「标准」，再点「开始训练」。

单机想立刻跑通链路，可以直接用开发者自测模式（自动拉起 2 个 Worker 并开跑）：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --test --rounds 20
```

Windows：

```bat
set PYTHONPATH=%CD%\python
python -m mahe.demo_train --test --rounds 20
```

训练结束后检查点写到你在界面里指定的路径（默认 `checkpoints/mahe_model.npz`）。
每轮日志包含 `workers`、`eval_loss`、`eval_acc`。

命令会同时启动 Web 控制台并自动打开 `http://127.0.0.1:47830/`。界面先播放启动
字幕，播放完成后才通知后端初始化训练模块；初始化信息在左上角菜单栏下方的信息区
按 `[INIT] 内容` 输出。`--no-browser` 可关闭自动打开，`--no-ui` 用于服务器场景
（无界面时按命令行参数直接训练）。

## 6. 局域网组网

拓扑：一台 Master + N 台 Worker，同一局域网；Master 监听 TCP 47820 与 UDP 47821。

### 6.1 启动 Master

```bash
PYTHONPATH=python python3 -m mahe.demo_train --min-workers 2
```

`--min-workers` 是 Master 每轮聚合前等待的最少 Worker 数，也是「训练」页里
「最少节点」的默认值；默认 1，单机测试可用。

Master 启动后是干净环境，**不会自动训练**：等 Worker 入网后，在控制台「训练」页
点「开始训练」。若先点开始而节点不够，信息流会提示“等待 worker 入网”，最多等
`--wait-timeout` 秒（默认 300），期间可随时点“停止训练”。

### 6.2 启动 Worker

同网段（能收到 UDP 广播）时无需指定地址：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker --name worker-a
```

跨网段、广播被交换机/防火墙拦截时直连 Master：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker --master 192.168.1.10
```

### 6.3 令牌

Master 与 Worker 通过 `--token` 指定，默认 `mahe-demo-token`
（[demo_train.py](../python/mahe/demo_train.py)）。真实部署务必替换为随机长字符串，
并保证 Master 与所有 Worker 一致；令牌不匹配的连接会被拒绝。

```bash
# Master
PYTHONPATH=python python3 -m mahe.demo_train --role master --token my-secret-token

# Worker
PYTHONPATH=python python3 -m mahe.demo_train --role worker --master 192.168.1.10 \
  --token my-secret-token
```

### 6.4 防火墙

| 端口 | 协议 | 用途 | 需要放行 |
|---|---|---|---|
| 47820 | TCP | 训练/心跳/入网 | Master 入站 |
| 47821 | UDP | 发现广播 | Master 出站、Worker 入站 |
| 47830 | TCP | Web 控制台与事件流 | 仅本机（绑定 127.0.0.1） |

Windows 首次运行会弹防火墙提示，请勾选“专用网络”。

### 6.5 通过 Web 控制台部署子端

Master 启动并完成初始化后，界面会弹出部署方式选择；也可随时点击“部署”页签。

1. **扫描**：点击“扫描”，Master 会探测本机私有 /24（可手动填写 CIDR），
   并发 ICMP + 端口探测，列出设备名称、系统类型、开放端口与是否可自动部署；
2. **筛选**：只有开放 SSH（22）的设备支持自动部署；仅开放 WinRM 的 Windows
   设备会标记为“需手动”；
3. **确认**：勾选目标设备，填写 SSH 用户名后点击“部署选中设备”；
4. **执行**：Master 通过 SSH 在目标机执行引导脚本：
   浅克隆指定 GitHub 仓库/分支 → 安装 numpy/pybind11 → `python build.py` →
   后台启动 Worker；
5. **手动部署**：点击“手动部署”下载 POSIX / Windows 脚本，在目标机执行。

自动部署要求 Master 已配置到目标机的 SSH 免密登录（`ssh-copy-id`）。
部署脚本与目标机配置可在命令行用 `--repo`、`--branch`、`--token` 指定。

### 6.6 批量生成部署包

在任意一台机器上按设备清单生成部署包：

```bash
python3 build.py --build-cluster \
  --systems "linux-x86_64,linux-aarch64,windows-x86_64,mac-arm64" \
  --master-index 0
```

`--master-index` 指定第几台设备是 Master（从 0 开始）。输出目录：

```text
build/cluster/device_00_master_linux-x86_64/
build/cluster/device_01_worker_linux-aarch64/
...
```

每个包含 `core/`、`mahe/`、`build.py`、`requirements.txt`、`run_<role>.sh|bat`、
`build_here.sh|bat` 与 `device_info.json`。与打包机同系统的设备会预编译扩展；
其余包内有 `BUILD_ON_TARGET.txt`，首次运行启动脚本时自动在目标机构建。

## 7. Web 控制台

控制台是浏览器中的独立页面，由 Master 的 Python 控制面托管在
`http://127.0.0.1:47830/`，启动后会自动打开。关闭或刷新浏览器都不影响训练，
重新打开会先收到完整状态快照。

| 角色 | 是否启动 Web 控制台 |
|---|---|
| 默认（Master） | 是（默认打开浏览器） |
| `--test` | 是（额外拉起 2 个本地 Worker 并自动训练） |
| `worker` | 否 |

### 7.1 初始化流程

1. 页面先播放启动字幕（终端自检序列）；
2. 字幕播放完毕，前端 POST `ui_ready` 给 Python 控制面；
3. 控制面收到信号后才初始化 C++ 核心、调度器与训练模块；
4. 随后加载数据集目录与助手状态，再弹出“子端部署”选择面板
   （扫描局域网 / 手动部署 / 稍后）。

初始化期间的所有信息都输出到左上角菜单栏下方的信息区，格式为
`[信息类型] 内容`，例如：

```text
[UI] 控制终端已启动：http://127.0.0.1:47830/
[INIT] 传输层就绪：TCP 47820 / UDP 47821
[INIT] 计算内核：avx2
[INIT] 默认数据集：三簇螺旋
[INIT] 环境已就绪：在「训练」页选择数据集与保存位置后开始
[INFO] round=1 workers=2 eval_loss=0.2816 eval_acc=0.9080
```

### 7.2 界面区域

- **总览**：轮次、loss、准确率、在线节点数；loss/acc 曲线；节点表；事件日志；
- **训练**：数据集、模型保存位置、训练量预设与参数、实时估算、开始/停止；
- **集群**：完整节点列表、在线数量、单节点「移出」与「清空节点」；
- **部署**：局域网设备扫描、设备筛选、自动/手动部署与部署日志；
- **助手**：DeepSeek 对话入口（未配置密钥时显示待启用）；
- **日志**：最近 200 条事件；
- 右上角 `◉ LINKED` / `◌ OFFLINE` 显示事件流连接状态，断线自动重连。

### 7.3 训练页怎么用

1. **数据集**：下拉框显示名称、类别数与样本量；下方是一句话说明与推荐参数。
   社区目录配置后会自动合并进列表（见 7.4）。
2. **模型保存位置**：点「浏览…」由引擎在**主机**弹出系统原生另存为对话框。
   浏览器无法直接拿到本地路径，所以这一步走 Python 后端；也可以手动输入。
   目录不存在会自动创建，后缀为空时自动补 `.npz`。
3. **训练量**：
   - `快速体验`：10 轮 × 10 步，用来确认链路；
   - `标准`：20 轮 × 20 步，默认；
   - `充分`：40 轮 × 30 步、batch 128，适合四类数据。
   估算行会实时显示 `总步数 · 每轮样本 · 等效 epoch`，并给出训练量是否偏少/偏大。
4. 点「开始训练」。没有足够节点时会先等待；训练中按钮变为「停止训练」。

> 口径：轮数 = 参数聚合次数；每轮步数 = 每个节点本地跑的 mini-batch 数；
> 等效 epoch = 全局处理样本 ÷ 数据集大小。1~50 属正常，`<1` 基本学不到东西，
> `>200` 在小数据集上容易过拟合。

### 7.4 数据集与社区目录

内置数据集由描述符参数在各节点本地确定性生成，不传输数据文件：

| id | 名称 | 类别 | 样本 | 用途 |
|---|---|---|---|---|
| `builtin:spiral3` | 三簇螺旋 | 3 | 6000 | 入门，非线性可分 |
| `builtin:gauss3` | 高斯团（三类） | 3 | 6000 | 最简，验证集群与链路 |
| `builtin:rings4` | 同心环（四类） | 4 | 8000 | 稍难，观察更多轮次 |

配置社区目录（可选）：

```bash
PYTHONPATH=python python3 -m mahe.demo_train --community https://example.com/mahe
# 或
export MAHE_COMMUNITY_URL=https://example.com/mahe
```

引擎会请求 `<base>/catalog.json`（也接受完整 `.json` 地址），格式为数组或
`{"datasets": [...]}`；条目字段与内置描述符一致。拉取失败会静默回退内置目录，
未知生成器条目会被跳过。目录里点「刷新」可立即重试。

### 7.5 节点分组管理

- 每个节点右侧的 `✕`：确认后把该节点移出分组。Master 先发 `BYE`，再摘除调度
  记录；用当前版本构建的 Worker 收到 `BYE` 会自行退出。
- 「清空节点」：一次性移出全部节点。
- 移出只影响当前分组/调度，不会卸载子端程序；需要时重新部署或重启 Worker 入网。

### 7.6 智能助手

「助手」页用于后续接入 DeepSeek：未配置密钥时输入框禁用并显示原因，不会发起
任何外部请求。启用方式：

```bash
export DEEPSEEK_API_KEY=sk-...
PYTHONPATH=python python3 -m mahe.demo_train
```

可选 `MAHE_DEEPSEEK_MODEL`（默认 `deepseek-chat`）与 `MAHE_DEEPSEEK_BASE`
（默认 `https://api.deepseek.com`）。密钥只在引擎进程内读取，不会下发到前端。

### 7.7 相关参数

| 参数 | 说明 |
|---|---|
| `--ui-port` | Web 控制台端口，默认 47830，仅绑定 127.0.0.1 |
| `--no-browser` | 不自动打开浏览器 |
| `--no-ui` | 不启动 Web 控制台（服务器/无桌面环境） |
| `--ui-timeout` | 等待界面就绪的秒数，超时后按无界面模式继续 |
| `--wait-timeout` | 开始训练后等待节点入网的最长秒数，默认 300 |

## 8. 安装器

### 8.1 使用安装器（终端用户）

在对应平台运行 `MAHE-Setup`（Windows 为 `MAHE-Setup.exe`）。安装器无需管理员
权限，会：

- 复制程序到用户目录（Windows：`%LOCALAPPDATA%\Programs\MAHE`；
  Linux：`~/.local/share/mahe`；macOS：`~/Applications/MAHE`）；
- **自动创建桌面快捷方式**；
- Windows 额外创建开始菜单项与卸载注册表项。

双击桌面快捷方式即启动 `mahe-engine` 并打开 Web 控制台。默认是**干净启动**
（只有 Master，不自动训练）；要作为子端运行可用环境变量切换：

```powershell
$env:MAHE_ENGINE_ROLE="worker"
```

日志：`~/.mahe/engine.log`；检查点：`~/.mahe/checkpoints/`。
卸载：运行安装目录下的 `uninstall.ps1` / `uninstall.sh`。

### 8.2 构建安装器（开发者）

前置：

```bash
python -m pip install pyinstaller
```

在目标平台上执行（PyInstaller 只产出本机平台产物）：

```bash
python build.py --bundle
```

| 平台 | 产物路径 |
|---|---|
| Windows | `build/installer/MAHE-Setup.exe` |
| macOS | `build/installer/MAHE-Setup` |
| Linux | `build/installer/MAHE-Setup` |

流程：编译扩展 → PyInstaller 冻结 `mahe-engine`（含 Web UI 静态资源）→
组装安装载荷 → PyInstaller onefile 生成安装器。体积约 100–130 MB。
整个流程不再需要 JDK、jpackage、WiX 或 Inno Setup。

## 9. 命令行参考

`mahe.demo_train` 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--role` | `master` | `master` / `worker`；`demo` 等同 `--test` |
| `-t`, `--test` | 关 | 开发者自测：拉起 2 个本地 Worker 并自动开始训练 |
| `--master` | 空 | Worker 直连地址；空则 UDP 自动发现 |
| `--master-port` | 47820 | Worker 连接 Master 的端口 |
| `--name` | 空 | 节点名，默认 `worker-<pid>` |
| `--cluster-name` | `mahe-demo` | 集群名（UDP beacon 展示） |
| `--token` | `mahe-demo-token` | 集群令牌，Master/Worker 必须一致 |
| `--tcp-port` | 47820 | Master TCP 端口 |
| `--udp-port` | 47821 | 发现广播端口 |
| `--ui-port` | 47830 | Web 控制台端口（仅回环） |
| `--no-ui` | 关 | 不启动 Web 控制台 |
| `--no-browser` | 关 | 不自动打开浏览器 |
| `--ui-timeout` | 30 | 等待界面就绪的秒数 |
| `--advertise-host` | 自动 | 告知子端的 Master 地址 |
| `--subnet` | 自动 | 扫描网段（CIDR），默认本机 /24 |
| `--repo` | 内置仓库 | 子端部署源码仓库 |
| `--branch` | `main` | 子端部署分支 |
| `--community` | 空 | 社区数据集目录地址（也可用 `MAHE_COMMUNITY_URL`） |
| `--dataset` | `builtin:spiral3` | 默认数据集 id |
| `--rounds` | 20 | 默认通信轮数（界面里可改） |
| `--k-steps` | 20 | 默认每轮本地训练步数 |
| `--batch` | 64 | 默认本地 batch 大小 |
| `--lr` | 0.5 | 默认学习率 |
| `--hidden` | 32 | 隐藏层宽度 |
| `--min-workers` | 1 | 每轮等待的最少 Worker 数 |
| `--wait-timeout` | 300 | 开始训练后等待节点入网的秒数 |
| `--ckpt` | `checkpoints/mahe_model.npz` | 默认模型保存路径（冻结版在 `~/.mahe/checkpoints/`） |

`build.py` 参数：

| 参数 | 说明 |
|---|---|
| `--test` | 构建并运行 C++ 单测 |
| `--bundle` | 生成当前平台一体化安装器（含桌面快捷方式） |
| `--build-cluster` + `--systems` + `--master-index` | 生成集群部署包 |
| `--no-asm` | 强制标量内核 |
| `--no-ext` | 只构建核心与单测 |
| `--compiler` | 指定编译器（如 `g++-13`、`cl`） |
| `--clean` | 清理产物（含旧版 Kotlin UI 残留） |

## 10. 调优建议

- **先看训练量**：训练页的估算行会给出等效 epoch；`<1` 说明还没开始收敛，
  `>200` 在小数据集上容易过拟合，优先调 `rounds` 与 `k_steps`。
- **通信太慢**：提高每轮步数（本地多算，减少轮次通信），或增加 batch。
- **收敛不稳**：调低学习率，或提高「最少节点」保证每轮参与者数量。
- **节点算力差异大**：MAHE 已按算力分加权分片；若差异仍明显，可让弱机只作
  Worker、强机作 Master，并适当降低总轮数观察曲线。
- **模型更大**：提高 `--hidden`，注意 NumPy 后端是 CPU 计算，收益取决于单机算力。
- **验证内核**：`kernel_impl()` 应为 `avx2`/`neon`；若为 `scalar`，检查 NASM 或
  编译器是否可用。

## 11. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `错误：mahe_core 扩展未构建` | 未构建扩展，或位数/架构不匹配 | 运行 `python build.py`；确认 Python 与编译器同为 64 位 |
| `未找到 C++ 编译器` | 编译器不在 PATH | Linux 装 `build-essential`；Windows 加 MinGW PATH 或用 VS x64 终端 |
| `未找到 NASM` | 未安装 NASM | `apt install nasm` / `winget install nasm`；或 `--no-asm` 回退 |
| `未找到 pybind11` | 缺依赖 | `python -m pip install pybind11 numpy` |
| Windows `undefined reference to '__imp_WSAStartup'` | MinGW 忽略 MSVC 的 `#pragma comment(lib)` | 用仓库最新 `build.py`（已自动加 `-lws2_32`） |
| Windows `cannot find -lpython3XX` | 找不到 Python 导入库 | 确认 64 位 Python，且 `sys.base_prefix\libs\python3XX.lib` 存在 |
| Windows `'cl' 不是内部或外部命令` | 没在 VS x64 开发终端 | 打开 *x64 Native Tools Command Prompt*，或先跑 `vcvars64.bat` |
| 链接期 `file in wrong format` | 32 位 Python 配 64 位编译器（或反之） | 统一 64 位：`python -c "import struct;print(struct.calcsize('P')*8)"` 应为 `64` |
| `DLL load failed while importing mahe_core: 找不到指定的模块` | MinGW 保留了动态依赖 `libwinpthread-1.dll`；Python 3.8+ 加载扩展不搜索 PATH | 用最新 `build.py`（已加 `-static` 全静态链接）`--clean` 重建；或把 `C:\msys64\mingw64\bin\libwinpthread-1.dll` 复制到 `python\mahe\`；建议改用 UCRT64 工具链或 MSVC |
| Worker 一直连不上 Master | 防火墙 / 跨网段广播被拦 | 放行 TCP 47820 + UDP 47821，或 `--master <IP>` |
| Web 控制台打不开 | 端口被占用，或训练未启动 | 确认进程已启动；浏览器访问 `http://127.0.0.1:47830/`；改用 `--ui-port` |
| 页面一直停在 BOOTING / 没有初始化日志 | 前端未发出 `ui_ready`，或事件流被代理拦截 | 检查 `/api/status`；关闭会拦截 SSE 的代理；刷新页面 |
| 扫描不到局域网设备 | 跨网段、ICMP 被拦、AP 隔离 | 填写正确 CIDR（`--subnet`）；关闭 AP 隔离；改用 `--master` 直连部署 |
| 自动部署失败（SSH） | 目标机未开 SSH / 未配置免密 / 用户名错误 | 目标机安装并启动 SSH；`ssh-copy-id user@host`；或用“手动部署”脚本 |
| 安装器没有创建桌面快捷方式 | PowerShell 不可用或执行策略拦截 | 在 cmd 中运行安装器查看报错；确认 `%USERPROFILE%\Desktop\MAHE.lnk`；必要时手动运行 `uninstall.ps1` 后重装 |
| 点「浏览…」没有弹出对话框 | Linux 缺 zenity/kdialog，或引擎在无桌面会话中运行 | 安装 `zenity`（或 `kdialog`）；也可直接在输入框手填路径 |
| 一直停在 READY、没有轮次 | 默认是干净启动，需要手动开始，且要等节点入网 | 在「训练」页点开始；单机自测改用 `--test` |
| 「开始训练」后一直等待节点 | 没有 Worker 入网，或令牌/端口不一致 | 检查信息流的“等待 worker 入网”；核对 `--token`、47820/UDP 47821；最多等 `--wait-timeout` 秒 |
| 节点移出后列表又出现 | 子端是旧版本，不响应 `BYE`，心跳重新注册 | 用当前分支源码重新部署/重启子端 |
| 助手页显示“待启用” | 未配置 `DEEPSEEK_API_KEY` | 设置后重启引擎；这是预期行为，不会发外部请求 |
| 数据集下拉框为空 | 目录尚未加载或 `refresh_datasets` 失败 | 初始化完成后再看；点「刷新」；查看信息流 `[ERROR]` |
| `--bundle` 提示缺 PyInstaller | 未安装 | `python -m pip install pyinstaller` |
| macOS 打开应用提示“已损坏” | 未签名/公证 | 右键“打开”，或 `xattr -dr com.apple.quarantine ~/Applications/MAHE.app` |

### 排查内核是否真的加速

```bash
python build.py --test
```

测试会逐元素比对 SIMD 与标量结果；在 Apple Silicon 上应能看到全 PASS，
且 `kernel_impl()` 返回 `neon`。若返回 `scalar`，说明 `.S` 未参与编译。

## 12. FAQ

**需要 GPU 吗？** 不需要。当前后端是 CPU NumPy MLP，GPU 不是前置条件。

**能训练自己的数据吗？** 目前支持“生成器描述符”形式的数据集：描述符随任务下发，
各节点在本地确定性重建同一份数据，因此不需要传数据文件。内置了三个示例；
社区目录可以下发更多描述符，但生成器必须是引擎已注册的类型（未知生成器会被跳过）。
接任意自有数据文件（CSV/NPZ 分发与分片）在路线图中。

**为什么默认一个节点都没有？** v0.1 起默认是干净环境，避免演示节点混进真实集群；
需要本机自测时加 `--test`，需要真实节点时在「部署」页扫描或手动部署。

**怎么把节点从分组里去掉？** 集群页节点右侧的 `✕`，或「清空节点」。

**不同系统能混组集群吗？** 可以。帧协议与端口一致，各节点用各自平台的扩展；
但版本与令牌必须一致。

**数据安全吗？** v0.1 无 TLS，令牌只做入网校验。仅在可信局域网使用；跨公网
请自行加隧道（如 WireGuard）。

**为什么不直接每步同步梯度？** 局域网上传参数量级的流量在每步同步时会成为瓶颈；
本地多步 + Δ 把每轮通信压到一次参数量传输。

**安装包为什么这么大？** 内含 Python 运行时、numpy 与 PyInstaller 自解压载荷，
换来终端用户无需安装 Python/numpy。不再包含 JRE。

## 13. 安全提示

- 演示令牌 `mahe-demo-token` 仅用于本地演示，部署前必须替换。
- Master 的 47820/47821 不要暴露到公网。
- Web 控制台 47830 只绑定本机回环地址，不要改成 `0.0.0.0`。
- 自动部署脚本包含集群令牌，下载或分发脚本前请确认接收方可信。
- 自动部署依赖 SSH 密钥，不要在脚本中写入密码。
- 检查点与日志可能包含训练数据统计信息，分发前请确认内容。
