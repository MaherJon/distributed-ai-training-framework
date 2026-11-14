**English** | [简体中文](README.zh-CN.md)

# MAHE · Distributed AI Training Framework

> Turn idle computers in dorms and offices into an AI training cluster.

**MAHE** organizes ordinary computers on a LAN into a training cluster using a
parameter-server architecture — no GPU budget required, starting from 2-core CPUs.
The master only aggregates parameters and schedules work; workers train on local
shards with their own compute. The core is C++17 with hand-written SIMD assembly,
Python drives training and orchestration, and a self-contained web terminal
(HTML/CSS/JS served by Python) provides the control console.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-HTML%20%2F%20CSS%20%2F%20JS-58E0B6?logo=html5&logoColor=white)
![Status](https://img.shields.io/badge/status-v0.1-orange)

If this project helps you, a star is appreciated.

![demo](./docs/gif/demo.gif)

---

## Contents

- [Why MAHE](#why-mahe)
- [Key features](#key-features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [SIMD kernels](#simd-kernels)
- [Cross-platform builds](#cross-platform-builds)
- [LAN cluster](#lan-cluster)
- [Web console](#web-console)
- [One-click installer](#one-click-installer)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

User guide (Chinese): [docs/USER_GUIDE.md](docs/USER_GUIDE.md) · Design doc (Chinese): [DESIGN.md](DESIGN.md)

---

## Why MAHE

- **Start without GPUs.** The hard part of distributed training is usually not
  FLOPs but the engineering cost of coordinating machines. MAHE needs 2 CPU cores
  and a LAN — retired office desktops become usable compute.
- **Friendly to weak networks.** Workers train K local steps and upload only the
  parameter delta Δ; the master merges deltas weighted by shard size. Communication
  drops by 1–2 orders of magnitude versus per-step gradient sync.
- **No platform or compiler lock-in.** One build script covers Windows (MinGW/MSVC),
  Linux and macOS. x86-64 uses AVX2, Apple Silicon/ARM64 uses NEON, and missing
  assembler toolchains fall back to scalar code automatically.
- **Distribute as one installer.** The Python engine, C++ kernels and web terminal
  are packed into a single installer that creates desktop shortcuts — end users
  install no Python, numpy or JDK.

## Key features

| Capability | Description |
|---|---|
| Parameter-server training | Master owns global params and optimizer state; workers hold replicas and data shards |
| Compute-weighted data parallelism | Workers report a micro-benchmark score at join time; shard size ∝ `score / Σscore`, so fast nodes don't wait on slow ones |
| Local multi-step aggregation | FedAvg-style: K local steps per round, only Δ is uploaded — big communication savings |
| Auto-discovery and direct connect | Master broadcasts over UDP periodically; use `--master <IP>` across subnets |
| Fault-tolerant scheduling | Heartbeat watchdog reclaims dead nodes' shards; each round waits for `--min-workers` |
| SIMD kernels | AVX2 (x86-64) / NEON (aarch64) for delta reduction and SGD update, dispatched at runtime |
| Checkpoints | `.npz` written at end/interruption, resumable |
| Observable UI | Browser web terminal on `127.0.0.1:47830`; boot subtitles finish before training init; logs stream as `[KIND] message` |
| LAN scan & deploy | Discover LAN hosts, filter by remote-management ports, then pull source from GitHub and build workers |
| One-click installer | `build.py --bundle` produces a single-file installer that creates desktop shortcuts |
| Cluster packaging | `--build-cluster` generates master/worker bundles by device count and OS |

## Quick start

Three commands for a single-machine end-to-end demo (1 master + 2 worker subprocesses):

```bash
python3 -m pip install numpy pybind11

python3 build.py --test        # build and run C++ tests (kernel/protocol/scheduler/handshake)
python3 build.py               # build the Python extension

PYTHONPATH=python python3 -m mahe.demo_train --role demo --rounds 20
```

The demo opens `http://127.0.0.1:47830/` in your browser. The boot animation runs
first; training modules are initialized only after the page sends `ui_ready`.
Initialization logs appear in the left info panel as `[INIT] ...`. Use
`--no-browser` to suppress auto-open and `--no-ui` for headless servers.

Real terminal output:

```text
INFO mahe.master: round=1  workers=2 eval_loss=0.2816 eval_acc=0.9080
INFO mahe.master: round=2  workers=2 eval_loss=0.2356 eval_acc=0.9140
...
```

A checkpoint is written to `checkpoints/mahe_ckpt.npz` (or `~/.mahe/checkpoints/`
in the packaged app). On Windows, replace `PYTHONPATH=python` with
`set PYTHONPATH=%CD%\python`; everything else is identical.

Full dependency, compiler and troubleshooting notes are in the
[user guide](docs/USER_GUIDE.md) (Chinese).

## Architecture

```text
Web terminal (browser, separate process, crash-isolated)
        │  127.0.0.1:47830 · HTTP + SSE + JSON
Python control plane — master  (aggregation / eval / checkpoints / UI events)
        │  pybind11
C++17 core — transport · discovery · scheduler · kernel
        │  TCP 47820 · 16-byte frame header (magic MAHE / v1)
   ┌────┴─────┐
 Worker A   Worker N   (Python + C++ extension)
```

## How it works

1. **Join.** Workers listen for the master's UDP broadcast (47821), then open a TCP
   connection carrying their compute score and token. The master verifies the token
   and assigns a `node_id`.
2. **Shard.** The scheduler splits this round's data into weighted shards by compute
   score and dispatches them to workers.
3. **Local training.** Each worker runs K steps (`--k-steps`) on its shard and computes
   only the parameter delta Δ.
4. **Aggregate.** The master averages Δ weighted by shard sample counts and applies the
   global update with a SIMD kernel.
5. **Recover.** Nodes whose heartbeat times out are marked dead; their shards return to
   the pending pool for the next round.

Communication per round (same model, same round):

| Strategy | Upload per round | Relative traffic |
|---|---|---|
| Per-step gradient sync | K × params | baseline |
| Local K steps + Δ (MAHE) | 1 × params | ~1/K |

## SIMD kernels

| Kernel | Platform | Implementation |
|---|---|---|
| `mahe_reduce_add_f32` | x86-64 | NASM / AVX2, 8×f32 per instruction |
| `mahe_reduce_add_f32` | aarch64 | GAS / NEON, 16×f32 per iteration (4×q registers) |
| `mahe_sgd_update_f32` | x86-64 | AVX2 FMA |
| `mahe_sgd_update_f32` | aarch64 | `dup + fneg + fmla` for `param -= lr*grad` |
| all | any | scalar C++ fallback, element-wise identical |

Dispatch is automatic:

```bash
python -c "import mahe; print(mahe.mahe_core.kernel_impl())"
```

prints `avx2`, `neon` or `scalar`. Use `--no-asm` to force the scalar path.

## Cross-platform builds

| Platform | Compiler | Assembly | Artifact |
|---|---|---|---|
| Windows x64 | MinGW-w64 (recommended) / MSVC | NASM `-f win64` | `mahe_core.cp3XX-win_amd64.pyd` |
| Linux x86-64 | g++ / clang++ | NASM `-f elf64` | `mahe_core.cpython-3XX-x86_64-linux-gnu.so` |
| Linux aarch64 | g++ / clang++ | NEON (assembled `.S`) | `mahe_core.cpython-3XX-aarch64-linux-gnu.so` |
| macOS Apple Silicon | clang++ | NEON (assembled `.S`) | `mahe_core.cpython-3XX-darwin.so` |
| macOS Intel | clang++ | NASM | same |

```bash
python build.py                     # core + Python extension
python build.py --test              # build and run C++ tests
python build.py --bundle            # one-click installer (engine + web UI)
python build.py --no-asm            # force scalar kernels
python build.py --compiler g++-13   # pick a compiler
python build.py --clean             # remove artifacts
```

On Windows, keep Python and the compiler both 64-bit. MinGW setup is described in the
[user guide · Windows](docs/USER_GUIDE.md#windows).

## LAN cluster

Master (listens on TCP 47820 / UDP 47821):

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role master --min-workers 2 --rounds 30
```

Worker (same subnet, discovered via UDP broadcast):

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker
```

Across subnets, or when broadcast is blocked:

```bash
PYTHONPATH=python python3 -m mahe.demo_train --role worker --master 192.168.1.10
```

> The demo token `mahe-demo-token` is hardcoded in
> [demo_train.py](python/mahe/demo_train.py). Replace it for any real deployment;
> connections with a wrong token are rejected.

Generate deployment bundles by device count and OS:

```bash
python3 build.py --build-cluster \
  --systems "linux-x86_64,linux-aarch64,windows-x86_64,mac-arm64" \
  --master-index 0
```

Output lands in `build/cluster/device_XX_<role>_<os>-<arch>/`, each containing sources,
build scripts, launch scripts and `device_info.json`. Devices matching the build host
get a prebuilt extension; others build on first launch on the target machine.

## Web console

The web terminal is served by the Python control plane on `127.0.0.1:47830` and
opens automatically (HTML/CSS/JS, no external dependencies).

```bash
python3 -m mahe.demo_train --role demo      # opens the browser
python3 -m mahe.demo_train --role master    # LAN master + console
python3 -m mahe.demo_train --role master --no-browser --no-ui   # headless
```

Flow:

1. The browser plays the boot subtitles.
2. On completion it posts `ui_ready`; only then does the backend initialize the
   C++ core, scheduler and training modules.
3. Initialization messages appear in the left info panel as `[INIT] …`, `[WARN] …`,
   `[ERROR] …`.
4. After initialization, a deployment panel offers LAN auto-deploy or manual scripts.

The layout is an intentionally restrained eDEX-style terminal: dark background,
hairline grid, monospace type and a single teal accent. The console shows boot
output, the node table, loss/acc curves, an event log and the deployment view.
Closing the browser does not affect training; reopening resyncs a full snapshot.

### Sub-node deployment

- **Scan**: the master probes its private /24 (or a user-supplied CIDR), resolves
  hostnames and checks SSH/SMB/RDP/WinRM/AFP/VNC ports. Results show device name,
  type and whether auto-deploy is possible.
- **Deploy**: for SSH-capable hosts, the master runs a generated bootstrap script
  that shallow-clones the configured GitHub repository/branch, installs
  numpy/pybind11, runs `build.py`, and starts a worker in the background.
- **Manual**: download the POSIX or Windows script and run it on the target machine.
  Automatic deployment uses SSH key auth (`BatchMode`); passwords are never stored.

## One-click installer

Pack the frozen Python engine and the web terminal into a single-file installer:

```bash
python -m pip install pyinstaller
python build.py --bundle
```

| Platform | Artifact |
|---|---|
| Windows | `build/installer/MAHE-Setup.exe` |
| macOS | `build/installer/MAHE-Setup` |
| Linux | `build/installer/MAHE-Setup` |

The installer runs per-user (no admin rights) and always creates a **desktop
shortcut**; on Windows it also creates a Start Menu entry and an uninstall
registry item. Launching the shortcut starts `mahe-engine` (default `--role demo`,
override with `MAHE_ENGINE_ROLE=master|worker`) and opens the web terminal.

PyInstaller only produces artifacts for the host platform, so build each target on
its own OS. Size is roughly 100–130 MB. Public macOS distribution additionally
requires signing and notarization.

## Project layout

```text
core/
  include/mahe/         protocol, transport, discovery, scheduler, kernel headers
  src/                  protocol / transport / discovery / scheduler / kernel / node
  asm/reduce_avx2.asm   AVX2 delta reduction and SGD update (NASM, x86-64)
  asm/reduce_neon.S     NEON delta reduction and SGD update (GAS, aarch64)
  bindings/pybind.cpp   Python extension (zero-copy NumPy buffers)
  tests/test_core.cpp   C++ unit tests (including end-to-end handshake)
python/mahe/
  model.py              NumPy MLP (CPU-only reference backend)
  worker_node.py        local multi-step training, Δ upload
  master_node.py        parameter server: weighted aggregation, eval, checkpoints
  web_server.py         HTTP + SSE + command channel
  webui/                static web terminal (HTML/CSS/JS)
  lan_scan.py           LAN device discovery and deployability filter
  deployer.py           GitHub source pull + remote build/deploy scripts
  demo_train.py         demo / master / worker entry point
packaging/
  engine_entry.py       PyInstaller entry for mahe-engine
  installer_entry.py    installer: copy files + desktop shortcuts
docs/USER_GUIDE.md      user guide (Chinese)
build.py                cross-platform build / test / cluster bundling / installer
```

## Testing

```bash
python build.py --test
```

Coverage:

- SIMD kernels vs scalar implementation, element-wise comparison (1e-6 tolerance)
- Frame codec: length, magic, header parsing
- Weighted sharding 3:1 → 750/250
- Task dispatch, completion, node removal
- Master/worker handshake, token rejection, bidirectional messages, graceful shutdown
- Web terminal: static assets, `ui_ready` handshake and `/api/status` snapshot (smoke test)

End-to-end: `--role demo` really spawns 2 worker subprocesses and writes a checkpoint.

## Roadmap

- **v0.1 (current)**: C++ core + AVX2/NEON kernels + Python parameter-server training +
  local demo + web terminal + LAN deployment flow
- **v0.2**: real multi-machine LAN validation, checkpoint resume, int8 gradient compression
- **v0.3**: PyTorch backend, richer web console views and control commands
- **v0.4**: TLS, web monitoring endpoint

## Known limitations

- The training backend is a NumPy MLP (CPU); PyTorch is not wired up yet.
- int8 gradient compression is designed but not implemented.
- Multi-machine validation is mostly local multi-process; real LAN deployment follows the guide.
- Automatic sub-node deployment requires SSH key authentication; Windows hosts that
  only expose WinRM must use the manual PowerShell script.

## Contributing

Issues and PRs are welcome. Before submitting:

1. Make `python build.py --test` pass.
2. Keep the C++ and Python sides in sync; new kernels must ship with a scalar fallback.
3. For assembly changes, test on the target architecture and state the `kernel_impl()`
   output in the PR description.

If this project helps you, a star is the best support.

## License

[GPL-3.0](LICENSE) · compatible with pybind11 (BSD) and NASM (BSD-2).
