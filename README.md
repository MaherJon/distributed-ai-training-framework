**English** | [简体中文](README.zh-CN.md)

# MAHE · Distributed AI Training Framework

> Turn idle computers in dorms and offices into an AI training cluster.

**MAHE** organizes ordinary computers on a LAN into a training cluster using a
parameter-server architecture — no GPU budget required, starting from 2-core CPUs.
The master only aggregates parameters and schedules work; workers train on local
shards with their own compute. The core is C++17 with hand-written SIMD assembly,
Python drives training and orchestration, and Kotlin Compose provides a desktop
console.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-Kotlin%20Compose-7F52FF?logo=kotlin&logoColor=white)
![Status](https://img.shields.io/badge/status-v0.1-orange)

If this project helps you, a star is appreciated.

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
- [Desktop console](#desktop-console)
- [One-click installers](#one-click-installers)
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
- **Distribute as one installer.** The Python engine, C++ kernels, Kotlin UI and a
  trimmed JRE are packed into a single MSI/DMG/DEB — end users install no Python,
  numpy or JDK.

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
| Observable UI | JSON Lines control channel; UI crash or disconnect never affects training, reconnect resyncs snapshots |
| One-click packaging | `build.py --bundle` produces MSI / DMG / DEB |
| Cluster packaging | `--build-cluster` generates master/worker bundles by device count and OS |

## Quick start

Three commands for a single-machine end-to-end demo (1 master + 2 worker subprocesses):

```bash
python3 -m pip install numpy pybind11

python3 build.py --test        # build and run C++ tests (kernel/protocol/scheduler/handshake)
python3 build.py               # build the Python extension

PYTHONPATH=python python3 -m mahe.demo_train --role demo --rounds 20
```

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
Kotlin Compose console  (separate process, crash-isolated)
        │  127.0.0.1:47830 · JSON Lines
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
python build.py --bundle            # one-click installer (engine + UI + JRE)
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

## Desktop console

A standalone Kotlin Compose Desktop process subscribing to events on `127.0.0.1:47830`:

```bash
cd ui
./gradlew run          # Windows: .\gradlew.bat run
```

Start training first (`demo` or `master` opens 47830), then the UI. The console shows a
boot self-check sequence, the node table, loss/acc curves and the event log; the
`◉ LINKED` / `◌ OFFLINE` indicator in the top-right shows connection state.

## One-click installers

Pack the Python engine, C++/NEON kernels, Kotlin UI and a trimmed JRE into one installer:

```bash
python -m pip install pyinstaller   # plus JDK 17+
python build.py --bundle
```

| Platform | Artifact | Path |
|---|---|---|
| Windows | MSI | `ui/build/compose/binaries/main/msi/MAHE-1.0.0.msi` |
| macOS | DMG | `ui/build/compose/binaries/main/dmg/MAHE-1.0.0.dmg` |
| Linux | DEB | `ui/build/compose/binaries/main/deb/*.deb` |

After installation, just launch the icon: when port 47830 is not listening, the UI
spawns the bundled `mahe-engine` (default `--role demo`; override with
`MAHE_ENGINE_ROLE=master|worker`).

PyInstaller and jpackage only produce artifacts for the host platform, so build each
target on its own OS. Size is roughly 100–130 MB. Public macOS distribution additionally
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
  ui_bridge.py          JSON Lines control channel
  demo_train.py         demo / master / worker entry point
ui/                     Kotlin Compose Desktop console
packaging/              PyInstaller entry point
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

End-to-end: `--role demo` really spawns 2 worker subprocesses and writes a checkpoint.

## Roadmap

- **v0.1 (current)**: C++ core + AVX2/NEON kernels + Python parameter-server training +
  local demo + UI scaffold
- **v0.2**: real multi-machine LAN validation, checkpoint resume, int8 gradient compression
- **v0.3**: PyTorch backend, full three-view UI and control commands
- **v0.4**: TLS, web monitoring endpoint

## Known limitations

- The training backend is a NumPy MLP (CPU); PyTorch is not wired up yet.
- int8 gradient compression is designed but not implemented.
- Multi-machine validation is mostly local multi-process; real LAN deployment follows the guide.
- The UI is a working console scaffold; create-cluster / start-stop interactions are not
  wired to control commands yet.

## Contributing

Issues and PRs are welcome. Before submitting:

1. Make `python build.py --test` pass.
2. Keep the C++ and Python sides in sync; new kernels must ship with a scalar fallback.
3. For assembly changes, test on the target architecture and state the `kernel_impl()`
   output in the PR description.

If this project helps you, a star is the best support.

## License

[GPL-3.0](LICENSE) · compatible with pybind11 (BSD) and NASM (BSD-2).
