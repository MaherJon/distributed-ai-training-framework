"""局域网设备发现：用于子端部署前的在线设备扫描与条件筛选。

只做“可部署性”判断，不执行任何远程操作：
1. 以本机 IPv4 推导 /24 网段（也可手动指定 CIDR）；
2. 并发 ICMP 探测在线主机（无 ping 时回退 TCP 探测）；
3. 解析主机名并探测常见远程管理端口（SSH / SMB / RDP / WinRM / AFP / VNC）；
4. 按端口特征推断系统类型，并给出可用的部署通道。

注意：扫描仅允许私有网段，避免误扫公网地址。
"""

from __future__ import annotations

import concurrent.futures as futures
import ipaddress
import os
import platform
import socket
import subprocess
import time
from typing import Any, Callable

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"
_CREATE_NO_WINDOW = 0x08000000 if _IS_WIN else 0

SSH_PORT = 22
SMB_PORT = 445
NETBIOS_PORT = 139
RDP_PORT = 3389
WINRM_PORT = 5985
AFP_PORT = 548
VNC_PORT = 5900

PROBE_PORTS = [SSH_PORT, SMB_PORT, NETBIOS_PORT, RDP_PORT, WINRM_PORT,
               AFP_PORT, VNC_PORT]

MAX_HOSTS = 1024
PING_WORKERS = 64
PROBE_WORKERS = 32


def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.IPv4Address(ip)
    except ipaddress.AddressValueError:
        return False
    return addr.is_private and not addr.is_loopback and not addr.is_link_local


def local_ipv4_addresses() -> list[str]:
    """本机 IPv4（主路由地址 + 主机名解析结果），去重且只保留私有地址。"""
    found: list[str] = []

    # 主出口地址：UDP connect 不会真正发包
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass

    out: list[str] = []
    for ip in found:
        if _is_private(ip) and ip not in out:
            out.append(ip)
    return out


def local_networks(prefix: int = 24) -> list[ipaddress.IPv4Network]:
    nets: list[ipaddress.IPv4Network] = []
    for ip in local_ipv4_addresses():
        net = ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False)
        if net not in nets:
            nets.append(net)
    return nets


def _ping_command(ip: str, timeout_ms: int) -> list[str]:
    if _IS_WIN:
        return ["ping", "-n", "1", "-w", str(timeout_ms), ip]
    if _IS_MAC:
        return ["ping", "-c", "1", "-W", str(timeout_ms), ip]
    return ["ping", "-c", "1", "-W", str(max(1, timeout_ms // 1000)), ip]


def _tcp_alive(ip: str, ports: list[int], timeout: float = 0.4) -> bool:
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((ip, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def ping_host(ip: str, timeout_ms: int = 400) -> tuple[bool, float]:
    """返回 (是否在线, 往返毫秒)。ping 不可用时回退到 TCP 探测。"""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            _ping_command(ip, timeout_ms),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, timeout_ms / 1000 + 1.0),
            creationflags=_CREATE_NO_WINDOW,
        )
        ok = proc.returncode == 0
    except FileNotFoundError:
        ok = _tcp_alive(ip, [SMB_PORT, SSH_PORT], timeout=timeout_ms / 1000)
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    dt = (time.perf_counter() - t0) * 1000
    return ok, round(dt, 1)


def probe_ports(ip: str, ports: list[int], timeout: float = 0.35) -> list[int]:
    open_ports: list[int] = []
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((ip, port)) == 0:
                    open_ports.append(port)
        except OSError:
            continue
    return open_ports


def resolve_name(ip: str, timeout: float = 1.0) -> str:
    """反向解析主机名；超时或失败时返回空串（DNS 可能阻塞，放到线程池里等）。"""
    pool = futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(socket.gethostbyaddr, ip)
        host, _, _ = fut.result(timeout=timeout)
        return host.split(".")[0]
    except Exception:
        return ""
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _classify(open_ports: list[int]) -> tuple[str, str]:
    """返回 (系统推断, 展示类型)。"""
    ports = set(open_ports)
    if ports & {SMB_PORT, NETBIOS_PORT, RDP_PORT, WINRM_PORT}:
        return "windows", "Windows 主机"
    if AFP_PORT in ports:
        return "macos", "macOS 主机"
    if SSH_PORT in ports:
        return "linux", "Linux / Unix 主机"
    return "unknown", "未知设备"


def _deploy_channel(os_kind: str, open_ports: list[int]) -> tuple[str, bool, str]:
    """返回 (部署通道, 是否支持自动部署, 说明)。"""
    ports = set(open_ports)
    if SSH_PORT in ports:
        return "ssh", True, "开放 SSH，可远程拉取源码并编译"
    if os_kind == "windows" and WINRM_PORT in ports:
        return "winrm", False, "开放 WinRM，需在 Master 安装 pywinrm 才能自动部署"
    if os_kind == "windows":
        return "manual", False, "未开放 SSH，请使用手动部署脚本"
    return "manual", False, "未发现远程管理通道，请使用手动部署"


def _device_from_host(ip: str, latency_ms: float) -> dict[str, Any]:
    name = resolve_name(ip)
    ports = probe_ports(ip, PROBE_PORTS)
    os_kind, type_label = _classify(ports)
    channel, deployable, note = _deploy_channel(os_kind, ports)
    return {
        "ip": ip,
        "name": name or "未命名设备",
        "os": os_kind,
        "type": type_label,
        "ports": ports,
        "channel": channel,
        "deployable": deployable,
        "note": note,
        "latency_ms": latency_ms,
        "scanned_at": time.strftime("%H:%M:%S"),
    }


def scan_devices(
    subnet: str | None = None,
    prefix: int = 24,
    *,
    on_device: Callable[[dict[str, Any]], None] | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """扫描局域网并返回设备列表（按 IP 排序）。

    ``on_device`` 在每台设备探测完成后立即回调，便于界面渐进展示；
    ``on_log(kind, text)`` 输出扫描过程信息。
    """
    def log(kind: str, text: str) -> None:
        if on_log:
            on_log(kind, text)

    if subnet:
        try:
            net = ipaddress.IPv4Network(subnet, strict=False)
        except ValueError as exc:
            raise ValueError(f"无效网段：{subnet}（{exc}）") from exc
        networks = [net]
    else:
        networks = local_networks(prefix)
    if not networks:
        raise RuntimeError("无法确定本机局域网网段，请手动指定（如 192.168.1.0/24）")

    own = set(local_ipv4_addresses())
    hosts: list[str] = []
    for net in networks:
        if not net.is_private:
            raise ValueError(f"仅允许扫描私有网段：{net}")
        if net.num_addresses > MAX_HOSTS:
            raise ValueError(f"网段过大：{net}（最多 {MAX_HOSTS} 个地址）")
        for host in net.hosts():
            ip = str(host)
            if ip not in own and ip not in hosts:
                hosts.append(ip)

    log("SCAN", f"扫描网段 {', '.join(str(n) for n in networks)}，"
                f"共 {len(hosts)} 个地址")

    alive: list[tuple[str, float]] = []
    with futures.ThreadPoolExecutor(max_workers=PING_WORKERS) as pool:
        results = pool.map(ping_host, hosts)
        for ip, (ok, latency) in zip(hosts, results):
            if ok:
                alive.append((ip, latency))

    if not alive:
        # 有些网络会丢弃 ICMP（或企业策略禁 ping），回退 TCP 探测常见端口
        log("WARN", "ICMP 无响应，回退 TCP 探测（445/22）…")
        with futures.ThreadPoolExecutor(max_workers=PING_WORKERS) as pool:
            futs = {pool.submit(_tcp_alive, ip, [SMB_PORT, SSH_PORT], 0.5): ip
                    for ip in hosts}
            for fut in futures.as_completed(futs):
                if fut.result():
                    alive.append((futs[fut], 0.0))

    log("SCAN", f"在线主机 {len(alive)} 台，开始探测远程管理端口…")

    devices: list[dict[str, Any]] = []
    if not alive:
        log("WARN", "未发现在线设备：请确认设备与 Master 处于同一局域网")
        return devices

    with futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        futs = {pool.submit(_device_from_host, ip, lat): ip
                for ip, lat in alive}
        for fut in futures.as_completed(futs):
            ip = futs[fut]
            try:
                dev = fut.result()
            except Exception as exc:  # 单台失败不影响整体
                log("WARN", f"{ip} 探测失败：{exc}")
                continue
            devices.append(dev)
            if on_device:
                on_device(dev)

    devices.sort(key=lambda d: ipaddress.IPv4Address(d["ip"]))
    ready = sum(1 for d in devices if d["deployable"])
    log("SCAN", f"扫描完成：{len(devices)} 台在线，{ready} 台可直接自动部署")
    return devices
