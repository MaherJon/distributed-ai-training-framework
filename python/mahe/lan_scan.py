"""局域网设备发现：为子端部署找出“真正的目标设备”。

只做“可部署性”判断，不执行任何远程操作：

1. 以本机 IPv4 推导 /24 网段（也可手动指定 CIDR）；
2. 并发 ICMP 探测在线主机（无 ping 时回退 TCP 探测）；
3. 用邻居表（ARP）补 MAC、按 OUI 推断厂商，反向 DNS / NetBIOS 取主机名；
4. 探测常见远程管理端口（SSH / SMB / RDP / WinRM / VNC / 打印机）以及 MAHE 端口；
5. 给每台主机归类（电脑 / NAS / 打印机 / 路由器 / 手机 / IoT），并标记它是否
   值得作为部署候选，供界面默认只展示候选设备。

注意：扫描仅允许私有网段，避免误扫公网地址。
"""

from __future__ import annotations

import concurrent.futures as futures
import ipaddress
import platform
import re
import shutil
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
PRINT_PORT = 9100
MAHE_PORT = 47820

PROBE_PORTS = [SSH_PORT, SMB_PORT, NETBIOS_PORT, RDP_PORT, WINRM_PORT,
               AFP_PORT, VNC_PORT, PRINT_PORT, MAHE_PORT]

MAX_HOSTS = 1024
PING_WORKERS = 64
PROBE_WORKERS = 32

# 常见厂商 OUI（前 3 字节）。这不是完整的 IEEE 数据库，只用于让设备更好辨认；
# 未命中时界面显示“厂商未知”，不影响扫描与部署。
_OUI: dict[str, str] = {
    # Apple
    "000393": "Apple", "000A27": "Apple", "000D93": "Apple", "001B63": "Apple",
    "001EC2": "Apple", "002312": "Apple", "002500": "Apple", "002608": "Apple",
    "3C0754": "Apple", "6C709F": "Apple", "8C8590": "Apple", "ACBC32": "Apple",
    "B8E856": "Apple", "D0E140": "Apple", "F01898": "Apple", "F0DBE2": "Apple",
    "78CA39": "Apple", "7CD1C3": "Apple",
    # Dell
    "001422": "Dell", "001C23": "Dell", "00219B": "Dell", "180373": "Dell",
    "24B6FD": "Dell", "44A842": "Dell", "544E90": "Dell", "782BCB": "Dell",
    "B82A72": "Dell", "BC305B": "Dell", "D4BED9": "Dell", "149182": "Dell",
    # HP
    "001B78": "HP", "00215A": "HP", "0025B3": "HP", "3C4A92": "HP",
    "3CD92B": "HP", "40A8F0": "HP", "9457A5": "HP", "A0481C": "HP",
    "B05ADA": "HP", "D89D67": "HP", "EC8EB5": "HP", "9C8E99": "HP",
    # Lenovo
    "0012FE": "Lenovo", "001E37": "Lenovo", "28D244": "Lenovo",
    "309C23": "Lenovo", "54EE75": "Lenovo", "68F728": "Lenovo",
    "98FA9B": "Lenovo", "C85B76": "Lenovo", "6025B3": "Lenovo",
    # Intel
    "001B21": "Intel", "001E64": "Intel", "0024D7": "Intel", "3C970E": "Intel",
    "448500": "Intel", "7C7A91": "Intel", "8086F2": "Intel", "B4D5BD": "Intel",
    # ASUS
    "001BFC": "ASUS", "00248C": "ASUS", "04D4C4": "ASUS", "1C872C": "ASUS",
    "2C56DC": "ASUS", "50465D": "ASUS", "704D7B": "ASUS", "9C5C8E": "ASUS",
    "AC220B": "ASUS", "D017C2": "ASUS",
    # Gigabyte / 主板整机
    "001A4D": "Gigabyte", "1C1B0D": "Gigabyte", "408D5C": "Gigabyte",
    "74D435": "Gigabyte", "94DE80": "Gigabyte", "B42E99": "Gigabyte",
    # 虚拟机 / 容器
    "000569": "VMware", "000C29": "VMware", "001C14": "VMware",
    "005056": "VMware", "080027": "VirtualBox", "00155D": "Hyper-V",
    "00163E": "Xen", "525400": "QEMU",
    # 单板机 / IoT
    "B827EB": "Raspberry Pi", "DCA632": "Raspberry Pi",
    "E45F01": "Raspberry Pi", "2CCF67": "Raspberry Pi",
    "240AC4": "Espressif", "2462AB": "Espressif", "30AEA4": "Espressif",
    "5CCF7F": "Espressif", "84CCA8": "Espressif",
    # 网络设备
    "001D0F": "TP-Link", "14CC20": "TP-Link", "50C7BF": "TP-Link",
    "A42BB0": "TP-Link", "C025E9": "TP-Link", "EC086B": "TP-Link",
    "F4F26D": "TP-Link", "98DAC4": "TP-Link",
    "00146C": "Netgear", "204E7F": "Netgear", "A040A0": "Netgear",
    "9C3DCF": "Netgear", "C40415": "Netgear",
    "001B11": "D-Link", "1C7EE5": "D-Link", "78321B": "D-Link",
    "0022B0": "D-Link", "14D64D": "D-Link",
    "C83A35": "Tenda", "D83214": "Tenda", "0023CD": "Tenda",
    "000C42": "MikroTik", "2CC81B": "MikroTik", "488F5A": "MikroTik",
    "00156D": "Ubiquiti", "245A4C": "Ubiquiti", "788A20": "Ubiquiti",
    "F09FC2": "Ubiquiti", "44D9E7": "Ubiquiti", "802AA8": "Ubiquiti",
    "001AA1": "Cisco", "001B54": "Cisco", "2C3F38": "Cisco",
    "706BB9": "Cisco", "00170F": "Cisco", "002155": "Cisco",
    "000B86": "Aruba", "6CF37F": "Aruba", "94B40F": "Aruba",
    "001349": "Zyxel", "5CF4AB": "Zyxel", "002389": "H3C", "3CE5A6": "H3C",
    # 手机 / 平板
    "009EC8": "Xiaomi", "286C07": "Xiaomi", "34CE00": "Xiaomi",
    "508F4C": "Xiaomi", "64B473": "Xiaomi", "7811DC": "Xiaomi",
    "8CBEBE": "Xiaomi", "F0B429": "Xiaomi", "50EC50": "Xiaomi",
    "00E0FC": "Huawei", "286ED4": "Huawei", "4846FB": "Huawei",
    "5C7D5E": "Huawei", "70723C": "Huawei", "80B686": "Huawei",
    "9CB2B2": "Huawei",
    "0012FB": "Samsung", "001599": "Samsung", "001D25": "Samsung",
    "342387": "Samsung", "5C0A5B": "Samsung", "781FDB": "Samsung",
    "8C7712": "Samsung", "A00798": "Samsung", "C81479": "Samsung",
    "3C5AB4": "Google", "546009": "Google", "F4F5D8": "Google",
    "94EB2C": "Google",
    # NAS / 打印机 / 电视盒子
    "001132": "Synology", "9009D0": "Synology", "00089B": "QNAP",
    "245EBE": "QNAP",
    "000085": "Canon", "001E8F": "Canon", "2C9EFC": "Canon",
    "F48139": "Canon", "88FB5B": "Canon",
    "000048": "Epson", "080037": "Epson", "64EB8C": "Epson",
    "A4EE57": "Epson",
    "008077": "Brother", "30055C": "Brother", "B42200": "Brother",
    "3C2AF4": "Brother",
    "0000AA": "Xerox", "001BA9": "Xerox",
    "000074": "Ricoh", "002673": "Ricoh", "58B961": "Ricoh",
    "44650D": "Amazon", "6837E9": "Amazon", "74C246": "Amazon",
    "FC65DE": "Amazon",
    "000E58": "Sonos", "5CAAFD": "Sonos", "7828CA": "Sonos",
}

_NETWORK_VENDORS = {"TP-Link", "Netgear", "D-Link", "Tenda", "MikroTik",
                    "Ubiquiti", "Cisco", "Aruba", "Zyxel", "H3C"}
_PRINTER_VENDORS = {"Canon", "Epson", "Brother", "Xerox", "Ricoh"}
_NAS_VENDORS = {"Synology", "QNAP"}
_PHONE_VENDORS = {"Apple", "Samsung", "Xiaomi", "Huawei", "Google"}
_TV_VENDORS = {"Amazon", "Roku", "LG", "Sony", "TCL", "Hisense"}
_IOT_VENDORS = {"Espressif", "Raspberry Pi", "Sonos"}
_PC_VENDORS = {"Dell", "HP", "Lenovo", "ASUS", "Gigabyte", "Intel", "Apple",
               "Samsung", "Microsoft", "Toshiba", "Fujitsu", "Acer", "VMware",
               "VirtualBox", "Hyper-V", "Xen", "QEMU"}

_ROUTER_NAME_RE = re.compile(
    r"(router|gateway|^gw[-_]?|accesspoint|^ap[-_]|switch|modem|firewall|"
    r"openwrt|unifi|udm|mesh)", re.I)
_NAS_NAME_RE = re.compile(r"(nas|diskstation|synology|qnap|fileserver)", re.I)
_PRINTER_NAME_RE = re.compile(
    r"(print|canon|epson|brother|xerox|ricoh|hp[-_]?[0-9a-f]{4})", re.I)

_MAC_RE = re.compile(r"(?:[0-9a-fA-F]{1,2}[:-]){5}[0-9a-fA-F]{1,2}")


# ---------------------------------------------------------------------------
# 网络基础
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 身份信息：MAC / 厂商 / NetBIOS
# ---------------------------------------------------------------------------

def normalize_mac(raw: str) -> str:
    """把 00-1A-2B-… / 0:1a:2b:… 统一成 00:1a:2b:…；无效返回空串。"""
    found = _MAC_RE.search(raw or "")
    if not found:
        return ""
    parts = re.split(r"[:\-]", found.group(0))
    try:
        octets = [int(p, 16) for p in parts]
    except ValueError:
        return ""
    if len(octets) != 6:
        return ""
    if all(o == 0 for o in octets) or all(o == 255 for o in octets):
        return ""
    if octets[0] & 1:  # 组播 / 广播地址
        return ""
    return ":".join(f"{o:02x}" for o in octets)


def mac_vendor(mac: str) -> str:
    """按 OUI 推断厂商；未收录时返回空串。"""
    mac = normalize_mac(mac)
    if not mac:
        return ""
    key = mac.replace(":", "").upper()[:6]
    return _OUI.get(key, "")


def read_arp_table() -> dict[str, str]:
    """读取本机邻居表（IP → MAC）。失败时返回空字典，不影响扫描。"""
    commands: list[list[str]]
    if _IS_WIN:
        commands = [["arp", "-a"]]
    elif _IS_MAC:
        commands = [["arp", "-an"]]
    else:
        commands = [["ip", "neigh"], ["arp", "-an"]]

    table: dict[str, str] = {}
    for cmd in commands:
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  errors="replace", timeout=5,
                                  creationflags=_CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in (proc.stdout or "").splitlines():
            ip_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", line)
            mac_match = _MAC_RE.search(line)
            if not ip_match or not mac_match:
                continue
            ip = ip_match.group(1)
            if ip in table:
                continue
            mac = normalize_mac(mac_match.group(0))
            if mac:
                table[ip] = mac
        if table:
            break
    return table


def _netbios_name(ip: str, timeout: float = 1.5) -> str:
    """Windows 主机名（NetBIOS）。Linux/macOS 有 nmblookup 时也试一次。"""
    if _IS_WIN:
        exe = shutil.which("nbtstat") or r"C:\Windows\System32\nbtstat.exe"
        cmd = [exe, "-A", ip]
    elif shutil.which("nmblookup"):
        cmd = ["nmblookup", "-A", ip]
    else:
        return ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", timeout=timeout,
                              creationflags=_CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (proc.stdout or "").splitlines():
        m = re.match(r"\s*(\S+)\s+<00>\s+UNIQUE", line)
        if m:
            name = m.group(1).strip()
            if name and name not in ("__MSBROWSE__",):
                return name
    return ""


# ---------------------------------------------------------------------------
# 归类与候选判断
# ---------------------------------------------------------------------------

def _os_from_ports(ports: list[int]) -> tuple[str, str]:
    """返回 (os_kind, 展示类型)。"""
    opened = set(ports)
    if opened & {SMB_PORT, NETBIOS_PORT, RDP_PORT, WINRM_PORT}:
        return "windows", "Windows 主机"
    if AFP_PORT in opened:
        return "macos", "macOS 主机"
    if SSH_PORT in opened:
        return "linux", "Linux / Unix 主机"
    return "unknown", "未知设备"


def _kind_of(vendor: str, hostname: str, netbios: str,
             ports: list[int]) -> tuple[str, str]:
    """推断设备类别，返回 (kind, 展示类型)。

    kind: computer / nas / printer / router / phone / tv / iot / unknown
    """
    name = f"{hostname} {netbios}".strip()
    opened = set(ports)
    has_windows_mgmt = bool(opened & {SMB_PORT, NETBIOS_PORT, RDP_PORT,
                                      WINRM_PORT})
    has_mgmt = bool(opened & {SSH_PORT, AFP_PORT, VNC_PORT,
                              SMB_PORT, NETBIOS_PORT, RDP_PORT, WINRM_PORT})

    if PRINT_PORT in opened:
        return "printer", "打印机"
    if vendor in _PRINTER_VENDORS and not has_windows_mgmt:
        return "printer", "打印机"
    if _PRINTER_NAME_RE.search(name) and not has_mgmt:
        return "printer", "打印机"

    if vendor in _NETWORK_VENDORS and not has_windows_mgmt:
        return "router", "路由器 / AP"
    if _ROUTER_NAME_RE.search(name) and not has_mgmt:
        return "router", "路由器 / AP"

    if vendor in _NAS_VENDORS or _NAS_NAME_RE.search(name):
        return "nas", "NAS / 存储"

    if not has_mgmt:
        # 没有管理端口时只能靠厂商猜类别：先排除手机/电视/IoT，避免把
        # 随手连上 WiFi 的手机当成训练节点；PC 厂商仍保留为候选。
        if vendor == "Apple":
            return "apple", "Apple 设备（Mac / iPhone）"
        if vendor in _PHONE_VENDORS:
            return "phone", "手机 / 平板"
        if vendor in _TV_VENDORS:
            return "tv", "电视 / 盒子"
        if vendor in _IOT_VENDORS:
            return "iot", "IoT 设备"
        if vendor in _PC_VENDORS:
            return "computer", "电脑（管理端口未开放）"
        return "unknown", "未知设备"

    os_kind, type_label = _os_from_ports(ports)
    return "computer", type_label


def _deploy_channel(kind: str, open_ports: list[int]) -> tuple[str, bool, str]:
    """返回 (部署通道, 是否支持自动部署, 说明)。"""
    if kind not in ("computer", "nas", "unknown"):
        return "none", False, "非训练目标（手机 / 打印机 / 路由器等）"
    opened = set(open_ports)
    if SSH_PORT in opened:
        return "ssh", True, "开放 SSH，可自动拉取源码并编译"
    if WINRM_PORT in opened:
        return "winrm", False, "开放 WinRM，需在 Master 安装 pywinrm 才能自动部署"
    if opened & {SMB_PORT, NETBIOS_PORT, RDP_PORT}:
        return "manual", False, "未开放 SSH，请使用手动部署脚本"
    return "manual", False, "未发现远程管理通道，请使用手动部署"


def _candidate(kind: str, channel: str, deployable: bool,
               mahe: bool) -> bool:
    """是否值得在界面默认展示（= 可能是可部署的训练节点）。"""
    if mahe or deployable:
        return True
    if channel == "winrm":
        return True
    return kind in ("computer", "nas")


def _describe(ip: str, *, latency_ms: float, hostname: str = "",
              netbios: str = "", mac: str = "",
              ports: list[int] | None = None) -> dict[str, Any]:
    """把探测到的原始信息整理成界面设备项（纯函数，便于测试）。"""
    ports = sorted(int(p) for p in (ports or []))
    mac = normalize_mac(mac)
    vendor = mac_vendor(mac)
    kind, type_label = _kind_of(vendor, hostname, netbios, ports)
    os_kind, _ = _os_from_ports(ports)
    channel, deployable, note = _deploy_channel(kind, ports)
    mahe = MAHE_PORT in ports
    candidate = _candidate(kind, channel, deployable, mahe)

    name = hostname or netbios or ""
    if not name and vendor:
        name = f"{vendor} 设备"
    if not name:
        name = "未命名设备"

    if mahe:
        note = "已运行 MAHE（端口 47820），可直接入网"
    elif kind not in ("computer", "nas", "unknown"):
        note = f"非训练目标：{type_label}"

    reason = note
    if not candidate:
        reason = "无远程管理端口的非目标设备，默认隐藏"

    return {
        "ip": ip,
        "name": name,
        "hostname": hostname,
        "netbios": netbios,
        "mac": mac,
        "vendor": vendor,
        "os": os_kind,
        "type": type_label,
        "kind": kind,
        "ports": ports,
        "channel": channel,
        "deployable": deployable,
        "mahe": mahe,
        "candidate": candidate,
        "note": note,
        "reason": reason,
        "latency_ms": latency_ms,
        "scanned_at": time.strftime("%H:%M:%S"),
    }


def _classify(open_ports: list[int]) -> tuple[str, str]:
    """兼容旧接口：仅按端口推断 (os_kind, 展示类型)。"""
    return _os_from_ports(open_ports)


def _device_from_host(ip: str, latency_ms: float,
                      arp: dict[str, str]) -> dict[str, Any]:
    hostname = resolve_name(ip)
    netbios = "" if hostname else _netbios_name(ip)
    ports = probe_ports(ip, PROBE_PORTS)
    return _describe(ip, latency_ms=latency_ms, hostname=hostname,
                     netbios=netbios, mac=arp.get(ip, ""), ports=ports)


def scan_devices(
    subnet: str | None = None,
    prefix: int = 24,
    *,
    on_device: Callable[[dict[str, Any]], None] | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """扫描局域网并返回全部在线设备（每项带 ``candidate`` 标记，按 IP 排序）。

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

    log("SCAN", f"在线主机 {len(alive)} 台，读取邻居表并探测远程管理端口…")

    devices: list[dict[str, Any]] = []
    if not alive:
        log("WARN", "未发现在线设备：请确认设备与 Master 处于同一局域网")
        return devices

    arp = read_arp_table()
    if arp:
        log("SCAN", f"邻居表：{len(arp)} 条 MAC 记录")

    with futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        futs = {pool.submit(_device_from_host, ip, lat, arp): ip
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
    candidates = sum(1 for d in devices if d["candidate"])
    hidden = len(devices) - candidates
    log("SCAN", f"扫描完成：{candidates} 台候选设备（{ready} 台可自动部署）；"
                f"另有 {hidden} 台手机/打印机/路由器等非目标设备")
    return devices
