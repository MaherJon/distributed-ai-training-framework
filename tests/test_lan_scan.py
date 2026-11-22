"""局域网扫描的纯函数测试：设备归类、候选过滤与身份信息整理。

这些测试不访问网络，可在任意平台的 CI 上运行。
"""

from __future__ import annotations

from mahe import lan_scan


# ---------------------------------------------------------------------------
# MAC / 厂商
# ---------------------------------------------------------------------------

def test_normalize_mac_accepts_common_forms():
    assert lan_scan.normalize_mac("00-1A-2B-3C-4D-5E") == "00:1a:2b:3c:4d:5e"
    assert lan_scan.normalize_mac("0:1a:2b:3c:4d:5e") == "00:1a:2b:3c:4d:5e"
    assert (lan_scan.normalize_mac(
        "at 00:11:22:33:44:55 on en0") == "00:11:22:33:44:55")


def test_normalize_mac_rejects_invalid_or_multicast():
    for raw in ("", "zz:zz:zz", "00:00:00:00:00:00",
                "ff:ff:ff:ff:ff:ff", "01:00:5e:00:00:01"):
        assert lan_scan.normalize_mac(raw) == ""


def test_mac_vendor_lookup():
    assert lan_scan.mac_vendor("00:14:22:aa:bb:cc") == "Dell"
    assert lan_scan.mac_vendor("8c:85:90:aa:bb:cc") == "Apple"
    assert lan_scan.mac_vendor("b8:27:eb:00:00:01") == "Raspberry Pi"
    assert lan_scan.mac_vendor("de:ad:be:ef:00:01") == ""
    assert lan_scan.mac_vendor("") == ""


# ---------------------------------------------------------------------------
# 归类
# ---------------------------------------------------------------------------

def test_classify_by_ports():
    assert lan_scan._classify([445]) == ("windows", "Windows 主机")
    assert lan_scan._classify([3389]) == ("windows", "Windows 主机")
    assert lan_scan._classify([22]) == ("linux", "Linux / Unix 主机")
    assert lan_scan._classify([548]) == ("macos", "macOS 主机")
    assert lan_scan._classify([]) == ("unknown", "未知设备")


def test_kind_of_recognizes_printer_router_phone():
    assert lan_scan._kind_of("", "", "", [9100])[0] == "printer"
    assert lan_scan._kind_of("TP-Link", "", "", [])[0] == "router"
    assert lan_scan._kind_of("", "HOME-ROUTER", "", [])[0] == "router"
    assert lan_scan._kind_of("Apple", "", "", [])[0] == "apple"
    assert lan_scan._kind_of("Samsung", "", "", [])[0] == "phone"
    assert lan_scan._kind_of("Synology", "", "", [445])[0] == "nas"
    assert lan_scan._kind_of("Dell", "", "", [])[0] == "computer"
    assert lan_scan._kind_of("", "", "", [])[0] == "unknown"


def test_kind_of_does_not_mistake_pc_with_ssh_for_phone():
    # 有 SSH 的 Apple 机器是 Mac，不是 iPhone
    kind, label = lan_scan._kind_of("Apple", "mac-studio", "", [22])
    assert kind == "computer"
    assert label == "Linux / Unix 主机"


def test_deploy_channel_rules():
    assert lan_scan._deploy_channel("computer", [22])[0:2] == ("ssh", True)
    assert lan_scan._deploy_channel("computer", [5985])[0:2] == ("winrm", False)
    assert lan_scan._deploy_channel("computer", [445])[0:2] == ("manual", False)
    assert lan_scan._deploy_channel("router", [22])[0:2] == ("none", False)
    assert lan_scan._deploy_channel("printer", [9100])[0:2] == ("none", False)


# ---------------------------------------------------------------------------
# 设备描述 / 候选过滤
# ---------------------------------------------------------------------------

def test_describe_ssh_pc_is_candidate():
    dev = lan_scan._describe(
        "192.168.1.20", latency_ms=3.2, hostname="DESKTOP-ABC",
        mac="00:14:22:aa:bb:cc", ports=[22, 445])
    assert dev["name"] == "DESKTOP-ABC"
    assert dev["vendor"] == "Dell"
    assert dev["os"] == "windows"
    assert dev["kind"] == "computer"
    assert dev["channel"] == "ssh"
    assert dev["deployable"] is True
    assert dev["candidate"] is True
    assert "SSH" in dev["note"]


def test_describe_winrm_pc_is_candidate_but_manual():
    dev = lan_scan._describe(
        "192.168.1.21", latency_ms=5.0, hostname="OFFICE-PC",
        mac="b8:2a:72:11:22:33", ports=[3389, 5985])
    assert dev["channel"] == "winrm"
    assert dev["deployable"] is False
    assert dev["candidate"] is True


def test_describe_mahe_host_is_candidate():
    dev = lan_scan._describe(
        "192.168.1.30", latency_ms=1.0, hostname="RACK-NODE",
        mac="1c:87:2c:00:00:01", ports=[lan_scan.MAHE_PORT])
    assert dev["mahe"] is True
    assert dev["candidate"] is True
    assert "MAHE" in dev["note"]


def test_describe_printer_and_phone_are_hidden():
    printer = lan_scan._describe(
        "192.168.1.40", latency_ms=2.0, netbios="HP-LaserJet",
        mac="00:1e:8f:aa:bb:cc", ports=[9100])
    assert printer["kind"] == "printer"
    assert printer["candidate"] is False
    assert "非训练目标" in printer["note"]

    phone = lan_scan._describe(
        "192.168.1.41", latency_ms=9.0, mac="8c:85:90:aa:bb:cc", ports=[])
    assert phone["kind"] == "apple"
    assert phone["candidate"] is False
    assert phone["deployable"] is False


def test_describe_unknown_host_is_hidden_and_named():
    dev = lan_scan._describe("192.168.1.50", latency_ms=12.0)
    assert dev["name"] == "未命名设备"
    assert dev["kind"] == "unknown"
    assert dev["candidate"] is False
    assert dev["ports"] == []


def test_describe_uses_vendor_when_hostname_missing():
    dev = lan_scan._describe(
        "192.168.1.60", latency_ms=1.0, mac="00:14:22:aa:bb:cc")
    assert dev["name"] == "Dell 设备"
    assert dev["candidate"] is True
    assert dev["channel"] == "manual"


def test_describe_never_defaults_to_ip_as_name():
    dev = lan_scan._describe("192.168.1.70", latency_ms=1.0)
    assert dev["name"] != dev["ip"]


def test_describe_ports_are_sorted_ints():
    dev = lan_scan._describe("192.168.1.80", latency_ms=1.0,
                             ports=["445", 22])  # type: ignore[list-item]
    assert dev["ports"] == [22, 445]
