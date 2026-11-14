"""社区数据集目录（可配置数据源）。

当前内置目录来自 :mod:`mahe.data`。当配置了社区地址后，会尝试从
``<base>/catalog.json`` 拉取目录并合并到内置列表；拉取失败时静默回退，
不影响离线使用。

约定的目录 JSON 格式（两种都支持）::

    ["<数据集描述符>", ...]
    {"datasets": ["<数据集描述符>", ...]}

描述符字段与 :mod:`mahe.data` 一致，生成器必须是本地已注册的类型，
否则该条目会被跳过（服务端无法安全执行未知生成器）。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from . import data

log = logging.getLogger("mahe.community")

CATALOG_TIMEOUT = 4.0


def community_url() -> str:
    return (os.environ.get("MAHE_COMMUNITY_URL") or "").strip().rstrip("/")


def _validate(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        spec = data.normalize(raw)
    except (ValueError, TypeError, KeyError):
        return None
    spec["source"] = "community"
    if not str(spec.get("id", "")).startswith("community:"):
        spec["id"] = f"community:{spec.get('id') or spec.get('name')}"
        spec["name"] = raw.get("name") or spec["id"]
    if raw.get("author"):
        spec["author"] = str(raw["author"])
    return spec


def fetch_catalog(url: str | None = None,
                  timeout: float = CATALOG_TIMEOUT) -> list[dict[str, Any]]:
    base = (url or community_url()).rstrip("/")
    if not base:
        return []
    target = base if base.endswith(".json") else f"{base}/catalog.json"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.info("社区目录不可用（%s）：%s", target, exc)
        return []

    items = payload.get("datasets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    out = []
    for raw in items:
        spec = _validate(raw)
        if spec:
            out.append(spec)
    return out


def catalog(url: str | None = None) -> dict[str, Any]:
    """合并「内置 + 社区」目录；返回值可直接下发给前端。"""
    base = url if url is not None else community_url()
    datasets = data.list_builtin()
    seen = {d["id"] for d in datasets}
    remote = fetch_catalog(base) if base else []
    for spec in remote:
        if spec["id"] not in seen:
            datasets.append(spec)
            seen.add(spec["id"])
    return {
        "datasets": datasets,
        "community": {
            "configured": bool(base),
            "url": base if base else "",
            "count": len(remote),
        },
    }
