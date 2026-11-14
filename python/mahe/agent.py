"""DeepSeek Agent 预留接口。

目标：为后续「AI 助手带新手训练」留出稳定的协议与入口，本期只做
最小可用实现——未配置密钥时不发起任何网络请求，界面显示待启用。

启用方式（后续）：设置环境变量 ``DEEPSEEK_API_KEY`` 后重启引擎。
可选：``MAHE_DEEPSEEK_MODEL`` 覆盖模型名，``MAHE_DEEPSEEK_BASE`` 覆盖服务地址。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("mahe.agent")

DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
KEY_ENV = "DEEPSEEK_API_KEY"
TIMEOUT = 60.0

SYSTEM_PROMPT = (
    "你是 MAHE 分布式训练终端的中文助手。用户多为新手，请用简短、可执行的"
    "中文回答：先给结论，再给 1-3 步操作。不要编造 MAHE 没有的功能。"
    "你了解的概念：轮数(rounds)=参数聚合次数；每轮步数(k_steps)=每个节点"
    "本地训练的 mini-batch 数；等效轮数(epoch)=全局处理样本数/数据集大小，"
    "1~50 为正常区间。模型默认保存为 .npz 文件。"
)


class AgentUnavailable(RuntimeError):
    """未配置密钥或服务不可用。"""


def _api_key() -> str:
    return (os.environ.get(KEY_ENV) or "").strip()


def status() -> dict[str, Any]:
    """给前端的状态描述；不含任何密钥内容。"""
    key = _api_key()
    return {
        "enabled": bool(key),
        "provider": "deepseek",
        "model": os.environ.get("MAHE_DEEPSEEK_MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get("MAHE_DEEPSEEK_BASE", DEFAULT_BASE),
        "hint": ("已配置密钥，可直接对话"
                 if key else
                 f"设置环境变量 {KEY_ENV} 并重启引擎后启用"),
    }


def ask(text: str, history: list[dict[str, str]] | None = None) -> str:
    """向 DeepSeek 发起一次对话，返回纯文本答复。"""
    key = _api_key()
    if not key:
        raise AgentUnavailable(f"未配置 {KEY_ENV}，智能助手暂不可用")
    text = (text or "").strip()
    if not text:
        raise AgentUnavailable("请输入问题")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})

    base = os.environ.get("MAHE_DEEPSEEK_BASE", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("MAHE_DEEPSEEK_MODEL", DEFAULT_MODEL)
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0.3, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise AgentUnavailable(f"服务返回 {exc.code}：{detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AgentUnavailable(f"请求失败：{exc}") from exc

    try:
        return str(payload["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentUnavailable(f"响应格式异常：{str(payload)[:200]}") from exc
