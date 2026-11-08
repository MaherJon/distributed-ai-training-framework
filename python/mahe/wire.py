"""任务/梯度负载的二级封装：8B 头(json_len, blob_len) + json + 原始字节。

C++ 帧协议只负责传输，本模块定义负载内部布局（v1，仅 Python 侧使用）：
  [0..3] json_len u32le
  [4..7] blob_len u32le
  ...    json (UTF-8)
  ...    blob (float32 小端数组等)
"""

from __future__ import annotations

import json
import struct

import numpy as np

_HEADER = struct.Struct("<II")


def pack(meta: dict, blob: bytes = b"") -> bytes:
    head = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(head), len(blob)) + head + blob


def unpack(payload: bytes) -> tuple[dict, bytes]:
    if len(payload) < _HEADER.size:
        raise ValueError("payload too short")
    json_len, blob_len = _HEADER.unpack_from(payload, 0)
    end = _HEADER.size + json_len
    meta = json.loads(payload[_HEADER.size:end].decode("utf-8"))
    blob = payload[end:end + blob_len]
    return meta, blob


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()
