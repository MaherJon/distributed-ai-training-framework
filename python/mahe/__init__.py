"""MAHE 分布式 AI 训练框架 · Python 控制面与训练核心"""

__version__ = "0.1.0"

try:
    from . import mahe_core  # noqa: F401  C++ 扩展
    HAS_CORE = True
    CORE_ERROR: ImportError | None = None
except ImportError as exc:  # 扩展未构建或加载失败时允许导入纯 Python 部分
    mahe_core = None  # type: ignore
    HAS_CORE = False
    CORE_ERROR = exc
