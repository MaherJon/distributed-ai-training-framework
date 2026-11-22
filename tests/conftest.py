"""测试入口：把 ``python/`` 加入 sys.path，允许直接 import mahe。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "python"
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))
