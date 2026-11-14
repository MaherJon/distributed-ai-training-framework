"""PyInstaller 入口：把 Python 控制面 + C++ 扩展冻结为 mahe-engine。

打包后（sys.frozen）此文件即引擎主程序，参数与 ``python -m mahe.demo_train`` 一致：
  mahe-engine                      # 干净启动（Master + Web 终端）
  mahe-engine --test               # 开发者自测（2 个本地 Worker + 自动训练）
  mahe-engine --role worker ...    # 子端
"""

import os
import sys

# 未冻结时（直接 python packaging/engine_entry.py）补充包路径，便于本地调试
_here = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(_here, "..", "python"))

from mahe.demo_train import main  # noqa: E402

if __name__ == "__main__":
    main()
