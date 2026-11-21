"""前端静态资源一致性：防止 app.js 引用了 index.html 里不存在的元素。"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "python" / "mahe" / "webui"


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_every_js_id_selector_exists_in_html():
    js = _read("app.js")
    html = _read("index.html")
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', js))
    defined = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(used - defined)
    assert not missing, f"app.js 引用了 HTML 中不存在的 id: {missing}"


def test_every_nav_target_has_a_view():
    html = _read("index.html")
    targets = set(re.findall(r'data-view="([A-Za-z0-9_-]+)"', html))
    views = set(re.findall(r'id="view-([A-Za-z0-9_-]+)"', html))
    assert targets, "导航菜单没有 data-view"
    assert targets == views, f"nav={sorted(targets)} views={sorted(views)}"


def test_dashboard_quick_actions_are_in_first_screen_markup():
    html = _read("index.html")
    dashboard = html.split('id="view-dashboard"', 1)[1].split("</section>", 1)[0]
    action_pos = dashboard.find('class="panel action-bar"')
    grid_pos = dashboard.find('class="grid"')
    assert action_pos != -1, "总览缺少操作条 action-bar"
    assert action_pos < grid_pos, "操作条必须排在图表/节点之前"


def test_deploy_view_has_device_filters():
    html = _read("index.html")
    for element_id in ("show-all", "device-search", "device-body", "check-all"):
        assert f'id="{element_id}"' in html, element_id


def test_refresh_guard_present_in_js():
    js = _read("app.js")
    # 刷新时先查后端是否已就绪，避免重复播放 init
    assert 'fetch("/api/status"' in js
    assert "snap.ui_ready" in js
    assert "sessionStorage" in js
