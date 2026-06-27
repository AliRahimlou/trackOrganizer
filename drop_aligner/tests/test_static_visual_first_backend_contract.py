from __future__ import annotations

from pathlib import Path


VISUAL_FIRST_PY = Path(__file__).resolve().parents[1] / "visual_first.py"


def test_visual_first_marker_does_not_call_disabled_visual_drop_v2_fallback() -> None:
    source = VISUAL_FIRST_PY.read_text(encoding="utf-8")
    marker_source = source[source.index("def visual_first_marker(") :]

    assert "visual_drop_v2_marker(" not in marker_source
    assert "from .visual_drop_v2 import visual_drop_v2_marker" not in marker_source
    assert "visual_drop_v2_fallback_disabled_for_visual_first" not in source
