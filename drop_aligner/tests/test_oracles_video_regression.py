from __future__ import annotations

from pathlib import Path

import pytest

from drop_aligner.visual_first import VISUAL_FIRST_PRODUCTION_SAMPLE_RATE, visual_first_marker


def test_oracles_video_manual_launch_point_regression() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/The Widdler - Oracles/"
        "drums_140_4A_6-The Widdler - Oracles.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    # use_cache=False: the canary must exercise the live analysis path — a
    # stale feature/tile cache must never be able to fake a pass here.
    result = visual_first_marker(
        audio_path,
        sample_rate=VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=False,
    )
    selected = result["selected_candidate"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(54.857142857142854, abs=0.003)
    assert selected["selected_by"] == "visual_gui_first_fat_block"
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert selected["bpm_clock"]["clock_zero_sec"] == pytest.approx(
        result["feature_map"]["beatgrid"]["bar_zero_sec"],
        abs=1e-9,
    )
    assert selected["bpm_clock"].get("source") not in {
        "gui_boom_front_edge_calibrated_grid",
        "visual_boom_grid_phase_calibration",
        "visual_boom_phase_calibrated",
    }
