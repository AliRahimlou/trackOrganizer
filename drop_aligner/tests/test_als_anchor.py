from __future__ import annotations

from pathlib import Path

import pytest

import drop_aligner.als_anchor as als_anchor_module
from drop_aligner.als_anchor import build_visual_first_als_anchor


def _visual_result(*, marker: float = 2.025, clock_source: str = "gui_boom_front_edge_track_grid"):
    return {
        "ok": True,
        "final_contract_failed": False,
        "marker": float(marker),
        "feature_map": {
            "beatgrid": {
                "bpm": 120.0,
                "beat_sec": 0.5,
                "bar_sec": 2.0,
                "bar_zero_sec": 0.0,
                "source_role": "drums:dj_drum_pattern",
            }
        },
        "selected_candidate": {
            "selected_by": "visual_gui_first_fat_block",
            "bpm_clock": {"source": clock_source},
        },
        "visual_audit": {"status": "pass", "flag_codes": []},
        "boom_proof": {"passes": True},
        "gui_mask_proof": {
            "passes": True,
            "reasons": [],
            "marker_signal_present": True,
            "marker_relevant_mask": True,
            "marker_immediate_body_present": True,
        },
    }


def _refined_attack(_path: str, downbeat: float, _bpm: float):
    sample_rate = 8000
    grid_sample = int(round(float(downbeat) * sample_rate))
    impact_sample = grid_sample + 200
    return {
        "ok": True,
        "refined": True,
        "reason": "bounded_attack_found",
        "sample_rate": sample_rate,
        "downbeat_sample": grid_sample,
        "downbeat_time_sec": float(downbeat),
        "attack_sample": impact_sample,
        "attack_time_sec": impact_sample / sample_rate,
        "attack_confidence": 0.90,
        "boom_body_crest": {
            "ok": True,
            "diagnostic_only": True,
            "moves_marker": False,
            "crest_sample": impact_sample + 120,
            "crest_time_sec": (impact_sample + 120) / sample_rate,
            "crest_offset_ms": 15.0,
            "crest_rms": 0.8,
            "crest_to_pre_db": 18.0,
        },
    }


def test_builds_sample_native_anchor_from_independent_one_and_attack() -> None:
    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(),
        impact_refiner=_refined_attack,
    )

    assert result["ok"] is True
    assert result["drop_sec"] == pytest.approx(2.025)
    assert result["grid_downbeat_sec"] == pytest.approx(2.0)
    assert result["grid_downbeat_sample"] == 16000
    assert result["impact_sample"] == 16200
    assert result["phase_translation_samples"] == 200
    assert result["beat_position"] == 1
    assert result["drop_sec"] != result["boom_body_crest"]["crest_time_sec"]
    assert result["boom_body_crest"]["crest_sample"] == 16320
    assert result["attack"]["boom_body_crest"] == result["boom_body_crest"]
    assert result["evidence"]["bounded_body_crest_measured"] is True


def test_rejects_visual_marker_on_beat_three() -> None:
    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(marker=3.0),
        impact_refiner=_refined_attack,
    )

    assert result["ok"] is False
    assert result["reason"] == "visual_marker_not_on_independent_one"


def test_rejects_a_candidate_that_rephased_its_own_clock() -> None:
    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(clock_source="gui_boom_front_edge_calibrated_grid"),
        impact_refiner=_refined_attack,
    )

    assert result["ok"] is False
    assert result["reason"] == "visual_candidate_self_calibrated_grid"


def test_rejects_when_bounded_attack_evidence_is_missing() -> None:
    def no_attack(_path: str, _downbeat: float, _bpm: float):
        return {"ok": True, "refined": False, "reason": "insufficient_attack_evidence"}

    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(),
        impact_refiner=no_attack,
    )

    assert result["ok"] is False
    assert result["reason"] == "bounded_attack_not_proven"


def test_rejects_wide_exact_gui_front_edge_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        als_anchor_module,
        "_gui_mask_proof_needs_front_edge_repair",
        lambda *_args, **_kwargs: True,
    )

    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(),
        impact_refiner=_refined_attack,
    )

    assert result["ok"] is False
    assert result["reason"] == "wide_exact_gui_front_edge_mismatch"


def test_one_sample_rounding_at_negative_window_edge_is_not_rejected() -> None:
    def quantized_edge(_path: str, downbeat: float, _bpm: float):
        sample_rate = 44100
        grid_sample = int(round(float(downbeat) * sample_rate))
        impact_sample = grid_sample - 353
        return {
            "ok": True,
            "refined": True,
            "reason": "bounded_attack_found",
            "sample_rate": sample_rate,
            "downbeat_sample": grid_sample,
            "attack_sample": impact_sample,
            "attack_time_sec": impact_sample / sample_rate,
            "attack_confidence": 0.9,
        }

    result = build_visual_first_als_anchor(
        "drums_120_test.wav",
        120.0,
        visual_result=_visual_result(marker=2.0),
        impact_refiner=quantized_edge,
    )

    assert result["ok"] is True
    assert result["impact_delta_ms"] < -8.0
    assert result["impact_delta_ms"] >= -8.0 - (1000.0 / 44100.0)


def test_oracles_screen_recording_truth_reaches_the_als_sample_anchor() -> None:
    audio_path = Path(
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/The Widdler - Oracles/"
        "drums_140_4A_6-The Widdler - Oracles.flac"
    )
    if not audio_path.exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = build_visual_first_als_anchor(str(audio_path), 140.0, use_cache=True)

    assert result["ok"] is True
    assert result["grid_downbeat_sec"] == pytest.approx(54.857142857142854, abs=0.001)
    assert result["drop_sec"] == pytest.approx(54.857710, abs=0.001)
    assert result["impact_delta_ms"] == pytest.approx(0.567, abs=0.5)
    assert result["phase_translation_samples"] == 25
    assert result["boom_body_crest"]["ok"] is True
    assert result["boom_body_crest"]["moves_marker"] is False
    assert result["boom_body_crest"]["crest_offset_ms"] == pytest.approx(57.64, abs=2.0)
    assert result["boom_body_crest"]["crest_to_pre_db"] > 12.0
    assert result["drop_sec"] == result["attack"]["attack_time_sec"]


def test_baboom_production_rate_keeps_first_drop_for_als_anchor() -> None:
    audio_path = Path(
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/9A/DECAP - Baboom/"
        "drums_105_9A_5-DECAP - Baboom.flac"
    )
    if not audio_path.exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = build_visual_first_als_anchor(
        str(audio_path),
        105.0,
        sample_rate=44100,
        use_cache=True,
    )

    assert result["ok"] is True
    assert result["grid_downbeat_sec"] == pytest.approx(36.57142857142857, abs=0.001)
    assert result["visual_marker_sec"] == pytest.approx(36.57142857142857, abs=0.001)
    assert result["drop_sec"] == pytest.approx(36.583039, abs=0.001)
    assert result["impact_delta_ms"] == pytest.approx(11.610, abs=0.5)
    assert result["phase_translation_samples"] == 512
    assert result["boom_body_crest"]["ok"] is True
    assert result["boom_body_crest"]["moves_marker"] is False
    assert result["boom_body_crest"]["crest_offset_ms"] == pytest.approx(14.4, abs=2.0)
    assert result["boom_body_crest"]["crest_to_pre_db"] > 12.0
    assert result["drop_sec"] == result["attack"]["attack_time_sec"]
