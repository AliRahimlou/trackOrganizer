from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from drop_aligner.downbeat_impact import refine_approved_downbeat_attack


def _write_attack(path, *, attack_sec: float, sample_rate: int = 8000) -> None:
    samples = np.zeros((3 * sample_rate, 2), dtype=np.float32)
    attack = int(round(float(attack_sec) * sample_rate))
    body = np.linspace(0.85, 0.55, int(round(0.030 * sample_rate)), dtype=np.float32)
    samples[attack : attack + body.size, 0] = body
    samples[attack : attack + body.size, 1] = -body
    sf.write(str(path), samples, sample_rate, subtype="FLOAT")


def test_refines_only_inside_the_approved_downbeat_window(tmp_path) -> None:
    audio_path = tmp_path / "drums_120_test.wav"
    _write_attack(audio_path, attack_sec=2.025)

    result = refine_approved_downbeat_attack(str(audio_path), 2.0, 120.0)

    assert result["ok"] is True
    assert result["refined"] is True
    assert result["downbeat_sample"] == 16000
    assert result["attack_time_sec"] == pytest.approx(2.025, abs=0.004)
    assert -8.0 <= result["attack_delta_ms"] <= 90.0
    assert result["zero_crossing_sample"] != result["attack_sample"] or result["zero_crossing_time_sec"] == result["attack_time_sec"]
    assert result["boom_body_crest"]["ok"] is True
    assert result["boom_body_crest"]["moves_marker"] is False
    assert result["boom_body_crest"]["crest_sample"] >= result["attack_sample"]
    assert (
        result["boom_body_crest"]["window_end_sample"]
        <= result["boom_body_crest"]["quarter_beat_guard_sample"]
    )


def test_rejects_when_the_only_attack_is_outside_the_one_window(tmp_path) -> None:
    audio_path = tmp_path / "drums_120_late.wav"
    _write_attack(audio_path, attack_sec=2.150)

    result = refine_approved_downbeat_attack(str(audio_path), 2.0, 120.0)

    assert result["ok"] is True
    assert result["refined"] is False
    assert result["reason"] == "insufficient_attack_evidence"
    assert result["search_end_sample"] <= int(round(2.090 * result["sample_rate"]))
    assert result["boom_body_crest"] == {
        "ok": False,
        "reason": "attack_not_refined",
        "diagnostic_only": True,
        "moves_marker": False,
    }


def test_prefers_first_credible_body_over_louder_later_hit(tmp_path) -> None:
    sample_rate = 8000
    audio_path = tmp_path / "drums_120_first_body.wav"
    samples = np.zeros((3 * sample_rate, 2), dtype=np.float32)
    first = int(round(2.010 * sample_rate))
    later = int(round(2.075 * sample_rate))
    samples[first : first + int(0.025 * sample_rate), 0] = 0.36
    samples[first : first + int(0.025 * sample_rate), 1] = -0.36
    samples[later : later + int(0.020 * sample_rate), 0] = 0.95
    samples[later : later + int(0.020 * sample_rate), 1] = -0.95
    sf.write(str(audio_path), samples, sample_rate, subtype="FLOAT")

    result = refine_approved_downbeat_attack(str(audio_path), 2.0, 120.0)

    assert result["refined"] is True
    assert result["attack_time_sec"] == pytest.approx(2.010, abs=0.004)
    assert result["attack_time_sec"] < 2.050
    # The later, louder hit is outside the bounded first-body crest window.
    assert result["boom_body_crest"]["crest_time_sec"] < 2.075


def test_measures_boom_envelope_crest_without_moving_the_attack(tmp_path) -> None:
    sample_rate = 8000
    audio_path = tmp_path / "drums_120_rising_boom.wav"
    samples = np.zeros((3 * sample_rate, 2), dtype=np.float32)
    attack_sample = int(round(2.0 * sample_rate))
    body_samples = int(round(0.055 * sample_rate))
    crest_offset = int(round(0.024 * sample_rate))
    rising = np.linspace(0.24, 0.92, crest_offset + 1, dtype=np.float32)
    falling = np.linspace(0.92, 0.38, body_samples - rising.size, dtype=np.float32)
    envelope = np.concatenate((rising, falling))
    carrier = np.sin(2.0 * np.pi * 70.0 * np.arange(body_samples) / sample_rate).astype(np.float32)
    boom = envelope * carrier
    samples[attack_sample : attack_sample + body_samples, 0] = boom
    samples[attack_sample : attack_sample + body_samples, 1] = -boom
    sf.write(str(audio_path), samples, sample_rate, subtype="FLOAT")

    result = refine_approved_downbeat_attack(str(audio_path), 2.0, 120.0)

    crest = result["boom_body_crest"]
    assert result["refined"] is True
    assert result["attack_time_sec"] == pytest.approx(2.0, abs=0.004)
    assert crest["ok"] is True
    assert crest["diagnostic_only"] is True
    assert crest["moves_marker"] is False
    assert crest["crest_time_sec"] == pytest.approx(2.024, abs=0.006)
    assert 0.0 < crest["crest_offset_ms"] <= 60.0
    assert crest["crest_to_pre_db"] > 12.0
    assert result["attack_sample"] != crest["crest_sample"]
