from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from drop_aligner.microalign import choose_microaligned_candidate, microalign_candidate_dicts, microalign_marker


def _write_wav(y: np.ndarray, sr: int) -> str:
    td = tempfile.TemporaryDirectory()
    path = Path(td.name) / "drums_micro_test.wav"
    wavfile.write(path, sr, np.clip(y, -0.98, 0.98).astype(np.float32))
    # Keep the temp directory alive by attaching it to the returned function object path.
    _TEMP_DIRS.append(td)
    return str(path)


_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _kick(sr: int, seconds: float, attack: float, *, pre_click: bool = False, riser: bool = False) -> np.ndarray:
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    if riser:
        start = int(max(0.0, attack - 0.35) * sr)
        end = int(attack * sr)
        n = max(1, end - start)
        y[start:end] += (np.linspace(0.0, 0.18, n) * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    if pre_click:
        click = int((attack - 0.040) * sr)
        if 0 <= click < len(y):
            y[click : click + 12] += 0.12
    start = int(attack * sr)
    n = int(0.110 * sr)
    env = np.exp(-np.linspace(0, 7, n))
    wave = np.sin(2 * np.pi * 75 * np.arange(n) / sr) * env
    if 0 <= start < len(y) - n:
        y[start : start + n] += wave.astype(np.float32)
    return y


def _noisy_buildup_to_kick(sr: int, seconds: float, attack: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    y = (0.018 * rng.standard_normal(int(sr * seconds))).astype(np.float32)
    start = int(max(0.0, attack - 0.42) * sr)
    end = int(attack * sr)
    n = max(1, end - start)
    phase = np.cumsum(np.linspace(180.0, 900.0, n)) / sr
    y[start:end] += (np.linspace(0.0, 0.20, n) * np.sin(2 * np.pi * phase)).astype(np.float32)
    for offset, amp in [(-0.090, 0.10), (-0.052, 0.14), (-0.026, 0.11)]:
        click = int((attack + offset) * sr)
        if 0 <= click < len(y) - 18:
            y[click : click + 18] += (amp * np.hanning(18)).astype(np.float32)
    start = int(attack * sr)
    n = int(0.120 * sr)
    env = np.exp(-np.linspace(0, 7, n))
    wave = 0.90 * np.sin(2 * np.pi * 72 * np.arange(n) / sr) * env
    if 0 <= start < len(y) - n:
        y[start : start + n] += wave.astype(np.float32)
    return y


def _loud_click_before_kick(sr: int, seconds: float, attack: float) -> np.ndarray:
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    click = int((attack - 0.055) * sr)
    if 0 <= click < len(y) - 36:
        y[click : click + 36] += (0.55 * np.hanning(36)).astype(np.float32)
    start = int(attack * sr)
    n = int(0.120 * sr)
    env = np.exp(-np.linspace(0, 7, n))
    wave = np.sin(2 * np.pi * 75 * np.arange(n) / sr) * env
    if 0 <= start < len(y) - n:
        y[start : start + n] += wave.astype(np.float32)
    return y


def _delayed_lobe_after_centerline(sr: int, seconds: float, boundary: float) -> np.ndarray:
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    tail_start = int((boundary - 0.020) * sr)
    tail_end = int(boundary * sr)
    if 0 <= tail_start < tail_end <= len(y):
        n = tail_end - tail_start
        y[tail_start:tail_end] += (0.006 * np.sin(2 * np.pi * 180 * np.arange(n) / sr)).astype(np.float32)
    start = int((boundary + 0.005) * sr)
    n = int(0.090 * sr)
    env = np.exp(-np.linspace(0, 5, n))
    wave = np.sin(2 * np.pi * 75 * np.arange(n) / sr) * env
    if 0 <= start < len(y) - n:
        y[start : start + n] += wave.astype(np.float32)
    return y


def _tail_before_hard_drop(sr: int, seconds: float, boundary: float, impact: float) -> np.ndarray:
    rng = np.random.default_rng(11)
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    tail_start = int((boundary - 0.080) * sr)
    tail_end = int(impact * sr)
    if 0 <= tail_start < tail_end <= len(y):
        n = tail_end - tail_start
        t = np.arange(n) / sr
        tail_env = np.linspace(0.22, 0.07, n)
        y[tail_start:tail_end] += (
            (tail_env * np.sin(2 * np.pi * 480 * t)) + (0.018 * rng.standard_normal(n))
        ).astype(np.float32)
    pre_hit = int((impact - 0.006) * sr)
    if 0 <= pre_hit < len(y) - 80:
        y[pre_hit : pre_hit + 80] += (0.12 * np.hanning(80)).astype(np.float32)
    start = int(impact * sr)
    n = int(0.130 * sr)
    kick_env = np.exp(-np.linspace(0, 6, n))
    kick = 0.90 * np.sin(2 * np.pi * 72 * np.arange(n) / sr) * kick_env
    bass = 0.55 * np.sin(2 * np.pi * 47 * np.arange(n) / sr) * np.exp(-np.linspace(0, 3.4, n))
    if 0 <= start < len(y) - n:
        y[start : start + n] += (kick + bass).astype(np.float32)
    return y


def test_microalign_snaps_late_candidate_back_to_kick_attack() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_kick(sr, 2.0, attack), sr)
    result = microalign_marker(path, attack + 0.045)
    assert abs(result["microaligned_time"] - attack) < 0.012
    assert result["snap_offset_ms"] < -25.0


def test_microalign_ignores_tiny_pre_click_before_kick() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_kick(sr, 2.0, attack, pre_click=True), sr)
    result = microalign_marker(path, attack + 0.030)
    assert result["microaligned_time"] > attack - 0.018
    assert abs(result["microaligned_time"] - attack) < 0.018


def test_microalign_does_not_snap_to_riser_tail() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_kick(sr, 2.0, attack, riser=True), sr)
    result = microalign_marker(path, attack + 0.025)
    assert result["microaligned_time"] > attack - 0.025
    assert abs(result["microaligned_time"] - attack) < 0.020


def test_microalign_filters_noisy_buildup_and_strengthens_real_impact() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_noisy_buildup_to_kick(sr, 2.0, attack), sr)
    result = microalign_marker(path, attack + 0.038)
    assert abs(result["microaligned_time"] - attack) < 0.006
    assert result["denoised_impact_strength"] > 0.80
    assert result["impact_contrast"] > 0.80
    assert result["impact_boundary_confidence"] > 0.88


def test_microalign_rejects_loud_pre_hit_click_as_boundary() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_loud_click_before_kick(sr, 2.0, attack), sr)
    result = microalign_marker(path, attack + 0.030)
    assert result["microaligned_time"] > attack - 0.020
    assert abs(result["microaligned_time"] - attack) < 0.014


def test_microalign_reports_rms_and_percentile_peak_rise_at_hit() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_noisy_buildup_to_kick(sr, 2.0, attack), sr)
    result = microalign_marker(path, attack + 0.038)
    assert abs(result["microaligned_time"] - attack) < 0.006
    assert result["rms_rise_score"] > 0.40
    assert result["peak_rise_score"] > 0.40
    assert not (result["crest_click_score"] > 0.75 and result["rms_rise_score"] < 0.30)


def test_microalign_keeps_correct_marker_close() -> None:
    sr = 44100
    attack = 1.0
    path = _write_wav(_kick(sr, 2.0, attack), sr)
    result = microalign_marker(path, attack)
    assert abs(result["snap_offset_ms"]) < 12.0


def test_microalign_keeps_input_boundary_when_visual_markers_bracket_real_stem() -> None:
    path = Path("/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/smith. - Crisp/drums_140_4A_6-smith. - Crisp.flac")
    if not path.exists():
        pytest.skip("local review stem is not available")
    boundary = 27.428571428571427
    result = microalign_marker(str(path), boundary)
    assert result["input_boundary_used"] == 1.0
    assert result["impact_body_used"] == 0.0
    assert abs(result["microaligned_time"] - boundary) < 0.0001
    assert abs(result["snap_offset_ms"]) < 0.1


def test_microalign_prefers_visual_knee_before_big_lobe() -> None:
    sr = 44100
    boundary = 1.0
    lobe_start = boundary + 0.005
    path = _write_wav(_delayed_lobe_after_centerline(sr, 2.0, boundary), sr)
    result = microalign_marker(path, boundary)
    assert result["microaligned_time"] < lobe_start
    assert abs(result["microaligned_time"] - boundary) < 0.002
    assert result["centerline_boundary_used"] is True
    assert result["visual_onset_knee_used"] == 0.0
    assert result["visual_onset_knee_quality"] > 0.80


def test_microalign_keeps_first_centerline_departure_before_hard_body() -> None:
    sr = 44100
    boundary = 1.0
    impact = boundary + 0.014
    path = _write_wav(_tail_before_hard_drop(sr, 2.0, boundary, impact), sr)
    result = microalign_marker(path, boundary)
    assert result["microaligned_time"] < impact
    assert boundary - 0.090 < result["microaligned_time"] < boundary - 0.040
    assert result["centerline_boundary_used"] is True or result["visual_onset_knee_used"] == 1.0
    assert result["impact_body_used"] == 0.0
    assert result["tail_bypass_ms"] == 0.0


def test_microalign_keeps_asd_reference_but_prefers_first_centerline() -> None:
    sr = 44100
    boundary = 1.0
    lobe_start = boundary + 0.005
    asd_marker = boundary + 0.0038
    path = _write_wav(_delayed_lobe_after_centerline(sr, 2.0, boundary), sr)
    result = microalign_marker(path, boundary, asd_marker_times=[asd_marker])
    assert result["microaligned_time"] < lobe_start
    assert abs(result["microaligned_time"] - boundary) < 0.002
    assert abs(result["ableton_asd_time"] - asd_marker) < 0.001
    assert result["ableton_asd_used"] == 0.0
    assert result["centerline_boundary_used"] is True
    assert result["visual_onset_knee_used"] == 0.0
    assert result["ableton_asd_quality"] > 0.80


def test_existing_microalign_keeps_strong_visual_onset_over_clock_lock() -> None:
    candidate = {
        "timestamp": 1.0,
        "bpm_clock": {"on_one": True, "nearest_one_time": 1.0},
        "microalign": {
            "ok": True,
            "microaligned_time": 1.080,
            "snap_offset_ms": 80.0,
            "micro_confidence": 0.78,
            "centerline_boundary_used": True,
            "centerline_boundary_quality": 0.86,
            "impact_boundary_confidence": 0.82,
        },
    }

    [result] = microalign_candidate_dicts("unused.wav", [candidate], limit=1)
    micro = result["microalign"]

    assert micro["microaligned_time"] == 1.080
    assert micro["clock_lock_skipped"] is True
    assert "clock_locked" not in micro


def test_auto_place_returns_review_suggestion_for_low_confidence_tracks() -> None:
    candidates = [
        {
            "rank": 1,
            "timestamp": 61.46,
            "confidence_score": 0.87,
            "microalign": {
                "ok": True,
                "microaligned_time": 61.4602,
                "micro_confidence": 0.89,
                "snap_offset_ms": 0.5,
            },
        },
        {
            "rank": 3,
            "timestamp": 60.98,
            "confidence_score": 0.91,
            "microalign": {
                "ok": True,
                "microaligned_time": 60.9851,
                "micro_confidence": 0.97,
                "snap_offset_ms": 4.7,
            },
        },
    ]

    result = choose_microaligned_candidate(
        candidates,
        confidence_tier="LOW",
        mode="normal",
        chooser_model_path="/tmp/nonexistent-drop-candidate-chooser.pkl",
    )

    assert result["auto_place"] is False
    assert result["review_needed"] is True
    assert result["suggested_time"] == 60.9851
    assert result["candidate"]["rank"] == 3
