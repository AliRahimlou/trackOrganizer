from __future__ import annotations

import numpy as np

from drop_aligner.detector import DropCandidate
from drop_aligner.drumprint import DRUMPRINT_FEATURE_KEYS, build_drumprint_analysis, score_candidate_drumprint
from drop_aligner.groove import FULL_GROOVE_FEATURE_KEYS
from drop_aligner.ranker import MODEL_FEATURES, candidate_feature_vector


def _synthetic_fake_hit_then_groove() -> tuple[np.ndarray, int, float]:
    sr = 22050
    bpm = 120.0
    beat = 60.0 / bpm
    y = np.zeros(int(sr * 24.0), dtype=np.float32)

    def add_pulse(t: float, *, freq: float = 80.0, dur: float = 0.08, amp: float = 0.9) -> None:
        start = int(t * sr)
        n = int(dur * sr)
        if not (0 <= start < len(y) - n):
            return
        env = np.exp(-np.linspace(0, 6, n))
        y[start : start + n] += (np.sin(2 * np.pi * freq * np.arange(n) / sr) * env * amp).astype(np.float32)

    def add_noise(t: float, *, dur: float = 0.04, amp: float = 0.35) -> None:
        start = int(t * sr)
        n = int(dur * sr)
        if not (0 <= start < len(y) - n):
            return
        rng = np.random.default_rng(7 + int(t * 1000))
        env = np.exp(-np.linspace(0, 5, n))
        y[start : start + n] += (rng.normal(0, amp, n) * env).astype(np.float32)

    add_pulse(3.5, amp=1.0)
    for section_start in (4.0, 12.0):
        for beat_index in range(16):
            t = section_start + beat_index * beat
            add_pulse(t, amp=0.85)
            if beat_index % 2 == 1:
                add_noise(t, amp=0.45)
            add_noise(t + 0.25, dur=0.025, amp=0.18)

    return np.clip(y, -1.0, 1.0), sr, bpm


def test_drumprint_prefers_sustained_groove_over_fake_hit() -> None:
    y, sr, bpm = _synthetic_fake_hit_then_groove()
    analysis = build_drumprint_analysis(y, sr, bpm)

    fake = score_candidate_drumprint(analysis, 3.5, transient_strength=1.0)
    drop = score_candidate_drumprint(analysis, 4.0, transient_strength=1.0)

    assert drop["drumprint_pattern_score"] > fake["drumprint_pattern_score"] + 0.08
    assert fake["fake_hit_penalty"] > drop["fake_hit_penalty"] + 0.20


def test_candidate_json_contains_drumprint_feature_values() -> None:
    candidate = DropCandidate(
        time_sec=4.0,
        snapped_sec=4.0,
        score=0.8,
        transient_strength=0.9,
        low_end_jump=0.8,
        post_drop_density=0.7,
        pre_post_energy_ratio=1.5,
        energy_contrast=0.6,
        rhythmic_consistency=0.5,
        drumprint={
            "enabled": True,
            "status": "ok",
            "drumprint_pattern_score": 0.75,
            "fake_hit_penalty": 0.05,
        },
        full_groove={
            "sustained_full_groove_score": 0.82,
            "immediate_groove_start_score": 0.91,
            "groove_stability": 0.77,
        },
    )

    payload = candidate.to_dict()

    assert payload["drumprint"]["drumprint_pattern_score"] == 0.75
    assert payload["drumprint_pattern_score"] == 0.75
    assert payload["full_groove"]["sustained_full_groove_score"] == 0.82
    assert payload["sustained_full_groove_score"] == 0.82
    for key in DRUMPRINT_FEATURE_KEYS:
        assert key in payload
    for key in FULL_GROOVE_FEATURE_KEYS:
        assert key in payload


def test_ranker_accepts_old_and_nested_drumprint_features() -> None:
    old_candidate = {"timestamp": 10.0, "score": 0.5}
    new_candidate = {
        "timestamp": 10.0,
        "score": 0.5,
        "drumprint": {
            "drumprint_pattern_score": 0.8,
            "fake_hit_penalty": 0.1,
        },
        "full_groove": {
            "sustained_full_groove_score": 0.7,
            "immediate_groove_start_score": 0.9,
        },
    }

    old_vector = candidate_feature_vector(old_candidate)
    new_vector = candidate_feature_vector(new_candidate)

    assert len(old_vector) == len(MODEL_FEATURES)
    assert len(new_vector) == len(MODEL_FEATURES)
    assert old_vector[MODEL_FEATURES.index("drumprint_pattern_score")] == 0.0
    assert new_vector[MODEL_FEATURES.index("drumprint_pattern_score")] == 0.8
    assert old_vector[MODEL_FEATURES.index("sustained_full_groove_score")] == 0.0
    assert new_vector[MODEL_FEATURES.index("sustained_full_groove_score")] == 0.7
