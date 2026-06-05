from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from drop_aligner.detector import DropDetectorConfig, detect_drop, extract_features, score_candidates


_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _write_wav(y: np.ndarray, sr: int) -> str:
    td = tempfile.TemporaryDirectory()
    path = Path(td.name) / "drums_full_groove_test.wav"
    wavfile.write(path, sr, np.clip(y, -0.98, 0.98).astype(np.float32))
    _TEMP_DIRS.append(td)
    return str(path)


def _add_kick(y: np.ndarray, sr: int, t: float, *, amp: float = 0.85) -> None:
    start = int(round(t * sr))
    n = int(round(0.105 * sr))
    if start < 0 or start + n >= len(y):
        return
    env = np.exp(-np.linspace(0.0, 7.0, n))
    wave = np.sin(2 * np.pi * 72.0 * np.arange(n) / sr) * env
    y[start : start + n] += (wave * amp).astype(np.float32)


def _add_snare(y: np.ndarray, sr: int, t: float, *, amp: float = 0.35) -> None:
    start = int(round(t * sr))
    n = int(round(0.050 * sr))
    if start < 0 or start + n >= len(y):
        return
    rng = np.random.default_rng(5100 + int(round(t * 1000)))
    env = np.exp(-np.linspace(0.0, 5.0, n))
    y[start : start + n] += (rng.normal(0.0, amp, n) * env).astype(np.float32)


def _fake_hit_then_sustained_groove() -> tuple[np.ndarray, int, float]:
    sr = 22050
    bpm = 120.0
    beat = 60.0 / bpm
    y = np.zeros(int(sr * 14.0), dtype=np.float32)

    _add_kick(y, sr, 3.0, amp=1.0)
    for beat_index in range(20):
        t = 4.0 + (beat_index * beat)
        _add_kick(y, sr, t, amp=0.86)
        if beat_index % 2 == 1:
            _add_snare(y, sr, t, amp=0.42)
        _add_snare(y, sr, t + (0.5 * beat), amp=0.16)

    return np.clip(y, -1.0, 1.0), sr, bpm


def _buildup_then_true_drop() -> tuple[np.ndarray, int, float]:
    sr = 22050
    bpm = 120.0
    beat = 60.0 / bpm
    y = np.zeros(int(sr * 18.0), dtype=np.float32)

    _add_kick(y, sr, 3.0, amp=1.0)
    for step in range(16):
        t = 4.0 + (step * 0.25)
        _add_snare(y, sr, t, amp=0.04 + (0.018 * step))

    for beat_index in range(24):
        t = 8.0 + (beat_index * beat)
        _add_kick(y, sr, t, amp=0.86)
        if beat_index % 2 == 1:
            _add_snare(y, sr, t, amp=0.38)
        _add_snare(y, sr, t + (0.5 * beat), amp=0.14)

    return np.clip(y, -1.0, 1.0), sr, bpm


def test_full_groove_layer_prefers_immediate_sustained_groove() -> None:
    y, sr, bpm = _fake_hit_then_sustained_groove()
    path = _write_wav(y, sr)
    cfg = DropDetectorConfig(
        sample_rate=sr,
        hpss=False,
        use_ranker_model=False,
        use_drumprint=False,
        min_drop_time_sec=1.0,
        max_drop_time_ratio=0.95,
    )
    features = extract_features(path, cfg, bpm=bpm)
    candidates = score_candidates(features, [3.0, 4.0], cfg)
    fake = min(candidates, key=lambda candidate: abs(candidate.time_sec - 3.0))
    drop = min(candidates, key=lambda candidate: abs(candidate.time_sec - 4.0))

    assert drop.full_groove["sustained_full_groove_score"] > fake.full_groove["sustained_full_groove_score"]
    assert drop.full_groove["immediate_groove_start_score"] > fake.full_groove["immediate_groove_start_score"]
    assert drop.score > fake.score


def test_edm_transition_prior_ignores_buildup_hits_before_kick_reentry() -> None:
    y, sr, bpm = _buildup_then_true_drop()
    path = _write_wav(y, sr)
    cfg = DropDetectorConfig(
        sample_rate=sr,
        hpss=False,
        use_ranker_model=False,
        use_region_model=False,
        use_drumprint=False,
        min_drop_time_sec=1.0,
        max_drop_time_ratio=0.95,
        candidate_prominence=0.05,
    )

    result = detect_drop(path, bpm=bpm, config=cfg)
    chosen = result.selected_candidate_dict()

    assert 7.90 <= result.drop_sec <= 8.10
    assert chosen is not None
    assert chosen["full_groove"]["kick_reentry_score"] >= 0.90
    assert chosen["full_groove"]["buildup_drop_score"] >= 0.40
