from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rhythm_engine.micro_refine import refine_estimate_to_attacks
from rhythm_engine.types import RhythmEngineConfig, RhythmEstimate


def _write_click_track(path: Path, *, sr: int, beats: list[float]) -> None:
    y = np.zeros(int(sr * 3.0), dtype=np.float32)
    for beat in beats:
        start = int(round(float(beat) * sr))
        n = int(round(0.030 * sr))
        if start < 0 or start + n >= len(y):
            continue
        env = np.exp(-np.linspace(0.0, 8.0, n))
        tone = np.sin(2.0 * np.pi * 880.0 * np.arange(n) / sr)
        y[start : start + n] += (tone * env * 0.9).astype(np.float32)
    wavfile.write(path, sr, np.clip(y, -0.98, 0.98).astype(np.float32))


def test_micro_refine_moves_grid_points_to_sample_attack(tmp_path: Path) -> None:
    path = tmp_path / "clicks.wav"
    reference = [0.50, 1.00, 1.50, 2.00]
    _write_click_track(path, sr=44100, beats=reference)
    estimate = RhythmEstimate(
        provider="test",
        beats=tuple(t + 0.018 for t in reference),
        downbeats=(reference[0] + 0.018,),
        bpm=120.0,
        confidence=0.8,
    )
    cfg = RhythmEngineConfig(
        providers=("test",),
        micro_refine_sample_rate=44100,
        micro_refine_window_ms=30.0,
    )

    refined = refine_estimate_to_attacks(str(path), estimate, cfg)

    assert refined.available
    assert refined.provider == "test+micro_refine"
    assert max(abs(a - b) for a, b in zip(refined.beats, reference)) <= 0.005
    assert abs(refined.downbeats[0] - reference[0]) <= 0.005
    assert refined.metadata["micro_refine_status"] == "ok"


def test_micro_refine_prefers_sibling_drums_stem(tmp_path: Path) -> None:
    inst = tmp_path / "inst_120_1A_Test.wav"
    drums = tmp_path / "drums_120_1A_Test.wav"
    reference = [0.50, 1.00, 1.50, 2.00]
    wavfile.write(inst, 44100, np.zeros(int(44100 * 3.0), dtype=np.float32))
    _write_click_track(drums, sr=44100, beats=reference)
    estimate = RhythmEstimate(
        provider="test",
        beats=tuple(t + 0.018 for t in reference),
        downbeats=(reference[0] + 0.018,),
        bpm=120.0,
        confidence=0.8,
    )
    cfg = RhythmEngineConfig(
        providers=("test",),
        micro_refine_sample_rate=44100,
        micro_refine_window_ms=30.0,
        micro_refine_stem_aware=True,
    )

    refined = refine_estimate_to_attacks(str(inst), estimate, cfg)

    assert max(abs(a - b) for a, b in zip(refined.beats, reference)) <= 0.005
    assert refined.metadata["micro_refine_source"] == "stem:drums"
    assert refined.metadata["micro_refine_audio_path"] == str(drums.resolve())
