from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rhythm_engine.engine import analyze_rhythm
import rhythm_engine.engine as engine_module
from rhythm_engine.types import RhythmEngineConfig
from rhythm_engine.types import RhythmEstimate


def test_engine_returns_unavailable_provider_status_without_failing(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    wavfile.write(path, 8000, np.zeros(8000, dtype=np.float32))
    cfg = RhythmEngineConfig(providers=("does_not_exist",), micro_refine=False)

    result = analyze_rhythm(str(path), config=cfg)

    assert result.providers[0].status == "unavailable"
    assert result.fused.status == "failed"
    assert result.final.status == "failed"
    assert result.quality is not None
    assert result.quality["tier"] == "LOW"


def test_engine_evaluates_each_provider_and_hypothesis(monkeypatch) -> None:
    def fake_run_providers(audio_path: str, cfg: RhythmEngineConfig):
        good_beats = tuple(1.0 + (0.5 * idx) for idx in range(8))
        late_beats = tuple(t + 0.04 for t in good_beats)
        return [
            RhythmEstimate(provider="good", beats=good_beats, downbeats=(1.0, 3.0), bpm=120.0, confidence=0.9),
            RhythmEstimate(provider="late", beats=late_beats, downbeats=(1.04, 3.04), bpm=120.0, confidence=0.8),
        ]

    monkeypatch.setattr(engine_module, "run_providers", fake_run_providers)

    result = analyze_rhythm(
        "dummy.wav",
        config=RhythmEngineConfig(providers=("good", "late"), micro_refine=False, use_hypothesis_selector=False),
        reference_beats=tuple(1.0 + (0.5 * idx) for idx in range(8)),
        reference_downbeats=(1.0, 3.0),
    )

    assert result.evaluation is not None
    assert result.evaluation["providers"]["good"]["median_abs_error_ms"] == 0.0
    assert result.evaluation["providers"]["late"]["median_abs_error_ms"] > 30.0
    assert "good:base" in result.evaluation["hypotheses"]
    assert "good:phase0" in result.evaluation["hypothesis_downbeats"]
