from __future__ import annotations

from pathlib import Path

import numpy as np

from rhythm_engine import providers
from rhythm_engine.providers import _beatnet_output_to_arrays
from rhythm_engine.types import RhythmEngineConfig, RhythmEstimate


def test_stem_ensemble_fuses_role_specific_provider_estimates(tmp_path: Path, monkeypatch) -> None:
    drums = tmp_path / "drums_120_1A_Test.wav"
    bass = tmp_path / "bass_120_1A_Test.wav"
    vocals = tmp_path / "vocals_120_1A_Test.wav"
    for path in (drums, bass, vocals):
        path.write_bytes(b"stub")

    def fake_track_organizer(path: str, config: RhythmEngineConfig) -> RhythmEstimate:
        role = Path(path).name.split("_", 1)[0]
        offset = {"drums": 0.000, "bass": 0.010, "vocals": 0.030}.get(role, 0.0)
        confidence = {"drums": 0.90, "bass": 0.80, "vocals": 0.55}.get(role, 0.4)
        return RhythmEstimate(
            provider="track_organizer",
            beats=(1.0 + offset, 1.5 + offset, 2.0 + offset),
            downbeats=(1.0 + offset,),
            bpm=120.0,
            confidence=confidence,
            metadata={"fake_role": role},
        )

    def fake_librosa(path: str, config: RhythmEngineConfig) -> RhythmEstimate:
        return RhythmEstimate.unavailable("librosa", "disabled_in_test")

    monkeypatch.setattr(providers, "track_organizer_provider", fake_track_organizer)
    monkeypatch.setattr(providers, "librosa_provider", fake_librosa)

    estimate = providers.stem_ensemble_provider(str(drums), RhythmEngineConfig(providers=("stem_ensemble",)))

    assert estimate.available
    assert estimate.provider == "stem_ensemble"
    assert len(estimate.beats) == 3
    assert abs(estimate.beats[0] - 1.0) < 1e-9
    assert estimate.metadata["stem_roles"] == ["bass", "drums", "vocals"]
    assert any(row["provider"].startswith("stem_ensemble:drums") for row in estimate.metadata["role_estimates"])


def test_beatnet_output_parser_extracts_downbeat_labels() -> None:
    beats, downbeats = _beatnet_output_to_arrays(
        np.asarray(
            [
                [0.50, 1.0],
                [1.00, 2.0],
                [1.50, 2.0],
                [2.00, 1.0],
            ],
            dtype=np.float64,
        ),
        beats_per_bar=4,
    )

    assert beats.tolist() == [0.5, 1.0, 1.5, 2.0]
    assert downbeats.tolist() == [0.5, 2.0]
