from __future__ import annotations

from rhythm_engine.fusion import fuse_estimates
from rhythm_engine.weights import provider_weight_for_name
from rhythm_engine.types import RhythmEngineConfig, RhythmEstimate


def test_fuse_estimates_uses_weighted_consensus_clusters() -> None:
    estimates = [
        RhythmEstimate(provider="a", beats=(1.000, 1.500, 2.000), downbeats=(1.000,), bpm=120.0, confidence=0.80),
        RhythmEstimate(provider="b", beats=(1.009, 1.508, 2.011), downbeats=(1.009,), bpm=120.2, confidence=0.90),
        RhythmEstimate(provider="c", beats=(1.040, 1.540, 2.040), downbeats=(1.040,), bpm=119.9, confidence=0.35),
    ]
    cfg = RhythmEngineConfig(providers=("a", "b", "c"), fusion_radius_ms=45.0, downbeat_fusion_radius_ms=55.0)

    fused = fuse_estimates(estimates, cfg)

    assert fused.available
    assert len(fused.beats) == 3
    assert abs(fused.beats[0] - 1.009) < 1e-9
    assert len(fused.downbeats) == 1
    assert fused.metadata["provider_count"] == 3
    assert fused.metadata["beat_clusters"][0]["provider_count"] == 3


def test_fuse_estimates_returns_failed_when_all_providers_missing() -> None:
    fused = fuse_estimates([RhythmEstimate.unavailable("beat_this", "not_installed")])

    assert not fused.available
    assert fused.status == "failed"
    assert fused.reason == "no_available_provider_estimates"


def test_fuse_estimates_applies_provider_weight_json(tmp_path) -> None:
    weights = tmp_path / "weights.json"
    weights.write_text('{"provider_weights":{"a":0.2,"b":3.0}}', encoding="utf-8")
    estimates = [
        RhythmEstimate(provider="a", beats=(1.000,), downbeats=(1.000,), bpm=120.0, confidence=0.95),
        RhythmEstimate(provider="b", beats=(1.035,), downbeats=(1.035,), bpm=121.0, confidence=0.40),
    ]
    cfg = RhythmEngineConfig(
        providers=("a", "b"),
        fusion_radius_ms=45.0,
        downbeat_fusion_radius_ms=45.0,
        provider_weights_json=str(weights),
    )

    fused = fuse_estimates(estimates, cfg)

    assert fused.beats == (1.035,)
    assert fused.downbeats == (1.035,)
    assert fused.bpm == 121.0
    assert fused.metadata["provider_weights"] == {"a": 0.2, "b": 3.0}


def test_fuse_estimates_suppresses_off_phase_duplicate_clusters() -> None:
    estimates = [
        RhythmEstimate(provider="good", beats=(1.000, 1.500, 2.000), downbeats=(1.000,), bpm=120.0, confidence=0.95),
        RhythmEstimate(provider="late", beats=(1.070, 1.570, 2.070), downbeats=(1.070,), bpm=120.0, confidence=0.40),
    ]
    cfg = RhythmEngineConfig(
        providers=("good", "late"),
        fusion_radius_ms=35.0,
        downbeat_fusion_radius_ms=35.0,
    )

    fused = fuse_estimates(estimates, cfg)

    assert fused.beats == (1.0, 1.5, 2.0)
    assert fused.downbeats == (1.0,)
    assert any(row["suppressed"] for row in fused.metadata["beat_clusters"])
    assert abs(fused.metadata["fusion_beat_min_gap_sec"] - 0.225) < 1e-12


def test_provider_weight_for_name_matches_nested_provider_names() -> None:
    weights = {"stem_ensemble:drums": 2.0, "librosa": 0.4}

    assert provider_weight_for_name("stem_ensemble:drums:track_organizer", weights) == 2.0
    assert provider_weight_for_name("stem_ensemble:bass:librosa", weights) == 0.4
    assert provider_weight_for_name("unknown", weights) == 1.0
