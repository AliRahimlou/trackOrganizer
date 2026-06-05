from __future__ import annotations

from rhythm_engine.quality import assess_grid_quality
from rhythm_engine.types import RhythmEstimate


def test_assess_grid_quality_marks_stable_supported_grid_high() -> None:
    beats = tuple(0.5 * idx for idx in range(80))
    final = RhythmEstimate(
        provider="final",
        beats=beats,
        downbeats=beats[::4],
        confidence=0.95,
        metadata={
            "hypothesis_selector_score": {"final_score": 0.86},
            "grid_repair_tempo_stability": 0.99,
            "micro_refine_median_abs_offset_ms": 4.0,
        },
    )
    fused = RhythmEstimate(provider="fusion", beats=beats, confidence=0.9, metadata={"provider_count": 3})

    report = assess_grid_quality(final, fused=fused)

    assert report.tier == "HIGH"
    assert report.score >= 0.78
    assert report.warnings == tuple()


def test_assess_grid_quality_flags_unstable_single_provider_grid_low() -> None:
    final = RhythmEstimate(
        provider="final",
        beats=(0.0, 0.5, 1.7, 2.1),
        downbeats=(0.0,),
        confidence=0.4,
        metadata={
            "hypothesis_selector_score": {"final_score": 0.42},
            "micro_refine_median_abs_offset_ms": 35.0,
        },
    )
    fused = RhythmEstimate(provider="fusion", beats=final.beats, confidence=0.4, metadata={"provider_count": 1})

    report = assess_grid_quality(final, fused=fused)

    assert report.tier == "LOW"
    assert "single_provider" in report.warnings
    assert "unstable_tempo" in report.warnings
    assert "large_micro_refine_offset" in report.warnings


def test_assess_grid_quality_flags_fusion_duplicate_pressure() -> None:
    beats = tuple(0.5 * idx for idx in range(80))
    final = RhythmEstimate(
        provider="final",
        beats=beats,
        downbeats=beats[::4],
        confidence=0.9,
        metadata={
            "hypothesis_selector_score": {"final_score": 0.80},
            "grid_repair_tempo_stability": 0.98,
        },
    )
    fused = RhythmEstimate(
        provider="fusion",
        beats=beats,
        confidence=0.8,
        metadata={
            "provider_count": 3,
            "beat_clusters": [
                {"time_sec": 0.0, "suppressed": False},
                {"time_sec": 0.07, "suppressed": True},
                {"time_sec": 0.5, "suppressed": False},
                {"time_sec": 0.57, "suppressed": True},
            ],
        },
    )

    report = assess_grid_quality(final, fused=fused)

    assert report.fusion_suppressed_cluster_ratio == 0.5
    assert "fusion_duplicate_pressure" in report.warnings
