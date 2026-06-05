from __future__ import annotations

from rhythm_engine.hypotheses import generate_hypotheses
from rhythm_engine.types import RhythmEngineConfig, RhythmEstimate


def test_generate_hypotheses_preserves_base_phase_half_and_double_time() -> None:
    estimate = RhythmEstimate(
        provider="provider",
        beats=tuple(i * 0.5 for i in range(12)),
        downbeats=(0.5, 2.5, 4.5),
        bpm=120.0,
        confidence=0.9,
    )
    cfg = RhythmEngineConfig(
        providers=("provider",),
        min_bpm=50.0,
        max_bpm=260.0,
        beats_per_bar=4,
        max_hypotheses=12,
    )

    hypotheses = generate_hypotheses([estimate], cfg)
    providers = {hyp.provider for hyp in hypotheses}

    assert "provider:base" in providers
    assert "provider:halftime" in providers
    assert "provider:doubletime" in providers
    assert any(provider.startswith("provider:phase") for provider in providers)
    phase = next(hyp for hyp in hypotheses if hyp.provider == "provider:phase1")
    assert phase.downbeats[:2] == (0.5, 2.5)
    half = next(hyp for hyp in hypotheses if hyp.provider == "provider:halftime")
    assert half.bpm == 60.0
    double = next(hyp for hyp in hypotheses if hyp.provider == "provider:doubletime")
    assert double.bpm == 240.0
