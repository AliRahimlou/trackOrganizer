from __future__ import annotations

from rhythm_engine.grid_repair import repair_steady_grid
from rhythm_engine.types import RhythmEngineConfig, RhythmEstimate


def test_repair_steady_grid_fills_missing_beats_and_downbeats() -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5, 1.5, 2.0),
        downbeats=(0.0,),
        bpm=120.0,
        confidence=0.8,
        duration_sec=2.0,
    )

    repaired = repair_steady_grid(estimate, RhythmEngineConfig(beats_per_bar=4))

    assert repaired.beats == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert repaired.downbeats == (0.0, 2.0)
    assert repaired.bpm == 120.0
    assert repaired.metadata["grid_repair_status"] == "ok"
    assert repaired.metadata["grid_repair_inserted_beats"] == 1


def test_repair_steady_grid_skips_unstable_tempo() -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5, 1.6, 2.05, 3.3),
        downbeats=(0.0,),
        bpm=120.0,
        confidence=0.8,
    )

    repaired = repair_steady_grid(
        estimate,
        RhythmEngineConfig(repair_min_tempo_stability=0.95),
    )

    assert repaired.beats == estimate.beats
    assert repaired.metadata["grid_repair_status"] == "skipped_unstable_tempo"


def test_repair_steady_grid_does_not_extend_beyond_duration() -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5, 1.0, 1.5, 2.0),
        downbeats=(0.0,),
        bpm=120.0,
        confidence=0.8,
        duration_sec=1.76,
    )

    repaired = repair_steady_grid(estimate, RhythmEngineConfig(beats_per_bar=4))

    assert repaired.beats == (0.0, 0.5, 1.0, 1.5)
    assert max(repaired.beats) <= estimate.duration_sec
    assert repaired.metadata["grid_repair_trimmed_beats"] == 1


def test_repair_steady_grid_keeps_best_lattice_candidate_for_duplicate_slot() -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5, 0.88, 1.0, 1.5, 2.0),
        downbeats=(0.0,),
        bpm=120.0,
        confidence=0.8,
        duration_sec=2.0,
    )

    repaired = repair_steady_grid(estimate, RhythmEngineConfig(beats_per_bar=4))

    assert repaired.beats == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert repaired.metadata["grid_repair_removed_beats"] >= 1
    assert abs(repaired.metadata["grid_repair_lattice_origin_sec"]) < 1e-12
