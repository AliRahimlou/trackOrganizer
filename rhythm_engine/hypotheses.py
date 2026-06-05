from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .providers import available_estimates
from .types import RhythmEngineConfig, RhythmEstimate


def _estimate_with(
    estimate: RhythmEstimate,
    *,
    provider_suffix: str,
    beats: Sequence[float],
    downbeats: Sequence[float],
    bpm: float | None,
    confidence_scale: float,
    variant: str,
    metadata_extra: dict[str, object] | None = None,
) -> RhythmEstimate:
    metadata = dict(estimate.metadata)
    metadata["hypothesis_variant"] = variant
    metadata["hypothesis_parent_provider"] = estimate.provider
    metadata.update(metadata_extra or {})
    return RhythmEstimate(
        provider=f"{estimate.provider}:{provider_suffix}",
        beats=tuple(float(t) for t in beats),
        downbeats=tuple(float(t) for t in downbeats),
        bpm=bpm,
        confidence=max(0.0, min(1.0, float(estimate.confidence) * float(confidence_scale))),
        duration_sec=estimate.duration_sec,
        sample_rate=estimate.sample_rate,
        metadata=metadata,
    )


def _phase_overlap_score(beats: np.ndarray, phase_downbeats: np.ndarray, provider_downbeats: Sequence[float]) -> float:
    known = np.asarray(list(provider_downbeats), dtype=np.float64)
    if known.size == 0 or phase_downbeats.size == 0 or beats.size < 2:
        return 0.0
    beat_sec = float(np.median(np.diff(beats))) if beats.size > 2 else 0.5
    tol = max(0.045, 0.18 * beat_sec)
    hits = 0
    for db in known[:64]:
        if np.min(np.abs(phase_downbeats - float(db))) <= tol:
            hits += 1
    return float(hits / max(1, min(len(known), 64)))


def _phase_hypotheses(estimate: RhythmEstimate, config: RhythmEngineConfig) -> List[RhythmEstimate]:
    beats = np.asarray(estimate.beats, dtype=np.float64)
    if beats.size < max(4, int(config.beats_per_bar) * 2):
        return []
    scored: List[tuple[float, int, np.ndarray]] = []
    for phase in range(max(1, int(config.beats_per_bar))):
        downbeats = beats[phase:: max(1, int(config.beats_per_bar))]
        overlap = _phase_overlap_score(beats, downbeats, estimate.downbeats)
        phase_bias = 0.04 * max(0.0, 1.0 - (phase / max(1, int(config.beats_per_bar) - 1)))
        score = overlap if overlap > 0.0 else phase_bias
        scored.append((float(score), int(phase), downbeats))
    scored.sort(key=lambda row: (-row[0], row[1]))

    out: List[RhythmEstimate] = []
    for score, phase, downbeats in scored[:2]:
        scale = 0.92 if score > 0.0 else 0.76
        out.append(
            _estimate_with(
                estimate,
                provider_suffix=f"phase{phase}",
                beats=estimate.beats,
                downbeats=tuple(float(t) for t in downbeats),
                bpm=estimate.bpm,
                confidence_scale=scale,
                variant="downbeat_phase",
                metadata_extra={"downbeat_phase": int(phase), "downbeat_phase_score": float(score)},
            )
        )
    return out


def _half_time_hypothesis(estimate: RhythmEstimate, config: RhythmEngineConfig) -> RhythmEstimate | None:
    beats = np.asarray(estimate.beats, dtype=np.float64)
    if beats.size < 6:
        return None
    bpm = None if estimate.bpm is None else float(estimate.bpm) * 0.5
    if bpm is not None and bpm < float(config.min_bpm):
        return None
    reduced = beats[::2]
    downbeats = reduced[:: max(1, int(config.beats_per_bar))]
    return _estimate_with(
        estimate,
        provider_suffix="halftime",
        beats=tuple(float(t) for t in reduced),
        downbeats=tuple(float(t) for t in downbeats),
        bpm=bpm,
        confidence_scale=0.70,
        variant="halftime",
    )


def _double_time_hypothesis(estimate: RhythmEstimate, config: RhythmEngineConfig) -> RhythmEstimate | None:
    beats = np.asarray(estimate.beats, dtype=np.float64)
    if beats.size < 4:
        return None
    bpm = None if estimate.bpm is None else float(estimate.bpm) * 2.0
    if bpm is not None and bpm > float(config.max_bpm):
        return None
    intervals = np.diff(beats)
    if intervals.size == 0:
        return None
    mids = beats[:-1] + (0.5 * intervals)
    doubled = np.sort(np.concatenate([beats, mids]))
    downbeats = doubled[:: max(1, int(config.beats_per_bar))]
    return _estimate_with(
        estimate,
        provider_suffix="doubletime",
        beats=tuple(float(t) for t in doubled),
        downbeats=tuple(float(t) for t in downbeats),
        bpm=bpm,
        confidence_scale=0.62,
        variant="doubletime",
    )


def generate_hypotheses(estimates: Iterable[RhythmEstimate], config: RhythmEngineConfig | None = None) -> List[RhythmEstimate]:
    cfg = config or RhythmEngineConfig()
    usable = available_estimates(estimates, cfg)
    out: List[RhythmEstimate] = []
    for estimate in usable:
        out.append(
            _estimate_with(
                estimate,
                provider_suffix="base",
                beats=estimate.beats,
                downbeats=estimate.downbeats,
                bpm=estimate.bpm,
                confidence_scale=1.0,
                variant="base",
            )
        )
        out.extend(_phase_hypotheses(estimate, cfg))
        half = _half_time_hypothesis(estimate, cfg)
        if half is not None:
            out.append(half)
        double = _double_time_hypothesis(estimate, cfg)
        if double is not None:
            out.append(double)

    out.sort(
        key=lambda estimate: (
            -float(estimate.confidence),
            str(estimate.metadata.get("hypothesis_variant", "")) != "base",
            estimate.provider,
        )
    )
    limit = max(0, int(cfg.max_hypotheses))
    return out[:limit] if limit else out
