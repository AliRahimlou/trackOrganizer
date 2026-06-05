from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .types import RhythmEngineConfig, RhythmEstimate


def _tempo_stability(beats: np.ndarray) -> tuple[float, float]:
    if beats.size < 4:
        return 0.0, 0.0
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.08) & (intervals < 4.0)]
    if intervals.size < 3:
        return 0.0, 0.0
    period = float(np.median(intervals))
    if period <= 1e-9:
        return 0.0, 0.0
    mad = float(np.median(np.abs(intervals - period)))
    stability = float(np.clip(1.0 - (mad / max(1e-6, 0.08 * period)), 0.0, 1.0))
    return stability, period


def _dedupe_close_beats(beats: Sequence[float], period: float) -> list[float]:
    out: list[float] = []
    min_gap = max(0.020, 0.45 * float(period))
    for beat in sorted(float(t) for t in beats if float(t) >= 0.0):
        if not out or beat - out[-1] >= min_gap:
            out.append(float(beat))
    return out


def _fill_missing_beats(beats: Sequence[float], period: float) -> tuple[list[float], int]:
    if len(beats) < 2:
        return list(beats), 0
    out = [float(beats[0])]
    inserted = 0
    for nxt in beats[1:]:
        prev = out[-1]
        gap = float(nxt) - float(prev)
        if gap > 1.55 * float(period):
            missing = int(round(gap / float(period))) - 1
            for step in range(max(0, missing)):
                candidate = prev + ((step + 1) * float(period))
                if candidate < float(nxt) - (0.35 * float(period)):
                    out.append(float(candidate))
                    inserted += 1
        out.append(float(nxt))
    return out, inserted


def _infer_lattice_origin(beats: Sequence[float], period: float) -> float | None:
    arr = np.asarray(list(beats), dtype=np.float64)
    if arr.size < 3 or period <= 0.0:
        return None
    phases = np.mod(arr, float(period)) / float(period)
    vectors = np.exp(2j * np.pi * phases)
    mean_vector = np.mean(vectors)
    if abs(mean_vector) < 0.30:
        return None
    phase = float(np.angle(mean_vector) / (2.0 * np.pi))
    if phase < 0.0:
        phase += 1.0
    origin_mod = phase * float(period)
    first = float(np.min(arr))
    origin = origin_mod + (round((first - origin_mod) / float(period)) * float(period))
    while origin > first + (0.5 * float(period)):
        origin -= float(period)
    while origin + float(period) <= first - (0.5 * float(period)):
        origin += float(period)
    return float(max(0.0, origin))


def _snap_to_lattice(
    beats: Sequence[float],
    period: float,
    *,
    snap_ratio: float,
) -> tuple[list[float], int, int, float | None]:
    origin = _infer_lattice_origin(beats, period)
    if origin is None:
        return list(float(t) for t in beats), 0, 0, None
    tolerance = max(0.020, float(snap_ratio) * float(period))
    slots: dict[int, tuple[float, float, float]] = {}
    removed = 0
    snapped = 0
    for beat in sorted(float(t) for t in beats if float(t) >= 0.0):
        idx = int(round((beat - origin) / float(period)))
        target = origin + (idx * float(period))
        if target < 0.0:
            target = 0.0
        residual = abs(beat - target)
        if residual > tolerance:
            removed += 1
            continue
        if residual > 0.001:
            snapped += 1
        previous = slots.get(idx)
        if previous is None or residual < previous[0]:
            if previous is not None:
                removed += 1
            slots[idx] = (float(residual), float(target), float(beat))
        else:
            removed += 1
    return [float(row[1]) for _idx, row in sorted(slots.items())], int(snapped), int(removed), float(origin)


def _trim_to_duration(beats: Sequence[float], duration_sec: float, period: float) -> tuple[list[float], int]:
    tolerance = min(0.010, max(0.001, 0.02 * float(period)))
    kept = [float(beat) for beat in beats if float(beat) <= float(duration_sec) + tolerance]
    return kept, int(len(beats) - len(kept))


def _infer_downbeat_phase(beats: Sequence[float], downbeats: Sequence[float], beats_per_bar: int) -> int:
    if not beats:
        return 0
    if not downbeats:
        return 0
    beat_arr = np.asarray(beats, dtype=np.float64)
    votes: dict[int, int] = {}
    for downbeat in downbeats:
        idx = int(np.argmin(np.abs(beat_arr - float(downbeat))))
        phase = idx % max(1, int(beats_per_bar))
        votes[phase] = votes.get(phase, 0) + 1
    if not votes:
        return 0
    return sorted(votes.items(), key=lambda row: (-row[1], row[0]))[0][0]


def repair_steady_grid(estimate: RhythmEstimate, config: RhythmEngineConfig | None = None) -> RhythmEstimate:
    cfg = config or RhythmEngineConfig()
    if not cfg.repair_steady_grid or not estimate.available:
        return estimate
    beats = np.asarray(estimate.beats, dtype=np.float64)
    stability, period = _tempo_stability(beats)
    metadata = dict(estimate.metadata)
    metadata["grid_repair_tempo_stability"] = float(stability)
    if stability < float(cfg.repair_min_tempo_stability) or period <= 0.0:
        metadata["grid_repair_status"] = "skipped_unstable_tempo"
        return estimate.with_updates(metadata=metadata)

    snapped = 0
    removed = 0
    lattice_origin = None
    if bool(cfg.repair_lattice_snap):
        deduped, snapped, removed, lattice_origin = _snap_to_lattice(
            estimate.beats,
            period,
            snap_ratio=float(cfg.repair_lattice_snap_ratio),
        )
        if lattice_origin is None:
            deduped = _dedupe_close_beats(estimate.beats, period)
    else:
        deduped = _dedupe_close_beats(estimate.beats, period)
    filled, inserted = _fill_missing_beats(deduped, period)
    duration = estimate.duration_sec
    trimmed = 0
    if duration is not None and filled:
        filled, trimmed = _trim_to_duration(filled, float(duration), period)
    if duration is not None and filled:
        tolerance = min(0.010, max(0.001, 0.02 * float(period)))
        while filled[-1] + period <= float(duration) + tolerance:
            filled.append(float(filled[-1] + period))
            inserted += 1
    phase = _infer_downbeat_phase(filled, estimate.downbeats, int(cfg.beats_per_bar))
    downbeats = filled[phase:: max(1, int(cfg.beats_per_bar))]

    metadata["grid_repair_status"] = "ok"
    metadata["grid_repair_period_sec"] = float(period)
    metadata["grid_repair_inserted_beats"] = int(inserted)
    metadata["grid_repair_trimmed_beats"] = int(trimmed)
    metadata["grid_repair_snapped_beats"] = int(snapped)
    metadata["grid_repair_removed_beats"] = int(removed)
    metadata["grid_repair_lattice_origin_sec"] = None if lattice_origin is None else float(lattice_origin)
    metadata["grid_repair_input_beats"] = int(len(estimate.beats))
    metadata["grid_repair_output_beats"] = int(len(filled))
    metadata["grid_repair_downbeat_phase"] = int(phase)
    bpm = 60.0 / period if period > 0 else estimate.bpm
    if bpm is not None and not math.isfinite(float(bpm)):
        bpm = estimate.bpm
    return RhythmEstimate(
        provider=f"{estimate.provider}:repaired",
        beats=tuple(float(t) for t in filled),
        downbeats=tuple(float(t) for t in downbeats),
        bpm=bpm,
        confidence=estimate.confidence,
        duration_sec=estimate.duration_sec,
        sample_rate=estimate.sample_rate,
        metadata=metadata,
    )
