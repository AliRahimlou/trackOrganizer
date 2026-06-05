from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .detector import FeatureBundle


@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    beat_sec: float
    bar_sec: float
    bar_zero_sec: float
    downbeat_confidence: float
    first_low_downbeat_sec: float
    source_role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm": float(self.bpm),
            "beat_sec": float(self.beat_sec),
            "bar_sec": float(self.bar_sec),
            "bar_zero_sec": float(self.bar_zero_sec),
            "downbeat_confidence": float(self.downbeat_confidence),
            "first_low_downbeat_sec": float(self.first_low_downbeat_sec),
            "source_role": self.source_role,
        }


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _window_max(values: np.ndarray, times: np.ndarray, center: float, radius: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(center) - float(radius)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(center) + float(radius)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(values[i0:i1]))


def _window_mean(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.mean(values[i0:i1]))


def _source_features(features_by_role: Mapping[str, FeatureBundle]) -> tuple[str, FeatureBundle]:
    for role in ("drums", "bass", "instrumental", "vocals"):
        features = features_by_role.get(role)
        if features is not None:
            return role, features
    if not features_by_role:
        raise ValueError("No feature bundles available for beatgrid resolution")
    role = sorted(features_by_role)[0]
    return role, features_by_role[role]


def _candidate_starts(features: FeatureBundle, *, max_candidates: int = 28) -> list[float]:
    times = np.asarray(features.frame_times, dtype=np.float64)
    if times.size < 3:
        return [0.0]
    beat = float(features.beat_sec)
    search_end = min(float(features.duration_sec), max(45.0, 32.0 * 4.0 * beat))
    curve = np.asarray((0.52 * features.low_jump_curve) + (0.30 * features.combined_attack) + (0.18 * features.low_energy), dtype=np.float64)
    limit = min(int(times.size), int(curve.size))
    times = times[:limit]
    curve = curve[:limit]
    valid = np.where((times >= 0.0) & (times <= search_end))[0]
    if valid.size == 0:
        return [0.0]
    threshold = max(float(np.percentile(curve[valid], 70.0)), float(np.max(curve[valid])) * 0.24)
    distance = max(1, int(round((0.75 * beat) / max(1e-6, float(times[1] - times[0])))))
    try:
        from scipy import signal

        peaks, _props = signal.find_peaks(curve, distance=distance, height=threshold, prominence=max(0.01, float(np.std(curve[valid])) * 0.20))
    except Exception:
        peaks = np.asarray([], dtype=np.int64)
    candidates = [0.0]
    ranked = [int(idx) for idx in peaks if int(idx) in set(int(v) for v in valid)]
    ranked.sort(key=lambda idx: float(curve[idx]), reverse=True)
    for idx in ranked[: max_candidates]:
        t = float(times[idx])
        if t >= 0.0 and all(abs(t - existing) > 0.25 * beat for existing in candidates):
            candidates.append(t)
    return sorted(candidates)


def _score_bar_zero(features: FeatureBundle, bar_zero: float) -> float:
    times = np.asarray(features.frame_times, dtype=np.float64)
    if times.size < 3:
        return 0.0
    beat = float(features.beat_sec)
    bar = 4.0 * beat
    duration = float(features.duration_sec)
    if bar <= 0.0 or beat <= 0.0 or bar_zero >= min(duration, 60.0):
        return 0.0

    downbeats = np.arange(float(bar_zero), duration, bar, dtype=np.float64)
    beats = np.arange(float(bar_zero), duration, beat, dtype=np.float64)
    if downbeats.size < 4 or beats.size < 8:
        return 0.0

    low = np.asarray(features.low_energy, dtype=np.float64)
    attack = np.asarray(features.combined_attack, dtype=np.float64)
    onset = np.asarray(features.onset, dtype=np.float64)
    rms = np.asarray(features.rms, dtype=np.float64)
    flux = np.asarray(features.spectral_flux, dtype=np.float64)

    beat_strength = np.mean([_window_max(low, times, t, 0.12 * beat) for t in beats[:96]])
    downbeat_strength = np.mean([_window_max(attack, times, t, 0.16 * beat) for t in downbeats[:32]])
    onset_strength = np.mean([_window_max(onset, times, t, 0.16 * beat) for t in downbeats[:32]])

    boundary_scores: list[float] = []
    phrase_scores: list[float] = []
    for index, t in enumerate(downbeats[:48]):
        pre_rms = _window_mean(rms, times, t - 2.0 * bar, t)
        post_rms = _window_mean(rms, times, t, t + 2.0 * bar)
        pre_low = _window_mean(low, times, t - 2.0 * bar, t)
        post_low = _window_mean(low, times, t, t + 2.0 * bar)
        local_flux = _window_max(flux, times, t, 0.30 * beat)
        boundary = (0.30 * abs(post_rms - pre_rms)) + (0.30 * abs(post_low - pre_low)) + (0.40 * local_flux)
        boundary_scores.append(float(boundary))
        bars = index + 1
        if bars % 8 == 0:
            phrase_scores.append(float(boundary))

    structural = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    phrase = float(np.mean(phrase_scores)) if phrase_scores else structural
    early_penalty = min(0.20, max(0.0, float(bar_zero) / max(1.0, 16.0 * bar)) * 0.20)
    return _clip01((0.28 * beat_strength) + (0.24 * downbeat_strength) + (0.16 * onset_strength) + (0.18 * structural) + (0.14 * phrase) - early_penalty)


def resolve_beatgrid(features_by_role: Mapping[str, FeatureBundle], *, bpm: Optional[float] = None) -> BeatGrid:
    role, features = _source_features(features_by_role)
    bpm_value = float(bpm if bpm is not None and bpm > 0 else features.bpm)
    if not math.isfinite(bpm_value) or bpm_value <= 0:
        bpm_value = 128.0
    beat = 60.0 / bpm_value
    bar = 4.0 * beat

    starts = _candidate_starts(features)
    scored = [(float(_score_bar_zero(features, start)), float(start)) for start in starts]
    if not scored:
        scored = [(0.0, 0.0)]
    scored.sort(key=lambda row: (-row[0], row[1]))
    best_score, best_start = scored[0]
    zero_score = float(_score_bar_zero(features, 0.0))
    earlier_phase = [
        (score, start)
        for score, start in scored
        if 0.35 * beat <= float(best_start) - float(start) <= 1.25 * beat
        and float(score) >= float(best_score) - 0.006
    ]
    if earlier_phase:
        best_score, best_start = sorted(earlier_phase, key=lambda row: (row[1], -row[0]))[0]
    phase_zero = math.fmod(float(best_start), float(bar))
    if phase_zero < 0.0:
        phase_zero += float(bar)
    phase_distance_from_zero = min(float(phase_zero), abs(float(bar) - float(phase_zero)))
    if phase_distance_from_zero <= 0.12 * float(beat):
        best_score, best_start = zero_score, 0.0
    elif zero_score >= 0.30 and float(best_score) < zero_score + 0.18:
        best_score, best_start = zero_score, 0.0
    score_values = np.asarray([score for score, _start in scored], dtype=np.float64)
    median = float(np.median(score_values)) if score_values.size else 0.0
    confidence = _clip01((float(best_score) - median + 0.06) / 0.30)
    phase_zero = math.fmod(float(best_start), float(bar))
    if phase_zero < 0.0:
        phase_zero += float(bar)

    return BeatGrid(
        bpm=float(bpm_value),
        beat_sec=float(beat),
        bar_sec=float(bar),
        bar_zero_sec=float(max(0.0, phase_zero)),
        downbeat_confidence=float(confidence),
        first_low_downbeat_sec=float(max(0.0, best_start)),
        source_role=role,
    )
