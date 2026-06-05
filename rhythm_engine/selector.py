from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import RhythmEngineConfig, RhythmEstimate


@dataclass(frozen=True)
class HypothesisScore:
    provider: str
    final_score: float
    beat_support: float
    beat_hit_rate: float
    offbeat_contrast: float
    tempo_stability: float
    coverage: float
    downbeat_support: float
    provider_confidence: float
    beat_count: int
    downbeat_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "final_score": float(self.final_score),
            "beat_support": float(self.beat_support),
            "beat_hit_rate": float(self.beat_hit_rate),
            "offbeat_contrast": float(self.offbeat_contrast),
            "tempo_stability": float(self.tempo_stability),
            "coverage": float(self.coverage),
            "downbeat_support": float(self.downbeat_support),
            "provider_confidence": float(self.provider_confidence),
            "beat_count": int(self.beat_count),
            "downbeat_count": int(self.downbeat_count),
        }


def _clip01(value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def _safe_norm(x: np.ndarray, *, hi_percentile: float = 94.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 8.0))
    hi = float(np.percentile(arr, float(hi_percentile)))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _window_peak(curve: np.ndarray, times: np.ndarray, center: float, radius: float) -> float:
    if curve.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(center) - float(radius)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(center) + float(radius)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(curve[i0:i1]))


def _sample_events(curve: np.ndarray, times: np.ndarray, events: Sequence[float], radius: float) -> np.ndarray:
    return np.asarray([_window_peak(curve, times, float(t), radius) for t in events], dtype=np.float64)


def _beats_in_duration(beats: Sequence[float], duration_sec: float) -> np.ndarray:
    arr = np.asarray([float(t) for t in beats if 0.0 <= float(t) <= float(duration_sec)], dtype=np.float64)
    if arr.size == 0:
        return arr
    return np.asarray(sorted(set(round(float(t), 9) for t in arr)), dtype=np.float64)


def _tempo_stability(beats: np.ndarray) -> float:
    if beats.size < 4:
        return 0.0
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.08) & (intervals < 4.0)]
    if intervals.size < 3:
        return 0.0
    median = float(np.median(intervals))
    if median <= 1e-9:
        return 0.0
    mad = float(np.median(np.abs(intervals - median)))
    return _clip01(1.0 - (mad / max(1e-6, 0.08 * median)))


def _coverage(beats: np.ndarray, duration_sec: float) -> float:
    if beats.size < 2 or duration_sec <= 0.0:
        return 0.0
    span = float(beats[-1] - beats[0])
    return _clip01(span / max(1e-6, float(duration_sec) * 0.82))


def _downbeat_support(
    curve: np.ndarray,
    times: np.ndarray,
    beats: np.ndarray,
    downbeats: np.ndarray,
    radius: float,
) -> float:
    if beats.size < 8 or downbeats.size < 2:
        return 0.42
    downbeat_vals = _sample_events(curve, times, downbeats, radius)
    if downbeat_vals.size == 0:
        return 0.42
    if beats.size:
        non_downbeats = []
        for beat in beats[: min(256, beats.size)]:
            if np.min(np.abs(downbeats - float(beat))) > max(0.035, 1.5 * radius):
                non_downbeats.append(float(beat))
        non_vals = _sample_events(curve, times, non_downbeats, radius) if non_downbeats else np.asarray([], dtype=np.float64)
    else:
        non_vals = np.asarray([], dtype=np.float64)
    down_mean = float(np.mean(downbeat_vals))
    non_mean = float(np.mean(non_vals)) if non_vals.size else max(0.0, down_mean - 0.10)
    return _clip01((down_mean - non_mean + 0.20) / 0.55)


def score_estimate_against_arrays(
    estimate: RhythmEstimate,
    *,
    frame_times: Sequence[float],
    onset: Sequence[float],
    low_jump: Sequence[float],
    rms: Sequence[float],
    spectral_flux: Sequence[float],
    duration_sec: float,
) -> HypothesisScore:
    times = np.asarray(frame_times, dtype=np.float64)
    n = min(len(times), len(onset), len(low_jump), len(rms), len(spectral_flux))
    if n <= 2 or not estimate.available:
        return HypothesisScore(estimate.provider, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, estimate.confidence, 0, 0)

    times = times[:n]
    onset_arr = _safe_norm(np.asarray(onset, dtype=np.float64)[:n])
    low_arr = _safe_norm(np.asarray(low_jump, dtype=np.float64)[:n])
    rms_arr = _safe_norm(np.asarray(rms, dtype=np.float64)[:n])
    raw_flux = np.asarray(spectral_flux, dtype=np.float64)[:n]
    flux_arr = _safe_norm(raw_flux)
    downbeat_flux_arr = _safe_norm(raw_flux, hi_percentile=99.5)
    beat_curve = _safe_norm((0.40 * onset_arr) + (0.25 * low_arr) + (0.20 * flux_arr) + (0.15 * rms_arr))
    downbeat_curve = _safe_norm(
        (0.20 * onset_arr) + (0.12 * low_arr) + (0.48 * downbeat_flux_arr) + (0.20 * rms_arr),
        hi_percentile=99.0,
    )

    beats = _beats_in_duration(estimate.beats, duration_sec)
    downbeats = _beats_in_duration(estimate.downbeats, duration_sec)
    if beats.size < 3:
        return HypothesisScore(estimate.provider, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, estimate.confidence, int(beats.size), int(downbeats.size))

    intervals = np.diff(beats)
    beat_sec = float(np.median(intervals[(intervals > 0.08) & (intervals < 4.0)])) if intervals.size else 0.5
    radius = max(0.030, min(0.085, 0.16 * beat_sec))
    beat_vals = _sample_events(beat_curve, times, beats, radius)
    beat_support = float(np.mean(beat_vals)) if beat_vals.size else 0.0
    threshold = max(0.18, float(np.percentile(beat_curve, 68.0)) if beat_curve.size else 0.18)
    beat_hit_rate = float(np.mean(beat_vals >= threshold)) if beat_vals.size else 0.0

    midpoints = beats[:-1] + (0.5 * np.diff(beats))
    off_vals = _sample_events(beat_curve, times, midpoints, radius)
    off_mean = float(np.mean(off_vals)) if off_vals.size else 0.0
    offbeat_contrast = _clip01((beat_support - off_mean + 0.16) / 0.50)
    tempo_stability = _tempo_stability(beats)
    coverage = _coverage(beats, float(duration_sec))
    downbeat_support = _downbeat_support(downbeat_curve, times, beats, downbeats, max(radius, 0.045))
    provider_conf = _clip01(estimate.confidence)

    final = _clip01(
        (0.24 * beat_support)
        + (0.21 * beat_hit_rate)
        + (0.18 * offbeat_contrast)
        + (0.13 * tempo_stability)
        + (0.09 * coverage)
        + (0.10 * downbeat_support)
        + (0.05 * provider_conf)
    )
    return HypothesisScore(
        provider=estimate.provider,
        final_score=final,
        beat_support=float(beat_support),
        beat_hit_rate=float(beat_hit_rate),
        offbeat_contrast=float(offbeat_contrast),
        tempo_stability=float(tempo_stability),
        coverage=float(coverage),
        downbeat_support=float(downbeat_support),
        provider_confidence=float(provider_conf),
        beat_count=int(beats.size),
        downbeat_count=int(downbeats.size),
    )


def _extract_selector_arrays(audio_path: str, config: RhythmEngineConfig) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
    from drop_aligner.detector import DropDetectorConfig, extract_features

    cfg = DropDetectorConfig(
        sample_rate=int(config.selector_sample_rate),
        hpss=False,
        use_ranker_model=False,
        use_region_model=False,
        use_drumprint=False,
    )
    features = extract_features(audio_path, cfg)
    arrays = {
        "frame_times": np.asarray(features.frame_times, dtype=np.float64),
        "onset": np.asarray(features.onset, dtype=np.float64),
        "low_jump": np.asarray(features.low_jump_curve, dtype=np.float64),
        "rms": np.asarray(features.rms, dtype=np.float64),
        "spectral_flux": np.asarray(features.spectral_flux, dtype=np.float64),
    }
    return arrays, float(features.duration_sec), {"selector_sample_rate": int(features.sr), "selector_bpm": float(features.bpm)}


def select_best_hypothesis(
    audio_path: str,
    candidates: Iterable[RhythmEstimate],
    *,
    fallback: RhythmEstimate,
    config: RhythmEngineConfig | None = None,
) -> RhythmEstimate:
    cfg = config or RhythmEngineConfig()
    if not cfg.use_hypothesis_selector:
        return fallback
    usable = [estimate for estimate in candidates if estimate.available]
    if not usable:
        return fallback
    try:
        arrays, duration, extraction_meta = _extract_selector_arrays(audio_path, cfg)
        scores = [
            score_estimate_against_arrays(
                estimate,
                frame_times=arrays["frame_times"],
                onset=arrays["onset"],
                low_jump=arrays["low_jump"],
                rms=arrays["rms"],
                spectral_flux=arrays["spectral_flux"],
                duration_sec=duration,
            )
            for estimate in usable
        ]
    except Exception as exc:
        metadata = dict(fallback.metadata)
        metadata["hypothesis_selector_status"] = "failed"
        metadata["hypothesis_selector_error"] = str(exc) or exc.__class__.__name__
        return fallback.with_updates(metadata=metadata)

    paired = list(zip(usable, scores))
    paired.sort(key=lambda row: (-row[1].final_score, -row[0].confidence, row[0].provider))
    chosen, chosen_score = paired[0]
    metadata = dict(chosen.metadata)
    metadata["hypothesis_selector_status"] = "ok"
    metadata["hypothesis_selector_score"] = chosen_score.to_dict()
    metadata["hypothesis_selector_top_scores"] = [score.to_dict() for _estimate, score in paired[:8]]
    metadata.update(extraction_meta)
    return RhythmEstimate(
        provider=f"{chosen.provider}:selected",
        beats=chosen.beats,
        downbeats=chosen.downbeats,
        bpm=chosen.bpm,
        confidence=float(np.clip((0.72 * chosen.confidence) + (0.28 * chosen_score.final_score), 0.0, 1.0)),
        duration_sec=chosen.duration_sec or duration,
        sample_rate=chosen.sample_rate,
        metadata=metadata,
    )
