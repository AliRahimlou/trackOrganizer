from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class BeatEvaluationReport:
    reference_count: int
    estimated_count: int
    matched_count: int
    median_abs_error_ms: float
    mean_abs_error_ms: float
    p90_abs_error_ms: float
    p95_abs_error_ms: float
    max_abs_error_ms: float
    hit_rate_5ms: float
    hit_rate_10ms: float
    hit_rate_20ms: float
    hit_rate_70ms: float
    continuity_10ms: float
    continuity_20ms: float
    continuity_70ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_count": int(self.reference_count),
            "estimated_count": int(self.estimated_count),
            "matched_count": int(self.matched_count),
            "median_abs_error_ms": float(self.median_abs_error_ms),
            "mean_abs_error_ms": float(self.mean_abs_error_ms),
            "p90_abs_error_ms": float(self.p90_abs_error_ms),
            "p95_abs_error_ms": float(self.p95_abs_error_ms),
            "max_abs_error_ms": float(self.max_abs_error_ms),
            "hit_rate_5ms": float(self.hit_rate_5ms),
            "hit_rate_10ms": float(self.hit_rate_10ms),
            "hit_rate_20ms": float(self.hit_rate_20ms),
            "hit_rate_70ms": float(self.hit_rate_70ms),
            "continuity_10ms": float(self.continuity_10ms),
            "continuity_20ms": float(self.continuity_20ms),
            "continuity_70ms": float(self.continuity_70ms),
        }


def _nearest_errors(reference: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.asarray([], dtype=np.float64)
    if estimated.size == 0:
        return np.full(reference.shape, np.inf, dtype=np.float64)
    errors = np.empty(reference.shape, dtype=np.float64)
    for idx, ref in enumerate(reference):
        nearest = float(estimated[int(np.argmin(np.abs(estimated - ref)))])
        errors[idx] = nearest - float(ref)
    return errors


def _hit_rate(abs_errors_ms: np.ndarray, threshold_ms: float) -> float:
    if abs_errors_ms.size == 0:
        return 0.0
    return float(np.mean(abs_errors_ms <= float(threshold_ms)))


def _continuity(abs_errors_ms: np.ndarray, threshold_ms: float) -> float:
    if abs_errors_ms.size == 0:
        return 0.0
    longest = 0
    current = 0
    for ok in abs_errors_ms <= float(threshold_ms):
        if bool(ok):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest / max(1, abs_errors_ms.size))


def evaluate_beat_grid(reference_beats: Sequence[float], estimated_beats: Sequence[float]) -> BeatEvaluationReport:
    reference = np.asarray(sorted(float(t) for t in reference_beats if float(t) >= 0.0), dtype=np.float64)
    estimated = np.asarray(sorted(float(t) for t in estimated_beats if float(t) >= 0.0), dtype=np.float64)
    errors_ms = _nearest_errors(reference, estimated) * 1000.0
    finite = errors_ms[np.isfinite(errors_ms)]
    abs_errors_ms = np.abs(errors_ms)
    finite_abs = abs_errors_ms[np.isfinite(abs_errors_ms)]

    if finite_abs.size == 0:
        median = mean = p90 = p95 = max_err = float("inf") if reference.size else 0.0
    else:
        median = float(np.median(finite_abs))
        mean = float(np.mean(finite_abs))
        p90 = float(np.percentile(finite_abs, 90.0))
        p95 = float(np.percentile(finite_abs, 95.0))
        max_err = float(np.max(finite_abs))

    return BeatEvaluationReport(
        reference_count=int(reference.size),
        estimated_count=int(estimated.size),
        matched_count=int(finite.size),
        median_abs_error_ms=median,
        mean_abs_error_ms=mean,
        p90_abs_error_ms=p90,
        p95_abs_error_ms=p95,
        max_abs_error_ms=max_err,
        hit_rate_5ms=_hit_rate(abs_errors_ms, 5.0),
        hit_rate_10ms=_hit_rate(abs_errors_ms, 10.0),
        hit_rate_20ms=_hit_rate(abs_errors_ms, 20.0),
        hit_rate_70ms=_hit_rate(abs_errors_ms, 70.0),
        continuity_10ms=_continuity(abs_errors_ms, 10.0),
        continuity_20ms=_continuity(abs_errors_ms, 20.0),
        continuity_70ms=_continuity(abs_errors_ms, 70.0),
    )
