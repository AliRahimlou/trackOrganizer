from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class BeatEvaluationReport:
    reference_count: int
    estimated_count: int
    matched_count: int
    median_error_ms: float
    mean_error_ms: float
    median_abs_error_ms: float
    mean_abs_error_ms: float
    p90_abs_error_ms: float
    p95_abs_error_ms: float
    max_abs_error_ms: float
    hit_rate_5ms: float
    hit_rate_10ms: float
    hit_rate_20ms: float
    hit_rate_70ms: float
    precision_20ms: float
    recall_20ms: float
    f1_20ms: float
    precision_70ms: float
    recall_70ms: float
    f1_70ms: float
    false_positive_count_70ms: int
    missed_count_70ms: int
    continuity_10ms: float
    continuity_20ms: float
    continuity_70ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_count": int(self.reference_count),
            "estimated_count": int(self.estimated_count),
            "matched_count": int(self.matched_count),
            "median_error_ms": float(self.median_error_ms),
            "mean_error_ms": float(self.mean_error_ms),
            "median_abs_error_ms": float(self.median_abs_error_ms),
            "mean_abs_error_ms": float(self.mean_abs_error_ms),
            "p90_abs_error_ms": float(self.p90_abs_error_ms),
            "p95_abs_error_ms": float(self.p95_abs_error_ms),
            "max_abs_error_ms": float(self.max_abs_error_ms),
            "hit_rate_5ms": float(self.hit_rate_5ms),
            "hit_rate_10ms": float(self.hit_rate_10ms),
            "hit_rate_20ms": float(self.hit_rate_20ms),
            "hit_rate_70ms": float(self.hit_rate_70ms),
            "precision_20ms": float(self.precision_20ms),
            "recall_20ms": float(self.recall_20ms),
            "f1_20ms": float(self.f1_20ms),
            "precision_70ms": float(self.precision_70ms),
            "recall_70ms": float(self.recall_70ms),
            "f1_70ms": float(self.f1_70ms),
            "false_positive_count_70ms": int(self.false_positive_count_70ms),
            "missed_count_70ms": int(self.missed_count_70ms),
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


def _one_to_one_match_count(reference: np.ndarray, estimated: np.ndarray, threshold_ms: float) -> int:
    if reference.size == 0 or estimated.size == 0:
        return 0
    threshold_sec = float(threshold_ms) * 0.001
    i = 0
    j = 0
    matched = 0
    while i < reference.size and j < estimated.size:
        diff = float(estimated[j] - reference[i])
        if abs(diff) <= threshold_sec:
            matched += 1
            i += 1
            j += 1
        elif diff < -threshold_sec:
            j += 1
        else:
            i += 1
    return int(matched)


def _precision_recall_f1(reference: np.ndarray, estimated: np.ndarray, threshold_ms: float) -> tuple[float, float, float, int]:
    matched = _one_to_one_match_count(reference, estimated, threshold_ms)
    precision = float(matched / max(1, estimated.size))
    recall = float(matched / max(1, reference.size))
    denom = precision + recall
    f1 = 0.0 if denom <= 0.0 else float((2.0 * precision * recall) / denom)
    return precision, recall, f1, matched


def evaluate_beat_grid(reference_beats: Sequence[float], estimated_beats: Sequence[float]) -> BeatEvaluationReport:
    reference = np.asarray(sorted(float(t) for t in reference_beats if float(t) >= 0.0), dtype=np.float64)
    estimated = np.asarray(sorted(float(t) for t in estimated_beats if float(t) >= 0.0), dtype=np.float64)
    errors_ms = _nearest_errors(reference, estimated) * 1000.0
    finite = errors_ms[np.isfinite(errors_ms)]
    abs_errors_ms = np.abs(errors_ms)
    finite_abs = abs_errors_ms[np.isfinite(abs_errors_ms)]

    if finite.size == 0:
        signed_median = signed_mean = float("inf") if reference.size else 0.0
    else:
        signed_median = float(np.median(finite))
        signed_mean = float(np.mean(finite))

    if finite_abs.size == 0:
        median = mean = p90 = p95 = max_err = float("inf") if reference.size else 0.0
    else:
        median = float(np.median(finite_abs))
        mean = float(np.mean(finite_abs))
        p90 = float(np.percentile(finite_abs, 90.0))
        p95 = float(np.percentile(finite_abs, 95.0))
        max_err = float(np.max(finite_abs))

    precision_20, recall_20, f1_20, _matched_20 = _precision_recall_f1(reference, estimated, 20.0)
    precision_70, recall_70, f1_70, matched_70 = _precision_recall_f1(reference, estimated, 70.0)

    return BeatEvaluationReport(
        reference_count=int(reference.size),
        estimated_count=int(estimated.size),
        matched_count=int(matched_70),
        median_error_ms=signed_median,
        mean_error_ms=signed_mean,
        median_abs_error_ms=median,
        mean_abs_error_ms=mean,
        p90_abs_error_ms=p90,
        p95_abs_error_ms=p95,
        max_abs_error_ms=max_err,
        hit_rate_5ms=_hit_rate(abs_errors_ms, 5.0),
        hit_rate_10ms=_hit_rate(abs_errors_ms, 10.0),
        hit_rate_20ms=_hit_rate(abs_errors_ms, 20.0),
        hit_rate_70ms=_hit_rate(abs_errors_ms, 70.0),
        precision_20ms=precision_20,
        recall_20ms=recall_20,
        f1_20ms=f1_20,
        precision_70ms=precision_70,
        recall_70ms=recall_70,
        f1_70ms=f1_70,
        false_positive_count_70ms=max(0, int(estimated.size) - int(matched_70)),
        missed_count_70ms=max(0, int(reference.size) - int(matched_70)),
        continuity_10ms=_continuity(abs_errors_ms, 10.0),
        continuity_20ms=_continuity(abs_errors_ms, 20.0),
        continuity_70ms=_continuity(abs_errors_ms, 70.0),
    )
