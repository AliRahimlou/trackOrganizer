from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .types import RhythmEstimate


@dataclass(frozen=True)
class GridQualityReport:
    tier: str
    score: float
    selector_score: float
    tempo_stability: float
    provider_count: int
    beat_count: int
    downbeat_count: int
    inserted_beat_ratio: float
    fusion_suppressed_cluster_ratio: float
    median_micro_offset_ms: float
    warnings: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "score": float(self.score),
            "selector_score": float(self.selector_score),
            "tempo_stability": float(self.tempo_stability),
            "provider_count": int(self.provider_count),
            "beat_count": int(self.beat_count),
            "downbeat_count": int(self.downbeat_count),
            "inserted_beat_ratio": float(self.inserted_beat_ratio),
            "fusion_suppressed_cluster_ratio": float(self.fusion_suppressed_cluster_ratio),
            "median_micro_offset_ms": float(self.median_micro_offset_ms),
            "warnings": list(self.warnings),
        }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _tempo_stability(beats: Sequence[float]) -> float:
    arr = np.asarray(list(beats), dtype=np.float64)
    if arr.size < 4:
        return 0.0
    intervals = np.diff(arr)
    intervals = intervals[(intervals > 0.08) & (intervals < 4.0)]
    if intervals.size < 3:
        return 0.0
    period = float(np.median(intervals))
    if period <= 1e-9:
        return 0.0
    mad = float(np.median(np.abs(intervals - period)))
    return float(np.clip(1.0 - (mad / max(1e-6, 0.08 * period)), 0.0, 1.0))


def assess_grid_quality(final: RhythmEstimate, *, fused: RhythmEstimate | None = None) -> GridQualityReport:
    metadata: Mapping[str, Any] = final.metadata or {}
    selector_payload = metadata.get("hypothesis_selector_score")
    selector_score = 0.0
    if isinstance(selector_payload, Mapping):
        selector_score = _finite_float(selector_payload.get("final_score"))
    tempo_stability = max(
        _finite_float(metadata.get("grid_repair_tempo_stability")),
        _tempo_stability(final.beats),
    )
    provider_count = 0
    if fused is not None and isinstance(fused.metadata, Mapping):
        provider_count = int(_finite_float(fused.metadata.get("provider_count"), 0.0))
    suppressed_ratio = 0.0
    if fused is not None and isinstance(fused.metadata, Mapping):
        clusters = fused.metadata.get("beat_clusters")
        if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
            total = len(clusters)
            suppressed = sum(1 for row in clusters if isinstance(row, Mapping) and bool(row.get("suppressed")))
            suppressed_ratio = float(suppressed / max(1, total))
    beat_count = len(final.beats)
    downbeat_count = len(final.downbeats)
    input_beats = int(_finite_float(metadata.get("grid_repair_input_beats"), beat_count))
    inserted = int(_finite_float(metadata.get("grid_repair_inserted_beats"), 0.0))
    inserted_ratio = float(inserted / max(1, input_beats))
    micro_offset = _finite_float(metadata.get("micro_refine_median_abs_offset_ms"), 0.0)

    warnings: list[str] = []
    if beat_count < 8:
        warnings.append("few_beats")
    if downbeat_count < 2:
        warnings.append("few_downbeats")
    if provider_count <= 1:
        warnings.append("single_provider")
    if selector_score and selector_score < 0.58:
        warnings.append("low_selector_score")
    if tempo_stability < 0.72:
        warnings.append("unstable_tempo")
    if inserted_ratio > 0.12:
        warnings.append("many_repaired_beats")
    if suppressed_ratio > 0.18:
        warnings.append("fusion_duplicate_pressure")
    if micro_offset > 28.0:
        warnings.append("large_micro_refine_offset")

    score = float(
        np.clip(
            (0.26 * (selector_score if selector_score > 0.0 else final.confidence))
            + (0.22 * tempo_stability)
            + (0.16 * np.clip(provider_count / 3.0, 0.0, 1.0))
            + (0.14 * np.clip(beat_count / 64.0, 0.0, 1.0))
            + (0.10 * np.clip(downbeat_count / 16.0, 0.0, 1.0))
            + (0.08 * (1.0 - np.clip(inserted_ratio / 0.20, 0.0, 1.0)))
            - (0.06 * np.clip(suppressed_ratio / 0.35, 0.0, 1.0))
            + (0.04 * (1.0 - np.clip(micro_offset / 45.0, 0.0, 1.0))),
            0.0,
            1.0,
        )
    )
    if score >= 0.78 and len(warnings) <= 1:
        tier = "HIGH"
    elif score >= 0.58 and len(warnings) <= 3:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return GridQualityReport(
        tier=tier,
        score=score,
        selector_score=float(selector_score),
        tempo_stability=float(tempo_stability),
        provider_count=int(provider_count),
        beat_count=int(beat_count),
        downbeat_count=int(downbeat_count),
        inserted_beat_ratio=float(inserted_ratio),
        fusion_suppressed_cluster_ratio=float(suppressed_ratio),
        median_micro_offset_ms=float(micro_offset),
        warnings=tuple(warnings),
    )
