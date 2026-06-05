from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .providers import available_estimates
from .types import RhythmEngineConfig, RhythmEstimate
from .weights import provider_weight_for_name, weights_from_config


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    pairs = sorted((float(v), max(1e-6, float(w))) for v, w in zip(values, weights))
    total = sum(weight for _value, weight in pairs)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= 0.5 * total:
            return float(value)
    return float(pairs[-1][0])


def _cluster_events(events: Sequence[Tuple[float, float, str]], radius_sec: float) -> List[List[Tuple[float, float, str]]]:
    clusters: List[List[Tuple[float, float, str]]] = []
    for time_sec, weight, provider in sorted(events, key=lambda row: row[0]):
        best_idx = None
        best_distance = float("inf")
        for idx, cluster in enumerate(clusters):
            center = _weighted_median([row[0] for row in cluster], [row[1] for row in cluster])
            distance = abs(float(time_sec) - center)
            if distance <= float(radius_sec) and distance < best_distance:
                best_idx = idx
                best_distance = distance
        if best_idx is None:
            clusters.append([(float(time_sec), float(weight), str(provider))])
        else:
            clusters[best_idx].append((float(time_sec), float(weight), str(provider)))
    return clusters


def _fuse_times(
    estimates: Sequence[RhythmEstimate],
    *,
    attr: str,
    radius_ms: float,
    provider_weights: Dict[str, float],
    min_gap_sec: float | None = None,
) -> Tuple[Tuple[float, ...], List[Dict[str, Any]]]:
    events: List[Tuple[float, float, str]] = []
    for estimate in estimates:
        weight = max(0.05, float(estimate.confidence) * provider_weight_for_name(estimate.provider, provider_weights))
        for time_sec in getattr(estimate, attr):
            events.append((float(time_sec), weight, estimate.provider))
    if not events:
        return tuple(), []

    clusters = _cluster_events(events, radius_sec=max(0.001, float(radius_ms) * 0.001))
    summaries: List[Dict[str, Any]] = []
    for cluster in clusters:
        times = [row[0] for row in cluster]
        weights = [row[1] for row in cluster]
        providers = sorted(set(row[2] for row in cluster))
        center = _weighted_median(times, weights)
        spread_ms = float(np.std(np.asarray(times, dtype=np.float64)) * 1000.0) if len(times) > 1 else 0.0
        summaries.append(
            {
                "time_sec": float(center),
                "support_count": int(len(cluster)),
                "provider_count": int(len(providers)),
                "weight_sum": float(sum(weights)),
                "max_weight": float(max(weights) if weights else 0.0),
                "providers": providers,
                "spread_ms": float(spread_ms),
                "suppressed": False,
            }
        )
    if min_gap_sec is not None and float(min_gap_sec) > 0.0:
        summaries = _suppress_close_clusters(summaries, float(min_gap_sec))
    fused = [float(row["time_sec"]) for row in summaries if not bool(row.get("suppressed"))]
    return tuple(sorted(set(round(t, 9) for t in fused))), summaries


def _cluster_quality(summary: Dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary.get("provider_count", 0.0) or 0.0),
        float(summary.get("weight_sum", 0.0) or 0.0),
        -float(summary.get("spread_ms", 0.0) or 0.0),
    )


def _suppress_close_clusters(summaries: Sequence[Dict[str, Any]], min_gap_sec: float) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []
    sorted_summaries = sorted(
        (dict(row) for row in summaries),
        key=lambda row: (
            -_cluster_quality(row)[0],
            -_cluster_quality(row)[1],
            -_cluster_quality(row)[2],
            float(row.get("time_sec", 0.0) or 0.0),
        ),
    )
    for summary in sorted_summaries:
        time_sec = float(summary.get("time_sec", 0.0) or 0.0)
        conflict = next(
            (
                kept
                for kept in accepted
                if abs(float(kept.get("time_sec", 0.0) or 0.0) - time_sec) < float(min_gap_sec)
            ),
            None,
        )
        if conflict is None:
            accepted.append(summary)
            continue
        summary["suppressed"] = True
        summary["suppressed_by_time_sec"] = float(conflict.get("time_sec", 0.0) or 0.0)

    return sorted([*accepted, *(row for row in sorted_summaries if bool(row.get("suppressed")))], key=lambda row: float(row.get("time_sec", 0.0) or 0.0))


def _fused_bpm(estimates: Sequence[RhythmEstimate], provider_weights: Dict[str, float]) -> float | None:
    bpms = [float(estimate.bpm) for estimate in estimates if estimate.bpm is not None]
    weights = [
        max(0.05, float(estimate.confidence) * provider_weight_for_name(estimate.provider, provider_weights))
        for estimate in estimates
        if estimate.bpm is not None
    ]
    if not bpms:
        return None
    return _weighted_median(bpms, weights)


def fuse_estimates(estimates: Sequence[RhythmEstimate], config: RhythmEngineConfig | None = None) -> RhythmEstimate:
    cfg = config or RhythmEngineConfig()
    usable = available_estimates(estimates, cfg)
    if not usable:
        return RhythmEstimate.failed("fusion", "no_available_provider_estimates")

    provider_weights = weights_from_config(cfg.provider_weights_json)
    bpm = _fused_bpm(usable, provider_weights)
    beat_period = None if bpm is None or bpm <= 0 else 60.0 / float(bpm)
    beat_min_gap = None
    downbeat_min_gap = None
    if bool(cfg.fusion_dedupe_events) and beat_period is not None:
        beat_min_gap = max(0.030, float(cfg.fusion_min_beat_gap_ratio) * float(beat_period))
        downbeat_min_gap = max(0.080, float(cfg.fusion_min_downbeat_gap_ratio) * float(beat_period) * max(1, int(cfg.beats_per_bar)))
    beats, beat_clusters = _fuse_times(
        usable,
        attr="beats",
        radius_ms=float(cfg.fusion_radius_ms),
        provider_weights=provider_weights,
        min_gap_sec=beat_min_gap,
    )
    downbeats, downbeat_clusters = _fuse_times(
        usable,
        attr="downbeats",
        radius_ms=float(cfg.downbeat_fusion_radius_ms),
        provider_weights=provider_weights,
        min_gap_sec=downbeat_min_gap,
    )
    provider_count = len(usable)
    confidence = float(np.clip(np.mean([estimate.confidence for estimate in usable]) + min(0.18, 0.045 * (provider_count - 1)), 0.0, 1.0))
    duration_values = [float(estimate.duration_sec) for estimate in usable if estimate.duration_sec is not None]
    sample_rates = [int(estimate.sample_rate) for estimate in usable if estimate.sample_rate is not None]
    return RhythmEstimate(
        provider="fusion",
        beats=beats,
        downbeats=downbeats,
        bpm=bpm,
        confidence=confidence,
        duration_sec=max(duration_values) if duration_values else None,
        sample_rate=sample_rates[0] if sample_rates else None,
        metadata={
            "provider_count": int(provider_count),
            "providers": [estimate.provider for estimate in usable],
            "provider_weights": dict(provider_weights),
            "fusion_dedupe_events": bool(cfg.fusion_dedupe_events),
            "fusion_beat_min_gap_sec": None if beat_min_gap is None else float(beat_min_gap),
            "fusion_downbeat_min_gap_sec": None if downbeat_min_gap is None else float(downbeat_min_gap),
            "beat_cluster_count": int(len(beat_clusters)),
            "downbeat_cluster_count": int(len(downbeat_clusters)),
            "beat_clusters": beat_clusters[:64],
            "downbeat_clusters": downbeat_clusters[:32],
        },
    )
