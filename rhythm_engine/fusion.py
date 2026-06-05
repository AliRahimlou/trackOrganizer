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
) -> Tuple[Tuple[float, ...], List[Dict[str, Any]]]:
    events: List[Tuple[float, float, str]] = []
    for estimate in estimates:
        weight = max(0.05, float(estimate.confidence) * provider_weight_for_name(estimate.provider, provider_weights))
        for time_sec in getattr(estimate, attr):
            events.append((float(time_sec), weight, estimate.provider))
    if not events:
        return tuple(), []

    clusters = _cluster_events(events, radius_sec=max(0.001, float(radius_ms) * 0.001))
    fused: List[float] = []
    summaries: List[Dict[str, Any]] = []
    for cluster in clusters:
        times = [row[0] for row in cluster]
        weights = [row[1] for row in cluster]
        providers = sorted(set(row[2] for row in cluster))
        center = _weighted_median(times, weights)
        spread_ms = float(np.std(np.asarray(times, dtype=np.float64)) * 1000.0) if len(times) > 1 else 0.0
        fused.append(center)
        summaries.append(
            {
                "time_sec": float(center),
                "support_count": int(len(cluster)),
                "provider_count": int(len(providers)),
                "providers": providers,
                "spread_ms": float(spread_ms),
            }
        )
    return tuple(sorted(set(round(t, 9) for t in fused))), summaries


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
    beats, beat_clusters = _fuse_times(
        usable,
        attr="beats",
        radius_ms=float(cfg.fusion_radius_ms),
        provider_weights=provider_weights,
    )
    downbeats, downbeat_clusters = _fuse_times(
        usable,
        attr="downbeats",
        radius_ms=float(cfg.downbeat_fusion_radius_ms),
        provider_weights=provider_weights,
    )
    provider_count = len(usable)
    confidence = float(np.clip(np.mean([estimate.confidence for estimate in usable]) + min(0.18, 0.045 * (provider_count - 1)), 0.0, 1.0))
    duration_values = [float(estimate.duration_sec) for estimate in usable if estimate.duration_sec is not None]
    sample_rates = [int(estimate.sample_rate) for estimate in usable if estimate.sample_rate is not None]
    return RhythmEstimate(
        provider="fusion",
        beats=beats,
        downbeats=downbeats,
        bpm=_fused_bpm(usable, provider_weights),
        confidence=confidence,
        duration_sec=max(duration_values) if duration_values else None,
        sample_rate=sample_rates[0] if sample_rates else None,
        metadata={
            "provider_count": int(provider_count),
            "providers": [estimate.provider for estimate in usable],
            "provider_weights": dict(provider_weights),
            "beat_cluster_count": int(len(beat_clusters)),
            "downbeat_cluster_count": int(len(downbeat_clusters)),
            "beat_clusters": beat_clusters[:64],
            "downbeat_clusters": downbeat_clusters[:32],
        },
    )
