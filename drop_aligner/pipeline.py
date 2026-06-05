from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate_chooser import candidate_effective_time


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_finite_float(value), 0.0, 1.0))


def _metric(candidate: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if key in candidate and candidate.get(key) not in (None, ""):
        return _finite_float(candidate.get(key), default)
    for nested_name in ("microalign", "drumprint", "full_groove", "debug", "drop_pipeline_score_components"):
        nested = candidate.get(nested_name)
        if isinstance(nested, Mapping) and nested.get(key) not in (None, ""):
            return _finite_float(nested.get(key), default)
    return float(default)


def _candidate_roles(candidate: Mapping[str, Any]) -> List[str]:
    roles = candidate.get("multistem_roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for role in roles:
        text = str(role or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _cluster_center(items: Sequence[Tuple[float, Mapping[str, Any]]]) -> float:
    times = sorted(float(t) for t, _candidate in items)
    if not times:
        return 0.0
    mid = len(times) // 2
    if len(times) % 2:
        return float(times[mid])
    return float((times[mid - 1] + times[mid]) * 0.5)


def _role_union(candidates: Sequence[Mapping[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for candidate in candidates:
        for role in _candidate_roles(candidate):
            if role in seen:
                continue
            seen.add(role)
            out.append(role)
    return out


def _source_union(candidates: Sequence[Mapping[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for candidate in candidates:
        reason = str(candidate.get("reason") or "")
        if reason.startswith("multistem_candidate:"):
            for source in reason.split(":", 1)[1].split(","):
                text = source.strip()
                if text and text not in seen:
                    seen.add(text)
                    out.append(text)
        source = str(candidate.get("source") or "")
        if source and source not in seen:
            seen.add(source)
            out.append(source)
    return out


def _base_score(candidate: Mapping[str, Any]) -> float:
    return _clip01(candidate.get("confidence_score", candidate.get("score", 0.0)))


def drop_score_components(
    candidate: Mapping[str, Any],
    *,
    cluster_size: int = 1,
    cluster_roles: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    roles = list(cluster_roles) if cluster_roles is not None else _candidate_roles(candidate)
    musical_roles = {role for role in roles if role not in {"saved", "unknown"}}
    role_count = len(musical_roles)

    original_score = _base_score(candidate)
    micro = _clip01(_metric(candidate, "micro_confidence"))
    drums = max(
        _clip01(_metric(candidate, "drums_transient_score")),
        _clip01(_metric(candidate, "transient_strength")),
        0.85 * _clip01(_metric(candidate, "attack_peak_strength")),
    )
    bass = max(_clip01(_metric(candidate, "bass_low_jump_score")), _clip01(_metric(candidate, "low_end_jump")))
    instrumental = max(
        _clip01(_metric(candidate, "inst_energy_jump_score")),
        _clip01(_metric(candidate, "energy_contrast")),
        _clip01(_metric(candidate, "spectral_flux_peak")),
    )
    vocal = max(
        _clip01(_metric(candidate, "vocal_transition_score")),
        0.85 * _clip01(_metric(candidate, "vocal_dropout_score")),
        0.65 * _clip01(_metric(candidate, "vocal_reentry_score")),
    )
    groove = max(
        _clip01(_metric(candidate, "immediate_groove_start_score")),
        _clip01(_metric(candidate, "sustained_full_groove_score")),
        _clip01(_metric(candidate, "buildup_drop_score")),
        _clip01(_metric(candidate, "drop_impact_score")),
        _clip01(_metric(candidate, "kick_reentry_score")),
        _clip01(_metric(candidate, "post_drop_density")),
    )
    knee_quality = _clip01(_metric(candidate, "visual_onset_knee_quality"))
    knee_used = _metric(candidate, "visual_onset_knee_used") > 0.0
    center_quality = _clip01(_metric(candidate, "centerline_boundary_quality"))
    center_used = bool(candidate.get("centerline_boundary_used")) or _metric(candidate, "centerline_boundary_used") > 0.0
    asd_quality = _clip01(_metric(candidate, "ableton_asd_quality"))
    asd_used = _metric(candidate, "ableton_asd_used") > 0.0
    boundary = max(
        knee_quality if knee_used else 0.55 * knee_quality,
        center_quality if center_used else 0.50 * center_quality,
        asd_quality if asd_used else 0.45 * asd_quality,
    )
    source = _clip01(_metric(candidate, "multistem_source_score"))
    agreement = max(_clip01(_metric(candidate, "multistem_agreement")), min(1.0, role_count / 3.0))
    cluster_bonus = _clip01((0.18 * max(0, int(cluster_size) - 1)) + (0.18 * role_count))
    offset_penalty = _clip01(abs(_metric(candidate, "snap_offset_ms")) / 220.0)
    fake_penalty = _clip01(_metric(candidate, "fake_hit_penalty"))

    positive = (
        (0.18 * original_score)
        + (0.18 * micro)
        + (0.13 * drums)
        + (0.11 * instrumental)
        + (0.10 * bass)
        + (0.08 * vocal)
        + (0.08 * groove)
        + (0.07 * agreement)
        + (0.05 * boundary)
        + (0.04 * source)
        + (0.05 * cluster_bonus)
    )
    negative = (0.10 * fake_penalty) + (0.06 * offset_penalty)
    final = _clip01(positive - negative)

    return {
        "original_score": float(original_score),
        "micro_boundary": float(micro),
        "drums_transient": float(drums),
        "bass_jump": float(bass),
        "instrumental_jump": float(instrumental),
        "vocal_transition": float(vocal),
        "groove_start": float(groove),
        "stem_agreement": float(agreement),
        "boundary_quality": float(boundary),
        "source_strength": float(source),
        "cluster_bonus": float(cluster_bonus),
        "fake_hit_penalty": float(fake_penalty),
        "offset_penalty": float(offset_penalty),
        "negative_penalty": float(negative),
        "final": float(final),
    }


def _ranked_cluster_choice(
    items: Sequence[Tuple[float, Mapping[str, Any]]],
    *,
    radius_sec: float,
) -> Dict[str, Any]:
    candidates = [candidate for _time, candidate in items]
    roles = _role_union(candidates)
    sources = _source_union(candidates)
    scored: List[Tuple[float, float, Mapping[str, Any], Dict[str, float]]] = []
    for time_sec, candidate in items:
        components = drop_score_components(candidate, cluster_size=len(items), cluster_roles=roles)
        rank = _finite_float(candidate.get("rank", candidate.get("handcrafted_rank")), 999.0)
        scored.append((float(components["final"]), rank, candidate, components))
    scored.sort(key=lambda row: (-row[0], row[1], candidate_effective_time(row[2]) or float("inf")))
    _score, _rank, selected, components = scored[0]
    out = dict(selected)
    effective_time = candidate_effective_time(out)
    original_score = _base_score(out)
    original_rank = _finite_float(out.get("rank", out.get("handcrafted_rank")), 0.0)
    original_time = _finite_float(out.get("timestamp"), default=0.0)
    out["pre_pipeline_score"] = float(original_score)
    out["pre_pipeline_rank"] = int(original_rank) if original_rank > 0 else 0
    out["pre_pipeline_timestamp"] = float(original_time) if original_time > 0.0 else None
    if effective_time is not None:
        out["timestamp"] = float(effective_time)
        out["snapped_sec"] = float(effective_time)
        out["microaligned_time"] = float(effective_time)
    out["drop_pipeline_score"] = float(components["final"])
    out["drop_pipeline_score_components"] = dict(components)
    out["drop_pipeline_cluster_size"] = int(len(items))
    out["drop_pipeline_cluster_radius_ms"] = float(radius_sec * 1000.0)
    out["drop_pipeline_cluster_center"] = float(_cluster_center(items))
    out["drop_pipeline_cluster_roles"] = list(roles)
    out["drop_pipeline_cluster_sources"] = list(sources)
    out["drop_pipeline_cluster_alternates"] = [
        {
            "time_sec": float(time_sec),
            "rank": int(_finite_float(candidate.get("rank", candidate.get("handcrafted_rank")), 0.0)),
            "score": float(drop_score_components(candidate, cluster_size=len(items), cluster_roles=roles)["final"]),
            "roles": _candidate_roles(candidate),
        }
        for time_sec, candidate in items
        if candidate is not selected
    ][:6]
    out["confidence_score"] = float(components["final"])
    out["score"] = float(components["final"])
    out["drop_pipeline_cluster_bonus"] = float(components["cluster_bonus"])
    out["drop_pipeline_stem_agreement"] = float(components["stem_agreement"])
    out["drop_pipeline_micro_boundary"] = float(components["micro_boundary"])
    out["drop_pipeline_negative_penalty"] = float(components["negative_penalty"])
    if roles:
        out["multistem_roles"] = list(roles)
    if sources:
        out["multistem_sources"] = list(sources)
    debug = dict(out.get("debug") or {}) if isinstance(out.get("debug"), Mapping) else {}
    debug["drop_pipeline_score"] = float(components["final"])
    debug["drop_pipeline_score_components"] = dict(components)
    out["debug"] = debug
    return out


def cluster_drop_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    radius_sec: float = 0.085,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    valid: List[Tuple[float, Mapping[str, Any]]] = []
    invalid: List[Dict[str, Any]] = []
    for candidate in candidates:
        t = candidate_effective_time(candidate)
        if t is None:
            invalid.append(dict(candidate))
            continue
        valid.append((float(t), candidate))

    clusters: List[List[Tuple[float, Mapping[str, Any]]]] = []
    for time_sec, candidate in sorted(valid, key=lambda row: row[0]):
        best_idx: Optional[int] = None
        best_distance = float("inf")
        for idx, cluster in enumerate(clusters):
            distance = abs(float(time_sec) - _cluster_center(cluster))
            if distance <= float(radius_sec) and distance < best_distance:
                best_idx = idx
                best_distance = distance
        if best_idx is None:
            clusters.append([(float(time_sec), candidate)])
        else:
            clusters[best_idx].append((float(time_sec), candidate))

    selected = [_ranked_cluster_choice(cluster, radius_sec=radius_sec) for cluster in clusters]
    selected.sort(key=lambda candidate: (-_finite_float(candidate.get("drop_pipeline_score")), candidate_effective_time(candidate) or float("inf")))
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = int(rank)
        candidate["handcrafted_rank"] = int(rank)
        candidate["drop_pipeline_rank"] = int(rank)
        candidate["selected"] = False

    return {
        "candidates": selected,
        "invalid_candidates": invalid,
        "summary": {
            "input_count": int(len(candidates)),
            "valid_count": int(len(valid)),
            "invalid_count": int(len(invalid)),
            "cluster_count": int(len(clusters)),
            "output_count": int(len(selected)),
            "deduped_count": int(max(0, len(valid) - len(clusters))),
            "cluster_radius_ms": float(radius_sec * 1000.0),
        },
    }


def run_drop_candidate_pipeline(
    candidates: Sequence[Mapping[str, Any]],
    *,
    cluster_radius_sec: float = 0.085,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    clustered = cluster_drop_candidates(candidates, radius_sec=cluster_radius_sec, limit=limit)
    summary = dict(clustered.get("summary") or {})
    summary["pipeline"] = "drop_candidate_pipeline_v1"
    summary["stages"] = [
        "hydrate_microaligned_candidates",
        "cluster_by_effective_time",
        "score_components",
        "select_cluster_representatives",
        "rank_for_selector",
    ]
    return {
        "ok": True,
        "candidates": clustered.get("candidates", []),
        "invalid_candidates": clustered.get("invalid_candidates", []),
        "summary": summary,
    }
