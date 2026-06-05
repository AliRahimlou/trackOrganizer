from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    nested = candidate.get("microalign")
    if isinstance(nested, Mapping):
        value = nested.get("microaligned_time")
        if value is not None:
            try:
                out = float(value)
                if out > 0.0:
                    return out
            except (TypeError, ValueError):
                pass
    for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec", "coarse_timestamp"):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            out = float(value)
            if out > 0.0:
                return out
        except (TypeError, ValueError):
            continue
    return None


def closest_candidate_to_pick(candidates: Sequence[Mapping[str, Any]], user_pick: float) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_delta = float("inf")
    for candidate in candidates:
        t = _candidate_time(candidate)
        if t is None:
            continue
        delta = abs(float(user_pick) - t)
        if delta < best_delta:
            best_delta = delta
            best = dict(candidate)
            best["abs_delta_to_user_pick"] = float(delta)
            best["signed_delta_to_user_pick"] = float(float(user_pick) - t)
    return best


def log_correction(
    *,
    track: str,
    ai_pick: float,
    user_pick: float,
    features: Optional[Dict[str, Any]] = None,
    top_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    selected_candidate: Optional[Mapping[str, Any]] = None,
    selected_by: Optional[str] = None,
    confidence_tier: Optional[str] = None,
    reviewed_from: Optional[str] = None,
    log_path: str = "drop_corrections.jsonl",
) -> str:
    candidates = [dict(candidate) for candidate in (top_candidates or [])]
    selected = dict(selected_candidate or {})
    feature_map = dict(features or {})
    selected_by_value = selected_by or selected.get("selected_by") or feature_map.get("selected_by") or ""
    confidence_tier_value = confidence_tier or feature_map.get("confidence_tier") or selected.get("confidence_tier") or "UNKNOWN"
    delta = float(user_pick) - float(ai_pick)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": str(track),
        "final_ai_pick": float(ai_pick),
        "ai_pick": float(ai_pick),
        "user_pick": float(user_pick),
        "delta": float(delta),
        "selected_by": str(selected_by_value),
        "confidence_tier": str(confidence_tier_value),
        "reviewed_from": str(reviewed_from or ""),
        "top_10_candidates": candidates,
        "selected_candidate": selected,
        "closest_candidate_to_user_pick": closest_candidate_to_pick(candidates, float(user_pick)),
        "features": feature_map,
    }
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return str(path)
