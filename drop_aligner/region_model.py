from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .groove import FULL_GROOVE_FEATURE_KEYS


DEFAULT_REGION_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_region_detector.pkl"
REGION_MODEL_ENV = "DROP_REGION_MODEL"

REGION_FEATURES: Tuple[str, ...] = (
    "time_ratio",
    "time_after_min_norm",
    "time_to_max_norm",
    "log_time_norm",
    "handcrafted_rank_norm",
    "handcrafted_rank_inv",
    "score",
    "score_delta_from_best",
    "score_percentile",
    "time_order_norm",
    "prev_gap_norm",
    "next_gap_norm",
    "density_1s",
    "density_4s",
    "snap_offset_abs_norm",
    "transient_strength",
    "low_end_jump",
    "post_drop_density",
    "pre_post_ratio_norm",
    "energy_contrast",
    "rhythmic_consistency",
    "drum_onset_spike",
    "rms_jump",
    "spectral_flux_peak",
    "pre_drop_contrast",
    "immediate_groove_start_score",
    "groove_stability",
    "sustained_full_groove_score",
    "pre_low",
    "post_low",
    "pre_rms",
    "post_rms",
    "post_pre_low_delta",
    "post_pre_rms_delta",
)


def _clip01(value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _mapping_get(candidate: Mapping[str, Any], key: str) -> Any:
    if key in candidate and candidate.get(key) not in (None, ""):
        return candidate.get(key)
    for nested_name in ("full_groove", "debug", "drumprint"):
        nested = candidate.get(nested_name)
        if isinstance(nested, Mapping) and nested.get(key) not in (None, ""):
            return nested.get(key)
    return None


def _get(candidate: Any, key: str, default: float = 0.0) -> float:
    if isinstance(candidate, Mapping):
        value = _mapping_get(candidate, key)
        return _finite_float(value, default)
    if hasattr(candidate, key):
        value = getattr(candidate, key)
        if value not in (None, ""):
            return _finite_float(value, default)
    for nested_name in ("full_groove", "debug", "drumprint"):
        nested = getattr(candidate, nested_name, None)
        if isinstance(nested, Mapping) and nested.get(key) not in (None, ""):
            return _finite_float(nested.get(key), default)
    return float(default)


def _norm_rank(value: Any, cap: float = 100.0) -> float:
    rank = _finite_float(value, default=cap)
    if rank <= 0.0:
        rank = cap
    return _clip01((rank - 1.0) / max(1.0, cap - 1.0))


def _rank_inv(value: Any, cap: float = 100.0) -> float:
    return 1.0 - _norm_rank(value, cap=cap)


def candidate_region_time(candidate: Any) -> Optional[float]:
    if isinstance(candidate, Mapping):
        for key in ("coarse_timestamp", "time_sec", "timestamp", "snapped_sec"):
            value = _optional_float(candidate.get(key))
            if value is not None and value > 0.0:
                return value
        return None
    for key in ("time_sec", "coarse_timestamp", "snapped_sec"):
        if hasattr(candidate, key):
            value = _optional_float(getattr(candidate, key))
            if value is not None and value > 0.0:
                return value
    return None


def _candidate_score(candidate: Any) -> float:
    if isinstance(candidate, Mapping):
        return _clip01(_mapping_get(candidate, "confidence_score") or _mapping_get(candidate, "score") or 0.0)
    return _clip01(getattr(candidate, "score", 0.0))


def _context(candidates: Sequence[Any], duration_sec: float) -> Dict[str, Any]:
    valid: List[Tuple[int, Any, float, float]] = []
    for index, candidate in enumerate(candidates):
        t = candidate_region_time(candidate)
        if t is None:
            continue
        valid.append((index, candidate, float(t), _candidate_score(candidate)))
    scores = [row[3] for row in valid]
    times = [row[2] for row in valid]
    sorted_scores = sorted(scores, reverse=True)
    sorted_times = sorted(times)
    return {
        "valid": valid,
        "duration_sec": max(1.0, float(duration_sec)),
        "best_score": max(scores, default=0.0),
        "sorted_scores": sorted_scores,
        "sorted_times": sorted_times,
    }


def candidate_feature_dict(candidate: Any, *, time_sec: float, context: Mapping[str, Any]) -> Dict[str, float]:
    duration = max(1.0, float(context.get("duration_sec", 1.0)))
    max_search_time = max(1.0, duration * 0.70)
    score = _candidate_score(candidate)
    sorted_scores = list(context.get("sorted_scores") or [])
    if len(sorted_scores) <= 1:
        score_percentile = 1.0
    else:
        lower_or_equal = sum(1 for value in sorted_scores if float(value) <= score)
        score_percentile = _clip01((lower_or_equal - 1.0) / max(1.0, len(sorted_scores) - 1.0))

    sorted_times = list(context.get("sorted_times") or [])
    time_order = 0
    for idx, value in enumerate(sorted_times):
        if abs(float(value) - float(time_sec)) <= 1e-6:
            time_order = idx
            break
    time_order_norm = 0.0 if len(sorted_times) <= 1 else _clip01(time_order / max(1.0, len(sorted_times) - 1.0))
    prev_gap = float("inf") if time_order <= 0 else float(time_sec) - float(sorted_times[time_order - 1])
    next_gap = float("inf") if time_order >= len(sorted_times) - 1 else float(sorted_times[time_order + 1]) - float(time_sec)
    finite_prev_gap = 8.0 if not math.isfinite(prev_gap) else max(0.0, prev_gap)
    finite_next_gap = 8.0 if not math.isfinite(next_gap) else max(0.0, next_gap)
    density_1s = sum(1 for value in sorted_times if abs(float(value) - float(time_sec)) <= 1.0)
    density_4s = sum(1 for value in sorted_times if abs(float(value) - float(time_sec)) <= 4.0)
    rank = _get(candidate, "handcrafted_rank", _get(candidate, "rank", 100.0))
    pre_low = _clip01(_get(candidate, "pre_low"))
    post_low = _clip01(_get(candidate, "post_low"))
    pre_rms = _clip01(_get(candidate, "pre_rms"))
    post_rms = _clip01(_get(candidate, "post_rms"))

    out = {
        "time_ratio": _clip01(float(time_sec) / duration),
        "time_after_min_norm": _clip01((float(time_sec) - 4.0) / max(1.0, max_search_time - 4.0)),
        "time_to_max_norm": _clip01((max_search_time - float(time_sec)) / max(1.0, max_search_time - 4.0)),
        "log_time_norm": _clip01(math.log1p(max(0.0, float(time_sec))) / math.log1p(max(2.0, max_search_time))),
        "handcrafted_rank_norm": _norm_rank(rank),
        "handcrafted_rank_inv": _rank_inv(rank),
        "score": score,
        "score_delta_from_best": _clip01(float(context.get("best_score", 0.0)) - score),
        "score_percentile": float(score_percentile),
        "time_order_norm": float(time_order_norm),
        "prev_gap_norm": _clip01(finite_prev_gap / 8.0),
        "next_gap_norm": _clip01(finite_next_gap / 8.0),
        "density_1s": _clip01((density_1s - 1.0) / 6.0),
        "density_4s": _clip01((density_4s - 1.0) / 16.0),
        "snap_offset_abs_norm": _clip01(abs(_get(candidate, "snap_offset", _get(candidate, "snap_offset_sec"))) / 0.25),
        "transient_strength": _clip01(_get(candidate, "transient_strength")),
        "low_end_jump": _clip01(_get(candidate, "low_end_jump")),
        "post_drop_density": _clip01(_get(candidate, "post_drop_density")),
        "pre_post_ratio_norm": _clip01(math.log1p(max(0.0, _get(candidate, "pre_post_energy_ratio"))) / math.log1p(12.0)),
        "energy_contrast": _clip01(_get(candidate, "energy_contrast")),
        "rhythmic_consistency": _clip01(_get(candidate, "rhythmic_consistency")),
        "pre_low": pre_low,
        "post_low": post_low,
        "pre_rms": pre_rms,
        "post_rms": post_rms,
        "post_pre_low_delta": _clip01((post_low - pre_low + 0.5) / 1.5),
        "post_pre_rms_delta": _clip01((post_rms - pre_rms + 0.5) / 1.5),
    }
    for name in FULL_GROOVE_FEATURE_KEYS:
        out[name] = _clip01(_get(candidate, name))
    return out


def candidate_feature_rows(candidates: Sequence[Any], *, duration_sec: float) -> List[Dict[str, Any]]:
    ctx = _context(candidates, duration_sec)
    rows: List[Dict[str, Any]] = []
    for index, candidate, time_sec, _score in list(ctx.get("valid") or []):
        features = candidate_feature_dict(candidate, time_sec=time_sec, context=ctx)
        rows.append(
            {
                "index": int(index),
                "candidate": candidate,
                "time_sec": float(time_sec),
                "features": features,
                "vector": [float(features.get(name, 0.0)) for name in REGION_FEATURES],
            }
        )
    return rows


def candidate_feature_matrix(candidates: Sequence[Any], *, duration_sec: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows = candidate_feature_rows(candidates, duration_sec=duration_sec)
    if not rows:
        return np.zeros((0, len(REGION_FEATURES)), dtype=np.float64), []
    return np.asarray([row["vector"] for row in rows], dtype=np.float64), rows


def resolve_model_path(model_path: Optional[str] = None) -> Path:
    env_path = os.environ.get(REGION_MODEL_ENV)
    if model_path:
        return Path(model_path).expanduser()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_REGION_MODEL_PATH


def load_region_model_payload(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = resolve_model_path(model_path)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid drop region model payload: {path}")
    payload.setdefault("feature_names", list(REGION_FEATURES))
    payload["path"] = str(path)
    return payload


def save_region_model_payload(payload: Mapping[str, Any], model_path: Optional[str] = None) -> str:
    path = resolve_model_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh)
    return str(path)


def predict_region_errors(
    payload: Mapping[str, Any],
    candidates: Sequence[Any],
    *,
    duration_sec: float,
) -> List[Dict[str, Any]]:
    x, rows = candidate_feature_matrix(candidates, duration_sec=duration_sec)
    if x.size == 0:
        return []
    model = payload["model"]
    pred = np.asarray(model.predict(x), dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for row, value in zip(rows, pred):
        predicted_error = max(0.0, _finite_float(value, default=999.0))
        out.append(
            {
                "index": int(row["index"]),
                "candidate": row["candidate"],
                "time_sec": float(row["time_sec"]),
                "predicted_abs_error_sec": float(predicted_error),
                "region_score": float(1.0 / (1.0 + predicted_error)),
            }
        )
    return out


def rank_region_candidates(
    candidates: Sequence[Any],
    *,
    duration_sec: float,
    model_path: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    payload = load_region_model_payload(model_path)
    if payload is None:
        return None
    predictions = predict_region_errors(payload, candidates, duration_sec=duration_sec)
    predictions.sort(
        key=lambda row: (
            float(row["predicted_abs_error_sec"]),
            _get(row["candidate"], "handcrafted_rank", _get(row["candidate"], "rank", 999.0)),
            float(row["time_sec"]),
        )
    )
    for rank, row in enumerate(predictions, start=1):
        row["region_rank"] = int(rank)
        row["model_path"] = str(payload.get("path", ""))
        row["training_rows"] = int(payload.get("training_rows", 0) or 0)
        row["correction_rows"] = int(payload.get("correction_rows", 0) or 0)
    return predictions
