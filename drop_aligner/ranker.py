from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .drumprint import DRUMPRINT_FEATURE_KEYS
from .groove import FULL_GROOVE_FEATURE_KEYS
from .microalign import MICROALIGN_FEATURE_KEYS


MODEL_FEATURES = [
    "transient_strength",
    "low_end_jump",
    "post_drop_density",
    "pre_post_energy_ratio",
    "rhythmic_consistency",
    "snap_offset",
    "timestamp",
    "confidence_score",
    *FULL_GROOVE_FEATURE_KEYS,
    *DRUMPRINT_FEATURE_KEYS,
    *MICROALIGN_FEATURE_KEYS,
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_ranker.pkl"


def _get(candidate: Any, *names: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        for name in names:
            if name in candidate and candidate[name] is not None:
                return candidate[name]
        for nested_name in ("drumprint", "full_groove", "microalign"):
            nested = candidate.get(nested_name)
            if isinstance(nested, Mapping):
                for name in names:
                    if name in nested and nested[name] is not None:
                        return nested[name]
        return default
    for name in names:
        if hasattr(candidate, name):
            value = getattr(candidate, name)
            if value is not None:
                return value
    for nested_name in ("drumprint", "full_groove", "microalign"):
        nested = getattr(candidate, nested_name, None)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]
    return default


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def candidate_timestamp(candidate: Any) -> Optional[float]:
    value = _get(candidate, "timestamp", "snapped_sec", "time_sec", "coarse_timestamp")
    if value is None:
        return None
    return _finite_float(value)


def candidate_feature_dict(candidate: Any) -> Dict[str, float]:
    timestamp = candidate_timestamp(candidate)
    features = {
        "transient_strength": _finite_float(_get(candidate, "transient_strength")),
        "low_end_jump": _finite_float(_get(candidate, "low_end_jump")),
        "post_drop_density": _finite_float(_get(candidate, "post_drop_density")),
        "pre_post_energy_ratio": _finite_float(_get(candidate, "pre_post_energy_ratio")),
        "rhythmic_consistency": _finite_float(_get(candidate, "rhythmic_consistency")),
        "snap_offset": _finite_float(_get(candidate, "snap_offset", "snap_offset_sec")),
        "timestamp": _finite_float(timestamp),
        "confidence_score": _finite_float(_get(candidate, "confidence_score", "score")),
    }
    for name in FULL_GROOVE_FEATURE_KEYS:
        features[name] = _finite_float(_get(candidate, name))
    for name in DRUMPRINT_FEATURE_KEYS:
        features[name] = _finite_float(_get(candidate, name))
    for name in MICROALIGN_FEATURE_KEYS:
        features[name] = _finite_float(_get(candidate, name))
    return features


def candidate_feature_vector(candidate: Any, feature_names: Sequence[str] = MODEL_FEATURES) -> List[float]:
    features = candidate_feature_dict(candidate)
    return [_finite_float(features.get(name)) for name in feature_names]


def candidate_feature_matrix(candidates: Iterable[Any], feature_names: Sequence[str] = MODEL_FEATURES) -> np.ndarray:
    rows = [candidate_feature_vector(candidate, feature_names) for candidate in candidates]
    return np.asarray(rows, dtype=np.float64)


def resolve_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        return Path(model_path).expanduser()
    return DEFAULT_MODEL_PATH


def load_ranker_payload(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = resolve_model_path(model_path)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid ranker model payload: {path}")
    payload.setdefault("feature_names", list(MODEL_FEATURES))
    payload["path"] = str(path)
    return payload


def save_ranker_payload(payload: Mapping[str, Any], model_path: Optional[str] = None) -> str:
    path = resolve_model_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh)
    return str(path)


def predict_candidate_distances(payload: Mapping[str, Any], candidates: Sequence[Any]) -> List[float]:
    feature_names = list(payload.get("feature_names") or MODEL_FEATURES)
    x = candidate_feature_matrix(candidates, feature_names)
    if x.size == 0:
        return []
    pred = np.asarray(payload["model"].predict(x), dtype=np.float64)
    return [max(0.0, _finite_float(value)) for value in pred]
