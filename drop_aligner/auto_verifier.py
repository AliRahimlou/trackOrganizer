from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate_chooser import CHOOSER_FEATURES, candidate_effective_time, candidate_feature_rows


DEFAULT_AUTO_VERIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_auto_verifier.pkl"

AUTO_VERIFIER_EXTRA_FEATURES: Tuple[str, ...] = (
    "candidate_time_norm",
    "candidate_rank_inv_50",
    "is_top_1",
    "is_top_3",
    "is_top_5",
    "score_gap_to_best",
    "pipeline_rank_inv",
    "cluster_size_norm",
    "cluster_has_alternates",
)
AUTO_VERIFIER_FEATURES: Tuple[str, ...] = (*CHOOSER_FEATURES, *AUTO_VERIFIER_EXTRA_FEATURES)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_finite_float(value), 0.0, 1.0))


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    for key in ("drop_pipeline_score", "confidence_score", "score"):
        if candidate.get(key) not in (None, ""):
            return _clip01(candidate.get(key))
    return 0.0


def _candidate_rank(candidate: Mapping[str, Any], fallback: int) -> int:
    for key in ("rank", "drop_pipeline_rank", "handcrafted_rank"):
        value = _finite_float(candidate.get(key), 0.0)
        if value > 0:
            return int(value)
    return int(fallback)


def _extra_features(candidate: Mapping[str, Any], *, time_sec: float, rank: int, best_score: float, max_time: float) -> Dict[str, float]:
    score = _candidate_score(candidate)
    pipeline_rank = _finite_float(candidate.get("drop_pipeline_rank"), float(rank))
    cluster_size = _finite_float(candidate.get("drop_pipeline_cluster_size"), 1.0)
    alternates = candidate.get("drop_pipeline_cluster_alternates")
    return {
        "candidate_time_norm": _clip01(float(time_sec) / max(1.0, float(max_time))),
        "candidate_rank_inv_50": _clip01(1.0 - ((max(1, int(rank)) - 1.0) / 49.0)),
        "is_top_1": 1.0 if int(rank) <= 1 else 0.0,
        "is_top_3": 1.0 if int(rank) <= 3 else 0.0,
        "is_top_5": 1.0 if int(rank) <= 5 else 0.0,
        "score_gap_to_best": _clip01(float(best_score) - float(score)),
        "pipeline_rank_inv": _clip01(1.0 - ((max(1.0, pipeline_rank) - 1.0) / 11.0)),
        "cluster_size_norm": _clip01(cluster_size / 5.0),
        "cluster_has_alternates": 1.0 if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes)) and len(alternates) > 0 else 0.0,
    }


def verifier_feature_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = AUTO_VERIFIER_FEATURES,
) -> List[Dict[str, Any]]:
    base_rows = candidate_feature_rows(candidates, feature_names=CHOOSER_FEATURES)
    valid_times = [float(row["time_sec"]) for row in base_rows]
    max_time = max(valid_times) if valid_times else 1.0
    best_score = max((_candidate_score(row["candidate"]) for row in base_rows), default=0.0)
    out: List[Dict[str, Any]] = []
    for position, row in enumerate(base_rows, start=1):
        candidate = row["candidate"]
        rank = _candidate_rank(candidate, position)
        features = dict(row["features"])
        features.update(
            _extra_features(
                candidate,
                time_sec=float(row["time_sec"]),
                rank=rank,
                best_score=float(best_score),
                max_time=float(max_time),
            )
        )
        out.append(
            {
                "index": int(row["index"]),
                "candidate": candidate,
                "time_sec": float(row["time_sec"]),
                "features": features,
                "vector": [float(features.get(str(name), 0.0)) for name in feature_names],
            }
        )
    return out


def verifier_feature_matrix(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = AUTO_VERIFIER_FEATURES,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows = verifier_feature_rows(candidates, feature_names=feature_names)
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float64), []
    return np.asarray([row["vector"] for row in rows], dtype=np.float64), rows


def save_auto_verifier_payload(payload: Mapping[str, Any], model_path: Optional[str] = None) -> str:
    path = Path(model_path).expanduser() if model_path else DEFAULT_AUTO_VERIFIER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh)
    return str(path)


def load_auto_verifier_payload(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = Path(model_path).expanduser() if model_path else DEFAULT_AUTO_VERIFIER_PATH
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"Invalid auto verifier payload: {path}")
    payload.setdefault("feature_names", list(AUTO_VERIFIER_FEATURES))
    payload["path"] = str(path)
    return payload


def _positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return proba[:, classes.index(1)]
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, -1]
    return np.asarray(model.predict(x), dtype=np.float64)


def predict_auto_verifier_candidates(
    payload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    feature_names = [str(name) for name in payload.get("feature_names") or AUTO_VERIFIER_FEATURES]
    x, rows = verifier_feature_matrix(candidates, feature_names=feature_names)
    if x.size == 0 or not rows:
        return []
    classifier = payload["classifier"]
    regressor = payload.get("regressor")
    probabilities = _positive_probability(classifier, x)
    if regressor is not None:
        raw_errors = np.asarray(regressor.predict(x), dtype=np.float64)
        probability_errors = np.maximum(0.0, 1.0 - probabilities) * 0.100
        predicted_errors = np.minimum(np.maximum(0.0, raw_errors), probability_errors)
    else:
        predicted_errors = np.maximum(0.0, 1.0 - probabilities) * 0.100
    out: List[Dict[str, Any]] = []
    for row, probability, predicted_error in zip(rows, probabilities, predicted_errors):
        out.append(
            {
                "index": int(row["index"]),
                "candidate": row["candidate"],
                "time_sec": float(row["time_sec"]),
                "p_within_25ms": _clip01(probability),
                "predicted_abs_error_sec": max(0.0, _finite_float(predicted_error, 999.0)),
                "model_path": str(payload.get("path", "")),
                "model_type": str(payload.get("model_type", "auto_verifier")),
            }
        )
    return out


def predict_auto_verifier(
    payload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_time: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    predictions = predict_auto_verifier_candidates(payload, candidates)
    if not predictions:
        return None
    if selected_time is not None:
        chosen_index = int(
            np.argmin(
                np.asarray(
                    [abs(float(row["time_sec"]) - float(selected_time)) for row in predictions],
                    dtype=np.float64,
                )
            )
        )
    else:
        chosen_index = int(
            max(
                range(len(predictions)),
                key=lambda idx: (
                    float(predictions[idx].get("p_within_25ms", 0.0)),
                    -float(predictions[idx].get("predicted_abs_error_sec", 999.0)),
                    -int(predictions[idx].get("index", 999999)),
                ),
            )
        )
    row = dict(predictions[chosen_index])
    row.pop("candidate", None)
    return row
