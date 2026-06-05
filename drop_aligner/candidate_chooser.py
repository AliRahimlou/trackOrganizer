from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CHOOSER_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_candidate_chooser.pkl"
CHOOSER_MODEL_ENV = "DROP_CANDIDATE_CHOOSER_MODEL"
_PAYLOAD_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}

CHOOSER_FEATURES: Tuple[str, ...] = (
    "rank_norm",
    "rank_inv",
    "handcrafted_rank_norm",
    "handcrafted_rank_inv",
    "model_rank_norm",
    "model_rank_inv",
    "model_rank_missing",
    "score",
    "score_delta_from_best",
    "score_percentile",
    "candidate_margin",
    "time_order_norm",
    "temporal_position_norm",
    "prev_gap_norm",
    "next_gap_norm",
    "density_500ms",
    "density_1500ms",
    "transient_strength",
    "low_end_jump",
    "post_drop_density",
    "pre_post_ratio_norm",
    "energy_contrast",
    "rhythmic_consistency",
    "snap_offset_abs_norm",
    "drum_onset_spike",
    "rms_jump",
    "spectral_flux_peak",
    "pre_drop_contrast",
    "immediate_groove_start_score",
    "groove_stability",
    "sustained_full_groove_score",
    "fake_hit_penalty",
    "drumprint_pattern_score",
    "post_drop_pattern_stability",
    "later_drop_match_score",
    "self_similarity_boundary_score",
    "micro_confidence",
    "snap_offset_ms_abs_norm",
    "attack_peak_strength",
    "attack_cleanliness",
    "sustained_after_attack",
    "zero_crossing_quality",
    "visual_onset_knee_quality",
    "visual_onset_knee_used",
    "visual_onset_knee_offset_abs_norm",
    "ableton_asd_quality",
    "ableton_asd_used",
    "ableton_asd_offset_abs_norm",
    "centerline_boundary_quality",
    "centerline_boundary_used",
    "multistem_agreement",
    "multistem_source_score",
    "multistem_role_count_norm",
    "multistem_saved_used",
    "multistem_drums_used",
    "multistem_instrumental_used",
    "multistem_vocals_used",
    "drums_transient_score",
    "bass_low_jump_score",
    "inst_energy_jump_score",
    "vocal_transition_score",
    "vocal_dropout_score",
    "vocal_reentry_score",
    "drop_pipeline_score",
    "drop_pipeline_cluster_bonus",
    "drop_pipeline_stem_agreement",
    "drop_pipeline_micro_boundary",
    "drop_pipeline_negative_penalty",
    "structure_candidate_used",
    "structure_first_drop_used",
    "structure_second_drop_used",
    "structure_on_one",
    "structure_phrase_prior",
    "structure_pre_space",
    "structure_mini_fill_penalty",
    "structure_clock_bar_norm",
    "saved_candidate_source",
    "saved_selected_used",
    "saved_rejected_used",
    "saved_rank_inv",
    "saved_model_rank_inv",
    "saved_score",
    "saved_micro_confidence",
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


def _get(candidate: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if key in candidate and candidate.get(key) not in (None, ""):
        return _finite_float(candidate.get(key), default)
    for nested_name in ("microalign", "drumprint", "full_groove", "structure_components", "debug"):
        nested = candidate.get(nested_name)
        if isinstance(nested, Mapping) and nested.get(key) not in (None, ""):
            return _finite_float(nested.get(key), default)
    return float(default)


def _has_role(candidate: Mapping[str, Any], role: str) -> float:
    roles = candidate.get("multistem_roles")
    if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)):
        return 1.0 if str(role) in {str(item) for item in roles} else 0.0
    return 0.0


def _has_source(candidate: Mapping[str, Any], source: str) -> float:
    sources = candidate.get("multistem_source_names")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        return 1.0 if str(source) in {str(item) for item in sources} else 0.0
    return 0.0


def _role_count_norm(candidate: Mapping[str, Any]) -> float:
    roles = candidate.get("multistem_roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        return 0.0
    return _clip01(len({str(item) for item in roles if str(item)}) / 4.0)


def _norm_rank(value: Any, cap: float = 10.0) -> float:
    rank = _finite_float(value, default=cap)
    if rank <= 0.0:
        rank = cap
    return _clip01((rank - 1.0) / max(1.0, cap - 1.0))


def _rank_inv(value: Any, cap: float = 10.0) -> float:
    return 1.0 - _norm_rank(value, cap=cap)


def _offset_norm_ms(value: Any, cap_ms: float) -> float:
    return _clip01(abs(_finite_float(value, default=0.0)) / max(1.0, cap_ms))


def candidate_effective_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for nested_name in ("microalign",):
        nested = candidate.get(nested_name)
        if isinstance(nested, Mapping):
            value = _optional_float(nested.get("microaligned_time"))
            if value is not None and value > 0.0:
                return value
    for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec", "coarse_timestamp"):
        value = _optional_float(candidate.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    return _clip01(_get(candidate, "confidence_score", _get(candidate, "score", 0.0)))


def _context(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid: List[Tuple[int, Mapping[str, Any], float, float]] = []
    for index, candidate in enumerate(candidates):
        t = candidate_effective_time(candidate)
        if t is None:
            continue
        valid.append((index, candidate, float(t), _candidate_score(candidate)))
    scores = [row[3] for row in valid]
    times = [row[2] for row in valid]
    best_score = max(scores, default=0.0)
    sorted_scores = sorted(scores, reverse=True)
    margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else (sorted_scores[0] if sorted_scores else 0.0)
    sorted_times = sorted(times)
    time_span = max(1e-6, (max(times) - min(times))) if times else 1.0
    return {
        "valid": valid,
        "best_score": float(best_score),
        "candidate_margin": float(max(0.0, margin)),
        "sorted_scores": sorted_scores,
        "sorted_times": sorted_times,
        "min_time": min(times) if times else 0.0,
        "time_span": float(time_span),
    }


def candidate_feature_dict(
    candidate: Mapping[str, Any],
    *,
    time_sec: float,
    context: Mapping[str, Any],
) -> Dict[str, float]:
    rank = candidate.get("rank", candidate.get("handcrafted_rank", 10))
    handcrafted_rank = candidate.get("handcrafted_rank", candidate.get("rank", 10))
    model_rank = candidate.get("model_rank", 0)
    model_rank_value = _finite_float(model_rank, default=0.0)
    model_rank_missing = 1.0 if model_rank_value <= 0.0 else 0.0
    if model_rank_value <= 0.0:
        model_rank_value = 10.0
    selected_by = str(candidate.get("selected_by") or "")
    structure_role = str(candidate.get("structure_role") or "")
    structure_components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
    structure_clock_bar = _finite_float(
        structure_components.get("clock_bar", candidate.get("structure_clock_bar", candidate.get("clock_bar", 0.0))),
        default=0.0,
    )

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
    temporal_position = _clip01((float(time_sec) - float(context.get("min_time", time_sec))) / max(1e-6, float(context.get("time_span", 1.0))))
    prev_gap = float("inf") if time_order <= 0 else float(time_sec) - float(sorted_times[time_order - 1])
    next_gap = float("inf") if time_order >= len(sorted_times) - 1 else float(sorted_times[time_order + 1]) - float(time_sec)
    finite_prev_gap = 4.0 if not math.isfinite(prev_gap) else max(0.0, prev_gap)
    finite_next_gap = 4.0 if not math.isfinite(next_gap) else max(0.0, next_gap)
    density_500ms = sum(1 for value in sorted_times if abs(float(value) - float(time_sec)) <= 0.5)
    density_1500ms = sum(1 for value in sorted_times if abs(float(value) - float(time_sec)) <= 1.5)

    return {
        "rank_norm": _norm_rank(rank),
        "rank_inv": _rank_inv(rank),
        "handcrafted_rank_norm": _norm_rank(handcrafted_rank),
        "handcrafted_rank_inv": _rank_inv(handcrafted_rank),
        "model_rank_norm": _norm_rank(model_rank_value),
        "model_rank_inv": _rank_inv(model_rank_value),
        "model_rank_missing": model_rank_missing,
        "score": score,
        "score_delta_from_best": _clip01(float(context.get("best_score", 0.0)) - score),
        "score_percentile": float(score_percentile),
        "candidate_margin": _clip01(float(context.get("candidate_margin", 0.0)) / 0.20),
        "time_order_norm": float(time_order_norm),
        "temporal_position_norm": float(temporal_position),
        "prev_gap_norm": _clip01(finite_prev_gap / 4.0),
        "next_gap_norm": _clip01(finite_next_gap / 4.0),
        "density_500ms": _clip01((density_500ms - 1.0) / 4.0),
        "density_1500ms": _clip01((density_1500ms - 1.0) / 8.0),
        "transient_strength": _clip01(_get(candidate, "transient_strength")),
        "low_end_jump": _clip01(_get(candidate, "low_end_jump")),
        "post_drop_density": _clip01(_get(candidate, "post_drop_density")),
        "pre_post_ratio_norm": _clip01(math.log1p(max(0.0, _get(candidate, "pre_post_energy_ratio"))) / math.log1p(12.0)),
        "energy_contrast": _clip01(_get(candidate, "energy_contrast")),
        "rhythmic_consistency": _clip01(_get(candidate, "rhythmic_consistency")),
        "snap_offset_abs_norm": _clip01(abs(_get(candidate, "snap_offset", _get(candidate, "snap_offset_sec"))) / 0.20),
        "drum_onset_spike": _clip01(_get(candidate, "drum_onset_spike")),
        "rms_jump": _clip01(_get(candidate, "rms_jump")),
        "spectral_flux_peak": _clip01(_get(candidate, "spectral_flux_peak")),
        "pre_drop_contrast": _clip01(_get(candidate, "pre_drop_contrast")),
        "immediate_groove_start_score": _clip01(_get(candidate, "immediate_groove_start_score")),
        "groove_stability": _clip01(_get(candidate, "groove_stability")),
        "sustained_full_groove_score": _clip01(_get(candidate, "sustained_full_groove_score")),
        "fake_hit_penalty": _clip01(_get(candidate, "fake_hit_penalty")),
        "drumprint_pattern_score": _clip01(_get(candidate, "drumprint_pattern_score")),
        "post_drop_pattern_stability": _clip01(_get(candidate, "post_drop_pattern_stability")),
        "later_drop_match_score": _clip01(_get(candidate, "later_drop_match_score")),
        "self_similarity_boundary_score": _clip01(_get(candidate, "self_similarity_boundary_score")),
        "micro_confidence": _clip01(_get(candidate, "micro_confidence")),
        "snap_offset_ms_abs_norm": _offset_norm_ms(_get(candidate, "snap_offset_ms"), 220.0),
        "attack_peak_strength": _clip01(_get(candidate, "attack_peak_strength")),
        "attack_cleanliness": _clip01(_get(candidate, "attack_cleanliness")),
        "sustained_after_attack": _clip01(_get(candidate, "sustained_after_attack")),
        "zero_crossing_quality": _clip01(_get(candidate, "zero_crossing_quality")),
        "visual_onset_knee_quality": _clip01(_get(candidate, "visual_onset_knee_quality")),
        "visual_onset_knee_used": 1.0 if _get(candidate, "visual_onset_knee_used") > 0.0 else 0.0,
        "visual_onset_knee_offset_abs_norm": _offset_norm_ms(_get(candidate, "visual_onset_knee_offset_ms"), 220.0),
        "ableton_asd_quality": _clip01(_get(candidate, "ableton_asd_quality")),
        "ableton_asd_used": 1.0 if _get(candidate, "ableton_asd_used") > 0.0 else 0.0,
        "ableton_asd_offset_abs_norm": _offset_norm_ms(_get(candidate, "ableton_asd_offset_ms"), 220.0),
        "centerline_boundary_quality": _clip01(_get(candidate, "centerline_boundary_quality")),
        "centerline_boundary_used": 1.0 if _get(candidate, "centerline_boundary_used") > 0.0 else 0.0,
        "multistem_agreement": _clip01(_get(candidate, "multistem_agreement")),
        "multistem_source_score": _clip01(_get(candidate, "multistem_source_score")),
        "multistem_role_count_norm": _role_count_norm(candidate),
        "multistem_saved_used": _has_role(candidate, "saved"),
        "multistem_drums_used": _has_role(candidate, "drums"),
        "multistem_instrumental_used": _has_role(candidate, "instrumental"),
        "multistem_vocals_used": _has_role(candidate, "vocals"),
        "drums_transient_score": _clip01(_get(candidate, "drums_transient_score")),
        "bass_low_jump_score": _clip01(_get(candidate, "bass_low_jump_score")),
        "inst_energy_jump_score": _clip01(_get(candidate, "inst_energy_jump_score")),
        "vocal_transition_score": _clip01(_get(candidate, "vocal_transition_score")),
        "vocal_dropout_score": _clip01(_get(candidate, "vocal_dropout_score")),
        "vocal_reentry_score": _clip01(_get(candidate, "vocal_reentry_score")),
        "drop_pipeline_score": _clip01(_get(candidate, "drop_pipeline_score")),
        "drop_pipeline_cluster_bonus": _clip01(_get(candidate, "drop_pipeline_cluster_bonus")),
        "drop_pipeline_stem_agreement": _clip01(_get(candidate, "drop_pipeline_stem_agreement")),
        "drop_pipeline_micro_boundary": _clip01(_get(candidate, "drop_pipeline_micro_boundary")),
        "drop_pipeline_negative_penalty": _clip01(_get(candidate, "drop_pipeline_negative_penalty")),
        "structure_candidate_used": 1.0 if selected_by == "structure_map" or bool(structure_role) else 0.0,
        "structure_first_drop_used": 1.0 if structure_role == "first_drop" else 0.0,
        "structure_second_drop_used": 1.0 if structure_role == "second_drop" else 0.0,
        "structure_on_one": 1.0 if bool(structure_components.get("on_one")) else 0.0,
        "structure_phrase_prior": _clip01(_get(candidate, "phrase_prior")),
        "structure_pre_space": _clip01(_get(candidate, "pre_space")),
        "structure_mini_fill_penalty": _clip01(_get(candidate, "mini_fill_penalty")),
        "structure_clock_bar_norm": _clip01(structure_clock_bar / 129.0),
        "saved_candidate_source": max(_has_role(candidate, "saved"), _has_source(candidate, "saved_candidate")),
        "saved_selected_used": 1.0 if _get(candidate, "saved_selected_used") > 0.0 else 0.0,
        "saved_rejected_used": 1.0 if _get(candidate, "saved_rejected_used") > 0.0 else 0.0,
        "saved_rank_inv": _rank_inv(_get(candidate, "saved_best_rank", 10.0), cap=12.0),
        "saved_model_rank_inv": _rank_inv(_get(candidate, "saved_best_model_rank", 10.0), cap=12.0),
        "saved_score": _clip01(_get(candidate, "saved_best_score")),
        "saved_micro_confidence": _clip01(_get(candidate, "saved_best_micro_confidence")),
    }


def candidate_feature_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = CHOOSER_FEATURES,
) -> List[Dict[str, Any]]:
    ctx = _context(candidates)
    rows: List[Dict[str, Any]] = []
    for index, candidate, time_sec, _score in list(ctx.get("valid") or []):
        features = candidate_feature_dict(candidate, time_sec=time_sec, context=ctx)
        rows.append(
            {
                "index": int(index),
                "candidate": candidate,
                "time_sec": float(time_sec),
                "features": features,
                "vector": [float(features.get(str(name), 0.0)) for name in feature_names],
            }
        )
    return rows


def candidate_feature_matrix(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = CHOOSER_FEATURES,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows = candidate_feature_rows(candidates, feature_names=feature_names)
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float64), []
    return np.asarray([row["vector"] for row in rows], dtype=np.float64), rows


def resolve_model_path(model_path: Optional[str] = None) -> Path:
    env_path = os.environ.get(CHOOSER_MODEL_ENV)
    if model_path:
        return Path(model_path).expanduser()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CHOOSER_MODEL_PATH


def load_candidate_chooser_payload(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = resolve_model_path(model_path)
    if not path.exists():
        return None
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = _PAYLOAD_CACHE.get(cache_key)
    if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
        return cached[2]
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid candidate chooser payload: {path}")
    payload.setdefault("feature_names", list(CHOOSER_FEATURES))
    payload["path"] = str(path)
    _PAYLOAD_CACHE[cache_key] = (int(stat.st_mtime_ns), int(stat.st_size), payload)
    return payload


def save_candidate_chooser_payload(payload: Mapping[str, Any], model_path: Optional[str] = None) -> str:
    path = resolve_model_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh)
    _PAYLOAD_CACHE.pop(str(path.resolve()), None)
    return str(path)


def predict_candidate_errors(payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)) or not feature_names:
        feature_names = CHOOSER_FEATURES
    model = payload["model"]
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None:
        try:
            expected_count = int(expected_features)
        except (TypeError, ValueError):
            expected_count = 0
        if expected_count > 0:
            names = [str(name) for name in feature_names]
            if len(names) > expected_count:
                feature_names = names[:expected_count]
            elif len(names) < expected_count:
                feature_names = [*names, *[f"missing_feature_{idx}" for idx in range(expected_count - len(names))]]
    x, rows = candidate_feature_matrix(candidates, feature_names=[str(name) for name in feature_names])
    if x.size == 0:
        return []
    model_type = str(payload.get("model_type") or "").lower()
    if "pairwise" in model_type:
        n_rows = int(x.shape[0])
        good_threshold = max(0.001, _finite_float(payload.get("good_candidate_threshold_sec"), 0.100))
        usable_threshold = max(good_threshold, _finite_float(payload.get("usable_candidate_threshold_sec"), 0.250))
        if n_rows == 1:
            win_scores = np.asarray([1.0], dtype=np.float64)
            win_margins = np.asarray([1.0], dtype=np.float64)
            comparisons = np.asarray([0], dtype=np.int64)
        else:
            win_totals = np.zeros(n_rows, dtype=np.float64)
            comparisons = np.zeros(n_rows, dtype=np.int64)
            pairs: List[Tuple[int, int]] = []
            pair_vectors: List[np.ndarray] = []
            for left in range(n_rows):
                for right in range(n_rows):
                    if left == right:
                        continue
                    pairs.append((left, right))
                    pair_vectors.append(x[left] - x[right])
            pair_x = np.asarray(pair_vectors, dtype=np.float64)
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(pair_x), dtype=np.float64)
                classes = list(getattr(model, "classes_", []))
                if 1 in classes:
                    pair_pred = proba[:, classes.index(1)]
                elif proba.ndim == 2 and proba.shape[1] >= 2:
                    pair_pred = proba[:, -1]
                else:
                    pair_pred = np.asarray(model.predict(pair_x), dtype=np.float64)
            else:
                pair_pred = np.asarray(model.predict(pair_x), dtype=np.float64)
            for (left, _right), value in zip(pairs, pair_pred):
                win_totals[left] += _clip01(_finite_float(value, default=0.0))
                comparisons[left] += 1
            win_scores = np.divide(win_totals, np.maximum(1, comparisons), dtype=np.float64)
            sorted_scores = np.sort(win_scores)[::-1]
            second_best = float(sorted_scores[1]) if sorted_scores.size > 1 else 0.0
            win_margins = np.asarray([max(0.0, float(score) - second_best) for score in win_scores], dtype=np.float64)
            best_index = int(np.argmax(win_scores))
            for idx in range(n_rows):
                if idx == best_index and sorted_scores.size > 1:
                    win_margins[idx] = max(0.0, float(sorted_scores[0]) - float(sorted_scores[1]))

        out: List[Dict[str, Any]] = []
        for row, score, margin, compared in zip(rows, win_scores, win_margins, comparisons):
            probability = _clip01(_finite_float(score, default=0.0))
            predicted_error = max(0.0, (1.0 - probability) * usable_threshold)
            out.append(
                {
                    "index": int(row["index"]),
                    "candidate": row["candidate"],
                    "time_sec": float(row["time_sec"]),
                    "predicted_abs_error_sec": float(predicted_error),
                    "selection_probability": float(probability),
                    "chooser_score": float(probability),
                    "probability_margin": float(_clip01(margin)),
                    "pairwise_comparisons": int(compared),
                    "model_type": str(payload.get("model_type") or ""),
                    "good_candidate_threshold_sec": float(good_threshold),
                    "usable_candidate_threshold_sec": float(usable_threshold),
                }
            )
        return out

    is_classifier = "classifier" in model_type or str(payload.get("target") or "").startswith("best_review_candidate")
    if is_classifier and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            pred = proba[:, classes.index(1)]
        elif proba.ndim == 2 and proba.shape[1] >= 2:
            pred = proba[:, -1]
        else:
            pred = np.asarray(model.predict(x), dtype=np.float64)
    else:
        pred = np.asarray(model.predict(x), dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for row, value in zip(rows, pred):
        if is_classifier:
            probability = _clip01(_finite_float(value, default=0.0))
            good_threshold = max(0.001, _finite_float(payload.get("good_candidate_threshold_sec"), 0.100))
            predicted_error = max(0.0, (1.0 - probability) * good_threshold)
            out.append(
                {
                    "index": int(row["index"]),
                    "candidate": row["candidate"],
                    "time_sec": float(row["time_sec"]),
                    "predicted_abs_error_sec": float(predicted_error),
                    "selection_probability": float(probability),
                    "chooser_score": float(probability),
                    "model_type": str(payload.get("model_type") or ""),
                }
            )
        else:
            predicted_error = max(0.0, _finite_float(value, default=999.0))
            out.append(
                {
                    "index": int(row["index"]),
                    "candidate": row["candidate"],
                    "time_sec": float(row["time_sec"]),
                    "predicted_abs_error_sec": float(predicted_error),
                    "chooser_score": float(1.0 / (1.0 + predicted_error)),
                    "model_type": str(payload.get("model_type") or ""),
                }
            )
    return out


def choose_learned_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    model_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    payload = load_candidate_chooser_payload(model_path)
    if payload is None:
        return None
    predictions = predict_candidate_errors(payload, candidates)
    if not predictions:
        return None
    model_type = str(payload.get("model_type") or "").lower()
    is_classifier = "classifier" in model_type or str(payload.get("target") or "").startswith("best_review_candidate")
    if is_classifier:
        ranked = sorted(
            predictions,
            key=lambda row: (
                -float(row.get("selection_probability", row.get("chooser_score", 0.0))),
                _finite_float(row["candidate"].get("rank"), 999.0) if isinstance(row.get("candidate"), Mapping) else 999.0,
                float(row["time_sec"]),
            ),
        )
    else:
        ranked = sorted(
            predictions,
            key=lambda row: (
                float(row["predicted_abs_error_sec"]),
                _finite_float(row["candidate"].get("rank"), 999.0) if isinstance(row.get("candidate"), Mapping) else 999.0,
                float(row["time_sec"]),
            ),
        )
    best = dict(ranked[0])
    if is_classifier:
        best_prob = _clip01(float(best.get("selection_probability", best.get("chooser_score", 0.0))))
        second_prob = _clip01(float(ranked[1].get("selection_probability", ranked[1].get("chooser_score", 0.0)))) if len(ranked) > 1 else 0.0
        probability_margin = max(0.0, best_prob - second_prob)
        margin = probability_margin
        confidence = _clip01((0.70 * best_prob) + (0.30 * _clip01(probability_margin / 0.30)))
        best["selection_probability"] = float(best_prob)
        best["probability_margin"] = float(probability_margin)
    else:
        second_error = float(ranked[1]["predicted_abs_error_sec"]) if len(ranked) > 1 else float(best["predicted_abs_error_sec"]) + 1.0
        margin = max(0.0, float(second_error) - float(best["predicted_abs_error_sec"]))
        confidence = _clip01((0.55 * (1.0 - _clip01(float(best["predicted_abs_error_sec"]) / 0.50))) + (0.45 * _clip01(margin / 0.15)))
    best["prediction_margin_sec"] = float(margin)
    best["selection_confidence"] = float(confidence)
    best["model_path"] = str(payload.get("path", ""))
    best["training_rows"] = int(payload.get("training_rows", 0) or 0)
    best["correction_rows"] = int(payload.get("correction_rows", 0) or 0)
    best["selector_correction_rows"] = int(payload.get("selector_correction_rows", 0) or 0)
    best["bad_candidate_sets"] = int(payload.get("bad_candidate_sets", 0) or 0)
    if isinstance(payload.get("auto_gate_thresholds"), Mapping):
        best["auto_gate_thresholds"] = dict(payload.get("auto_gate_thresholds") or {})
    return best
