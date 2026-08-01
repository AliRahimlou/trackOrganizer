from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_VISUAL_SELECTOR_PATH = Path(__file__).resolve().parents[1] / "models" / "visual_candidate_selector.pkl"

VISUAL_SELECTOR_FEATURES = (
    "rank_inv",
    "rank_norm",
    "score",
    "score_delta_from_best",
    "score_percentile",
    "time_order_norm",
    "temporal_position_norm",
    "prev_gap_norm",
    "next_gap_norm",
    "clock_bar_norm",
    "clock_bar_early_8",
    "clock_bar_early_17",
    "clock_bar_17_33",
    "clock_bar_33_65",
    "clock_bar_late_65",
    "on_one",
    "one_distance_norm",
    "visual_phrase_prior",
    "visual_post4_height",
    "visual_post8_height",
    "visual_pre4_height",
    "visual_jump4",
    "visual_jump8",
    "visual_post_bass8",
    "visual_post_drum8",
    "visual_post_inst8",
    "visual_pre_inst4",
    "visual_pre_drum_cont4",
    "visual_post_drum_cont4",
    "visual_post_drum_cont8",
    "visual_local_reentry_gap",
    "visual_frame_body",
    "visual_frame_pre_body",
    "visual_frame_post_body",
    "visual_frame_attack_peak",
    "visual_frame_low_peak",
    "visual_frame_score",
    "visual_local_reentry",
    "visual_phrase_body_shift",
    "visual_opening_body_start",
    "visual_phrase_dropout_reentry",
    "visual_later_definitive_guard",
    "visual_blank_waveform_guard",
    "source_chunk",
    "source_first_fat_block",
    "source_v2",
    "source_opening",
    "source_static_guard",
    "source_late_guard",
    "source_fusion_guard",
    "source_track_zero_guard",
    "source_audit_replacement",
    "source_body_peak",
    "source_body_edge",
    "drop_strength_proxy",
    "visual_first_top_boost_align",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _clip01(value: Any) -> float:
    return float(np.clip(_finite_float(value), 0.0, 1.0))


def _rank_inv(value: Any, *, cap: float = 24.0) -> float:
    rank = max(1.0, min(float(cap), _finite_float(value, cap)))
    return float(1.0 / rank)


def _rank_norm(value: Any, *, cap: float = 24.0) -> float:
    rank = max(1.0, min(float(cap), _finite_float(value, cap)))
    return float((rank - 1.0) / max(1.0, cap - 1.0))


def candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
    for source in (micro, candidate):
        for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec", "visual_raw_chunk_time"):
            try:
                value = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return float(value)
    return None


def _visual(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("visual_components")
    return value if isinstance(value, Mapping) else {}


def _score(candidate: Mapping[str, Any]) -> float:
    return _clip01(candidate.get("score", candidate.get("confidence_score", 0.0)))


def _clock_bar(candidate: Mapping[str, Any]) -> int:
    visual = _visual(candidate)
    try:
        value = int(visual.get("clock_bar", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    if value:
        return int(value)
    clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
    try:
        return int(clock.get("nearest_one_bar", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _on_one(candidate: Mapping[str, Any]) -> float:
    clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
    if "on_one" in clock:
        return 1.0 if bool(clock.get("on_one")) else 0.0
    one_distance = _finite_float(clock.get("one_distance_ms"), 999999.0)
    return 1.0 if one_distance <= 55.0 else 0.0


def _source_is(candidate: Mapping[str, Any], needle: str) -> float:
    return 1.0 if needle in str(candidate.get("selected_by") or "") else 0.0


def _drop_strength_proxy(candidate: Mapping[str, Any]) -> float:
    visual = _visual(candidate)
    phrase_bonus = 0.5 * _clip01((_finite_float(visual.get("phrase_prior")) - 0.48) / 0.44)
    section_shift = 0.16 if bool(visual.get("phrase_body_shift")) else 0.0
    local_reentry = 0.10 if bool(visual.get("local_reentry")) else 0.0
    return _clip01(
        (0.24 * _finite_float(visual.get("post8_height")))
        + (0.20 * _finite_float(visual.get("post_bass8")))
        + (0.18 * _finite_float(visual.get("post_drum8")))
        + (0.14 * _finite_float(visual.get("post_inst8")))
        + (0.09 * _finite_float(visual.get("jump8")))
        + (0.07 * _finite_float(visual.get("post_drum_cont8")))
        + phrase_bonus
        + section_shift
        + local_reentry
        - (0.08 * _finite_float(visual.get("pre_drum_cont4")))
    )


def _context(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid: List[Tuple[int, Mapping[str, Any], float, float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        time_sec = candidate_time(candidate)
        if time_sec is None:
            continue
        valid.append((index, candidate, float(time_sec), _score(candidate)))
    scores = sorted(score for _index, _candidate, _time, score in valid)
    times = sorted(time for _index, _candidate, time, _score_value in valid)
    return {
        "valid": valid,
        "sorted_scores": scores,
        "sorted_times": times,
        "best_score": max(scores) if scores else 0.0,
        "min_time": min(times) if times else 0.0,
        "time_span": max(times) - min(times) if len(times) > 1 else 1.0,
    }


def feature_dict(candidate: Mapping[str, Any], *, time_sec: float, context: Mapping[str, Any]) -> Dict[str, float]:
    visual = _visual(candidate)
    score = _score(candidate)
    rank = candidate.get("rank", candidate.get("handcrafted_rank", 24.0))
    clock_bar = _clock_bar(candidate)
    clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
    sorted_scores = list(context.get("sorted_scores") or [])
    if len(sorted_scores) <= 1:
        score_percentile = 1.0
    else:
        lower_or_equal = sum(1 for value in sorted_scores if float(value) <= score)
        score_percentile = _clip01((lower_or_equal - 1.0) / max(1.0, len(sorted_scores) - 1.0))
    sorted_times = list(context.get("sorted_times") or [])
    time_order = 0
    for index, value in enumerate(sorted_times):
        if abs(float(value) - float(time_sec)) <= 1e-6:
            time_order = index
            break
    time_order_norm = 0.0 if len(sorted_times) <= 1 else _clip01(time_order / max(1.0, len(sorted_times) - 1.0))
    temporal_position = _clip01((float(time_sec) - float(context.get("min_time", time_sec))) / max(1e-6, float(context.get("time_span", 1.0))))
    prev_gap = float("inf") if time_order <= 0 else float(time_sec) - float(sorted_times[time_order - 1])
    next_gap = float("inf") if time_order >= len(sorted_times) - 1 else float(sorted_times[time_order + 1]) - float(time_sec)
    finite_prev_gap = 4.0 if not math.isfinite(prev_gap) else max(0.0, prev_gap)
    finite_next_gap = 4.0 if not math.isfinite(next_gap) else max(0.0, next_gap)
    return {
        "rank_inv": _rank_inv(rank),
        "rank_norm": _rank_norm(rank),
        "score": float(score),
        "score_delta_from_best": _clip01(float(context.get("best_score", 0.0)) - score),
        "score_percentile": float(score_percentile),
        "time_order_norm": float(time_order_norm),
        "temporal_position_norm": float(temporal_position),
        "prev_gap_norm": _clip01(finite_prev_gap / 8.0),
        "next_gap_norm": _clip01(finite_next_gap / 8.0),
        "clock_bar_norm": _clip01(clock_bar / 129.0),
        "clock_bar_early_8": 1.0 if 1 <= clock_bar <= 8 else 0.0,
        "clock_bar_early_17": 1.0 if 1 <= clock_bar <= 17 else 0.0,
        "clock_bar_17_33": 1.0 if 17 <= clock_bar <= 33 else 0.0,
        "clock_bar_33_65": 1.0 if 33 <= clock_bar <= 65 else 0.0,
        "clock_bar_late_65": 1.0 if clock_bar >= 65 else 0.0,
        "on_one": _on_one(candidate),
        "one_distance_norm": _clip01(abs(_finite_float(clock.get("one_distance_ms"), 999999.0)) / 220.0),
        "visual_phrase_prior": _clip01(visual.get("phrase_prior")),
        "visual_post4_height": _clip01(visual.get("post4_height")),
        "visual_post8_height": _clip01(visual.get("post8_height")),
        "visual_pre4_height": _clip01(visual.get("pre4_height")),
        "visual_jump4": _clip01(_finite_float(visual.get("jump4")) + 0.5),
        "visual_jump8": _clip01(_finite_float(visual.get("jump8")) + 0.5),
        "visual_post_bass8": _clip01(visual.get("post_bass8")),
        "visual_post_drum8": _clip01(visual.get("post_drum8")),
        "visual_post_inst8": _clip01(visual.get("post_inst8")),
        "visual_pre_inst4": _clip01(visual.get("pre_inst4")),
        "visual_pre_drum_cont4": _clip01(visual.get("pre_drum_cont4")),
        "visual_post_drum_cont4": _clip01(visual.get("post_drum_cont4")),
        "visual_post_drum_cont8": _clip01(visual.get("post_drum_cont8")),
        "visual_local_reentry_gap": _clip01(_finite_float(visual.get("local_reentry_gap")) + 0.5),
        "visual_frame_body": _clip01(visual.get("frame_body")),
        "visual_frame_pre_body": _clip01(visual.get("frame_pre_body")),
        "visual_frame_post_body": _clip01(visual.get("frame_post_body")),
        "visual_frame_attack_peak": _clip01(visual.get("frame_attack_peak")),
        "visual_frame_low_peak": _clip01(visual.get("frame_low_peak")),
        "visual_frame_score": _clip01(visual.get("frame_score")),
        "visual_local_reentry": 1.0 if bool(visual.get("local_reentry")) else 0.0,
        "visual_phrase_body_shift": 1.0 if bool(visual.get("phrase_body_shift")) else 0.0,
        "visual_opening_body_start": 1.0 if bool(visual.get("opening_body_start")) else 0.0,
        "visual_phrase_dropout_reentry": 1.0 if bool(visual.get("phrase_dropout_reentry")) else 0.0,
        "visual_later_definitive_guard": 1.0 if bool(visual.get("later_definitive_drop_guard")) else 0.0,
        "visual_blank_waveform_guard": 1.0 if bool(visual.get("blank_waveform_marker_guard")) else 0.0,
        "source_chunk": _source_is(candidate, "visual_gui_chunk"),
        "source_first_fat_block": _source_is(candidate, "visual_gui_first_fat_block"),
        "source_v2": _source_is(candidate, "visual_drop_v2"),
        "source_opening": _source_is(candidate, "visual_gui_opening_drop"),
        "source_static_guard": _source_is(candidate, "visual_static_marker"),
        "source_late_guard": _source_is(candidate, "visual_late_reset"),
        "source_fusion_guard": _source_is(candidate, "visual_fusion"),
        "source_track_zero_guard": _source_is(candidate, "visual_track_zero"),
        "source_audit_replacement": _source_is(candidate, "visual_audit_replacement"),
        "source_body_peak": _source_is(candidate, "visual_body_peak"),
        "source_body_edge": _source_is(candidate, "visual_body_edge"),
        "drop_strength_proxy": _drop_strength_proxy(candidate),
        "visual_first_top_boost_align": _finite_float(visual.get("first_top_boost_align"), 0.5),
    }


def feature_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = VISUAL_SELECTOR_FEATURES,
) -> List[Dict[str, Any]]:
    ctx = _context(candidates)
    rows: List[Dict[str, Any]] = []
    for index, candidate, time_sec, _score_value in list(ctx.get("valid") or []):
        features = feature_dict(candidate, time_sec=time_sec, context=ctx)
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


def load_payload(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = Path(model_path).expanduser() if model_path else DEFAULT_VISUAL_SELECTOR_PATH
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid visual candidate selector payload: {path}")
    return payload


def save_payload(payload: Mapping[str, Any], model_path: Optional[str] = None) -> str:
    path = Path(model_path).expanduser() if model_path else DEFAULT_VISUAL_SELECTOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh)
    return str(path)


def predict_candidates(payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    model = payload.get("model")
    feature_names = [str(name) for name in payload.get("feature_names") or VISUAL_SELECTOR_FEATURES]
    rows = feature_rows(candidates, feature_names=feature_names)
    if not rows:
        return []
    x = np.asarray([row["vector"] for row in rows], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            scores = proba[:, classes.index(1)]
        elif proba.ndim == 2 and proba.shape[1] >= 2:
            scores = proba[:, -1]
        else:
            scores = proba.reshape(-1)
    else:
        scores = np.asarray(model.predict(x), dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        out.append(
            {
                "candidate": row["candidate"],
                "time_sec": float(row["time_sec"]),
                "score": float(np.clip(score, 0.0, 1.0)),
                "features": row["features"],
                "index": int(row["index"]),
            }
        )
    out.sort(key=lambda row: (-float(row["score"]), float(row["time_sec"]), int(row["index"])))
    return out


def choose_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    model_path: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    loaded = dict(payload) if isinstance(payload, Mapping) else load_payload(model_path)
    if not loaded:
        return None
    ranked = predict_candidates(loaded, candidates)
    if not ranked:
        return None
    best = dict(ranked[0]["candidate"])
    original_source = str(best.get("selected_by") or "")
    best["selected_by"] = "visual_candidate_selector"
    best["visual_candidate_selector"] = {
        "score": float(ranked[0]["score"]),
        "margin": float(ranked[0]["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)),
        "source_selected_by": original_source,
        "training_rows": int(loaded.get("training_rows", 0) or 0),
        "training_tracks": int(loaded.get("training_tracks", 0) or 0),
        "model_path": str(model_path or DEFAULT_VISUAL_SELECTOR_PATH),
    }
    best["reason"] = (
        "visual candidate selector chose the reviewed-pattern waveform candidate; "
        f"{best.get('reason') or 'visual candidate'}"
    )
    return best
