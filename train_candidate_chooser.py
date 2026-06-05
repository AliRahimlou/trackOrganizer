#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from drop_aligner.candidate_chooser import (
    CHOOSER_FEATURES,
    candidate_feature_rows,
    save_candidate_chooser_payload,
)
from drop_aligner.exclusions import row_has_excluded_path


GOOD_CANDIDATE_SEC = 0.100
USABLE_CANDIDATE_SEC = 0.250


@dataclass
class CandidateRow:
    correction_id: int
    track: str
    reviewed_from: str
    selected_by: str
    ai_pick: float
    user_pick: float
    candidate_index: int
    candidate_time: float
    target_abs_error_sec: float
    oracle_abs_error_sec: float
    is_oracle: bool
    selector_label: int
    sample_weight: float
    vector: List[float]


@dataclass
class TrainingData:
    rows: List[CandidateRow]
    correction_count: int
    selector_correction_count: int
    bad_candidate_sets: int
    oracle_within_25ms: int
    oracle_within_100ms: int
    oracle_within_250ms: int
    accepted_picks: int
    corrected_picks: int


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _candidate_rows(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    source = row.get("top_10_candidates")
    if not source:
        source = row.get("candidates")
    candidates = list(source) if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) else []
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        candidates.append(selected)
    out: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        t = None
        micro = candidate.get("microalign")
        if isinstance(micro, Mapping):
            t = _safe_float(micro.get("microaligned_time"))
        for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec"):
            if t is None:
                t = _safe_float(candidate.get(key))
        key = f"{t:.6f}" if t is not None else json.dumps(dict(candidate), sort_keys=True, default=str)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _ai_pick(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _row_weight(row: Mapping[str, Any], *, target: float, is_oracle: bool, correction_id: int, total_rows: int) -> float:
    reviewed_from = str(row.get("reviewed_from") or "")
    selected_by = str(row.get("selected_by") or "")
    weight = 1.0
    if is_oracle:
        weight *= 6.0
    if target <= 0.025:
        weight *= 2.6
    elif target <= 0.100:
        weight *= 1.8
    if reviewed_from.startswith("web_accept_"):
        weight *= 1.75
    if reviewed_from == "web_candidate_pick":
        weight *= 2.25
    if reviewed_from == "web_manual_marker":
        weight *= 1.35
    if selected_by == "review_auto_place":
        weight *= 1.50
    if total_rows > 1:
        recency = correction_id / float(max(1, total_rows - 1))
        weight *= 0.75 + (0.75 * recency)
    return float(weight)


def load_training_data(corrections_path: str) -> TrainingData:
    path = Path(corrections_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Correction log not found: {path}")
    raw_rows = [row for row in _iter_jsonl(path) if not row_has_excluded_path(row)]
    rows: List[CandidateRow] = []
    accepted = 0
    corrected = 0
    usable_corrections = 0
    selector_corrections = 0
    bad_candidate_sets = 0
    oracle_25 = 0
    oracle_100 = 0
    oracle_250 = 0

    for correction_id, row in enumerate(raw_rows):
        user_pick = _safe_float(row.get("user_pick"))
        ai_pick = _ai_pick(row)
        if user_pick is None or ai_pick is None:
            continue
        candidates = _candidate_rows(row)
        feature_rows = candidate_feature_rows(candidates)
        if not feature_rows:
            continue
        targets = [abs(float(item["time_sec"]) - float(user_pick)) for item in feature_rows]
        oracle_error = min(targets)
        oracle_index = int(np.argmin(np.asarray(targets, dtype=np.float64)))
        usable_corrections += 1
        if abs(float(user_pick) - float(ai_pick)) <= 0.001:
            accepted += 1
        else:
            corrected += 1
        if oracle_error <= 0.025:
            oracle_25 += 1
        if oracle_error <= GOOD_CANDIDATE_SEC:
            oracle_100 += 1
        if oracle_error <= USABLE_CANDIDATE_SEC:
            oracle_250 += 1
        else:
            bad_candidate_sets += 1
            continue
        selector_corrections += 1
        for item_index, item in enumerate(feature_rows):
            target = float(targets[item_index])
            is_oracle = bool(item_index == oracle_index)
            rows.append(
                CandidateRow(
                    correction_id=int(correction_id),
                    track=str(row.get("track") or row.get("filename") or ""),
                    reviewed_from=str(row.get("reviewed_from") or ""),
                    selected_by=str(row.get("selected_by") or ""),
                    ai_pick=float(ai_pick),
                    user_pick=float(user_pick),
                    candidate_index=int(item["index"]),
                    candidate_time=float(item["time_sec"]),
                    target_abs_error_sec=target,
                    oracle_abs_error_sec=float(oracle_error),
                    is_oracle=is_oracle,
                    selector_label=1 if is_oracle else 0,
                    sample_weight=_row_weight(
                        row,
                        target=target,
                        is_oracle=is_oracle,
                        correction_id=correction_id,
                        total_rows=len(raw_rows),
                    ),
                    vector=[float(v) for v in item["vector"]],
                )
            )
    if not rows:
        raise ValueError(f"No usable candidate rows found in correction log: {path}")
    return TrainingData(
        rows=rows,
        correction_count=usable_corrections,
        selector_correction_count=selector_corrections,
        bad_candidate_sets=bad_candidate_sets,
        oracle_within_25ms=oracle_25,
        oracle_within_100ms=oracle_100,
        oracle_within_250ms=oracle_250,
        accepted_picks=accepted,
        corrected_picks=corrected,
    )


def _make_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=int(random_state),
        n_estimators=700,
        min_samples_leaf=1,
        max_features=0.75,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def _matrix(rows: Sequence[CandidateRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row.vector for row in rows], dtype=np.float64)
    y = np.asarray([int(row.selector_label) for row in rows], dtype=np.int64)
    w = np.asarray([row.sample_weight for row in rows], dtype=np.float64)
    return x, y, w


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


def _split_rows(rows: Sequence[CandidateRow], valid_frac: float) -> Tuple[List[CandidateRow], List[CandidateRow]]:
    if valid_frac <= 0.0:
        return list(rows), []
    validation_tracks = {
        row.track or str(row.correction_id)
        for row in rows
        if _stable_bucket(row.track or str(row.correction_id)) < float(valid_frac)
    }
    train: List[CandidateRow] = []
    valid: List[CandidateRow] = []
    for row in rows:
        key = row.track or str(row.correction_id)
        if key in validation_tracks:
            valid.append(row)
        else:
            train.append(row)
    if not train or not valid:
        return list(rows), []
    return train, valid


def _split_diagnostics(train_rows: Sequence[CandidateRow], valid_rows: Sequence[CandidateRow]) -> Dict[str, Any]:
    train_tracks = {row.track or str(row.correction_id) for row in train_rows}
    valid_tracks = {row.track or str(row.correction_id) for row in valid_rows}
    overlap = sorted(track for track in train_tracks.intersection(valid_tracks) if track)
    return {
        "strategy": "track_grouped_stable_hash",
        "train_tracks": int(len(train_tracks)),
        "validation_tracks": int(len(valid_tracks)),
        "train_candidates": int(len(train_rows)),
        "validation_candidates": int(len(valid_rows)),
        "train_corrections": int(len({int(row.correction_id) for row in train_rows})),
        "validation_corrections": int(len({int(row.correction_id) for row in valid_rows})),
        "leakage_check_passed": bool(not overlap),
        "overlap_track_count": int(len(overlap)),
        "overlap_tracks_sample": overlap[:20],
    }


def _label_diagnostics(rows: Sequence[CandidateRow]) -> Dict[str, Any]:
    by_correction: Dict[int, int] = {}
    positive = 0
    negative = 0
    positive_weight = 0.0
    negative_weight = 0.0
    oracle_errors: List[float] = []
    for row in rows:
        by_correction[int(row.correction_id)] = by_correction.get(int(row.correction_id), 0) + 1
        if int(row.selector_label) == 1:
            positive += 1
            positive_weight += float(row.sample_weight)
        else:
            negative += 1
            negative_weight += float(row.sample_weight)
        if bool(row.is_oracle):
            oracle_errors.append(float(row.oracle_abs_error_sec))
    group_sizes = np.asarray(list(by_correction.values()), dtype=np.float64)
    return {
        "candidate_rows": int(len(rows)),
        "corrections": int(len(by_correction)),
        "positive_rows": int(positive),
        "negative_rows": int(negative),
        "positive_percent": _pct(positive, len(rows)),
        "positive_weight_sum": float(positive_weight),
        "negative_weight_sum": float(negative_weight),
        "positive_weight_percent": _pct(int(round(positive_weight * 1000.0)), int(round((positive_weight + negative_weight) * 1000.0))),
        "candidates_per_correction_median": float(np.median(group_sizes)) if group_sizes.size else None,
        "candidates_per_correction_p90": float(np.percentile(group_sizes, 90.0)) if group_sizes.size else None,
        "candidates_per_correction_max": int(np.max(group_sizes)) if group_sizes.size else 0,
        "oracle_error_median_sec": float(np.median(np.asarray(oracle_errors, dtype=np.float64))) if oracle_errors else None,
        "oracle_error_max_sec": float(np.max(np.asarray(oracle_errors, dtype=np.float64))) if oracle_errors else None,
    }


def _selection_metrics(rows: Sequence[CandidateRow], scores: Sequence[float]) -> Dict[str, Any]:
    grouped: Dict[int, List[Tuple[CandidateRow, float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row.correction_id, []).append((row, _clip_probability(score)))
    selected_errors: List[float] = []
    oracle_errors: List[float] = []
    ai_errors: List[float] = []
    selected_scores: List[float] = []
    selected_is_oracle = 0
    for group in grouped.values():
        chosen, _pred = min(
            group,
            key=lambda item: (
                -float(item[1]),
                int(item[0].candidate_index),
                float(item[0].candidate_time),
            ),
        )
        selected_errors.append(float(chosen.target_abs_error_sec))
        selected_scores.append(float(_pred))
        oracle = min(float(row.oracle_abs_error_sec) for row, _ in group)
        oracle_errors.append(float(oracle))
        ai_errors.append(abs(float(chosen.ai_pick) - float(chosen.user_pick)))
        selected_is_oracle += 1 if bool(chosen.is_oracle) else 0
    return _error_summary(
        selected_errors,
        prefix="selected",
        extra={
            "corrections": int(len(selected_errors)),
            "selected_oracle_percent": _pct(selected_is_oracle, len(selected_errors)),
            "oracle_within_25ms_percent": _pct(sum(err <= 0.025 for err in oracle_errors), len(oracle_errors)),
            "oracle_within_100ms_percent": _pct(sum(err <= 0.100 for err in oracle_errors), len(oracle_errors)),
            "ai_within_25ms_percent": _pct(sum(err <= 0.025 for err in ai_errors), len(ai_errors)),
            "ai_within_100ms_percent": _pct(sum(err <= 0.100 for err in ai_errors), len(ai_errors)),
            "selected_probability_median": float(np.median(np.asarray(selected_scores, dtype=np.float64))) if selected_scores else None,
            "selected_probability_p10": float(np.percentile(np.asarray(selected_scores, dtype=np.float64), 10.0)) if selected_scores else None,
        },
    )


def _selected_details(rows: Sequence[CandidateRow], scores: Sequence[float]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Tuple[CandidateRow, float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row.correction_id, []).append((row, _clip_probability(score)))
    details: List[Dict[str, Any]] = []
    for correction_id, group in grouped.items():
        ranked = sorted(
            group,
            key=lambda item: (
                -float(item[1]),
                int(item[0].candidate_index),
                float(item[0].candidate_time),
            ),
        )
        chosen, score = ranked[0]
        second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        oracle = min(group, key=lambda item: (float(item[0].target_abs_error_sec), int(item[0].candidate_index)))[0]
        details.append(
            {
                "correction_id": int(correction_id),
                "track": str(chosen.track),
                "score": float(score),
                "margin": float(max(0.0, float(score) - second)),
                "selected_error_sec": float(chosen.target_abs_error_sec),
                "oracle_error_sec": float(oracle.target_abs_error_sec),
                "selected_is_oracle": bool(chosen.is_oracle),
            }
        )
    return details


def _gate_metrics(details: Sequence[Mapping[str, Any]], *, probability: float, margin: float) -> Dict[str, Any]:
    accepted = [
        row
        for row in details
        if float(row.get("score", 0.0)) >= float(probability)
        and float(row.get("margin", 0.0)) >= float(margin)
    ]
    out: Dict[str, Any] = {
        "probability": float(probability),
        "margin": float(margin),
        "accepted": int(len(accepted)),
        "coverage_percent": _pct(int(len(accepted)), int(len(details))),
    }
    if accepted:
        errors = np.asarray([float(row.get("selected_error_sec", 999.0)) for row in accepted], dtype=np.float64)
        out.update(
            {
                "median_error_sec": float(np.median(errors)),
                "mean_error_sec": float(np.mean(errors)),
                "max_error_sec": float(np.max(errors)),
                "within_25ms_percent": _pct(int(np.sum(errors <= 0.025)), int(errors.size)),
                "within_50ms_percent": _pct(int(np.sum(errors <= 0.050)), int(errors.size)),
                "within_100ms_percent": _pct(int(np.sum(errors <= 0.100)), int(errors.size)),
                "over_250ms": int(np.sum(errors > 0.250)),
                "over_1s": int(np.sum(errors > 1.0)),
            }
        )
    return out


def _pick_gate(details: Sequence[Mapping[str, Any]], *, max_error_sec: float, min_accept: int) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for probability in [round(v, 3) for v in np.arange(0.99, 0.19, -0.01)]:
        for margin in [round(v, 3) for v in np.arange(0.50, -0.001, -0.01)]:
            metrics = _gate_metrics(details, probability=float(probability), margin=float(margin))
            accepted = int(metrics.get("accepted", 0) or 0)
            if accepted < int(min_accept):
                continue
            if float(metrics.get("max_error_sec", 999.0)) > float(max_error_sec):
                continue
            if int(metrics.get("over_250ms", 0) or 0) > 0:
                continue
            if best is None:
                best = metrics
                continue
            if accepted > int(best.get("accepted", 0) or 0):
                best = metrics
                continue
            if accepted == int(best.get("accepted", 0) or 0) and float(metrics.get("mean_error_sec", 999.0)) < float(best.get("mean_error_sec", 999.0)):
                best = metrics
    if best is None:
        return {
            "probability": 0.99,
            "margin": 0.50,
            "accepted": 0,
            "coverage_percent": 0.0,
            "max_error_sec": None,
            "note": "no validation threshold met the requested autonomous safety target",
        }
    return best


def _auto_gate_thresholds(validation_details: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    conservative = _pick_gate(validation_details, max_error_sec=0.025, min_accept=1)
    normal = _pick_gate(validation_details, max_error_sec=0.050, min_accept=1)
    aggressive = _pick_gate(validation_details, max_error_sec=0.100, min_accept=1)

    def thresholds(metrics: Mapping[str, Any], *, fallback_probability: float, fallback_margin: float, micro: float, offset: float, fake: float) -> Dict[str, float]:
        probability = float(metrics.get("probability", fallback_probability) or fallback_probability)
        margin = float(metrics.get("margin", fallback_margin) or fallback_margin)
        confidence = float(np.clip((0.70 * probability) + (0.30 * min(1.0, margin / 0.30)) - 0.02, 0.0, 1.0))
        predicted_error = max(0.005, (1.0 - probability) * float(GOOD_CANDIDATE_SEC) + 0.005)
        return {
            "probability": float(probability),
            "confidence": float(confidence),
            "margin": float(margin),
            "predicted_error": float(predicted_error),
            "micro": float(micro),
            "offset": float(offset),
            "fake": float(fake),
            "disagree_probability": min(1.01, float(probability) + 0.06),
        }

    return {
        "conservative": thresholds(conservative, fallback_probability=0.99, fallback_margin=0.50, micro=0.96, offset=20.0, fake=0.25),
        "normal": thresholds(normal, fallback_probability=0.99, fallback_margin=0.50, micro=0.93, offset=45.0, fake=0.35),
        "aggressive": thresholds(aggressive, fallback_probability=0.99, fallback_margin=0.50, micro=0.88, offset=90.0, fake=0.60),
        "validation_gate_metrics": {
            "conservative": conservative,
            "normal": normal,
            "aggressive": aggressive,
        },
    }


def _clip_probability(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def _classification_metrics(labels: Sequence[int], scores: Sequence[float], weights: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray([_clip_probability(value) for value in scores], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out: Dict[str, Any] = {
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
    }
    if y.size == 0 or len(set(int(v) for v in y.tolist())) < 2:
        return out
    p_loss = np.clip(p, 1e-6, 1.0 - 1e-6)
    out["candidate_log_loss"] = float(log_loss(y, p_loss, sample_weight=w, labels=[0, 1]))
    out["candidate_roc_auc"] = float(roc_auc_score(y, p, sample_weight=w))
    out["candidate_average_precision"] = float(average_precision_score(y, p, sample_weight=w))
    return out


def _pct(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (100.0 * float(count)) / float(total)


def _error_summary(errors: Sequence[float], *, prefix: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    values = np.asarray(list(errors), dtype=np.float64)
    if values.size == 0:
        out: Dict[str, Any] = {
            f"{prefix}_count": 0,
            f"{prefix}_median_abs_error_sec": None,
            f"{prefix}_mean_abs_error_sec": None,
            f"{prefix}_within_25ms_percent": None,
            f"{prefix}_within_100ms_percent": None,
        }
    else:
        out = {
            f"{prefix}_count": int(values.size),
            f"{prefix}_median_abs_error_sec": float(np.median(values)),
            f"{prefix}_mean_abs_error_sec": float(np.mean(values)),
            f"{prefix}_within_5ms_percent": _pct(int(np.sum(values <= 0.005)), int(values.size)),
            f"{prefix}_within_10ms_percent": _pct(int(np.sum(values <= 0.010)), int(values.size)),
            f"{prefix}_within_25ms_percent": _pct(int(np.sum(values <= 0.025)), int(values.size)),
            f"{prefix}_within_50ms_percent": _pct(int(np.sum(values <= 0.050)), int(values.size)),
            f"{prefix}_within_100ms_percent": _pct(int(np.sum(values <= 0.100)), int(values.size)),
            f"{prefix}_over_1s": int(np.sum(values > 1.0)),
        }
    if extra:
        out.update(dict(extra))
    return out


def _feature_importances(model: object) -> Dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return {name: float(value) for name, value in zip(CHOOSER_FEATURES, np.asarray(values, dtype=np.float64))}


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
    return str(out)


def train_candidate_chooser(
    *,
    corrections: str,
    output: str,
    report: str,
    random_state: int = 42,
    valid_frac: float = 0.20,
) -> Dict[str, Any]:
    data = load_training_data(corrections)
    train_rows, valid_rows = _split_rows(data.rows, valid_frac)
    split_diagnostics = _split_diagnostics(train_rows, valid_rows)
    x_train, y_train, w_train = _matrix(train_rows)
    validation_metrics: Dict[str, Any] = {}
    auto_gate_thresholds: Dict[str, Any] = {}
    validation_gate_metrics: Dict[str, Any] = {}
    if valid_rows:
        validation_model = _make_model(random_state)
        validation_model.fit(x_train, y_train, sample_weight=w_train)
        x_valid, y_valid, w_valid = _matrix(valid_rows)
        valid_pred = _positive_probability(validation_model, x_valid)
        validation_details = _selected_details(valid_rows, valid_pred)
        auto_gate_thresholds = _auto_gate_thresholds(validation_details)
        validation_gate_metrics = dict(auto_gate_thresholds.pop("validation_gate_metrics", {}))
        validation_metrics = {
            **_classification_metrics(y_valid, valid_pred, w_valid),
            **_selection_metrics(valid_rows, valid_pred),
            "auto_gate_shadow": validation_gate_metrics,
        }

    x_all, y_all, w_all = _matrix(data.rows)
    model = _make_model(random_state)
    model.fit(x_all, y_all, sample_weight=w_all)
    pred_all = _positive_probability(model, x_all)
    trained_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "model": model,
        "model_type": "random_forest_classifier",
        "feature_names": list(CHOOSER_FEATURES),
        "target": "best_review_candidate_with_oracle_within_250ms",
        "training_label_diagnostics": _label_diagnostics(data.rows),
        "good_candidate_threshold_sec": float(GOOD_CANDIDATE_SEC),
        "usable_candidate_threshold_sec": float(USABLE_CANDIDATE_SEC),
        "trained_at": trained_at,
        "corrections_path": str(Path(corrections).expanduser()),
        "training_rows": int(len(data.rows)),
        "correction_rows": int(data.correction_count),
        "selector_correction_rows": int(data.selector_correction_count),
        "bad_candidate_sets": int(data.bad_candidate_sets),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "feature_importances": _feature_importances(model),
        "auto_gate_thresholds": auto_gate_thresholds,
        "validation_split": split_diagnostics,
    }
    saved_path = save_candidate_chooser_payload(payload, output)

    report_payload = {
        "model_path": saved_path,
        "model_type": "random_forest_classifier",
        "feature_names": list(CHOOSER_FEATURES),
        "training_rows": int(len(data.rows)),
        "training_label_diagnostics": _label_diagnostics(data.rows),
        "correction_rows": int(data.correction_count),
        "selector_correction_rows": int(data.selector_correction_count),
        "bad_candidate_sets": int(data.bad_candidate_sets),
        "oracle_within_25ms_percent": _pct(data.oracle_within_25ms, data.correction_count),
        "oracle_within_100ms_percent": _pct(data.oracle_within_100ms, data.correction_count),
        "oracle_within_250ms_percent": _pct(data.oracle_within_250ms, data.correction_count),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        **_classification_metrics(y_all, pred_all, w_all),
        "good_candidate_threshold_sec": float(GOOD_CANDIDATE_SEC),
        "usable_candidate_threshold_sec": float(USABLE_CANDIDATE_SEC),
        "target_min_sec": float(np.min([row.target_abs_error_sec for row in data.rows])),
        "target_median_sec": float(np.median([row.target_abs_error_sec for row in data.rows])),
        "target_max_sec": float(np.max([row.target_abs_error_sec for row in data.rows])),
        "sample_weight_sum": float(np.sum(w_all)),
        "train_selection_metrics": _selection_metrics(data.rows, pred_all),
        "validation_metrics": validation_metrics,
        "validation_split": split_diagnostics,
        "feature_importances": _feature_importances(model),
        "auto_gate_thresholds": auto_gate_thresholds,
        "trained_at": trained_at,
    }
    report_path = _write_json(report, report_payload)
    report_payload["training_report"] = report_path
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MicroSnap candidate chooser from review corrections.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL path")
    parser.add_argument("--output", default="models/drop_candidate_chooser.pkl", help="Output model pickle path")
    parser.add_argument("--report", default="models/candidate_chooser_report.json", help="Training report JSON path")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_candidate_chooser(
        corrections=args.corrections,
        output=args.output,
        report=args.report,
        random_state=int(args.random_state),
        valid_frac=float(args.valid_frac),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
