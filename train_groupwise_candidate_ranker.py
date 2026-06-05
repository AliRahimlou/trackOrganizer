#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from drop_aligner.candidate_chooser import CHOOSER_FEATURES, save_candidate_chooser_payload
from train_candidate_chooser import (
    GOOD_CANDIDATE_SEC,
    USABLE_CANDIDATE_SEC,
    CandidateRow,
    _pct,
    _selection_metrics,
    _split_diagnostics,
    _split_rows,
    load_training_data,
)


DEFAULT_CORRECTIONS = "models/multistem_training_corrections.jsonl"
DEFAULT_OUTPUT = "models/drop_multistem_groupwise_ranker.pkl"
DEFAULT_REPORT = "models/multistem_groupwise_ranker_report.json"


@dataclass(frozen=True)
class PairwiseData:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    pair_count: int
    correction_count: int


def _group_rows(rows: Sequence[CandidateRow]) -> Dict[int, List[CandidateRow]]:
    groups: Dict[int, List[CandidateRow]] = {}
    for row in rows:
        groups.setdefault(int(row.correction_id), []).append(row)
    return groups


def _make_model(random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        random_state=int(random_state),
        n_estimators=900,
        min_samples_leaf=2,
        max_features=0.85,
        bootstrap=False,
        n_jobs=-1,
    )


def _pairwise_training_data(rows: Sequence[CandidateRow], *, min_gap_sec: float) -> PairwiseData:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    weights: List[float] = []
    group_count = 0
    for group in _group_rows(rows).values():
        if len(group) < 2:
            continue
        oracle = min(group, key=lambda row: (float(row.target_abs_error_sec), int(row.candidate_index)))
        made_pair = False
        for other in group:
            if other is oracle:
                continue
            gap = float(other.target_abs_error_sec) - float(oracle.target_abs_error_sec)
            if gap < float(min_gap_sec):
                continue
            diff = np.asarray(oracle.vector, dtype=np.float64) - np.asarray(other.vector, dtype=np.float64)
            weight = 0.5 * (float(oracle.sample_weight) + float(other.sample_weight))
            weight *= min(5.0, 1.0 + (gap / 0.250))
            x_rows.append(diff.tolist())
            y_rows.append(1)
            weights.append(float(weight))
            x_rows.append((-diff).tolist())
            y_rows.append(0)
            weights.append(float(weight))
            made_pair = True
        if made_pair:
            group_count += 1
    if not x_rows:
        raise ValueError("No pairwise training rows were produced")
    return PairwiseData(
        x=np.asarray(x_rows, dtype=np.float64),
        y=np.asarray(y_rows, dtype=np.int64),
        w=np.asarray(weights, dtype=np.float64),
        pair_count=int(len(x_rows)),
        correction_count=int(group_count),
    )


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


def _pairwise_scores(model: object, rows: Sequence[CandidateRow]) -> List[float]:
    if not rows:
        return []
    x = np.asarray([row.vector for row in rows], dtype=np.float64)
    n_rows = int(x.shape[0])
    if n_rows == 1:
        return [1.0]
    pairs: List[Tuple[int, int]] = []
    pair_vectors: List[np.ndarray] = []
    for left in range(n_rows):
        for right in range(n_rows):
            if left == right:
                continue
            pairs.append((left, right))
            pair_vectors.append(x[left] - x[right])
    pred = _positive_probability(model, np.asarray(pair_vectors, dtype=np.float64))
    totals = np.zeros(n_rows, dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.float64)
    for (left, _right), value in zip(pairs, pred):
        totals[left] += float(np.clip(value, 0.0, 1.0))
        counts[left] += 1.0
    return (totals / np.maximum(1.0, counts)).tolist()


def _score_rows(model: object, rows: Sequence[CandidateRow]) -> List[float]:
    scores_by_index: Dict[Tuple[int, int], float] = {}
    for correction_id, group in _group_rows(rows).items():
        scores = _pairwise_scores(model, group)
        for row, score in zip(group, scores):
            scores_by_index[(int(correction_id), int(row.candidate_index))] = float(score)
    return [scores_by_index[(int(row.correction_id), int(row.candidate_index))] for row in rows]


def _classification_metrics(labels: Sequence[int], scores: Sequence[float], weights: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out: Dict[str, Any] = {
        "positive_pairs": int(np.sum(y == 1)),
        "negative_pairs": int(np.sum(y == 0)),
    }
    if y.size == 0 or len(set(int(v) for v in y.tolist())) < 2:
        return out
    p_loss = np.clip(p, 1e-6, 1.0 - 1e-6)
    out["pair_log_loss"] = float(log_loss(y, p_loss, sample_weight=w, labels=[0, 1]))
    out["pair_roc_auc"] = float(roc_auc_score(y, p, sample_weight=w))
    out["pair_average_precision"] = float(average_precision_score(y, p, sample_weight=w))
    return out


def _selected_details(rows: Sequence[CandidateRow], scores: Sequence[float]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Tuple[CandidateRow, float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(int(row.correction_id), []).append((row, float(np.clip(score, 0.0, 1.0))))
    details: List[Dict[str, Any]] = []
    for correction_id, group in grouped.items():
        ranked = sorted(
            group,
            key=lambda item: (-float(item[1]), int(item[0].candidate_index), float(item[0].candidate_time)),
        )
        chosen, score = ranked[0]
        second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = max(0.0, float(score) - second)
        oracle = min(group, key=lambda item: (float(item[0].target_abs_error_sec), int(item[0].candidate_index)))[0]
        details.append(
            {
                "correction_id": int(correction_id),
                "track": str(chosen.track),
                "score": float(score),
                "margin": float(margin),
                "selected_error_sec": float(chosen.target_abs_error_sec),
                "oracle_error_sec": float(oracle.target_abs_error_sec),
                "selected_is_oracle": bool(chosen.is_oracle),
                "selected_candidate_index": int(chosen.candidate_index),
                "oracle_candidate_index": int(oracle.candidate_index),
            }
        )
    return details


def _error_metrics(details: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    errors = np.asarray([float(row.get("selected_error_sec", 999.0)) for row in details], dtype=np.float64)
    if errors.size == 0:
        return {}
    return {
        "selected_count": int(errors.size),
        "selected_median_abs_error_sec": float(np.median(errors)),
        "selected_mean_abs_error_sec": float(np.mean(errors)),
        "selected_within_25ms_percent": _pct(int(np.sum(errors <= 0.025)), int(errors.size)),
        "selected_within_50ms_percent": _pct(int(np.sum(errors <= 0.050)), int(errors.size)),
        "selected_within_100ms_percent": _pct(int(np.sum(errors <= 0.100)), int(errors.size)),
        "selected_within_250ms_percent": _pct(int(np.sum(errors <= 0.250)), int(errors.size)),
        "selected_over_1s": int(np.sum(errors > 1.0)),
        "selected_max_abs_error_sec": float(np.max(errors)),
    }


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
    probability_grid = [round(v, 3) for v in np.arange(0.99, 0.49, -0.01)]
    margin_grid = [round(v, 3) for v in np.arange(0.40, -0.001, -0.01)]
    for probability in probability_grid:
        for margin in margin_grid:
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
            "margin": 0.40,
            "accepted": 0,
            "coverage_percent": 0.0,
            "max_error_sec": None,
            "note": "no validation threshold met the requested autonomous safety target",
        }
    return best


def _auto_gate_thresholds(validation_details: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    conservative = _pick_gate(validation_details, max_error_sec=0.025, min_accept=1)
    normal = _pick_gate(validation_details, max_error_sec=0.050, min_accept=1)
    aggressive = _pick_gate(validation_details, max_error_sec=0.100, min_accept=1)

    def thresholds(metrics: Mapping[str, Any], *, fallback_probability: float, fallback_margin: float, micro: float, offset: float, fake: float) -> Dict[str, float]:
        probability = float(metrics.get("probability", fallback_probability) or fallback_probability)
        margin = float(metrics.get("margin", fallback_margin) or fallback_margin)
        confidence = float(np.clip((0.70 * probability) + (0.30 * min(1.0, margin / 0.30)) - 0.02, 0.0, 1.0))
        predicted_error = max(0.005, (1.0 - probability) * float(USABLE_CANDIDATE_SEC) + 0.005)
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
        "conservative": thresholds(conservative, fallback_probability=0.99, fallback_margin=0.40, micro=0.96, offset=20.0, fake=0.25),
        "normal": thresholds(normal, fallback_probability=0.99, fallback_margin=0.40, micro=0.93, offset=45.0, fake=0.35),
        "aggressive": thresholds(aggressive, fallback_probability=0.99, fallback_margin=0.40, micro=0.88, offset=90.0, fake=0.60),
        "validation_gate_metrics": {
            "conservative": conservative,
            "normal": normal,
            "aggressive": aggressive,
        },
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return str(out)


def train_groupwise_ranker(
    *,
    corrections: str,
    output: str,
    report: str,
    random_state: int = 42,
    valid_frac: float = 0.20,
    min_pair_gap_sec: float = 0.015,
) -> Dict[str, Any]:
    data = load_training_data(corrections)
    train_rows, valid_rows = _split_rows(data.rows, valid_frac)
    split_diagnostics = _split_diagnostics(train_rows, valid_rows)
    train_pairwise = _pairwise_training_data(train_rows, min_gap_sec=float(min_pair_gap_sec))
    validation_metrics: Dict[str, Any] = {}
    validation_gate_metrics: Dict[str, Any] = {}
    auto_gate_thresholds: Dict[str, Any] = {}
    if valid_rows:
        validation_model = _make_model(random_state)
        validation_model.fit(train_pairwise.x, train_pairwise.y, sample_weight=train_pairwise.w)
        valid_scores = _score_rows(validation_model, valid_rows)
        validation_details = _selected_details(valid_rows, valid_scores)
        auto_gate_thresholds = _auto_gate_thresholds(validation_details)
        validation_gate_metrics = dict(auto_gate_thresholds.pop("validation_gate_metrics", {}))
        validation_metrics = {
            **_selection_metrics(valid_rows, valid_scores),
            "groupwise": _error_metrics(validation_details),
            "auto_gate_shadow": validation_gate_metrics,
        }

    all_pairwise = _pairwise_training_data(data.rows, min_gap_sec=float(min_pair_gap_sec))
    model = _make_model(random_state)
    model.fit(all_pairwise.x, all_pairwise.y, sample_weight=all_pairwise.w)
    pair_pred = _positive_probability(model, all_pairwise.x)
    train_scores = _score_rows(model, data.rows)
    train_details = _selected_details(data.rows, train_scores)
    trained_at = datetime.now(timezone.utc).isoformat()
    if not auto_gate_thresholds:
        auto_gate_thresholds = {
            "conservative": {
                "probability": 0.99,
                "confidence": 0.95,
                "margin": 0.40,
                "predicted_error": 0.010,
                "micro": 0.96,
                "offset": 20.0,
                "fake": 0.25,
                "disagree_probability": 1.01,
            },
            "normal": {
                "probability": 0.99,
                "confidence": 0.95,
                "margin": 0.40,
                "predicted_error": 0.010,
                "micro": 0.93,
                "offset": 45.0,
                "fake": 0.35,
                "disagree_probability": 1.01,
            },
            "aggressive": {
                "probability": 0.99,
                "confidence": 0.95,
                "margin": 0.40,
                "predicted_error": 0.010,
                "micro": 0.88,
                "offset": 90.0,
                "fake": 0.60,
                "disagree_probability": 1.01,
            },
        }
    payload = {
        "model": model,
        "model_type": "pairwise_ranker_classifier",
        "feature_names": list(CHOOSER_FEATURES),
        "target": "oracle_candidate_beats_other_candidates_in_same_track",
        "good_candidate_threshold_sec": float(GOOD_CANDIDATE_SEC),
        "usable_candidate_threshold_sec": float(USABLE_CANDIDATE_SEC),
        "trained_at": trained_at,
        "corrections_path": str(Path(corrections).expanduser()),
        "training_rows": int(len(data.rows)),
        "training_pair_rows": int(all_pairwise.pair_count),
        "correction_rows": int(data.correction_count),
        "selector_correction_rows": int(data.selector_correction_count),
        "bad_candidate_sets": int(data.bad_candidate_sets),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "min_pair_gap_sec": float(min_pair_gap_sec),
        "auto_gate_thresholds": auto_gate_thresholds,
        "validation_split": split_diagnostics,
    }
    saved_path = save_candidate_chooser_payload(payload, output)
    report_payload = {
        "model_path": saved_path,
        "model_type": "pairwise_ranker_classifier",
        "feature_names": list(CHOOSER_FEATURES),
        "training_rows": int(len(data.rows)),
        "training_pair_rows": int(all_pairwise.pair_count),
        "pairwise_correction_rows": int(all_pairwise.correction_count),
        "correction_rows": int(data.correction_count),
        "selector_correction_rows": int(data.selector_correction_count),
        "bad_candidate_sets": int(data.bad_candidate_sets),
        "oracle_within_25ms_percent": _pct(data.oracle_within_25ms, data.correction_count),
        "oracle_within_100ms_percent": _pct(data.oracle_within_100ms, data.correction_count),
        "oracle_within_250ms_percent": _pct(data.oracle_within_250ms, data.correction_count),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "good_candidate_threshold_sec": float(GOOD_CANDIDATE_SEC),
        "usable_candidate_threshold_sec": float(USABLE_CANDIDATE_SEC),
        "min_pair_gap_sec": float(min_pair_gap_sec),
        **_classification_metrics(all_pairwise.y, pair_pred, all_pairwise.w),
        "train_selection_metrics": {
            **_selection_metrics(data.rows, train_scores),
            "groupwise": _error_metrics(train_details),
        },
        "validation_metrics": validation_metrics,
        "validation_split": split_diagnostics,
        "auto_gate_thresholds": auto_gate_thresholds,
        "trained_at": trained_at,
    }
    report_path = _write_json(report, report_payload)
    report_payload["training_report"] = report_path
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a groupwise pairwise ranker for drop candidate selection.")
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS, help="Correction JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output model pickle path")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Training report JSON path")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    parser.add_argument("--min-pair-gap-sec", type=float, default=0.015)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_groupwise_ranker(
        corrections=str(args.corrections),
        output=str(args.output),
        report=str(args.report),
        random_state=int(args.random_state),
        valid_frac=float(args.valid_frac),
        min_pair_gap_sec=float(args.min_pair_gap_sec),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
