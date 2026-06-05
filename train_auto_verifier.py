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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, log_loss, precision_recall_fscore_support, roc_auc_score

from drop_aligner.auto_verifier import AUTO_VERIFIER_FEATURES, save_auto_verifier_payload, verifier_feature_rows
from drop_aligner.exclusions import row_has_excluded_path


DEFAULT_CORRECTIONS = Path("models/multistem_training_corrections.jsonl")
DEFAULT_OUTPUT = Path("models/drop_auto_verifier.pkl")
DEFAULT_REPORT = Path("models/auto_verifier_report.json")


@dataclass(frozen=True)
class VerifierRow:
    correction_id: int
    track: str
    ai_pick: Optional[float]
    user_pick: float
    candidate_index: int
    candidate_time: float
    abs_error_sec: float
    within_5ms: int
    within_10ms: int
    within_25ms: int
    within_50ms: int
    sample_weight: float
    vector: List[float]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _candidate_rows(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    source: List[Any] = []
    candidates = row.get("top_10_candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        source.extend(candidates)
    else:
        candidates = row.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            source.extend(candidates)
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        source.append(selected)

    out: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in source:
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


def _split_rows(rows: Sequence[VerifierRow], valid_frac: float) -> Tuple[List[VerifierRow], List[VerifierRow], Dict[str, Any]]:
    if valid_frac <= 0.0:
        return list(rows), [], {
            "strategy": "none",
            "train_tracks": len({row.track for row in rows}),
            "validation_tracks": 0,
            "leakage_check_passed": True,
        }
    validation_tracks = {row.track for row in rows if _stable_bucket(row.track or str(row.correction_id)) < float(valid_frac)}
    train = [row for row in rows if row.track not in validation_tracks]
    valid = [row for row in rows if row.track in validation_tracks]
    if not train or not valid:
        train, valid = list(rows), []
    train_tracks = {row.track for row in train}
    valid_tracks = {row.track for row in valid}
    overlap = sorted(track for track in train_tracks.intersection(valid_tracks) if track)
    split = {
        "strategy": "track_grouped_stable_hash",
        "train_tracks": int(len(train_tracks)),
        "validation_tracks": int(len(valid_tracks)),
        "train_candidates": int(len(train)),
        "validation_candidates": int(len(valid)),
        "train_corrections": int(len({row.correction_id for row in train})),
        "validation_corrections": int(len({row.correction_id for row in valid})),
        "leakage_check_passed": bool(not overlap),
        "overlap_track_count": int(len(overlap)),
        "overlap_tracks_sample": overlap[:20],
    }
    return train, valid, split


def _sample_weight(error: float) -> float:
    if error <= 0.005:
        return 6.0
    if error <= 0.010:
        return 4.5
    if error <= 0.025:
        return 3.5
    if error <= 0.050:
        return 2.0
    return 1.0


def load_verifier_rows(corrections: str) -> List[VerifierRow]:
    path = Path(corrections).expanduser()
    rows: List[VerifierRow] = []
    for correction_id, row in enumerate(_iter_jsonl(path)):
        if row_has_excluded_path(row):
            continue
        track = str(row.get("track") or row.get("filename") or "")
        user_pick = _safe_float(row.get("user_pick"))
        if not track or user_pick is None:
            continue
        ai_pick = _ai_pick(row)
        feature_rows = verifier_feature_rows(_candidate_rows(row))
        for feature_row in feature_rows:
            candidate_time = float(feature_row["time_sec"])
            error = abs(candidate_time - float(user_pick))
            rows.append(
                VerifierRow(
                    correction_id=int(correction_id),
                    track=track,
                    ai_pick=ai_pick,
                    user_pick=float(user_pick),
                    candidate_index=int(feature_row["index"]),
                    candidate_time=float(candidate_time),
                    abs_error_sec=float(error),
                    within_5ms=1 if error <= 0.005 else 0,
                    within_10ms=1 if error <= 0.010 else 0,
                    within_25ms=1 if error <= 0.025 else 0,
                    within_50ms=1 if error <= 0.050 else 0,
                    sample_weight=_sample_weight(error),
                    vector=[float(value) for value in feature_row["vector"]],
                )
            )
    if not rows:
        raise ValueError(f"No verifier rows found in {path}")
    return rows


def _matrix(rows: Sequence[VerifierRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row.vector for row in rows], dtype=np.float64)
    y = np.asarray([row.within_25ms for row in rows], dtype=np.int64)
    target = np.asarray([min(float(row.abs_error_sec), 5.0) for row in rows], dtype=np.float64)
    w = np.asarray([row.sample_weight for row in rows], dtype=np.float64)
    return x, y, target, w


def _make_classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=int(random_state),
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.75,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def _make_regressor(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        random_state=int(random_state),
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
    )


def _positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, -1]
    return np.asarray(model.predict(x), dtype=np.float64)


def _classification_metrics(labels: Sequence[int], scores: Sequence[float], weights: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    w = np.asarray(weights, dtype=np.float64)
    out: Dict[str, Any] = {
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
    }
    if y.size == 0 or len(set(int(value) for value in y.tolist())) < 2:
        return out
    out["log_loss"] = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    out["roc_auc"] = float(roc_auc_score(y, p, sample_weight=w))
    out["average_precision"] = float(average_precision_score(y, p, sample_weight=w))
    hard = (p >= 0.50).astype(np.int64)
    precision, recall, f1, _support = precision_recall_fscore_support(y, hard, labels=[1], zero_division=0)
    out["precision_at_0_50"] = float(precision[0]) if len(precision) else 0.0
    out["recall_at_0_50"] = float(recall[0]) if len(recall) else 0.0
    out["f1_at_0_50"] = float(f1[0]) if len(f1) else 0.0
    return out


def _calibration_curve(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> List[Dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for idx in range(int(bins)):
        low = idx / float(bins)
        high = (idx + 1) / float(bins)
        mask = (p >= low) & (p < high if idx < bins - 1 else p <= high)
        if not np.any(mask):
            continue
        out.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(np.sum(mask)),
                "mean_predicted": float(np.mean(p[mask])),
                "actual_within_25ms_rate": float(np.mean(y[mask])),
            }
        )
    return out


def _selected_validation_rows(rows: Sequence[VerifierRow]) -> List[VerifierRow]:
    grouped: Dict[int, List[VerifierRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.correction_id), []).append(row)
    selected: List[VerifierRow] = []
    for group in grouped.values():
        ai_pick = next((row.ai_pick for row in group if row.ai_pick is not None), None)
        if ai_pick is None:
            chosen = min(group, key=lambda row: (float(row.abs_error_sec), int(row.candidate_index)))
        else:
            chosen = min(group, key=lambda row: (abs(float(row.candidate_time) - float(ai_pick)), int(row.candidate_index)))
        selected.append(chosen)
    return selected


def _threshold_coverage(selected_rows: Sequence[VerifierRow], scores: Sequence[float], predicted_errors: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    errors = np.asarray([row.abs_error_sec for row in selected_rows], dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    pe = np.asarray(predicted_errors, dtype=np.float64)
    for threshold in (0.90, 0.95, 0.975, 0.99):
        accepted = (p >= float(threshold))
        accepted_count = int(np.sum(accepted))
        key = f"p_{str(threshold).replace('.', '_')}"
        item: Dict[str, Any] = {
            "probability_threshold": float(threshold),
            "accepted": accepted_count,
            "coverage_percent": None if len(selected_rows) == 0 else 100.0 * accepted_count / float(len(selected_rows)),
        }
        if accepted_count:
            accepted_errors = errors[accepted]
            item.update(
                {
                    "false_auto_save_count": int(np.sum(accepted_errors > 0.025)),
                    "false_auto_save_rate_percent": 100.0 * float(np.sum(accepted_errors > 0.025)) / float(accepted_count),
                    "within_25ms_percent": 100.0 * float(np.sum(accepted_errors <= 0.025)) / float(accepted_count),
                    "max_error_ms": float(np.max(accepted_errors) * 1000.0),
                    "median_predicted_error_ms": float(np.median(pe[accepted]) * 1000.0),
                }
            )
        out[key] = item
    return out


def train_auto_verifier(
    *,
    corrections: str,
    output: str,
    report: str,
    random_state: int = 42,
    valid_frac: float = 0.20,
) -> Dict[str, Any]:
    rows = load_verifier_rows(corrections)
    train_rows, valid_rows, split = _split_rows(rows, valid_frac)
    x_train, y_train, target_train, w_train = _matrix(train_rows)
    classifier = _make_classifier(random_state)
    regressor = _make_regressor(random_state)
    classifier.fit(x_train, y_train, sample_weight=w_train)
    regressor.fit(x_train, target_train, sample_weight=w_train)

    validation_metrics: Dict[str, Any] = {}
    if valid_rows:
        x_valid, y_valid, target_valid, w_valid = _matrix(valid_rows)
        valid_scores = _positive_probability(classifier, x_valid)
        raw_valid_errors = np.maximum(0.0, np.asarray(regressor.predict(x_valid), dtype=np.float64))
        valid_errors = np.minimum(raw_valid_errors, np.maximum(0.0, 1.0 - valid_scores) * 0.100)
        selected_rows = _selected_validation_rows(valid_rows)
        selected_lookup = {(row.correction_id, row.candidate_index): idx for idx, row in enumerate(valid_rows)}
        selected_indices = [selected_lookup[(row.correction_id, row.candidate_index)] for row in selected_rows]
        selected_scores = valid_scores[selected_indices] if selected_indices else np.asarray([], dtype=np.float64)
        selected_predicted_errors = valid_errors[selected_indices] if selected_indices else np.asarray([], dtype=np.float64)
        validation_metrics = {
            "candidate_classifier": _classification_metrics(y_valid, valid_scores, w_valid),
            "candidate_regressor_mae_ms": float(np.mean(np.abs(valid_errors - target_valid)) * 1000.0),
            "candidate_regressor_median_abs_error_ms": float(np.median(np.abs(valid_errors - target_valid)) * 1000.0),
            "calibration_curve": _calibration_curve(y_valid, valid_scores),
            "selected_count": int(len(selected_rows)),
            "selected_actual_within_25ms_percent": None
            if not selected_rows
            else 100.0 * float(sum(row.abs_error_sec <= 0.025 for row in selected_rows)) / float(len(selected_rows)),
            "threshold_coverage": _threshold_coverage(selected_rows, selected_scores, selected_predicted_errors),
        }

    x_all, y_all, target_all, w_all = _matrix(rows)
    final_classifier = _make_classifier(random_state)
    final_regressor = _make_regressor(random_state)
    final_classifier.fit(x_all, y_all, sample_weight=w_all)
    final_regressor.fit(x_all, target_all, sample_weight=w_all)
    trained_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "classifier": final_classifier,
        "regressor": final_regressor,
        "model_type": "random_forest_auto_verifier",
        "feature_names": list(AUTO_VERIFIER_FEATURES),
        "target": "candidate_within_25ms_and_abs_error_sec",
        "trained_at": trained_at,
        "corrections_path": str(Path(corrections).expanduser()),
        "training_rows": int(len(rows)),
        "correction_rows": int(len({row.correction_id for row in rows})),
        "validation_split": split,
    }
    saved_path = save_auto_verifier_payload(payload, output)
    report_payload = {
        "model_path": saved_path,
        "model_type": "random_forest_auto_verifier",
        "feature_names": list(AUTO_VERIFIER_FEATURES),
        "training_rows": int(len(rows)),
        "correction_rows": int(len({row.correction_id for row in rows})),
        "positive_within_25ms_rows": int(np.sum(y_all == 1)),
        "negative_rows": int(np.sum(y_all == 0)),
        "validation_split": split,
        "validation_metrics": validation_metrics,
        "trained_at": trained_at,
    }
    out_path = Path(report).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    report_payload["training_report"] = str(out_path)
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an auto-save verifier for selected drop candidates.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS), help="Expanded multistem correction JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output auto verifier pickle")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output verifier report JSON")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = train_auto_verifier(
        corrections=str(args.corrections),
        output=str(args.output),
        report=str(args.report),
        random_state=int(args.random_state),
        valid_frac=float(args.valid_frac),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
