#!/usr/bin/env python3

"""
Train a lightweight logistic marker-ranker from candidate-level evaluation rows.

The resulting JSON model can be consumed by ``first_downbeat_detector.py`` via:

- ``DetectorOptions(marker_rank_model_path="...")``
- or ``DOWNBEAT_MARKER_RANK_MODEL=/path/to/model.json``

This intentionally avoids heavyweight ML dependencies so it can run in the same
environment as the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from first_downbeat_detector import MARKER_MODEL_FEATURES


LOG = logging.getLogger("train_downbeat_marker_ranker")


def _safe_float(value: object) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool01(value: object) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1.0
    if text in {"0", "false", "no"}:
        return 0.0
    try:
        return 1.0 if float(text) > 0.0 else 0.0
    except Exception:
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _group_split(rows: Sequence[Dict[str, object]], valid_frac: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if valid_frac <= 0.0:
        return list(rows), []
    train_rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    for row in rows:
        track_rel = str(row.get("track_rel") or "")
        bucket = (sum(ord(ch) for ch in track_rel) % 1000) / 1000.0
        if bucket < valid_frac:
            valid_rows.append(dict(row))
        else:
            train_rows.append(dict(row))
    if not train_rows:
        return list(rows), []
    return train_rows, valid_rows


def _prepare_matrix(rows: Sequence[Dict[str, object]], label_col: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    filtered: List[Dict[str, object]] = []
    x_rows: List[List[float]] = []
    y_vals: List[float] = []
    for row in rows:
        y = _safe_bool01(row.get(label_col))
        if y is None:
            continue
        feats: List[float] = []
        valid = True
        for name in MARKER_MODEL_FEATURES:
            value = _feature_value_from_row(row, name)
            if value is None:
                valid = False
                break
            feats.append(float(value))
        if not valid:
            continue
        filtered.append(dict(row))
        x_rows.append(feats)
        y_vals.append(float(y))
    if not x_rows:
        return np.zeros((0, len(MARKER_MODEL_FEATURES)), dtype=np.float64), np.zeros((0,), dtype=np.float64), []
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_vals, dtype=np.float64), filtered


def _feature_value_from_row(row: Dict[str, object], name: str) -> Optional[float]:
    direct = _safe_float(row.get(f"feature_{name}"))
    if direct is not None:
        return direct
    if name == "cluster_index_norm":
        cluster_index = _safe_float(row.get("feature_cluster_index"))
        marker_count = _safe_float(row.get("event_marker_count"))
        if cluster_index is None or marker_count is None:
            return None
        denom = max(1.0, float(marker_count) - 1.0)
        return float(cluster_index) / denom
    if name == "event_score_norm":
        return _safe_float(row.get("event_score"))
    if name == "event_marker_count_norm":
        marker_count = _safe_float(row.get("event_marker_count"))
        if marker_count is None:
            return None
        return min(1.0, max(0.0, float(marker_count) - 1.0) / 3.0)
    if name == "event_plausible_count_norm":
        plausible_count = _safe_float(row.get("event_plausible_marker_count"))
        marker_count = _safe_float(row.get("event_marker_count"))
        if plausible_count is None or marker_count is None:
            return None
        return min(1.0, float(plausible_count) / max(1.0, float(marker_count)))
    return None


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pos = float(np.sum(y > 0.5))
    neg = float(np.sum(y <= 0.5))
    if pos <= 0.0 or neg <= 0.0:
        return np.ones_like(y, dtype=np.float64)
    pos_w = neg / max(1.0, pos)
    return np.where(y > 0.5, pos_w, 1.0).astype(np.float64)


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iters: int,
    learning_rate: float,
    l2: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if x.size == 0:
        raise ValueError("No training rows were available after filtering.")
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    xn = (x - means) / scales

    weights = np.zeros((x.shape[1],), dtype=np.float64)
    bias = 0.0
    sample_weights = _balanced_weights(y)

    for _ in range(max_iters):
        logits = xn.dot(weights) + bias
        preds = _sigmoid(logits)
        err = (preds - y) * sample_weights
        grad_w = (xn.T.dot(err) / max(1.0, float(xn.shape[0]))) + (l2 * weights)
        grad_b = float(np.mean(err))
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
    return weights, float(bias), means, scales


def _binary_metrics(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, means: np.ndarray, scales: np.ndarray) -> Dict[str, Optional[float]]:
    if x.size == 0 or y.size == 0:
        return {"rows": 0, "logloss": None, "accuracy": None, "precision": None, "recall": None}
    xn = (x - means) / np.where(scales < 1e-8, 1.0, scales)
    probs = _sigmoid(xn.dot(weights) + float(bias))
    preds = probs >= 0.5
    eps = 1e-8
    logloss = -float(np.mean((y * np.log(np.clip(probs, eps, 1.0 - eps))) + ((1.0 - y) * np.log(np.clip(1.0 - probs, eps, 1.0 - eps)))))
    accuracy = float(np.mean(preds == (y > 0.5)))
    tp = float(np.sum((preds == 1) & (y > 0.5)))
    fp = float(np.sum((preds == 1) & (y <= 0.5)))
    fn = float(np.sum((preds == 0) & (y > 0.5)))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else None
    return {
        "rows": int(y.size),
        "logloss": logloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def _read_candidate_rows(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def train_model(
    *,
    candidate_csv: str,
    out_model: str,
    label_col: str,
    max_iters: int,
    learning_rate: float,
    l2: float,
    valid_frac: float,
) -> Dict[str, object]:
    rows = _read_candidate_rows(candidate_csv)
    train_rows, valid_rows = _group_split(rows, valid_frac=valid_frac)
    x_train, y_train, train_filtered = _prepare_matrix(train_rows, label_col)
    x_valid, y_valid, _ = _prepare_matrix(valid_rows, label_col)
    weights, bias, means, scales = _fit_logistic_regression(
        x_train,
        y_train,
        max_iters=max_iters,
        learning_rate=learning_rate,
        l2=l2,
    )

    payload: Dict[str, object] = {
        "model_type": "logistic_regression",
        "version": 1,
        "label_name": str(label_col),
        "feature_order": list(MARKER_MODEL_FEATURES),
        "means": [float(v) for v in means.tolist()],
        "scales": [float(v) for v in scales.tolist()],
        "weights": [float(v) for v in weights.tolist()],
        "bias": float(bias),
        "score_scale": 1.0,
        "training_rows": int(x_train.shape[0]),
        "validation_rows": int(x_valid.shape[0]),
        "train_metrics": _binary_metrics(x_train, y_train, weights, bias, means, scales),
        "validation_metrics": _binary_metrics(x_valid, y_valid, weights, bias, means, scales),
        "source_candidate_csv": os.path.abspath(candidate_csv),
    }
    with open(out_model, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a simple logistic Ableton marker ranker from candidate-level CSV rows.")
    ap.add_argument("--candidate-csv", required=True, help="CSV produced by evaluate_downbeat_detector.py --out-candidate-csv")
    ap.add_argument("--out-model", required=True, help="Output JSON model path")
    ap.add_argument("--label-col", default="candidate_is_manual_match_25ms", help="Binary label column to learn")
    ap.add_argument("--max-iters", type=int, default=1200, help="Gradient descent iterations")
    ap.add_argument("--learning-rate", type=float, default=0.08, help="Gradient descent learning rate")
    ap.add_argument("--l2", type=float, default=0.001, help="L2 regularization strength")
    ap.add_argument("--valid-frac", type=float, default=0.2, help="Group-holdout fraction by track_rel")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(message)s")

    payload = train_model(
        candidate_csv=args.candidate_csv,
        out_model=args.out_model,
        label_col=str(args.label_col),
        max_iters=int(args.max_iters),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        valid_frac=float(args.valid_frac),
    )
    summary = {
        "out_model": os.path.abspath(args.out_model),
        "label_name": payload["label_name"],
        "training_rows": payload["training_rows"],
        "validation_rows": payload["validation_rows"],
        "train_metrics": payload["train_metrics"],
        "validation_metrics": payload["validation_metrics"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
