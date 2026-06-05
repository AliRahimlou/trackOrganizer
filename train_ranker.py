#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.ranker import MODEL_FEATURES, candidate_feature_vector, candidate_timestamp, save_ranker_payload


ACCEPT_TOLERANCE_SEC = 1e-3


@dataclass
class TrainingData:
    x: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    meta: List[Dict[str, Any]]
    total_corrections: int
    accepted_picks: int
    corrected_picks: int
    deltas: List[float]


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


def _candidate_rows(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
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
        t = candidate_timestamp(candidate)
        key = f"{t:.6f}" if t is not None else json.dumps(dict(candidate), sort_keys=True, default=str)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("user_pick")
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _ai_pick(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick"):
        value = row.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _selected_candidate_timestamp(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Optional[float]:
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        timestamp = candidate_timestamp(selected)
        if timestamp is not None:
            return timestamp
    for candidate in candidates:
        if candidate.get("selected"):
            timestamp = candidate_timestamp(candidate)
            if timestamp is not None:
                return timestamp
    return None


def _is_accepted_pick(user_pick: float, ai_pick: float) -> bool:
    return abs(float(user_pick) - float(ai_pick)) <= ACCEPT_TOLERANCE_SEC


def _accepted_candidate_label(candidate_timestamp_sec: float, selected_timestamp: Optional[float], ai_pick: float) -> float:
    if selected_timestamp is not None and abs(float(candidate_timestamp_sec) - float(selected_timestamp)) <= ACCEPT_TOLERANCE_SEC:
        return 0.0
    if abs(float(candidate_timestamp_sec) - float(ai_pick)) <= ACCEPT_TOLERANCE_SEC:
        return 0.0
    return abs(float(candidate_timestamp_sec) - float(ai_pick))


def load_training_rows(corrections_path: str) -> TrainingData:
    path = Path(corrections_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Correction log not found: {path}")

    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    weights: List[float] = []
    meta: List[Dict[str, Any]] = []
    total_corrections = 0
    accepted_picks = 0
    corrected_picks = 0
    deltas: List[float] = []

    for row in _iter_jsonl(path):
        if row_has_excluded_path(row):
            continue
        user_pick = _user_pick(row)
        ai_pick = _ai_pick(row)
        if user_pick is None or ai_pick is None:
            continue
        candidates = _candidate_rows(row)
        if not candidates:
            continue
        accepted = _is_accepted_pick(user_pick, ai_pick)
        selected_timestamp = _selected_candidate_timestamp(row, candidates)
        sample_weight = 2.0 if accepted else 1.0
        delta = float(user_pick) - float(ai_pick)
        total_corrections += 1
        accepted_picks += 1 if accepted else 0
        corrected_picks += 0 if accepted else 1
        deltas.append(delta)

        for candidate in candidates:
            candidate_time = candidate_timestamp(candidate)
            if candidate_time is None:
                continue
            if accepted:
                label = _accepted_candidate_label(float(candidate_time), selected_timestamp, float(ai_pick))
            else:
                label = abs(float(user_pick) - float(candidate_time))
            x_rows.append(candidate_feature_vector(candidate, MODEL_FEATURES))
            y_rows.append(float(label))
            weights.append(float(sample_weight))
            meta.append(
                {
                    "track": row.get("track", ""),
                    "ai_pick": float(ai_pick),
                    "user_pick": float(user_pick),
                    "candidate_timestamp": float(candidate_time),
                    "label_abs_delta_sec": float(label),
                    "sample_weight": float(sample_weight),
                    "accepted_pick": bool(accepted),
                    "handcrafted_rank": candidate.get("handcrafted_rank", candidate.get("rank")),
                    "model_rank": candidate.get("model_rank"),
                }
            )

    if not x_rows:
        raise ValueError(f"No usable candidate rows found in correction log: {path}")
    return TrainingData(
        x=np.asarray(x_rows, dtype=np.float64),
        y=np.asarray(y_rows, dtype=np.float64),
        sample_weight=np.asarray(weights, dtype=np.float64),
        meta=meta,
        total_corrections=int(total_corrections),
        accepted_picks=int(accepted_picks),
        corrected_picks=int(corrected_picks),
        deltas=deltas,
    )


def _make_model(model_type: str, random_state: int):
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=160,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            random_state=random_state,
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _feature_importances(model: object) -> Dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return {name: float(value) for name, value in zip(MODEL_FEATURES, np.asarray(values, dtype=np.float64))}


def _write_report(report: Mapping[str, Any], report_path: str) -> str:
    path = Path(report_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, indent=2, ensure_ascii=True)
    return str(path)


def train_ranker(
    *,
    corrections: str,
    output: str,
    report: str = "models/training_report.json",
    model_type: str = "random_forest",
    random_state: int = 42,
) -> Dict[str, Any]:
    data = load_training_rows(corrections)
    model = _make_model(model_type, random_state)
    model.fit(data.x, data.y, sample_weight=data.sample_weight)
    pred = np.asarray(model.predict(data.x), dtype=np.float64)
    mae = float(mean_absolute_error(data.y, pred, sample_weight=data.sample_weight))
    importances = _feature_importances(model)
    trained_at = datetime.now(timezone.utc).isoformat()
    avg_delta = float(np.mean(data.deltas)) if data.deltas else 0.0
    avg_abs_delta = float(np.mean(np.abs(data.deltas))) if data.deltas else 0.0

    payload = {
        "model": model,
        "model_type": model_type,
        "feature_names": list(MODEL_FEATURES),
        "target": "abs_delta_to_user_pick_sec",
        "trained_at": trained_at,
        "corrections_path": str(Path(corrections).expanduser()),
        "training_rows": int(len(data.y)),
        "correction_rows": int(data.total_corrections),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "average_delta_sec": avg_delta,
        "average_abs_delta_sec": avg_abs_delta,
        "train_mae_sec": mae,
        "target_min_sec": float(np.min(data.y)),
        "target_median_sec": float(np.median(data.y)),
        "target_max_sec": float(np.max(data.y)),
        "feature_importances": importances,
        "sample_meta": data.meta[:25],
    }
    saved_path = save_ranker_payload(payload, output)
    report_payload = {
        "model_path": saved_path,
        "model_type": model_type,
        "feature_names": list(MODEL_FEATURES),
        "training_rows": int(len(data.y)),
        "total_corrections": int(data.total_corrections),
        "correction_rows": int(data.total_corrections),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "average_delta_sec": avg_delta,
        "average_abs_delta_sec": avg_abs_delta,
        "train_mae_sec": mae,
        "target_min_sec": float(np.min(data.y)),
        "target_median_sec": float(np.median(data.y)),
        "target_max_sec": float(np.max(data.y)),
        "sample_weight_sum": float(np.sum(data.sample_weight)),
        "feature_importances": importances,
        "trained_at": trained_at,
    }
    report_path = _write_report(report_payload, report)
    report_payload["training_report"] = report_path
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Ali's 1.1.1 candidate re-ranker from correction logs.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL path")
    parser.add_argument("--output", default="models/drop_ranker.pkl", help="Output model pickle path")
    parser.add_argument("--report", default="models/training_report.json", help="Training report JSON path")
    parser.add_argument("--model", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_ranker(
        corrections=args.corrections,
        output=args.output,
        report=args.report,
        model_type=args.model,
        random_state=int(args.random_state),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
