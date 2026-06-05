#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from drop_aligner.detector import (
    DropDetectorConfig,
    _full_track_candidate_peak_times,
    _snap_and_rank_candidates,
    extract_features,
    score_candidates,
)
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.region_model import (
    REGION_FEATURES,
    candidate_feature_rows,
    candidate_region_time,
    save_region_model_payload,
)


BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.IGNORECASE)
TARGET_CAP_SEC = 8.0


@dataclass
class RegionRow:
    correction_id: int
    track: str
    ai_pick: float
    user_pick: float
    candidate_index: int
    candidate_time: float
    target_abs_error_sec: float
    oracle_abs_error_sec: float
    is_oracle: bool
    sample_weight: float
    vector: List[float]


@dataclass
class TrainingData:
    rows: List[RegionRow]
    correction_count: int
    accepted_picks: int
    corrected_picks: int
    skipped_rows: int
    failed_rows: int
    failures: List[Dict[str, str]]


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


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(row.get("user_pick"))


def _ai_pick(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _track_path(row: Mapping[str, Any]) -> Optional[Path]:
    for key in ("track", "filename", "audio_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return path
    return None


def _infer_bpm_from_path(path: Path) -> Optional[float]:
    match = BPM_RE.match(path.name)
    if match:
        return float(match.group(1))
    for parent in [path.parent, *path.parents]:
        try:
            bpm = int(parent.name)
        except ValueError:
            continue
        if 60 <= bpm <= 220:
            return float(bpm)
    return None


def _stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _row_weight(
    *,
    target: float,
    is_oracle: bool,
    correction_id: int,
    total_rows: int,
    reviewed_from: str = "",
) -> float:
    weight = 1.0
    if target <= 0.025:
        weight *= 12.0
    elif target <= 0.050:
        weight *= 8.0
    elif target <= 0.100:
        weight *= 5.0
    elif target <= 0.250:
        weight *= 2.5
    elif is_oracle:
        weight *= 2.0
    if str(reviewed_from) == "web_candidate_pick":
        weight *= 2.25
    if total_rows > 1:
        recency = correction_id / float(max(1, total_rows - 1))
        weight *= 0.75 + (0.75 * recency)
    return float(weight)


def _build_track_candidates(
    track: Path,
    *,
    user_pick: float,
    ai_pick: float,
    cfg: DropDetectorConfig,
    max_candidates_per_track: int,
) -> Tuple[float, List[Any]]:
    bpm = _infer_bpm_from_path(track)
    features = extract_features(str(track), cfg, bpm=bpm)
    cues = [float(user_pick), float(ai_pick)]
    peak_times = _full_track_candidate_peak_times(features, cfg, external_cues=cues)
    if not peak_times:
        peak_times = cues
    candidates = score_candidates(features, peak_times, cfg, drumprint_analysis=None, drumprint_status="disabled")
    ranked = _snap_and_rank_candidates(candidates, features, cfg)
    if not ranked:
        return float(features.duration_sec), []

    closest = min(ranked, key=lambda candidate: abs(float(candidate.snapped_sec) - float(user_pick)))
    ranked_by_score = sorted(ranked, key=lambda candidate: (-float(candidate.score), float(candidate.time_sec)))
    limit = max(10, int(max_candidates_per_track))
    kept = ranked_by_score[:limit]
    if closest not in kept:
        kept.append(closest)
    kept.sort(key=lambda candidate: float(candidate.time_sec))
    for rank, candidate in enumerate(sorted(kept, key=lambda candidate: (-float(candidate.score), float(candidate.time_sec))), start=1):
        candidate.handcrafted_rank = int(rank)
        candidate.rank = int(rank)
    return float(features.duration_sec), kept


def load_training_data(
    corrections_path: str,
    *,
    sample_rate: Optional[int],
    max_candidates_per_track: int,
    limit: int,
) -> TrainingData:
    path = Path(corrections_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Correction log not found: {path}")
    raw_rows = [row for row in _iter_jsonl(path) if not row_has_excluded_path(row)]
    if limit > 0:
        raw_rows = raw_rows[-int(limit) :]

    cfg = DropDetectorConfig(
        sample_rate=sample_rate,
        use_drumprint=False,
        use_microalign=False,
        use_ranker_model=False,
        use_region_model=False,
        hpss=False,
        candidate_prominence=0.055,
        max_drop_time_ratio=0.82,
    )
    rows: List[RegionRow] = []
    accepted = 0
    corrected = 0
    usable_corrections = 0
    skipped = 0
    failed = 0
    failures: List[Dict[str, str]] = []

    for correction_id, row in enumerate(raw_rows):
        user = _user_pick(row)
        ai = _ai_pick(row)
        track = _track_path(row)
        if user is None or ai is None or track is None:
            skipped += 1
            continue
        try:
            duration, candidates = _build_track_candidates(
                track,
                user_pick=float(user),
                ai_pick=float(ai),
                cfg=cfg,
                max_candidates_per_track=max_candidates_per_track,
            )
        except Exception as exc:
            failed += 1
            failures.append({"track": str(track), "error": str(exc) or exc.__class__.__name__})
            continue
        feature_rows = candidate_feature_rows(candidates, duration_sec=duration)
        if not feature_rows:
            skipped += 1
            continue

        candidate_times = []
        for item in feature_rows:
            candidate = item["candidate"]
            snapped = _safe_float(getattr(candidate, "snapped_sec", None))
            candidate_times.append(float(snapped if snapped is not None else item["time_sec"]))
        targets = [abs(float(t) - float(user)) for t in candidate_times]
        oracle_error = min(targets)
        oracle_index = int(np.argmin(np.asarray(targets, dtype=np.float64)))
        usable_corrections += 1
        if abs(float(user) - float(ai)) <= 0.001:
            accepted += 1
        else:
            corrected += 1

        for item_index, item in enumerate(feature_rows):
            target = float(targets[item_index])
            rows.append(
                RegionRow(
                    correction_id=int(correction_id),
                    track=str(track),
                    ai_pick=float(ai),
                    user_pick=float(user),
                    candidate_index=int(item["index"]),
                    candidate_time=float(candidate_times[item_index]),
                    target_abs_error_sec=target,
                    oracle_abs_error_sec=float(oracle_error),
                    is_oracle=bool(item_index == oracle_index),
                    sample_weight=_row_weight(
                        target=target,
                        is_oracle=bool(item_index == oracle_index),
                        correction_id=correction_id,
                        total_rows=len(raw_rows),
                        reviewed_from=str(row.get("reviewed_from") or ""),
                    ),
                    vector=[float(v) for v in item["vector"]],
                )
            )
        print(
            f"[{usable_corrections}/{len(raw_rows)}] training candidates: {track.name} ({len(feature_rows)} rows)",
            flush=True,
        )

    if not rows:
        raise ValueError(f"No usable region training rows found in correction log: {path}")
    return TrainingData(
        rows=rows,
        correction_count=usable_corrections,
        accepted_picks=accepted,
        corrected_picks=corrected,
        skipped_rows=skipped,
        failed_rows=failed,
        failures=failures,
    )


def _make_model(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        random_state=int(random_state),
        n_estimators=450,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
    )


def _matrix(rows: Sequence[RegionRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row.vector for row in rows], dtype=np.float64)
    y = np.asarray([min(float(row.target_abs_error_sec), TARGET_CAP_SEC) for row in rows], dtype=np.float64)
    w = np.asarray([row.sample_weight for row in rows], dtype=np.float64)
    return x, y, w


def _split_rows(rows: Sequence[RegionRow], valid_frac: float) -> Tuple[List[RegionRow], List[RegionRow]]:
    if valid_frac <= 0.0:
        return list(rows), []
    train: List[RegionRow] = []
    valid: List[RegionRow] = []
    for row in rows:
        key = row.track or str(row.correction_id)
        if _stable_bucket(key) < float(valid_frac):
            valid.append(row)
        else:
            train.append(row)
    if not train or not valid:
        return list(rows), []
    return train, valid


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
            f"{prefix}_within_25ms_percent": _pct(int(np.sum(values <= 0.025)), int(values.size)),
            f"{prefix}_within_50ms_percent": _pct(int(np.sum(values <= 0.050)), int(values.size)),
            f"{prefix}_within_100ms_percent": _pct(int(np.sum(values <= 0.100)), int(values.size)),
            f"{prefix}_within_250ms_percent": _pct(int(np.sum(values <= 0.250)), int(values.size)),
            f"{prefix}_within_1s_percent": _pct(int(np.sum(values <= 1.000)), int(values.size)),
            f"{prefix}_over_1s": int(np.sum(values > 1.0)),
        }
    if extra:
        out.update(dict(extra))
    return out


def _selection_metrics(rows: Sequence[RegionRow], predictions: Sequence[float]) -> Dict[str, Any]:
    grouped: Dict[int, List[Tuple[RegionRow, float]]] = {}
    for row, pred in zip(rows, predictions):
        grouped.setdefault(row.correction_id, []).append((row, max(0.0, float(pred))))
    selected_errors: List[float] = []
    oracle_errors: List[float] = []
    ai_errors: List[float] = []
    selected_is_oracle = 0
    for group in grouped.values():
        chosen, _pred = min(
            group,
            key=lambda item: (
                float(item[1]),
                int(item[0].candidate_index),
                float(item[0].candidate_time),
            ),
        )
        selected_errors.append(float(chosen.target_abs_error_sec))
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
            "oracle_within_250ms_percent": _pct(sum(err <= 0.250 for err in oracle_errors), len(oracle_errors)),
            "ai_within_25ms_percent": _pct(sum(err <= 0.025 for err in ai_errors), len(ai_errors)),
            "ai_within_100ms_percent": _pct(sum(err <= 0.100 for err in ai_errors), len(ai_errors)),
            "ai_within_250ms_percent": _pct(sum(err <= 0.250 for err in ai_errors), len(ai_errors)),
        },
    )


def _feature_importances(model: object) -> Dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return {name: float(value) for name, value in zip(REGION_FEATURES, np.asarray(values, dtype=np.float64))}


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
    return str(out)


def train_drop_region_detector(
    *,
    corrections: str,
    output: str,
    report: str,
    sample_rate: Optional[int],
    max_candidates_per_track: int,
    random_state: int = 42,
    valid_frac: float = 0.20,
    limit: int = 0,
) -> Dict[str, Any]:
    data = load_training_data(
        corrections,
        sample_rate=sample_rate,
        max_candidates_per_track=max_candidates_per_track,
        limit=limit,
    )
    train_rows, valid_rows = _split_rows(data.rows, valid_frac)
    x_train, y_train, w_train = _matrix(train_rows)
    validation_metrics: Dict[str, Any] = {}
    if valid_rows:
        validation_model = _make_model(random_state)
        validation_model.fit(x_train, y_train, sample_weight=w_train)
        x_valid, y_valid, w_valid = _matrix(valid_rows)
        valid_pred = np.asarray(validation_model.predict(x_valid), dtype=np.float64)
        validation_metrics = {
            "candidate_mae_sec": float(mean_absolute_error(y_valid, valid_pred, sample_weight=w_valid)),
            **_selection_metrics(valid_rows, valid_pred),
        }

    x_all, y_all, w_all = _matrix(data.rows)
    model = _make_model(random_state)
    model.fit(x_all, y_all, sample_weight=w_all)
    pred_all = np.asarray(model.predict(x_all), dtype=np.float64)
    trained_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "model": model,
        "model_type": "random_forest_regressor",
        "feature_names": list(REGION_FEATURES),
        "target": "full_track_candidate_abs_error_to_user_marker_sec",
        "target_cap_sec": float(TARGET_CAP_SEC),
        "trained_at": trained_at,
        "corrections_path": str(Path(corrections).expanduser()),
        "training_rows": int(len(data.rows)),
        "correction_rows": int(data.correction_count),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "sample_rate": None if sample_rate is None else int(sample_rate),
        "max_candidates_per_track": int(max_candidates_per_track),
        "feature_importances": _feature_importances(model),
    }
    saved_path = save_region_model_payload(payload, output)

    report_payload = {
        "model_path": saved_path,
        "model_type": "random_forest_regressor",
        "feature_names": list(REGION_FEATURES),
        "training_rows": int(len(data.rows)),
        "correction_rows": int(data.correction_count),
        "accepted_picks": int(data.accepted_picks),
        "corrected_picks": int(data.corrected_picks),
        "skipped_rows": int(data.skipped_rows),
        "failed_rows": int(data.failed_rows),
        "failures": data.failures[:20],
        "candidate_mae_sec": float(mean_absolute_error(y_all, pred_all, sample_weight=w_all)),
        "target_cap_sec": float(TARGET_CAP_SEC),
        "target_min_sec": float(np.min(y_all)),
        "target_median_sec": float(np.median(y_all)),
        "target_max_sec": float(np.max(y_all)),
        "sample_weight_sum": float(np.sum(w_all)),
        "train_selection_metrics": _selection_metrics(data.rows, pred_all),
        "validation_metrics": validation_metrics,
        "feature_importances": _feature_importances(model),
        "trained_at": trained_at,
    }
    report_path = _write_json(report, report_payload)
    report_payload["training_report"] = report_path
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a whole-track learned drop-region detector from review corrections.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL path")
    parser.add_argument("--output", default="models/drop_region_detector.pkl", help="Output model pickle path")
    parser.add_argument("--report", default="models/drop_region_detector_report.json", help="Training report JSON path")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Analysis sample rate for training. Use 0 to preserve source rate")
    parser.add_argument("--max-candidates-per-track", type=int, default=260)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=0, help="Train on only the most recent N correction rows")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sample_rate = int(args.sample_rate) if int(args.sample_rate) > 0 else None
    summary = train_drop_region_detector(
        corrections=args.corrections,
        output=args.output,
        report=args.report,
        sample_rate=sample_rate,
        max_candidates_per_track=int(args.max_candidates_per_track),
        random_state=int(args.random_state),
        valid_frac=float(args.valid_frac),
        limit=int(args.limit),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
