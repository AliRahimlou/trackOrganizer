#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from drop_aligner.visual_candidate_selector import (
    VISUAL_SELECTOR_FEATURES,
    save_payload,
)


DEFAULT_ATLAS = "models/visual_candidate_atlas.jsonl"
DEFAULT_MODEL = "models/visual_candidate_selector.pkl"
DEFAULT_REPORT = "models/visual_candidate_selector_report.json"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                yield row


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_float(value), 0.0, 1.0))


def _stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _rank_inv(value: Any, *, cap: float = 24.0) -> float:
    rank = max(1.0, min(float(cap), _float(value, cap)))
    return float(1.0 / rank)


def _rank_norm(value: Any, *, cap: float = 24.0) -> float:
    rank = max(1.0, min(float(cap), _float(value, cap)))
    return float((rank - 1.0) / max(1.0, cap - 1.0))


def _source_is(row: Mapping[str, Any], needle: str) -> float:
    return 1.0 if needle in str(row.get("selected_by") or "") else 0.0


def _drop_strength_proxy(row: Mapping[str, Any]) -> float:
    phrase_bonus = 0.5 * _clip01((_float(row.get("visual_phrase_prior")) - 0.48) / 0.44)
    section_shift = 0.16 if bool(row.get("visual_phrase_body_shift")) else 0.0
    local_reentry = 0.10 if bool(row.get("visual_local_reentry")) else 0.0
    return _clip01(
        (0.24 * _float(row.get("visual_post8_height")))
        + (0.20 * _float(row.get("visual_post_bass8")))
        + (0.18 * _float(row.get("visual_post_drum8")))
        + (0.14 * _float(row.get("visual_post_inst8")))
        + (0.09 * _float(row.get("visual_jump8")))
        + (0.07 * _float(row.get("visual_post_drum_cont8")))
        + phrase_bonus
        + section_shift
        + local_reentry
        - (0.08 * _float(row.get("visual_pre_drum_cont4")))
    )


def _feature_rows(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    by_track: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[str(row.get("track") or "")].append(row)
    out: List[Dict[str, Any]] = []
    for track, group in by_track.items():
        valid = [row for row in group if row.get("candidate_time") not in (None, "")]
        if not valid:
            continue
        scores = sorted(_float(row.get("score")) for row in valid)
        times = sorted(_float(row.get("candidate_time")) for row in valid)
        best_score = max(scores) if scores else 0.0
        min_time = min(times) if times else 0.0
        span = max(times) - min(times) if len(times) > 1 else 1.0
        for row in valid:
            score = _clip01(row.get("score"))
            time_sec = _float(row.get("candidate_time"))
            clock_bar = int(_float(row.get("nearest_one_bar"), _float(row.get("visual_clock_bar"))))
            if not clock_bar:
                clock_bar = int(_float(row.get("visual_clock_bar")))
            time_order = 0
            for index, value in enumerate(times):
                if abs(float(value) - time_sec) <= 1e-6:
                    time_order = index
                    break
            if len(scores) <= 1:
                score_percentile = 1.0
            else:
                lower_or_equal = sum(1 for value in scores if float(value) <= score)
                score_percentile = _clip01((lower_or_equal - 1.0) / max(1.0, len(scores) - 1.0))
            prev_gap = float("inf") if time_order <= 0 else time_sec - float(times[time_order - 1])
            next_gap = float("inf") if time_order >= len(times) - 1 else float(times[time_order + 1]) - time_sec
            finite_prev_gap = 4.0 if not math.isfinite(prev_gap) else max(0.0, prev_gap)
            finite_next_gap = 4.0 if not math.isfinite(next_gap) else max(0.0, next_gap)
            features = {
                "rank_inv": _rank_inv(row.get("rank")),
                "rank_norm": _rank_norm(row.get("rank")),
                "score": score,
                "score_delta_from_best": _clip01(best_score - score),
                "score_percentile": float(score_percentile),
                "time_order_norm": 0.0 if len(times) <= 1 else _clip01(time_order / max(1.0, len(times) - 1.0)),
                "temporal_position_norm": _clip01((time_sec - min_time) / max(1e-6, span)),
                "prev_gap_norm": _clip01(finite_prev_gap / 8.0),
                "next_gap_norm": _clip01(finite_next_gap / 8.0),
                "clock_bar_norm": _clip01(clock_bar / 129.0),
                "clock_bar_early_8": 1.0 if 1 <= clock_bar <= 8 else 0.0,
                "clock_bar_early_17": 1.0 if 1 <= clock_bar <= 17 else 0.0,
                "clock_bar_17_33": 1.0 if 17 <= clock_bar <= 33 else 0.0,
                "clock_bar_33_65": 1.0 if 33 <= clock_bar <= 65 else 0.0,
                "clock_bar_late_65": 1.0 if clock_bar >= 65 else 0.0,
                "on_one": 1.0 if str(row.get("on_one")).lower() == "true" or bool(row.get("on_one") is True) else 0.0,
                "one_distance_norm": _clip01(abs(_float(row.get("one_distance_ms"), 999999.0)) / 220.0),
                "visual_phrase_prior": _clip01(row.get("visual_phrase_prior")),
                "visual_post4_height": _clip01(row.get("visual_post4_height")),
                "visual_post8_height": _clip01(row.get("visual_post8_height")),
                "visual_pre4_height": _clip01(row.get("visual_pre4_height")),
                "visual_jump4": _clip01(_float(row.get("visual_jump4")) + 0.5),
                "visual_jump8": _clip01(_float(row.get("visual_jump8")) + 0.5),
                "visual_post_bass8": _clip01(row.get("visual_post_bass8")),
                "visual_post_drum8": _clip01(row.get("visual_post_drum8")),
                "visual_post_inst8": _clip01(row.get("visual_post_inst8")),
                "visual_pre_inst4": _clip01(row.get("visual_pre_inst4")),
                "visual_pre_drum_cont4": _clip01(row.get("visual_pre_drum_cont4")),
                "visual_post_drum_cont4": _clip01(row.get("visual_post_drum_cont4")),
                "visual_post_drum_cont8": _clip01(row.get("visual_post_drum_cont8")),
                "visual_local_reentry_gap": _clip01(_float(row.get("visual_local_reentry_gap")) + 0.5),
                "visual_frame_body": _clip01(row.get("visual_frame_body")),
                "visual_frame_pre_body": _clip01(row.get("visual_frame_pre_body")),
                "visual_frame_post_body": _clip01(row.get("visual_frame_post_body")),
                "visual_frame_attack_peak": _clip01(row.get("visual_frame_attack_peak")),
                "visual_frame_low_peak": _clip01(row.get("visual_frame_low_peak")),
                "visual_frame_score": _clip01(row.get("visual_frame_score")),
                "visual_local_reentry": 1.0 if str(row.get("visual_local_reentry")).lower() == "true" or bool(row.get("visual_local_reentry") is True) else 0.0,
                "visual_phrase_body_shift": 1.0 if str(row.get("visual_phrase_body_shift")).lower() == "true" or bool(row.get("visual_phrase_body_shift") is True) else 0.0,
                "visual_opening_body_start": 1.0 if str(row.get("visual_opening_body_start")).lower() == "true" or bool(row.get("visual_opening_body_start") is True) else 0.0,
                "visual_phrase_dropout_reentry": 1.0 if str(row.get("visual_phrase_dropout_reentry")).lower() == "true" or bool(row.get("visual_phrase_dropout_reentry") is True) else 0.0,
                "visual_later_definitive_guard": 1.0 if str(row.get("visual_later_definitive_guard")).lower() == "true" or bool(row.get("visual_later_definitive_guard") is True) else 0.0,
                "visual_blank_waveform_guard": 1.0 if str(row.get("visual_blank_waveform_guard")).lower() == "true" or bool(row.get("visual_blank_waveform_guard") is True) else 0.0,
                "source_chunk": _source_is(row, "visual_gui_chunk"),
                "source_first_fat_block": _source_is(row, "visual_gui_first_fat_block"),
                "source_v2": _source_is(row, "visual_drop_v2"),
                "source_opening": _source_is(row, "visual_gui_opening_drop"),
                "source_static_guard": _source_is(row, "visual_static_marker"),
                "source_late_guard": _source_is(row, "visual_late_reset"),
                "source_fusion_guard": _source_is(row, "visual_fusion"),
                "source_track_zero_guard": _source_is(row, "visual_track_zero"),
                "source_audit_replacement": _source_is(row, "visual_audit_replacement"),
                "source_body_peak": _source_is(row, "visual_body_peak"),
                "source_body_edge": _source_is(row, "visual_body_edge"),
                "drop_strength_proxy": _drop_strength_proxy(row),
            }
            out.append(
                {
                    "track": track,
                    "row": row,
                    "label": 1 if str(row.get("candidate_is_oracle")).lower() == "true" or bool(row.get("candidate_is_oracle") is True) else 0,
                    "target_error_ms": _float(row.get("candidate_error_ms"), 999999.0),
                    "candidate_time": time_sec,
                    "vector": [float(features.get(str(name), 0.0)) for name in feature_names],
                    "features": features,
                }
            )
    return out


def _split(rows: Sequence[Mapping[str, Any]], valid_frac: float) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    if valid_frac <= 0.0:
        return list(rows), []
    valid_tracks = {str(row["track"]) for row in rows if _stable_bucket(str(row["track"])) < valid_frac}
    train = [row for row in rows if str(row["track"]) not in valid_tracks]
    valid = [row for row in rows if str(row["track"]) in valid_tracks]
    if not train or not valid:
        return list(rows), []
    return train, valid


def _make_model(random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        random_state=int(random_state),
        n_estimators=900,
        min_samples_leaf=1,
        max_features=0.85,
        class_weight="balanced",
        bootstrap=False,
        n_jobs=-1,
    )


def _predict(model: object, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float64)
    x = np.asarray([row["vector"] for row in rows], dtype=np.float64)
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, -1]
    return proba.reshape(-1)


def _selection_metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> Dict[str, Any]:
    grouped: Dict[str, List[Tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[str(row["track"])].append((row, float(score)))
    errors: List[float] = []
    oracle_errors: List[float] = []
    oracle_hits = 0
    for group in grouped.values():
        selected, _score_value = max(group, key=lambda item: (float(item[1]), -_float(item[0]["candidate_time"]) * 0.0))
        oracle = min(group, key=lambda item: float(item[0]["target_error_ms"]))[0]
        err = float(selected["target_error_ms"])
        errors.append(err)
        oracle_errors.append(float(oracle["target_error_ms"]))
        if bool(selected["label"]):
            oracle_hits += 1
    return {
        "tracks": int(len(errors)),
        "selected_oracle_count": int(oracle_hits),
        "selected_oracle_pct": 0.0 if not errors else round(100.0 * oracle_hits / len(errors), 2),
        "selected_within_25ms": int(sum(err <= 25.0 for err in errors)),
        "selected_within_50ms": int(sum(err <= 50.0 for err in errors)),
        "selected_within_250ms": int(sum(err <= 250.0 for err in errors)),
        "selected_within_50ms_pct": 0.0 if not errors else round(100.0 * sum(err <= 50.0 for err in errors) / len(errors), 2),
        "selected_wrong_section": int(sum(err >= 8000.0 for err in errors)),
        "selected_median_error_ms": float(np.median(np.asarray(errors, dtype=np.float64))) if errors else None,
        "selected_mean_error_ms": float(np.mean(np.asarray(errors, dtype=np.float64))) if errors else None,
        "oracle_within_50ms": int(sum(err <= 50.0 for err in oracle_errors)),
        "oracle_within_250ms": int(sum(err <= 250.0 for err in oracle_errors)),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    raw_rows = list(_iter_jsonl(Path(args.atlas_jsonl).expanduser()))
    feature_names = list(VISUAL_SELECTOR_FEATURES)
    rows = _feature_rows(raw_rows, feature_names)
    if not rows:
        raise ValueError("No candidate rows found")
    train_rows, valid_rows = _split(rows, float(args.valid_frac))
    model = _make_model(int(args.random_state))
    x_train = np.asarray([row["vector"] for row in train_rows], dtype=np.float64)
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    sample_weight = np.asarray(
        [
            (7.5 if int(row["label"]) else 1.0)
            * (2.0 if float(row["target_error_ms"]) <= 50.0 else 1.0)
            for row in train_rows
        ],
        dtype=np.float64,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    train_scores = _predict(model, train_rows)
    valid_scores = _predict(model, valid_rows) if valid_rows else np.asarray([], dtype=np.float64)
    all_scores = _predict(model, rows)
    y_all = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    report: Dict[str, Any] = {
        "atlas_jsonl": str(Path(args.atlas_jsonl).expanduser()),
        "candidate_rows": int(len(rows)),
        "tracks": int(len({str(row["track"]) for row in rows})),
        "feature_names": feature_names,
        "train_rows": int(len(train_rows)),
        "train_tracks": int(len({str(row["track"]) for row in train_rows})),
        "valid_rows": int(len(valid_rows)),
        "valid_tracks": int(len({str(row["track"]) for row in valid_rows})),
        "train_selection": _selection_metrics(train_rows, train_scores),
        "valid_selection": _selection_metrics(valid_rows, valid_scores) if valid_rows else {},
        "all_selection": _selection_metrics(rows, all_scores),
    }
    if len(set(y_all.tolist())) > 1:
        clipped = np.clip(all_scores, 1e-6, 1.0 - 1e-6)
        report["all_classifier"] = {
            "roc_auc": float(roc_auc_score(y_all, all_scores)),
            "average_precision": float(average_precision_score(y_all, all_scores)),
            "log_loss": float(log_loss(y_all, clipped, labels=[0, 1])),
        }
    payload = {
        "model": model,
        "feature_names": feature_names,
        "training_rows": int(len(rows)),
        "training_tracks": int(report["tracks"]),
        "report": report,
    }
    saved = save_payload(payload, args.output)
    report["model_path"] = saved
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a visual-only candidate selector from the visual candidate atlas.")
    parser.add_argument("--atlas-jsonl", default=DEFAULT_ATLAS)
    parser.add_argument("--output", default=DEFAULT_MODEL)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
