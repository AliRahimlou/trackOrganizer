#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from drop_aligner.auto_verifier import AUTO_VERIFIER_FEATURES, load_auto_verifier_payload
from train_auto_verifier import _matrix, _positive_probability, _selected_validation_rows, _split_rows, load_verifier_rows


DEFAULT_CORRECTIONS = Path("models/multistem_training_corrections.jsonl")
DEFAULT_MODEL = Path("models/drop_auto_verifier.pkl")
DEFAULT_OUTPUT = Path("models/auto_gate_config.json")


def _feature(row_vector: Sequence[float], name: str) -> float:
    try:
        index = list(AUTO_VERIFIER_FEATURES).index(name)
    except ValueError:
        return 0.0
    if index >= len(row_vector):
        return 0.0
    return float(row_vector[index])


def _selected_calibration_rows(corrections: str, valid_frac: float) -> List[Any]:
    rows = load_verifier_rows(corrections)
    _train, valid, _split = _split_rows(rows, valid_frac)
    return _selected_validation_rows(valid if valid else rows)


def _verifier_selected_rows(
    rows: Sequence[Any],
    probabilities: np.ndarray,
    predicted_errors: np.ndarray,
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    grouped: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row.correction_id), []).append(int(index))
    selected_indices: List[int] = []
    for indices in grouped.values():
        chosen = max(
            indices,
            key=lambda idx: (
                float(probabilities[idx]),
                -float(predicted_errors[idx]),
                -int(rows[idx].candidate_index),
            ),
        )
        selected_indices.append(int(chosen))
    selected_indices.sort(key=lambda idx: int(rows[idx].correction_id))
    return (
        [rows[idx] for idx in selected_indices],
        np.asarray([float(probabilities[idx]) for idx in selected_indices], dtype=np.float64),
        np.asarray([float(predicted_errors[idx]) for idx in selected_indices], dtype=np.float64),
    )


def _evaluate_gate(
    rows: Sequence[Any],
    probabilities: np.ndarray,
    predicted_errors: np.ndarray,
    *,
    min_probability: float,
    max_predicted_error_ms: float,
    min_micro_confidence: float,
    max_snap_offset_ms: float,
    max_fake_hit_penalty: float,
    min_candidate_margin: float,
    error_limit_ms: float,
) -> Dict[str, Any]:
    actual_errors = np.asarray([float(row.abs_error_sec) for row in rows], dtype=np.float64)
    micro = np.asarray([_feature(row.vector, "micro_confidence") for row in rows], dtype=np.float64)
    snap_ms = np.asarray([_feature(row.vector, "snap_offset_ms_abs_norm") * 220.0 for row in rows], dtype=np.float64)
    fake = np.asarray([_feature(row.vector, "fake_hit_penalty") for row in rows], dtype=np.float64)
    margin = np.asarray([_feature(row.vector, "candidate_margin") for row in rows], dtype=np.float64)
    mask = (
        (probabilities >= float(min_probability))
        & ((predicted_errors * 1000.0) <= float(max_predicted_error_ms))
        & (micro >= float(min_micro_confidence))
        & (snap_ms <= float(max_snap_offset_ms))
        & (fake <= float(max_fake_hit_penalty))
        & (margin >= float(min_candidate_margin))
    )
    accepted_errors = actual_errors[mask]
    accepted = int(np.sum(mask))
    false_count = int(np.sum(accepted_errors > (float(error_limit_ms) / 1000.0))) if accepted else 0
    return {
        "accepted": accepted,
        "coverage_percent": None if len(rows) == 0 else 100.0 * float(accepted) / float(len(rows)),
        "false_auto_save_count": false_count,
        "false_auto_save_rate_percent": None if accepted == 0 else 100.0 * float(false_count) / float(accepted),
        "max_actual_error_ms": None if accepted == 0 else float(np.max(accepted_errors) * 1000.0),
        "median_actual_error_ms": None if accepted == 0 else float(np.median(accepted_errors) * 1000.0),
        "within_error_limit_percent": None if accepted == 0 else 100.0 * float(np.sum(accepted_errors <= (float(error_limit_ms) / 1000.0))) / float(accepted),
        "thresholds": {
            "min_p_within_25ms": float(min_probability),
            "max_predicted_error_ms": float(max_predicted_error_ms),
            "min_micro_confidence": float(min_micro_confidence),
            "max_snap_offset_ms": float(max_snap_offset_ms),
            "max_fake_hit_penalty": float(max_fake_hit_penalty),
            "min_candidate_margin": float(min_candidate_margin),
        },
    }


def _search(
    rows: Sequence[Any],
    probabilities: np.ndarray,
    predicted_errors: np.ndarray,
    *,
    error_limit_ms: float,
    max_false_rate_percent: float,
    require_zero_false: bool,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for min_probability in np.arange(0.995, 0.799, -0.005):
        for max_predicted_error_ms in (5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0):
            for min_micro_confidence in (0.98, 0.96, 0.93, 0.90, 0.86, 0.80, 0.0):
                for max_snap_offset_ms in (15.0, 25.0, 45.0, 70.0, 90.0, 140.0, 220.0):
                    for max_fake_hit_penalty in (0.10, 0.20, 0.35, 0.60, 1.0):
                        for min_candidate_margin in (0.30, 0.20, 0.10, 0.05, 0.0):
                            metrics = _evaluate_gate(
                                rows,
                                probabilities,
                                predicted_errors,
                                min_probability=float(min_probability),
                                max_predicted_error_ms=float(max_predicted_error_ms),
                                min_micro_confidence=float(min_micro_confidence),
                                max_snap_offset_ms=float(max_snap_offset_ms),
                                max_fake_hit_penalty=float(max_fake_hit_penalty),
                                min_candidate_margin=float(min_candidate_margin),
                                error_limit_ms=float(error_limit_ms),
                            )
                            accepted = int(metrics.get("accepted", 0) or 0)
                            if accepted <= 0:
                                continue
                            false_count = int(metrics.get("false_auto_save_count", 0) or 0)
                            false_rate_value = metrics.get("false_auto_save_rate_percent")
                            false_rate = 999.0 if false_rate_value is None else float(false_rate_value)
                            if require_zero_false and false_count > 0:
                                continue
                            if false_rate > float(max_false_rate_percent):
                                continue
                            if best is None:
                                best = metrics
                                continue
                            if accepted > int(best.get("accepted", 0) or 0):
                                best = metrics
                                continue
                            if accepted == int(best.get("accepted", 0) or 0):
                                current_max = float(metrics.get("max_actual_error_ms") or 999999.0)
                                best_max = float(best.get("max_actual_error_ms") or 999999.0)
                                if current_max < best_max:
                                    best = metrics
    if best is None:
        return {
            "accepted": 0,
            "coverage_percent": 0.0,
            "false_auto_save_count": 0,
            "false_auto_save_rate_percent": None,
            "max_actual_error_ms": None,
            "thresholds": {
                "min_p_within_25ms": 0.995,
                "max_predicted_error_ms": 5.0,
                "min_micro_confidence": 0.98,
                "max_snap_offset_ms": 15.0,
                "max_fake_hit_penalty": 0.10,
                "min_candidate_margin": 0.30,
            },
            "note": "no threshold accepted calibration rows under the requested safety target",
        }
    return best


def tune_auto_gate(
    *,
    corrections: str,
    model: str,
    output: str,
    valid_frac: float = 0.20,
) -> Dict[str, Any]:
    payload = load_auto_verifier_payload(model)
    if payload is None:
        raise FileNotFoundError(f"Auto verifier model not found: {model}")
    all_rows = load_verifier_rows(corrections)
    _train_rows, valid_rows, _split = _split_rows(all_rows, valid_frac)
    calibration_rows = valid_rows if valid_rows else all_rows
    x, _y, _target, _w = _matrix(calibration_rows)
    probabilities = _positive_probability(payload["classifier"], x)
    raw_predicted_errors = np.maximum(0.0, np.asarray(payload["regressor"].predict(x), dtype=np.float64))
    predicted_errors = np.minimum(raw_predicted_errors, np.maximum(0.0, 1.0 - probabilities) * 0.100)
    selected_rows, selected_probabilities, selected_predicted_errors = _verifier_selected_rows(
        calibration_rows,
        probabilities,
        predicted_errors,
    )
    config = {
        "model_path": str(Path(model).expanduser()),
        "corrections": str(Path(corrections).expanduser()),
        "calibration_selected_tracks": int(len(selected_rows)),
        "selection_strategy": "max_auto_verifier_probability_per_track",
        "safe": _search(
            selected_rows,
            selected_probabilities,
            selected_predicted_errors,
            error_limit_ms=25.0,
            max_false_rate_percent=0.0,
            require_zero_false=True,
        ),
        "balanced": _search(
            selected_rows,
            selected_probabilities,
            selected_predicted_errors,
            error_limit_ms=25.0,
            max_false_rate_percent=1.0,
            require_zero_false=False,
        ),
        "aggressive": _search(
            selected_rows,
            selected_probabilities,
            selected_predicted_errors,
            error_limit_ms=50.0,
            max_false_rate_percent=5.0,
            require_zero_false=False,
        ),
    }
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    config["output"] = str(out_path)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune calibrated auto-save gate thresholds on held-out tracks.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS), help="Expanded correction JSONL")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Auto verifier model")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output auto gate config JSON")
    parser.add_argument("--valid-frac", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = tune_auto_gate(
        corrections=str(args.corrections),
        model=str(args.model),
        output=str(args.output),
        valid_frac=float(args.valid_frac),
    )
    print(json.dumps(config, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
