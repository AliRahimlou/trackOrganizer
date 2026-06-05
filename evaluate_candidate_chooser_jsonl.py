#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.candidate_chooser import (
    candidate_effective_time,
    choose_learned_candidate,
    load_candidate_chooser_payload,
)


DEFAULT_THRESHOLDS_MS = (5, 10, 25, 50, 100, 250)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
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
                yield line_no, row


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or row.get("filename") or "")


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    time_sec = candidate_effective_time(candidate)
    if time_sec is not None:
        return f"time:{float(time_sec):.9f}"
    return "body:" + json.dumps(dict(candidate), sort_keys=True, default=str)[:500]


def _candidate_boundary_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec"):
        value = _float_or_none(candidate.get(key))
        if value is not None and value > 0.0:
            return float(value)
    return candidate_effective_time(candidate)


def _candidate_rows(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    source: List[Any] = []
    for key in (
        "candidates",
        "top_10_candidates",
        "post_structure_candidates",
        "merged_candidates",
        "candidate_debug",
    ):
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = value.get("candidates")
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                source.extend(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            source.extend(value)

    for key in ("selected_candidate", "closest_candidate_to_user_pick"):
        value = row.get(key)
        if isinstance(value, Mapping):
            source.append(value)

    out: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in source:
        if not isinstance(candidate, Mapping):
            continue
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _candidate_rank(candidate: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not candidate:
        return None
    for key in ("handcrafted_rank", "rank", "model_rank"):
        try:
            value = int(candidate.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _prediction_score(selection: Mapping[str, Any]) -> Optional[float]:
    for key in ("selection_probability", "chooser_score"):
        value = _float_or_none(selection.get(key))
        if value is not None:
            return float(value)
    return None


def _closest_candidate(candidates: Sequence[Mapping[str, Any]], user_pick: float) -> Tuple[Optional[Mapping[str, Any]], Optional[float]]:
    best: Optional[Mapping[str, Any]] = None
    best_error: Optional[float] = None
    for candidate in candidates:
        time_sec = candidate_effective_time(candidate)
        if time_sec is None:
            continue
        error = abs(float(time_sec) - float(user_pick))
        if best_error is None or error < best_error:
            best = candidate
            best_error = float(error)
    return best, best_error


def _percent(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return 100.0 * float(count) / float(total)


def _error_summary(errors: Sequence[float], tolerance_sec: float) -> Dict[str, Any]:
    if not errors:
        return {
            "count": 0,
            "within_tolerance": 0,
            "within_tolerance_percent": None,
            "mean_abs_error_sec": None,
            "median_abs_error_sec": None,
            "max_abs_error_sec": None,
            "percent_within": {f"{threshold}ms": None for threshold in DEFAULT_THRESHOLDS_MS},
        }
    values = sorted(float(error) for error in errors)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return {
        "count": int(len(values)),
        "within_tolerance": int(sum(error <= float(tolerance_sec) for error in values)),
        "within_tolerance_percent": _percent(int(sum(error <= float(tolerance_sec) for error in values)), int(len(values))),
        "mean_abs_error_sec": float(sum(values) / len(values)),
        "median_abs_error_sec": float(median),
        "max_abs_error_sec": float(max(values)),
        "percent_within": {
            f"{threshold}ms": _percent(
                int(sum(error <= (float(threshold) / 1000.0) for error in values)),
                int(len(values)),
            )
            for threshold in DEFAULT_THRESHOLDS_MS
        },
    }


def _evaluate_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    model_path: str,
    tolerance_sec: float,
) -> Optional[Dict[str, Any]]:
    user_pick = _float_or_none(row.get("user_pick"))
    if user_pick is None:
        return None
    candidates = _candidate_rows(row)
    if not candidates:
        return None

    selected = choose_learned_candidate(candidates, model_path=model_path)
    if not selected:
        return None

    selected_candidate = selected.get("candidate") if isinstance(selected.get("candidate"), Mapping) else None
    model_time = _float_or_none(selected.get("time_sec"))
    selected_time = _candidate_boundary_time(selected_candidate) if selected_candidate is not None else model_time
    if selected_time is None:
        return None

    closest, closest_error = _closest_candidate(candidates, float(user_pick))
    closest_time = None if closest is None else candidate_effective_time(closest)
    selected_error = abs(float(selected_time) - float(user_pick))
    candidate_count = sum(1 for candidate in candidates if candidate_effective_time(candidate) is not None)
    return {
        "line_no": int(line_no),
        "track": _track(row),
        "user_pick": float(user_pick),
        "selected_time": float(selected_time),
        "selected_model_time": model_time,
        "selected_abs_error_sec": float(selected_error),
        "hit": bool(selected_error <= float(tolerance_sec)),
        "selected_rank": _candidate_rank(selected_candidate),
        "selected_score": _prediction_score(selected),
        "selected_confidence": _float_or_none(selected.get("selection_confidence")),
        "selected_margin": _float_or_none(selected.get("prediction_margin_sec")),
        "closest_time": closest_time,
        "closest_abs_error_sec": closest_error,
        "closest_rank": _candidate_rank(closest),
        "selected_is_closest": bool(closest_time is not None and abs(float(selected_time) - float(closest_time)) <= 1e-9),
        "candidate_count": int(candidate_count),
        "model_type": str(selected.get("model_type") or ""),
    }


def evaluate_candidate_chooser_jsonl(
    *,
    model: str,
    corrections: str,
    tolerance_ms: float = 25.0,
    track_contains: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model_path = str(Path(model).expanduser())
    corrections_path = Path(corrections).expanduser()
    payload = load_candidate_chooser_payload(model_path)
    if payload is None:
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not corrections_path.exists():
        raise FileNotFoundError(f"Corrections JSONL not found: {corrections_path}")

    tolerance_sec = float(tolerance_ms) / 1000.0
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {
        "track_filter": 0,
        "missing_user_pick": 0,
        "no_candidates": 0,
        "no_selection": 0,
    }
    seen_rows = 0

    for line_no, row in _iter_jsonl(corrections_path):
        seen_rows += 1
        track = _track(row)
        if track_contains and track_contains.lower() not in track.lower():
            skipped["track_filter"] += 1
            continue
        if _float_or_none(row.get("user_pick")) is None:
            skipped["missing_user_pick"] += 1
            continue
        if not _candidate_rows(row):
            skipped["no_candidates"] += 1
            continue
        evaluated = _evaluate_row(row, line_no=line_no, model_path=model_path, tolerance_sec=tolerance_sec)
        if evaluated is None:
            skipped["no_selection"] += 1
            continue
        rows.append(evaluated)

    errors = [float(row["selected_abs_error_sec"]) for row in rows]
    oracle_errors = [
        float(row["closest_abs_error_sec"])
        for row in rows
        if row.get("closest_abs_error_sec") is not None
    ]
    misses = sorted(
        [row for row in rows if not bool(row.get("hit"))],
        key=lambda row: float(row["selected_abs_error_sec"]),
        reverse=True,
    )
    summary = {
        "model": model_path,
        "corrections": str(corrections_path),
        "track_contains": track_contains,
        "tolerance_ms": float(tolerance_ms),
        "input_rows": int(seen_rows),
        "evaluated_rows": int(len(rows)),
        "skipped": skipped,
        "model_metadata": {
            "model_type": str(payload.get("model_type") or ""),
            "training_rows": int(payload.get("training_rows", 0) or 0),
            "correction_rows": int(payload.get("correction_rows", 0) or 0),
            "selector_correction_rows": int(payload.get("selector_correction_rows", 0) or 0),
            "feature_count": int(len(payload.get("feature_names") or [])),
        },
        "selected": _error_summary(errors, tolerance_sec),
        "oracle_closest_candidate": _error_summary(oracle_errors, tolerance_sec),
        "selected_closest_candidate_count": int(sum(1 for row in rows if row.get("selected_is_closest"))),
        "selected_closest_candidate_percent": _percent(int(sum(1 for row in rows if row.get("selected_is_closest"))), int(len(rows))),
        "miss_count": int(len(misses)),
        "worst_misses": [
            {
                "line_no": row["line_no"],
                "track": row["track"],
                "user_pick": row["user_pick"],
                "selected_time": row["selected_time"],
                "selected_abs_error_sec": row["selected_abs_error_sec"],
                "selected_rank": row["selected_rank"],
                "selected_score": row["selected_score"],
                "closest_time": row["closest_time"],
                "closest_abs_error_sec": row["closest_abs_error_sec"],
                "closest_rank": row["closest_rank"],
                "candidate_count": row["candidate_count"],
            }
            for row in misses[:20]
        ],
    }
    return summary, misses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a learned drop candidate chooser pickle against post-structure correction JSONL rows."
    )
    parser.add_argument("--model", required=True, help="Candidate chooser model pickle path")
    parser.add_argument("--corrections", required=True, help="Correction JSONL path with merged candidate rows")
    parser.add_argument("--tolerance-ms", type=float, default=25.0, help="Hit tolerance in milliseconds")
    parser.add_argument("--track-contains", default=None, help="Only evaluate tracks whose path contains this substring")
    parser.add_argument("--show-misses", action="store_true", help="Print miss rows as JSONL to stderr")
    parser.add_argument("--miss-limit", type=int, default=20, help="Maximum miss lines to print with --show-misses")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary, misses = evaluate_candidate_chooser_jsonl(
        model=args.model,
        corrections=args.corrections,
        tolerance_ms=float(args.tolerance_ms),
        track_contains=args.track_contains,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.show_misses:
        for row in misses[: max(0, int(args.miss_limit))]:
            print(json.dumps(row, sort_keys=True, ensure_ascii=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
