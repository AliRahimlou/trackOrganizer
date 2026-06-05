#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from drop_aligner.historical_markers import HistoricalMarker, load_historical_markers
from drop_aligner.musical_clock import bpm_clock_for_time
from drop_aligner.multistem import choose_multistem_candidate
from web_review import (
    _apply_even_bar_prior,
    _apply_structure_map_prior,
    _float_or_none,
    _infer_bpm_from_path,
    _normalize_tier,
    _read_json,
)


DEFAULT_SUMMARY = "/Users/alirahimlou/Desktop/MUSIC/STEMS/drop_batch_summary.csv"


def _fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    minutes = int(seconds // 60)
    sec = float(seconds) - (minutes * 60)
    return f"{minutes}:{sec:06.3f}"


def _load_summary_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with open(summary_csv, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _row_bpm(row: Mapping[str, str], payload: Mapping[str, Any]) -> float:
    feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    return (
        _infer_bpm_from_path(row.get("filename", ""))
        or _float_or_none(payload.get("bpm"))
        or _float_or_none(feature_summary.get("bpm"))
        or 128.0
    )


def _clock_summary(time_sec: Optional[float], bpm: Optional[float]) -> Dict[str, Any]:
    clock = bpm_clock_for_time(time_sec, bpm) if time_sec is not None else None
    if not clock:
        return {"on_one": False, "label": "--", "one_distance_ms": None}
    return {
        "on_one": bool(clock.get("on_one")),
        "label": f"b{clock.get('nearest_one_bar')} beat{clock.get('beat_in_bar')} {float(clock.get('one_distance_ms', 0.0) or 0.0):.1f}ms",
        "one_distance_ms": float(clock.get("one_distance_ms", 0.0) or 0.0),
        "nearest_one_bar": int(clock.get("nearest_one_bar", 0) or 0),
        "beat_in_bar": int(clock.get("beat_in_bar", 0) or 0),
    }


def _candidate_mode_prediction(row: Mapping[str, str]) -> tuple[Optional[float], str]:
    payload = _read_json(row.get("candidates_json", ""))
    candidate = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    for source in (candidate, payload):
        if not isinstance(source, Mapping):
            continue
        for key in ("microaligned_time", "corrected_drop_time", "user_pick", "final_ai_pick", "detected_drop_time", "timestamp", "snapped_sec"):
            value = _float_or_none(source.get(key))
            if value is not None and value > 0:
                return float(value), f"saved_payload:{key}"
    value = _float_or_none(row.get("detected_drop_time"))
    return (float(value), "summary:detected_drop_time") if value is not None else (None, "missing")


def _raw_full_auto_prediction(row: Mapping[str, str]) -> tuple[Optional[float], str]:
    track = str(row.get("filename") or "")
    payload = _read_json(row.get("candidates_json", ""))
    feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    tier = _normalize_tier(row.get("confidence_tier") or payload.get("confidence_tier") or feature_summary.get("confidence_tier"))
    base = payload.get("top_10_candidates") if isinstance(payload.get("top_10_candidates"), list) else []
    result = choose_multistem_candidate(
        track,
        saved_candidates=base,
        confidence_tier=tier,
        mode="normal",
        expanded_limit=120,
        microalign_limit=50,
        sample_rate=16000,
    )
    structure_prior = _apply_structure_map_prior(
        track,
        list(result.get("candidates", []) or []),
        result.get("suggestion", {}) if isinstance(result.get("suggestion"), Mapping) else {},
        confidence_tier=tier,
        sample_rate=16000,
    )
    structure_map = structure_prior.get("structure_map") if isinstance(structure_prior.get("structure_map"), Mapping) else {}
    beatgrid = structure_map.get("beatgrid") if isinstance(structure_map.get("beatgrid"), Mapping) else {}
    bpm = _row_bpm(row, payload)
    bar_prior = _apply_even_bar_prior(
        list(structure_prior.get("candidates", []) or []),
        structure_prior.get("suggestion", {}) if isinstance(structure_prior.get("suggestion"), Mapping) else {},
        bpm=bpm,
        confidence_tier=tier,
        bar_zero_sec=_float_or_none(beatgrid.get("bar_zero_sec")),
        allow_promotion=not bool(structure_map.get("first_drop")),
    )
    suggestion = bar_prior.get("suggestion") if isinstance(bar_prior.get("suggestion"), Mapping) else {}
    predicted = _float_or_none(suggestion.get("suggested_time"))
    reason = str(suggestion.get("reason") or "")
    return predicted, reason


def _classify(delta: Optional[float], clock: Mapping[str, Any]) -> str:
    if delta is None:
        return "no_prediction"
    abs_delta = abs(float(delta))
    if abs_delta <= 0.001:
        return "exact"
    if abs_delta <= 0.025:
        return "within_25ms"
    if not clock.get("on_one"):
        return "off_bpm_one"
    if abs_delta >= 16.0:
        return "wrong_section"
    if abs_delta >= 1.0:
        return "wrong_bar_or_phrase"
    return "micro_offset"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    correction_logs = [Path(path).expanduser() for path in args.corrections]
    index = load_historical_markers(
        correction_logs=correction_logs,
        marker_db_path=Path(args.marker_db).expanduser() if args.marker_db else None,
        min_marker_db_sec=float(args.min_marker_db_sec),
    )
    rows = _load_summary_rows(Path(args.summary).expanduser())
    matched: List[tuple[Mapping[str, str], HistoricalMarker]] = []
    for row in rows:
        payload = _read_json(row.get("candidates_json", ""))
        marker = index.find(row.get("filename", ""), bpm=_row_bpm(row, payload))
        if marker is not None and (not args.require_existing_audio or Path(str(row.get("filename", ""))).expanduser().exists()):
            matched.append((row, marker))

    if args.limit:
        matched = matched[: int(args.limit)]

    results: List[Dict[str, Any]] = []
    started = time.time()
    for idx, (row, marker) in enumerate(matched, start=1):
        payload = _read_json(row.get("candidates_json", ""))
        bpm = _row_bpm(row, payload)
        expected = float(marker.user_pick)
        elapsed = None
        if args.full_auto:
            t0 = time.time()
            predicted, source = _raw_full_auto_prediction(row)
            elapsed = time.time() - t0
        elif args.saved_payload:
            predicted, source = _candidate_mode_prediction(row)
        else:
            predicted = expected
            source = f"historical:{marker.source}"
        delta = None if predicted is None else float(predicted) - expected
        clock = _clock_summary(predicted, bpm)
        result = {
            "index": idx,
            "track": str(row.get("filename", "")),
            "expected": expected,
            "predicted": predicted,
            "delta": delta,
            "abs_delta": None if delta is None else abs(float(delta)),
            "bpm": bpm,
            "clock": clock,
            "category": _classify(delta, clock),
            "source": source,
            "historical_source": marker.source,
            "elapsed_sec": elapsed,
        }
        results.append(result)
        if not args.quiet:
            predicted_text = "--" if predicted is None else f"{predicted:.6f}"
            delta_text = "--" if delta is None else f"{delta:+.6f}"
            elapsed_text = "" if elapsed is None else f" {elapsed:.1f}s"
            print(f"{idx:02d}. {Path(str(row.get('filename', ''))).name}")
            print(f"    expected {_fmt_time(expected)} {expected:.6f} | predicted {predicted_text} | delta {delta_text}{elapsed_text}")
            print(f"    clock {clock['label']} | {result['category']} | {source[:120]}")

    abs_deltas = [float(row["abs_delta"]) for row in results if row.get("abs_delta") is not None]
    summary = {
        "summary_csv": str(Path(args.summary).expanduser()),
        "historical_marker_count": len(index),
        "matched_summary_tracks": len(matched),
        "evaluated": len(results),
        "mode": "full_auto_raw" if args.full_auto else "saved_payload" if args.saved_payload else "historical_prior",
        "elapsed_sec": time.time() - started,
        "exact_1ms": sum(1 for value in abs_deltas if value <= 0.001),
        "within_25ms": sum(1 for value in abs_deltas if value <= 0.025),
        "within_100ms": sum(1 for value in abs_deltas if value <= 0.100),
        "median_abs_delta": statistics.median(abs_deltas) if abs_deltas else None,
        "mean_abs_delta": statistics.mean(abs_deltas) if abs_deltas else None,
        "categories": {},
        "results": results,
    }
    for row in results:
        summary["categories"][row["category"]] = int(summary["categories"].get(row["category"], 0)) + 1
    if not args.quiet:
        print("\nSummary")
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2, ensure_ascii=True))
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current drop placement against historical human-approved markers.")
    parser.add_argument("summary", nargs="?", default=DEFAULT_SUMMARY, help="drop_batch_summary.csv path")
    parser.add_argument(
        "--corrections",
        action="append",
        default=["drop_corrections.jsonl", "models/multistem_training_corrections.jsonl", "drop_corrections.before_notToBeOrganized_filter_20260501_022657.jsonl"],
        help="Correction JSONL to include. Can be repeated.",
    )
    parser.add_argument("--marker-db", default="drop_marker_db.json", help="Historical marker DB JSON path")
    parser.add_argument("--min-marker-db-sec", type=float, default=1.0, help="Ignore marker-db entries before this time")
    parser.add_argument("--limit", type=int, default=20, help="Maximum matched tracks to evaluate; 0 means all")
    parser.add_argument("--require-existing-audio", action="store_true", help="Skip rows whose audio file is missing")
    parser.add_argument("--saved-payload", action="store_true", help="Replay the saved payload prediction instead of the historical prior")
    parser.add_argument("--full-auto", action="store_true", help="Run the current raw full detector. Slow.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final JSON summary")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
