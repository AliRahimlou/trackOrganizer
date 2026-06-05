#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.musical_clock import bpm_clock_for_time
from web_review import ReviewApp, _candidate_marker_time, _default_summary_path, _default_template_path


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _time_from_candidate(candidate: Any) -> Optional[float]:
    if isinstance(candidate, Mapping):
        return _candidate_marker_time(candidate)
    return None


def _suggested_time(payload: Mapping[str, Any]) -> Optional[float]:
    suggestion = payload.get("suggestion") if isinstance(payload.get("suggestion"), Mapping) else {}
    for value in (suggestion.get("suggested_time"), _time_from_candidate(suggestion.get("candidate"))):
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _suggested_candidate(payload: Mapping[str, Any]) -> Dict[str, Any]:
    suggestion = payload.get("suggestion") if isinstance(payload.get("suggestion"), Mapping) else {}
    candidate = suggestion.get("candidate")
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _payload_clock_zero(payload: Mapping[str, Any]) -> float:
    source_info = payload.get("source_info") if isinstance(payload.get("source_info"), Mapping) else {}
    structure = source_info.get("structure_map") if isinstance(source_info.get("structure_map"), Mapping) else {}
    beatgrid = structure.get("beatgrid") if isinstance(structure.get("beatgrid"), Mapping) else {}
    value = _float_or_none(beatgrid.get("bar_zero_sec"))
    return float(value or 0.0)


def _top_candidate_times(payload: Mapping[str, Any], limit: int = 20) -> List[float]:
    out: List[float] = []
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    for candidate in candidates[:limit]:
        t = _time_from_candidate(candidate)
        if t is not None:
            out.append(float(t))
    return out


def _final_truth_items(app: ReviewApp) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in app.items:
        review = item.get("review") if isinstance(item.get("review"), Mapping) else {}
        user_pick = _float_or_none(review.get("user_pick"))
        if user_pick is None:
            continue
        if not (review.get("reviewed") or review.get("corrected") or review.get("approved")):
            continue
        row = dict(item)
        row["truth_time"] = float(user_pick)
        row["truth_source"] = (
            "approved" if review.get("approved") else "corrected" if review.get("corrected") else "reviewed"
        )
        row["truth_reviewed_at"] = str(review.get("timestamp_reviewed") or "")
        out.append(row)
    out.sort(key=lambda row: (str(row.get("truth_reviewed_at") or ""), str(row.get("audio_path") or "")))
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _truth_class(row: Mapping[str, Any]) -> str:
    reviewed_from = str(row.get("reviewed_from") or "")
    delta = _float_or_none(row.get("delta"))
    if delta is not None and abs(delta) <= 0.001:
        return "approve"
    if reviewed_from == "web_manual_marker":
        return "manual_marker"
    if reviewed_from == "web_candidate_pick" or str(row.get("selected_by") or "") == "user_candidate_pick":
        return "candidate_pick"
    if reviewed_from.startswith("web_accept_"):
        return "accepted_marker"
    if reviewed_from == "web_review":
        return "ambiguous_web_correction"
    return reviewed_from or "unknown"


def _correction_truth_items(app: ReviewApp, correction_logs: Sequence[Path]) -> List[Dict[str, Any]]:
    item_by_track = {str(item.get("audio_path") or ""): item for item in app.items}
    latest_by_track: Dict[str, Dict[str, Any]] = {}
    order = 0
    for path in correction_logs:
        for row in _read_jsonl(path):
            order += 1
            track = str(row.get("track") or "")
            user_pick = _float_or_none(row.get("user_pick"))
            if not track or user_pick is None:
                continue
            merged = dict(row)
            merged["_truth_order"] = order
            merged["_truth_log"] = str(path)
            previous = latest_by_track.get(track)
            if previous is None or int(merged.get("_truth_order", 0)) >= int(previous.get("_truth_order", 0)):
                latest_by_track[track] = merged

    truth: List[Dict[str, Any]] = []
    for track, row in latest_by_track.items():
        item = item_by_track.get(track)
        if not item:
            continue
        out = dict(item)
        out["truth_time"] = float(row["user_pick"])
        out["truth_source"] = _truth_class(row)
        out["truth_reviewed_at"] = str(row.get("timestamp") or row.get("timestamp_logged") or "")
        out["truth_log"] = str(row.get("_truth_log") or "")
        out["truth_row"] = row
        truth.append(out)
    truth.sort(key=lambda row: (str(row.get("truth_reviewed_at") or ""), str(row.get("audio_path") or "")))
    return truth


def _classify(row: Mapping[str, Any]) -> List[str]:
    flags: List[str] = []
    predicted = _float_or_none(row.get("predicted_time"))
    expected = _float_or_none(row.get("expected_time"))
    if predicted is None:
        return ["no_prediction"]
    if expected is None:
        return ["no_truth"]
    abs_ms = abs(predicted - expected) * 1000.0
    if abs_ms <= 40.0:
        flags.append("pass_40ms")
    elif abs_ms <= 250.0:
        flags.append("near_miss_250ms")
    elif abs_ms <= 1000.0:
        flags.append("miss_under_1s")
    else:
        flags.append("miss_over_1s")
    pred_clock = row.get("predicted_clock") if isinstance(row.get("predicted_clock"), Mapping) else {}
    if pred_clock and float(pred_clock.get("one_distance_ms", 999999.0) or 999999.0) > 40.0:
        flags.append("predicted_off_one")
    if not row.get("expected_in_top_candidates"):
        flags.append("truth_not_in_top_candidates")
    return flags


def evaluate(app: ReviewApp, items: Iterable[Mapping[str, Any]], *, mode: str = "detector") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        start = time.time()
        expected = float(item["truth_time"])
        item_id = str(item["id"])
        try:
            payload = app.auto_place(item_id, mode=mode)
        except Exception as exc:
            rows.append(
                {
                    "index": index,
                    "item_id": item_id,
                    "track": str(item.get("audio_path") or ""),
                    "expected_time": expected,
                    "ok": False,
                    "error": str(exc) or exc.__class__.__name__,
                    "elapsed_sec": time.time() - start,
                    "flags": ["auto_place_exception"],
                }
            )
            continue
        predicted = _suggested_time(payload)
        candidate = _suggested_candidate(payload)
        top_times = _top_candidate_times(payload)
        bpm = _float_or_none(item.get("bpm"))
        clock_zero_sec = _payload_clock_zero(payload)
        expected_clock = bpm_clock_for_time(expected, bpm, clock_zero_sec=clock_zero_sec) or {}
        candidate_clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
        predicted_clock = dict(candidate_clock) if candidate_clock else (bpm_clock_for_time(predicted, bpm, clock_zero_sec=clock_zero_sec) if predicted is not None else {})
        closest_top_delta = min((abs(t - expected) for t in top_times), default=None)
        row = {
            "index": index,
            "item_id": item_id,
            "track": str(item.get("audio_path") or ""),
            "bpm": bpm,
            "truth_source": item.get("truth_source"),
            "expected_time": expected,
            "predicted_time": predicted,
            "delta_sec": None if predicted is None else float(predicted - expected),
            "abs_delta_ms": None if predicted is None else float(abs(predicted - expected) * 1000.0),
            "source": payload.get("source"),
            "selected_by": candidate.get("selected_by"),
            "candidate_rank": candidate.get("rank"),
            "candidate_reason": candidate.get("reason"),
            "suggestion_reason": (payload.get("suggestion") or {}).get("reason")
            if isinstance(payload.get("suggestion"), Mapping)
            else "",
            "structure_bar": candidate.get("structure_bar"),
            "structure_clock_bar": candidate.get("structure_clock_bar"),
            "expected_clock": expected_clock,
            "predicted_clock": predicted_clock or {},
            "clock_zero_sec": float(clock_zero_sec),
            "expected_in_top_candidates": closest_top_delta is not None and closest_top_delta <= 0.250,
            "closest_top_candidate_delta_ms": None if closest_top_delta is None else float(closest_top_delta * 1000.0),
            "ok": bool(payload.get("ok")),
            "error": payload.get("error"),
            "elapsed_sec": time.time() - start,
        }
        row["flags"] = _classify(row)
        rows.append(row)
    return rows


def summarize(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    pass_40 = sum(1 for row in rows if "pass_40ms" in row.get("flags", []))
    near_250 = sum(1 for row in rows if "near_miss_250ms" in row.get("flags", []))
    miss_1s = sum(1 for row in rows if "miss_over_1s" in row.get("flags", []))
    off_one = sum(1 for row in rows if "predicted_off_one" in row.get("flags", []))
    recall = sum(1 for row in rows if row.get("expected_in_top_candidates"))
    elapsed = sum(float(row.get("elapsed_sec", 0.0) or 0.0) for row in rows)
    return {
        "total": total,
        "pass_40ms": pass_40,
        "pass_40ms_rate": 0.0 if total == 0 else pass_40 / total,
        "near_miss_250ms": near_250,
        "miss_over_1s": miss_1s,
        "predicted_off_one": off_one,
        "truth_recall_top_candidates_250ms": recall,
        "truth_recall_top_candidates_250ms_rate": 0.0 if total == 0 else recall / total,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay auto-place against prior human-approved/manual markers.")
    parser.add_argument("--summary", default=str(_default_summary_path()))
    parser.add_argument("--template", default=str(_default_template_path()))
    parser.add_argument("--correction-log", default="drop_corrections.jsonl")
    parser.add_argument(
        "--truth-source",
        choices=("corrections", "state"),
        default="corrections",
        help="Use correction JSONL latest user_pick rows, or review_state user_pick rows.",
    )
    parser.add_argument(
        "--include-correction-backups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include drop_corrections.before*.jsonl before the current correction log.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--track-contains", default="")
    parser.add_argument("--output", default="models/historical_regression_results.jsonl")
    parser.add_argument(
        "--mode",
        default="detector",
        help="auto_place mode to evaluate. Default detector bypasses historical marker replay.",
    )
    parser.add_argument(
        "--use-review-memory",
        action="store_true",
        help="Allow auto-place to replay historical review memory. Off by default for fresh detector regression.",
    )
    args = parser.parse_args()

    app = ReviewApp(
        summary_csv=args.summary,
        template=args.template,
        correction_log=args.correction_log,
        auto_retrain_every=0,
        review_low_only=False,
        review_medium_and_low=False,
        regenerate_als_on_correction=False,
    )
    if not args.use_review_memory:
        app.review_memory = {}
    if args.truth_source == "state":
        truth = _final_truth_items(app)
    else:
        correction_logs: List[Path] = []
        if args.include_correction_backups:
            correction_logs.extend(sorted(Path(".").glob("drop_corrections.before*.jsonl")))
        correction_logs.append(Path(args.correction_log))
        truth = _correction_truth_items(app, correction_logs)
    if args.track_contains:
        needle = args.track_contains.lower()
        truth = [item for item in truth if needle in str(item.get("audio_path") or "").lower()]
    if args.offset:
        truth = truth[int(args.offset) :]
    if args.limit:
        truth = truth[: int(args.limit)]

    rows = evaluate(app, truth, mode=str(args.mode or "detector"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary = summarize(rows)
    summary["review_memory_enabled"] = bool(args.use_review_memory)
    summary["truth_source"] = args.truth_source
    summary["mode"] = str(args.mode or "detector")
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": summary}, indent=2, ensure_ascii=True))
    worst = sorted(
        [row for row in rows if row.get("abs_delta_ms") is not None],
        key=lambda row: float(row.get("abs_delta_ms", 0.0) or 0.0),
        reverse=True,
    )[:10]
    for row in worst:
        print(
            f"MISS {float(row.get('abs_delta_ms', 0.0) or 0.0):8.1f}ms "
            f"expected={row.get('expected_time')} predicted={row.get('predicted_time')} "
            f"flags={','.join(row.get('flags') or [])} track={row.get('track')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
