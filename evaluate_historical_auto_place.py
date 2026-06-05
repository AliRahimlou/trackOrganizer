#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _reexec_with_local_venv() -> None:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_with_local_venv()

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.multistem import choose_multistem_candidate
from drop_aligner.musical_clock import bpm_clock_for_time, feature_grid_for_time
from web_review import (
    _apply_even_bar_prior,
    _apply_structure_map_prior,
    _candidate_marker_time,
    _float_or_none,
    _infer_bpm_from_path,
    _normalize_tier,
    _read_json,
    _stable_id,
)


DEFAULT_CORRECTIONS = Path("drop_corrections.jsonl")
DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_REVIEW_STATE = DEFAULT_SUMMARY.parent / "review_state.json"
DEFAULT_JSON = Path("models/historical_auto_place_regression.json")
DEFAULT_CSV = Path("models/historical_auto_place_regression.csv")
DEFAULT_ROWS_JSONL = Path("models/historical_auto_place_regression_rows.jsonl")
THRESHOLDS_MS = (5, 10, 25, 40, 50, 100, 250, 1000)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_line_no"] = line_no
                row["_truth_source"] = str(path)
                yield row


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or row.get("filename") or row.get("audio_path") or "")


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    return _float_or_none(row.get("user_pick"))


def _read_summary(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {str(row.get("filename", "")): row for row in csv.DictReader(fh) if row.get("filename")}


def _read_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _review_state_rows(review_state: Path, summary_path: Path) -> List[Dict[str, Any]]:
    if not review_state.exists():
        return []
    try:
        with open(review_state, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return []
    items = state.get("items") if isinstance(state, Mapping) else {}
    if not isinstance(items, Mapping):
        return []
    row_by_id = {_stable_id(row): row for row in _read_summary_rows(summary_path)}
    out: List[Dict[str, Any]] = []
    for item_index, (item_id, review) in enumerate(items.items(), start=1):
        if not isinstance(review, Mapping) or not bool(review.get("reviewed")):
            continue
        user_pick = _float_or_none(review.get("user_pick"))
        if user_pick is None:
            continue
        summary_row = row_by_id.get(str(item_id))
        if not summary_row:
            continue
        track = str(summary_row.get("filename") or "")
        if not track:
            continue
        out.append(
            {
                "_line_no": item_index,
                "_truth_source": str(review_state),
                "track": track,
                "user_pick": float(user_pick),
                "reviewed_from": "review_state",
                "selected_by": summary_row.get("selected_by", ""),
                "confidence_tier": summary_row.get("confidence_tier", ""),
                "ai_pick": _float_or_none(summary_row.get("detected_drop_time")),
                "timestamp": review.get("timestamp_reviewed", ""),
            }
        )
    return out


def _candidate_rows(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = row.get("top_10_candidates")
    if not candidates:
        candidates = row.get("candidates")
    if not candidates:
        selected = row.get("selected_candidate")
        candidates = [selected] if isinstance(selected, Mapping) else []
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _summary_payload(summary_row: Mapping[str, str]) -> Dict[str, Any]:
    path = str(summary_row.get("candidates_json") or "")
    if not path:
        return {}
    return _read_json(path)


def _summary_candidates(summary_row: Mapping[str, str], fallback_row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    payload = _summary_payload(summary_row)
    candidates = payload.get("top_10_candidates") if isinstance(payload.get("top_10_candidates"), list) else []
    if not candidates:
        candidates = _candidate_rows(fallback_row)
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else None
    closest = payload.get("closest_candidate_to_user_pick") if isinstance(payload.get("closest_candidate_to_user_pick"), Mapping) else None
    out = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    if closest:
        row = dict(closest)
        row.setdefault("source", "saved_closest_to_review_pick")
        row.setdefault("selected_by", "saved_closest_to_review_pick")
        out.append(row)
    if selected:
        out.append(selected)
    return out


def _summary_tier(summary_row: Mapping[str, str], fallback_row: Mapping[str, Any]) -> str:
    payload = _summary_payload(summary_row)
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    return _normalize_tier(
        summary_row.get("confidence_tier")
        or payload.get("confidence_tier")
        or features.get("confidence_tier")
        or fallback_row.get("confidence_tier")
    )


def _summary_bpm(track: str, summary_row: Mapping[str, str]) -> Optional[float]:
    payload = _summary_payload(summary_row)
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    return _infer_bpm_from_path(track) or _float_or_none(payload.get("bpm")) or _float_or_none(features.get("bpm"))


def _latest_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        track = _track(row)
        if track:
            out[track] = row
    return list(out.values())


def _dedupe(candidates: Sequence[Mapping[str, Any]], radius_sec: float = 0.010) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    times: List[float] = []
    for candidate in candidates:
        row = dict(candidate)
        t = _candidate_marker_time(row)
        if t is not None and any(abs(float(t) - existing) <= float(radius_sec) for existing in times):
            continue
        out.append(row)
        if t is not None:
            times.append(float(t))
    return out


def _closest_candidate(candidates: Sequence[Mapping[str, Any]], target_time: float) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_error: Optional[float] = None
    for index, candidate in enumerate(candidates, start=1):
        marker = _candidate_marker_time(candidate)
        if marker is None:
            continue
        error = abs(float(marker) - float(target_time))
        if best_error is None or error < best_error:
            best_error = float(error)
            best = {
                "rank": candidate.get("rank") or candidate.get("handcrafted_rank") or index,
                "time": float(marker),
                "error_ms": float(error * 1000.0),
                "selected_by": candidate.get("selected_by", ""),
            }
    return best


def _run_detector(
    track: str,
    *,
    saved_candidates: Sequence[Mapping[str, Any]],
    bpm: Optional[float],
    confidence_tier: str,
    mode: str,
    expanded_limit: int,
    microalign_limit: int,
    sample_rate: int,
) -> Dict[str, Any]:
    result = choose_multistem_candidate(
        track,
        saved_candidates=saved_candidates,
        confidence_tier=confidence_tier,
        mode=mode,
        expanded_limit=int(expanded_limit),
        microalign_limit=int(microalign_limit),
        sample_rate=int(sample_rate),
    )
    candidate_pool = list(result.get("candidates", []) or [])
    candidate_pool.extend(dict(candidate) for candidate in saved_candidates if isinstance(candidate, Mapping))
    structure_prior = _apply_structure_map_prior(
        track,
        candidate_pool,
        result.get("suggestion", {}) if isinstance(result.get("suggestion"), Mapping) else {},
        confidence_tier=confidence_tier,
        sample_rate=int(sample_rate),
    )
    structure_map = structure_prior.get("structure_map") if isinstance(structure_prior.get("structure_map"), Mapping) else {}
    beatgrid = structure_map.get("beatgrid") if isinstance(structure_map.get("beatgrid"), Mapping) else {}
    bar_prior = _apply_even_bar_prior(
        list(structure_prior.get("candidates", []) or []),
        structure_prior.get("suggestion", {}) if isinstance(structure_prior.get("suggestion"), Mapping) else {},
        bpm=bpm or _float_or_none(result.get("bpm")),
        confidence_tier=confidence_tier,
        bar_zero_sec=_float_or_none(beatgrid.get("bar_zero_sec")),
        allow_promotion=False,
    )
    suggestion = bar_prior.get("suggestion", {}) if isinstance(bar_prior.get("suggestion"), Mapping) else {}
    candidate = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else {}
    predicted = _float_or_none(suggestion.get("suggested_time"))
    if predicted is None:
        predicted = _candidate_marker_time(candidate)
    return {
        "predicted": predicted,
        "suggestion": suggestion,
        "candidate": candidate,
        "candidates": _dedupe(bar_prior.get("candidates", []) if isinstance(bar_prior.get("candidates"), Sequence) else []),
        "structure_map": structure_map,
        "source_count": result.get("source_count"),
        "candidate_count": result.get("candidate_count"),
    }


def _clock_fields(prefix: str, time_sec: Optional[float], bpm: Optional[float], grid_zero: Optional[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    clock = bpm_clock_for_time(time_sec, bpm, clock_zero_sec=float(grid_zero or 0.0))
    grid = feature_grid_for_time(time_sec, bpm, bar_zero_sec=grid_zero)
    if clock:
        out[f"{prefix}_clock_bar"] = clock.get("nearest_one_bar")
        out[f"{prefix}_beat_in_bar"] = clock.get("beat_in_bar")
        out[f"{prefix}_one_distance_ms"] = round(float(clock.get("one_distance_ms", 0.0) or 0.0), 3)
        out[f"{prefix}_phrase"] = clock.get("phrase")
    if grid:
        out[f"{prefix}_grid_bar"] = grid.get("nearest_grid_bar")
        out[f"{prefix}_grid_distance_beats"] = round(float(grid.get("grid_distance_beats", 0.0) or 0.0), 4)
    return out


def evaluate_historical_auto_place(
    *,
    corrections: str,
    batch_summary: str,
    review_state: str = "",
    output_json: str,
    output_csv: str,
    rows_jsonl: str = "",
    limit: int = 0,
    offset: int = 0,
    latest_per_track: bool = True,
    shard_count: int = 1,
    shard_index: int = 0,
    mode: str = "normal",
    expanded_limit: int = 120,
    microalign_limit: int = 50,
    sample_rate: int = 16000,
    target_ms: float = 25.0,
    progress_every: int = 1,
    resume: bool = False,
    track_contains: str = "",
) -> Dict[str, Any]:
    corrections_path = Path(corrections).expanduser()
    summary_path = Path(batch_summary).expanduser()
    review_state_path = Path(review_state).expanduser() if review_state else None
    rows_jsonl_path = Path(rows_jsonl).expanduser() if rows_jsonl else None
    summary = _read_summary(summary_path)
    rows = [row for row in _iter_jsonl(corrections_path) if not row_has_excluded_path(row)]
    if review_state_path:
        rows.extend(_review_state_rows(review_state_path, summary_path))
    rows = [row for row in rows if _track(row) and _user_pick(row) is not None]
    if latest_per_track:
        rows = _latest_rows(rows)
    track_filter = str(track_contains or "").strip().lower()
    if track_filter:
        rows = [row for row in rows if track_filter in _track(row).lower()]
    rows.sort(key=lambda row: _track(row).lower())
    total_truth_rows = len(rows)
    shard_count = max(1, int(shard_count))
    shard_index = max(0, min(shard_count - 1, int(shard_index)))
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if offset > 0:
        rows = rows[int(offset) :]
    if limit > 0:
        rows = rows[: int(limit)]
    if rows_jsonl_path:
        rows_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            completed = {str(row.get("track") or "") for row in _iter_jsonl(rows_jsonl_path)}
            rows = [row for row in rows if _track(row) not in completed]
        elif rows_jsonl_path.exists():
            rows_jsonl_path.unlink()

    out_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    errors_ms: List[float] = []
    started = time.time()

    for index, row in enumerate(rows, start=1):
        track = _track(row)
        user_pick = _user_pick(row)
        assert user_pick is not None
        summary_row = summary.get(track, {})
        bpm = _summary_bpm(track, summary_row)
        tier = _summary_tier(summary_row, row)
        saved_candidates = _summary_candidates(summary_row, row)
        base: Dict[str, Any] = {
            "index": index,
            "line_no": row.get("_line_no"),
            "track": track,
            "user_pick": float(user_pick),
            "truth_source": row.get("_truth_source", str(corrections_path)),
            "reviewed_from": row.get("reviewed_from", ""),
            "historical_selected_by": row.get("selected_by", ""),
            "confidence_tier": tier,
            "bpm": "" if bpm is None else float(bpm),
            "status": "ok",
        }
        if not Path(track).expanduser().exists():
            counts["missing_file"] += 1
            base.update({"status": "missing_file", "error": "audio file does not exist"})
            out_rows.append(base)
            continue
        try:
            result = _run_detector(
                track,
                saved_candidates=saved_candidates,
                bpm=bpm,
                confidence_tier=tier,
                mode=mode,
                expanded_limit=expanded_limit,
                microalign_limit=microalign_limit,
                sample_rate=sample_rate,
            )
            predicted = _float_or_none(result.get("predicted"))
            structure_map = result.get("structure_map") if isinstance(result.get("structure_map"), Mapping) else {}
            beatgrid = structure_map.get("beatgrid") if isinstance(structure_map.get("beatgrid"), Mapping) else {}
            grid_zero = _float_or_none(beatgrid.get("bar_zero_sec"))
            candidate = result.get("candidate") if isinstance(result.get("candidate"), Mapping) else {}
            suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), Mapping) else {}
            candidate_pool = result.get("candidates") if isinstance(result.get("candidates"), Sequence) else []
            closest = _closest_candidate([candidate for candidate in candidate_pool if isinstance(candidate, Mapping)], float(user_pick))
            if predicted is None:
                counts["no_prediction"] += 1
                base.update({"status": "no_prediction", "error": suggestion.get("reason", "")})
                out_rows.append(base)
                continue
            error_ms = abs(float(predicted) - float(user_pick)) * 1000.0
            errors_ms.append(float(error_ms))
            status = "miss"
            if error_ms <= float(target_ms):
                status = f"exact_{int(round(float(target_ms)))}ms"
            elif error_ms <= 100.0:
                status = "close_100ms"
            elif _float_or_none(closest.get("error_ms")) is not None and float(closest.get("error_ms")) <= float(target_ms):
                status = "selection_miss"
            elif _float_or_none(closest.get("error_ms")) is not None and float(closest.get("error_ms")) <= 100.0:
                status = "near_candidate_selection_miss"
            elif not closest:
                status = "candidate_generation_miss"
            counts[status] += 1
            clock = bpm_clock_for_time(predicted, bpm)
            if clock and not bool(clock.get("on_one")):
                counts["predicted_off_one"] += 1
            base.update(
                {
                    "status": status,
                    "predicted": float(predicted),
                    "error_ms": round(float(error_ms), 3),
                    "selected_by": candidate.get("selected_by", suggestion.get("mode", "")),
                    "reason": suggestion.get("reason", candidate.get("reason", "")),
                    "structure_first_drop": _candidate_marker_time(structure_map.get("first_drop", {}))
                    if isinstance(structure_map.get("first_drop"), Mapping)
                    else "",
                    "structure_second_drop": _candidate_marker_time(structure_map.get("second_drop", {}))
                    if isinstance(structure_map.get("second_drop"), Mapping)
                    else "",
                    "bar_count": structure_map.get("bar_count", ""),
                    "feature_grid_zero": "" if grid_zero is None else round(float(grid_zero), 6),
                    "source_count": result.get("source_count", ""),
                    "candidate_count": result.get("candidate_count", ""),
                    "closest_candidate_rank": closest.get("rank", ""),
                    "closest_candidate_time": closest.get("time", ""),
                    "closest_candidate_error_ms": ""
                    if _float_or_none(closest.get("error_ms")) is None
                    else round(float(closest.get("error_ms")), 3),
                }
            )
            base.update(_clock_fields("predicted", predicted, bpm, grid_zero))
            base.update(_clock_fields("user", user_pick, bpm, grid_zero))
            out_rows.append(base)
        except Exception as exc:
            counts["detector_error"] += 1
            base.update({"status": "detector_error", "error": str(exc) or exc.__class__.__name__})
            out_rows.append(base)
        if rows_jsonl_path:
            with open(rows_jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out_rows[-1], ensure_ascii=True, sort_keys=True) + "\n")
        if progress_every and index % int(progress_every) == 0:
            print(f"[{index}/{len(rows)}] {out_rows[-1].get('status')} {Path(track).name}", flush=True)

    total = len(out_rows)
    usable = [row for row in out_rows if row.get("error_ms") not in (None, "")]
    error_values = [float(row["error_ms"]) for row in usable]
    within = {
        f"within_{threshold_ms}ms": int(sum(1 for value in error_values if value <= threshold_ms))
        for threshold_ms in THRESHOLDS_MS
    }
    within_percent = {
        key + "_percent": (100.0 * value / max(1, len(error_values)))
        for key, value in within.items()
    }
    exact_status = f"exact_{int(round(float(target_ms)))}ms"
    misses = sorted(
        [row for row in out_rows if row.get("status") not in {exact_status, "close_100ms"}],
        key=lambda item: float(item.get("error_ms", 999999.0) or 999999.0),
        reverse=True,
    )
    report: Dict[str, Any] = {
        "corrections": str(corrections_path),
        "batch_summary": str(summary_path),
        "review_state": "" if review_state_path is None else str(review_state_path),
        "latest_per_track": bool(latest_per_track),
        "total_truth_rows": int(total_truth_rows),
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "limit": int(limit),
        "offset": int(offset),
        "mode": str(mode),
        "expanded_limit": int(expanded_limit),
        "microalign_limit": int(microalign_limit),
        "sample_rate": int(sample_rate),
        "target_ms": float(target_ms),
        "elapsed_sec": round(float(time.time() - started), 3),
        "rows": int(total),
        "usable_predictions": int(len(error_values)),
        "counts": dict(counts),
        "within": within,
        "within_percent": within_percent,
        "median_error_ms": None if not error_values else float(sorted(error_values)[len(error_values) // 2]),
        "mean_error_ms": None if not error_values else float(sum(error_values) / len(error_values)),
        "worst_misses": misses[:50],
    }

    json_path = Path(output_json).expanduser()
    csv_path = Path(output_csv).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    fieldnames = [
        "index",
        "line_no",
        "track",
        "truth_source",
        "status",
        "error",
        "user_pick",
        "predicted",
        "error_ms",
        "bpm",
        "user_clock_bar",
        "user_beat_in_bar",
        "user_one_distance_ms",
        "predicted_clock_bar",
        "predicted_beat_in_bar",
        "predicted_one_distance_ms",
        "predicted_phrase",
        "predicted_grid_bar",
        "selected_by",
        "historical_selected_by",
        "reviewed_from",
        "confidence_tier",
        "reason",
        "structure_first_drop",
        "structure_second_drop",
        "bar_count",
        "feature_grid_zero",
        "source_count",
        "candidate_count",
        "closest_candidate_rank",
        "closest_candidate_time",
        "closest_candidate_error_ms",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    report["output_json"] = str(json_path)
    report["output_csv"] = str(csv_path)
    if rows_jsonl_path:
        report["rows_jsonl"] = str(rows_jsonl_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run current auto-place detector against historical approved/corrected markers.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS))
    parser.add_argument("--batch-summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--review-state", default=str(DEFAULT_REVIEW_STATE), help="Review state JSON to merge; pass an empty string to ignore it")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--rows-jsonl", default=str(DEFAULT_ROWS_JSONL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--all-rows", action="store_true", help="Use every correction row instead of the latest row per track")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--expanded-limit", type=int, default=120)
    parser.add_argument("--microalign-limit", type=int, default=50)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--target-ms", type=float, default=25.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip tracks already present in --rows-jsonl")
    parser.add_argument("--track-contains", default="", help="Only evaluate historical rows whose track path contains this text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_historical_auto_place(
        corrections=str(args.corrections),
        batch_summary=str(args.batch_summary),
        review_state=str(args.review_state or ""),
        output_json=str(args.output_json),
        output_csv=str(args.output_csv),
        rows_jsonl=str(args.rows_jsonl or ""),
        limit=int(args.limit),
        offset=int(args.offset),
        latest_per_track=not bool(args.all_rows),
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
        mode=str(args.mode),
        expanded_limit=int(args.expanded_limit),
        microalign_limit=int(args.microalign_limit),
        sample_rate=int(args.analysis_sr),
        target_ms=float(args.target_ms),
        progress_every=int(args.progress_every),
        resume=bool(args.resume),
        track_contains=str(args.track_contains or ""),
    )
    compact = {
        "total_truth_rows": report["total_truth_rows"],
        "rows": report["rows"],
        "usable_predictions": report["usable_predictions"],
        "counts": report["counts"],
        "within_percent": report["within_percent"],
        "median_error_ms": report["median_error_ms"],
        "elapsed_sec": report["elapsed_sec"],
        "output_json": report["output_json"],
        "output_csv": report["output_csv"],
        "rows_jsonl": report.get("rows_jsonl", ""),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
