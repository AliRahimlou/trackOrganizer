#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from drop_aligner.historical_markers import is_human_review_source  # noqa: E402
from drop_aligner.multistem import infer_bpm_from_path  # noqa: E402
from drop_aligner.musical_clock import bpm_clock_for_time  # noqa: E402
from drop_aligner.visual_first import visual_first_marker  # noqa: E402


DEFAULT_CORRECTIONS = (
    "models/multistem_training_corrections.jsonl",
    "drop_corrections.jsonl",
)
DEFAULT_JSON = "models/visual_first_audit.json"
DEFAULT_CSV = "models/visual_first_audit_misses.csv"
THRESHOLDS_MS = (25, 50, 100, 250, 500, 1000)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _path_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve()).lower()
    except OSError:
        return str(Path(path).expanduser()).lower()


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
                row["_source_path"] = str(path)
                yield row


def _load_truth_rows(paths: Sequence[str], *, track_contains: str = "", all_observations: bool = False) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    order = 0
    needle = track_contains.lower().strip()
    for path_text in paths:
        path = Path(path_text).expanduser()
        for row in _iter_jsonl(path):
            track = str(row.get("track") or row.get("filename") or row.get("audio_path") or "")
            user_pick = _float_or_none(row.get("user_pick"))
            if not track or user_pick is None or user_pick <= 0.0:
                continue
            if needle and needle not in track.lower():
                continue
            reviewed_from = str(row.get("reviewed_from") or "")
            if not is_human_review_source(reviewed_from):
                continue
            if not Path(track).expanduser().exists():
                continue
            order += 1
            truth = {
                "track": track,
                "expected": float(user_pick),
                "reviewed_from": reviewed_from,
                "selected_by": str(row.get("selected_by") or ""),
                "source_path": str(path),
                "line_no": int(row.get("_line_no") or 0),
                "order": order,
            }
            if all_observations:
                out.append(truth)
            else:
                latest[_path_key(track)] = truth
    if all_observations:
        return out
    return list(latest.values())


def _clock_label(time_sec: Optional[float], bpm: Optional[float], *, clock_zero_sec: float = 0.0) -> str:
    clock = bpm_clock_for_time(time_sec, bpm, clock_zero_sec=float(clock_zero_sec)) if time_sec is not None else None
    if not clock:
        return "--"
    distance = _float_or_none(clock.get("one_distance_ms"))
    distance_text = "--" if distance is None else f"{distance:.1f}ms"
    return f"b{clock.get('nearest_one_bar')} beat{clock.get('beat_in_bar')} {distance_text}"


def _failure_family(error_ms: Optional[float], audit: Mapping[str, Any], reason: str) -> str:
    if error_ms is None:
        return "detector_error"
    flags = {str(flag) for flag in audit.get("flag_codes") or []}
    reason_text = str(reason or "").lower()
    if "selected_matches_rejected_section" in flags:
        return "rejected_section_selected"
    if "late_body_after_section_entry" in flags:
        return "late_inside_drop_body"
    if "intro_before_stronger_drop" in flags:
        return "intro_or_buildup_before_stronger_drop"
    if "blank" in reason_text:
        return "blank_waveform_guard"
    if "grid" in reason_text or "track-zero" in reason_text or "half-bar" in reason_text or "one-beat" in reason_text:
        return "grid_phase_or_off_one"
    if float(error_ms) >= 8000.0:
        return "wrong_section"
    if float(error_ms) >= 1000.0:
        return "bar_or_phrase_offset"
    if float(error_ms) > 250.0:
        return "front_edge_or_grid_offset"
    if float(error_ms) > 50.0:
        return "micro_timing_offset"
    return "pass"


def _evaluate_one(payload: Mapping[str, Any]) -> Dict[str, Any]:
    track = str(payload["track"])
    expected = float(payload["expected"])
    sample_rate = int(payload["sample_rate"])
    started = time.time()
    try:
        result = visual_first_marker(track, sample_rate=sample_rate, use_cache=True)
    except Exception as exc:
        return {
            **{key: payload.get(key) for key in ("track", "expected", "reviewed_from", "selected_by", "source_path", "line_no")},
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
            "predicted": None,
            "error_ms": None,
            "elapsed_sec": time.time() - started,
        }
    selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
    audit = result.get("visual_audit") if isinstance(result.get("visual_audit"), Mapping) else {}
    predicted = _float_or_none(result.get("marker"))
    delta = None if predicted is None else float(predicted) - expected
    error_ms = None if delta is None else abs(float(delta) * 1000.0)
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    feature_map = result.get("feature_map") if isinstance(result.get("feature_map"), Mapping) else {}
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    bpm = _float_or_none(beatgrid.get("bpm")) or infer_bpm_from_path(track)
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_clock = selected.get("bpm_clock") if isinstance(selected.get("bpm_clock"), Mapping) else {}
    clock = (
        dict(selected_clock)
        if selected_clock and ("on_one" in selected_clock or "one_distance_ms" in selected_clock)
        else bpm_clock_for_time(predicted, bpm, clock_zero_sec=float(clock_zero_sec))
        if predicted is not None and bpm
        else None
    )
    one_distance_ms = _float_or_none(clock.get("one_distance_ms")) if isinstance(clock, Mapping) else None
    top_candidates = []
    for candidate in result.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        value = _float_or_none(candidate.get("timestamp"))
        if value is None:
            value = _float_or_none(candidate.get("time_sec"))
        top_candidates.append(
            {
                "time": value,
                "selected_by": str(candidate.get("selected_by") or ""),
                "score": _float_or_none(candidate.get("score")),
                "reason": str(candidate.get("reason") or "")[:160],
            }
        )
        if len(top_candidates) >= 5:
            break
    return {
        **{key: payload.get(key) for key in ("track", "expected", "reviewed_from", "selected_by", "source_path", "line_no")},
        "ok": bool(result.get("ok")),
        "predicted": predicted,
        "delta": delta,
        "error_ms": error_ms,
        "raw_visual_time": _float_or_none(result.get("raw_visual_time")),
        "source": str(selected.get("selected_by") or ""),
        "reason": str(selected.get("reason") or ""),
        "clock_bar": visual.get("clock_bar"),
        "on_one": bool(clock.get("on_one")) if isinstance(clock, Mapping) else False,
        "one_distance_ms": one_distance_ms,
        "clock": _clock_label(predicted, bpm, clock_zero_sec=float(clock_zero_sec)),
        "clock_zero_sec": float(clock_zero_sec),
        "visual_audit_status": str(audit.get("status") or ""),
        "visual_audit_action": str(audit.get("recommended_action") or ""),
        "visual_audit_flags": list(audit.get("flag_codes") or []),
        "failure_family": _failure_family(error_ms, audit, str(selected.get("reason") or "")),
        "elapsed_sec": time.time() - started,
        "top_candidates": top_candidates,
    }


def _percent(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round((100.0 * float(count)) / float(total), 2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "error_ms",
        "expected",
        "predicted",
        "delta",
        "track",
        "source",
        "clock_bar",
        "on_one",
        "one_distance_ms",
        "clock",
        "clock_zero_sec",
        "visual_audit_status",
        "visual_audit_action",
        "visual_audit_flags",
        "failure_family",
        "reviewed_from",
        "selected_by",
        "reason",
        "source_path",
        "line_no",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    truth_rows = _load_truth_rows(
        args.corrections,
        track_contains=str(args.track_contains or ""),
        all_observations=bool(args.all_observations),
    )
    if args.limit:
        truth_rows = truth_rows[: int(args.limit)]
    started = time.time()
    jobs = [dict(row, sample_rate=int(args.sample_rate)) for row in truth_rows]
    results: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_evaluate_one(job))
            if args.progress_every and index % int(args.progress_every) == 0:
                print(f"evaluated {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_evaluate_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"evaluated {index}/{len(jobs)}", flush=True)

    results.sort(key=lambda row: float(row.get("error_ms") if row.get("error_ms") is not None else 1e18), reverse=True)
    ok_results = [row for row in results if row.get("ok") and row.get("error_ms") is not None]
    errors = [float(row["error_ms"]) for row in ok_results]
    total = len(results)
    summary = {
        "evaluated": total,
        "ok": len(ok_results),
        "failed": total - len(ok_results),
        "elapsed_sec": time.time() - started,
        "sample_rate": int(args.sample_rate),
        "corrections": list(args.corrections),
        "track_contains": str(args.track_contains or ""),
        "median_error_ms": statistics.median(errors) if errors else None,
        "mean_error_ms": statistics.mean(errors) if errors else None,
        "max_error_ms": max(errors) if errors else None,
        "thresholds": {
            f"within_{threshold}_ms": {
                "count": sum(1 for error in errors if error <= float(threshold)),
                "pct": _percent(sum(1 for error in errors if error <= float(threshold)), len(errors)),
            }
            for threshold in THRESHOLDS_MS
        },
        "wrong_section_count": sum(1 for error in errors if error >= 8000.0),
        "worst": results[: int(args.worst)],
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({**summary, "results": results}, indent=2, ensure_ascii=True), encoding="utf-8")
    misses = [row for row in results if row.get("error_ms") is None or float(row.get("error_ms") or 0.0) > float(args.miss_ms)]
    _write_csv(Path(args.output_csv).expanduser(), misses)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote misses CSV: {Path(args.output_csv).expanduser()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual_first_marker against human-reviewed web placements.")
    parser.add_argument("--corrections", action="append", default=list(DEFAULT_CORRECTIONS), help="Human correction JSONL. Repeatable.")
    parser.add_argument("--track-contains", default="", help="Only evaluate tracks whose path contains this text.")
    parser.add_argument("--all-observations", action="store_true", help="Evaluate every correction row instead of latest per track.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum tracks to evaluate; 0 means all.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--miss-ms", type=float, default=250.0, help="Rows above this error go to the misses CSV.")
    parser.add_argument("--worst", type=int, default=25, help="Number of worst rows to include in printed summary.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N completed tracks; 0 disables.")
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
