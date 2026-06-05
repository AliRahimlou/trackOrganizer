#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.detector import DropDetectorConfig, detect_drop
from drop_aligner.exclusions import row_has_excluded_path
from verify_als import verify_als


BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.IGNORECASE)
OUTPUT_JSON = "drumprint_ablation_report.json"
OUTPUT_CSV = "drumprint_ablation_report.csv"
CSV_COLUMNS = [
    "track",
    "no_drumprint_time",
    "drumprint_time",
    "user_or_golden_time",
    "no_drumprint_error_ms",
    "drumprint_error_ms",
    "improvement_ms",
    "no_drumprint_confidence_tier",
    "drumprint_confidence_tier",
    "drumprint_pattern_score",
    "fake_hit_penalty",
    "result",
    "notes",
]
THRESHOLDS_MS = [5, 10, 25, 50, 100]
UNCHANGED_TOLERANCE_MS = 1.0
FAKE_HIT_TIME_DELTA_SEC = 0.025
SIGNIFICANT_REGRESSION_MS = 25.0


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_float(value: Any, digits: int = 6) -> str:
    out = _float_or_none(value)
    if out is None:
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def _read_json(path: str) -> Any:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_summary(path: str) -> List[Dict[str, str]]:
    summary = Path(path).expanduser()
    if not summary.exists():
        raise FileNotFoundError(f"Batch summary not found: {summary}")
    rows: List[Dict[str, str]] = []
    with open(summary, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row_has_excluded_path(row):
                continue
            filename = str(row.get("filename") or "").strip()
            if filename:
                rows.append(dict(row))
    return rows


def _track_aliases(track: str) -> List[str]:
    raw = str(track or "").strip()
    if not raw:
        return []
    aliases = [raw]
    path = Path(raw).expanduser()
    try:
        aliases.append(str(path.resolve()))
    except Exception:
        pass
    aliases.append(path.name)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _add_reference(lookup: Dict[str, Dict[str, Any]], track: str, value: float, source: str, meta: Mapping[str, Any]) -> None:
    ref = {
        "track": str(track),
        "time": float(value),
        "source": str(source),
        "meta": dict(meta),
    }
    for alias in _track_aliases(track):
        lookup[alias] = ref


def _reference_for_track(lookup: Mapping[str, Dict[str, Any]], track: str) -> Optional[Dict[str, Any]]:
    for alias in _track_aliases(track):
        ref = lookup.get(alias)
        if ref is not None:
            return dict(ref)
    return None


def _load_corrections(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not path:
        return lookup
    corrections = Path(path).expanduser()
    if not corrections.exists():
        return lookup
    with open(corrections, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping) or row_has_excluded_path(row):
                continue
            track = str(row.get("track") or row.get("filename") or row.get("audio_path") or "").strip()
            user_pick = _float_or_none(row.get("user_pick"))
            if track and user_pick is not None:
                _add_reference(lookup, track, user_pick, "correction", {"line_no": line_no, "timestamp": row.get("timestamp", "")})
    return lookup


def _golden_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("tracks"), Sequence):
        for row in payload["tracks"]:
            if isinstance(row, Mapping):
                yield row
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for row in payload:
            if isinstance(row, Mapping):
                yield row
        return
    if isinstance(payload, Mapping):
        for track, value in payload.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("track", track)
                yield row
            else:
                yield {"track": track, "expected_drop_time": value}


def _load_golden(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not path:
        return lookup
    golden = Path(path).expanduser()
    if not golden.exists():
        return lookup
    payload = _read_json(str(golden))
    for index, row in enumerate(_golden_records(payload), start=1):
        if row_has_excluded_path(row):
            continue
        track = str(row.get("track") or row.get("filename") or row.get("audio_path") or "").strip()
        expected = None
        for key in ("expected_drop_time", "drop_sec", "user_pick", "final_ai_pick", "detected_drop_time"):
            expected = _float_or_none(row.get(key))
            if expected is not None:
                break
        if track and expected is not None:
            _add_reference(lookup, track, expected, "golden", {"index": index})
    return lookup


def _candidate_json_bpm(row: Mapping[str, str]) -> Optional[float]:
    path = str(row.get("candidates_json") or "").strip()
    if not path:
        return None
    json_path = Path(path).expanduser()
    if not json_path.exists():
        return None
    try:
        payload = _read_json(str(json_path))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    value = _float_or_none(payload.get("bpm"))
    if value is not None:
        return value
    features = payload.get("feature_summary")
    if isinstance(features, Mapping):
        return _float_or_none(features.get("bpm"))
    return None


def _infer_bpm_from_path(path: str) -> Optional[float]:
    audio = Path(path)
    match = BPM_RE.match(audio.name)
    if match:
        return float(match.group(1))
    for parent in [audio.parent, *audio.parents]:
        value = _float_or_none(parent.name)
        if value is not None and 60.0 <= value <= 220.0:
            return value
    return None


def _looks_like_drums(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("drums") or name.startswith("drum_") or name.startswith("drum-")


def _metric(candidate: Mapping[str, Any], feature_summary: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        nested = candidate.get("drumprint")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    if value is None:
        value = feature_summary.get(f"chosen_{key}")
    return _float_or_none(value) or 0.0


def _mode_payload(result: Any) -> Dict[str, Any]:
    selected = result.selected_candidate_dict() or {}
    top = result.top_candidate_dicts(1)
    feature_summary = dict(result.features_summary)
    return {
        "ok": True,
        "time": float(result.drop_sec),
        "coarse_time": float(result.coarse_sec),
        "bpm": float(result.bpm),
        "confidence": float(result.confidence),
        "confidence_tier": str(result.confidence_tier),
        "selected_by": str(result.selected_by),
        "selected_candidate_rank": selected.get("rank"),
        "selected_candidate_handcrafted_rank": selected.get("handcrafted_rank"),
        "selected_candidate_model_rank": selected.get("model_rank"),
        "drumprint_pattern_score": _metric(selected, feature_summary, "drumprint_pattern_score"),
        "fake_hit_penalty": _metric(selected, feature_summary, "fake_hit_penalty"),
        "post_drop_pattern_stability": _metric(selected, feature_summary, "post_drop_pattern_stability"),
        "later_drop_match_score": _metric(selected, feature_summary, "later_drop_match_score"),
        "selected_candidate": selected,
        "top_candidate": top[0] if top else {},
        "feature_summary": feature_summary,
    }


def _run_detection_mode(audio_path: str, bpm: Optional[float], use_drumprint: bool) -> Dict[str, Any]:
    cfg = DropDetectorConfig(
        sample_rate=22050,
        hpss=not _looks_like_drums(audio_path),
        use_drumprint=bool(use_drumprint),
    )
    result = detect_drop(audio_path, bpm=bpm, config=cfg)
    return _mode_payload(result)


def _verify_existing_als(row: Mapping[str, str]) -> Dict[str, Any]:
    output_als = str(row.get("output_als") or "").strip()
    if not output_als:
        return {"checked": False, "valid": None, "reason": "missing output_als column"}
    als_path = Path(output_als).expanduser()
    if not als_path.exists():
        return {"checked": False, "valid": None, "reason": f"missing ALS: {als_path}"}

    candidates_json = str(row.get("candidates_json") or "").strip()
    candidates_arg = candidates_json if candidates_json and Path(candidates_json).expanduser().exists() else None
    try:
        report = verify_als(str(als_path), candidates_json=candidates_arg)
    except Exception as exc:
        return {"checked": True, "valid": False, "reason": str(exc) or exc.__class__.__name__}
    return {
        "checked": True,
        "valid": bool(report.get("valid")),
        "drop_marker_time": report.get("drop_marker_time"),
        "errors": list(report.get("errors", [])),
    }


def _process_track(task: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(task["row"])
    track = str(row.get("filename") or "")
    bpm = _float_or_none(task.get("bpm"))
    out: Dict[str, Any] = {
        "track": track,
        "row": row,
        "bpm": bpm,
        "no_drumprint": {"ok": False, "error": "not_run"},
        "drumprint": {"ok": False, "error": "not_run"},
        "als_verification": _verify_existing_als(row),
    }
    audio = Path(track).expanduser()
    if not audio.exists():
        out["error"] = f"audio not found: {audio}"
        return out

    for key, use_drumprint in (("no_drumprint", False), ("drumprint", True)):
        try:
            out[key] = _run_detection_mode(str(audio), bpm, use_drumprint)
        except Exception as exc:
            out[key] = {"ok": False, "error": str(exc) or exc.__class__.__name__}
    return out


def _error_ms(time_sec: Optional[float], target_sec: Optional[float]) -> Optional[float]:
    if time_sec is None or target_sec is None:
        return None
    return abs(float(time_sec) - float(target_sec)) * 1000.0


def _result_label(no_error: Optional[float], drum_error: Optional[float]) -> str:
    if no_error is None or drum_error is None:
        return "unchanged"
    improvement = float(no_error) - float(drum_error)
    if improvement > UNCHANGED_TOLERANCE_MS:
        return "improved"
    if improvement < -UNCHANGED_TOLERANCE_MS:
        return "worsened"
    return "unchanged"


def _csv_row(detail: Mapping[str, Any]) -> Dict[str, str]:
    no_mode = detail.get("no_drumprint") if isinstance(detail.get("no_drumprint"), Mapping) else {}
    drum_mode = detail.get("drumprint") if isinstance(detail.get("drumprint"), Mapping) else {}
    return {
        "track": str(detail.get("track", "")),
        "no_drumprint_time": _format_float(no_mode.get("time"), 9),
        "drumprint_time": _format_float(drum_mode.get("time"), 9),
        "user_or_golden_time": _format_float(detail.get("reference_time"), 9),
        "no_drumprint_error_ms": _format_float(detail.get("no_drumprint_error_ms"), 3),
        "drumprint_error_ms": _format_float(detail.get("drumprint_error_ms"), 3),
        "improvement_ms": _format_float(detail.get("improvement_ms"), 3),
        "no_drumprint_confidence_tier": str(no_mode.get("confidence_tier") or ""),
        "drumprint_confidence_tier": str(drum_mode.get("confidence_tier") or ""),
        "drumprint_pattern_score": _format_float(drum_mode.get("drumprint_pattern_score"), 6),
        "fake_hit_penalty": _format_float(drum_mode.get("fake_hit_penalty"), 6),
        "result": str(detail.get("result", "")),
        "notes": str(detail.get("notes", "")),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))
    return str(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
    return str(path)


def _percent_within(errors: Sequence[float], threshold_ms: float) -> float:
    if not errors:
        return 0.0
    return float(100.0 * sum(1 for err in errors if err <= float(threshold_ms)) / len(errors))


def _metrics(details: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    referenced = [
        row
        for row in details
        if _float_or_none(row.get("no_drumprint_error_ms")) is not None
        and _float_or_none(row.get("drumprint_error_ms")) is not None
    ]
    no_errors = [float(row["no_drumprint_error_ms"]) for row in referenced]
    drum_errors = [float(row["drumprint_error_ms"]) for row in referenced]
    improvements = [float(row["improvement_ms"]) for row in referenced if _float_or_none(row.get("improvement_ms")) is not None]

    return {
        "referenced_tracks": int(len(referenced)),
        "no_drumprint": {
            "mean_absolute_error_ms": float(mean(no_errors)) if no_errors else 0.0,
            "median_absolute_error_ms": float(median(no_errors)) if no_errors else 0.0,
            "percent_within": {f"{threshold}ms": _percent_within(no_errors, threshold) for threshold in THRESHOLDS_MS},
        },
        "drumprint": {
            "mean_absolute_error_ms": float(mean(drum_errors)) if drum_errors else 0.0,
            "median_absolute_error_ms": float(median(drum_errors)) if drum_errors else 0.0,
            "percent_within": {f"{threshold}ms": _percent_within(drum_errors, threshold) for threshold in THRESHOLDS_MS},
        },
        "mean_improvement_ms": float(mean(improvements)) if improvements else 0.0,
        "median_improvement_ms": float(median(improvements)) if improvements else 0.0,
        "tracks_improved_by_drumprint": int(sum(1 for row in referenced if row.get("result") == "improved")),
        "tracks_worsened_by_drumprint": int(sum(1 for row in referenced if row.get("result") == "worsened")),
        "tracks_unchanged": int(sum(1 for row in referenced if row.get("result") == "unchanged")),
        "best_drumprint_improvements": sorted(
            [row for row in referenced if _float_or_none(row.get("improvement_ms")) is not None],
            key=lambda row: float(row["improvement_ms"]),
            reverse=True,
        )[:20],
        "worst_drumprint_regressions": sorted(
            [row for row in referenced if _float_or_none(row.get("improvement_ms")) is not None],
            key=lambda row: float(row["improvement_ms"]),
        )[:20],
    }


def _fake_hit_analysis(details: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rescues: List[Mapping[str, Any]] = []
    regressions: List[Mapping[str, Any]] = []
    for row in details:
        no_mode = row.get("no_drumprint") if isinstance(row.get("no_drumprint"), Mapping) else {}
        drum_mode = row.get("drumprint") if isinstance(row.get("drumprint"), Mapping) else {}
        no_time = _float_or_none(no_mode.get("time"))
        drum_time = _float_or_none(drum_mode.get("time"))
        improvement = _float_or_none(row.get("improvement_ms"))
        if no_time is None or drum_time is None or improvement is None:
            continue
        if drum_time <= no_time + FAKE_HIT_TIME_DELTA_SEC:
            continue
        if improvement > UNCHANGED_TOLERANCE_MS:
            rescues.append(row)
        elif improvement < -UNCHANGED_TOLERANCE_MS:
            regressions.append(row)
    return {
        "fake_hit_rescues": int(len(rescues)),
        "fake_hit_regressions": int(len(regressions)),
        "examples": [
            {
                "track": row.get("track"),
                "reference_time": row.get("reference_time"),
                "no_drumprint_time": row.get("no_drumprint", {}).get("time"),
                "drumprint_time": row.get("drumprint", {}).get("time"),
                "improvement_ms": row.get("improvement_ms"),
                "fake_hit_penalty": row.get("drumprint", {}).get("fake_hit_penalty"),
                "drumprint_pattern_score": row.get("drumprint", {}).get("drumprint_pattern_score"),
            }
            for row in rescues[:10]
        ],
        "regression_examples": [
            {
                "track": row.get("track"),
                "reference_time": row.get("reference_time"),
                "no_drumprint_time": row.get("no_drumprint", {}).get("time"),
                "drumprint_time": row.get("drumprint", {}).get("time"),
                "improvement_ms": row.get("improvement_ms"),
                "fake_hit_penalty": row.get("drumprint", {}).get("fake_hit_penalty"),
                "drumprint_pattern_score": row.get("drumprint", {}).get("drumprint_pattern_score"),
            }
            for row in regressions[:10]
        ],
    }


def _high_confidence_regressions(details: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in details:
        no_mode = row.get("no_drumprint") if isinstance(row.get("no_drumprint"), Mapping) else {}
        no_error = _float_or_none(row.get("no_drumprint_error_ms"))
        drum_error = _float_or_none(row.get("drumprint_error_ms"))
        if no_mode.get("confidence_tier") != "HIGH" or no_error is None or drum_error is None:
            continue
        if no_error <= 25.0 and (drum_error - no_error) > SIGNIFICANT_REGRESSION_MS:
            out.append(row)
    return out


def _recommendation(metrics: Mapping[str, Any], fake_hit: Mapping[str, Any], high_regressions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    no_metrics = metrics.get("no_drumprint", {}) if isinstance(metrics.get("no_drumprint"), Mapping) else {}
    drum_metrics = metrics.get("drumprint", {}) if isinstance(metrics.get("drumprint"), Mapping) else {}
    no_p25 = float((no_metrics.get("percent_within") or {}).get("25ms", 0.0))
    drum_p25 = float((drum_metrics.get("percent_within") or {}).get("25ms", 0.0))
    no_median = float(no_metrics.get("median_absolute_error_ms", 0.0))
    drum_median = float(drum_metrics.get("median_absolute_error_ms", 0.0))
    rescues = int(fake_hit.get("fake_hit_rescues", 0))
    regressions = int(fake_hit.get("fake_hit_regressions", 0))

    positive_gate = drum_median < no_median or drum_p25 > no_p25 or rescues > regressions
    high_safe = len(high_regressions) == 0
    recommend = bool(metrics.get("referenced_tracks", 0)) and positive_gate and high_safe
    reasons: List[str] = []
    if drum_median < no_median:
        reasons.append(f"median error improved {no_median:.3f}ms -> {drum_median:.3f}ms")
    if drum_p25 > no_p25:
        reasons.append(f"25ms accuracy improved {no_p25:.2f}% -> {drum_p25:.2f}%")
    if rescues > regressions:
        reasons.append(f"fake-hit rescues beat regressions {rescues} -> {regressions}")
    if high_regressions:
        reasons.append(f"{len(high_regressions)} already-correct HIGH-confidence tracks regressed by >{SIGNIFICANT_REGRESSION_MS:.0f}ms")
    if not metrics.get("referenced_tracks", 0):
        reasons.append("no correction/golden references were available")
    if not reasons:
        reasons.append("no pass criteria improved")
    return {
        "recommend_drumprint_default": recommend,
        "positive_gate_passed": bool(positive_gate),
        "high_confidence_regression_gate_passed": bool(high_safe),
        "reasons": reasons,
    }


def _attach_reference_and_comparison(result: Mapping[str, Any], reference: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    detail = dict(result)
    no_mode = detail.get("no_drumprint") if isinstance(detail.get("no_drumprint"), Mapping) else {}
    drum_mode = detail.get("drumprint") if isinstance(detail.get("drumprint"), Mapping) else {}
    target = _float_or_none(reference.get("time")) if reference else None
    no_time = _float_or_none(no_mode.get("time"))
    drum_time = _float_or_none(drum_mode.get("time"))
    no_error = _error_ms(no_time, target)
    drum_error = _error_ms(drum_time, target)
    improvement = None if no_error is None or drum_error is None else float(no_error - drum_error)
    notes: List[str] = []
    if reference:
        notes.append(str(reference.get("source", "reference")))
    else:
        notes.append("no reference target")
    if not no_mode.get("ok"):
        notes.append(f"no-drumprint failed: {no_mode.get('error')}")
    if not drum_mode.get("ok"):
        notes.append(f"drumprint failed: {drum_mode.get('error')}")
    als = detail.get("als_verification")
    if isinstance(als, Mapping):
        if als.get("valid") is True:
            notes.append("existing ALS valid")
        elif als.get("checked"):
            notes.append("existing ALS invalid")
        else:
            notes.append(str(als.get("reason", "existing ALS not checked")))

    detail["reference_time"] = target
    detail["reference_source"] = reference.get("source") if reference else ""
    detail["no_drumprint_error_ms"] = no_error
    detail["drumprint_error_ms"] = drum_error
    detail["improvement_ms"] = improvement
    detail["result"] = _result_label(no_error, drum_error)
    detail["notes"] = "; ".join(notes)
    return detail


def _print_summary(report: Mapping[str, Any], json_path: str, csv_path: str) -> None:
    metrics = report.get("metrics", {})
    fake_hit = report.get("fake_hit_analysis", {})
    recommendation = report.get("recommendation", {})
    no_metrics = metrics.get("no_drumprint", {}) if isinstance(metrics.get("no_drumprint"), Mapping) else {}
    drum_metrics = metrics.get("drumprint", {}) if isinstance(metrics.get("drumprint"), Mapping) else {}

    print("\nDrumPrint A/B summary")
    print(f"Tracks scanned: {report.get('tracks_scanned', 0)}")
    print(f"Tracks with references: {metrics.get('referenced_tracks', 0)}")
    print(f"Detection failures: no-drumprint={report.get('no_drumprint_failures', 0)}, drumprint={report.get('drumprint_failures', 0)}")
    if metrics.get("referenced_tracks", 0):
        print(
            "Median error: "
            f"no-drumprint={float(no_metrics.get('median_absolute_error_ms', 0.0)):.3f}ms, "
            f"drumprint={float(drum_metrics.get('median_absolute_error_ms', 0.0)):.3f}ms"
        )
        print(
            "25ms accuracy: "
            f"no-drumprint={float((no_metrics.get('percent_within') or {}).get('25ms', 0.0)):.2f}%, "
            f"drumprint={float((drum_metrics.get('percent_within') or {}).get('25ms', 0.0)):.2f}%"
        )
        print(
            "Track outcomes: "
            f"improved={metrics.get('tracks_improved_by_drumprint', 0)}, "
            f"worsened={metrics.get('tracks_worsened_by_drumprint', 0)}, "
            f"unchanged={metrics.get('tracks_unchanged', 0)}"
        )
        print(
            "Fake-hit analysis: "
            f"rescues={fake_hit.get('fake_hit_rescues', 0)}, "
            f"regressions={fake_hit.get('fake_hit_regressions', 0)}"
        )
    print(f"Recommendation: {'KEEP DrumPrint default' if recommendation.get('recommend_drumprint_default') else 'DO NOT promote yet'}")
    for reason in recommendation.get("reasons", []):
        print(f"  - {reason}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare drop detection with and without DrumPrint.")
    parser.add_argument("summary_csv", help="drop_batch_summary.csv from batch.py")
    parser.add_argument("--template", required=True, help="Ableton .als template path. Only validated; not modified.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL with user_pick references")
    parser.add_argument("--golden", help="Optional golden_tracks.json with expected_drop_time references")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="models/drumprint_eval")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary_csv).expanduser()
    template = Path(args.template).expanduser()
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    rows = _read_summary(str(summary_path))
    corrections = _load_corrections(args.corrections)
    golden = _load_golden(args.golden)

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        bpm = _candidate_json_bpm(row) or _infer_bpm_from_path(str(row.get("filename", "")))
        tasks.append({"row": row, "bpm": bpm})

    raw_results: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            result = _process_track(task)
            raw_results.append(result)
            print(f"[{index}/{len(tasks)}] compared: {result.get('track', '')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_track = {pool.submit(_process_track, task): str(task["row"].get("filename", "")) for task in tasks}
            for index, future in enumerate(as_completed(future_to_track), start=1):
                track = future_to_track[future]
                try:
                    raw_results.append(future.result())
                except Exception as exc:
                    raw_results.append({"track": track, "error": str(exc) or exc.__class__.__name__})
                print(f"[{index}/{len(tasks)}] compared: {track}", flush=True)
        raw_results.sort(key=lambda row: str(row.get("track", "")).lower())

    details: List[Dict[str, Any]] = []
    for result in raw_results:
        track = str(result.get("track", ""))
        reference = _reference_for_track(corrections, track) or _reference_for_track(golden, track)
        details.append(_attach_reference_and_comparison(result, reference))

    metrics = _metrics(details)
    fake_hit = _fake_hit_analysis(details)
    high_regressions = _high_confidence_regressions(details)
    recommendation = _recommendation(metrics, fake_hit, high_regressions)

    output_dir = Path(args.output_dir).expanduser()
    csv_path = output_dir / OUTPUT_CSV
    json_path = output_dir / OUTPUT_JSON
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_csv": str(summary_path),
        "template": str(template),
        "corrections": str(Path(args.corrections).expanduser()) if args.corrections else "",
        "golden": str(Path(args.golden).expanduser()) if args.golden else "",
        "output_dir": str(output_dir),
        "tracks_scanned": int(len(rows)),
        "no_drumprint_failures": int(sum(1 for row in details if not (row.get("no_drumprint") or {}).get("ok"))),
        "drumprint_failures": int(sum(1 for row in details if not (row.get("drumprint") or {}).get("ok"))),
        "metrics": metrics,
        "fake_hit_analysis": fake_hit,
        "high_confidence_regressions": list(high_regressions[:20]),
        "recommendation": recommendation,
        "rows": details,
    }
    written_csv = _write_csv(csv_path, details)
    written_json = _write_json(json_path, report)
    _print_summary(report, written_json, written_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
