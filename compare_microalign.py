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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.detector import DropDetectorConfig, detect_drop
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.microalign import should_auto_accept


BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.IGNORECASE)
OUTPUT_JSON = "microalign_ablation_report.json"
OUTPUT_CSV = "microalign_ablation_report.csv"
THRESHOLDS_MS = [1, 5, 10, 25, 50, 100]
UNCHANGED_TOLERANCE_MS = 1.0
SIGNIFICANT_REGRESSION_MS = 25.0
CSV_COLUMNS = [
    "track",
    "no_microalign_time",
    "microalign_time",
    "user_or_golden_time",
    "no_microalign_error_ms",
    "microalign_error_ms",
    "improvement_ms",
    "snap_offset_ms",
    "micro_confidence",
    "attack_cleanliness",
    "zero_crossing_quality",
    "sustained_after_attack",
    "confidence_tier",
    "auto_accept_conservative",
    "auto_accept_normal",
    "auto_accept_aggressive",
    "result",
    "notes",
]


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    out = _float_or_none(value)
    if out is None:
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def _read_json(path: str) -> Any:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_summary(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(Path(path).expanduser(), "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row_has_excluded_path(row) and str(row.get("filename") or "").strip():
                rows.append(dict(row))
    return rows


def _track_aliases(track: str) -> List[str]:
    raw = str(track or "").strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    aliases = [raw, path.name]
    try:
        aliases.append(str(path.resolve()))
    except Exception:
        pass
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _add_reference(lookup: Dict[str, Dict[str, Any]], track: str, value: float, source: str, meta: Mapping[str, Any]) -> None:
    ref = {"track": str(track), "time": float(value), "source": str(source), "meta": dict(meta)}
    for alias in _track_aliases(track):
        lookup[alias] = ref


def _reference_for_track(lookup: Mapping[str, Dict[str, Any]], track: str) -> Optional[Dict[str, Any]]:
    for alias in _track_aliases(track):
        if alias in lookup:
            return dict(lookup[alias])
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
        nested = candidate.get("microalign")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    if value is None:
        nested = candidate.get("drumprint")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    if value is None:
        value = feature_summary.get(f"chosen_{key}")
    return _float_or_none(value) or 0.0


def _mode_payload(result: Any) -> Dict[str, Any]:
    selected = result.selected_candidate_dict() or {}
    top_10 = result.top_candidate_dicts(10)
    feature_summary = dict(result.features_summary)
    selected_with_tier = dict(selected)
    selected_with_tier["confidence_tier"] = result.confidence_tier
    gates = {
        mode: should_auto_accept(selected_with_tier, mode, candidates=top_10, confidence_tier=str(result.confidence_tier))
        for mode in ("conservative", "normal", "aggressive")
    }
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
        "microaligned_time": _metric(selected, feature_summary, "microaligned_time"),
        "snap_offset_ms": _metric(selected, feature_summary, "snap_offset_ms"),
        "micro_confidence": _metric(selected, feature_summary, "micro_confidence"),
        "attack_cleanliness": _metric(selected, feature_summary, "attack_cleanliness"),
        "zero_crossing_quality": _metric(selected, feature_summary, "zero_crossing_quality"),
        "sustained_after_attack": _metric(selected, feature_summary, "sustained_after_attack"),
        "drumprint_pattern_score": _metric(selected, feature_summary, "drumprint_pattern_score"),
        "fake_hit_penalty": _metric(selected, feature_summary, "fake_hit_penalty"),
        "auto_accept": gates,
        "selected_candidate": selected,
        "top_10_candidates": top_10,
        "feature_summary": feature_summary,
    }


def _run_detection(audio_path: str, bpm: Optional[float], use_microalign: bool) -> Dict[str, Any]:
    cfg = DropDetectorConfig(
        sample_rate=22050,
        hpss=not _looks_like_drums(audio_path),
        use_microalign=bool(use_microalign),
    )
    result = detect_drop(audio_path, bpm=bpm, config=cfg)
    return _mode_payload(result)


def _process_track(task: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(task["row"])
    track = str(row.get("filename") or "")
    bpm = _float_or_none(task.get("bpm"))
    out: Dict[str, Any] = {
        "track": track,
        "row": row,
        "bpm": bpm,
        "no_microalign": {"ok": False, "error": "not_run"},
        "microalign": {"ok": False, "error": "not_run"},
    }
    audio = Path(track).expanduser()
    if not audio.exists():
        out["error"] = f"audio not found: {audio}"
        return out
    for key, enabled in (("no_microalign", False), ("microalign", True)):
        try:
            out[key] = _run_detection(str(audio), bpm, enabled)
        except Exception as exc:
            out[key] = {"ok": False, "error": str(exc) or exc.__class__.__name__}
    return out


def _error_ms(time_sec: Optional[float], target_sec: Optional[float]) -> Optional[float]:
    if time_sec is None or target_sec is None:
        return None
    return abs(float(time_sec) - float(target_sec)) * 1000.0


def _result_label(no_error: Optional[float], micro_error: Optional[float]) -> str:
    if no_error is None or micro_error is None:
        return "unchanged"
    improvement = float(no_error) - float(micro_error)
    if improvement > UNCHANGED_TOLERANCE_MS:
        return "improved"
    if improvement < -UNCHANGED_TOLERANCE_MS:
        return "worsened"
    return "unchanged"


def _attach_reference(result: Mapping[str, Any], reference: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    detail = dict(result)
    no_mode = detail.get("no_microalign") if isinstance(detail.get("no_microalign"), Mapping) else {}
    micro_mode = detail.get("microalign") if isinstance(detail.get("microalign"), Mapping) else {}
    target = _float_or_none(reference.get("time")) if reference else None
    no_time = _float_or_none(no_mode.get("time"))
    micro_time = _float_or_none(micro_mode.get("time"))
    no_error = _error_ms(no_time, target)
    micro_error = _error_ms(micro_time, target)
    improvement = None if no_error is None or micro_error is None else float(no_error - micro_error)
    notes: List[str] = []
    if reference:
        notes.append(str(reference.get("source", "reference")))
    else:
        notes.append("no reference target")
    if not no_mode.get("ok"):
        notes.append(f"no-microalign failed: {no_mode.get('error')}")
    if not micro_mode.get("ok"):
        notes.append(f"microalign failed: {micro_mode.get('error')}")
    detail["reference_time"] = target
    detail["reference_source"] = reference.get("source") if reference else ""
    detail["no_microalign_error_ms"] = no_error
    detail["microalign_error_ms"] = micro_error
    detail["improvement_ms"] = improvement
    detail["result"] = _result_label(no_error, micro_error)
    detail["notes"] = "; ".join(notes)
    return detail


def _percent_within(errors: Sequence[float], threshold_ms: float) -> float:
    if not errors:
        return 0.0
    return float(100.0 * sum(1 for err in errors if err <= float(threshold_ms)) / len(errors))


def _high_confidence_regressions(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in rows:
        no_mode = row.get("no_microalign") if isinstance(row.get("no_microalign"), Mapping) else {}
        no_error = _float_or_none(row.get("no_microalign_error_ms"))
        micro_error = _float_or_none(row.get("microalign_error_ms"))
        if no_mode.get("confidence_tier") != "HIGH" or no_error is None or micro_error is None:
            continue
        if no_error <= 25.0 and (micro_error - no_error) > SIGNIFICANT_REGRESSION_MS:
            out.append(row)
    return out


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    referenced = [
        row
        for row in rows
        if _float_or_none(row.get("no_microalign_error_ms")) is not None
        and _float_or_none(row.get("microalign_error_ms")) is not None
    ]
    no_errors = [float(row["no_microalign_error_ms"]) for row in referenced]
    micro_errors = [float(row["microalign_error_ms"]) for row in referenced]
    improvements = [float(row["improvement_ms"]) for row in referenced if _float_or_none(row.get("improvement_ms")) is not None]
    offsets = [
        abs(float((row.get("microalign") or {}).get("snap_offset_ms")))
        for row in rows
        if isinstance(row.get("microalign"), Mapping) and _float_or_none(row.get("microalign", {}).get("snap_offset_ms")) is not None
    ]
    eligible = {
        mode: sum(
            1
            for row in rows
            if isinstance(row.get("microalign"), Mapping)
            and isinstance(row["microalign"].get("auto_accept"), Mapping)
            and bool(row["microalign"]["auto_accept"].get(mode, {}).get("auto_accept"))
        )
        for mode in ("conservative", "normal", "aggressive")
    }
    high_regressions = _high_confidence_regressions(referenced)
    return {
        "referenced_tracks": int(len(referenced)),
        "no_microalign": {
            "mean_absolute_error_ms": float(mean(no_errors)) if no_errors else 0.0,
            "median_absolute_error_ms": float(median(no_errors)) if no_errors else 0.0,
            "percent_within": {f"{threshold}ms": _percent_within(no_errors, threshold) for threshold in THRESHOLDS_MS},
        },
        "microalign": {
            "mean_absolute_error_ms": float(mean(micro_errors)) if micro_errors else 0.0,
            "median_absolute_error_ms": float(median(micro_errors)) if micro_errors else 0.0,
            "percent_within": {f"{threshold}ms": _percent_within(micro_errors, threshold) for threshold in THRESHOLDS_MS},
        },
        "tracks_improved_by_microalign": int(sum(1 for row in referenced if row.get("result") == "improved")),
        "tracks_worsened_by_microalign": int(sum(1 for row in referenced if row.get("result") == "worsened")),
        "tracks_unchanged": int(sum(1 for row in referenced if row.get("result") == "unchanged")),
        "mean_improvement_ms": float(mean(improvements)) if improvements else 0.0,
        "median_improvement_ms": float(median(improvements)) if improvements else 0.0,
        "average_snap_offset_ms": float(mean(offsets)) if offsets else 0.0,
        "median_snap_offset_ms": float(median(offsets)) if offsets else 0.0,
        "percent_snap_offsets_over_50ms": float(100.0 * sum(1 for off in offsets if off > 50.0) / len(offsets)) if offsets else 0.0,
        "percent_snap_offsets_over_100ms": float(100.0 * sum(1 for off in offsets if off > 100.0) / len(offsets)) if offsets else 0.0,
        "auto_accept_eligible": eligible,
        "high_confidence_regressions": high_regressions[:20],
        "best_microalign_improvements": sorted(
            [
                row
                for row in referenced
                if _float_or_none(row.get("improvement_ms")) is not None
                and float(row.get("improvement_ms", 0.0)) > UNCHANGED_TOLERANCE_MS
            ],
            key=lambda row: float(row["improvement_ms"]),
            reverse=True,
        )[:20],
        "worst_microalign_regressions": sorted(
            [
                row
                for row in referenced
                if _float_or_none(row.get("improvement_ms")) is not None
                and float(row.get("improvement_ms", 0.0)) < -UNCHANGED_TOLERANCE_MS
            ],
            key=lambda row: float(row["improvement_ms"]),
        )[:20],
    }


def _recommendation(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    no_metrics = metrics.get("no_microalign", {}) if isinstance(metrics.get("no_microalign"), Mapping) else {}
    micro_metrics = metrics.get("microalign", {}) if isinstance(metrics.get("microalign"), Mapping) else {}
    no_p10 = float((no_metrics.get("percent_within") or {}).get("10ms", 0.0))
    micro_p10 = float((micro_metrics.get("percent_within") or {}).get("10ms", 0.0))
    no_p25 = float((no_metrics.get("percent_within") or {}).get("25ms", 0.0))
    micro_p25 = float((micro_metrics.get("percent_within") or {}).get("25ms", 0.0))
    no_median = float(no_metrics.get("median_absolute_error_ms", 0.0))
    micro_median = float(micro_metrics.get("median_absolute_error_ms", 0.0))
    high_regressions = list(metrics.get("high_confidence_regressions", []) if isinstance(metrics.get("high_confidence_regressions"), list) else [])
    over_100 = float(metrics.get("percent_snap_offsets_over_100ms", 0.0))
    worst = list(metrics.get("worst_microalign_regressions", []) if isinstance(metrics.get("worst_microalign_regressions"), list) else [])
    bad_worst = [row for row in worst if float(row.get("improvement_ms", 0.0)) < -100.0]

    positive = micro_median < no_median or micro_p10 > no_p10 or micro_p25 > no_p25
    safety = len(high_regressions) == 0 and over_100 <= 10.0 and len(bad_worst) <= 2
    recommend = bool(metrics.get("referenced_tracks", 0)) and positive and safety
    reasons: List[str] = []
    if micro_median < no_median:
        reasons.append(f"median error improved {no_median:.3f}ms -> {micro_median:.3f}ms")
    if micro_p10 > no_p10:
        reasons.append(f"10ms accuracy improved {no_p10:.2f}% -> {micro_p10:.2f}%")
    if micro_p25 > no_p25:
        reasons.append(f"25ms accuracy improved {no_p25:.2f}% -> {micro_p25:.2f}%")
    if high_regressions:
        reasons.append(f"{len(high_regressions)} HIGH-confidence tracks regressed badly")
    if over_100 > 10.0:
        reasons.append(f"{over_100:.2f}% of snap offsets exceeded 100ms")
    if bad_worst:
        reasons.append(f"{len(bad_worst)} regressions exceeded 100ms")
    if not metrics.get("referenced_tracks", 0):
        reasons.append("no correction/golden references were available")
    if not reasons:
        reasons.append("no pass criteria improved")
    return {
        "recommend_microalign_default": bool(recommend),
        "positive_gate_passed": bool(positive),
        "safety_gate_passed": bool(safety),
        "reasons": reasons,
    }


def _csv_row(row: Mapping[str, Any]) -> Dict[str, str]:
    no_mode = row.get("no_microalign") if isinstance(row.get("no_microalign"), Mapping) else {}
    micro_mode = row.get("microalign") if isinstance(row.get("microalign"), Mapping) else {}
    gates = micro_mode.get("auto_accept") if isinstance(micro_mode.get("auto_accept"), Mapping) else {}
    return {
        "track": str(row.get("track", "")),
        "no_microalign_time": _fmt(no_mode.get("time"), 9),
        "microalign_time": _fmt(micro_mode.get("time"), 9),
        "user_or_golden_time": _fmt(row.get("reference_time"), 9),
        "no_microalign_error_ms": _fmt(row.get("no_microalign_error_ms"), 3),
        "microalign_error_ms": _fmt(row.get("microalign_error_ms"), 3),
        "improvement_ms": _fmt(row.get("improvement_ms"), 3),
        "snap_offset_ms": _fmt(micro_mode.get("snap_offset_ms"), 3),
        "micro_confidence": _fmt(micro_mode.get("micro_confidence"), 6),
        "attack_cleanliness": _fmt(micro_mode.get("attack_cleanliness"), 6),
        "zero_crossing_quality": _fmt(micro_mode.get("zero_crossing_quality"), 6),
        "sustained_after_attack": _fmt(micro_mode.get("sustained_after_attack"), 6),
        "confidence_tier": str(micro_mode.get("confidence_tier") or no_mode.get("confidence_tier") or ""),
        "auto_accept_conservative": str(bool((gates.get("conservative") or {}).get("auto_accept"))).lower(),
        "auto_accept_normal": str(bool((gates.get("normal") or {}).get("auto_accept"))).lower(),
        "auto_accept_aggressive": str(bool((gates.get("aggressive") or {}).get("auto_accept"))).lower(),
        "result": str(row.get("result", "")),
        "notes": str(row.get("notes", "")),
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


def _print_summary(report: Mapping[str, Any], json_path: str, csv_path: str) -> None:
    metrics = report.get("metrics", {})
    recommendation = report.get("recommendation", {})
    no_metrics = metrics.get("no_microalign", {}) if isinstance(metrics.get("no_microalign"), Mapping) else {}
    micro_metrics = metrics.get("microalign", {}) if isinstance(metrics.get("microalign"), Mapping) else {}
    print("\nMicroSnap A/B summary")
    print(f"Tracks scanned: {report.get('tracks_scanned', 0)}")
    print(f"Tracks with references: {metrics.get('referenced_tracks', 0)}")
    print(f"Detection failures: no-microalign={report.get('no_microalign_failures', 0)}, microalign={report.get('microalign_failures', 0)}")
    if metrics.get("referenced_tracks", 0):
        print(
            "Median error: "
            f"no-microalign={float(no_metrics.get('median_absolute_error_ms', 0.0)):.3f}ms, "
            f"microalign={float(micro_metrics.get('median_absolute_error_ms', 0.0)):.3f}ms"
        )
        print(
            "10ms / 25ms accuracy: "
            f"no={float((no_metrics.get('percent_within') or {}).get('10ms', 0.0)):.2f}%/"
            f"{float((no_metrics.get('percent_within') or {}).get('25ms', 0.0)):.2f}%, "
            f"micro={float((micro_metrics.get('percent_within') or {}).get('10ms', 0.0)):.2f}%/"
            f"{float((micro_metrics.get('percent_within') or {}).get('25ms', 0.0)):.2f}%"
        )
        print(
            "Track outcomes: "
            f"improved={metrics.get('tracks_improved_by_microalign', 0)}, "
            f"worsened={metrics.get('tracks_worsened_by_microalign', 0)}, "
            f"unchanged={metrics.get('tracks_unchanged', 0)}"
        )
        print(
            "Snap offsets: "
            f"median={float(metrics.get('median_snap_offset_ms', 0.0)):.2f}ms, "
            f">100ms={float(metrics.get('percent_snap_offsets_over_100ms', 0.0)):.2f}%"
        )
        eligible = metrics.get("auto_accept_eligible", {})
        print(
            "Auto-accept eligible: "
            f"conservative={eligible.get('conservative', 0)}, "
            f"normal={eligible.get('normal', 0)}, aggressive={eligible.get('aggressive', 0)}"
        )
    print(f"Recommendation: {'ENABLE MicroSnap default' if recommendation.get('recommend_microalign_default') else 'KEEP optional for now'}")
    for reason in recommendation.get("reasons", []):
        print(f"  - {reason}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare drop detection with and without sample-level MicroSnap alignment.")
    parser.add_argument("summary_csv", help="drop_batch_summary.csv from batch.py")
    parser.add_argument("--template", required=True, help="Ableton .als template path. Only validated; not modified.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl")
    parser.add_argument("--golden")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="models/microalign_eval")
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
    tasks = [
        {"row": row, "bpm": _candidate_json_bpm(row) or _infer_bpm_from_path(str(row.get("filename", "")))}
        for row in rows
    ]

    raw: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            result = _process_track(task)
            raw.append(result)
            print(f"[{index}/{len(tasks)}] compared: {result.get('track', '')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_track = {pool.submit(_process_track, task): str(task["row"].get("filename", "")) for task in tasks}
            for index, future in enumerate(as_completed(future_to_track), start=1):
                track = future_to_track[future]
                try:
                    raw.append(future.result())
                except Exception as exc:
                    raw.append({"track": track, "error": str(exc) or exc.__class__.__name__})
                print(f"[{index}/{len(tasks)}] compared: {track}", flush=True)
        raw.sort(key=lambda row: str(row.get("track", "")).lower())

    details: List[Dict[str, Any]] = []
    for result in raw:
        track = str(result.get("track", ""))
        reference = _reference_for_track(corrections, track) or _reference_for_track(golden, track)
        details.append(_attach_reference(result, reference))

    metrics = _metrics(details)
    recommendation = _recommendation(metrics)
    output_dir = Path(args.output_dir).expanduser()
    json_path = output_dir / OUTPUT_JSON
    csv_path = output_dir / OUTPUT_CSV
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_csv": str(summary_path),
        "template": str(template),
        "corrections": str(Path(args.corrections).expanduser()) if args.corrections else "",
        "golden": str(Path(args.golden).expanduser()) if args.golden else "",
        "tracks_scanned": int(len(rows)),
        "no_microalign_failures": int(sum(1 for row in details if not (row.get("no_microalign") or {}).get("ok"))),
        "microalign_failures": int(sum(1 for row in details if not (row.get("microalign") or {}).get("ok"))),
        "metrics": metrics,
        "recommendation": recommendation,
        "rows": details,
    }
    written_csv = _write_csv(csv_path, details)
    written_json = _write_json(json_path, report)
    _print_summary(report, written_json, written_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
