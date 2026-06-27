#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.boom_profile import boom_proof_front_edge_freshness
from project_config import GENERATED_SET_DIR


DISALLOWED_PASS_SOURCES = {
    "historical_human_marker",
    "historical_review_memory",
    "manual_review_marker",
    "review_auto_place",
    "saved_closest_to_review_pick",
    "visual_drop_v2",
    "visual_drop_v2_candidate",
    "visual_first_rms_body_fallback",
    "web_accept_blue_marker",
    "web_save_placed_marker",
}
DISALLOWED_PASS_SOURCE_PREFIXES = ("historical_", "saved_")


def _unsafe_selected_source(selected_by: str) -> bool:
    source = str(selected_by or "").strip()
    return source in DISALLOWED_PASS_SOURCES or any(
        source.startswith(prefix) for prefix in DISALLOWED_PASS_SOURCE_PREFIXES
    )

FLAG_PRIORITY = {
    "off_one_bpm_grid": 0,
    "later_definitive_drop_available": 1,
    "earlier_definitive_drop_available": 1,
    "earlier_real_drop_available_before_late_marker": 2,
    "earlier_phrase_body_edge_available": 2,
    "intro_before_stronger_drop": 3,
    "late_body_after_section_entry": 4,
    "late_dense_continuation_without_reset": 5,
    "ambiguous_visual_drop_evidence": 6,
    "missing_visual_waveform_components": 7,
    "rms_body_fallback": 8,
}

STATUS_PRIORITY = {
    "replace": 0,
    "fallback": 1,
    "review": 2,
    "": 3,
    "pass": 99,
}

SUMMARY_FIELDS = [
    "filename",
    "detected_drop_time",
    "confidence",
    "confidence_tier",
    "selected_by",
    "output_als",
    "candidates_json",
    "debug_png",
    "als_valid",
    "als_validation_error",
    "status",
    "error",
    "micro_confidence",
    "snap_offset_ms",
    "microaligned_time",
    "visual_audit_status",
    "visual_audit_flags",
    "boom_proof_status",
    "boom_proof_reasons",
    "boom_profile_score",
    "visual_first_batch_auto_time",
    "visual_first_batch_auto_source",
]

QUEUE_FIELDS = [
    "rank",
    "track",
    "filename",
    "reason_to_review",
    "priority",
    "bpm",
    "key",
    "energy",
    "track_folder",
    "marker_sec",
    "audit_status",
    "audit_flags",
    "boom_proof_status",
    "boom_proof_reasons",
    "selected_by",
    "candidates_json",
    "output_als",
    "drums_path",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_report() -> Path:
    root = GENERATED_SET_DIR / "VisualFirstFresh"
    reports = sorted(
        (
            path
            for path in root.glob("VISUAL_FIRST_FRESH_ALL_DRUMS_*_report.json")
            if "_detector_pass_only_report" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError(f"No VISUAL_FIRST_FRESH_ALL_DRUMS report found in {root}")
    return reports[-1]


def _track(row: Mapping[str, Any]) -> Mapping[str, Any]:
    track = row.get("track")
    return track if isinstance(track, Mapping) else {}


def _flags(row: Mapping[str, Any]) -> List[str]:
    return [str(flag) for flag in row.get("audit_flags") or [] if str(flag)]


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("audit_status") or "").strip().lower()


def _selected_by(row: Mapping[str, Any]) -> str:
    return str(row.get("selected_by") or "").strip()


def _boom_proof(row: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = row.get("boom_proof")
    return proof if isinstance(proof, Mapping) else {}


def _boom_passes(row: Mapping[str, Any]) -> bool:
    proof = _boom_proof(row)
    return bool(proof.get("passes")) and bool(boom_proof_front_edge_freshness(proof).get("fresh"))


def _boom_reasons(row: Mapping[str, Any]) -> List[str]:
    return [str(reason) for reason in _boom_proof(row).get("reasons") or [] if str(reason)]


def _boom_profile_score(row: Mapping[str, Any]) -> str:
    proof = _boom_proof(row)
    nearest = proof.get("nearest_profile") if isinstance(proof.get("nearest_profile"), Mapping) else {}
    try:
        score = float(nearest.get("profile_score"))
    except (TypeError, ValueError):
        return ""
    return f"{score:.6f}" if math.isfinite(score) else ""


def row_passes_production_gate(row: Mapping[str, Any]) -> bool:
    return (
        _status(row) == "pass"
        and not _flags(row)
        and not _unsafe_selected_source(_selected_by(row))
        and _boom_passes(row)
    )


def _reason(row: Mapping[str, Any]) -> str:
    status = _status(row) or "missing audit status"
    flags = _flags(row)
    selected_by = _selected_by(row)
    reasons = []
    if status != "pass":
        reasons.append(f"audit_status={status}")
    if flags:
        reasons.append("flags=" + ";".join(flags))
    if _unsafe_selected_source(selected_by):
        reasons.append(f"unsafe_source={selected_by}")
    if not _boom_passes(row):
        proof = _boom_proof(row)
        if bool(proof.get("passes")):
            freshness = boom_proof_front_edge_freshness(proof)
            reasons.append(f"boom_proof=stale_front_edge:{freshness.get('reason') or 'unknown'}")
        else:
            boom_reasons = _boom_reasons(row)
            reasons.append("boom_proof=hold" + (":" + ";".join(boom_reasons) if boom_reasons else ":missing"))
    return " | ".join(reasons) or "held by production gate"


def _priority(row: Mapping[str, Any]) -> Tuple[int, int, int, str]:
    flags = _flags(row)
    flag_priority = min((FLAG_PRIORITY.get(flag, 50) for flag in flags), default=50)
    status_priority = STATUS_PRIORITY.get(_status(row), 50)
    track = _track(row)
    bpm = int(_safe_float(track.get("bpm"), 999))
    folder = str(track.get("folder") or "")
    return (status_priority, flag_priority, bpm, folder.lower())


def _confidence_for(row: Mapping[str, Any]) -> Tuple[str, str]:
    if row_passes_production_gate(row):
        return "0.990000", "HIGH"
    status = _status(row)
    if status == "replace":
        return "0.050000", "LOW"
    if status == "review":
        return "0.250000", "LOW"
    return "0.000000", "LOW"


def _summary_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    confidence, tier = _confidence_for(row)
    marker = _safe_float(row.get("marker"))
    return {
        "filename": str(row.get("drums_path") or ""),
        "detected_drop_time": f"{marker:.6f}",
        "confidence": confidence,
        "confidence_tier": tier,
        "selected_by": _selected_by(row),
        "output_als": str(row.get("output_als") or ""),
        "candidates_json": str(row.get("candidates_json") or ""),
        "debug_png": "",
        "als_valid": "true" if (row.get("verification") or {}).get("valid") else "false",
        "als_validation_error": ";".join(str(err) for err in (row.get("verification") or {}).get("errors") or []),
        "status": str(row.get("status") or "processed"),
        "error": "",
        "micro_confidence": "",
        "snap_offset_ms": "",
        "microaligned_time": f"{marker:.6f}",
        "visual_audit_status": _status(row),
        "visual_audit_flags": ";".join(_flags(row)),
        "boom_proof_status": "pass" if _boom_passes(row) else "hold",
        "boom_proof_reasons": ";".join(_boom_reasons(row)),
        "boom_profile_score": _boom_profile_score(row),
        "visual_first_batch_auto_time": "",
        "visual_first_batch_auto_source": "visual_first_fresh_library_set",
    }


def _queue_row(rank: int, row: Mapping[str, Any], *, reason: Optional[str] = None) -> Dict[str, Any]:
    track = _track(row)
    flags = _flags(row)
    priority = _priority(row)
    return {
        "rank": rank,
        "track": str(row.get("drums_path") or ""),
        "filename": str(row.get("drums_path") or ""),
        "reason_to_review": reason or _reason(row),
        "priority": ".".join(str(part) for part in priority[:3]),
        "bpm": track.get("bpm", ""),
        "key": track.get("key", ""),
        "energy": track.get("energy", ""),
        "track_folder": track.get("folder", ""),
        "marker_sec": f"{_safe_float(row.get('marker')):.6f}",
        "audit_status": _status(row),
        "audit_flags": ";".join(flags),
        "boom_proof_status": "pass" if _boom_passes(row) else "hold",
        "boom_proof_reasons": ";".join(_boom_reasons(row)),
        "selected_by": _selected_by(row),
        "candidates_json": str(row.get("candidates_json") or ""),
        "output_als": str(row.get("output_als") or ""),
        "drums_path": str(row.get("drums_path") or ""),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_gate_artifacts(report_path: Path, out_dir: Path, prefix: Optional[str] = None) -> Dict[str, Any]:
    report = _load_json(report_path)
    processed = [dict(row) for row in report.get("processed_rows") or [] if isinstance(row, Mapping)]
    failures = [dict(row) for row in report.get("failure_rows") or [] if isinstance(row, Mapping)]
    pass_rows = [row for row in processed if row_passes_production_gate(row)]
    held_rows = [row for row in processed if not row_passes_production_gate(row)]
    held_rows.sort(key=_priority)
    pass_rows.sort(key=lambda row: (_safe_float(_track(row).get("bpm"), 999), str(_track(row).get("key") or ""), str(_track(row).get("folder") or "").lower()))

    stem = prefix or report_path.name.removesuffix("_report.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_summary_csv = out_dir / f"{stem}_fresh_detector_summary.csv"
    pass_summary_csv = out_dir / f"{stem}_detector_pass_only_summary.csv"
    full_queue_csv = out_dir / f"{stem}_full_library_review_queue.csv"
    review_queue_csv = out_dir / f"{stem}_needs_human_review_queue.csv"
    review_detail_csv = out_dir / f"{stem}_needs_human_review_details.csv"
    pass_report_path = out_dir / f"{stem}_detector_pass_only_report.json"
    gate_summary_path = out_dir / f"{stem}_production_gate_summary.json"

    _write_csv(fresh_summary_csv, SUMMARY_FIELDS, (_summary_row(row) for row in processed))
    _write_csv(pass_summary_csv, SUMMARY_FIELDS, (_summary_row(row) for row in pass_rows))
    full_queue_rows = [
        _queue_row(
            idx,
            row,
            reason=_reason(row) if not row_passes_production_gate(row) else "full-library confirmation pass",
        )
        for idx, row in enumerate(processed, start=1)
    ]
    _write_csv(full_queue_csv, QUEUE_FIELDS, full_queue_rows)
    queue_rows = [_queue_row(idx, row) for idx, row in enumerate(held_rows, start=1)]
    _write_csv(review_queue_csv, QUEUE_FIELDS, queue_rows)
    _write_csv(review_detail_csv, QUEUE_FIELDS, queue_rows)

    pass_report = dict(report)
    pass_report["source_report"] = str(report_path)
    pass_report["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    pass_report["processed_rows"] = pass_rows
    pass_report["failure_rows"] = []
    pass_report["processed_count"] = len(pass_rows)
    pass_report["failure_count"] = 0
    pass_report["production_gate"] = {
        "mode": "detector_pass_only",
        "source_processed_count": len(processed),
        "passed_count": len(pass_rows),
        "held_count": len(held_rows),
        "source_failure_count": len(failures),
        "held_status_counts": dict(Counter(_status(row) or "missing" for row in held_rows)),
        "held_flag_counts": dict(Counter(flag for row in held_rows for flag in _flags(row))),
        "held_boom_reason_counts": dict(Counter(reason for row in held_rows for reason in _boom_reasons(row))),
        "review_queue_csv": str(review_queue_csv),
        "full_library_review_queue_csv": str(full_queue_csv),
        "fresh_summary_csv": str(fresh_summary_csv),
        "pass_summary_csv": str(pass_summary_csv),
    }
    pass_report_path.write_text(json.dumps(pass_report, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")

    gate_summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(report_path),
        "fresh_summary_csv": str(fresh_summary_csv),
        "pass_summary_csv": str(pass_summary_csv),
        "full_library_review_queue_csv": str(full_queue_csv),
        "review_queue_csv": str(review_queue_csv),
        "review_detail_csv": str(review_detail_csv),
        "detector_pass_report": str(pass_report_path),
        "processed_count": len(processed),
        "detector_pass_count": len(pass_rows),
        "needs_human_review_count": len(held_rows),
        "source_failure_count": len(failures),
        "held_status_counts": dict(Counter(_status(row) or "missing" for row in held_rows)),
        "held_flag_counts": dict(Counter(flag for row in held_rows for flag in _flags(row))),
        "held_boom_reason_counts": dict(Counter(reason for row in held_rows for reason in _boom_reasons(row))),
        "selected_by_counts": dict(Counter(_selected_by(row) for row in processed)),
    }
    gate_summary_path.write_text(json.dumps(gate_summary, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    return gate_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a fresh visual-first library report into detector-pass rows and human-review queue rows."
    )
    parser.add_argument("report", nargs="?", default="", help="Fresh visual-first all-drums report JSON. Defaults to latest.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults beside the input report.")
    parser.add_argument("--prefix", default="", help="Output filename prefix. Defaults to input report stem.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = Path(args.report).expanduser().resolve() if args.report else _latest_report().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else report.parent
    summary = build_gate_artifacts(report, out_dir, prefix=args.prefix or None)
    print(json.dumps(summary, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
