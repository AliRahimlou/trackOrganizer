#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.boom_profile import boom_proof_front_edge_freshness
from drop_aligner.production_contract import (
    DISALLOWED_PASS_SOURCES,
    DISALLOWED_PASS_SOURCE_PREFIXES,
    unsafe_selected_source as _unsafe_selected_source,
)
from project_config import GENERATED_SET_DIR

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


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_report() -> Path:
    root = GENERATED_SET_DIR / "VisualFirstFresh"
    patterns = (
        "VISUAL_FIRST_FRESH_ALL_TRACKS_*_report.json",
        "VISUAL_FIRST_FRESH_ALL_DRUMS_*_report.json",
    )
    reports = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if "_detector_pass_only_report" not in path.name
        },
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError(
            "No VISUAL_FIRST_FRESH_ALL_TRACKS or legacy "
            f"VISUAL_FIRST_FRESH_ALL_DRUMS report found in {root}"
        )
    return reports[-1]


def _track(row: Mapping[str, Any]) -> Mapping[str, Any]:
    track = row.get("track")
    return track if isinstance(track, Mapping) else {}


def _accepted_als_anchor(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    for source in (row, result):
        for key in ("als_anchor", "visual_first_als_anchor"):
            anchor = source.get(key) if isinstance(source.get(key), Mapping) else {}
            if (
                bool(anchor.get("ok"))
                and bool(anchor.get("accepted"))
                and _finite_float(anchor.get("drop_sec")) is not None
            ):
                return anchor
    return {}


def _als_anchor(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    for source in (row, result):
        for key in ("als_anchor", "visual_first_als_anchor"):
            anchor = source.get(key) if isinstance(source.get(key), Mapping) else {}
            if anchor:
                return anchor
    return {}


def _row_marker(row: Mapping[str, Any]) -> Optional[float]:
    anchor = _accepted_als_anchor(row)
    anchor_marker = _finite_float(anchor.get("drop_sec"))
    if anchor_marker is not None:
        return float(anchor_marker)

    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    for value in (row.get("marker"), result.get("marker")):
        marker = _finite_float(value)
        if marker is not None:
            return float(marker)
    for source in (row, result):
        selected = source.get("selected_candidate") if isinstance(source.get("selected_candidate"), Mapping) else {}
        for key in ("timestamp", "microaligned_time", "snapped_sec", "time_sec"):
            marker = _finite_float(selected.get(key))
            if marker is not None:
                return float(marker)
    return None


def _visual_marker(row: Mapping[str, Any], fallback: Optional[float]) -> Optional[float]:
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    anchor = _accepted_als_anchor(row)
    for value in (
        row.get("visual_marker"),
        row.get("visual_marker_sec"),
        result.get("visual_marker"),
        result.get("visual_marker_sec"),
        anchor.get("visual_marker_sec"),
        result.get("marker"),
        row.get("marker"),
    ):
        marker = _finite_float(value)
        if marker is not None:
            return float(marker)
    return float(fallback) if fallback is not None else None


def _library_sort_key(row: Mapping[str, Any]) -> Tuple[float, str, str]:
    track = _track(row)
    return (
        _safe_float(track.get("bpm"), 999.0),
        str(track.get("key") or ""),
        str(track.get("folder") or row.get("drums_path") or "").lower(),
    )


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


def _gui_mask_proof(row: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = row.get("gui_mask_proof")
    return proof if isinstance(proof, Mapping) else {}


def _gui_mask_passes(row: Mapping[str, Any]) -> bool:
    proof = _gui_mask_proof(row)
    return (
        bool(proof.get("passes"))
        and proof.get("marker_signal_present") is True
        and proof.get("marker_relevant_mask") is True
    )


def _boom_profile_score(row: Mapping[str, Any]) -> str:
    proof = _boom_proof(row)
    nearest = proof.get("nearest_profile") if isinstance(proof.get("nearest_profile"), Mapping) else {}
    try:
        score = float(nearest.get("profile_score"))
    except (TypeError, ValueError):
        return ""
    return f"{score:.6f}" if math.isfinite(score) else ""


def _materialize_review_candidate_json(
    row: Mapping[str, Any],
    *,
    candidate_dir: Path,
    marker: Optional[float],
) -> Path:
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    track = _track(row)
    selected_source = (
        result.get("selected_candidate")
        if isinstance(result.get("selected_candidate"), Mapping)
        else row.get("selected_candidate")
        if isinstance(row.get("selected_candidate"), Mapping)
        else {}
    )
    visual_selected = dict(selected_source)
    selected = dict(visual_selected)
    visual_marker = _visual_marker(row, marker)
    selected.update({"selected": bool(marker is not None), "review_only": True})
    if marker is not None:
        selected.update(
            {
                "timestamp": float(marker),
                "snapped_sec": float(marker),
                "microaligned_time": float(marker),
            }
        )
    if visual_marker is not None:
        selected["visual_marker_sec"] = float(visual_marker)
    selected_by = str(
        row.get("selected_by")
        or selected.get("selected_by")
        or result.get("source")
        or "visual_first_review_failure"
    )
    selected.setdefault("selected_by", selected_by)
    visual_audit = result.get("visual_audit") if isinstance(result.get("visual_audit"), Mapping) else {}
    if not visual_audit:
        visual_audit = {
            "status": str(row.get("audit_status") or "review"),
            "flag_codes": [str(flag) for flag in row.get("audit_flags") or [] if str(flag)],
        }
    boom_proof = (
        result.get("boom_proof")
        if isinstance(result.get("boom_proof"), Mapping)
        else row.get("boom_proof")
        if isinstance(row.get("boom_proof"), Mapping)
        else {}
    )
    gui_mask_proof = (
        result.get("gui_mask_proof")
        if isinstance(result.get("gui_mask_proof"), Mapping)
        else row.get("gui_mask_proof")
        if isinstance(row.get("gui_mask_proof"), Mapping)
        else {}
    )
    anchor = dict(_als_anchor(row))
    raw_candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
    raw_boom_candidates = result.get("boom_candidates") if isinstance(result.get("boom_candidates"), list) else []
    feature_map = result.get("feature_map") if isinstance(result.get("feature_map"), Mapping) else {}
    audio_path = str(row.get("drums_path") or result.get("audio_path") or "")
    payload = {
        "source": "visual_first_full_review_sidecar",
        "reviewed_from": "visual_first_failure_report",
        "review_only": True,
        "source_status": str(row.get("status") or "failure"),
        "source_error": str(row.get("error") or ""),
        "audio_path": audio_path,
        "track_folder": str(track.get("folder") or ""),
        "bpm": _safe_float(track.get("bpm"), 0.0),
        "key": str(track.get("key") or ""),
        "energy": track.get("energy"),
        "final_ai_pick": float(marker) if marker is not None else None,
        "drop_sec": float(marker) if marker is not None else None,
        "visual_marker_sec": float(visual_marker) if visual_marker is not None else None,
        "visual_selected_candidate": visual_selected,
        "als_anchor": anchor,
        "marker_contract": {
            "als_marker": "first_bounded_drum_body_attack",
            "visual_marker": "approved_drop_section_on_independent_one",
            "boom_body_crest": "post_attack_diagnostic_only",
        },
        "selected_by": selected_by,
        "selected_candidate": selected,
        "top_10_candidates": [dict(item) for item in raw_candidates if isinstance(item, Mapping)][:10],
        "visual_audit": dict(visual_audit),
        "boom_proof": dict(boom_proof),
        "gui_mask_proof": dict(gui_mask_proof),
        "boom_candidates": [dict(item) for item in raw_boom_candidates if isinstance(item, Mapping)][:10],
        "feature_map": dict(feature_map),
        "output_als": str(row.get("output_als") or ""),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    marker_key = f"{float(marker):.9f}" if marker is not None else "unplaced"
    digest = hashlib.sha1(
        f"{audio_path}|{marker_key}|{row.get('status') or ''}|{row.get('error') or ''}".encode("utf-8")
    ).hexdigest()[:16]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    path = candidate_dir / f"review_{digest}_drop_candidates.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    return path


def _reviewable_failure_row(row: Mapping[str, Any], *, candidate_dir: Path) -> Optional[Dict[str, Any]]:
    drums_path = str(row.get("drums_path") or "").strip()
    marker = _row_marker(row)
    if not drums_path:
        return None

    review_row = dict(row)
    if marker is not None:
        review_row["marker"] = float(marker)
    review_row["_review_only"] = True
    review_row["_source_status"] = str(row.get("status") or "failure")
    if not str(review_row.get("output_als") or "").strip():
        output_digest = hashlib.sha1(drums_path.encode("utf-8")).hexdigest()[:16]
        review_als_dir = candidate_dir.with_name(
            candidate_dir.name.removesuffix("_review_candidates") + "_review_als"
        )
        review_row["output_als"] = str(review_als_dir / f"review_{output_digest}_DROP_ALIGNED.als")
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    for key in ("boom_proof", "gui_mask_proof", "als_anchor"):
        if not isinstance(review_row.get(key), Mapping) and isinstance(result.get(key), Mapping):
            review_row[key] = dict(result[key])
    if not str(review_row.get("selected_by") or ""):
        selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
        review_row["selected_by"] = str(selected.get("selected_by") or result.get("source") or "")
    if not str(review_row.get("audit_status") or ""):
        audit = result.get("visual_audit") if isinstance(result.get("visual_audit"), Mapping) else {}
        review_row["audit_status"] = str(audit.get("status") or "review")
        review_row["audit_flags"] = [str(flag) for flag in audit.get("flag_codes") or [] if str(flag)]

    raw_candidates_path = str(row.get("candidates_json") or "").strip()
    candidates_path = Path(raw_candidates_path).expanduser() if raw_candidates_path else None
    if candidates_path is None or not candidates_path.is_file():
        candidates_path = _materialize_review_candidate_json(
            review_row,
            candidate_dir=candidate_dir,
            marker=float(marker) if marker is not None else None,
        )
    review_row["candidates_json"] = str(candidates_path)
    return review_row


def row_passes_production_gate(row: Mapping[str, Any]) -> bool:
    return (
        not bool(row.get("_review_only"))
        and _status(row) == "pass"
        and not _flags(row)
        and not _unsafe_selected_source(_selected_by(row))
        and _boom_passes(row)
        and _gui_mask_passes(row)
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
    if not _gui_mask_passes(row):
        gui_proof = _gui_mask_proof(row)
        gui_reasons = [str(reason) for reason in gui_proof.get("reasons") or [] if str(reason)]
        if not gui_proof:
            reasons.append("gui_mask=missing")
        elif not bool(gui_proof.get("passes")):
            reasons.append("gui_mask=hold" + (":" + ";".join(gui_reasons) if gui_reasons else ":missing"))
        elif gui_proof.get("marker_signal_present") is not True:
            reasons.append("gui_mask=blank_marker:no_visible_signal")
        else:
            reasons.append("gui_mask=stale_missing_marker_relevant_mask")
    return " | ".join(reasons) or "held by production gate"


def _review_reason(row: Mapping[str, Any]) -> str:
    reason = _reason(row)
    error = str(row.get("error") or "").strip()
    if not error:
        return reason
    if reason == "held by production gate":
        return error
    if error in reason:
        return reason
    return f"{reason} | source_error={error}"


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
    marker = _row_marker(row)
    marker_text = f"{float(marker):.6f}" if marker is not None else ""
    review_only = bool(row.get("_review_only"))
    verification = row.get("verification") if isinstance(row.get("verification"), Mapping) else {}
    verification_errors = [str(err) for err in verification.get("errors") or [] if str(err)]
    if review_only:
        als_valid = ""
        als_validation_error = "review_only:no_verified_als"
        if verification_errors:
            als_validation_error += ";" + ";".join(verification_errors)
    else:
        als_valid = "true" if verification.get("valid") else "false"
        als_validation_error = ";".join(verification_errors)
    return {
        "filename": str(row.get("drums_path") or ""),
        "detected_drop_time": marker_text,
        "confidence": confidence,
        "confidence_tier": tier,
        "selected_by": _selected_by(row),
        "output_als": str(row.get("output_als") or ""),
        "candidates_json": str(row.get("candidates_json") or ""),
        "debug_png": "",
        "als_valid": als_valid,
        "als_validation_error": als_validation_error,
        "status": "partial" if review_only else str(row.get("status") or "processed"),
        "error": str(row.get("error") or ""),
        "micro_confidence": "",
        "snap_offset_ms": "",
        "microaligned_time": marker_text,
        "visual_audit_status": _status(row),
        "visual_audit_flags": ";".join(_flags(row)),
        "boom_proof_status": "pass" if _boom_passes(row) else "hold",
        "boom_proof_reasons": ";".join(_boom_reasons(row)),
        "boom_profile_score": _boom_profile_score(row),
        "visual_first_batch_auto_time": "",
        "visual_first_batch_auto_source": (
            "visual_first_full_review_sidecar" if review_only else "visual_first_fresh_library_set"
        ),
    }


def _queue_row(rank: int, row: Mapping[str, Any], *, reason: Optional[str] = None) -> Dict[str, Any]:
    track = _track(row)
    flags = _flags(row)
    priority = _priority(row)
    marker = _row_marker(row)
    marker_text = f"{float(marker):.6f}" if marker is not None else ""
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
        "marker_sec": marker_text,
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
    stem = prefix or report_path.name.removesuffix("_report.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    review_candidate_dir = out_dir / f"{stem}_review_candidates"

    processed_drums_paths = {str(row.get("drums_path") or "").strip() for row in processed}
    reviewable_failures: List[Dict[str, Any]] = []
    for row in failures:
        review_row = _reviewable_failure_row(row, candidate_dir=review_candidate_dir)
        if review_row is None:
            continue
        if str(review_row.get("drums_path") or "").strip() in processed_drums_paths:
            continue
        reviewable_failures.append(review_row)

    pass_rows = [row for row in processed if row_passes_production_gate(row)]
    held_rows = [row for row in processed if not row_passes_production_gate(row)]
    needs_review_rows = held_rows + reviewable_failures
    full_review_rows = processed + reviewable_failures
    held_rows.sort(key=_priority)
    needs_review_rows.sort(key=_priority)
    full_review_rows.sort(key=_library_sort_key)
    pass_rows.sort(key=_library_sort_key)

    fresh_summary_csv = out_dir / f"{stem}_fresh_detector_summary.csv"
    full_review_summary_csv = out_dir / f"{stem}_full_library_review_summary.csv"
    pass_summary_csv = out_dir / f"{stem}_detector_pass_only_summary.csv"
    full_queue_csv = out_dir / f"{stem}_full_library_review_queue.csv"
    review_queue_csv = out_dir / f"{stem}_needs_human_review_queue.csv"
    review_detail_csv = out_dir / f"{stem}_needs_human_review_details.csv"
    pass_report_path = out_dir / f"{stem}_detector_pass_only_report.json"
    gate_summary_path = out_dir / f"{stem}_production_gate_summary.json"

    _write_csv(fresh_summary_csv, SUMMARY_FIELDS, (_summary_row(row) for row in processed))
    _write_csv(full_review_summary_csv, SUMMARY_FIELDS, (_summary_row(row) for row in full_review_rows))
    _write_csv(pass_summary_csv, SUMMARY_FIELDS, (_summary_row(row) for row in pass_rows))
    full_queue_rows = [
        _queue_row(
            idx,
            row,
            reason=_review_reason(row) if not row_passes_production_gate(row) else "full-library confirmation pass",
        )
        for idx, row in enumerate(full_review_rows, start=1)
    ]
    _write_csv(full_queue_csv, QUEUE_FIELDS, full_queue_rows)
    queue_rows = [
        _queue_row(idx, row, reason=_review_reason(row))
        for idx, row in enumerate(needs_review_rows, start=1)
    ]
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
        "reviewable_failure_count": len(reviewable_failures),
        "non_reviewable_failure_count": len(failures) - len(reviewable_failures),
        "full_review_count": len(full_review_rows),
        "needs_human_review_count": len(needs_review_rows),
        "held_status_counts": dict(Counter(_status(row) or "missing" for row in needs_review_rows)),
        "held_flag_counts": dict(Counter(flag for row in needs_review_rows for flag in _flags(row))),
        "held_boom_reason_counts": dict(Counter(reason for row in needs_review_rows for reason in _boom_reasons(row))),
        "review_queue_csv": str(review_queue_csv),
        "full_library_review_queue_csv": str(full_queue_csv),
        "fresh_summary_csv": str(fresh_summary_csv),
        "full_review_summary_csv": str(full_review_summary_csv),
        "pass_summary_csv": str(pass_summary_csv),
    }
    pass_report_path.write_text(json.dumps(pass_report, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")

    gate_summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(report_path),
        "fresh_summary_csv": str(fresh_summary_csv),
        "full_review_summary_csv": str(full_review_summary_csv),
        "pass_summary_csv": str(pass_summary_csv),
        "full_library_review_queue_csv": str(full_queue_csv),
        "review_queue_csv": str(review_queue_csv),
        "review_detail_csv": str(review_detail_csv),
        "detector_pass_report": str(pass_report_path),
        "processed_count": len(processed),
        "detector_pass_count": len(pass_rows),
        "full_review_count": len(full_review_rows),
        "needs_human_review_count": len(needs_review_rows),
        "source_failure_count": len(failures),
        "reviewable_failure_count": len(reviewable_failures),
        "non_reviewable_failure_count": len(failures) - len(reviewable_failures),
        "held_status_counts": dict(Counter(_status(row) or "missing" for row in needs_review_rows)),
        "held_flag_counts": dict(Counter(flag for row in needs_review_rows for flag in _flags(row))),
        "held_boom_reason_counts": dict(Counter(reason for row in needs_review_rows for reason in _boom_reasons(row))),
        "selected_by_counts": dict(Counter(_selected_by(row) for row in processed)),
        "full_review_selected_by_counts": dict(Counter(_selected_by(row) for row in full_review_rows)),
    }
    gate_summary_path.write_text(json.dumps(gate_summary, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    return gate_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a fresh visual-first library report into detector-pass rows and human-review queue rows."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="",
        help="Fresh visual-first all-tracks report JSON. Defaults to latest (legacy all-drums names are supported).",
    )
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
