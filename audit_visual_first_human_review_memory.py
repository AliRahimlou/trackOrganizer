from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.historical_markers import exact_key_for_path, slug_for_path
from drop_aligner.boom_profile import marker_boom_proof, boom_proof_front_edge_freshness
from drop_aligner.waveform import accept_gui_boom_mask_with_front_edge_proof, gui_boom_mask_proof


DEFAULT_REPORT = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_contract_hardened_v5_freshproof_report.json"
)
DEFAULT_CORRECTION_LOG = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/drop_corrections.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/human_review_audit"
)
DEFAULT_CACHE_DIR = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/.waveform_cache"
)

MANUAL_REVIEW_SOURCES = {
    "web_manual_marker",
    "web_candidate_pick",
    "web_review",
    "web_ai_refined_accept",
}
BLUE_APPROVAL_SOURCES = {
    "web_accept_blue_marker",
    "web_accept_grid_marker",
    "web_accept_knee_marker",
    "web_accept_attack_marker",
    "web_accept_asd_marker",
    "web_accept_micro_marker",
}
ALL_REVIEW_SOURCES = MANUAL_REVIEW_SOURCES | BLUE_APPROVAL_SOURCES


@dataclass(frozen=True)
class ReviewMarker:
    track: str
    user_pick: float
    reviewed_from: str
    timestamp: str
    selected_by: str
    source_path: str
    line_number: int

    @property
    def strength(self) -> str:
        source = self.reviewed_from.strip().lower()
        if source in MANUAL_REVIEW_SOURCES:
            return "manual"
        if source in BLUE_APPROVAL_SOURCES:
            return "blue_approval"
        return "unknown"


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _path_key(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve()).lower()
    except OSError:
        return str(Path(text).expanduser()).lower()


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.expanduser().open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                yield line_number, payload


def _track_from_review(row: Mapping[str, Any]) -> str:
    for key in ("track", "filename", "audio_path", "drums_path"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _marker_from_review(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("user_pick", "corrected_drop_time", "final_ai_pick", "drop_sec"):
        value = row.get(key)
        if value is None:
            continue
        marker = _finite_float(value)
        if math.isfinite(marker) and marker > 0.0:
            return float(marker)
    return None


def load_review_markers(correction_logs: Sequence[Path]) -> List[ReviewMarker]:
    markers: List[ReviewMarker] = []
    for log_path in correction_logs:
        path = Path(log_path).expanduser()
        if not path.exists():
            continue
        for line_number, row in _iter_jsonl(path):
            reviewed_from = str(row.get("reviewed_from") or "").strip().lower()
            if reviewed_from not in ALL_REVIEW_SOURCES:
                continue
            track = _track_from_review(row)
            if not track:
                continue
            marker = _marker_from_review(row)
            if marker is None:
                continue
            markers.append(
                ReviewMarker(
                    track=track,
                    user_pick=float(marker),
                    reviewed_from=reviewed_from,
                    timestamp=str(row.get("timestamp") or ""),
                    selected_by=str(row.get("selected_by") or ""),
                    source_path=str(path),
                    line_number=int(line_number),
                )
            )
    return markers


def _processed_rows(report_path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(report_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Report is not a JSON object: {report_path}")
    rows = payload.get("processed_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Report has no processed_rows list: {report_path}")
    return [row for row in rows if isinstance(row, Mapping)]


def _review_indexes(markers: Sequence[ReviewMarker]) -> tuple[Dict[str, ReviewMarker], Dict[str, ReviewMarker], Dict[str, ReviewMarker]]:
    exact: Dict[str, ReviewMarker] = {}
    stem: Dict[str, ReviewMarker] = {}
    slug: Dict[str, ReviewMarker] = {}
    for marker in markers:
        keys = (
            (_path_key(marker.track), exact),
            (exact_key_for_path(marker.track), stem),
            (slug_for_path(marker.track), slug),
        )
        for key, index in keys:
            if key:
                index[key] = marker
    return exact, stem, slug


def _find_review_for_row(
    row: Mapping[str, Any],
    exact: Mapping[str, ReviewMarker],
    stem: Mapping[str, ReviewMarker],
    slug: Mapping[str, ReviewMarker],
) -> Optional[ReviewMarker]:
    candidates = [
        row.get("drums_path"),
        row.get("track"),
        (row.get("track") or {}).get("src") if isinstance(row.get("track"), Mapping) else None,
    ]
    for candidate in candidates:
        key = _path_key(candidate)
        if key and key in exact:
            return exact[key]
    for candidate in candidates:
        key = exact_key_for_path(candidate)
        if key and key in stem:
            return stem[key]
    for candidate in candidates:
        key = slug_for_path(candidate)
        if key and key in slug:
            return slug[key]
    return None


def build_review_indexes(
    markers: Sequence[ReviewMarker],
) -> tuple[Dict[str, ReviewMarker], Dict[str, ReviewMarker], Dict[str, ReviewMarker]]:
    return _review_indexes(markers)


def find_review_for_row(
    row: Mapping[str, Any],
    exact: Mapping[str, ReviewMarker],
    stem: Mapping[str, ReviewMarker],
    slug: Mapping[str, ReviewMarker],
) -> Optional[ReviewMarker]:
    return _find_review_for_row(row, exact, stem, slug)


def review_marker_gate(
    row: Mapping[str, Any],
    marker: float,
    *,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    return _review_marker_gate(row, marker, cache_dir=cache_dir)


def audit_human_review_memory(
    report_path: Path,
    *,
    correction_logs: Sequence[Path],
    out_dir: Path,
    cache_dir: Optional[Path] = None,
    manual_tolerance_sec: float = 0.050,
    blue_tolerance_sec: float = 0.050,
) -> Dict[str, Any]:
    rows = _processed_rows(report_path)
    markers = load_review_markers(correction_logs)
    exact, stem, slug = _review_indexes(markers)
    out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    audit_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        review = _find_review_for_row(row, exact, stem, slug)
        if review is None:
            continue
        marker = _finite_float(row.get("marker"))
        if not math.isfinite(marker):
            continue
        delta = float(marker) - float(review.user_pick)
        abs_delta = abs(delta)
        strength = review.strength
        tolerance = float(manual_tolerance_sec if strength == "manual" else blue_tolerance_sec)
        if abs_delta <= tolerance:
            review_gate: Dict[str, Any] = {}
            review_gate_passes = None
            status = "pass"
        elif strength == "manual":
            review_gate = _review_marker_gate(row, review.user_pick, cache_dir=cache_dir)
            review_gate_passes = bool(review_gate.get("passes"))
            if review_gate_passes:
                status = "validated_hard_mismatch"
            else:
                status = "stale_manual_mismatch"
        else:
            review_gate = {}
            review_gate_passes = None
            status = "advisory_mismatch"
        audit_rows.append(
            {
                "index": int(index),
                "status": status,
                "strength": strength,
                "delta_sec": float(delta),
                "abs_delta_sec": float(abs_delta),
                "freshproof_marker": float(marker),
                "review_marker": float(review.user_pick),
                "review_marker_gate_passes": "" if review_gate_passes is None else bool(review_gate_passes),
                "review_marker_gate_reasons": ";".join(str(reason) for reason in review_gate.get("reasons", []) if str(reason)) if review_gate else "",
                "review_marker_boom_passes": "" if not review_gate else bool(review_gate.get("boom_passes")),
                "review_marker_gui_passes": "" if not review_gate else bool(review_gate.get("gui_passes")),
                "reviewed_from": review.reviewed_from,
                "review_timestamp": review.timestamp,
                "review_selected_by": review.selected_by,
                "selected_by": str(row.get("selected_by") or ""),
                "track": str((row.get("track") or {}).get("folder") if isinstance(row.get("track"), Mapping) else row.get("track") or ""),
                "drums_path": str(row.get("drums_path") or ""),
                "review_track": review.track,
                "review_source_path": review.source_path,
                "review_line_number": int(review.line_number),
            }
        )
    validated_hard = [row for row in audit_rows if row["status"] == "validated_hard_mismatch"]
    stale_manual = [row for row in audit_rows if row["status"] == "stale_manual_mismatch"]
    advisory = [row for row in audit_rows if row["status"] == "advisory_mismatch"]
    matched = [row for row in audit_rows if row["status"] == "pass"]
    fieldnames = [
        "index",
        "status",
        "strength",
        "delta_sec",
        "abs_delta_sec",
        "freshproof_marker",
        "review_marker",
        "review_marker_gate_passes",
        "review_marker_gate_reasons",
        "review_marker_boom_passes",
        "review_marker_gui_passes",
        "reviewed_from",
        "review_timestamp",
        "review_selected_by",
        "selected_by",
        "track",
        "drums_path",
        "review_track",
        "review_source_path",
        "review_line_number",
    ]
    csv_path = out_dir / f"{report_path.stem}_human_review_audit.csv"
    hard_csv = out_dir / f"{report_path.stem}_human_review_validated_hard_mismatches.csv"
    stale_csv = out_dir / f"{report_path.stem}_human_review_stale_manual_mismatches.csv"
    advisory_csv = out_dir / f"{report_path.stem}_human_review_advisory_mismatches.csv"
    for path, selected_rows in (
        (csv_path, audit_rows),
        (hard_csv, validated_hard),
        (stale_csv, stale_manual),
        (advisory_csv, advisory),
    ):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for audit_row in selected_rows:
                writer.writerow({name: audit_row.get(name, "") for name in fieldnames})
    summary = {
        "source_report": str(report_path.expanduser()),
        "correction_logs": [str(Path(path).expanduser()) for path in correction_logs],
        "processed_rows": len(rows),
        "review_marker_count": len(markers),
        "matched_review_rows": len(audit_rows),
        "passed_count": len(matched),
        "hard_mismatch_count": len(validated_hard),
        "validated_hard_mismatch_count": len(validated_hard),
        "stale_manual_mismatch_count": len(stale_manual),
        "advisory_mismatch_count": len(advisory),
        "manual_review_rows": sum(1 for row in audit_rows if row["strength"] == "manual"),
        "blue_approval_rows": sum(1 for row in audit_rows if row["strength"] == "blue_approval"),
        "manual_tolerance_sec": float(manual_tolerance_sec),
        "blue_tolerance_sec": float(blue_tolerance_sec),
        "passed": len(validated_hard) == 0,
        "csv": str(csv_path),
        "hard_mismatch_csv": str(hard_csv),
        "stale_manual_mismatch_csv": str(stale_csv),
        "advisory_mismatch_csv": str(advisory_csv),
    }
    json_path = out_dir / f"{report_path.stem}_human_review_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(json_path)
    return summary


def _review_marker_gate(
    row: Mapping[str, Any],
    marker: float,
    *,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    reasons: List[str] = []
    candidates_json_raw = str(row.get("candidates_json") or "").strip()
    candidates_json = Path(candidates_json_raw).expanduser() if candidates_json_raw else None
    boom_candidates: List[Mapping[str, Any]] = []
    profile = None
    beatgrid = None
    if candidates_json is not None and candidates_json.exists():
        try:
            payload = json.loads(candidates_json.read_text(encoding="utf-8"))
            raw_boom = payload.get("boom_candidates") if isinstance(payload, Mapping) else None
            if isinstance(raw_boom, list):
                boom_candidates = [candidate for candidate in raw_boom if isinstance(candidate, Mapping)]
            feature_map = payload.get("feature_map") if isinstance(payload, Mapping) else None
            if isinstance(feature_map, Mapping):
                beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else None
        except Exception as exc:
            reasons.append(f"candidate_json_error:{str(exc) or exc.__class__.__name__}")
    else:
        reasons.append("missing_candidates_json")
    boom = marker_boom_proof(float(marker), boom_candidates, profile=profile, beatgrid=beatgrid)
    freshness = boom_proof_front_edge_freshness(boom)
    if not bool(boom.get("passes")):
        reasons.append("boom_proof_hold:" + ";".join(str(reason) for reason in boom.get("reasons") or [] if str(reason)))
    elif not bool(freshness.get("fresh")):
        reasons.append(f"boom_proof_stale:{freshness.get('reason') or 'unknown'}")
    gui = {"passes": False, "reasons": ["missing_cache_dir"]}
    drums_path = str(row.get("drums_path") or "")
    if cache_dir is not None and drums_path:
        try:
            gui = gui_boom_mask_proof(drums_path, float(marker), cache_dir=cache_dir)
            gui = accept_gui_boom_mask_with_front_edge_proof(gui, boom)
        except Exception as exc:
            gui = {"passes": False, "reasons": [f"gui_error:{str(exc) or exc.__class__.__name__}"]}
    if not bool(gui.get("passes")):
        reasons.append("gui_mask_hold:" + ";".join(str(reason) for reason in gui.get("reasons") or [] if str(reason)))
    return {
        "passes": bool(boom.get("passes")) and bool(freshness.get("fresh")) and bool(gui.get("passes")),
        "reasons": reasons,
        "boom_passes": bool(boom.get("passes")),
        "boom_fresh": bool(freshness.get("fresh")),
        "gui_passes": bool(gui.get("passes")),
        "boom_proof": boom,
        "gui_proof": gui,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fresh visual-first markers against latest human review/correction log markers.",
    )
    parser.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    parser.add_argument("--correction-log", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--manual-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--blue-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--require-no-hard-mismatches", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logs = [Path(path).expanduser() for path in (args.correction_log or [str(DEFAULT_CORRECTION_LOG)])]
    summary = audit_human_review_memory(
        Path(args.report),
        correction_logs=logs,
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir).expanduser() if args.cache_dir else None,
        manual_tolerance_sec=float(args.manual_tolerance_sec),
        blue_tolerance_sec=float(args.blue_tolerance_sec),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_no_hard_mismatches and int(summary.get("hard_mismatch_count") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
