#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from drop_aligner.historical_markers import is_human_review_source
from project_config import DEFAULT_ALS_TEMPLATE, DROP_BATCH_SUMMARY
from web_review import (
    ReviewApp,
    _candidate_marker_time,
    _float_or_none,
    _now_iso,
    _read_json,
    _stable_id,
    _write_json,
)


DETECTOR_PREP_SOURCE = "visual_detector_prep"
STALE_BATCH_AUTO_SOURCE = "visual_first_batch_auto"
STALE_DETECTOR_SOURCES = {STALE_BATCH_AUTO_SOURCE, "batch_auto", DETECTOR_PREP_SOURCE}
REVIEWED_FROM = DETECTOR_PREP_SOURCE
SUMMARY_EXTRA_COLUMNS = [
    "micro_confidence",
    "snap_offset_ms",
    "microaligned_time",
    "visual_detector_prep_time",
    "visual_detector_prep_source",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _write_summary(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})
    tmp.replace(path)


def _format_float(value: Any, digits: int = 9) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{float(number):.{int(digits)}f}".rstrip("0").rstrip(".")


def _summary_fieldnames(existing: Sequence[str]) -> list[str]:
    out = list(existing)
    for column in SUMMARY_EXTRA_COLUMNS:
        if column not in out:
            out.append(column)
    return out


def _backup_file(path: Path, backup_dir: Path, manifest: list[dict[str, str]]) -> None:
    if not path.exists():
        return
    digest = _stable_backup_name(path)
    backup = backup_dir / f"{digest}{path.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    manifest.append({"path": str(path), "backup": str(backup)})


def _stable_backup_name(path: Path) -> str:
    import hashlib

    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _candidate_source(candidate: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    return str(candidate.get("selected_by") or item.get("selected_by") or "visual_first_selected_candidate")


def _candidate_rank(candidate: Mapping[str, Any]) -> Optional[int]:
    value = candidate.get("rank") or candidate.get("picked_candidate_rank")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _marker_candidate(item: Mapping[str, Any]) -> tuple[Optional[float], Dict[str, Any], str]:
    candidate = dict(item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else {})
    marker = _candidate_marker_time(candidate)
    if marker is None:
        marker = _float_or_none(item.get("initial_candidate_time"))
    if marker is None:
        marker = _float_or_none(item.get("ai_pick"))
    source = _candidate_source(candidate, item)
    return (None if marker is None else float(marker), candidate, source)


def _payload_for_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(item.get("candidates_json", ""))).expanduser()
    if not path.exists():
        return {}
    payload = _read_json(str(path))
    return payload if isinstance(payload, dict) else {}


def _manual_preserve_reason(item: Mapping[str, Any], row: Optional[Mapping[str, str]] = None) -> str:
    payload = _payload_for_item(item)
    review = item.get("review") if isinstance(item.get("review"), Mapping) else {}
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    sources = [
        review.get("reviewed_from"),
        payload.get("reviewed_from"),
        selected.get("reviewed_from"),
        row.get("reviewed_from") if row else None,
    ]
    for source in sources:
        if is_human_review_source(source):
            return f"human_review:{source}"

    selected_by_values = [
        review.get("selected_by"),
        item.get("selected_by"),
        payload.get("selected_by"),
        selected.get("selected_by"),
        row.get("selected_by") if row else None,
    ]
    for selected_by in selected_by_values:
        value = str(selected_by or "").strip().lower()
        if value == "user_candidate_pick":
            return "user_candidate_pick"
        if value == "historical_human_marker" and str(payload.get("reviewed_from") or "").strip().lower() not in STALE_DETECTOR_SOURCES:
            return "historical_human_marker"

    if bool(review.get("corrected")) and not bool(review.get("visual_first_batch_auto")) and not bool(review.get("detector_prep")):
        return "state_corrected_without_batch_auto_flag"
    if _float_or_none(payload.get("corrected_drop_time")) is not None and not payload.get("visual_first_batch_auto") and not payload.get("visual_detector_prep"):
        source = str(payload.get("reviewed_from") or "")
        if source.strip().lower() not in STALE_DETECTOR_SOURCES:
            return "candidate_json_corrected_marker"
    return ""


def _replace_item_selection(
    item: Dict[str, Any],
    marker: float,
    candidate: Mapping[str, Any],
    selected_by: str,
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    item["initial_candidate_time"] = float(marker)
    item["selected_by"] = str(selected_by)
    item["selected_candidate"] = dict(candidate)
    item["top_10_candidates"] = [dict(row) for row in candidates[:10] if isinstance(row, Mapping)]


def _previous_batch_auto_candidate(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    payload = _payload_for_item(item)
    if str(payload.get("reviewed_from") or "").strip().lower() not in STALE_DETECTOR_SOURCES and not payload.get("visual_first_batch_auto"):
        return None
    marker = _float_or_none(payload.get("final_ai_pick")) or _float_or_none(payload.get("drop_sec")) or _float_or_none(item.get("ai_pick"))
    if marker is None or marker <= 0.0:
        return None
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    candidate = dict(selected)
    candidate["timestamp"] = float(marker)
    candidate["snapped_sec"] = float(marker)
    candidate["microaligned_time"] = float(marker)
    candidate["time_sec"] = float(marker)
    candidate["selected"] = False
    candidate["selected_by"] = "saved_visual_batch_auto_marker"
    candidate["source"] = "saved_visual_batch_auto_marker"
    candidate["reason"] = "previous visual-first batch marker retained as a normal rescan candidate"
    candidate["score"] = min(0.74, _float_or_none(candidate.get("score")) or 0.70)
    candidate["confidence_score"] = min(0.74, _float_or_none(candidate.get("confidence_score")) or 0.70)
    return candidate


def _is_previous_batch_auto_candidate(candidate: Mapping[str, Any]) -> bool:
    selected_by = str(candidate.get("selected_by") or "").strip().lower()
    source = str(candidate.get("source") or "").strip().lower()
    reviewed_from = str(candidate.get("reviewed_from") or "").strip().lower()
    reason = str(candidate.get("reason") or "").strip().lower()
    return bool(
        selected_by in {"saved_visual_batch_auto_marker", "visual_first_batch_auto", DETECTOR_PREP_SOURCE}
        or source in {"saved_visual_batch_auto_marker", "visual_first_batch_auto", DETECTOR_PREP_SOURCE}
        or reviewed_from in STALE_DETECTOR_SOURCES
        or "previous visual-first batch marker retained" in reason
    )


def _rescan_marker_candidate(
    app: ReviewApp,
    item: Dict[str, Any],
    mode: str,
    *,
    allow_visual_audit_review: bool = False,
) -> tuple[Optional[float], Dict[str, Any], str, list[Dict[str, Any]]]:
    mode_text = str(mode or "raw").strip().lower()
    keep_previous_batch_auto = mode_text in {"with_previous_batch_auto", "raw_with_previous_batch_auto"}
    previous_candidate = _previous_batch_auto_candidate(item) if keep_previous_batch_auto else None
    original_candidates = [dict(row) for row in item.get("top_10_candidates") or [] if isinstance(row, Mapping)]
    if not keep_previous_batch_auto:
        original_candidates = [row for row in original_candidates if not _is_previous_batch_auto_candidate(row)]
        item["top_10_candidates"] = [dict(row) for row in original_candidates]
    if previous_candidate is not None:
        item["top_10_candidates"] = [previous_candidate] + [
            row
            for row in original_candidates
            if (_candidate_marker_time(row) is None or abs(float(_candidate_marker_time(row) or 0.0) - float(previous_candidate["timestamp"])) > 0.010)
        ]
    result = app.auto_place(str(item.get("id")), mode=mode)
    if not result.get("ok"):
        return None, {}, str(result.get("source") or "rescan_failed"), []
    candidates = [dict(row) for row in result.get("candidates") or [] if isinstance(row, Mapping)]
    suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), Mapping) else {}
    candidate = dict(suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else {})
    if not candidate and candidates:
        candidate = dict(candidates[0])
    marker = _float_or_none(suggestion.get("suggested_time"))
    if marker is None and candidate:
        marker = _candidate_marker_time(candidate)
    source_info = result.get("source_info") if isinstance(result.get("source_info"), Mapping) else {}
    visual_info = source_info.get("visual_first") if isinstance(source_info.get("visual_first"), Mapping) else {}
    visual_audit = visual_info.get("audit") if isinstance(visual_info.get("audit"), Mapping) else {}
    if visual_audit:
        candidate["visual_audit"] = dict(visual_audit)
        audit_status = str(visual_audit.get("status") or "unknown")
        if audit_status != "pass" and not bool(allow_visual_audit_review):
            return None, candidate, f"visual_audit_{audit_status}", candidates
    visual_boom_proof = visual_info.get("boom_proof") if isinstance(visual_info.get("boom_proof"), Mapping) else {}
    if visual_boom_proof and not isinstance(candidate.get("boom_proof"), Mapping):
        candidate["boom_proof"] = dict(visual_boom_proof)
    selected_by = str(candidate.get("selected_by") or result.get("source") or "visual_first_rescan")
    if marker is not None and candidate:
        _replace_item_selection(item, float(marker), candidate, selected_by, candidates or [candidate])
    return (None if marker is None else float(marker), candidate, selected_by, candidates)


def _update_candidate_payload(
    item: Mapping[str, Any],
    marker: float,
    candidate: Mapping[str, Any],
    selected_by: str,
    regen: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(str(item.get("candidates_json", ""))).expanduser()
    if not path.exists():
        return
    payload = _read_json(str(path))
    selected = dict(candidate)
    boom_proof = (
        regen.get("boom_proof")
        if isinstance(regen, Mapping) and isinstance(regen.get("boom_proof"), Mapping)
        else selected.get("boom_proof")
        if isinstance(selected.get("boom_proof"), Mapping)
        else {}
    )
    gui_mask_proof = (
        regen.get("gui_mask_proof")
        if isinstance(regen, Mapping) and isinstance(regen.get("gui_mask_proof"), Mapping)
        else selected.get("gui_mask_proof")
        if isinstance(selected.get("gui_mask_proof"), Mapping)
        else {}
    )
    selected["timestamp"] = float(marker)
    selected["snapped_sec"] = float(marker)
    selected["microaligned_time"] = float(marker)
    selected["selected"] = True
    selected["selected_by"] = selected_by
    if isinstance(boom_proof, Mapping):
        selected["boom_proof"] = dict(boom_proof)
    if isinstance(gui_mask_proof, Mapping):
        selected["gui_mask_proof"] = dict(gui_mask_proof)
    selected.setdefault("reason", f"selected by {DETECTOR_PREP_SOURCE}")
    top_candidates = [dict(row) for row in item.get("top_10_candidates") or [] if isinstance(row, Mapping)]
    if top_candidates:
        selected_time = _candidate_marker_time(selected)
        replaced = False
        for index, row in enumerate(top_candidates):
            row_time = _candidate_marker_time(row)
            if selected_time is not None and row_time is not None and abs(float(row_time) - float(selected_time)) <= 0.010:
                top_candidates[index] = dict(selected)
                replaced = True
                break
        if not replaced:
            top_candidates.insert(0, dict(selected))
    else:
        top_candidates = [dict(selected)]

    previous_final_ai_pick = _float_or_none(payload.get("final_ai_pick"))
    payload["final_ai_pick"] = float(marker)
    payload["drop_sec"] = float(marker)
    payload["selected_by"] = selected_by
    payload["selected_candidate"] = selected
    payload["top_10_candidates"] = top_candidates[:10]
    payload["reviewed_from"] = DETECTOR_PREP_SOURCE
    if isinstance(selected.get("visual_audit"), Mapping):
        payload["visual_audit"] = dict(selected.get("visual_audit") or {})
    if isinstance(boom_proof, Mapping):
        payload["boom_proof"] = dict(boom_proof)
    if isinstance(gui_mask_proof, Mapping):
        payload["gui_mask_proof"] = dict(gui_mask_proof)
    payload.pop("reviewed_at", None)
    payload.pop("visual_first_batch_auto", None)
    payload["detector_prepared_at"] = _now_iso()
    payload["visual_detector_prep"] = {
        "applied_at": _now_iso(),
        "suggested_time": float(marker),
        "selected_by": selected_by,
        "candidate_rank": _candidate_rank(selected),
        "previous_final_ai_pick": previous_final_ai_pick,
    }
    features = payload.get("feature_summary")
    if isinstance(features, dict):
        features["selected_by"] = selected_by
        features["chosen_microaligned_time"] = float(marker)
        micro = selected.get("microalign") if isinstance(selected.get("microalign"), Mapping) else selected
        if _float_or_none(micro.get("micro_confidence")) is not None:
            features["chosen_micro_confidence"] = float(micro.get("micro_confidence"))
        if _float_or_none(micro.get("snap_offset_ms")) is not None:
            features["chosen_snap_offset_ms"] = float(micro.get("snap_offset_ms"))
        if micro.get("reason"):
            features["chosen_microalign_reason"] = str(micro.get("reason"))
    _write_json(path, payload)


def _update_summary_row(row: dict[str, str], item: Mapping[str, Any], marker: float, selected_by: str, regen: Mapping[str, Any]) -> None:
    candidate = item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else {}
    micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else candidate
    verification = regen.get("verification") if isinstance(regen.get("verification"), Mapping) else {}
    row["detected_drop_time"] = _format_float(marker)
    row["selected_by"] = selected_by
    row["microaligned_time"] = _format_float(marker)
    row["micro_confidence"] = _format_float(micro.get("micro_confidence"), digits=6)
    row["snap_offset_ms"] = _format_float(micro.get("snap_offset_ms"), digits=6)
    row["visual_detector_prep_time"] = _format_float(marker)
    row["visual_detector_prep_source"] = selected_by
    if "visual_first_batch_auto_time" in row:
        row["visual_first_batch_auto_time"] = ""
    if "visual_first_batch_auto_source" in row:
        row["visual_first_batch_auto_source"] = ""
    if verification:
        row["als_valid"] = "true" if verification.get("valid") else "false"
        row["als_validation_error"] = "" if verification.get("valid") else "; ".join(str(err) for err in verification.get("errors", []))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply the current visual-first review marker to all remaining tracks.")
    parser.add_argument("--summary", default=str(DROP_BATCH_SUMMARY), help="drop_batch_summary.csv to update")
    parser.add_argument("--template", default=str(DEFAULT_ALS_TEMPLATE), help="ALS template used to regenerate per-track ALS files")
    parser.add_argument("--correction-log", default="drop_corrections.jsonl", help="Existing correction log used only for historical memory")
    parser.add_argument("--limit", type=int, default=0, help="Only process this many remaining rows")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index within the selected target list to start from. Intended for disjoint worker shards.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="1-based inclusive index within the selected target list to stop at. 0 means the end of the list.",
    )
    parser.add_argument("--all", action="store_true", help="Refresh all eligible rows, including rows already marked reviewed")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Run the current visual-first auto placer for each target instead of reusing the saved candidate JSON pick",
    )
    parser.add_argument(
        "--overwrite-manual",
        action="store_true",
        help="Allow rewriting rows that look manually reviewed. By default human placements are preserved.",
    )
    parser.add_argument(
        "--min-change-ms",
        type=float,
        default=0.0,
        help="Skip refreshed rows whose new marker differs from the current summary marker by less than this many ms.",
    )
    parser.add_argument(
        "--max-change-ms",
        type=float,
        default=0.0,
        help="Skip refreshed rows whose new marker differs from the current summary marker by more than this many ms. 0 disables the cap.",
    )
    parser.add_argument(
        "--mode",
        default="visual_only",
        help=(
            "Auto-place mode used with --rescan. Defaults to visual_only: raw waveform/section visual scan, "
            "then zoomed micro placement, with history/models/saved candidates bypassed. Use raw only for the old mixed path."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing")
    parser.add_argument(
        "--no-shared-writes",
        action="store_true",
        help=(
            "Do not write review_state.json or the shared summary CSV. Use only for disjoint shards; "
            "run sync_visual_first_summary.py after all shards finish."
        ),
    )
    parser.add_argument("--review-low-only", action="store_true")
    parser.add_argument("--review-medium-and-low", action="store_true")
    parser.add_argument(
        "--allow-visual-audit-review",
        action="store_true",
        help="Allow --rescan visual-first writes even when the detector audit recommends review or replacement.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = Path(args.summary).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}")
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    app = ReviewApp(
        summary_csv=str(summary),
        template=str(template),
        correction_log=str(args.correction_log),
        auto_retrain_every=0,
        review_low_only=bool(args.review_low_only),
        review_medium_and_low=bool(args.review_medium_and_low),
        regenerate_als_on_correction=True,
        visual_first=True,
    )
    if args.no_shared_writes:
        app._save_state = lambda: None  # type: ignore[method-assign]
    rows, fieldnames = _read_summary(summary)
    fieldnames = _summary_fieldnames(fieldnames)
    row_by_id = {_stable_id(row): row for row in rows}
    all_targets = list(app.items) if args.all else [item for item in app.items if not app._completed(item)]
    total_target_count = len(all_targets)
    start_index = max(1, int(args.start_index or 1))
    end_index = int(args.end_index or 0)
    if end_index <= 0 or end_index > total_target_count:
        end_index = total_target_count
    if start_index > end_index:
        targets = []
    else:
        targets = all_targets[start_index - 1 : end_index]
    if int(args.limit) > 0:
        targets = targets[: int(args.limit)]
    target_offset = start_index - 1

    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    if args.dry_run:
        for item in targets:
            row = row_by_id.get(str(item.get("id")))
            preserve_reason = "" if args.overwrite_manual else _manual_preserve_reason(item, row)
            if preserve_reason:
                status_counts["skipped_manual_preserve"] += 1
                continue
            if args.rescan:
                marker, candidate, selected_by, _ = _rescan_marker_candidate(
                    app,
                    item,
                    str(args.mode or "raw"),
                    allow_visual_audit_review=bool(args.allow_visual_audit_review),
                )
            else:
                marker, candidate, selected_by = _marker_candidate(item)
            if marker is None:
                if selected_by.startswith("visual_audit_"):
                    status_counts[selected_by] += 1
                else:
                    status_counts["missing_marker"] += 1
                continue
            current = _float_or_none(row.get("detected_drop_time") if row else item.get("ai_pick"))
            if float(args.min_change_ms or 0.0) > 0 and current is not None:
                delta_ms = abs(float(marker) - float(current)) * 1000.0
                if delta_ms < float(args.min_change_ms):
                    status_counts["skipped_below_min_change"] += 1
                    continue
            if float(args.max_change_ms or 0.0) > 0 and current is not None:
                delta_ms = abs(float(marker) - float(current)) * 1000.0
                if delta_ms > float(args.max_change_ms):
                    status_counts["skipped_above_max_change"] += 1
                    continue
            source_counts[selected_by] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "track": item.get("track_name"),
                        "marker": marker,
                        "previous_marker": current,
                        "delta_ms": None if current is None else round((float(marker) - float(current)) * 1000.0, 3),
                        "selected_by": selected_by,
                        "rank": _candidate_rank(candidate),
                    }
                )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "summary": str(summary),
                    "targets": len(targets),
                    "target_range": {
                        "start_index": start_index,
                        "end_index": end_index,
                        "total_targets": total_target_count,
                    },
                    "eligible": sum(source_counts.values()),
                    "all": bool(args.all),
                        "rescan": bool(args.rescan),
                        "overwrite_manual": bool(args.overwrite_manual),
                        "no_shared_writes": bool(args.no_shared_writes),
                        "allow_visual_audit_review": bool(args.allow_visual_audit_review),
                        "min_change_ms": float(args.min_change_ms or 0.0),
                        "max_change_ms": float(args.max_change_ms or 0.0),
                        "source_counts": dict(source_counts),
                    "status_counts": dict(status_counts),
                    "examples": examples,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    stamp = _stamp()
    backup_dir = summary.parent / ".visual_detector_prep_backups" / stamp
    report_dir = summary.parent / ".visual_detector_prep_reports"
    backup_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    if not args.no_shared_writes:
        _backup_file(summary, backup_dir, manifest)
        _backup_file(app.state_path, backup_dir, manifest)

    processed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for local_index, item in enumerate(targets, start=1):
        index = target_offset + local_index
        progress_total = total_target_count
        row = row_by_id.get(str(item.get("id")))
        preserve_reason = "" if args.overwrite_manual else _manual_preserve_reason(item, row)
        if preserve_reason:
            status_counts["skipped_manual_preserve"] += 1
            print(f"[{index}/{progress_total}] skip manual preserve ({preserve_reason}): {item.get('audio_path')}", flush=True)
            continue
        if args.rescan:
            marker, candidate, selected_by, candidates = _rescan_marker_candidate(
                app,
                item,
                str(args.mode or "raw"),
                allow_visual_audit_review=bool(args.allow_visual_audit_review),
            )
        else:
            marker, candidate, selected_by = _marker_candidate(item)
            candidates = [dict(row) for row in item.get("top_10_candidates") or [] if isinstance(row, Mapping)]
        if marker is None:
            if selected_by.startswith("visual_audit_"):
                status_counts[selected_by] += 1
                failures.append(
                    {
                        "id": item.get("id"),
                        "track": item.get("audio_path"),
                        "error": selected_by,
                        "visual_audit": candidate.get("visual_audit") if isinstance(candidate.get("visual_audit"), Mapping) else {},
                    }
                )
                print(f"[{index}/{progress_total}] HOLD {selected_by}: {item.get('audio_path')}", flush=True)
            else:
                status_counts["missing_marker"] += 1
                failures.append({"id": item.get("id"), "track": item.get("audio_path"), "error": "missing_marker"})
            continue
        current = _float_or_none(row.get("detected_drop_time") if row else item.get("ai_pick"))
        if float(args.min_change_ms or 0.0) > 0 and current is not None:
            delta_ms = abs(float(marker) - float(current)) * 1000.0
            if delta_ms < float(args.min_change_ms):
                status_counts["skipped_below_min_change"] += 1
                print(
                    f"[{index}/{progress_total}] skip below min change {delta_ms:.3f}ms: {item.get('audio_path')}",
                    flush=True,
                )
                continue
        if float(args.max_change_ms or 0.0) > 0 and current is not None:
            delta_ms = abs(float(marker) - float(current)) * 1000.0
            if delta_ms > float(args.max_change_ms):
                status_counts["skipped_above_max_change"] += 1
                print(
                    f"[{index}/{progress_total}] skip above max change {delta_ms:.3f}ms: {item.get('audio_path')}",
                    flush=True,
                )
                continue
        source_counts[selected_by] += 1
        candidate_path = Path(str(item.get("candidates_json", ""))).expanduser()
        _backup_file(candidate_path, backup_dir, manifest)
        regen = app.regenerate_als(
            str(item.get("id")),
            float(marker),
            reviewed_from=REVIEWED_FROM,
            top_candidates=candidates or item.get("top_10_candidates") or [],
            selected_candidate=candidate,
            selected_by=selected_by,
        )
        if not regen.get("ok"):
            status_counts["regenerate_failed"] += 1
            failures.append({"id": item.get("id"), "track": item.get("audio_path"), "marker": marker, "error": regen.get("error") or regen})
            print(f"[{index}/{progress_total}] FAIL {marker:.6f}: {item.get('audio_path')} :: {regen.get('error') or regen}", flush=True)
            continue

        _update_candidate_payload(item, float(marker), candidate, selected_by, regen=regen)
        if row is not None:
            _update_summary_row(row, item, float(marker), selected_by, regen)
        original = _float_or_none(item.get("ai_pick")) or float(marker)
        review = item["review"]
        review.update(
            {
                "reviewed": False,
                "skipped": False,
                "approved": False,
                "corrected": False,
                "user_pick": None,
                "timestamp_reviewed": "",
                "reviewed_from": "",
                "selected_by": selected_by,
                "selected_candidate_rank": _candidate_rank(candidate),
                "detector_prep": True,
                "detector_prep_source": DETECTOR_PREP_SOURCE,
                "detector_prep_marker": float(marker),
                "detector_prep_at": _now_iso(),
                "visual_first_rescan": bool(args.rescan),
                "regenerated_als_path": str(regen.get("output_als") or ""),
                "als_valid": bool((regen.get("verification") or {}).get("valid")),
                "als_validation_error": "; ".join(str(err) for err in (regen.get("verification") or {}).get("errors", [])),
            }
        )
        review.pop("visual_first_batch_auto", None)
        status_counts["updated"] += 1
        processed.append(
            {
                "id": item.get("id"),
                "track": item.get("audio_path"),
                "marker": float(marker),
                "previous_marker": current,
                "delta_ms": None if current is None else round((float(marker) - float(current)) * 1000.0, 3),
                "selected_by": selected_by,
                "rank": _candidate_rank(candidate),
                "output_als": regen.get("output_als"),
            }
        )
        print(f"[{index}/{progress_total}] updated {marker:.6f}: {item.get('audio_path')}", flush=True)

    if args.no_shared_writes:
        status_counts["skipped_shared_writes"] += 1
    else:
        app.state["current_index"] = app._first_active_index(int(app.state.get("current_index", 0)))
        app.state.pop("visual_first_batch_auto", None)
        app.state["visual_detector_prep"] = {
            "applied_at": _now_iso(),
            "targets": len(targets),
            "updated": int(status_counts.get("updated", 0)),
            "failures": len(failures),
            "backup_dir": str(backup_dir),
            "source": DETECTOR_PREP_SOURCE,
            "all": bool(args.all),
            "rescan": bool(args.rescan),
            "overwrite_manual": bool(args.overwrite_manual),
            "allow_visual_audit_review": bool(args.allow_visual_audit_review),
            "min_change_ms": float(args.min_change_ms or 0.0),
            "max_change_ms": float(args.max_change_ms or 0.0),
        }
        app._save_state()
        _write_summary(summary, rows, fieldnames)
    manifest_path = backup_dir / "manifest.json"
    _write_json(manifest_path, {"created_at": _now_iso(), "files": manifest})
    report_path = report_dir / f"visual_detector_prep_{stamp}.json"
    report = {
        "summary": str(summary),
        "template": str(template),
        "targets": len(targets),
        "target_range": {
            "start_index": start_index,
            "end_index": end_index,
            "total_targets": total_target_count,
        },
        "counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "all": bool(args.all),
        "rescan": bool(args.rescan),
        "overwrite_manual": bool(args.overwrite_manual),
        "no_shared_writes": bool(args.no_shared_writes),
        "allow_visual_audit_review": bool(args.allow_visual_audit_review),
        "min_change_ms": float(args.min_change_ms or 0.0),
        "max_change_ms": float(args.max_change_ms or 0.0),
        "backup_dir": str(backup_dir),
        "manifest": str(manifest_path),
        "failures": failures,
        "processed_sample": processed[:20],
    }
    _write_json(report_path, report)
    print(json.dumps({**report, "report": str(report_path)}, indent=2, ensure_ascii=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
