#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.als import modify_als
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.microalign import choose_microaligned_candidate, microalign_candidate_dicts
from verify_als import verify_als


DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "alsFiles" / "128.als"
SUMMARY_EXTRA_COLUMNS = [
    "micro_confidence",
    "snap_offset_ms",
    "microaligned_time",
]


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    tmp.replace(path)


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _stable_id(row: Mapping[str, str]) -> str:
    raw = "|".join([row.get("filename", ""), row.get("output_als", ""), row.get("candidates_json", "")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _summary_fieldnames(existing: Sequence[str]) -> List[str]:
    out = list(existing)
    for column in SUMMARY_EXTRA_COLUMNS:
        if column not in out:
            out.append(column)
    return out


def _read_summary(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        return list(reader), fieldnames


def _write_summary(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})
    tmp.replace(path)


def _load_reviewed_ids(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    state = _read_json(state_path)
    items = state.get("items")
    if not isinstance(items, Mapping):
        return set()
    return {
        str(item_id)
        for item_id, item_state in items.items()
        if isinstance(item_state, Mapping) and bool(item_state.get("reviewed") or item_state.get("skipped"))
    }


def _tier(row: Mapping[str, str], payload: Mapping[str, Any]) -> str:
    feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    value = row.get("confidence_tier") or payload.get("confidence_tier") or feature_summary.get("confidence_tier") or "UNKNOWN"
    text = str(value).strip().upper()
    return text if text in {"LOW", "MEDIUM", "HIGH"} else "UNKNOWN"


def _mark_selected_candidate(candidate: Mapping[str, Any], micro: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(candidate)
    original_timestamp = _float_or_none(out.get("timestamp"))
    suggested = float(micro["microaligned_time"])
    source_selected_by = str(out.get("selected_by") or "")
    out["pre_auto_place_timestamp"] = original_timestamp
    out["timestamp"] = suggested
    out["snapped_sec"] = suggested
    out["microalign"] = dict(micro)
    out["microaligned_time"] = suggested
    out["snap_offset_ms"] = float(micro.get("snap_offset_ms", 0.0) or 0.0)
    out["micro_confidence"] = float(micro.get("micro_confidence", 0.0) or 0.0)
    out["selected"] = True
    out["auto_place_initial_selected_by"] = source_selected_by
    out["selected_by"] = source_selected_by if source_selected_by == "candidate_chooser" else "review_auto_place"
    out["reason"] = "selected_by_review_auto_place_initial"
    return out


def _update_top_candidates(
    candidates: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    selected_rank = selected.get("rank")
    selected_pre = _float_or_none(selected.get("pre_auto_place_timestamp"))
    out: List[Dict[str, Any]] = []
    replaced = False
    for candidate in candidates:
        item = dict(candidate)
        item_rank = item.get("rank")
        item_time = _float_or_none(item.get("timestamp"))
        is_selected = False
        if selected_rank is not None and item_rank == selected_rank:
            is_selected = True
        elif selected_pre is not None and item_time is not None and abs(item_time - selected_pre) <= 1e-6:
            is_selected = True
        if is_selected and not replaced:
            out.append(dict(selected))
            replaced = True
        else:
            item["selected"] = False
            out.append(item)
    if not replaced:
        out.insert(0, dict(selected))
    return out


def _backup_file(path: Path, backup_dir: Path, manifest: List[Dict[str, str]]) -> None:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    backup = backup_dir / f"{digest}{path.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    manifest.append({"path": str(path), "backup": str(backup)})


def _iter_target_rows(
    rows: Sequence[Mapping[str, str]],
    reviewed_ids: set[str],
    *,
    remaining_only: bool,
) -> Iterable[Tuple[int, Mapping[str, str]]]:
    for index, row in enumerate(rows):
        if row_has_excluded_path(row):
            continue
        if remaining_only and _stable_id(row) in reviewed_ids:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status and status not in {"processed", "skipped", "partial"}:
            continue
        als_valid = str(row.get("als_valid", "")).strip().lower()
        if als_valid and als_valid != "true":
            continue
        yield index, row


def _apply_one(
    row: Mapping[str, str],
    *,
    template: Path,
    mode: str,
    regenerate_als: bool,
    force: bool,
    auto_accept_only: bool,
) -> Tuple[Dict[str, str], Optional[Dict[str, Any]], str]:
    updated_row = dict(row)
    audio_path = str(row.get("filename", ""))
    candidates_path = Path(str(row.get("candidates_json", ""))).expanduser()
    if not audio_path or not candidates_path.exists():
        return updated_row, None, "missing_candidates_json"

    payload = _read_json(candidates_path)
    if not force and isinstance(payload.get("auto_place_initial"), Mapping):
        return updated_row, None, "already_updated"
    candidates = payload.get("top_10_candidates")
    if not isinstance(candidates, list) or not candidates:
        return updated_row, None, "no_candidates"

    aligned = microalign_candidate_dicts(audio_path, candidates, limit=10)
    suggestion = choose_microaligned_candidate(aligned, confidence_tier=_tier(row, payload), mode=mode)
    gate = suggestion.get("auto_accept") if isinstance(suggestion.get("auto_accept"), Mapping) else {}
    mode_gate = gate.get(mode) if isinstance(gate.get(mode), Mapping) else {}
    if auto_accept_only and not bool(mode_gate.get("auto_accept")):
        return updated_row, None, "review_gate_failed"
    micro = suggestion.get("microalign")
    if not isinstance(micro, Mapping) or micro.get("microaligned_time") is None:
        return updated_row, None, "no_microalign_suggestion"

    suggested_time = float(micro["microaligned_time"])
    previous_pick = _float_or_none(payload.get("final_ai_pick")) or _float_or_none(row.get("detected_drop_time"))
    selected_candidate = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else {}
    selected_candidate = _mark_selected_candidate(selected_candidate, micro)

    payload["auto_place_initial"] = {
        "applied_at": _now_iso(),
        "mode": str(mode),
        "previous_final_ai_pick": previous_pick,
        "suggested_time": suggested_time,
        "reason": suggestion.get("reason", ""),
        "review_needed": bool(suggestion.get("review_needed", True)),
        "auto_place": bool(suggestion.get("auto_place", False)),
        "auto_accept": gate,
        "candidate_chooser": suggestion.get("candidate_chooser") if isinstance(suggestion.get("candidate_chooser"), Mapping) else None,
    }
    payload["final_ai_pick"] = suggested_time
    payload["drop_sec"] = suggested_time
    payload["selected_by"] = "review_auto_place"
    payload["selected_candidate"] = selected_candidate
    payload["top_10_candidates"] = _update_top_candidates(aligned, selected_candidate)
    features = payload.get("feature_summary")
    if isinstance(features, dict):
        features["selected_by"] = "review_auto_place"
        features["chosen_microaligned_time"] = suggested_time
        features["chosen_micro_confidence"] = float(micro.get("micro_confidence", 0.0) or 0.0)
        features["chosen_snap_offset_ms"] = float(micro.get("snap_offset_ms", 0.0) or 0.0)
        features["chosen_microalign_reason"] = str(micro.get("reason", ""))

    updated_row["detected_drop_time"] = f"{suggested_time:.9f}".rstrip("0").rstrip(".")
    updated_row["selected_by"] = "review_auto_place"
    updated_row["microaligned_time"] = f"{suggested_time:.9f}".rstrip("0").rstrip(".")
    updated_row["micro_confidence"] = f"{float(micro.get('micro_confidence', 0.0) or 0.0):.6f}".rstrip("0").rstrip(".")
    updated_row["snap_offset_ms"] = f"{float(micro.get('snap_offset_ms', 0.0) or 0.0):.6f}".rstrip("0").rstrip(".")

    if regenerate_als:
        output = Path(str(row.get("output_als", ""))).expanduser()
        feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
        bpm = _float_or_none(payload.get("bpm")) or _float_or_none(feature_summary.get("bpm")) or 128.0
        if output.exists():
            previous = output.with_name(output.stem + ".previous" + output.suffix)
            shutil.copy2(output, previous)
        modify_als(
            template_path=str(template),
            audio_path=audio_path,
            drop_sec=suggested_time,
            bpm=float(bpm),
            output_path=str(output),
            strict_stems=True,
        )
        verify_payload = candidates_path.with_suffix(candidates_path.suffix + ".auto_place_verify")
        with open(verify_payload, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
        try:
            report = verify_als(str(output), candidates_json=str(verify_payload))
        finally:
            verify_payload.unlink(missing_ok=True)
        updated_row["als_valid"] = "true" if report.get("valid") else "false"
        updated_row["als_validation_error"] = "" if report.get("valid") else "; ".join(str(err) for err in report.get("errors", []))
        if not report.get("valid"):
            return updated_row, payload, "als_validation_failed"

    return updated_row, payload, "updated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote review Auto Place suggestions into initial AI picks for review rows.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv to update")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="ALS template used when regenerating output ALS files")
    parser.add_argument("--state", help="review_state.json. Defaults to beside --summary")
    parser.add_argument("--mode", choices=["conservative", "normal", "aggressive"], default="normal")
    parser.add_argument("--all", dest="remaining_only", action="store_false", help="Update reviewed/skipped rows too")
    parser.set_defaults(remaining_only=True)
    parser.add_argument("--no-regenerate-als", dest="regenerate_als", action="store_false", help="Only update CSV/JSON markers")
    parser.set_defaults(regenerate_als=True)
    parser.add_argument("--limit", type=int, default=0, help="Only update this many target rows")
    parser.add_argument("--force", action="store_true", help="Re-apply even when auto_place_initial already exists")
    parser.add_argument("--auto-accept-only", action="store_true", help="Only apply suggestions that pass the selected auto-accept gate")
    parser.add_argument("--dry-run", action="store_true", help="Analyze but do not write files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = Path(args.summary).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    state = Path(args.state).expanduser().resolve() if args.state else summary.parent / "review_state.json"
    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}")
    if bool(args.regenerate_als) and not template.exists():
        raise SystemExit(f"Template not found: {template}")

    rows, fieldnames = _read_summary(summary)
    output_fields = _summary_fieldnames(fieldnames)
    reviewed_ids = _load_reviewed_ids(state)
    target_rows = list(_iter_target_rows(rows, reviewed_ids, remaining_only=bool(args.remaining_only)))
    if int(args.limit) > 0:
        target_rows = target_rows[: int(args.limit)]

    stamp = _now_stamp()
    backup_dir = summary.parent / ".auto_place_initial_backups" / stamp
    manifest: List[Dict[str, str]] = []
    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        _backup_file(summary, backup_dir, manifest)

    counts: Dict[str, int] = {}
    for number, (row_index, row) in enumerate(target_rows, start=1):
        updated_row, updated_payload, status = _apply_one(
            row,
            template=template,
            mode=str(args.mode),
            regenerate_als=bool(args.regenerate_als) and not bool(args.dry_run),
            force=bool(args.force),
            auto_accept_only=bool(args.auto_accept_only),
        )
        counts[status] = counts.get(status, 0) + 1
        if updated_payload is not None and not args.dry_run:
            candidates_path = Path(str(row.get("candidates_json", ""))).expanduser()
            _backup_file(candidates_path, backup_dir, manifest)
            _write_json(candidates_path, updated_payload)
            rows[row_index] = updated_row
        print(f"[{number}/{len(target_rows)}] {status}: {row.get('filename', '')}", flush=True)

    if not args.dry_run:
        _write_summary(summary, rows, output_fields)
        manifest_path = backup_dir / "manifest.json"
        _write_json(manifest_path, {"created_at": _now_iso(), "files": manifest})

    print(
        json.dumps(
            {
                "summary": str(summary),
                "state": str(state),
                "remaining_only": bool(args.remaining_only),
                "targets": len(target_rows),
                "counts": counts,
                "dry_run": bool(args.dry_run),
                "backup_dir": "" if args.dry_run else str(backup_dir),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 1 if counts.get("als_validation_failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
