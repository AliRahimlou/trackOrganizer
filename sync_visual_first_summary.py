#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from apply_visual_first_remaining import REVIEWED_FROM, SUMMARY_EXTRA_COLUMNS, _format_float
from drop_aligner.boom_profile import boom_proof_front_edge_freshness
from drop_aligner.historical_markers import is_human_review_source
from project_config import DROP_BATCH_SUMMARY


DISALLOWED_PASS_SOURCES = {
    "historical_human_marker",
    "historical_review_memory",
    "manual_review_marker",
    "review_auto_place",
    "saved_closest_to_review_pick",
    "visual_drop_v2",
    "visual_drop_v2_candidate",
    "visual_first_hold",
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


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _write_summary(path: Path, rows: list[Mapping[str, str]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)


def _fieldnames(existing: list[str]) -> list[str]:
    out = list(existing)
    for name in SUMMARY_EXTRA_COLUMNS:
        if name not in out:
            out.append(name)
    return out


def _marker_from_payload(payload: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "corrected_drop_time", "user_pick", "drop_sec", "downbeat_seconds"):
        value = _float_or_none(payload.get(key))
        if value is not None:
            return float(value)
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    for key in ("microaligned_time", "snapped_sec", "timestamp", "time_sec"):
        value = _float_or_none(selected.get(key))
        if value is not None:
            return float(value)
    return None


def _selected_by(payload: Mapping[str, Any]) -> str:
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    return str(selected.get("selected_by") or payload.get("selected_by") or "")


def _selected_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_candidate")
    return selected if isinstance(selected, Mapping) else {}


def _visual_audit(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = payload.get("visual_audit")
    if isinstance(audit, Mapping):
        return audit
    selected = _selected_candidate(payload)
    audit = selected.get("visual_audit")
    return audit if isinstance(audit, Mapping) else {}


def _boom_proof(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = payload.get("boom_proof")
    if isinstance(proof, Mapping):
        return proof
    selected = _selected_candidate(payload)
    proof = selected.get("boom_proof")
    return proof if isinstance(proof, Mapping) else {}


def _gui_mask_proof(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = payload.get("gui_mask_proof")
    if isinstance(proof, Mapping):
        return proof
    selected = _selected_candidate(payload)
    proof = selected.get("gui_mask_proof")
    return proof if isinstance(proof, Mapping) else {}


def _is_detector_payload(payload: Mapping[str, Any], selected_by: str) -> bool:
    reviewed_from = str(payload.get("reviewed_from") or "").strip()
    if is_human_review_source(reviewed_from):
        return False
    source = str(payload.get("source") or "").strip().lower()
    selected_source = str(_selected_candidate(payload).get("source") or "").strip().lower()
    selected_by_l = str(selected_by or "").strip().lower()
    return bool(
        reviewed_from.lower() == REVIEWED_FROM
        or bool(payload.get("visual_detector_prep"))
        or bool(payload.get("visual_first_batch_auto"))
        or source.startswith("visual_first")
        or source.startswith("visual_")
        or selected_source.startswith("visual_first")
        or selected_source.startswith("visual_")
        or selected_by_l.startswith("visual_")
    )


def _production_gate_reasons(payload: Mapping[str, Any], selected_by: str) -> list[str]:
    audit = _visual_audit(payload)
    proof = _boom_proof(payload)
    gui_proof = _gui_mask_proof(payload)
    status = str(audit.get("status") or "").strip().lower()
    flags = [str(flag) for flag in audit.get("flag_codes") or [] if str(flag)]
    reasons: list[str] = []
    if status != "pass":
        reasons.append(f"audit_status={status or 'missing'}")
    if flags:
        reasons.append("audit_flags=" + ";".join(flags))
    if _unsafe_selected_source(str(selected_by or "")):
        reasons.append(f"unsafe_source={selected_by}")
    if not bool(proof.get("passes")):
        proof_reasons = [str(reason) for reason in proof.get("reasons") or [] if str(reason)]
        reasons.append("boom_proof=hold" + (":" + ";".join(proof_reasons) if proof_reasons else ":missing"))
    else:
        freshness = boom_proof_front_edge_freshness(proof)
        if not bool(freshness.get("fresh")):
            reasons.append(f"boom_proof=stale_front_edge:{freshness.get('reason') or 'unknown'}")
    if not bool(gui_proof.get("passes")):
        gui_reasons = [str(reason) for reason in gui_proof.get("reasons") or [] if str(reason)]
        reasons.append("gui_mask=hold" + (":" + ";".join(gui_reasons) if gui_reasons else ":missing"))
    return reasons


def sync_summary(summary: Path, *, dry_run: bool = False) -> dict[str, Any]:
    rows, names = _read_summary(summary)
    fieldnames = _fieldnames(names)
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    held_examples: list[dict[str, Any]] = []
    for row in rows:
        candidates_path = Path(str(row.get("candidates_json") or "")).expanduser()
        if not candidates_path.exists():
            counts["missing_candidates_json"] += 1
            continue
        payload = _read_json(candidates_path)
        marker = _marker_from_payload(payload)
        if marker is None:
            counts["missing_marker"] += 1
            continue
        current = _float_or_none(row.get("detected_drop_time"))
        selected_by = _selected_by(payload)
        reviewed_from = str(payload.get("reviewed_from") or "")
        if _is_detector_payload(payload, selected_by):
            gate_reasons = _production_gate_reasons(payload, selected_by)
            if gate_reasons:
                counts["held_production_gate"] += 1
                for reason in gate_reasons:
                    counts[f"held:{reason}"] += 1
                if len(held_examples) < 12:
                    held_examples.append(
                        {
                            "filename": row.get("filename"),
                            "marker": float(marker),
                            "selected_by": selected_by,
                            "reasons": gate_reasons,
                            "candidates_json": str(candidates_path),
                        }
                    )
                continue
        row["detected_drop_time"] = _format_float(marker)
        row["microaligned_time"] = _format_float(marker)
        if selected_by:
            row["selected_by"] = selected_by
        if reviewed_from.strip().lower() == REVIEWED_FROM or payload.get("visual_detector_prep"):
            row["visual_detector_prep_time"] = _format_float(marker)
            row["visual_detector_prep_source"] = selected_by or REVIEWED_FROM
            if "visual_first_batch_auto_time" in row:
                row["visual_first_batch_auto_time"] = ""
            if "visual_first_batch_auto_source" in row:
                row["visual_first_batch_auto_source"] = ""
        counts["synced"] += 1
        if current is None or abs(float(marker) - float(current)) > 0.001:
            counts["changed"] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "filename": row.get("filename"),
                        "previous": current,
                        "marker": float(marker),
                        "selected_by": selected_by,
                    }
                )
    backup_path = ""
    if not dry_run:
        backup_dir = summary.parent / ".visual_detector_prep_backups" / f"summary_sync_{_now_stamp()}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / summary.name
        shutil.copy2(summary, backup)
        backup_path = str(backup)
        _write_summary(summary, rows, fieldnames)
    return {
        "summary": str(summary),
        "dry_run": bool(dry_run),
        "counts": dict(counts),
        "backup": backup_path,
        "examples": examples,
        "held_examples": held_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync drop_batch_summary.csv from per-track candidate JSON markers.")
    parser.add_argument("--summary", default=str(DROP_BATCH_SUMMARY), help="drop_batch_summary.csv to update")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = Path(args.summary).expanduser().resolve()
    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}")
    print(json.dumps(sync_summary(summary, dry_run=bool(args.dry_run)), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
