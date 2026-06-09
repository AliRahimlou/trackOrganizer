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
from project_config import DROP_BATCH_SUMMARY


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


def sync_summary(summary: Path, *, dry_run: bool = False) -> dict[str, Any]:
    rows, names = _read_summary(summary)
    fieldnames = _fieldnames(names)
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
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
        row["detected_drop_time"] = _format_float(marker)
        row["microaligned_time"] = _format_float(marker)
        if selected_by:
            row["selected_by"] = selected_by
        if reviewed_from.strip().lower() == REVIEWED_FROM:
            row["visual_first_batch_auto_time"] = _format_float(marker)
            row["visual_first_batch_auto_source"] = selected_by or REVIEWED_FROM
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
        backup_dir = summary.parent / ".visual_first_batch_backups" / f"summary_sync_{_now_stamp()}"
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
