#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import trackOrganizerAndAlsGen as tog
from apply_folder_drop_candidates_to_set import (
    _clip_bpm,
    _clip_name,
    _iter_triplet_rows,
    _last_marker_sec,
    _replace_warp_markers,
    _resolve_clip_audio_path,
)
from drop_aligner.historical_markers import HistoricalMarker, load_historical_markers


DEFAULT_CORRECTION_LOGS = (
    Path("drop_corrections.jsonl"),
    Path("models/multistem_training_corrections.jsonl"),
)
DEFAULT_MARKER_DB = Path("drop_marker_db.json")


def _parse_warp_markers(clip: ET.Element) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        try:
            rows.append((float(marker.get("SecTime")), float(marker.get("BeatTime"))))
        except Exception:
            continue
    rows.sort(key=lambda item: item[0])
    return rows


def _has_drop_anchor_111(clip: ET.Element, min_drop_sec: float) -> bool:
    markers = _parse_warp_markers(clip)
    if len(markers) < 3:
        return False
    if not any(beat < -1e-6 for _, beat in markers):
        return False
    return any(sec > float(min_drop_sec) and abs(beat) <= 1e-6 for sec, beat in markers)


def _row_has_drop_anchor_111(row: Mapping[str, ET.Element], min_drop_sec: float) -> bool:
    return any(_has_drop_anchor_111(clip, min_drop_sec=min_drop_sec) for clip in row.values())


def _duration_seconds(path: str, cache: Dict[str, float]) -> Optional[float]:
    key = os.path.abspath(path)
    if key not in cache:
        try:
            cache[key] = float(tog.get_duration_seconds(key) or 0.0)
        except Exception:
            cache[key] = 0.0
    value = float(cache[key])
    return value if value > 0 else None


def _row_end_sec(row: Mapping[str, ET.Element], als_path: str, duration_cache: Dict[str, float]) -> Optional[float]:
    values: List[float] = []
    for clip in row.values():
        audio_path = _resolve_clip_audio_path(clip, als_path)
        if audio_path:
            duration = _duration_seconds(audio_path, duration_cache)
            if duration and duration > 0:
                values.append(float(duration))
                continue
        last_marker = _last_marker_sec(clip)
        if last_marker and last_marker > 0:
            values.append(float(last_marker))
    return max(values) if values else None


def _match_marker(
    index: Any,
    row: Mapping[str, ET.Element],
    als_path: str,
    bpm: Optional[int],
) -> Tuple[Optional[HistoricalMarker], str]:
    for role in ("drums", "inst", "vocals"):
        clip = row.get(role)
        if clip is None:
            continue
        audio_path = _resolve_clip_audio_path(clip, als_path)
        if audio_path:
            marker = index.find(audio_path, bpm=bpm)
            if marker is not None:
                return marker, audio_path

    for role in ("drums", "inst", "vocals"):
        clip = row.get(role)
        if clip is None:
            continue
        name = _clip_name(clip)
        marker = index.find(name, bpm=bpm)
        if marker is not None:
            return marker, name

    return None, ""


def _write_report(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "slot",
        "name",
        "bpm",
        "drop_sec",
        "match_key",
        "marker_track",
        "marker_source",
        "reviewed_from",
        "source_path",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def apply_manual_markers(
    als_in: str,
    als_out: str,
    *,
    correction_logs: Sequence[str],
    marker_db: str,
    min_drop_sec_existing: float,
    dry_run: bool,
    report_csv: str,
) -> Dict[str, int]:
    index = load_historical_markers(
        correction_logs=[path for path in correction_logs if path],
        marker_db_path=marker_db or None,
    )
    with gzip.open(als_in, "rb") as fh:
        root = ET.fromstring(fh.read())

    stats = {
        "rows_total": 0,
        "rows_protected_existing_anchor": 0,
        "rows_missing_anchor": 0,
        "rows_matched_manual": 0,
        "rows_changed": 0,
        "rows_skipped_no_bpm": 0,
        "rows_skipped_no_manual_marker": 0,
        "rows_skipped_no_end": 0,
        "clips_retimed": 0,
    }
    report_rows: List[Dict[str, Any]] = []
    duration_cache: Dict[str, float] = {}

    for slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        drums = row["drums"]
        if _row_has_drop_anchor_111(row, min_drop_sec=float(min_drop_sec_existing)):
            stats["rows_protected_existing_anchor"] += 1
            continue

        stats["rows_missing_anchor"] += 1
        bpm = _clip_bpm(drums)
        if not bpm:
            stats["rows_skipped_no_bpm"] += 1
            continue

        marker, match_key = _match_marker(index, row, als_in, bpm)
        if marker is None:
            stats["rows_skipped_no_manual_marker"] += 1
            continue

        stats["rows_matched_manual"] += 1
        drop_sec = float(marker.user_pick)
        end_sec = _row_end_sec(row, als_in, duration_cache)
        if end_sec is None or end_sec <= 0:
            stats["rows_skipped_no_end"] += 1
            continue
        end_sec = max(float(end_sec), drop_sec + 1.0)

        if not dry_run:
            for role in ("drums", "inst", "vocals"):
                _replace_warp_markers(row[role], int(bpm), drop_sec, float(end_sec))

        stats["rows_changed"] += 1
        stats["clips_retimed"] += 3
        report_rows.append(
            {
                "slot": int(slot_idx),
                "name": _clip_name(drums),
                "bpm": int(bpm),
                "drop_sec": f"{drop_sec:.9f}".rstrip("0").rstrip("."),
                "match_key": match_key,
                "marker_track": marker.track,
                "marker_source": marker.source,
                "reviewed_from": marker.reviewed_from,
                "source_path": marker.source_path,
            }
        )
        print(
            f"[MANUAL] slot={slot_idx} drop={drop_sec:.3f}s "
            f"source={marker.source}:{marker.reviewed_from or '-'} name={_clip_name(drums)}"
        )

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
        with gzip.open(als_out, "wb") as fh:
            fh.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    if report_csv:
        _write_report(report_csv, report_rows)
    return stats


def _default_report_path(als_out: str) -> str:
    stem, _ext = os.path.splitext(os.path.abspath(als_out))
    return f"{stem}_manual_111_report.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply human-reviewed 1.1.1 anchors to missing rows in a combined Ableton set."
    )
    parser.add_argument("--als", required=True, help="Input .als path")
    parser.add_argument("--out", required=True, help="Output .als path")
    parser.add_argument(
        "--correction-log",
        action="append",
        default=[],
        help="Correction JSONL to use as human markers. Can be repeated.",
    )
    parser.add_argument(
        "--marker-db",
        default=str(DEFAULT_MARKER_DB),
        help="Optional drop_marker_db.json path. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--min-drop-sec-existing",
        type=float,
        default=1.0,
        help="Protect existing BeatTime=0 anchors after this many seconds.",
    )
    parser.add_argument("--report-csv", default="", help="Report path. Defaults beside --out.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    correction_logs = args.correction_log or [str(path) for path in DEFAULT_CORRECTION_LOGS if path.exists()]
    report_csv = args.report_csv or _default_report_path(args.out)
    stats = apply_manual_markers(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        correction_logs=[os.path.abspath(str(Path(path).expanduser())) for path in correction_logs],
        marker_db=os.path.abspath(str(Path(args.marker_db).expanduser())) if str(args.marker_db).strip() else "",
        min_drop_sec_existing=float(args.min_drop_sec_existing),
        dry_run=bool(args.dry_run),
        report_csv=os.path.abspath(str(Path(report_csv).expanduser())) if report_csv else "",
    )
    print(stats)
    if args.dry_run:
        print("[DONE] Dry run only; no ALS written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
