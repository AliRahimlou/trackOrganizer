#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from ableton_analysis_adapter import extract_ableton_onset_markers
from apply_folder_drop_candidates_to_set import (
    _clip_bpm,
    _clip_name,
    _current_anchor_sec,
    _iter_triplet_rows,
    _last_marker_sec,
    _replace_warp_markers,
    _resolve_clip_audio_path,
)

try:
    from drop_aligner.microalign import microalign_marker
except Exception:
    microalign_marker = None


def _nearest_marker(markers: List[float], current: float, max_offset_sec: float) -> Optional[float]:
    if not markers:
        return None
    lo = float(current) - float(max_offset_sec)
    hi = float(current) + float(max_offset_sec)
    nearby = [float(marker) for marker in markers if lo <= float(marker) <= hi]
    if not nearby:
        return None
    return min(nearby, key=lambda marker: abs(float(marker) - float(current)))


def _microalign_fallback(audio_path: str, current: float, max_offset_sec: float) -> Tuple[Optional[float], str]:
    if microalign_marker is None:
        return None, "microalign_unavailable"
    try:
        result = microalign_marker(
            audio_path,
            float(current),
            search_before_ms=max(250.0, float(max_offset_sec) * 1000.0),
            search_after_ms=max(350.0, float(max_offset_sec) * 1000.0),
        )
    except Exception as exc:
        return None, f"microalign_error:{type(exc).__name__}"

    try:
        snapped = float(result.get("microaligned_time"))
        confidence = float(result.get("micro_confidence") or 0.0)
        impact_confidence = float(result.get("impact_boundary_confidence") or 0.0)
    except Exception:
        return None, "microalign_bad_result"
    if not math.isfinite(snapped):
        return None, "microalign_nonfinite"
    if abs(snapped - float(current)) > float(max_offset_sec):
        return None, "microalign_too_far"
    if max(confidence, impact_confidence) < 0.72:
        return None, "microalign_low_confidence"
    return float(snapped), f"microalign:{result.get('reason') or 'accepted'}"


def _choose_snap(
    audio_path: str,
    current: float,
    *,
    max_asd_offset_sec: float,
    max_micro_offset_sec: float,
    microalign_fallback: bool,
) -> Tuple[Optional[float], str, int]:
    try:
        markers = extract_ableton_onset_markers(audio_path)
    except Exception as exc:
        markers = None
        asd_error = f"asd_error:{type(exc).__name__}"
    else:
        asd_error = "no_asd_markers"

    if markers is not None and markers.candidate_seconds:
        candidate = _nearest_marker([float(t) for t in markers.candidate_seconds], float(current), float(max_asd_offset_sec))
        if candidate is not None:
            return float(candidate), str(markers.source), int(len(markers.candidate_seconds))
        asd_error = f"{markers.source}_none_near"

    if microalign_fallback:
        snapped, reason = _microalign_fallback(audio_path, float(current), float(max_micro_offset_sec))
        if snapped is not None:
            return snapped, reason, 0
        return None, f"{asd_error};{reason}", 0
    return None, asd_error, 0


def snap_set(
    als_in: str,
    als_out: str,
    *,
    report_csv: Optional[str],
    max_asd_offset_ms: float,
    max_micro_offset_ms: float,
    min_delta_ms: float,
    microalign_fallback: bool,
    dry_run: bool,
) -> Dict[str, int]:
    root = ET.fromstring(gzip.open(als_in, "rb").read())
    stats = {
        "rows_total": 0,
        "rows_changed": 0,
        "rows_skipped_no_anchor": 0,
        "rows_skipped_no_audio": 0,
        "rows_skipped_no_bpm": 0,
        "rows_skipped_no_near_transient": 0,
        "clips_retimed": 0,
    }
    report_rows: List[Dict[str, object]] = []
    max_asd_offset_sec = float(max_asd_offset_ms) * 0.001
    max_micro_offset_sec = float(max_micro_offset_ms) * 0.001
    min_delta_sec = float(min_delta_ms) * 0.001

    for slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        drums = row["drums"]
        name = _clip_name(drums)
        current = _current_anchor_sec(drums)
        if current is None:
            stats["rows_skipped_no_anchor"] += 1
            continue

        bpm = _clip_bpm(drums)
        if not bpm:
            stats["rows_skipped_no_bpm"] += 1
            continue

        audio_path = _resolve_clip_audio_path(drums, als_in)
        if not audio_path:
            stats["rows_skipped_no_audio"] += 1
            continue

        snapped, source, marker_count = _choose_snap(
            audio_path,
            float(current),
            max_asd_offset_sec=max_asd_offset_sec,
            max_micro_offset_sec=max_micro_offset_sec,
            microalign_fallback=bool(microalign_fallback),
        )
        if snapped is None:
            stats["rows_skipped_no_near_transient"] += 1
            report_rows.append(
                {
                    "slot": slot_idx,
                    "name": name,
                    "current_sec": f"{float(current):.9f}",
                    "snapped_sec": "",
                    "offset_ms": "",
                    "source": source,
                    "marker_count": marker_count,
                    "changed": 0,
                }
            )
            continue

        offset_ms = (float(snapped) - float(current)) * 1000.0
        changed = abs(float(snapped) - float(current)) >= min_delta_sec
        if changed:
            end_sec = max((_last_marker_sec(clip) or 0.0) for clip in row.values())
            end_sec = max(float(end_sec), float(snapped) + 1.0)
            if not dry_run:
                for role in ("drums", "inst", "vocals"):
                    _replace_warp_markers(row[role], int(bpm), float(snapped), float(end_sec))
            stats["rows_changed"] += 1
            stats["clips_retimed"] += 3
            print(f"[ASD-SNAP] slot={slot_idx} {float(current):.6f}s -> {float(snapped):.6f}s ({offset_ms:+.1f}ms) name={name}")

        report_rows.append(
            {
                "slot": slot_idx,
                "name": name,
                "current_sec": f"{float(current):.9f}",
                "snapped_sec": f"{float(snapped):.9f}",
                "offset_ms": f"{offset_ms:.3f}",
                "source": source,
                "marker_count": marker_count,
                "changed": int(changed),
            }
        )

    if report_csv:
        os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)
        with open(report_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["slot", "name", "current_sec", "snapped_sec", "offset_ms", "source", "marker_count", "changed"],
            )
            writer.writeheader()
            writer.writerows(report_rows)

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
        with gzip.open(als_out, "wb") as fh:
            fh.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Snap Session View 1.1.1 anchors to nearby Ableton .asd transient markers.")
    parser.add_argument("--als", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-csv")
    parser.add_argument("--max-asd-offset-ms", type=float, default=160.0)
    parser.add_argument("--max-micro-offset-ms", type=float, default=140.0)
    parser.add_argument("--min-delta-ms", type=float, default=1.0)
    parser.add_argument("--microalign-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stats = snap_set(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        report_csv=os.path.abspath(args.report_csv) if args.report_csv else None,
        max_asd_offset_ms=float(args.max_asd_offset_ms),
        max_micro_offset_ms=float(args.max_micro_offset_ms),
        min_delta_ms=float(args.min_delta_ms),
        microalign_fallback=bool(args.microalign_fallback),
        dry_run=bool(args.dry_run),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
