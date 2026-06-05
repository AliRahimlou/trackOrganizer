#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Tuple

from apply_folder_drop_candidates_to_set import (
    _clip_bpm,
    _clip_name,
    _current_anchor_sec,
    _iter_triplet_rows,
    _last_marker_sec,
    _replace_warp_markers,
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def load_overrides(path: str) -> List[Dict[str, str]]:
    overrides: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.lstrip().startswith("#")):
            name_contains = (row.get("name_contains") or "").strip()
            drop_sec = (row.get("drop_sec") or "").strip()
            if not name_contains or not drop_sec:
                continue
            float(drop_sec)
            overrides.append(
                {
                    "name_contains": name_contains,
                    "drop_sec": drop_sec,
                    "source": (row.get("source") or "manual_override").strip(),
                }
            )
    overrides.sort(key=lambda item: len(_norm(item["name_contains"])), reverse=True)
    return overrides


def _match_override(name: str, overrides: Iterable[Dict[str, str]]) -> Dict[str, str] | None:
    normalized_name = _norm(name)
    for override in overrides:
        if _norm(override["name_contains"]) in normalized_name:
            return override
    return None


def apply_overrides(als_in: str, als_out: str, overrides_path: str, *, dry_run: bool) -> Dict[str, int]:
    overrides = load_overrides(overrides_path)
    root = ET.fromstring(gzip.open(als_in, "rb").read())
    stats = {
        "rows_total": 0,
        "overrides_loaded": len(overrides),
        "rows_matched": 0,
        "rows_changed": 0,
        "rows_skipped_no_bpm": 0,
        "clips_retimed": 0,
    }

    for slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        drums = row["drums"]
        override = _match_override(_clip_name(drums), overrides)
        if override is None:
            continue
        stats["rows_matched"] += 1

        bpm = _clip_bpm(drums)
        if not bpm:
            stats["rows_skipped_no_bpm"] += 1
            continue

        drop_sec = float(override["drop_sec"])
        current = _current_anchor_sec(drums)
        end_sec = max((_last_marker_sec(clip) or 0.0) for clip in row.values())
        end_sec = max(float(end_sec), drop_sec + 1.0)

        if not dry_run:
            for role in ("drums", "inst", "vocals"):
                _replace_warp_markers(row[role], int(bpm), drop_sec, float(end_sec))

        stats["rows_changed"] += 1
        stats["clips_retimed"] += 3
        print(
            f"[OVERRIDE] slot={slot_idx} {float(current or 0.0):.3f}s -> {drop_sec:.3f}s "
            f"source={override['source']} name={_clip_name(drums)}"
        )

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
        with gzip.open(als_out, "wb") as fh:
            fh.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply explicit first-drop overrides to CH1/CH2/CH3 Session View triplets.")
    parser.add_argument("--als", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = apply_overrides(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        os.path.abspath(args.overrides),
        dry_run=bool(args.dry_run),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
