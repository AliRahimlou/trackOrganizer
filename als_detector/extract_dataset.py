#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from als_detector.utils.als_io import extract_labeled_clips_from_als
from als_detector.utils.labels import collapse_rows_by_audio


def _collect_als_paths(explicit: List[str], globs: List[str]) -> List[str]:
    out: List[str] = []
    for p in explicit:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
    for patt in globs:
        for p in glob.glob(patt, recursive=True):
            if os.path.isfile(p) and p.lower().endswith(".als"):
                out.append(os.path.abspath(p))
    out = sorted(set(out))
    return out


def _dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Keep one row per (audio_path, rounded_target_sec), favor exact-beat source.
    by: Dict[Tuple[str, float], Dict[str, object]] = {}
    for r in rows:
        audio = str(r.get("audio_path", "")).strip()
        target = float(r.get("target_sec", 0.0))
        key = (audio, round(target, 3))
        prev = by.get(key)
        if prev is None:
            by[key] = r
            continue
        prev_src = str(prev.get("target_source", ""))
        cur_src = str(r.get("target_source", ""))
        if prev_src != "exact" and cur_src == "exact":
            by[key] = r
    return sorted(by.values(), key=lambda x: (str(x.get("audio_path", "")), float(x.get("target_sec", 0.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract labeled 1.1.1 targets from Ableton .als files.")
    ap.add_argument("--als", action="append", default=[], help="Explicit .als path (repeatable).")
    ap.add_argument("--als-glob", action="append", default=[], help="Glob for .als files, supports ** (repeatable).")
    ap.add_argument("--out", default="als_detector/data/labels.jsonl", help="Output JSONL path.")
    ap.add_argument("--include-unwarped", action="store_true", help="Include clips with IsWarped=false.")
    ap.add_argument("--no-resolve-audio", action="store_true", help="Do not resolve relative audio paths.")
    ap.add_argument(
        "--no-collapse-audio",
        action="store_true",
        help="Keep all clip-level rows. Default behavior collapses to one canonical row per audio file.",
    )
    args = ap.parse_args()

    als_paths = _collect_als_paths(explicit=args.als, globs=args.als_glob)
    if not als_paths:
        print("[ERROR] No .als files found. Provide --als or --als-glob.")
        return 2

    rows: List[Dict[str, object]] = []
    failed = 0
    for p in als_paths:
        try:
            clips = extract_labeled_clips_from_als(
                p,
                resolve_audio_paths=not args.no_resolve_audio,
                include_unwarped=args.include_unwarped,
            )
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")
            failed += 1
            continue
        for c in clips:
            rows.append(c.to_json())

    rows = _dedupe_rows(rows)
    exact_before = sum(1 for r in rows if str(r.get("target_source", "")) == "exact")
    if not args.no_collapse_audio:
        rows = collapse_rows_by_audio(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"[OK] Wrote {len(rows)} labels to {os.path.abspath(args.out)}")
    print(f"[OK] ALS parsed: {len(als_paths) - failed} / {len(als_paths)}")
    if rows:
        with_audio = sum(1 for r in rows if str(r.get("audio_path", "")).strip())
        exact = sum(1 for r in rows if str(r.get("target_source", "")) == "exact")
        print(f"[OK] Rows with audio path: {with_audio}")
        if args.no_collapse_audio:
            print(f"[OK] Exact BeatTime==0 markers: {exact}")
        else:
            print(f"[OK] Exact BeatTime==0 markers (pre-collapse): {exact_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
