#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a reusable drop-marker database from one or more Ableton .als files.

The DB stores manually set 1.1.1 anchors (BeatTime ~= 0 warp marker) keyed by
drums stem filename so future runs can reuse ground-truth drop times.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


ROLE_PREFIX_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_KEY_PREFIX_RE = re.compile(r"^\d{2,3}_[0-9]{1,2}[ab](?:_\d{1,2})?[-_]", re.I)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
BPM_FROM_EXACT_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)


def exact_key(path_or_name: str) -> str:
    base = os.path.basename((path_or_name or "").strip())
    stem, _ = os.path.splitext(base)
    return stem.lower().strip()


def slug_key(exact: str) -> str:
    s = (exact or "").lower().strip()
    s = ROLE_PREFIX_RE.sub("", s)
    s = BPM_KEY_PREFIX_RE.sub("", s)
    s = NON_ALNUM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def bpm_from_exact(exact: str) -> Optional[int]:
    m = BPM_FROM_EXACT_RE.match(exact or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    return v if v > 0 else None


def _iter_audio_clips(root: ET.Element) -> Iterable[ET.Element]:
    for clip in root.iter("AudioClip"):
        yield clip


def _clip_name(clip: ET.Element) -> str:
    n = clip.find("Name")
    if n is not None:
        v = n.get("Value")
        if v:
            return v
    n = clip.find("EffectiveName")
    if n is not None:
        v = n.get("Value")
        if v:
            return v
    return ""


def _clip_file_path(clip: ET.Element) -> str:
    p = clip.find(".//SampleRef/FileRef/Path")
    if p is not None:
        v = p.get("Value")
        if v:
            return v
    p = clip.find(".//SampleRef/FileRef/RelativePath")
    if p is not None:
        v = p.get("Value")
        if v:
            return v
    return ""


def _warp_markers(clip: ET.Element) -> List[Tuple[float, float]]:
    wm = clip.find("WarpMarkers")
    if wm is None:
        return []
    out: List[Tuple[float, float]] = []
    for m in wm.findall("WarpMarker"):
        try:
            sec = float(m.get("SecTime"))
            beat = float(m.get("BeatTime"))
        except Exception:
            continue
        out.append((sec, beat))
    return out


def _robust_center(vals: List[float]) -> Tuple[float, int]:
    if not vals:
        return 0.0, 0
    v = sorted(float(x) for x in vals)
    if len(v) < 4:
        return float(statistics.median(v)), len(v)

    try:
        q1, _, q3 = statistics.quantiles(v, n=4, method="inclusive")
    except Exception:
        q1, q3 = v[len(v) // 4], v[(3 * len(v)) // 4]
    iqr = max(0.0, q3 - q1)
    lo = q1 - (1.5 * iqr)
    hi = q3 + (1.5 * iqr)
    kept = [x for x in v if lo <= x <= hi]
    if not kept:
        kept = v
    return float(statistics.median(kept)), len(kept)


def parse_als_labels(als_path: str) -> Dict[str, List[Dict[str, object]]]:
    with gzip.open(als_path, "rb") as f:
        root = ET.fromstring(f.read())

    out: Dict[str, List[Dict[str, object]]] = {}
    for clip in _iter_audio_clips(root):
        name = _clip_name(clip)
        ref_path = _clip_file_path(clip)
        ref = ref_path or name
        if not ref:
            continue

        ex = exact_key(ref)
        if not ex:
            continue
        if not ex.startswith("drums_"):
            # Only use drums clips for drop ground truth.
            continue

        markers = _warp_markers(clip)
        if not markers:
            continue

        beat0 = [sec for sec, beat in markers if abs(float(beat)) <= 1e-4]
        if not beat0:
            continue

        rec = {
            "drop_sec": float(sorted(beat0)[0]),
            "ref_path": ref_path,
            "clip_name": name,
            "source_als": als_path,
        }
        out.setdefault(ex, []).append(rec)
    return out


def build_db(als_files: List[str]) -> Dict[str, object]:
    merged: Dict[str, List[Dict[str, object]]] = {}
    parsed_clips = 0
    for als in als_files:
        labels = parse_als_labels(als)
        for ex, rows in labels.items():
            parsed_clips += len(rows)
            merged.setdefault(ex, []).extend(rows)

    exact: Dict[str, Dict[str, object]] = {}
    slug: Dict[str, List[Dict[str, object]]] = {}
    for ex, rows in merged.items():
        vals = [float(r["drop_sec"]) for r in rows]
        drop_sec, used = _robust_center(vals)
        sk = slug_key(ex)
        bpm = bpm_from_exact(ex)
        obs = len(vals)
        src_paths = sorted({str(r.get("ref_path") or "") for r in rows if str(r.get("ref_path") or "")})
        ex_row = {
            "drop_sec": round(float(drop_sec), 6),
            "obs": int(obs),
            "obs_used": int(used),
            "bpm": bpm,
            "slug": sk,
            "sample_path": src_paths[0] if src_paths else "",
        }
        exact[ex] = ex_row
        slug.setdefault(sk, []).append({
            "exact_key": ex,
            "drop_sec": ex_row["drop_sec"],
            "obs": ex_row["obs"],
            "bpm": bpm,
        })

    for sk, rows in slug.items():
        rows.sort(key=lambda r: (-int(r.get("obs") or 0), str(r.get("exact_key") or "")))

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_als": sorted(als_files),
        "stats": {
            "drums_clips_with_beat0": int(parsed_clips),
            "unique_exact_keys": int(len(exact)),
            "unique_slug_keys": int(len(slug)),
        },
        "exact": exact,
        "slug": slug,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build drop-marker DB from manually cued Ableton .als files.")
    ap.add_argument("--als", nargs="+", required=True, help="One or more .als files to parse.")
    ap.add_argument("--out", default="drop_marker_db.json", help="Output JSON path.")
    args = ap.parse_args()

    als_files = [os.path.abspath(p) for p in args.als]
    missing = [p for p in als_files if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] Missing ALS: {p}")
        return 2

    db = build_db(als_files)
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")

    st = db.get("stats", {})
    print(f"[OK] Wrote drop DB: {out_path}")
    print(f"[OK] drums_clips_with_beat0={st.get('drums_clips_with_beat0', 0)}")
    print(f"[OK] unique_exact_keys={st.get('unique_exact_keys', 0)}")
    print(f"[OK] unique_slug_keys={st.get('unique_slug_keys', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

