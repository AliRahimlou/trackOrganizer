#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)


def _value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or "").strip()


def _set_value(node: Optional[ET.Element], value: object) -> None:
    if node is not None:
        node.set("Value", str(value))


def _clip_name(clip: ET.Element) -> str:
    return _value(clip.find("./Name"))


def _clip_role(clip: ET.Element) -> Optional[str]:
    m = ROLE_RE.match(_clip_name(clip))
    if not m:
        return None
    return m.group(1).lower()


def _clip_bpm(clip: ET.Element) -> Optional[int]:
    m = BPM_RE.match(_clip_name(clip))
    if not m:
        return None
    try:
        bpm = int(m.group(1))
    except Exception:
        return None
    return bpm if bpm > 0 else None


def _track_name(track: ET.Element) -> str:
    return _value(track.find("./Name/EffectiveName")) or _value(track.find("./Name/UserName"))


def _track_slots(track: ET.Element) -> List[ET.Element]:
    return track.findall("./DeviceChain/MainSequencer/ClipSlotList/ClipSlot")


def _slot_audio_clip(slot: ET.Element) -> Optional[ET.Element]:
    for path in ("./ClipSlot/Value/AudioClip", "./Value/AudioClip", ".//AudioClip"):
        clip = slot.find(path)
        if clip is not None:
            return clip
    return None


def _choose_tracks(root: ET.Element) -> List[ET.Element]:
    tracks = root.findall(".//AudioTrack")
    by_name = {_track_name(track): track for track in tracks}
    preferred = [by_name[name] for name in ("CH1", "CH2", "CH3") if name in by_name]
    return preferred if len(preferred) == 3 else tracks[:3]


def _iter_rows(root: ET.Element) -> List[Tuple[int, Dict[str, ET.Element]]]:
    tracks = _choose_tracks(root)
    if len(tracks) < 3:
        return []
    slot_lists = [_track_slots(track) for track in tracks]
    rows: List[Tuple[int, Dict[str, ET.Element]]] = []
    for slot_idx in range(max((len(slots) for slots in slot_lists), default=0)):
        row: Dict[str, ET.Element] = {}
        for slots in slot_lists:
            if slot_idx >= len(slots):
                continue
            clip = _slot_audio_clip(slots[slot_idx])
            if clip is None:
                continue
            role = _clip_role(clip)
            if role in {"drums", "inst", "vocals"} and role not in row:
                row[role] = clip
        if {"drums", "inst", "vocals"}.issubset(row):
            rows.append((slot_idx, row))
    return rows


def _warp_markers(clip: ET.Element) -> ET.Element:
    node = clip.find("./WarpMarkers")
    if node is None:
        node = ET.Element("WarpMarkers")
        clip.insert(0, node)
    return node


def _marker_pairs(clip: ET.Element) -> List[Tuple[ET.Element, float, float]]:
    out: List[Tuple[ET.Element, float, float]] = []
    for marker in _warp_markers(clip).findall("./WarpMarker"):
        try:
            sec = float(marker.get("SecTime"))
            beat = float(marker.get("BeatTime"))
        except Exception:
            continue
        out.append((marker, sec, beat))
    out.sort(key=lambda item: item[1])
    return out


def _beat0_sec(clip: ET.Element, min_sec: float) -> Optional[float]:
    pairs = _marker_pairs(clip)
    if len(pairs) < 2:
        return None
    if not any(beat < -1e-6 for _, _, beat in pairs):
        return None
    zero_secs = [sec for _, sec, beat in pairs if abs(beat) <= 1e-6 and sec > min_sec]
    return min(zero_secs) if zero_secs else None


def _beat_at(markers: Sequence[Tuple[ET.Element, float, float]], sec: float, bpm: Optional[int]) -> float:
    if not markers:
        return float(sec) * float(bpm or 128) / 60.0
    for _, m_sec, m_beat in markers:
        if abs(float(m_sec) - float(sec)) <= 1e-7:
            return float(m_beat)
    for idx in range(len(markers) - 1):
        _, a_sec, a_beat = markers[idx]
        _, b_sec, b_beat = markers[idx + 1]
        if (a_sec <= sec <= b_sec) or (b_sec <= sec <= a_sec):
            span = b_sec - a_sec
            if abs(span) <= 1e-9:
                return float(a_beat)
            t = (float(sec) - a_sec) / span
            return float(a_beat + t * (b_beat - a_beat))
    if len(markers) >= 2:
        a = markers[0]
        b = markers[1]
        if sec > markers[-1][1]:
            a = markers[-2]
            b = markers[-1]
        span = b[1] - a[1]
        if abs(span) > 1e-9:
            t = (float(sec) - a[1]) / span
            return float(a[2] + t * (b[2] - a[2]))
    return float(sec) * float(bpm or 128) / 60.0


def _next_marker_id(markers: Sequence[Tuple[ET.Element, float, float]]) -> int:
    ids: List[int] = []
    for marker, _, _ in markers:
        try:
            ids.append(int(marker.get("Id") or "0"))
        except Exception:
            pass
    return (max(ids) + 1) if ids else 0


def _make_marker(markers: Sequence[Tuple[ET.Element, float, float]], marker_id: int) -> ET.Element:
    if markers:
        marker = deepcopy(markers[-1][0])
    else:
        marker = ET.Element("WarpMarker")
    marker.set("Id", str(marker_id))
    return marker


def _sort_markers(container: ET.Element) -> None:
    nodes = container.findall("./WarpMarker")
    nodes.sort(key=lambda marker: float(marker.get("SecTime") or "inf"))
    for node in nodes:
        container.remove(node)
    for node in nodes:
        container.append(node)


def _retime_clip_to_anchor(clip: ET.Element, anchor_sec: float, bpm: Optional[int]) -> None:
    container = _warp_markers(clip)
    markers = _marker_pairs(clip)
    if not markers:
        markers = []
    offset = _beat_at(markers, anchor_sec, bpm)

    for marker, _, beat in markers:
        marker.set("BeatTime", f"{float(beat - offset):.9f}".rstrip("0").rstrip("."))

    markers = _marker_pairs(clip)
    next_id = _next_marker_id(markers)
    if not any(abs(sec) <= 1e-7 for _, sec, _ in markers):
        zero = _make_marker(markers, next_id)
        next_id += 1
        zero.set("SecTime", "0")
        zero.set("BeatTime", f"{float(_beat_at(markers, 0.0, bpm) - offset):.9f}".rstrip("0").rstrip("."))
        container.append(zero)
    if not any(abs(sec - anchor_sec) <= 1e-5 for _, sec, _ in markers):
        anchor = _make_marker(markers, next_id)
        anchor.set("SecTime", f"{float(anchor_sec):.9f}".rstrip("0").rstrip("."))
        anchor.set("BeatTime", "0")
        container.append(anchor)

    for marker, sec, _ in _marker_pairs(clip):
        if abs(sec - anchor_sec) <= 1e-5:
            marker.set("BeatTime", "0")

    is_warped = clip.find("./IsWarped")
    if is_warped is None:
        is_warped = ET.Element("IsWarped", {"Value": "true"})
        clip.insert(0, is_warped)
    else:
        _set_value(is_warped, "true")
    _sort_markers(container)


def _choose_anchor(row: Dict[str, ET.Element], min_sec: float, tolerance: float) -> Optional[float]:
    anchors: Dict[str, float] = {}
    for role in ("drums", "inst", "vocals"):
        sec = _beat0_sec(row[role], min_sec=min_sec)
        if sec is not None:
            anchors[role] = float(sec)
    if not anchors:
        return None

    vals = list(anchors.values())
    clusters: List[List[float]] = []
    for value in sorted(vals):
        if clusters and abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    clusters.sort(key=lambda item: (-len(item), abs(sum(item) / len(item) - anchors.get("drums", item[0]))))
    if len(clusters[0]) >= 2:
        return float(sum(clusters[0]) / len(clusters[0]))
    return anchors.get("drums") or vals[0]


def sync_als(als_in: str, als_out: str, min_sec: float, tolerance: float) -> Dict[str, int]:
    with gzip.open(als_in, "rb") as fh:
        root = ET.fromstring(fh.read())

    stats = {"rows_total": 0, "rows_with_anchor": 0, "rows_changed": 0, "clips_retimed": 0}
    for _, row in _iter_rows(root):
        stats["rows_total"] += 1
        anchor = _choose_anchor(row, min_sec=min_sec, tolerance=tolerance)
        if anchor is None:
            continue
        stats["rows_with_anchor"] += 1
        changed = False
        bpm = _clip_bpm(row["drums"])
        for role in ("drums", "inst", "vocals"):
            current = _beat0_sec(row[role], min_sec=min_sec)
            if current is None or abs(float(current) - float(anchor)) > tolerance:
                _retime_clip_to_anchor(row[role], anchor, bpm)
                stats["clips_retimed"] += 1
                changed = True
        if changed:
            stats["rows_changed"] += 1

    out_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
    with gzip.open(als_out, "wb") as fh:
        fh.write(out_xml)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize drums/inst/vocals 1.1.1 anchors inside an Ableton ALS file.")
    parser.add_argument("--als", required=True, help="Input .als path")
    parser.add_argument("--out", required=True, help="Output .als path")
    parser.add_argument("--min-sec", type=float, default=1.0, help="Ignore BeatTime=0 markers at or before this second.")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Seconds of tolerance for matching anchors.")
    args = parser.parse_args()
    stats = sync_als(args.als, args.out, min_sec=float(args.min_sec), tolerance=float(args.tolerance))
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
