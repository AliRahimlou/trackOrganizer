#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)


def _value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or "").strip()


def _set_value(node: Optional[ET.Element], value: object) -> bool:
    if node is None:
        return False
    text = str(value)
    if node.get("Value") == text:
        return False
    node.set("Value", text)
    return True


def _fmt(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", "-0.0", ""} else text


def _clip_name(clip: ET.Element) -> str:
    return _value(clip.find("./Name"))


def _clip_role(clip: ET.Element) -> Optional[str]:
    match = ROLE_RE.match(_clip_name(clip))
    if not match:
        return None
    return match.group(1).lower()


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


def _iter_triplet_rows(root: ET.Element) -> List[Tuple[int, Dict[str, ET.Element]]]:
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


def _marker_rows(clip: ET.Element) -> List[Tuple[ET.Element, float, float]]:
    container = clip.find("./WarpMarkers")
    if container is None:
        return []
    rows: List[Tuple[ET.Element, float, float]] = []
    for marker in container.findall("./WarpMarker"):
        try:
            rows.append((marker, float(marker.get("SecTime")), float(marker.get("BeatTime"))))
        except Exception:
            continue
    rows.sort(key=lambda item: item[1])
    return rows


def _anchor_marker(clip: ET.Element, min_sec: float) -> Optional[Tuple[ET.Element, float, float]]:
    candidates = [(m, sec, beat) for m, sec, beat in _marker_rows(clip) if sec > min_sec and abs(beat) <= 0.01]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (abs(item[2]), item[1]))
    return candidates[0]


def _zero_sec_marker(clip: ET.Element) -> Optional[Tuple[ET.Element, float, float]]:
    rows = _marker_rows(clip)
    candidates = [(m, sec, beat) for m, sec, beat in rows if abs(sec) <= 1e-7]
    if candidates:
        return candidates[0]
    return rows[0] if rows else None


def _repair_clip(clip: ET.Element, min_sec: float) -> bool:
    anchor = _anchor_marker(clip, min_sec=min_sec)
    zero = _zero_sec_marker(clip)
    if anchor is None or zero is None:
        return False

    changed = False
    anchor_marker, _anchor_sec, _anchor_beat = anchor
    zero_marker, _zero_sec, zero_beat = zero
    hidden_start = float(zero_beat)
    if hidden_start >= -1e-6:
        return False

    if anchor_marker.get("BeatTime") != "0":
        anchor_marker.set("BeatTime", "0")
        changed = True
    hidden_text = _fmt(hidden_start)
    hidden_end_text = _fmt(hidden_start + 4.0)

    loop = clip.find("./Loop")
    if loop is not None:
        changed = _set_value(loop.find("./HiddenLoopStart"), hidden_text) or changed
        changed = _set_value(loop.find("./HiddenLoopEnd"), hidden_end_text) or changed

    scroller = clip.find("./ScrollerTimePreserver")
    if scroller is not None:
        changed = _set_value(scroller.find("./LeftTime"), hidden_text) or changed
        current_end = _value(clip.find("./CurrentEnd"))
        if current_end:
            changed = _set_value(scroller.find("./RightTime"), current_end) or changed

    selection = clip.find("./TimeSelection")
    if selection is not None:
        changed = _set_value(selection.find("./AnchorTime"), hidden_text) or changed
        changed = _set_value(selection.find("./OtherTime"), hidden_text) or changed

    return changed


def repair_als(als_in: str, als_out: str, min_sec: float) -> Dict[str, int]:
    with gzip.open(als_in, "rb") as fh:
        root = ET.fromstring(fh.read())

    stats = {"rows_total": 0, "rows_repaired": 0, "clips_repaired": 0}
    for _slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        repaired = 0
        for role in ("drums", "inst", "vocals"):
            if _repair_clip(row[role], min_sec=min_sec):
                repaired += 1
        if repaired:
            stats["rows_repaired"] += 1
            stats["clips_repaired"] += repaired

    out_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
    with gzip.open(als_out, "wb") as fh:
        fh.write(out_xml)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Live-visible clip loop/start display around generated 1.1.1 drop markers.")
    parser.add_argument("--als", required=True, help="Input ALS path")
    parser.add_argument("--out", required=True, help="Output ALS path")
    parser.add_argument("--min-sec", type=float, default=1.0, help="Ignore BeatTime=0 markers at or before this second.")
    args = parser.parse_args()
    print(repair_als(args.als, args.out, min_sec=float(args.min_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
