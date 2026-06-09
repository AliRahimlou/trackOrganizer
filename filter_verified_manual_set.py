#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)


def _value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or "").strip()


def _track_name(track: ET.Element) -> str:
    return _value(track.find("./Name/EffectiveName")) or _value(track.find("./Name/UserName"))


def _clip_name(clip: ET.Element) -> str:
    return _value(clip.find("./Name"))


def _clip_role(clip: ET.Element) -> Optional[str]:
    match = ROLE_RE.match(_clip_name(clip))
    return match.group(1).lower() if match else None


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


def _marker_pairs(clip: ET.Element) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        try:
            pairs.append((float(marker.get("SecTime")), float(marker.get("BeatTime"))))
        except Exception:
            continue
    pairs.sort(key=lambda item: item[0])
    return pairs


def _has_manual_anchor(clip: ET.Element, min_sec: float) -> bool:
    pairs = _marker_pairs(clip)
    if len(pairs) < 3:
        return False
    if not any(beat < -1e-6 for _, beat in pairs):
        return False
    return any(sec > float(min_sec) and abs(beat) <= 1e-6 for sec, beat in pairs)


def _row_has_manual_anchor(row: Mapping[str, ET.Element], min_sec: float) -> bool:
    return any(_has_manual_anchor(clip, min_sec=min_sec) for clip in row.values())


def _load_als(path: str) -> ET.Element:
    with gzip.open(path, "rb") as fh:
        return ET.fromstring(fh.read())


def _live_set(root: ET.Element) -> ET.Element:
    if root.tag == "Ableton":
        live_set = root.find("./LiveSet")
        if live_set is not None:
            return live_set
    return root


def _manual_slots_from_source(path: str, min_sec: float) -> Set[int]:
    root = _load_als(path)
    keep: Set[int] = set()
    for slot_idx, row in _iter_triplet_rows(root):
        if _row_has_manual_anchor(row, min_sec=min_sec):
            keep.add(int(slot_idx))
    return keep


def _manual_slots_from_report(path: str, source_filter: Set[str], reviewed_from_filter: Set[str]) -> Set[int]:
    if not path:
        return set()
    keep: Set[int] = set()
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if source_filter and str(row.get("marker_source") or "") not in source_filter:
                continue
            if reviewed_from_filter and str(row.get("reviewed_from") or "") not in reviewed_from_filter:
                continue
            try:
                keep.add(int(row.get("slot") or ""))
            except Exception:
                continue
    return keep


def _remove_unkept_indexed_children(parent: Optional[ET.Element], tag: str, keep_slots: Set[int]) -> int:
    if parent is None:
        return 0
    children = parent.findall(f"./{tag}")
    removed = 0
    for index, child in reversed(list(enumerate(children))):
        if index not in keep_slots:
            parent.remove(child)
            removed += 1
    return removed


def _set_value(node: Optional[ET.Element], value: object) -> None:
    if node is not None:
        node.set("Value", str(value))


def _blank_slot(slot: ET.Element) -> bool:
    changed = False
    for value in slot.findall("./ClipSlot/Value") + slot.findall("./Value"):
        if list(value):
            for child in list(value):
                value.remove(child)
            changed = True
        if value.text:
            value.text = None
            changed = True
    _set_value(slot.find("./HasStop"), "true")
    _set_value(slot.find("./NeedRefreeze"), "true")
    return changed


def _clear_scene(scene: ET.Element) -> bool:
    changed = False
    name = scene.find("./Name")
    if name is not None and name.get("Value"):
        name.set("Value", "")
        changed = True
    annotation = scene.find("./Annotation")
    if annotation is not None and annotation.get("Value"):
        annotation.set("Value", "")
        changed = True
    return changed


def _blank_unkept_indexed_children(parent: Optional[ET.Element], tag: str, keep_slots: Set[int]) -> int:
    if parent is None:
        return 0
    changed = 0
    for index, child in enumerate(parent.findall(f"./{tag}")):
        if index in keep_slots:
            continue
        if tag == "ClipSlot":
            if _blank_slot(child):
                changed += 1
        elif tag == "Scene":
            if _clear_scene(child):
                changed += 1
    return changed


def _filter_rows(
    root: ET.Element,
    keep_slots: Set[int],
    *,
    mode: str,
    blank_extra_audio_tracks: bool,
    preserve_extra_audio_tracks: bool,
) -> Dict[str, int]:
    live_set = _live_set(root)
    scenes = live_set.find("./Scenes")
    original_scenes = len(scenes.findall("./Scene")) if scenes is not None else 0
    if mode == "blank":
        removed_scenes = 0
        blanked_scenes = _blank_unkept_indexed_children(scenes, "Scene", keep_slots)
    else:
        blanked_scenes = 0
        removed_scenes = _remove_unkept_indexed_children(scenes, "Scene", keep_slots)

    stats = {
        "original_scenes": int(original_scenes),
        "kept_scenes": int(original_scenes - removed_scenes),
        "removed_scenes": int(removed_scenes),
        "blanked_scenes": int(blanked_scenes),
        "track_slot_lists_filtered": 0,
        "track_slots_removed": 0,
        "track_slots_blanked": 0,
        "extra_track_slots_blanked": 0,
        "extra_track_slot_lists_preserved": 0,
    }
    tracks_parent = live_set.find("./Tracks")
    if tracks_parent is None:
        return stats

    for track in list(tracks_parent):
        if track.tag not in {"AudioTrack", "MidiTrack", "GroupTrack"}:
            continue
        track_name = _track_name(track)
        if preserve_extra_audio_tracks and track.tag == "AudioTrack" and track_name not in {"CH1", "CH2", "CH3"}:
            stats["extra_track_slot_lists_preserved"] += len(track.findall(".//ClipSlotList"))
            continue
        if blank_extra_audio_tracks and track.tag == "AudioTrack" and track_name not in {"CH1", "CH2", "CH3"}:
            for csl in track.findall(".//ClipSlotList"):
                for slot in csl.findall("./ClipSlot"):
                    if _blank_slot(slot):
                        stats["extra_track_slots_blanked"] += 1
            continue
        for csl in track.findall(".//ClipSlotList"):
            # Return/folded/internal lists may have independent lengths. Only lists
            # that matched the original scene count are scene-row aligned.
            if len(csl.findall("./ClipSlot")) != original_scenes:
                continue
            if mode == "blank":
                changed = _blank_unkept_indexed_children(csl, "ClipSlot", keep_slots)
                removed = 0
            else:
                removed = _remove_unkept_indexed_children(csl, "ClipSlot", keep_slots)
                changed = 0
            stats["track_slot_lists_filtered"] += 1
            stats["track_slots_removed"] += int(removed)
            stats["track_slots_blanked"] += int(changed)
    return stats


def _write_kept_report(path: str, root: ET.Element) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["row", "name"])
        writer.writeheader()
        for row_idx, row in _iter_triplet_rows(root):
            writer.writerow({"row": row_idx, "name": _clip_name(row["drums"])})


def filter_verified_manual_set(
    als_in: str,
    als_out: str,
    *,
    source_manual_als: str,
    manual_report_csv: str,
    manual_report_sources: Iterable[str],
    reviewed_from_values: Iterable[str],
    include_source_manual: bool,
    mode: str,
    blank_extra_audio_tracks: bool,
    preserve_extra_audio_tracks: bool,
    min_sec: float,
    kept_report_csv: str,
) -> Dict[str, int]:
    source_slots = _manual_slots_from_source(source_manual_als, min_sec=min_sec) if include_source_manual else set()
    keep_slots = set(source_slots)
    report_slots = _manual_slots_from_report(
        manual_report_csv,
        {str(source) for source in manual_report_sources if str(source)},
        {str(value) for value in reviewed_from_values if str(value)},
    )
    keep_slots.update(report_slots)

    root = _load_als(als_in)
    stats = {
        "source_manual_slots": len(source_slots),
        "manual_report_slots": len(report_slots),
        "keep_slots": len(keep_slots),
    }
    stats.update(
        _filter_rows(
            root,
            keep_slots,
            mode=mode,
            blank_extra_audio_tracks=blank_extra_audio_tracks,
            preserve_extra_audio_tracks=preserve_extra_audio_tracks,
        )
    )

    os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
    with gzip.open(als_out, "wb") as fh:
        fh.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    if kept_report_csv:
        _write_kept_report(kept_report_csv, root)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an Ableton set containing only manually verified 1.1.1 rows.")
    parser.add_argument("--als", required=True, help="Input ALS to filter, usually the manual-applied copy.")
    parser.add_argument("--out", required=True, help="Filtered output ALS.")
    parser.add_argument("--source-manual-als", required=True, help="Original ALS whose existing anchors count as manual.")
    parser.add_argument("--manual-report-csv", required=True, help="Manual/web marker report CSV from apply_manual_111_to_set.py.")
    parser.add_argument(
        "--manual-report-source",
        action="append",
        default=[],
        help="Only include manual report rows with this marker_source. Can be repeated. Default includes all.",
    )
    parser.add_argument(
        "--reviewed-from",
        action="append",
        default=[],
        help="Only include manual report rows with this reviewed_from value. Can be repeated. Default includes all.",
    )
    parser.add_argument(
        "--no-source-manual",
        action="store_true",
        help="Do not include existing anchors from --source-manual-als; keep only report-matched rows.",
    )
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=["remove", "blank"],
        default="remove",
        help="remove compresses rows; blank keeps original scene count and clears unkept clips.",
    )
    parser.add_argument(
        "--blank-extra-audio-tracks",
        action="store_true",
        help="Clear all clips from audio tracks outside CH1/CH2/CH3.",
    )
    parser.add_argument(
        "--preserve-extra-audio-tracks",
        action="store_true",
        help="Do not filter or blank audio tracks outside CH1/CH2/CH3.",
    )
    parser.add_argument("--kept-report-csv", default="")
    args = parser.parse_args()
    kept_report = args.kept_report_csv
    if not kept_report:
        base, _ext = os.path.splitext(os.path.abspath(args.out))
        kept_report = f"{base}_kept_rows.csv"
    stats = filter_verified_manual_set(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        source_manual_als=os.path.abspath(args.source_manual_als),
        manual_report_csv=os.path.abspath(args.manual_report_csv),
        manual_report_sources=args.manual_report_source,
        reviewed_from_values=args.reviewed_from,
        include_source_manual=not bool(args.no_source_manual),
        mode=str(args.mode),
        blank_extra_audio_tracks=bool(args.blank_extra_audio_tracks),
        preserve_extra_audio_tracks=bool(args.preserve_extra_audio_tracks),
        min_sec=float(args.min_sec),
        kept_report_csv=os.path.abspath(kept_report),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
