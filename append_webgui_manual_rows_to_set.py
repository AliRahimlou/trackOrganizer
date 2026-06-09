#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)


def _value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or "").strip()


def _track_name(track: ET.Element) -> str:
    return _value(track.find("./Name/EffectiveName")) or _value(track.find("./Name/UserName")) or track.tag


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


def _load_als(path: str) -> ET.Element:
    with gzip.open(path, "rb") as fh:
        return ET.fromstring(fh.read())


def _live_set(root: ET.Element) -> ET.Element:
    if root.tag == "Ableton":
        live_set = root.find("./LiveSet")
        if live_set is not None:
            return live_set
    return root


def _choose_triplet_tracks(root: ET.Element) -> List[ET.Element]:
    tracks = root.findall(".//AudioTrack")
    by_name = {_track_name(track): track for track in tracks}
    preferred = [by_name[name] for name in ("CH1", "CH2", "CH3") if name in by_name]
    return preferred if len(preferred) == 3 else tracks[:3]


def _iter_triplet_rows(root: ET.Element) -> List[Tuple[int, Dict[str, ET.Element]]]:
    tracks = _choose_triplet_tracks(root)
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


def _track_key(name: str) -> str:
    text = str(name or "").lower().strip()
    text = re.sub(r"^(?:drums|inst|vocals)_\d{2,3}_[0-9]{1,2}[ab]_(?:\d+[-_])?", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _existing_keys(root: ET.Element) -> Set[str]:
    keys: Set[str] = set()
    for _slot_idx, row in _iter_triplet_rows(root):
        key = _track_key(_clip_name(row["drums"]))
        if key:
            keys.add(key)
    return keys


def _report_rows(path: str, *, marker_source: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if marker_source and str(row.get("marker_source") or "") != marker_source:
                continue
            try:
                int(row.get("slot") or "")
            except Exception:
                continue
            rows.append(dict(row))
    rows.sort(key=lambda row: int(row.get("slot") or "0"))
    return rows


def _scene_count(root: ET.Element) -> int:
    scenes = _live_set(root).find("./Scenes")
    return len(scenes.findall("./Scene")) if scenes is not None else 0


def _scene_aligned_lists(root: ET.Element) -> List[Tuple[str, ET.Element]]:
    count = _scene_count(root)
    out: List[Tuple[str, ET.Element]] = []
    live_set = _live_set(root)
    tracks = live_set.find("./Tracks")
    if tracks is None:
        return out
    for track in list(tracks):
        if track.tag not in {"AudioTrack", "MidiTrack", "GroupTrack"}:
            continue
        track_name = _track_name(track)
        for csl in track.findall(".//ClipSlotList"):
            if len(csl.findall("./ClipSlot")) == count:
                out.append((track_name, csl))
    return out


def _next_attr_id(parent: ET.Element, tag: Optional[str] = None) -> int:
    ids: List[int] = []
    children = parent.findall(f"./{tag}") if tag else list(parent)
    for child in children:
        try:
            ids.append(int(child.get("Id") or "0"))
        except Exception:
            continue
    return (max(ids) + 1) if ids else 0


def _copy_with_new_id(node: ET.Element, parent: ET.Element, tag: Optional[str] = None) -> ET.Element:
    copied = deepcopy(node)
    if "Id" in copied.attrib:
        copied.set("Id", str(_next_attr_id(parent, tag=tag or copied.tag)))
    return copied


def _blank_slot_like(parent: ET.Element) -> ET.Element:
    slots = parent.findall("./ClipSlot")
    if slots:
        slot = deepcopy(slots[-1])
        if "Id" in slot.attrib:
            slot.set("Id", str(_next_attr_id(parent, tag="ClipSlot")))
        for value in slot.findall("./ClipSlot/Value") + slot.findall("./Value"):
            for child in list(value):
                value.remove(child)
            value.text = None
        has_stop = slot.find("./HasStop")
        if has_stop is not None:
            has_stop.set("Value", "true")
        need_refreeze = slot.find("./NeedRefreeze")
        if need_refreeze is not None:
            need_refreeze.set("Value", "true")
        return slot
    return ET.Element("ClipSlot", {"Id": str(_next_attr_id(parent, tag="ClipSlot"))})


def _scene_name(scene: ET.Element) -> str:
    return _value(scene.find("./Name"))


def _append_rows(
    dest_root: ET.Element,
    source_root: ET.Element,
    source_slots: Sequence[int],
    *,
    preserve_extra_audio_tracks: bool,
) -> List[Dict[str, str]]:
    dest_live = _live_set(dest_root)
    source_live = _live_set(source_root)
    dest_scenes = dest_live.find("./Scenes")
    source_scenes = source_live.find("./Scenes")
    if dest_scenes is None or source_scenes is None:
        raise ValueError("Missing Scenes node.")

    dest_lists = _scene_aligned_lists(dest_root)
    source_lists = _scene_aligned_lists(source_root)
    if not dest_lists:
        raise ValueError("No destination scene-aligned ClipSlotList nodes found.")
    if len(dest_lists) != len(source_lists):
        raise ValueError(f"Scene-aligned list count mismatch: dest={len(dest_lists)} source={len(source_lists)}")

    source_scene_nodes = source_scenes.findall("./Scene")
    appended: List[Dict[str, str]] = []
    for source_slot in source_slots:
        if source_slot < 0 or source_slot >= len(source_scene_nodes):
            continue
        source_scene = source_scene_nodes[source_slot]
        new_scene = _copy_with_new_id(source_scene, dest_scenes, tag="Scene")
        dest_scenes.append(new_scene)

        for (dest_track_name, dest_list), (_source_track_name, source_list) in zip(dest_lists, source_lists):
            source_slot_nodes = source_list.findall("./ClipSlot")
            copy_source_slot = (not preserve_extra_audio_tracks) or dest_track_name in {"CH1", "CH2", "CH3"}
            if copy_source_slot and source_slot < len(source_slot_nodes):
                new_slot = _copy_with_new_id(source_slot_nodes[source_slot], dest_list, tag="ClipSlot")
            else:
                new_slot = _blank_slot_like(dest_list)
            dest_list.append(new_slot)

        appended.append({"source_slot": str(source_slot), "scene_name": _scene_name(source_scene)})
    return appended


def _write_report(path: str, rows: Iterable[Mapping[str, str]]) -> None:
    fields = ["source_slot", "scene_name", "reviewed_from", "drop_sec", "name"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_webgui_manual_rows(
    dest_als: str,
    source_als: str,
    report_csv: str,
    out_als: str,
    *,
    marker_source: str,
    report_out: str,
    preserve_extra_audio_tracks: bool,
) -> Dict[str, int]:
    dest_root = _load_als(dest_als)
    source_root = _load_als(source_als)

    existing = _existing_keys(dest_root)
    report_rows = _report_rows(report_csv, marker_source=marker_source)
    missing_rows = [row for row in report_rows if _track_key(row.get("name", "")) not in existing]
    missing_slots = [int(row["slot"]) for row in missing_rows]

    appended = _append_rows(
        dest_root,
        source_root,
        missing_slots,
        preserve_extra_audio_tracks=preserve_extra_audio_tracks,
    )
    detail_rows: List[Dict[str, str]] = []
    by_slot = {int(row["slot"]): row for row in missing_rows}
    for row in appended:
        source_slot = int(row["source_slot"])
        report = by_slot.get(source_slot, {})
        detail = dict(row)
        detail.update(
            {
                "reviewed_from": str(report.get("reviewed_from") or ""),
                "drop_sec": str(report.get("drop_sec") or ""),
                "name": str(report.get("name") or ""),
            }
        )
        detail_rows.append(detail)

    os.makedirs(os.path.dirname(os.path.abspath(out_als)), exist_ok=True)
    with gzip.open(out_als, "wb") as fh:
        fh.write(ET.tostring(dest_root, encoding="utf-8", xml_declaration=True))
    if report_out:
        _write_report(report_out, detail_rows)
    return {
        "webgui_report_rows": len(report_rows),
        "already_present": len(report_rows) - len(missing_rows),
        "missing_appended": len(detail_rows),
        "dest_original_scenes": _scene_count(_load_als(dest_als)),
        "dest_output_scenes": _scene_count(dest_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append missing web-GUI verified manual 1.1.1 rows into an existing Ableton set.")
    parser.add_argument("--dest-als", required=True, help="Existing ALS to start from. Not modified.")
    parser.add_argument("--source-als", required=True, help="ALS containing source rows to copy from.")
    parser.add_argument("--manual-report-csv", required=True, help="Manual marker report from apply_manual_111_to_set.py.")
    parser.add_argument("--out", required=True, help="Output ALS path.")
    parser.add_argument("--marker-source", default="correction_log")
    parser.add_argument("--report-out", default="")
    parser.add_argument(
        "--preserve-extra-audio-tracks",
        action="store_true",
        help="Append blank slots for tracks outside CH1/CH2/CH3 instead of copying source content.",
    )
    args = parser.parse_args()
    report_out = args.report_out
    if not report_out:
        base, _ext = os.path.splitext(os.path.abspath(args.out))
        report_out = f"{base}_appended_rows.csv"
    stats = append_webgui_manual_rows(
        os.path.abspath(args.dest_als),
        os.path.abspath(args.source_als),
        os.path.abspath(args.manual_report_csv),
        os.path.abspath(args.out),
        marker_source=str(args.marker_source),
        report_out=os.path.abspath(report_out),
        preserve_extra_audio_tracks=bool(args.preserve_extra_audio_tracks),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
