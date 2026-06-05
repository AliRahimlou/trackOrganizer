#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

from rekordbox_mik_prior import lookup_first_drop_cue

try:
    from drop_aligner.microalign import microalign_marker
except Exception:
    microalign_marker = None


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)
ENERGY_LABEL_RE = re.compile(r"\benergy\s*([0-9]+)\b", re.I)
DROP_LABEL_RE = re.compile(r"\b(drop|chorus|hook|main|peak)\b", re.I)
NON_DROP_LABEL_RE = re.compile(r"\b(intro|outro|break|breakdown|build|buildup|verse)\b", re.I)


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
    return match.group(1).lower() if match else None


def _clip_bpm(clip: ET.Element) -> Optional[int]:
    match = BPM_RE.match(_clip_name(clip))
    if not match:
        return None
    try:
        bpm = int(match.group(1))
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


def _resolve_clip_audio_path(clip: ET.Element, als_path: str) -> Optional[str]:
    path_node = clip.find("./SampleRef/FileRef/Path")
    if path_node is not None:
        path = _value(path_node)
        if path and os.path.exists(path):
            return os.path.abspath(path)

    rel_node = clip.find("./SampleRef/FileRef/RelativePath")
    if rel_node is not None:
        rel = _value(rel_node)
        if rel:
            candidate = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(als_path)), rel)))
            if os.path.exists(candidate):
                return candidate
    return None


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


def _current_anchor_sec(clip: ET.Element, min_sec: float) -> Optional[float]:
    candidates = [(sec, beat) for _m, sec, beat in _marker_rows(clip) if sec > min_sec and abs(beat) <= 0.02]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (abs(item[1]), item[0]))
    return float(candidates[0][0])


def _last_marker_sec(clip: ET.Element) -> Optional[float]:
    rows = _marker_rows(clip)
    if rows:
        return max(float(sec) for _m, sec, _beat in rows)
    return None


def _sec_to_beats(sec: float, bpm: int) -> float:
    return (float(sec) * float(bpm)) / 60.0


def _replace_warp_markers(clip: ET.Element, bpm: int, drop_sec: float, end_sec: float) -> None:
    old = clip.find("./WarpMarkers")
    if old is not None:
        clip.remove(old)

    points = sorted(set(round(max(0.0, value), 6) for value in (0.0, float(drop_sec), float(end_sec))))
    phi = -_sec_to_beats(float(drop_sec), int(bpm))
    container = ET.Element("WarpMarkers")
    for idx, sec in enumerate(points):
        beat = _sec_to_beats(float(sec), int(bpm)) + phi
        container.append(
            ET.Element(
                "WarpMarker",
                {
                    "Id": str(idx),
                    "SecTime": f"{float(sec):.6f}",
                    "BeatTime": f"{float(beat):.6f}",
                },
            )
        )
    clip.insert(0, container)

    warped = clip.find("./IsWarped")
    if warped is None:
        clip.insert(0, ET.Element("IsWarped", {"Value": "true"}))
    else:
        warped.set("Value", "true")

    hidden_start = _fmt(phi)
    hidden_end = _fmt(phi + 4.0)
    loop = clip.find("./Loop")
    if loop is not None:
        _set_value(loop.find("./HiddenLoopStart"), hidden_start)
        _set_value(loop.find("./HiddenLoopEnd"), hidden_end)

    scroller = clip.find("./ScrollerTimePreserver")
    if scroller is not None:
        _set_value(scroller.find("./LeftTime"), hidden_start)
        current_end = _value(clip.find("./CurrentEnd"))
        if current_end:
            _set_value(scroller.find("./RightTime"), current_end)

    selection = clip.find("./TimeSelection")
    if selection is not None:
        _set_value(selection.find("./AnchorTime"), hidden_start)
        _set_value(selection.find("./OtherTime"), hidden_start)


def _snap_drop_to_audio(
    audio_path: str,
    cue_sec: float,
    *,
    enabled: bool,
    min_confidence: float,
    max_offset_ms: float,
) -> Tuple[float, str, float, float]:
    if not enabled or microalign_marker is None:
        return float(cue_sec), "cue", 0.0, 0.0
    try:
        result = microalign_marker(
            audio_path,
            float(cue_sec),
            search_before_ms=250.0,
            search_after_ms=500.0,
        )
    except Exception:
        return float(cue_sec), "cue_snap_failed", 0.0, 0.0

    aligned = result.get("microaligned_time")
    confidence = float(result.get("micro_confidence") or 0.0)
    offset_ms = float(result.get("snap_offset_ms") or 0.0)
    if aligned is None:
        return float(cue_sec), "cue_no_snap", confidence, offset_ms
    if confidence >= float(min_confidence) and abs(offset_ms) <= float(max_offset_ms):
        return float(aligned), "cue_microaligned", confidence, offset_ms
    return float(cue_sec), "cue_snap_low_conf", confidence, offset_ms


def _cue_energy(name: str) -> Optional[int]:
    match = ENERGY_LABEL_RE.search(name or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _is_explicit_drop_label(name: str) -> bool:
    if not DROP_LABEL_RE.search(name or ""):
        return False
    return not bool(NON_DROP_LABEL_RE.search(name or ""))


def refine_als(
    als_in: str,
    als_out: str,
    *,
    rekordbox_xml: str,
    min_existing_sec: float,
    min_delta_sec: float,
    snap_audio: bool,
    snap_min_confidence: float,
    snap_max_offset_ms: float,
    max_cue_delta_sec: float,
    max_missing_cue_sec: float,
    min_cue_energy: int,
    only_name: str,
    max_rows: int,
    dry_run: bool,
) -> Tuple[Dict[str, int], str]:
    with gzip.open(als_in, "rb") as fh:
        root = ET.fromstring(fh.read())

    stats = {
        "rows_total": 0,
        "rows_with_cue": 0,
        "rows_changed": 0,
        "rows_skipped_no_audio": 0,
        "rows_skipped_no_bpm": 0,
        "rows_skipped_no_cue": 0,
        "rows_skipped_far_cue": 0,
        "clips_retimed": 0,
    }
    review_rows: List[Dict[str, object]] = []

    for slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        drums = row["drums"]
        if only_name and only_name.lower() not in _clip_name(drums).lower():
            continue
        if max_rows > 0 and stats["rows_changed"] >= int(max_rows):
            break
        bpm = _clip_bpm(drums)
        if not bpm:
            stats["rows_skipped_no_bpm"] += 1
            continue

        drums_audio = _resolve_clip_audio_path(drums, als_in)
        if not drums_audio:
            stats["rows_skipped_no_audio"] += 1
            continue

        stem_paths: List[str] = []
        for role in ("drums", "inst", "vocals"):
            path = _resolve_clip_audio_path(row[role], als_in)
            if path:
                stem_paths.append(path)

        cue, _confidence, reason = lookup_first_drop_cue(
            xml_path=rekordbox_xml,
            track_dir=os.path.dirname(drums_audio),
            source_audio_path=drums_audio,
            stem_paths=stem_paths,
            preferred_num=1,
            allow_fuzzy=False,
        )
        if cue is None:
            stats["rows_skipped_no_cue"] += 1
            continue
        energy = _cue_energy(cue.name)
        if (
            int(min_cue_energy) > 0
            and not _is_explicit_drop_label(cue.name)
            and (energy is None or int(energy) < int(min_cue_energy))
        ):
            stats["rows_skipped_no_cue"] += 1
            continue
        stats["rows_with_cue"] += 1

        cue_sec = float(cue.start_sec)
        current_sec = _current_anchor_sec(drums, min_sec=float(min_existing_sec))
        cue_delta = None if current_sec is None else float(cue_sec) - float(current_sec)
        if current_sec is None and float(max_missing_cue_sec) > 0.0 and float(cue_sec) > float(max_missing_cue_sec):
            stats["rows_skipped_far_cue"] += 1
            continue
        if current_sec is not None and float(max_cue_delta_sec) > 0.0 and abs(cue_delta or 0.0) > float(max_cue_delta_sec):
            stats["rows_skipped_far_cue"] += 1
            continue

        drop_sec, snap_source, snap_conf, snap_offset_ms = _snap_drop_to_audio(
            drums_audio,
            cue_sec,
            enabled=bool(snap_audio),
            min_confidence=float(snap_min_confidence),
            max_offset_ms=float(snap_max_offset_ms),
        )

        delta = None if current_sec is None else float(drop_sec) - float(current_sec)
        if current_sec is not None and abs(delta or 0.0) < float(min_delta_sec):
            continue

        end_sec = max((_last_marker_sec(clip) or 0.0) for clip in row.values())
        if end_sec <= 0.0:
            continue
        end_sec = max(float(end_sec), float(drop_sec) + 1.0)

        if not dry_run:
            for role in ("drums", "inst", "vocals"):
                _replace_warp_markers(row[role], int(bpm), float(drop_sec), float(end_sec))

        stats["rows_changed"] += 1
        stats["clips_retimed"] += 3
        review_rows.append(
            {
                "slot": int(slot_idx),
                "name": _clip_name(drums),
                "audio_path": os.path.abspath(drums_audio),
                "bpm": int(bpm),
                "old_sec": "" if current_sec is None else f"{float(current_sec):.6f}",
                "cue_sec": f"{float(cue_sec):.6f}",
                "new_sec": f"{float(drop_sec):.6f}",
                "delta_sec": "" if delta is None else f"{float(delta):.6f}",
                "cue_name": cue.name,
                "cue_num": "" if cue.num is None else int(cue.num),
                "snap_source": snap_source,
                "snap_confidence": f"{float(snap_conf):.6f}",
                "snap_offset_ms": f"{float(snap_offset_ms):.3f}",
                "reason": reason,
            }
        )
        print(
            f"[MIK] slot={slot_idx} {float(current_sec or 0.0):.3f}s -> {float(drop_sec):.3f}s "
            f"cue={float(cue_sec):.3f}s {cue.name or 'cue'} snap={snap_source} name={_clip_name(drums)}"
        )

    report_path = os.path.splitext(os.path.abspath(als_out))[0] + "_mik_refine.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "slot",
            "name",
            "audio_path",
            "bpm",
            "old_sec",
            "cue_sec",
            "new_sec",
            "delta_sec",
            "cue_name",
            "cue_num",
            "snap_source",
            "snap_confidence",
            "snap_offset_ms",
            "reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    if not dry_run:
        out_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
        with gzip.open(als_out, "wb") as fh:
            fh.write(out_xml)

    return stats, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Retimes generated 1.1.1 anchors to the first Mixed In Key/Rekordbox drop cue.")
    parser.add_argument("--als", required=True, help="Input ALS path")
    parser.add_argument("--out", required=True, help="Output ALS path")
    parser.add_argument("--rekordbox-xml", default="/Users/alirahimlou/Documents/rekordbox_mikcues_001.xml")
    parser.add_argument("--min-existing-sec", type=float, default=1.0)
    parser.add_argument("--min-delta-sec", type=float, default=0.040, help="Skip rows whose existing marker is already this close.")
    parser.add_argument("--no-snap-audio", action="store_true", help="Use cue seconds exactly; do not micro-align to audio attacks.")
    parser.add_argument("--snap-min-confidence", type=float, default=0.70)
    parser.add_argument("--snap-max-offset-ms", type=float, default=180.0)
    parser.add_argument(
        "--max-cue-delta-sec",
        type=float,
        default=2.0,
        help="Skip cue refinements farther than this from the current marker (0 disables this guard).",
    )
    parser.add_argument(
        "--max-missing-cue-sec",
        type=float,
        default=120.0,
        help="When no current marker exists, skip first cue candidates later than this (0 disables this guard).",
    )
    parser.add_argument(
        "--min-cue-energy",
        type=int,
        default=6,
        help="Minimum Mixed In Key Energy N cue to apply, unless cue label explicitly says drop/chorus/hook/main/peak (0 disables).",
    )
    parser.add_argument("--only-name", default="", help="Only process rows whose drums clip name contains this text.")
    parser.add_argument("--max-rows", type=int, default=0, help="Stop after this many changed rows (0=all).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats, report = refine_als(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        rekordbox_xml=os.path.abspath(args.rekordbox_xml),
        min_existing_sec=float(args.min_existing_sec),
        min_delta_sec=float(args.min_delta_sec),
        snap_audio=not bool(args.no_snap_audio),
        snap_min_confidence=float(args.snap_min_confidence),
        snap_max_offset_ms=float(args.snap_max_offset_ms),
        max_cue_delta_sec=float(args.max_cue_delta_sec),
        max_missing_cue_sec=float(args.max_missing_cue_sec),
        min_cue_energy=int(args.min_cue_energy),
        only_name=str(args.only_name or ""),
        max_rows=int(args.max_rows),
        dry_run=bool(args.dry_run),
    )
    print(stats)
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
