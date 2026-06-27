#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from drop_aligner.als import _duration_info
from drop_aligner.legacy_write_guard import add_legacy_detector_write_arg, require_legacy_detector_write_opt_in


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


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
    path = _value(clip.find("./SampleRef/FileRef/Path"))
    if path and os.path.exists(path):
        return os.path.abspath(path)
    rel = _value(clip.find("./SampleRef/FileRef/RelativePath"))
    if rel:
        candidate = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(als_path)), rel)))
        if os.path.exists(candidate):
            return candidate
    return None


def _as_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out >= 0.0 else None


def _candidate_json_paths(audio_path: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(audio_path))
    base = os.path.splitext(os.path.basename(audio_path))[0]
    exact = os.path.join(folder, f"{base}_drop_candidates.json")
    paths: List[str] = []
    if os.path.exists(exact):
        paths.append(exact)
    for path in sorted(glob.glob(os.path.join(folder, "*_drop_candidates.json"))):
        if path not in paths:
            paths.append(path)
    return paths


def _drop_sec_from_json(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    selected = data.get("selected_candidate")
    if isinstance(selected, dict):
        for key in ("microaligned_time", "snapped_sec", "timestamp", "time_abs_sec"):
            sec = _as_float(selected.get(key))
            if sec is not None and sec > 1.0:
                return float(sec)

    for key in ("final_ai_pick", "downbeat_seconds", "drop_sec"):
        sec = _as_float(data.get(key))
        if sec is not None and sec > 1.0:
            return float(sec)

    candidates = data.get("top_10_candidates") or data.get("candidates") or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("microaligned_time", "snapped_sec", "timestamp", "time_abs_sec"):
                sec = _as_float(candidate.get(key))
                if sec is not None and sec > 1.0:
                    return float(sec)
    return None


def _drop_sec_from_local_aligned_als(audio_path: str) -> Optional[float]:
    folder = os.path.dirname(os.path.abspath(audio_path))
    base = os.path.splitext(os.path.basename(audio_path))[0]
    paths = [os.path.join(folder, f"{base}_DROP_ALIGNED.als")]
    paths.extend(sorted(glob.glob(os.path.join(folder, "*_DROP_ALIGNED.als"))))
    seen = set()
    for path in paths:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            root = ET.fromstring(gzip.open(path, "rb").read())
        except Exception:
            continue
        best: Optional[float] = None
        for clip in root.findall(".//AudioClip"):
            clip_name = _clip_name(clip)
            if clip_name and _nfc(base) not in _nfc(clip_name):
                continue
            for marker in clip.findall("./WarpMarkers/WarpMarker"):
                try:
                    sec = float(marker.get("SecTime"))
                    beat = float(marker.get("BeatTime"))
                except Exception:
                    continue
                if sec > 1.0 and abs(beat) <= 0.02:
                    if best is None or sec < best:
                        best = float(sec)
        if best is not None:
            return best
    return None


def _local_drop_sec(audio_path: str) -> Tuple[Optional[float], str]:
    sec = _drop_sec_from_local_aligned_als(audio_path)
    if sec is not None:
        return float(sec), "local_DROP_ALIGNED.als"
    for path in _candidate_json_paths(audio_path):
        sec = _drop_sec_from_json(path)
        if sec is not None:
            return float(sec), os.path.basename(path)
    return None, ""


def _sec_to_beats(sec: float, bpm: int) -> float:
    return (float(sec) * float(bpm)) / 60.0


def _last_marker_sec(clip: ET.Element) -> Optional[float]:
    vals: List[float] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        try:
            vals.append(float(marker.get("SecTime")))
        except Exception:
            pass
    return max(vals) if vals else None


def _clip_duration_sec(clip: ET.Element, als_path: str) -> Optional[float]:
    audio_path = _resolve_clip_audio_path(clip, als_path)
    if not audio_path:
        return None
    try:
        _, _, duration = _duration_info(audio_path)
    except Exception:
        return None
    return float(duration) if duration and duration > 0.0 else None


def _row_end_sec(row: Dict[str, ET.Element], als_path: str, drop_sec: float) -> float:
    values: List[float] = [float(drop_sec) + 1.0]
    for clip in row.values():
        marker_sec = _last_marker_sec(clip)
        if marker_sec is not None:
            values.append(float(marker_sec))
        duration = _clip_duration_sec(clip, als_path)
        if duration is not None:
            values.append(float(duration))
    return max(values)


def _current_anchor_sec(clip: ET.Element) -> Optional[float]:
    anchors: List[float] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        try:
            sec = float(marker.get("SecTime"))
            beat = float(marker.get("BeatTime"))
        except Exception:
            continue
        if sec > 1.0 and abs(beat) <= 0.02:
            anchors.append(float(sec))
    return min(anchors) if anchors else None


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

    end_beat = _sec_to_beats(float(max(float(end_sec), float(drop_sec) + 0.01)), int(bpm)) + phi
    end_beat = max(0.01, float(end_beat))
    hidden_start = _fmt(phi)
    hidden_end = _fmt(phi + 4.0)
    _set_value(clip.find("./CurrentStart"), "0")
    _set_value(clip.find("./CurrentEnd"), _fmt(end_beat))
    loop = clip.find("./Loop")
    if loop is not None:
        _set_value(loop.find("./LoopStart"), "0")
        _set_value(loop.find("./LoopEnd"), _fmt(end_beat))
        _set_value(loop.find("./OutMarker"), _fmt(end_beat))
        _set_value(loop.find("./HiddenLoopStart"), hidden_start)
        _set_value(loop.find("./HiddenLoopEnd"), hidden_end)

    scroller = clip.find("./ScrollerTimePreserver")
    if scroller is not None:
        _set_value(scroller.find("./LeftTime"), hidden_start)
        _set_value(scroller.find("./RightTime"), _fmt(end_beat))

    selection = clip.find("./TimeSelection")
    if selection is not None:
        _set_value(selection.find("./AnchorTime"), hidden_start)
        _set_value(selection.find("./OtherTime"), hidden_start)


def apply_local_candidates(
    als_in: str,
    als_out: str,
    *,
    min_delta_sec: float,
    only_missing: bool,
    dry_run: bool,
    allow_legacy_detector_write: bool = False,
) -> Dict[str, int]:
    if not dry_run:
        require_legacy_detector_write_opt_in(
            "apply_folder_drop_candidates_to_set.py",
            action="retiming a combined ALS from local drop_candidates/DROP_ALIGNED files",
            explicit=bool(allow_legacy_detector_write),
        )
    root = ET.fromstring(gzip.open(als_in, "rb").read())
    stats = {
        "rows_total": 0,
        "rows_with_local_candidate": 0,
        "rows_changed": 0,
        "rows_skipped_no_candidate": 0,
        "rows_skipped_no_audio": 0,
        "rows_skipped_no_bpm": 0,
        "clips_retimed": 0,
    }
    for slot_idx, row in _iter_triplet_rows(root):
        stats["rows_total"] += 1
        drums = row["drums"]
        bpm = _clip_bpm(drums)
        if not bpm:
            stats["rows_skipped_no_bpm"] += 1
            continue
        audio_path = _resolve_clip_audio_path(drums, als_in)
        if not audio_path:
            stats["rows_skipped_no_audio"] += 1
            continue
        drop_sec, source = _local_drop_sec(audio_path)
        if drop_sec is None:
            stats["rows_skipped_no_candidate"] += 1
            continue
        stats["rows_with_local_candidate"] += 1
        current = _current_anchor_sec(drums)
        if only_missing and current is not None:
            continue
        if current is not None and abs(float(current) - float(drop_sec)) < float(min_delta_sec):
            continue

        end_sec = _row_end_sec(row, als_in, float(drop_sec))
        if not dry_run:
            for role in ("drums", "inst", "vocals"):
                _replace_warp_markers(row[role], int(bpm), float(drop_sec), float(end_sec))
        stats["rows_changed"] += 1
        stats["clips_retimed"] += 3
        print(
            f"[LOCAL] slot={slot_idx} {float(current or 0.0):.3f}s -> {float(drop_sec):.3f}s "
            f"source={source} name={_clip_name(drums)}"
        )

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(als_out)), exist_ok=True)
        with gzip.open(als_out, "wb") as fh:
            fh.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply per-track local drop_candidates/DROP_ALIGNED anchors to a combined Session View ALS set.")
    parser.add_argument("--als", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-delta-sec", type=float, default=0.040)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_legacy_detector_write_arg(parser)
    args = parser.parse_args()
    stats = apply_local_candidates(
        os.path.abspath(args.als),
        os.path.abspath(args.out),
        min_delta_sec=float(args.min_delta_sec),
        only_missing=bool(args.only_missing),
        dry_run=bool(args.dry_run),
        allow_legacy_detector_write=bool(args.allow_legacy_detector_write),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
