#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ZERO_TOL = 1e-6
ANCHOR_BEAT_TOL = 0.02
AUDIO_EXTS = {".als"}


def _read_root(path: str) -> ET.Element:
    with gzip.open(path, "rb") as fh:
        return ET.fromstring(fh.read())


def _write_root(path: str, root: ET.Element) -> None:
    buf = BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    with gzip.open(path, "wb") as fh:
        fh.write(buf.getvalue())


def _value(node: Optional[ET.Element]) -> str:
    return "" if node is None else str(node.get("Value") or "").strip()


def _set_value(node: Optional[ET.Element], value: object) -> bool:
    if node is None:
        return False
    text = _fmt(float(value)) if isinstance(value, float) else str(value)
    old_float = _as_float(node.get("Value"))
    new_float = _as_float(text)
    if old_float is not None and new_float is not None and abs(float(old_float) - float(new_float)) <= 1e-6:
        return False
    if node.get("Value") == text:
        return False
    node.set("Value", text)
    return True


def _fmt(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0", "-0.0"} else text


def _as_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _markers(clip: ET.Element) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        sec = _as_float(marker.get("SecTime"))
        beat = _as_float(marker.get("BeatTime"))
        if sec is None or beat is None:
            continue
        rows.append((float(sec), float(beat)))
    rows.sort(key=lambda row: row[0])
    return rows


def _anchor_sec(markers: Sequence[Tuple[float, float]], min_sec: float) -> Optional[float]:
    anchors = [sec for sec, beat in markers if sec > float(min_sec) and abs(float(beat)) <= ANCHOR_BEAT_TOL]
    return min(anchors) if anchors else None


def _bpm_from_markers(markers: Sequence[Tuple[float, float]]) -> Optional[float]:
    usable = [(sec, beat) for sec, beat in markers if sec >= 0.0]
    for i, (sec_a, beat_a) in enumerate(usable):
        for sec_b, beat_b in usable[i + 1 :]:
            dsec = float(sec_b) - float(sec_a)
            dbeat = float(beat_b) - float(beat_a)
            if dsec <= 1e-6 or abs(dbeat) <= 1e-6:
                continue
            bpm = (dbeat / dsec) * 60.0
            if 20.0 <= bpm <= 320.0:
                return float(bpm)
    return None


def _duration_sec_from_sample_ref(clip: ET.Element) -> Optional[float]:
    frames = _as_float(_value(clip.find("./SampleRef/DefaultDuration")))
    sr = _as_float(_value(clip.find("./SampleRef/DefaultSampleRate")))
    if frames and sr and frames > 0 and sr > 0:
        return float(frames) / float(sr)
    return None


def _end_sec(clip: ET.Element, markers: Sequence[Tuple[float, float]], anchor_sec: float) -> Optional[float]:
    duration = _duration_sec_from_sample_ref(clip)
    if duration and duration > float(anchor_sec):
        return float(duration)
    marker_end = max((sec for sec, _beat in markers), default=0.0)
    if marker_end > float(anchor_sec) + 0.01:
        return float(marker_end)
    return None


def _warp_markers_container(clip: ET.Element) -> ET.Element:
    container = clip.find("./WarpMarkers")
    if container is None:
        container = ET.Element("WarpMarkers")
        clip.insert(0, container)
    return container


def _rewrite_markers(clip: ET.Element, *, bpm: float, anchor_sec: float, end_sec: float) -> bool:
    container = _warp_markers_container(clip)
    old = [(m.get("SecTime"), m.get("BeatTime")) for m in container.findall("./WarpMarker")]
    for child in list(container):
        container.remove(child)
    points = sorted(set(round(max(0.0, value), 6) for value in (0.0, float(anchor_sec), float(end_sec))))
    for idx, sec in enumerate(points):
        beat = ((float(sec) - float(anchor_sec)) * float(bpm)) / 60.0
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
    new = [(m.get("SecTime"), m.get("BeatTime")) for m in container.findall("./WarpMarker")]
    if len(old) != len(new):
        return True
    for (old_sec, old_beat), (new_sec, new_beat) in zip(old, new):
        old_sec_f = _as_float(old_sec)
        old_beat_f = _as_float(old_beat)
        new_sec_f = _as_float(new_sec)
        new_beat_f = _as_float(new_beat)
        if (
            old_sec_f is None
            or old_beat_f is None
            or new_sec_f is None
            or new_beat_f is None
            or abs(float(old_sec_f) - float(new_sec_f)) > 1e-6
            or abs(float(old_beat_f) - float(new_beat_f)) > 1e-6
        ):
            return True
    return False


def _has_ancestor(node: ET.Element, parent_map: Dict[ET.Element, ET.Element], tag: str) -> bool:
    parent = parent_map.get(node)
    while parent is not None:
        if parent.tag == tag:
            return True
        parent = parent_map.get(parent)
    return False


def _normalize_clip(clip: ET.Element, min_sec: float, rewrite_warp_grid: bool = False) -> bool:
    markers = _markers(clip)
    anchor = _anchor_sec(markers, min_sec=min_sec)
    if anchor is None:
        return False
    bpm = _bpm_from_markers(markers)
    end = _end_sec(clip, markers, anchor)
    if bpm is None or end is None:
        return False

    changed = False
    if rewrite_warp_grid:
        changed = _rewrite_markers(clip, bpm=float(bpm), anchor_sec=float(anchor), end_sec=float(end))
    phase = -((float(anchor) * float(bpm)) / 60.0)
    end_beat = max(0.01, ((float(end) * float(bpm)) / 60.0) + phase)

    warped = clip.find("./IsWarped")
    if warped is None:
        warped = ET.Element("IsWarped", {"Value": "true"})
        clip.insert(0, warped)
        changed = True
    else:
        changed = _set_value(warped, "true") or changed

    changed = _set_value(clip.find("./CurrentStart"), "0") or changed

    loop = clip.find("./Loop")
    if loop is not None:
        changed = _set_value(loop.find("./LoopStart"), "0") or changed
        changed = _set_value(loop.find("./HiddenLoopStart"), float(phase)) or changed
        changed = _set_value(loop.find("./HiddenLoopEnd"), float(phase + 4.0)) or changed

    scroller = clip.find("./ScrollerTimePreserver")
    if scroller is not None:
        changed = _set_value(scroller.find("./LeftTime"), float(phase)) or changed

    selection = clip.find("./TimeSelection")
    if selection is not None:
        changed = _set_value(selection.find("./AnchorTime"), float(phase)) or changed
        changed = _set_value(selection.find("./OtherTime"), float(phase)) or changed

    return changed


def _iter_als_paths(paths: Sequence[str], recursive: bool) -> Iterable[str]:
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            full = str(path.resolve())
            if full not in seen:
                seen.add(full)
                yield full
            continue
        if path.is_dir():
            pattern = "**/*.als" if recursive else "*.als"
            for child in path.glob(pattern):
                if child.is_file():
                    full = str(child.resolve())
                    if full not in seen:
                        seen.add(full)
                        yield full


def normalize_als(
    path: str,
    *,
    apply: bool,
    backup: bool,
    min_sec: float,
    clip_slots_only: bool = False,
    rewrite_warp_grid: bool = False,
) -> Tuple[int, int]:
    root = _read_root(path)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    changed_clips = 0
    seen_clips = 0
    for clip in root.iter("AudioClip"):
        if clip_slots_only and not _has_ancestor(clip, parent_map, "ClipSlot"):
            continue
        seen_clips += 1
        if _normalize_clip(clip, min_sec=float(min_sec), rewrite_warp_grid=bool(rewrite_warp_grid)):
            changed_clips += 1
    if changed_clips and apply:
        if backup:
            backup_path = f"{path}.timing-backup"
            if not os.path.exists(backup_path):
                shutil.copy2(path, backup_path)
        _write_root(path, root)
    return changed_clips, seen_clips


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Ableton ALS clip launch timing from existing BeatTime=0 anchors.")
    parser.add_argument("paths", nargs="+", help="ALS files or folders to scan")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan folders")
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is audit only.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .timing-backup files when applying")
    parser.add_argument("--min-sec", type=float, default=1.0, help="Ignore BeatTime=0 anchors at or before this second")
    parser.add_argument("--clip-slots-only", action="store_true", help="Only normalize clips inside Session View ClipSlot nodes")
    parser.add_argument(
        "--rewrite-warp-grid",
        action="store_true",
        help="Destructively replace each clip's warp markers with sample-start/drop/end markers. Off by default to preserve SDK/transient corrections.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    args = parser.parse_args()

    files = list(_iter_als_paths(args.paths, recursive=bool(args.recursive)))
    changed_files = 0
    changed_clips = 0
    total_clips = 0
    for path in files:
        try:
            clips_changed, clips_total = normalize_als(
                path,
                apply=bool(args.apply),
                backup=not bool(args.no_backup),
                min_sec=float(args.min_sec),
                clip_slots_only=bool(args.clip_slots_only),
                rewrite_warp_grid=bool(args.rewrite_warp_grid),
            )
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            continue
        total_clips += int(clips_total)
        if clips_changed:
            changed_files += 1
            changed_clips += int(clips_changed)
            if not args.quiet:
                action = "UPDATED" if args.apply else "WOULD_UPDATE"
                print(f"[{action}] clips={clips_changed} {path}")

    mode = "apply" if args.apply else "dry-run"
    print(
        f"[SUMMARY] mode={mode} files_scanned={len(files)} files_changed={changed_files} "
        f"clips_changed={changed_clips} clips_seen={total_clips}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
