#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
import gzip
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple


_ROLE_BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)


@dataclass
class WarpMarker:
    marker_id: Optional[int]
    sec_time: float
    beat_time: float


@dataclass
class LabeledClip:
    audio_path: str
    target_sec: float
    bpm: Optional[float]
    als_path: str
    clip_name: str
    clip_id: Optional[str]
    warp_enabled: bool
    target_source: str
    marker_count: int

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        return d


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _decode_als_bytes(raw: bytes) -> bytes:
    try:
        return gzip.decompress(raw)
    except Exception:
        return raw


def read_als_root(als_path: str) -> ET.Element:
    raw = _read_bytes(als_path)
    xml_bytes = _decode_als_bytes(raw)
    return ET.fromstring(xml_bytes)


def write_als_root(als_path: str, root: ET.Element) -> None:
    # Keep output format stable: UTF-8 xml declaration + gzip compression.
    buf = BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    data = gzip.compress(buf.getvalue())
    with open(als_path, "wb") as f:
        f.write(data)


def _node_value(elem: Optional[ET.Element], default: str = "") -> str:
    if elem is None:
        return default
    return elem.get("Value", default)


def clip_name(clip: ET.Element) -> str:
    n = clip.find("Name")
    v = _node_value(n).strip()
    if v:
        return v
    n = clip.find("EffectiveName")
    return _node_value(n).strip()


def _clip_file_path_nodes(clip: ET.Element) -> Tuple[Optional[ET.Element], Optional[ET.Element]]:
    path_node = clip.find(".//SampleRef/FileRef/Path")
    rel_node = clip.find(".//SampleRef/FileRef/RelativePath")
    return path_node, rel_node


def clip_audio_path(clip: ET.Element, als_path: str, resolve: bool = True) -> str:
    path_node, rel_node = _clip_file_path_nodes(clip)
    raw_path = _node_value(path_node).strip()
    raw_rel = _node_value(rel_node).strip()

    if resolve:
        if raw_path and os.path.exists(raw_path):
            return os.path.abspath(raw_path)
        if raw_rel:
            base = os.path.dirname(os.path.abspath(als_path))
            maybe = os.path.abspath(os.path.join(base, raw_rel))
            if os.path.exists(maybe):
                return maybe

    if raw_path:
        return raw_path
    if raw_rel:
        if resolve:
            base = os.path.dirname(os.path.abspath(als_path))
            return os.path.abspath(os.path.join(base, raw_rel))
        return raw_rel
    return clip_name(clip)


def set_clip_audio_path(clip: ET.Element, audio_path: str, out_als_path: str) -> None:
    path_node, rel_node = _clip_file_path_nodes(clip)
    abs_audio = os.path.abspath(audio_path)
    rel_audio = os.path.relpath(abs_audio, start=os.path.dirname(os.path.abspath(out_als_path)))

    if path_node is not None:
        path_node.set("Value", abs_audio)
    if rel_node is not None:
        rel_node.set("Value", rel_audio)


def parse_warp_markers(clip: ET.Element) -> List[WarpMarker]:
    wm = clip.find("WarpMarkers")
    if wm is None:
        return []
    out: List[WarpMarker] = []
    for m in wm.findall("WarpMarker"):
        sec = m.get("SecTime")
        beat = m.get("BeatTime")
        if sec is None or beat is None:
            continue
        try:
            sec_f = float(sec)
            beat_f = float(beat)
        except Exception:
            continue
        marker_id = None
        try:
            marker_id = int(m.get("Id")) if m.get("Id") is not None else None
        except Exception:
            marker_id = None
        out.append(WarpMarker(marker_id=marker_id, sec_time=sec_f, beat_time=beat_f))
    out.sort(key=lambda x: x.sec_time)
    return out


def _interpolate_sec_for_beat_zero(markers: List[WarpMarker]) -> Tuple[Optional[float], str]:
    if not markers:
        return None, "missing"

    # Exact beat-zero marker.
    exact = [m for m in markers if abs(m.beat_time) <= 1e-6]
    if exact:
        exact.sort(key=lambda m: m.sec_time)
        return float(exact[0].sec_time), "exact"

    # Interpolate across marker pair that brackets beat 0.
    for i in range(len(markers) - 1):
        a = markers[i]
        b = markers[i + 1]
        if (a.beat_time <= 0.0 <= b.beat_time) or (b.beat_time <= 0.0 <= a.beat_time):
            dbeat = b.beat_time - a.beat_time
            if abs(dbeat) <= 1e-9:
                continue
            alpha = (0.0 - a.beat_time) / dbeat
            sec = a.sec_time + alpha * (b.sec_time - a.sec_time)
            return float(sec), "interp"

    # Extrapolate from nearest two markers in beat domain.
    by_beat = sorted(markers, key=lambda m: abs(m.beat_time))
    if len(by_beat) < 2:
        return None, "missing"
    a = by_beat[0]
    b = by_beat[1]
    dbeat = b.beat_time - a.beat_time
    if abs(dbeat) <= 1e-9:
        return None, "missing"
    alpha = (0.0 - a.beat_time) / dbeat
    sec = a.sec_time + alpha * (b.sec_time - a.sec_time)
    return float(sec), "extrap"


def infer_target_sec_from_clip(clip: ET.Element) -> Tuple[Optional[float], str]:
    markers = parse_warp_markers(clip)
    return _interpolate_sec_for_beat_zero(markers)


def infer_clip_bpm(clip: ET.Element, audio_ref: str) -> Optional[float]:
    # Preferred source: clip Tempo/Manual.
    manual = clip.find("Tempo/Manual")
    if manual is not None:
        try:
            bpm = float(manual.get("Value", ""))
            if bpm > 0:
                return bpm
        except Exception:
            pass

    # Fallback from filename convention drums_140_2A_...
    name = os.path.basename(audio_ref or "")
    stem, _ = os.path.splitext(name)
    m = _ROLE_BPM_RE.match(stem)
    if m:
        try:
            bpm = float(int(m.group(1)))
            if bpm > 0:
                return bpm
        except Exception:
            pass
    return None


def iter_audio_clips(root: ET.Element) -> Iterable[ET.Element]:
    for clip in root.iter("AudioClip"):
        yield clip


def extract_labeled_clips_from_als(
    als_path: str,
    resolve_audio_paths: bool = True,
    include_unwarped: bool = False,
) -> List[LabeledClip]:
    root = read_als_root(als_path)
    out: List[LabeledClip] = []
    for clip in iter_audio_clips(root):
        is_warped = True
        iw = clip.find("IsWarped")
        if iw is not None:
            is_warped = _node_value(iw, "true").strip().lower() in {"1", "true", "yes"}
        if (not include_unwarped) and (not is_warped):
            continue

        target_sec, source = infer_target_sec_from_clip(clip)
        if target_sec is None:
            continue

        audio_ref = clip_audio_path(clip, als_path=als_path, resolve=resolve_audio_paths)
        bpm = infer_clip_bpm(clip, audio_ref)
        markers = parse_warp_markers(clip)
        out.append(
            LabeledClip(
                audio_path=audio_ref,
                target_sec=float(target_sec),
                bpm=bpm,
                als_path=os.path.abspath(als_path),
                clip_name=clip_name(clip),
                clip_id=clip.get("Id"),
                warp_enabled=is_warped,
                target_source=source,
                marker_count=len(markers),
            )
        )
    return out


def replace_warp_markers_with_anchor(
    clip: ET.Element,
    anchor_sec: float,
    bpm: float,
    end_sec: float,
) -> None:
    wm = clip.find("WarpMarkers")
    if wm is None:
        wm = ET.Element("WarpMarkers")
        clip.insert(0, wm)
    for child in list(wm):
        wm.remove(child)

    # Linear map with explicit beat-zero anchor.
    phase = -((float(anchor_sec) * float(bpm)) / 60.0)
    points = sorted(set([0.0, max(0.0, float(anchor_sec)), max(0.0, float(end_sec))]))
    for i, sec in enumerate(points):
        beat = ((sec * float(bpm)) / 60.0) + phase
        wm.append(
            ET.Element(
                "WarpMarker",
                {
                    "Id": str(i),
                    "SecTime": f"{sec:.6f}",
                    "BeatTime": f"{beat:.6f}",
                },
            )
        )

    is_warped = clip.find("IsWarped")
    if is_warped is None:
        is_warped = ET.SubElement(clip, "IsWarped")
    is_warped.set("Value", "true")

    # Keep loop/out markers consistent with warped endpoint.
    end_beat = ((max(0.0, float(end_sec)) * float(bpm)) / 60.0) + phase
    current_start = clip.find("CurrentStart")
    if current_start is not None:
        current_start.set("Value", "0")
    current_end = clip.find("CurrentEnd")
    if current_end is not None:
        current_end.set("Value", f"{end_beat:.6f}")

    loop = clip.find("Loop")
    if loop is not None:
        for tag, value in (
            ("LoopStart", 0.0),
            ("LoopEnd", end_beat),
            ("OutMarker", end_beat),
            ("HiddenLoopStart", phase),
            ("HiddenLoopEnd", phase + 4.0),
        ):
            node = loop.find(tag)
            if node is not None:
                node.set("Value", f"{float(value):.6f}")

    scroller = clip.find("ScrollerTimePreserver")
    if scroller is not None:
        left = scroller.find("LeftTime")
        if left is not None:
            left.set("Value", f"{phase:.6f}")
        right = scroller.find("RightTime")
        if right is not None:
            right.set("Value", f"{end_beat:.6f}")

    selection = clip.find("TimeSelection")
    if selection is not None:
        for tag in ("AnchorTime", "OtherTime"):
            node = selection.find(tag)
            if node is not None:
                node.set("Value", f"{phase:.6f}")


def nearest_time(value: float, candidates: List[float]) -> float:
    if not candidates:
        return float(value)
    return min(candidates, key=lambda t: abs(float(t) - float(value)))
