#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import gzip
import html
import os
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from .utils import as_float, as_int


@dataclass
class WarpMarker:
    marker_id: Optional[int]
    beat: float
    sec: float


@dataclass
class ClipLabel:
    audio_path: str
    als_path: str
    target_sec: float
    bpm_hint: Optional[float]
    warp_markers: List[Dict[str, float]]
    sr: Optional[int]
    metadata: Dict[str, object]

    def to_row(self) -> Dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "als_path": self.als_path,
            "target_sec": float(self.target_sec),
            "sr": self.sr,
            "bpm_hint": float(self.bpm_hint) if self.bpm_hint else None,
            "warp_markers": self.warp_markers,
            "metadata": self.metadata,
        }


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
    buf = BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    parent = os.path.dirname(os.path.abspath(als_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(als_path, "wb") as f:
        f.write(gzip.compress(buf.getvalue()))


def iter_audio_clips(root: ET.Element) -> Iterable[ET.Element]:
    for clip in root.iter("AudioClip"):
        yield clip


def node_val(elem: Optional[ET.Element], default: str = "") -> str:
    if elem is None:
        return default
    return elem.get("Value", default)


def clip_name(clip: ET.Element) -> str:
    n = clip.find("Name")
    v = node_val(n).strip()
    if v:
        return v
    n = clip.find("EffectiveName")
    return node_val(n).strip()


def clip_warp_enabled(clip: ET.Element) -> bool:
    iw = clip.find("IsWarped")
    if iw is None:
        return True
    return node_val(iw, "true").strip().lower() in {"true", "1", "yes"}


def clip_tempo_hint(clip: ET.Element) -> Optional[float]:
    for p in (
        "Tempo/Manual",
        "Warping/Segmentation/Tempo",
        "ClipTempo",
    ):
        n = clip.find(p)
        if n is None:
            continue
        v = as_float(node_val(n))
        if v and v > 0:
            return float(v)
    return None


def _clip_path_nodes(clip: ET.Element) -> Tuple[Optional[ET.Element], Optional[ET.Element]]:
    path_node = clip.find(".//SampleRef/FileRef/Path")
    rel_node = clip.find(".//SampleRef/FileRef/RelativePath")
    return path_node, rel_node


def clip_audio_path(clip: ET.Element, als_path: str, resolve: bool = True) -> str:
    path_node, rel_node = _clip_path_nodes(clip)
    raw_path = node_val(path_node).strip()
    raw_rel = node_val(rel_node).strip()

    if resolve:
        if raw_path and os.path.exists(raw_path):
            return os.path.abspath(raw_path)
        if raw_rel:
            p = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(als_path)), raw_rel))
            if os.path.exists(p):
                return p
    if raw_path:
        return raw_path
    if raw_rel:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(als_path)), raw_rel)) if resolve else raw_rel
    return clip_name(clip)


def set_clip_audio_path(clip: ET.Element, audio_path: str, out_als_path: str) -> None:
    pnode, rnode = _clip_path_nodes(clip)
    abs_audio = os.path.abspath(audio_path)
    rel_audio = os.path.relpath(abs_audio, start=os.path.dirname(os.path.abspath(out_als_path)))
    if pnode is not None:
        pnode.set("Value", abs_audio)
    if rnode is not None:
        rnode.set("Value", rel_audio)


def parse_warp_markers(clip: ET.Element) -> List[WarpMarker]:
    wm = clip.find("WarpMarkers")
    if wm is None:
        return []
    rows: List[WarpMarker] = []
    for m in wm.findall("WarpMarker"):
        b = as_float(m.get("BeatTime"))
        s = as_float(m.get("SecTime"))
        if b is None or s is None:
            continue
        rows.append(WarpMarker(marker_id=as_int(m.get("Id")), beat=float(b), sec=float(s)))
    rows.sort(key=lambda x: x.sec)
    return rows


def beat0_sec_from_markers(markers: Sequence[WarpMarker]) -> Tuple[Optional[float], str]:
    if not markers:
        return None, "missing"

    exact = sorted([m.sec for m in markers if abs(float(m.beat)) <= 1e-7])
    if exact:
        return float(exact[0]), "exact"

    # bracket interpolation
    for i in range(len(markers) - 1):
        a = markers[i]
        b = markers[i + 1]
        if (a.beat <= 0.0 <= b.beat) or (b.beat <= 0.0 <= a.beat):
            d = b.beat - a.beat
            if abs(d) <= 1e-9:
                continue
            t = (0.0 - a.beat) / d
            sec = a.sec + t * (b.sec - a.sec)
            return float(sec), "interp"

    # extrapolate from closest beats around zero in beat-distance
    by = sorted(markers, key=lambda m: abs(float(m.beat)))
    if len(by) < 2:
        return None, "missing"
    a, b = by[0], by[1]
    d = b.beat - a.beat
    if abs(d) <= 1e-9:
        return None, "missing"
    t = (0.0 - a.beat) / d
    sec = a.sec + t * (b.sec - a.sec)
    return float(sec), "extrap"


def clip_sample_rate_hint(clip: ET.Element) -> Optional[int]:
    for p in (
        ".//SampleRef/DefaultDuration/TimeSignatures/RemoteableTimeSignature/Numerator",
        ".//SampleRef/FileRef/SampleRate",
    ):
        n = clip.find(p)
        if n is None:
            continue
        v = as_int(node_val(n))
        if v and 4000 <= v <= 384000:
            return int(v)
    return None


def clip_extra_offsets(clip: ET.Element) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tag in ("CurrentStart", "CurrentEnd", "SampleOffset", "StartMarker", "EndMarker", "LoopStart", "LoopEnd"):
        n = clip.find(tag)
        if n is None:
            continue
        v = as_float(node_val(n))
        if v is not None:
            out[tag] = float(v)
    return out


def extract_labels_from_als(als_path: str, resolve_paths: bool = True, include_unwarped: bool = False) -> List[ClipLabel]:
    root = read_als_root(als_path)
    out: List[ClipLabel] = []
    als_abs = os.path.abspath(als_path)

    for clip in iter_audio_clips(root):
        warped = clip_warp_enabled(clip)
        if (not include_unwarped) and (not warped):
            continue

        markers = parse_warp_markers(clip)
        sec, source = beat0_sec_from_markers(markers)
        if sec is None:
            continue

        audio_path = clip_audio_path(clip, als_abs, resolve=resolve_paths)
        tempo = clip_tempo_hint(clip)
        sr_hint = clip_sample_rate_hint(clip)
        rows = [{"beat": float(m.beat), "sec": float(m.sec)} for m in markers]
        label = ClipLabel(
            audio_path=audio_path,
            als_path=als_abs,
            target_sec=float(sec),
            bpm_hint=tempo,
            warp_markers=rows,
            sr=sr_hint,
            metadata={
                "clip_name": clip_name(clip),
                "clip_id": clip.get("Id"),
                "warp_enabled": bool(warped),
                "target_source": source,
                "marker_count": len(rows),
                "offsets": clip_extra_offsets(clip),
            },
        )
        out.append(label)
    return out


def _cluster_values(vals: List[float], gap_sec: float) -> List[List[float]]:
    if not vals:
        return []
    s = sorted(vals)
    out: List[List[float]] = [[s[0]]]
    for v in s[1:]:
        if abs(v - out[-1][-1]) <= gap_sec:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def dedupe_labels(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_audio: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        ap = str(r.get("audio_path", "")).strip()
        if not ap:
            continue
        by_audio.setdefault(os.path.abspath(ap), []).append(dict(r))

    out: List[Dict[str, object]] = []
    for ap, group in sorted(by_audio.items()):
        valid = [g for g in group if as_float(g.get("target_sec")) is not None]
        if not valid:
            continue
        vals = [float(g["target_sec"]) for g in valid if float(g["target_sec"]) > 0.0]
        if not vals:
            continue

        bpm_vals = [float(v) for v in [as_float(g.get("bpm_hint")) for g in valid] if v and v > 0]
        bpm = float(sum(bpm_vals) / len(bpm_vals)) if bpm_vals else 128.0
        beat_sec = 60.0 / max(1e-6, bpm)
        clusters = _cluster_values(vals, gap_sec=max(0.12, beat_sec))

        scored: List[Tuple[int, float, float, List[float]]] = []
        for c in clusters:
            med = float(c[len(c) // 2])
            std = float((sum((x - med) ** 2 for x in c) / max(1, len(c))) ** 0.5)
            scored.append((len(c), std, med, c))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        best = scored[0]
        target = float(best[2])

        # Choose representative row closest to chosen target.
        rep = min(valid, key=lambda g: abs(float(g["target_sec"]) - target))
        rep["audio_path"] = ap
        rep["target_sec"] = target
        rep["metadata"] = dict(rep.get("metadata") or {})
        rep["metadata"]["dedupe_obs"] = int(len(group))
        rep["metadata"]["dedupe_cluster_size"] = int(best[0])
        out.append(rep)

    return out


def _collect_numeric_ids(root: ET.Element) -> set[int]:
    used: set[int] = set()
    for e in root.iter():
        v = e.get("Id")
        if v is None:
            continue
        n = as_int(v)
        if n is not None:
            used.add(int(n))
    return used


def _next_id(used: set[int]) -> int:
    x = 1
    while x in used:
        x += 1
    used.add(x)
    return x


def select_target_clips(root: ET.Element, template_als: str, audio_path: str, apply_to_all: bool) -> List[ET.Element]:
    clips = list(iter_audio_clips(root))
    if apply_to_all:
        return clips
    base = os.path.basename(audio_path).lower()
    stem, _ = os.path.splitext(base)
    hit: List[ET.Element] = []
    for c in clips:
        ref = clip_audio_path(c, template_als, resolve=False)
        rbase = os.path.basename(ref).lower()
        rstem, _ = os.path.splitext(rbase)
        nm = clip_name(c).lower()
        if rbase == base or (rstem and rstem == stem) or (stem and stem in nm):
            hit.append(c)
    if hit:
        return hit
    return clips[:1] if clips else []


def rewrite_clip_warp_markers(
    root: ET.Element,
    clip: ET.Element,
    target_sec: float,
    bpm: float,
    duration_sec: float,
) -> None:
    wm = clip.find("WarpMarkers")
    if wm is None:
        wm = ET.SubElement(clip, "WarpMarkers")
    for child in list(wm):
        wm.remove(child)

    used = _collect_numeric_ids(root)
    target = max(0.0, float(target_sec))
    dur = max(target + 0.01, float(duration_sec))
    beat_per_sec = float(bpm) / 60.0

    points = [
        (0.0, -target * beat_per_sec),
        (target, 0.0),
        (dur, (dur - target) * beat_per_sec),
    ]
    for sec, beat in points:
        m = ET.Element(
            "WarpMarker",
            {
                "Id": str(_next_id(used)),
                "SecTime": f"{sec:.6f}",
                "BeatTime": f"{beat:.6f}",
            },
        )
        wm.append(m)

    iw = clip.find("IsWarped")
    if iw is None:
        iw = ET.SubElement(clip, "IsWarped")
    iw.set("Value", "true")

    end_beat = (dur - target) * beat_per_sec
    for tag in ("LoopEnd", "OutMarker"):
        for n in clip.iter(tag):
            n.set("Value", f"{end_beat:.6f}")


def collect_list_ids(root: ET.Element) -> List[str]:
    vals: List[str] = []
    for e in root.iter():
        v = e.get("ListId")
        if v is not None:
            vals.append(str(v))
    return vals


def validate_als(root: ET.Element, expected_target_sec: Optional[float] = None) -> Dict[str, object]:
    # BeatTime==0 marker existence check
    beat0 = 0
    beat0_secs: List[float] = []
    for clip in iter_audio_clips(root):
        for wm in parse_warp_markers(clip):
            if abs(float(wm.beat)) <= 1e-6:
                beat0 += 1
                beat0_secs.append(float(wm.sec))

    list_ids = collect_list_ids(root)
    dup_list_ids = sorted({x for x in list_ids if list_ids.count(x) > 1})

    sec_ok = None
    if expected_target_sec is not None and beat0_secs:
        sec_ok = min(abs(float(expected_target_sec) - s) for s in beat0_secs) <= 0.001

    preview_n = 32
    return {
        "ok": bool(beat0 >= 1 and len(dup_list_ids) == 0 and (sec_ok if sec_ok is not None else True)),
        "beat0_markers": int(beat0),
        "beat0_secs_total": int(len(beat0_secs)),
        "beat0_secs_preview": beat0_secs[:preview_n],
        "duplicate_list_ids": dup_list_ids,
        "target_match_1ms": sec_ok,
    }
