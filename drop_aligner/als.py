from __future__ import annotations

import gzip
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
from lxml import etree


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}
STEM_ROLES = ("drums", "inst", "vocals")
ROLE_RE = re.compile(r"(^|[^a-z])(drums|inst|vocals)(?:[-_/ .]|$)", re.IGNORECASE)


def _value(node: Optional[etree._Element]) -> Optional[str]:
    if node is None:
        return None
    if "Value" in node.attrib:
        return node.attrib.get("Value")
    return node.text


def _set_value(node: Optional[etree._Element], value: object) -> None:
    if node is None:
        return
    text = str(value)
    if "Value" in node.attrib:
        node.attrib["Value"] = text
    else:
        node.text = text


def _marker_get(marker: etree._Element, name: str) -> Optional[float]:
    if name in marker.attrib:
        try:
            return float(marker.attrib[name])
        except Exception:
            return None
    child = marker.find(name)
    raw = _value(child)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _marker_set(marker: etree._Element, name: str, value: float) -> None:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    if name in marker.attrib or marker.find(name) is None:
        marker.attrib[name] = text
        return
    _set_value(marker.find(name), text)


def _marker_id(marker: etree._Element) -> Optional[int]:
    if "Id" in marker.attrib:
        try:
            return int(marker.attrib["Id"])
        except Exception:
            return None
    child = marker.find("Id")
    raw = _value(child)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _marker_set_id(marker: etree._Element, marker_id: int) -> None:
    if "Id" in marker.attrib or marker.find("Id") is None:
        marker.attrib["Id"] = str(int(marker_id))
        return
    _set_value(marker.find("Id"), int(marker_id))


def _load_als(template_path: str) -> etree._ElementTree:
    with gzip.open(template_path, "rb") as fh:
        data = fh.read()
    parser = etree.XMLParser(remove_blank_text=False, recover=False)
    return etree.ElementTree(etree.fromstring(data, parser=parser))


def _first_audio_clip(root: etree._Element) -> etree._Element:
    clip = root.find(".//AudioClip")
    if clip is None:
        raise ValueError("Template does not contain an AudioClip.")
    return clip


def _audio_clips(root: etree._Element) -> Tuple[etree._Element, ...]:
    return tuple(root.findall(".//AudioClip"))


def _role_from_text(text: object) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = ROLE_RE.search(raw)
    if not match:
        return None
    role = match.group(2).lower()
    return role if role in STEM_ROLES else None


def _clip_role(clip: etree._Element) -> Optional[str]:
    values = []
    for path in ("Name", "EffectiveName", "UserName", ".//SampleRef/FileRef/Path", ".//SampleRef/FileRef/RelativePath"):
        node = clip.find(path)
        value = _value(node)
        if value:
            values.append(value)
    for value in values:
        role = _role_from_text(value)
        if role:
            return role
    return None


def _stem_role(path: Path) -> Optional[str]:
    name = path.name.lower()
    for role in STEM_ROLES:
        if name.startswith(role):
            return role
    return _role_from_text(name)


def _discover_stem_paths(audio_path: str) -> Dict[str, str]:
    audio = Path(audio_path).expanduser().resolve()
    stems: Dict[str, str] = {}
    role = _stem_role(audio)
    if role:
        stems[role] = str(audio)
    if not audio.parent.exists():
        return stems
    for candidate in sorted(audio.parent.iterdir(), key=lambda p: p.name.lower()):
        if not candidate.is_file() or candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        candidate_role = _stem_role(candidate)
        if candidate_role and candidate_role not in stems:
            stems[candidate_role] = str(candidate.resolve())
    return stems


def _file_ref_nodes(clip: etree._Element):
    sample_ref = clip.find("SampleRef")
    if sample_ref is None:
        return []
    return sample_ref.findall(".//FileRef")


def _set_audio_file_reference(clip: etree._Element, audio_path: str, output_path: str) -> None:
    audio_abs = os.path.abspath(audio_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    rel = os.path.relpath(audio_abs, output_dir)
    size = os.path.getsize(audio_abs)
    mod_time = int(os.path.getmtime(audio_abs))
    stem = os.path.splitext(os.path.basename(audio_abs))[0]

    for file_ref in _file_ref_nodes(clip):
        _set_value(file_ref.find("Path"), audio_abs)
        _set_value(file_ref.find("RelativePath"), rel)
        _set_value(file_ref.find("OriginalFileSize"), int(size))

    name_node = clip.find("Name")
    _set_value(name_node, stem)

    sample_ref = clip.find("SampleRef")
    if sample_ref is not None:
        _set_value(sample_ref.find("LastModDate"), int(mod_time))
        browser = sample_ref.find(".//BrowserContentPath")
        if browser is not None:
            _set_value(browser, f"userfolder:{os.path.dirname(audio_abs)}#{os.path.basename(audio_abs)}")


def _duration_info(audio_path: str) -> Tuple[int, int, float]:
    duration = float(librosa.get_duration(path=audio_path))
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        sr = int(info.samplerate)
        frames = int(info.frames)
    except Exception:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        frames = int(len(y))
    return int(sr), int(frames), float(duration)


def _format_float(value: float) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".")


def _update_clip_timing_fields(clip: etree._Element, *, drop_sec: float, duration_sec: float, bpm: float) -> None:
    """Keep launch, loop, and view state aligned with the drop-at-1.1.1 warp grid."""
    beat_per_sec = float(bpm) / 60.0
    phase = -(max(0.0, float(drop_sec)) * beat_per_sec)
    end_beat = (max(float(duration_sec), float(drop_sec) + 0.01) * beat_per_sec) + phase
    end_beat = max(0.01, float(end_beat))

    _set_value(clip.find("CurrentStart"), "0")
    _set_value(clip.find("CurrentEnd"), _format_float(end_beat))

    loop = clip.find("Loop")
    if loop is not None:
        _set_value(loop.find("LoopStart"), "0")
        _set_value(loop.find("LoopEnd"), _format_float(end_beat))
        _set_value(loop.find("OutMarker"), _format_float(end_beat))
        _set_value(loop.find("HiddenLoopStart"), _format_float(phase))
        _set_value(loop.find("HiddenLoopEnd"), _format_float(phase + 4.0))

    scroller = clip.find("ScrollerTimePreserver")
    if scroller is not None:
        _set_value(scroller.find("LeftTime"), _format_float(phase))
        _set_value(scroller.find("RightTime"), _format_float(end_beat))

    selection = clip.find("TimeSelection")
    if selection is not None:
        _set_value(selection.find("AnchorTime"), _format_float(phase))
        _set_value(selection.find("OtherTime"), _format_float(phase))


def _update_duration_fields(clip: etree._Element, audio_path: str, bpm: float, drop_sec: float) -> float:
    sr, frames, duration = _duration_info(audio_path)
    sample_ref = clip.find("SampleRef")
    if sample_ref is not None:
        _set_value(sample_ref.find("DefaultDuration"), int(frames))
        _set_value(sample_ref.find("DefaultSampleRate"), int(sr))
    _update_clip_timing_fields(clip, drop_sec=drop_sec, duration_sec=duration, bpm=bpm)
    return float(duration)


def _warp_markers(clip: etree._Element) -> etree._Element:
    markers = clip.find("WarpMarkers")
    if markers is None:
        markers = etree.SubElement(clip, "WarpMarkers")
    return markers


def _ensure_zero_marker(markers: etree._Element, bpm: float, drop_sec: float) -> etree._Element:
    marker_list = list(markers.findall("WarpMarker"))
    zero = None
    for marker in marker_list:
        sec = _marker_get(marker, "SecTime")
        if sec is not None and abs(sec) <= 1e-6:
            zero = marker
            break
    if zero is None:
        zero = deepcopy(marker_list[0]) if marker_list else etree.Element("WarpMarker")
        markers.insert(0, zero)
    _marker_set(zero, "SecTime", 0.0)
    _marker_set(zero, "BeatTime", -(float(drop_sec) * float(bpm) / 60.0))
    return zero


def _retime_existing_markers(markers: etree._Element, bpm: float, drop_sec: float) -> None:
    """Keep existing marker SecTime values but retime beats around the drop."""
    for marker in markers.findall("WarpMarker"):
        sec = _marker_get(marker, "SecTime")
        if sec is None:
            continue
        beat_time = (float(sec) - float(drop_sec)) * float(bpm) / 60.0
        _marker_set(marker, "BeatTime", beat_time)


def _new_marker_like(markers: etree._Element) -> etree._Element:
    marker_list = list(markers.findall("WarpMarker"))
    if marker_list:
        marker = deepcopy(marker_list[-1])
        for child in list(marker):
            child.text = child.text
    else:
        marker = etree.Element("WarpMarker")
    max_id = max((_marker_id(m) or 0) for m in marker_list) if marker_list else 0
    _marker_set_id(marker, max_id + 1)
    return marker


def _sort_warp_markers(markers: etree._Element) -> None:
    marker_nodes = markers.findall("WarpMarker")
    if len(marker_nodes) < 2:
        return
    marker_nodes.sort(key=lambda m: (_marker_get(m, "SecTime") if _marker_get(m, "SecTime") is not None else float("inf")))
    for marker in marker_nodes:
        markers.remove(marker)
    for marker in marker_nodes:
        markers.append(marker)


def _replace_warp_markers_with_drop_grid(
    clip: etree._Element,
    *,
    drop_sec: float,
    duration_sec: float,
    bpm: float,
) -> None:
    markers = _warp_markers(clip)
    for child in list(markers):
        markers.remove(child)

    target = max(0.0, float(drop_sec))
    end = max(target + 0.01, float(duration_sec))
    beat_per_sec = float(bpm) / 60.0
    points = sorted(set(round(max(0.0, p), 6) for p in (0.0, target, end)))
    for idx, sec in enumerate(points):
        beat = (float(sec) - target) * beat_per_sec
        marker = etree.Element(
            "WarpMarker",
            Id=str(idx),
            SecTime=f"{float(sec):.6f}",
            BeatTime=f"{float(beat):.6f}",
        )
        markers.append(marker)


def modify_als(
    template_path: str,
    audio_path: str,
    drop_sec: float,
    bpm: float,
    output_path: Optional[str] = None,
    strict_stems: bool = False,
) -> str:
    """Modify an Ableton template in-place in memory and save a drop-aligned ALS.

    The marker at SecTime 0.0 is preserved. Its BeatTime is shifted negative so
    the inserted drop marker can be BeatTime 0.0, which is Ableton's bar 1 beat 1.
    """

    if output_path is None:
        audio = Path(audio_path)
        output_path = str(audio.with_name(f"{audio.stem}_DROP_ALIGNED.als"))

    tree = _load_als(template_path)
    root = tree.getroot()
    clips = _audio_clips(root)
    if not clips:
        raise ValueError("Template does not contain an AudioClip.")

    role_to_clip: Dict[str, etree._Element] = {}
    for clip in clips:
        role = _clip_role(clip)
        if role and role not in role_to_clip:
            role_to_clip[role] = clip

    stem_paths = _discover_stem_paths(audio_path)
    use_stem_set = bool(role_to_clip) and len(stem_paths) > 1
    if strict_stems and role_to_clip:
        missing = [role for role in role_to_clip if role not in stem_paths]
        if missing:
            raise FileNotFoundError(f"Missing required stem(s) for ALS template: {', '.join(missing)} beside {audio_path}")
        use_stem_set = True

    target_clips: Tuple[Tuple[etree._Element, str], ...]
    if use_stem_set:
        pairs = []
        for role, clip in role_to_clip.items():
            path = stem_paths.get(role)
            if path:
                pairs.append((clip, path))
        target_clips = tuple(pairs)
    else:
        target_clips = ((_first_audio_clip(root), audio_path),)

    for clip, clip_audio_path in target_clips:
        _set_audio_file_reference(clip, clip_audio_path, output_path)
        duration_sec = _update_duration_fields(clip, clip_audio_path, bpm, drop_sec)
        _replace_warp_markers_with_drop_grid(
            clip,
            drop_sec=float(drop_sec),
            duration_sec=float(duration_sec),
            bpm=float(bpm),
        )

    save_als(tree, output_path)
    return output_path


def save_als(tree: etree._ElementTree, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = etree.tostring(
        tree,
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=False,
        standalone=None,
    )
    with gzip.open(output_path, "wb", compresslevel=6) as fh:
        fh.write(xml_bytes)
