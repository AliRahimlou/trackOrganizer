#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_fusion_audit import _json_default, build_audit


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)
DEFAULT_REKORDBOX_XML = "/Users/alirahimlou/Documents/rekordbox.xml"


def _value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or node.text or "").strip()


def _set_value(node: Optional[ET.Element], value: object) -> bool:
    if node is None:
        return False
    text = _fmt(float(value)) if isinstance(value, float) else str(value)
    old = _value(node)
    old_f = _as_float(old)
    new_f = _as_float(text)
    if old_f is not None and new_f is not None and abs(float(old_f) - float(new_f)) <= 1e-6:
        return False
    if old == text:
        return False
    if "Value" in node.attrib:
        node.set("Value", text)
    else:
        node.text = text
    return True


def _fmt(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0", "-0.0"} else text


def _as_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _marker_get(marker: ET.Element, name: str) -> Optional[float]:
    if name in marker.attrib:
        return _as_float(marker.get(name))
    return _as_float(_value(marker.find(name)))


def _marker_set(marker: ET.Element, name: str, value: float) -> None:
    text = _fmt(float(value))
    if name in marker.attrib or marker.find(name) is None:
        marker.set(name, text)
    else:
        _set_value(marker.find(name), text)


def _marker_id(marker: ET.Element) -> int:
    raw = marker.get("Id") or _value(marker.find("Id"))
    try:
        return int(raw)
    except Exception:
        return 0


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


def _iter_rows(root: ET.Element) -> Iterable[Tuple[int, Dict[str, ET.Element]]]:
    tracks = _choose_tracks(root)
    if len(tracks) < 3:
        return
    slot_lists = [_track_slots(track) for track in tracks]
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
            yield slot_idx, row


def _resolve_clip_audio_path(clip: ET.Element, als_path: str) -> Optional[str]:
    raw = _value(clip.find("./SampleRef/FileRef/Path"))
    if raw and os.path.exists(raw):
        return os.path.abspath(raw)
    rel = _value(clip.find("./SampleRef/FileRef/RelativePath"))
    if rel:
        candidate = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(als_path)), rel)))
        if os.path.exists(candidate):
            return candidate
    return None


def _markers_container(clip: ET.Element) -> ET.Element:
    container = clip.find("./WarpMarkers")
    if container is None:
        container = ET.Element("WarpMarkers")
        clip.insert(0, container)
    return container


def _marker_pairs(clip: ET.Element) -> List[Tuple[ET.Element, float, float]]:
    rows: List[Tuple[ET.Element, float, float]] = []
    for marker in _markers_container(clip).findall("./WarpMarker"):
        sec = _marker_get(marker, "SecTime")
        beat = _marker_get(marker, "BeatTime")
        if sec is None or beat is None:
            continue
        rows.append((marker, float(sec), float(beat)))
    rows.sort(key=lambda item: item[1])
    return rows


def _beat_at(markers: Sequence[Tuple[ET.Element, float, float]], sec: float, bpm: Optional[int]) -> float:
    if not markers:
        return float(sec) * float(bpm or 128) / 60.0
    for _marker, m_sec, m_beat in markers:
        if abs(float(m_sec) - float(sec)) <= 1e-7:
            return float(m_beat)
    for idx in range(len(markers) - 1):
        _m_a, sec_a, beat_a = markers[idx]
        _m_b, sec_b, beat_b = markers[idx + 1]
        if sec_a <= float(sec) <= sec_b or sec_b <= float(sec) <= sec_a:
            span = sec_b - sec_a
            if abs(span) <= 1e-9:
                return float(beat_a)
            frac = (float(sec) - sec_a) / span
            return float(beat_a + (frac * (beat_b - beat_a)))
    if len(markers) >= 2:
        a = markers[0]
        b = markers[1]
        if sec > markers[-1][1]:
            a = markers[-2]
            b = markers[-1]
        span = b[1] - a[1]
        if abs(span) > 1e-9:
            frac = (float(sec) - a[1]) / span
            return float(a[2] + (frac * (b[2] - a[2])))
    return float(sec) * float(bpm or 128) / 60.0


def _next_marker_id(markers: Sequence[Tuple[ET.Element, float, float]]) -> int:
    ids = [_marker_id(marker) for marker, _sec, _beat in markers]
    return max(ids, default=-1) + 1


def _make_marker(markers: Sequence[Tuple[ET.Element, float, float]], marker_id: int) -> ET.Element:
    marker = deepcopy(markers[-1][0]) if markers else ET.Element("WarpMarker")
    marker.set("Id", str(int(marker_id)))
    return marker


def _sort_markers(container: ET.Element) -> None:
    nodes = container.findall("./WarpMarker")
    nodes.sort(key=lambda marker: _marker_get(marker, "SecTime") if _marker_get(marker, "SecTime") is not None else float("inf"))
    for node in nodes:
        container.remove(node)
    for node in nodes:
        container.append(node)


def _current_anchor_sec(clip: ET.Element, min_sec: float = 1.0) -> Optional[float]:
    anchors = [
        sec
        for _marker, sec, beat in _marker_pairs(clip)
        if sec > float(min_sec) and abs(float(beat)) <= 0.02
    ]
    return min(anchors) if anchors else None


def _retime_clip_to_anchor(clip: ET.Element, anchor_sec: float, bpm: Optional[int]) -> Tuple[bool, float, int, int]:
    container = _markers_container(clip)
    markers = _marker_pairs(clip)
    old_count = len(markers)
    if not markers:
        return False, 0.0, old_count, old_count

    offset = _beat_at(markers, float(anchor_sec), bpm)
    changed = abs(offset) > 1e-8
    for marker, _sec, beat in markers:
        _marker_set(marker, "BeatTime", float(beat - offset))

    next_id = _next_marker_id(markers)
    current = _marker_pairs(clip)
    zero_beat = _beat_at(markers, 0.0, bpm) - offset
    if not any(abs(sec) <= 1e-7 for _marker, sec, _beat in current):
        zero = _make_marker(current or markers, next_id)
        next_id += 1
        _marker_set(zero, "SecTime", 0.0)
        _marker_set(zero, "BeatTime", zero_beat)
        container.append(zero)
        changed = True

    current = _marker_pairs(clip)
    if not any(abs(sec - float(anchor_sec)) <= 1e-5 for _marker, sec, _beat in current):
        anchor = _make_marker(current or markers, next_id)
        _marker_set(anchor, "SecTime", float(anchor_sec))
        _marker_set(anchor, "BeatTime", 0.0)
        container.append(anchor)
        changed = True

    for marker, sec, _beat in _marker_pairs(clip):
        if abs(sec - float(anchor_sec)) <= 1e-5:
            _marker_set(marker, "BeatTime", 0.0)

    is_warped = clip.find("./IsWarped")
    if is_warped is None:
        clip.insert(0, ET.Element("IsWarped", {"Value": "true"}))
        changed = True
    else:
        changed = _set_value(is_warped, "true") or changed
    _sort_markers(container)
    return changed, float(offset), old_count, len(_marker_pairs(clip))


def _shift_field(node: Optional[ET.Element], offset: float, *, minimum: Optional[float] = None) -> bool:
    raw = _as_float(_value(node))
    if raw is None:
        return False
    value = float(raw) - float(offset)
    if minimum is not None:
        value = max(float(minimum), value)
    return _set_value(node, float(value))


def _update_launch_fields(clip: ET.Element, offset: float, bpm: Optional[int]) -> bool:
    markers = _marker_pairs(clip)
    hidden = _beat_at(markers, 0.0, bpm)
    changed = False
    changed = _set_value(clip.find("./CurrentStart"), "0") or changed
    changed = _shift_field(clip.find("./CurrentEnd"), offset, minimum=0.01) or changed

    loop = clip.find("./Loop")
    if loop is not None:
        changed = _set_value(loop.find("./LoopStart"), "0") or changed
        changed = _shift_field(loop.find("./LoopEnd"), offset, minimum=0.01) or changed
        changed = _shift_field(loop.find("./OutMarker"), offset, minimum=0.01) or changed
        changed = _set_value(loop.find("./HiddenLoopStart"), float(hidden)) or changed
        changed = _set_value(loop.find("./HiddenLoopEnd"), float(hidden + 4.0)) or changed

    scroller = clip.find("./ScrollerTimePreserver")
    if scroller is not None:
        changed = _set_value(scroller.find("./LeftTime"), float(hidden)) or changed
        changed = _shift_field(scroller.find("./RightTime"), offset, minimum=0.01) or changed

    selection = clip.find("./TimeSelection")
    if selection is not None:
        changed = _set_value(selection.find("./AnchorTime"), float(hidden)) or changed
        changed = _set_value(selection.find("./OtherTime"), float(hidden)) or changed
    return changed


def _set_global_quantization(root: ET.Element, value: int = 4) -> int:
    changed = 0
    for tag in ("GlobalQuantisation", "GlobalQuantization"):
        for node in root.iter(tag):
            if _set_value(node, str(int(value))):
                changed += 1
    return changed


def _folder_key(path: str) -> str:
    return os.path.abspath(os.path.dirname(path))


def _current_als_seed(clip: ET.Element, role: str) -> Optional[Dict[str, Any]]:
    sec = _current_anchor_sec(clip)
    if sec is None:
        return None
    return {
        "seed_sec": float(sec),
        "source": "current_als_111",
        "role": role,
        "source_score": 0.60,
        "snap_window_beats": 0.60,
        "clip_name": _clip_name(clip),
    }


def _load_cached_report(path: str, force: bool) -> Optional[Dict[str, Any]]:
    if force or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("analysis_version") == "drop_fusion_audit_v1" else None


def _analyze_folder(job: Mapping[str, Any]) -> Dict[str, Any]:
    folder = str(job["folder"])
    report_path = str(job["report_path"])
    cached = _load_cached_report(report_path, bool(job.get("force_analysis")))
    if cached is not None:
        cached["_cache_status"] = "reused"
        return {"folder": folder, "audit": cached, "report_path": report_path}

    audit = build_audit(
        folder,
        als_path=None,
        current_als_seeds=list(job.get("current_als_seeds") or []),
        current_als_meta=dict(job.get("current_als_meta") or {}),
        source_audio_path=None,
        rekordbox_xml_path=str(job.get("rekordbox_xml") or DEFAULT_REKORDBOX_XML),
        sample_rate=int(job.get("sample_rate") or 22050),
        per_role_limit=int(job.get("per_role_limit") or 28),
        candidate_limit=int(job.get("candidate_limit") or 40),
    )
    audit["_cache_status"] = "generated"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    return {"folder": folder, "audit": audit, "report_path": report_path}


def _suggestion(audit: Mapping[str, Any]) -> Dict[str, Any]:
    suggestion = audit.get("suggestion")
    return dict(suggestion) if isinstance(suggestion, Mapping) else {}


def _safe_suggestion_time(audit: Mapping[str, Any], require_safe: bool) -> Optional[float]:
    suggestion = _suggestion(audit)
    if not suggestion.get("available"):
        return None
    if require_safe and not bool(suggestion.get("safe_to_write")):
        return None
    sec = _as_float(suggestion.get("time_sec"))
    return float(sec) if sec is not None and sec > 0.0 else None


def build_set(
    *,
    als_in: str,
    als_out: str,
    workers: int,
    force_analysis: bool,
    require_safe: bool,
    sample_rate: int,
    per_role_limit: int,
    candidate_limit: int,
    rekordbox_xml: str,
) -> Dict[str, Any]:
    start = time.time()
    with gzip.open(als_in, "rb") as fh:
        root = ET.fromstring(fh.read())

    rows = list(_iter_rows(root))
    folders: Dict[str, Dict[str, Any]] = {}
    row_infos: List[Dict[str, Any]] = []
    original_marker_count = sum(1 for clip in root.iter("AudioClip") for _ in clip.findall("./WarpMarkers/WarpMarker"))

    for slot_idx, row in rows:
        paths = {role: _resolve_clip_audio_path(clip, als_in) for role, clip in row.items()}
        drums_path = paths.get("drums")
        if not drums_path:
            continue
        folder = _folder_key(drums_path)
        bpm = _clip_bpm(row["drums"])
        seeds = []
        for role, clip in row.items():
            seed_role = "instrumental" if role == "inst" else role
            seed = _current_als_seed(clip, seed_role)
            if seed is not None:
                seeds.append(seed)
        info = folders.setdefault(
            folder,
            {
                "folder": folder,
                "bpm": bpm,
                "current_als_seeds": [],
                "rows": [],
                "report_path": os.path.join(folder, "drop_fusion_audit.json"),
            },
        )
        info["current_als_seeds"].extend(seeds)
        info["rows"].append(slot_idx)
        row_infos.append({"slot_idx": slot_idx, "folder": folder, "row": row, "paths": paths, "bpm": bpm})

    jobs = []
    for folder, info in sorted(folders.items()):
        deduped: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int]] = set()
        for seed in info.get("current_als_seeds", []):
            key = (str(seed.get("role") or ""), int(round(float(seed.get("seed_sec", 0.0) or 0.0) * 1000.0)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(seed))
        jobs.append(
            {
                "folder": folder,
                "report_path": info["report_path"],
                "current_als_seeds": deduped,
                "current_als_meta": {
                    "available": True,
                    "source": "set_precomputed_current_als_111",
                    "anchor_count": len(deduped),
                    "rows": list(info.get("rows", [])),
                },
                "force_analysis": bool(force_analysis),
                "rekordbox_xml": rekordbox_xml,
                "sample_rate": int(sample_rate),
                "per_role_limit": int(per_role_limit),
                "candidate_limit": int(candidate_limit),
            }
        )

    print(f"[FUSION-SET] rows={len(rows)} folders={len(jobs)} workers={workers}")
    audits: Dict[str, Dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {executor.submit(_analyze_folder, job): job["folder"] for job in jobs}
        for future in as_completed(future_map):
            folder = future_map[future]
            completed += 1
            try:
                result = future.result()
                audits[folder] = dict(result["audit"])
                suggestion = _suggestion(audits[folder])
                sec = _as_float(suggestion.get("time_sec"))
                safe = bool(suggestion.get("safe_to_write"))
                cache = audits[folder].get("_cache_status", "")
                print(
                    f"[FUSION-SET] {completed}/{len(jobs)} {cache} "
                    f"{Path(folder).name}: {sec if sec is not None else 'n/a'} safe={safe}"
                )
            except Exception as exc:
                audits[folder] = {"ok": False, "error": str(exc), "suggestion": {"available": False}}
                print(f"[FUSION-SET] ERROR {Path(folder).name}: {exc}")

    stats = {
        "rows_total": len(rows),
        "folders_total": len(jobs),
        "folders_safe": 0,
        "folders_review": 0,
        "folders_error": 0,
        "rows_retimed": 0,
        "clips_retimed": 0,
        "clips_seen": 0,
        "marker_count_before": original_marker_count,
        "marker_count_after": 0,
        "global_quantization_updates": 0,
        "elapsed_sec": 0.0,
    }
    report_rows: List[Dict[str, Any]] = []

    for folder, audit in audits.items():
        if not audit.get("ok", True):
            stats["folders_error"] += 1
        elif _safe_suggestion_time(audit, require_safe=require_safe) is not None:
            stats["folders_safe"] += 1
        else:
            stats["folders_review"] += 1

    for info in row_infos:
        folder = str(info["folder"])
        audit = audits.get(folder, {})
        suggestion = _suggestion(audit)
        anchor_sec = _safe_suggestion_time(audit, require_safe=require_safe)
        changed_in_row = 0
        row_status = "applied" if anchor_sec is not None else "held"
        if anchor_sec is not None:
            bpm = info.get("bpm")
            for role in ("drums", "inst", "vocals"):
                clip = info["row"][role]
                stats["clips_seen"] += 1
                changed, offset, old_count, new_count = _retime_clip_to_anchor(clip, float(anchor_sec), bpm)
                fields_changed = _update_launch_fields(clip, float(offset), bpm)
                if changed or fields_changed:
                    changed_in_row += 1
                    stats["clips_retimed"] += 1
                if new_count < old_count:
                    raise RuntimeError(f"Marker count decreased for {_clip_name(clip)}")
            if changed_in_row:
                stats["rows_retimed"] += 1
        report_rows.append(
            {
                "slot_idx": int(info["slot_idx"]),
                "folder": folder,
                "status": row_status,
                "suggested_sec": "" if suggestion.get("time_sec") is None else f"{float(suggestion.get('time_sec')):.6f}",
                "score": "" if suggestion.get("score") is None else f"{float(suggestion.get('score')):.6f}",
                "confidence": "" if suggestion.get("confidence") is None else f"{float(suggestion.get('confidence')):.6f}",
                "margin": "" if suggestion.get("margin_from_runner_up") is None else f"{float(suggestion.get('margin_from_runner_up')):.6f}",
                "safe_to_write": str(bool(suggestion.get("safe_to_write"))),
                "sources": ",".join(str(src) for src in (suggestion.get("sources") or [])),
                "report": os.path.join(folder, "drop_fusion_audit.json"),
            }
        )

    stats["global_quantization_updates"] = _set_global_quantization(root, 4)
    stats["marker_count_after"] = sum(1 for clip in root.iter("AudioClip") for _ in clip.findall("./WarpMarkers/WarpMarker"))
    stats["elapsed_sec"] = round(time.time() - start, 3)

    if stats["marker_count_after"] < stats["marker_count_before"]:
        raise RuntimeError("Output marker count decreased; refusing to write.")

    out_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    out_path = Path(als_out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb") as fh:
        fh.write(out_xml)

    csv_path = str(out_path.with_suffix("")) + "_fusion_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "slot_idx",
                "folder",
                "status",
                "suggested_sec",
                "score",
                "confidence",
                "margin",
                "safe_to_write",
                "sources",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    json_path = str(out_path.with_suffix("")) + "_fusion_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        payload = {"input": als_in, "output": str(out_path), "stats": stats}
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    stats["output"] = str(out_path)
    stats["csv_report"] = csv_path
    stats["json_report"] = json_path
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a new set ALS using fused drop anchors while preserving warp markers.")
    parser.add_argument("--als", required=True, help="Input ALS set")
    parser.add_argument("--out", required=True, help="Output ALS set")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-analysis", action="store_true", help="Regenerate drop_fusion_audit.json reports even if cached reports exist.")
    parser.add_argument("--allow-review-candidates", action="store_true", help="Apply suggestions even when the fusion safe gate is false.")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--per-role-limit", type=int, default=28)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--rekordbox-xml", default=os.environ.get("REKORDBOX_XML_PATH", DEFAULT_REKORDBOX_XML))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.als):
        print(f"Input ALS not found: {args.als}", file=sys.stderr)
        return 2
    stats = build_set(
        als_in=args.als,
        als_out=args.out,
        workers=int(args.workers),
        force_analysis=bool(args.force_analysis),
        require_safe=not bool(args.allow_review_candidates),
        sample_rate=int(args.sample_rate),
        per_role_limit=int(args.per_role_limit),
        candidate_limit=int(args.candidate_limit),
        rekordbox_xml=str(args.rekordbox_xml),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
