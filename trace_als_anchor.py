#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from first_downbeat_detector import build_als_anchor_map, detect_track_folder
from trackOrganizerAndAlsGen import (
    DROP_ANCHOR_OVERRIDES_SEC,
    FLAC_FOLDER,
    _as_float,
    _clip_role_from_audio_clip,
    _detect_alignment_with_first_downbeat,
    _lookup_drop_from_db,
    _manual_drop_from_project_als,
    _max_audible_end_seconds,
    _phi_map_from_anchor_map,
    _role_audible_end_seconds,
    _shift_anchor_map,
    _uniform_anchor_map,
    collect_track_names_for_folder,
)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_clip_state(als_path: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(gzip.open(als_path, "rb").read())
    rows: List[Dict[str, Any]] = []
    for clip in root.iter("AudioClip"):
        role = _clip_role_from_audio_clip(clip)
        if role not in {"drums", "inst", "vocals"}:
            continue
        warp_markers = []
        for wm in clip.findall("WarpMarkers/WarpMarker"):
            warp_markers.append(
                {
                    "id": wm.get("Id"),
                    "sec_time": _safe_float(wm.get("SecTime")),
                    "beat_time": _safe_float(wm.get("BeatTime")),
                }
            )
        loop_end = None
        out_marker = None
        for node in clip.iter("LoopEnd"):
            loop_end = _safe_float(node.get("Value"))
            break
        for node in clip.iter("OutMarker"):
            out_marker = _safe_float(node.get("Value"))
            break
        beat_zero_marker = None
        for marker in warp_markers:
            beat = _safe_float(marker.get("beat_time"))
            if beat is not None and abs(beat) <= 1e-5:
                beat_zero_marker = marker
                break
        rows.append(
            {
                "role": role,
                "beat_zero_marker": beat_zero_marker,
                "warp_markers": warp_markers,
                "loop_end": loop_end,
                "out_marker": out_marker,
            }
        )
    return rows


def _resolve_bpm(track_dir: str, detected_bpm: Optional[float], override_bpm: Optional[int]) -> Optional[int]:
    if override_bpm:
        return int(override_bpm)
    if detected_bpm and detected_bpm > 0:
        return int(round(float(detected_bpm)))
    for part in os.path.normpath(track_dir).split(os.sep):
        try:
            bpm = int(part)
        except Exception:
            continue
        if 40 <= bpm <= 220:
            return bpm
    return None


def build_trace(track_dir: str, bpm_override: Optional[int] = None) -> Dict[str, Any]:
    track_dir = os.path.abspath(track_dir)
    track_names = collect_track_names_for_folder(track_dir)
    detection = detect_track_folder(track_dir, debug_dir=None, generate_plots=False)
    detected_bpm = _safe_float(detection.get("bpm"))
    bpm = _resolve_bpm(track_dir, detected_bpm, bpm_override)

    detector_anchor_map = build_als_anchor_map(detection) if build_als_anchor_map is not None else {}
    fd_anchor_map, fd_drums_sec, fd_conf, fd_strategy = _detect_alignment_with_first_downbeat(track_dir, track_names)
    role_anchor_map = dict(fd_anchor_map)
    drop_sec = fd_drums_sec
    drop_conf = float(fd_conf or 0.0)
    used_first_downbeat = fd_drums_sec is not None and bpm is not None
    used_db_anchor = False
    used_manual_anchor = False
    drums_source_path = None
    if track_names.get("drums"):
        flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
        if os.path.exists(flac_path):
            drums_source_path = flac_path

    db_lookup = None
    if bpm and drums_source_path:
        db_sec, db_conf, db_reason = _lookup_drop_from_db(drums_source_path, bpm)
        db_lookup = {"seconds": db_sec, "confidence": db_conf, "reason": db_reason}
        if db_sec is not None:
            drop_sec = float(db_sec)
            drop_conf = max(drop_conf, float(db_conf))
            used_db_anchor = True
            if used_first_downbeat and fd_drums_sec is not None and role_anchor_map:
                role_anchor_map = _shift_anchor_map(role_anchor_map, float(db_sec) - float(fd_drums_sec))
            else:
                role_anchor_map = _uniform_anchor_map(track_names, drop_sec)

    manual_sec = _manual_drop_from_project_als(track_dir, bpm) if bpm else None
    if manual_sec is not None:
        drop_sec = float(manual_sec)
        drop_conf = max(drop_conf, 0.99)
        used_manual_anchor = True
        if used_first_downbeat and fd_drums_sec is not None and role_anchor_map:
            role_anchor_map = _shift_anchor_map(role_anchor_map, float(manual_sec) - float(fd_drums_sec))
        else:
            role_anchor_map = _uniform_anchor_map(track_names, drop_sec)

    offset_override = None
    folder_name = os.path.basename(track_dir)
    if drop_sec is not None:
        offset_override = _safe_float(DROP_ANCHOR_OVERRIDES_SEC.get(folder_name))
        if offset_override:
            drop_sec = max(0.0, float(drop_sec) + float(offset_override))
            if role_anchor_map:
                role_anchor_map = _shift_anchor_map(role_anchor_map, float(offset_override))

    if (not role_anchor_map) and drop_sec is not None:
        role_anchor_map = _uniform_anchor_map(track_names, drop_sec)

    role_end_sec_map = _role_audible_end_seconds(track_names)
    end_sec = max(role_end_sec_map.values()) if role_end_sec_map else _max_audible_end_seconds(track_names)
    phi_map = _phi_map_from_anchor_map(role_anchor_map, bpm) if (role_anchor_map and bpm) else {}
    output_als = os.path.join(track_dir, "CH1.als")
    saved_clip_state = _extract_clip_state(output_als) if os.path.exists(output_als) else []

    debug = detection.get("debug") or {}
    ableton_snap = debug.get("ableton_snap") or {}
    return {
        "track_dir": track_dir,
        "output_als": output_als,
        "bpm": bpm,
        "track_names": track_names,
        "detector": {
            "rough_custom_seconds": debug.get("rough_custom_candidate", {}).get("time_abs_sec") if isinstance(debug.get("rough_custom_candidate"), dict) else None,
            "custom_reference_seconds": debug.get("custom_reference_candidate", {}).get("time_abs_sec") if isinstance(debug.get("custom_reference_candidate"), dict) else None,
            "final_chosen_asd_marker_seconds": ableton_snap.get("chosen_marker_seconds"),
            "final_detector_drums_seconds": detection.get("drums", {}).get("downbeat_seconds"),
            "confidence": detection.get("drums", {}).get("confidence"),
            "strategy": debug.get("candidate_strategy"),
            "debug_reason": ableton_snap.get("chosen_marker_reason"),
        },
        "writer": {
            "anchor_map_from_detector": detector_anchor_map,
            "anchor_map_seen_by_writer": fd_anchor_map,
            "effective_anchor_map_passed_to_stamper": role_anchor_map,
            "drop_sec_after_overrides": drop_sec,
            "drop_conf_after_overrides": drop_conf,
            "used_first_downbeat": used_first_downbeat,
            "used_db_anchor": used_db_anchor,
            "used_manual_anchor": used_manual_anchor,
            "db_lookup": db_lookup,
            "manual_override_seconds": manual_sec,
            "drop_anchor_offset_seconds": offset_override,
            "role_end_sec_map": role_end_sec_map,
            "end_sec": end_sec,
            "phi_map": phi_map,
            "first_downbeat_strategy": fd_strategy,
        },
        "saved_als": {
            "clip_states": saved_clip_state,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace first-downbeat detection -> ALS writer -> saved CH1.als warp markers.")
    ap.add_argument("--track-dir", required=True, help="Track folder containing stems and CH1.als")
    ap.add_argument("--bpm", type=int, default=None, help="Optional BPM override")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    payload = build_trace(args.track_dir, bpm_override=args.bpm)
    text = json.dumps(payload, indent=2, default=_json_default)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    print(text)


if __name__ == "__main__":
    main()
