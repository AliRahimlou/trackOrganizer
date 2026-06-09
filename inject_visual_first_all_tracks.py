#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import buildSetAndGenerateAls as builder
from apply_folder_drop_candidates_to_set import (
    _clip_bpm,
    _clip_duration_sec,
    _clip_name,
    _current_anchor_sec,
    _fmt,
    _iter_triplet_rows,
    _last_marker_sec,
    _local_drop_sec,
    _replace_warp_markers,
    _resolve_clip_audio_path,
    _row_end_sec,
    _sec_to_beats,
    _set_value,
    _slot_audio_clip,
)


DEFAULT_SOURCE_ALS = Path("/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158.als")
DEFAULT_STEMS_DIR = Path("/Users/alirahimlou/Desktop/MUSIC/STEMS")
ROLE_TO_CH = {"drums": 1, "inst": 2, "vocals": 3}


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_path(source_als: Path) -> Path:
    return source_als.with_name(f"{source_als.stem}_VISUAL_FIRST_ALL_TRACKS_{_timestamp()}{source_als.suffix}")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    ts = _timestamp()
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def _read_root(path: Path) -> ET.Element:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return ET.fromstring(raw)


def _write_root(path: Path, root: ET.Element) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    builder.write_als_gz(path, root)


def _format_scene_name(track: Dict[str, Any]) -> str:
    energy = track.get("energy")
    folder_name = builder.swap_artist_title(track["folder"]) if builder.RENAME_TRACKS else track["folder"]
    if energy is not None:
        return f"{track['bpm']}_{track['key']}_{energy}-{folder_name}"
    return f"{track['bpm']}_{track['key']}-{folder_name}"


def _first_present(*items: Optional[ET.Element]) -> Optional[ET.Element]:
    for item in items:
        if item is not None:
            return item
    return None


def _row_audio_clips(ch_map: Dict[int, ET.Element], row_idx: int, chs: Sequence[int] = (1, 2, 3)) -> List[ET.Element]:
    clips: List[ET.Element] = []
    for ch in chs:
        track = ch_map.get(ch)
        if track is None:
            continue
        slots = builder.track_slot_list(track) or []
        if row_idx >= len(slots):
            continue
        clip = _slot_audio_clip(slots[row_idx])
        if clip is not None:
            clips.append(clip)
    return clips


def _row_anchor_seconds(ch_map: Dict[int, ET.Element], row_idx: int) -> List[float]:
    anchors: List[float] = []
    for clip in _row_audio_clips(ch_map, row_idx):
        anchor = _current_anchor_sec(clip)
        if anchor is not None:
            anchors.append(float(anchor))
    return anchors


def _row_has_middle_anchor(ch_map: Dict[int, ET.Element], row_idx: int) -> bool:
    return bool(_row_anchor_seconds(ch_map, row_idx))


def _count_middle_anchor_rows(root: ET.Element) -> int:
    ch_map = builder.ch_tracks_map(root)
    if not ch_map:
        return 0
    ref = _first_present(ch_map.get(1), ch_map.get(2), ch_map.get(3))
    slots = builder.track_slot_list(ref) if ref is not None else []
    total = len(slots or [])
    return sum(1 for row_idx in range(total) if _row_has_middle_anchor(ch_map, row_idx))


def repair_existing_missing_anchors(
    root: ET.Element,
    source_als: Path,
    *,
    dry_run: bool,
) -> Dict[str, int]:
    stats = {
        "rows_total": 0,
        "rows_already_anchored": 0,
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
        current = _current_anchor_sec(drums)
        if current is not None:
            stats["rows_already_anchored"] += 1
            continue
        audio_path = _resolve_clip_audio_path(drums, str(source_als))
        if not audio_path:
            stats["rows_skipped_no_audio"] += 1
            continue
        drop_sec, source = _local_drop_sec(audio_path)
        if drop_sec is None:
            stats["rows_skipped_no_candidate"] += 1
            continue
        stats["rows_with_local_candidate"] += 1
        end_sec = _row_end_sec(row, str(source_als), float(drop_sec))
        if not dry_run:
            for role in ("drums", "inst", "vocals"):
                _replace_warp_markers(row[role], int(bpm), float(drop_sec), float(end_sec))
        stats["rows_changed"] += 1
        stats["clips_retimed"] += 3
        print(
            f"[REPAIR] slot={slot_idx} -> {float(drop_sec):.3f}s "
            f"source={source} name={_clip_name(drums)}"
        )
    return stats


def _source_clip_names_from_stems(track: Dict[str, Any]) -> Dict[int, str]:
    folder = Path(track["src"]).parent
    out: Dict[int, str] = {}
    try:
        children = list(folder.iterdir())
    except OSError:
        return out
    for child in children:
        if not child.is_file():
            continue
        match = builder.STEM_RE.match(child.stem)
        if not match:
            continue
        role, bpm, key, energy = match.groups()
        ch = ROLE_TO_CH.get(role.lower())
        if not ch or ch in out:
            continue
        if int(bpm) != int(track["bpm"]) or key.upper() != str(track["key"]).upper():
            continue
        if track.get("energy") is not None and int(energy) != int(track["energy"]):
            continue
        out[ch] = child.stem
    return out


def _load_source_names(track: Dict[str, Any], src_als: Path) -> Dict[int, str]:
    fast = _source_clip_names_from_stems(track)
    if fast:
        return fast
    try:
        return builder.source_clip_names(src_als)
    except Exception:
        return {}


def _write_playlist_csv(path: Path, tracks: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["#", "BPM", "Key", "Energy", "TrackFolder", "SourcePath", "PreferredAls"])
        for idx, track in enumerate(tracks, start=1):
            src_path = Path(track["src"])
            preferred = builder.preferred_source_als(src_path)
            writer.writerow([
                idx,
                track["bpm"],
                track["key"],
                track.get("energy") if track.get("energy") is not None else "",
                track["folder"],
                src_path,
                preferred,
            ])


def _existing_row_infos(root: ET.Element, ch_map: Dict[int, ET.Element]) -> List[Dict[str, Any]]:
    scenes = builder.scenes_node(root).findall("./Scene")
    reference_track = _first_present(ch_map.get(1), ch_map.get(2), ch_map.get(3), ch_map.get(4))
    ref_slots = builder.track_slot_list(reference_track) if reference_track is not None else []
    max_rows = max(
        [len(scenes), len(ref_slots or [])]
        + [len(builder.track_slot_list(track) or []) for track in ch_map.values()]
    )
    rows: List[Dict[str, Any]] = []
    for row_idx in range(max_rows):
        names = set()
        anchors = _row_anchor_seconds(ch_map, row_idx)
        for clip in _row_audio_clips(ch_map, row_idx):
            name = _clip_name(clip)
            if name:
                names.add(builder.norm_text(name))

        key = None
        if ref_slots is not None and row_idx < len(ref_slots):
            key = builder.scene_sort_key_from_clipname(builder.clip_name_from_slot(ref_slots[row_idx]))
        if key is None and row_idx < len(scenes):
            key = builder.scene_sort_key_from_name(builder.scene_label(scenes[row_idx]))

        rows.append({
            "row": row_idx,
            "names": names,
            "anchors": anchors,
            "protected": bool(anchors),
            "key": key,
        })
    return rows


def _name_to_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for row in rows:
        row_idx = int(row["row"])
        for name in row["names"]:
            index.setdefault(name, []).append(row_idx)
    return index


def _current_row_keys(root: ET.Element, ch_map: Dict[int, ET.Element]) -> List[Any]:
    return [row["key"] for row in _existing_row_infos(root, ch_map)]


def _find_insert_index_from_keys(row_keys: Sequence[Any], new_key: Any) -> int:
    parsed = [(idx, key) for idx, key in enumerate(row_keys) if key is not None]
    if not parsed:
        return len(row_keys)
    last_le = None
    for idx, key in parsed:
        if key <= new_key:
            last_le = idx
    if last_le is None:
        return parsed[0][0]
    return last_le + 1


def inject_visual_first_tracks(
    root: ET.Element,
    tracks: Sequence[Dict[str, Any]],
    *,
    dry_run: bool,
    allow_unaligned_fallback: bool,
) -> Dict[str, Any]:
    tp = builder.tracks_parent(root)
    if tp is None:
        raise RuntimeError("Base ALS has no <Tracks> node")
    ch_map = builder.ch_tracks_map(root)
    if not ch_map:
        raise RuntimeError("Base ALS has no CH1..CH4 tracks")

    all_tracks = [track for track in list(tp) if track.tag in ("AudioTrack", "MidiTrack", "GroupTrack")]
    reference_track = _first_present(ch_map.get(1), ch_map.get(2), ch_map.get(3), ch_map.get(4))
    if reference_track is None:
        raise RuntimeError("Could not find a reference CH track")

    existing_rows = _existing_row_infos(root, ch_map)
    name_index = _name_to_rows(existing_rows)
    rows_by_idx = {int(row["row"]): row for row in existing_rows}
    rows_to_remove = set()
    scheduled: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "tracks_seen": 0,
        "inserted_missing": 0,
        "replaced_unanchored_duplicates": 0,
        "skipped_protected_duplicates": 0,
        "skipped_no_source_names": 0,
        "skipped_missing_drop_aligned_sources": 0,
        "missing_drop_aligned_sources": 0,
        "duplicate_rows_removed": 0,
        "events": [],
    }

    for track in tracks:
        stats["tracks_seen"] += 1
        scene_name = _format_scene_name(track)
        source_ch1 = Path(track["src"])
        src_als = builder.preferred_source_als(source_ch1)
        if src_als == source_ch1:
            stats["missing_drop_aligned_sources"] += 1
            if not allow_unaligned_fallback:
                stats["skipped_missing_drop_aligned_sources"] += 1
                stats["events"].append({
                    "action": "skip_missing_drop_aligned_source",
                    "scene": scene_name,
                    "source": str(src_als),
                })
                continue
        src_names = _load_source_names(track, src_als)
        if not src_names:
            stats["skipped_no_source_names"] += 1
            stats["events"].append({
                "action": "skip_no_source_names",
                "scene": scene_name,
                "source": str(src_als),
            })
            continue

        wanted_names = [builder.norm_text(name) for name in src_names.values() if name]
        dup_rows = sorted({row_idx for name in wanted_names for row_idx in name_index.get(name, [])})
        protected_rows = [row_idx for row_idx in dup_rows if rows_by_idx.get(row_idx, {}).get("protected")]
        if protected_rows:
            stats["skipped_protected_duplicates"] += 1
            stats["events"].append({
                "action": "skip_protected_duplicate",
                "scene": scene_name,
                "rows": protected_rows,
                "anchors": {str(row): rows_by_idx.get(row, {}).get("anchors", []) for row in protected_rows},
                "source": str(src_als),
            })
            continue

        if dup_rows:
            stats["replaced_unanchored_duplicates"] += 1
            stats["duplicate_rows_removed"] += len(dup_rows)
            rows_to_remove.update(dup_rows)
            stats["events"].append({
                "action": "replace_unanchored_duplicate",
                "scene": scene_name,
                "rows": dup_rows,
                "source": str(src_als),
            })
        else:
            stats["inserted_missing"] += 1

        scheduled.append({
            "track": track,
            "scene": scene_name,
            "source": src_als,
            "replace": bool(dup_rows),
        })

    print(
        "[PLAN] "
        f"tracks={stats['tracks_seen']} "
        f"keep_protected={stats['skipped_protected_duplicates']} "
        f"replace={stats['replaced_unanchored_duplicates']} "
        f"insert_missing={stats['inserted_missing']} "
        f"remove_rows={len(rows_to_remove)} "
        f"missing_drop_aligned={stats['missing_drop_aligned_sources']} "
        f"skip_missing_drop_aligned={stats['skipped_missing_drop_aligned_sources']} "
        f"skip_no_names={stats['skipped_no_source_names']}",
        flush=True,
    )

    if not dry_run and rows_to_remove:
        for row_idx in sorted(rows_to_remove, reverse=True):
            builder.remove_scene_row(root, row_idx)
        print(f"[APPLY] Removed {len(rows_to_remove)} unanchored duplicate row(s)", flush=True)

    if dry_run:
        row_keys = [row["key"] for row in existing_rows if int(row["row"]) not in rows_to_remove]
    else:
        row_keys = _current_row_keys(root, ch_map)

    for idx, item in enumerate(scheduled, start=1):
        track = item["track"]
        scene_name = item["scene"]
        src_als = item["source"]
        insert_idx = _find_insert_index_from_keys(row_keys, builder.scene_sort_key_from_track(track))
        if not dry_run:
            builder.insert_scene(root, all_tracks, ch_map, src_als, scene_name, insert_idx)
        row_keys.insert(insert_idx, builder.scene_sort_key_from_track(track))
        stats["events"].append({
            "action": "insert_replacement" if item["replace"] else "insert_missing",
            "scene": scene_name,
            "insert_idx": insert_idx,
            "source": str(src_als),
        })
        if idx % 100 == 0 or idx == len(scheduled):
            print(f"[APPLY] Planned/applied {idx}/{len(scheduled)} insert row(s)", flush=True)

    return stats


def _verify_als(path: Path) -> Dict[str, Any]:
    root = _read_root(path)
    scenes = builder.scenes_node(root).findall("./Scene")
    ch_map = builder.ch_tracks_map(root)
    slot_counts = {}
    for ch, track in sorted(ch_map.items()):
        slots = builder.track_slot_list(track) or []
        slot_counts[f"CH{ch}"] = len(slots)
    return {
        "valid_xml": True,
        "audio_tracks": len(root.findall(".//AudioTrack")),
        "audio_clips": len(root.findall(".//AudioClip")),
        "scenes": len(scenes),
        "slot_counts": slot_counts,
        "middle_anchor_rows": _count_middle_anchor_rows(root),
    }


def _extend_anchored_clip_ends_to_audio_duration(root: ET.Element, als_path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "anchored_clips_seen": 0,
        "clips_extended": 0,
        "clips_skipped_no_bpm": 0,
        "clips_skipped_no_duration": 0,
    }
    for clip in root.findall(".//AudioClip"):
        anchor_sec = _current_anchor_sec(clip)
        if anchor_sec is None:
            continue
        stats["anchored_clips_seen"] += 1
        bpm = _clip_bpm(clip)
        if not bpm:
            stats["clips_skipped_no_bpm"] += 1
        duration_sec = _clip_duration_sec(clip, str(als_path))
        if duration_sec is None:
            stats["clips_skipped_no_duration"] += 1
        end_values: List[float] = []
        for node in (clip.find("./CurrentEnd"), clip.find("./Loop/LoopEnd"), clip.find("./Loop/OutMarker")):
            if node is None:
                continue
            try:
                end_values.append(float(node.get("Value")))
            except Exception:
                pass
        if bpm and duration_sec is not None:
            end_sec = max(float(duration_sec), float(_last_marker_sec(clip) or 0.0), float(anchor_sec) + 1.0)
            desired_end_beat = max(0.01, _sec_to_beats(float(end_sec) - float(anchor_sec), int(bpm)))
        else:
            desired_end_beat = max([value for value in end_values if value > 0.0], default=0.0)
            if desired_end_beat <= 0.0:
                continue
        current_end = min(end_values) if end_values else 0.0
        if current_end >= desired_end_beat - 0.25:
            continue
        _set_value(clip.find("./CurrentEnd"), _fmt(desired_end_beat))
        loop = clip.find("./Loop")
        if loop is not None:
            _set_value(loop.find("./LoopEnd"), _fmt(desired_end_beat))
            _set_value(loop.find("./OutMarker"), _fmt(desired_end_beat))
        scroller = clip.find("./ScrollerTimePreserver")
        if scroller is not None:
            _set_value(scroller.find("./RightTime"), _fmt(desired_end_beat))
        stats["clips_extended"] += 1
    return stats


def run(args: argparse.Namespace) -> int:
    source_als = Path(args.als).expanduser().resolve()
    stems_dir = Path(args.stems).expanduser().resolve()
    if not source_als.exists():
        raise FileNotFoundError(source_als)
    if not stems_dir.exists():
        raise FileNotFoundError(stems_dir)

    out_path = Path(args.out).expanduser().resolve() if args.out else _default_output_path(source_als)
    out_path = _unique_path(out_path)
    report_path = out_path.with_name(f"{out_path.stem}_report.json")
    playlist_path = out_path.with_suffix(".csv")

    print(f"[LOAD] {source_als}")
    root = _read_root(source_als)
    before_anchor_rows = _count_middle_anchor_rows(root)
    repair_stats = repair_existing_missing_anchors(root, source_als, dry_run=bool(args.dry_run))
    after_repair_anchor_rows = _count_middle_anchor_rows(root)

    tracks = builder.sort_by_bpm_key_energy(builder.collect_tracks(str(stems_dir)))
    print(f"[POOL] {len(tracks)} STEMS tracks found")
    _write_playlist_csv(playlist_path, tracks)
    print(f"[LOG] Playlist CSV -> {playlist_path}")

    inject_stats = inject_visual_first_tracks(
        root,
        tracks,
        dry_run=bool(args.dry_run),
        allow_unaligned_fallback=bool(args.allow_unaligned_fallback),
    )

    verification: Optional[Dict[str, Any]] = None
    final_end_repair: Optional[Dict[str, Any]] = None
    if not args.dry_run:
        final_end_repair = _extend_anchored_clip_ends_to_audio_duration(root, out_path)
        print(
            "[REPAIR] Final end sweep "
            f"anchored={final_end_repair['anchored_clips_seen']} "
            f"extended={final_end_repair['clips_extended']} "
            f"skip_no_bpm={final_end_repair['clips_skipped_no_bpm']} "
            f"skip_no_duration={final_end_repair['clips_skipped_no_duration']}",
            flush=True,
        )
        _write_root(out_path, root)
        verification = _verify_als(out_path)
        print(f"[DONE] Wrote ALS -> {out_path}")
    else:
        print(f"[DRY RUN] ALS not written. Planned output would be -> {out_path}")

    report = {
        "source_als": str(source_als),
        "output_als": str(out_path),
        "stems_dir": str(stems_dir),
        "dry_run": bool(args.dry_run),
        "before_middle_anchor_rows": before_anchor_rows,
        "after_repair_middle_anchor_rows": after_repair_anchor_rows,
        "repair": repair_stats,
        "inject": inject_stats,
        "final_end_repair": final_end_repair,
        "verification": verification,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[LOG] Report JSON -> {report_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of an ALS set, repair default-at-start clips from local visual-first "
            "DROP_ALIGNED markers, and inject all missing STEMS tracks while preserving existing "
            "middle/drop anchors."
        )
    )
    parser.add_argument("--als", default=str(DEFAULT_SOURCE_ALS), help="Source ALS to copy/inject into.")
    parser.add_argument("--stems", default=str(DEFAULT_STEMS_DIR), help="STEMS root containing BPM/key track folders.")
    parser.add_argument("--out", default="", help="Output ALS path. Defaults beside source with timestamp.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and report without writing the ALS.")
    parser.add_argument(
        "--allow-unaligned-fallback",
        action="store_true",
        help="Allow inserting plain CH1.als sources when no visual-first DROP_ALIGNED ALS exists.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
