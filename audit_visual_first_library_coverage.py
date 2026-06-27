#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import buildSetAndGenerateAls as builder
from drop_aligner.exclusions import (
    drums_stem_signal_stats,
    is_near_empty_drums_stem,
    row_has_excluded_path,
    row_is_acapella,
)
from project_config import GENERATED_SET_DIR, STEMS_ROOT_DIR


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}
DEFAULT_SUMMARY = (
    GENERATED_SET_DIR
    / "VisualFirstFresh"
    / "VISUAL_FIRST_PRODUCTION_ALL_TRACKS_20260619_visual_boom_v7_fresh_detector_fresh_detector_summary.csv"
)


def _path_key(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        path = Path(str(value)).expanduser()
        if path.exists():
            return str(path.resolve())
        return str(path)
    except Exception:
        return str(value)


def _track_public(track: Mapping[str, Any], drums_path: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "bpm": track.get("bpm"),
        "key": track.get("key"),
        "energy": track.get("energy"),
        "folder": track.get("folder"),
        "src": track.get("src"),
        "drums_path": str(drums_path) if drums_path else "",
    }


def _find_role_audio(folder: Path, role: str, track: Mapping[str, Any]) -> Optional[Path]:
    bpm = int(track["bpm"])
    key = str(track["key"]).upper()
    energy = track.get("energy")
    matches: List[Path] = []
    try:
        children = list(folder.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_file() or child.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        match = builder.STEM_RE.match(child.name)
        if not match:
            continue
        child_role, child_bpm, child_key, child_energy = match.groups()
        if child_role.lower() != role:
            continue
        if int(child_bpm) != bpm or child_key.upper() != key:
            continue
        if energy is not None and int(child_energy) != int(energy):
            continue
        matches.append(child)
    if not matches and role == "drums":
        matches = [
            child
            for child in children
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS and child.name.lower().startswith("drums_")
        ]
    return sorted(matches, key=lambda path: path.name.lower())[0] if matches else None


def _load_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_audio_path(row: Mapping[str, Any]) -> str:
    for key in (
        "filename",
        "drums_path",
        "DrumsPath",
        "audio_path",
        "AudioPath",
        "track",
        "Track",
    ):
        value = _path_key(row.get(key))
        if value:
            return value
    return ""


def _limited(rows: Iterable[Mapping[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return [dict(row) for row in list(rows)[: max(0, int(limit))]]


def _eligible_library(stems: str | Path, sample_limit: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stems_path = Path(stems).expanduser().resolve()
    all_tracks = builder.sort_by_bpm_key_energy(builder.collect_tracks(str(stems_path)))
    acapella_tracks = [track for track in all_tracks if row_is_acapella({"track": track})]
    non_acapella_tracks = [track for track in all_tracks if track not in acapella_tracks]

    eligible: List[Dict[str, Any]] = []
    missing_drums: List[Dict[str, Any]] = []
    near_empty_drums: List[Dict[str, Any]] = []
    for track in non_acapella_tracks:
        drums_path = _find_role_audio(Path(str(track["src"])).parent, "drums", track)
        if drums_path is None:
            missing_drums.append(_track_public(track))
            continue
        signal_stats = drums_stem_signal_stats(drums_path)
        if is_near_empty_drums_stem(drums_path, stats=signal_stats):
            near_empty_drums.append(
                {
                    **_track_public(track, drums_path),
                    "signal_stats": signal_stats,
                }
            )
            continue
        eligible.append({**_track_public(track, drums_path), "drums_path_key": _path_key(drums_path)})

    report: Dict[str, Any] = {
        "stems": str(stems_path),
        "input_track_count": len(all_tracks),
        "acapella_track_count": len(acapella_tracks),
        "non_acapella_track_count": len(non_acapella_tracks),
        "missing_drums_count": len(missing_drums),
        "near_empty_drums_count": len(near_empty_drums),
        "eligible_track_count": len(eligible),
        "missing_drums_samples": _limited(missing_drums, sample_limit),
        "near_empty_drums_samples": _limited(near_empty_drums, sample_limit),
    }
    return report, eligible


def compare_rows_to_eligible(
    rows: Iterable[Mapping[str, Any]],
    *,
    stems: str | Path = STEMS_ROOT_DIR,
    source: str = "",
    sample_limit: int = 25,
) -> Dict[str, Any]:
    report, eligible = _eligible_library(stems, sample_limit)
    input_rows = [dict(row) for row in rows]
    filtered_rows = [row for row in input_rows if not (row_has_excluded_path(row) or row_is_acapella(row))]
    row_paths = [_row_audio_path(row) for row in filtered_rows]
    path_counts = Counter(path for path in row_paths if path)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    empty_path_rows = [row for row, path in zip(filtered_rows, row_paths) if not path]
    nonexistent_paths = sorted(path for path in path_counts if not Path(path).exists())

    eligible_paths = {str(row["drums_path_key"]) for row in eligible}
    summary_paths = set(path_counts)
    missing_from_summary = sorted(eligible_paths - summary_paths)
    extra_in_summary = sorted(summary_paths - eligible_paths)

    report.update(
        {
            "summary": str(source),
            "summary_row_count": len(input_rows),
            "summary_non_excluded_row_count": len(filtered_rows),
            "summary_unique_audio_count": len(summary_paths),
            "duplicate_summary_path_count": len(duplicate_paths),
            "empty_summary_path_count": len(empty_path_rows),
            "nonexistent_summary_path_count": len(nonexistent_paths),
            "missing_from_summary_count": len(missing_from_summary),
            "extra_in_summary_count": len(extra_in_summary),
            "coverage_checked": True,
            "all_covered": not (
                duplicate_paths
                or empty_path_rows
                or nonexistent_paths
                or missing_from_summary
                or extra_in_summary
                or len(summary_paths) != len(eligible_paths)
            ),
            "missing_from_summary_samples": missing_from_summary[:sample_limit],
            "extra_in_summary_samples": extra_in_summary[:sample_limit],
            "duplicate_summary_path_samples": duplicate_paths[:sample_limit],
            "empty_summary_path_samples": _limited(empty_path_rows, sample_limit),
            "nonexistent_summary_path_samples": nonexistent_paths[:sample_limit],
        }
    )
    return report


def build_coverage_report(
    stems: str | Path = STEMS_ROOT_DIR,
    summary: str | Path | None = DEFAULT_SUMMARY,
    *,
    sample_limit: int = 25,
) -> Dict[str, Any]:
    if summary in (None, ""):
        report, _eligible = _eligible_library(stems, sample_limit)
        report["summary"] = ""
        report["all_covered"] = False
        report["coverage_checked"] = False
        return report

    summary_path = Path(str(summary)).expanduser().resolve()
    rows = _load_summary_rows(summary_path)
    return compare_rows_to_eligible(rows, stems=stems, source=str(summary_path), sample_limit=sample_limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether a visual-first production summary covers every eligible non-acapella drums-stem track."
    )
    parser.add_argument("--stems", default=str(STEMS_ROOT_DIR), help="STEMS root with /BPM/Key/Track folders.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Visual-first production summary CSV to audit.")
    parser.add_argument("--sample-limit", type=int, default=25, help="Number of mismatch samples to include in JSON output.")
    parser.add_argument("--require-complete", action="store_true", help="Exit non-zero unless the summary exactly covers the eligible library.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_coverage_report(args.stems, args.summary, sample_limit=int(args.sample_limit))
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if bool(args.require_complete) and not bool(report.get("all_covered")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
