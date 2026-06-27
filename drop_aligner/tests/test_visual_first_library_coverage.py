from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_visual_first_library_coverage import build_coverage_report


def _write_audio(path: Path, *, amp: float = 0.25, sr: int = 16000) -> None:
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (amp * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


def _track_folder(root: Path, *, bpm: int, key: str, folder: str, energy: int = 7) -> Path:
    path = root / str(bpm) / key / folder
    path.mkdir(parents=True)
    (path / "CH1.als").write_text("", encoding="utf-8")
    _write_audio(path / f"drums_{bpm}_{key}_{energy}-{folder}.flac")
    return path


def _write_summary(path: Path, filenames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "detected_drop_time", "selected_by"])
        writer.writeheader()
        for filename in filenames:
            writer.writerow(
                {
                    "filename": filename,
                    "detected_drop_time": "32.0",
                    "selected_by": "visual_boom_grid_one_snap",
                }
            )


def test_coverage_report_accepts_exact_non_acapella_summary(tmp_path: Path) -> None:
    stems = tmp_path / "STEMS"
    track = _track_folder(stems, bpm=128, key="1A", folder="Artist - Boom")
    _track_folder(stems, bpm=128, key="2A", folder="Artist - Acapella Tool")
    drums = track / "drums_128_1A_7-Artist - Boom.flac"
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [str(drums)])

    report = build_coverage_report(stems, summary)

    assert report["input_track_count"] == 2
    assert report["acapella_track_count"] == 1
    assert report["eligible_track_count"] == 1
    assert report["summary_non_excluded_row_count"] == 1
    assert report["all_covered"] is True


def test_coverage_report_accepts_production_csv_drums_path_column(tmp_path: Path) -> None:
    stems = tmp_path / "STEMS"
    track = _track_folder(stems, bpm=128, key="1A", folder="Artist - Boom")
    drums = track / "drums_128_1A_7-Artist - Boom.flac"
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "#",
                "BPM",
                "Key",
                "Energy",
                "TrackFolder",
                "MarkerSec",
                "SelectedBy",
                "FreshAlignedAls",
                "DrumsPath",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "#": "1",
                "BPM": "128",
                "Key": "1A",
                "Energy": "7",
                "TrackFolder": "Artist - Boom",
                "MarkerSec": "32.0",
                "SelectedBy": "visual_boom_grid_one_snap",
                "FreshAlignedAls": str(track / "aligned.als"),
                "DrumsPath": str(drums),
            }
        )

    report = build_coverage_report(stems, summary)

    assert report["eligible_track_count"] == 1
    assert report["summary_unique_audio_count"] == 1
    assert report["empty_summary_path_count"] == 0
    assert report["missing_from_summary_count"] == 0
    assert report["all_covered"] is True


def test_coverage_report_flags_missing_extra_duplicate_and_nonexistent_rows(tmp_path: Path) -> None:
    stems = tmp_path / "STEMS"
    covered = _track_folder(stems, bpm=128, key="1A", folder="Artist - Covered")
    missing = _track_folder(stems, bpm=129, key="1A", folder="Artist - Missing")
    covered_drums = covered / "drums_128_1A_7-Artist - Covered.flac"
    missing_drums = missing / "drums_129_1A_7-Artist - Missing.flac"
    extra = tmp_path / "outside.flac"
    extra.write_text("", encoding="utf-8")
    nonexistent = tmp_path / "does-not-exist.flac"
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [str(covered_drums), str(covered_drums), str(extra), str(nonexistent)])

    report = build_coverage_report(stems, summary)

    assert report["eligible_track_count"] == 2
    assert report["all_covered"] is False
    assert report["missing_from_summary_count"] == 1
    assert str(missing_drums.resolve()) in report["missing_from_summary_samples"]
    assert report["duplicate_summary_path_count"] == 1
    assert report["extra_in_summary_count"] == 2
    assert report["nonexistent_summary_path_count"] == 1


def test_coverage_report_excludes_near_empty_drums_stems(tmp_path: Path) -> None:
    stems = tmp_path / "STEMS"
    audible = _track_folder(stems, bpm=128, key="1A", folder="Artist - Audible")
    near_empty = _track_folder(stems, bpm=128, key="2A", folder="Artist - Near Empty")
    near_empty_drums = near_empty / "drums_128_2A_7-Artist - Near Empty.flac"
    _write_audio(near_empty_drums, amp=0.001)
    audible_drums = audible / "drums_128_1A_7-Artist - Audible.flac"
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [str(audible_drums)])

    report = build_coverage_report(stems, summary)

    assert report["near_empty_drums_count"] == 1
    assert report["eligible_track_count"] == 1
    assert report["all_covered"] is True
