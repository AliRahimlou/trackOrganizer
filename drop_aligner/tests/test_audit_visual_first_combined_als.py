from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import audit_visual_first_combined_als as audit


def _track(tmp_path: Path) -> dict:
    folder = tmp_path / "Artist - Track"
    folder.mkdir()
    src = folder / "CH1.als"
    src.write_text("", encoding="utf-8")
    return {
        "src": str(src),
        "bpm": 128,
        "key": "1A",
        "energy": 7,
        "folder": "Artist - Track",
    }


def _mini_combined_als(path: Path, refs: dict[int, Path], *, marker: float = 32.0) -> None:
    root = ET.Element("Ableton")
    live_set = ET.SubElement(root, "LiveSet")
    scenes = ET.SubElement(live_set, "Scenes")
    scene = ET.SubElement(scenes, "Scene", {"Id": "0"})
    ET.SubElement(scene, "Name", {"Value": "128_1A_7-Artist - Track"})
    tracks = ET.SubElement(live_set, "Tracks")
    for ch in (1, 2, 3):
        track = ET.SubElement(tracks, "AudioTrack")
        ET.SubElement(track, "Name", {"Value": f"CH{ch}"})
        device_chain = ET.SubElement(track, "DeviceChain")
        sequencer = ET.SubElement(device_chain, "MainSequencer")
        slot_list = ET.SubElement(sequencer, "ClipSlotList")
        slot = ET.SubElement(slot_list, "ClipSlot", {"Id": "0"})
        value = ET.SubElement(slot, "Value")
        clip = ET.SubElement(value, "AudioClip")
        ET.SubElement(clip, "Name", {"Value": refs[ch].stem})
        sample_ref = ET.SubElement(clip, "SampleRef")
        file_ref = ET.SubElement(sample_ref, "FileRef")
        ET.SubElement(file_ref, "Path", {"Value": str(refs[ch])})
        ET.SubElement(file_ref, "RelativePath", {"Value": refs[ch].name})
        markers = ET.SubElement(clip, "WarpMarkers")
        ET.SubElement(markers, "WarpMarker", {"Id": "0", "SecTime": "0", "BeatTime": "-64"})
        ET.SubElement(markers, "WarpMarker", {"Id": "1", "SecTime": str(marker), "BeatTime": "0"})
    path.write_bytes(gzip.compress(ET.tostring(root, encoding="utf-8", xml_declaration=True)))


def _report(path: Path, track: dict, drums: Path, *, marker: float) -> None:
    path.write_text(
        json.dumps(
            {
                "processed_rows": [
                    {
                        "track": track,
                        "marker": marker,
                        "drums_path": str(drums),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_combined_als_audit_passes_matching_triplet(tmp_path: Path) -> None:
    track = _track(tmp_path)
    folder = Path(track["src"]).parent
    drums = folder / "drums_128_1A_7-Artist - Track.flac"
    inst = folder / "inst_128_1A_7-Artist - Track.flac"
    vocals = folder / "vocals_128_1A_7-Artist - Track.flac"
    for path in (drums, inst, vocals):
        path.write_text("", encoding="utf-8")
    als = tmp_path / "combined.als"
    report = tmp_path / "report.json"
    _mini_combined_als(als, {1: drums, 2: inst, 3: vocals}, marker=32.0)
    _report(report, track, drums, marker=32.0)

    result = audit.audit_combined_als(als, report)

    assert result["passed"] is True
    assert result["complete_triplet_rows"] == 1
    assert result["anchor_match_rows"] == 1
    assert result["file_ref_match_rows"] == 1


def test_combined_als_audit_rejects_marker_mismatch(tmp_path: Path) -> None:
    track = _track(tmp_path)
    folder = Path(track["src"]).parent
    drums = folder / "drums_128_1A_7-Artist - Track.flac"
    inst = folder / "inst_128_1A_7-Artist - Track.flac"
    vocals = folder / "vocals_128_1A_7-Artist - Track.flac"
    for path in (drums, inst, vocals):
        path.write_text("", encoding="utf-8")
    als = tmp_path / "combined.als"
    report = tmp_path / "report.json"
    _mini_combined_als(als, {1: drums, 2: inst, 3: vocals}, marker=32.5)
    _report(report, track, drums, marker=32.0)

    result = audit.audit_combined_als(als, report)

    assert result["passed"] is False
    assert result["anchor_mismatch_count"] == 1
    assert result["all_rows_match_expected"] is False
