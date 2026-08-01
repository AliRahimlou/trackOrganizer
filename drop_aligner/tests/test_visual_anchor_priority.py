from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET

import pytest

import trackOrganizerAndAlsGen as organizer
from verify_als import verify_als


def _write_template(template_path, clip_names) -> None:
    root = ET.Element("Ableton")
    for name in clip_names:
        clip = ET.SubElement(root, "AudioClip")
        ET.SubElement(clip, "Name", {"Value": name})
    ET.SubElement(root, "IsWarped", {"Value": "false"})
    with gzip.open(template_path, "wb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def _run_modify_als(monkeypatch, tmp_path, *, db_lookup):
    target = tmp_path / "track"
    target.mkdir()
    template = tmp_path / "120.als"
    drums = tmp_path / "drums_120_test.flac"
    drums.touch()
    _write_template(template, ["drums_120_test.flac"])

    monkeypatch.setattr(organizer, "FLAC_FOLDER", str(tmp_path))
    monkeypatch.setattr(organizer, "SKIP_EXISTING", False)
    monkeypatch.setattr(organizer, "ALS_BACKUP_BEFORE_WRITE", False)
    monkeypatch.setattr(organizer, "USE_VISUAL_FIRST_ALS_ANCHOR", True)
    monkeypatch.setattr(organizer, "USE_DROP_FUSION_AUTOMATION", False)
    monkeypatch.setattr(organizer, "USE_REKORDBOX_MIK_CUE_STAMP_TEST", False)
    monkeypatch.setattr(organizer, "DROP_ANCHOR_OFFSET_SEC", 0.0)
    monkeypatch.setattr(organizer, "DROP_ANCHOR_OVERRIDES_SEC", {})
    monkeypatch.setattr(organizer, "_shared_stem_duration_seconds", lambda _tracks: (10.0, {"drums": 10.0}))
    monkeypatch.setattr(
        organizer,
        "_detect_alignment_with_first_downbeat",
        lambda *_args, **_kwargs: ({"drums": 2.0}, 2.0, 0.9, "test"),
    )
    monkeypatch.setattr(
        organizer,
        "_lookup_rekordbox_first_drop_prior",
        lambda *_args, **_kwargs: (None, 0.0, ""),
    )
    monkeypatch.setattr(organizer, "get_duration_in_beats", lambda *_args, **_kwargs: 20.0)
    monkeypatch.setattr(
        organizer,
        "_detect_visual_first_als_anchor",
        lambda *_args, **_kwargs: {
            "ok": True,
            "accepted": True,
            "reason": "visual_grid_one_with_bounded_drum_attack",
            "drop_sec": 2.025,
            "confidence": 0.95,
            "grid_downbeat_sec": 2.0,
            "grid_downbeat_sample": 16000,
            "impact_sample": 16200,
            "phase_translation_samples": 200,
        },
    )
    monkeypatch.setattr(organizer, "_lookup_drop_from_db", db_lookup)
    monkeypatch.setattr(organizer, "_manual_drop_from_project_als", lambda *_args, **_kwargs: None)

    organizer.modify_als_file(
        str(template),
        str(target),
        {"drums": drums.name, "inst": None, "vocals": None},
        120,
        force=True,
    )
    return target


def _anchor_sec_times(als_path) -> list:
    with gzip.open(als_path, "rb") as handle:
        generated = ET.fromstring(handle.read())
    return [
        float(point.get("SecTime"))
        for point in generated.iter("WarpMarker")
        if abs(float(point.get("BeatTime"))) <= 1e-9 and float(point.get("SecTime")) > 0.0
    ]


def test_drop_db_does_not_override_an_approved_visual_anchor(monkeypatch, tmp_path) -> None:
    target = _run_modify_als(
        monkeypatch,
        tmp_path,
        db_lookup=lambda *_args, **_kwargs: (5.5, 0.9, "slug:some-other-edit (obs=4)"),
    )

    output = target / "CH1.als"
    assert output.exists()
    anchors = _anchor_sec_times(output)
    assert anchors == [pytest.approx(2.025)]

    audit = json.loads((target / organizer.VISUAL_FIRST_ALS_AUDIT_BASENAME).read_text(encoding="utf-8"))
    assert audit["applied_to_als"] is True
    assert audit["final_drop_sec"] == pytest.approx(2.025)


def test_drop_db_env_escape_hatch_restores_the_old_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(organizer, "DROP_DB_OVERRIDES_VISUAL_ANCHOR", True)
    target = _run_modify_als(
        monkeypatch,
        tmp_path,
        db_lookup=lambda *_args, **_kwargs: (5.5, 0.9, "exact:drums_120_test (obs=4)"),
    )

    anchors = _anchor_sec_times(target / "CH1.als")
    assert anchors == [pytest.approx(5.5)]


def test_verify_als_requires_every_clip_to_share_the_drop_anchor(tmp_path) -> None:
    aligned = tmp_path / "aligned.als"
    root = ET.Element("Ableton")
    for name, anchor in (("drums.flac", 2.025), ("inst.flac", 2.025), ("vocals.flac", 2.025)):
        clip = ET.SubElement(root, "AudioClip")
        ET.SubElement(clip, "Name", {"Value": name})
        markers = ET.SubElement(clip, "WarpMarkers")
        ET.SubElement(markers, "WarpMarker", {"Id": "1", "SecTime": "0.0", "BeatTime": "-4.05"})
        ET.SubElement(markers, "WarpMarker", {"Id": "2", "SecTime": str(anchor), "BeatTime": "0.0"})
    with gzip.open(aligned, "wb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    report = verify_als(str(aligned), allow_multiple=True)
    by_name = {check["name"]: check["passed"] for check in report["checks"]}
    assert by_name["all_clips_have_drop_anchor"] is True
    assert by_name["all_clips_share_drop_anchor"] is True
    assert report["clip_anchor_times"] == [pytest.approx(2.025)] * 3


def test_verify_als_flags_a_drifted_stem_anchor(tmp_path) -> None:
    aligned = tmp_path / "drifted.als"
    root = ET.Element("Ableton")
    for name, anchor in (("drums.flac", 2.025), ("inst.flac", 2.025), ("vocals.flac", 2.500)):
        clip = ET.SubElement(root, "AudioClip")
        ET.SubElement(clip, "Name", {"Value": name})
        markers = ET.SubElement(clip, "WarpMarkers")
        ET.SubElement(markers, "WarpMarker", {"Id": "1", "SecTime": "0.0", "BeatTime": "-4.05"})
        ET.SubElement(markers, "WarpMarker", {"Id": "2", "SecTime": str(anchor), "BeatTime": "0.0"})
    with gzip.open(aligned, "wb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    report = verify_als(str(aligned), allow_multiple=True)
    by_name = {check["name"]: check["passed"] for check in report["checks"]}
    assert by_name["all_clips_share_drop_anchor"] is False
    assert report["valid"] is False
