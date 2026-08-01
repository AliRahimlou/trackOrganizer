from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET

import pytest

import trackOrganizerAndAlsGen as organizer


def test_retarget_anchor_map_preserves_inter_stem_offsets() -> None:
    result = organizer._retarget_anchor_map(
        {"drums": 10.0, "inst": 10.018, "vocals": 9.992},
        {"drums": "drums.flac", "inst": "inst.flac", "vocals": "vocals.flac"},
        20.0,
    )

    assert result["drums"] == pytest.approx(20.0)
    assert result["inst"] == pytest.approx(20.018)
    assert result["vocals"] == pytest.approx(19.992)


def test_visual_anchor_wrapper_passes_production_settings(monkeypatch, tmp_path) -> None:
    drums_path = tmp_path / "drums_120_test.wav"
    drums_path.touch()
    observed = {}

    def fake_builder(path, bpm, *, sample_rate, use_cache):
        observed.update(
            path=path,
            bpm=bpm,
            sample_rate=sample_rate,
            use_cache=use_cache,
        )
        return {"ok": True, "accepted": True, "drop_sec": 2.025}

    monkeypatch.setattr(organizer, "USE_VISUAL_FIRST_ALS_ANCHOR", True)
    monkeypatch.setattr(organizer, "VISUAL_FIRST_ALS_SAMPLE_RATE", 44100)
    monkeypatch.setattr(organizer, "VISUAL_FIRST_ALS_USE_CACHE", True)
    monkeypatch.setattr(organizer, "build_visual_first_als_anchor", fake_builder)

    result = organizer._detect_visual_first_als_anchor(str(drums_path), 120)

    assert result["accepted"] is True
    assert observed == {
        "path": str(drums_path),
        "bpm": 120.0,
        "sample_rate": 44100,
        "use_cache": True,
    }


def test_als_warp_marker_uses_the_bounded_impact_as_beat_zero() -> None:
    root = ET.Element("Ableton")
    clip = ET.SubElement(root, "AudioClip")
    ET.SubElement(clip, "Name", {"Value": "drums_120_test.flac"})
    ET.SubElement(root, "IsWarped", {"Value": "false"})

    count = organizer._stamp_drop_anchor_warpmarkers(
        root,
        120,
        {"drums": 2.025},
        {"drums": 10.0},
        10.0,
    )

    points = clip.find("WarpMarkers").findall("WarpMarker")
    impact = next(point for point in points if point.get("SecTime") == "2.025000")
    assert count == 3
    assert float(impact.get("BeatTime")) == pytest.approx(0.0, abs=1e-9)
    assert root.find("IsWarped").get("Value") == "true"


def test_visual_contract_failure_does_not_create_an_als(monkeypatch, tmp_path) -> None:
    target = tmp_path / "track"
    target.mkdir()
    template = tmp_path / "120.als"
    template.write_bytes(b"template must remain uncopied")
    drums = tmp_path / "drums_120_test.wav"
    drums.touch()

    monkeypatch.setattr(organizer, "FLAC_FOLDER", str(tmp_path))
    monkeypatch.setattr(organizer, "SKIP_EXISTING", False)
    monkeypatch.setattr(organizer, "ALS_BACKUP_BEFORE_WRITE", False)
    monkeypatch.setattr(organizer, "USE_VISUAL_FIRST_ALS_ANCHOR", True)
    monkeypatch.setattr(organizer, "USE_REKORDBOX_MIK_CUE_STAMP_TEST", False)
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
            "ok": False,
            "accepted": False,
            "reason": "visual_marker_not_on_independent_one",
        },
    )
    monkeypatch.setattr(organizer, "_lookup_drop_from_db", lambda *_args, **_kwargs: (None, 0.0, ""))
    monkeypatch.setattr(organizer, "_manual_drop_from_project_als", lambda *_args, **_kwargs: None)

    organizer.modify_als_file(
        str(template),
        str(target),
        {"drums": drums.name, "inst": None, "vocals": None},
        120,
        force=True,
    )

    assert not (target / "CH1.als").exists()
    audit = json.loads((target / organizer.VISUAL_FIRST_ALS_AUDIT_BASENAME).read_text(encoding="utf-8"))
    assert audit["reason"] == "visual_marker_not_on_independent_one"
    assert audit["applied_to_als"] is False


def test_visual_attack_sample_reaches_generated_als_unchanged(monkeypatch, tmp_path) -> None:
    target = tmp_path / "track"
    target.mkdir()
    template = tmp_path / "120.als"
    drums = tmp_path / "drums_120_test.flac"
    drums.touch()

    root = ET.Element("Ableton")
    clip = ET.SubElement(root, "AudioClip")
    ET.SubElement(clip, "Name", {"Value": "drums_120_test.flac"})
    ET.SubElement(root, "IsWarped", {"Value": "false"})
    with gzip.open(template, "wb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))

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
    monkeypatch.setattr(organizer, "_lookup_drop_from_db", lambda *_args, **_kwargs: (None, 0.0, ""))
    monkeypatch.setattr(organizer, "_manual_drop_from_project_als", lambda *_args, **_kwargs: None)

    organizer.modify_als_file(
        str(template),
        str(target),
        {"drums": drums.name, "inst": None, "vocals": None},
        120,
        force=True,
    )

    output = target / "CH1.als"
    with gzip.open(output, "rb") as handle:
        generated = ET.fromstring(handle.read())
    impact = next(
        point
        for point in generated.iter("WarpMarker")
        if point.get("SecTime") == "2.025000"
    )
    assert float(impact.get("BeatTime")) == pytest.approx(0.0, abs=1e-9)
    audit = json.loads((target / organizer.VISUAL_FIRST_ALS_AUDIT_BASENAME).read_text(encoding="utf-8"))
    assert audit["applied_to_als"] is True
    assert audit["impact_sample"] == 16200
    assert audit["final_drop_sec"] == pytest.approx(2.025)
