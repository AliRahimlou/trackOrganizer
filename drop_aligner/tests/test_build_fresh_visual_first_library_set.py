from __future__ import annotations

import argparse
import gzip
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import build_fresh_visual_first_library_set as fresh_build


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


def _fresh_boom_proof(marker: float = 32.0) -> dict:
    return {
        "passes": True,
        "marker_sec": marker,
        "nearest": {
            "edge_time": marker,
            "offset_sec": 0.0,
            "abs_offset_sec": 0.0,
        },
        "reasons": [],
    }


def _fresh_gui_mask_proof() -> dict:
    return {
        "passes": True,
        "reasons": [],
        "placeable_count": 2,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": True,
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
        role = {1: "drums", 2: "inst", 3: "vocals"}[ch]
        ET.SubElement(clip, "Name", {"Value": refs[ch].stem})
        sample_ref = ET.SubElement(clip, "SampleRef")
        file_ref = ET.SubElement(sample_ref, "FileRef")
        ET.SubElement(file_ref, "Path", {"Value": str(refs[ch])})
        ET.SubElement(file_ref, "RelativePath", {"Value": refs[ch].name})
        markers = ET.SubElement(clip, "WarpMarkers")
        ET.SubElement(markers, "WarpMarker", {"Id": "0", "SecTime": "0", "BeatTime": "-64"})
        ET.SubElement(markers, "WarpMarker", {"Id": "1", "SecTime": str(marker), "BeatTime": "0"})
    path.write_bytes(gzip.compress(ET.tostring(root, encoding="utf-8", xml_declaration=True)))


def test_production_gate_blocks_stale_and_memory_sources() -> None:
    for selected_by in (
        "historical_review_memory",
        "historical_human_marker",
        "saved_closest_to_review_pick",
        "saved_detector_seed",
        "visual_drop_v2",
        "web_accept_blue_marker",
        "web_save_placed_marker",
    ):
        reasons = fresh_build._production_gate_reasons(
            audit_status="pass",
            audit_flags=[],
            selected_by=selected_by,
            boom_proof=_fresh_boom_proof(32.0),
            gui_mask_proof=_fresh_gui_mask_proof(),
        )
        assert f"unsafe_source={selected_by}" in reasons


def test_production_gate_allows_normalized_visual_contract_source() -> None:
    reasons = fresh_build._production_gate_reasons(
        audit_status="pass",
        audit_flags=[],
        selected_by="visual_gui_boom_front_edge_contract",
        boom_proof=_fresh_boom_proof(32.0),
        gui_mask_proof=_fresh_gui_mask_proof(),
    )

    assert reasons == []


def test_production_gate_rejects_stale_gui_proof_without_immediate_body() -> None:
    stale_gui_proof = _fresh_gui_mask_proof()
    stale_gui_proof.pop("marker_immediate_body_present")

    reasons = fresh_build._production_gate_reasons(
        audit_status="pass",
        audit_flags=[],
        selected_by="visual_gui_boom_front_edge_contract",
        boom_proof=_fresh_boom_proof(32.0),
        gui_mask_proof=stale_gui_proof,
    )

    assert "gui_mask=strict_contract_hold:stale_missing_marker_immediate_body" in reasons


def test_production_gate_allows_exact_sparse_front_edge_proof() -> None:
    gui_proof = _fresh_gui_mask_proof()
    gui_proof.update(
        {
            "marker_immediate_body_present": False,
            "marker_body_mask": False,
            "nearest_placeable_offset_sec": 0.0,
            "accepted_by_exact_sparse_front_edge_proof": True,
        }
    )

    reasons = fresh_build._production_gate_reasons(
        audit_status="pass",
        audit_flags=[],
        selected_by="visual_final_contract_gui_mask_nearest_repair",
        boom_proof=_fresh_boom_proof(32.0),
        gui_mask_proof=gui_proof,
    )

    assert reasons == []


def test_visual_first_marker_timeout_raises_bounded_error() -> None:
    with pytest.raises(fresh_build._TrackTimeoutError, match="visual_first_marker_timeout:0.010s"):
        with fresh_build._time_limit(0.010, "visual_first_marker"):
            time.sleep(0.100)


def test_combined_set_verification_rejects_wrong_channel_file_ref(tmp_path: Path) -> None:
    track = _track(tmp_path)
    folder = Path(track["src"]).parent
    drums = folder / "drums_128_1A_7-Artist - Track.flac"
    inst = folder / "inst_128_1A_7-Artist - Track.flac"
    vocals = folder / "vocals_128_1A_7-Artist - Track.flac"
    wrong_inst = folder / "inst_128_1A_7-Wrong - Track.flac"
    for path in (drums, inst, vocals, wrong_inst):
        path.write_text("", encoding="utf-8")
    combined = tmp_path / "combined.als"
    _mini_combined_als(combined, {1: drums, 2: wrong_inst, 3: vocals})

    result = fresh_build._verify_combined_set(
        combined,
        [
            {
                "track": track,
                "marker": 32.0,
                "drums_path": str(drums),
            }
        ],
    )

    assert result["anchor_match_rows"] == 1
    assert result["file_ref_mismatch_count"] == 1
    assert result["all_file_refs_match_expected"] is False
    assert result["all_rows_match_expected"] is False
    assert result["file_ref_mismatch_samples"][0]["reason"] == "file_ref_mismatch"


def test_existing_fresh_row_requires_current_boom_proof(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    output = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_DROP_ALIGNED.als")
    candidates = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_drop_candidates.json")
    output.write_text("not real als but exists", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "final_ai_pick": 32.0,
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": {"passes": False, "reasons": ["profile_score below threshold"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fresh_build, "_verify_anchor", lambda *args, **kwargs: {"valid": True})

    row = fresh_build._load_existing_fresh_row(output, candidates, track, drums)

    assert row is None


def test_existing_fresh_row_reuses_only_production_gated_payload(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    output = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_DROP_ALIGNED.als")
    candidates = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_drop_candidates.json")
    output.write_text("not real als but exists", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "final_ai_pick": 32.0,
                "drop_sec": 32.0,
                "visual_marker_sec": 31.995,
                "als_anchor": {
                    "ok": True,
                    "accepted": True,
                    "drop_sec": 32.0,
                    "sample_rate": 44100,
                },
                "selected_by": "visual_boom_grid_one_snap",
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": _fresh_boom_proof(32.0),
                "gui_mask_proof": {
                    "passes": True,
                    "reasons": [],
                    "placeable_count": 2,
                    "marker_relevant_mask": True,
                    "marker_signal_present": True,
                    "marker_immediate_body_present": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fresh_build,
        "_verify_anchor",
        lambda *args, **kwargs: {"valid": True, "drop_marker_time": 32.0, "drop_marker_count": 1, "errors": []},
    )

    row = fresh_build._load_existing_fresh_row(output, candidates, track, drums)

    assert row is not None
    assert row["marker"] == 32.0
    assert row["visual_marker"] == 31.995
    assert row["als_anchor"]["accepted"] is True
    assert row["selected_by"] == "visual_boom_grid_one_snap"
    assert row["boom_proof"]["passes"] is True
    assert row["gui_mask_proof"]["passes"] is True


def test_existing_fresh_row_rejects_stale_persisted_boom_front_edge(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    output = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_DROP_ALIGNED.als")
    candidates = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_drop_candidates.json")
    output.write_text("not real als but exists", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "final_ai_pick": 32.421,
                "drop_sec": 32.421,
                "selected_by": "visual_boom_grid_phase_calibration",
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": {
                    "passes": True,
                    "marker_sec": 32.421,
                    "nearest": {
                        "edge_time": 32.0,
                        "offset_sec": 0.421,
                        "abs_offset_sec": 0.421,
                    },
                    "reasons": [],
                },
                "gui_mask_proof": {
                    "passes": True,
                    "reasons": [],
                    "placeable_count": 2,
                    "marker_signal_present": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fresh_build, "_verify_anchor", lambda *args, **kwargs: {"valid": True})

    row = fresh_build._load_existing_fresh_row(output, candidates, track, drums)

    assert row is None


def test_existing_fresh_row_requires_current_gui_mask_proof(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    output = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_DROP_ALIGNED.als")
    candidates = drums.with_name(f"{drums.stem}_VISUAL_FIRST_FRESH_test_drop_candidates.json")
    output.write_text("not real als but exists", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "final_ai_pick": 32.0,
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": _fresh_boom_proof(32.0),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fresh_build, "_verify_anchor", lambda *args, **kwargs: {"valid": True})

    row = fresh_build._load_existing_fresh_row(output, candidates, track, drums)

    assert row is None


def test_process_track_holds_unproven_marker_before_writing_als(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("", encoding="utf-8")
    calls = []

    monkeypatch.setattr(fresh_build, "_find_role_audio", lambda folder, role, track_arg: drums if role == "drums" else None)
    monkeypatch.setattr(
        fresh_build,
        "drums_stem_signal_stats",
        lambda *_args, **_kwargs: {"peak_abs": 0.5, "window_rms_p99": 0.25},
    )
    monkeypatch.setattr(fresh_build, "is_near_empty_drums_stem", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        fresh_build,
        "visual_first_marker",
        lambda *args, **kwargs: {
            "ok": True,
            "marker": 32.0,
            "selected_candidate": {
                "timestamp": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
            },
            "visual_audit": {"status": "pass", "flag_codes": []},
            "boom_proof": {"passes": False, "reasons": ["no_boom_body_section_candidate"]},
        },
    )
    monkeypatch.setattr(fresh_build, "_write_visual_candidate_json", lambda *args, **kwargs: calls.append("json"))
    monkeypatch.setattr(fresh_build, "modify_als", lambda *args, **kwargs: calls.append("als"))
    monkeypatch.setattr(
        fresh_build,
        "build_visual_first_als_anchor",
        lambda *args, **kwargs: {
            "ok": True,
            "accepted": True,
            "reason": "visual_grid_one_with_bounded_drum_attack",
            "drop_sec": 32.01,
            "impact_sec": 32.01,
        },
    )

    row = fresh_build._process_track(
        {
            "track": track,
            "template": str(tmp_path / "template.als"),
            "run_stamp": "test",
            "sample_rate": 16000,
            "use_cache": True,
            "strict_stems": False,
            "force": True,
            "dry_run": False,
        }
    )

    assert row["status"] == "hold"
    assert "boom_proof=hold" in row["error"]
    assert row["selected_by"] == "visual_boom_grid_one_snap"
    assert row["marker"] == 32.0
    assert row["als_anchor"]["accepted"] is False
    assert row["als_anchor"]["reason"] == "visual_production_gate_hold"
    assert row["als_anchor"]["stage_b_attack_accepted"] is True
    assert calls == ["json"]


def test_process_track_holds_non_placeable_gui_mask_before_writing_als(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("", encoding="utf-8")
    calls = []

    monkeypatch.setattr(fresh_build, "_find_role_audio", lambda folder, role, track_arg: drums if role == "drums" else None)
    monkeypatch.setattr(
        fresh_build,
        "drums_stem_signal_stats",
        lambda *_args, **_kwargs: {"peak_abs": 0.5, "window_rms_p99": 0.25},
    )
    monkeypatch.setattr(fresh_build, "is_near_empty_drums_stem", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        fresh_build,
        "visual_first_marker",
        lambda *args, **kwargs: {
            "ok": True,
            "marker": 32.0,
            "selected_candidate": {
                "timestamp": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
            },
            "visual_audit": {"status": "pass", "flag_codes": []},
            "boom_proof": _fresh_boom_proof(32.0),
        },
    )
    monkeypatch.setattr(
        fresh_build,
        "visual_gui_mask_proof",
        lambda *args, **kwargs: {
            "passes": False,
            "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
            "placeable_count": 0,
        },
    )
    monkeypatch.setattr(fresh_build, "_write_visual_candidate_json", lambda *args, **kwargs: calls.append("json"))
    monkeypatch.setattr(fresh_build, "modify_als", lambda *args, **kwargs: calls.append("als"))

    row = fresh_build._process_track(
        {
            "track": track,
            "template": str(tmp_path / "template.als"),
            "run_stamp": "test",
            "sample_rate": 16000,
            "use_cache": True,
            "strict_stems": False,
            "force": True,
            "dry_run": False,
            "waveform_cache_dir": str(tmp_path / "cache"),
        }
    )

    assert row["status"] == "hold"
    assert "gui_mask=hold" in row["error"]
    assert row["gui_mask_proof"]["passes"] is False
    assert calls == ["json"]


def test_validated_human_override_rewrites_processed_row(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("", encoding="utf-8")
    output = tmp_path / "track.als"
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "selected_candidate": {"timestamp": 33.0, "selected_by": "visual_gui_chunk"},
                "top_10_candidates": [{"timestamp": 33.0}],
                "boom_candidates": [{"timestamp": 32.0}],
                "feature_map": {"beatgrid": {"bpm": 128}},
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "drop_corrections.jsonl"
    log.write_text(
        json.dumps(
            {
                "track": str(drums),
                "user_pick": 32.0,
                "reviewed_from": "web_manual_marker",
                "timestamp": "2026-06-20T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = {
        "status": "processed",
        "track": track,
        "drums_path": str(drums),
        "output_als": str(output),
        "candidates_json": str(candidates),
        "marker": 33.0,
        "selected_by": "visual_gui_chunk",
        "audit_status": "pass",
        "audit_flags": [],
    }
    calls = []

    monkeypatch.setattr(
        fresh_build,
        "review_marker_gate",
        lambda *args, **kwargs: {
            "passes": True,
            "reasons": [],
            "boom_proof": fresh_build._fresh_boom_proof(32.0) if hasattr(fresh_build, "_fresh_boom_proof") else _fresh_boom_proof(32.0),
            "gui_proof": {
                "passes": True,
                "reasons": [],
                "placeable_count": 2,
                "marker_relevant_mask": True,
                "marker_signal_present": True,
                "marker_immediate_body_present": True,
            },
        },
    )
    monkeypatch.setattr(fresh_build, "modify_als", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        fresh_build,
        "_verify_anchor",
        lambda *args, **kwargs: {"valid": True, "drop_marker_time": 32.0, "drop_marker_count": 1, "errors": []},
    )

    result = fresh_build._apply_validated_human_overrides(
        [row],
        correction_logs=[log],
        template=tmp_path / "template.als",
        strict_stems=True,
        waveform_cache_dir=tmp_path / "cache",
    )

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert result["applied_count"] == 1
    assert result["failure_count"] == 0
    assert row["marker"] == 32.0
    assert row["selected_by"] == fresh_build.VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
    assert row["human_review_override"]["reviewed_from"] == "web_manual_marker"
    assert payload["final_ai_pick"] == 32.0
    assert payload["selected_by"] == fresh_build.VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
    assert calls and calls[0]["drop_sec"] == 32.0


def test_unproven_human_override_is_stale_and_does_not_rewrite_row(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    candidates.write_text("{}", encoding="utf-8")
    log = tmp_path / "drop_corrections.jsonl"
    log.write_text(
        json.dumps({"track": str(drums), "user_pick": 32.0, "reviewed_from": "web_manual_marker"}) + "\n",
        encoding="utf-8",
    )
    row = {
        "track": track,
        "drums_path": str(drums),
        "output_als": str(tmp_path / "track.als"),
        "candidates_json": str(candidates),
        "marker": 33.0,
        "selected_by": "visual_gui_chunk",
    }
    calls = []

    monkeypatch.setattr(
        fresh_build,
        "review_marker_gate",
        lambda *args, **kwargs: {"passes": False, "reasons": ["boom_proof_hold:no_boom_body_section_candidate"]},
    )
    monkeypatch.setattr(fresh_build, "modify_als", lambda **kwargs: calls.append(kwargs))

    result = fresh_build._apply_validated_human_overrides(
        [row],
        correction_logs=[log],
        template=tmp_path / "template.als",
        strict_stems=True,
        waveform_cache_dir=tmp_path / "cache",
    )

    assert result["applied_count"] == 0
    assert result["stale_manual_mismatch_count"] == 1
    assert row["marker"] == 33.0
    assert calls == []


def test_write_visual_candidate_json_persists_gui_mask_proof(tmp_path: Path) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    candidates = tmp_path / "candidate.json"
    output = tmp_path / "out.als"

    fresh_build._write_visual_candidate_json(
        candidates,
        audio_path=drums,
        marker=32.0,
        result={
            "selected_candidate": {"selected_by": "visual_boom_grid_one_snap"},
            "visual_audit": {"status": "pass", "flag_codes": []},
            "boom_proof": _fresh_boom_proof(32.0),
            "gui_mask_proof": {
                "passes": True,
                "reasons": [],
                "placeable_count": 3,
                "marker_relevant_mask": True,
                "marker_signal_present": True,
                "marker_immediate_body_present": True,
            },
        },
        track=track,
        output_als=output,
        als_anchor={
            "ok": True,
            "accepted": True,
            "drop_sec": 32.012,
            "impact_sec": 32.012,
            "visual_marker_sec": 32.0,
            "sample_rate": 44100,
        },
    )

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert payload["gui_mask_proof"]["passes"] is True
    assert payload["selected_candidate"]["gui_mask_proof"]["passes"] is True
    assert payload["visual_marker_sec"] == 32.0
    assert payload["als_anchor"]["impact_sec"] == 32.012
    assert payload["marker_contract"]["boom_body_crest"] == "post_attack_diagnostic_only"


def test_process_track_writes_bounded_anchor_impact_not_visual_stage_a(tmp_path, monkeypatch) -> None:
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("audio placeholder", encoding="utf-8")
    anchor_calls = []
    als_calls = []

    monkeypatch.setattr(fresh_build, "_find_role_audio", lambda folder, role, track_arg: drums if role == "drums" else None)
    monkeypatch.setattr(
        fresh_build,
        "drums_stem_signal_stats",
        lambda *_args, **_kwargs: {"peak_abs": 0.5, "window_rms_p99": 0.25},
    )
    monkeypatch.setattr(fresh_build, "is_near_empty_drums_stem", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        fresh_build,
        "visual_first_marker",
        lambda *args, **kwargs: {
            "ok": True,
            "marker": 32.0,
            "selected_candidate": {
                "timestamp": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
            },
            "visual_audit": {"status": "pass", "flag_codes": []},
            "boom_proof": _fresh_boom_proof(32.0),
            "gui_mask_proof": _fresh_gui_mask_proof(),
            "feature_map": {"beatgrid": {"bpm": 128.0, "bar_zero_sec": 0.0}},
        },
    )
    monkeypatch.setattr(fresh_build, "visual_gui_mask_proof", lambda *args, **kwargs: _fresh_gui_mask_proof())

    def fake_anchor(audio_path, bpm, **kwargs):
        anchor_calls.append((audio_path, bpm, kwargs))
        return {
            "ok": True,
            "accepted": True,
            "drop_sec": 32.012,
            "impact_sec": 32.012,
            "impact_sample": 1411729,
            "visual_marker_sec": 32.0,
            "grid_downbeat_sec": 32.0,
            "sample_rate": 44100,
            "attack": {"zero_crossing_time_sec": 32.0121, "alternate_candidates": []},
        }

    monkeypatch.setattr(fresh_build, "build_visual_first_als_anchor", fake_anchor)
    monkeypatch.setattr(fresh_build, "modify_als", lambda **kwargs: als_calls.append(kwargs))
    monkeypatch.setattr(
        fresh_build,
        "_verify_anchor",
        lambda *args, **kwargs: {
            "valid": True,
            "drop_marker_time": 32.012,
            "drop_marker_count": 1,
            "errors": [],
        },
    )

    row = fresh_build._process_track(
        {
            "track": track,
            "template": str(tmp_path / "template.als"),
            "run_stamp": "impact_test",
            "sample_rate": 44100,
            "use_cache": True,
            "strict_stems": False,
            "force": True,
            "dry_run": False,
            "waveform_cache_dir": str(tmp_path / "cache"),
        }
    )

    payload = json.loads(Path(row["candidates_json"]).read_text(encoding="utf-8"))
    assert row["status"] == "processed"
    assert row["visual_marker"] == 32.0
    assert row["marker"] == 32.012
    assert row["als_anchor"]["impact_sample"] == 1411729
    assert anchor_calls[0][2]["visual_result"]["marker"] == 32.0
    assert als_calls[0]["drop_sec"] == 32.012
    assert payload["final_ai_pick"] == 32.012
    assert payload["visual_marker_sec"] == 32.0
    assert payload["visual_selected_candidate"]["timestamp"] == 32.0
    assert payload["selected_candidate"]["timestamp"] == 32.012
    assert payload["als_anchor"]["impact_sample"] == 1411729


def test_run_holds_reused_report_with_incomplete_library_coverage(tmp_path, monkeypatch) -> None:
    stems = tmp_path / "STEMS"
    stems.mkdir()
    template = tmp_path / "template.als"
    base_als = tmp_path / "base.als"
    template.write_text("", encoding="utf-8")
    base_als.write_text("", encoding="utf-8")
    output = tmp_path / "combined.als"
    report = tmp_path / "combined_report.json"
    reuse_report = tmp_path / "reuse_report.json"
    reuse_report.write_text(json.dumps({"processed_rows": []}), encoding="utf-8")
    track = _track(tmp_path)
    drums = Path(track["src"]).with_name("drums_128_1A_7-Artist - Track.flac")
    drums.write_text("", encoding="utf-8")
    build_calls = []

    monkeypatch.setattr(fresh_build.builder, "collect_tracks", lambda stems_arg: [track])
    monkeypatch.setattr(fresh_build.builder, "sort_by_bpm_key_energy", lambda tracks: list(tracks))
    monkeypatch.setattr(fresh_build, "_find_role_audio", lambda folder, role, track_arg: drums if role == "drums" else None)
    monkeypatch.setattr(
        fresh_build,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": False,
            "eligible_track_count": 1,
            "missing_from_summary_count": 1,
            "extra_in_summary_count": 0,
        },
    )
    monkeypatch.setattr(fresh_build, "_build_combined_set", lambda *args, **kwargs: build_calls.append(True) or {})

    code = fresh_build.run(
        argparse.Namespace(
            stems=str(stems),
            template=str(template),
            base_als=str(base_als),
            out_dir=str(tmp_path),
            out=str(output),
            report=str(report),
            run_stamp="test",
            workers=1,
            sample_rate=16000,
            use_cache=True,
            strict_stems=True,
            force=False,
            dry_run=False,
            limit=0,
            allow_partial=False,
            allow_unsafe_audit=False,
            reuse_report=str(reuse_report),
        )
    )

    saved_report = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert build_calls == []
    assert saved_report["coverage_checked"] is True
    assert saved_report["coverage_all_covered"] is False
    assert saved_report["coverage_missing_from_summary_count"] == 1
