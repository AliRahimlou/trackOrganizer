from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_visual_first_current_detector as audit


def _report_row(audio: Path, marker: float = 24.0) -> dict:
    return {
        "drums_path": str(audio),
        "marker": marker,
        "selected_by": "visual_final_contract_gui_mask_nearest_repair",
    }


def _detector_result(marker: float, *, ok: bool = True, boom: bool = True, gui: bool = True, one: bool = True) -> dict:
    selected = {
        "timestamp": marker,
        "selected_by": "visual_final_contract_gui_mask_nearest_repair",
        "bpm_clock": {
            "on_one": one,
            "one_distance_ms": 0.0 if one else 95.0,
            "source": "test_clock",
        },
        "boom_proof": {"passes": boom, "reasons": [] if boom else ["test_boom_hold"]},
        "gui_mask_proof": {
            "passes": gui,
            "reasons": [] if gui else ["test_gui_hold"],
            "marker_signal_present": bool(gui),
            "marker_immediate_body_present": bool(gui),
        },
    }
    return {
        "ok": ok,
        "drop_sec": marker,
        "selected_by": selected["selected_by"],
        "selected_candidate": selected,
        "boom_proof": selected["boom_proof"],
        "gui_mask_proof": selected["gui_mask_proof"],
        "visual_audit": {"status": "pass", "flag_codes": []},
    }


def _staccato_detector_result(marker: float) -> dict:
    payload = _detector_result(marker)
    proof = {
        "passes": True,
        "accepted_by_staccato_front_body_proof": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": False,
        "reasons": [
            "marker_not_on_gui_boom_relevant_mask",
            "marker_not_on_gui_boom_front_edge_mask",
            "gui_tile_has_no_placeable_boom_front_edge",
        ],
    }
    payload["gui_mask_proof"] = proof
    payload["selected_candidate"]["gui_mask_proof"] = proof
    return payload


def _args(tmp_path: Path, report: Path) -> argparse.Namespace:
    return argparse.Namespace(
        report=str(report),
        out_dir=str(tmp_path / "audit"),
        indices="",
        offset=0,
        limit=0,
        workers=1,
        sample_rate=44100,
        use_cache=True,
        resume=True,
        force=False,
        progress_every=0,
        max_runtime_sec=0.0,
        per_row_timeout_sec=0.0,
        strict_tolerance_sec=0.002,
        near_tolerance_sec=0.050,
        require_all_near_match=True,
        require_all_strict_match=False,
    )


def test_parse_indices_supports_ranges_and_dedupes() -> None:
    assert audit._parse_indices("3,1-2,2,6-4") == [3, 1, 2, 4, 5, 6]


def test_classify_delta_separates_strict_near_and_mismatch() -> None:
    strict = audit._classify_delta(10.0, 10.001, strict_tolerance_sec=0.002, near_tolerance_sec=0.050)
    near = audit._classify_delta(10.0, 10.025, strict_tolerance_sec=0.002, near_tolerance_sec=0.050)
    miss = audit._classify_delta(10.0, 10.125, strict_tolerance_sec=0.002, near_tolerance_sec=0.050)

    assert strict["match_status"] == "strict_match"
    assert near["match_status"] == "near_match"
    assert miss["match_status"] == "mismatch"


def test_audit_one_requires_detector_proofs_not_just_marker_match(tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_bytes(b"")
    task = {
        "index": 1,
        "row": _report_row(audio, marker=24.0),
        "sample_rate": 44100,
        "use_cache": True,
        "strict_tolerance_sec": 0.002,
        "near_tolerance_sec": 0.050,
    }

    row = audit._audit_one(task, detector_fn=lambda *_args, **_kwargs: _detector_result(24.0, gui=False))

    assert row["strict_match"] is True
    assert row["status"] == "proof_hold"
    assert row["current_boom_proof_pass"] is True
    assert row["current_gui_mask_proof_pass"] is False


def test_audit_one_treats_staccato_front_body_proof_as_ready(tmp_path: Path) -> None:
    audio = tmp_path / "drums_80_1A_7-Staccato.flac"
    audio.write_bytes(b"")
    task = {
        "index": 1,
        "row": _report_row(audio, marker=33.0),
        "sample_rate": 44100,
        "use_cache": True,
        "strict_tolerance_sec": 0.002,
        "near_tolerance_sec": 0.050,
    }

    row = audit._audit_one(task, detector_fn=lambda *_args, **_kwargs: _staccato_detector_result(33.0))

    assert row["strict_match"] is True
    assert row["status"] == "strict_match"
    assert row["current_gui_mask_proof_clean"] is True
    assert row["current_gui_staccato_contract_relief"] is True
    assert row["current_gui_strict_contract_issue"] == ""


def test_audit_one_allows_zero_second_opening_drop_marker(tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Opening Drop.flac"
    audio.write_bytes(b"")
    task = {
        "index": 1,
        "row": _report_row(audio, marker=0.0),
        "sample_rate": 44100,
        "use_cache": True,
        "strict_tolerance_sec": 0.002,
        "near_tolerance_sec": 0.050,
    }

    row = audit._audit_one(task, detector_fn=lambda *_args, **_kwargs: _detector_result(0.0))

    assert row["current_marker"] == 0.0
    assert row["status"] == "strict_match"


def test_timeout_result_records_resumeable_audit_error(tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Slow.flac"
    task = {
        "index": 9,
        "row": _report_row(audio, marker=18.0),
        "sample_rate": 44100,
        "strict_tolerance_sec": 0.002,
        "near_tolerance_sec": 0.050,
    }

    row = audit._timeout_result(task, timeout_sec=30.0, elapsed_sec=30.25)

    assert row["audit_key"].startswith("9:")
    assert row["status"] == "timeout"
    assert row["detector_ok"] is False
    assert row["detector_error"] == "detector_timeout:30.000s"
    assert row["match_status"] == "missing_current_marker"


def test_audit_resume_skips_completed_rows(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_bytes(b"")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"processed_rows": [_report_row(audio, marker=12.0)]}), encoding="utf-8")
    calls = {"count": 0}

    def fake_detector(path: str, *, sample_rate: int, use_cache: bool) -> dict:
        calls["count"] += 1
        return _detector_result(12.0)

    monkeypatch.setattr(audit, "_run_detector", fake_detector)

    first = audit.audit(_args(tmp_path, report))
    second = audit.audit(_args(tmp_path, report))

    assert calls["count"] == 1
    assert first["all_selected_near_match"] is True
    assert second["all_selected_near_match"] is True
    assert second["completed_count"] == 1
