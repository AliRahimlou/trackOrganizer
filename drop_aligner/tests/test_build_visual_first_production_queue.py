from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import build_visual_first_production_queue as queue


def _clean_gui_mask_proof() -> dict:
    return {
        "passes": True,
        "marker_signal_present": True,
        "marker_relevant_mask": True,
    }


def _row_with_boom_proof(proof: dict) -> dict:
    return {
        "audit_status": "pass",
        "audit_flags": [],
        "selected_by": "visual_boom_grid_one_snap",
        "boom_proof": proof,
        "gui_mask_proof": _clean_gui_mask_proof(),
    }


def test_production_queue_accepts_fresh_front_edge_proof() -> None:
    row = _row_with_boom_proof(
        {
            "passes": True,
            "marker_sec": 32.0,
            "nearest": {
                "edge_time": 32.0,
                "offset_sec": 0.0,
                "abs_offset_sec": 0.0,
            },
            "reasons": [],
        }
    )

    assert queue.row_passes_production_gate(row) is True
    assert queue._reason(row) == "held by production gate"


def test_production_queue_rejects_stale_front_edge_proof() -> None:
    row = _row_with_boom_proof(
        {
            "passes": True,
            "marker_sec": 32.421,
            "nearest": {
                "edge_time": 32.0,
                "offset_sec": 0.421,
                "abs_offset_sec": 0.421,
            },
            "reasons": [],
        }
    )

    assert queue.row_passes_production_gate(row) is False
    assert "boom_proof=stale_front_edge" in queue._reason(row)


def _fresh_boom(marker: float) -> dict:
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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_gate_artifacts_add_reviewable_failures_to_full_review_only(tmp_path: Path) -> None:
    pass_candidates = tmp_path / "pass_candidates.json"
    pass_candidates.write_text("{}", encoding="utf-8")
    held_candidates = tmp_path / "held_candidates.json"
    held_candidates.write_text("{}", encoding="utf-8")
    missing_source_candidates = tmp_path / "must_not_be_created_in_source_tree.json"
    pass_row = {
        "status": "processed",
        "track": {"bpm": 120, "key": "1A", "folder": "Pass"},
        "drums_path": "/library/drums_120_1A_Pass.wav",
        "marker": 16.0,
        "output_als": "/library/pass.als",
        "candidates_json": str(pass_candidates),
        "audit_status": "pass",
        "audit_flags": [],
        "selected_by": "visual_boom_grid_one_snap",
        "boom_proof": _fresh_boom(16.0),
        "gui_mask_proof": _clean_gui_mask_proof(),
        "als_anchor": {
            "ok": True,
            "accepted": True,
            "drop_sec": 16.004,
            "visual_marker_sec": 16.0,
        },
        "verification": {"valid": True, "errors": []},
    }
    processed_hold = {
        "status": "processed",
        "track": {"bpm": 121, "key": "2A", "folder": "Processed hold"},
        "drums_path": "/library/drums_121_2A_Held.wav",
        "marker": 24.0,
        "output_als": "/library/held.als",
        "candidates_json": str(held_candidates),
        "audit_status": "review",
        "audit_flags": ["ambiguous_visual_drop_evidence"],
        "selected_by": "visual_boom_grid_one_snap",
        "boom_proof": {"passes": False, "marker_sec": 24.0, "reasons": ["weak_body"]},
        "verification": {"valid": True, "errors": []},
    }
    reviewable_failure = {
        "status": "hold",
        "track": {"bpm": 122, "key": "3A", "folder": "Review failure"},
        "drums_path": "/library/drums_122_3A_Review.wav",
        "marker": 30.0,
        "visual_marker": 30.0,
        "output_als": "/library/review.als",
        "candidates_json": str(missing_source_candidates),
        "audit_status": "pass",
        "audit_flags": [],
        "selected_by": "visual_candidate_selector",
        "boom_proof": _fresh_boom(30.0),
        "als_anchor": {
            "ok": True,
            "accepted": True,
            "drop_sec": 30.012,
            "visual_marker_sec": 30.0,
            "impact_sample": 1323529,
        },
        "error": "als_verification_failed",
        "result": {
            "marker": 30.0,
            "source": "visual_candidate_selector",
            "selected_candidate": {"timestamp": 30.0, "selected_by": "visual_candidate_selector"},
            "candidates": [{"timestamp": 30.0, "rank": 1}],
            "feature_map": {
                "beatgrid": {"bpm": 122.0, "bar_zero_sec": 0.0},
            },
            "visual_audit": {"status": "pass", "flag_codes": []},
        },
    }
    no_marker_failure = {
        "status": "error",
        "track": {"bpm": 123, "key": "4A", "folder": "No marker"},
        "drums_path": "/library/drums_123_4A_NoMarker.wav",
        "error": "visual_first_marker_missing_time",
    }
    report_path = tmp_path / "VISUAL_FIRST_FRESH_ALL_TRACKS_test_report.json"
    report_path.write_text(
        json.dumps(
            {
                "processed_rows": [pass_row, processed_hold],
                "failure_rows": [reviewable_failure, no_marker_failure],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "artifacts"

    summary = queue.build_gate_artifacts(report_path, out_dir, prefix="test")

    assert summary["processed_count"] == 2
    assert summary["detector_pass_count"] == 1
    assert summary["full_review_count"] == 4
    assert summary["needs_human_review_count"] == 3
    assert summary["reviewable_failure_count"] == 2
    assert summary["non_reviewable_failure_count"] == 0

    fresh_summary = _read_csv(summary["fresh_summary_csv"])
    assert len(fresh_summary) == 2
    assert next(row for row in fresh_summary if row["filename"].endswith("Pass.wav"))["detected_drop_time"] == "16.004000"
    pass_summary = _read_csv(summary["pass_summary_csv"])
    assert len(pass_summary) == 1
    assert pass_summary[0]["detected_drop_time"] == "16.004000"
    full_summary = _read_csv(summary["full_review_summary_csv"])
    assert len(full_summary) == 4
    failure_summary = next(row for row in full_summary if row["filename"].endswith("Review.wav"))
    assert failure_summary["detected_drop_time"] == "30.012000"
    assert failure_summary["microaligned_time"] == "30.012000"
    assert failure_summary["status"] == "partial"
    assert failure_summary["als_valid"] == ""
    assert failure_summary["confidence_tier"] == "LOW"
    assert failure_summary["confidence"] == "0.000000"
    assert failure_summary["als_validation_error"].startswith("review_only:no_verified_als")
    assert failure_summary["error"] == "als_verification_failed"
    no_marker_summary = next(row for row in full_summary if row["filename"].endswith("NoMarker.wav"))
    assert no_marker_summary["detected_drop_time"] == ""
    assert no_marker_summary["status"] == "partial"
    assert no_marker_summary["als_valid"] == ""
    assert no_marker_summary["output_als"].endswith("_DROP_ALIGNED.als")
    assert Path(no_marker_summary["output_als"]).is_relative_to(out_dir)
    no_marker_sidecar = json.loads(Path(no_marker_summary["candidates_json"]).read_text(encoding="utf-8"))
    assert no_marker_sidecar["review_only"] is True
    assert no_marker_sidecar["final_ai_pick"] is None
    assert no_marker_sidecar["selected_candidate"]["selected"] is False

    sidecar = Path(failure_summary["candidates_json"])
    assert sidecar.is_file()
    assert sidecar.is_relative_to(out_dir)
    assert not missing_source_candidates.exists()
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["source"] == "visual_first_full_review_sidecar"
    assert sidecar_payload["review_only"] is True
    assert sidecar_payload["final_ai_pick"] == 30.012
    assert sidecar_payload["drop_sec"] == 30.012
    assert sidecar_payload["visual_marker_sec"] == 30.0
    assert sidecar_payload["als_anchor"]["impact_sample"] == 1323529
    assert sidecar_payload["feature_map"]["beatgrid"]["bpm"] == 122.0

    full_queue = _read_csv(summary["full_library_review_queue_csv"])
    assert len(full_queue) == 4
    assert next(row for row in full_queue if row["filename"].endswith("Review.wav"))["marker_sec"] == "30.012000"
    assert next(row for row in full_queue if row["filename"].endswith("NoMarker.wav"))["marker_sec"] == ""
    assert len(_read_csv(summary["review_queue_csv"])) == 3

    pass_report = json.loads(Path(summary["detector_pass_report"]).read_text(encoding="utf-8"))
    assert len(pass_report["processed_rows"]) == 1
    assert pass_report["processed_rows"][0]["drums_path"].endswith("Pass.wav")
    assert pass_report["failure_rows"] == []


def test_review_failure_preserves_existing_candidate_json(tmp_path: Path) -> None:
    source_candidates = tmp_path / "source_candidates.json"
    source_candidates.write_text('{"source":"builder_hold"}', encoding="utf-8")
    row = {
        "status": "hold",
        "drums_path": "/library/drums_140_1A_Test.wav",
        "marker": 42.0,
        "candidates_json": str(source_candidates),
    }

    review_row = queue._reviewable_failure_row(row, candidate_dir=tmp_path / "sidecars")

    assert review_row is not None
    assert review_row["candidates_json"] == str(source_candidates)
    assert json.loads(source_candidates.read_text(encoding="utf-8")) == {"source": "builder_hold"}
    assert not (tmp_path / "sidecars").exists()


def test_latest_report_accepts_current_and_legacy_names(tmp_path: Path, monkeypatch) -> None:
    report_dir = tmp_path / "VisualFirstFresh"
    report_dir.mkdir()
    current = report_dir / "VISUAL_FIRST_FRESH_ALL_TRACKS_current_report.json"
    legacy = report_dir / "VISUAL_FIRST_FRESH_ALL_DRUMS_legacy_report.json"
    derived = report_dir / "VISUAL_FIRST_FRESH_ALL_TRACKS_newer_detector_pass_only_report.json"
    for path in (current, legacy, derived):
        path.write_text("{}", encoding="utf-8")
    os.utime(current, (100.0, 100.0))
    os.utime(legacy, (200.0, 200.0))
    os.utime(derived, (300.0, 300.0))
    monkeypatch.setattr(queue, "GENERATED_SET_DIR", tmp_path)

    assert queue._latest_report() == legacy

    legacy.unlink()
    assert queue._latest_report() == current
