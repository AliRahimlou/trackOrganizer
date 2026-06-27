from __future__ import annotations

import csv
import json
from pathlib import Path

from apply_visual_first_remaining import REVIEWED_FROM, _update_candidate_payload
from sync_visual_first_summary import sync_summary


def _write_summary(path: Path, candidates_json: Path, marker: float = 32.0) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "detected_drop_time",
                "selected_by",
                "candidates_json",
                "visual_detector_prep_time",
                "visual_detector_prep_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "/music/drums_128_1A_7-Artist - Track.flac",
                "detected_drop_time": str(marker),
                "selected_by": "old_marker",
                "candidates_json": str(candidates_json),
            }
        )


def _read_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.DictReader(fh))


def _fresh_boom_proof(marker: float = 64.0) -> dict:
    return {
        "passes": True,
        "reasons": [],
        "marker_sec": marker,
        "nearest": {
            "edge_time": marker,
            "offset_sec": 0.0,
            "abs_offset_sec": 0.0,
        },
    }


def test_sync_holds_unproven_visual_detector_payload(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    candidates = tmp_path / "candidates.json"
    _write_summary(summary, candidates)
    candidates.write_text(
        json.dumps(
            {
                "reviewed_from": REVIEWED_FROM,
                "final_ai_pick": 64.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {"selected_by": "visual_boom_grid_one_snap", "timestamp": 64.0},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": {"passes": False, "reasons": ["profile_score below threshold"]},
            }
        ),
        encoding="utf-8",
    )

    result = sync_summary(summary, dry_run=False)
    row = _read_row(summary)

    assert result["counts"]["held_production_gate"] == 1
    assert row["detected_drop_time"] == "32.0"
    assert row["selected_by"] == "old_marker"


def test_sync_holds_visual_detector_payload_missing_gui_mask_proof(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    candidates = tmp_path / "candidates.json"
    _write_summary(summary, candidates)
    candidates.write_text(
        json.dumps(
            {
                "reviewed_from": REVIEWED_FROM,
                "final_ai_pick": 64.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {"selected_by": "visual_boom_grid_one_snap", "timestamp": 64.0},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": _fresh_boom_proof(64.0),
            }
        ),
        encoding="utf-8",
    )

    result = sync_summary(summary, dry_run=False)
    row = _read_row(summary)

    assert result["counts"]["held_production_gate"] == 1
    assert result["counts"]["held:gui_mask=hold:missing"] == 1
    assert row["detected_drop_time"] == "32.0"
    assert row["selected_by"] == "old_marker"


def test_sync_allows_proven_visual_detector_payload(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    candidates = tmp_path / "candidates.json"
    _write_summary(summary, candidates)
    candidates.write_text(
        json.dumps(
            {
                "reviewed_from": REVIEWED_FROM,
                "final_ai_pick": 64.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {"selected_by": "visual_boom_grid_one_snap", "timestamp": 64.0},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": _fresh_boom_proof(64.0),
                "gui_mask_proof": {"passes": True, "reasons": [], "placeable_count": 3},
            }
        ),
        encoding="utf-8",
    )

    result = sync_summary(summary, dry_run=False)
    row = _read_row(summary)

    assert result["counts"]["synced"] == 1
    assert result["counts"].get("held_production_gate", 0) == 0
    assert row["detected_drop_time"] == "64"
    assert row["selected_by"] == "visual_boom_grid_one_snap"
    assert row["visual_detector_prep_time"] == "64"


def test_sync_holds_visual_detector_payload_with_stale_boom_front_edge(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    candidates = tmp_path / "candidates.json"
    _write_summary(summary, candidates)
    candidates.write_text(
        json.dumps(
            {
                "reviewed_from": REVIEWED_FROM,
                "final_ai_pick": 64.421,
                "selected_by": "visual_boom_grid_phase_calibration",
                "selected_candidate": {"selected_by": "visual_boom_grid_phase_calibration", "timestamp": 64.421},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": {
                    "passes": True,
                    "reasons": [],
                    "marker_sec": 64.421,
                    "nearest": {
                        "edge_time": 64.0,
                        "offset_sec": 0.421,
                        "abs_offset_sec": 0.421,
                    },
                },
                "gui_mask_proof": {"passes": True, "reasons": [], "placeable_count": 3},
            }
        ),
        encoding="utf-8",
    )

    result = sync_summary(summary, dry_run=False)
    row = _read_row(summary)

    assert result["counts"]["held_production_gate"] == 1
    assert result["counts"]["held:boom_proof=stale_front_edge:boom_edge_offset 0.421s above 0.080s"] == 1
    assert row["detected_drop_time"] == "32.0"
    assert row["selected_by"] == "old_marker"


def test_visual_first_prep_payload_persists_regeneration_guard_proofs(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"top_10_candidates": []}), encoding="utf-8")
    item = {
        "candidates_json": str(candidates),
        "top_10_candidates": [{"timestamp": 64.0, "selected_by": "visual_boom_grid_one_snap"}],
    }

    _update_candidate_payload(
        item,
        64.0,
        {"timestamp": 64.0, "selected_by": "visual_boom_grid_one_snap"},
        "visual_boom_grid_one_snap",
        regen={
            "boom_proof": _fresh_boom_proof(64.0),
            "gui_mask_proof": {"passes": True, "placeable_count": 4},
        },
    )

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert payload["reviewed_from"] == REVIEWED_FROM
    assert payload["boom_proof"] == _fresh_boom_proof(64.0)
    assert payload["gui_mask_proof"] == {"passes": True, "placeable_count": 4}
    assert payload["selected_candidate"]["boom_proof"]["passes"] is True
    assert payload["selected_candidate"]["gui_mask_proof"]["passes"] is True


def test_sync_allows_human_review_payload_without_detector_proof(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    candidates = tmp_path / "candidates.json"
    _write_summary(summary, candidates)
    candidates.write_text(
        json.dumps(
            {
                "reviewed_from": "web_manual_marker",
                "user_pick": 72.0,
                "selected_by": "user_candidate_pick",
                "selected_candidate": {"selected_by": "user_candidate_pick", "timestamp": 72.0},
            }
        ),
        encoding="utf-8",
    )

    result = sync_summary(summary, dry_run=False)
    row = _read_row(summary)

    assert result["counts"]["synced"] == 1
    assert result["counts"].get("held_production_gate", 0) == 0
    assert row["detected_drop_time"] == "72"
    assert row["selected_by"] == "user_candidate_pick"
