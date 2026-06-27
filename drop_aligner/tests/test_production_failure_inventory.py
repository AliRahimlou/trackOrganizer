from __future__ import annotations

from pathlib import Path

from drop_aligner.production_failure_inventory import (
    KNOWN_VISUAL_REGRESSIONS,
    build_failure_inventory,
    build_regression_seed_payload,
    classify_failure,
    write_jsonl,
)


def test_classify_failure_maps_blank_off_one_and_late_body() -> None:
    labels = classify_failure(
        {
            "reasons": [
                "gui_mask=hold:blank_marker_no_visible_signal",
                "boom_proof=hold:one_distance_ms 250.0 above 90.0",
                "persisted_boom_proof=stale_front_edge:marker 0.421 after boom front edge",
            ]
        }
    )

    assert "blank_waveform_marker" in labels
    assert "off_grid_or_not_on_one" in labels
    assert "late_inside_body_or_tail" in labels


def test_failure_inventory_keeps_actionable_marker_context() -> None:
    inventory = build_failure_inventory(
        [
            {
                "index": 7,
                "track": "/tmp/drums_140_4A_6-Test.flac",
                "marker": 55.118,
                "selected_by": "visual_boom_grid_phase_calibration",
                "suggested_marker_sec": 54.857,
                "boom_nearest_edge": 54.857,
                "boom_edge_offset_sec": 0.261,
                "reasons": ["persisted_boom_proof=stale_front_edge:marker after boom front edge"],
                "candidates_json": "/tmp/candidates.json",
            }
        ],
        source_report="/tmp/report.json",
    )

    assert inventory[0]["index"] == 7
    assert inventory[0]["taxonomy"] == "late_inside_body_or_tail"
    assert inventory[0]["marker_sec"] == 55.118
    assert inventory[0]["suggested_marker_sec"] == 54.857
    assert inventory[0]["source_report"] == "/tmp/report.json"


def test_regression_seed_payload_separates_known_ground_truth_from_proposals() -> None:
    payload = build_regression_seed_payload(
        [
            {
                "index": 7,
                "drums_path": "/tmp/drums_140_4A_6-Test.flac",
                "marker_sec": 55.118,
                "suggested_marker_sec": 54.857,
                "taxonomy": "late_inside_body_or_tail",
                "reasons": ["marker was late"],
                "source_report": "/tmp/report.json",
            }
        ]
    )

    assert payload["known_regressions"] == KNOWN_VISUAL_REGRESSIONS
    assert payload["known_regressions"][0]["needs_manual_confirmation"] is False
    assert payload["proposed_regression_seeds"][0]["needs_manual_confirmation"] is True
    assert payload["proposed_regression_seeds"][0]["expected_marker_sec"] == 54.857


def test_write_jsonl_writes_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "inventory.jsonl"

    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert path.read_text(encoding="utf-8").splitlines() == ['{"a": 1}', '{"b": 2}']
