from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import validate_visual_first_production as validator


def _args(
    tmp_path: Path,
    report: Path,
    *,
    require_all_pass: bool = True,
    skip_combined_als: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        report=str(report),
        profile="",
        out_dir=str(tmp_path),
        limit=0,
        workers=1,
        sample_rate=16000,
        progress_every=0,
        rerun_detector=False,
        skip_rerun_report_match=False,
        rerun_report_tolerance_sec=0.002,
        require_all_pass=require_all_pass,
        include_excluded=False,
        stems=str(tmp_path / "STEMS"),
        skip_coverage=False,
        skip_combined_als=skip_combined_als,
        skip_human_review_audit=True,
        human_correction_log=None,
        human_review_audit_dir="",
        manual_review_tolerance_sec=0.050,
        blue_review_tolerance_sec=0.050,
        skip_gui_mask=True,
        skip_persisted_proof=False,
        waveform_cache_dir="",
        coverage_sample_limit=25,
    )


def _report(tmp_path: Path) -> Path:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"processed_rows": [{"filename": str(audio), "selected_by": "visual_boom_grid_one_snap"}]}),
        encoding="utf-8",
    )
    return report


def _touch_json(path: Path, mtime: int) -> Path:
    path.write_text(json.dumps({"processed_rows": []}), encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _fresh_boom_proof(marker: float = 32.0) -> dict:
    return {
        "passes": True,
        "reasons": [],
        "marker_sec": marker,
        "nearest": {
            "edge_time": marker,
            "offset_sec": 0.0,
            "abs_offset_sec": 0.0,
        },
        "nearest_profile": {
            "metrics": {
                "one_distance_ms": 0.0,
            },
        },
    }


def _fresh_gui_proof(marker: float = 32.0) -> dict:
    return {
        "passes": True,
        "reasons": [],
        "marker": marker,
        "placeable_count": 2,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": True,
    }


def _visual_body_components() -> dict:
    return {
        "post8_height": 0.72,
        "post4_height": 0.69,
        "post_bass8": 0.58,
        "post_drum8": 0.94,
        "post_drum_cont8": 0.91,
    }


def test_latest_report_prefers_current_all_tracks_and_ignores_derived(tmp_path: Path) -> None:
    legacy_newer = _touch_json(tmp_path / "VISUAL_FIRST_FRESH_ALL_DRUMS_99999999_report.json", 400)
    current_valid = _touch_json(tmp_path / "VISUAL_FIRST_FRESH_ALL_TRACKS_20260619_report.json", 100)
    detector_pass_only = _touch_json(
        tmp_path / "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_detector_pass_only_report.json",
        500,
    )
    production_validation = _touch_json(
        tmp_path / "VISUAL_FIRST_FRESH_ALL_TRACKS_20260621_production_validation_report.json",
        600,
    )

    assert validator._latest_report(tmp_path) == current_valid
    assert legacy_newer.exists()
    assert detector_pass_only.exists()
    assert production_validation.exists()


def test_latest_report_falls_back_to_legacy_all_drums(tmp_path: Path) -> None:
    legacy_old = _touch_json(tmp_path / "VISUAL_FIRST_FRESH_ALL_DRUMS_20260618_report.json", 100)
    legacy_new = _touch_json(tmp_path / "VISUAL_FIRST_FRESH_ALL_DRUMS_20260619_report.json", 200)

    assert validator._latest_report(tmp_path) == legacy_new
    assert legacy_old.exists()


def test_validator_requires_full_library_coverage_for_all_pass(tmp_path: Path, monkeypatch) -> None:
    report = _report(tmp_path)
    monkeypatch.setattr(validator, "_validate_one", lambda job: {"passes": True, "reasons": [], "selected_by": "visual_boom_grid_one_snap"})
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": False,
            "eligible_track_count": 2,
            "missing_from_summary_count": 1,
            "extra_in_summary_count": 0,
        },
    )

    result = validator.validate(_args(tmp_path, report))

    assert result["pass_count"] == 1
    assert result["hold_count"] == 0
    assert result["coverage_checked"] is True
    assert result["coverage_all_covered"] is False
    assert result["all_passed"] is False
    assert result["exit_code"] == 1


def test_validator_all_pass_requires_rows_and_coverage(tmp_path: Path, monkeypatch) -> None:
    report = _report(tmp_path)
    monkeypatch.setattr(validator, "_validate_one", lambda job: {"passes": True, "reasons": [], "selected_by": "visual_boom_grid_one_snap"})
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": True,
            "eligible_track_count": 1,
            "missing_from_summary_count": 0,
            "extra_in_summary_count": 0,
        },
    )

    result = validator.validate(_args(tmp_path, report))

    assert result["coverage_all_covered"] is True
    assert result["all_passed"] is True
    assert result["exit_code"] == 0


def test_validator_all_pass_requires_human_review_audit_when_enabled(tmp_path: Path, monkeypatch) -> None:
    report = _report(tmp_path)
    correction_log = tmp_path / "drop_corrections.jsonl"
    correction_log.write_text("", encoding="utf-8")
    monkeypatch.setattr(validator, "_validate_one", lambda job: {"passes": True, "reasons": [], "selected_by": "visual_boom_grid_one_snap"})
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": True,
            "eligible_track_count": 1,
            "missing_from_summary_count": 0,
            "extra_in_summary_count": 0,
        },
    )
    monkeypatch.setattr(
        validator,
        "audit_human_review_memory",
        lambda *args, **kwargs: {
            "checked": True,
            "passed": False,
            "review_marker_count": 1,
            "matched_review_rows": 1,
            "manual_review_rows": 1,
            "blue_approval_rows": 0,
            "hard_mismatch_count": 1,
            "validated_hard_mismatch_count": 1,
            "stale_manual_mismatch_count": 0,
            "advisory_mismatch_count": 0,
        },
    )
    args = _args(tmp_path, report)
    args.skip_human_review_audit = False
    args.human_correction_log = [str(correction_log)]

    result = validator.validate(args)

    assert result["human_review_audit_checked"] is True
    assert result["human_review_audit_passed"] is False
    assert result["human_review_validated_hard_mismatch_count"] == 1
    assert result["all_passed"] is False
    assert result["exit_code"] == 1


def test_validator_summary_reports_rerun_detector_counts(tmp_path: Path, monkeypatch) -> None:
    report = _report(tmp_path)
    monkeypatch.setattr(
        validator,
        "_validate_one",
        lambda job: {
            "passes": True,
            "reasons": [],
            "selected_by": "visual_boom_grid_one_snap",
            "marker": 10.0,
            "report_marker": 10.0,
            "rerun_report_delta_sec": 0.0,
        },
    )
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": True,
            "eligible_track_count": 1,
            "missing_from_summary_count": 0,
            "extra_in_summary_count": 0,
        },
    )
    args = _args(tmp_path, report)
    args.rerun_detector = True

    result = validator.validate(args)

    assert result["rerun_detector"] is True
    assert result["rerun_report_match_required"] is True
    assert result["rerun_checked_count"] == 1
    assert result["rerun_match_count"] == 1
    assert result["rerun_mismatch_count"] == 0


def test_validator_writes_failure_inventory_and_regression_seeds(tmp_path: Path, monkeypatch) -> None:
    report = _report(tmp_path)
    monkeypatch.setattr(
        validator,
        "_validate_one",
        lambda job: {
            "index": job["index"],
            "passes": False,
            "reasons": [
                "gui_mask=hold:blank_marker_no_visible_signal",
                "boom_proof=hold:one_distance_ms 250.0 above 90.0",
            ],
            "track": str(tmp_path / "STEMS" / "128" / "1A" / "Test" / "drums_128_1A_7-Test.flac"),
            "marker": 32.25,
            "suggested_marker_sec": 32.0,
            "selected_by": "visual_boom_grid_phase_calibration",
        },
    )
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": True,
            "eligible_track_count": 1,
            "missing_from_summary_count": 0,
            "extra_in_summary_count": 0,
        },
    )

    result = validator.validate(_args(tmp_path, report, require_all_pass=False))

    inventory_path = tmp_path / f"{report.stem}_production_failure_inventory.jsonl"
    seed_path = tmp_path / f"{report.stem}_production_regression_seeds.json"
    assert result["failure_taxonomy_counts"]["blank_waveform_marker"] == 1
    assert inventory_path.exists()
    assert seed_path.exists()
    assert "blank_waveform_marker" in inventory_path.read_text(encoding="utf-8")
    assert "proposed_regression_seeds" in seed_path.read_text(encoding="utf-8")


def test_validator_cli_reruns_detector_by_default() -> None:
    args = validator.parse_args([])

    assert args.rerun_detector is True


def test_validator_cli_can_explicitly_skip_detector_rerun() -> None:
    args = validator.parse_args(["--no-rerun-detector"])

    assert args.rerun_detector is False


def test_validator_cli_runs_human_review_audit_by_default() -> None:
    args = validator.parse_args([])

    assert args.skip_human_review_audit is False


def test_validator_requires_combined_als_anchor_mapping_for_all_pass(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    als = tmp_path / "combined.als"
    als.write_text("not real in this unit test", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "output_als": str(als),
                "processed_rows": [
                    {
                        "filename": str(audio),
                        "marker": 32.0,
                        "selected_by": "visual_boom_grid_one_snap",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "_validate_one", lambda job: {"passes": True, "reasons": [], "selected_by": "visual_boom_grid_one_snap"})
    monkeypatch.setattr(
        validator,
        "compare_rows_to_eligible",
        lambda *args, **kwargs: {
            "coverage_checked": True,
            "all_covered": True,
            "eligible_track_count": 1,
            "missing_from_summary_count": 0,
            "extra_in_summary_count": 0,
        },
    )
    monkeypatch.setattr(
        validator,
        "_verify_combined_set",
        lambda *args, **kwargs: {
            "valid_xml": True,
            "row_count": 1,
            "expected_row_count": 1,
            "all_rows_match_expected": False,
            "anchor_mismatch_count": 1,
            "anchor_mismatch_samples": [{"row": 0, "reason": "anchor_marker_mismatch"}],
        },
    )

    result = validator.validate(_args(tmp_path, report, skip_combined_als=False))

    assert result["pass_count"] == 1
    assert result["combined_als_checked"] is True
    assert result["combined_als_all_rows_match_expected"] is False
    assert result["combined_als_anchor_mismatch_count"] == 1
    assert result["all_passed"] is False
    assert result["exit_code"] == 1


def test_gui_mask_tile_proof_accepts_marker_on_placeable_front_edge() -> None:
    proof = validator._gui_mask_proof_for_tile(
        {
            "start_sec": 10.0,
            "end_sec": 16.0,
            "boom_relevant_mask": [False, False, True, False],
            "boom_placeable_mask": [False, False, True, False],
            "boom_placeable_count": 1,
            "boom_bin_span_sec": 1.5,
        },
        13.1,
    )

    assert proof["passes"] is True
    assert proof["mask_index"] == 2


def test_gui_mask_tile_proof_rejects_flat_or_non_placeable_marker() -> None:
    proof = validator._gui_mask_proof_for_tile(
        {
            "start_sec": 10.0,
            "end_sec": 16.0,
            "boom_relevant_mask": [False, False, False, False],
            "boom_placeable_mask": [False, False, False, False],
            "boom_placeable_count": 0,
        },
        13.1,
    )

    assert proof["passes"] is False
    assert "marker_not_on_gui_boom_front_edge_mask" in proof["reasons"]
    assert "gui_tile_has_no_placeable_boom_front_edge" in proof["reasons"]


def test_gui_mask_tile_proof_rejects_missing_relevance_mask() -> None:
    proof = validator._gui_mask_proof_for_tile(
        {
            "start_sec": 10.0,
            "end_sec": 16.0,
            "boom_placeable_mask": [False, False, True, False],
            "boom_placeable_count": 1,
        },
        13.1,
    )

    assert proof["passes"] is False
    assert "missing_gui_boom_relevant_mask" in proof["reasons"]


def test_validate_one_requires_gui_mask_when_enabled(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})
    monkeypatch.setattr(
        validator,
        "_gui_mask_proof",
        lambda *args, **kwargs: {
            "passes": False,
            "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
            "placeable_count": 0,
        },
    )

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": False,
            "waveform_cache_dir": str(tmp_path / ".waveform_cache"),
        }
    )

    assert result["passes"] is False
    assert result["gui_mask"]["checked"] is True
    assert any(reason.startswith("gui_mask=hold:") for reason in result["reasons"])


def test_validate_one_requires_persisted_proofs_by_default(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {"timestamp": 32.0, "selected_by": "visual_boom_grid_one_snap"},
                "visual_audit": {"status": "pass", "flag_codes": []},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert "persisted_boom_proof=hold:missing" in result["reasons"]
    assert "persisted_gui_mask_proof=hold:missing" in result["reasons"]


def test_validate_one_accepts_persisted_proofs_from_candidate_json(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    proof = _fresh_boom_proof(32.0)
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {
                    "timestamp": 32.0,
                    "selected_by": "visual_boom_grid_one_snap",
                    "visual_components": {
                        "post8_height": 0.72,
                        "post4_height": 0.69,
                        "post_bass8": 0.58,
                        "post_drum8": 0.94,
                        "post_drum_cont8": 0.91,
                    },
                },
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": proof,
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
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is True
    assert result["persisted_boom_proof"]["passes"] is True
    assert result["persisted_gui_mask_proof"]["passes"] is True
    assert result["actual_visual_body_contract_pass"] is True


def test_validate_one_accepts_current_rerun_strict_proofs_when_raw_recompute_misses_relief(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import drop_aligner.visual_first as visual_first

    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    marker = 32.0
    selected = {
        "timestamp": marker,
        "selected_by": "visual_final_contract_gui_front_edge_repair",
        "visual_components": _visual_body_components(),
    }
    detector_payload = {
        "ok": True,
        "audio_path": str(audio),
        "drop_sec": marker,
        "final_ai_pick": marker,
        "selected_by": "visual_final_contract_gui_front_edge_repair",
        "selected_candidate": selected,
        "visual_audit": {"status": "pass", "flag_codes": []},
        "boom_proof": _fresh_boom_proof(marker),
        "gui_mask_proof": _fresh_gui_proof(marker),
    }
    candidates = tmp_path / "candidate.json"
    candidates.write_text(json.dumps(detector_payload), encoding="utf-8")

    monkeypatch.setattr(visual_first, "visual_first_marker", lambda *args, **kwargs: detector_payload)
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        validator,
        "marker_boom_proof",
        lambda *args, **kwargs: {
            "passes": False,
            "reasons": ["earlier_dominant_boom_available at 16.000s b8"],
        },
    )
    monkeypatch.setattr(
        validator,
        "_gui_mask_proof",
        lambda *args, **kwargs: {
            "passes": False,
            "reasons": ["marker_has_no_immediate_drop_body"],
            "marker": marker,
            "marker_relevant_mask": True,
            "marker_signal_present": True,
        },
    )

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": marker,
                "selected_by": "visual_final_contract_gui_front_edge_repair",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": True,
            "require_rerun_matches_report": True,
            "rerun_report_tolerance_sec": 0.002,
            "skip_gui_mask": False,
            "waveform_cache_dir": str(tmp_path),
        }
    )

    assert result["passes"] is True
    assert result["rerun_report_delta_sec"] == 0.0
    assert result["boom_proof"]["accepted_by_trusted_persisted_strict_proof"] is True
    assert result["boom_proof"]["trusted_persisted_proof_source"] == "current_detector_rerun"
    assert result["boom_proof"]["raw_validation_recomputed_reasons"] == [
        "earlier_dominant_boom_available at 16.000s b8"
    ]
    assert result["gui_mask"]["accepted_by_trusted_persisted_strict_gui_mask_proof"] is True
    assert result["gui_mask"]["trusted_persisted_proof_source"] == "current_detector_rerun"
    assert result["gui_mask"]["raw_validation_recomputed_reasons"] == ["marker_has_no_immediate_drop_body"]


def test_validate_one_accepts_trusted_sparse_groove_gui_proof_as_actual_body_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "drums_122_7A_6-Test.wav"
    audio.write_text("", encoding="utf-8")
    marker = 11.602409
    boom = _fresh_boom_proof(marker)
    gui = {
        **_fresh_gui_proof(marker),
        "marker_immediate_body_present": False,
        "raw_reasons": ["marker_has_no_immediate_drop_body"],
        "accepted_by_gui_sparse_pulse_proof": True,
        "accepted_by_sparse_groove_front_edge_proof": True,
        "placeable_count": 93,
        "nearest_placeable_offset_sec": 0.0,
        "marker_post_relevant_occupancy_250ms": 0.80,
        "marker_post_rms_max_250ms": 0.53,
        "sparse_groove_front_sparse_score": 0.567,
        "sparse_groove_front_actual_body": 0.540,
        "sparse_groove_front_profile_score": 0.582,
        "sparse_groove_front_body_score": 0.635,
        "sparse_groove_front_darkness": 0.721,
        "sparse_groove_front_post8": 0.540,
        "sparse_groove_front_drum_cont": 0.700,
        "sparse_groove_front_bass": 0.280,
        "sparse_groove_front_simultaneity": 0.604,
        "sparse_groove_front_post_relevant": 0.800,
        "sparse_groove_front_post_rms_peak": 0.688,
    }
    candidates = tmp_path / "candidate.json"
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": marker,
                "selected_by": "visual_absolute_final_context_fine_front_edge_repair",
                "selected_candidate": {
                    "timestamp": marker,
                    "selected_by": "visual_absolute_final_context_fine_front_edge_repair",
                    "visual_components": {
                        "boom_section_darkness": 0.106,
                        "bar_height": 0.106,
                        "max_post8_height": 0.106,
                        "post4_height": 0.540,
                        "post8_height": 0.540,
                        "post_bass8": 0.280,
                        "post_drum8": 0.900,
                        "post_drum_cont8": 0.700,
                        "jump4": 0.160,
                        "jump8": 0.160,
                        "phrase_prior": 0.480,
                    },
                },
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": boom,
                "gui_mask_proof": gui,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: boom)

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": marker,
                "selected_by": "visual_absolute_final_context_fine_front_edge_repair",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is True
    assert result["actual_visual_body_contract_pass"] is True
    assert result["actual_visual_body_contract"]["source"] == "trusted_sparse_groove_gui_proof"


def test_validate_one_rejects_selected_candidate_without_visual_body_metrics(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    proof = _fresh_boom_proof(32.0)
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {"timestamp": 32.0, "selected_by": "visual_boom_grid_one_snap"},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": proof,
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
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert result["actual_visual_body_contract_pass"] is False
    assert "actual_visual_body_contract=hold:missing_visual_body_metrics" in result["reasons"]


def test_validate_one_rejects_candidate_that_is_not_on_the_one(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    proof = _fresh_boom_proof(32.0)
    proof["nearest_profile"]["metrics"]["one_distance_ms"] = 250.0
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {
                    "timestamp": 32.0,
                    "selected_by": "visual_boom_grid_one_snap",
                    "bpm_clock": {"bpm": 128.0, "bar_sec": 1.875, "one_distance_ms": 250.0},
                    "visual_components": {
                        "one_distance_ms": 250.0,
                        "post8_height": 0.72,
                        "post4_height": 0.69,
                        "post_bass8": 0.58,
                        "post_drum8": 0.94,
                        "post_drum_cont8": 0.91,
                    },
                },
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": proof,
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
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert result["grid_on_one_contract_pass"] is False
    assert "grid_on_one_contract=hold:one_distance_ms 250.0 above 90.0" in result["reasons"]


def test_validate_one_rejects_stale_persisted_gui_proof_without_relevance_mask(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    proof = _fresh_boom_proof(32.0)
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "selected_candidate": {
                    "timestamp": 32.0,
                    "selected_by": "visual_boom_grid_one_snap",
                    "visual_components": {
                        "post8_height": 0.72,
                        "post4_height": 0.69,
                        "post_bass8": 0.58,
                        "post_drum8": 0.94,
                        "post_drum_cont8": 0.91,
                    },
                },
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": proof,
                    "gui_mask_proof": {
                        "passes": True,
                        "reasons": [],
                        "placeable_count": 2,
                        "marker_signal_present": True,
                        "marker_immediate_body_present": True,
                    },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.0,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert "persisted_gui_mask_proof=hold:stale_missing_marker_relevant_mask" in result["reasons"]


def test_validate_one_rejects_stale_persisted_boom_front_edge(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "drums_128_1A_7-Test.flac"
    audio.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidate.json"
    candidates.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "drop_sec": 32.421,
                "selected_by": "visual_boom_grid_phase_calibration",
                "selected_candidate": {"timestamp": 32.421, "selected_by": "visual_boom_grid_phase_calibration"},
                "visual_audit": {"status": "pass", "flag_codes": []},
                "boom_proof": {
                    "passes": True,
                    "reasons": [],
                    "marker_sec": 32.421,
                    "nearest": {
                        "edge_time": 32.0,
                        "offset_sec": 0.421,
                        "abs_offset_sec": 0.421,
                    },
                },
                "gui_mask_proof": {
                    "passes": True,
                    "reasons": [],
                    "placeable_count": 2,
                    "marker_relevant_mask": True,
                    "marker_signal_present": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: _fresh_boom_proof(32.421))

    result = validator._validate_one(
        {
            "row": {
                "drums_path": str(audio),
                "marker": 32.421,
                "selected_by": "visual_boom_grid_phase_calibration",
                "audit_status": "pass",
                "audit_flags": [],
                "candidates_json": str(candidates),
            },
            "sample_rate": 16000,
            "rerun_detector": False,
            "require_rerun_matches_report": False,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert any(reason.startswith("persisted_boom_proof=stale_front_edge:") for reason in result["reasons"])


def test_rerun_detector_marker_must_match_saved_report_marker(monkeypatch) -> None:
    import drop_aligner.visual_first as visual_first

    track = "/tmp/drums_128_1A_7-Test.flac"
    monkeypatch.setattr(
        visual_first,
        "visual_first_marker",
        lambda *args, **kwargs: {
            "ok": True,
            "marker": 10.010,
            "final_ai_pick": 10.010,
            "selected_by": "visual_boom_grid_one_snap",
            "selected_candidate": {
                "timestamp": 10.010,
                "selected_by": "visual_boom_grid_one_snap",
            },
            "visual_audit": {"status": "pass", "flag_codes": []},
        },
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": track,
                "marker": 10.000,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
            },
            "sample_rate": 16000,
            "rerun_detector": True,
            "require_rerun_matches_report": True,
            "rerun_report_tolerance_sec": 0.002,
            "skip_gui_mask": True,
        }
    )

    assert result["passes"] is False
    assert result["report_marker"] == 10.0
    assert result["marker"] == 10.01
    assert result["rerun_report_delta_sec"] > 0.002
    assert any(reason.startswith("rerun_marker_mismatch:") for reason in result["reasons"])


def test_rerun_detector_failure_does_not_fall_back_to_saved_report_marker(monkeypatch) -> None:
    import drop_aligner.visual_first as visual_first

    track = "/tmp/drums_128_1A_7-Test.flac"
    monkeypatch.setattr(
        visual_first,
        "visual_first_marker",
        lambda *args, **kwargs: {"ok": False, "error": "no_boom_body_section_candidate"},
    )
    monkeypatch.setattr(validator, "compute_bar_feature_map", lambda *args, **kwargs: {"bar_count": 64, "beatgrid": {}})
    monkeypatch.setattr(validator, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(validator, "marker_boom_proof", lambda *args, **kwargs: {"passes": True, "reasons": []})

    result = validator._validate_one(
        {
            "row": {
                "drums_path": track,
                "marker": 10.000,
                "selected_by": "visual_boom_grid_one_snap",
                "audit_status": "pass",
                "audit_flags": [],
            },
            "sample_rate": 16000,
            "rerun_detector": True,
            "require_rerun_matches_report": True,
            "rerun_report_tolerance_sec": 0.002,
            "skip_gui_mask": True,
            "skip_persisted_proof": True,
        }
    )

    assert result["passes"] is False
    assert result["marker"] is None
    assert "missing_marker" in result["reasons"]
    assert "rerun_marker_missing" in result["reasons"]
    assert result["audit_status"] == "error"
    assert "detector_not_ok:no_boom_body_section_candidate" in result["audit_flags"]
