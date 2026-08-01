from __future__ import annotations

import json

import pytest

import web_review
from web_review import ReviewApp, _drop_point_contract


def _accepted_anchor() -> dict:
    return {
        "ok": True,
        "accepted": True,
        "reason": "visual_grid_one_with_bounded_drum_attack",
        "source": "visual_first_bounded_attack_v1",
        "drop_sec": 54.85770975056689,
        "grid_downbeat_sec": 54.857142857142854,
        "grid_downbeat_sample": 2_419_200,
        "impact_sec": 54.85770975056689,
        "impact_sample": 2_419_225,
        "phase_translation_samples": 25,
        "sample_rate": 44_100,
        "visual_marker_sec": 54.857142857142854,
        "selected_by": "visual_gui_first_fat_block",
        "attack": {
            "sample_rate": 44_100,
            "zero_crossing_sample": 2_419_226,
            "zero_crossing_time_sec": 2_419_226 / 44_100,
            "alternate_candidates": [
                {"sample": 2_419_500, "time_sec": 2_419_500 / 44_100, "confidence": 0.72}
            ],
            "boom_body_crest": {
                "ok": True,
                "diagnostic_only": True,
                "moves_marker": False,
                "crest_sample": 2_420_000,
                "crest_time_sec": 2_420_000 / 44_100,
                "crest_offset_ms": 17.5737,
            },
        },
    }


def test_drop_point_contract_keeps_impact_and_diagnostics_separate() -> None:
    points = _drop_point_contract(_accepted_anchor())

    assert points["status"] == "accepted"
    assert points["musical_one"]["sample"] == 2_419_200
    assert points["als_impact"]["sample"] == 2_419_225
    assert points["phase_translation"] == {"samples": 25, "milliseconds": pytest.approx(25_000 / 44_100)}
    assert points["zero_crossing"]["diagnostic_only"] is True
    assert points["body_crest"]["sample"] == 2_420_000
    assert points["body_crest"]["diagnostic_only"] is True
    assert points["body_crest_evidence"]["moves_marker"] is False
    assert points["alternate_attacks"][0]["diagnostic_only"] is True


def test_review_app_prefers_persisted_als_anchor_without_rescanning(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "drums_140_1A_7-Artist - Track.flac"
    candidates = tmp_path / "track_drop_candidates.json"
    summary = tmp_path / "drop_batch_summary.csv"
    anchor = _accepted_anchor()
    selected = {
        "rank": 1,
        "timestamp": anchor["visual_marker_sec"],
        "snapped_sec": anchor["visual_marker_sec"],
        "selected_by": "visual_gui_first_fat_block",
    }
    candidates.write_text(
        json.dumps(
            {
                "final_ai_pick": anchor["drop_sec"],
                "bpm": 140,
                "selected_candidate": selected,
                "top_10_candidates": [selected],
                "feature_map": {
                    "beatgrid": {
                        "bpm": 140,
                        "bar_zero_sec": 0.0,
                        "bar_sec": 240 / 140,
                        "beat_sec": 60 / 140,
                    }
                },
                "als_anchor": anchor,
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        "filename,detected_drop_time,output_als,candidates_json,status,als_valid,confidence_tier,selected_by\n"
        f"{audio},,{tmp_path / 'track.als'},{candidates},processed,true,HIGH,visual_gui_first_fat_block\n",
        encoding="utf-8",
    )
    app = ReviewApp(
        summary_csv=str(summary),
        template=str(tmp_path / "template.als"),
        correction_log=str(tmp_path / "corrections.jsonl"),
        auto_retrain_every=0,
        review_low_only=False,
        review_medium_and_low=False,
        regenerate_als_on_correction=False,
        visual_first=True,
    )
    monkeypatch.setattr(web_review, "visual_first_marker", lambda *args, **kwargs: pytest.fail("persisted anchor rescanned"))

    public = app._item_public(app.items[0])

    assert public is not None
    assert public["ai_pick"] == anchor["drop_sec"]
    assert public["drop_points"]["visual_marker"]["time_sec"] == anchor["visual_marker_sec"]
    assert public["drop_points"]["als_impact"]["time_sec"] == anchor["impact_sec"]
    assert public["visual_first_scan"]["version"] == "persisted_als_anchor_v1"


def test_fresh_visual_scan_promotes_bounded_impact_to_server_blue(monkeypatch) -> None:
    anchor = _accepted_anchor()
    monkeypatch.setattr(web_review, "build_visual_first_als_anchor", lambda *args, **kwargs: anchor)
    app = ReviewApp.__new__(ReviewApp)
    visual_marker = anchor["visual_marker_sec"]
    selected = {
        "rank": 1,
        "timestamp": visual_marker,
        "snapped_sec": visual_marker,
        "selected_by": "visual_gui_first_fat_block",
    }
    item = {
        "audio_path": "/unused/drums.flac",
        "bpm": 140.0,
        "top_10_candidates": [selected],
        "selected_candidate": selected,
        "confidence_tier": "HIGH",
    }

    ok = app._apply_visual_first_result(
        item,
        {
            "ok": True,
            "marker": visual_marker,
            "selected_candidate": selected,
            "candidates": [selected],
            "feature_map": {"beatgrid": {"bpm": 140.0, "bar_zero_sec": 0.0}},
            "visual_audit": {"status": "pass", "flag_codes": []},
        },
    )

    assert ok is True
    assert item["ai_pick"] == anchor["impact_sec"]
    assert item["drop_points"]["visual_marker"]["time_sec"] == visual_marker
    assert item["drop_points"]["als_impact"]["time_sec"] == anchor["impact_sec"]
    assert item["visual_first_scan"]["als_impact"] == anchor["impact_sec"]


def test_visual_first_review_keeps_unplaced_detector_failure_in_library(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "drums_128_1A_5-Unplaced.wav"
    candidates = tmp_path / "unplaced_drop_candidates.json"
    summary = tmp_path / "full_library_review_summary.csv"
    candidates.write_text(
        json.dumps(
            {
                "source": "visual_first_full_review_sidecar",
                "review_only": True,
                "final_ai_pick": None,
                "selected_by": "visual_first_review_failure",
                "selected_candidate": {
                    "selected": False,
                    "review_only": True,
                    "selected_by": "visual_first_review_failure",
                },
                "top_10_candidates": [],
                "als_anchor": {
                    "ok": False,
                    "accepted": False,
                    "reason": "visual_contract_failed",
                },
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        "filename,detected_drop_time,output_als,candidates_json,status,als_valid,confidence_tier,selected_by\n"
        f"{audio},,,{candidates},partial,,LOW,visual_first_review_failure\n",
        encoding="utf-8",
    )
    app = ReviewApp(
        summary_csv=str(summary),
        template=str(tmp_path / "template.als"),
        correction_log=str(tmp_path / "corrections.jsonl"),
        auto_retrain_every=0,
        review_low_only=False,
        review_medium_and_low=False,
        regenerate_als_on_correction=False,
        visual_first=True,
    )
    monkeypatch.setattr(
        web_review,
        "visual_first_marker",
        lambda *args, **kwargs: {"ok": False, "error": "final_gui_mask_strict_contract_failed"},
    )

    assert len(app.items) == 1
    assert app.items[0]["ai_pick"] is None
    public = app._item_public(app.items[0])
    assert public is not None
    assert public["ai_pick"] is None
    assert public["drop_points"]["status"] == "rejected"
    assert public["visual_first_hold"]["reason"].startswith("fresh visual-first scan failed")


def test_static_gui_renders_all_drop_point_roles() -> None:
    source = (web_review.Path(web_review.__file__).resolve().parent / "static" / "app.js").read_text(encoding="utf-8")
    html = (web_review.Path(web_review.__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")

    for token in ("musicalOne", "visualEdge", "als_impact", "zeroDiagnostic", "bodyCrest", "alternateAttack"):
        assert token in source
    for element_id in (
        "gridOneMarker",
        "visualDropMarker",
        "alsImpactMarker",
        "phaseTranslation",
        "diagnosticZeroMarker",
        "boomCrestMarker",
        "alternateAttackMarkers",
    ):
        assert f'id="{element_id}"' in html
