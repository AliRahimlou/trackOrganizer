from __future__ import annotations

import json
import threading

import web_review
from web_review import (
    ReviewApp,
    _apply_even_bar_prior,
    _apply_post_structure_selection_guard,
    _boundary_variant_candidates,
    _load_review_memory,
    _primary_visual_phrase_candidate,
    _track_zero_grid_variant_candidates,
)


def _candidate(time_sec: float, *, rank: int, score: float, micro: float, snap_ms: float = 0.0) -> dict:
    return {
        "rank": rank,
        "handcrafted_rank": rank,
        "timestamp": time_sec,
        "confidence_score": score,
        "score": score,
        "fake_hit_penalty": 0.0,
        "microalign": {
            "ok": True,
            "microaligned_time": time_sec,
            "micro_confidence": micro,
            "snap_offset_ms": snap_ms,
        },
    }


def test_review_app_loads_fresh_visual_first_library_csv(tmp_path) -> None:
    summary = tmp_path / "fresh.csv"
    audio = tmp_path / "drums_128_1A_7-Artist - Track.flac"
    als = tmp_path / "fresh.als"
    summary.write_text(
        "\n".join(
            [
                "#,BPM,Key,Energy,TrackFolder,MarkerSec,SelectedBy,FreshAlignedAls,DrumsPath",
                f"1,128,1A,7,Artist - Track,48.125,visual_boom_grid_one_snap,{als},{audio}",
            ]
        )
        + "\n",
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

    assert len(app.items) == 1
    item = app.items[0]
    assert item["audio_path"] == str(audio)
    assert item["ai_pick"] == 48.125
    assert item["output_als"] == str(als)
    assert item["selected_by"] == "visual_boom_grid_one_snap"
    assert item["confidence_tier"] == "HIGH"
    assert item["bpm"] == 128.0
    assert item["selected_candidate"]["source"] == "fresh_visual_first_library_csv"


def test_review_item_public_exposes_persisted_feature_map_beatgrid(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    audio = tmp_path / "drums_145_1A_7-Artist - Track.flac"
    candidates_json = tmp_path / "drums_145_1A_7-Artist - Track_drop_candidates.json"
    beatgrid = {
        "bpm": 145.0,
        "beat_sec": 60.0 / 145.0,
        "bar_sec": 240.0 / 145.0,
        "bar_zero_sec": 0.173,
        "downbeat_confidence": 0.94,
    }
    selected = _candidate(48.173, rank=1, score=0.91, micro=0.96)
    selected["selected_by"] = "visual_boom_grid_one_snap"
    candidates_json.write_text(
        json.dumps(
            {
                "final_ai_pick": 48.173,
                "bpm": 145.0,
                "selected_candidate": selected,
                "top_10_candidates": [selected],
                "feature_map": {"ok": True, "bar_count": 96, "beatgrid": beatgrid},
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        "filename,detected_drop_time,output_als,candidates_json,status,als_valid,confidence_tier,selected_by\n"
        f"{audio},48.173,{tmp_path / 'track.als'},{candidates_json},processed,true,HIGH,visual_boom_grid_one_snap\n",
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
    app._prime_item_with_visual_candidate = lambda item, scan=False: None

    public = app._item_public(app.items[0])

    assert public is not None
    assert public["feature_map"]["beatgrid"] == beatgrid
    assert public["beatgrid"] == beatgrid


def test_even_bar_prior_rescues_one_beat_early_marker_to_same_bar_one() -> None:
    early = _candidate(40.736272108843536, rank=1, score=0.746, micro=0.800, snap_ms=0.27)
    on_one = _candidate(41.14321088435374, rank=2, score=0.875, micro=0.856, snap_ms=-72.79)
    result = _apply_even_bar_prior(
        [early, on_one],
        {
            "candidate": early,
            "suggested_time": early["timestamp"],
            "reason": "model selected an early centerline marker",
        },
        bpm=140.0,
        confidence_tier="LOW",
    )

    suggestion = result["suggestion"]

    assert suggestion["candidate"]["rank"] == 2
    assert suggestion["candidate"]["selected_by"] == "musical_clock_rescue"
    assert suggestion["suggested_time"] == on_one["timestamp"]


def test_visual_primary_phrase_prior_allows_32_bar_drop_over_24_bar_probe() -> None:
    early_phrase_probe = _candidate(41.1407029478458, rank=1, score=0.744, micro=0.934)
    true_32_bar_drop = _candidate(54.85582766439909, rank=7, score=0.713, micro=0.940)

    result = _primary_visual_phrase_candidate(
        [early_phrase_probe, true_32_bar_drop],
        early_phrase_probe,
        bpm=140.0,
    )

    assert result is not None
    assert result["rank"] == 7
    assert result["selected_by"] == "visual_primary_phrase_prior"
    assert result["structure_role"] == "first_drop"


def test_visual_primary_phrase_prior_keeps_clean_16_bar_drop_candidate() -> None:
    clean_16_bar_drop = _candidate(27.428571428571427, rank=1, score=0.801, micro=0.958)
    later_32_bar_section = _candidate(54.857142857142854, rank=2, score=0.798, micro=0.988)

    result = _primary_visual_phrase_candidate(
        [clean_16_bar_drop, later_32_bar_section],
        clean_16_bar_drop,
        bpm=140.0,
    )

    assert result is None


def test_visual_primary_phrase_prior_keeps_zoomed_visual_first_primary_bar_marker() -> None:
    zoomed_visual_drop = _candidate(55.72462585034013, rank=1, score=0.600, micro=0.475, snap_ms=867.48)
    zoomed_visual_drop.update(
        {
            "selected_by": "visual_gui_first_fat_block",
            "visual_raw_chunk_time": 54.857142857142854,
            "visual_components": {
                "clock_bar": 33,
                "post_bass8": 0.482,
                "post_drum_cont8": 0.927,
            },
        }
    )
    raw_grid_edge = _candidate(54.85732426303854, rank=2, score=0.600, micro=0.820)

    result = _primary_visual_phrase_candidate(
        [raw_grid_edge, zoomed_visual_drop],
        zoomed_visual_drop,
        bpm=140.0,
    )

    assert result is None


def test_apply_visual_first_result_selects_candidate_matching_blue_marker() -> None:
    app = ReviewApp.__new__(ReviewApp)
    stale_phrase = _candidate(55.35327891156462, rank=1, score=0.70, micro=0.90)
    stale_phrase.update({"selected_by": "visual_primary_phrase_prior"})
    visual_marker = _candidate(82.60951537798219, rank=2, score=0.66, micro=0.92)
    visual_marker.update(
        {
            "selected_by": "visual_gui_first_fat_block",
            "visual_components": {"clock_bar": 49},
            "reason": "visual-only selected first real fat waveform block",
        }
    )
    item = {
        "top_10_candidates": [stale_phrase],
        "confidence_tier": "LOW",
    }
    beatgrid = {
        "bpm": 140.0,
        "beat_sec": 60.0 / 140.0,
        "bar_sec": 240.0 / 140.0,
        "bar_zero_sec": 0.137,
        "downbeat_confidence": 0.91,
    }

    ok = app._apply_visual_first_result(
        item,
        {
            "ok": True,
            "version": 1,
            "marker": visual_marker["timestamp"],
            "raw_visual_time": 82.40135211267607,
            "selected_candidate": stale_phrase,
            "candidates": [visual_marker],
            "feature_map": {"ok": True, "bar_count": 96, "beatgrid": beatgrid},
        },
    )

    assert ok is True
    assert item["ai_pick"] == visual_marker["timestamp"]
    assert item["selected_candidate"]["timestamp"] == visual_marker["timestamp"]
    assert item["top_10_candidates"][0]["timestamp"] == visual_marker["timestamp"]
    assert item["feature_map"]["beatgrid"] == beatgrid
    assert item["beatgrid"] == beatgrid


def test_visual_first_refresh_preserves_successful_visual_scan() -> None:
    app = ReviewApp.__new__(ReviewApp)
    visual_marker = _candidate(82.60951537798219, rank=1, score=0.66, micro=0.92)
    visual_marker.update(
        {
            "selected_by": "visual_gui_first_fat_block",
            "visual_components": {"clock_bar": 49},
        }
    )
    stale_phrase = _candidate(55.35327891156462, rank=2, score=0.70, micro=0.90)
    stale_phrase.update({"selected_by": "visual_primary_phrase_prior"})
    item = {
        "top_10_candidates": [visual_marker, stale_phrase],
        "selected_candidate": dict(visual_marker),
        "selected_by": "visual_gui_first_fat_block",
        "visual_first_scanned": True,
        "visual_first_scan": {"marker": visual_marker["timestamp"], "source": "visual_gui_first_fat_block"},
        "confidence_tier": "LOW",
    }

    app._prime_item_with_visual_candidate(item, scan=True)

    assert item["selected_candidate"]["timestamp"] == visual_marker["timestamp"]
    assert item["top_10_candidates"][0]["timestamp"] == visual_marker["timestamp"]


def test_visual_first_scan_ignores_historical_review_memory(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    track = "/music/STEMS/128/1A/Artist - Track/drums_128_1A_7-Artist - Track.flac"
    app.review_memory = {
        track: {
            "track": track,
            "user_pick": 48.0,
            "reviewed_from": "web_candidate_pick",
            "source": "historical_review_memory",
        }
    }
    historical = _candidate(48.0, rank=1, score=0.99, micro=0.99)
    historical.update({"selected_by": "historical_review_memory"})
    visual_marker = _candidate(64.0, rank=1, score=0.72, micro=0.94)
    visual_marker.update(
        {
            "selected_by": "visual_gui_first_fat_block",
            "visual_components": {"clock_bar": 33},
            "reason": "visual detector marker",
        }
    )
    calls = []

    def fake_visual_first_marker(audio_path, **kwargs):
        calls.append((audio_path, kwargs))
        return {
            "ok": True,
            "version": 1,
            "marker": visual_marker["timestamp"],
            "raw_visual_time": visual_marker["timestamp"],
            "selected_candidate": dict(visual_marker),
            "candidates": [dict(visual_marker)],
        }

    monkeypatch.setattr(web_review, "visual_first_marker", fake_visual_first_marker)
    item = {
        "audio_path": track,
        "top_10_candidates": [historical],
        "selected_candidate": dict(historical),
        "selected_by": "historical_review_memory",
        "confidence_tier": "LOW",
        "review": {},
    }

    app._prime_item_with_visual_candidate(item, scan=True)

    assert calls
    assert item["selected_by"] == "visual_gui_first_fat_block"
    assert item["ai_pick"] == visual_marker["timestamp"]
    assert item["selected_candidate"]["selected_by"] == "visual_gui_first_fat_block"


def test_visual_first_scan_failure_hides_stale_detector_marker(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    stale = _candidate(48.0, rank=1, score=0.81, micro=0.94)
    stale.update({"selected_by": "visual_drop_v2", "reason": "old batch marker"})
    calls = []

    def fake_visual_first_marker(audio_path, **kwargs):
        calls.append((audio_path, kwargs))
        return {"ok": False, "error": "no_boom_body_section_candidate"}

    monkeypatch.setattr(web_review, "visual_first_marker", fake_visual_first_marker)
    item = {
        "audio_path": "/music/STEMS/128/1A/Artist - Track/drums_128_1A_7-Artist - Track.flac",
        "ai_pick": 48.0,
        "ai_pick_label": "0:48.000",
        "top_10_candidates": [dict(stale)],
        "selected_candidate": dict(stale),
        "selected_by": "visual_drop_v2",
        "confidence_tier": "LOW",
        "review": {},
    }

    app._prime_item_with_visual_candidate(item, scan=True)

    assert calls
    assert item["selected_by"] == "visual_first_hold"
    assert item["ai_pick"] is None
    assert item["selected_candidate"] == {}
    assert item["top_10_candidates"] == []
    assert item["visual_first_scan_error"] == "no_boom_body_section_candidate"
    assert item["visual_first_hold"]["scan_error"] == "no_boom_body_section_candidate"
    assert item["visual_first_stale_marker"] == 48.0
    assert item["visual_first_stale_selected_by"] == "visual_drop_v2"


def test_visual_first_previous_scan_failure_stays_hold_instead_of_old_fallback() -> None:
    app = ReviewApp.__new__(ReviewApp)
    stale = _candidate(48.0, rank=1, score=0.81, micro=0.94)
    stale.update({"selected_by": "visual_primary_phrase_prior", "reason": "old fallback marker"})
    item = {
        "audio_path": "/music/STEMS/128/1A/Artist - Track/drums_128_1A_7-Artist - Track.flac",
        "ai_pick": 48.0,
        "ai_pick_label": "0:48.000",
        "top_10_candidates": [dict(stale)],
        "selected_candidate": dict(stale),
        "selected_by": "visual_primary_phrase_prior",
        "confidence_tier": "LOW",
        "review": {},
        "visual_first_scanned": True,
        "visual_first_scan_error": "fresh detector failed",
    }

    app._prime_item_with_visual_candidate(item, scan=True)

    assert item["selected_by"] == "visual_first_hold"
    assert item["ai_pick"] is None
    assert item["selected_candidate"] == {}
    assert item["top_10_candidates"] == []
    assert item["visual_first_hold"]["scan_error"] == "fresh detector failed"
    assert item["visual_first_stale_marker"] == 48.0


def test_regenerate_als_rejects_visual_first_marker_before_writing(monkeypatch, tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {
        "track-1": {
            "id": "track-1",
            "audio_path": str(tmp_path / "drums_128_1A_7-Artist - Track.flac"),
            "output_als": str(tmp_path / "out.als"),
            "candidates_json": str(tmp_path / "candidates.json"),
            "selected_candidate": {"timestamp": 48.0, "selected_by": "visual_drop_v2"},
            "review": {},
            "bpm": 128.0,
        }
    }
    calls = []
    monkeypatch.setattr(
        app,
        "_visual_save_guard",
        lambda *args, **kwargs: {"ok": False, "error": "visual-first save rejected: boom_proof_failed"},
    )
    monkeypatch.setattr(web_review, "modify_als", lambda *args, **kwargs: calls.append("modify"))

    result = app.regenerate_als("track-1", 48.0)

    assert result["ok"] is False
    assert "boom_proof_failed" in result["error"]
    assert calls == []


def test_regenerate_als_visual_first_manual_marker_does_not_use_stale_item_candidate(monkeypatch, tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    output = tmp_path / "out.als"
    app.item_by_id = {
        "track-1": {
            "id": "track-1",
            "audio_path": str(tmp_path / "drums_128_1A_7-Artist - Track.flac"),
            "output_als": str(output),
            "candidates_json": str(tmp_path / "candidates.json"),
            "selected_candidate": {"rank": 99, "selected_by": "stale_blue_marker"},
            "review": {},
            "bpm": 128.0,
        }
    }
    seen = []
    app._visual_save_guard = lambda *args, **kwargs: seen.append(kwargs.get("selected_candidate")) or {"ok": False, "error": "stop before write"}
    monkeypatch.setattr(web_review, "modify_als", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write ALS")))

    result = app.regenerate_als("track-1", 48.0, reviewed_from="web_manual_marker")

    assert result["ok"] is False
    assert seen == [None]


def test_regenerate_als_visual_first_manual_marker_ignores_explicit_guard_candidate(monkeypatch, tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    output = tmp_path / "out.als"
    app.template = tmp_path / "template.als"
    app.item_by_id = {
        "track-1": {
            "id": "track-1",
            "audio_path": str(tmp_path / "drums_128_1A_7-Artist - Track.flac"),
            "output_als": str(output),
            "candidates_json": str(tmp_path / "candidates.json"),
            "selected_candidate": {"rank": 99, "selected_by": "stale_blue_marker"},
            "review": {},
            "bpm": 128.0,
        }
    }
    seen = []
    app._visual_save_guard = lambda *args, **kwargs: seen.append(kwargs.get("selected_candidate")) or {"ok": False, "error": "stop before write"}
    monkeypatch.setattr(web_review, "modify_als", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write ALS")))

    result = app.regenerate_als(
        "track-1",
        48.0,
        reviewed_from="web_manual_marker",
        guard_candidate={"rank": 4, "timestamp": 48.0},
    )

    assert result["ok"] is False
    assert seen == [None]


def test_visual_first_save_guard_rejects_marker_before_boom_front_edge(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    marker = 32.0

    monkeypatch.setattr(
        web_review,
        "compute_bar_feature_map",
        lambda *args, **kwargs: {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}},
    )
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        web_review,
        "marker_boom_proof",
        lambda *args, **kwargs: {
            "passes": True,
            "reasons": [],
            "nearest": {"offset_sec": -0.050, "edge_time": 32.050},
            "nearest_profile": {"passes_profile": True, "reasons": []},
        },
    )

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
            "selected_candidate": {"timestamp": marker},
        },
        marker,
    )

    assert result["ok"] is False
    assert "before Boom front edge" in result["error"]
    assert "limit 20ms" in result["error"]


def test_visual_first_save_guard_uses_visual_first_production_sample_rate(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app._visual_gui_mask_proof = lambda *args, **kwargs: {
        "passes": True,
        "reasons": [],
        "placeable_count": 1,
        "marker_signal_present": True,
        "marker_relevant_mask": True,
        "marker_immediate_body_present": True,
    }
    marker = 32.0
    sample_rates = []

    def fake_feature_map(_audio_path, *, sample_rate, use_cache):
        sample_rates.append((sample_rate, use_cache))
        return {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}}

    monkeypatch.setattr(web_review, "compute_bar_feature_map", fake_feature_map)
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        web_review,
        "marker_boom_proof",
        lambda *args, **kwargs: {
            "passes": True,
            "reasons": [],
            "nearest": {"offset_sec": 0.0, "edge_time": marker},
            "nearest_profile": {"passes_profile": True, "profile_score": 0.72, "reasons": []},
        },
    )

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
        },
        marker,
    )

    assert result["ok"] is True
    assert sample_rates == [(web_review.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE, True)]
    assert web_review.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE == 44100


def test_visual_first_save_guard_rejects_non_placeable_gui_boom_mask(monkeypatch) -> None:
    class FakeWaveformCache:
        def info(self, _audio_path):
            return {"duration": 24.0}

        def tile(self, _audio_path, *, start_sec, end_sec, width):
            return {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "width": int(width),
                "boom_relevant_mask": [False, False, False, False, False, False],
                "boom_placeable_mask": [False, False, False, False, False, False],
                "boom_placeable_count": 0,
                "boom_bin_span_sec": (float(end_sec) - float(start_sec)) / 6.0,
                "cache_hit": False,
            }

    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app.waveform_cache = FakeWaveformCache()
    marker = 12.0

    monkeypatch.setattr(
        web_review,
        "compute_bar_feature_map",
        lambda *args, **kwargs: {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}},
    )
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        web_review,
        "marker_boom_proof",
        lambda *args, **kwargs: {
            "passes": True,
            "reasons": [],
            "nearest": {"offset_sec": 0.0, "edge_time": marker},
            "nearest_profile": {"passes_profile": True, "profile_score": 0.49, "reasons": []},
        },
    )

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
            "selected_candidate": {"timestamp": marker},
        },
        marker,
    )

    assert result["ok"] is False
    assert "gui_mask_failed" in result["error"]
    assert "marker_not_on_gui_boom_front_edge_mask" in result["error"]
    assert "gui_tile_has_no_placeable_boom_front_edge" in result["error"]


def test_visual_first_save_guard_accepts_marker_on_gui_boom_front_edge(monkeypatch) -> None:
    class FakeWaveformCache:
        def info(self, _audio_path):
            return {"duration": 24.0}

        def tile(self, _audio_path, *, start_sec, end_sec, width):
            return {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "width": int(width),
                "rms": [0.0, 0.0, 0.0, 0.75, 0.70, 0.68],
                "rms_peak": 0.75,
                "global_rms_peak": 0.75,
                "boom_body_density": [0.02, 0.03, 0.04, 0.92, 0.88, 0.86],
                "boom_body_mask": [False, False, False, True, True, True],
                "boom_relevant_mask": [False, False, False, True, True, True],
                "boom_placeable_mask": [False, False, False, True, False, False],
                "boom_placeable_count": 1,
                "boom_bin_span_sec": (float(end_sec) - float(start_sec)) / 6.0,
                "cache_hit": False,
            }

    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app.waveform_cache = FakeWaveformCache()
    marker = 12.0

    monkeypatch.setattr(
        web_review,
        "compute_bar_feature_map",
        lambda *args, **kwargs: {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}},
    )
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        web_review,
        "marker_boom_proof",
        lambda *args, **kwargs: {
            "passes": True,
            "reasons": [],
            "nearest": {"offset_sec": 0.0, "edge_time": marker},
            "nearest_profile": {"passes_profile": True, "profile_score": 0.49, "reasons": []},
        },
    )

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
            "selected_candidate": {"timestamp": marker},
        },
        marker,
    )

    assert result["ok"] is True
    assert result["gui_mask"]["passes"] is True
    assert result["gui_mask"]["placeable_count"] == 1


def test_visual_first_save_guard_derives_manual_marker_proof_from_server_gui_mask(monkeypatch) -> None:
    class FakeWaveformCache:
        def info(self, _audio_path):
            return {"duration": 24.0}

        def tile(self, _audio_path, *, start_sec, end_sec, width):
            return {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "width": int(width),
                "boom_body_density": [
                    0.08,
                    0.10,
                    0.11,
                    0.12,
                    0.15,
                    0.18,
                    0.92,
                    0.94,
                    0.91,
                    0.88,
                    0.86,
                    0.84,
                ],
                "boom_body_mask": [False, False, False, False, False, False, True, True, True, True, True, True],
                "boom_relevant_mask": [False, False, False, False, False, False, True, True, True, True, True, True],
                "boom_placeable_mask": [False, False, False, False, False, False, True, False, False, False, False, False],
                "boom_placeable_count": 1,
                "boom_bin_span_sec": (float(end_sec) - float(start_sec)) / 12.0,
                "cache_hit": False,
            }

    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app.waveform_cache = FakeWaveformCache()
    marker = 12.0

    monkeypatch.setattr(
        web_review,
        "compute_bar_feature_map",
        lambda *args, **kwargs: {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}},
    )
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])
    calls = []

    def fake_marker_boom_proof(_marker, _boom_candidates, *, selected_candidate, **_kwargs):
        calls.append(dict(selected_candidate))
        if selected_candidate.get("selected_by") == "server_gui_boom_front_edge_contract":
            return {
                "passes": True,
                "reasons": [],
                "nearest": {"offset_sec": 0.0, "edge_time": marker},
                "nearest_profile": {"passes_profile": True, "profile_score": 0.72, "reasons": []},
            }
        return {
            "passes": False,
            "reasons": ["no_boom_body_section_candidate"],
            "nearest": {"offset_sec": 0.0, "edge_time": marker},
            "nearest_profile": {"passes_profile": False, "profile_score": 0.0, "reasons": ["no profile"]},
        }

    monkeypatch.setattr(web_review, "marker_boom_proof", fake_marker_boom_proof)

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
        },
        marker,
    )

    assert result["ok"] is True
    assert result["boom_proof"]["server_gui_front_edge_candidate_used"] is True
    assert result["gui_mask"]["passes"] is True
    assert len(calls) == 2
    assert calls[0]["timestamp"] == marker
    assert calls[0].get("selected_by") is None
    assert calls[1]["selected_by"] == "server_gui_boom_front_edge_contract"
    assert calls[1]["visual_components"]["server_gui_front_edge_contract"] is True


def test_visual_first_save_guard_does_not_inherit_stale_item_candidate_for_manual_marker(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app._visual_gui_mask_proof = lambda *args, **kwargs: {
        "passes": True,
        "reasons": [],
        "placeable_count": 1,
        "marker_signal_present": True,
        "marker_relevant_mask": True,
        "marker_immediate_body_present": True,
    }
    marker = 12.0
    captured = []

    monkeypatch.setattr(
        web_review,
        "compute_bar_feature_map",
        lambda *args, **kwargs: {"bar_count": 96, "beatgrid": {"bpm": 120.0, "bar_zero_sec": 0.0}},
    )
    monkeypatch.setattr(web_review, "boom_body_section_candidates", lambda *args, **kwargs: [])

    def fake_marker_boom_proof(_marker, _boom_candidates, *, selected_candidate, **_kwargs):
        captured.append(dict(selected_candidate))
        return {
            "passes": True,
            "reasons": [],
            "nearest": {"offset_sec": 0.0, "edge_time": marker},
            "nearest_profile": {"passes_profile": True, "profile_score": 0.49, "reasons": []},
        }

    monkeypatch.setattr(web_review, "marker_boom_proof", fake_marker_boom_proof)

    result = app._visual_save_guard(
        {
            "id": "track-1",
            "audio_path": "/music/drums_120_1A_7-Test.flac",
            "bpm": 120.0,
            "selected_candidate": {
                "rank": 99,
                "selected_by": "stale_blue_marker",
                "visual_components": {"stale_blue_context": True},
            },
        },
        marker,
    )

    assert result["ok"] is True
    assert captured
    assert captured[0]["timestamp"] == marker
    assert captured[0]["snapped_sec"] == marker
    assert "rank" not in captured[0]
    assert captured[0].get("selected_by") is None
    assert captured[0]["visual_components"].get("stale_blue_context") is None


def test_visual_first_refine_marker_endpoint_is_disabled() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {"track-1": {"id": "track-1", "ai_pick": 48.0, "audio_path": "/music/drums_128_1A_7-Test.flac"}}

    result = app.refine_marker("track-1", 48.0)

    assert result["ok"] is False
    assert "disabled in visual-first mode" in result["error"]


def test_visual_first_non_visual_auto_place_modes_are_disabled() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {"track-1": {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac"}}

    result = app.auto_place("track-1", "normal")

    assert result["ok"] is False
    assert result["source"] == "visual_first_only"
    assert "non-visual auto_place modes are disabled" in result["error"]


def test_visual_first_label_section_endpoint_is_disabled() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {"track-1": {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac", "review": {}}}

    result = app.label_section("track-1", "drop", marker_time=32.0)

    assert result["ok"] is False
    assert "label_section is disabled in visual-first mode" in result["error"]


def test_visual_first_retrain_endpoint_is_disabled(monkeypatch) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    calls = []
    monkeypatch.setattr(web_review, "_run_retrain", lambda *args, **kwargs: calls.append("retrain") or 0)

    result = app.run_retrain()

    assert result["ok"] is False
    assert "retrain is disabled in visual-first mode" in result["error"]
    assert calls == []


def test_visual_first_after_logged_review_does_not_schedule_retrain() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.visual_first = True
    app.auto_retrain_every = 0
    app.state = {"new_reviews_since_retrain": 0}
    advanced = []
    saved = []
    app._advance = lambda: advanced.append(True)
    app._save_state = lambda: saved.append(dict(app.state))

    app._after_logged_review()

    assert app.state["new_reviews_since_retrain"] == 1
    assert app.state.get("retrain_due") is None
    assert advanced == [True]
    assert saved


def test_visual_first_correct_rejects_blue_acceptance_semantics() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {"track-1": {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac", "review": {}}}
    calls = []
    app._visual_save_guard = lambda *args, **kwargs: calls.append("guard") or {"ok": True}

    result = app.correct("track-1", 32.0, reviewed_from="web_accept_blue_marker")

    assert result["ok"] is False
    assert "must use /api/approve" in result["error"]
    assert calls == []


def test_visual_first_approve_rejects_browser_server_blue_marker_mismatch() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.item_by_id = {
        "track-1": {
            "id": "track-1",
            "audio_path": "/music/drums_128_1A_7-Test.flac",
            "ai_pick": 32.0,
            "selected_candidate": {"timestamp": 32.0},
            "review": {},
        }
    }
    calls = []
    app._visual_save_guard = lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True}

    result = app.approve("track-1", marker_time=32.25, picked_candidate={"rank": 1, "timestamp": 32.25})

    assert result["ok"] is False
    assert "blue marker mismatch" in result["error"]
    assert calls == []


def test_visual_first_approve_uses_client_candidate_context_only_after_marker_match() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "ai_pick": 32.0,
        "selected_candidate": {"rank": 99, "timestamp": 32.0, "selected_by": "stale_context"},
        "review": {},
    }
    app.item_by_id = {"track-1": item}
    guard_candidates = []
    logged = []
    app._visual_save_guard = lambda *args, **kwargs: guard_candidates.append(kwargs.get("selected_candidate")) or {"ok": True}
    app._log_review = lambda *args, **kwargs: logged.append(kwargs)
    app._after_logged_review = lambda: None
    app.state_payload = lambda: {"current": "ok"}

    result = app.approve(
        "track-1",
        marker_time=32.0005,
        picked_candidate={"rank": 2, "timestamp": 32.0005, "selected_by": "client_validated_blue"},
    )

    assert result["ok"] is True
    assert item["review"]["approved"] is True
    assert guard_candidates
    assert guard_candidates[0]["rank"] == 2
    assert guard_candidates[0]["timestamp"] == 32.0
    assert guard_candidates[0]["snapped_sec"] == 32.0
    assert logged[0]["reviewed_from"] == "web_accept_blue_marker"
    assert logged[0]["selected_candidate"]["rank"] == 2


def test_visual_first_correct_is_logged_as_green_manual_not_candidate_pick() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.regenerate_als_on_correction = False
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "ai_pick": 32.0,
        "review": {},
    }
    app.item_by_id = {"track-1": item}
    guard_candidates = []
    app._visual_save_guard = lambda *args, **kwargs: guard_candidates.append(kwargs.get("selected_candidate")) or {"ok": True}
    logged = []
    updated = []
    app._log_review = lambda *args, **kwargs: logged.append(kwargs)
    app._update_candidate_json_user_pick = lambda *args, **kwargs: updated.append(kwargs)
    app._after_logged_review = lambda: None
    app.state_payload = lambda: {"ok": True}

    result = app.correct(
        "track-1",
        32.0,
        reviewed_from="web_candidate_pick",
        picked_candidate={"rank": 4, "timestamp": 32.0},
        top_candidates=[{"rank": 4, "timestamp": 32.0}],
    )

    assert result["ok"] is True
    assert item["review"]["corrected"] is True
    assert guard_candidates
    assert guard_candidates[0] is None
    assert logged[0]["reviewed_from"] == "web_manual_marker"
    assert logged[0]["selected_candidate"] is None
    assert logged[0]["selected_by"] is None
    assert updated[0]["reviewed_from"] == "web_manual_marker"
    assert updated[0]["selected_candidate"] is None
    assert updated[0]["selected_by"] is None


def test_visual_first_manual_correct_does_not_fallback_to_stale_blue_candidate(tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.regenerate_als_on_correction = False
    candidates_json = tmp_path / "candidates.json"
    candidates_json.write_text(
        json.dumps(
            {
                "selected_by": "stale_blue_marker",
                "selected_candidate": {"rank": 99, "timestamp": 28.0, "selected_by": "stale_blue_marker"},
            }
        ),
        encoding="utf-8",
    )
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "ai_pick": 28.0,
        "selected_by": "stale_blue_marker",
        "selected_candidate": {"rank": 99, "timestamp": 28.0, "selected_by": "stale_blue_marker"},
        "candidates_json": str(candidates_json),
        "review": {},
    }
    app.item_by_id = {"track-1": item}
    app._visual_save_guard = lambda *args, **kwargs: {"ok": True, "boom_proof": {"passes": True}, "gui_mask": {"passes": True}}
    logged = []
    app._log_review = lambda *args, **kwargs: logged.append(kwargs)
    app._after_logged_review = lambda: None
    app.state_payload = lambda: {"ok": True}

    result = app.correct("track-1", 32.0, reviewed_from="web_manual_marker")

    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert logged[0]["reviewed_from"] == "web_manual_marker"
    assert logged[0]["selected_candidate"] is None
    assert logged[0]["selected_by"] is None
    assert "selected_candidate" not in payload
    assert "selected_by" not in payload
    assert payload["user_pick"] == 32.0
    assert payload["reviewed_from"] == "web_manual_marker"


def test_visual_first_correct_does_not_log_or_mark_reviewed_when_regeneration_fails() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.regenerate_als_on_correction = True
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "ai_pick": 32.0,
        "review": {
            "reviewed": False,
            "skipped": False,
            "approved": False,
            "corrected": False,
            "user_pick": None,
        },
    }
    app.item_by_id = {"track-1": item}
    app._visual_save_guard = lambda *args, **kwargs: {"ok": True}
    app.regenerate_als = lambda *args, **kwargs: {"ok": False, "error": "verify failed"}
    app._log_review = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not log failed ALS writes"))
    app._after_logged_review = lambda: (_ for _ in ()).throw(AssertionError("must not advance failed ALS writes"))

    result = app.correct("track-1", 32.0, reviewed_from="web_manual_marker")

    assert result["ok"] is False
    assert result["error"] == "verify failed"
    assert result["regeneration"] == {"ok": False, "error": "verify failed"}
    assert item["review"] == {
        "reviewed": False,
        "skipped": False,
        "approved": False,
        "corrected": False,
        "user_pick": None,
    }


def test_regenerate_als_persists_visual_guard_proofs_to_candidate_json(monkeypatch, tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.template = tmp_path / "template.als"
    app.state = {}
    app._save_state = lambda: None
    output = tmp_path / "out.als"
    candidates_json = tmp_path / "candidates.json"
    candidates_json.write_text(json.dumps({"top_10_candidates": []}), encoding="utf-8")
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "output_als": str(output),
        "candidates_json": str(candidates_json),
        "review": {},
        "bpm": 128.0,
    }
    app.item_by_id = {"track-1": item}
    app._visual_save_guard = lambda *args, **kwargs: {
        "ok": True,
        "boom_proof": {"passes": True, "nearest": {"edge_time": 32.0}},
        "gui_mask": {"passes": True, "placeable_count": 2},
    }
    monkeypatch.setattr(web_review, "modify_als", lambda **kwargs: output.write_text("als", encoding="utf-8"))
    monkeypatch.setattr(web_review, "verify_als", lambda *args, **kwargs: {"valid": True, "errors": []})

    result = app.regenerate_als(
        "track-1",
        32.0,
        reviewed_from="visual_detector_prep",
        selected_candidate={"timestamp": 32.0, "selected_by": "visual_boom_grid_one_snap"},
        selected_by="visual_boom_grid_one_snap",
    )

    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert payload["boom_proof"] == {"passes": True, "nearest": {"edge_time": 32.0}}
    assert payload["gui_mask_proof"] == {"passes": True, "placeable_count": 2}
    assert payload["selected_candidate"]["boom_proof"]["passes"] is True
    assert payload["selected_candidate"]["gui_mask_proof"]["passes"] is True


def test_regenerate_als_manual_marker_does_not_persist_stale_blue_candidate(monkeypatch, tmp_path) -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    app.visual_first = True
    app.template = tmp_path / "template.als"
    app.state = {}
    app._save_state = lambda: None
    output = tmp_path / "out.als"
    candidates_json = tmp_path / "candidates.json"
    candidates_json.write_text(
        json.dumps(
            {
                "selected_by": "stale_blue_marker",
                "selected_candidate": {"rank": 99, "timestamp": 28.0, "selected_by": "stale_blue_marker"},
            }
        ),
        encoding="utf-8",
    )
    item = {
        "id": "track-1",
        "audio_path": "/music/drums_128_1A_7-Test.flac",
        "output_als": str(output),
        "candidates_json": str(candidates_json),
        "selected_by": "stale_blue_marker",
        "selected_candidate": {"rank": 99, "timestamp": 28.0, "selected_by": "stale_blue_marker"},
        "review": {},
        "bpm": 128.0,
    }
    app.item_by_id = {"track-1": item}
    app._visual_save_guard = lambda *args, **kwargs: {
        "ok": True,
        "boom_proof": {"passes": True},
        "gui_mask": {"passes": True},
    }
    monkeypatch.setattr(web_review, "modify_als", lambda **kwargs: output.write_text("als", encoding="utf-8"))
    monkeypatch.setattr(web_review, "verify_als", lambda *args, **kwargs: {"valid": True, "errors": []})

    result = app.regenerate_als("track-1", 32.0, reviewed_from="web_manual_marker")

    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert "selected_candidate" not in payload
    assert "selected_by" not in payload
    assert payload["boom_proof"]["passes"] is True
    assert payload["gui_mask_proof"]["passes"] is True


def test_validate_visual_marker_returns_guard_payload_without_writing() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    item = {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac", "selected_candidate": {"timestamp": 32.0}}
    app.item_by_id = {"track-1": item}
    app.get_item = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate must not build a public item"))
    calls = []
    app._visual_save_guard = lambda item_arg, marker, **kwargs: calls.append((item_arg, marker, kwargs)) or {
        "ok": True,
        "boom_proof": {"passes": True},
        "gui_mask": {"passes": True},
        "bpm_clock": {"one_distance_ms": 0.0},
    }
    app._log_review = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validate must not log review"))
    app._after_logged_review = lambda: (_ for _ in ()).throw(AssertionError("validate must not advance review"))

    result = app.validate_visual_marker(
        "track-1",
        32.0,
        picked_candidate={"rank": 2, "timestamp": 32.0},
        context="blue",
    )

    assert result["ok"] is True
    assert result["valid"] is True
    assert result["marker_time"] == 32.0
    assert result["error"] == ""
    assert result["boom_proof"] == {"passes": True}
    assert result["gui_mask"] == {"passes": True}
    assert result["bpm_clock"] == {"one_distance_ms": 0.0}
    assert calls == [(item, 32.0, {"selected_candidate": {"rank": 2, "timestamp": 32.0}})]


def test_validate_visual_marker_ignores_candidate_context_for_manual_green_marker() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    item = {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac", "selected_candidate": {"timestamp": 32.0}}
    app.item_by_id = {"track-1": item}
    calls = []
    app._visual_save_guard = lambda item_arg, marker, **kwargs: calls.append((item_arg, marker, kwargs)) or {
        "ok": False,
        "error": "visual-first save rejected: marker-only proof failed",
        "boom_proof": {"passes": False},
        "gui_mask": {"passes": False},
    }
    app._log_review = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validate must not log review"))
    app._after_logged_review = lambda: (_ for _ in ()).throw(AssertionError("validate must not advance review"))

    result = app.validate_visual_marker(
        "track-1",
        32.0,
        picked_candidate={"rank": 2, "timestamp": 32.0, "visual_components": {"stale": True}},
        context="manual",
    )

    assert result["ok"] is True
    assert result["valid"] is False
    assert "marker-only proof failed" in result["error"]
    assert calls == [(item, 32.0, {"selected_candidate": None})]


def test_validate_visual_marker_reports_rejection_without_writing() -> None:
    app = ReviewApp.__new__(ReviewApp)
    app.lock = threading.RLock()
    item = {"id": "track-1", "audio_path": "/music/drums_128_1A_7-Test.flac"}
    app.item_by_id = {"track-1": item}
    app.get_item = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate must not build a public item"))
    app._visual_save_guard = lambda *args, **kwargs: {
        "ok": False,
        "error": "visual-first save rejected: marker_not_on_gui_boom_front_edge_mask",
        "boom_proof": {"passes": False, "reasons": ["nearest_boom_edge 3.000s away"]},
        "gui_mask": {"passes": False, "reasons": ["marker_not_on_gui_boom_front_edge_mask"]},
    }
    app._log_review = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validate must not log review"))
    app._after_logged_review = lambda: (_ for _ in ()).throw(AssertionError("validate must not advance review"))

    result = app.validate_visual_marker("track-1", 12.0)

    assert result["ok"] is True
    assert result["valid"] is False
    assert "marker_not_on_gui_boom_front_edge_mask" in result["error"]
    assert result["boom_proof"]["passes"] is False
    assert result["gui_mask"]["passes"] is False


def test_boundary_variants_do_not_promote_raw_clock_boundary_without_visual_support() -> None:
    candidate = _candidate(27.428571, rank=1, score=0.80, micro=0.91)
    candidate["microalign"].update(
        {
            "input_candidate_time": 27.428571,
            "input_boundary_quality": 0.0,
            "input_boundary_used": 0.0,
            "attack_start_time": 27.492,
        }
    )

    variants = _boundary_variant_candidates([candidate], max_candidates=1)

    assert not any(row.get("boundary_variant") == "clock_boundary" for row in variants)
    assert not any(row.get("boundary_variant") == "input_boundary" for row in variants)


def test_boundary_variants_keep_input_boundary_when_visually_bracketed() -> None:
    candidate = _candidate(27.428571, rank=1, score=0.80, micro=0.91)
    candidate["microalign"].update(
        {
            "input_candidate_time": 27.428571,
            "input_boundary_quality": 0.88,
            "input_boundary_used": 1.0,
        }
    )

    variants = _boundary_variant_candidates([candidate], max_candidates=1)

    assert any(row.get("boundary_variant") == "input_boundary" for row in variants)


def test_track_zero_grid_variants_do_not_move_structure_first_drop_far_from_visual_marker() -> None:
    first_drop = _candidate(54.854603, rank=2, score=0.74, micro=0.90)
    first_drop.update(
        {
            "structure_role": "first_drop",
            "section_label": "first_drop",
            "selected_by": "saved_visual_batch_auto_marker",
        }
    )

    variants = _track_zero_grid_variant_candidates(
        {"beatgrid": {"bpm": 140.0}, "audio_path": "/music/drums_140_2A_7-Test.flac"},
        [first_drop],
    )

    assert variants
    assert max(abs(row["timestamp"] - first_drop["timestamp"]) for row in variants) <= 0.120


def test_review_memory_ignores_batch_auto_rows(tmp_path) -> None:
    log_path = tmp_path / "drop_corrections.jsonl"
    track = "/music/STEMS/140/4A/Artist - Track/drums_140_4A_7-Artist - Track.flac"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"track": track, "user_pick": 27.428, "reviewed_from": "visual_first_batch_auto"}),
                json.dumps({"track": track, "user_pick": 41.149, "reviewed_from": "web_candidate_pick"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    memory = _load_review_memory(str(log_path))

    assert memory[track]["user_pick"] == 41.149
    assert memory[track]["reviewed_from"] == "web_candidate_pick"


def test_post_structure_guard_restores_earlier_visual_first_drop_over_late_learned_pick() -> None:
    visual_first_drop = _candidate(38.4, rank=2, score=0.58, micro=0.92)
    visual_first_drop.update(
        {
            "structure_role": "first_drop",
            "section_label": "first_drop",
            "structure_components": {
                "sustained_groove": 0.62,
                "low_to_high": 0.58,
                "novelty": 0.55,
                "phrase_prior": 0.86,
            },
        }
    )
    late_model_pick = _candidate(153.6, rank=1, score=0.86, micro=0.84)
    learned = {
        "candidate": late_model_pick,
        "selection_probability": 0.70,
        "chooser_score": 0.70,
        "selection_confidence": 0.70,
    }

    guarded = _apply_post_structure_selection_guard(
        learned,
        [late_model_pick, visual_first_drop],
        {"bpm": 100.0, "bar_zero_sec": 0.0},
    )

    assert guarded["candidate"]["structure_role"] == "first_drop"
    assert guarded["candidate"]["timestamp"] == 38.4
    assert "visual-first guard" in guarded["post_structure_guard_reason"]


def test_post_structure_guard_blocks_far_later_saved_pick_over_first_drop_cluster() -> None:
    saved_first_drop = _candidate(54.854603, rank=2, score=0.74, micro=0.0)
    saved_first_drop.update(
        {
            "selected_by": "saved_visual_batch_auto_marker",
            "source": "saved_visual_batch_auto_marker",
            "reason": "previous visual-first batch marker retained as a normal rescan candidate",
            "structure_role": "first_drop",
            "section_label": "first_drop",
        }
    )
    late_saved_pick = _candidate(219.425510, rank=6, score=0.88, micro=0.91)
    late_saved_pick.update(
        {
            "selected_by": "candidate_chooser",
            "reason": "multistem_candidate:saved",
        }
    )
    learned = {
        "candidate": late_saved_pick,
        "selection_probability": 0.91,
        "chooser_score": 0.91,
        "selection_confidence": 0.91,
    }

    guarded = _apply_post_structure_selection_guard(
        learned,
        [late_saved_pick, saved_first_drop],
        {"bpm": 140.0, "bar_zero_sec": 0.0},
    )

    assert guarded["candidate"]["timestamp"] == saved_first_drop["timestamp"]
    assert guarded["candidate"]["structure_role"] == "first_drop"
    assert "first-drop cluster" in guarded["post_structure_guard_reason"]


def test_post_structure_guard_restores_clean_16_bar_candidate_over_later_primary_phrase() -> None:
    clean_16_bar_drop = _candidate(27.428571428571427, rank=3, score=0.801, micro=0.958)
    clean_16_bar_drop["selected_by"] = "candidate_chooser"
    later_primary = _candidate(54.86163265306122, rank=1, score=0.740, micro=0.988)
    later_primary.update(
        {
            "selected_by": "visual_primary_phrase_prior",
            "structure_role": "first_drop",
            "section_label": "first_drop",
            "reason": "visual 32-bar phrase prior selected over early/off-phrase marker",
        }
    )
    learned = {
        "candidate": later_primary,
        "selection_probability": 0.61,
        "chooser_score": 0.61,
        "selection_confidence": 0.61,
    }

    guarded = _apply_post_structure_selection_guard(
        learned,
        [later_primary, clean_16_bar_drop],
        {"bpm": 140.0, "bar_zero_sec": 0.0},
    )

    assert guarded["candidate"]["timestamp"] == clean_16_bar_drop["timestamp"]
    assert "clean on-ONE visual candidate" in guarded["post_structure_guard_reason"]


def test_post_structure_guard_restores_nearby_clean_one_over_offphrase_tail() -> None:
    clean_one = _candidate(27.428571428571427, rank=4, score=0.796, micro=0.956)
    clean_one["selected_by"] = "multistem_candidate"
    offphrase_tail = _candidate(29.142857142857142, rank=1, score=0.526, micro=0.70)
    offphrase_tail["selected_by"] = "structure_map"
    learned = {
        "candidate": offphrase_tail,
        "selection_probability": 0.20,
        "chooser_score": 0.20,
        "selection_confidence": 0.20,
    }

    guarded = _apply_post_structure_selection_guard(
        learned,
        [offphrase_tail, clean_one],
        {"bpm": 140.0, "bar_zero_sec": 0.0},
    )

    assert guarded["candidate"]["timestamp"] == clean_one["timestamp"]
    assert "off-phrase/tail pick" in guarded["post_structure_guard_reason"]


def test_post_structure_guard_replaces_intro_block_with_first_big_structure_drop() -> None:
    intro_block = _candidate(7.868852459, rank=1, score=0.60, micro=0.80)
    intro_block.update(
        {
            "structure_role": "drop_candidate",
            "structure_components": {
                "clock_bar": 5,
                "phrase_prior": 0.66,
                "sustained_groove": 0.66,
                "low_to_high": 0.50,
            },
        }
    )
    first_big_block = _candidate(15.737704918, rank=4, score=0.58, micro=0.82)
    first_big_block.update(
        {
            "structure_role": "first_drop",
            "section_label": "first_drop",
            "structure_components": {
                "clock_bar": 9,
                "phrase_prior": 0.86,
                "sustained_groove": 0.76,
                "low_to_high": 0.43,
                "block_height": 0.61,
            },
        }
    )
    learned = {
        "candidate": intro_block,
        "selection_probability": 0.02,
        "chooser_score": 0.02,
        "selection_confidence": 0.15,
    }

    guarded = _apply_post_structure_selection_guard(
        learned,
        [intro_block, first_big_block],
        {"bpm": 122.0, "bar_zero_sec": 0.0},
    )

    assert guarded["candidate"]["structure_role"] == "first_drop"
    assert guarded["candidate"]["timestamp"] == 15.737704918
