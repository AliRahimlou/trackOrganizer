from __future__ import annotations

import json

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

    ok = app._apply_visual_first_result(
        item,
        {
            "ok": True,
            "version": 1,
            "marker": visual_marker["timestamp"],
            "raw_visual_time": 82.40135211267607,
            "selected_candidate": stale_phrase,
            "candidates": [visual_marker],
        },
    )

    assert ok is True
    assert item["ai_pick"] == visual_marker["timestamp"]
    assert item["selected_candidate"]["timestamp"] == visual_marker["timestamp"]
    assert item["top_10_candidates"][0]["timestamp"] == visual_marker["timestamp"]


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
