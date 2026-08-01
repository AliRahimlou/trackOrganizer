from __future__ import annotations

from drop_aligner.boom_profile import (
    boom_proof_front_edge_freshness,
    boom_front_edge_correction,
    earliest_dominant_boom_replacement,
    marker_boom_proof,
    score_candidate_against_profile,
)


def _boom_candidate(time_sec: float = 32.0, *, score: float = 0.74, clock_bar: int = 17) -> dict:
    return {
        "timestamp": time_sec,
        "score": score,
        "rank": 1,
        "bpm_clock": {"bpm": 150.0, "bar_sec": 1.6, "one_distance_ms": 0.0},
        "visual_components": {
            "feature_bar": clock_bar,
            "clock_bar": clock_bar,
            "post4_height": 0.74,
            "post8_height": 0.76,
            "pre4_height": 0.18,
            "jump4": 0.42,
            "jump8": 0.40,
            "post_bass8": 0.62,
            "post_drum8": 0.98,
            "post_drum_cont8": 0.96,
            "boom_section_darkness": 0.78,
            "boom_section_simultaneity": 0.82,
            "sustained": 1.0,
            "pre_space": 0.82,
            "local_reentry": True,
            "local_reentry_gap": 0.42,
            "phrase_prior": 0.94,
            "one_distance_ms": 0.0,
            "boom_body_section": True,
            "boom_section_end_bar": clock_bar + 7,
        },
    }


def test_boom_profile_accepts_sustained_drum_bass_body() -> None:
    scored = score_candidate_against_profile(_boom_candidate())

    assert scored["passes_profile"]
    assert scored["profile_score"] >= 0.60


def test_boom_profile_rejects_weak_non_simultaneous_body() -> None:
    candidate = _boom_candidate(score=0.25)
    candidate["visual_components"].update(
        {
            "post8_height": 0.25,
            "post_bass8": 0.04,
            "post_drum8": 0.18,
            "post_drum_cont8": 0.12,
            "boom_section_darkness": 0.25,
            "boom_section_simultaneity": 0.12,
            "sustained": 0.05,
        }
    )

    scored = score_candidate_against_profile(candidate)

    assert not scored["passes_profile"]
    assert any("simultaneity" in reason for reason in scored["reasons"])


def test_boom_profile_derives_sustain_when_visual_candidate_omits_field() -> None:
    candidate = _boom_candidate()
    candidate["visual_components"].pop("sustained", None)
    candidate["visual_components"].update(
        {
            "post4_height": 0.69,
            "post8_height": 0.67,
            "post_bass8": 0.46,
            "post_drum8": 1.0,
            "post_drum_cont4": 0.96,
            "post_drum_cont8": 0.97,
            "body_like": True,
        }
    )
    profile = {
        "thresholds": {
            "min_profile_score": 0.569,
            "min_darkness": 0.668,
            "min_post8_height": 0.557,
            "min_simultaneity": 0.637,
            "min_post_bass8": 0.278,
            "min_post_drum8": 0.947,
            "min_post_drum_cont8": 0.664,
            "min_contrast": 0.197,
            "min_sustain": 0.98,
            "max_one_distance_ms": 70.0,
        }
    }

    scored = score_candidate_against_profile(candidate, profile)

    assert scored["passes_profile"]
    assert scored["metrics"]["sustain"] >= 0.98


def test_boom_profile_missing_sustain_still_rejects_weak_one_shot() -> None:
    candidate = _boom_candidate(score=0.50)
    candidate["visual_components"].pop("sustained", None)
    candidate["visual_components"].update(
        {
            "post4_height": 0.62,
            "post8_height": 0.42,
            "post_bass8": 0.14,
            "post_drum8": 1.0,
            "post_drum_cont4": 0.30,
            "post_drum_cont8": 0.12,
            "body_like": False,
        }
    )
    profile = {
        "thresholds": {
            "min_profile_score": 0.569,
            "min_darkness": 0.668,
            "min_post8_height": 0.557,
            "min_simultaneity": 0.637,
            "min_post_bass8": 0.278,
            "min_post_drum8": 0.947,
            "min_post_drum_cont8": 0.664,
            "min_contrast": 0.197,
            "min_sustain": 0.98,
            "max_one_distance_ms": 70.0,
        }
    }

    scored = score_candidate_against_profile(candidate, profile)

    assert not scored["passes_profile"]
    assert scored["metrics"]["sustain"] < 0.50


def test_boom_profile_compensates_lower_bass_sustained_groove_body() -> None:
    candidate = _boom_candidate(score=0.60)
    candidate["visual_components"].update(
        {
            "post4_height": 0.56,
            "post8_height": 0.53,
            "post_bass8": 0.20,
            "post_drum8": 0.97,
            "post_drum_cont8": 0.74,
            "boom_section_darkness": 0.60,
            "boom_section_simultaneity": 0.61,
            "sustained": 1.0,
            "local_reentry_gap": 0.42,
            "pre_space": 0.90,
        }
    )
    profile = {
        "thresholds": {
            "min_profile_score": 0.569,
            "min_darkness": 0.668,
            "min_post8_height": 0.557,
            "min_simultaneity": 0.637,
            "min_post_bass8": 0.278,
            "min_post_drum8": 0.947,
            "min_post_drum_cont8": 0.664,
            "min_contrast": 0.197,
            "min_sustain": 0.98,
            "max_one_distance_ms": 70.0,
        }
    }

    scored = score_candidate_against_profile(candidate, profile)

    assert scored["passes_profile"]
    assert scored["compensated_reasons"]


def test_boom_profile_compensates_continuous_bass_body_with_lower_raw_drum_density() -> None:
    candidate = _boom_candidate(time_sec=14.790364556078842, score=0.675, clock_bar=9)
    candidate["visual_components"].update(
        {
            "post4_height": 0.674,
            "post8_height": 0.684,
            "pre4_height": 0.610,
            "jump4": 0.064,
            "jump8": 0.237,
            "post_bass8": 0.629,
            "post_drum8": 0.841,
            "post_drum_cont4": 0.854,
            "post_drum_cont8": 0.904,
            "boom_section_darkness": 0.684,
            "boom_section_simultaneity": 0.771,
            "phrase_prior": 0.86,
            "pre_space": 0.390,
        }
    )
    profile = {
        "thresholds": {
            "min_profile_score": 0.569,
            "min_darkness": 0.668,
            "min_post8_height": 0.557,
            "min_simultaneity": 0.637,
            "min_post_bass8": 0.278,
            "min_post_drum8": 0.947,
            "min_post_drum_cont8": 0.664,
            "min_contrast": 0.197,
            "min_sustain": 0.98,
            "max_one_distance_ms": 70.0,
        }
    }

    scored = score_candidate_against_profile(candidate, profile)

    assert scored["passes_profile"]
    assert scored["compensated_reasons"] == ["post_drum8 0.841 below 0.947"]


def test_boom_profile_compensation_rejects_thin_unsustained_body() -> None:
    candidate = _boom_candidate(score=0.56)
    candidate["visual_components"].update(
        {
            "post4_height": 0.54,
            "post8_height": 0.50,
            "post_bass8": 0.14,
            "post_drum8": 0.89,
            "post_drum_cont8": 0.42,
            "boom_section_darkness": 0.56,
            "boom_section_simultaneity": 0.48,
            "sustained": 0.35,
            "local_reentry_gap": 0.05,
            "pre_space": 0.20,
        }
    )
    profile = {
        "thresholds": {
            "min_profile_score": 0.569,
            "min_darkness": 0.668,
            "min_post8_height": 0.557,
            "min_simultaneity": 0.637,
            "min_post_bass8": 0.278,
            "min_post_drum8": 0.947,
            "min_post_drum_cont8": 0.664,
            "min_contrast": 0.197,
            "min_sustain": 0.98,
            "max_one_distance_ms": 70.0,
        }
    }

    scored = score_candidate_against_profile(candidate, profile)

    assert not scored["passes_profile"]


def test_marker_boom_proof_rejects_marker_late_inside_body() -> None:
    candidate = _boom_candidate(time_sec=40.0)

    proof = marker_boom_proof(41.2, [candidate])

    assert not proof["passes"]
    assert any("after boom front edge" in reason for reason in proof["reasons"])


def test_marker_boom_proof_accepts_front_edge_marker() -> None:
    candidate = _boom_candidate(time_sec=40.0)

    proof = marker_boom_proof(40.02, [candidate])

    assert proof["passes"]


def test_marker_boom_proof_cannot_erase_authoritative_grid_failure() -> None:
    candidate = _boom_candidate(32.144)
    candidate["bpm_clock"].update({"clock_zero_sec": 0.144, "on_one": True})
    candidate["visual_components"]["one_distance_ms"] = 0.0

    proof = marker_boom_proof(
        32.144,
        [candidate],
        selected_candidate=candidate,
        beatgrid={"bpm": 150.0, "bar_sec": 1.6, "bar_zero_sec": 0.0},
    )

    assert proof["passes"] is False
    assert any("authoritative beatgrid" in reason for reason in proof["reasons"])


def test_boom_proof_front_edge_freshness_rejects_stale_pass_payload() -> None:
    freshness = boom_proof_front_edge_freshness(
        {
            "passes": True,
            "marker_sec": 27.464,
            "nearest": {
                "edge_time": 27.042,
                "offset_sec": 0.422,
                "abs_offset_sec": 0.422,
            },
        }
    )

    assert freshness["fresh"] is False
    assert "above 0.080s" in freshness["reason"]


def test_boom_proof_front_edge_freshness_accepts_normalized_proof() -> None:
    freshness = boom_proof_front_edge_freshness(
        {
            "passes": True,
            "marker_sec": 27.464,
            "nearest": {
                "edge_time": 27.464,
                "offset_sec": 0.0,
                "abs_offset_sec": 0.0,
            },
        }
    )

    assert freshness["fresh"] is True


def test_marker_boom_proof_rejects_marker_before_body_front_edge() -> None:
    candidate = _boom_candidate(time_sec=40.0)

    proof = marker_boom_proof(39.95, [candidate])

    assert not proof["passes"]
    assert any("before boom front edge" in reason for reason in proof["reasons"])


def test_boom_front_edge_correction_rejects_body_edge_off_authoritative_one() -> None:
    candidate = _boom_candidate(time_sec=27.47367832167832, clock_bar=17)
    selected = _boom_candidate(time_sec=26.85278911564626, score=0.70, clock_bar=17)
    selected["bpm_clock"] = {
        "bpm": 143.0,
        "bar_sec": 240.0 / 143.0,
        "one_distance_ms": 620.8892060320608,
        "on_one": False,
    }
    selected["visual_components"]["one_distance_ms"] = 620.8892060320608

    correction = boom_front_edge_correction(
        26.85278911564626,
        [candidate],
        selected_candidate=selected,
        profile={
            "thresholds": {
                "min_profile_score": 0.569,
                "min_darkness": 0.668,
                "min_post8_height": 0.557,
                "min_simultaneity": 0.637,
                "min_post_bass8": 0.278,
                "min_post_drum8": 0.947,
                "min_post_drum_cont8": 0.664,
                "min_contrast": 0.197,
                "min_sustain": 0.98,
                "max_one_distance_ms": 70.0,
                "max_front_edge_offset_sec": 0.9,
            }
        },
        beatgrid={"bpm": 143.0, "bar_sec": 240.0 / 143.0, "bar_zero_sec": 0.0},
    )

    assert correction is None


def test_boom_front_edge_correction_moves_small_pre_edge_blank_marker() -> None:
    candidate = _boom_candidate(time_sec=40.0)
    selected = _boom_candidate(time_sec=39.95)
    selected["visual_components"]["one_distance_ms"] = 50.0

    correction = boom_front_edge_correction(39.95, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["marker"] == 40.0
    assert correction["direction"] == "before_body_front_edge"


def test_marker_boom_proof_accepts_selected_local_transition_inside_coarse_section() -> None:
    section = _boom_candidate(time_sec=20.571, score=0.68, clock_bar=13)
    section["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    section["visual_components"].update({"feature_bar": 13, "clock_bar": 13, "boom_section_end_bar": 22})
    selected = _boom_candidate(time_sec=27.428, score=0.61, clock_bar=17)
    selected["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    selected["visual_components"].update(
        {
            "feature_bar": 17,
            "clock_bar": 17,
            "phrase_prior": 0.94,
            "post8_height": 0.63,
            "post_bass8": 0.58,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.89,
            "boom_section_darkness": 0.63,
            "boom_section_simultaneity": 0.80,
            "jump4": 0.03,
            "jump8": 0.03,
            "local_reentry_gap": 0.03,
        }
    )

    proof = marker_boom_proof(27.428, [section], selected_candidate=selected)

    assert proof["passes"]
    assert proof["nearest"]["selected_marker_candidate"]


def test_marker_boom_proof_rejects_separate_earlier_dominant_body() -> None:
    early = _boom_candidate(time_sec=26.666, score=0.72, clock_bar=12)
    later = _boom_candidate(time_sec=94.545, score=0.76, clock_bar=40)
    selected = _boom_candidate(time_sec=96.084, score=0.74, clock_bar=41)
    for candidate in (early, later, selected):
        candidate["bpm_clock"] = {"bpm": 99.0, "bar_sec": 240.0 / 99.0, "one_distance_ms": 0.0}
    early["visual_components"].update({"feature_bar": 12, "clock_bar": 12, "boom_section_end_bar": 31})
    later["visual_components"].update({"feature_bar": 40, "clock_bar": 40, "boom_section_end_bar": 55})
    selected["visual_components"].update({"feature_bar": 41, "clock_bar": 41, "phrase_prior": 0.94})

    proof = marker_boom_proof(96.084, [early, later], selected_candidate=selected)

    assert not proof["passes"]
    assert any("earlier_dominant_boom_available" in reason for reason in proof["reasons"])


def test_marker_boom_proof_does_not_let_prephrase_intro_disqualify_phrase_drop() -> None:
    intro = _boom_candidate(time_sec=12.0, score=0.715, clock_bar=8)
    phrase_drop = _boom_candidate(time_sec=41.143, score=0.630, clock_bar=25)
    for candidate in (intro, phrase_drop):
        candidate["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    intro["visual_components"].update(
        {
            "feature_bar": 8,
            "clock_bar": 8,
            "phrase_prior": 0.18,
            "post4_height": 0.678,
            "post8_height": 0.717,
            "post_bass8": 0.607,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.905,
            "boom_section_darkness": 0.773,
            "boom_section_simultaneity": 0.816,
            "local_reentry": True,
            "local_reentry_gap": 0.356,
            "opening_body_start": True,
        }
    )
    phrase_drop["visual_components"].update(
        {
            "feature_bar": 25,
            "clock_bar": 25,
            "phrase_prior": 0.96,
            "post4_height": 0.618,
            "post8_height": 0.602,
            "post_bass8": 0.509,
            "post_drum8": 0.983,
            "post_drum_cont8": 0.786,
            "boom_section_darkness": 0.630,
            "boom_section_simultaneity": 0.746,
            "local_reentry": True,
            "local_reentry_gap": 0.247,
            "phrase_body_shift": True,
            "phrase_dropout_reentry": True,
        }
    )

    proof = marker_boom_proof(41.143, [intro], selected_candidate=phrase_drop)

    assert proof["passes"]
    assert not any("earlier_dominant_boom_available" in reason for reason in proof["reasons"])


def test_earliest_dominant_replacement_rejects_fast_off_one_repeated_body() -> None:
    early = _boom_candidate(time_sec=29.340, score=0.64, clock_bar=16)
    later = _boom_candidate(time_sec=104.931, score=0.68, clock_bar=56)
    selected = _boom_candidate(time_sec=105.949, score=0.68, clock_bar=57)
    for candidate in (early, later, selected):
        candidate["bpm_clock"] = {
            "bpm": 127.0,
            "bar_sec": 240.0 / 127.0,
            "one_distance_ms": 0.0,
        }
    early["visual_components"].update(
        {
            "feature_bar": 16,
            "clock_bar": 16,
            "boom_section_end_bar": 39,
            "post8_height": 0.59,
            "post_bass8": 0.31,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.86,
            "boom_section_darkness": 0.68,
            "boom_section_simultaneity": 0.69,
            "local_reentry_gap": 0.29,
            "pre_space": 0.69,
        }
    )
    later["visual_components"].update(
        {
            "feature_bar": 56,
            "clock_bar": 56,
            "boom_section_end_bar": 80,
            "post8_height": 0.59,
            "post_bass8": 0.37,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.85,
            "boom_section_darkness": 0.68,
            "boom_section_simultaneity": 0.69,
            "local_reentry_gap": 0.27,
            "pre_space": 0.63,
        }
    )
    selected["bpm_clock"]["one_distance_ms"] = 1018.0
    selected["visual_components"].update(
        {
            "feature_bar": 57,
            "clock_bar": 57,
            "one_distance_ms": 1018.0,
            "post8_height": 0.59,
            "post_bass8": 0.37,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.85,
            "boom_section_darkness": 0.68,
            "boom_section_simultaneity": 0.69,
            "local_reentry_gap": 0.27,
            "pre_space": 0.63,
        }
    )

    replacement = earliest_dominant_boom_replacement(
        105.949,
        [early, later],
        selected_candidate=selected,
        beatgrid={"bpm": 127.0, "bar_sec": 240.0 / 127.0, "bar_zero_sec": 0.0},
    )

    assert replacement is None


def test_marker_boom_proof_does_not_treat_file_start_as_earlier_dominant_drop() -> None:
    opening = _boom_candidate(time_sec=0.0, score=0.76, clock_bar=1)
    later = _boom_candidate(time_sec=32.0, score=0.74, clock_bar=17)

    proof = marker_boom_proof(32.0, [opening, later])

    assert proof["passes"]
    assert not any("earlier_dominant_boom_available" in reason for reason in proof["reasons"])


def test_marker_boom_proof_allows_track_relative_dominant_body() -> None:
    candidate = _boom_candidate(time_sec=32.0, score=0.58, clock_bar=13)
    candidate["visual_components"].update(
        {
            "post4_height": 0.53,
            "post8_height": 0.54,
            "post_bass8": 0.26,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.56,
            "boom_section_darkness": 0.61,
            "boom_section_simultaneity": 0.58,
            "sustained": 1.0,
            "local_reentry_gap": 0.16,
            "jump4": 0.13,
            "jump8": 0.16,
        }
    )

    proof = marker_boom_proof(32.0, [candidate])

    assert proof["passes"]
    assert proof["nearest_profile"].get("relative_boom_profile") or proof["nearest_profile"].get("compensated_reasons")


def test_boom_front_edge_correction_allows_slow_long_same_section() -> None:
    candidate = _boom_candidate(time_sec=120.0, score=0.72, clock_bar=41)
    candidate["bpm_clock"] = {"bpm": 82.0, "bar_sec": 240.0 / 82.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 41,
            "clock_bar": 41,
            "boom_section_end_bar": 64,
            "phrase_prior": 0.48,
        }
    )

    correction = boom_front_edge_correction(138.0, [candidate], selected_candidate=candidate)

    assert correction is not None
    assert correction["marker"] == 120.0
    assert correction["direction"] == "late_inside_body"


def test_boom_front_edge_correction_allows_midtempo_same_section_body() -> None:
    candidate = _boom_candidate(time_sec=78.624, score=0.70, clock_bar=33)
    candidate["bpm_clock"] = {"bpm": 100.0, "bar_sec": 2.4, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 33,
            "clock_bar": 33,
            "boom_section_start_bar": 34,
            "boom_section_end_bar": 44,
            "phrase_prior": 1.0,
            "post_bass8": 0.36,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.72,
            "local_reentry_gap": 0.36,
        }
    )
    selected = _boom_candidate(time_sec=90.622, score=0.72, clock_bar=38)
    selected["bpm_clock"] = {"bpm": 100.0, "bar_sec": 2.4, "one_distance_ms": 0.0}
    selected["visual_components"].update({"clock_bar": 38, "feature_bar": 38, "phrase_prior": 0.18})

    correction = boom_front_edge_correction(90.622, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["marker"] == 78.624
    assert correction["direction"] == "late_inside_body"


def test_boom_front_edge_correction_does_not_pull_fast_micro_transition_to_previous_grid_edge() -> None:
    candidate = _boom_candidate(time_sec=12.307692, score=0.70, clock_bar=9)
    candidate["bpm_clock"] = {"bpm": 156.0, "bar_sec": 240.0 / 156.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 9,
            "clock_bar": 9,
            "boom_section_end_bar": 25,
            "phrase_prior": 0.86,
            "post_bass8": 0.33,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.95,
            "local_reentry": True,
            "local_reentry_gap": 0.37,
        }
    )
    selected = _boom_candidate(time_sec=12.697746, score=0.77, clock_bar=9)
    selected["bpm_clock"] = {"bpm": 156.0, "bar_sec": 240.0 / 156.0, "one_distance_ms": 390.0}
    selected["visual_components"].update(
        {
            "feature_bar": 9,
            "clock_bar": 9,
            "phrase_prior": 0.86,
            "post_bass8": 0.33,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.95,
            "local_reentry": True,
            "local_reentry_gap": 0.31,
        }
    )

    correction = boom_front_edge_correction(12.697746, [candidate], selected_candidate=selected)

    assert correction is None


def test_boom_front_edge_correction_allows_fast_marker_late_inside_body() -> None:
    candidate = _boom_candidate(time_sec=44.000, score=0.74, clock_bar=25)
    candidate["bpm_clock"] = {"bpm": 150.0, "bar_sec": 1.6, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 25,
            "clock_bar": 25,
            "boom_section_start_bar": 25,
            "boom_section_end_bar": 40,
            "phrase_prior": 0.90,
            "post_bass8": 0.48,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.95,
            "local_reentry": True,
            "local_reentry_gap": 0.34,
        }
    )
    selected = _boom_candidate(time_sec=45.180, score=0.72, clock_bar=26)
    selected["bpm_clock"] = {"bpm": 150.0, "bar_sec": 1.6, "one_distance_ms": 180.0}
    selected["visual_components"].update(
        {
            "feature_bar": 26,
            "clock_bar": 26,
            "phrase_prior": 0.88,
            "post_bass8": 0.48,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.95,
            "local_reentry": True,
            "local_reentry_gap": 0.30,
        }
    )

    correction = boom_front_edge_correction(45.180, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["marker"] == 44.000
    assert correction["direction"] == "late_inside_body"


def test_boom_front_edge_correction_moves_marker_several_bars_before_strong_body() -> None:
    candidate = _boom_candidate(time_sec=55.0, score=0.76, clock_bar=33)
    candidate["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 33,
            "clock_bar": 33,
            "boom_section_start_bar": 33,
            "boom_section_end_bar": 48,
            "post4_height": 0.72,
            "post8_height": 0.70,
            "post_bass8": 0.50,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.82,
            "boom_section_darkness": 0.76,
            "boom_section_simultaneity": 0.78,
            "local_reentry": True,
            "local_reentry_gap": 0.22,
            "sustained": 1.0,
        }
    )
    selected = _boom_candidate(time_sec=50.8, score=0.52, clock_bar=31)
    selected["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 250.0}
    selected["visual_components"].update(
        {
            "clock_bar": 31,
            "one_distance_ms": 250.0,
            "post4_height": 0.28,
            "post8_height": 0.30,
            "post_bass8": 0.08,
            "post_drum8": 0.30,
            "post_drum_cont8": 0.12,
            "boom_section_darkness": 0.30,
            "boom_section_simultaneity": 0.14,
            "local_reentry": False,
            "local_reentry_gap": 0.02,
            "sustained": 0.10,
        }
    )

    correction = boom_front_edge_correction(50.8, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["direction"] == "before_body_front_edge"
    assert correction["marker"] == 55.0


def test_boom_front_edge_correction_keeps_visible_body_before_later_body() -> None:
    later = _boom_candidate(time_sec=55.0, score=0.76, clock_bar=33)
    later["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    later["visual_components"].update({"feature_bar": 33, "clock_bar": 33, "boom_section_start_bar": 33, "boom_section_end_bar": 48})
    selected = _boom_candidate(time_sec=44.571, score=0.70, clock_bar=27)
    selected["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    selected["visual_components"].update(
        {
            "feature_bar": 27,
            "clock_bar": 27,
            "post4_height": 0.68,
            "post8_height": 0.64,
            "post_bass8": 0.42,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.78,
            "boom_section_darkness": 0.68,
            "boom_section_simultaneity": 0.70,
            "local_reentry": True,
            "local_reentry_gap": 0.20,
            "one_distance_ms": 0.0,
        }
    )

    correction = boom_front_edge_correction(44.571, [later], selected_candidate=selected)

    assert correction is None


def test_boom_front_edge_correction_does_not_jump_far_before_unrelated_body() -> None:
    candidate = _boom_candidate(time_sec=72.0, score=0.76, clock_bar=45)
    candidate["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update({"feature_bar": 45, "clock_bar": 45, "boom_section_start_bar": 45, "boom_section_end_bar": 60})
    selected = _boom_candidate(time_sec=50.8, score=0.52, clock_bar=31)
    selected["bpm_clock"] = {"bpm": 140.0, "bar_sec": 240.0 / 140.0, "one_distance_ms": 250.0}
    selected["visual_components"].update({"clock_bar": 31, "one_distance_ms": 250.0})

    correction = boom_front_edge_correction(50.8, [candidate], selected_candidate=selected)

    assert correction is None


def test_boom_front_edge_correction_moves_fast_marker_deep_inside_long_body() -> None:
    candidate = _boom_candidate(time_sec=31.875, score=0.68, clock_bar=18)
    candidate["bpm_clock"] = {"bpm": 128.0, "bar_sec": 240.0 / 128.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 18,
            "clock_bar": 18,
            "boom_section_start_bar": 20,
            "boom_section_end_bar": 96,
            "boom_section_darkness": 0.82,
            "boom_section_simultaneity": 0.78,
            "phrase_prior": 0.18,
            "post4_height": 0.62,
            "post8_height": 0.66,
            "post_bass8": 0.56,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.83,
            "local_reentry": True,
            "local_reentry_gap": 0.09,
            "jump4": 0.08,
            "jump8": 0.08,
        }
    )
    selected = _boom_candidate(time_sec=75.966, score=0.76, clock_bar=42)
    selected["bpm_clock"] = {"bpm": 128.0, "bar_sec": 240.0 / 128.0, "one_distance_ms": 966.0}
    selected["visual_components"].update(
        {
            "feature_bar": 42,
            "clock_bar": 42,
            "phrase_prior": 0.86,
            "post4_height": 0.81,
            "post8_height": 0.81,
            "post_bass8": 0.66,
            "post_drum8": 1.0,
            "post_drum_cont8": 1.0,
            "local_reentry": True,
            "local_reentry_gap": 0.44,
        }
    )

    correction = boom_front_edge_correction(75.966, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["marker"] == 31.875
    assert correction["direction"] == "late_inside_body"


def test_boom_front_edge_correction_allows_lower_phrase_front_edge_in_same_section() -> None:
    candidate = _boom_candidate(time_sec=30.304780, score=0.80, clock_bar=16)
    candidate["bpm_clock"] = {"bpm": 123.0, "bar_sec": 240.0 / 123.0, "one_distance_ms": 0.0}
    candidate["visual_components"].update(
        {
            "feature_bar": 16,
            "clock_bar": 16,
            "boom_section_start_bar": 17,
            "boom_section_end_bar": 31,
            "phrase_prior": 0.18,
            "post4_height": 0.76,
            "post8_height": 0.75,
            "post_bass8": 0.50,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.94,
            "local_reentry": True,
            "local_reentry_gap": 0.61,
            "jump4": 0.64,
            "jump8": 0.69,
        }
    )
    selected = _boom_candidate(time_sec=31.280648, score=0.74, clock_bar=17)
    selected["bpm_clock"] = {"bpm": 123.0, "bar_sec": 240.0 / 123.0, "one_distance_ms": 975.0}
    selected["visual_components"].update(
        {
            "feature_bar": 17,
            "clock_bar": 17,
            "phrase_prior": 0.94,
            "post_bass8": 0.50,
            "post_drum8": 1.0,
            "post_drum_cont8": 1.0,
            "local_reentry": True,
            "local_reentry_gap": 0.31,
        }
    )

    correction = boom_front_edge_correction(31.280648, [candidate], selected_candidate=selected)

    assert correction is not None
    assert correction["marker"] == 30.304780


def test_earliest_dominant_boom_replacement_allows_midtempo_tracks() -> None:
    early = _boom_candidate(time_sec=26.666, score=0.72, clock_bar=12)
    late = _boom_candidate(time_sec=94.545, score=0.76, clock_bar=40)
    for candidate in (early, late):
        candidate["bpm_clock"] = {"bpm": 99.0, "bar_sec": 240.0 / 99.0, "one_distance_ms": 0.0}
    early["visual_components"].update({"feature_bar": 12, "clock_bar": 12, "boom_section_end_bar": 31})
    late["visual_components"].update({"feature_bar": 40, "clock_bar": 40, "boom_section_end_bar": 55})

    replacement = earliest_dominant_boom_replacement(
        96.084,
        [early, late],
        selected_candidate=late,
    )

    assert replacement is not None
    assert replacement["marker"] == 26.666
    assert replacement["direction"] == "earlier_dominant_boom"


def test_earliest_dominant_boom_replacement_allows_fast_tracks() -> None:
    early = _boom_candidate(time_sec=15.484, score=0.74, clock_bar=9)
    late = _boom_candidate(time_sec=77.483, score=0.76, clock_bar=41)
    for candidate in (early, late):
        candidate["bpm_clock"] = {"bpm": 124.0, "bar_sec": 240.0 / 124.0, "one_distance_ms": 0.0}
        candidate["visual_components"].update(
            {
                "post4_height": 0.74,
                "post8_height": 0.73,
                "post_bass8": 0.55,
                "post_drum8": 1.0,
                "post_drum_cont8": 0.92,
                "local_reentry": True,
                "local_reentry_gap": 0.36,
                "jump4": 0.34,
                "jump8": 0.35,
            }
        )
    early["visual_components"].update({"feature_bar": 9, "clock_bar": 9, "boom_section_end_bar": 24})
    late["visual_components"].update({"feature_bar": 41, "clock_bar": 41, "boom_section_end_bar": 56})

    replacement = earliest_dominant_boom_replacement(
        77.483,
        [early, late],
        selected_candidate=late,
    )

    assert replacement is not None
    assert replacement["marker"] == 15.484
    assert replacement["direction"] == "earlier_dominant_boom"
