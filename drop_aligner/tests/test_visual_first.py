from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drop_aligner import visual_first as visual_first_module
from drop_aligner import visual_drop_v2 as visual_drop_v2_module
from drop_aligner.exclusions import drums_stem_signal_stats, is_near_empty_drums_stem
from drop_aligner.visual_first import (
    _adjacent_phrase_edge_after_bad_snap_candidate,
    _absolute_final_proven_front_edge_audit_relief,
    _blank_waveform_marker_guard_candidate,
    _boom_candidate_auto_takeover,
    _boom_proof_with_grid_phase_neutralized,
    _clock_for_visual_edge,
    _earlier_phrase_edge_after_bad_snap_candidate,
    _filter_rejected_sections,
    _final_overbroad_reclaim_arbitration,
    _first_strongest_dark_section_candidate,
    _boom_proven_global_replacement_candidate,
    _fusion_audit_guard_candidate,
    _gui_mask_proof,
    _gui_proof_has_exact_sparse_front_edge,
    _late_reset_body_guard_candidate,
    _later_proven_boom_repair_for_failed_selected,
    _later_definitive_drop_guard_candidate,
    _opening_drop_profile,
    _proven_boom_phase_calibrated_clock,
    _proven_boom_grid_snap_time,
    _review_pattern_body_candidate_after_boom_repair,
    _stronger_real_drop_after_static_marker_candidate,
    _structure_section_guard_candidate,
    _selector_locked_beat_phase_probe_refinement,
    _selector_locked_marker_should_resist_boom_replacement,
    _track_zero_grid_phase_guard_candidate,
    _use_visual_drop_v2_result,
    _visual_drop_v2_has_earlier_comparable_drop,
    _zoomed_marker_time,
    audit_visual_selection,
    boom_body_section_candidates,
    select_first_visual_chunk,
    visual_chunk_candidates,
    visual_first_marker,
)
from drop_aligner.visual_drop_v2 import select_visual_drop_v2, visual_drop_v2_candidates
from drop_aligner.waveform import gui_boom_mask_strict_contract_issue


def _assert_visual_first_production_contract(result: dict) -> dict:
    assert result["ok"] is True
    selected = result["selected_candidate"]
    assert selected["selected_by"] not in visual_first_module._STALE_FINAL_VISUAL_SOURCES
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"].get("flag_codes", []) == []
    assert result["boom_proof"]["passes"] is True
    assert selected["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert result["gui_mask_proof"]["marker_signal_present"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["marker_signal_present"] is True
    return selected


def _assert_fresh_builder_gate_accepts(audio_path: str, result: dict) -> dict:
    import build_fresh_visual_first_library_set as fresh_build
    from drop_aligner.waveform import accept_gui_boom_mask_with_front_edge_proof

    selected_for_gui = dict(result["selected_candidate"])
    marker = float(result["marker"])
    boom_proof = fresh_build._boom_proof(result)
    detector_gui_proof = dict(fresh_build._gui_mask_proof(result))
    builder_gui = fresh_build.visual_gui_mask_proof(
        audio_path,
        marker,
        cache_dir=fresh_build.DEFAULT_WAVEFORM_CACHE_DIR,
    )
    selected_source = str(selected_for_gui.get("selected_by") or fresh_build._selected_by(result))
    repair_gui_lag_sec = 0.300 if selected_source.startswith("visual_final_contract_") else 0.040
    repair_gui_profile = 0.620 if selected_source.startswith("visual_final_contract_") else 0.560
    builder_gui = accept_gui_boom_mask_with_front_edge_proof(
        builder_gui,
        boom_proof,
        near_offset_sec=repair_gui_lag_sec,
        near_profile_score=repair_gui_profile,
    )
    builder_gui = fresh_build._accept_gui_contract_by_staccato_front_body_proof(
        builder_gui,
        boom_proof,
        selected_for_gui,
    )
    builder_gui = fresh_build._accept_gui_contract_by_sparse_groove_front_edge_proof(
        builder_gui,
        boom_proof,
        selected_for_gui,
    )
    builder_gui = fresh_build._accept_gui_contract_by_sustained_body_section(
        builder_gui,
        boom_proof,
        selected_for_gui,
    )
    builder_gui = fresh_build._accept_gui_contract_by_actual_body_proof(
        builder_gui,
        boom_proof,
        selected_for_gui,
    )
    builder_gui = fresh_build._accept_gui_contract_by_opening_body_start_proof(
        builder_gui,
        boom_proof,
        selected_for_gui,
    )
    if fresh_build._gui_proof_has_exact_sparse_front_edge(builder_gui, boom_proof, selected_for_gui):
        builder_gui = {**dict(builder_gui), "passes": True, "reasons": []}
    if (
        (not bool(builder_gui.get("passes")) or gui_boom_mask_strict_contract_issue(builder_gui))
        and bool(detector_gui_proof.get("passes"))
        and not gui_boom_mask_strict_contract_issue(detector_gui_proof)
        and detector_gui_proof.get("marker_signal_present") is True
    ):
        builder_gui = {**dict(detector_gui_proof), "builder_preserved_detector_gui_mask_proof": True}
    result_for_gate = dict(result)
    selected_for_gui["gui_mask_proof"] = dict(builder_gui)
    result_for_gate["selected_candidate"] = selected_for_gui
    result_for_gate["gui_mask_proof"] = dict(builder_gui)

    assert fresh_build._result_production_gate_reasons(result_for_gate) == []
    return builder_gui


def _feature_map(heights: list[float], *, bpm: float = 60.0) -> dict:
    bar_sec = 240.0 / float(bpm)
    bars = []
    for index, height in enumerate(heights):
        bars.append(
            {
                "bar": index + 1,
                "start_sec": index * bar_sec,
                "end_sec": (index + 1) * bar_sec,
                "aggregate_energy": height,
                "groove_energy": height,
                "bass_low_energy": height,
                "drum_density": height,
                "instrumental_energy": height,
                "timbre_novelty": 0.20,
            }
        )
    return {
        "ok": True,
        "bar_count": len(bars),
        "beatgrid": {
            "bpm": bpm,
            "beat_sec": 60.0 / float(bpm),
            "bar_sec": bar_sec,
            "bar_zero_sec": 0.0,
        },
        "bars": bars,
    }


def _component_feature_map(rows: list[dict], *, bpm: float = 60.0, with_roles: bool = False) -> dict:
    bar_sec = 240.0 / float(bpm)
    bars = []
    for index, row in enumerate(rows):
        bar = {
            "bar": index + 1,
            "start_sec": index * bar_sec,
            "end_sec": (index + 1) * bar_sec,
            "aggregate_energy": row.get("aggregate", row.get("height", 0.0)),
            "groove_energy": row.get("groove", row.get("height", 0.0)),
            "bass_low_energy": row.get("bass", row.get("height", 0.0)),
            "drum_density": row.get("drum", row.get("height", 0.0)),
            "instrumental_energy": row.get("inst", row.get("height", 0.0)),
            "vocal_presence": row.get("vocal", 0.0),
            "timbre_novelty": row.get("novelty", 0.20),
        }
        if with_roles:
            bar["roles"] = {"drums": {"rms_occupancy": row.get("drum_cont", row.get("drum", 0.0))}}
        bars.append(bar)
    return {
        "ok": True,
        "bar_count": len(bars),
        "beatgrid": {
            "bpm": bpm,
            "beat_sec": 60.0 / float(bpm),
            "bar_sec": bar_sec,
            "bar_zero_sec": 0.0,
        },
        "bars": bars,
    }


def _candidate(
    timestamp: float,
    clock_bar: int,
    *,
    score: float,
    phrase_prior: float = 0.18,
    post4: float = 0.60,
    post8: float = 0.60,
    bass: float = 0.40,
    drum: float = 1.0,
    pre_drum: float = 0.10,
    local_gap: float = 0.20,
    local_reentry: bool = True,
    phrase_body_shift: bool = False,
    opening_body_start: bool = False,
    prev_drum: float = 1.0,
    bpm: float = 144.0,
) -> dict:
    return {
        "timestamp": timestamp,
        "score": score,
        "confidence_score": score,
        "reason": "test candidate",
        "visual_components": {
            "clock_bar": clock_bar,
            "phrase_prior": phrase_prior,
            "post4_height": post4,
            "post8_height": post8,
            "post_bass8": bass,
            "post_drum8": drum,
            "pre4_height": 0.25,
            "pre8_height": 0.20,
            "pre_inst4": 0.20,
            "post_inst8": 0.20,
            "pre_vocal4": 0.20,
            "post_vocal8": 0.20,
            "drum_continuity": drum,
            "prev_drum_continuity": prev_drum,
            "pre_drum_cont4": pre_drum,
            "post_drum_cont4": drum,
            "post_drum_cont8": drum,
            "local_reentry": local_reentry,
            "local_reentry_gap": local_gap,
            "phrase_body_shift": phrase_body_shift,
            "opening_body_start": opening_body_start,
            "jump4": 0.15,
            "jump8": 0.15,
            "prev1_height": 0.20,
            "prev2_height": 0.20,
        },
        "bpm_clock": {"bpm": bpm},
    }


def _attach_passing_boom_proof(
    candidate: dict,
    *,
    profile_score: float = 0.700,
    body_score: float = 0.730,
    transition_score: float = 0.520,
) -> dict:
    visual = candidate["visual_components"]
    metrics = {
        "score": candidate.get("score", 0.0),
        "profile_score": profile_score,
        "body_score": body_score,
        "transition_score": transition_score,
        "darkness": max(candidate.get("score", 0.0), visual.get("post8_height", 0.0)),
        "post4_height": visual.get("post4_height", 0.0),
        "post8_height": visual.get("post8_height", 0.0),
        "post_bass8": visual.get("post_bass8", 0.0),
        "post_drum8": visual.get("post_drum8", 0.0),
        "post_drum_cont4": visual.get("post_drum_cont4", 0.0),
        "post_drum_cont8": visual.get("post_drum_cont8", 0.0),
        "simultaneity": 0.760,
        "contrast": max(visual.get("local_reentry_gap", 0.0), visual.get("jump4", 0.0), visual.get("jump8", 0.0)),
        "sustain": 1.0,
        "pre_space": 0.560,
        "phrase_prior": visual.get("phrase_prior", 0.0),
        "one_distance_ms": 0.0,
    }
    candidate["boom_proof"] = {
        "passes": True,
        "nearest": {
            "edge_time": candidate["timestamp"],
            "offset_sec": 0.0,
            "abs_offset_sec": 0.0,
            "contains_marker": True,
            "selected_marker_candidate": True,
        },
        "nearest_profile": {
            "passes_profile": True,
            "profile_score": profile_score,
            "metrics": metrics,
            "reasons": [],
        },
        "reasons": [],
    }
    visual["boom_proof_pass"] = True
    visual["boom_profile_score"] = profile_score
    return candidate


def test_exact_sparse_front_edge_helper_accepts_calibrated_on_one_body_edge() -> None:
    candidate = _candidate(
        41.40032258064516,
        20,
        score=0.840,
        phrase_prior=0.940,
        post4=0.760,
        post8=0.780,
        bass=0.820,
        drum=1.0,
        pre_drum=0.0,
        local_gap=0.420,
    )
    candidate["bpm_clock"] = {"on_one": True, "one_distance_ms": 0.0}
    boom_proof = _attach_passing_boom_proof(candidate, profile_score=0.760, body_score=0.780)["boom_proof"]
    gui_proof = {
        "passes": True,
        "reasons": [],
        "raw_reasons": ["marker_has_no_immediate_drop_body"],
        "marker_signal_present": True,
        "marker_relevant_mask": True,
        "marker_body_mask": False,
        "marker_immediate_body_present": False,
        "nearest_placeable_offset_sec": 0.0,
        "placeable_count": 1,
        "marker_post_relevant_occupancy_250ms": 0.100,
        "marker_post_relevant_occupancy_500ms": 0.120,
        "marker_post_rms_max_250ms": 0.180,
        "marker_post_rms_max_500ms": 0.200,
    }

    assert _gui_proof_has_exact_sparse_front_edge(gui_proof, boom_proof, candidate) is True


def _v2_candidate(
    timestamp: float,
    clock_bar: int,
    *,
    score: float,
    phrase_prior: float = 0.18,
    body: float = 0.62,
    post4: float = 0.62,
    post8: float = 0.62,
    pre4: float = 0.24,
    bass: float = 0.42,
    drum: float = 1.0,
    pre_drum: float = 0.10,
    post_drum4: float = 0.90,
    post_drum8: float = 0.90,
    transition: float = 0.24,
    inst: float = 0.25,
    vocal: float = 0.25,
    local_gap: float = 0.18,
) -> dict:
    return {
        "timestamp": timestamp,
        "snapped_sec": timestamp,
        "time_sec": timestamp,
        "score": score,
        "confidence_score": score,
        "selected_by": "visual_drop_v2_candidate",
        "reason": "test visual-v2 candidate",
        "visual_components": {
            "clock_bar": clock_bar,
            "phrase_prior": phrase_prior,
            "body_score": body,
            "post4_height": post4,
            "post8_height": post8,
            "pre4_height": pre4,
            "post_bass8": bass,
            "post_drum8": drum,
            "pre_drum_cont4": pre_drum,
            "post_drum_cont4": post_drum4,
            "post_drum_cont8": post_drum8,
            "transition": transition,
            "post_inst8": inst,
            "post_vocal8": vocal,
            "local_reentry_gap": local_gap,
        },
        "bpm_clock": {"bpm": 147.0},
    }


def test_visual_first_skips_smaller_buildup_when_next_block_is_bigger() -> None:
    heights = [0.16] * 16 + [0.50] * 8 + [0.72] * 24
    candidates = visual_chunk_candidates(_feature_map(heights))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 25


def test_boom_body_section_detector_sees_whole_track_before_selecting() -> None:
    rows = (
        [{"height": 0.10, "bass": 0.08, "drum": 0.12, "drum_cont": 0.10}] * 8
        + [{"height": 0.42, "bass": 0.30, "drum": 0.55, "drum_cont": 0.50}] * 8
        + [{"height": 0.18, "bass": 0.12, "drum": 0.18, "drum_cont": 0.10}] * 8
        + [{"height": 0.88, "bass": 0.82, "drum": 0.96, "drum_cont": 0.95}] * 12
    )

    candidates = boom_body_section_candidates(_component_feature_map(rows, with_roles=True))

    assert candidates
    assert candidates[0]["selected_by"] == "visual_boom_body_section"
    assert candidates[0]["visual_components"]["feature_bar"] == 25


def test_boom_body_section_detector_keeps_first_dominant_boom() -> None:
    rows = (
        [{"height": 0.10, "bass": 0.08, "drum": 0.12, "drum_cont": 0.10}] * 8
        + [{"height": 0.78, "bass": 0.70, "drum": 0.94, "drum_cont": 0.94}] * 8
        + [{"height": 0.20, "bass": 0.12, "drum": 0.20, "drum_cont": 0.12}] * 8
        + [{"height": 0.84, "bass": 0.76, "drum": 0.96, "drum_cont": 0.96}] * 12
    )

    candidates = boom_body_section_candidates(_component_feature_map(rows, with_roles=True))

    assert candidates
    assert candidates[0]["visual_components"]["feature_bar"] == 9


def test_boom_candidate_auto_takeover_accepts_dynamic_dark_body() -> None:
    candidate = _candidate(
        32.0,
        17,
        score=0.72,
        phrase_prior=0.86,
        post4=0.70,
        post8=0.68,
        bass=0.44,
        drum=1.0,
        pre_drum=0.10,
        local_gap=0.26,
        local_reentry=True,
    )
    candidate["selected_by"] = "visual_boom_body_section"
    candidate["visual_components"].update(
        {
            "post_drum_cont4": 0.92,
            "post_drum_cont8": 0.92,
            "boom_section_darkness": 0.72,
            "boom_section_simultaneity": 0.74,
            "sustained": 1.0,
            "one_distance_ms": 0.0,
            "boom_body_section": True,
        }
    )

    assert _boom_candidate_auto_takeover(candidate)


def test_boom_candidate_auto_takeover_rejects_thin_non_body() -> None:
    candidate = _candidate(
        32.0,
        17,
        score=0.58,
        post4=0.52,
        post8=0.49,
        bass=0.16,
        drum=0.82,
        pre_drum=0.50,
        local_gap=0.04,
        local_reentry=False,
    )
    candidate["selected_by"] = "visual_boom_body_section"
    candidate["visual_components"].update(
        {
            "post_drum_cont4": 0.32,
            "post_drum_cont8": 0.32,
            "boom_section_darkness": 0.50,
            "boom_section_simultaneity": 0.32,
            "sustained": 0.20,
            "one_distance_ms": 0.0,
            "boom_body_section": True,
        }
    )

    assert not _boom_candidate_auto_takeover(candidate)


def test_boom_global_replacement_promotes_proven_front_edge_after_bad_marker() -> None:
    selected = _candidate(
        48.0,
        17,
        score=0.52,
        phrase_prior=0.18,
        post4=0.44,
        post8=0.45,
        bass=0.18,
        drum=0.42,
        pre_drum=0.80,
        local_gap=0.02,
        bpm=80.0,
    )
    selected["selected_by"] = "visual_gui_first_fat_block"
    boom = _candidate(
        24.0,
        9,
        score=0.78,
        phrase_prior=0.86,
        post4=0.78,
        post8=0.80,
        bass=0.66,
        drum=0.98,
        pre_drum=0.08,
        local_gap=0.44,
        phrase_body_shift=True,
        bpm=80.0,
    )
    boom.update(
        {
            "snapped_sec": 24.0,
            "time_sec": 24.0,
            "visual_raw_chunk_time": 24.0,
            "selected_by": "visual_boom_body_section",
        }
    )
    boom["bpm_clock"] = {"bpm": 80.0, "bar_sec": 3.0, "one_distance_ms": 0.0, "on_one": True}
    boom["visual_components"].update(
        {
            "feature_bar": 9,
            "boom_body_section": True,
            "boom_section_darkness": 0.82,
            "boom_section_simultaneity": 0.84,
            "boom_section_end_bar": 16,
            "sustained": 1.0,
            "pre_space": 0.86,
            "one_distance_ms": 0.0,
        }
    )

    replacement = _boom_proven_global_replacement_candidate(
        selected,
        [selected, boom],
        [boom],
        beatgrid={"bpm": 80.0, "bar_sec": 3.0, "bar_zero_sec": 0.0},
    )

    assert replacement is not None
    assert replacement["selected_by"] == "visual_boom_global_front_edge"
    assert replacement["timestamp"] == pytest.approx(24.0)
    assert replacement["boom_proof"]["passes"]


def test_boom_global_replacement_rejects_fast_tempo_body_off_authoritative_one() -> None:
    selected = _candidate(
        4.0,
        2,
        score=0.50,
        phrase_prior=0.18,
        post4=0.50,
        post8=0.52,
        bass=0.18,
        drum=1.0,
        pre_drum=0.10,
        local_gap=0.49,
        bpm=134.0,
    )
    selected["selected_by"] = "visual_candidate_selector"
    selected["bpm_clock"] = {"bpm": 134.0, "bar_sec": 240.0 / 134.0, "one_distance_ms": 700.0, "on_one": False}
    boom = _candidate(
        89.107,
        50,
        score=0.73,
        phrase_prior=0.18,
        post4=0.64,
        post8=0.66,
        bass=0.50,
        drum=1.0,
        pre_drum=0.06,
        local_gap=0.51,
        bpm=134.0,
    )
    boom.update(
        {
            "snapped_sec": 89.107,
            "time_sec": 89.107,
            "visual_raw_chunk_time": 89.107,
            "selected_by": "visual_boom_body_section",
        }
    )
    boom["bpm_clock"] = {"bpm": 134.0, "bar_sec": 240.0 / 134.0, "one_distance_ms": 0.0, "on_one": True}
    boom["visual_components"].update(
        {
            "feature_bar": 50,
            "boom_body_section": True,
            "boom_section_darkness": 0.78,
            "boom_section_simultaneity": 0.74,
            "post_drum_cont4": 0.69,
            "post_drum_cont8": 0.77,
            "boom_section_end_bar": 128,
            "sustained": 1.0,
            "pre_space": 0.86,
            "one_distance_ms": 0.0,
        }
    )

    replacement = _boom_proven_global_replacement_candidate(
        selected,
        [selected, boom],
        [boom],
        beatgrid={"bpm": 134.0, "bar_sec": 240.0 / 134.0, "bar_zero_sec": 0.0},
    )

    assert replacement is None


def test_visual_first_shifts_opening_edge_to_stronger_phrase_body() -> None:
    candidates = [
        _candidate(15.0, 10, score=0.685, phrase_prior=0.18, post4=0.763, post8=0.760, bass=0.566),
        _candidate(
            26.667,
            17,
            score=0.729,
            phrase_prior=0.94,
            post4=0.800,
            post8=0.807,
            bass=0.636,
            pre_drum=1.0,
            local_gap=0.072,
            local_reentry=False,
            phrase_body_shift=True,
        ),
    ]

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 17
    assert selected["visual_edge_replaced_candidate"]["clock_bar"] == 10


def test_visual_first_uses_opening_body_start_after_low_drum_phrase_edge() -> None:
    candidates = [
        _candidate(
            26.667,
            17,
            score=0.637,
            phrase_prior=0.94,
            post4=0.647,
            post8=0.659,
            bass=0.442,
            drum=0.731,
            local_reentry=False,
            phrase_body_shift=True,
        ),
        _candidate(
            28.333,
            18,
            score=0.614,
            phrase_prior=0.18,
            post4=0.675,
            post8=0.671,
            bass=0.475,
            drum=1.0,
            local_reentry=False,
            opening_body_start=True,
            prev_drum=0.173,
        ),
        _candidate(
            31.667,
            20,
            score=0.613,
            phrase_prior=0.18,
            post4=0.689,
            post8=0.677,
            bass=0.483,
            drum=1.0,
            local_gap=0.169,
            local_reentry=True,
        ),
    ]

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 18


def test_selector_late_halfbeat_probe_preserves_visible_body_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_microalign(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("visible-body raw entries should not launch a late half-beat probe")

    monkeypatch.setattr(visual_first_module, "microalign_marker", fail_microalign)

    probe = _selector_locked_beat_phase_probe_refinement(
        "/tmp/drums_130_2A_5-ALTR.flac",
        121.846153846,
        121.845292168,
        {
            "clock_bar": 67,
            "local_reentry": True,
            "local_reentry_gap": 0.204731496,
            "phrase_prior": 0.48,
            "post4_height": 0.658551186,
            "post_bass8": 0.37128424,
            "post_drum_cont4": 1.0,
        },
        {
            "source_selected_by": "visual_gui_chunk",
            "accepted_by": "visible_body_gate",
            "score": 0.964444444,
        },
    )

    assert probe is None


def test_visual_first_shifts_off_phrase_edge_to_adjacent_phrase_edge() -> None:
    candidates = [
        _candidate(42.488, 26, score=0.703, phrase_prior=0.18, post4=0.652, post8=0.679, bass=0.546, pre_drum=0.07),
        _candidate(44.155, 27, score=0.714, phrase_prior=0.48, post4=0.685, post8=0.677, bass=0.544, pre_drum=0.20),
    ]

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 27
    assert selected["visual_edge_replaced_candidate"]["clock_bar"] == 26


def test_strongest_dark_section_guard_prefers_first_equal_dark_drop() -> None:
    earlier = _candidate(
        24.303,
        17,
        score=0.624,
        phrase_prior=0.86,
        post4=0.604,
        post8=0.606,
        bass=0.466,
        drum=1.0,
        pre_drum=0.270,
        local_gap=0.24,
        local_reentry=True,
        phrase_body_shift=True,
    )
    later = _candidate(
        48.608,
        33,
        score=0.625,
        phrase_prior=1.0,
        post4=0.605,
        post8=0.615,
        bass=0.471,
        drum=1.0,
        pre_drum=0.725,
        local_gap=0.38,
        local_reentry=True,
        phrase_body_shift=True,
    )
    weak_intro = _candidate(
        12.152,
        9,
        score=0.533,
        phrase_prior=0.86,
        post4=0.490,
        post8=0.491,
        bass=0.225,
        drum=0.99,
        pre_drum=0.010,
        local_gap=0.20,
    )

    guarded = _first_strongest_dark_section_candidate(later, [weak_intro, earlier, later])

    assert guarded is not None
    assert guarded["timestamp"] == earlier["timestamp"]
    assert guarded["selected_by"] == "visual_first_strongest_dark_section_guard"


def test_visual_first_selects_phrase_body_release_after_nearby_fill() -> None:
    early_fill = _candidate(
        32.0,
        21,
        score=0.691,
        phrase_prior=0.66,
        post4=0.758,
        post8=0.688,
        bass=0.546,
        drum=1.0,
        pre_drum=0.287,
        local_gap=0.200,
        local_reentry=True,
    )
    early_fill["visual_components"].update(
        {
            "post_inst8": 0.481,
            "pre_inst4": 0.941,
            "post_drum_cont4": 0.827,
            "post_drum_cont8": 0.913,
        }
    )
    leadin_body = _candidate(
        35.2,
        23,
        score=0.621,
        phrase_prior=0.48,
        post4=0.686,
        post8=0.651,
        bass=0.536,
        drum=1.0,
        pre_drum=0.595,
        local_gap=0.127,
        local_reentry=False,
    )
    leadin_body["visual_components"].update(
        {
            "post_inst8": 0.258,
            "pre_inst4": 0.964,
            "post_drum_cont4": 0.910,
            "post_drum_cont8": 0.955,
        }
    )
    phrase_body = _candidate(
        38.4,
        25,
        score=0.594,
        phrase_prior=0.96,
        post4=0.618,
        post8=0.627,
        bass=0.541,
        drum=1.0,
        pre_drum=0.827,
        local_gap=-0.034,
        local_reentry=False,
        phrase_body_shift=True,
    )
    phrase_body["visual_components"].update(
        {
            "post_inst8": 0.087,
            "pre_inst4": 0.870,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
        }
    )
    later_peak = _candidate(
        64.0,
        41,
        score=0.697,
        phrase_prior=0.86,
        post4=0.721,
        post8=0.735,
        bass=0.541,
        drum=1.0,
        pre_drum=1.0,
        local_gap=0.144,
        local_reentry=False,
        phrase_body_shift=True,
    )
    later_peak["visual_components"].update(
        {
            "post_inst8": 0.705,
            "pre_inst4": 0.103,
            "post_drum_cont4": 0.917,
            "post_drum_cont8": 0.958,
        }
    )

    selected = select_first_visual_chunk([early_fill, leadin_body, phrase_body, later_peak])

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 25


def test_visual_first_shifts_adjacent_body_leadin_to_phrase_edge_with_better_continuity() -> None:
    candidates = [
        _candidate(
            53.143,
            32,
            score=0.685,
            phrase_prior=0.18,
            post4=0.690,
            post8=0.706,
            bass=0.590,
            drum=1.0,
            pre_drum=0.597,
            local_gap=0.439,
        ),
        _candidate(
            54.857,
            33,
            score=0.762,
            phrase_prior=1.0,
            post4=0.711,
            post8=0.717,
            bass=0.603,
            drum=1.0,
            pre_drum=0.426,
            local_gap=0.341,
        ),
    ]
    candidates[0]["visual_components"].update({"post_drum_cont4": 0.782, "post_drum_cont8": 0.821})
    candidates[1]["visual_components"].update({"post_drum_cont4": 0.856, "post_drum_cont8": 0.861})

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33
    assert selected["visual_edge_replaced_candidate"]["clock_bar"] == 32


def test_bad_snap_guard_uses_adjacent_phrase_edge() -> None:
    selected = _candidate(
        53.143,
        32,
        score=0.685,
        phrase_prior=0.18,
        post4=0.690,
        post8=0.706,
        bass=0.590,
        drum=1.0,
        pre_drum=0.597,
        local_gap=0.439,
    )
    adjacent = _candidate(
        54.857,
        33,
        score=0.762,
        phrase_prior=1.0,
        post4=0.744,
        post8=0.717,
        bass=0.603,
        drum=0.98,
        pre_drum=0.426,
        local_gap=0.341,
    )
    selected["visual_components"].update({"post_drum_cont4": 0.782})
    adjacent["visual_components"].update({"post_drum_cont4": 0.856})

    guarded = _adjacent_phrase_edge_after_bad_snap_candidate(
        selected,
        [selected, adjacent],
        {"snap_offset_ms": 858.9, "micro_confidence": 0.472},
    )

    assert guarded is not None
    assert guarded["visual_components"]["clock_bar"] == 33
    assert guarded["selected_by"] == "visual_adjacent_phrase_edge_after_bad_snap"


def test_static_marker_guard_replaces_late_non_drop_with_earlier_real_edge() -> None:
    selected = _candidate(
        53.184,
        30,
        score=0.645,
        phrase_prior=0.18,
        post4=0.716,
        post8=0.713,
        bass=0.556,
        drum=1.0,
        pre_drum=0.908,
        local_gap=0.179,
    )
    earlier = _candidate(
        44.093,
        25,
        score=0.627,
        phrase_prior=0.96,
        post4=0.628,
        post8=0.653,
        bass=0.546,
        drum=0.99,
        pre_drum=0.789,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _stronger_real_drop_after_static_marker_candidate(
        selected,
        [selected, earlier],
        marker=53.170,
    )

    assert guarded is not None
    assert guarded["visual_components"]["clock_bar"] == 25
    assert guarded["selected_by"] == "visual_static_marker_stronger_real_drop_guard"


def test_static_marker_guard_replaces_early_intro_with_later_real_edge() -> None:
    selected = _candidate(
        18.857,
        12,
        score=0.597,
        phrase_prior=0.18,
        post4=0.570,
        post8=0.583,
        bass=0.333,
        drum=0.95,
        pre_drum=0.750,
        local_gap=0.183,
    )
    later = _candidate(
        27.429,
        17,
        score=0.633,
        phrase_prior=0.94,
        post4=0.636,
        post8=0.649,
        bass=0.439,
        drum=1.0,
        pre_drum=0.875,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _stronger_real_drop_after_static_marker_candidate(
        selected,
        [selected, later],
        marker=18.857,
    )

    assert guarded is not None
    assert guarded["visual_components"]["clock_bar"] == 17


def test_static_marker_guard_keeps_build_edge_when_microsnap_found_body() -> None:
    selected = _candidate(
        60.809,
        36,
        score=0.668,
        phrase_prior=0.18,
        post4=0.814,
        post8=0.729,
        bass=0.595,
        drum=1.0,
        pre_drum=0.750,
        local_gap=0.166,
    )
    earlier = _candidate(
        55.666,
        33,
        score=0.652,
        phrase_prior=1.0,
        post4=0.622,
        post8=0.723,
        bass=0.534,
        drum=1.0,
        pre_drum=0.649,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _stronger_real_drop_after_static_marker_candidate(
        selected,
        [selected, earlier],
        marker=68.574,
    )

    assert guarded is None


def test_static_marker_guard_does_not_promote_opening_prehit_before_body_start() -> None:
    selected = _candidate(
        28.333,
        18,
        score=0.614,
        phrase_prior=0.18,
        post4=0.675,
        post8=0.671,
        bass=0.475,
        drum=1.0,
        pre_drum=0.793,
        local_gap=0.134,
        opening_body_start=True,
    )
    prehit = _candidate(
        26.667,
        17,
        score=0.637,
        phrase_prior=0.94,
        post4=0.647,
        post8=0.659,
        bass=0.442,
        drum=1.0,
        pre_drum=1.0,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _stronger_real_drop_after_static_marker_candidate(
        selected,
        [selected, prehit],
        marker=28.333,
    )

    assert guarded is None


def test_static_marker_guard_preserves_later_definitive_body_peak() -> None:
    selected = _candidate(
        75.872,
        49,
        score=0.740,
        phrase_prior=0.18,
        post4=0.623,
        post8=0.630,
        bass=0.507,
        drum=1.0,
        pre_drum=0.559,
        local_gap=0.004,
    )
    selected["visual_components"].update(
        {
            "later_definitive_drop_guard": True,
            "frame_body": 0.966,
            "frame_post_body": 0.791,
            "frame_low_peak": 0.999,
            "post_drum_cont4": 0.955,
            "post_drum_cont8": 0.978,
        }
    )
    earlier_dense_block = _candidate(
        38.689,
        25,
        score=0.632,
        phrase_prior=0.96,
        post4=0.662,
        post8=0.687,
        bass=0.518,
        drum=1.0,
        pre_drum=1.0,
        local_gap=-0.017,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _stronger_real_drop_after_static_marker_candidate(
        selected,
        [selected, earlier_dense_block],
        marker=75.805,
    )

    assert guarded is None


def test_visual_chunk_candidates_include_phrase_dropout_to_body_on_one() -> None:
    rows = [
        {"height": 0.18, "bass": 0.18, "drum": 0.20, "inst": 0.25, "vocal": 0.30, "drum_cont": 0.10}
        for _ in range(32)
    ]
    for index in range(18, 23):
        rows[index].update({"height": 0.55, "bass": 0.35, "drum": 1.0, "inst": 0.68, "vocal": 0.78, "drum_cont": 0.55})
    rows[23].update({"height": 0.27, "bass": 0.16, "drum": 0.95, "inst": 0.20, "vocal": 0.55, "drum_cont": 0.12})
    for index in range(24, 32):
        rows[index].update({"height": 0.52, "bass": 0.48, "drum": 1.0, "inst": 0.16, "vocal": 0.42, "drum_cont": 0.70})
    rows[24].update({"height": 0.50, "bass": 0.42, "drum": 1.0, "inst": 0.12, "vocal": 0.40, "drum_cont": 0.56})

    candidates = visual_chunk_candidates(_component_feature_map(rows, bpm=60, with_roles=True))
    selected = select_first_visual_chunk(candidates)
    phrase_dropout = [
        row
        for row in candidates
        if row["visual_components"]["clock_bar"] == 25
        and row["visual_components"].get("phrase_dropout_reentry")
    ]

    assert phrase_dropout
    assert phrase_dropout[0]["visual_components"]["preserve_visual_on_one"] is True
    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 25


def test_visual_first_protects_phrase_dropout_body_from_later_section() -> None:
    build = _candidate(
        31.448,
        20,
        score=0.569,
        phrase_prior=0.18,
        post4=0.637,
        post8=0.560,
        bass=0.386,
        drum=1.0,
        pre_drum=0.145,
        local_gap=0.196,
    )
    build["visual_components"].update({"post_drum_cont4": 0.584, "post_drum_cont8": 0.495})
    phrase_dropout = _candidate(
        39.724137931034484,
        25,
        score=0.537,
        phrase_prior=0.96,
        post4=0.524,
        post8=0.553,
        bass=0.438,
        drum=1.0,
        pre_drum=0.556,
        local_gap=0.133,
        local_reentry=False,
        phrase_body_shift=False,
    )
    phrase_dropout["visual_components"].update(
        {
            "phrase_dropout_reentry": True,
            "preserve_visual_on_one": True,
            "prev1_height": 0.391,
            "prev2_height": 0.567,
            "post_inst8": 0.204,
            "pre_inst4": 0.598,
            "post_vocal8": 0.444,
            "pre_vocal4": 0.719,
            "post_drum_cont4": 0.502,
            "post_drum_cont8": 0.643,
        }
    )
    later_section = _candidate(
        76.138,
        47,
        score=0.645,
        phrase_prior=0.48,
        post4=0.653,
        post8=0.635,
        bass=0.366,
        drum=1.0,
        pre_drum=0.410,
        local_gap=0.131,
        local_reentry=False,
    )
    later_section["visual_components"].update({"post_drum_cont4": 0.917, "post_drum_cont8": 0.932})

    selected = select_first_visual_chunk([build, phrase_dropout, later_section])

    assert selected is not None
    assert selected["timestamp"] == pytest.approx(39.724137931034484)
    assert selected["visual_components"]["clock_bar"] == 25


def test_visual_first_filters_rejected_visual_section_before_selection() -> None:
    candidates = [
        _candidate(55.0, 33, score=0.70, phrase_prior=1.0),
        _candidate(82.0, 49, score=0.66, phrase_prior=0.96),
    ]

    filtered = _filter_rejected_sections(candidates, [{"timestamp": 55.0, "clock_bar": 33}])
    selected = select_first_visual_chunk(filtered)

    assert len(filtered) == 1
    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 49


def test_visual_first_does_not_hide_drop_from_backfilled_skip_rejection() -> None:
    candidates = [
        _candidate(44.113, 27, score=0.714, phrase_prior=0.48, post4=0.66, post8=0.68, bass=0.54),
        _candidate(70.834, 43, score=0.659, phrase_prior=0.48, post4=0.72, post8=0.74, bass=0.51),
    ]

    filtered = _filter_rejected_sections(
        candidates,
        [{"timestamp": 44.113, "raw_time": 44.155, "clock_bar": 27, "backfilled_from_latest_skip": True}],
    )
    selected = select_first_visual_chunk(filtered)
    audit = audit_visual_selection(
        candidates[0],
        filtered,
        rejected_sections=[{"timestamp": 44.113, "raw_time": 44.155, "clock_bar": 27, "backfilled_from_latest_skip": True}],
    )

    assert [row["visual_components"]["clock_bar"] for row in filtered] == [27, 43]
    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 27
    assert audit["status"] == "pass"


def test_visual_first_does_not_retreat_to_intro_after_later_rejection() -> None:
    candidates = [
        _candidate(15.0, 9, score=0.90, phrase_prior=1.0),
        _candidate(55.0, 33, score=0.70, phrase_prior=1.0),
        _candidate(82.0, 49, score=0.66, phrase_prior=0.96),
    ]

    filtered = _filter_rejected_sections(candidates, [{"timestamp": 55.0, "clock_bar": 33}])
    selected = select_first_visual_chunk(filtered)

    assert [row["visual_components"]["clock_bar"] for row in filtered] == [49]
    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 49


def test_zoomed_marker_prefers_clean_impact_when_visual_knee_is_early() -> None:
    marker = _zoomed_marker_time(
        82.40135211267607,
        {
            "microaligned_time": 82.60951537798219,
            "visual_onset_knee_time": 82.60951537798219,
            "visual_onset_knee_used": 1.0,
            "attack_start_time": 83.22142013988694,
            "zero_crossing_time": 83.22078521925202,
            "zero_crossing_quality": 0.845,
            "attack_cleanliness": 0.931,
            "attack_peak_strength": 0.688,
            "impact_boundary_confidence": 0.797,
            "denoised_impact_strength": 0.829,
        },
        {
            "post_drum8": 0.992,
            "post_bass8": 0.302,
            "pre_drum_cont4": 0.0,
        },
    )

    assert marker == pytest.approx(83.22078521925202)


def test_zoomed_marker_prefers_late_bang_over_busy_visual_edge() -> None:
    marker = _zoomed_marker_time(
        98.02816901408451,
        {
            "microaligned_time": 98.93215994378973,
            "attack_start_time": 99.14079939957203,
            "zero_crossing_time": 99.13798760818882,
            "zero_crossing_quality": 0.926,
            "attack_cleanliness": 0.566,
            "attack_peak_strength": 1.0,
            "impact_boundary_confidence": 0.468,
            "denoised_impact_strength": 1.0,
            "rms_rise_score": 0.972,
            "peak_rise_score": 1.0,
            "micro_confidence": 0.419,
        },
        {
            "pre4_height": 0.611,
            "jump4": 0.057,
            "post_drum8": 1.0,
            "post_bass8": 0.460,
            "pre_drum_cont4": 0.532,
        },
    )

    assert marker == pytest.approx(99.13798760818882)


def test_zoomed_marker_moves_early_knee_to_dark_sustained_body_entry() -> None:
    marker = _zoomed_marker_time(
        55.26315789473684,
        {
            "microaligned_time": 55.585765604487406,
            "visual_onset_knee_time": 55.585765604487406,
            "visual_onset_knee_quality": 0.935,
            "visual_onset_knee_used": 1.0,
            "attack_start_time": 56.258668098818475,
            "zero_crossing_time": 56.25596968612006,
            "zero_crossing_quality": 0.867,
            "micro_confidence": 0.419,
            "attack_cleanliness": 0.547,
            "attack_peak_strength": 1.0,
            "sustained_after_attack": 0.982,
            "denoised_impact_strength": 0.799,
            "impact_boundary_confidence": 0.433,
        },
        {
            "local_reentry": True,
            "local_reentry_gap": 0.390,
            "pre_drum_cont4": 0.071,
            "post_drum8": 1.0,
            "post_drum_cont4": 0.940,
            "post_bass8": 0.367,
        },
    )

    assert marker == pytest.approx(56.25596968612006)


def test_zoomed_marker_moves_later_body_peak_to_first_attack_boundary() -> None:
    marker = _zoomed_marker_time(
        75.872,
        {
            "microaligned_time": 75.87351927437642,
            "attack_start_time": 75.81902947845805,
            "zero_crossing_time": 75.81737414965986,
            "zero_crossing_quality": 0.943,
            "micro_confidence": 0.829,
            "attack_cleanliness": 0.355,
            "attack_peak_strength": 1.0,
            "sustained_after_attack": 1.0,
            "denoised_impact_strength": 0.720,
            "impact_boundary_confidence": 0.757,
        },
        {
            "later_definitive_drop_guard": True,
            "frame_body": 0.966,
            "frame_post_body": 0.791,
            "frame_low_peak": 0.999,
            "post_drum8": 1.0,
            "post_bass8": 0.507,
            "post_drum_cont4": 0.955,
        },
    )

    assert marker == pytest.approx(75.81737414965986)


def test_visual_first_prefers_instrumental_section_entry_over_later_body_peak() -> None:
    drum_only_body = _candidate(
        67.6056338028169,
        41,
        score=0.572,
        phrase_prior=0.86,
        post4=0.566,
        post8=0.579,
        bass=0.473,
        phrase_body_shift=True,
    )
    drum_only_body["visual_components"].update({"pre_inst4": 0.044, "post_inst8": 0.069})
    instrumental_entry = _candidate(
        94.64788732394366,
        57,
        score=0.616,
        phrase_prior=0.86,
        post4=0.630,
        post8=0.656,
        bass=0.426,
        drum=0.977,
        pre_drum=0.778,
        phrase_body_shift=True,
    )
    instrumental_entry["visual_components"].update({"pre_inst4": 0.226, "post_inst8": 0.684})
    later_body_peak = _candidate(
        98.02816901408451,
        59,
        score=0.625,
        phrase_prior=0.48,
        post4=0.668,
        post8=0.679,
        bass=0.460,
        phrase_body_shift=False,
    )
    later_body_peak["visual_components"].update({"pre_inst4": 0.581, "post_inst8": 0.642})

    selected = select_first_visual_chunk([drum_only_body, instrumental_entry, later_body_peak])

    assert selected is not None
    assert selected["timestamp"] == pytest.approx(94.64788732394366)
    assert "instrumental/bass section entry" in selected["reason"]


def test_visual_audit_flags_late_body_peak_after_section_entry() -> None:
    instrumental_entry = _candidate(
        94.64788732394366,
        57,
        score=0.616,
        phrase_prior=0.86,
        post4=0.630,
        post8=0.656,
        bass=0.426,
        drum=0.977,
        pre_drum=0.778,
        phrase_body_shift=True,
    )
    instrumental_entry["visual_components"].update({"pre_inst4": 0.226, "post_inst8": 0.684})
    later_body_peak = _candidate(
        98.02816901408451,
        59,
        score=0.625,
        phrase_prior=0.48,
        post4=0.668,
        post8=0.679,
        bass=0.460,
        phrase_body_shift=False,
    )
    later_body_peak["visual_components"].update({"pre_inst4": 0.581, "post_inst8": 0.642})

    audit = audit_visual_selection(later_body_peak, [instrumental_entry, later_body_peak])

    assert audit["status"] == "replace"
    assert "late_body_after_section_entry" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 57


def test_absolute_final_relief_clears_stale_late_body_flag_for_proven_front_edge() -> None:
    earlier_section_entry = _candidate(
        104.18793650793651,
        57,
        score=0.451,
        phrase_prior=0.86,
        post4=0.599,
        post8=0.610,
        bass=0.417,
        drum=1.0,
        pre_drum=0.502,
        local_gap=0.013,
        local_reentry=False,
        phrase_body_shift=True,
        bpm=129.0,
    )
    earlier_section_entry["visual_components"].update(
        {
            "jump4": 0.061,
            "jump8": 0.105,
            "post_inst8": 0.600,
            "post_drum_cont4": 0.540,
            "post_drum_cont8": 0.541,
        }
    )
    selected_front_edge = _candidate(
        141.3953488372093,
        77,
        score=0.602,
        phrase_prior=0.66,
        post4=0.596,
        post8=0.587,
        bass=0.467,
        drum=1.0,
        pre_drum=0.281,
        local_gap=0.066,
        local_reentry=False,
        phrase_body_shift=False,
        bpm=129.0,
    )
    selected_front_edge["selected_by"] = "visual_boom_off_one_front_edge_replacement"
    selected_front_edge["bpm_clock"] = {
        "bpm": 129.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 77,
    }
    selected_front_edge["visual_components"].update(
        {
            "jump4": 0.137,
            "jump8": 0.190,
            "post_inst8": 0.417,
            "post_drum_cont4": 0.629,
            "post_drum_cont8": 0.537,
        }
    )
    _attach_passing_boom_proof(
        selected_front_edge,
        profile_score=0.606,
        body_score=0.653,
        transition_score=0.455,
    )
    selected_front_edge["gui_mask_proof"] = {"passes": True, "accepted_by_boom_front_edge_proof": True}
    audit = {
        "status": "replace",
        "recommended_action": "replace",
        "flags": [{"code": "late_body_after_section_entry", "severity": "high"}],
        "flag_codes": ["late_body_after_section_entry"],
        "preferred_candidate": earlier_section_entry,
    }

    relieved = _absolute_final_proven_front_edge_audit_relief(
        audit,
        selected_front_edge,
        selected_front_edge["boom_proof"],
    )

    assert relieved["status"] == "pass"
    assert relieved["flag_codes"] == []
    assert relieved["cleared_flag_codes"] == ["late_body_after_section_entry"]


def test_absolute_final_relief_accepts_continuous_bass_body_with_lower_raw_drum_density() -> None:
    earlier_phrase_edge = _candidate(
        7.4071655328798185,
        5,
        score=0.673,
        phrase_prior=0.66,
        post4=0.610,
        post8=0.642,
        bass=0.534,
        drum=0.818,
        pre_drum=0.283,
        local_gap=0.186,
        local_reentry=True,
        phrase_body_shift=True,
        bpm=130.0,
    )
    earlier_phrase_edge["visual_components"].update(
        {
            "jump4": 0.328,
            "jump8": 0.360,
            "post_inst8": 0.669,
            "post_drum_cont4": 0.951,
            "post_drum_cont8": 0.903,
        }
    )
    selected_front_edge = _candidate(
        14.790364556078842,
        9,
        score=0.675,
        phrase_prior=0.86,
        post4=0.674,
        post8=0.684,
        bass=0.629,
        drum=0.841,
        pre_drum=0.951,
        local_gap=0.124,
        local_reentry=False,
        phrase_body_shift=True,
        bpm=130.0,
    )
    selected_front_edge["selected_by"] = "visual_gui_first_fat_block"
    selected_front_edge["bpm_clock"] = {
        "bpm": 130.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 9,
    }
    selected_front_edge["visual_components"].update(
        {
            "jump4": 0.064,
            "jump8": 0.237,
            "post_inst8": 0.670,
            "post_drum_cont4": 0.854,
            "post_drum_cont8": 0.904,
        }
    )
    _attach_passing_boom_proof(
        selected_front_edge,
        profile_score=0.681,
        body_score=0.733,
        transition_score=0.473,
    )
    selected_front_edge["gui_mask_proof"] = {"passes": True}
    audit = {
        "status": "replace",
        "recommended_action": "replace",
        "flags": [
            {"code": "ambiguous_visual_drop_evidence", "severity": "high"},
            {"code": "earlier_phrase_body_edge_available", "severity": "high"},
        ],
        "flag_codes": ["ambiguous_visual_drop_evidence", "earlier_phrase_body_edge_available"],
        "preferred_candidate": earlier_phrase_edge,
    }

    relieved = _absolute_final_proven_front_edge_audit_relief(
        audit,
        selected_front_edge,
        selected_front_edge["boom_proof"],
    )

    assert relieved["status"] == "pass"
    assert relieved["flag_codes"] == []


def test_select_first_visual_chunk_keeps_clear_first_drop_before_late_instrumental_entry() -> None:
    first_drop = _candidate(
        31.136,
        17,
        score=0.804,
        phrase_prior=0.94,
        post4=0.689,
        post8=0.689,
        bass=0.452,
        drum=1.0,
        pre_drum=0.155,
        local_gap=0.477,
        phrase_body_shift=True,
    )
    first_drop["visual_components"].update(
        {
            "jump4": 0.549,
            "jump8": 0.549,
            "post_drum_cont4": 0.917,
            "post_drum_cont8": 0.926,
        }
    )
    later_entry = _candidate(
        119.751,
        65,
        score=0.805,
        phrase_prior=1.0,
        post4=0.686,
        post8=0.685,
        bass=0.420,
        drum=1.0,
        pre_drum=0.155,
        local_gap=0.478,
        phrase_body_shift=True,
    )
    later_entry["visual_components"].update(
        {
            "jump4": 0.544,
            "jump8": 0.544,
            "post_drum_cont4": 0.914,
            "post_drum_cont8": 0.922,
            "pre_inst4": 0.150,
            "post_inst8": 0.650,
        }
    )

    selected = select_first_visual_chunk([first_drop, later_entry])

    assert selected is not None
    assert selected["timestamp"] == pytest.approx(31.136)


def test_select_first_visual_chunk_keeps_primary_bar_33_instrumental_entry() -> None:
    early_body = _candidate(
        27.429,
        17,
        score=0.704,
        phrase_prior=0.94,
        post4=0.604,
        post8=0.606,
        bass=0.457,
        drum=1.0,
        pre_drum=0.137,
        local_gap=0.290,
        phrase_body_shift=True,
    )
    early_body["visual_components"].update(
        {
            "jump4": 0.337,
            "jump8": 0.337,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
        }
    )
    primary_entry = _candidate(
        54.857,
        33,
        score=0.718,
        phrase_prior=1.0,
        post4=0.705,
        post8=0.710,
        bass=0.512,
        drum=1.0,
        pre_drum=0.900,
        local_gap=0.058,
        local_reentry=False,
        phrase_body_shift=True,
    )
    primary_entry["visual_components"].update(
        {
            "jump4": 0.068,
            "jump8": 0.068,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "pre_inst4": 0.100,
            "post_inst8": 0.650,
        }
    )

    selected = select_first_visual_chunk([early_body, primary_entry])

    assert selected is not None
    assert selected["timestamp"] == pytest.approx(54.857)
    assert "instrumental/bass section entry" in selected["reason"]


def test_visual_audit_flags_rejected_blue_section() -> None:
    rejected = _candidate(55.0, 33, score=0.70, phrase_prior=1.0)
    later = _candidate(82.0, 49, score=0.66, phrase_prior=0.96)

    audit = audit_visual_selection(rejected, [rejected, later], rejected_sections=[{"timestamp": 55.0, "clock_bar": 33}])

    assert audit["status"] == "replace"
    assert "selected_matches_rejected_section" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 49


def test_visual_audit_marks_early_intro_when_later_drop_is_stronger() -> None:
    intro = _candidate(13.714, 9, score=0.68, phrase_prior=0.18, post4=0.54, post8=0.52, bass=0.20)
    later = _candidate(
        55.714,
        33,
        score=0.77,
        phrase_prior=0.92,
        post4=0.72,
        post8=0.74,
        bass=0.58,
        drum=1.0,
        phrase_body_shift=True,
    )

    audit = audit_visual_selection(intro, [intro, later])

    assert audit["status"] == "replace"
    assert "intro_before_stronger_drop" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 33


def test_visual_audit_flags_off_one_bpm_grid_marker() -> None:
    selected = _candidate(31.545482993197282, 17, score=0.62, bpm=128.0)
    selected["bpm_clock"] = {
        "bpm": 128.0,
        "on_one": False,
        "one_distance_ms": 329.5,
        "beat_in_bar": 4,
        "nearest_one_bar": 18,
    }

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "review"
    assert "off_one_bpm_grid" in audit["flag_codes"]


def test_proven_boom_grid_snap_only_repairs_small_one_miss() -> None:
    proof = {"passes": True, "nearest": {"offset_sec": 0.0}}
    profile = {"thresholds": {"max_one_distance_ms": 90.0}}
    clock = {"nearest_one_time": 133.953488, "one_distance_ms": 45.9}
    wider_clock = {"nearest_one_time": 48.0, "one_distance_ms": 144.0}

    assert _proven_boom_grid_snap_time(133.907592, clock, proof, profile, 240.0 / 86.0) == 133.953488
    assert _proven_boom_grid_snap_time(48.144, wider_clock, proof, profile, 2.0) == 48.0

    far_clock = {"nearest_one_time": 134.200000, "one_distance_ms": 292.4}

    assert _proven_boom_grid_snap_time(133.907592, far_clock, proof, profile, 240.0 / 86.0) is None
    assert _proven_boom_grid_snap_time(133.907592, clock, {"passes": False}, profile, 240.0 / 86.0) is None


def test_proven_boom_body_cannot_rephase_its_own_grid() -> None:
    selected = _candidate(
        77.48352380952382,
        41,
        score=0.78,
        phrase_prior=0.86,
        post4=0.732,
        post8=0.729,
        bass=0.645,
        drum=1.0,
        local_gap=0.379,
        phrase_body_shift=True,
        bpm=124.0,
    )
    clock = {
        "bpm": 124.0,
        "bar_sec": 240.0 / 124.0,
        "clock_zero_sec": 0.0,
        "nearest_one_bar": 41,
        "one_distance_ms": 64.169,
        "on_one": False,
    }
    proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.80},
        "nearest_profile": {
            "profile_score": 0.765,
            "metrics": {
                "profile_score": 0.765,
                "post4_height": 0.732,
                "post8_height": 0.729,
                "post_bass8": 0.645,
                "post_drum8": 1.0,
                "post_drum_cont8": 1.0,
                "contrast": 0.379,
                "phrase_prior": 0.86,
            },
        },
    }

    calibrated = _proven_boom_phase_calibrated_clock(
        77.48352380952382,
        124.0,
        clock,
        selected,
        proof,
        {"thresholds": {"max_one_distance_ms": 90.0, "max_front_edge_offset_sec": 0.9}},
        240.0 / 124.0,
    )

    assert calibrated is None

    too_far = dict(proof)
    too_far["nearest"] = {"offset_sec": 0.95}
    assert (
        _proven_boom_phase_calibrated_clock(
            77.48352380952382,
            124.0,
            clock,
            selected,
            too_far,
            {"thresholds": {"max_one_distance_ms": 90.0, "max_front_edge_offset_sec": 0.9}},
            240.0 / 124.0,
        )
        is None
    )

    half_bar_clock = dict(clock)
    half_bar_clock["one_distance_ms"] = 828.0
    half_bar_calibrated = _proven_boom_phase_calibrated_clock(
        77.48352380952382,
        124.0,
        half_bar_clock,
        selected,
        proof,
        {"thresholds": {"max_one_distance_ms": 90.0, "max_front_edge_offset_sec": 0.9}},
        240.0 / 124.0,
    )

    assert half_bar_calibrated is None


def test_proven_boom_phase_calibration_rejects_weak_visual_body() -> None:
    selected = _candidate(48.144, 25, score=0.45, post4=0.48, post8=0.47, bass=0.12, drum=0.80)
    clock = {
        "bpm": 120.0,
        "bar_sec": 2.0,
        "clock_zero_sec": 0.0,
        "nearest_one_bar": 25,
        "one_distance_ms": 144.0,
        "on_one": False,
    }
    proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.0},
        "nearest_profile": {
            "profile_score": 0.52,
            "metrics": {"profile_score": 0.52, "post8_height": 0.47, "post_bass8": 0.12},
        },
    }

    assert (
        _proven_boom_phase_calibrated_clock(
            48.144,
            120.0,
            clock,
            selected,
            proof,
            {"thresholds": {"max_one_distance_ms": 90.0}},
            2.0,
        )
        is None
    )


def test_grid_phase_failure_remains_failed_despite_visual_boom_body() -> None:
    selected = _candidate(
        56.783472,
        30,
        score=0.76,
        phrase_prior=0.90,
        post4=0.72,
        post8=0.71,
        bass=0.58,
        drum=1.0,
        local_gap=0.34,
        phrase_body_shift=True,
        bpm=128.0,
    )
    selected["bpm_clock"] = {
        "bpm": 128.0,
        "bar_sec": 240.0 / 128.0,
        "clock_zero_sec": 0.0,
        "nearest_one_bar": 31,
        "one_distance_ms": 128.0,
        "on_one": False,
    }
    selected["visual_components"]["one_distance_ms"] = 128.0
    failed_proof = {"passes": False, "reasons": ["one_distance_ms 128.0 above 70.0"]}
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
            "max_front_edge_offset_sec": 0.9,
        }
    }

    proof = _boom_proof_with_grid_phase_neutralized(
        56.783472,
        selected,
        [selected],
        failed_proof,
        profile,
        {"bpm": 128.0, "bar_sec": 240.0 / 128.0, "bar_zero_sec": 0.0},
    )

    assert proof is None

    late_body_proof = _boom_proof_with_grid_phase_neutralized(
        58.200,
        selected,
        [selected],
        {"passes": False, "reasons": ["marker 1.417s after boom front edge"]},
        profile,
        {"bpm": 128.0, "bar_sec": 240.0 / 128.0, "bar_zero_sec": 0.0},
    )

    assert late_body_proof is None


def test_visual_audit_flags_missing_visual_waveform_components() -> None:
    selected = {
        "timestamp": 95.99854875283447,
        "score": 0.868,
        "confidence_score": 0.868,
        "selected_by": "visual_fusion_audit_guard",
        "reason": "fusion guard without full waveform components",
        "bpm_clock": {"bpm": 120.0, "on_one": True, "one_distance_ms": 1.5, "nearest_one_bar": 49},
    }

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "review"
    assert "missing_visual_waveform_components" in audit["flag_codes"]


def test_visual_audit_flags_late_dense_continuation_without_reset() -> None:
    selected = _candidate(
        119.17241379310344,
        73,
        score=0.635,
        phrase_prior=0.96,
        post4=0.641,
        post8=0.643,
        bass=0.403,
        drum=1.0,
        pre_drum=0.793,
        local_gap=0.180,
        phrase_body_shift=True,
    )
    selected["visual_components"].update({"jump4": 0.088, "jump8": 0.046, "post_drum_cont4": 0.986, "post_drum_cont8": 0.963})

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "review"
    assert "late_dense_continuation_without_reset" in audit["flag_codes"]


def test_visual_audit_flags_dense_continuation_without_front_edge() -> None:
    selected = _candidate(
        35.024662131519274,
        19,
        score=0.555,
        phrase_prior=0.48,
        post4=0.545,
        post8=0.556,
        bass=0.343,
        drum=1.0,
        pre_drum=0.552,
        local_gap=0.043,
    )
    selected["selected_by"] = "visual_gui_first_fat_block"
    selected["visual_components"].update({"jump4": 0.067, "jump8": 0.137, "post_drum_cont4": 0.911, "post_drum_cont8": 0.913})

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "review"
    assert "ambiguous_visual_drop_evidence" in audit["flag_codes"]


def test_visual_audit_flags_weak_sustained_visual_body() -> None:
    selected = _candidate(
        16.000771,
        9,
        score=0.628,
        phrase_prior=0.86,
        post4=0.509,
        post8=0.493,
        bass=0.327,
        drum=0.912,
        pre_drum=0.012,
        local_gap=0.329,
    )
    selected["selected_by"] = "visual_gui_first_fat_block"
    selected["visual_components"].update({"jump4": 0.424, "jump8": 0.322, "post_drum_cont4": 0.460, "post_drum_cont8": 0.558})

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "review"
    assert "ambiguous_visual_drop_evidence" in audit["flag_codes"]


def test_visual_audit_flags_earlier_real_drop_before_late_structure_marker() -> None:
    selected = {
        "timestamp": 88.64635967207396,
        "visual_raw_chunk_time": 88.61538461538461,
        "score": 0.709,
        "confidence_score": 0.709,
        "selected_by": "visual_structure_section_guard",
        "structure_clock_bar": 49,
        "structure_components": {
            "clock_bar": 49,
            "phrase_prior": 0.96,
            "post_4_energy": 0.674,
            "post_8_energy": 0.647,
            "pre_4_energy": 0.327,
            "pre_8_energy": 0.393,
            "post_bass": 0.652,
            "post_density": 0.999,
            "low_to_high": 0.470,
            "instrumental_reentry": 0.841,
        },
        "bpm_clock": {"bpm": 130.0, "on_one": True, "one_distance_ms": 31.0, "nearest_one_bar": 49},
    }
    earlier = _candidate(
        29.53846153846154,
        17,
        score=0.678,
        phrase_prior=0.94,
        post4=0.674,
        post8=0.664,
        bass=0.656,
        drum=1.0,
        pre_drum=0.513,
        local_gap=0.269,
        phrase_body_shift=True,
    )
    earlier["visual_components"].update({"jump4": 0.168, "jump8": 0.150, "post_drum_cont4": 1.0, "post_drum_cont8": 0.994})

    audit = audit_visual_selection(selected, [selected, earlier])

    assert audit["status"] == "replace"
    assert audit["selected"]["clock_bar"] == 49
    assert "earlier_real_drop_available_before_late_marker" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 17


def test_visual_audit_flags_earlier_real_drop_before_bar33_marker() -> None:
    earlier = _candidate(
        27.428571428571427,
        17,
        score=0.684,
        phrase_prior=0.94,
        post4=0.667,
        post8=0.635,
        bass=0.409,
        drum=1.0,
        pre_drum=0.537,
        local_gap=0.374,
    )
    earlier["visual_components"].update({"jump4": 0.374, "jump8": 0.320, "post_drum_cont4": 1.0, "post_drum_cont8": 0.940})
    selected = _candidate(
        54.857142857142854,
        33,
        score=0.669,
        phrase_prior=1.0,
        post4=0.663,
        post8=0.668,
        bass=0.501,
        drum=1.0,
        pre_drum=0.787,
        local_gap=0.326,
    )
    selected["visual_components"].update({"jump4": 0.200, "jump8": 0.200, "post_drum_cont4": 0.916, "post_drum_cont8": 0.906})

    audit = audit_visual_selection(selected, [selected, earlier])

    assert audit["status"] == "replace"
    assert "earlier_real_drop_available_before_late_marker" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 17


def test_visual_audit_flags_later_definitive_drop_after_early_intro_block() -> None:
    early = _candidate(
        15.0,
        9,
        score=0.686,
        phrase_prior=0.86,
        post4=0.641,
        post8=0.639,
        bass=0.624,
        drum=1.0,
        pre_drum=0.257,
        local_gap=0.235,
        phrase_body_shift=True,
    )
    later = _candidate(
        195.0,
        105,
        score=0.796,
        phrase_prior=0.86,
        post4=0.735,
        post8=0.721,
        bass=0.600,
        drum=1.0,
        pre_drum=0.125,
        local_gap=0.439,
        phrase_body_shift=True,
    )
    early["visual_components"].update({"jump4": 0.228, "jump8": 0.228, "post_drum_cont4": 0.996, "post_drum_cont8": 0.987})
    later["visual_components"].update({"jump4": 0.464, "jump8": 0.424, "post_drum_cont4": 1.0, "post_drum_cont8": 0.943})

    audit = audit_visual_selection(early, [early, later])

    assert audit["status"] == "replace"
    assert "later_definitive_drop_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 105


def test_visual_audit_flags_comparable_later_main_phrase_after_bar_nine_intro() -> None:
    early = _candidate(
        14.998820861678004,
        9,
        score=0.725,
        phrase_prior=0.86,
        post4=0.620,
        post8=0.628,
        bass=0.374,
        drum=1.0,
        pre_drum=0.004,
        local_gap=0.326,
        phrase_body_shift=True,
    )
    later = _candidate(
        135.0,
        73,
        score=0.724,
        phrase_prior=0.96,
        post4=0.620,
        post8=0.619,
        bass=0.376,
        drum=1.0,
        pre_drum=0.009,
        local_gap=0.312,
        phrase_body_shift=True,
    )
    early["visual_components"].update({"jump4": 0.366, "jump8": 0.385, "post_drum_cont4": 1.0, "post_drum_cont8": 1.0})
    later["visual_components"].update({"jump4": 0.372, "jump8": 0.362, "post_drum_cont4": 1.0, "post_drum_cont8": 1.0})

    audit = audit_visual_selection(early, [early, later])

    assert audit["status"] == "replace"
    assert "later_definitive_drop_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 73


def test_visual_audit_flags_later_body_when_early_block_has_weak_sustain() -> None:
    early = _candidate(
        16.70173221926452,
        9,
        score=0.716,
        phrase_prior=0.86,
        post4=0.636,
        post8=0.631,
        bass=0.454,
        drum=1.0,
        pre_drum=0.049,
        local_gap=0.438,
    )
    early["visual_components"].update({"jump4": 0.407, "jump8": 0.407, "post_drum_cont4": 0.353, "post_drum_cont8": 0.330})
    later = _candidate(
        141.91304347826087,
        69,
        score=0.634,
        phrase_prior=0.66,
        post4=0.655,
        post8=0.636,
        bass=0.473,
        drum=1.0,
        pre_drum=0.612,
        local_gap=0.486,
    )
    later["visual_components"].update({"jump4": 0.137, "jump8": 0.137, "post_drum_cont4": 0.863, "post_drum_cont8": 0.912})

    audit = audit_visual_selection(early, [early, later])

    assert audit["status"] == "replace"
    assert "later_definitive_drop_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 69


def test_visual_audit_flags_later_clean_body_after_weak_early_continuation() -> None:
    early = _candidate(
        33.22977149834293,
        19,
        score=0.608,
        phrase_prior=0.48,
        post4=0.624,
        post8=0.608,
        bass=0.373,
        drum=1.0,
        pre_drum=0.879,
        local_gap=0.242,
    )
    early["visual_components"].update({"jump4": 0.150, "jump8": 0.150, "post_drum_cont4": 0.800, "post_drum_cont8": 0.766})
    later = _candidate(
        121.84615384615385,
        67,
        score=0.627,
        phrase_prior=0.48,
        post4=0.659,
        post8=0.622,
        bass=0.371,
        drum=1.0,
        pre_drum=0.871,
        local_gap=0.205,
    )
    later["visual_components"].update({"jump4": 0.205, "jump8": 0.205, "post_drum_cont4": 1.0, "post_drum_cont8": 0.950})

    audit = audit_visual_selection(early, [early, later])

    assert audit["status"] == "replace"
    assert "later_definitive_drop_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 67


def test_visual_audit_replaces_late_marker_when_earlier_definitive_body_exists() -> None:
    earlier = _candidate(
        28.444444444444443,
        17,
        score=0.710,
        phrase_prior=0.94,
        post4=0.599,
        post8=0.604,
        bass=0.429,
        drum=1.0,
        pre_drum=0.005,
        local_gap=0.524,
        phrase_body_shift=True,
    )
    earlier["visual_components"].update({"jump4": 0.445, "jump8": 0.445, "post_drum_cont4": 0.875, "post_drum_cont8": 0.877})
    late = _candidate(
        113.77510204081632,
        65,
        score=0.871,
        phrase_prior=1.0,
        post4=0.651,
        post8=0.645,
        bass=0.360,
        drum=1.0,
        pre_drum=0.230,
        local_gap=0.479,
    )
    late["selected_by"] = "visual_fusion_audit_guard"
    late["bpm_clock"] = {"bpm": 135.0, "on_one": True, "one_distance_ms": 2.7, "nearest_one_bar": 65}
    late["visual_components"].update({"jump4": 0.216, "jump8": 0.216, "post_drum_cont4": 0.923, "post_drum_cont8": 0.911})

    audit = audit_visual_selection(late, [late, earlier])

    assert audit["status"] == "replace"
    assert "earlier_definitive_drop_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 17


def test_visual_audit_replaces_late_marker_with_earlier_phrase_body_edge() -> None:
    earlier = _candidate(
        28.444444444444443,
        17,
        score=0.690,
        phrase_prior=0.94,
        post4=0.640,
        post8=0.640,
        bass=0.460,
        drum=1.0,
        pre_drum=0.080,
        local_gap=0.330,
        phrase_body_shift=True,
    )
    earlier["visual_components"].update({"jump4": 0.300, "jump8": 0.310, "post_drum_cont4": 0.920, "post_drum_cont8": 0.900})
    selected = _candidate(
        43.000,
        25,
        score=0.720,
        phrase_prior=0.66,
        post4=0.650,
        post8=0.648,
        bass=0.470,
        drum=1.0,
        pre_drum=0.430,
        local_gap=0.140,
        local_reentry=False,
    )
    selected["visual_components"].update({"jump4": 0.090, "jump8": 0.080, "post_drum_cont4": 0.970, "post_drum_cont8": 0.940})

    audit = audit_visual_selection(selected, [selected, earlier])

    assert audit["status"] == "replace"
    assert "earlier_phrase_body_edge_available" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 17


def test_visual_audit_rejects_thin_early_peak_as_phrase_body_edge() -> None:
    selected = _candidate(
        27.428571428571427,
        17,
        score=0.649,
        phrase_prior=0.94,
        post4=0.616,
        post8=0.613,
        bass=0.424,
        drum=1.0,
        pre_drum=0.194,
        local_gap=0.303,
        local_reentry=True,
        phrase_body_shift=True,
    )
    selected["visual_components"].update(
        {
            "jump4": 0.144,
            "jump8": 0.150,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "preserve_visual_on_one": True,
        }
    )
    thin_early_peak = _candidate(
        13.728,
        9,
        score=0.553,
        phrase_prior=0.86,
        post4=0.454,
        post8=0.463,
        bass=0.130,
        drum=0.987,
        pre_drum=0.009,
        local_gap=0.120,
        local_reentry=True,
        phrase_body_shift=True,
        opening_body_start=True,
    )
    thin_early_peak["selected_by"] = "visual_body_peak_candidate"
    thin_early_peak["visual_components"].update(
        {
            "jump4": 0.201,
            "jump8": 0.279,
            "post_drum_cont4": 0.154,
            "post_drum_cont8": 0.174,
            "frame_body": 0.826,
            "frame_post_body": 0.278,
            "frame_low_peak": 0.918,
        }
    )

    audit = audit_visual_selection(selected, [selected, thin_early_peak])

    assert "earlier_phrase_body_edge_available" not in audit["flag_codes"]
    assert audit["preferred_candidate"] is None


def test_visual_audit_relieves_opening_body_artifact_against_proven_phrase_boom() -> None:
    selected = _candidate(
        55.401,
        25,
        score=0.657,
        phrase_prior=0.96,
        post4=0.644,
        post8=0.638,
        bass=0.514,
        drum=1.0,
        pre_drum=0.427,
        local_gap=0.194,
        local_reentry=True,
        phrase_body_shift=True,
    )
    selected["visual_components"].update({"jump4": 0.190, "jump8": 0.097, "post_drum_cont4": 0.865, "post_drum_cont8": 0.860})
    _attach_passing_boom_proof(selected, profile_score=0.692, body_score=0.725, transition_score=0.509)
    opening = _candidate(
        9.312,
        5,
        score=0.727,
        phrase_prior=0.66,
        post4=0.640,
        post8=0.639,
        bass=0.521,
        drum=1.0,
        pre_drum=0.222,
        local_gap=0.193,
        local_reentry=True,
        phrase_body_shift=True,
        opening_body_start=True,
    )
    opening["selected_by"] = "visual_body_peak_candidate"
    opening["visual_components"].update({"jump4": 0.302, "jump8": 0.300, "post_drum_cont4": 0.850, "post_drum_cont8": 0.854})

    audit = audit_visual_selection(selected, [selected, opening])

    assert audit["status"] == "pass"
    assert audit["flag_codes"] == []
    assert audit["cleared_flag_codes"] == ["earlier_phrase_body_edge_available"]


def test_visual_audit_relieves_ambiguous_flag_for_proven_visible_boom() -> None:
    selected = _candidate(
        13.717,
        9,
        score=0.605,
        phrase_prior=0.86,
        post4=0.577,
        post8=0.512,
        bass=0.330,
        drum=0.957,
        pre_drum=0.080,
        local_gap=0.236,
        local_reentry=True,
    )
    selected["selected_by"] = "visual_gui_first_fat_block"
    selected["visual_components"].update({"post_drum_cont4": 0.566, "post_drum_cont8": 0.566})
    _attach_passing_boom_proof(selected, profile_score=0.605, body_score=0.604, transition_score=0.522)

    audit = audit_visual_selection(selected, [selected])

    assert audit["status"] == "pass"
    assert audit["flag_codes"] == []
    assert audit["cleared_flag_codes"] == ["ambiguous_visual_drop_evidence"]


def test_visual_audit_keeps_stronger_earlier_real_drop_before_proven_late_boom() -> None:
    selected = _candidate(
        97.198,
        33,
        score=0.692,
        phrase_prior=1.0,
        post4=0.694,
        post8=0.710,
        bass=0.449,
        drum=1.0,
        pre_drum=1.0,
        local_gap=0.113,
        local_reentry=False,
        phrase_body_shift=True,
    )
    selected["visual_components"].update({"jump4": 0.142, "jump8": 0.129, "post_drum_cont4": 0.992, "post_drum_cont8": 0.996})
    _attach_passing_boom_proof(selected, profile_score=0.704, body_score=0.752, transition_score=0.467)
    earlier = _candidate(
        30.380,
        11,
        score=0.804,
        phrase_prior=0.48,
        post4=0.664,
        post8=0.692,
        bass=0.437,
        drum=1.0,
        pre_drum=0.0,
        local_gap=0.557,
        local_reentry=True,
        phrase_body_shift=False,
    )
    earlier["visual_components"].update({"jump4": 0.564, "jump8": 0.610, "post_drum_cont4": 0.889, "post_drum_cont8": 0.939})

    audit = audit_visual_selection(selected, [selected, earlier])

    assert audit["status"] == "replace"
    assert "earlier_real_drop_available_before_late_marker" in audit["flag_codes"]
    assert audit["preferred_candidate"]["clock_bar"] == 11


def test_later_definitive_guard_replaces_early_fake_fat_block_in_later_drop_window() -> None:
    early = _candidate(
        15.0,
        9,
        score=0.686,
        phrase_prior=0.86,
        post4=0.641,
        post8=0.639,
        bass=0.624,
        drum=1.0,
        pre_drum=0.257,
        local_gap=0.235,
        phrase_body_shift=True,
    )
    early["visual_components"].update({"jump4": 0.228, "post_drum_cont4": 0.996, "post_drum_cont8": 0.987})
    later_drop = _candidate(
        121.875,
        65,
        score=0.774,
        phrase_prior=0.66,
        post4=0.733,
        post8=0.720,
        bass=0.601,
        drum=1.0,
        pre_drum=0.082,
        local_gap=0.479,
    )
    later_drop["visual_components"].update({"jump4": 0.402, "post_drum_cont4": 1.0, "post_drum_cont8": 0.940})
    definitive = _candidate(
        195.0,
        105,
        score=0.796,
        phrase_prior=0.86,
        post4=0.735,
        post8=0.721,
        bass=0.600,
        drum=1.0,
        pre_drum=0.125,
        local_gap=0.439,
        phrase_body_shift=True,
    )
    definitive["visual_components"].update({"jump4": 0.464, "post_drum_cont4": 1.0, "post_drum_cont8": 0.940})

    guarded = _later_definitive_drop_guard_candidate(
        early,
        [early, later_drop, definitive],
        {"duration_sec": 315.0},
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(121.875)
    assert guarded["selected_by"] == "visual_later_definitive_drop_guard"
    assert guarded["visual_components"]["later_definitive_drop_guard"] is True


def test_later_definitive_guard_replaces_early_block_with_later_texture_release_body_peak() -> None:
    early = _candidate(
        19.742315789473682,
        13,
        score=0.727,
        phrase_prior=0.80,
        post4=0.595,
        post8=0.618,
        bass=0.495,
        drum=1.0,
        pre_drum=0.148,
        local_gap=0.333,
    )
    early["visual_components"].update({"jump4": 0.364, "jump8": 0.461, "post_drum_cont4": 1.0, "post_drum_cont8": 1.0})
    later_body_peak = _candidate(
        75.872,
        49,
        score=0.740,
        phrase_prior=0.18,
        post4=0.623,
        post8=0.630,
        bass=0.507,
        drum=1.0,
        pre_drum=0.559,
        local_gap=0.004,
        local_reentry=False,
    )
    later_body_peak["selected_by"] = "visual_body_peak_candidate"
    later_body_peak["visual_components"].update(
        {
            "frame_body": 0.966,
            "frame_post_body": 0.791,
            "frame_low_peak": 0.999,
            "post_drum_cont4": 0.955,
            "post_drum_cont8": 0.955,
            "pre_inst4": 0.831,
            "post_inst8": 0.248,
        }
    )

    guarded = _later_definitive_drop_guard_candidate(
        early,
        [early, later_body_peak],
        {"duration_sec": 120.0},
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(75.872)
    assert guarded["selected_by"] == "visual_later_definitive_drop_guard"


def test_later_definitive_guard_keeps_strong_first_drop_body() -> None:
    first_drop = _candidate(
        30.476190476190474,
        17,
        score=0.801,
        phrase_prior=0.94,
        post4=0.698,
        post8=0.684,
        bass=0.673,
        drum=1.0,
        pre_drum=0.183,
        local_gap=0.487,
        local_reentry=True,
        phrase_body_shift=True,
    )
    first_drop["visual_components"].update(
        {
            "jump4": 0.507,
            "jump8": 0.412,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 0.943,
            "preserve_visual_on_one": True,
        }
    )
    later_body_peak = _candidate(
        104.832,
        56,
        score=0.716,
        phrase_prior=0.18,
        post4=0.740,
        post8=0.734,
        bass=0.692,
        drum=1.0,
        pre_drum=1.0,
        local_gap=-0.002,
        local_reentry=False,
    )
    later_body_peak["selected_by"] = "visual_body_peak_candidate"
    later_body_peak["visual_components"].update(
        {
            "frame_body": 0.984,
            "frame_post_body": 0.801,
            "frame_low_peak": 1.0,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "pre_inst4": 0.753,
            "post_inst8": 0.367,
        }
    )

    guarded = _later_definitive_drop_guard_candidate(
        first_drop,
        [first_drop, later_body_peak],
        {"duration_sec": 190.0},
    )

    assert guarded is None


def test_visual_audit_accepts_later_definitive_guard_over_weaker_early_edge() -> None:
    early = _candidate(
        31.875,
        17,
        score=0.656,
        phrase_prior=0.94,
        post4=0.657,
        post8=0.655,
        bass=0.678,
        drum=1.0,
        pre_drum=0.687,
        local_gap=0.139,
        phrase_body_shift=True,
    )
    early["visual_components"].update({"jump4": 0.101, "post_drum_cont4": 1.0, "post_drum_cont8": 1.0})
    selected = _candidate(
        126.511,
        69,
        score=0.681,
        phrase_prior=0.66,
        post4=0.713,
        post8=0.702,
        bass=0.683,
        drum=1.0,
        pre_drum=0.227,
        local_gap=0.143,
    )
    selected["visual_components"].update(
        {
            "jump4": 0.160,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "later_definitive_drop_guard": True,
        }
    )

    audit = audit_visual_selection(selected, [early, selected])

    assert audit["status"] == "pass"
    assert "earlier_phrase_body_edge_available" not in audit["flag_codes"]


def test_later_definitive_guard_replaces_same_height_intro_with_clean_reset_drop() -> None:
    early = _candidate(
        38.095,
        21,
        score=0.633,
        phrase_prior=0.66,
        post4=0.677,
        post8=0.679,
        bass=0.465,
        drum=1.0,
        pre_drum=0.634,
        local_gap=0.081,
    )
    early["visual_components"].update({"jump4": 0.081, "post_drum_cont4": 0.624, "post_drum_cont8": 0.616})
    later = _candidate(
        121.905,
        65,
        score=0.750,
        phrase_prior=1.0,
        post4=0.680,
        post8=0.679,
        bass=0.467,
        drum=1.0,
        pre_drum=0.008,
        local_gap=0.379,
        phrase_body_shift=True,
    )
    later["visual_components"].update({"jump4": 0.456, "post_drum_cont4": 0.622, "post_drum_cont8": 0.630})

    guarded = _later_definitive_drop_guard_candidate(
        early,
        [early, later],
        {"duration_sec": 271.24},
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(121.905)
    assert guarded["visual_components"]["clock_bar"] == 65


def test_later_definitive_guard_does_not_jump_to_late_song_peak() -> None:
    early = _candidate(
        13.714,
        9,
        score=0.579,
        phrase_prior=0.86,
        post4=0.600,
        post8=0.590,
        bass=0.450,
        drum=1.0,
        pre_drum=0.300,
        local_gap=0.250,
    )
    early["visual_components"].update({"jump4": 0.210, "post_drum_cont4": 0.900, "post_drum_cont8": 0.900})
    late_peak = _candidate(
        150.857,
        89,
        score=0.718,
        phrase_prior=0.86,
        post4=0.760,
        post8=0.750,
        bass=0.620,
        drum=1.0,
        pre_drum=0.020,
        local_gap=0.460,
        phrase_body_shift=True,
    )
    late_peak["visual_components"].update({"jump4": 0.480, "post_drum_cont4": 1.0, "post_drum_cont8": 1.0})

    guarded = _later_definitive_drop_guard_candidate(
        early,
        [early, late_peak],
        {"duration_sec": 220.0},
    )

    assert guarded is None


def test_later_definitive_guard_does_not_replace_bar_33_drop_with_repeat() -> None:
    selected = _candidate(
        60.952,
        33,
        score=0.674,
        phrase_prior=1.0,
        post4=0.645,
        post8=0.620,
        bass=0.628,
        drum=1.0,
        pre_drum=0.349,
        local_gap=0.286,
        phrase_body_shift=True,
    )
    selected["visual_components"].update({"jump4": 0.188, "post_drum_cont4": 0.950, "post_drum_cont8": 0.941})
    repeat = _candidate(
        129.522,
        69,
        score=0.779,
        phrase_prior=0.66,
        post4=0.696,
        post8=0.683,
        bass=0.623,
        drum=1.0,
        pre_drum=0.008,
        local_gap=0.463,
        local_reentry=True,
    )
    repeat["visual_components"].update({"jump4": 0.465, "post_drum_cont4": 0.929, "post_drum_cont8": 0.922})

    guarded = _later_definitive_drop_guard_candidate(
        selected,
        [selected, repeat],
        {"duration_sec": 230.0},
    )

    assert guarded is None


def test_later_definitive_guard_keeps_real_early_drop_without_stronger_later_body() -> None:
    early = _candidate(
        31.875,
        17,
        score=0.760,
        phrase_prior=1.0,
        post4=0.720,
        post8=0.715,
        bass=0.560,
        drum=1.0,
        pre_drum=0.090,
        local_gap=0.360,
        phrase_body_shift=True,
    )
    early["visual_components"].update({"jump4": 0.360, "post_drum_cont4": 0.970, "post_drum_cont8": 0.960})
    later = _candidate(
        95.625,
        49,
        score=0.805,
        phrase_prior=1.0,
        post4=0.742,
        post8=0.738,
        bass=0.575,
        drum=1.0,
        pre_drum=0.260,
        local_gap=0.320,
        phrase_body_shift=True,
    )
    later["visual_components"].update({"jump4": 0.390, "post_drum_cont4": 0.980, "post_drum_cont8": 0.970})

    guarded = _later_definitive_drop_guard_candidate(
        early,
        [early, later],
        {"duration_sec": 240.0},
    )

    assert guarded is None


def test_visual_first_skips_early_jump_when_later_block_is_clearly_taller() -> None:
    heights = [0.34] * 16 + [0.56] * 8 + [0.61] * 8 + [0.70] * 24
    candidates = visual_chunk_candidates(_feature_map(heights))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33


def test_visual_first_keeps_first_sustained_fat_block() -> None:
    heights = [0.16] * 16 + [0.70] * 24 + [0.74] * 16
    candidates = visual_chunk_candidates(_feature_map(heights))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 17


def test_visual_first_keeps_strong_phrase_reentry_before_denser_bass_repeat() -> None:
    first_drop = _candidate(
        36.57142857142857,
        17,
        score=0.6692887525825705,
        phrase_prior=0.94,
        post4=0.669433114445183,
        post8=0.7099251896609506,
        bass=0.6013641085172138,
        drum=1.0,
        pre_drum=0.41751269035533,
        local_gap=0.3194941460029845,
        bpm=105.0,
    )
    first_drop["visual_components"].update(
        {
            "drum_continuity": 0.883248730964467,
            "post_drum_cont4": 0.6091370558375635,
            "post_drum_cont8": 0.6909898477157361,
            "jump4": 0.10780561957851675,
            "jump8": 0.13919982549929721,
        }
    )
    first_drop["bpm_clock"].update({"on_one": True, "one_distance_ms": 0.0})
    later_repeat = _candidate(
        45.71428571428571,
        21,
        score=0.6973723910612749,
        phrase_prior=0.66,
        post4=0.7504172648767181,
        post8=0.7682450523679232,
        bass=0.7199118677504258,
        drum=1.0,
        pre_drum=0.6091370558375635,
        local_gap=0.1944286025388846,
        bpm=105.0,
    )
    later_repeat["visual_components"].update(
        {
            "drum_continuity": 0.766497461928934,
            "post_drum_cont4": 0.7728426395939086,
            "post_drum_cont8": 0.815989847715736,
            "jump4": 0.08098415043153506,
            "jump8": 0.15271474771199856,
        }
    )
    later_repeat["bpm_clock"].update({"on_one": True, "one_distance_ms": 0.0})

    selected = select_first_visual_chunk([first_drop, later_repeat])

    assert selected is not None
    assert selected["timestamp"] == pytest.approx(first_drop["timestamp"])


def test_visual_first_skips_buildup_to_later_bass_body_drop() -> None:
    rows = [{"height": 0.16} for _ in range(40)]
    for idx in range(8, 16):
        rows[idx] = {"aggregate": 0.55, "groove": 0.56, "bass": 0.13, "drum": 1.0, "inst": 0.84, "vocal": 0.58}
    for idx in range(24, 31):
        rows[idx] = {"aggregate": 0.58, "groove": 0.62, "bass": 0.40, "drum": 1.0, "inst": 0.55, "vocal": 0.50}
    rows[31] = {"height": 0.18, "drum": 0.8, "vocal": 0.7}
    for idx in range(32, 40):
        rows[idx] = {"aggregate": 0.62, "groove": 0.70, "bass": 0.52, "drum": 1.0, "inst": 0.26, "vocal": 0.24}
    candidates = visual_chunk_candidates(_component_feature_map(rows))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33


def test_visual_first_keeps_real_early_drop_when_later_block_is_not_bass_upgrade() -> None:
    rows = [{"height": 0.12} for _ in range(40)]
    for idx in range(8, 16):
        rows[idx] = {"aggregate": 0.48, "groove": 0.55, "bass": 0.33, "drum": 1.0, "inst": 0.30, "vocal": 0.67}
    rows[31] = {"height": 0.20, "drum": 0.65, "vocal": 0.60}
    for idx in range(32, 40):
        rows[idx] = {"aggregate": 0.60, "groove": 0.65, "bass": 0.36, "drum": 1.0, "inst": 0.56, "vocal": 0.22}
    candidates = visual_chunk_candidates(_component_feature_map(rows))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 9


def test_visual_first_skips_intro_phrase_when_later_drum_drop_has_cleaner_reset() -> None:
    rows = [{"height": 0.16, "drum_cont": 0.10} for _ in range(44)]
    for idx in range(12, 15):
        rows[idx] = {"height": 0.25, "drum": 0.70, "drum_cont": 0.20}
    rows[15] = {"aggregate": 0.64, "groove": 0.66, "bass": 0.54, "drum": 1.0, "inst": 0.49, "vocal": 0.50, "drum_cont": 0.74}
    for idx in range(16, 24):
        rows[idx] = {"aggregate": 0.70, "groove": 0.69, "bass": 0.56, "drum": 1.0, "inst": 0.48, "vocal": 0.50, "drum_cont": 0.80}
    for idx in range(28, 32):
        rows[idx] = {"height": 0.20, "drum": 0.55, "drum_cont": 0.18}
    for idx in range(32, 40):
        rows[idx] = {"aggregate": 0.68, "groove": 0.69, "bass": 0.49, "drum": 1.0, "inst": 0.47, "vocal": 0.47, "drum_cont": 0.93}
    candidates = visual_chunk_candidates(_component_feature_map(rows, with_roles=True))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33


def test_visual_first_keeps_first_clean_dense_reentry_over_nearby_body() -> None:
    rows = [{"height": 0.07, "drum": 0.0, "drum_cont": 0.0} for _ in range(24)]
    rows[8] = {
        "aggregate": 0.56,
        "groove": 0.58,
        "bass": 0.42,
        "drum": 1.0,
        "inst": 0.27,
        "vocal": 0.36,
        "drum_cont": 0.84,
    }
    for idx in range(9, 16):
        rows[idx] = {
            "aggregate": 0.59,
            "groove": 0.59,
            "bass": 0.43,
            "drum": 1.0,
            "inst": 0.28,
            "vocal": 0.36,
            "drum_cont": 1.0,
        }
    candidates = visual_chunk_candidates(_component_feature_map(rows, bpm=140.0, with_roles=True))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 9


def test_visual_first_keeps_first_body_transition_inside_opening_cluster() -> None:
    candidates = [
        _candidate(
            27.386,
            17,
            score=0.673,
            phrase_prior=0.94,
            post4=0.458,
            post8=0.559,
            bass=0.381,
            drum=0.875,
            pre_drum=0.0,
            local_gap=0.421,
            local_reentry=False,
        ),
        _candidate(
            29.042,
            18,
            score=0.729,
            phrase_prior=0.18,
            post4=0.605,
            post8=0.630,
            bass=0.425,
            drum=1.0,
            pre_drum=0.0,
            local_gap=0.530,
        ),
        _candidate(
            30.697,
            19,
            score=0.746,
            phrase_prior=0.48,
            post4=0.652,
            post8=0.652,
            bass=0.466,
            drum=1.0,
            pre_drum=0.106,
            local_gap=0.334,
        ),
    ]

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 18


def test_visual_first_keeps_phrase_start_body_over_adjacent_bump() -> None:
    first = _candidate(
        26.301369863013697,
        17,
        score=0.678,
        phrase_prior=0.94,
        post4=0.647,
        post8=0.654,
        bass=0.392,
        drum=1.0,
        pre_drum=0.391,
        local_gap=0.283,
        phrase_body_shift=True,
    )
    first["visual_components"].update(
        {
            "jump4": 0.155,
            "jump8": 0.217,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
        }
    )
    adjacent_bump = _candidate(
        27.945205479452053,
        18,
        score=0.631,
        phrase_prior=0.18,
        post4=0.660,
        post8=0.658,
        bass=0.434,
        drum=1.0,
        pre_drum=0.557,
        local_gap=0.211,
    )
    adjacent_bump["visual_components"].update(
        {
            "jump4": 0.130,
            "jump8": 0.155,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 0.978,
        }
    )

    selected = select_first_visual_chunk([first, adjacent_bump])

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 17


def test_visual_first_replaces_late_bad_snap_with_earlier_phrase_body_edge() -> None:
    earlier = _candidate(
        26.301369863013697,
        17,
        score=0.678,
        phrase_prior=0.94,
        post4=0.647,
        post8=0.654,
        bass=0.392,
        drum=1.0,
        pre_drum=0.391,
        local_gap=0.283,
        phrase_body_shift=True,
    )
    earlier["visual_components"].update(
        {
            "jump4": 0.155,
            "jump8": 0.217,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
        }
    )
    selected = _candidate(
        27.945205479452053,
        18,
        score=0.631,
        phrase_prior=0.18,
        post4=0.660,
        post8=0.658,
        bass=0.434,
        drum=1.0,
        pre_drum=0.557,
        local_gap=0.211,
    )
    selected["visual_components"].update(
        {
            "jump4": 0.158,
            "jump8": 0.196,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 0.978,
        }
    )

    guarded = _earlier_phrase_edge_after_bad_snap_candidate(
        selected,
        [selected, earlier],
        {"snap_offset_ms": 401.38, "micro_confidence": 0.437},
    )

    assert guarded is not None
    assert guarded["visual_components"]["clock_bar"] == 17
    assert guarded["selected_by"] == "visual_earlier_phrase_edge_after_bad_snap"


def test_visual_first_skips_mid_phrase_buildup_for_primary_drop_body() -> None:
    buildup = _candidate(
        39.45205479452055,
        25,
        score=0.586,
        phrase_prior=0.96,
        post4=0.535,
        post8=0.561,
        bass=0.378,
        drum=1.0,
        pre_drum=0.132,
        local_reentry=False,
        local_gap=0.079,
    )
    buildup["visual_components"].update(
        {
            "jump4": 0.126,
            "jump8": 0.117,
            "post_drum_cont4": 0.569,
            "post_drum_cont8": 0.526,
            "post_vocal8": 0.589,
            "post_inst8": 0.425,
        }
    )
    primary_drop = _candidate(
        52.602739726027394,
        33,
        score=0.618,
        phrase_prior=1.0,
        post4=0.609,
        post8=0.620,
        bass=0.410,
        drum=0.995,
        pre_drum=0.484,
        local_reentry=False,
        local_gap=-0.063,
        phrase_body_shift=True,
    )
    primary_drop["visual_components"].update(
        {
            "jump4": 0.022,
            "jump8": 0.059,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "post_vocal8": 0.076,
            "post_inst8": 0.324,
        }
    )

    selected = select_first_visual_chunk([buildup, primary_drop])

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33


def test_opening_drop_profile_detects_tracks_that_start_in_the_drop() -> None:
    rows = []
    rows.append({"aggregate": 0.46, "groove": 0.54, "bass": 0.22, "drum": 1.0, "inst": 0.58, "vocal": 0.20})
    for _ in range(1, 12):
        rows.append({"aggregate": 0.68, "groove": 0.72, "bass": 0.50, "drum": 1.0, "inst": 0.62, "vocal": 0.10})
    for _ in range(12, 16):
        rows.append({"aggregate": 0.60, "groove": 0.64, "bass": 0.42, "drum": 1.0, "inst": 0.52, "vocal": 0.10})
    feature_map = _component_feature_map(rows, bpm=142.0)

    profile = _opening_drop_profile(feature_map)

    assert profile is not None
    assert profile["avg2_9_height"] >= 0.62
    assert profile["dense_opening_bars"] >= 10


def test_opening_drop_profile_ignores_sparse_intro_tracks() -> None:
    rows = [{"height": 0.08, "bass": 0.02, "drum": 0.0, "inst": 0.30} for _ in range(16)]
    for idx in range(8, 16):
        rows[idx] = {"aggregate": 0.52, "groove": 0.54, "bass": 0.28, "drum": 0.70, "inst": 0.45}
    feature_map = _component_feature_map(rows, bpm=140.0)

    assert _opening_drop_profile(feature_map) is None


def test_visual_first_allows_guarded_early_body_before_bar_nine() -> None:
    rows = [{"height": 0.14, "drum_cont": 0.0} for _ in range(20)]
    rows[4] = {"aggregate": 0.46, "groove": 0.52, "bass": 0.42, "drum": 0.80, "inst": 0.54, "vocal": 0.15, "drum_cont": 0.50}
    rows[5] = {"height": 0.20, "drum": 0.75, "drum_cont": 0.30}
    for idx in range(6, 14):
        rows[idx] = {"aggregate": 0.82, "groove": 0.86, "bass": 0.84, "drum": 1.0, "inst": 0.62, "vocal": 0.30, "drum_cont": 1.0}
    candidates = visual_chunk_candidates(_component_feature_map(rows, bpm=142.0, with_roles=True))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 7


def test_visual_first_skips_early_dense_intro_when_cleaner_32_bar_body_appears() -> None:
    rows = [{"height": 0.12, "drum_cont": 0.0} for _ in range(44)]
    for idx in range(4, 16):
        rows[idx] = {"aggregate": 0.62, "groove": 0.66, "bass": 0.38, "drum": 1.0, "inst": 0.72, "vocal": 0.10, "drum_cont": 0.98}
    for idx in range(16, 24):
        rows[idx] = {"aggregate": 0.62, "groove": 0.66, "bass": 0.39, "drum": 1.0, "inst": 0.30, "vocal": 0.10, "drum_cont": 0.98}
    for idx in range(28, 32):
        rows[idx] = {"aggregate": 0.30, "groove": 0.34, "bass": 0.18, "drum": 0.65, "inst": 0.48, "vocal": 0.38, "drum_cont": 0.30}
    for idx in range(32, 40):
        rows[idx] = {"aggregate": 0.58, "groove": 0.66, "bass": 0.46, "drum": 1.0, "inst": 0.10, "vocal": 0.26, "drum_cont": 1.0}
    candidates = visual_chunk_candidates(_component_feature_map(rows, bpm=140.0, with_roles=True))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33


def test_visual_late_reset_guard_replaces_low_bass_intro_with_real_body_drop() -> None:
    intro = _candidate(
        20.307,
        13,
        score=0.789,
        post4=0.666,
        post8=0.658,
        bass=0.388,
        drum=1.0,
        pre_drum=0.072,
        local_gap=0.423,
    )
    intro["visual_components"].update({"post_inst8": 0.680})
    later_body = _candidate(
        149.410,
        91,
        score=0.680,
        post4=0.730,
        post8=0.719,
        bass=0.491,
        drum=1.0,
        pre_drum=0.328,
        local_gap=0.322,
    )
    later_body["visual_components"].update({"post_inst8": 0.718, "pre4_height": 0.572})
    feature_map = {"duration_sec": 223.54, "beatgrid": {"first_low_downbeat_sec": 41.824}}

    guarded = _late_reset_body_guard_candidate(intro, [intro, later_body], feature_map)

    assert guarded is not None
    assert guarded["visual_components"]["clock_bar"] == 91
    assert guarded["selected_by"] == "visual_late_reset_body_guard"


def test_structure_section_guard_replaces_early_intro_fat_block(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _candidate(
        39.724,
        25,
        score=0.666,
        phrase_prior=0.96,
        post4=0.685,
        post8=0.686,
        bass=0.577,
        drum=1.0,
        pre_drum=0.188,
        local_gap=0.125,
        local_reentry=False,
        phrase_body_shift=True,
    )
    selected["visual_components"].update(
        {
            "pre4_height": 0.561,
            "jump4": 0.124,
            "post_drum_cont4": 0.638,
            "post_drum_cont8": 0.617,
            "post_inst8": 0.699,
        }
    )

    def fake_analyze_track_structure(*args, **kwargs):
        return {
            "first_drop": {
                "timestamp": 144.762,
                "score": 0.741,
                "confidence_score": 0.741,
                "selected_by": "structure_map",
                "reason": "drop_candidate: 4-bar phrase start | clock b77",
                "structure_clock_bar": 77,
                "structure_components": {
                    "clock_bar": 77,
                    "phrase_prior": 0.66,
                    "block_height": 0.650,
                    "sustained_groove": 0.800,
                    "post_density": 1.0,
                    "timbre_novelty": 0.819,
                    "instrumental_reentry": 0.567,
                },
            }
        }

    monkeypatch.setattr("drop_aligner.structure_map.analyze_track_structure", fake_analyze_track_structure)

    guarded = _structure_section_guard_candidate("track.flac", selected)

    assert guarded is not None
    assert guarded["timestamp"] == 144.762
    assert guarded["structure_role"] == "first_drop"
    assert guarded["selected_by"] == "visual_structure_section_guard"
    assert guarded["visual_guard_replaced_candidate"]["clock_bar"] == 25
    assert guarded["visual_guard_replaced_candidate"]["guard_reason"] == "early intro fat block"


def test_structure_section_guard_uses_later_definitive_drop_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _candidate(
        28.371891891891888,
        18,
        score=0.732,
        phrase_prior=0.18,
        post4=0.765,
        post8=0.742,
        bass=0.506,
        drum=1.0,
        pre_drum=0.385,
        local_gap=0.057,
        local_reentry=False,
    )
    selected["visual_components"].update(
        {
            "pre4_height": 0.468,
            "jump4": 0.297,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 0.968,
        }
    )
    section_entry = _candidate(
        51.07459459459459,
        32,
        score=0.622,
        phrase_prior=0.18,
        post4=0.707,
        post8=0.690,
        bass=0.486,
        drum=1.0,
        pre_drum=0.578,
        local_gap=0.277,
        local_reentry=True,
    )

    def fake_analyze_track_structure(*args, **kwargs):
        return {
            "first_drop": {
                "timestamp": 52.696216216216214,
                "score": 0.544,
                "confidence_score": 0.544,
                "selected_by": "structure_map",
                "reason": "drop_candidate: 32-bar phrase start | clock b33",
                "structure_clock_bar": 33,
                "structure_components": {
                    "clock_bar": 33,
                    "phrase_prior": 1.0,
                    "block_height": 0.696,
                    "sustained_groove": 0.759,
                    "post_density": 1.0,
                    "timbre_novelty": 0.571,
                    "instrumental_reentry": 0.175,
                },
            }
        }

    monkeypatch.setattr("drop_aligner.structure_map.analyze_track_structure", fake_analyze_track_structure)

    guarded = _structure_section_guard_candidate("track.wav", selected, [selected, section_entry])

    assert guarded is not None
    assert guarded["timestamp"] == 51.07459459459459
    assert guarded["selected_by"] == "visual_structure_section_guard"
    assert guarded["structure_section_anchor_time"] == 52.696216216216214
    assert guarded["visual_guard_replaced_candidate"]["clock_bar"] == 18
    assert guarded["visual_guard_replaced_candidate"]["guard_reason"] == "early off-phrase sustained block"


def test_fusion_audit_guard_replaces_early_visual_block(tmp_path: Path) -> None:
    audio = tmp_path / "drums_145_8A_Test Track.flac"
    audio.write_bytes(b"")
    (tmp_path / "drop_fusion_audit.json").write_text(
        """
{
  "ok": true,
  "bpm": 145,
  "suggestion": {
    "available": true,
    "safe_to_write": true,
    "time_sec": 92.68562358276644,
    "score": 0.8239663526370087,
    "confidence": 0.6729114127104177,
    "roles": ["clock", "cue"],
    "sources": ["bpm_phrase_bar", "rekordbox_mik_cue"],
    "write_policy": "high_confidence_candidate_available_but_not_written_by_audit",
    "components": {
      "drums_bass_impact": 0.7975417403507166,
      "drums_energy_jump": 0.5298307770267074,
      "instrumental_jump": 0.85,
      "vocal_transition": 0.6340425535775076,
      "fake_hit_penalty": 0.0,
      "weak_body_penalty": 0.0
    }
  },
  "candidates": [
    {
      "rank": 1,
      "time_sec": 92.68562358276644,
      "score": 0.8239663526370087,
      "roles": ["clock", "cue"],
      "sources": ["bpm_phrase_bar", "rekordbox_mik_cue"],
      "components": {
        "drums_bass_impact": 0.7975417403507166,
        "drums_energy_jump": 0.5298307770267074,
        "instrumental_jump": 0.85,
        "vocal_transition": 0.6340425535775076,
        "fake_hit_penalty": 0.0,
        "weak_body_penalty": 0.0
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    selected = _candidate(
        39.724,
        25,
        score=0.666,
        phrase_prior=0.96,
        post4=0.685,
        post8=0.686,
        bass=0.577,
        local_reentry=False,
        phrase_body_shift=True,
    )

    guarded = _fusion_audit_guard_candidate(str(audio), selected)

    assert guarded is not None
    assert guarded["timestamp"] == 92.68562358276644
    assert guarded["selected_by"] == "visual_fusion_audit_guard"
    assert guarded["visual_guard_replaced_candidate"]["clock_bar"] == 25
    assert guarded["fusion_audit"]["safe_to_write"] is True


def test_fusion_audit_guard_defers_one_bar_to_body_drop(tmp_path: Path) -> None:
    audio = tmp_path / "drums_145_8A_Test Track.flac"
    audio.write_bytes(b"")
    (tmp_path / "drop_fusion_audit.json").write_text(
        """
{
  "ok": true,
  "bpm": 145,
  "suggestion": {
    "available": true,
    "safe_to_write": true,
    "time_sec": 92.68562358276644,
    "score": 0.8239663526370087,
    "confidence": 0.6729114127104177,
    "roles": ["clock", "cue"],
    "sources": ["bpm_phrase_bar", "rekordbox_mik_cue"],
    "write_policy": "high_confidence_candidate_available_but_not_written_by_audit",
    "components": {
      "drums_bass_impact": 0.7975417403507166,
      "drums_energy_jump": 0.5298307770267074,
      "instrumental_jump": 0.85,
      "vocal_transition": 0.6340425535775076,
      "fake_hit_penalty": 0.0,
      "weak_body_penalty": 0.0
    }
  },
  "candidates": []
}
""".strip(),
        encoding="utf-8",
    )
    selected = _candidate(
        39.724,
        25,
        score=0.666,
        phrase_prior=0.96,
        post4=0.685,
        post8=0.686,
        bass=0.577,
        local_reentry=False,
        phrase_body_shift=True,
    )
    body = _candidate(
        94.34482758620689,
        58,
        score=0.633,
        phrase_prior=0.18,
        post4=0.725,
        post8=0.676,
        bass=0.603,
        drum=1.0,
        pre_drum=0.346,
        local_gap=0.258,
    )
    body["visual_components"].update({"post_drum_cont4": 0.856, "post_drum_cont8": 0.772})

    guarded = _fusion_audit_guard_candidate(str(audio), selected, [body])

    assert guarded is not None
    assert guarded["timestamp"] == 94.34482758620689
    assert guarded["selected_by"] == "visual_fusion_audit_body_defer"
    assert guarded["fusion_audit_pre_drop_candidate"]["timestamp"] == 92.68562358276644
    assert guarded["visual_guard_replaced_candidate"]["guard_reason"] == "fusion audit cue was one bar before body drop"
    assert guarded["bpm_clock"]["on_one"] is True


def test_fusion_audit_guard_prefers_stronger_second_bar_body_drop(tmp_path: Path) -> None:
    audio = tmp_path / "drums_145_8A_Test Track.flac"
    audio.write_bytes(b"")
    (tmp_path / "drop_fusion_audit.json").write_text(
        """
{
  "ok": true,
  "bpm": 145,
  "suggestion": {
    "available": true,
    "safe_to_write": true,
    "time_sec": 92.68562358276644,
    "score": 0.8239663526370087,
    "confidence": 0.6729114127104177,
    "roles": ["clock", "cue"],
    "sources": ["bpm_phrase_bar", "rekordbox_mik_cue"],
    "write_policy": "high_confidence_candidate_available_but_not_written_by_audit",
    "components": {
      "drums_bass_impact": 0.7975417403507166,
      "drums_energy_jump": 0.5298307770267074,
      "instrumental_jump": 0.85,
      "vocal_transition": 0.6340425535775076,
      "fake_hit_penalty": 0.0,
      "weak_body_penalty": 0.0
    }
  },
  "candidates": []
}
""".strip(),
        encoding="utf-8",
    )
    selected = _candidate(
        39.724,
        25,
        score=0.666,
        phrase_prior=0.96,
        post4=0.685,
        post8=0.686,
        bass=0.577,
        local_reentry=False,
        phrase_body_shift=True,
    )
    partial_body = _candidate(
        94.34482758620689,
        58,
        score=0.6334974114495231,
        phrase_prior=0.18,
        post4=0.7254337557204508,
        post8=0.6758217173368177,
        bass=0.6027825997308659,
        drum=0.9055152805653318,
        pre_drum=0.3460595776772248,
        local_gap=0.2579518812591951,
    )
    partial_body["visual_components"].update(
        {
            "post_drum_cont4": 0.8557692307692307,
            "post_drum_cont8": 0.7716346153846154,
            "jump4": 0.17312613851244263,
            "jump8": 0.048790502071512476,
        }
    )
    true_body = _candidate(
        96.0,
        59,
        score=0.6733465364430108,
        phrase_prior=0.18,
        post4=0.753898460285814,
        post8=0.6849214538771216,
        bass=0.6220014238058743,
        drum=0.8873699044649331,
        pre_drum=0.3460595776772248,
        local_gap=0.2385145004879703,
    )
    true_body["visual_components"].update(
        {
            "post_drum_cont4": 0.9038461538461539,
            "post_drum_cont8": 0.7809671945701357,
            "jump4": 0.2254899321560816,
            "jump8": 0.07085464674712283,
        }
    )

    guarded = _fusion_audit_guard_candidate(str(audio), selected, [partial_body, true_body])

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(96.0)
    assert guarded["visual_components"]["clock_bar"] == 59
    assert guarded["visual_components"]["fusion_audit_body_offset_bars"] == 2
    assert guarded["selected_by"] == "visual_fusion_audit_body_defer"
    assert "deferred 2 bars" in guarded["reason"]
    assert guarded["visual_guard_replaced_candidate"]["guard_reason"] == "fusion audit cue was 2 bars before body drop"
    assert guarded["bpm_clock"]["on_one"] is True


def test_zoomed_marker_preserves_deferred_visual_one() -> None:
    marker = _zoomed_marker_time(
        94.34482758620689,
        {
            "microaligned_time": 95.1133536633044,
            "micro_confidence": 0.92,
            "snap_offset_ms": 768.5,
        },
        {
            "preserve_visual_on_one": True,
            "fusion_audit_body_defer": True,
            "local_reentry": True,
            "post_drum8": 1.0,
            "post_bass8": 0.603,
            "post_drum_cont4": 0.856,
        },
    )

    assert marker == pytest.approx(94.34482758620689)

    tiny_snap = _zoomed_marker_time(
        96.0,
        {
            "microaligned_time": 95.99780045351474,
            "micro_confidence": 0.88,
            "snap_offset_ms": -2.2,
        },
        {
            "preserve_visual_on_one": True,
            "fusion_audit_body_defer": True,
            "local_reentry": True,
            "post_drum8": 0.887,
            "post_bass8": 0.622,
            "post_drum_cont4": 0.904,
        },
    )

    assert tiny_snap == pytest.approx(96.0)


def test_blank_waveform_guard_replaces_empty_blue_with_first_body_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    times = np.round(np.arange(25.80, 27.25, 0.032), 3)
    rms = np.zeros_like(times, dtype=np.float64)
    low = np.zeros_like(times, dtype=np.float64)
    attack = np.zeros_like(times, dtype=np.float64)

    def add_hit(center: float, rms_value: float, low_value: float, attack_value: float) -> None:
        mask = np.abs(times - center) <= 0.070
        rms[mask] = np.maximum(rms[mask], rms_value)
        low[mask] = np.maximum(low[mask], low_value)
        attack[mask] = np.maximum(attack[mask], attack_value)

    add_hit(26.284, 0.94, 0.95, 1.0)
    add_hit(26.690, 0.99, 0.98, 1.0)
    add_hit(27.100, 0.97, 0.95, 1.0)
    features = SimpleNamespace(
        frame_times=times,
        rms=rms,
        low_energy=low,
        combined_attack=attack,
        duration_sec=120.0,
    )

    monkeypatch.setattr("drop_aligner.visual_first.find_stem_group", lambda path: SimpleNamespace(roles={"drums": path}))
    monkeypatch.setattr("drop_aligner.visual_first.infer_bpm_from_path", lambda _path: 148.0)
    monkeypatch.setattr("drop_aligner.visual_first.extract_features", lambda *_args, **_kwargs: features)

    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        if abs(marker_time - 26.690) <= 0.090:
            return {
                "microaligned_time": 26.690,
                "micro_confidence": 0.988,
                "impact_boundary_confidence": 0.976,
                "attack_peak_strength": 1.0,
                "denoised_impact_strength": 1.0,
                "rms_rise_score": 0.689,
                "peak_rise_score": 0.456,
                "snap_offset_ms": 0.0,
            }
        if abs(marker_time - 27.100) <= 0.090:
            return {
                "microaligned_time": 27.100,
                "micro_confidence": 0.840,
                "impact_boundary_confidence": 0.850,
                "attack_peak_strength": 1.0,
                "denoised_impact_strength": 0.720,
                "rms_rise_score": 1.0,
                "peak_rise_score": 1.0,
                "snap_offset_ms": 0.0,
            }
        return {
            "microaligned_time": 26.284,
            "micro_confidence": 0.930,
            "impact_boundary_confidence": 0.900,
            "attack_peak_strength": 1.0,
            "denoised_impact_strength": 0.720,
            "rms_rise_score": 0.380,
            "peak_rise_score": 0.040,
            "snap_offset_ms": 0.0,
        }

    monkeypatch.setattr("drop_aligner.visual_first.microalign_marker", fake_microalign)

    selected = _candidate(
        25.945945945945947,
        17,
        score=0.708,
        phrase_prior=0.94,
        post4=0.634,
        post8=0.641,
        bass=0.525,
        pre_drum=0.0,
        local_gap=0.368,
    )
    selected["visual_components"]["preserve_visual_on_one"] = True

    guarded = _blank_waveform_marker_guard_candidate(
        "dummy.flac",
        25.945945945945947,
        25.945945945945947,
        selected,
        {
            "microaligned_time": 26.284,
            "micro_confidence": 0.474,
            "impact_boundary_confidence": 0.446,
            "snap_offset_ms": 338.0,
            "review_needed": True,
        },
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(26.284)
    assert guarded["selected_by"] == "visual_blank_waveform_guard"
    assert guarded["visual_components"]["blank_waveform_marker_guard"] is True


def test_blank_waveform_guard_scans_past_nearby_gap_to_body_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    times = np.round(np.arange(25.20, 30.20, 0.032), 3)
    rms = np.zeros_like(times, dtype=np.float64)
    low = np.zeros_like(times, dtype=np.float64)
    attack = np.zeros_like(times, dtype=np.float64)

    mask = np.abs(times - 28.610) <= 0.070
    rms[mask] = 0.92
    low[mask] = 0.88
    attack[mask] = 0.96
    features = SimpleNamespace(
        frame_times=times,
        rms=rms,
        low_energy=low,
        combined_attack=attack,
        duration_sec=120.0,
    )

    monkeypatch.setattr("drop_aligner.visual_first.find_stem_group", lambda path: SimpleNamespace(roles={"drums": path}))
    monkeypatch.setattr("drop_aligner.visual_first.infer_bpm_from_path", lambda _path: 151.0)
    monkeypatch.setattr("drop_aligner.visual_first.extract_features", lambda *_args, **_kwargs: features)

    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        return {
            "microaligned_time": 28.610 if abs(marker_time - 28.610) <= 0.100 else marker_time,
            "micro_confidence": 0.960,
            "impact_boundary_confidence": 0.920,
            "attack_peak_strength": 1.0,
            "denoised_impact_strength": 0.900,
            "rms_rise_score": 0.900,
            "peak_rise_score": 0.900,
            "snap_offset_ms": 0.0,
        }

    monkeypatch.setattr("drop_aligner.visual_first.microalign_marker", fake_microalign)

    selected = _candidate(
        25.43046357615894,
        17,
        score=0.623,
        phrase_prior=0.94,
        post4=0.609,
        post8=0.652,
        bass=0.480,
        pre_drum=0.812,
        local_gap=0.100,
        local_reentry=False,
        phrase_body_shift=True,
        bpm=151.0,
    )

    guarded = _blank_waveform_marker_guard_candidate(
        "dummy_151.flac",
        25.43046357615894,
        25.382277635115855,
        selected,
        {
            "microaligned_time": 25.382277635115855,
            "micro_confidence": 0.750,
            "impact_boundary_confidence": 0.437,
            "snap_offset_ms": -48.19,
            "review_needed": True,
        },
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(28.610)
    assert guarded["selected_by"] == "visual_blank_waveform_guard"


def test_visual_drop_v2_keeps_clear_song_start_drop_over_slightly_later_body() -> None:
    rows = [{"height": 0.05, "drum_cont": 0.0} for _ in range(44)]
    for idx in range(8, 16):
        rows[idx] = {
            "aggregate": 0.64,
            "groove": 0.68,
            "bass": 0.54,
            "drum": 1.0,
            "inst": 0.42,
            "vocal": 0.28,
            "drum_cont": 0.90,
        }
    for idx in range(28, 32):
        rows[idx] = {"height": 0.26, "bass": 0.18, "drum": 0.55, "drum_cont": 0.20}
    for idx in range(32, 40):
        rows[idx] = {
            "aggregate": 0.68,
            "groove": 0.70,
            "bass": 0.56,
            "drum": 1.0,
            "inst": 0.44,
            "vocal": 0.26,
            "drum_cont": 1.0,
        }
    candidates = visual_drop_v2_candidates(_component_feature_map(rows, bpm=144.0, with_roles=True))

    selected = select_visual_drop_v2(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 9
    assert "protected first clear song-start drop section" in selected["reason"]


def test_visual_drop_v2_uses_phrase_body_entry_after_intro_fill() -> None:
    rows = [{"height": 0.05, "drum_cont": 0.0} for _ in range(44)]
    for idx in range(14, 18):
        rows[idx] = {
            "aggregate": 0.42,
            "groove": 0.44,
            "bass": 0.28,
            "drum": 0.35,
            "inst": 0.88,
            "vocal": 0.28,
            "drum_cont": 0.08,
        }
    for idx in range(18, 24):
        rows[idx] = {
            "aggregate": 0.66,
            "groove": 0.70,
            "bass": 0.48,
            "drum": 1.0,
            "inst": 0.62,
            "vocal": 0.30,
            "drum_cont": 0.74,
        }
    for idx in range(24, 32):
        rows[idx] = {
            "aggregate": 0.68,
            "groove": 0.72,
            "bass": 0.58,
            "drum": 1.0,
            "inst": 0.08,
            "vocal": 0.20,
            "drum_cont": 1.0,
        }

    candidates = visual_drop_v2_candidates(_component_feature_map(rows, bpm=150.0, with_roles=True))
    selected = select_visual_drop_v2(candidates)

    assert any(row["visual_components"]["clock_bar"] == 25 for row in candidates)
    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 25
    assert selected["visual_components"]["phrase_body_entry"] is True
    assert "skipped nearby intro fill/build" in selected["reason"]


def test_visual_drop_v2_zoomed_marker_keeps_phrase_body_grid_edge() -> None:
    marker = visual_drop_v2_module._zoomed_marker(
        38.4,
        {
            "microaligned_time": 38.3668253968254,
            "micro_confidence": 0.991,
            "impact_boundary_confidence": 0.944,
            "attack_cleanliness": 1.0,
            "snap_offset_ms": -33.1746031746,
        },
        {
            "clock_bar": 25,
            "phrase_prior": 0.96,
            "body_score": 0.702,
            "post4_height": 0.675,
            "post8_height": 0.683,
            "post_bass8": 0.541,
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "post_inst8": 0.087,
            "post_vocal8": 0.207,
            "phrase_body_entry": True,
        },
    )

    assert marker == pytest.approx(38.4)


def test_visual_drop_v2_skips_vocal_buildup_for_darker_primary_drop() -> None:
    early_buildup = _v2_candidate(
        26.122448979591837,
        17,
        score=0.643,
        phrase_prior=0.94,
        body=0.658,
        post4=0.622,
        post8=0.601,
        pre4=0.229,
        bass=0.415,
        pre_drum=0.010,
        post_drum4=0.956,
        post_drum8=0.951,
        transition=0.414,
        inst=0.286,
        vocal=0.646,
        local_gap=0.390,
    )
    earlier_dark_build = _v2_candidate(
        45.714285714285715,
        29,
        score=0.399,
        phrase_prior=0.66,
        body=0.615,
        post4=0.646,
        post8=0.639,
        pre4=0.605,
        bass=0.485,
        pre_drum=0.309,
        post_drum4=0.559,
        post_drum8=0.676,
        transition=0.069,
        inst=0.484,
        vocal=0.135,
        local_gap=0.052,
    )
    primary_drop = _v2_candidate(
        52.244897959183675,
        33,
        score=0.420,
        phrase_prior=1.0,
        body=0.621,
        post4=0.632,
        post8=0.629,
        pre4=0.646,
        bass=0.458,
        pre_drum=0.559,
        post_drum4=0.794,
        post_drum8=0.777,
        transition=0.037,
        inst=0.254,
        vocal=0.236,
        local_gap=0.012,
    )

    selected = select_visual_drop_v2([early_buildup, earlier_dark_build, primary_drop])

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 33
    assert selected["visual_components"]["vocal_texture_buildup_release"] is True
    assert "skipped vocal/texture buildup" in selected["reason"]


def test_visual_drop_v2_body_beat_probe_uses_stronger_drop_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    early_buildup = _v2_candidate(
        26.122448979591837,
        17,
        score=0.643,
        phrase_prior=0.94,
        body=0.658,
        post4=0.622,
        post8=0.601,
        bass=0.415,
        pre_drum=0.010,
        post_drum4=0.956,
        post_drum8=0.951,
        transition=0.414,
        inst=0.286,
        vocal=0.646,
        local_gap=0.390,
    )
    primary_drop = _v2_candidate(
        52.244897959183675,
        33,
        score=0.420,
        phrase_prior=1.0,
        body=0.621,
        post4=0.632,
        post8=0.629,
        pre4=0.646,
        bass=0.458,
        pre_drum=0.559,
        post_drum4=0.794,
        post_drum8=0.777,
        transition=0.037,
        inst=0.254,
        vocal=0.236,
        local_gap=0.012,
    )
    beat_sec = 60.0 / 147.0
    target = 52.244897959183675 + beat_sec

    monkeypatch.setattr(
        visual_drop_v2_module,
        "compute_bar_feature_map",
        lambda *_args, **_kwargs: {
            "ok": True,
            "bar_count": 64,
            "duration_sec": 120.0,
            "beatgrid": {"bpm": 147.0},
            "bars": [],
        },
    )
    monkeypatch.setattr(
        visual_drop_v2_module,
        "visual_drop_v2_candidates",
        lambda *_args, **_kwargs: [early_buildup, primary_drop],
    )

    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        if abs(marker_time - target) <= 0.010:
            return {
                "microaligned_time": target,
                "snap_offset_ms": 0.0,
                "micro_confidence": 0.963,
                "impact_boundary_confidence": 0.990,
                "attack_peak_strength": 1.0,
                "denoised_impact_strength": 1.0,
                "rms_rise_score": 1.0,
                "peak_rise_score": 1.0,
                "reason": "input boundary kept inside tight visual onset bracket",
            }
        return {
            "microaligned_time": marker_time,
            "snap_offset_ms": 0.0,
            "micro_confidence": 0.918,
            "impact_boundary_confidence": 0.908,
            "attack_peak_strength": 0.714,
            "denoised_impact_strength": 0.720,
            "rms_rise_score": 1.0,
            "peak_rise_score": 1.0,
            "reason": "quiet center-line departure before sustained attack",
        }

    monkeypatch.setattr(visual_drop_v2_module, "microalign_marker", fake_microalign)

    result = visual_drop_v2_module.visual_drop_v2_marker("dummy.flac", sample_rate=16000, use_cache=False)

    assert result["marker"] == pytest.approx(target)
    assert result["raw_visual_time"] == pytest.approx(52.244897959183675)
    assert result["selected_candidate"]["microalign"]["visual_drop_v2_body_beat_probe_used"] is True


def test_visual_first_only_uses_v2_clear_start_for_later_intro_phrase() -> None:
    base_result = {
        "ok": True,
        "raw_visual_time": 14.8,
        "feature_map": {
            "beatgrid": {
                "bpm": 142.0,
                "first_low_downbeat_sec": 18.2,
            }
        },
        "selected_candidate": {
            "score": 0.62,
            "reason": "visual-v2 protected first clear song-start drop section",
            "visual_components": {
                "clock_bar": 9,
                "body_score": 0.64,
                "post4_height": 0.70,
                "post8_height": 0.66,
                "post_bass8": 0.45,
                "pre_drum_cont4": 0.10,
                "post_drum_cont4": 0.90,
                "post_drum_cont8": 0.90,
                "post_inst8": 0.25,
                "transition": 0.24,
            },
        },
    }

    assert _use_visual_drop_v2_result(base_result) is False

    later_phrase = dict(base_result)
    later_phrase["raw_visual_time"] = 30.4
    later_phrase["feature_map"] = {"beatgrid": {"bpm": 142.0, "first_low_downbeat_sec": 30.4}}
    later_phrase["selected_candidate"] = dict(base_result["selected_candidate"])
    later_phrase["selected_candidate"]["visual_components"] = dict(base_result["selected_candidate"]["visual_components"])
    later_phrase["selected_candidate"]["visual_components"]["clock_bar"] = 19

    assert _use_visual_drop_v2_result(later_phrase) is True


def test_visual_first_uses_v2_nearby_intro_fill_phrase_body_result() -> None:
    result = {
        "ok": True,
        "raw_visual_time": 38.4,
        "feature_map": {
            "beatgrid": {
                "bpm": 150.0,
                "first_low_downbeat_sec": 0.0,
            }
        },
        "selected_candidate": {
            "score": 0.460,
            "reason": "visual-v2 skipped nearby intro fill/build and selected the first phrase body drop entry",
            "visual_components": {
                "clock_bar": 25,
                "phrase_prior": 0.96,
                "body_score": 0.702,
                "post4_height": 0.675,
                "post8_height": 0.683,
                "post_bass8": 0.541,
                "post_drum_cont4": 1.0,
                "post_drum_cont8": 1.0,
                "post_inst8": 0.087,
                "post_vocal8": 0.207,
            },
        },
    }

    assert _use_visual_drop_v2_result(result) is True


def test_visual_drop_v2_gate_rejects_late_section_when_earlier_comparable_drop_exists() -> None:
    selected = {
        "timestamp": 63.172671532846714,
        "visual_raw_chunk_time": 63.172671532846714,
        "score": 0.636,
        "visual_components": {
            "clock_bar": 37,
            "body_score": 0.716,
            "post4_height": 0.713,
            "post8_height": 0.717,
            "post_bass8": 0.535,
            "post_drum8": 1.0,
            "pre_drum_cont4": 0.064,
            "post_drum_cont4": 0.877,
            "post_drum_cont8": 0.861,
            "phrase_prior": 0.80,
            "transition": 0.390,
        },
    }
    earlier = {
        "timestamp": 28.136175182481754,
        "visual_raw_chunk_time": 28.136175182481754,
        "score": 0.627,
        "visual_components": {
            "clock_bar": 17,
            "body_score": 0.684,
            "post4_height": 0.682,
            "post8_height": 0.692,
            "post_bass8": 0.512,
            "post_drum8": 1.0,
            "pre_drum_cont4": 0.014,
            "post_drum_cont4": 0.782,
            "post_drum_cont8": 0.770,
            "phrase_prior": 0.94,
            "transition": 0.433,
        },
    }

    assert _visual_drop_v2_has_earlier_comparable_drop(selected, [selected, earlier]) is True


def test_visual_first_uses_phrase_body_shift_when_waveform_is_dense_from_start() -> None:
    rows = [{"aggregate": 0.60, "groove": 0.65, "bass": 0.42, "drum": 1.0, "inst": 0.55, "vocal": 0.68, "drum_cont": 0.72} for _ in range(72)]
    for idx in range(48, 56):
        rows[idx] = {"aggregate": 0.60, "groove": 0.70, "bass": 0.58, "drum": 1.0, "inst": 0.30, "vocal": 0.22, "drum_cont": 0.86}
    candidates = visual_chunk_candidates(_component_feature_map(rows, bpm=142.0, with_roles=True))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 49


def test_zoomed_marker_allows_forward_move_to_drop_body() -> None:
    marker = _zoomed_marker_time(
        27.428571,
        {
            "microaligned_time": 27.924898,
            "visual_onset_knee_time": 27.924898,
            "impact_body_time": 27.93068,
        },
    )

    assert marker == 27.924898


def test_zoomed_marker_rejects_large_backwards_tail_move() -> None:
    marker = _zoomed_marker_time(
        41.142857,
        {
            "microaligned_time": 40.712381,
            "impact_body_time": 41.145057,
        },
    )

    assert marker == 41.145057


def test_zoomed_marker_keeps_strong_knee_before_late_grid_edge() -> None:
    marker = _zoomed_marker_time(
        27.08865753424658,
        {
            "microaligned_time": 27.03035821451869,
            "visual_onset_knee_time": 27.03035821451869,
            "visual_onset_knee_quality": 0.644186765615337,
            "visual_onset_knee_used": 1.0,
            "attack_start_time": 27.03412238685429,
            "impact_body_time": 27.08489336191098,
            "impact_body_quality": 0.6021176062873269,
            "micro_confidence": 0.5932295074318062,
            "attack_peak_strength": 1.0,
            "denoised_impact_strength": 1.0,
            "impact_boundary_confidence": 0.6355116747438849,
            "peak_rise_score": 1.0,
        },
        {
            "clock_bar": 17,
            "phrase_prior": 0.94,
            "local_reentry": True,
            "local_reentry_gap": 0.2152578610461986,
            "pre_drum_cont4": 0.12745098039215685,
            "post_drum8": 1.0,
            "post_bass8": 0.291066170312234,
        },
    )

    assert marker == pytest.approx(27.03035821451869)


def test_zoomed_marker_rejects_small_forward_snap_into_decay_tail() -> None:
    marker = _zoomed_marker_time(
        26.814270270270267,
        {
            "microaligned_time": 26.828283875712444,
            "snap_offset_ms": 14.013605442176669,
            "micro_confidence": 0.7062534482860731,
            "impact_boundary_confidence": 0.8705473943468849,
            "attack_cleanliness": 0.5126843045483164,
            "zero_crossing_quality": 0.6600675820302488,
        },
        {
            "clock_bar": 17,
            "phrase_prior": 0.94,
            "local_reentry": True,
            "local_reentry_gap": 0.19994751332125882,
            "pre4_height": 0.35830541075348404,
            "post4_height": 0.5585265859082124,
            "post8_height": 0.5572819697044273,
            "jump4": 0.20022117515472831,
            "jump8": 0.27158834467477816,
            "post_bass8": 0.25124957558592986,
            "post_drum8": 1.0,
            "pre_drum_cont4": 0.16941176470588237,
            "post_drum_cont4": 0.8165686274509804,
            "post_drum_cont8": 0.8175980392156863,
        },
    )

    assert marker == pytest.approx(26.814270270270267)


def test_zoomed_marker_rejects_peak_fallback_into_decay_tail() -> None:
    marker = _zoomed_marker_time(
        26.814270270270267,
        {
            "microaligned_time": 26.757830360973216,
            "impact_body_time": 26.757830360973216,
            "attack_start_time": 26.757830360973216,
            "peak_time": 26.828283875712444,
            "zero_crossing_time": 26.75433829748115,
            "snap_offset_ms": -56.439909297051116,
            "micro_confidence": 0.7062534482860731,
            "impact_boundary_confidence": 0.8705473943468849,
            "attack_cleanliness": 0.5126843045483164,
            "zero_crossing_quality": 0.6600675820302488,
        },
        {
            "clock_bar": 17,
            "phrase_prior": 0.94,
            "local_reentry": True,
            "local_reentry_gap": 0.19994751332125882,
            "pre4_height": 0.35830541075348404,
            "post4_height": 0.5585265859082124,
            "post8_height": 0.5572819697044273,
            "jump4": 0.20022117515472831,
            "jump8": 0.27158834467477816,
            "post_bass8": 0.25124957558592986,
            "post_drum8": 1.0,
            "pre_drum_cont4": 0.16941176470588237,
            "post_drum_cont4": 0.8165686274509804,
            "post_drum_cont8": 0.8175980392156863,
        },
    )

    assert marker == pytest.approx(26.814270270270267)


def test_track_zero_grid_phase_guard_replaces_half_bar_late_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        assert marker_time == pytest.approx(26.301369863013697)
        return {
            "microaligned_time": 26.303968321995466,
            "ableton_asd_time": 26.303968253968254,
            "attack_start_time": 26.305192811791382,
            "visual_onset_knee_time": 26.30371888888889,
            "visual_onset_knee_quality": 0.8413354267603077,
            "micro_confidence": 0.9654785214471509,
            "impact_boundary_confidence": 0.9124144381138972,
            "snap_offset_ms": -0.861678004532962,
        }

    monkeypatch.setattr("drop_aligner.visual_first.microalign_marker", fake_microalign)
    selected = _candidate(
        27.08865753424658,
        17,
        score=0.665,
        phrase_prior=0.94,
        post4=0.536,
        post8=0.577,
        bass=0.291,
        drum=1.0,
        pre_drum=0.127,
        local_gap=0.215,
    )
    feature_map = {
        "beatgrid": {
            "bpm": 146.0,
            "beat_sec": 60.0 / 146.0,
            "bar_sec": 240.0 / 146.0,
            "bar_zero_sec": 0.78728767123288,
        }
    }

    guarded = _track_zero_grid_phase_guard_candidate(
        "drums_146_12A_5-DnA, Dov1, An-Ten-Nae, OAKK - Dreams - OAKK Remix.flac",
        selected,
        feature_map,
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(26.303968253968254)
    assert guarded["selected_by"] == "visual_track_zero_grid_phase_guard"
    assert guarded["visual_components"]["clock_bar"] == 17
    assert guarded["visual_components"]["track_zero_grid_phase_guard"] is True
    assert guarded["bpm_clock"]["on_one"] is True
    assert guarded["visual_grid_replaced_candidate"]["timestamp"] == pytest.approx(27.08865753424658)


def test_track_zero_grid_phase_guard_replaces_one_beat_late_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        assert marker_time == pytest.approx(32.87671232876712)
        return {
            "microaligned_time": 32.89948752834467,
            "visual_onset_knee_time": 32.89948752834467,
            "visual_onset_knee_quality": 0.88,
            "micro_confidence": 0.998,
            "impact_boundary_confidence": 0.922,
            "snap_offset_ms": 22.77519957755203,
        }

    monkeypatch.setattr("drop_aligner.visual_first.microalign_marker", fake_microalign)
    selected = _candidate(
        33.30016438356164,
        21,
        score=0.640,
        phrase_prior=0.66,
        post4=0.639,
        post8=0.640,
        bass=0.415,
        drum=1.0,
        pre_drum=0.882,
        local_gap=0.168,
    )
    selected["visual_components"].update(
        {
            "post_drum_cont4": 1.0,
            "post_drum_cont8": 1.0,
            "local_reentry": True,
        }
    )
    feature_map = {
        "beatgrid": {
            "bpm": 146.0,
            "beat_sec": 60.0 / 146.0,
            "bar_sec": 240.0 / 146.0,
            "bar_zero_sec": 0.4234520547945251,
        }
    }

    guarded = _track_zero_grid_phase_guard_candidate(
        "drums_146_9A_5-Mo Bamba - Sheck Wes.wav",
        selected,
        feature_map,
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(32.89948752834467)
    assert guarded["selected_by"] == "visual_track_zero_grid_phase_guard"
    assert guarded["visual_components"]["clock_bar"] == 21
    assert guarded["visual_components"]["track_zero_grid_phase_guard"] is True
    assert guarded["bpm_clock"]["on_one"] is True
    assert guarded["visual_grid_replaced_candidate"]["phase_beats"] == pytest.approx(1.0304)
    assert "one-beat-late" in guarded["reason"]


def test_track_zero_grid_phase_guard_replaces_beat_four_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_microalign(_audio_path: str, marker_time: float, **_kwargs: object) -> dict:
        assert marker_time == pytest.approx(47.671232876712324)
        return {
            "microaligned_time": 47.66923741185972,
            "visual_onset_knee_time": 47.66923741185972,
            "visual_onset_knee_quality": 0.86,
            "micro_confidence": 0.993,
            "impact_boundary_confidence": 0.909,
            "snap_offset_ms": -1.9954648526052665,
        }

    monkeypatch.setattr("drop_aligner.visual_first.microalign_marker", fake_microalign)
    selected = _candidate(
        47.27101369863014,
        29,
        score=0.765,
        phrase_prior=0.66,
        post4=0.756,
        post8=0.775,
        bass=0.638,
        drum=1.0,
        pre_drum=0.0,
        local_gap=0.365,
    )
    selected["visual_components"].update(
        {
            "post_drum_cont4": 0.956,
            "post_drum_cont8": 0.978,
            "local_reentry": True,
        }
    )
    feature_map = {
        "beatgrid": {
            "bpm": 146.0,
            "beat_sec": 60.0 / 146.0,
            "bar_sec": 240.0 / 146.0,
            "bar_zero_sec": 1.2436164383561676,
        }
    }

    guarded = _track_zero_grid_phase_guard_candidate(
        "drums_146_9B_5-jigitz - make believe.flac",
        selected,
        feature_map,
    )

    assert guarded is not None
    assert guarded["timestamp"] == pytest.approx(47.66923741185972)
    assert guarded["selected_by"] == "visual_track_zero_grid_phase_guard"
    assert guarded["bpm_clock"]["on_one"] is True
    assert guarded["visual_grid_replaced_candidate"]["phase_beats"] == pytest.approx(3.0261333333333416)
    assert "beat-four/one-beat-early" in guarded["reason"]


def test_zoomed_marker_rejects_large_forward_low_confidence_snap() -> None:
    marker = _zoomed_marker_time(
        54.006857,
        {
            "microaligned_time": 54.746426,
            "impact_body_time": 54.746426,
            "snap_offset_ms": 739.6,
            "micro_confidence": 0.465,
            "reason": "large microalignment offset; review recommended",
        },
        {
            "pre4_height": 0.579,
            "jump4": 0.078,
        },
    )

    assert marker == 54.006857


def test_zoomed_marker_rejects_full_beat_jump_from_clean_visual_reentry() -> None:
    marker = _zoomed_marker_time(
        29.04165517241379,
        {
            "microaligned_time": 30.059047462663223,
            "snap_offset_ms": 1017.392,
            "micro_confidence": 0.4817,
            "impact_boundary_confidence": 0.72,
        },
        {
            "local_reentry": True,
            "local_reentry_gap": 0.529,
            "pre_drum_cont4": 0.0,
            "post_drum8": 1.0,
            "post_bass8": 0.425,
        },
    )

    assert marker == pytest.approx(29.04165517241379)


def test_zoomed_marker_rejects_low_confidence_forward_jump_to_segment_middle() -> None:
    marker = _zoomed_marker_time(
        16.11940298507463,
        {
            "microaligned_time": 16.46221477645785,
            "snap_offset_ms": 342.812,
            "micro_confidence": 0.492,
            "impact_boundary_confidence": 0.439,
            "attack_cleanliness": 1.0,
            "attack_peak_strength": 1.0,
            "denoised_impact_strength": 0.772,
            "rms_rise_score": 1.0,
            "peak_rise_score": 1.0,
        },
        {
            "local_reentry": True,
            "local_reentry_gap": 0.688,
            "pre_drum_cont4": 0.0,
            "post_drum8": 1.0,
            "post_bass8": 0.491,
        },
    )

    assert marker == pytest.approx(16.11940298507463)


def test_zoomed_marker_allows_sustained_body_snap_when_visual_edge_is_not_clean_one() -> None:
    marker = _zoomed_marker_time(
        43.711999999999996,
        {
            "microaligned_time": 44.55789569160997,
            "attack_start_time": 44.567034013605436,
            "zero_crossing_time": 44.56662585034013,
            "impact_body_time": 44.571138321995456,
            "micro_confidence": 0.494,
            "impact_boundary_confidence": 0.432,
            "attack_cleanliness": 1.0,
            "attack_peak_strength": 1.0,
            "denoised_impact_strength": 0.72,
            "rms_rise_score": 1.0,
            "peak_rise_score": 1.0,
            "zero_crossing_quality": 0.989,
        },
        {
            "local_reentry": True,
            "local_reentry_gap": 0.495,
            "pre_drum_cont4": 0.196,
            "post_drum8": 1.0,
            "post_bass8": 0.525,
            "pre4_height": 0.311,
            "jump4": 0.283,
        },
    )

    assert marker == pytest.approx(44.55789569160997)


def test_zoomed_marker_uses_transition_edge_after_short_prehit() -> None:
    marker = _zoomed_marker_time(
        26.666666666666668,
        {
            "microaligned_time": 26.67698412698413,
            "impact_body_time": 26.69185941043084,
            "impact_body_quality": 0.622128910283896,
            "micro_confidence": 0.9418233714580599,
            "attack_cleanliness": 0.9955085737125926,
            "attack_peak_strength": 1.0,
            "impact_boundary_confidence": 0.7985634703079503,
            "denoised_impact_strength": 0.617832536572747,
            "rms_rise_score": 1.0,
            "peak_rise_score": 1.0,
            "zero_crossing_quality": 0.6578543361900151,
        },
        {
            "clock_bar": 17,
            "pre4_height": 0.40207146339017075,
            "jump4": 0.25092718557370797,
            "post_bass8": 0.574879342238052,
            "post_drum8": 1.0,
            "post_inst8": 0.13779078708566878,
            "pre_drum_cont4": 0.004807692307692308,
            "local_reentry": True,
            "local_reentry_gap": 0.3189355146915343,
            "phrase_prior": 0.94,
        },
    )

    assert marker == pytest.approx(26.690520634020638)


@pytest.mark.parametrize(
    ("audio_path", "target", "tolerance"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/borne - Silence/drums_140_5A_6-borne - Silence.flac",
            44.57104308390023,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/WITH U - SVDKO, MAYV/drums_140_5A_6-WITH U - SVDKO, MAYV.flac",
            57.48180839002268,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/Xotix - BUTTERFACE/drums_140_5A_6-Xotix - BUTTERFACE.flac",
            15.742012471655329,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/CloZee, David Starfire - Soma Dreams - CloZee remix/drums_140_5A_7-CloZee, David Starfire - Soma Dreams - CloZee remix.flac",
            73.536,
            0.12,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/Ganja White Night - Blackberries/drums_140_5A_7-Ganja White Night - Blackberries.flac",
            82.23027210884354,
            0.12,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/Perception Disorder - Alix Perez/drums_140_5A_7-Perception Disorder - Alix Perez.flac",
            54.857142857142854,
            0.12,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/6A/SHOSH - Dizzy/drums_140_6A_5-SHOSH - Dizzy.flac",
            27.428503401360544,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/DO IT - Currly/drums_140_7A_7-DO IT - Currly.wav",
            29.142857142857142,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/LYNY, FLY - Pulse/drums_140_7A_7-LYNY, FLY - Pulse.flac",
            27.427868480725625,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Mirage - Phrva, Arya/drums_140_7A_7-Mirage - Phrva, Arya.flac",
            54.857561,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Seth David - Sexy Mushroom Lady/drums_140_7A_7-Seth David - Sexy Mushroom Lady.flac",
            13.716208616780046,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Wanna Go - Phrva/drums_140_7A_7-Wanna Go - Phrva.wav",
            27.42625850340136,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Bassnectar, PEEKABOO - Disrupt The System - Underground Mix/drums_140_7A_8-Bassnectar, PEEKABOO - Disrupt The System - Underground Mix.flac",
            0.0,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Seance - PEEKABOO/drums_140_7A_8-Seance - PEEKABOO.flac",
            68.573968,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/8A/Mersiv - Depth Perception/drums_140_8A_6-Mersiv - Depth Perception.flac",
            96.0,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Ravenscoon - Free Your Mind/drums_140_9A_8-Ravenscoon - Free Your Mind.flac",
            58.239864,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/142/9A/When I Look at You - Emalkay/drums_142_9A_8-When I Look at You - Emalkay.flac",
            1.231360544217687,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/144/12A/Covex, Lexi Shanley, SVDKO - I Miss U - SVDKO Remix/drums_144_12A_6-Covex, Lexi Shanley, SVDKO - I Miss U - SVDKO Remix.flac",
            43.334535,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/144/4A/Saint Mary’s Lake - Mfinity/drums_144_4A_7-Saint Mary’s Lake - Mfinity.flac",
            26.690431,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/144/7A/Move Your Body - Nyrus Dub - Morillo, Scott Nice, Nyrus/drums_144_7A_6-Move Your Body - Nyrus Dub - Morillo, Scott Nice, Nyrus.wav",
            28.33298866213152,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/144/9A/Jkyl & Hyde - Sensory Session/drums_144_9A_3-Jkyl & Hyde - Sensory Session.flac",
            59.19466666666666,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/145/11A/Lapalux - Don't Mean A Thing/drums_145_11A_5-Lapalux - Don't Mean A Thing.flac",
            149.477101,
            0.08,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/145/6A/Lana Del Rey - Doin' Time/drums_145_6A_5-Lana Del Rey - Doin' Time.flac",
            30.059043083900225,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/146/7A/SubDocta - Torqued/drums_146_7A_8-SubDocta - Torqued.flac",
            26.495912698412695,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/146/8A/Tape B - About Me Too/drums_146_8A_6-Tape B - About Me Too.flac",
            53.56651360544218,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/146/9A/Mo Bamba - Sheck Wes/drums_146_9A_5-Mo Bamba - Sheck Wes.wav",
            32.89948752834467,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/146/9B/jigitz - make believe/drums_146_9B_5-jigitz - make believe.flac",
            47.671233,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/147/7A/Decktrik, Ashez - YALLAH - Ashez Remix/drums_147_7A_6-Decktrik, Ashez - YALLAH - Ashez Remix.flac",
            24.48979591836735,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/148/12A/Funk Tribu - DR34M$/drums_148_12A_5-Funk Tribu - DR34M$.flac",
            26.283859410430836,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/148/5A/Depths - The Rad Hatter, Peace Sine/drums_148_5A_5-Depths - The Rad Hatter, Peace Sine.wav",
            25.94542857142857,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/148/7A/La Câlin - Serhat Durmus/drums_148_7A_4-La Câlin - Serhat Durmus.wav",
            25.999093,
            0.01,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/152/9A/Lokal - Like We/drums_152_9A_5-Lokal - Like We.flac",
            56.318520706528226,
            0.04,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/152/9A/Nothing More - Monuman/drums_152_9A_6-Nothing More - Monuman.wav",
            50.528231,
            0.01,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/156/7A/I Got Something - rSUN/drums_156_7A_7-I Got Something - rSUN.wav",
            12.697710,
            0.005,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/158/10A/Know Good - Dust/drums_158_10A_5-Know Good - Dust.flac",
            48.607594936708864,
            0.01,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Tomb (feat. Rasha Kamal) - Fady Haroun, Adrinaline, Rasha Kamal/drums_140_9A_8-Tomb (feat. Rasha Kamal) - Fady Haroun, Adrinaline, Rasha Kamal.wav",
            13.714285714285714,
            0.08,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Kito - take your vibes and go - Taiki Nulight Remix/drums_140_9A_8-Kito - take your vibes and go - Taiki Nulight Remix.flac",
            55.719705,
            0.05,
        ),
    ],
)
def test_visual_first_matches_saved_webgui_examples(audio_path: str, target: float, tolerance: float) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    _assert_visual_first_production_contract(result)


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/100/5A/AHEE, Sømething - The Action/drums_100_5A_7-AHEE, Sømething - The Action.flac",
            38.400016666666666,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/100/5A/The Action - AHEE, Sømething/drums_100_5A_7-The Action - AHEE, Sømething.wav",
            38.40000833333333,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/4A/AC Slater - My Peoples/drums_126_4A_6-AC Slater - My Peoples.flac",
            60.98485260770975,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/4A/AC Slater, Taiki Nulight - My Peoples/drums_126_4A_6-AC Slater, Taiki Nulight - My Peoples.flac",
            60.98485260770975,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/10A/WHIPPED CREAM - Rewind.. (But I Love You)/drums_130_10A_6-WHIPPED CREAM - Rewind.. (But I Love You).flac",
            55.46077097505669,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/SVDKO, MAYV - WITH U/drums_140_5A_6-SVDKO, MAYV - WITH U.flac",
            54.85845804988662,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/WITH U - SVDKO, MAYV/drums_140_5A_6-WITH U - SVDKO, MAYV.flac",
            54.857142857142854,
        ),
    ],
)
def test_visual_first_no_longer_needs_v6_human_review_overrides(audio_path: str, target: float) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = _assert_visual_first_production_contract(result)

    assert selected["selected_by"] != "visual_validated_human_review_override"


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/9A/Skrillex, Habstrakt - Chicken Soup/drums_126_9A_7-Skrillex, Habstrakt - Chicken Soup.flac",
            55.39809523809523,
            "visual_audit_boom_replacement",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/79/5A/Aerials - System of a Down/drums_79_5A_5-Aerials - System of a Down.flac",
            30.37974683544304,
            "visual_audit_boom_replacement",
        ),
    ],
)
def test_visual_first_promotes_audit_proven_boom_front_edge(audio_path: str, target: float, selected_by: str) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    _assert_visual_first_production_contract(result)

    assert result["boom_proof"]["nearest"]["contains_marker"] is True


def test_visual_first_promotes_far_earlier_proven_drop_over_stale_late_marker() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Seth David - Sexy Mushroom Lady/drums_140_7A_7-Seth David - Sexy Mushroom Lady.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    _assert_visual_first_production_contract(result)


def test_visual_first_repairs_blank_grid_snap_to_gui_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/79/3A/Thouxanbanfauni - STYRO STAINS/drums_79_3A_5-Thouxanbanfauni - STYRO STAINS.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = _assert_visual_first_production_contract(result)

    assert result["marker"] == pytest.approx(24.351904044318154, abs=0.006)
    assert selected["selected_by"] != "visual_full_track_stronger_boom_front_edge"
    assert result["marker"] != pytest.approx(9.111001133040691, abs=0.250)


def test_visual_first_repairs_stale_aerials_context_to_zoomed_gui_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/79/5A/Aerials - System of a Down/drums_79_5A_5-Aerials - System of a Down.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    _assert_visual_first_production_contract(result)

    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    assert result["marker"] == pytest.approx(32.597426303854874, abs=0.006)
    assert selected["selected_by"] == "visual_absolute_final_context_fine_front_edge_repair"
    assert visual["context_local_fine_front_edge_repair"] is True
    assert visual["context_local_fine_seed_marker"] == pytest.approx(32.612426303854875, abs=0.001)
    assert visual["context_local_fine_previous_marker"] == pytest.approx(151.75401360544217, abs=0.001)
    assert result["gui_mask_proof"].get("fine_front_edge_passes") is True
    assert abs(float(result["boom_proof"]["nearest"]["offset_sec"])) <= 0.040


def test_visual_first_repairs_coarse_gui_pass_to_zoomed_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/79/6A/Kid Cudi - Solo Dolo (Nightmare)/"
        "drums_79_6A_3-Kid Cudi - Solo Dolo (Nightmare).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(47.31096485260772, abs=0.006)
    assert selected["selected_by"] == "visual_final_contract_visible_signal_repair"
    assert visual["final_blank_signal_gui_front_edge_repair"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert result["gui_mask_proof"]["marker_signal_present"] is True
    assert result["boom_proof"]["passes"] is True


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/158/10A/Know Good - Dust/"
            "drums_158_10A_5-Know Good - Dust.flac",
            48.607594936708864,
            "visual_boom_earliest_dominant_replacement",
        ),
    ],
)
def test_visual_first_proven_boom_marker_uses_stable_wide_zoom_relief(
    audio_path: str,
    target: float,
    selected_by: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(target, abs=0.002)
    assert selected["selected_by"] == selected_by
    assert result["gui_mask_proof"]["accepted_by_full_track_stable_wide_zoom_conflict"] is True
    assert selected["visual_components"]["full_track_stable_wide_zoom_conflict_relief"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_keeps_first_phrase_body_over_later_in_section_point() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/3A/Apashe, Wasiu, CloZee - Majesty - CloZee Remix/"
        "drums_80_3A_7-Apashe, Wasiu, CloZee - Majesty - CloZee Remix.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(72.0, abs=0.020)
    assert selected["selected_by"] == "visual_gui_first_fat_block"
    assert result["marker"] != pytest.approx(84.1233973613079, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_preserves_first_clear_early_drop_over_later_darker_section() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/156/7A/I Got Something - rSUN/"
        "drums_156_7A_7-I Got Something - rSUN.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(12.620192307692308, abs=0.006)
    assert selected["selected_by"] == "visual_early_clear_drop_context_front_edge_repair"
    assert visual["early_clear_drop_reclaim"] is True
    assert visual["early_clear_drop_replaced_marker"] == pytest.approx(51.15423076923077, abs=0.001)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_does_not_restore_sparse_pulse_before_dominant_later_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/85/5A/Josh Teed - The Jungle/"
        "drums_85_5A_7-Josh Teed - The Jungle.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected.get("visual_components", {})

    assert result["marker"] == pytest.approx(141.1756243623965, abs=0.020)
    assert selected["selected_by"] == "visual_full_track_stronger_boom_front_edge"
    assert visual.get("early_clear_drop_reclaim") is not True
    assert result["marker"] != pytest.approx(22.58823529411765, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_vetoes_early_reclaim_when_immediate_body_is_later() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/85/6A/An-Ten-Nae - Always Find the Hidden Path/"
        "drums_85_6A_6-An-Ten-Nae - Always Find the Hidden Path.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected.get("visual_components", {})

    assert result["marker"] == pytest.approx(43.917238827530646, abs=0.020)
    assert selected["selected_by"] == "visual_full_track_stronger_boom_front_edge"
    assert visual.get("early_clear_drop_reclaim") is not True
    assert result["marker"] != pytest.approx(22.590235294117647, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/4A/G Jones, Minnesota - Thunderdome/"
            "drums_80_4A_7-G Jones, Minnesota - Thunderdome.flac",
            97.131,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/4A/Flozone - South/"
            "drums_80_4A_8-Flozone - South.flac",
            35.998548752834466,
        ),
    ],
)
def test_visual_first_does_not_reclaim_sparse_intro_pulse_before_dense_drop(
    audio_path: str,
    target: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected.get("visual_components", {})

    assert result["marker"] == pytest.approx(target, abs=0.250)
    assert selected["selected_by"] != "visual_early_clear_drop_context_front_edge_repair"
    assert visual.get("early_clear_drop_reclaim") is not True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_replaces_relevant_signal_only_intro_pulse_with_sustained_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Rage Money Power - ZISO/"
        "drums_80_9A_5-Rage Money Power - ZISO.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(78.07709526298335, abs=0.020)
    assert selected["selected_by"] == "visual_final_contract_visible_signal_repair"
    assert visual["full_track_stronger_selected_relevant_signal_only"] is True
    assert visual["full_track_stronger_previous_marker"] == pytest.approx(26.923537414965985, abs=0.002)
    assert visual["full_track_stronger_repair_kind"] == "first_body_after_weak_marker"
    assert result["marker"] > 70.0
    assert result["marker"] < 82.0
    assert result["marker"] != pytest.approx(26.891037414965986, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_prefers_first_credible_body_before_weak_marker_over_darker_repeat() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/4A/Beats Antique, Fanfara Kalashnikov - Oriental Uno/"
        "drums_80_4A_7-Beats Antique, Fanfara Kalashnikov - Oriental Uno.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(7.905069648190733, abs=0.020)
    assert selected["selected_by"] in {
        "visual_full_track_stronger_boom_front_edge",
        "visual_gui_boom_front_edge_contract",
    }
    if selected["selected_by"] == "visual_full_track_stronger_boom_front_edge":
        assert visual["full_track_stronger_repair_kind"] == "first_body_before_weak_marker"
    else:
        assert result["gui_mask_proof"]["passes"] is True
    assert result["marker"] < 10.0
    assert result["marker"] != pytest.approx(13.905122512092618, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_restores_sparse_staccato_first_body_before_later_darker_repeat() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/6A/CloZee, LSDREAM - MOONLIGHT/"
        "drums_80_6A_5-CloZee, LSDREAM - MOONLIGHT.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    gui_proof = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(33.0, abs=0.030)
    assert selected["selected_by"] == "visual_first_credible_body_front_edge_repair"
    assert result["marker"] < 40.0
    assert result["marker"] != pytest.approx(111.0, abs=0.500)
    assert result["marker"] != pytest.approx(198.031, abs=0.500)
    assert visual["first_credible_body_front_edge_repair"] is True
    assert gui_proof["accepted_by_staccato_front_body_proof"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_restores_nearby_first_body_over_later_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/11A/Teddy Swims - Lose Control/"
        "drums_80_11A_6-Teddy Swims - Lose Control.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    gui_proof = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(36.2, abs=0.060)
    assert selected["selected_by"] == "visual_first_credible_body_front_edge_repair"
    assert result["marker"] < 40.0
    assert result["marker"] != pytest.approx(42.837, abs=0.500)
    assert result["marker"] != pytest.approx(96.719, abs=0.500)
    assert visual["first_credible_body_front_edge_repair"] is True
    assert gui_proof["accepted_by_staccato_front_body_proof"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_accepts_minimal_sparse_fine_front_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/87/9A/Dub Phizix, Skeptical, Strategy - Marka/"
        "drums_87_9A_6-Dub Phizix, Skeptical, Strategy - Marka.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    gui_proof = result["gui_mask_proof"]

    assert result["marker"] == pytest.approx(66.185, abs=0.050)
    assert selected["selected_by"] == "visual_gui_first_fat_block"
    assert gui_proof["accepted_by_staccato_front_body_proof"] is True
    assert gui_proof["staccato_front_fine_sparse_body"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_replaces_stale_v2_with_sustained_body_section_front() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/90/8A/Emancipator - Ghost Pong/"
        "drums_90_8A_5-Emancipator - Ghost Pong.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    gui_proof = result["gui_mask_proof"]
    boom_proof = result["boom_proof"]

    assert result["marker"] == pytest.approx(66.667, abs=0.050)
    assert selected["selected_by"] == "visual_sustained_body_section_front_edge"
    assert boom_proof["accepted_by_sustained_body_section_boom_proof"] is True
    assert gui_proof["accepted_by_sustained_body_section_gui_proof"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert boom_proof["passes"] is True
    assert gui_proof["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_preserves_gui_first_fat_block_over_intro_like_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Smoakland - Set You Free/"
        "drums_80_9A_7-Smoakland - Set You Free.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(23.984, abs=0.030)
    assert selected["selected_by"] == "visual_gui_first_fat_block"
    assert result["marker"] != pytest.approx(12.783, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_keeps_contextual_first_body_over_late_end_cluster() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/82/7A/Swangin - HEXED/"
        "drums_82_7A_8-Swangin - HEXED.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    assert result["marker"] == pytest.approx(46.908268292682926, abs=0.020)
    assert result["selected_candidate"]["selected_by"] == "visual_final_contract_visible_signal_repair"
    assert result["marker"] < 60.0
    assert result["marker"] != pytest.approx(164.41939241548536, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_promotes_audit_preferred_earlier_phrase_body_over_late_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/7A/Champagne Drip - Oni/"
        "drums_80_7A_7-Champagne Drip - Oni.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(72.13278911564626, abs=0.020)
    assert selected["selected_by"] in {
        "visual_audit_earlier_phrase_body_replacement",
        "visual_final_contract_gui_sparse_pulse_repair",
    }
    assert (
        visual.get("final_audit_preferred_earlier_phrase_body_repair") is True
        or selected["selected_by"] == "visual_final_contract_gui_sparse_pulse_repair"
    )
    assert result["marker"] != pytest.approx(174.84357702070767, abs=0.250)
    assert result["marker"] != pytest.approx(69.0, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_keeps_first_body_instead_of_late_definitive_repeat() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Saturna, CØNTRA, Savej - What I Can Do - Savej Remix/"
        "drums_80_9A_6-Saturna, CØNTRA, Savej - What I Can Do - Savej Remix.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    assert result.get("ok") is True
    assert result.get("error") is None
    assert result["marker"] == pytest.approx(35.97786848072562, abs=0.020)
    assert result["selected_candidate"]["selected_by"] == "visual_gui_first_fat_block"
    assert result["marker"] < 40.0
    assert result["marker"] != pytest.approx(168.00210657596372, abs=0.250)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


def test_visual_first_accepts_clean_coarse_gui_edge_when_zoom_has_no_better_body_mask() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/CLUSTER - Muadeep/"
        "drums_80_9A_6-CLUSTER - Muadeep.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    assert result.get("ok") is True
    assert result.get("error") is None
    assert result["marker"] == pytest.approx(29.974557823129253, abs=0.002)
    assert result["selected_candidate"]["selected_by"] == "visual_candidate_selector"
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert result["gui_mask_proof"]["accepted_by_full_track_stable_wide_zoom_conflict"] is True


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/127/9A/Peanut Butter Jelly - T.I., Young Thug, Young Dro/drums_127_9A_8-Peanut Butter Jelly - T.I., Young Thug, Young Dro.wav",
            29.34047244094488,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/1A/YOOKiE - FiGHT FOR YOUR RiGHT/drums_130_1A_6-YOOKiE - FiGHT FOR YOUR RiGHT.flac",
            23.66523076923077,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Pink Peonies - Zingara/drums_80_9A_5-Pink Peonies - Zingara.wav",
            18.0,
        ),
    ],
)
def test_visual_first_moves_fast_off_one_marker_to_earliest_dominant_boom(audio_path: str, target: float) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )

    _assert_visual_first_production_contract(result)


@pytest.mark.parametrize(
    "audio_path",
    [
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/WITH U - SVDKO, MAYV/drums_140_5A_6-WITH U - SVDKO, MAYV.flac",
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Power - borne/drums_140_7A_7-Power - borne.flac",
    ],
)
def test_visual_first_attaches_boom_proof_before_visual_audit(audio_path: str) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = _assert_visual_first_production_contract(result)

    assert selected["visual_components"]["boom_proof_pass"] is True
    assert _gui_mask_proof(audio_path, result["marker"])["passes"] is True


def test_visual_first_vetoes_blank_snap_and_keeps_gui_boom_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/LIKE ME - Nikita, the Wicked/drums_140_7A_6-LIKE ME - Nikita, the Wicked.wav"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(55.67085714285714, abs=0.012)
    assert selected["selected_by"] == "visual_boom_grid_one_snap"
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert selected["bpm_clock"]["on_one"] is True
    assert selected["boom_proof"]["passes"] is True
    assert _gui_mask_proof(audio_path, 68.57393990929705)["passes"] is True
    assert abs(float(result["boom_proof"]["nearest"]["offset_sec"])) <= 0.040


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/122/7A/ZHU, partywithray - Zhudio54/drums_122_7A_6-ZHU, partywithray - Zhudio54.flac",
            19.477131147540984,
            "visual_final_contract_gui_mask_nearest_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/4A/Chris Lake - 400/drums_125_4A_6-Chris Lake - 400.flac",
            15.584999999999999,
            "visual_final_contract_gui_mask_nearest_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/6A/ero808, NXSTY, RYA - SHAKE A LIL’ SOMETHIN’ (Original Mix)/drums_130_6A_7-ero808, NXSTY, RYA - SHAKE A LIL’ SOMETHIN’ (Original Mix).flac",
            88.01337553081838,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/131/2A/Future - LIL DEMON/drums_131_2A_6-Future - LIL DEMON.flac",
            61.87631643458712,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/131/9A/Sunday Scaries, XKYLAR - Want That (Extended Mix)/drums_131_9A_7-Sunday Scaries, XKYLAR - Want That (Extended Mix).flac",
            102.66791984732825,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/135/4A/ero808, Manila Killa - BABYLON/drums_135_4A_6-ero808, Manila Killa - BABYLON.flac",
            47.87100973493546,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Tape B - Backspin Bass/drums_140_9A_7-Tape B - Backspin Bass.flac",
            102.8571655328798,
            "visual_later_definitive_drop_guard",
        ),
    ],
)
def test_visual_first_resolves_full_library_gui_contract_blockers(audio_path: str, target: float, selected_by: str) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(target, abs=0.012)
    assert selected["selected_by"] == selected_by
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    gui_proof = _gui_mask_proof(audio_path, result["marker"])
    boom_offset = abs(float(result["boom_proof"]["nearest"]["offset_sec"]))
    assert gui_proof["passes"] is True or boom_offset <= 0.040
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by", "min_marker"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/84/4A/HARD - SOPHIE/drums_84_4A_8-HARD - SOPHIE.wav",
            23.015379912271452,
            "visual_full_track_stronger_boom_front_edge",
            20.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/84/9A/Silk Static - Bloom Tender/drums_84_9A_6-Silk Static - Bloom Tender.flac",
            33.93278107883175,
            "visual_final_contract_gui_front_edge_repair",
            20.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/85/9A/An-Ten-Nae - Way Up/drums_85_9A_4-An-Ten-Nae - Way Up.flac",
            52.115669349511535,
            "visual_full_track_stronger_boom_front_edge",
            30.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/88/12A/NOTION, Cameron Hayes - SECRETS/drums_88_12A_6-NOTION, Cameron Hayes - SECRETS.flac",
            104.16294311965711,
            "visual_final_contract_gui_front_edge_repair",
            90.0,
        ),
    ],
)
def test_visual_first_final_strict_recovery_uses_dominant_contract_front_edge(
    audio_path: str,
    target: float,
    selected_by: str,
    min_marker: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    gui_proof = result["gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(target, abs=0.020)
    assert result["marker"] >= min_marker
    assert selected["selected_by"] == selected_by
    assert selected["visual_components"]["final_strict_contract_pool_recovery"] is True
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_proof["marker_signal_present"] is True
    assert gui_boom_mask_strict_contract_issue(gui_proof) is None
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by", "requires_wide_proof"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/103/9B/Bien Loco - Nova y Jory/drums_103_9B_4-Bien Loco - Nova y Jory.wav",
            18.385679611650485,
            "visual_final_contract_gui_mask_nearest_repair",
            False,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Flow - rSUN/drums_140_9A_5-Flow - rSUN.wav",
            41.99366213151928,
            "visual_final_contract_gui_mask_nearest_repair",
            False,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/Marauder - dêtre/drums_140_2A_7-Marauder - dêtre.flac",
            27.429727891156464,
            "visual_gui_first_fat_block",
            True,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Criso - Marked/drums_140_7A_6-Criso - Marked.flac",
            27.428571428571427,
            "visual_candidate_selector",
            False,
        ),
    ],
)
def test_visual_first_repairs_full_library_44100_gui_contract_holds(
    audio_path: str,
    target: float,
    selected_by: str,
    requires_wide_proof: bool,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]

    assert result["marker"] == pytest.approx(target, abs=0.012)
    assert selected["selected_by"] == selected_by
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert bool(gui_proof.get("stable_wide_gui_mask_proof")) is requires_wide_proof
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "stale_marker"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/123/9A/NXSTY, Coka Cobra, One True God - No Promises/drums_123_9A_6-NXSTY, Coka Cobra, One True God - No Promises.flac",
            91.70731707317073,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/134/5A/Cocky - Mirror Maze, Niles/drums_134_5A_6-Cocky - Mirror Maze, Niles.flac",
            40.22666859044559,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/136/6A/Darby, OOTORO - PARTY/drums_136_6A_7-Darby, OOTORO - PARTY.flac",
            14.86077097505669,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/Ready - Klinical, Killa P/drums_140_2A_6-Ready - Klinical, Killa P.flac",
            152.3381540769248,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/144/2A/Tech N9ne - Demons/drums_144_2A_8-Tech N9ne - Demons.flac",
            104.47636388888888,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/150/9A/DVBBS, Wiz Khalifa, Urfavxboyfriend, Goldsoul - SH SH SH (Hit That) (feat. Wiz Khalifa, Urfavxboyfriend & Goldsoul)/drums_150_9A_7-DVBBS, Wiz Khalifa, Urfavxboyfriend, Goldsoul - SH SH SH (Hit That) (feat. Wiz Khalifa, Urfavxboyfriend & Goldsoul).flac",
            28.622909656161863,
        ),
    ],
)
def test_visual_first_repairs_wide_exact_boom_gui_relief_to_raw_gui_front_edge(
    audio_path: str,
    stale_marker: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]
    nearest_offset = float(gui_proof.get("nearest_placeable_offset_sec") or 0.0)

    assert result["ok"] is True
    assert abs(float(result["marker"]) - stale_marker) > 0.040
    assert selected["selected_by"].startswith("visual_")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_proof.get("accepted_by_boom_front_edge_proof") is None
    assert visual_first_module._gui_mask_proof_needs_front_edge_repair(gui_proof) is False
    assert abs(nearest_offset) <= 0.040
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_metaform_crush_repairs_final_gui_offset_to_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/100/12A/Metaform - Crush/drums_100_12A_4-Metaform - Crush.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(76.745, abs=0.003)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_proof.get("accepted_by_coarse_gui_front_edge_after_fine_probe") is True
    assert visual_first_module._gui_mask_proof_needs_front_edge_repair(
        gui_proof,
        trust_reasonless_relief_offset=True,
    ) is False
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_saturna_keeps_clean_coarse_front_edge_over_fine_texture_offset() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Saturna, CØNTRA, Savej - What I Can Do - Savej Remix/drums_80_9A_6-Saturna, CØNTRA, Savej - What I Can Do - Savej Remix.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]
    coarse_gui = gui_proof["coarse_gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(35.97786848072562, abs=0.003)
    assert selected["selected_by"] == "visual_gui_first_fat_block"
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_proof.get("coarse_passed_but_zoomed_front_edge_failed") is True
    assert coarse_gui["passes"] is True
    assert abs(float(coarse_gui["nearest_placeable_offset_sec"])) <= 0.001
    assert visual_first_module._gui_mask_proof_needs_front_edge_repair(
        gui_proof,
        trust_reasonless_relief_offset=True,
    ) is False
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "selected_by"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/4A/The American Dollar - Age Of Wonder/drums_80_4A_4-The American Dollar - Age Of Wonder.flac",
            24.130862811791385,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/94/4A/Pharoahe Monch - Simon Says/drums_94_4A_5-Pharoahe Monch - Simon Says.flac",
            46.816,
            "visual_final_contract_gui_front_edge_repair",
        ),
    ],
)
def test_visual_first_repairs_blank_markers_to_visible_gui_front_edges(
    audio_path: str,
    target: float,
    selected_by: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(target, abs=0.012)
    assert selected["selected_by"] == selected_by
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_proof.get("accepted_by_actual_visual_body_proof") is None
    assert int(gui_proof.get("placeable_count") or 0) > 0
    assert abs(float(gui_proof.get("nearest_placeable_offset_sec") or 0.0)) <= 0.010
    assert selected["visual_components"]["actual_visual_body_contract_pass"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize("relevance_value", [False, None])
def test_actual_body_gui_relief_requires_relevant_gui_mask(relevance_value: bool | None) -> None:
    gui_proof = {
        "passes": False,
        "marker_signal_present": True,
        "marker_placeable_mask": False,
        "reasons": ["marker_not_on_gui_boom_relevant_mask"],
    }
    if relevance_value is not None:
        gui_proof["marker_relevant_mask"] = relevance_value
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.0, "contains_marker": True},
        "nearest_profile": {"profile_score": 0.90, "passes_profile": True},
    }
    candidate = {
        "timestamp": 32.0,
        "visual_components": {
            "actual_visual_body": 0.92,
            "section_body_score": 0.90,
            "boom_section_darkness": 0.90,
            "post8_height": 0.90,
            "max_post8_height": 0.90,
            "drum_density": 0.95,
            "drum_continuity": 0.80,
            "bass_low_energy": 0.70,
            "drum_bass_simultaneity": 0.85,
            "drop_transition_score": 0.80,
        },
    }

    relieved = visual_first_module._accept_gui_contract_by_actual_body_proof(
        gui_proof,
        boom_proof,
        candidate,
    )

    assert relieved["passes"] is False
    assert relieved.get("accepted_by_actual_visual_body_proof") is None


def test_actual_body_gui_relief_allows_tiny_profile_rounding_margin() -> None:
    gui_proof = {
        "passes": False,
        "marker_signal_present": True,
        "marker_relevant_mask": True,
        "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.0, "contains_marker": True},
        "nearest_profile": {
            "profile_score": 0.5999685819868473,
            "passes_profile": True,
            "metrics": {
                "profile_score": 0.5999685819868473,
                "body_score": 0.90,
                "transition_score": 0.80,
                "darkness": 0.90,
                "post8_height": 0.90,
                "post_bass8": 0.70,
                "post_drum8": 0.95,
                "post_drum_cont8": 0.80,
                "simultaneity": 0.85,
            },
        },
    }
    candidate = {
        "timestamp": 32.0,
        "boom_proof": boom_proof,
        "visual_components": {
            "actual_visual_body": 0.92,
            "section_body_score": 0.90,
            "boom_section_darkness": 0.90,
            "boom_body_section": True,
            "boom_section_simultaneity": 0.85,
            "post8_height": 0.90,
            "max_post8_height": 0.90,
            "post_bass8": 0.70,
            "post_drum8": 0.95,
            "post_drum_cont8": 0.80,
            "phrase_prior": 0.90,
            "pre_drum_cont4": 0.0,
            "drop_transition_score": 0.80,
        },
    }

    relieved = visual_first_module._accept_gui_contract_by_actual_body_proof(
        gui_proof,
        boom_proof,
        candidate,
    )

    assert relieved["passes"] is True
    assert relieved["accepted_by_actual_visual_body_proof"] is True
    assert relieved["raw_reasons"] == ["marker_not_on_gui_boom_front_edge_mask"]


def test_sparse_groove_gui_relief_requires_current_placeable_signal() -> None:
    gui_proof = {
        "passes": True,
        "reasons": [],
        "marker_signal_present": True,
        "marker_relevant_mask": True,
        "marker_immediate_body_present": False,
        "nearest_placeable_offset_sec": 0.0,
        "placeable_count": 64,
        "marker_rms_max": 0.31,
        "marker_post_relevant_occupancy_500ms": 0.080,
        "marker_post_rms_max_500ms": 0.31,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.0, "contains_marker": True},
        "nearest_profile": {"profile_score": 0.52, "passes_profile": True},
    }
    candidate = {
        "timestamp": 49.1039,
        "bpm_clock": {"on_one": True, "one_distance_ms": 0.0},
        "visual_components": {
            "bar_height": 0.520,
            "body_score": 0.585,
            "post8_height": 0.594,
            "post_bass8": 0.358,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.747,
            "boom_section_darkness": 0.594,
            "boom_section_simultaneity": 0.682,
        },
    }

    relieved = visual_first_module._accept_gui_contract_by_sparse_groove_front_edge_proof(
        gui_proof,
        boom_proof,
        candidate,
    )
    assert relieved["passes"] is True
    assert relieved["accepted_by_sparse_groove_front_edge_proof"] is True
    assert gui_boom_mask_strict_contract_issue(relieved) is None

    blank_gui = {**gui_proof, "marker_signal_present": False}
    rejected = visual_first_module._accept_gui_contract_by_sparse_groove_front_edge_proof(
        blank_gui,
        boom_proof,
        candidate,
    )
    assert rejected.get("accepted_by_sparse_groove_front_edge_proof") is None


def test_visual_first_prefers_first_credible_body_over_later_sparse_groove_signal() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/97/6A/"
        "A Tribe Called Quest - Award Tour (feat. Trugoy The Dove)/"
        "drums_97_6A_6-A Tribe Called Quest - Award Tour (feat. Trugoy The Dove).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(42.3505, abs=0.012)
    assert selected["selected_by"] == "visual_final_contract_gui_front_edge_repair"
    assert result["boom_proof"]["passes"] is True
    assert gui_proof["passes"] is True
    assert gui_boom_mask_strict_contract_issue(gui_proof) is None
    assert selected["bpm_clock"]["on_one"] is True
    assert gui_proof["marker_signal_present"] is True


def test_visual_first_rejects_stable_wide_sparse_opening_without_local_body() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/101/4A/Intoxicated - Aaryan Shah/drums_101_4A_4-Intoxicated - Aaryan Shah.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    gui_proof = selected["gui_mask_proof"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(53.472, abs=0.012)
    assert selected["selected_by"] == "visual_final_contract_actual_body_boom_repair"
    assert gui_proof["passes"] is True
    assert gui_proof["marker_body_mask"] is True
    assert float(gui_proof["marker_body_density_max"]) >= 0.120
    assert result["boom_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_near_empty_drums_stem_is_excluded_instead_of_forced_to_drop_marker() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/176/10B/Drone - Pale Blue/drums_176_10B_2-Drone - Pale Blue.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    stats = drums_stem_signal_stats(audio_path)

    assert is_near_empty_drums_stem(audio_path, stats=stats) is True
    assert stats["peak_abs"] < stats["near_empty_peak_floor"]
    assert stats["window_rms_p99"] < stats["near_empty_window_rms_p99_floor"]


def test_visual_first_does_not_restore_same_bar_front_edge_off_the_one() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/132/5A/Pìjus - Daydreamin/drums_132_5A_6-Pìjus - Daydreamin.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(58.18358276643991, abs=0.012)
    assert selected["selected_by"] != "visual_same_bar_proven_front_edge_restore"
    assert result["visual_audit"]["status"] == "pass"
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_replaces_weak_recovered_candidate_with_visible_gui_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/137/2A/Dopplereffekt - Rocket Scientist/drums_137_2A_6-Dopplereffekt - Rocket Scientist.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(21.00266878527567, abs=0.012)
    assert selected["selected_by"] == "visual_early_clear_drop_context_front_edge_repair"
    assert result["visual_audit"]["status"] == "pass"
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["visual_components"]["actual_visual_body_contract_pass"] is True


def test_visual_first_repairs_to_gui_placeable_later_body_when_early_body_has_no_gui_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/Ravenscoon, CASHFORGOLD - Awake/drums_140_2A_8-Ravenscoon, CASHFORGOLD - Awake.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(68.5965022675737, abs=0.012)
    assert selected["selected_by"] == "visual_final_contract_gui_front_edge_repair"
    assert "analysis_rate_recovery" not in result
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["visual_audit"]["absolute_contract_ignored_unproven_preferred"] is True
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["placeable_count"] > 0
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_repairs_44100_marker_with_proof_clean_direct_result() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/The Widdler - System Failure/drums_140_9A_7-The Widdler - System Failure.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(54.857142857142854, abs=0.012)
    assert selected["selected_by"] == "visual_same_bar_proven_front_edge_restore"
    assert "analysis_rate_recovery" not in result
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["placeable_count"] > 0
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_sparse_impact_blank_marker_repairs_to_later_gui_body() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/94/4A/Pharoahe Monch - Simon Says/drums_94_4A_5-Pharoahe Monch - Simon Says.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(46.816, abs=0.012)
    assert selected["selected_by"] == "visual_final_contract_gui_front_edge_repair"
    assert result["boom_proof"].get("accepted_by_sparse_impact_boom_proof") is None
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"].get("accepted_by_actual_visual_body_proof") is None
    assert int(selected["gui_mask_proof"].get("placeable_count") or 0) > 0


def test_visual_first_final_blank_signal_repair_moves_to_visible_gui_front_edge() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/11A/AC Slater, Chris Lorenzo, Wax Motif - Fly Kicks - Wax Motif Remix/drums_125_11A_7-AC Slater, Chris Lorenzo, Wax Motif - Fly Kicks - Wax Motif Remix.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["ok"] is True
    assert result["marker"] == pytest.approx(92.1967573696145, abs=0.050)
    assert selected["selected_by"] == "visual_final_first_true_body_before_later_repeat"
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert result["gui_mask_proof"]["marker_signal_present"] is True
    assert visual["final_first_true_body_before_later_repeat"] is True
    assert visual["final_first_true_body_previous_marker"] == pytest.approx(123.42566893424036, abs=0.001)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_repairs_to_stronger_adjacent_gui_body_when_audit_preferred_edge_is_unproven() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/7A/Champagne Drip - Oni/drums_80_7A_7-Champagne Drip - Oni.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(72.13278911564626, abs=0.012)
    assert selected["selected_by"] == "visual_final_contract_gui_sparse_pulse_repair"
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["placeable_count"] > 0
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_preserves_first_true_drop_over_later_louder_booms() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Been About - rSUN/drums_80_9A_6-Been About - rSUN.wav"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(23.998594104308392, abs=0.012)
    assert selected["selected_by"] == "visual_candidate_selector"
    assert selected["visual_components"].get("full_track_stronger_boom_front_edge_repair") is None
    assert visual_first_module._candidate_has_visible_boom_body(
        selected,
        profile_floor=0.580,
        body_floor=0.600,
        darkness_floor=0.560,
        post8_floor=0.540,
        drum_floor=0.850,
        drum_cont_floor=0.520,
        bass_floor=0.200,
        simultaneity_floor=0.560,
        require_transition=False,
    ) is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_full_track_stronger_repair_uses_fresh_candidate_proof() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/94/9A/Thornato, Benjamín Vanegas - Mama Clo/drums_94_9A_5-Thornato, Benjamín Vanegas - Mama Clo.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(71.8341206895685, abs=0.012)
    assert selected["selected_by"] == "visual_full_track_stronger_boom_front_edge"
    assert selected["visual_components"]["full_track_stronger_boom_front_edge_repair"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_replaces_weak_early_marker_with_first_later_proven_body() -> None:
    audio_path = "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/Champagne Drip, 9 Theory, Ashley Mazanec - Circles/drums_140_9A_7-Champagne Drip, 9 Theory, Ashley Mazanec - Circles.flac"
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(53.85916345724873, abs=0.012)
    assert selected["selected_by"] == "visual_full_track_stronger_boom_front_edge"
    assert visual["full_track_stronger_boom_front_edge_repair"] is True
    assert visual["full_track_stronger_previous_marker"] == pytest.approx(44.43133786848073, abs=0.012)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"].get("accepted_by_boom_front_edge_proof") is None
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "previous", "kind"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/123/5A/Aphex Twin - Windowlicker/drums_123_5A_6-Aphex Twin - Windowlicker.flac",
            303.0512125,
            21.99529672031414,
            "later_sustained_body",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/Quavo, Young Thug - Focused (feat. Young Thug)/drums_140_2A_6-Quavo, Young Thug - Focused (feat. Young Thug).flac",
            5.594984,
            48.01659863945578,
            "first_body_before_weak_marker",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/SHIMA - KARMA (Original Mix)/drums_140_2A_7-SHIMA - KARMA (Original Mix).flac",
            111.862366,
            122.1463888888889,
            "earlier_sustained_body",
        ),
    ],
)
def test_visual_first_repairs_suspicious_sustained_body_advisories(
    audio_path: str,
    target: float,
    previous: float,
    kind: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(
        audio_path,
        sample_rate=visual_first_module.VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        use_cache=True,
    )
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(target, abs=0.012)
    assert selected["selected_by"] == "visual_full_track_stronger_boom_front_edge"
    assert visual["full_track_stronger_boom_front_edge_repair"] is True
    assert visual["full_track_stronger_repair_kind"] == kind
    assert visual["full_track_stronger_previous_marker"] == pytest.approx(previous, abs=0.012)
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "source"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/CLUSTER - Muadeep/drums_80_9A_6-CLUSTER - Muadeep.flac",
            29.974557823129253,
            "visual_candidate_selector",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/11A/Who Run It - Three 6 Mafia/drums_80_11A_6-Who Run It - Three 6 Mafia.wav",
            26.186930138378674,
            "visual_final_contract_gui_front_edge_repair",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/80/9A/Zingara - Pink Peonies/drums_80_9A_5-Zingara - Pink Peonies.flac",
            105.7530437106228,
            "visual_final_contract_gui_front_edge_repair",
        ),
    ],
)
def test_visual_first_repairs_strict_gui_mask_holds_to_proven_front_edges(
    audio_path: str,
    target: float,
    source: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(target, abs=0.012)
    assert selected["selected_by"] == source
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["placeable_count"] > 0
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


@pytest.mark.parametrize(
    ("audio_path", "target", "tolerance"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/6A/Omar Souleyman Warni Warni - Omar Solaiman/drums_140_6A_7-Omar Souleyman Warni Warni - Omar Solaiman.wav",
            111.42857142857142,
            0.050,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/6A/Omar Solaiman - Omar Souleyman Warni Warni/drums_140_6A_7-Omar Solaiman - Omar Souleyman Warni Warni.flac",
            111.42857142857142,
            0.050,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/Tape B - Mac Miller - Loud (Tape B X STVSH Flip)/drums_140_4A_8-Tape B - Mac Miller - Loud (Tape B X STVSH Flip).flac",
            54.852018140589564,
            0.050,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/Nikita, the Wicked - BACK ONCE AGAIN/drums_140_4A_6-Nikita, the Wicked - BACK ONCE AGAIN.flac",
            27.428571428571427,
            0.050,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/Throwing Snow - Forged/drums_140_4A_7-Throwing Snow - Forged.flac",
            27.528816326530613,
            0.120,
        ),
    ],
)
def test_visual_first_preserves_reviewed_selector_markers_over_weak_boom_replacements(
    audio_path: str,
    target: float,
    tolerance: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)

    assert result["marker"] == pytest.approx(target, abs=tolerance)
    assert result["selected_candidate"]["selected_by"] != "visual_boom_earliest_dominant_replacement"


def test_clean_reviewed_selector_resists_distant_earliest_boom_override() -> None:
    selected = {
        "selected_by": "visual_candidate_selector",
        "reason": "visual candidate selector chose the reviewed-pattern waveform candidate",
        "timestamp": 120.0,
        "score": 0.72,
        "visual_components": {
            "clock_bar": 65,
            "phrase_prior": 1.0,
            "post4_height": 0.62,
            "post8_height": 0.63,
            "post_bass8": 0.42,
            "post_drum8": 1.0,
            "post_drum_cont4": 0.90,
            "post_drum_cont8": 0.90,
            "local_reentry": True,
            "local_reentry_gap": 0.35,
            "body_score": 0.72,
        },
    }
    weak_earlier_boom = {
        "selected_by": "visual_boom_earliest_dominant_replacement",
        "timestamp": 15.0,
        "score": 0.79,
        "force_selector_override": True,
        "force_selector_override_reasons": ["earlier_dominant_boom_available"],
        "visual_components": {
            "clock_bar": 9,
            "phrase_prior": 0.86,
            "post4_height": 0.66,
            "post8_height": 0.66,
            "post_bass8": 0.46,
            "post_drum8": 1.0,
            "post_drum_cont4": 0.92,
            "post_drum_cont8": 0.92,
            "body_score": 0.76,
            "boom_body_section": True,
        },
    }
    decisive_earlier_boom = {
        **weak_earlier_boom,
        "score": 0.90,
        "visual_components": {
            **weak_earlier_boom["visual_components"],
            "post4_height": 0.75,
            "post8_height": 0.76,
            "post_bass8": 0.54,
            "body_score": 0.84,
            "boom_section_simultaneity": 0.82,
            "post_inst8": 0.85,
            "jump4": 0.80,
            "jump8": 0.80,
            "local_reentry": True,
            "local_reentry_gap": 0.55,
            "post_drum_cont4": 0.96,
            "post_drum_cont8": 0.96,
        },
    }

    assert _selector_locked_marker_should_resist_boom_replacement(
        selected,
        weak_earlier_boom,
        selected_audit_flags=set(),
    )
    assert not _selector_locked_marker_should_resist_boom_replacement(
        selected,
        decisive_earlier_boom,
        selected_audit_flags=set(),
    )


def test_gui_front_edge_clock_uses_repaired_time_not_previous_marker_phase() -> None:
    selected = {
        "bpm_clock": {"nearest_one_bar": 65, "clock_zero_sec": 0.0},
        "visual_components": {"clock_bar": 65},
    }

    clock = _clock_for_visual_edge(15.023497732426303, 128.0, 0.0, selected)

    assert clock["source"] == "gui_boom_front_edge_track_grid"
    assert clock["nearest_one_bar"] == 9
    assert clock["beat_in_bar"] == 1
    assert clock["on_one"] is True

    off_grid = _clock_for_visual_edge(15.144, 128.0, 0.0, selected)

    assert off_grid["source"] == "gui_boom_front_edge_track_grid"
    assert off_grid["clock_zero_sec"] == pytest.approx(0.0)
    assert off_grid["one_distance_ms"] == pytest.approx(144.0)
    assert off_grid["on_one"] is False


def test_visual_first_ignores_file_start_intro_boom_for_later_definitive_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/128/9A/Odd Mob - LEFT TO RIGHT/"
        "drums_128_9A_6-Odd Mob - LEFT TO RIGHT.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(120.0, abs=0.050)
    assert selected["selected_by"] == "visual_final_audit_boom_front_edge"
    assert selected.get("visual_audit_ignored_file_start_intro_boom")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/145/9A/Jack Harlow - WHATS POPPIN/"
            "drums_145_9A_5-Jack Harlow - WHATS POPPIN.flac",
            6.696213151927438,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/142/12A/Kings of Convenience - Love Is No Big Truth/"
            "drums_142_12A_6-Kings of Convenience - Love Is No Big Truth.flac",
            14.181338028169014,
        ),
    ],
)
def test_visual_first_reclaims_early_review_pattern_body_from_late_boom_repair(
    audio_path: str,
    target: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(target, abs=0.050)
    assert selected.get("visual_body_reclaim_from_high_risk_boom_repair")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []


def test_visual_first_uses_trusted_anchor_body_over_generic_early_reclaim() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/128/8A/Planet Funk, Odd Mob - Chase the Sun (Odd Mob Remix)/"
        "drums_128_8A_7-Planet Funk, Odd Mob - Chase the Sun (Odd Mob Remix).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(90.00172335600908, abs=0.050)
    assert selected["selected_by"] in {
        "visual_gui_chunk",
        "visual_gui_boom_front_edge_contract",
    }
    assert selected["selected_by"] != "visual_first_strongest_dark_section_guard"
    if selected["selected_by"] == "visual_gui_boom_front_edge_contract":
        assert result["boom_proof"]["passes"] is True
        assert selected["gui_mask_proof"]["passes"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []


def test_visual_v2_reclaim_prefers_earliest_adequate_body_candidate() -> None:
    def candidate(time_sec: float, score: float, *, body: float, bass: float, pre_space: float) -> dict:
        return {
            "selected_by": "visual_drop_v2_candidate",
            "timestamp": time_sec,
            "score": score,
            "confidence_score": score,
            "reason": "visual-v2 full-song waveform section edge candidate",
            "visual_components": {
                "clock_bar": int(time_sec // 2),
                "post4_height": max(score, 0.58),
                "post8_height": max(score, 0.56),
                "post_bass8": bass,
                "post_drum8": 1.0,
                "post_drum_cont4": 0.70,
                "post_drum_cont8": 0.70,
                "body_score": body,
                "pre_space": pre_space,
                "local_reentry": True,
            },
        }

    selected = {
        **candidate(128.0, 0.408, body=0.69, bass=0.41, pre_space=0.40),
        "selected_by": "visual_drop_v2",
    }
    early = candidate(20.5, 0.418, body=0.59, bass=0.26, pre_space=0.71)
    later = candidate(36.5, 0.407, body=0.68, bass=0.34, pre_space=0.47)

    reclaimed = _review_pattern_body_candidate_after_boom_repair(
        selected,
        [selected, later, early],
        [],
        beatgrid={"bpm": 105.0, "bar_zero_sec": 0.0},
        rejected_sections=[],
    )

    assert reclaimed is not None
    assert reclaimed["timestamp"] == early["timestamp"]
    assert reclaimed["selected_by"] == "visual_drop_v2_candidate"


def test_visual_first_rejects_off_grid_early_body_for_later_proven_drop() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/9A/DECAP - Baboom/"
        "drums_105_9A_5-DECAP - Baboom.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(36.57142857142857, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_actual_body_boom_repair"
    assert selected["bpm_clock"]["source"] == "gui_boom_front_edge_track_grid"
    assert selected["bpm_clock"]["clock_zero_sec"] == pytest.approx(0.0)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert _gui_mask_proof(audio_path, result["marker"])["passes"] is True


def test_visual_first_baboom_keeps_same_first_drop_at_production_rate() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/9A/DECAP - Baboom/"
        "drums_105_9A_5-DECAP - Baboom.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(36.57142857142857, abs=0.050)
    assert result["marker"] != pytest.approx(45.71428571428571, abs=0.050)
    assert selected["bpm_clock"]["clock_zero_sec"] == pytest.approx(0.0)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert result["gui_mask_proof"]["passes"] is True


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/3B/1999 - Skeler/"
            "drums_130_3B_7-1999 - Skeler.flac",
            29.632,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/6A/Subtronics - Quantum Queso/"
            "drums_140_6A_7-Subtronics - Quantum Queso.flac",
            41.14285714285714,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/128/6A/GENESI - Done/"
            "drums_128_6A_5-GENESI - Done.flac",
            30.0,
        ),
    ],
)
def test_visual_first_final_arbitration_restores_reviewed_body_over_late_boom_reclaim(
    audio_path: str,
    target: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(target, abs=0.060)
    assert selected.get("visual_overbroad_reclaim_arbitration")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_sparse_repair_yields_to_dominant_boom_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/150/2A/Shanghai Doom - Dune/"
        "drums_150_2A_7-Shanghai Doom - Dune.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(30.0, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert selected["selected_by"] != "visual_final_contract_gui_sparse_pulse_repair"
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_final_overbroad_arbitration_requires_reset_front_for_later_stronger_body() -> None:
    selected = _candidate(
        27.851899881691807,
        17,
        score=0.585,
        phrase_prior=0.94,
        post4=0.695,
        post8=0.586,
        bass=0.573,
        drum=1.0,
        pre_drum=0.524,
        local_gap=0.175,
        phrase_body_shift=True,
        bpm=138.0,
    )
    selected["selected_by"] = "visual_final_contract_visible_signal_repair"
    selected["bpm_clock"] = {
        "bpm": 138.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 17,
    }
    selected["visual_components"]["one_distance_ms"] = 0.0
    _attach_passing_boom_proof(selected, profile_score=0.656, body_score=0.686, transition_score=0.477)
    selected["boom_proof"]["nearest_profile"]["metrics"].update(
        {"contrast": 0.175, "pre_space": 0.465, "post_drum_cont8": 0.696}
    )

    no_reset_later = _candidate(
        41.795918367346935,
        25,
        score=0.622,
        phrase_prior=0.96,
        post4=0.587,
        post8=0.566,
        bass=0.538,
        drum=0.994,
        pre_drum=0.698,
        local_gap=0.001,
        local_reentry=False,
        phrase_body_shift=True,
        bpm=138.0,
    )
    no_reset_later["selected_by"] = "visual_body_peak_candidate"
    no_reset_later["bpm_clock"] = {
        "bpm": 138.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 25,
    }
    no_reset_later["visual_components"]["one_distance_ms"] = 0.0
    no_reset_later["visual_components"]["pre_space"] = 0.414
    no_reset_later["visual_components"]["jump4"] = 0.001
    no_reset_later["visual_components"]["jump8"] = 0.001
    _attach_passing_boom_proof(no_reset_later, profile_score=0.628, body_score=0.673, transition_score=0.387)
    no_reset_later["boom_proof"]["nearest_profile"]["metrics"].update(
        {
            "contrast": 0.001,
            "pre_space": 0.414,
            "post8_height": 0.566,
            "post_bass8": 0.538,
            "post_drum8": 0.994,
            "post_drum_cont8": 0.621,
        }
    )

    reset_later = _candidate(
        111.30434782608695,
        65,
        score=0.777,
        phrase_prior=1.0,
        post4=0.707,
        post8=0.704,
        bass=0.518,
        drum=1.0,
        pre_drum=0.220,
        local_gap=0.652,
        local_reentry=True,
        phrase_body_shift=True,
        bpm=138.0,
    )
    reset_later["selected_by"] = "visual_boom_body_section"
    reset_later["bpm_clock"] = {
        "bpm": 138.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 65,
    }
    reset_later["visual_components"]["one_distance_ms"] = 0.0
    reset_later["visual_components"]["pre_space"] = 0.573
    reset_later["visual_components"]["boom_body_section"] = True
    _attach_passing_boom_proof(reset_later, profile_score=0.779, body_score=0.769, transition_score=0.733)
    reset_later["boom_proof"]["nearest_profile"]["metrics"].update(
        {
            "contrast": 0.652,
            "pre_space": 0.573,
            "post8_height": 0.704,
            "post_bass8": 0.518,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.862,
        }
    )

    replacement = _final_overbroad_reclaim_arbitration(
        selected,
        [no_reset_later, reset_later],
        beatgrid={"bpm": 138.0, "bar_sec": 240.0 / 138.0, "bar_zero_sec": 0.0},
        rejected_sections=[],
    )

    assert replacement is not None
    assert replacement["timestamp"] == pytest.approx(111.30434782608695, abs=0.001)
    assert replacement["selected_by"] == "visual_boom_body_section"
    assert "stronger later GUI body" in replacement["reason"]


def test_later_proven_boom_repair_rejects_moderate_no_reset_audit_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dense early continuation is not a drop when the whole-track Boom has a reset."""

    selected = _candidate(
        22.674285714285713,
        13,
        score=0.670,
        phrase_prior=0.86,
        post4=0.640,
        post8=0.620,
        bass=0.430,
        drum=1.0,
        pre_drum=0.615,
        local_gap=0.125,
        local_reentry=False,
        phrase_body_shift=True,
        bpm=130.0,
    )
    selected["selected_by"] = "visual_body_peak_candidate"
    selected["bpm_clock"] = {
        "bpm": 130.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 13,
    }
    selected["visual_components"].update(
        {
            "jump4": 0.125,
            "jump8": 0.125,
            "one_distance_ms": 0.0,
        }
    )
    _attach_passing_boom_proof(selected, profile_score=0.690, body_score=0.700, transition_score=0.410)
    selected["boom_proof"]["nearest_profile"]["metrics"].update(
        {
            "contrast": 0.125,
            "pre_space": 0.300,
            "post8_height": 0.620,
            "post_bass8": 0.430,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.720,
            "phrase_prior": 0.86,
        }
    )

    later_boom = _candidate(
        192.02521541950114,
        105,
        score=0.800,
        phrase_prior=0.86,
        post4=0.800,
        post8=0.743,
        bass=0.636,
        drum=1.0,
        pre_drum=0.203,
        local_gap=0.587,
        local_reentry=True,
        phrase_body_shift=True,
        bpm=130.0,
    )
    later_boom["selected_by"] = "visual_boom_body_section"
    later_boom["bpm_clock"] = {
        "bpm": 130.0,
        "on_one": True,
        "one_distance_ms": 0.0,
        "nearest_one_bar": 105,
    }
    later_boom["visual_components"].update(
        {
            "boom_body_section": True,
            "boom_section_darkness": 0.800,
            "boom_section_simultaneity": 0.760,
            "post_drum_cont8": 0.972,
            "one_distance_ms": 0.0,
        }
    )
    _attach_passing_boom_proof(later_boom, profile_score=0.784, body_score=0.816, transition_score=0.733)
    later_boom["boom_proof"]["nearest_profile"]["metrics"].update(
        {
            "contrast": 0.587,
            "pre_space": 0.527,
            "post8_height": 0.743,
            "post_bass8": 0.636,
            "post_drum8": 1.0,
            "post_drum_cont8": 0.972,
            "darkness": 0.800,
            "simultaneity": 0.760,
            "phrase_prior": 0.86,
        }
    )

    monkeypatch.setattr(
        visual_first_module,
        "audit_visual_selection",
        lambda *_args, **_kwargs: {"status": "pass", "flag_codes": [], "flags": []},
    )

    def fake_marker_boom_proof(
        _marker_sec: float,
        _boom_candidates: list[dict],
        *,
        selected_candidate: dict | None = None,
        **_kwargs: object,
    ) -> dict:
        proof = selected_candidate.get("boom_proof") if isinstance(selected_candidate, dict) else {}
        return dict(proof) if isinstance(proof, dict) and proof.get("passes") else {"passes": False}

    monkeypatch.setattr(visual_first_module, "marker_boom_proof", fake_marker_boom_proof)

    replacement = _later_proven_boom_repair_for_failed_selected(
        selected,
        [selected],
        [later_boom],
        selected["boom_proof"],
        {"passes": True},
        beatgrid={"bpm": 130.0, "bar_sec": 240.0 / 130.0, "bar_zero_sec": 0.0},
    )

    assert replacement is not None
    assert replacement["timestamp"] == pytest.approx(192.02521541950114, abs=0.001)
    assert replacement["selected_by"] == "visual_later_proven_boom_contract_repair"
    assert replacement["visual_later_proven_boom_repair_strength_votes"] >= 2
    assert replacement["visual_components"]["later_proven_boom_contract_repair"] is True


def test_visual_first_burn_dem_bridges_yields_no_reset_body_to_later_reset_drop() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/138/9A/Burn Dem Bridges - Extended Mix - Skin On Skin/"
        "drums_138_9A_7-Burn Dem Bridges - Extended Mix - Skin On Skin.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(111.329, abs=0.050)
    assert selected["selected_by"] != "visual_body_peak_candidate"
    assert body["contrast"] >= 0.550
    assert body["pre_drum"] <= 0.300
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_run_it_up_does_not_reclaim_sparse_no_reset_body_peak() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/12A/"
        "Wax Motif, Juos, 24hrs - Run It Up (feat. 24Hrs) (Extended Mix)/"
        "drums_130_12A_6-Wax Motif, Juos, 24hrs - Run It Up (feat. 24Hrs) (Extended Mix).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(29.646, abs=0.060)
    assert result["marker"] > 27.0
    assert not visual.get("early_body_peak_reclaimed_from_later_gui_repair")
    assert not visual.get("early_body_peak_reclaimed_from_final_later_gui_repair")
    assert body["contrast"] >= 0.150
    assert body["pre_space"] >= 0.900
    assert body["bass"] >= 0.400
    assert body["drum"] >= 0.880
    assert body["drum_cont"] >= 0.650
    assert body["simultaneity"] >= 0.600
    assert selected["gui_mask_proof"]["marker_immediate_body_present"] is True
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_enigma_sunken_place_uses_later_main_phrase_title_grid_front() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/7A/ENiGMA Dubz - The Sunken Place - VIP/"
        "drums_105_7A_5-ENiGMA Dubz - The Sunken Place - VIP.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(54.857143, abs=0.003)
    assert result["marker"] < 55.0
    assert result["marker"] != pytest.approx(57.088, abs=0.040)
    assert selected["selected_by"] == "visual_final_later_main_phrase_body_repair"
    assert visual.get("final_later_main_phrase_body_override") is True
    assert body["phrase"] >= 0.900
    assert body["darkness"] >= 0.680
    assert body["body_score"] >= 0.660
    assert body["contrast"] >= 0.330
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["bpm_clock"]["source"] != "gui_boom_front_edge_calibrated_grid"
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["marker_signal_present"] is True


def test_visual_first_key_glock_remix_rejects_bar_nine_warmup_body_for_main_phrase_drop() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/128/7A/"
        "KEY GLOCK SINCE 6IX (SOUNDS NASTE REMIX) - Sounds na$te/"
        "drums_128_7A_8-KEY GLOCK SINCE 6IX (SOUNDS NASTE REMIX) - Sounds na$te.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(30.000, abs=0.050)
    assert result["marker"] != pytest.approx(15.000, abs=0.050)
    assert selected["selected_by"] == "visual_final_later_main_phrase_body_repair"
    assert visual.get("final_later_main_phrase_body_override") is True
    assert visual.get("final_later_main_phrase_previous_marker") == pytest.approx(15.000, abs=0.001)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert body["phrase"] >= 0.900
    assert body["body_score"] >= 0.780
    assert body["post8"] >= 0.700
    assert body["bass"] >= 0.650
    assert body["drum"] >= 0.950
    assert selected["boom_proof"].get("later_main_phrase_over_earlier_dominant_relief") is True


def test_visual_first_love_me_now_rejects_early_phase_calibrated_body_peak_for_true_drop() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/12A/LOVE ME NOW! - Desren/"
        "drums_130_12A_7-LOVE ME NOW! - Desren.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(66.461519, abs=0.050)
    assert result["marker"] != pytest.approx(14.816, abs=0.050)
    assert selected["selected_by"] == "visual_final_later_main_phrase_body_repair"
    assert visual.get("final_later_main_phrase_body_override") is True
    assert visual.get("final_later_main_phrase_previous_marker") == pytest.approx(14.816, abs=0.001)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert body["body_score"] >= 0.880
    assert body["darkness"] >= 0.920
    assert body["post8"] >= 0.820
    assert body["bass"] >= 0.820
    assert body["contrast"] >= 0.500


def test_visual_first_sexy_bitch_rejects_early_stable_wide_nonbody_for_dominant_drop() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/132/10A/Sexy Bitch (Extended Mix) - Discip/"
        "drums_132_10A_6-Sexy Bitch (Extended Mix) - Discip.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(87.273175, abs=0.050)
    assert result["marker"] != pytest.approx(29.090909, abs=0.050)
    assert selected["selected_by"] == "visual_final_later_main_phrase_body_repair"
    assert visual.get("final_later_main_phrase_body_override") is True
    assert visual.get("final_later_main_phrase_previous_marker") == pytest.approx(29.090909, abs=0.001)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert body["body_score"] >= 0.920
    assert body["darkness"] >= 0.920
    assert body["post8"] >= 0.900
    assert body["bass"] >= 0.920
    assert body["drum_cont"] >= 0.950
    assert body["contrast"] >= 0.600


@pytest.mark.parametrize(
    ("audio_path", "target", "target_abs", "expected_selected_by"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/5A/"
            "John Summit - Make Me Feel - Extended Mix/"
            "drums_126_5A_6-John Summit - Make Me Feel - Extended Mix.flac",
            60.951451,
            0.010,
            "visual_candidate_selector",
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/6A/"
            "FISHER, Shermanology - It's A Killa/"
            "drums_126_6A_5-FISHER, Shermanology - It's A Killa.flac",
            60.952381,
            0.010,
            "visual_gui_first_fat_block",
        ),
    ],
)
def test_visual_first_timeout_cluster_exact_passes_without_visual_drop_v2_fallback(
    audio_path: str,
    target: float,
    target_abs: float,
    expected_selected_by: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)

    assert result["marker"] == pytest.approx(target, abs=target_abs)
    assert selected["selected_by"] == expected_selected_by
    assert "visual_drop_v2" not in selected["selected_by"]
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_timeout_cluster_stale_review_row_still_finishes_with_contract() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/4A/"
        "John Summit, Nic Fanciulli - Witch Doctor/"
        "drums_126_4A_7-John Summit, Nic Fanciulli - Witch Doctor.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)

    assert result["marker"] == pytest.approx(30.476190, abs=0.010)
    assert selected["selected_by"] == "visual_sustained_body_section_front_edge"
    assert "visual_drop_v2" not in selected["selected_by"]
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_if_you_know_moves_pre_body_grid_to_visual_front_edge() -> None:
    if os.environ.get("RUN_SLOW_VISUAL_AUDIO_TESTS") != "1":
        pytest.skip("slow full-detector local audio regression; enable with RUN_SLOW_VISUAL_AUDIO_TESTS=1")
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/93/10B/DEDRO - If You Know/"
        "drums_93_10B_7-DEDRO - If You Know.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(41.40032258064516, abs=0.012)
    assert result["marker"] != pytest.approx(41.29032258064516, abs=0.040)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert visual.get("final_gui_mask_nearest_front_edge_repair") is True
    assert visual.get("final_gui_mask_nearest_previous_marker") == pytest.approx(41.29032258064516, abs=0.012)
    assert selected["bpm_clock"]["source"] == "gui_boom_front_edge_calibrated_grid"
    assert selected["bpm_clock"]["clock_zero_sec"] == pytest.approx(0.110, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert gui["marker_signal_present"] is True
    assert gui["marker_relevant_mask"] is True
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui_boom_mask_strict_contract_issue(gui) in ("", "marker_has_no_immediate_drop_body")


def test_visual_first_lets_try_it_prefers_earlier_same_section_reset_front() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/6A/"
        "Marten Hørger, BRANDON - Lets Try It/"
        "drums_125_6A_8-Marten Hørger, BRANDON - Lets Try It.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(32.670476, abs=0.010)
    assert result["marker"] != pytest.approx(36.480, abs=0.050)
    assert selected["selected_by"] == "visual_earlier_same_section_reset_front_edge"
    assert visual.get("earlier_same_section_reset_front_edge_repair") is True
    assert visual.get("earlier_same_section_late_repeat_marker") == pytest.approx(36.480, abs=0.010)
    assert body["contrast"] >= 0.500
    assert body["pre_space"] >= 0.550
    assert body["pre_drum"] <= 0.080
    assert selected["bpm_clock"]["source"] == "title_bpm_track_zero"
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 40.0


def test_visual_first_alibi_rejects_wide_same_section_reset_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/130/5A/"
        "Ella Henderson, Rudimental - Alibi (feat. Rudimental)/"
        "drums_130_5A_7-Ella Henderson, Rudimental - Alibi (feat. Rudimental).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]

    assert result["marker"] == pytest.approx(33.190769, abs=0.010)
    assert result["marker"] != pytest.approx(33.230769, abs=0.010)
    assert selected["selected_by"] != "visual_earlier_same_section_reset_front_edge"
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


def test_visual_first_move_too_slow_repairs_wide_final_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/133/11A/"
        "Move Too Slow - Cesco/drums_133_11A_6-Move Too Slow - Cesco.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(58.492204, abs=0.010)
    assert result["marker"] != pytest.approx(58.927204, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert visual.get("final_gui_mask_nearest_front_edge_repair") is True
    assert visual.get("final_gui_mask_nearest_previous_marker") == pytest.approx(58.927204, abs=0.010)
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


def test_visual_first_acyan_repairs_wide_final_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/"
        "LEADPOISON - Acyan/drums_140_2A_6-LEADPOISON - Acyan.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(41.134558, abs=0.010)
    assert result["marker"] != pytest.approx(41.819138, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert visual.get("final_gui_mask_nearest_front_edge_repair") is True
    assert visual.get("final_gui_mask_nearest_previous_marker") == pytest.approx(42.074558, abs=0.010)
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


def test_visual_first_psychosis_repairs_wide_final_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/"
        "Alix Perez - Psychosis/drums_140_7A_6-Alix Perez - Psychosis.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(54.911179, abs=0.010)
    assert result["marker"] != pytest.approx(55.801179, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert visual.get("final_gui_mask_nearest_front_edge_repair") is True
    assert visual.get("final_gui_mask_nearest_previous_marker") == pytest.approx(55.801179, abs=0.010)
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


def test_visual_first_kula_kula_repairs_wide_final_gui_front_edge() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/9A/"
        "Pushloop, The Widdler - Kula Kula - The Widdler Remix/"
        "drums_140_9A_6-Pushloop, The Widdler - Kula Kula - The Widdler Remix.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]

    assert result["marker"] == pytest.approx(55.239333, abs=0.010)
    assert result["marker"] != pytest.approx(56.118277, abs=0.050)
    assert selected["selected_by"] == "visual_candidate_selector"
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui.get("marker_immediate_body_present") is True
    assert gui.get("accepted_by_coarse_gui_front_edge_after_fine_probe") is not True
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


def test_visual_first_overkill_requires_fine_gui_front_edge_over_unproven_earlier_boom() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/"
        "BLVZE - OVERKILL (Original Mix)/"
        "drums_140_7A_8-BLVZE - OVERKILL (Original Mix).flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    boom = result["boom_proof"]
    audit = result["visual_audit"]

    assert result["marker"] == pytest.approx(69.850714, abs=0.010)
    assert result["marker"] != pytest.approx(12.000000, abs=0.050)
    assert result["marker"] != pytest.approx(27.428571, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_mask_nearest_repair"
    assert boom.get("accepted_by_gui_front_edge_over_unproven_earlier_boom") is True
    assert boom.get("ignored_unproven_earlier_boom_times") == pytest.approx([13.714], abs=0.001)
    assert audit.get("gui_front_edge_over_unproven_earlier_boom_audit_relief") is True
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui.get("marker_signal_present") is True
    assert gui.get("marker_relevant_mask") is True
    assert gui.get("marker_immediate_body_present") is True
    assert gui.get("accepted_by_coarse_gui_front_edge_after_fine_probe") is not True
    assert gui["nearest_placeable_offset_sec"] == pytest.approx(0.0, abs=0.006)
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


@pytest.mark.parametrize(
    "audio_path",
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/"
            "Fred again.. - Victory Lap Four/"
            "drums_140_2A_8-Fred again.. - Victory Lap Four.flac"
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/"
            "Fred again.., Skepta, PlaqueBoyMax, Denzel Curry, Hanumankind, That Mexican OT - Victory Lap Four/"
            "drums_140_2A_8-Fred again.., Skepta, PlaqueBoyMax, Denzel Curry, Hanumankind, That Mexican OT - Victory Lap Four.flac"
        ),
    ],
)
def test_visual_first_victory_lap_four_rejects_no_immediate_body_recovery(
    audio_path: str,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)
    gui = result["gui_mask_proof"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(166.784000, abs=0.010)
    assert result["marker"] != pytest.approx(15.428571, abs=0.050)
    assert selected["selected_by"] == "visual_final_contract_gui_sparse_pulse_repair"
    assert visual.get("analysis_rate_last_mile_strict_gui_recovery") is True
    assert gui_boom_mask_strict_contract_issue(gui) in ("", None)
    assert (
        visual_first_module._gui_mask_proof_needs_front_edge_repair(
            gui,
            trust_reasonless_relief_offset=True,
        )
        is False
    )
    assert gui.get("marker_signal_present") is True
    assert gui.get("marker_relevant_mask") is True
    assert gui.get("marker_immediate_body_present") is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0

    _assert_fresh_builder_gate_accepts(audio_path, result)


@pytest.mark.parametrize(
    ("audio_path", "target"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/6A/"
            "Locked Club, RLGN - Electro Hund/"
            "drums_125_6A_7-Locked Club, RLGN - Electro Hund.flac",
            61.4415,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/11A/"
            "AC Slater, Chris Lorenzo, Wax Motif - Fly Kicks - Wax Motif Remix/"
            "drums_125_11A_7-AC Slater, Chris Lorenzo, Wax Motif - Fly Kicks - Wax Motif Remix.flac",
            92.196848,
        ),
    ],
)
def test_visual_first_same_section_reset_repair_does_not_pull_clean_125_bpm_cases(
    audio_path: str,
    target: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = _assert_visual_first_production_contract(result)

    assert result["marker"] == pytest.approx(target, abs=0.050)
    assert selected["selected_by"] != "visual_earlier_same_section_reset_front_edge"
    assert not selected["visual_components"].get("earlier_same_section_reset_front_edge_repair")
    assert selected["bpm_clock"]["on_one"] is True


@pytest.mark.parametrize(
    ("audio_path", "target", "max_marker"),
    [
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/8A/Break Your Fall - BOII/"
            "drums_105_8A_6-Break Your Fall - BOII.wav",
            54.931,
            56.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/"
            "Roll in Peace (OkayJake x VCTRE remix) - Kodak Black/"
            "drums_140_2A_6-Roll in Peace (OkayJake x VCTRE remix) - Kodak Black.wav",
            27.430,
            28.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/128/6A/GENESI - Done/"
            "drums_128_6A_5-GENESI - Done.flac",
            30.000,
            31.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/131/2A/InntRaw - Breathe/"
            "drums_131_2A_6-InntRaw - Breathe.flac",
            31.145,
            32.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/131/5A/And Back - JAKKOB/"
            "drums_131_5A_6-And Back - JAKKOB.wav",
            29.313,
            30.0,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/132/5A/"
            "PAWSA, The Adventures Of Stevie V - Dirty Cash (Money Talks)/"
            "drums_132_5A_7-PAWSA, The Adventures Of Stevie V - Dirty Cash (Money Talks).flac",
            43.865,
            44.5,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/132/10A/Layton Giordani - Paranoid/"
            "drums_132_10A_7-Layton Giordani - Paranoid.flac",
            18.182,
            19.0,
        ),
    ],
)
def test_visual_first_later_main_phrase_repair_does_not_pull_clean_cases_later(
    audio_path: str,
    target: float,
    max_marker: float,
) -> None:
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]

    assert result["marker"] == pytest.approx(target, abs=0.060)
    assert result["marker"] < max_marker
    assert not visual.get("final_later_main_phrase_body_override")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True


def test_visual_first_roll_in_peace_keeps_earlier_proven_phrase_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/2A/"
        "Roll in Peace (OkayJake x VCTRE remix) - Kodak Black/"
        "drums_140_2A_6-Roll in Peace (OkayJake x VCTRE remix) - Kodak Black.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(27.430, abs=0.060)
    assert result["marker"] < 28.0
    assert selected["selected_by"] != "visual_final_contract_gui_front_edge_repair"
    assert body["phrase"] >= 0.900
    assert body["post8"] >= 0.600
    assert body["bass"] >= 0.350
    assert body["drum_cont"] >= 0.700
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_ddd_uses_raw_earlier_dominant_boom_warning() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/12A/DDD - Dank Frank/"
        "drums_140_12A_5-DDD - Dank Frank.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(27.433, abs=0.060)
    assert result["marker"] < 28.0
    assert selected["selected_by"] == "visual_boom_earliest_dominant_replacement"
    assert visual.get("main_path_raw_earlier_dominant_repair") is True
    assert body["contrast"] >= 0.300
    assert body["pre_space"] >= 0.600
    assert body["post8"] >= 0.580
    assert body["bass"] >= 0.400
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_who_zero_point_prefers_first_clean_body_before_later_repeat() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/100/2A/Who - zero point/"
        "drums_100_2A_5-Who - zero point.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(48.014, abs=0.060)
    assert result["marker"] < 49.0
    assert selected["selected_by"] == "visual_final_first_true_body_before_later_repeat"
    assert visual.get("final_first_true_body_before_later_repeat") is True
    assert body["pre_drum"] <= 0.120
    assert body["post8"] >= 0.600
    assert body["bass"] >= 0.400
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True


def test_visual_first_break_your_fall_rejects_opening_intro_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/105/8A/Break Your Fall - BOII/"
        "drums_105_8A_6-Break Your Fall - BOII.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(54.931, abs=0.060)
    assert result["marker"] > 50.0
    assert selected["selected_by"] in {
        "visual_final_opening_intro_to_later_drop_repair",
        "visual_return_opening_intro_to_later_drop_repair",
    }
    assert (
        visual.get("final_opening_intro_to_later_drop_repair") is True
        or visual.get("return_opening_intro_to_later_drop_repair") is True
    )
    assert body["phrase"] >= 0.900
    assert body["bass"] >= 0.480
    assert body["drum_cont"] >= 0.900
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["gui_mask_proof"].get("accepted_by_low_density_transition_body_start") is True
    assert gui_boom_mask_strict_contract_issue(selected["gui_mask_proof"]) is None


def test_visual_first_calypso_prefers_first_strict_drop_front_over_later_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/120/12A/"
        "CALYPSO VIP (CALYPSO VIP.flac) - ero808, NXSTY/"
        "drums_120_12A_7-CALYPSO VIP (CALYPSO VIP.flac) - ero808, NXSTY.wav"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(14.000, abs=0.050)
    assert selected["selected_by"] in {
        "visual_return_first_true_body_before_later_repeat",
        "visual_final_first_true_body_before_later_repeat",
    }
    assert (
        visual.get("return_first_true_body_before_later_repeat") is True
        or visual.get("final_first_true_body_before_later_repeat") is True
    )
    assert max(body["pre_space"], body["contrast"]) >= 0.520
    assert body["drum"] >= 0.950
    assert body["simultaneity"] >= 0.600
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert gui_boom_mask_strict_contract_issue(selected["gui_mask_proof"]) is None


def test_visual_first_locked_club_prefers_first_proven_drop_over_later_heavier_body() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/125/6A/Locked Club, RLGN - Electro Hund/"
        "drums_125_6A_7-Locked Club, RLGN - Electro Hund.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    visual = selected["visual_components"]
    body = visual_first_module._candidate_visual_body_summary(selected)

    assert result["marker"] == pytest.approx(61.440, abs=0.050)
    assert selected["selected_by"] == "visual_final_first_true_body_before_later_repeat"
    assert visual.get("final_first_true_body_before_later_repeat") is True
    assert visual.get("final_first_true_body_previous_marker") == pytest.approx(153.600, abs=0.001)
    assert body["body_score"] >= 0.620
    assert body["post8"] >= 0.520
    assert body["drum_cont"] >= 0.700
    assert body["simultaneity"] >= 0.600
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_my_peoples_preserves_pre_snap_transition_boundary() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/4A/AC Slater, Taiki Nulight - My Peoples/"
        "drums_126_4A_6-AC Slater, Taiki Nulight - My Peoples.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]

    assert result["marker"] == pytest.approx(60.98485260770975, abs=0.050)
    assert selected["selected_by"] in {
        "visual_gui_first_fat_block",
        "visual_final_audit_boom_front_edge",
    }
    assert not selected["visual_components"].get("zoomed_gui_nearest_front_edge_snap")
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["gui_mask_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0


def test_visual_first_my_peoples_keeps_transition_start_before_later_body_snap() -> None:
    audio_path = (
        "/Users/alirahimlou/Desktop/MUSIC/STEMS/126/4A/AC Slater, Taiki Nulight - My Peoples/"
        "drums_126_4A_6-AC Slater, Taiki Nulight - My Peoples.flac"
    )
    if not Path(audio_path).exists():
        pytest.skip(f"local audio fixture not available: {audio_path}")

    result = visual_first_marker(audio_path, sample_rate=44100, use_cache=True)
    selected = result["selected_candidate"]
    body = visual_first_module._candidate_visual_body_summary(selected)
    gui = selected["gui_mask_proof"]

    assert result["marker"] == pytest.approx(60.984853, abs=0.006)
    assert result["marker"] < 61.000
    assert selected["selected_by"] in {
        "visual_gui_first_fat_block",
        "visual_final_audit_boom_front_edge",
    }
    assert gui.get("accepted_by_low_density_transition_body_start") is True
    assert gui.get("marker_signal_present") is True
    assert gui.get("marker_relevant_mask") is True
    assert gui.get("marker_immediate_body_present") is False
    assert gui_boom_mask_strict_contract_issue(gui) is None
    assert body["phrase"] >= 0.940
    assert body["darkness"] >= 0.600
    assert body["drum_cont"] >= 0.840
    assert result["visual_audit"]["status"] == "pass"
    assert result["visual_audit"]["flag_codes"] == []
    assert result["boom_proof"]["passes"] is True
    assert selected["bpm_clock"]["on_one"] is True
    assert abs(float(selected["bpm_clock"]["one_distance_ms"])) <= 1.0
