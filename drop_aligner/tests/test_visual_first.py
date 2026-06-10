from __future__ import annotations

from pathlib import Path

import pytest

from drop_aligner.visual_first import (
    _filter_rejected_sections,
    _use_visual_drop_v2_result,
    _zoomed_marker_time,
    select_first_visual_chunk,
    visual_chunk_candidates,
    visual_first_marker,
)
from drop_aligner.visual_drop_v2 import select_visual_drop_v2, visual_drop_v2_candidates


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
            "pre_drum_cont4": pre_drum,
            "post_drum_cont4": drum,
            "post_drum_cont8": drum,
            "local_reentry": local_reentry,
            "local_reentry_gap": local_gap,
            "phrase_body_shift": phrase_body_shift,
            "jump4": 0.15,
            "jump8": 0.15,
            "prev1_height": 0.20,
            "prev2_height": 0.20,
        },
        "bpm_clock": {"bpm": bpm},
    }


def test_visual_first_skips_smaller_buildup_when_next_block_is_bigger() -> None:
    heights = [0.16] * 16 + [0.50] * 8 + [0.72] * 24
    candidates = visual_chunk_candidates(_feature_map(heights))

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 25


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


def test_visual_first_shifts_off_phrase_edge_to_adjacent_phrase_edge() -> None:
    candidates = [
        _candidate(42.488, 26, score=0.703, phrase_prior=0.18, post4=0.652, post8=0.679, bass=0.546, pre_drum=0.07),
        _candidate(44.155, 27, score=0.714, phrase_prior=0.48, post4=0.685, post8=0.677, bass=0.544, pre_drum=0.20),
    ]

    selected = select_first_visual_chunk(candidates)

    assert selected is not None
    assert selected["visual_components"]["clock_bar"] == 27
    assert selected["visual_edge_replaced_candidate"]["clock_bar"] == 26


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
            54.857142857142854,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/Xotix - BUTTERFACE/drums_140_5A_6-Xotix - BUTTERFACE.flac",
            13.715578231292517,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/5A/CloZee, David Starfire - Soma Dreams - CloZee remix/drums_140_5A_7-CloZee, David Starfire - Soma Dreams - CloZee remix.flac",
            28.060544217687074,
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
            41.14285714285714,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Wanna Go - Phrva/drums_140_7A_7-Wanna Go - Phrva.wav",
            27.42625850340136,
            0.05,
        ),
        (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/7A/Bassnectar, PEEKABOO - Disrupt The System - Underground Mix/drums_140_7A_8-Bassnectar, PEEKABOO - Disrupt The System - Underground Mix.flac",
            82.28650793650793,
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

    marker = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)["marker"]

    assert abs(marker - target) <= tolerance
