from __future__ import annotations

from drop_aligner.structure_map import _pick_first_second


def _structure_candidate(bar: int, *, score: float, sustain: float, low_to_high: float, block_height: float | None = None) -> dict:
    return {
        "rank": bar,
        "handcrafted_rank": bar,
        "timestamp": float(bar),
        "snapped_sec": float(bar),
        "time_sec": float(bar),
        "score": float(score),
        "confidence_score": float(score),
        "structure_bar": int(bar),
        "structure_clock_bar": int(bar),
        "structure_components": {
            "clock_bar": int(bar),
            "phrase_bar": int(bar),
            "phrase_prior": 0.95,
            "sustained_groove": float(sustain),
            "block_height": float(block_height if block_height is not None else sustain),
            "low_to_high": float(low_to_high),
            "timbre_novelty": 0.62,
            "pre_space": 0.58,
            "mini_fill_penalty": 0.0,
            "on_one": True,
            "one_distance_ms": 0.0,
        },
    }


def test_structure_map_prefers_stronger_later_visual_chunk_over_weak_early_phrase_probe() -> None:
    early_probe = _structure_candidate(9, score=0.64, sustain=0.50, low_to_high=0.42)
    visible_chunk = _structure_candidate(17, score=0.59, sustain=0.60, low_to_high=0.62)

    first, _second = _pick_first_second([early_probe, visible_chunk])

    assert first is not None
    assert first["structure_clock_bar"] == 17
    assert first["structure_role"] == "first_drop"


def test_structure_map_skips_smaller_buildup_block_when_next_phrase_is_visibly_bigger() -> None:
    buildup = _structure_candidate(17, score=0.66, sustain=0.52, low_to_high=0.48, block_height=0.50)
    drop = _structure_candidate(25, score=0.58, sustain=0.57, low_to_high=0.64, block_height=0.63)

    first, _second = _pick_first_second([buildup, drop])

    assert first is not None
    assert first["structure_clock_bar"] == 25
    assert first["structure_role"] == "first_drop"


def test_structure_map_keeps_first_sufficient_big_block_over_much_later_louder_block() -> None:
    first_big_block = _structure_candidate(9, score=0.57, sustain=0.76, low_to_high=0.43, block_height=0.61)
    much_later_louder_block = _structure_candidate(75, score=0.73, sustain=0.83, low_to_high=0.58, block_height=0.78)

    first, _second = _pick_first_second([much_later_louder_block, first_big_block])

    assert first is not None
    assert first["structure_clock_bar"] == 9
    assert first["structure_role"] == "first_drop"
    assert first["first_drop_selection_rule"] == "first_sufficient_visual_block"
