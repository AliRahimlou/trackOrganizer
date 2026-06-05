from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .musical_clock import bpm_clock_for_time, phrase_strength_for_bar
from .structure_features import compute_bar_feature_map


STRUCTURE_MAP_VERSION = 6


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _mean_bars(bars: List[Mapping[str, Any]], key: str, start: int, end: int) -> float:
    subset = bars[max(0, int(start)) : min(len(bars), int(end))]
    if not subset:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0) or 0.0) for row in subset]))


def _slope_bars(bars: List[Mapping[str, Any]], key: str, start: int, end: int) -> float:
    subset = bars[max(0, int(start)) : min(len(bars), int(end))]
    if len(subset) < 2:
        return 0.0
    values = np.asarray([float(row.get(key, 0.0) or 0.0) for row in subset], dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    try:
        slope = float(np.polyfit(x, values, deg=1)[0])
    except Exception:
        return 0.0
    return _clip01((slope + 0.035) / 0.12)


def _bar_transition_score(
    bars: List[Mapping[str, Any]],
    idx: int,
    *,
    bpm: Optional[float] = None,
    clock_zero_sec: float = 0.0,
) -> Dict[str, float]:
    bar_number = int(bars[idx].get("bar", idx + 1) or idx + 1)
    start_sec = float(bars[idx].get("start_sec", 0.0) or 0.0)
    clock = bpm_clock_for_time(start_sec, bpm, clock_zero_sec=float(clock_zero_sec or 0.0))
    clock_bar = int((clock or {}).get("nearest_one_bar", 0) or 0)
    phrase_bar = int(clock_bar or bar_number)
    pre_8_energy = _mean_bars(bars, "aggregate_energy", idx - 8, idx)
    pre_4_energy = _mean_bars(bars, "aggregate_energy", idx - 4, idx)
    post_4_energy = _mean_bars(bars, "aggregate_energy", idx, idx + 4)
    post_8_energy = _mean_bars(bars, "aggregate_energy", idx, idx + 8)
    post_groove = _mean_bars(bars, "groove_energy", idx, idx + 8)
    post_density = _mean_bars(bars, "drum_density", idx, idx + 8)
    post_bass = _mean_bars(bars, "bass_low_energy", idx, idx + 8)
    novelty = max(
        float(bars[idx].get("timbre_novelty", 0.0) or 0.0),
        _mean_bars(bars, "timbre_novelty", idx - 1, idx + 2),
    )
    vocal_dropout = max(float(bars[idx].get("vocal_dropout", 0.0) or 0.0), _mean_bars(bars, "vocal_dropout", idx - 1, idx + 2))
    inst_reentry = max(float(bars[idx].get("instrumental_reentry", 0.0) or 0.0), _mean_bars(bars, "instrumental_reentry", idx - 1, idx + 2))
    build_slope = max(
        _slope_bars(bars, "aggregate_energy", idx - 8, idx),
        _slope_bars(bars, "timbre_novelty", idx - 8, idx),
    )
    low_to_high = _clip01((0.62 * (post_8_energy - pre_8_energy + 0.18)) + (0.38 * (post_4_energy - pre_4_energy + 0.18)))
    sustained = _clip01((0.42 * post_groove) + (0.30 * post_density) + (0.28 * post_bass))
    pre_space = _clip01(1.0 - (0.68 * pre_4_energy) - (0.32 * _mean_bars(bars, "drum_density", idx - 4, idx)))
    phrase, phrase_name = phrase_strength_for_bar(phrase_bar)
    one_distance_value = (clock or {}).get("one_distance_ms", 999999.0)
    one_distance_ms = float(999999.0 if one_distance_value is None else one_distance_value)
    on_one = bool((clock or {}).get("on_one")) if clock else False
    if not on_one:
        phrase = 0.0
        phrase_name = "off-one"
    mini_fill_penalty = _clip01((0.42 - post_groove) / 0.42) if post_groove < 0.42 else 0.0
    score = _clip01(
        (0.27 * low_to_high)
        + (0.22 * sustained)
        + (0.15 * novelty)
        + (0.11 * build_slope)
        + (0.09 * inst_reentry)
        + (0.06 * vocal_dropout)
        + (0.10 * phrase)
        - (0.12 * mini_fill_penalty)
    )
    return {
        "score": float(score),
        "low_to_high": float(low_to_high),
        "sustained_groove": float(sustained),
        "post_groove": float(post_groove),
        "post_density": float(post_density),
        "post_bass": float(post_bass),
        "pre_space": float(pre_space),
        "timbre_novelty": float(novelty),
        "build_slope": float(build_slope),
        "instrumental_reentry": float(inst_reentry),
        "vocal_dropout": float(vocal_dropout),
        "phrase_prior": float(phrase),
        "phrase_bar": int(phrase_bar),
        "clock_bar": int(clock_bar) if clock_bar is not None else 0,
        "clock_one_time": float((clock or {}).get("nearest_one_time", start_sec) if clock else start_sec),
        "one_distance_ms": float(one_distance_ms),
        "on_one": bool(on_one),
        "beatgrid_bar": int(bar_number),
        "mini_fill_penalty": float(mini_fill_penalty),
        "phrase": phrase_name,
    }


def _candidate_from_bar(bar: Mapping[str, Any], components: Mapping[str, float], *, rank: int, role: str) -> Dict[str, Any]:
    bar_number = int(bar.get("bar", rank) or rank)
    marker_time = float(components.get("clock_one_time", bar.get("start_sec", 0.0)) or bar.get("start_sec", 0.0) or 0.0)
    reason = (
        f"{role}: {components.get('phrase', 'phrase')} | "
        f"clock b{int(components.get('phrase_bar', bar_number) or bar_number)} | "
        f"low->high {float(components.get('low_to_high', 0.0)):.2f}, "
        f"sustain {float(components.get('sustained_groove', 0.0)):.2f}, "
        f"novelty {float(components.get('timbre_novelty', 0.0)):.2f}"
    )
    return {
        "rank": int(rank),
        "handcrafted_rank": int(rank),
        "timestamp": float(marker_time),
        "snapped_sec": float(marker_time),
        "time_sec": float(marker_time),
        "confidence_score": float(components.get("score", 0.0) or 0.0),
        "score": float(components.get("score", 0.0) or 0.0),
        "selected_by": "structure_map",
        "reason": reason,
        "structure_role": role,
        "section_label": role,
        "structure_bar": int(bar_number),
        "structure_clock_bar": int(components.get("clock_bar", 0) or 0),
        "structure_components": dict(components),
        "post_drop_density": float(components.get("post_density", 0.0) or 0.0),
        "sustained_full_groove_score": float(components.get("sustained_groove", 0.0) or 0.0),
        "pre_drop_contrast": float(components.get("low_to_high", 0.0) or 0.0),
        "immediate_groove_start_score": float(components.get("post_groove", 0.0) or 0.0),
        "vocal_transition_score": float(components.get("vocal_dropout", 0.0) or 0.0),
        "inst_energy_jump_score": float(components.get("instrumental_reentry", 0.0) or 0.0),
        "self_similarity_boundary_score": float(components.get("timbre_novelty", 0.0) or 0.0),
    }


def _bar_lane(bar: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bar": int(bar.get("bar", 0) or 0),
        "start_sec": float(bar.get("start_sec", 0.0) or 0.0),
        "end_sec": float(bar.get("end_sec", 0.0) or 0.0),
        "aggregate_energy": float(bar.get("aggregate_energy", 0.0) or 0.0),
        "groove_energy": float(bar.get("groove_energy", 0.0) or 0.0),
        "bass_low_energy": float(bar.get("bass_low_energy", 0.0) or 0.0),
        "drum_density": float(bar.get("drum_density", 0.0) or 0.0),
        "instrumental_energy": float(bar.get("instrumental_energy", 0.0) or 0.0),
        "instrumental_reentry": float(bar.get("instrumental_reentry", 0.0) or 0.0),
        "vocal_presence": float(bar.get("vocal_presence", 0.0) or 0.0),
        "vocal_dropout": float(bar.get("vocal_dropout", 0.0) or 0.0),
        "timbre_novelty": float(bar.get("timbre_novelty", 0.0) or 0.0),
    }


def _pick_first_second(candidates: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    def one_distance(candidate: Mapping[str, Any]) -> float:
        value = (candidate.get("structure_components") or {}).get("one_distance_ms", 999999.0)
        return float(999999.0 if value is None else value)

    viable = [
        candidate
        for candidate in candidates
        if float(candidate.get("score", 0.0) or 0.0) >= 0.42
        and float((candidate.get("structure_components") or {}).get("sustained_groove", 0.0) or 0.0) >= 0.34
        and float((candidate.get("structure_components") or {}).get("mini_fill_penalty", 0.0) or 0.0) < 0.55
    ]
    on_one_viable = [
        candidate
        for candidate in viable
        if bool((candidate.get("structure_components") or {}).get("on_one"))
        and one_distance(candidate) <= 40.0
    ]
    viable = on_one_viable
    first = None
    if viable:
        strongest = max(viable, key=lambda row: float(row.get("score", 0.0) or 0.0))
        strongest_score = float(strongest.get("score", 0.0) or 0.0)
        preferred_floor = max(0.44, strongest_score - 0.14)
        preferred_phrase_starts = [
            candidate
            for candidate in viable
            if float(candidate.get("score", 0.0) or 0.0) >= preferred_floor
            and float((candidate.get("structure_components") or {}).get("phrase_prior", 0.0) or 0.0) >= 0.86
            and int((candidate.get("structure_components") or {}).get("clock_bar", 0) or 0) >= 9
        ]
        if preferred_phrase_starts:
            preferred_phrase_starts.sort(
                key=lambda row: (
                    int((row.get("structure_components") or {}).get("clock_bar", row.get("structure_bar", 999999)) or 999999),
                    -float(row.get("score", 0.0) or 0.0),
                )
            )
            first = dict(preferred_phrase_starts[0])
        else:
            floor = max(0.44, strongest_score - 0.12)
            near_best = [
                candidate
                for candidate in viable
                if float(candidate.get("score", 0.0) or 0.0) >= floor
                and bool((candidate.get("structure_components") or {}).get("on_one"))
            ]
            near_best.sort(key=lambda row: (int(row.get("structure_bar", 999999) or 999999), -float(row.get("score", 0.0) or 0.0)))
            first = dict(near_best[0]) if near_best else None
        if first is not None:
            first["structure_role"] = "first_drop"
            first["section_label"] = "first_drop"

    second = None
    if first:
        first_bar = int(first.get("structure_bar", 0) or 0)
        first_clock_bar = int((first.get("structure_components") or {}).get("clock_bar", first.get("structure_clock_bar", 0)) or 0)
        later = [
            candidate
            for candidate in candidates
            if int(candidate.get("structure_bar", 0) or 0) >= first_bar + 16
            and int((candidate.get("structure_components") or {}).get("clock_bar", candidate.get("structure_clock_bar", 0)) or 0)
            >= max(first_clock_bar + 16, 1)
            and bool((candidate.get("structure_components") or {}).get("on_one"))
            and one_distance(candidate) <= 40.0
            and float(candidate.get("score", 0.0) or 0.0) >= 0.38
            and float((candidate.get("structure_components") or {}).get("sustained_groove", 0.0) or 0.0) >= 0.32
            and float((candidate.get("structure_components") or {}).get("mini_fill_penalty", 0.0) or 0.0) < 0.55
        ]
        if later:
            later.sort(
                key=lambda row: (
                    -float((row.get("structure_components") or {}).get("phrase_prior", 0.0) or 0.0),
                    -float(row.get("score", 0.0) or 0.0),
                    int((row.get("structure_components") or {}).get("clock_bar", row.get("structure_bar", 999999)) or 999999),
                )
            )
            second = dict(later[0])
            second["structure_role"] = "second_drop"
            second["section_label"] = "second_drop"
    return first, second


def _section_ranges(bar_count: int, first: Optional[Mapping[str, Any]], second: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if bar_count <= 0:
        return []
    first_bar = int((first or {}).get("structure_bar", 0) or 0)
    second_bar = int((second or {}).get("structure_bar", 0) or 0)
    sections: List[Dict[str, Any]] = []

    def add(label: str, start: int, end: int) -> None:
        start = max(1, int(start))
        end = min(int(bar_count), int(end))
        if end >= start:
            sections.append({"label": label, "start_bar": start, "end_bar": end})

    if first_bar > 0:
        build_start = max(1, first_bar - 8)
        add("intro", 1, build_start - 1)
        add("build", build_start, first_bar - 1)
        if second_bar > first_bar:
            breakdown_start = max(first_bar + 8, second_bar - 8)
            add("drop", first_bar, breakdown_start - 1)
            add("breakdown", breakdown_start, max(breakdown_start, second_bar - 5))
            add("second_build", max(breakdown_start, second_bar - 4), second_bar - 1)
            add("second_drop", second_bar, min(bar_count, second_bar + 15))
            add("outro", min(bar_count, second_bar + 16), bar_count)
        else:
            add("drop", first_bar, min(bar_count, first_bar + 15))
            add("outro", min(bar_count, first_bar + 16), bar_count)
    else:
        add("intro", 1, bar_count)
    return sections


def _apply_section_labels(bar_lanes: List[Dict[str, Any]], sections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labeled = [dict(row, section_label="unknown") for row in bar_lanes]
    for section in sections:
        label = str(section.get("label") or "unknown")
        start = int(section.get("start_bar", 0) or 0)
        end = int(section.get("end_bar", 0) or 0)
        for row in labeled:
            bar = int(row.get("bar", 0) or 0)
            if start <= bar <= end:
                row["section_label"] = label
    return labeled


def analyze_track_structure(audio_path: str, *, sample_rate: int = 16000, use_cache: bool = True) -> Dict[str, Any]:
    feature_map = compute_bar_feature_map(audio_path, sample_rate=int(sample_rate), use_cache=use_cache)
    bars = list(feature_map.get("bars") or [])
    bar_lanes = [_bar_lane(bar) for bar in bars]
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    bpm = beatgrid.get("bpm")
    try:
        clock_zero_sec = float(beatgrid.get("bar_zero_sec", 0.0) or 0.0)
    except (TypeError, ValueError):
        clock_zero_sec = 0.0
    if len(bars) < 32:
        return {
            "ok": False,
            "version": STRUCTURE_MAP_VERSION,
            "audio_path": str(audio_path),
            "error": "insufficient_bar_count_for_structure",
            "beatgrid": feature_map.get("beatgrid") or {},
            "stem_group": feature_map.get("stem_group") or {},
            "bar_count": int(feature_map.get("bar_count", 0) or 0),
            "feature_cache_hit": bool(feature_map.get("cache_hit")),
            "bar_lanes": bar_lanes,
            "candidates": [],
            "first_drop": None,
            "second_drop": None,
            "sections": [],
        }
    scored: List[Dict[str, Any]] = []
    for idx in range(2, max(2, len(bars) - 4)):
        components = _bar_transition_score(bars, idx, bpm=bpm, clock_zero_sec=clock_zero_sec)
        bar_number = int(bars[idx].get("bar", idx + 1) or idx + 1)
        # The search space stays broad, but structurally implausible off-grid
        # peaks are left as low-ranked alternatives instead of hard failures.
        candidate = _candidate_from_bar(bars[idx], components, rank=0, role="drop_candidate")
        candidate["structure_bar"] = bar_number
        scored.append(candidate)
    scored.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), int(row.get("structure_bar", 999999) or 999999)))
    candidates = [dict(candidate, rank=index, handcrafted_rank=index) for index, candidate in enumerate(scored[:40], start=1)]
    first, second = _pick_first_second(candidates)
    sections = _section_ranges(int(feature_map.get("bar_count", 0) or 0), first, second)
    labeled_bar_lanes = _apply_section_labels(bar_lanes, sections)
    return {
        "ok": True,
        "version": STRUCTURE_MAP_VERSION,
        "audio_path": str(audio_path),
        "beatgrid": feature_map.get("beatgrid") or {},
        "stem_group": feature_map.get("stem_group") or {},
        "bar_count": int(feature_map.get("bar_count", 0) or 0),
        "feature_cache_hit": bool(feature_map.get("cache_hit")),
        "bar_lanes": labeled_bar_lanes,
        "candidates": candidates,
        "first_drop": first,
        "second_drop": second,
        "sections": sections,
    }
