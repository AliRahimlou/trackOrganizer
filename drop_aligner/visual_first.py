from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .detector import DropDetectorConfig, extract_features
from .microalign import microalign_marker
from .musical_clock import bpm_clock_for_time, phrase_strength_for_bar
from .multistem import find_stem_group, infer_bpm_from_path
from .structure_features import compute_bar_feature_map


VISUAL_FIRST_VERSION = 1
REJECTED_SECTION_BAR_RADIUS = 2
REJECTED_SECTION_TIME_RADIUS_SEC = 6.0


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _use_visual_drop_v2_result(result: Mapping[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    feature_map = result.get("feature_map") if isinstance(result.get("feature_map"), Mapping) else {}
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    reason = str(selected.get("reason") or "")
    try:
        bpm = float(beatgrid.get("bpm", 0.0) or 0.0)
        clock_bar = int(visual.get("clock_bar", 0) or 0)
        score = float(selected.get("score", selected.get("confidence_score", 0.0)) or 0.0)
        body = float(visual.get("body_score", 0.0) or 0.0)
        post4 = float(visual.get("post4_height", 0.0) or 0.0)
        post8 = float(visual.get("post8_height", 0.0) or 0.0)
        bass = float(visual.get("post_bass8", 0.0) or 0.0)
        pre_drum = float(visual.get("pre_drum_cont4", 0.0) or 0.0)
        transition = float(visual.get("transition", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    later_drop_gate = bool(
        "skipped full intro/build section" in reason
        and 33 <= clock_bar <= 37
        and score >= 0.580
        and body >= 0.700
    )
    raw_time = (
        _finite_float(result.get("raw_visual_time"))
        or _finite_float(selected.get("visual_raw_chunk_time"))
        or _finite_float(selected.get("timestamp"))
        or 0.0
    )
    first_low_downbeat = _finite_float(beatgrid.get("first_low_downbeat_sec"))
    honors_clock_intro = bool(
        first_low_downbeat is None
        or first_low_downbeat <= 0.0
        or raw_time >= first_low_downbeat - 0.250
    )
    # This branch is intentionally narrower than the v2 selector. It is only a
    # shortcut for later intro phrase starts; very early b9/b10 hits still fall
    # back to the older visual/candidate flow, which has better reviewed data.
    clear_song_start_gate = bool(
        "protected first clear song-start drop section" in reason
        and bpm >= 141.5
        and 17 <= clock_bar <= 21
        and score >= 0.430
        and body >= 0.560
        and post4 >= 0.550
        and post8 >= 0.560
        and bass >= 0.300
        and pre_drum <= 0.250
        and transition >= 0.150
        and honors_clock_intro
    )
    return bool(later_drop_gate or clear_song_start_gate)


def _candidate_rejected_by_section(
    candidate: Mapping[str, Any],
    rejected_sections: Sequence[Mapping[str, Any]],
    *,
    bar_radius: int = REJECTED_SECTION_BAR_RADIUS,
    time_radius_sec: float = REJECTED_SECTION_TIME_RADIUS_SEC,
) -> bool:
    if not rejected_sections:
        return False
    visual = candidate.get("visual_components") if isinstance(candidate.get("visual_components"), Mapping) else {}
    try:
        candidate_bar = int(visual.get("clock_bar", 0) or 0)
    except (TypeError, ValueError):
        candidate_bar = 0
    candidate_time = (
        _finite_float(candidate.get("visual_raw_chunk_time"))
        or _finite_float(candidate.get("timestamp"))
        or _finite_float(candidate.get("snapped_sec"))
        or _finite_float(candidate.get("time_sec"))
    )
    for rejected in rejected_sections:
        if not isinstance(rejected, Mapping):
            continue
        try:
            rejected_bar = int(rejected.get("clock_bar", 0) or 0)
        except (TypeError, ValueError):
            rejected_bar = 0
        if candidate_bar and rejected_bar and abs(candidate_bar - rejected_bar) <= int(bar_radius):
            return True
        rejected_time = _finite_float(rejected.get("raw_time")) or _finite_float(rejected.get("timestamp"))
        if (
            candidate_time is not None
            and rejected_time is not None
            and abs(float(candidate_time) - float(rejected_time)) <= float(time_radius_sec)
        ):
            return True
    return False


def _filter_rejected_sections(
    candidates: Sequence[Mapping[str, Any]],
    rejected_sections: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not rejected_sections:
        return rows
    kept = [row for row in rows if not _candidate_rejected_by_section(row, rejected_sections)]
    latest_rejected_bar = 0
    latest_rejected_time: Optional[float] = None
    for rejected in rejected_sections:
        if not isinstance(rejected, Mapping):
            continue
        try:
            rejected_bar = int(rejected.get("clock_bar", 0) or 0)
        except (TypeError, ValueError):
            rejected_bar = 0
        latest_rejected_bar = max(latest_rejected_bar, rejected_bar)
        rejected_time = _finite_float(rejected.get("raw_time")) or _finite_float(rejected.get("timestamp"))
        if rejected_time is not None:
            latest_rejected_time = max(latest_rejected_time or 0.0, float(rejected_time))

    if latest_rejected_bar >= 17 or (latest_rejected_time is not None and latest_rejected_time >= 24.0):
        later_rows: List[Dict[str, Any]] = []
        for row in kept:
            visual = row.get("visual_components") if isinstance(row.get("visual_components"), Mapping) else {}
            try:
                candidate_bar = int(visual.get("clock_bar", 0) or 0)
            except (TypeError, ValueError):
                candidate_bar = 0
            candidate_time = (
                _finite_float(row.get("visual_raw_chunk_time"))
                or _finite_float(row.get("timestamp"))
                or _finite_float(row.get("snapped_sec"))
                or _finite_float(row.get("time_sec"))
            )
            later_by_bar = bool(candidate_bar and candidate_bar > latest_rejected_bar + REJECTED_SECTION_BAR_RADIUS)
            later_by_time = bool(
                candidate_time is not None
                and latest_rejected_time is not None
                and float(candidate_time) > latest_rejected_time + REJECTED_SECTION_TIME_RADIUS_SEC
            )
            if later_by_bar or later_by_time:
                later_rows.append(row)
        if later_rows:
            return later_rows
    return kept or rows


def _visual_components(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("visual_components")
    return value if isinstance(value, Mapping) else {}


def _visual_clock_bar(candidate: Mapping[str, Any]) -> int:
    try:
        return int(_visual_components(candidate).get("clock_bar", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _visual_candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "snapped_sec", "time_sec", "visual_raw_chunk_time", "microaligned_time"):
        value = _finite_float(candidate.get(key))
        if value is not None:
            return float(value)
    return None


def _visual_candidate_raw_time(candidate: Mapping[str, Any]) -> Optional[float]:
    return _finite_float(candidate.get("visual_raw_chunk_time")) or _visual_candidate_time(candidate)


def _visual_candidate_score(candidate: Mapping[str, Any]) -> float:
    return _finite_float(candidate.get("score")) or _finite_float(candidate.get("confidence_score")) or 0.0


def _visual_drop_strength(candidate: Mapping[str, Any]) -> float:
    visual = _visual_components(candidate)
    phrase_bonus = 0.5 * _clip01((float(visual.get("phrase_prior", 0.0) or 0.0) - 0.48) / 0.44)
    section_shift = 0.16 if bool(visual.get("phrase_body_shift")) else 0.0
    local_reentry = 0.10 if bool(visual.get("local_reentry")) else 0.0
    return _clip01(
        (0.24 * float(visual.get("post8_height", 0.0) or 0.0))
        + (0.20 * float(visual.get("post_bass8", 0.0) or 0.0))
        + (0.18 * float(visual.get("post_drum8", 0.0) or 0.0))
        + (0.14 * float(visual.get("post_inst8", 0.0) or 0.0))
        + (0.09 * float(visual.get("jump8", 0.0) or 0.0))
        + (0.07 * float(visual.get("post_drum_cont8", 0.0) or 0.0))
        + phrase_bonus
        + section_shift
        + local_reentry
        - (0.08 * float(visual.get("pre_drum_cont4", 0.0) or 0.0))
    )


def _visual_real_drop_like(candidate: Mapping[str, Any]) -> bool:
    visual = _visual_components(candidate)
    clock_bar = _visual_clock_bar(candidate)
    post4 = float(visual.get("post4_height", 0.0) or 0.0)
    post8 = float(visual.get("post8_height", 0.0) or 0.0)
    bass = float(visual.get("post_bass8", 0.0) or 0.0)
    drum = float(visual.get("post_drum8", 0.0) or 0.0)
    phrase = float(visual.get("phrase_prior", 0.0) or 0.0)
    local_gap = float(visual.get("local_reentry_gap", 0.0) or 0.0)
    local_reentry = bool(visual.get("local_reentry"))
    phrase_shift = bool(visual.get("phrase_body_shift"))
    if clock_bar <= 0:
        return False
    return bool(
        (
            local_reentry
            and local_gap >= 0.220
            and bass >= 0.420
            and drum >= 0.880
            and post4 >= 0.560
            and post8 >= 0.560
        )
        or (
            phrase_shift
            and phrase >= 0.780
            and bass >= 0.320
            and drum >= 0.880
            and post4 >= 0.560
            and post8 >= 0.560
        )
    )


def _visual_instrumental_bass_section_entry(candidate: Mapping[str, Any]) -> bool:
    visual = _visual_components(candidate)
    inst = float(visual.get("post_inst8", 0.0) or 0.0)
    pre_inst = float(visual.get("pre_inst4", 0.0) or 0.0)
    return bool(
        _visual_clock_bar(candidate) >= 33
        and float(visual.get("phrase_prior", 0.0) or 0.0) >= 0.80
        and bool(visual.get("phrase_body_shift"))
        and inst >= 0.500
        and pre_inst <= 0.380
        and inst >= pre_inst + 0.220
        and float(visual.get("post_bass8", 0.0) or 0.0) >= 0.320
        and float(visual.get("post_drum8", 0.0) or 0.0) >= 0.900
        and float(visual.get("post4_height", 0.0) or 0.0) >= 0.580
        and float(visual.get("post8_height", 0.0) or 0.0) >= 0.580
    )


def _visual_has_earlier_real_drop_before(
    candidates: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> bool:
    candidate_bar = _visual_clock_bar(candidate)
    candidate_time = _visual_candidate_raw_time(candidate)
    for earlier in candidates:
        if earlier is candidate:
            continue
        earlier_bar = _visual_clock_bar(earlier)
        earlier_time = _visual_candidate_raw_time(earlier)
        if candidate_bar and earlier_bar:
            if earlier_bar >= candidate_bar:
                continue
        elif candidate_time is not None and earlier_time is not None:
            if earlier_time >= candidate_time:
                continue
        else:
            continue
        if _visual_real_drop_like(earlier):
            return True
    return False


def summarize_visual_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    visual = _visual_components(candidate)
    keys = (
        "clock_bar",
        "phrase_prior",
        "post4_height",
        "post8_height",
        "pre4_height",
        "jump4",
        "jump8",
        "post_bass8",
        "post_drum8",
        "post_inst8",
        "pre_inst4",
        "pre_drum_cont4",
        "post_drum_cont4",
        "post_drum_cont8",
        "local_reentry",
        "local_reentry_gap",
        "phrase_body_shift",
    )
    components = {key: visual.get(key) for key in keys if key in visual}
    return {
        "rank": candidate.get("rank"),
        "time": _visual_candidate_time(candidate),
        "raw_time": _visual_candidate_raw_time(candidate),
        "clock_bar": _visual_clock_bar(candidate),
        "score": _visual_candidate_score(candidate),
        "drop_strength": _visual_drop_strength(candidate),
        "selected_by": str(candidate.get("selected_by") or ""),
        "reason": str(candidate.get("reason") or ""),
        "real_drop_like": _visual_real_drop_like(candidate),
        "instrumental_bass_section_entry": _visual_instrumental_bass_section_entry(candidate),
        "visual_components": components,
    }


def audit_visual_selection(
    selected: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    rejected_sections: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    rows = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if selected:
        selected_time = _visual_candidate_time(selected)
        if not any(
            selected_time is not None
            and _visual_candidate_time(row) is not None
            and abs(float(_visual_candidate_time(row)) - float(selected_time)) <= 0.010
            for row in rows
        ):
            rows.insert(0, dict(selected))

    flags: List[Dict[str, Any]] = []
    selected_bar = _visual_clock_bar(selected)
    selected_time = _visual_candidate_time(selected)
    selected_strength = _visual_drop_strength(selected)
    preferred: Optional[Dict[str, Any]] = None
    recommended_action = "accept"

    if _candidate_rejected_by_section(selected, list(rejected_sections or [])):
        later = [
            row
            for row in rows
            if not _candidate_rejected_by_section(row, list(rejected_sections or []))
            and _visual_candidate_time(row) is not None
            and (selected_time is None or float(_visual_candidate_time(row)) > float(selected_time) + 0.750)
        ]
        later.sort(key=lambda row: (-_visual_drop_strength(row), _visual_candidate_time(row) or 1e18))
        if later:
            preferred = dict(later[0])
            recommended_action = "replace"
        flags.append(
            {
                "code": "selected_matches_rejected_section",
                "severity": "high",
                "message": "Selected blue marker lands in a section the user already skipped/rejected.",
            }
        )

    earlier_entries = [
        row
        for row in rows
        if _visual_instrumental_bass_section_entry(row)
        and not _visual_has_earlier_real_drop_before(rows, row)
        and _visual_candidate_time(row) is not None
        and selected_time is not None
        and float(_visual_candidate_time(row)) + 1.200 < float(selected_time)
        and (_visual_clock_bar(row) == 0 or selected_bar == 0 or _visual_clock_bar(row) + 1 < selected_bar)
    ]
    if earlier_entries:
        earlier_entries.sort(key=lambda row: (_visual_candidate_time(row) or 1e18, -_visual_drop_strength(row)))
        entry = dict(earlier_entries[0])
        if preferred is None or _visual_drop_strength(entry) >= _visual_drop_strength(preferred) - 0.050:
            preferred = entry
            recommended_action = "replace"
        flags.append(
            {
                "code": "late_body_after_section_entry",
                "severity": "high",
                "message": "Selected blue marker appears to be inside the drop body after an earlier first drop section entry.",
            }
        )

    if selected_bar and selected_bar <= 17:
        later_stronger = [
            row
            for row in rows
            if _visual_candidate_time(row) is not None
            and selected_time is not None
            and float(_visual_candidate_time(row)) > float(selected_time) + 6.0
            and _visual_clock_bar(row) >= selected_bar + 4
            and _visual_real_drop_like(row)
            and _visual_drop_strength(row) >= selected_strength + 0.140
        ]
        later_stronger.sort(key=lambda row: (-_visual_drop_strength(row), _visual_candidate_time(row) or 1e18))
        if later_stronger:
            if recommended_action != "replace":
                recommended_action = "review"
            flags.append(
                {
                    "code": "intro_before_stronger_drop",
                    "severity": "medium",
                    "message": "Selected blue marker is early while a later candidate has a stronger full drop profile.",
                    "preferred_candidate": summarize_visual_candidate(later_stronger[0]),
                }
            )

    body_shift_candidates = []
    if not _visual_instrumental_bass_section_entry(selected):
        body_shift_candidates = [
            row
            for row in rows
            if _visual_candidate_time(row) is not None
            and selected_time is not None
            and float(_visual_candidate_time(row)) < float(selected_time) - 0.500
            and bool(_visual_components(row).get("phrase_body_shift"))
            and _visual_drop_strength(row) >= selected_strength - 0.080
        ]
    if body_shift_candidates and selected_bar and _visual_clock_bar(body_shift_candidates[0]) < selected_bar:
        if recommended_action == "accept":
            recommended_action = "review"
        body_shift_candidates.sort(key=lambda row: (_visual_candidate_time(row) or 1e18))
        flags.append(
            {
                "code": "earlier_phrase_body_edge_available",
                "severity": "medium",
                "message": "An earlier phrase/body edge is close in strength; verify the blue marker is not late.",
                "preferred_candidate": summarize_visual_candidate(body_shift_candidates[0]),
            }
        )

    status = "pass"
    if recommended_action == "replace":
        status = "replace"
    elif flags:
        status = "review"

    return {
        "ok": bool(selected),
        "status": status,
        "recommended_action": recommended_action,
        "flags": flags,
        "flag_codes": [str(flag.get("code")) for flag in flags],
        "selected": summarize_visual_candidate(selected) if selected else {},
        "preferred_candidate": summarize_visual_candidate(preferred) if preferred else None,
        "candidate_count": len(rows),
        "candidates": [summarize_visual_candidate(row) for row in rows[:10]],
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _bar_height(bar: Mapping[str, Any]) -> float:
    return _clip01(
        (0.34 * float(bar.get("aggregate_energy", 0.0) or 0.0))
        + (0.26 * float(bar.get("groove_energy", 0.0) or 0.0))
        + (0.16 * float(bar.get("bass_low_energy", 0.0) or 0.0))
        + (0.14 * float(bar.get("drum_density", 0.0) or 0.0))
        + (0.10 * float(bar.get("instrumental_energy", 0.0) or 0.0))
    )


def _window_mean(heights: Sequence[float], start: int, end: int) -> float:
    return _mean(list(heights[max(0, int(start)) : min(len(heights), int(end))]))


def _bar_clock(bar: Mapping[str, Any], beatgrid: Mapping[str, Any]) -> Dict[str, Any]:
    return bpm_clock_for_time(
        float(bar.get("start_sec", 0.0) or 0.0),
        beatgrid.get("bpm"),
        clock_zero_sec=float(beatgrid.get("bar_zero_sec", 0.0) or 0.0),
    ) or {}


def _role_feature(bar: Mapping[str, Any], role: str, key: str) -> float:
    roles = bar.get("roles")
    if not isinstance(roles, Mapping):
        return 0.0
    data = roles.get(role)
    if not isinstance(data, Mapping):
        return 0.0
    return _clip01(data.get(key, 0.0))


def visual_chunk_candidates(feature_map: Mapping[str, Any], *, min_clock_bar: int = 5, max_clock_bar: int = 81) -> List[Dict[str, Any]]:
    bars = [dict(row) for row in feature_map.get("bars") or [] if isinstance(row, Mapping)]
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    if len(bars) < 8:
        return []

    heights = [_bar_height(bar) for bar in bars]
    bass_heights = [_clip01(bar.get("bass_low_energy", 0.0)) for bar in bars]
    drum_heights = [_clip01(bar.get("drum_density", 0.0)) for bar in bars]
    inst_heights = [_clip01(bar.get("instrumental_energy", 0.0)) for bar in bars]
    vocal_heights = [_clip01(bar.get("vocal_presence", 0.0)) for bar in bars]
    drum_continuity = [_role_feature(bar, "drums", "rms_occupancy") for bar in bars]
    max_post8 = max((_window_mean(heights, idx, idx + 8) for idx in range(max(1, len(heights) - 7))), default=0.0)
    candidates: List[Dict[str, Any]] = []
    for idx, bar in enumerate(bars[:-3]):
        clock = _bar_clock(bar, beatgrid)
        clock_bar = int(clock.get("nearest_one_bar", 0) or 0)
        one_distance_value = clock.get("one_distance_ms")
        one_distance_ms = 999999.0 if one_distance_value is None else float(one_distance_value)
        if clock_bar < int(min_clock_bar) or clock_bar > int(max_clock_bar):
            continue
        bpm_value = float(beatgrid.get("bpm", 0.0) or 0.0)
        if clock_bar < 9 and bpm_value < 141.5:
            continue

        post4 = _window_mean(heights, idx, idx + 4)
        post8 = _window_mean(heights, idx, idx + 8)
        pre4 = _window_mean(heights, idx - 4, idx)
        pre8 = _window_mean(heights, idx - 8, idx)
        prev1 = float(heights[idx - 1]) if idx > 0 else pre4
        prev2 = _window_mean(heights, idx - 2, idx)
        jump4 = post4 - pre4
        jump8 = post8 - pre8
        local_bar = float(heights[idx])
        phrase_prior, phrase_name = phrase_strength_for_bar(clock_bar)
        post_bass8 = _window_mean(bass_heights, idx, idx + 8)
        pre_bass4 = _window_mean(bass_heights, idx - 4, idx)
        post_drum8 = _window_mean(drum_heights, idx, idx + 8)
        post_inst8 = _window_mean(inst_heights, idx, idx + 8)
        pre_inst4 = _window_mean(inst_heights, idx - 4, idx)
        post_vocal8 = _window_mean(vocal_heights, idx, idx + 8)
        pre_vocal4 = _window_mean(vocal_heights, idx - 4, idx)
        local_drum_continuity = float(drum_continuity[idx]) if idx < len(drum_continuity) else 0.0
        pre_drum_cont4 = _window_mean(drum_continuity, idx - 4, idx)
        post_drum_cont4 = _window_mean(drum_continuity, idx, idx + 4)
        post_drum_cont8 = _window_mean(drum_continuity, idx, idx + 8)
        local_reentry_gap = max(local_bar - prev1, local_bar - prev2, post4 - prev1)
        local_reentry = bool(
            local_reentry_gap >= 0.155
            and local_bar >= 0.42
            and post4 >= max(0.48, 0.68 * max_post8)
            and post8 >= max(0.50, 0.68 * max_post8)
        )
        on_phrase_one = bool(clock.get("on_one") and one_distance_ms <= 55.0 and phrase_prior >= 0.60)
        continuity_transition = bool(
            local_drum_continuity >= 0.85
            and post_drum_cont4 >= 0.85
            and pre_drum_cont4 <= 0.62
            and post4 >= max(0.50, 0.78 * max_post8)
            and post8 >= max(0.50, 0.78 * max_post8)
        )
        phrase_body_shift = bool(
            on_phrase_one
            and phrase_prior >= 0.86
            and post4 >= max(0.50, 0.74 * max_post8)
            and post8 >= max(0.50, 0.74 * max_post8)
            and post_drum_cont4 >= 0.62
            and post_bass8 >= 0.36
            and (
                post_bass8 >= pre_bass4 + 0.055
                or pre_inst4 >= post_inst8 + 0.180
                or pre_vocal4 >= post_vocal8 + 0.150
                or pre4 <= post4 - 0.085
            )
        )
        if not on_phrase_one and not local_reentry and not continuity_transition and not phrase_body_shift:
            continue
        sustained_bars = sum(1 for value in heights[idx : min(len(heights), idx + 8)] if value >= max(0.42, post8 - 0.08))
        sustained = _clip01(sustained_bars / 6.0)
        novelty = float(bar.get("timbre_novelty", 0.0) or 0.0)
        pre_space = _clip01(1.0 - pre4)

        starts_block = bool(
            post4 >= pre4 + 0.075
            or (local_bar >= pre4 + 0.105 and post4 >= max(0.42, pre4 + 0.035))
            or local_reentry
            or continuity_transition
            or phrase_body_shift
        )
        visually_large = bool(post8 >= max(0.44, 0.72 * max_post8) or post4 >= max(0.48, 0.74 * max_post8))
        if not starts_block or not visually_large:
            continue

        visual_score = _clip01(
            (0.36 * post8)
            + (0.20 * post4)
            + (0.15 * _clip01(jump8 + 0.18))
            + (0.10 * _clip01(jump4 + 0.15))
            + (0.08 * sustained)
            + (0.06 * float(phrase_prior))
            + (0.03 * novelty)
            + (0.02 * pre_space)
            + (0.04 * _clip01(post_bass8 + max(0.0, local_reentry_gap)))
            + (0.04 * _clip01(post_drum_cont4))
        )
        if visual_score < 0.42:
            continue
        marker_time = float(bar.get("start_sec", 0.0) or 0.0) if local_reentry else float(
            clock.get("nearest_one_time", bar.get("start_sec", 0.0)) or bar.get("start_sec", 0.0)
        )

        candidates.append(
            {
                "rank": 0,
                "handcrafted_rank": 0,
                "timestamp": float(marker_time),
                "snapped_sec": float(marker_time),
                "time_sec": float(marker_time),
                "score": float(visual_score),
                "confidence_score": float(visual_score),
                "selected_by": "visual_gui_chunk",
                "reason": (
                    "visual-only local waveform re-entry candidate"
                    if local_reentry
                    else "visual-only phrase body-shift drop candidate"
                    if phrase_body_shift
                    else "visual-only first sustained waveform block candidate"
                ),
                "structure_role": "first_drop",
                "section_label": "first_drop",
                "visual_first_version": VISUAL_FIRST_VERSION,
                "visual_components": {
                    "feature_bar": int(idx + 1),
                    "clock_bar": int(clock_bar),
                    "phrase": phrase_name,
                    "phrase_prior": float(phrase_prior),
                    "post4_height": float(post4),
                    "post8_height": float(post8),
                    "pre4_height": float(pre4),
                    "pre8_height": float(pre8),
                    "prev1_height": float(prev1),
                    "prev2_height": float(prev2),
                    "jump4": float(jump4),
                    "jump8": float(jump8),
                    "bar_height": float(local_bar),
                    "max_post8_height": float(max_post8),
                    "sustained": float(sustained),
                    "timbre_novelty": float(novelty),
                    "pre_space": float(pre_space),
                    "local_reentry": bool(local_reentry),
                    "local_reentry_gap": float(local_reentry_gap),
                    "post_bass8": float(post_bass8),
                    "pre_bass4": float(pre_bass4),
                    "post_drum8": float(post_drum8),
                    "post_inst8": float(post_inst8),
                    "pre_inst4": float(pre_inst4),
                    "post_vocal8": float(post_vocal8),
                    "pre_vocal4": float(pre_vocal4),
                    "drum_continuity": float(local_drum_continuity),
                    "pre_drum_cont4": float(pre_drum_cont4),
                    "post_drum_cont4": float(post_drum_cont4),
                    "post_drum_cont8": float(post_drum_cont8),
                    "phrase_body_shift": bool(phrase_body_shift),
                    "one_distance_ms": float(one_distance_ms),
                },
                "bpm_clock": dict(clock),
            }
        )

    candidates.sort(key=lambda row: (float(row["timestamp"]), -float(row["score"])))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
        row["handcrafted_rank"] = rank
    return candidates


def select_first_visual_chunk(candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not rows:
        return None

    def comp(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value = row.get("visual_components")
        return value if isinstance(value, Mapping) else {}

    def clock_bar(row: Mapping[str, Any]) -> int:
        return int(comp(row).get("clock_bar", 0) or 0)

    def post8(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post8_height", 0.0) or 0.0)

    def post4(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post4_height", 0.0) or 0.0)

    def pre4(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("pre4_height", 0.0) or 0.0)

    def jump(row: Mapping[str, Any]) -> float:
        return max(float(comp(row).get("jump4", 0.0) or 0.0), float(comp(row).get("jump8", 0.0) or 0.0))

    def bass(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_bass8", 0.0) or 0.0)

    def inst(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_inst8", 0.0) or 0.0)

    def vocal(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_vocal8", 0.0) or 0.0)

    def pre_inst(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("pre_inst4", 0.0) or 0.0)

    def pre_vocal(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("pre_vocal4", 0.0) or 0.0)

    def drum(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_drum8", 0.0) or 0.0)

    def phrase_prior(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("phrase_prior", 0.0) or 0.0)

    def bpm_for(row: Mapping[str, Any]) -> float:
        clock = row.get("bpm_clock") if isinstance(row.get("bpm_clock"), Mapping) else {}
        return float(clock.get("bpm", 0.0) or 0.0)

    def prev1(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("prev1_height", 0.0) or 0.0)

    def prev2(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("prev2_height", 0.0) or 0.0)

    def drum_cont(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("drum_continuity", 0.0) or 0.0)

    def pre_drum_cont(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("pre_drum_cont4", 0.0) or 0.0)

    def post_drum_cont(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_drum_cont4", 0.0) or 0.0)

    def post_drum_cont8(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("post_drum_cont8", 0.0) or 0.0)

    def first_dense_transition(row: Mapping[str, Any]) -> bool:
        return bool(
            drum_cont(row) >= 0.850
            and post_drum_cont(row) >= 0.850
            and pre_drum_cont(row) <= 0.500
            and post4(row) >= 0.560
            and post8(row) >= 0.520
        )

    def local_gap(row: Mapping[str, Any]) -> float:
        return float(comp(row).get("local_reentry_gap", 0.0) or 0.0)

    def is_local_reentry(row: Mapping[str, Any]) -> bool:
        return bool(comp(row).get("local_reentry"))

    def is_phrase_body_shift(row: Mapping[str, Any]) -> bool:
        return bool(comp(row).get("phrase_body_shift"))

    def first_clean_dense_reentry(row: Mapping[str, Any]) -> bool:
        return bool(
            is_local_reentry(row)
            and clock_bar(row) <= 17
            and local_gap(row) >= 0.300
            and jump(row) >= 0.250
            and pre_drum_cont(row) <= 0.080
            and post_drum_cont(row) >= 0.900
            and post_drum_cont8(row) >= 0.900
            and drum(row) >= 0.950
        )

    def instrumental_bass_section_entry(row: Mapping[str, Any]) -> bool:
        return bool(
            clock_bar(row) >= 33
            and phrase_prior(row) >= 0.80
            and is_phrase_body_shift(row)
            and inst(row) >= 0.500
            and pre_inst(row) <= 0.380
            and inst(row) >= pre_inst(row) + 0.220
            and bass(row) >= 0.320
            and drum(row) >= 0.900
            and post4(row) >= 0.580
            and post8(row) >= 0.580
        )

    def earlier_real_drop_before(row: Mapping[str, Any]) -> bool:
        row_bar = clock_bar(row)
        for earlier in rows:
            if clock_bar(earlier) >= row_bar:
                continue
            local_reentry_drop = bool(
                is_local_reentry(earlier)
                and local_gap(earlier) >= 0.220
                and bass(earlier) >= 0.480
                and drum(earlier) >= 0.900
                and post4(earlier) >= 0.580
                and post8(earlier) >= 0.580
            )
            phrase_body_drop = bool(
                is_phrase_body_shift(earlier)
                and phrase_prior(earlier) >= 0.800
                and bass(earlier) >= 0.500
                and drum(earlier) >= 0.900
                and post4(earlier) >= 0.600
                and post8(earlier) >= 0.600
            )
            if local_reentry_drop or phrase_body_drop:
                return True
        return False

    for row in rows:
        if not instrumental_bass_section_entry(row) or earlier_real_drop_before(row):
            continue
        selected = dict(row)
        selected["selected_by"] = "visual_gui_first_fat_block"
        selected["reason"] = (
            "visual-only selected first instrumental/bass section entry before later body peak; "
            f"{row.get('reason') or 'instrumental/bass section entry'}"
        )
        return selected

    def same_opening_block_upgrade(selected: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        selected_bar = clock_bar(selected)
        selected_score = float(selected.get("score", selected.get("confidence_score", 0.0)) or 0.0)
        selected_bpm = bpm_for(selected)
        if selected_bpm < 141.5:
            return None
        for later in rows:
            later_bar = clock_bar(later)
            if later_bar <= selected_bar:
                continue
            later_score = float(later.get("score", later.get("confidence_score", 0.0)) or 0.0)
            early_phrase_body_edge = bool(
                selected_bar <= 12
                and later_bar <= selected_bar + 8
                and later_score >= selected_score + 0.035
                and phrase_prior(later) >= 0.860
                and (is_phrase_body_shift(later) or phrase_prior(later) >= phrase_prior(selected) + 0.300)
                and post4(later) >= post4(selected) - 0.020
                and post8(later) >= post8(selected) - 0.020
                and bass(later) >= bass(selected) + 0.035
                and drum(later) >= 0.950
            )
            adjacent_phrase_edge = bool(
                21 <= selected_bar <= 33
                and later_bar <= selected_bar + 1
                and phrase_prior(selected) <= 0.250
                and phrase_prior(later) >= phrase_prior(selected) + 0.250
                and later_score >= selected_score + 0.005
                and post4(later) >= post4(selected) + 0.015
                and post8(later) >= post8(selected) - 0.015
                and bass(later) >= bass(selected) - 0.015
                and pre_drum_cont(later) <= 0.260
            )
            if early_phrase_body_edge or adjacent_phrase_edge:
                upgraded = dict(later)
                upgraded["selected_by"] = "visual_gui_first_fat_block"
                upgraded["reason"] = (
                    "visual-only shifted to stronger phrase/body edge inside same opening block; "
                    f"{later.get('reason') or 'nearby phrase/body edge'}"
                )
                upgraded["visual_edge_replaced_candidate"] = {
                    "timestamp": float(selected.get("timestamp", 0.0) or 0.0),
                    "clock_bar": int(selected_bar),
                    "score": float(selected_score),
                    "phrase_prior": float(phrase_prior(selected)),
                    "reason": str(selected.get("reason") or ""),
                }
                return upgraded
        return None

    for row in rows:
        row_bar = clock_bar(row)
        stronger_soon = False
        for later in rows:
            later_bar = clock_bar(later)
            if later_bar <= row_bar:
                continue
            bass_body_release = bool(
                later_bar <= row_bar + 24
                and row_bar <= 17
                and local_gap(later) >= 0.145
                and bass(later) >= bass(row) + 0.100
                and post8(later) >= post8(row) - 0.025
                and post4(later) >= post4(row) - 0.020
            )
            phrase_intro_to_drop = bool(
                later_bar <= row_bar + 24
                and row_bar <= 17
                and local_gap(later) >= 0.145
                and bass(later) >= bass(row) + 0.070
                and post8(later) >= post8(row) + 0.090
                and post4(later) >= post4(row) + 0.080
                and jump(later) >= max(0.120, jump(row) + 0.040)
            )
            post_intro_breakdown_reentry = bool(
                row_bar <= 13
                and 21 <= later_bar <= min(row_bar + 24, 29)
                and is_local_reentry(later)
                and phrase_prior(later) >= 0.86
                and local_gap(later) >= 0.245
                and post4(later) >= post4(row) - 0.055
                and post8(later) >= post8(row) - 0.045
                and bass(later) >= bass(row) - 0.020
                and drum(later) >= 0.950
                and post_drum_cont(later) >= 0.850
                and min(prev1(later), prev2(later)) <= 0.500
                and (
                    inst(later) <= inst(row) - 0.100
                    or vocal(later) <= vocal(row) - 0.180
                    or phrase_prior(later) >= phrase_prior(row) + 0.080
                )
            )
            nearby_phrase_body_after_predrop = bool(
                13 <= row_bar <= 16
                and later_bar <= row_bar + 3
                and phrase_prior(later) >= 0.94
                and post4(later) >= post4(row) - 0.060
                and post8(later) >= post8(row) - 0.055
                and bass(later) >= bass(row) - 0.045
                and drum(later) >= 0.950
                and post_drum_cont(later) >= post_drum_cont(row) + 0.070
                and pre_drum_cont(later) <= 0.550
                and (
                    inst(later) <= inst(row) - 0.060
                    or phrase_prior(later) >= phrase_prior(row) + 0.300
                    or jump(row) <= 0.120
                )
            )
            buildup_to_bass_drop = bool(
                later_bar <= row_bar + 8
                and local_gap(later) >= 0.210
                and post8(later) >= post8(row) - 0.030
                and post4(later) >= post4(row) - 0.030
                and (
                    (bass(later) >= bass(row) + 0.070 and inst(later) <= inst(row) - 0.140)
                    or (
                        bass(later) >= bass(row) - 0.020
                        and inst(later) <= inst(row) - 0.140
                        and vocal(later) <= vocal(row) - 0.080
                    )
                )
            )
            post_gap_drum_slam = bool(
                later_bar <= row_bar + 8
                and row_bar <= 17
                and is_local_reentry(later)
                and local_gap(later) >= max(0.240, local_gap(row) + 0.035)
                and drum(later) >= drum(row) + 0.050
                and post8(later) >= post8(row) - 0.025
                and min(prev1(later), prev2(later)) <= min(prev1(row), prev2(row)) - 0.090
            )
            intro_break_to_later_drum_drop = bool(
                15 <= row_bar <= 17
                and later_bar <= row_bar + 18
                and 31 <= later_bar <= 35
                and is_local_reentry(later)
                and post4(later) >= post4(row) - 0.040
                and post8(later) >= post8(row) - 0.020
                and drum(later) >= 0.950
                and post_drum_cont(later) >= 0.860
                and post_drum_cont(later) >= post_drum_cont(row) + 0.030
                and post_drum_cont8(later) >= post_drum_cont8(row) + 0.095
                and (
                    pre_drum_cont(later) <= pre_drum_cont(row) - 0.075
                    or (
                        pre_drum_cont(later) <= 0.300
                        and post_drum_cont8(later) >= post_drum_cont8(row) + 0.110
                    )
                )
                and (
                    pre4(later) <= pre4(row) - 0.055
                    or jump(later) >= jump(row) + 0.040
                    or (
                        pre_drum_cont(later) <= 0.300
                        and post_drum_cont8(later) >= post_drum_cont8(row) + 0.110
                    )
                )
            )
            early_dense_intro_to_later_body = bool(
                row_bar <= 17
                and 31 <= later_bar <= 37
                and (is_phrase_body_shift(later) or is_local_reentry(later) or phrase_prior(later) >= 0.86)
                and post_drum_cont(later) >= 0.780
                and drum(later) >= 0.950
                and post4(later) >= post4(row) - 0.120
                and post8(later) >= post8(row) - 0.130
                and bass(later) >= bass(row) - 0.020
                and (
                    bass(later) >= bass(row) + 0.050
                    or inst(later) <= inst(row) - 0.300
                    or pre_inst(later) >= inst(later) + 0.180
                    or pre_vocal(later) >= vocal(later) + 0.150
                    or (
                        pre4(later) <= post4(later) - 0.075
                        and post_drum_cont(later) >= 0.780
                    )
                )
            )
            mid_intro_to_phrase_body = bool(
                21 <= row_bar <= 25
                and 31 <= later_bar <= 37
                and later_bar <= row_bar + 12
                and (is_phrase_body_shift(later) or is_local_reentry(later) or phrase_prior(later) >= 0.86)
                and post_drum_cont(later) >= 0.780
                and drum(later) >= 0.950
                and post4(later) >= post4(row) - 0.055
                and post8(later) >= post8(row) - 0.050
                and bass(later) >= bass(row) + 0.035
                and (
                    pre4(later) <= post4(later) - 0.070
                    or post_drum_cont(later) >= post_drum_cont(row) + 0.120
                    or phrase_prior(later) >= phrase_prior(row) + 0.120
                )
            )
            nearby_bass_body_upgrade = bool(
                21 <= row_bar <= 33
                and later_bar <= row_bar + 8
                and local_gap(later) >= 0.140
                and bass(later) >= bass(row) + 0.040
                and post4(later) >= post4(row) + 0.040
                and post8(later) >= post8(row) + 0.020
                and drum(later) >= 0.950
                and post_drum_cont(later) >= 0.820
            )
            drum_stem_body_takeover = bool(
                33 <= row_bar <= 57
                and later_bar <= row_bar + 8
                and phrase_prior(later) >= 0.80
                and post_drum_cont(row) <= 0.52
                and post_drum_cont(later) >= 0.76
                and bass(later) >= max(0.35, bass(row) + 0.070)
                and inst(later) <= inst(row) - 0.240
                and post4(later) >= post4(row) - 0.030
                and post8(later) >= post8(row) - 0.030
                and drum(later) >= 0.950
            )
            sparse_to_dense_drum_drop = bool(
                later_bar <= row_bar + 8
                and row_bar <= 17
                and not first_clean_dense_reentry(row)
                and drum_cont(later) >= 0.850
                and post_drum_cont(later) >= 0.850
                and pre_drum_cont(later) <= 0.620
                and post4(later) >= post4(row) - 0.025
                and post8(later) >= post8(row) - 0.070
                and bass(later) >= bass(row) - 0.040
            )
            if later_bar > row_bar + 16 and not (
                bass_body_release
                or phrase_intro_to_drop
                or post_intro_breakdown_reentry
                or intro_break_to_later_drum_drop
                or early_dense_intro_to_later_body
                or mid_intro_to_phrase_body
            ):
                continue
            height_wins = post8(later) >= post8(row) + max(0.050, 0.075 * max(0.1, post8(row)))
            body_wins = post4(later) >= post4(row) + max(0.045, 0.065 * max(0.1, post4(row)))
            clearly_bigger_block = bool(
                post8(later) >= post8(row) + 0.095
                and post4(later) >= post4(row) + 0.085
            )
            immediate_stronger_reentry = bool(
                later_bar <= row_bar + 2
                and is_local_reentry(later)
                and local_gap(later) >= 0.200
                and post8(later) >= post8(row) + 0.035
                and post4(later) >= post4(row) + 0.035
                and jump(later) >= jump(row) + 0.040
            )
            overwhelming_later_drop = bool(
                later_bar <= row_bar + 24
                and post8(later) >= post8(row) + 0.130
                and post4(later) >= post4(row) + 0.115
                and bass(later) >= bass(row) + 0.070
            )
            texture_heavy_intro = bool(inst(row) >= 0.650 and vocal(row) >= 0.500)
            if first_clean_dense_reentry(row) and not texture_heavy_intro and not overwhelming_later_drop:
                continue
            protected_first_dense_transition = bool(
                row_bar <= 33
                and first_dense_transition(row)
                and not bass_body_release
                and not phrase_intro_to_drop
                and not post_intro_breakdown_reentry
                and not nearby_phrase_body_after_predrop
                and not early_dense_intro_to_later_body
                and not mid_intro_to_phrase_body
                and not overwhelming_later_drop
            )
            if protected_first_dense_transition:
                continue
            protected_local_reentry = bool(
                is_local_reentry(row)
                and local_gap(row) >= 0.240
                and (
                    bass(row) >= 0.430
                    or (
                        drum(row) >= 0.950
                        and post_drum_cont(row) >= 0.900
                        and post4(row) >= 0.580
                        and post8(row) >= 0.560
                    )
                )
                and not bass_body_release
                and not phrase_intro_to_drop
                and not post_intro_breakdown_reentry
                and not nearby_phrase_body_after_predrop
                and not buildup_to_bass_drop
                and not post_gap_drum_slam
                and not intro_break_to_later_drum_drop
                and not early_dense_intro_to_later_body
                and not mid_intro_to_phrase_body
                and not nearby_bass_body_upgrade
                and not drum_stem_body_takeover
                and not sparse_to_dense_drum_drop
                and not immediate_stronger_reentry
            )
            if protected_local_reentry:
                continue
            contrast_wins = bool(
                jump(later) >= jump(row) + 0.090
                and post8(later) >= post8(row) - 0.040
                and post4(later) >= post4(row) - 0.040
            )
            if (
                bass_body_release
                or phrase_intro_to_drop
                or post_intro_breakdown_reentry
                or nearby_phrase_body_after_predrop
                or buildup_to_bass_drop
                or post_gap_drum_slam
                or intro_break_to_later_drum_drop
                or early_dense_intro_to_later_body
                or mid_intro_to_phrase_body
                or nearby_bass_body_upgrade
                or drum_stem_body_takeover
                or sparse_to_dense_drum_drop
                or immediate_stronger_reentry
                or clearly_bigger_block
                or ((height_wins or body_wins) and jump(later) >= max(0.050, jump(row) - 0.060))
                or contrast_wins
            ):
                stronger_soon = True
                break
        if stronger_soon:
            continue
        selected = dict(row)
        selected["selected_by"] = "visual_gui_first_fat_block"
        selected["reason"] = "visual-only selected first real fat waveform block; smaller earlier buildup/fake block skipped"
        upgraded = same_opening_block_upgrade(selected)
        if upgraded is not None:
            return upgraded
        return selected
    selected = dict(rows[0])
    selected["selected_by"] = "visual_gui_first_fat_block"
    selected["reason"] = "visual-only selected earliest sustained waveform block"
    return selected


def _micro_time(micro: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        value = float(micro.get(key))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _zoomed_marker_time(
    raw_time: float,
    micro: Mapping[str, Any],
    visual_components: Optional[Mapping[str, Any]] = None,
) -> float:
    marker = _micro_time(micro, "microaligned_time")
    if marker is None:
        return float(raw_time)
    micro_confidence = micro.get("micro_confidence")
    try:
        micro_conf = float(micro_confidence)
    except (TypeError, ValueError):
        micro_conf = None
    visual = visual_components if isinstance(visual_components, Mapping) else {}
    pre4 = _clip01(visual.get("pre4_height", 0.0))
    jump4 = float(visual.get("jump4", 0.0) or 0.0) if visual else 0.0
    post_drum = _clip01(visual.get("post_drum8", 0.0))
    post_bass = _clip01(visual.get("post_bass8", 0.0))
    pre_drum = _clip01(visual.get("pre_drum_cont4", 0.0))
    attack_time = _micro_time(micro, "attack_start_time")
    zero_time = _micro_time(micro, "zero_crossing_time")
    knee_time = _micro_time(micro, "visual_onset_knee_time")
    try:
        knee_used = bool(float(micro.get("visual_onset_knee_used", 0.0) or 0.0) > 0.0)
    except (TypeError, ValueError):
        knee_used = False
    try:
        attack_clean = float(micro.get("attack_cleanliness", 0.0) or 0.0)
    except (TypeError, ValueError):
        attack_clean = 0.0
    try:
        attack_strength = float(micro.get("attack_peak_strength", 0.0) or 0.0)
    except (TypeError, ValueError):
        attack_strength = 0.0
    try:
        impact_conf = float(micro.get("impact_boundary_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        impact_conf = 0.0
    try:
        denoised_impact = float(micro.get("denoised_impact_strength", 0.0) or 0.0)
    except (TypeError, ValueError):
        denoised_impact = 0.0
    try:
        zero_quality = float(micro.get("zero_crossing_quality", 0.0) or 0.0)
    except (TypeError, ValueError):
        zero_quality = 0.0
    try:
        rms_rise = float(micro.get("rms_rise_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        rms_rise = 0.0
    try:
        peak_rise = float(micro.get("peak_rise_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        peak_rise = 0.0
    if (
        marker > float(raw_time) + 0.350
        and micro_conf is not None
        and micro_conf < 0.80
        and pre4 >= 0.500
        and jump4 <= 0.120
    ):
        busy_buildup_late_bang = bool(
            attack_time is not None
            and float(raw_time) + 0.550 <= float(attack_time) <= float(raw_time) + 1.350
            and post_drum >= 0.78
            and post_bass >= 0.30
            and attack_strength >= 0.65
            and denoised_impact >= 0.75
            and (
                (attack_clean >= 0.54 and impact_conf >= 0.40)
                or (rms_rise >= 0.75 and peak_rise >= 0.75)
            )
        )
        if busy_buildup_late_bang:
            if (
                zero_time is not None
                and zero_quality >= 0.45
                and float(attack_time) - 0.035 <= float(zero_time) <= float(attack_time) + 0.015
            ):
                return float(zero_time)
            return float(attack_time)
        return float(raw_time)
    if marker < float(raw_time) - 0.050:
        for key in ("impact_body_time", "visual_onset_knee_time", "centerline_boundary_time", "attack_start_time", "peak_time"):
            value = _micro_time(micro, key)
            if value is not None and float(raw_time) - 0.050 <= value <= float(raw_time) + 1.250:
                return float(value)
        return float(raw_time)
    if marker > float(raw_time) + 1.250:
        return float(raw_time)
    early_knee_before_bang = bool(
        knee_used
        and knee_time is not None
        and attack_time is not None
        and abs(float(marker) - float(knee_time)) <= 0.035
        and float(attack_time) >= float(marker) + 0.300
        and float(attack_time) <= float(raw_time) + 1.150
        and post_drum >= 0.78
        and post_bass >= 0.24
        and pre_drum <= 0.20
        and attack_clean >= 0.78
        and attack_strength >= 0.52
        and impact_conf >= 0.70
        and denoised_impact >= 0.66
    )
    if early_knee_before_bang:
        if (
            zero_time is not None
            and zero_quality >= 0.45
            and float(attack_time) - 0.030 <= float(zero_time) <= float(attack_time) + 0.010
        ):
            return float(zero_time)
        return float(attack_time)
    return float(marker)


def _visual_body_onset_time(audio_path: str, raw_time: float, visual_components: Mapping[str, Any]) -> Optional[float]:
    clock_bar = int(visual_components.get("clock_bar", 0) or 0)
    pre4 = _clip01(visual_components.get("pre4_height", 0.0))
    jump4 = float(visual_components.get("jump4", 0.0) or 0.0)
    post_bass = _clip01(visual_components.get("post_bass8", 0.0))
    if not (21 <= clock_bar <= 49 and pre4 >= 0.500 and jump4 <= 0.140 and post_bass >= 0.500):
        return None
    try:
        group = find_stem_group(audio_path)
        drums_path = group.roles.get("drums") or audio_path
        bpm = infer_bpm_from_path(audio_path)
        features = extract_features(
            drums_path,
            DropDetectorConfig(sample_rate=16000, use_drumprint=False),
            bpm=bpm,
        )
    except Exception:
        return None
    times = np.asarray(features.frame_times, dtype=np.float64)
    rms = np.asarray(features.rms, dtype=np.float64)
    low = np.asarray(features.low_energy, dtype=np.float64)
    start = float(raw_time) + 0.350
    end = float(raw_time) + 1.100
    mask = (times >= start) & (times <= end)
    if not bool(np.any(mask)):
        return None
    t = times[mask]
    r = rms[mask]
    l = low[mask]
    if t.size < 3 or float(np.max(r)) <= 0.0 or float(np.max(l)) <= 0.0:
        return None
    rms_threshold = max(float(np.quantile(r, 0.75)), 0.65 * float(np.max(r)), 0.35)
    low_threshold = max(float(np.quantile(l, 0.65)), 0.55 * float(np.max(l)), 0.25)
    hits = np.flatnonzero((r >= rms_threshold) & (l >= low_threshold))
    if hits.size == 0:
        return None
    coarse = float(t[int(hits[0])])
    try:
        import librosa

        offset = max(start, coarse - 0.160)
        duration = max(0.050, min(end, coarse + 0.160) - offset)
        audio, sr = librosa.load(drums_path, sr=44100, mono=True, offset=float(offset), duration=float(duration))
        window = max(1, int(0.004 * sr))
        hop = max(1, int(0.001 * sr))
        if audio.size > window:
            xs: list[float] = []
            peaks: list[float] = []
            for index in range(0, int(audio.size) - window, hop):
                chunk = audio[index : index + window]
                xs.append(float(offset) + ((index + (window / 2.0)) / float(sr)))
                peaks.append(float(np.max(np.abs(chunk))))
            peak_values = np.asarray(peaks, dtype=np.float64)
            if peak_values.size and float(np.max(peak_values)) > 0.0:
                peak_hits = np.flatnonzero(peak_values >= (0.98 * float(np.max(peak_values))))
                if peak_hits.size:
                    return float(xs[int(peak_hits[0])])
    except Exception:
        pass
    return float(coarse)


def _buildup_release_body_entry_marker(
    audio_path: str,
    raw_time: float,
    visual_components: Mapping[str, Any],
    micro: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    clock_bar = int(visual_components.get("clock_bar", 0) or 0)
    phrase_prior = float(visual_components.get("phrase_prior", 0.0) or 0.0)
    post_drum = _clip01(visual_components.get("post_drum8", 0.0))
    post_bass = _clip01(visual_components.get("post_bass8", 0.0))
    post_inst = _clip01(visual_components.get("post_inst8", 0.0))
    post_vocal = _clip01(visual_components.get("post_vocal8", 0.0))
    try:
        micro_conf = float(micro.get("micro_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        micro_conf = 0.0
    try:
        impact_conf = float(micro.get("impact_boundary_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        impact_conf = 0.0
    try:
        attack_clean = float(micro.get("attack_cleanliness", 0.0) or 0.0)
    except (TypeError, ValueError):
        attack_clean = 0.0

    build_texture = bool(
        17 <= clock_bar <= 49
        and post_drum >= 0.85
        and post_bass >= 0.38
        and post_inst >= 0.50
        and post_vocal >= 0.18
    )
    current_boundary_is_suspect = bool(
        micro_conf < 0.78
        or impact_conf < 0.60
        or attack_clean < 0.60
        or phrase_prior < 0.55
    )
    if not (build_texture and current_boundary_is_suspect):
        return None

    try:
        group = find_stem_group(audio_path)
        bpm = infer_bpm_from_path(audio_path)
        cfg = DropDetectorConfig(sample_rate=16000, use_drumprint=False)
        features_by_role = {
            role: extract_features(path, cfg, bpm=bpm)
            for role, path in group.roles.items()
            if role in {"drums", "instrumental", "vocals"}
        }
    except Exception:
        return None
    drums = features_by_role.get("drums")
    inst = features_by_role.get("instrumental")
    vocals = features_by_role.get("vocals")
    if drums is None or inst is None or vocals is None:
        return None

    def feature_mean(role: str, key: str, start: float, end: float) -> float:
        features = features_by_role.get(role)
        if features is None:
            return 0.0
        times = np.asarray(features.frame_times, dtype=np.float64)
        values = np.asarray(getattr(features, key), dtype=np.float64)
        if times.size == 0 or values.size == 0:
            return 0.0
        mask = (times >= float(start)) & (times < float(end))
        if not bool(np.any(mask)):
            return 0.0
        return float(np.mean(values[mask]))

    def feature_max(role: str, key: str, start: float, end: float) -> float:
        features = features_by_role.get(role)
        if features is None:
            return 0.0
        times = np.asarray(features.frame_times, dtype=np.float64)
        values = np.asarray(getattr(features, key), dtype=np.float64)
        if times.size == 0 or values.size == 0:
            return 0.0
        mask = (times >= float(start)) & (times < float(end))
        if not bool(np.any(mask)):
            return 0.0
        return float(np.max(values[mask]))

    duration = max(float(getattr(features, "duration_sec", 0.0) or 0.0) for features in features_by_role.values())
    scan_start = float(raw_time) + 1.80
    scan_end = min(float(raw_time) + 10.00, max(scan_start, duration * 0.55))
    if scan_end <= scan_start + 0.50:
        return None

    rows: List[Dict[str, float]] = []
    for time in np.arange(scan_start, scan_end, 0.032):
        t = float(time)
        drum_pre = feature_mean("drums", "rms", t - 1.20, t)
        drum_post = feature_mean("drums", "rms", t, t + 1.60)
        low_pre = feature_mean("drums", "low_energy", t - 1.20, t)
        low_post = feature_mean("drums", "low_energy", t, t + 1.60)
        inst_pre = feature_mean("instrumental", "rms", t - 1.80, t)
        inst_post = feature_mean("instrumental", "rms", t, t + 1.80)
        vocal_pre = feature_mean("vocals", "rms", t - 1.80, t)
        vocal_post = feature_mean("vocals", "rms", t, t + 1.80)
        inst_drop = max(0.0, inst_pre - inst_post)
        vocal_drop = max(0.0, vocal_pre - vocal_post)
        drum_hold = _clip01((0.62 * (drum_post / 0.72)) + (0.38 * (low_post / 0.45)))
        drum_gain = max(0.0, drum_post - drum_pre) + (0.50 * max(0.0, low_post - low_pre))
        attack = max(
            feature_max("drums", "combined_attack", t - 0.080, t + 0.120),
            0.70 * feature_max("instrumental", "combined_attack", t - 0.080, t + 0.120),
            0.70 * feature_max("vocals", "combined_attack", t - 0.080, t + 0.120),
        )
        texture_drop = (0.62 * inst_drop) + (0.38 * vocal_drop)
        paired_release = (0.55 * _clip01(inst_drop / 0.34)) + (0.45 * _clip01(vocal_drop / 0.18))
        score = _clip01(
            (0.32 * drum_hold)
            + (0.23 * _clip01(texture_drop / 0.45))
            + (0.18 * paired_release)
            + (0.15 * _clip01(attack))
            + (0.12 * _clip01(drum_gain / 0.18))
        )
        if (
            score >= 0.86
            and inst_drop >= 0.36
            and vocal_drop >= 0.16
            and drum_hold >= 0.72
            and attack >= 0.60
        ):
            rows.append(
                {
                    "score": float(score),
                    "time": t,
                    "inst_drop": float(inst_drop),
                    "vocal_drop": float(vocal_drop),
                    "drum_hold": float(drum_hold),
                    "drum_gain": float(drum_gain),
                    "attack": float(attack),
                }
            )
    if not rows:
        return None

    clusters: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = []
    previous_time: Optional[float] = None
    for row in rows:
        row_time = float(row["time"])
        if previous_time is None or row_time - previous_time <= 0.080:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
        previous_time = row_time
    if current:
        clusters.append(current)
    if not clusters:
        return None

    peaks = [max(cluster, key=lambda row: float(row["score"])) for cluster in clusters]
    max_peak = max(float(row["score"]) for row in peaks)
    viable_peaks = [peak for peak in peaks if float(peak["score"]) >= max(0.88, max_peak - 0.12)]
    if not viable_peaks:
        return None
    selected_peak = viable_peaks[0]
    scan_choice = selected_peak
    scan_time = float(scan_choice["time"])
    if scan_time <= float(raw_time) + 1.20:
        return None

    try:
        refined = microalign_marker(audio_path, scan_time, search_before_ms=120, search_after_ms=450)
    except Exception:
        refined = {
            "microaligned_time": scan_time,
            "micro_confidence": 0.0,
            "impact_boundary_confidence": 0.0,
            "snap_offset_ms": 0.0,
            "reason": "buildup release verifier MicroSnap failed",
        }
    marker = _micro_time(refined, "microaligned_time") or scan_time
    try:
        refined_micro = float(refined.get("micro_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        refined_micro = 0.0
    try:
        refined_impact = float(refined.get("impact_boundary_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        refined_impact = 0.0
    if marker <= float(raw_time) + 1.20 or marker > float(raw_time) + 10.50:
        return None
    if abs(marker - scan_time) > 0.250:
        return None
    if refined_micro < 0.82 and refined_impact < 0.88:
        return None

    return {
        "marker": float(marker),
        "scan_time": float(scan_time),
        "scan_score": float(scan_choice["score"]),
        "section_peak_score": float(selected_peak["score"]),
        "max_scan_score": float(max_peak),
        "inst_drop": float(scan_choice["inst_drop"]),
        "vocal_drop": float(scan_choice["vocal_drop"]),
        "drum_hold": float(scan_choice["drum_hold"]),
        "drum_gain": float(scan_choice["drum_gain"]),
        "attack": float(scan_choice["attack"]),
        "microalign": dict(refined),
    }


def _structure_section_guard_candidate(audio_path: str, selected: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    if not visual:
        return None

    selected_bar = int(visual.get("clock_bar", 0) or 0)
    selected_time = _micro_time(selected, "timestamp")
    selected_score = float(selected.get("score", selected.get("confidence_score", 0.0)) or 0.0)
    phrase_prior = float(visual.get("phrase_prior", 0.0) or 0.0)
    post_drum_cont = float(visual.get("post_drum_cont4", 0.0) or 0.0)
    local_gap = float(visual.get("local_reentry_gap", 0.0) or 0.0)

    weak_off_phrase_reentry = bool(
        25 <= selected_bar <= 33
        and bool(visual.get("local_reentry"))
        and phrase_prior <= 0.48
        and post_drum_cont <= 0.58
        and selected_score <= 0.61
        and local_gap >= 0.24
    )
    if not weak_off_phrase_reentry or selected_time is None:
        return None

    try:
        from .structure_map import analyze_track_structure

        structure = analyze_track_structure(audio_path, sample_rate=16000, use_cache=True)
    except Exception:
        return None
    first = structure.get("first_drop") if isinstance(structure.get("first_drop"), Mapping) else None
    if not first:
        return None

    first_time = _micro_time(first, "timestamp")
    components = first.get("structure_components") if isinstance(first.get("structure_components"), Mapping) else {}
    first_bar = int(components.get("clock_bar", first.get("structure_clock_bar", 0)) or 0)
    if first_time is None or first_time <= selected_time + 16.0:
        return None
    if not (33 <= first_bar <= 57):
        return None

    structure_score = float(first.get("score", first.get("confidence_score", 0.0)) or 0.0)
    structure_phrase = float(components.get("phrase_prior", 0.0) or 0.0)
    structure_height = float(components.get("block_height", 0.0) or 0.0)
    structure_sustain = float(components.get("sustained_groove", 0.0) or 0.0)
    structure_density = float(components.get("post_density", 0.0) or 0.0)
    structure_novelty = float(components.get("timbre_novelty", 0.0) or 0.0)
    structure_reentry = float(components.get("instrumental_reentry", 0.0) or 0.0)
    if not (
        structure_score >= 0.48
        and structure_phrase >= 0.86
        and structure_height >= 0.62
        and structure_sustain >= 0.62
        and structure_density >= 0.86
        and max(structure_novelty, structure_reentry) >= 0.45
    ):
        return None

    guarded = dict(first)
    guarded["selected_by"] = "visual_structure_section_guard"
    guarded["structure_role"] = "first_drop"
    guarded["section_label"] = "first_drop"
    guarded["reason"] = (
        "visual section guard replaced weak off-phrase local re-entry with section-map first drop; "
        f"{first.get('reason') or 'section-map first drop'}"
    )
    guarded["visual_guard_replaced_candidate"] = {
        "timestamp": float(selected_time),
        "clock_bar": int(selected_bar),
        "score": float(selected_score),
        "phrase_prior": float(phrase_prior),
        "post_drum_cont4": float(post_drum_cont),
        "local_reentry_gap": float(local_gap),
        "reason": str(selected.get("reason") or ""),
    }
    return guarded


def visual_first_marker(
    audio_path: str,
    *,
    sample_rate: int = 16000,
    use_cache: bool = True,
    rejected_sections: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    rejected_sections = list(rejected_sections or [])
    visual_v2: Optional[Dict[str, Any]] = None
    try:
        from .visual_drop_v2 import visual_drop_v2_marker

        visual_v2 = dict(visual_drop_v2_marker(audio_path, sample_rate=int(sample_rate), use_cache=use_cache))
        selected_v2 = (
            visual_v2.get("selected_candidate")
            if isinstance(visual_v2.get("selected_candidate"), Mapping)
            else {}
        )
        if _use_visual_drop_v2_result(visual_v2) and not _candidate_rejected_by_section(selected_v2, rejected_sections):
            visual_v2 = dict(visual_v2)
            visual_v2["visual_audit"] = audit_visual_selection(
                selected_v2,
                [row for row in visual_v2.get("candidates") or [] if isinstance(row, Mapping)],
                rejected_sections=rejected_sections,
            )
            return visual_v2
    except Exception:
        visual_v2 = None

    feature_map = compute_bar_feature_map(audio_path, sample_rate=int(sample_rate), use_cache=use_cache)
    candidates = visual_chunk_candidates(feature_map)
    candidates_for_selection = _filter_rejected_sections(candidates, rejected_sections)
    selected = select_first_visual_chunk(candidates_for_selection)
    if selected is None:
        return {
            "ok": False,
            "error": "no_visual_chunk_candidate",
            "feature_map": {
                "ok": bool(feature_map.get("ok")),
                "bar_count": int(feature_map.get("bar_count", 0) or 0),
                "beatgrid": feature_map.get("beatgrid") or {},
                "cache_hit": bool(feature_map.get("cache_hit")),
            },
            "candidates": [],
        }

    guarded = _structure_section_guard_candidate(audio_path, selected)
    if guarded is not None:
        selected = guarded
        candidates_for_selection = [dict(guarded)] + [dict(row) for row in candidates_for_selection]

    raw_time = float(selected.get("timestamp", 0.0) or 0.0)
    try:
        micro = microalign_marker(audio_path, raw_time, search_before_ms=60, search_after_ms=1200)
    except Exception as exc:
        micro = {
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
            "input_candidate_time": raw_time,
            "microaligned_time": raw_time,
            "snap_offset_ms": 0.0,
            "reason": "visual-only MicroSnap failed; kept visual block boundary",
        }
    marker = _zoomed_marker_time(
        raw_time,
        micro if isinstance(micro, Mapping) else {},
        selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {},
    )
    body_onset = _visual_body_onset_time(
        audio_path,
        raw_time,
        selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {},
    )
    original_micro_time = _micro_time(micro, "microaligned_time") if isinstance(micro, Mapping) else None
    try:
        original_micro_conf = float(micro.get("micro_confidence")) if isinstance(micro, Mapping) else None
    except (TypeError, ValueError):
        original_micro_conf = None
    body_override_allowed = bool(
        original_micro_time is not None
        and original_micro_time > float(raw_time) + 0.350
        and original_micro_conf is not None
        and original_micro_conf < 0.80
    )
    if body_onset is not None and body_override_allowed:
        marker = float(body_onset)
        if isinstance(micro, Mapping):
            micro = dict(micro)
            micro["visual_body_onset_time"] = float(body_onset)
            micro["visual_body_onset_used"] = True
            micro["reason"] = f"{micro.get('reason') or 'MicroSnap reviewed'}; visual body onset used"
    release_entry = None
    if body_onset is None or not body_override_allowed:
        release_entry = _buildup_release_body_entry_marker(
            audio_path,
            raw_time,
            selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {},
            micro if isinstance(micro, Mapping) else {},
        )
    if release_entry is not None:
        marker = float(release_entry["marker"])
        if isinstance(micro, Mapping):
            micro = dict(micro)
        else:
            micro = {}
        release_micro = release_entry.get("microalign") if isinstance(release_entry.get("microalign"), Mapping) else {}
        if release_micro:
            micro.update(dict(release_micro))
        micro["buildup_release_body_entry_time"] = float(release_entry["marker"])
        micro["buildup_release_body_entry_scan_time"] = float(release_entry["scan_time"])
        micro["buildup_release_body_entry_score"] = float(release_entry["scan_score"])
        micro["buildup_release_body_entry_peak_score"] = float(release_entry["section_peak_score"])
        micro["buildup_release_body_entry_used"] = True
        micro["buildup_release_body_entry_features"] = {
            "inst_drop": float(release_entry["inst_drop"]),
            "vocal_drop": float(release_entry["vocal_drop"]),
            "drum_hold": float(release_entry["drum_hold"]),
            "drum_gain": float(release_entry["drum_gain"]),
            "attack": float(release_entry["attack"]),
            "max_scan_score": float(release_entry["max_scan_score"]),
        }
        micro["reason"] = f"{micro.get('reason') or 'MicroSnap reviewed'}; buildup release body-entry verifier used"
    selected["visual_raw_chunk_time"] = float(raw_time)
    selected["timestamp"] = float(marker)
    selected["snapped_sec"] = float(marker)
    selected["microaligned_time"] = float(marker)
    if isinstance(micro, Mapping):
        micro = dict(micro)
        original_micro = _micro_time(micro, "microaligned_time")
        if original_micro is not None and abs(float(original_micro) - float(marker)) > 0.010:
            micro["original_microaligned_time"] = float(original_micro)
        micro["microaligned_time"] = float(marker)
        micro["snap_offset_ms"] = float((float(marker) - float(raw_time)) * 1000.0)
    selected["microalign"] = dict(micro)
    selected["reason"] = (
        f"{selected.get('reason')}; zoomed from visual block edge "
        f"{raw_time:.6f}s to transient/body {marker:.6f}s"
    )
    if candidates_for_selection:
        deduped: List[Dict[str, Any]] = [dict(selected)]
        for candidate in candidates_for_selection:
            if abs(float(candidate.get("timestamp", 0.0) or 0.0) - marker) > 0.010:
                deduped.append(dict(candidate))
        candidates = deduped[:10]
    audit = audit_visual_selection(selected, candidates, rejected_sections=rejected_sections)
    return {
        "ok": True,
        "version": VISUAL_FIRST_VERSION,
        "audio_path": str(audio_path),
        "marker": float(marker),
        "raw_visual_time": float(raw_time),
        "selected_candidate": selected,
        "candidates": [dict(row) for row in candidates[:10]],
        "visual_audit": audit,
        "feature_map": {
            "ok": bool(feature_map.get("ok")),
            "bar_count": int(feature_map.get("bar_count", 0) or 0),
            "duration_sec": float(feature_map.get("duration_sec", 0.0) or 0.0),
            "beatgrid": feature_map.get("beatgrid") or {},
            "cache_hit": bool(feature_map.get("cache_hit")),
        },
    }
