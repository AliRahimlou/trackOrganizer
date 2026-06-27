from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .microalign import microalign_marker
from .musical_clock import bpm_clock_for_time, phrase_strength_for_bar
from .multistem import infer_bpm_from_path
from .structure_features import compute_bar_feature_map


VISUAL_DROP_V2_VERSION = 1


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _window(values: Sequence[float], start: int, end: int) -> List[float]:
    return list(values[max(0, int(start)) : min(len(values), int(end))])


def _window_mean(values: Sequence[float], start: int, end: int) -> float:
    return _mean(_window(values, start, end))


def _role_value(bar: Mapping[str, Any], role: str, key: str) -> float:
    roles = bar.get("roles")
    if not isinstance(roles, Mapping):
        return 0.0
    data = roles.get(role)
    if not isinstance(data, Mapping):
        return 0.0
    return _clip01(data.get(key, 0.0))


def _bar_visual_height(bar: Mapping[str, Any]) -> float:
    drum_rms = _role_value(bar, "drums", "rms")
    drum_occ = _role_value(bar, "drums", "rms_occupancy")
    return _clip01(
        (0.28 * float(bar.get("aggregate_energy", 0.0) or 0.0))
        + (0.24 * float(bar.get("groove_energy", 0.0) or 0.0))
        + (0.18 * float(bar.get("bass_low_energy", 0.0) or 0.0))
        + (0.14 * max(float(bar.get("drum_density", 0.0) or 0.0), drum_rms))
        + (0.10 * drum_occ)
        + (0.06 * float(bar.get("instrumental_energy", 0.0) or 0.0))
    )


def visual_drop_v2_candidates(
    feature_map: Mapping[str, Any],
    *,
    min_clock_bar: int = 9,
    max_clock_bar: int = 81,
) -> List[Dict[str, Any]]:
    bars = [dict(row) for row in feature_map.get("bars") or [] if isinstance(row, Mapping)]
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    if len(bars) < 8:
        return []

    heights = [_bar_visual_height(bar) for bar in bars]
    bass = [_clip01(bar.get("bass_low_energy", 0.0)) for bar in bars]
    drum_density = [_clip01(bar.get("drum_density", 0.0)) for bar in bars]
    inst = [_clip01(bar.get("instrumental_energy", 0.0)) for bar in bars]
    vocal = [_clip01(bar.get("vocal_presence", 0.0)) for bar in bars]
    drum_occ = [_role_value(bar, "drums", "rms_occupancy") for bar in bars]
    drum_rms = [_role_value(bar, "drums", "rms") for bar in bars]

    scan_end = min(len(bars), max(49, min(int(max_clock_bar), int(len(bars) * 0.58))))
    max_body8 = max((_window_mean(heights, idx, idx + 8) for idx in range(0, max(1, scan_end - 7))), default=0.0)
    candidates: List[Dict[str, Any]] = []
    for idx, bar in enumerate(bars[:-3]):
        clock = bpm_clock_for_time(
            float(bar.get("start_sec", 0.0) or 0.0),
            beatgrid.get("bpm"),
            clock_zero_sec=float(beatgrid.get("bar_zero_sec", 0.0) or 0.0),
        ) or {}
        clock_bar = int(clock.get("nearest_one_bar", 0) or 0)
        if clock_bar < int(min_clock_bar) or clock_bar > int(max_clock_bar):
            continue

        post2 = _window_mean(heights, idx, idx + 2)
        post4 = _window_mean(heights, idx, idx + 4)
        post8 = _window_mean(heights, idx, idx + 8)
        pre2 = _window_mean(heights, idx - 2, idx)
        pre4 = _window_mean(heights, idx - 4, idx)
        pre8 = _window_mean(heights, idx - 8, idx)
        prev_floor = min(pre2, pre4, pre8)
        post_bass8 = _window_mean(bass, idx, idx + 8)
        post_drum_density8 = _window_mean(drum_density, idx, idx + 8)
        post_inst8 = _window_mean(inst, idx, idx + 8)
        post_vocal8 = _window_mean(vocal, idx, idx + 8)
        pre_inst8 = _window_mean(inst, idx - 8, idx)
        pre_vocal8 = _window_mean(vocal, idx - 8, idx)
        pre_drum_occ4 = _window_mean(drum_occ, idx - 4, idx)
        post_drum_occ4 = _window_mean(drum_occ, idx, idx + 4)
        post_drum_occ8 = _window_mean(drum_occ, idx, idx + 8)
        post_drum_rms4 = _window_mean(drum_rms, idx, idx + 4)
        pre_drum_rms4 = _window_mean(drum_rms, idx - 4, idx)
        local_height = float(heights[idx])
        prev1 = float(heights[idx - 1]) if idx > 0 else pre4
        prev2 = _window_mean(heights, idx - 2, idx)
        local_reentry_gap = max(local_height - prev1, local_height - prev2, post4 - prev1)
        transition = _clip01(max(post2, post4) - prev_floor)
        continuity_jump = _clip01(post_drum_occ4 - pre_drum_occ4)
        rms_jump = _clip01(post_drum_rms4 - pre_drum_rms4 + 0.08)
        body = _clip01(
            (0.28 * post8)
            + (0.24 * post4)
            + (0.18 * post_drum_occ8)
            + (0.16 * post_bass8)
            + (0.08 * post_drum_density8)
            + (0.06 * max(0.0, post8 - prev_floor))
        )
        phrase_prior, phrase_name = phrase_strength_for_bar(clock_bar)
        on_one = bool(clock.get("on_one"))
        one_distance = clock.get("one_distance_ms")
        one_distance_ms = 999999.0 if one_distance is None else float(one_distance)
        texture_release = bool(pre_inst8 >= post_inst8 + 0.220 and post_inst8 <= 0.320)
        phrase_body_entry = bool(
            phrase_prior >= 0.860
            and body >= 0.650
            and post4 >= max(0.620, 0.82 * max_body8)
            and post8 >= max(0.620, 0.84 * max_body8)
            and post_bass8 >= 0.500
            and post_drum_occ4 >= 0.880
            and post_drum_occ8 >= 0.900
            and post_vocal8 <= 0.360
            and texture_release
        )
        edge_like = bool(
            transition >= 0.095
            or local_reentry_gap >= 0.135
            or continuity_jump >= 0.180
            or (pre_drum_occ4 <= 0.360 and post_drum_occ4 >= 0.780)
            or phrase_body_entry
        )
        body_like = bool(
            body >= max(0.500, 0.72 * max_body8)
            and post4 >= max(0.500, 0.72 * max_body8)
            and post_drum_occ4 >= 0.520
        )
        if not edge_like or not body_like:
            continue

        score = _clip01(
            (0.32 * body)
            + (0.22 * transition)
            + (0.14 * continuity_jump)
            + (0.10 * post_bass8)
            + (0.08 * post_drum_occ8)
            + (0.06 * _clip01(local_reentry_gap + 0.12))
            + (0.05 * float(phrase_prior))
            + (0.03 * rms_jump)
        )
        marker_time = float(bar.get("start_sec", 0.0) or 0.0)
        if on_one and one_distance_ms <= 65.0:
            marker_time = float(clock.get("nearest_one_time", marker_time) or marker_time)
        candidates.append(
            {
                "rank": 0,
                "handcrafted_rank": 0,
                "timestamp": float(marker_time),
                "snapped_sec": float(marker_time),
                "time_sec": float(marker_time),
                "score": float(score),
                "confidence_score": float(score),
                "selected_by": "visual_drop_v2_candidate",
                "reason": "visual-v2 full-song waveform section edge candidate",
                "structure_role": "first_drop",
                "section_label": "first_drop",
                "visual_drop_v2_version": VISUAL_DROP_V2_VERSION,
                "visual_components": {
                    "feature_bar": int(idx + 1),
                    "clock_bar": int(clock_bar),
                    "phrase": phrase_name,
                    "phrase_prior": float(phrase_prior),
                    "post2_height": float(post2),
                    "post4_height": float(post4),
                    "post8_height": float(post8),
                    "pre2_height": float(pre2),
                    "pre4_height": float(pre4),
                    "pre8_height": float(pre8),
                    "prev_floor_height": float(prev_floor),
                    "bar_height": float(local_height),
                    "max_body8_height": float(max_body8),
                    "transition": float(transition),
                    "local_reentry_gap": float(local_reentry_gap),
                    "continuity_jump": float(continuity_jump),
                    "rms_jump": float(rms_jump),
                    "body_score": float(body),
                    "post_bass8": float(post_bass8),
                    "post_drum8": float(post_drum_density8),
                    "post_inst8": float(post_inst8),
                    "post_vocal8": float(post_vocal8),
                    "pre_inst8": float(pre_inst8),
                    "pre_vocal8": float(pre_vocal8),
                    "pre_drum_cont4": float(pre_drum_occ4),
                    "post_drum_cont4": float(post_drum_occ4),
                    "post_drum_cont8": float(post_drum_occ8),
                    "one_distance_ms": float(one_distance_ms),
                    "edge_like": bool(edge_like),
                    "body_like": bool(body_like),
                    "texture_release": bool(texture_release),
                    "phrase_body_entry": bool(phrase_body_entry),
                },
                "bpm_clock": dict(clock),
            }
        )

    candidates.sort(key=lambda row: (float(row["timestamp"]), -float(row["score"])))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
        row["handcrafted_rank"] = rank
    return candidates


def _components(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("visual_components")
    return value if isinstance(value, Mapping) else {}


def _clock_bar(candidate: Mapping[str, Any]) -> int:
    return int(_components(candidate).get("clock_bar", 0) or 0)


def _metric(candidate: Mapping[str, Any], key: str) -> float:
    return float(_components(candidate).get(key, 0.0) or 0.0)


def _micro_time(micro: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        value = float(micro.get(key))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _micro_metric(micro: Mapping[str, Any], key: str) -> float:
    try:
        value = float(micro.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _micro_body_hit_score(micro: Mapping[str, Any]) -> float:
    return _clip01(
        (0.24 * _micro_metric(micro, "micro_confidence"))
        + (0.24 * _micro_metric(micro, "impact_boundary_confidence"))
        + (0.18 * _micro_metric(micro, "attack_peak_strength"))
        + (0.16 * _micro_metric(micro, "denoised_impact_strength"))
        + (0.10 * _micro_metric(micro, "rms_rise_score"))
        + (0.08 * _micro_metric(micro, "peak_rise_score"))
    )


def _clear_song_start_drop(candidate: Mapping[str, Any]) -> bool:
    return bool(
        _clock_bar(candidate) <= 21
        and float(candidate.get("score", 0.0) or 0.0) >= 0.430
        and _metric(candidate, "body_score") >= 0.560
        and _metric(candidate, "post4_height") >= 0.550
        and _metric(candidate, "post8_height") >= 0.560
        and _metric(candidate, "post_bass8") >= 0.300
        and _metric(candidate, "pre_drum_cont4") <= 0.250
        and _metric(candidate, "post_drum_cont4") >= 0.840
        and _metric(candidate, "post_drum_cont8") >= 0.870
        and _metric(candidate, "post_inst8") <= 0.430
        and _metric(candidate, "transition") >= 0.150
    )


def _phrase_body_entry(candidate: Mapping[str, Any]) -> bool:
    visual = _components(candidate)
    return bool(
        visual.get("phrase_body_entry")
        or (
            _metric(candidate, "phrase_prior") >= 0.860
            and _metric(candidate, "body_score") >= 0.650
            and _metric(candidate, "post4_height") >= 0.620
            and _metric(candidate, "post8_height") >= 0.620
            and _metric(candidate, "post_bass8") >= 0.500
            and _metric(candidate, "post_drum_cont4") >= 0.880
            and _metric(candidate, "post_drum_cont8") >= 0.900
            and _metric(candidate, "post_inst8") <= 0.320
            and _metric(candidate, "post_vocal8") <= 0.360
        )
    )


def _vocal_texture_buildup_to_later_primary_drop(current: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    row_bar = _clock_bar(current)
    later_bar = _clock_bar(later)
    if row_bar > 21 or later_bar <= row_bar + 8 or later_bar > 57:
        return False
    if _metric(later, "phrase_prior") < 0.860:
        return False

    current_vocal = _metric(current, "post_vocal8")
    later_vocal = _metric(later, "post_vocal8")
    current_texture = max(current_vocal, _metric(current, "post_inst8"))
    later_texture = max(later_vocal, _metric(later, "post_inst8"))
    vocal_build = bool(
        current_vocal >= 0.500
        and later_vocal <= current_vocal - 0.240
        and later_vocal <= 0.420
    )
    darker_release = bool(
        later_texture <= current_texture - 0.140
        or _metric(later, "post_bass8") >= _metric(current, "post_bass8") + 0.035
    )
    comparable_body = bool(
        _metric(later, "body_score") >= max(0.560, _metric(current, "body_score") - 0.055)
        and _metric(later, "post4_height") >= _metric(current, "post4_height") - 0.030
        and _metric(later, "post8_height") >= _metric(current, "post8_height") - 0.030
    )
    real_drop_body = bool(
        _metric(later, "post_bass8") >= max(0.400, _metric(current, "post_bass8") + 0.030)
        and _metric(later, "post_drum8") >= 0.940
        and _metric(later, "post_drum_cont4") >= 0.720
        and _metric(later, "post_drum_cont8") >= 0.700
        and float(later.get("score", 0.0) or 0.0) >= max(0.390, float(current.get("score", 0.0) or 0.0) - 0.250)
    )
    return bool(vocal_build and darker_release and comparable_body and real_drop_body)


def _overwhelming_later_upgrade(current: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    return bool(
        _metric(later, "post4_height") >= _metric(current, "post4_height") + 0.110
        and _metric(later, "post8_height") >= _metric(current, "post8_height") + 0.080
        and _metric(later, "post_bass8") >= _metric(current, "post_bass8") + 0.080
        and float(later.get("score", 0.0) or 0.0) >= float(current.get("score", 0.0) or 0.0) - 0.020
    )


def _near_later_phrase_body_entry(current: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    row_bar = _clock_bar(current)
    later_bar = _clock_bar(later)
    if row_bar > 24 or later_bar <= row_bar or later_bar > row_bar + 8 or later_bar > 57:
        return False
    if not _phrase_body_entry(later):
        return False
    if float(later.get("score", 0.0) or 0.0) < max(0.390, float(current.get("score", 0.0) or 0.0) - 0.140):
        return False

    current_build_or_fill = bool(
        _metric(current, "phrase_prior") < 0.860
        or _metric(current, "post_drum_cont4") <= 0.860
        or _metric(current, "post_drum_cont8") <= 0.890
        or _metric(current, "post_inst8") >= _metric(later, "post_inst8") + 0.160
    )
    later_is_real_body = bool(
        _metric(later, "body_score") >= max(0.650, _metric(current, "body_score") - 0.020)
        and _metric(later, "post4_height") >= _metric(current, "post4_height") - 0.055
        and _metric(later, "post8_height") >= _metric(current, "post8_height") - 0.040
        and _metric(later, "post_bass8") >= _metric(current, "post_bass8") - 0.010
        and (
            _metric(later, "post_drum_cont8") >= _metric(current, "post_drum_cont8") + 0.040
            or _metric(later, "post_inst8") <= _metric(current, "post_inst8") - 0.140
            or _metric(later, "post_bass8") >= _metric(current, "post_bass8") + 0.035
        )
    )
    return bool(current_build_or_fill and later_is_real_body)


def _later_beats_intro_candidate(current: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    row_bar = _clock_bar(current)
    later_bar = _clock_bar(later)
    if row_bar > 21 or later_bar <= row_bar + 8 or later_bar > 57:
        return False
    if float(later.get("score", 0.0) or 0.0) < max(0.420, float(current.get("score", 0.0) or 0.0) - 0.120):
        return False
    if _clear_song_start_drop(current) and not _overwhelming_later_upgrade(current, later):
        return False
    comparable_body = bool(
        _metric(later, "body_score") >= max(0.54, _metric(current, "body_score") - 0.025)
        and _metric(later, "post4_height") >= _metric(current, "post4_height") - 0.055
        and _metric(later, "post8_height") >= _metric(current, "post8_height") - 0.035
    )
    stronger_body = bool(
        _metric(later, "post4_height") >= _metric(current, "post4_height") + 0.050
        or _metric(later, "post8_height") >= _metric(current, "post8_height") + 0.050
        or _metric(later, "post_bass8") >= _metric(current, "post_bass8") + 0.085
    )
    cleaner_drum_body = bool(
        _metric(later, "post_drum_cont8") >= _metric(current, "post_drum_cont8") + 0.095
        and _metric(later, "pre_drum_cont4") <= _metric(current, "pre_drum_cont4") - 0.070
        and _metric(later, "post4_height") >= _metric(current, "post4_height") - 0.060
    )
    later_has_reset = bool(
        _metric(later, "transition") >= 0.120
        or _metric(later, "continuity_jump") >= 0.150
        or _metric(later, "pre4_height") <= _metric(later, "post4_height") - 0.120
    )
    current_is_intro_activation = bool(
        _metric(current, "pre4_height") <= 0.360
        or _metric(current, "pre_drum_cont4") <= 0.180
        or row_bar <= 17
    )
    return bool(current_is_intro_activation and comparable_body and later_has_reset and (stronger_body or cleaner_drum_body))


def select_visual_drop_v2(candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not rows:
        return None

    scan_rows = [row for row in rows if _clock_bar(row) <= 81]
    early_intro_rows = [row for row in scan_rows if _clock_bar(row) <= 21]
    if early_intro_rows:
        texture_release_rows: List[Dict[str, Any]] = []
        for later in scan_rows:
            if _clock_bar(later) <= 21:
                continue
            matched_early = next(
                (early for early in early_intro_rows if _vocal_texture_buildup_to_later_primary_drop(early, later)),
                None,
            )
            if matched_early is None:
                continue
            promoted = dict(later)
            promoted_visual = dict(_components(promoted))
            promoted_visual["vocal_texture_buildup_release"] = True
            promoted["visual_components"] = promoted_visual
            promoted["visual_guard_replaced_candidate"] = {
                "timestamp": float(matched_early.get("timestamp", 0.0) or 0.0),
                "clock_bar": int(_clock_bar(matched_early)),
                "score": float(matched_early.get("score", 0.0) or 0.0),
                "post_vocal8": float(_metric(matched_early, "post_vocal8")),
                "post_bass8": float(_metric(matched_early, "post_bass8")),
                "reason": str(matched_early.get("reason") or ""),
            }
            texture_release_rows.append(promoted)
        if texture_release_rows:
            selected = dict(texture_release_rows[0])
            selected["selected_by"] = "visual_drop_v2"
            selected["reason"] = (
                "visual-v2 skipped vocal/texture buildup and selected first darker primary drop block"
            )
            return selected

        protected_starts = [row for row in early_intro_rows if _clear_song_start_drop(row)]
        if protected_starts:
            first_start = protected_starts[0]
            has_overwhelming_upgrade = any(
                _clock_bar(later) > _clock_bar(first_start) + 8
                and _clock_bar(later) <= 57
                and _overwhelming_later_upgrade(first_start, later)
                for later in scan_rows
            )
            if not has_overwhelming_upgrade:
                selected = dict(first_start)
                selected["selected_by"] = "visual_drop_v2"
                selected["reason"] = (
                    "visual-v2 protected first clear song-start drop section; later sections were not stronger enough"
                )
                return selected
        near_phrase_body_rows = [
            later
            for later in scan_rows
            if any(_near_later_phrase_body_entry(early, later) for early in early_intro_rows)
        ]
        if near_phrase_body_rows:
            selected = dict(near_phrase_body_rows[0])
            selected["selected_by"] = "visual_drop_v2"
            selected["reason"] = (
                "visual-v2 skipped nearby intro fill/build and selected the first phrase body drop entry"
            )
            return selected
        intro_beating_later = [
            later
            for later in scan_rows
            if _clock_bar(later) > 21
            and any(_later_beats_intro_candidate(early, later) for early in early_intro_rows)
        ]
        if intro_beating_later:
            selected = dict(intro_beating_later[0])
            selected["selected_by"] = "visual_drop_v2"
            selected["reason"] = (
                "visual-v2 skipped full intro/build section and selected the first later real drop block"
            )
            return selected

    for row in scan_rows:
        if any(_near_later_phrase_body_entry(row, later) for later in scan_rows):
            continue
        if any(_later_beats_intro_candidate(row, later) for later in scan_rows):
            continue
        selected = dict(row)
        selected["selected_by"] = "visual_drop_v2"
        selected["reason"] = (
            "visual-v2 selected first full-song drop section; intro/build/fake block skipped"
        )
        if _clear_song_start_drop(selected):
            selected["reason"] = (
                "visual-v2 protected first clear song-start drop section; later sections were not stronger enough"
            )
        return selected

    selected = dict(scan_rows[0] if scan_rows else rows[0])
    selected["selected_by"] = "visual_drop_v2"
    selected["reason"] = "visual-v2 selected earliest full-song waveform drop section"
    return selected


def _zoomed_marker(raw_time: float, micro: Mapping[str, Any], visual: Mapping[str, Any]) -> float:
    try:
        marker = float(micro.get("microaligned_time"))
    except (TypeError, ValueError):
        return float(raw_time)
    if not np.isfinite(marker) or marker <= 0.0:
        return float(raw_time)
    try:
        micro_conf = float(micro.get("micro_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        micro_conf = 0.0
    offset = marker - float(raw_time)
    if _phrase_body_entry({"visual_components": visual}) and offset < -0.012:
        return float(raw_time)
    if -0.080 <= offset <= 1.200 and micro_conf >= 0.40:
        return float(marker)
    impact = micro.get("impact_body_time")
    try:
        impact_time = float(impact)
    except (TypeError, ValueError):
        impact_time = 0.0
    if impact_time > 0 and -0.080 <= impact_time - float(raw_time) <= 1.200:
        return float(impact_time)
    if abs(offset) <= 0.150:
        return float(marker)
    return float(raw_time)


def _body_beat_probe_marker(
    audio_path: str,
    raw_time: float,
    selected: Mapping[str, Any],
    base_micro: Mapping[str, Any],
    bpm: float,
) -> Optional[Dict[str, Any]]:
    visual = _components(selected)
    if not (
        bool(visual.get("vocal_texture_buildup_release"))
        and _clock_bar(selected) >= 29
        and _metric(selected, "phrase_prior") >= 0.860
        and _metric(selected, "post_drum8") >= 0.940
        and _metric(selected, "post_bass8") >= 0.400
        and (
            _metric(selected, "transition") <= 0.080
            or _metric(selected, "pre4_height") >= _metric(selected, "post4_height") - 0.060
            or _metric(selected, "pre_drum_cont4") >= 0.350
        )
    ):
        return None
    if bpm <= 0.0:
        return None
    beat_sec = 60.0 / float(bpm)
    if not (0.250 <= beat_sec <= 0.750):
        return None

    base_score = _micro_body_hit_score(base_micro)
    best: Optional[Dict[str, Any]] = None
    for beat_index in (1, 2, 3):
        probe_time = float(raw_time) + (float(beat_index) * beat_sec)
        try:
            probe_micro = microalign_marker(audio_path, probe_time, search_before_ms=90, search_after_ms=220)
        except Exception:
            continue
        marker = _micro_time(probe_micro, "microaligned_time") or probe_time
        if abs(float(marker) - probe_time) > 0.140:
            continue
        score = _micro_body_hit_score(probe_micro)
        row = {
            "marker": float(marker),
            "probe_time": float(probe_time),
            "beat_index": int(beat_index),
            "score": float(score),
            "microalign": dict(probe_micro),
        }
        if best is None or score > float(best["score"]):
            best = row
    if best is None:
        return None

    best_micro = best["microalign"] if isinstance(best.get("microalign"), Mapping) else {}
    stronger_attack = bool(
        _micro_metric(best_micro, "attack_peak_strength") >= _micro_metric(base_micro, "attack_peak_strength") + 0.180
        or _micro_metric(best_micro, "denoised_impact_strength") >= _micro_metric(base_micro, "denoised_impact_strength") + 0.180
        or _micro_metric(best_micro, "impact_boundary_confidence")
        >= _micro_metric(base_micro, "impact_boundary_confidence") + 0.070
    )
    if not (
        float(best["score"]) >= max(0.880, base_score + 0.030)
        and _micro_metric(best_micro, "micro_confidence") >= 0.880
        and _micro_metric(best_micro, "impact_boundary_confidence") >= 0.880
        and stronger_attack
    ):
        return None
    return best


def visual_drop_v2_marker(audio_path: str, *, sample_rate: int = 16000, use_cache: bool = True) -> Dict[str, Any]:
    feature_map = compute_bar_feature_map(audio_path, sample_rate=int(sample_rate), use_cache=use_cache)
    candidates = visual_drop_v2_candidates(feature_map)
    selected = select_visual_drop_v2(candidates)
    if selected is None:
        return {
            "ok": False,
            "error": "no_visual_drop_v2_candidate",
            "candidates": [],
            "feature_map": {
                "ok": bool(feature_map.get("ok")),
                "bar_count": int(feature_map.get("bar_count", 0) or 0),
                "beatgrid": feature_map.get("beatgrid") or {},
                "cache_hit": bool(feature_map.get("cache_hit")),
            },
        }

    raw_time = float(selected.get("timestamp", 0.0) or 0.0)
    try:
        micro = microalign_marker(audio_path, raw_time, search_before_ms=80, search_after_ms=1400)
    except Exception as exc:
        micro = {
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
            "input_candidate_time": raw_time,
            "microaligned_time": raw_time,
            "snap_offset_ms": 0.0,
            "reason": "visual-v2 MicroSnap failed; kept visual section edge",
        }
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    marker = _zoomed_marker(raw_time, micro if isinstance(micro, Mapping) else {}, visual)
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    try:
        bpm = float(beatgrid.get("bpm", 0.0) or 0.0)
    except (TypeError, ValueError):
        bpm = 0.0
    beat_probe = _body_beat_probe_marker(
        audio_path,
        raw_time,
        selected,
        micro if isinstance(micro, Mapping) else {},
        bpm,
    )
    if beat_probe is not None:
        marker = float(beat_probe["marker"])
        if isinstance(micro, Mapping):
            base_micro = dict(micro)
        else:
            base_micro = {}
        probe_micro = dict(beat_probe.get("microalign") if isinstance(beat_probe.get("microalign"), Mapping) else {})
        probe_micro["visual_drop_v2_body_beat_probe_used"] = True
        probe_micro["visual_drop_v2_body_beat_probe_time"] = float(beat_probe["probe_time"])
        probe_micro["visual_drop_v2_body_beat_probe_index"] = int(beat_probe["beat_index"])
        probe_micro["visual_drop_v2_body_beat_probe_score"] = float(beat_probe["score"])
        probe_micro["visual_drop_v2_body_beat_probe_base_score"] = float(_micro_body_hit_score(base_micro))
        probe_micro["visual_drop_v2_body_beat_probe_base_time"] = float(raw_time)
        probe_micro["visual_drop_v2_body_beat_probe_base_micro"] = base_micro
        probe_micro["reason"] = (
            f"{probe_micro.get('reason') or 'MicroSnap reviewed'}; "
            "visual-v2 body beat probe selected stronger drop impact"
        )
        micro = probe_micro
    selected["visual_raw_chunk_time"] = float(raw_time)
    selected["timestamp"] = float(marker)
    selected["snapped_sec"] = float(marker)
    selected["time_sec"] = float(marker)
    selected["microaligned_time"] = float(marker)
    selected["source"] = "visual_drop_v2"
    selected["visual_first_version"] = VISUAL_DROP_V2_VERSION
    if isinstance(micro, Mapping):
        micro = dict(micro)
    else:
        micro = {}
    original_micro = micro.get("microaligned_time")
    try:
        original_micro_float = float(original_micro)
    except (TypeError, ValueError):
        original_micro_float = None
    if original_micro_float is not None and abs(original_micro_float - marker) > 0.010:
        micro["original_microaligned_time"] = original_micro_float
    micro["microaligned_time"] = float(marker)
    micro["snap_offset_ms"] = float((marker - raw_time) * 1000.0)
    micro["visual_drop_v2_used"] = True
    micro["reason"] = f"{micro.get('reason') or 'MicroSnap reviewed'}; visual-v2 zoomed section edge"
    selected["microalign"] = micro
    selected["reason"] = (
        f"{selected.get('reason')}; zoomed from visual section edge "
        f"{raw_time:.6f}s to transient/body {marker:.6f}s"
    )
    deduped: List[Dict[str, Any]] = [dict(selected)]
    for candidate in candidates:
        if abs(float(candidate.get("timestamp", 0.0) or 0.0) - marker) > 0.010:
            deduped.append(dict(candidate))
    return {
        "ok": True,
        "version": VISUAL_DROP_V2_VERSION,
        "audio_path": str(audio_path),
        "marker": float(marker),
        "raw_visual_time": float(raw_time),
        "selected_candidate": dict(selected),
        "candidates": deduped[:10],
        "feature_map": {
            "ok": bool(feature_map.get("ok")),
            "bar_count": int(feature_map.get("bar_count", 0) or 0),
            "duration_sec": float(feature_map.get("duration_sec", 0.0) or 0.0),
            "beatgrid": feature_map.get("beatgrid") or {
                "bpm": infer_bpm_from_path(audio_path),
            },
            "cache_hit": bool(feature_map.get("cache_hit")),
        },
    }
