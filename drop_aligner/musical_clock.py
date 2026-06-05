from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


ONE_TOLERANCE_MS = 40.0


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def phrase_strength_for_bar(bar_number: Any) -> Tuple[float, str]:
    try:
        bar = int(bar_number)
    except (TypeError, ValueError):
        return 0.18, "off-phrase"
    phase = max(0, bar - 1)
    if phase > 0 and phase % 32 == 0:
        return 1.00, "32-bar phrase start"
    if phase > 0 and phase % 24 == 0:
        return 0.96, "24-bar phrase start"
    if phase > 0 and phase % 16 == 0:
        return 0.94, "16-bar phrase start"
    if phase > 0 and phase % 8 == 0:
        return 0.86, "8-bar phrase start"
    if phase > 0 and phase % 12 == 0:
        return 0.80, "12-bar phrase start"
    if phase > 0 and phase % 4 == 0:
        return 0.66, "4-bar phrase start"
    if phase > 0 and phase % 2 == 0:
        return 0.48, "even-bar phrase start"
    return 0.18, "off-phrase"


def bpm_clock_for_time(
    time_sec: Any,
    bpm: Any,
    *,
    clock_zero_sec: float = 0.0,
    one_tolerance_ms: float = ONE_TOLERANCE_MS,
) -> Optional[Dict[str, Any]]:
    time_value = _float_or_none(time_sec)
    bpm_value = _float_or_none(bpm)
    if time_value is None or bpm_value is None or bpm_value <= 0.0:
        return None
    beat_sec = 60.0 / bpm_value
    bar_sec = beat_sec * 4.0
    if beat_sec <= 0.0 or bar_sec <= 0.0:
        return None

    t = max(0.0, float(time_value) - float(clock_zero_sec))
    beat_float = t / beat_sec
    nearest_beat = int(math.floor(beat_float + 0.5))
    nearest_beat_time = float(clock_zero_sec) + (nearest_beat * beat_sec)
    beat_distance_sec = abs(float(time_value) - nearest_beat_time)
    beat_distance_beats = beat_distance_sec / beat_sec

    one_float = t / bar_sec
    nearest_one_zero_based = int(math.floor(one_float + 0.5))
    nearest_one_time = float(clock_zero_sec) + (nearest_one_zero_based * bar_sec)
    one_distance_sec = abs(float(time_value) - nearest_one_time)
    one_distance_beats = one_distance_sec / beat_sec

    floored_bar = int(math.floor(t / bar_sec)) + 1
    nearest_one_bar = int(nearest_one_zero_based + 1)
    beat_in_bar = int((nearest_beat % 4) + 1)
    phrase_strength, phrase = phrase_strength_for_bar(nearest_one_bar)
    on_one = one_distance_sec * 1000.0 <= float(one_tolerance_ms)
    return {
        "source": "title_bpm_track_zero",
        "bpm": float(bpm_value),
        "beat_sec": float(beat_sec),
        "bar_sec": float(bar_sec),
        "clock_zero_sec": float(clock_zero_sec),
        "beat_index": int(nearest_beat),
        "beat_number": int(nearest_beat + 1),
        "beat_in_bar": int(beat_in_bar),
        "bar_number": int(floored_bar),
        "nearest_beat_time": float(nearest_beat_time),
        "distance_sec": float(beat_distance_sec),
        "distance_ms": float(beat_distance_sec * 1000.0),
        "distance_beats": float(beat_distance_beats),
        "nearest_one_bar": int(nearest_one_bar),
        "nearest_one_time": float(nearest_one_time),
        "one_distance_sec": float(one_distance_sec),
        "one_distance_ms": float(one_distance_sec * 1000.0),
        "one_distance_beats": float(one_distance_beats),
        "on_beat_score": float(max(0.0, 1.0 - (beat_distance_beats / 0.50))),
        "on_one_score": float(max(0.0, 1.0 - (one_distance_beats / 0.50))),
        "on_one": bool(on_one),
        "one_tolerance_ms": float(one_tolerance_ms),
        "phrase_bar": int(nearest_one_bar),
        "phrase": phrase if on_one else "off-one",
        "phrase_strength": float(phrase_strength if on_one else 0.0),
    }


def feature_grid_for_time(time_sec: Any, bpm: Any, *, bar_zero_sec: Any = None) -> Optional[Dict[str, Any]]:
    time_value = _float_or_none(time_sec)
    bpm_value = _float_or_none(bpm)
    bar_zero = _float_or_none(bar_zero_sec)
    if time_value is None or bpm_value is None or bpm_value <= 0.0 or bar_zero is None:
        return None
    beat_sec = 60.0 / bpm_value
    bar_sec = beat_sec * 4.0
    if beat_sec <= 0.0 or bar_sec <= 0.0:
        return None
    grid_index = ((float(time_value) - float(bar_zero)) / bar_sec) + 1.0
    nearest_grid = int(math.floor(grid_index + 0.5))
    nearest_time = float(bar_zero) + ((nearest_grid - 1) * bar_sec)
    distance_sec = abs(float(time_value) - nearest_time)
    return {
        "bpm": float(bpm_value),
        "bar_zero_sec": float(bar_zero),
        "grid_bar_index": float(grid_index),
        "nearest_grid_bar": int(max(1, nearest_grid)),
        "nearest_grid_time": float(nearest_time),
        "grid_distance_sec": float(distance_sec),
        "grid_distance_beats": float(distance_sec / beat_sec),
    }
