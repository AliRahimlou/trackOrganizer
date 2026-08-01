from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .detector import FeatureBundle


@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    beat_sec: float
    bar_sec: float
    bar_zero_sec: float
    downbeat_confidence: float
    first_low_downbeat_sec: float
    source_role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm": float(self.bpm),
            "beat_sec": float(self.beat_sec),
            "bar_sec": float(self.bar_sec),
            "bar_zero_sec": float(self.bar_zero_sec),
            "downbeat_confidence": float(self.downbeat_confidence),
            "first_low_downbeat_sec": float(self.first_low_downbeat_sec),
            "source_role": self.source_role,
        }


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _window_max(values: np.ndarray, times: np.ndarray, center: float, radius: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(center) - float(radius)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(center) + float(radius)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(values[i0:i1]))


def _window_mean(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.mean(values[i0:i1]))


def _robust_norm(values: np.ndarray, *, lo: float = 20.0, hi: float = 96.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    floor = float(np.percentile(finite, lo))
    ceiling = float(np.percentile(finite, hi))
    peak = float(np.max(finite))
    if not math.isfinite(floor):
        floor = 0.0
    if not math.isfinite(ceiling) or ceiling <= floor + 1e-9:
        ceiling = peak
    if not math.isfinite(ceiling) or ceiling <= floor + 1e-9:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - floor) / max(1e-9, ceiling - floor), 0.0, 1.0)


def _positive_delta(values: np.ndarray, *, lookback: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    frames = max(1, int(lookback))
    out = np.zeros_like(arr, dtype=np.float64)
    for idx in range(arr.size):
        start = max(0, idx - frames)
        previous = arr[start:idx]
        baseline = float(np.mean(previous)) if previous.size else float(arr[idx])
        out[idx] = max(0.0, float(arr[idx]) - baseline)
    return out


def _pattern_arrays(features: FeatureBundle) -> Optional[Dict[str, np.ndarray]]:
    arrays = [
        np.asarray(features.frame_times, dtype=np.float64),
        np.asarray(features.low_energy, dtype=np.float64),
        np.asarray(features.low_jump_curve, dtype=np.float64),
        np.asarray(features.rms, dtype=np.float64),
        np.asarray(features.onset, dtype=np.float64),
        np.asarray(features.combined_attack, dtype=np.float64),
        np.asarray(features.spectral_flux, dtype=np.float64),
    ]
    n = min(arr.size for arr in arrays)
    if n < 8:
        return None
    times = arrays[0][:n]
    if times.size < 8 or not np.all(np.isfinite(times[: min(8, times.size)])):
        return None
    hop = float(np.median(np.diff(times[: min(times.size, 64)]))) if times.size > 1 else 0.025
    if not math.isfinite(hop) or hop <= 0.0:
        hop = 0.025
    low = _robust_norm(arrays[1][:n])
    low_jump = _robust_norm(arrays[2][:n])
    rms = _robust_norm(arrays[3][:n])
    onset = _robust_norm(arrays[4][:n])
    attack = _robust_norm(arrays[5][:n])
    flux = _robust_norm(arrays[6][:n])
    low_onset = _robust_norm(_positive_delta(low, lookback=max(2, int(round(0.12 / max(1e-6, hop))))))
    hit = np.clip(
        (0.30 * attack)
        + (0.24 * onset)
        + (0.22 * rms)
        + (0.26 * low)
        + (0.28 * low_jump)
        + (0.22 * low_onset)
        + (0.10 * flux),
        0.0,
        1.0,
    )
    kick = np.clip(
        (0.42 * low_jump)
        + (0.34 * low)
        + (0.20 * low_onset)
        + (0.12 * rms)
        + (0.08 * attack),
        0.0,
        1.0,
    )
    snare = np.clip(
        (0.44 * flux)
        + (0.34 * onset)
        + (0.18 * attack)
        + (0.08 * rms)
        - (0.16 * low),
        0.0,
        1.0,
    )
    return {
        "times": times,
        "hit": hit,
        "kick": kick,
        "snare": snare,
        "combined": np.clip((0.55 * hit) + (0.45 * rms), 0.0, 1.0),
        "hop_sec": np.asarray([hop], dtype=np.float64),
    }


def _local_peak(values: np.ndarray, times: np.ndarray, center: float, radius: float) -> tuple[Optional[float], float, float]:
    if values.size == 0 or times.size == 0:
        return None, 0.0, 1.0
    radius_f = max(0.001, float(radius))
    i0 = int(np.searchsorted(times, max(0.0, float(center) - radius_f), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(center) + radius_f), side="right"))
    if i1 <= i0:
        return None, 0.0, 1.0
    local_values = np.asarray(values[i0:i1], dtype=np.float64)
    local_times = np.asarray(times[i0:i1], dtype=np.float64)
    distance_penalty = np.abs(local_times - float(center)) / radius_f * 0.04
    idx = int(np.argmax(local_values - distance_penalty))
    peak_time = float(local_times[idx])
    peak_value = float(local_values[idx])
    delta_norm = float(np.clip(abs(peak_time - float(center)) / radius_f, 0.0, 1.0))
    return peak_time, peak_value, delta_norm


def _window_peak(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(values[i0:i1]))


def _first_strong_time(values: np.ndarray, times: np.ndarray, center: float, *, back: float, forward: float) -> float:
    i0 = int(np.searchsorted(times, max(0.0, float(center) - float(back)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(center) + float(forward)), side="right"))
    if i1 <= i0:
        return float(center)
    local = np.asarray(values[i0:i1], dtype=np.float64)
    if local.size == 0:
        return float(center)
    peak = float(np.max(local))
    if peak <= 1e-9:
        return float(center)
    threshold = max(float(np.percentile(local, 72.0)), peak * 0.56)
    for offset, value in enumerate(local):
        if float(value) >= threshold:
            return float(times[i0 + offset])
    return float(times[i0 + int(np.argmax(local))])


def _dedupe_times(values: list[float], *, min_distance: float) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0.0):
        if all(abs(value - existing) > float(min_distance) for existing in out):
            out.append(value)
    return out


def _drum_pattern_candidate_starts(features: FeatureBundle, *, bpm: float, max_candidates: int = 72) -> list[float]:
    arrays = _pattern_arrays(features)
    if arrays is None:
        return []
    times = arrays["times"]
    hit = arrays["hit"]
    kick = arrays["kick"]
    hop = float(arrays["hop_sec"][0])
    beat = 60.0 / max(1e-6, float(bpm))
    if times.size < 8 or beat <= 0.0:
        return []
    curve = np.clip((0.56 * kick) + (0.32 * hit) + (0.12 * arrays["combined"]), 0.0, 1.0)
    valid = np.where((times >= 0.0) & (times <= max(0.0, float(features.duration_sec) - min(0.25, 0.5 * beat))))[0]
    if valid.size == 0:
        return []
    threshold = max(float(np.percentile(curve[valid], 74.0)), float(np.max(curve[valid])) * 0.34)
    distance = max(1, int(round((0.62 * beat) / max(1e-6, hop))))
    try:
        from scipy import signal

        peaks, _props = signal.find_peaks(
            curve,
            distance=distance,
            height=threshold,
            prominence=max(0.006, float(np.std(curve[valid])) * 0.16),
        )
    except Exception:
        peaks = np.asarray(
            [
                idx
                for idx in valid[1:-1]
                if curve[idx] >= threshold and curve[idx] >= curve[idx - 1] and curve[idx] >= curve[idx + 1]
            ],
            dtype=np.int64,
        )
    valid_set = {int(v) for v in valid}
    ranked = [int(idx) for idx in peaks if int(idx) in valid_set]
    ranked.sort(key=lambda idx: float(curve[idx]), reverse=True)
    starts: list[float] = []
    for idx in ranked[: max(1, int(max_candidates))]:
        peak_time = float(times[idx])
        edge_time = _first_strong_time(
            np.maximum(kick, hit),
            times,
            peak_time,
            back=max(0.080, 0.30 * beat),
            forward=max(0.035, 0.12 * beat),
        )
        starts.append(edge_time)
        starts.append(peak_time)
    return _dedupe_times(starts, min_distance=max(0.035, 0.16 * beat))


def _phase_distance(a: float, b: float, period: float) -> float:
    if period <= 0.0:
        return abs(float(a) - float(b))
    delta = abs(float(a) - float(b)) % float(period)
    return float(min(delta, abs(float(period) - delta)))


def _score_drum_pattern_bar_zero(
    features: FeatureBundle,
    *,
    bpm: float,
    bar_zero: float,
    arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, float]:
    arrays = arrays if arrays is not None else _pattern_arrays(features)
    if arrays is None:
        return {"confidence": 0.0, "score": 0.0}
    times = arrays["times"]
    hit = arrays["hit"]
    kick = arrays["kick"]
    snare = arrays["snare"]
    combined = arrays["combined"]
    bpm_value = float(bpm)
    if not math.isfinite(bpm_value) or bpm_value <= 0.0:
        return {"confidence": 0.0, "score": 0.0}
    beat = 60.0 / bpm_value
    bar = 4.0 * beat
    duration = min(float(features.duration_sec), float(times[-1]) if times.size else 0.0)
    if beat <= 0.0 or bar <= 0.0 or duration <= 0.0:
        return {"confidence": 0.0, "score": 0.0}
    hit_radius = float(np.clip(0.16 * beat, 0.035, 0.090))
    score_total = 0.0
    weight_total = 0.0
    penalty_total = 0.0
    penalty_weight = 0.0
    downbeat_total = 0.0
    downbeat_kick_total = 0.0
    downbeat_combined_total = 0.0
    downbeat_penalty_total = 0.0
    downbeat_tight_weight = 0.0
    downbeat_weight = 0.0
    offbeat_total = 0.0
    offbeat_kick_total = 0.0
    offbeat_combined_total = 0.0
    offbeat_weight = 0.0
    backbeat_snare_total = 0.0
    backbeat_snare_weight = 0.0
    non_backbeat_snare_total = 0.0
    non_backbeat_snare_weight = 0.0
    active_bar_count = 0

    max_bars = max(4, min(128, int(math.ceil(max(0.0, duration - float(bar_zero)) / bar)) + 1))
    for bar_index in range(max_bars):
        bar_start = float(bar_zero) + (bar_index * bar)
        if bar_start > duration + hit_radius:
            break
        activity = _window_peak(hit, times, bar_start - hit_radius, bar_start + bar + hit_radius)
        kick_activity = _window_peak(kick, times, bar_start - hit_radius, bar_start + bar + hit_radius)
        bar_activity = max(float(activity), float(kick_activity))
        if bar_activity < 0.055:
            continue
        active_bar_count += 1
        bar_weight = (0.30 + (0.70 * _clip01(bar_activity))) / (1.0 + (0.010 * bar_index))
        for beat_index in range(4):
            target = bar_start + (beat_index * beat)
            if target < 0.0 or target > duration + hit_radius:
                continue
            peak_time, peak_value, delta_norm = _local_peak(hit, times, target, hit_radius)
            del peak_time
            beat_weight = 1.50 if beat_index == 0 else (1.00 if beat_index == 2 else 0.74)
            weight = beat_weight * bar_weight
            score_total += peak_value * (1.0 - (0.52 * delta_norm)) * weight
            penalty_total += delta_norm * weight
            penalty_weight += weight
            weight_total += weight
            if beat_index == 0:
                kick_peak_time, kick_peak_value, kick_delta_norm = _local_peak(kick, times, target, hit_radius)
                combined_peak = _local_peak(combined, times, target, hit_radius)[1]
                if kick_peak_time is None:
                    kick_delta_norm = delta_norm
                downbeat_total += peak_value * bar_weight
                downbeat_kick_total += kick_peak_value * bar_weight
                downbeat_combined_total += combined_peak * bar_weight
                downbeat_penalty_total += kick_delta_norm * bar_weight
                if kick_delta_norm <= 0.52:
                    downbeat_tight_weight += bar_weight
                downbeat_weight += bar_weight
            else:
                offbeat_total += peak_value * bar_weight
                offbeat_kick_total += _local_peak(kick, times, target, hit_radius)[1] * bar_weight
                offbeat_combined_total += _local_peak(combined, times, target, hit_radius)[1] * bar_weight
                offbeat_weight += bar_weight
            snare_hit = _local_peak(snare, times, target, hit_radius)[1]
            if beat_index in {1, 3}:
                backbeat_snare_total += snare_hit * bar_weight
                backbeat_snare_weight += bar_weight
            else:
                non_backbeat_snare_total += snare_hit * bar_weight
                non_backbeat_snare_weight += bar_weight

    if weight_total <= 0.0 or downbeat_weight <= 0.0 or active_bar_count < 2:
        return {"confidence": 0.0, "score": 0.0}
    average_score = score_total / weight_total
    average_penalty = penalty_total / max(1e-9, penalty_weight)
    downbeat_average = downbeat_total / max(1e-9, downbeat_weight)
    downbeat_kick_average = downbeat_kick_total / max(1e-9, downbeat_weight)
    downbeat_combined_average = downbeat_combined_total / max(1e-9, downbeat_weight)
    downbeat_penalty = downbeat_penalty_total / max(1e-9, downbeat_weight)
    downbeat_tightness = downbeat_tight_weight / max(1e-9, downbeat_weight)
    offbeat_average = offbeat_total / max(1e-9, offbeat_weight)
    offbeat_kick_average = offbeat_kick_total / max(1e-9, offbeat_weight)
    offbeat_combined_average = offbeat_combined_total / max(1e-9, offbeat_weight)
    backbeat_snare_average = backbeat_snare_total / max(1e-9, backbeat_snare_weight)
    non_backbeat_snare_average = non_backbeat_snare_total / max(1e-9, non_backbeat_snare_weight)
    kick_advantage = max(0.0, downbeat_kick_average - offbeat_kick_average)
    combined_advantage = max(0.0, downbeat_combined_average - offbeat_combined_average)
    backbeat_advantage = max(0.0, backbeat_snare_average - non_backbeat_snare_average)
    score = (
        (1.22 * average_score)
        + (0.34 * downbeat_average)
        + (0.58 * downbeat_kick_average)
        + (0.30 * downbeat_combined_average)
        + (0.60 * kick_advantage)
        + (0.32 * combined_advantage)
        + (0.22 * backbeat_snare_average)
        + (0.56 * backbeat_advantage)
        + (0.26 * downbeat_tightness)
        - (0.46 * average_penalty)
        - (0.72 * downbeat_penalty)
    )
    confidence = _clip01((score + 0.20) / 2.20)
    return {
        "confidence": float(confidence),
        "score": float(score),
        "active_bar_count": float(active_bar_count),
        "average_score": float(average_score),
        "downbeat_average": float(downbeat_average),
        "downbeat_kick_average": float(downbeat_kick_average),
        "offbeat_kick_average": float(offbeat_kick_average),
        "kick_advantage": float(kick_advantage),
        "backbeat_snare_average": float(backbeat_snare_average),
        "non_backbeat_snare_average": float(non_backbeat_snare_average),
        "backbeat_advantage": float(backbeat_advantage),
        "downbeat_penalty": float(downbeat_penalty),
    }


def find_visual_drop_drum_downbeat(
    features: FeatureBundle,
    *,
    visual_time_sec: float,
    bpm: float,
    clock_zero_sec: Optional[float] = None,
    current_marker_sec: Optional[float] = None,
    search_before_sec: Optional[float] = None,
    search_after_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the local musical one inside an already-chosen visual drop region.

    This deliberately does not solve the whole-track drop problem. The caller
    must already have chosen a visible drop section; this function only decides
    which local drum/bass impact should become `1.1.1`.
    """

    arrays = _pattern_arrays(features)
    bpm_value = float(bpm)
    visual_time = float(visual_time_sec)
    if arrays is None or not math.isfinite(bpm_value) or bpm_value <= 0.0 or not math.isfinite(visual_time):
        return None
    beat = 60.0 / bpm_value
    bar = 4.0 * beat
    if beat <= 0.0 or bar <= 0.0:
        return None
    current_marker = current_marker_sec
    try:
        current_marker = float(current_marker) if current_marker is not None else None
    except (TypeError, ValueError):
        current_marker = None
    if current_marker is not None and not math.isfinite(current_marker):
        current_marker = None

    times = arrays["times"]
    hit = arrays["hit"]
    kick = arrays["kick"]
    snare = arrays["snare"]
    combined = arrays["combined"]
    hop = float(arrays["hop_sec"][0])
    duration = min(float(features.duration_sec), float(times[-1]) if times.size else 0.0)
    if times.size < 8 or duration <= 0.0:
        return None

    before = (
        float(search_before_sec)
        if search_before_sec is not None
        else max(0.18, min(3.0, 2.35 * beat))
    )
    after = (
        float(search_after_sec)
        if search_after_sec is not None
        else max(0.40, min(5.0, 3.20 * beat))
    )
    if not math.isfinite(before) or before <= 0.0:
        before = max(0.18, 2.35 * beat)
    if not math.isfinite(after) or after <= 0.0:
        after = max(0.40, 3.20 * beat)
    if current_marker is not None and visual_time <= current_marker <= visual_time + max(after, 1.50 * bar):
        after = max(after, (current_marker - visual_time) + (0.55 * beat))

    window_start = max(0.0, visual_time - before)
    window_end = min(duration, visual_time + after)
    if window_end <= window_start + max(0.080, 0.20 * beat):
        return None

    local_mask = np.where((times >= window_start) & (times <= window_end))[0]
    if local_mask.size < 3:
        return None

    curve = np.clip((0.52 * kick) + (0.28 * hit) + (0.20 * combined), 0.0, 1.0)
    local_curve = np.asarray(curve[local_mask], dtype=np.float64)
    local_peak = float(np.max(local_curve)) if local_curve.size else 0.0
    if local_peak < 0.055:
        return None
    local_threshold = max(
        float(np.percentile(local_curve, 66.0)),
        local_peak * 0.30,
        0.075,
    )
    peak_distance = max(1, int(round((0.30 * beat) / max(1e-6, hop))))
    try:
        from scipy import signal

        peak_indexes, _props = signal.find_peaks(
            curve,
            distance=peak_distance,
            height=local_threshold,
            prominence=max(0.004, float(np.std(local_curve)) * 0.12),
        )
        local_index_set = {int(value) for value in local_mask}
        peak_indexes = np.asarray([idx for idx in peak_indexes if int(idx) in local_index_set], dtype=np.int64)
    except Exception:
        peak_indexes = np.asarray(
            [
                int(idx)
                for idx in local_mask[1:-1]
                if curve[idx] >= local_threshold and curve[idx] >= curve[idx - 1] and curve[idx] >= curve[idx + 1]
            ],
            dtype=np.int64,
        )
    if peak_indexes.size == 0:
        ranked_local = sorted((int(idx) for idx in local_mask), key=lambda idx: float(curve[idx]), reverse=True)
        peak_indexes = np.asarray(ranked_local[:8], dtype=np.int64)

    candidate_times: list[float] = [float(visual_time)]
    if current_marker is not None and window_start <= current_marker <= window_end:
        candidate_times.append(float(current_marker))
    ranked_peaks = sorted((int(idx) for idx in peak_indexes), key=lambda idx: float(curve[idx]), reverse=True)
    for idx in ranked_peaks[:32]:
        peak_time = float(times[idx])
        edge_time = _first_strong_time(
            curve,
            times,
            peak_time,
            back=max(0.050, 0.24 * beat),
            forward=max(0.030, 0.10 * beat),
        )
        if window_start <= edge_time <= window_end:
            candidate_times.append(edge_time)
        candidate_times.append(peak_time)

    body_threshold = max(float(np.percentile(local_curve, 70.0)), local_peak * 0.38, 0.095)
    for idx in sorted(int(value) for value in local_mask):
        t = float(times[idx])
        if float(curve[idx]) < body_threshold:
            continue
        post = _window_mean(combined, times, t, min(duration, t + max(0.35, 1.10 * beat)))
        if post >= max(0.060, body_threshold * 0.38):
            candidate_times.append(t)
            break

    zero = None
    if clock_zero_sec is not None:
        try:
            zero = float(clock_zero_sec)
        except (TypeError, ValueError):
            zero = None
    if zero is not None and math.isfinite(zero):
        start_bar = int(math.floor((window_start - zero) / bar)) - 1
        end_bar = int(math.ceil((window_end - zero) / bar)) + 1
        for bar_index in range(start_bar, end_bar + 1):
            grid_time = zero + (bar_index * bar)
            if window_start <= grid_time <= window_end:
                candidate_times.append(float(grid_time))
                peak_time, peak_value, _delta = _local_peak(curve, times, grid_time, max(0.050, 0.18 * beat))
                if peak_time is not None and peak_value >= local_threshold * 0.70:
                    edge_time = _first_strong_time(
                        curve,
                        times,
                        peak_time,
                        back=max(0.050, 0.22 * beat),
                        forward=max(0.030, 0.10 * beat),
                    )
                    if window_start <= edge_time <= window_end:
                        candidate_times.append(float(edge_time))

    candidates = _dedupe_times(
        [value for value in candidate_times if window_start <= float(value) <= window_end],
        min_distance=max(0.018, 0.055 * beat),
    )
    if not candidates:
        return None

    hit_radius = float(np.clip(0.16 * beat, 0.030, 0.095))
    scored: list[Dict[str, Any]] = []
    for candidate_time in candidates:
        t = float(candidate_time)
        hit_peak_time, hit_peak, hit_delta = _local_peak(hit, times, t, hit_radius)
        kick_peak_time, kick_peak, kick_delta = _local_peak(kick, times, t, hit_radius)
        combined_peak = _local_peak(combined, times, t, hit_radius)[1]
        curve_peak_time, curve_peak, curve_delta = _local_peak(curve, times, t, hit_radius)
        peak_time = curve_peak_time or kick_peak_time or hit_peak_time or t
        edge_time = _first_strong_time(
            curve,
            times,
            float(peak_time),
            back=max(0.035, 0.16 * beat),
            forward=max(0.025, 0.08 * beat),
        )
        if window_start <= edge_time <= window_end:
            t = float(edge_time)
            hit_peak_time, hit_peak, hit_delta = _local_peak(hit, times, t, hit_radius)
            kick_peak_time, kick_peak, kick_delta = _local_peak(kick, times, t, hit_radius)
            combined_peak = _local_peak(combined, times, t, hit_radius)[1]
            curve_peak_time, curve_peak, curve_delta = _local_peak(curve, times, t, hit_radius)

        pre_body = _window_mean(combined, times, t - max(0.30, 1.25 * beat), t - max(0.050, 0.12 * beat))
        post_body = _window_mean(combined, times, t, min(duration, t + max(0.35, 1.60 * beat)))
        post_peak = _window_peak(combined, times, t, min(duration, t + max(0.25, 1.25 * beat)))
        body_jump = max(0.0, post_body - pre_body)
        visible_energy = max(float(hit_peak), float(kick_peak), float(combined_peak), float(post_peak), float(curve_peak))
        if visible_energy < 0.105 and post_body < 0.085:
            continue

        backbeat_values: list[float] = []
        non_backbeat_values: list[float] = []
        offbeat_kick_values: list[float] = []
        for beat_index in range(1, 4):
            target = t + (beat_index * beat)
            if target > duration + hit_radius:
                continue
            snare_value = _local_peak(snare, times, target, hit_radius)[1]
            if beat_index in {1, 3}:
                backbeat_values.append(float(snare_value))
            else:
                non_backbeat_values.append(float(snare_value))
            offbeat_kick_values.append(float(_local_peak(kick, times, target, hit_radius)[1]))
        backbeat_snare = float(np.mean(backbeat_values)) if backbeat_values else 0.0
        non_backbeat_snare = float(np.mean(non_backbeat_values)) if non_backbeat_values else 0.0
        offbeat_kick = float(np.mean(offbeat_kick_values)) if offbeat_kick_values else 0.0
        kick_advantage = max(0.0, float(kick_peak) - (0.72 * offbeat_kick))
        backbeat_advantage = max(0.0, backbeat_snare - non_backbeat_snare)

        grid_bonus = 0.0
        grid_distance_ms: Optional[float] = None
        if zero is not None and math.isfinite(zero):
            grid_distance = _phase_distance(t, zero, bar)
            grid_distance_ms = float(grid_distance * 1000.0)
            grid_bonus = 1.0 - _clip01(grid_distance / max(1e-9, 0.22 * beat))

        delta_from_visual = t - visual_time
        early_penalty = _clip01((abs(min(0.0, delta_from_visual)) - (1.08 * beat)) / max(1e-9, 1.25 * beat))
        late_penalty = _clip01((max(0.0, delta_from_visual) - (0.55 * beat)) / max(1e-9, 1.75 * beat))
        earliest_bonus = 1.0 - _clip01((t - window_start) / max(1e-9, window_end - window_start))
        tightness = 1.0 - min(1.0, max(float(hit_delta), float(kick_delta), float(curve_delta)))
        score = (
            (0.40 * float(kick_peak))
            + (0.26 * float(hit_peak))
            + (0.24 * float(combined_peak))
            + (0.24 * float(post_body))
            + (0.16 * float(post_peak))
            + (0.24 * kick_advantage)
            + (0.14 * backbeat_snare)
            + (0.16 * backbeat_advantage)
            + (0.14 * body_jump)
            + (0.08 * grid_bonus)
            + (0.05 * tightness)
            + (0.05 * earliest_bonus)
            - (0.18 * early_penalty)
            - (0.14 * late_penalty)
        )
        downbeat_evidence = max(float(kick_peak), 0.78 * float(combined_peak), 0.70 * float(curve_peak))
        confidence = _clip01((score - 0.12) / 0.98)
        if downbeat_evidence < 0.100 and (post_body < 0.140 or body_jump < 0.030):
            confidence *= 0.55
            score -= 0.080
        scored.append(
            {
                "time_sec": float(t),
                "score": float(score),
                "confidence": float(confidence),
                "kick_peak": float(kick_peak),
                "hit_peak": float(hit_peak),
                "combined_peak": float(combined_peak),
                "curve_peak": float(curve_peak),
                "post_body": float(post_body),
                "post_peak": float(post_peak),
                "pre_body": float(pre_body),
                "body_jump": float(body_jump),
                "backbeat_snare": float(backbeat_snare),
                "backbeat_advantage": float(backbeat_advantage),
                "offbeat_kick": float(offbeat_kick),
                "kick_advantage": float(kick_advantage),
                "grid_distance_ms": grid_distance_ms,
                "visual_delta_sec": float(delta_from_visual),
                "visible_energy": float(visible_energy),
            }
        )

    if not scored:
        return None
    scored.sort(key=lambda row: (-float(row["score"]), float(row["time_sec"])))
    best = scored[0]
    if float(best.get("confidence", 0.0) or 0.0) < 0.260 or float(best.get("score", 0.0) or 0.0) < 0.210:
        return None
    first_credible = [
        row
        for row in scored
        if float(row.get("score", 0.0) or 0.0) >= max(0.210, float(best["score"]) - 0.075)
        and float(row.get("visible_energy", 0.0) or 0.0) >= 0.105
        and (
            float(row.get("kick_peak", 0.0) or 0.0) >= 0.100
            or float(row.get("combined_peak", 0.0) or 0.0) >= 0.145
            or float(row.get("body_jump", 0.0) or 0.0) >= 0.080
        )
    ]
    chosen = min(first_credible, key=lambda row: float(row["time_sec"])) if first_credible else best
    result = dict(chosen)
    result["bpm"] = float(bpm_value)
    result["beat_sec"] = float(beat)
    result["bar_sec"] = float(bar)
    result["visual_time_sec"] = float(visual_time)
    result["window_start_sec"] = float(window_start)
    result["window_end_sec"] = float(window_end)
    result["candidate_count"] = int(len(scored))
    result["best_score"] = float(best.get("score", 0.0) or 0.0)
    result["best_time_sec"] = float(best.get("time_sec", chosen["time_sec"]) or chosen["time_sec"])
    return result


def _source_features(features_by_role: Mapping[str, FeatureBundle]) -> tuple[str, FeatureBundle]:
    for role in ("drums", "bass", "instrumental", "vocals"):
        features = features_by_role.get(role)
        if features is not None:
            return role, features
    if not features_by_role:
        raise ValueError("No feature bundles available for beatgrid resolution")
    role = sorted(features_by_role)[0]
    return role, features_by_role[role]


def _candidate_starts(features: FeatureBundle, *, max_candidates: int = 28) -> list[float]:
    times = np.asarray(features.frame_times, dtype=np.float64)
    if times.size < 3:
        return [0.0]
    beat = float(features.beat_sec)
    search_end = min(float(features.duration_sec), max(45.0, 32.0 * 4.0 * beat))
    curve = np.asarray((0.52 * features.low_jump_curve) + (0.30 * features.combined_attack) + (0.18 * features.low_energy), dtype=np.float64)
    limit = min(int(times.size), int(curve.size))
    times = times[:limit]
    curve = curve[:limit]
    valid = np.where((times >= 0.0) & (times <= search_end))[0]
    if valid.size == 0:
        return [0.0]
    threshold = max(float(np.percentile(curve[valid], 70.0)), float(np.max(curve[valid])) * 0.24)
    distance = max(1, int(round((0.75 * beat) / max(1e-6, float(times[1] - times[0])))))
    try:
        from scipy import signal

        peaks, _props = signal.find_peaks(curve, distance=distance, height=threshold, prominence=max(0.01, float(np.std(curve[valid])) * 0.20))
    except Exception:
        peaks = np.asarray([], dtype=np.int64)
    candidates = [0.0]
    ranked = [int(idx) for idx in peaks if int(idx) in set(int(v) for v in valid)]
    ranked.sort(key=lambda idx: float(curve[idx]), reverse=True)
    for idx in ranked[: max_candidates]:
        t = float(times[idx])
        if t >= 0.0 and all(abs(t - existing) > 0.25 * beat for existing in candidates):
            candidates.append(t)
    return sorted(candidates)


def _score_bar_zero(features: FeatureBundle, bar_zero: float) -> float:
    times = np.asarray(features.frame_times, dtype=np.float64)
    if times.size < 3:
        return 0.0
    beat = float(features.beat_sec)
    bar = 4.0 * beat
    duration = float(features.duration_sec)
    if bar <= 0.0 or beat <= 0.0 or bar_zero >= min(duration, 60.0):
        return 0.0

    downbeats = np.arange(float(bar_zero), duration, bar, dtype=np.float64)
    beats = np.arange(float(bar_zero), duration, beat, dtype=np.float64)
    if downbeats.size < 4 or beats.size < 8:
        return 0.0

    low = np.asarray(features.low_energy, dtype=np.float64)
    attack = np.asarray(features.combined_attack, dtype=np.float64)
    onset = np.asarray(features.onset, dtype=np.float64)
    rms = np.asarray(features.rms, dtype=np.float64)
    flux = np.asarray(features.spectral_flux, dtype=np.float64)

    beat_strength = np.mean([_window_max(low, times, t, 0.12 * beat) for t in beats[:96]])
    downbeat_strength = np.mean([_window_max(attack, times, t, 0.16 * beat) for t in downbeats[:32]])
    onset_strength = np.mean([_window_max(onset, times, t, 0.16 * beat) for t in downbeats[:32]])

    boundary_scores: list[float] = []
    phrase_scores: list[float] = []
    for index, t in enumerate(downbeats[:48]):
        pre_rms = _window_mean(rms, times, t - 2.0 * bar, t)
        post_rms = _window_mean(rms, times, t, t + 2.0 * bar)
        pre_low = _window_mean(low, times, t - 2.0 * bar, t)
        post_low = _window_mean(low, times, t, t + 2.0 * bar)
        local_flux = _window_max(flux, times, t, 0.30 * beat)
        boundary = (0.30 * abs(post_rms - pre_rms)) + (0.30 * abs(post_low - pre_low)) + (0.40 * local_flux)
        boundary_scores.append(float(boundary))
        bars = index + 1
        if bars % 8 == 0:
            phrase_scores.append(float(boundary))

    structural = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    phrase = float(np.mean(phrase_scores)) if phrase_scores else structural
    early_penalty = min(0.20, max(0.0, float(bar_zero) / max(1.0, 16.0 * bar)) * 0.20)
    return _clip01((0.28 * beat_strength) + (0.24 * downbeat_strength) + (0.16 * onset_strength) + (0.18 * structural) + (0.14 * phrase) - early_penalty)


def resolve_beatgrid(features_by_role: Mapping[str, FeatureBundle], *, bpm: Optional[float] = None) -> BeatGrid:
    role, features = _source_features(features_by_role)
    bpm_value = float(bpm if bpm is not None and bpm > 0 else features.bpm)
    if not math.isfinite(bpm_value) or bpm_value <= 0:
        bpm_value = 128.0
    beat = 60.0 / bpm_value
    bar = 4.0 * beat

    starts = _candidate_starts(features)
    for start in _drum_pattern_candidate_starts(features, bpm=bpm_value):
        if all(abs(float(start) - float(existing)) > 0.18 * beat for existing in starts):
            starts.append(float(start))
    starts = sorted(starts)
    scored = [(float(_score_bar_zero(features, start)), float(start)) for start in starts]
    if not scored:
        scored = [(0.0, 0.0)]
    scored.sort(key=lambda row: (-row[0], row[1]))
    best_score, best_start = scored[0]
    zero_score = float(_score_bar_zero(features, 0.0))
    earlier_phase = [
        (score, start)
        for score, start in scored
        if 0.35 * beat <= float(best_start) - float(start) <= 1.25 * beat
        and float(score) >= float(best_score) - 0.006
    ]
    if earlier_phase:
        best_score, best_start = sorted(earlier_phase, key=lambda row: (row[1], -row[0]))[0]
    phase_zero = math.fmod(float(best_start), float(bar))
    if phase_zero < 0.0:
        phase_zero += float(bar)
    phase_distance_from_zero = min(float(phase_zero), abs(float(bar) - float(phase_zero)))
    if phase_distance_from_zero <= 0.12 * float(beat):
        best_score, best_start = zero_score, 0.0
    elif zero_score >= 0.30 and float(best_score) < zero_score + 0.18:
        best_score, best_start = zero_score, 0.0

    old_best_score = float(best_score)
    old_best_start = float(best_start)
    old_phase_zero = math.fmod(float(old_best_start), float(bar))
    if old_phase_zero < 0.0:
        old_phase_zero += float(bar)
    pattern_arrays = _pattern_arrays(features)
    pattern_records: list[Dict[str, Any]] = []
    phase_seen: list[float] = []
    for start in starts:
        phase = math.fmod(float(start), float(bar))
        if phase < 0.0:
            phase += float(bar)
        if any(_phase_distance(phase, seen, bar) <= 0.060 * beat for seen in phase_seen):
            continue
        phase_seen.append(float(phase))
        pattern_score = _score_drum_pattern_bar_zero(features, bpm=bpm_value, bar_zero=phase, arrays=pattern_arrays)
        if float(pattern_score.get("confidence", 0.0) or 0.0) <= 0.0:
            continue
        pattern_records.append(
            {
                "start": float(start),
                "phase": float(phase),
                **pattern_score,
            }
        )
    zero_pattern = _score_drum_pattern_bar_zero(features, bpm=bpm_value, bar_zero=0.0, arrays=pattern_arrays)
    pattern_records.append({"start": 0.0, "phase": 0.0, **zero_pattern})
    pattern_records.sort(
        key=lambda row: (
            -float(row.get("confidence", 0.0) or 0.0),
            -float(row.get("score", 0.0) or 0.0),
            float(row.get("start", 0.0) or 0.0),
        )
    )
    pattern_source = False
    if pattern_records:
        pattern_best = pattern_records[0]
        pattern_runner = next(
            (
                row
                for row in pattern_records[1:]
                if _phase_distance(float(row.get("phase", 0.0) or 0.0), float(pattern_best.get("phase", 0.0) or 0.0), bar)
                > 0.18 * beat
            ),
            None,
        )
        pattern_conf = float(pattern_best.get("confidence", 0.0) or 0.0)
        pattern_raw_score = float(pattern_best.get("score", 0.0) or 0.0)
        runner_conf = float(pattern_runner.get("confidence", 0.0) or 0.0) if pattern_runner else 0.0
        runner_raw_score = float(pattern_runner.get("score", 0.0) or 0.0) if pattern_runner else -999.0
        old_pattern = _score_drum_pattern_bar_zero(
            features,
            bpm=bpm_value,
            bar_zero=old_phase_zero,
            arrays=pattern_arrays,
        )
        old_pattern_conf = float(old_pattern.get("confidence", 0.0) or 0.0)
        phase_disagreement = _phase_distance(float(pattern_best.get("phase", 0.0) or 0.0), old_phase_zero, bar)
        strong_pattern_override = bool(
            pattern_conf >= 0.520
            and (
                pattern_conf >= runner_conf + 0.020
                or pattern_raw_score >= runner_raw_score + 0.045
            )
            and (
                pattern_conf >= old_pattern_conf + 0.040
                or (old_best_score < 0.240 and pattern_conf >= 0.580)
                or (phase_disagreement >= 0.30 * beat and pattern_conf >= 0.650)
            )
        )
        same_phase_pattern_boost = bool(
            phase_disagreement <= 0.14 * beat
            and pattern_conf >= max(0.500, old_pattern_conf - 0.010)
        )
        if strong_pattern_override or same_phase_pattern_boost:
            # A drum-pattern hit may strengthen an existing phase hypothesis,
            # but a nearby hit is not an independent reason to translate that
            # phase by a few analysis frames.  Preserve the prior phase when
            # both hypotheses already agree; otherwise every visible onset can
            # slowly become its own grid origin.
            if phase_disagreement > 0.14 * beat:
                best_start = float(pattern_best.get("start", 0.0) or 0.0)
            else:
                best_start = float(old_best_start)
            best_score = max(float(old_best_score), pattern_conf)
            pattern_source = True

    score_values = np.asarray([score for score, _start in scored], dtype=np.float64)
    median = float(np.median(score_values)) if score_values.size else 0.0
    confidence = _clip01((float(best_score) - median + 0.06) / 0.30)
    phase_zero = math.fmod(float(best_start), float(bar))
    if phase_zero < 0.0:
        phase_zero += float(bar)
    if pattern_source and pattern_records:
        confidence = max(confidence, float(pattern_records[0].get("confidence", 0.0) or 0.0))

    return BeatGrid(
        bpm=float(bpm_value),
        beat_sec=float(beat),
        bar_sec=float(bar),
        bar_zero_sec=float(max(0.0, phase_zero)),
        downbeat_confidence=float(confidence),
        first_low_downbeat_sec=float(max(0.0, best_start)),
        source_role=f"{role}:dj_drum_pattern" if pattern_source else role,
    )
