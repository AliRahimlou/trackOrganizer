from __future__ import annotations

from collections import OrderedDict
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw


CACHE_VERSION = 23
RAW_SAMPLE_LIMIT = 96_000
RAW_SAMPLES_PER_PIXEL = 16.0
MAX_TILE_WIDTH = 48_000
PEAK_FULL_READ_LIMIT = 12_000_000
PEAK_STREAM_BLOCK_FRAMES = 1_000_000
GLOBAL_RMS_PROFILE_BINS = 8192
PNG_VERSION = 14
PNG_MAX_WIDTH = 16_000
PNG_MAX_HEIGHT = 4_000
SVG_MAX_DETAIL_BINS = 48_000
RMS_MAXIMIZER_TARGET = 0.78
RMS_MAXIMIZER_CEILING = 0.985
RMS_MAXIMIZER_KNEE = 0.72
RMS_MAXIMIZER_MIN_GAIN = 0.85
RMS_MAXIMIZER_MAX_GAIN = 7.5
BOOM_BODY_SMOOTH_SECONDS = 0.180
BOOM_BODY_NOISE_FLOOR = 0.180
BOOM_BODY_FLOOR_BLEND = 0.74
BOOM_BODY_MIN_NORMALIZED = 0.380
BOOM_GUI_BODY_VISIBLE_DENSITY = 0.220
BOOM_GUI_RELEVANT_DENSITY = BOOM_GUI_BODY_VISIBLE_DENSITY
BOOM_BODY_CEILING_PERCENTILE = 96.5
BOOM_PLACE_POST_SECONDS = 0.260
BOOM_PLACE_PRE_SECONDS = 0.110
BOOM_PLACE_EDGE_GRACE_SECONDS = 0.080
BOOM_PLACE_MIN_POST_NORMALIZED = 0.340
BOOM_PLACE_MIN_PEAK_NORMALIZED = 0.430
BOOM_PLACE_MIN_CONTRAST = 0.070
BOOM_PLACE_IMPACT_LEAD_SECONDS = 0.650
BOOM_PLACE_FUTURE_BODY_SECONDS = 0.420
BOOM_PLACE_MIN_IMPACT_NORMALIZED = 0.300
BOOM_SPARSE_PULSE_COARSE_SECONDS = 0.020
GUI_STRICT_FRONT_EDGE_RELIEF_SECONDS = 0.040
GZIP_JSON_MEMORY_CACHE_MAX_ITEMS = 512
GZIP_JSON_MEMORY_CACHE_MAX_FILE_BYTES = 1_000_000
_GZIP_JSON_MEMORY_CACHE: "OrderedDict[str, tuple[int, int, Dict[str, Any]]]" = OrderedDict()


def _read_gzip_json_with_memory_cache(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = str(resolved)
    cached = _GZIP_JSON_MEMORY_CACHE.get(key)
    if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
        _GZIP_JSON_MEMORY_CACHE.move_to_end(key)
        return dict(cached[2])
    with gzip.open(resolved, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return payload
    if int(stat.st_size) <= GZIP_JSON_MEMORY_CACHE_MAX_FILE_BYTES:
        _GZIP_JSON_MEMORY_CACHE[key] = (int(stat.st_mtime_ns), int(stat.st_size), dict(payload))
        _GZIP_JSON_MEMORY_CACHE.move_to_end(key)
        while len(_GZIP_JSON_MEMORY_CACHE) > GZIP_JSON_MEMORY_CACHE_MAX_ITEMS:
            _GZIP_JSON_MEMORY_CACHE.popitem(last=False)
    return dict(payload)


def _remember_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
    except OSError:
        return
    if int(stat.st_size) > GZIP_JSON_MEMORY_CACHE_MAX_FILE_BYTES:
        return
    key = str(resolved)
    _GZIP_JSON_MEMORY_CACHE[key] = (int(stat.st_mtime_ns), int(stat.st_size), dict(payload))
    _GZIP_JSON_MEMORY_CACHE.move_to_end(key)
    while len(_GZIP_JSON_MEMORY_CACHE) > GZIP_JSON_MEMORY_CACHE_MAX_ITEMS:
        _GZIP_JSON_MEMORY_CACHE.popitem(last=False)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clip01(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _audio_key(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    raw = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _json_safe(values: np.ndarray, digits: int = 6) -> List[float]:
    if values.size == 0:
        return []
    rounded = np.round(np.asarray(values, dtype=np.float32), digits)
    return [float(v) for v in rounded]


def _json_bool(values: Sequence[bool]) -> List[bool]:
    return [bool(value) for value in values]


def _mono_preserve_peak(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    indices = np.argmax(np.abs(arr), axis=1)
    return arr[np.arange(arr.shape[0]), indices].astype(np.float32, copy=False)


def _frame_min_max(data: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.min(arr)), float(np.max(arr))


def _frame_rms(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float32), dtype=np.float64)))


def _rms_bins_from_mono(data: np.ndarray, bins: int) -> np.ndarray:
    mono = np.asarray(data, dtype=np.float32)
    safe_bins = max(1, int(bins))
    rms = np.zeros(safe_bins, dtype=np.float32)
    if mono.size == 0:
        return rms
    edges = np.floor(np.linspace(0, int(mono.size), safe_bins + 1)).astype(np.int64)
    counts = np.diff(edges)
    non_empty = counts > 0
    if not bool(np.any(non_empty)):
        return rms
    squared = np.square(mono, dtype=np.float32)
    prefix = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(squared, dtype=np.float64)])
    sums = prefix[edges[1:]] - prefix[edges[:-1]]
    rms[non_empty] = np.sqrt(sums[non_empty] / counts[non_empty]).astype(np.float32, copy=False)
    return rms


def _rms_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([max(0.0, float(v)) for v in values], dtype=np.float32)
    if arr.size == 0:
        return {
            "rms_peak": 1.0,
            "rms_percentile_10": 0.0,
            "rms_percentile_25": 0.0,
            "rms_percentile_50": 0.0,
            "rms_percentile_55": 0.0,
            "rms_percentile_70": 0.0,
            "rms_percentile_75": 0.0,
            "rms_percentile_84": 0.0,
            "rms_percentile_90": 0.0,
            "rms_percentile_94": 0.0,
            "rms_percentile_95": 1.0,
            "rms_percentile_99": 1.0,
            "rms_silence_floor": 0.0,
            "rms_boom_floor": 0.0,
            "rms_boom_ceiling": 1.0,
            "rms_visual_ceiling": 1.0,
            "visual_maximizer_target": RMS_MAXIMIZER_TARGET,
            "visual_maximizer_makeup_gain": 1.0,
            "visual_maximizer_ceiling": RMS_MAXIMIZER_CEILING,
            "visual_maximizer_knee": RMS_MAXIMIZER_KNEE,
        }
    peak = float(np.max(arr))
    p10 = float(np.percentile(arr, 10.0))
    p25 = float(np.percentile(arr, 25.0))
    p50 = float(np.percentile(arr, 50.0))
    p55 = float(np.percentile(arr, 55.0))
    p70 = float(np.percentile(arr, 70.0))
    p75 = float(np.percentile(arr, 75.0))
    p84 = float(np.percentile(arr, 84.0))
    p90 = float(np.percentile(arr, 90.0))
    p94 = float(np.percentile(arr, 94.0))
    p95 = float(np.percentile(arr, 95.0))
    p99 = float(np.percentile(arr, 99.0))
    ceiling = max(1e-5, min(max(p95 * 1.35, p99 * 0.82), max(peak, 1e-5)))
    reference = max(p95, p99 * 0.74, peak * 0.58, 1e-5)
    makeup = float(np.clip(RMS_MAXIMIZER_TARGET / reference, RMS_MAXIMIZER_MIN_GAIN, RMS_MAXIMIZER_MAX_GAIN))
    silence_floor = max(0.0, min(max(p10 * 1.65, p25 * 0.72, peak * 0.006), max(peak, 1e-9)))
    boom_floor = max(
        p55 + max(0.0, p90 - p55) * 0.74,
        p70 * 0.95,
        p84 * 0.78,
        p95 * 0.58,
        p99 * 0.36,
        peak * 0.10,
        silence_floor * 2.2,
    )
    boom_ceiling = max(boom_floor + 1e-6, p94, p95 * 0.92, p99 * 0.70, peak * 0.44)
    return {
        "rms_peak": float(peak if peak > 1e-9 else 1.0),
        "rms_percentile_10": float(p10),
        "rms_percentile_25": float(p25),
        "rms_percentile_50": float(p50),
        "rms_percentile_55": float(p55),
        "rms_percentile_70": float(p70),
        "rms_percentile_75": float(p75),
        "rms_percentile_84": float(p84),
        "rms_percentile_90": float(p90),
        "rms_percentile_94": float(p94),
        "rms_percentile_95": float(p95 if p95 > 1e-9 else peak if peak > 1e-9 else 1.0),
        "rms_percentile_99": float(p99 if p99 > 1e-9 else peak if peak > 1e-9 else 1.0),
        "rms_silence_floor": float(silence_floor),
        "rms_boom_floor": float(boom_floor if boom_floor > 1e-9 else 0.0),
        "rms_boom_ceiling": float(boom_ceiling if boom_ceiling > 1e-9 else 1.0),
        "rms_visual_ceiling": float(ceiling),
        "visual_maximizer_target": RMS_MAXIMIZER_TARGET,
        "visual_maximizer_makeup_gain": makeup,
        "visual_maximizer_ceiling": RMS_MAXIMIZER_CEILING,
        "visual_maximizer_knee": RMS_MAXIMIZER_KNEE,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, float(percentile)))


def _moving_average(values: Sequence[float], radius: int) -> List[float]:
    safe_radius = max(0, int(round(radius)))
    source = np.asarray([float(v) for v in values], dtype=np.float64)
    if safe_radius <= 0 or source.size < 3:
        return [float(v) for v in source]
    prefix = np.concatenate([np.asarray([0.0]), np.cumsum(source)])
    indices = np.arange(int(source.size), dtype=np.int64)
    starts = np.maximum(0, indices - safe_radius)
    ends = np.minimum(int(source.size), indices + safe_radius + 1)
    counts = np.maximum(1, ends - starts)
    return ((prefix[ends] - prefix[starts]) / counts).astype(np.float64, copy=False).tolist()


def _bool_runs(values: Sequence[bool]) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    active_start: Optional[int] = None
    for index, active in enumerate(values):
        if bool(active) and active_start is None:
            active_start = index
        elif not bool(active):
            if active_start is not None:
                runs.append((int(active_start), int(index)))
            active_start = None
    if active_start is not None:
        runs.append((int(active_start), len(values)))
    return runs


def _metric_value(local_stats: Mapping[str, Any], global_profile: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = _float_or_none(global_profile.get(f"global_{key}"))
    if value is None:
        value = _float_or_none(local_stats.get(key))
    return float(value) if value is not None else float(default)


def _metric_has_global(global_profile: Mapping[str, Any], key: str) -> bool:
    return _float_or_none(global_profile.get(f"global_{key}")) is not None


def _boom_rms_amplitude(value: float, local_stats: Mapping[str, Any], global_profile: Mapping[str, Any]) -> float:
    makeup = max(
        RMS_MAXIMIZER_MIN_GAIN,
        min(RMS_MAXIMIZER_MAX_GAIN, _metric_value(local_stats, global_profile, "visual_maximizer_makeup_gain", 1.0)),
    )
    ceiling = max(
        0.5,
        min(1.0, _metric_value(local_stats, global_profile, "visual_maximizer_ceiling", RMS_MAXIMIZER_CEILING)),
    )
    knee = max(
        0.2,
        min(ceiling - 0.02, _metric_value(local_stats, global_profile, "visual_maximizer_knee", RMS_MAXIMIZER_KNEE)),
    )
    lifted = max(0.0, float(value)) * makeup
    if lifted <= knee:
        limited = lifted
    else:
        limiter_range = max(0.001, ceiling - knee)
        limited = knee + limiter_range * math.tanh((lifted - knee) / limiter_range)
    return float(np.clip((limited / ceiling) ** 0.82, 0.0, 1.0))


def _boom_masks(
    rms_values: Sequence[float],
    *,
    start_sec: float,
    end_sec: float,
    local_stats: Mapping[str, Any],
    global_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    raw = [max(0.0, float(value)) for value in rms_values]
    if len(raw) < 2:
        return {
            "boom_bin_span_sec": 0.0,
            "boom_floor": BOOM_BODY_NOISE_FLOOR,
            "boom_ceiling": 1.0,
            "boom_body_density": [],
            "boom_body_mask": [],
            "boom_relevant_mask": [],
            "boom_placeable_mask": [],
            "boom_placeable_count": 0,
        }
    span = max(1e-9, float(end_sec) - float(start_sec))
    bin_span = span / max(1, len(raw))
    energy = [_boom_rms_amplitude(value, local_stats, global_profile) for value in raw]
    radius = max(1, int(round((BOOM_BODY_SMOOTH_SECONDS / max(bin_span, 1e-9)) / 2.0)))
    body = _moving_average(energy, radius)

    local_q55 = _percentile(body, 55.0)
    local_q70 = _percentile(body, 70.0)
    local_q84 = _percentile(body, 84.0)
    local_q90 = _percentile(body, 90.0)
    local_q94 = _percentile(body, 94.0)
    local_q96 = _percentile(body, BOOM_BODY_CEILING_PERCENTILE)
    use_track_thresholds = bool(_metric_has_global(global_profile, "rms_percentile_90"))

    def profile_q(percentile_key: str, fallback: float) -> float:
        value = _float_or_none(global_profile.get(f"global_boom_body_percentile_{percentile_key}"))
        if value is not None:
            return float(value)
        raw_value = _metric_value(local_stats, global_profile, f"rms_percentile_{percentile_key}", 0.0)
        if raw_value <= 0.0:
            return float(fallback)
        return _boom_rms_amplitude(raw_value, local_stats, global_profile)

    if use_track_thresholds:
        q55 = profile_q("55", local_q55)
        q70 = profile_q("70", local_q70)
        q84 = profile_q("84", local_q84)
        q90 = profile_q("90", local_q90)
        q94 = profile_q("94", local_q94)
        q96 = max(profile_q("96_5", local_q96), q94, q90)
    else:
        q55 = local_q55
        q70 = local_q70
        q84 = local_q84
        q90 = local_q90
        q94 = local_q94
        q96 = local_q96
    body_peak = max(body) if body else 0.0
    raw_peak = max(raw) if raw else 0.0
    raw_p95 = _percentile(raw, 95.0)
    raw_p99 = _percentile(raw, 99.0)
    global_floor = _boom_rms_amplitude(_metric_value(local_stats, global_profile, "rms_boom_floor", 0.0), local_stats, global_profile)
    global_ceiling = _boom_rms_amplitude(_metric_value(local_stats, global_profile, "rms_boom_ceiling", 0.0), local_stats, global_profile)
    floor = max(
        BOOM_BODY_NOISE_FLOOR,
        global_floor,
        q55 + max(0.0, q90 - q55) * BOOM_BODY_FLOOR_BLEND,
        q70 * 0.98,
        q84 * 0.82,
        q94 * 0.52,
    )
    ceiling = max(floor + 0.045, global_ceiling, q96, q94, q90)
    global_raw_peak = _metric_value(local_stats, global_profile, "rms_peak", raw_peak)
    global_silence_floor = _metric_value(local_stats, global_profile, "rms_silence_floor", 0.0)
    audible_dense_body = bool(raw_peak >= 0.160 or raw_p95 >= 0.095 or raw_p99 >= 0.135)
    relative_dense_body = bool(
        raw_peak >= max(0.012, global_silence_floor * 1.55)
        and raw_p95 >= max(0.006, global_silence_floor * 1.25, raw_peak * 0.18)
        and raw_peak >= max(0.012, float(global_raw_peak) * 0.34)
    )
    dense_body_has_signal = bool(audible_dense_body or relative_dense_body)
    if dense_body_has_signal and body_peak > BOOM_BODY_NOISE_FLOOR and floor >= body_peak * 0.985:
        adaptive_cap = max(
            BOOM_BODY_NOISE_FLOOR,
            min(q70 * 1.04, body_peak * 0.84),
            min(q84 * 0.96, body_peak * 0.88),
            min(q90 * 0.91, body_peak * 0.90),
            body_peak * 0.78,
        )
        floor = min(floor, adaptive_cap)
        if relative_dense_body and not audible_dense_body:
            ceiling = max(floor + 0.045, q96, q94, q90, body_peak)
        else:
            ceiling = max(floor + 0.045, global_ceiling, q96, q94, q90, body_peak)
    track_body_range = max(0.0, float(q90) - float(q55), float(q94) - float(q70))
    flat_texture_veto = bool(
        not audible_dense_body
        and raw_peak < 0.120
        and raw_p99 < 0.100
        and body_peak < 0.400
        and track_body_range < 0.025
    )

    def normalized(value: float) -> float:
        return float(np.clip((float(value) - floor) / max(ceiling - floor, 0.001), 0.0, 1.0))

    density = [normalized(value) for value in body]
    raw_density = [normalized(value) for value in energy]
    body_mask = [
        value >= BOOM_BODY_MIN_NORMALIZED
        or (body[index] >= q90 and body[index] >= floor * 0.96)
        or (body[index] >= q84 and body[index] >= floor)
        for index, value in enumerate(density)
    ]
    hard_energy_mask = [
        energy[index] >= floor
        and (
            raw_density[index] >= min(0.280, BOOM_BODY_MIN_NORMALIZED * 0.78)
            or energy[index] >= q90
        )
        for index in range(len(energy))
    ]
    bridge_gap_bins = max(1, int(round(0.035 / max(bin_span, 1e-9))))
    index = 0
    while index < len(hard_energy_mask):
        if hard_energy_mask[index]:
            index += 1
            continue
        gap_start = index
        while index < len(hard_energy_mask) and not hard_energy_mask[index]:
            index += 1
        gap_end = index
        left_active = gap_start > 0 and hard_energy_mask[gap_start - 1]
        right_active = gap_end < len(hard_energy_mask) and hard_energy_mask[gap_end]
        if left_active and right_active and (gap_end - gap_start) <= bridge_gap_bins:
            for fill_index in range(gap_start, gap_end):
                hard_energy_mask[fill_index] = True
    hard_energy_run_start: List[Optional[int]] = [None] * len(hard_energy_mask)
    active_start: Optional[int] = None
    hard_energy_runs: List[tuple[int, int]] = []
    for index, active in enumerate(hard_energy_mask):
        if active and active_start is None:
            active_start = index
        elif not active:
            if active_start is not None:
                hard_energy_runs.append((int(active_start), int(index)))
            active_start = None
        if active_start is not None:
            hard_energy_run_start[index] = active_start
    if active_start is not None:
        hard_energy_runs.append((int(active_start), len(hard_energy_mask)))

    sparse_raw_floor = max(
        0.012,
        float(global_silence_floor) * 1.75,
        raw_p95 * 0.72,
        raw_peak * 0.30,
        float(global_raw_peak) * 0.16,
    )
    if raw_peak > 0.0:
        sparse_raw_floor = min(max(sparse_raw_floor, raw_peak * 0.24), raw_peak * 0.86)
    raw_sparse_mask = [
        value >= sparse_raw_floor
        and value >= max(0.010, float(global_silence_floor) * 1.45)
        for value in raw
    ]
    raw_sparse_runs = _bool_runs(raw_sparse_mask)

    post_bins = max(1, int(round(BOOM_PLACE_POST_SECONDS / max(bin_span, 1e-9))))
    future_body_bins = max(post_bins, int(round(BOOM_PLACE_FUTURE_BODY_SECONDS / max(bin_span, 1e-9))))
    pre_bins = max(1, int(round(BOOM_PLACE_PRE_SECONDS / max(bin_span, 1e-9))))
    edge_grace_bins = max(1, int(round(BOOM_PLACE_EDGE_GRACE_SECONDS / max(bin_span, 1e-9))))
    impact_lead_bins = max(1, int(round(BOOM_PLACE_IMPACT_LEAD_SECONDS / max(bin_span, 1e-9))))
    sustained_run_bins = max(2, int(round(0.050 / max(bin_span, 1e-9))))
    sparse_train_bins = max(1, int(round(1.250 / max(bin_span, 1e-9))))
    sparse_pre_bins = max(pre_bins, int(round(0.450 / max(bin_span, 1e-9))))
    sparse_min_run_bins = max(1, int(round(0.010 / max(bin_span, 1e-9))))
    raw_sparse_min_run_bins = max(1, int(round(0.004 / max(bin_span, 1e-9))))
    coarse_factor = max(1, int(round(BOOM_SPARSE_PULSE_COARSE_SECONDS / max(bin_span, 1e-9))))
    coarse_raw: List[float] = []
    if coarse_factor > 1:
        for coarse_index in range(0, len(raw), coarse_factor):
            coarse_raw.append(float(max(raw[coarse_index : coarse_index + coarse_factor], default=0.0)))
    else:
        coarse_raw = list(raw)
    coarse_sparse_mask = [
        value >= sparse_raw_floor
        and value >= max(0.010, float(global_silence_floor) * 1.45)
        for value in coarse_raw
    ]
    coarse_sparse_runs = _bool_runs(coarse_sparse_mask)
    coarse_span = bin_span * max(1, coarse_factor)
    next_sustained_run_start: List[Optional[int]] = [None] * len(hard_energy_mask)
    next_start: Optional[int] = None
    run_index = len(hard_energy_runs) - 1
    for index in range(len(hard_energy_mask) - 1, -1, -1):
        while run_index >= 0 and hard_energy_runs[run_index][0] > index:
            run_start, run_end = hard_energy_runs[run_index]
            if (int(run_end) - int(run_start)) >= sustained_run_bins:
                next_start = int(run_start)
            run_index -= 1
        next_sustained_run_start[index] = next_start
    body_arr = np.asarray(body, dtype=np.float64)

    def forward_percentile(window_bins: int, percentile: float) -> np.ndarray:
        safe_window = max(1, int(window_bins))
        if body_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        if safe_window == 1:
            return body_arr.copy()
        padded = np.pad(body_arr, (0, safe_window - 1), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, safe_window)[: body_arr.size]
        return np.percentile(windows, float(percentile), axis=1)

    def forward_max(window_bins: int) -> np.ndarray:
        safe_window = max(1, int(window_bins))
        if body_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        if safe_window == 1:
            return body_arr.copy()
        padded = np.pad(body_arr, (0, safe_window - 1), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, safe_window)[: body_arr.size]
        return np.max(windows, axis=1)

    def backward_percentile(window_bins: int, percentile: float, pad_value: float) -> np.ndarray:
        safe_window = max(1, int(window_bins))
        if body_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        padded = np.concatenate([np.full(safe_window, float(pad_value), dtype=np.float64), body_arr])
        windows = np.lib.stride_tricks.sliding_window_view(padded, safe_window)[: body_arr.size]
        return np.percentile(windows, float(percentile), axis=1)

    body_p34 = _percentile(body, 34.0)
    post_body_values = forward_percentile(post_bins, 72.0)
    post_peak_values = forward_max(post_bins)
    pre_floor_values = backward_percentile(pre_bins, 58.0, body_p34)
    future_post_values = forward_percentile(future_body_bins, 72.0)
    future_peak_values = forward_max(future_body_bins)
    placeable: List[bool] = []
    for index in range(len(body)):
        if len(body) <= 0:
            placeable.append(False)
            continue
        pre_floor = float(pre_floor_values[index]) if index < len(pre_floor_values) else body_p34
        post_body = float(post_body_values[index]) if index < len(post_body_values) else 0.0
        post_peak = float(post_peak_values[index]) if index < len(post_peak_values) else 0.0
        normalized_post = normalized(post_body)
        normalized_peak = normalized(post_peak)
        contrast = max(0.0, post_body - pre_floor)
        hard_body_floor = post_peak >= floor and (post_body >= floor * 0.92 or post_peak >= q90)
        enough_body = normalized_post >= BOOM_PLACE_MIN_POST_NORMALIZED or post_body >= q84
        enough_peak = normalized_peak >= BOOM_PLACE_MIN_PEAK_NORMALIZED or post_peak >= q90
        enough_contrast = contrast >= BOOM_PLACE_MIN_CONTRAST or (
            post_body >= q90 and normalized_post >= BOOM_PLACE_MIN_POST_NORMALIZED + 0.08
        )
        run_start = hard_energy_run_start[index]
        in_front_edge_window = bool(run_start is not None and index <= int(run_start) + edge_grace_bins)
        future_run_start = next_sustained_run_start[index]
        future_post = (
            float(future_post_values[int(future_run_start)])
            if future_run_start is not None and int(future_run_start) < len(future_post_values)
            else 0.0
        )
        future_peak = (
            float(future_peak_values[int(future_run_start)])
            if future_run_start is not None and int(future_run_start) < len(future_peak_values)
            else 0.0
        )
        future_contrast = max(0.0, future_post - pre_floor)
        raw_impact = energy[index]
        raw_impact_normalized = raw_density[index]
        future_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in hard_energy_runs
            if int(run_start_i) >= index
            and int(run_start_i) <= index + sparse_train_bins
            and (int(run_end_i) - int(run_start_i)) >= sparse_min_run_bins
        ]
        pre_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in hard_energy_runs
            if int(run_end_i) > index - sparse_pre_bins and int(run_start_i) < index
        ]
        future_sparse_active = sum(int(run_end_i) - int(run_start_i) for run_start_i, run_end_i in future_sparse_runs)
        pre_sparse_active = sum(
            max(0, min(index, int(run_end_i)) - max(0, int(run_start_i), index - sparse_pre_bins))
            for run_start_i, run_end_i in pre_sparse_runs
        )
        first_sparse_start = int(future_sparse_runs[0][0]) if future_sparse_runs else None
        future_sparse_slice = energy[index : min(len(energy), index + sparse_train_bins)]
        pre_sparse_slice = energy[max(0, index - sparse_pre_bins) : index]
        future_sparse_peak = max(future_sparse_slice) if future_sparse_slice else 0.0
        pre_sparse_peak = max(pre_sparse_slice) if pre_sparse_slice else 0.0
        sparse_pulse_train_front = bool(
            audible_dense_body
            and first_sparse_start is not None
            and abs(int(first_sparse_start) - index) <= edge_grace_bins
            and len(future_sparse_runs) >= 3
            and future_sparse_active >= max(3, int(round(0.040 / max(bin_span, 1e-9))))
            and future_sparse_peak >= floor
            and raw_impact >= max(floor * 0.78, q84 * 0.84)
            and raw_impact_normalized >= min(0.260, BOOM_PLACE_MIN_IMPACT_NORMALIZED)
            and pre_sparse_active <= max(1, int(round(future_sparse_active * 0.35)))
            and pre_sparse_peak <= max(floor * 0.92, future_sparse_peak * 0.72)
        )
        future_raw_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in raw_sparse_runs
            if int(run_start_i) >= index
            and int(run_start_i) <= index + sparse_train_bins
            and (int(run_end_i) - int(run_start_i)) >= raw_sparse_min_run_bins
        ]
        pre_raw_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in raw_sparse_runs
            if int(run_end_i) > index - sparse_pre_bins and int(run_start_i) < index
        ]
        first_raw_sparse_start = int(future_raw_sparse_runs[0][0]) if future_raw_sparse_runs else None
        future_raw_sparse_active = sum(
            int(run_end_i) - int(run_start_i) for run_start_i, run_end_i in future_raw_sparse_runs
        )
        pre_raw_sparse_active = sum(
            max(0, min(index, int(run_end_i)) - max(0, int(run_start_i), index - sparse_pre_bins))
            for run_start_i, run_end_i in pre_raw_sparse_runs
        )
        future_raw_sparse_slice = raw[index : min(len(raw), index + sparse_train_bins)]
        pre_raw_sparse_slice = raw[max(0, index - sparse_pre_bins) : index]
        future_raw_sparse_peak = max(future_raw_sparse_slice) if future_raw_sparse_slice else 0.0
        pre_raw_sparse_peak = max(pre_raw_sparse_slice) if pre_raw_sparse_slice else 0.0
        sparse_train_span_bins = (
            int(future_raw_sparse_runs[-1][0]) - int(first_raw_sparse_start)
            if future_raw_sparse_runs and first_raw_sparse_start is not None
            else 0
        )
        raw_sparse_pulse_train_front = bool(
            dense_body_has_signal
            and not flat_texture_veto
            and first_raw_sparse_start is not None
            and abs(int(first_raw_sparse_start) - index) <= edge_grace_bins
            and len(future_raw_sparse_runs) >= 3
            and sparse_train_span_bins >= max(2, int(round(0.240 / max(bin_span, 1e-9))))
            and future_raw_sparse_active >= max(3, int(round(0.020 / max(bin_span, 1e-9))))
            and future_raw_sparse_peak >= sparse_raw_floor
            and raw_impact >= max(sparse_raw_floor * 0.82, future_raw_sparse_peak * 0.52)
            and pre_raw_sparse_active <= max(1, int(round(future_raw_sparse_active * 0.40)))
            and pre_raw_sparse_peak <= max(sparse_raw_floor * 1.06, future_raw_sparse_peak * 0.72)
        )
        coarse_index = min(len(coarse_raw) - 1, max(0, index // max(1, coarse_factor))) if coarse_raw else 0
        coarse_train_bins = max(1, int(math.ceil(sparse_train_bins / max(1, coarse_factor))))
        coarse_pre_bins = max(1, int(math.ceil(sparse_pre_bins / max(1, coarse_factor))))
        future_coarse_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in coarse_sparse_runs
            if int(run_start_i) >= coarse_index
            and int(run_start_i) <= coarse_index + coarse_train_bins
        ]
        pre_coarse_sparse_runs = [
            (run_start_i, run_end_i)
            for run_start_i, run_end_i in coarse_sparse_runs
            if int(run_end_i) > coarse_index - coarse_pre_bins and int(run_start_i) < coarse_index
        ]
        first_coarse_sparse_start = (
            int(future_coarse_sparse_runs[0][0]) * max(1, coarse_factor)
            if future_coarse_sparse_runs
            else None
        )
        future_coarse_active_sec = sum(
            (int(run_end_i) - int(run_start_i)) * coarse_span
            for run_start_i, run_end_i in future_coarse_sparse_runs
        )
        pre_coarse_active_sec = sum(
            max(0, min(coarse_index, int(run_end_i)) - max(0, int(run_start_i), coarse_index - coarse_pre_bins))
            * coarse_span
            for run_start_i, run_end_i in pre_coarse_sparse_runs
        )
        coarse_future_end = min(len(coarse_raw), coarse_index + coarse_train_bins)
        coarse_pre_start = max(0, coarse_index - coarse_pre_bins)
        future_coarse_peak = max(coarse_raw[coarse_index:coarse_future_end], default=0.0)
        pre_coarse_peak = max(coarse_raw[coarse_pre_start:coarse_index], default=0.0)
        coarse_train_span_sec = (
            (int(future_coarse_sparse_runs[-1][0]) - int(future_coarse_sparse_runs[0][0])) * coarse_span
            if len(future_coarse_sparse_runs) >= 2
            else 0.0
        )
        raw_edge_peak = max(raw[index : min(len(raw), index + edge_grace_bins + 1)], default=0.0)
        coarse_sparse_pulse_train_front = bool(
            dense_body_has_signal
            and not flat_texture_veto
            and first_coarse_sparse_start is not None
            and abs(int(first_coarse_sparse_start) - index) <= edge_grace_bins
            and len(future_coarse_sparse_runs) >= 3
            and coarse_train_span_sec >= 0.240
            and future_coarse_active_sec >= 0.020
            and future_coarse_peak >= sparse_raw_floor
            and raw_edge_peak >= max(sparse_raw_floor * 0.82, future_coarse_peak * 0.48)
            and pre_coarse_active_sec <= max(coarse_span, future_coarse_active_sec * 0.42)
            and pre_coarse_peak <= max(sparse_raw_floor * 1.08, future_coarse_peak * 0.74)
        )
        impact_leads_to_body = bool(
            future_run_start is not None
            and int(future_run_start) >= index
            and int(future_run_start) - index <= impact_lead_bins
            and raw_impact >= max(floor * 0.72, q84 * 0.82)
            and raw_impact_normalized >= min(BOOM_PLACE_MIN_IMPACT_NORMALIZED, BOOM_BODY_MIN_NORMALIZED * 0.86)
            and future_peak >= floor
            and (
                normalized(future_post) >= BOOM_PLACE_MIN_POST_NORMALIZED
                or future_post >= q84
                or future_peak >= q90
            )
            and (
                future_contrast >= BOOM_PLACE_MIN_CONTRAST * 0.70
                or future_peak >= q90
            )
        )
        placeable.append(
            bool(
                not flat_texture_veto
                and (
                    (in_front_edge_window and hard_body_floor and enough_body and enough_peak and enough_contrast)
                    or impact_leads_to_body
                    or sparse_pulse_train_front
                    or raw_sparse_pulse_train_front
                    or coarse_sparse_pulse_train_front
                )
            )
        )

    has_placeable_front_edge = any(bool(value) for value in placeable)
    sparse_train_relevant = [False] * len(body_mask)
    if has_placeable_front_edge and not flat_texture_veto:
        for place_index, is_placeable in enumerate(placeable):
            if not is_placeable:
                continue
            end_index = min(len(sparse_train_relevant), place_index + sparse_train_bins)
            for relevant_index in range(place_index, end_index):
                coarse_index = (
                    min(len(coarse_sparse_mask) - 1, max(0, relevant_index // max(1, coarse_factor)))
                    if coarse_sparse_mask
                    else 0
                )
                if (
                    raw_sparse_mask[relevant_index]
                    or hard_energy_mask[relevant_index]
                    or (coarse_sparse_mask[coarse_index] if coarse_sparse_mask else False)
                ):
                    sparse_train_relevant[relevant_index] = True
    relevant_mask = [
        bool(
            placeable[index]
            or sparse_train_relevant[index]
            or (
                has_placeable_front_edge
                and body_mask[index]
                and density[index] >= BOOM_GUI_RELEVANT_DENSITY
            )
        )
        for index in range(len(body_mask))
    ]
    return {
        "boom_bin_span_sec": float(bin_span),
        "boom_mask_profile_scope": "track" if use_track_thresholds else "tile",
        "boom_floor": float(floor),
        "boom_ceiling": float(ceiling),
        "boom_body_density": _json_safe(np.asarray(density, dtype=np.float32), digits=5),
        "boom_body_mask": _json_bool(body_mask),
        "boom_relevant_mask": _json_bool(relevant_mask),
        "boom_placeable_mask": _json_bool(placeable),
        "boom_placeable_count": int(sum(1 for value in placeable if value)),
    }


def _resolution_level(samples_per_pixel: float) -> int:
    spp = max(1.0, float(samples_per_pixel))
    return int(2 ** round(math.log2(spp))) if spp > 1.0 else 1


class WaveformCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def info(self, audio_path: str) -> Dict[str, Any]:
        path = Path(audio_path).expanduser()
        with sf.SoundFile(str(path)) as fh:
            sample_rate = int(fh.samplerate)
            frames = int(fh.frames)
            channels = int(fh.channels)
        return {
            "sample_rate": sample_rate,
            "total_samples": frames,
            "duration": float(frames / sample_rate) if sample_rate else 0.0,
            "channels": channels,
            "audio_key": _audio_key(path),
        }

    def _cache_path(
        self,
        path: Path,
        *,
        start_sample: int,
        end_sample: int,
        width: int,
        mode: str,
        level: int,
    ) -> Path:
        key = _audio_key(path)
        shard = self.cache_dir / key[:2] / key[2:4]
        shard.mkdir(parents=True, exist_ok=True)
        raw = f"v{CACHE_VERSION}|{key}|{start_sample}|{end_sample}|{width}|{mode}|{level}"
        name = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return shard / f"{name}.json.gz"

    def _profile_cache_path(self, path: Path) -> Path:
        key = _audio_key(path)
        shard = self.cache_dir / key[:2] / key[2:4]
        shard.mkdir(parents=True, exist_ok=True)
        raw = f"v{CACHE_VERSION}|{key}|global_rms_profile|{GLOBAL_RMS_PROFILE_BINS}"
        name = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return shard / f"{name}.json.gz"

    def _read_frames(self, path: Path, start_sample: int, end_sample: int) -> np.ndarray:
        frames = max(0, int(end_sample) - int(start_sample))
        if frames <= 0:
            return np.zeros(0, dtype=np.float32)
        with sf.SoundFile(str(path)) as fh:
            safe_start = max(0, min(int(start_sample), int(fh.frames)))
            fh.seek(safe_start)
            data = fh.read(max(0, min(frames, int(fh.frames) - safe_start)), dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32)

    def _raw_tile(self, path: Path, start_sample: int, end_sample: int, width: int) -> Dict[str, Any]:
        data = self._read_frames(path, start_sample, end_sample)
        mono = _mono_preserve_peak(data)
        rms_width = max(1, min(int(width), int(mono.size) if mono.size else 1))
        rms = _rms_bins_from_mono(mono, rms_width)
        return {
            "mode": "samples",
            "sample_start": int(start_sample),
            "samples": _json_safe(np.clip(mono, -1.0, 1.0), digits=7),
            "rms": _json_safe(np.clip(rms, 0.0, 1.0), digits=7),
        }

    def _peak_tile(self, path: Path, start_sample: int, end_sample: int, width: int) -> Dict[str, Any]:
        total = max(0, int(end_sample) - int(start_sample))
        bins = max(1, int(width))
        mins = np.zeros(bins, dtype=np.float32)
        maxs = np.zeros(bins, dtype=np.float32)
        rms = np.zeros(bins, dtype=np.float32)
        if total <= 0:
            return {"mode": "peaks", "mins": _json_safe(mins), "maxs": _json_safe(maxs), "rms": _json_safe(rms)}

        edges = np.floor(np.linspace(0, total, bins + 1)).astype(np.int64)
        if total <= PEAK_FULL_READ_LIMIT:
            data = self._read_frames(path, start_sample, end_sample)
            data = _mono_preserve_peak(data)
            counts = np.diff(edges)
            if data.size and bool(np.all(counts > 0)):
                starts = edges[:-1]
                mins = np.minimum.reduceat(data, starts).astype(np.float32, copy=False)
                maxs = np.maximum.reduceat(data, starts).astype(np.float32, copy=False)
                sums = np.add.reduceat(np.square(data, dtype=np.float32), starts).astype(np.float64, copy=False)
                rms = np.sqrt(sums / counts).astype(np.float32, copy=False)
            else:
                for idx in range(bins):
                    seg = data[int(edges[idx]) : int(edges[idx + 1])]
                    mins[idx], maxs[idx] = _frame_min_max(seg)
                    rms[idx] = _frame_rms(seg)
        else:
            mins.fill(np.inf)
            maxs.fill(-np.inf)
            sums = np.zeros(bins, dtype=np.float64)
            counts = np.zeros(bins, dtype=np.int64)
            absolute_edges = int(start_sample) + edges
            with sf.SoundFile(str(path)) as fh:
                frame_count = int(fh.frames)
                cursor = max(0, min(int(start_sample), frame_count))
                final = max(cursor, min(int(end_sample), frame_count))
                fh.seek(cursor)
                while cursor < final:
                    count = min(PEAK_STREAM_BLOCK_FRAMES, final - cursor)
                    data = fh.read(count, dtype="float32", always_2d=True)
                    mono = _mono_preserve_peak(data)
                    if mono.size == 0:
                        break
                    block_start = cursor
                    block_end = cursor + int(mono.size)
                    first_bin = max(0, min(bins - 1, int(np.searchsorted(absolute_edges, block_start, side="right") - 1)))
                    last_bin = max(0, min(bins - 1, int(np.searchsorted(absolute_edges, block_end - 1, side="right") - 1)))
                    for idx in range(first_bin, last_bin + 1):
                        local_start = max(block_start, int(absolute_edges[idx])) - block_start
                        local_end = min(block_end, int(absolute_edges[idx + 1])) - block_start
                        if local_end <= local_start:
                            continue
                        seg = mono[int(local_start) : int(local_end)]
                        lo, hi = _frame_min_max(seg)
                        mins[idx] = min(float(mins[idx]), lo)
                        maxs[idx] = max(float(maxs[idx]), hi)
                        sums[idx] += float(np.sum(np.square(seg, dtype=np.float32), dtype=np.float64))
                        counts[idx] += int(seg.size)
                    cursor = block_end
            mins = np.where(np.isfinite(mins), mins, 0.0).astype(np.float32, copy=False)
            maxs = np.where(np.isfinite(maxs), maxs, 0.0).astype(np.float32, copy=False)
            rms = np.sqrt(np.divide(sums, np.maximum(counts, 1), dtype=np.float64)).astype(np.float32, copy=False)
        return {
            "mode": "peaks",
            "mins": _json_safe(np.clip(mins, -1.0, 1.0)),
            "maxs": _json_safe(np.clip(maxs, -1.0, 1.0)),
            "rms": _json_safe(np.clip(rms, 0.0, 1.0), digits=7),
        }

    def _global_rms_profile(self, path: Path, total_samples: int, sample_rate: int) -> Dict[str, Any]:
        cache_path = self._profile_cache_path(path)
        if cache_path.exists():
            try:
                cached = _read_gzip_json_with_memory_cache(cache_path)
                if isinstance(cached, dict):
                    return cached
            except Exception:
                cache_path.unlink(missing_ok=True)

        bins = max(1, min(GLOBAL_RMS_PROFILE_BINS, int(total_samples) if total_samples > 0 else 1))
        rms = np.zeros(bins, dtype=np.float32)
        if total_samples > 0:
            if total_samples <= PEAK_FULL_READ_LIMIT:
                data = self._read_frames(path, 0, total_samples)
                mono = _mono_preserve_peak(data)
                rms = _rms_bins_from_mono(mono, bins)
            else:
                sums = np.zeros(bins, dtype=np.float64)
                counts = np.zeros(bins, dtype=np.int64)
                edges = np.floor(np.linspace(0, int(total_samples), bins + 1)).astype(np.int64)
                with sf.SoundFile(str(path)) as fh:
                    cursor = 0
                    final = int(min(total_samples, int(fh.frames)))
                    fh.seek(0)
                    while cursor < final:
                        count = min(PEAK_STREAM_BLOCK_FRAMES, final - cursor)
                        data = fh.read(count, dtype="float32", always_2d=True)
                        mono = _mono_preserve_peak(data)
                        if mono.size == 0:
                            break
                        block_start = cursor
                        block_end = cursor + int(mono.size)
                        first_bin = max(0, min(bins - 1, int(np.searchsorted(edges, block_start, side="right") - 1)))
                        last_bin = max(0, min(bins - 1, int(np.searchsorted(edges, block_end - 1, side="right") - 1)))
                        for idx in range(first_bin, last_bin + 1):
                            local_start = max(block_start, int(edges[idx])) - block_start
                            local_end = min(block_end, int(edges[idx + 1])) - block_start
                            if local_end <= local_start:
                                continue
                            seg = mono[int(local_start) : int(local_end)]
                            sums[idx] += float(np.sum(np.square(seg, dtype=np.float32), dtype=np.float64))
                            counts[idx] += int(seg.size)
                        cursor = block_end
                rms = np.sqrt(np.divide(sums, np.maximum(counts, 1), dtype=np.float64)).astype(np.float32, copy=False)

        stats = _rms_stats(rms)
        profile: Dict[str, Any] = {
            "global_rms_profile_bins": int(bins),
            "global_rms_profile_samples_per_bin": float(total_samples / max(1, bins)),
        }
        profile.update({f"global_{key}": value for key, value in stats.items()})
        profile_sample_rate = max(1, int(sample_rate))
        body_bin_span = float((total_samples / max(1, bins)) / profile_sample_rate) if total_samples > 0 else 0.0
        body_radius = max(1, int(round((BOOM_BODY_SMOOTH_SECONDS / max(body_bin_span, 1e-9)) / 2.0)))
        energy = [_boom_rms_amplitude(float(value), stats, {}) for value in rms]
        body = _moving_average(energy, body_radius)
        for percentile in (55.0, 70.0, 84.0, 90.0, 94.0, BOOM_BODY_CEILING_PERCENTILE):
            key = str(int(percentile)) if float(percentile).is_integer() else str(percentile).replace(".", "_")
            profile[f"global_boom_body_percentile_{key}"] = float(_percentile(body, percentile))
        profile["global_boom_body_peak"] = float(max(body) if body else 0.0)
        with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
            json.dump(profile, fh, ensure_ascii=True)
        _remember_gzip_json(cache_path, profile)
        return profile

    def tile(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        width: int,
        force_mode: str = "auto",
    ) -> Dict[str, Any]:
        path = Path(audio_path).expanduser()
        info = self.info(str(path))
        sr = int(info["sample_rate"])
        total_samples = int(info["total_samples"])
        start = max(0, min(total_samples, int(round(float(start_sec) * sr))))
        end = max(start + 1, min(total_samples, int(round(float(end_sec) * sr))))
        safe_width = max(1, min(MAX_TILE_WIDTH, int(width or 1)))
        samples_per_pixel = max(1.0, (end - start) / safe_width)
        use_raw = (
            force_mode == "samples"
            or (
                force_mode == "auto"
                and (end - start) <= RAW_SAMPLE_LIMIT
                and samples_per_pixel <= RAW_SAMPLES_PER_PIXEL
            )
        )
        mode = "samples" if use_raw else "peaks"
        level = 1 if use_raw else _resolution_level(samples_per_pixel)
        cache_path = self._cache_path(path, start_sample=start, end_sample=end, width=safe_width, mode=mode, level=level)
        if cache_path.exists():
            try:
                cached = _read_gzip_json_with_memory_cache(cache_path)
                if isinstance(cached, dict):
                    cached["cache_hit"] = True
                    return cached
            except Exception:
                cache_path.unlink(missing_ok=True)

        payload = self._raw_tile(path, start, end, safe_width) if use_raw else self._peak_tile(path, start, end, safe_width)
        amplitude_values: List[float] = []
        if payload["mode"] == "samples":
            amplitude_values = list(payload.get("samples", []))
        else:
            amplitude_values = list(payload.get("mins", [])) + list(payload.get("maxs", []))
        peak = max((abs(float(v)) for v in amplitude_values), default=1.0)
        rms_stats = _rms_stats(payload.get("rms", []))
        global_profile = self._global_rms_profile(path, total_samples, sr)
        boom_masks = _boom_masks(
            payload.get("rms", []),
            start_sec=float(start / sr),
            end_sec=float(end / sr),
            local_stats=rms_stats,
            global_profile=global_profile,
        )
        out = {
            "ok": True,
            "audio_key": info["audio_key"],
            "sample_rate": sr,
            "total_samples": total_samples,
            "duration": float(info["duration"]),
            "start_sec": float(start / sr),
            "end_sec": float(end / sr),
            "start_sample": int(start),
            "end_sample": int(end),
            "width": int(safe_width),
            "samples_per_pixel": float(samples_per_pixel),
            "resolution_level": int(level),
            "amplitude_peak": float(peak if peak > 1e-9 else 1.0),
            **rms_stats,
            **global_profile,
            **boom_masks,
            "cache_hit": False,
            **payload,
        }
        with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=True)
        _remember_gzip_json(cache_path, out)
        return out

    def render_png(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        width: int,
        height: int,
        markers: Sequence[Mapping[str, Any]] = (),
    ) -> bytes:
        width = max(400, min(PNG_MAX_WIDTH, int(width or 2400)))
        height = max(180, min(PNG_MAX_HEIGHT, int(height or 720)))
        tile = self.tile(audio_path, start_sec=start_sec, end_sec=end_sec, width=width, force_mode="auto")
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        mid = height // 2
        draw.line([(0, mid), (width, mid)], fill="#d7dce2", width=max(1, height // 360))
        gain = 0.92 / max(float(tile.get("amplitude_peak", 1.0) or 1.0), 1e-9)

        rms_values = [max(0.0, float(v)) for v in tile.get("rms", [])]
        if len(rms_values) >= 2:
            use_global_profile = bool(tile.get("global_visual_maximizer_makeup_gain"))

            def tile_metric(key: str, default: float = 0.0) -> float:
                raw = tile.get(f"global_{key}") if use_global_profile else tile.get(key)
                if raw is None and use_global_profile:
                    raw = tile.get(key)
                out = _float_or_none(raw)
                return float(out) if out is not None else float(default)

            def rms_amp(value: float) -> float:
                makeup = tile_metric("visual_maximizer_makeup_gain", 1.0)
                ceiling = max(0.5, min(1.0, tile_metric("visual_maximizer_ceiling", RMS_MAXIMIZER_CEILING)))
                knee = max(0.2, min(ceiling - 0.02, tile_metric("visual_maximizer_knee", RMS_MAXIMIZER_KNEE)))
                lifted = max(0.0, float(value)) * makeup
                if lifted <= knee:
                    limited = lifted
                else:
                    limiter_range = max(0.001, ceiling - knee)
                    limited = knee + limiter_range * math.tanh((lifted - knee) / limiter_range)
                return float(np.clip((limited / ceiling) ** 0.82, 0.0, 1.0))

            tile_span = max(1e-9, float(tile.get("end_sec", end_sec)) - float(tile.get("start_sec", start_sec)))
            bin_span = tile_span / max(1, len(rms_values))
            energy = [rms_amp(value) for value in rms_values]
            body_radius = max(1, int(round((0.180 / max(bin_span, 1e-9)) / 2.0)))
            if body_radius > 1:
                source = np.asarray(energy, dtype=np.float64)
                prefix = np.concatenate([np.asarray([0.0]), np.cumsum(source)])
                body: List[float] = []
                for index, value in enumerate(source):
                    i0 = max(0, index - body_radius)
                    i1 = min(len(source), index + body_radius + 1)
                    body.append(float((prefix[i1] - prefix[i0]) / max(1, i1 - i0)))
            else:
                body = list(energy)

            server_density = [max(0.0, min(1.0, float(value))) for value in tile.get("boom_body_density", [])]
            server_body_mask = [bool(value) for value in tile.get("boom_body_mask", [])]
            server_relevant_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])]
            server_placeable_mask = [bool(value) for value in tile.get("boom_placeable_mask", [])]
            use_server_masks = (
                len(server_density) == len(body)
                and len(server_body_mask) == len(body)
                and len(server_relevant_mask) == len(body)
                and len(server_placeable_mask) == len(body)
            )
            if use_server_masks:
                body = server_density
                floor = 0.0
                ceiling = 1.0
            else:
                body_array = np.asarray(body, dtype=np.float64)
                q55 = float(np.percentile(body_array, 55.0))
                q70 = float(np.percentile(body_array, 70.0))
                q84 = float(np.percentile(body_array, 84.0))
                q90 = float(np.percentile(body_array, 90.0))
                q94 = float(np.percentile(body_array, 94.0))
                q96 = float(np.percentile(body_array, 96.5))
                global_floor = rms_amp(tile_metric("rms_boom_floor", 0.0))
                global_ceiling = rms_amp(tile_metric("rms_boom_ceiling", 0.0))
                floor = max(
                    BOOM_BODY_NOISE_FLOOR,
                    global_floor,
                    q55 + max(0.0, q90 - q55) * BOOM_BODY_FLOOR_BLEND,
                    q70 * 0.98,
                    q84 * 0.82,
                    q94 * 0.52,
                )
                ceiling = max(floor + 0.045, global_ceiling, q96, q94, q90)
            bin_px = width / max(1, len(body))
            bins_per_bar = max(1, int(round(5.0 / max(bin_px, 0.001))))
            for i in range(0, len(body), bins_per_bar):
                end_index = min(len(body), i + bins_per_bar)
                chunk = body[i:end_index]
                if not chunk:
                    continue
                if use_server_masks:
                    has_body = any(server_relevant_mask[i:end_index])
                    has_placeable = any(server_placeable_mask[i:end_index])
                    if not has_body and not has_placeable:
                        continue
                peak = max(chunk)
                avg = sum(chunk) / max(1, len(chunk))
                density = max(avg, peak * 0.88)
                if use_server_masks:
                    if not has_placeable and density < BOOM_GUI_BODY_VISIBLE_DENSITY:
                        continue
                    normalized = (
                        max(density, BOOM_PLACE_MIN_POST_NORMALIZED)
                        if has_placeable
                        else density
                    )
                else:
                    normalized = max(0.0, min(1.0, (density - floor) / max(ceiling - floor, 0.001)))
                if not use_server_masks and normalized < BOOM_BODY_MIN_NORMALIZED:
                    continue
                x0 = int(round(i * width / max(1, len(body))))
                x1 = int(round(end_index * width / max(1, len(body))))
                x = int(round((x0 + x1) / 2))
                amp = float(normalized**0.54)
                half_height = max(1, int(round(amp * height * 0.43)))
                line_width = max(1, min(max(2, height // 80), int(round(max(1.0, (x1 - x0) * 0.62)))))
                draw.line([(x, mid - half_height), (x, mid + half_height)], fill="#263746", width=line_width)
        elif tile.get("mode") == "samples":
            samples = [float(v) for v in tile.get("samples", [])]
            if len(samples) >= 2:
                span = max(1, len(samples) - 1)
                sample_mask = (
                    [bool(value) for value in tile.get("boom_relevant_mask", [])]
                    if isinstance(tile.get("boom_relevant_mask"), list)
                    else []
                )
                sample_placeable = (
                    [bool(value) for value in tile.get("boom_placeable_mask", [])]
                    if isinstance(tile.get("boom_placeable_mask"), list)
                    else []
                )

                def sample_visible(index: int) -> bool:
                    if not sample_mask and not sample_placeable:
                        return True
                    mask_index = max(0, min(max(len(sample_mask), len(sample_placeable)) - 1, int(round(index * (max(len(sample_mask), len(sample_placeable)) - 1) / max(1, span)))))
                    return bool(
                        (sample_mask[mask_index] if mask_index < len(sample_mask) else False)
                        or (sample_placeable[mask_index] if mask_index < len(sample_placeable) else False)
                    )

                points = [
                    (
                        int(round(i * (width - 1) / span)),
                        int(round(mid - _clip01((sample if sample_visible(i) else 0.0) * gain) * (height * 0.46))),
                    )
                    for i, sample in enumerate(samples)
                ]
                draw.line(points, fill="#263746", width=max(1, height // 320))
                if width / span >= 8:
                    radius = max(2, height // 180)
                    for point_index, (x, y) in enumerate(points):
                        if not sample_visible(point_index):
                            continue
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#263746")
        else:
            mins = [float(v) for v in tile.get("mins", [])]
            maxs = [float(v) for v in tile.get("maxs", [])]
            peak_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])] if isinstance(tile.get("boom_relevant_mask"), list) else []
            peak_placeable = [bool(value) for value in tile.get("boom_placeable_mask", [])] if isinstance(tile.get("boom_placeable_mask"), list) else []
            for x, (lo, hi) in enumerate(zip(mins, maxs)):
                if peak_mask or peak_placeable:
                    mask_index = max(
                        0,
                        min(
                            max(len(peak_mask), len(peak_placeable)) - 1,
                            int(round(x * (max(len(peak_mask), len(peak_placeable)) - 1) / max(1, len(mins) - 1))),
                        ),
                    )
                    visible = bool(
                        (peak_mask[mask_index] if mask_index < len(peak_mask) else False)
                        or (peak_placeable[mask_index] if mask_index < len(peak_placeable) else False)
                    )
                    if not visible:
                        lo = 0.0
                        hi = 0.0
                y1 = int(round(mid - _clip01(hi * gain) * (height * 0.46)))
                y2 = int(round(mid - _clip01(lo * gain) * (height * 0.46)))
                draw.line([(x, y1), (x, y2)], fill="#263746")

        if tile.get("mode") == "samples":
            samples = [float(v) for v in tile.get("samples", [])]
            span = max(1, len(samples) - 1)
            if len(samples) >= 2 and (width / span) >= 0.25:
                sample_mask = (
                    [bool(value) for value in tile.get("boom_relevant_mask", [])]
                    if isinstance(tile.get("boom_relevant_mask"), list)
                    else []
                )
                sample_placeable = (
                    [bool(value) for value in tile.get("boom_placeable_mask", [])]
                    if isinstance(tile.get("boom_placeable_mask"), list)
                    else []
                )
                mask_len = max(len(sample_mask), len(sample_placeable))

                def sample_overlay_visible(index: int) -> bool:
                    if mask_len <= 0:
                        return True
                    mask_index = max(0, min(mask_len - 1, int(round(index * (mask_len - 1) / max(1, span)))))
                    return bool(
                        (sample_mask[mask_index] if mask_index < len(sample_mask) else False)
                        or (sample_placeable[mask_index] if mask_index < len(sample_placeable) else False)
                    )

                segment: List[tuple[int, int]] = []
                for index, sample in enumerate(samples):
                    if not sample_overlay_visible(index):
                        if len(segment) >= 2:
                            draw.line(segment, fill="#111820", width=max(1, height // 420))
                        segment = []
                        continue
                    segment.append(
                        (
                            int(round(index * (width - 1) / span)),
                            int(round(mid - _clip01(sample * gain) * (height * 0.46))),
                        )
                    )
                if len(segment) >= 2:
                    draw.line(segment, fill="#111820", width=max(1, height // 420))

        start = float(tile.get("start_sec", start_sec))
        end = float(tile.get("end_sec", end_sec))
        span_sec = max(1e-9, end - start)
        colors = {
            "ai": "#c2352b",
            "candidate": "#d88716",
            "micro": "#005f73",
            "knee": "#008f7a",
            "asd": "#7a4cff",
            "attack": "#6a3d9a",
            "zero": "#111820",
            "user": "#177245",
        }
        for marker in markers:
            t = _float_or_none(marker.get("time"))
            if t is None or t < start or t > end:
                continue
            x = int(round((t - start) / span_sec * (width - 1)))
            kind = str(marker.get("kind", "user"))
            color = colors.get(kind, "#111820")
            draw.line([(x, 0), (x, height)], fill=color, width=max(2, width // 1400))
            label = str(marker.get("label") or kind.upper())
            draw.rectangle((x + 4, 6, x + 4 + max(28, len(label) * 8), 26), fill="#ffffff", outline=color)
            draw.text((x + 8, 10), label, fill=color)

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def render_svg(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        width: int,
        height: int,
        markers: Sequence[Mapping[str, Any]] = (),
    ) -> bytes:
        width = max(800, min(SVG_MAX_DETAIL_BINS, int(width or 9600)))
        height = max(240, min(PNG_MAX_HEIGHT, int(height or 900)))
        tile = self.tile(audio_path, start_sec=start_sec, end_sec=end_sec, width=width, force_mode="auto")
        mid = height / 2.0
        gain = 0.92 / max(float(tile.get("amplitude_peak", 1.0) or 1.0), 1e-9)
        elements: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" shape-rendering="geometricPrecision">'
            ),
            "<title>Boom-gated drums waveform</title>",
            "<desc>Vector waveform export using the same Boom relevance and placeable masks as the review UI.</desc>",
            '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
            (
                f'<line class="centerline" x1="0" y1="{mid:.3f}" x2="{width}" y2="{mid:.3f}" '
                'stroke="#d7dce2" stroke-width="1" vector-effect="non-scaling-stroke"/>'
            ),
        ]

        def add_line(
            class_name: str,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            *,
            stroke: str = "#263746",
            stroke_width: float = 1.0,
            opacity: Optional[float] = None,
        ) -> None:
            opacity_attr = "" if opacity is None else f' opacity="{max(0.0, min(1.0, float(opacity))):.3f}"'
            elements.append(
                (
                    f'<line class="{class_name}" x1="{x1:.3f}" y1="{y1:.3f}" '
                    f'x2="{x2:.3f}" y2="{y2:.3f}" stroke="{stroke}" '
                    f'stroke-width="{max(0.1, float(stroke_width)):.3f}" '
                    f'vector-effect="non-scaling-stroke"{opacity_attr}/>'
                )
            )

        def add_polyline(class_name: str, points: Sequence[tuple[float, float]], *, stroke: str = "#263746") -> None:
            if len(points) < 2:
                return
            point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
            elements.append(
                (
                    f'<polyline class="{class_name}" points="{point_text}" fill="none" '
                    f'stroke="{stroke}" stroke-width="1" vector-effect="non-scaling-stroke"/>'
                )
            )

        rms_values = [max(0.0, float(v)) for v in tile.get("rms", [])]
        if len(rms_values) >= 2:
            server_density = [max(0.0, min(1.0, float(value))) for value in tile.get("boom_body_density", [])]
            server_relevant_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])]
            server_placeable_mask = [bool(value) for value in tile.get("boom_placeable_mask", [])]
            use_server_masks = (
                len(server_density) == len(rms_values)
                and len(server_relevant_mask) == len(rms_values)
                and len(server_placeable_mask) == len(rms_values)
            )
            if use_server_masks:
                body = server_density
                floor = 0.0
                ceiling = 1.0
            else:
                local_stats = _rms_stats(rms_values)
                global_profile = {
                    key: value
                    for key, value in tile.items()
                    if isinstance(key, str) and key.startswith("global_")
                }
                tile_span = max(1e-9, float(tile.get("end_sec", end_sec)) - float(tile.get("start_sec", start_sec)))
                bin_span = tile_span / max(1, len(rms_values))
                energy = [_boom_rms_amplitude(value, local_stats, global_profile) for value in rms_values]
                radius = max(1, int(round((BOOM_BODY_SMOOTH_SECONDS / max(bin_span, 1e-9)) / 2.0)))
                body = _moving_average(energy, radius)
                body_array = np.asarray(body, dtype=np.float64)
                q55 = float(np.percentile(body_array, 55.0))
                q70 = float(np.percentile(body_array, 70.0))
                q84 = float(np.percentile(body_array, 84.0))
                q90 = float(np.percentile(body_array, 90.0))
                q94 = float(np.percentile(body_array, 94.0))
                q96 = float(np.percentile(body_array, 96.5))
                global_floor = _boom_rms_amplitude(_metric_value(local_stats, global_profile, "rms_boom_floor", 0.0), local_stats, global_profile)
                global_ceiling = _boom_rms_amplitude(_metric_value(local_stats, global_profile, "rms_boom_ceiling", 0.0), local_stats, global_profile)
                floor = max(
                    BOOM_BODY_NOISE_FLOOR,
                    global_floor,
                    q55 + max(0.0, q90 - q55) * BOOM_BODY_FLOOR_BLEND,
                    q70 * 0.98,
                    q84 * 0.82,
                    q94 * 0.52,
                )
                ceiling = max(floor + 0.045, global_ceiling, q96, q94, q90)

            bin_px = width / max(1, len(body))
            bins_per_bar = max(1, int(round(5.0 / max(bin_px, 0.001))))
            for i in range(0, len(body), bins_per_bar):
                end_index = min(len(body), i + bins_per_bar)
                chunk = body[i:end_index]
                if not chunk:
                    continue
                has_relevant = any(server_relevant_mask[i:end_index]) if use_server_masks else True
                has_placeable = any(server_placeable_mask[i:end_index]) if use_server_masks else False
                if use_server_masks and not has_relevant and not has_placeable:
                    continue
                density = max(sum(chunk) / max(1, len(chunk)), max(chunk) * 0.88)
                if use_server_masks:
                    if not has_placeable and density < BOOM_GUI_BODY_VISIBLE_DENSITY:
                        continue
                    normalized = max(density, BOOM_PLACE_MIN_POST_NORMALIZED) if has_placeable else density
                else:
                    normalized = max(0.0, min(1.0, (density - floor) / max(ceiling - floor, 0.001)))
                    if normalized < BOOM_BODY_MIN_NORMALIZED:
                        continue
                x0 = i * width / max(1, len(body))
                x1 = end_index * width / max(1, len(body))
                x = (x0 + x1) / 2.0
                amp = float(normalized**0.54)
                half_height = max(1.0, amp * height * 0.43)
                line_width = max(0.6, min(max(2.0, height / 80.0), max(1.0, (x1 - x0) * 0.62)))
                class_name = "waveform-bar placeable" if has_placeable else "waveform-bar relevant"
                opacity = 0.98 if has_placeable else 0.56 if use_server_masks else 0.90
                add_line(class_name, x, mid - half_height, x, mid + half_height, stroke_width=line_width, opacity=opacity)
        elif tile.get("mode") == "samples":
            samples = [float(v) for v in tile.get("samples", [])]
            if len(samples) >= 2:
                span = max(1, len(samples) - 1)
                sample_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])] if isinstance(tile.get("boom_relevant_mask"), list) else []
                sample_placeable = [bool(value) for value in tile.get("boom_placeable_mask", [])] if isinstance(tile.get("boom_placeable_mask"), list) else []
                mask_len = max(len(sample_mask), len(sample_placeable))

                def sample_visible(index: int) -> bool:
                    if mask_len <= 0:
                        return True
                    mask_index = max(0, min(mask_len - 1, int(round(index * (mask_len - 1) / max(1, span)))))
                    return bool(
                        (sample_mask[mask_index] if mask_index < len(sample_mask) else False)
                        or (sample_placeable[mask_index] if mask_index < len(sample_placeable) else False)
                    )

                segment: List[tuple[float, float]] = []
                for index, sample in enumerate(samples):
                    if not sample_visible(index):
                        add_polyline("waveform-samples", segment)
                        segment = []
                        continue
                    x = index * (width - 1) / span
                    y = mid - _clip01(sample * gain) * (height * 0.46)
                    segment.append((x, y))
                add_polyline("waveform-samples", segment)
        else:
            mins = [float(v) for v in tile.get("mins", [])]
            maxs = [float(v) for v in tile.get("maxs", [])]
            peak_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])] if isinstance(tile.get("boom_relevant_mask"), list) else []
            peak_placeable = [bool(value) for value in tile.get("boom_placeable_mask", [])] if isinstance(tile.get("boom_placeable_mask"), list) else []
            mask_len = max(len(peak_mask), len(peak_placeable))
            for x, (lo, hi) in enumerate(zip(mins, maxs)):
                if mask_len > 0:
                    mask_index = max(0, min(mask_len - 1, int(round(x * (mask_len - 1) / max(1, len(mins) - 1)))))
                    visible = bool(
                        (peak_mask[mask_index] if mask_index < len(peak_mask) else False)
                        or (peak_placeable[mask_index] if mask_index < len(peak_placeable) else False)
                    )
                    if not visible:
                        continue
                y1 = mid - _clip01(hi * gain) * (height * 0.46)
                y2 = mid - _clip01(lo * gain) * (height * 0.46)
                add_line("waveform-peak", float(x), y1, float(x), y2)

        if tile.get("mode") == "samples":
            samples = [float(v) for v in tile.get("samples", [])]
            span = max(1, len(samples) - 1)
            if len(samples) >= 2 and (width / span) >= 0.25:
                sample_mask = [bool(value) for value in tile.get("boom_relevant_mask", [])] if isinstance(tile.get("boom_relevant_mask"), list) else []
                sample_placeable = [bool(value) for value in tile.get("boom_placeable_mask", [])] if isinstance(tile.get("boom_placeable_mask"), list) else []
                mask_len = max(len(sample_mask), len(sample_placeable))

                def sample_detail_visible(index: int) -> bool:
                    if mask_len <= 0:
                        return True
                    mask_index = max(0, min(mask_len - 1, int(round(index * (mask_len - 1) / max(1, span)))))
                    return bool(
                        (sample_mask[mask_index] if mask_index < len(sample_mask) else False)
                        or (sample_placeable[mask_index] if mask_index < len(sample_placeable) else False)
                    )

                segment: List[tuple[float, float]] = []
                for index, sample in enumerate(samples):
                    if not sample_detail_visible(index):
                        add_polyline("waveform-samples", segment, stroke="#111820")
                        segment = []
                        continue
                    x = index * (width - 1) / span
                    y = mid - _clip01(sample * gain) * (height * 0.46)
                    segment.append((x, y))
                add_polyline("waveform-samples", segment, stroke="#111820")

        start = float(tile.get("start_sec", start_sec))
        end = float(tile.get("end_sec", end_sec))
        span_sec = max(1e-9, end - start)
        colors = {
            "ai": "#c2352b",
            "candidate": "#d88716",
            "micro": "#005f73",
            "knee": "#008f7a",
            "asd": "#7a4cff",
            "attack": "#6a3d9a",
            "zero": "#111820",
            "user": "#177245",
        }
        for marker in markers:
            t = _float_or_none(marker.get("time"))
            if t is None or t < start or t > end:
                continue
            x = (t - start) / span_sec * (width - 1)
            kind = str(marker.get("kind", "user"))
            color = colors.get(kind, "#111820")
            label = html.escape(str(marker.get("label") or kind.upper()), quote=True)
            add_line(f"marker marker-{html.escape(kind, quote=True)}", x, 0.0, x, float(height), stroke=color, stroke_width=max(1.5, width / 2200.0))
            label_width = max(34, len(label) * 8 + 12)
            label_x = min(max(2.0, x + 5.0), max(2.0, width - label_width - 2.0))
            elements.append(
                (
                    f'<rect class="marker-label-bg" x="{label_x:.3f}" y="7" width="{label_width:.3f}" height="22" '
                    f'fill="#ffffff" stroke="{color}" vector-effect="non-scaling-stroke"/>'
                )
            )
            elements.append(
                f'<text class="marker-label" x="{label_x + 6:.3f}" y="22" fill="{color}" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="13">'
                f"{label}</text>"
            )

        elements.append("</svg>")
        return ("\n".join(elements) + "\n").encode("utf-8")


def gui_boom_mask_proof_for_tile(
    tile: Mapping[str, Any],
    marker: float,
    *,
    radius_bins: int = 2,
) -> Dict[str, Any]:
    mask = tile.get("boom_placeable_mask") if isinstance(tile.get("boom_placeable_mask"), list) else []
    reasons: List[str] = []
    start = _float_or_none(tile.get("start_sec")) or 0.0
    end = _float_or_none(tile.get("end_sec")) or start
    marker_f = float(marker)
    if not mask:
        return {
            "passes": False,
            "reasons": ["missing_gui_boom_placeable_mask"],
            "placeable_count": int(tile.get("boom_placeable_count") or 0),
            "mask_index": None,
            "start_sec": float(start),
            "end_sec": float(end),
        }
    if marker_f < float(start) or marker_f > float(end):
        reasons.append("marker_outside_gui_waveform_tile")
    span = max(float(end) - float(start), 1e-9)
    mask_index = max(0, min(len(mask) - 1, int(math.floor(((marker_f - float(start)) / span) * len(mask)))))
    bin_span = span / max(1, len(mask))
    radius = max(0, int(radius_bins))
    relevant_radius = max(radius, int(math.ceil(min(0.035, BOOM_PLACE_EDGE_GRACE_SECONDS) / max(bin_span, 1e-9))))
    allowed = any(
        bool(mask[index])
        for index in range(max(0, mask_index - radius), min(len(mask), mask_index + radius + 1))
    )
    placeable_indices = [index for index, value in enumerate(mask) if value]
    nearest_placeable_index: Optional[int] = None
    nearest_placeable_offset_sec: Optional[float] = None
    if placeable_indices:
        nearest_placeable_index = min(placeable_indices, key=lambda index: abs(index - mask_index))
        nearest_placeable_offset_sec = float((int(nearest_placeable_index) - int(mask_index)) * (span / max(1, len(mask))))
    rms_values = [max(0.0, float(value)) for value in tile.get("rms", [])] if isinstance(tile.get("rms"), list) else []
    density_values = (
        [max(0.0, min(1.0, float(value))) for value in tile.get("boom_body_density", [])]
        if isinstance(tile.get("boom_body_density"), list)
        else []
    )
    body_mask = (
        [bool(value) for value in tile.get("boom_body_mask", [])]
        if isinstance(tile.get("boom_body_mask"), list)
        else []
    )
    relevant_mask = (
        [bool(value) for value in tile.get("boom_relevant_mask", [])]
        if isinstance(tile.get("boom_relevant_mask"), list)
        else []
    )
    signal_start = max(0, mask_index - relevant_radius)
    signal_end = min(len(mask), mask_index + relevant_radius + 1)
    marker_rms_max = max((rms_values[index] for index in range(signal_start, min(signal_end, len(rms_values)))), default=0.0)
    marker_density_max = max(
        (density_values[index] for index in range(signal_start, min(signal_end, len(density_values)))),
        default=0.0,
    )
    marker_body_mask = any(bool(body_mask[index]) for index in range(signal_start, min(signal_end, len(body_mask))))
    marker_relevant_mask = any(
        bool(relevant_mask[index])
        for index in range(signal_start, min(signal_end, len(relevant_mask)))
    )
    immediate_bins_250 = max(1, int(math.ceil(0.250 / max(bin_span, 1e-9))))
    immediate_bins_500 = max(immediate_bins_250, int(math.ceil(0.500 / max(bin_span, 1e-9))))
    immediate_end_250 = min(len(mask), mask_index + immediate_bins_250)
    immediate_end_500 = min(len(mask), mask_index + immediate_bins_500)

    def _mean_window(values: Sequence[float], end_index: int) -> float:
        safe_end = min(int(end_index), len(values))
        if safe_end <= mask_index:
            return 0.0
        window = values[mask_index:safe_end]
        return float(sum(window) / max(1, len(window))) if window else 0.0

    def _max_window(values: Sequence[float], end_index: int) -> float:
        safe_end = min(int(end_index), len(values))
        if safe_end <= mask_index:
            return 0.0
        return float(max(values[mask_index:safe_end], default=0.0))

    def _occupancy_window(values: Sequence[bool], end_index: int) -> float:
        safe_end = min(int(end_index), len(values))
        if safe_end <= mask_index:
            return 0.0
        window = values[mask_index:safe_end]
        return float(sum(1 for value in window if value) / max(1, len(window))) if window else 0.0

    marker_post_density_mean_250ms = _mean_window(density_values, immediate_end_250)
    marker_post_density_mean_500ms = _mean_window(density_values, immediate_end_500)
    marker_post_density_max_250ms = _max_window(density_values, immediate_end_250)
    marker_post_density_max_500ms = _max_window(density_values, immediate_end_500)
    marker_post_body_occupancy_250ms = _occupancy_window(body_mask, immediate_end_250)
    marker_post_body_occupancy_500ms = _occupancy_window(body_mask, immediate_end_500)
    marker_post_relevant_occupancy_250ms = _occupancy_window(relevant_mask, immediate_end_250)
    marker_post_relevant_occupancy_500ms = _occupancy_window(relevant_mask, immediate_end_500)
    marker_post_rms_mean_250ms = _mean_window(rms_values, immediate_end_250)
    marker_post_rms_mean_500ms = _mean_window(rms_values, immediate_end_500)
    marker_post_rms_max_250ms = _max_window(rms_values, immediate_end_250)
    marker_post_rms_max_500ms = _max_window(rms_values, immediate_end_500)
    marker_immediate_body_present = bool(
        marker_body_mask
        or marker_post_body_occupancy_250ms >= 0.180
        or marker_post_body_occupancy_500ms >= 0.140
        or (
            marker_relevant_mask
            and marker_post_density_mean_500ms >= 0.080
            and marker_post_density_max_500ms >= 0.300
        )
    )
    tile_rms_peak = _float_or_none(tile.get("rms_peak")) or max(rms_values or [0.0])
    global_rms_peak = _float_or_none(tile.get("global_rms_peak")) or float(tile_rms_peak)
    signal_floor = max(0.006, float(global_rms_peak) * 0.10)
    marker_signal_present = bool(
        marker_relevant_mask
        or marker_body_mask
        or marker_density_max >= 0.220
        or marker_rms_max >= signal_floor
    )
    if not relevant_mask:
        reasons.append("missing_gui_boom_relevant_mask")
    elif not marker_relevant_mask:
        reasons.append("marker_not_on_gui_boom_relevant_mask")
    if not allowed:
        reasons.append("marker_not_on_gui_boom_front_edge_mask")
    placeable_count = int(tile.get("boom_placeable_count") or sum(1 for value in mask if value))
    if placeable_count <= 0:
        reasons.append("gui_tile_has_no_placeable_boom_front_edge")
    return {
        "passes": not reasons,
        "reasons": reasons,
        "placeable_count": int(placeable_count),
        "mask_index": int(mask_index),
        "radius_bins": int(radius),
        "relevant_radius_bins": int(relevant_radius),
        "nearest_placeable_index": int(nearest_placeable_index) if nearest_placeable_index is not None else None,
        "nearest_placeable_offset_sec": (
            float(nearest_placeable_offset_sec) if nearest_placeable_offset_sec is not None else None
        ),
        "marker_rms_max": float(marker_rms_max),
        "marker_body_density_max": float(marker_density_max),
        "marker_body_mask": bool(marker_body_mask),
        "marker_relevant_mask": bool(marker_relevant_mask),
        "marker_post_density_mean_250ms": float(marker_post_density_mean_250ms),
        "marker_post_density_mean_500ms": float(marker_post_density_mean_500ms),
        "marker_post_density_max_250ms": float(marker_post_density_max_250ms),
        "marker_post_density_max_500ms": float(marker_post_density_max_500ms),
        "marker_post_body_occupancy_250ms": float(marker_post_body_occupancy_250ms),
        "marker_post_body_occupancy_500ms": float(marker_post_body_occupancy_500ms),
        "marker_post_relevant_occupancy_250ms": float(marker_post_relevant_occupancy_250ms),
        "marker_post_relevant_occupancy_500ms": float(marker_post_relevant_occupancy_500ms),
        "marker_post_rms_mean_250ms": float(marker_post_rms_mean_250ms),
        "marker_post_rms_mean_500ms": float(marker_post_rms_mean_500ms),
        "marker_post_rms_max_250ms": float(marker_post_rms_max_250ms),
        "marker_post_rms_max_500ms": float(marker_post_rms_max_500ms),
        "marker_immediate_body_present": bool(marker_immediate_body_present),
        "marker_signal_floor": float(signal_floor),
        "marker_signal_present": bool(marker_signal_present),
        "start_sec": float(start),
        "end_sec": float(end),
        "bin_span_sec": float(_float_or_none(tile.get("boom_bin_span_sec")) or bin_span),
    }


def gui_boom_mask_proof(
    audio_path: str,
    marker: float,
    *,
    cache_dir: str | Path,
    view_sec: float = 6.0,
    width: int = 1200,
    radius_bins: int = 2,
    cache: Optional[WaveformCache] = None,
) -> Dict[str, Any]:
    waveform_cache = cache or WaveformCache(Path(cache_dir).expanduser())
    info = waveform_cache.info(audio_path)
    duration = max(0.0, _float_or_none(info.get("duration")) or 0.0)
    marker_f = max(0.0, float(marker))
    half = max(0.050, float(view_sec) / 2.0)
    start = max(0.0, marker_f - half)
    end = min(duration, marker_f + half) if duration > 0.0 else marker_f + half
    if end - start < 0.100:
        end = min(max(duration, end), start + 0.100)
    if marker_f > end:
        start = max(0.0, marker_f - 0.050)
        end = marker_f + 0.050
    tile = waveform_cache.tile(audio_path, start_sec=start, end_sec=end, width=int(width))
    proof = gui_boom_mask_proof_for_tile(tile, marker_f, radius_bins=radius_bins)
    proof.update(
        {
            "audio_path": str(audio_path),
            "marker": float(marker_f),
            "duration": float(duration),
            "width": int(width),
            "cache_hit": bool(tile.get("cache_hit")),
        }
    )
    return proof


def gui_boom_mask_strict_contract_issue(gui_mask: Mapping[str, Any]) -> Optional[str]:
    """Return the production GUI-contract issue for a nominally passing proof.

    `passes` can be relaxed by Boom/front-edge relief for tiny mask
    quantization. The production contract still needs visible marker signal,
    relevant Boom mask evidence, and some immediate post-marker body evidence.
    This helper keeps that stricter interpretation shared by the detector, GUI,
    validators, and audits.
    """

    if not isinstance(gui_mask, Mapping) or not gui_mask:
        return "missing_gui_mask_proof"
    if not bool(gui_mask.get("passes")):
        return None

    reasons = {
        str(reason)
        for reason in list(gui_mask.get("raw_reasons") or []) + list(gui_mask.get("reasons") or [])
        if str(reason)
    }
    staccato_front_relief = bool(gui_mask.get("accepted_by_staccato_front_body_proof"))
    if staccato_front_relief:
        allowed_sparse_front_reasons = {
            "marker_has_no_immediate_drop_body",
            "marker_not_on_gui_boom_relevant_mask",
            "marker_not_on_relevant_boom_mask",
            "marker_not_on_gui_boom_front_edge_mask",
            "gui_tile_has_no_placeable_boom_front_edge",
        }
        strong_staccato_body = bool(
            reasons <= allowed_sparse_front_reasons
            and gui_mask.get("marker_signal_present") is True
            and (_float_or_none(gui_mask.get("staccato_front_profile_score")) or 0.0) >= 0.565
            and (_float_or_none(gui_mask.get("staccato_front_body_score")) or 0.0) >= 0.640
            and (_float_or_none(gui_mask.get("staccato_front_darkness")) or 0.0) >= 0.580
            and (_float_or_none(gui_mask.get("staccato_front_post8")) or 0.0) >= 0.540
            and (_float_or_none(gui_mask.get("staccato_front_drum_cont")) or 0.0) >= 0.740
        )
        massive_immediate_body = bool(
            gui_mask.get("staccato_front_massive_immediate_body")
            and reasons <= allowed_sparse_front_reasons
            and gui_mask.get("marker_signal_present") is True
            and (_float_or_none(gui_mask.get("staccato_front_profile_score")) or 0.0) >= 0.700
            and (_float_or_none(gui_mask.get("staccato_front_body_score")) or 0.0) >= 0.760
            and (_float_or_none(gui_mask.get("staccato_front_darkness")) or 0.0) >= 0.760
            and (_float_or_none(gui_mask.get("staccato_front_post8")) or 0.0) >= 0.560
            and (_float_or_none(gui_mask.get("staccato_front_drum_cont")) or 0.0) >= 0.650
            and (
                (_float_or_none(gui_mask.get("staccato_front_bass")) or 0.0) >= 0.450
                or (_float_or_none(gui_mask.get("staccato_front_simultaneity")) or 0.0) >= 0.700
            )
        )
        if strong_staccato_body or massive_immediate_body:
            return None
    opening_body_start_relief = bool(gui_mask.get("accepted_by_opening_body_start_proof"))
    if opening_body_start_relief:
        allowed_opening_reasons = {
            "marker_not_on_gui_boom_relevant_mask",
            "marker_not_on_relevant_boom_mask",
            "marker_not_on_gui_boom_front_edge_mask",
        }
        opening_body = bool(
            reasons <= allowed_opening_reasons
            and gui_mask.get("marker_signal_present") is True
            and gui_mask.get("marker_immediate_body_present") is True
            and (_float_or_none(gui_mask.get("opening_body_start_body_score")) or 0.0) >= 0.760
            and (_float_or_none(gui_mask.get("opening_body_start_darkness")) or 0.0) >= 0.740
            and (_float_or_none(gui_mask.get("opening_body_start_post8")) or 0.0) >= 0.520
            and (_float_or_none(gui_mask.get("opening_body_start_bass")) or 0.0) >= 0.560
            and (_float_or_none(gui_mask.get("opening_body_start_drum_cont")) or 0.0) >= 0.620
            and 1 <= int(gui_mask.get("opening_body_start_clock_bar") or 0) <= 5
        )
        if opening_body:
            return None
    sustained_body_section_relief = bool(gui_mask.get("accepted_by_sustained_body_section_gui_proof"))
    if sustained_body_section_relief:
        if (
            reasons <= {"marker_not_on_gui_boom_front_edge_mask"}
            and gui_mask.get("marker_signal_present") is True
            and gui_mask.get("marker_relevant_mask") is True
            and gui_mask.get("marker_immediate_body_present") is True
            and gui_mask.get("marker_body_mask") is True
            and (_float_or_none(gui_mask.get("marker_post_body_occupancy_250ms")) or 0.0) >= 0.450
            and (_float_or_none(gui_mask.get("marker_post_relevant_occupancy_250ms")) or 0.0) >= 0.650
            and (_float_or_none(gui_mask.get("marker_post_rms_mean_250ms")) or 0.0) >= 0.180
        ):
            return None
    if "missing_gui_boom_relevant_mask" in reasons:
        return "missing_gui_boom_relevant_mask"
    if "missing_gui_boom_placeable_mask" in reasons:
        return "missing_gui_boom_placeable_mask"
    if "marker_outside_gui_waveform_tile" in reasons:
        return "marker_outside_gui_waveform_tile"
    if gui_mask.get("marker_relevant_mask") is False or "marker_not_on_gui_boom_relevant_mask" in reasons:
        return "marker_not_on_relevant_boom_mask"
    if gui_mask.get("marker_signal_present") is False:
        return "blank_marker_no_visible_signal"
    if "gui_tile_has_no_placeable_boom_front_edge" in reasons:
        return "gui_tile_has_no_placeable_boom_front_edge"

    nearest = _float_or_none(gui_mask.get("nearest_placeable_offset_sec"))
    if "marker_not_on_gui_boom_front_edge_mask" in reasons and (
        nearest is None or abs(float(nearest)) > GUI_STRICT_FRONT_EDGE_RELIEF_SECONDS
    ):
        return "marker_not_on_gui_boom_front_edge_mask"

    transition_relief_relevant = max(
        _float_or_none(gui_mask.get("marker_post_relevant_occupancy_250ms")) or 0.0,
        _float_or_none(gui_mask.get("marker_post_relevant_occupancy_500ms")) or 0.0,
    )
    transition_relief_rms = max(
        _float_or_none(gui_mask.get("marker_post_rms_mean_250ms")) or 0.0,
        _float_or_none(gui_mask.get("marker_post_rms_mean_500ms")) or 0.0,
    )
    transition_front_relief = bool(
        (
            gui_mask.get("accepted_by_sparse_transition_front_edge")
            or gui_mask.get("accepted_by_low_density_transition_body_start")
        )
        and gui_mask.get("marker_signal_present") is True
        and gui_mask.get("marker_relevant_mask") is True
        and nearest is not None
        and abs(float(nearest)) <= GUI_STRICT_FRONT_EDGE_RELIEF_SECONDS
        and int(gui_mask.get("placeable_count") or 0) >= 2
        and transition_relief_relevant >= 0.600
        and transition_relief_rms >= 0.180
    )
    sparse_relief = bool(
        gui_mask.get("accepted_by_staccato_front_body_proof")
        or gui_mask.get("accepted_by_sustained_body_section_gui_proof")
        or gui_mask.get("accepted_by_full_track_stable_wide_zoom_conflict")
        or gui_mask.get("accepted_by_actual_visual_body_proof")
        or gui_mask.get("accepted_by_gui_sparse_pulse_proof")
        or transition_front_relief
        or gui_mask.get("stable_wide_gui_mask_proof")
        or gui_mask.get("fine_front_edge_immediate_body_relief")
    )
    if gui_mask.get("marker_immediate_body_present") is False and not sparse_relief:
        post_body = max(
            _float_or_none(gui_mask.get("marker_post_body_occupancy_250ms")) or 0.0,
            _float_or_none(gui_mask.get("marker_post_body_occupancy_500ms")) or 0.0,
        )
        post_density_mean = max(
            _float_or_none(gui_mask.get("marker_post_density_mean_250ms")) or 0.0,
            _float_or_none(gui_mask.get("marker_post_density_mean_500ms")) or 0.0,
        )
        post_density_peak = max(
            _float_or_none(gui_mask.get("marker_post_density_max_250ms")) or 0.0,
            _float_or_none(gui_mask.get("marker_post_density_max_500ms")) or 0.0,
        )
        post_relevant = max(
            _float_or_none(gui_mask.get("marker_post_relevant_occupancy_250ms")) or 0.0,
            _float_or_none(gui_mask.get("marker_post_relevant_occupancy_500ms")) or 0.0,
        )
        if post_body <= 0.050:
            return "marker_has_no_immediate_drop_body"
        if post_body <= 0.035 and post_density_mean <= 0.035 and post_relevant <= 0.060:
            return "marker_has_no_immediate_drop_body"
        if post_body <= 0.010 and post_density_mean <= 0.055 and post_density_peak <= 0.160:
            return "marker_has_no_immediate_drop_body"
    elif "marker_immediate_body_present" not in gui_mask:
        return "stale_missing_marker_immediate_body"

    return None


def enforce_strict_gui_boom_mask_contract(gui_mask: Mapping[str, Any]) -> Dict[str, Any]:
    proof = dict(gui_mask) if isinstance(gui_mask, Mapping) else {}
    issue = gui_boom_mask_strict_contract_issue(proof)
    if not issue:
        return proof
    reasons = [str(reason) for reason in proof.get("reasons") or [] if str(reason)]
    if issue not in reasons:
        reasons.append(issue)
    proof.update(
        {
            "passes": False,
            "strict_gui_contract_failed": True,
            "strict_gui_contract_issue": issue,
            "reasons": reasons,
        }
    )
    return proof


def accept_gui_boom_mask_with_front_edge_proof(
    gui_mask: Mapping[str, Any],
    boom_proof: Mapping[str, Any],
    *,
    exact_offset_sec: float = 0.015,
    exact_profile_score: float = 0.495,
    near_offset_sec: float = 0.040,
    near_profile_score: float = 0.560,
) -> Dict[str, Any]:
    proof = dict(gui_mask)
    if bool(proof.get("passes")):
        return proof
    if not bool(boom_proof.get("passes")):
        return proof
    reasons = [str(reason) for reason in proof.get("reasons") or [] if str(reason)]
    if proof.get("marker_relevant_mask") is False:
        if "marker_not_on_relevant_boom_mask" not in reasons:
            reasons.append("marker_not_on_relevant_boom_mask")
        proof["reasons"] = reasons
        return proof
    if "missing_gui_boom_relevant_mask" in reasons or "marker_not_on_gui_boom_relevant_mask" in reasons:
        return proof
    if proof.get("marker_signal_present") is False:
        if "marker_has_no_visible_signal" not in reasons:
            reasons.append("marker_has_no_visible_signal")
        proof["reasons"] = reasons
        return proof
    placeable_count = int(proof.get("placeable_count") or 0)
    if placeable_count <= 0 or "gui_tile_has_no_placeable_boom_front_edge" in reasons:
        return proof
    nearest_placeable_offset = _float_or_none(proof.get("nearest_placeable_offset_sec"))
    if "marker_not_on_gui_boom_front_edge_mask" in reasons and (
        nearest_placeable_offset is None or abs(float(nearest_placeable_offset)) > float(near_offset_sec)
    ):
        return proof
    nearest = boom_proof.get("nearest") if isinstance(boom_proof.get("nearest"), Mapping) else {}
    nearest_profile = boom_proof.get("nearest_profile") if isinstance(boom_proof.get("nearest_profile"), Mapping) else {}
    offset_raw = _float_or_none(nearest.get("offset_sec"))
    offset = abs(float(offset_raw)) if offset_raw is not None else 999999.0
    profile_score_raw = _float_or_none(nearest_profile.get("profile_score"))
    profile_score = float(profile_score_raw) if profile_score_raw is not None else 0.0
    exact_front_edge = bool(offset <= float(exact_offset_sec) and profile_score >= float(exact_profile_score))
    proven_near_front_edge = bool(offset <= float(near_offset_sec) and profile_score >= float(near_profile_score))
    if bool(nearest_profile.get("passes_profile")) and (exact_front_edge or proven_near_front_edge):
        proof.update(
            {
                "passes": True,
                "accepted_by_boom_front_edge_proof": True,
                "raw_reasons": reasons,
                "reasons": [],
                "front_edge_offset_sec": float(offset_raw if offset_raw is not None else 0.0),
                "front_edge_profile_score": float(profile_score),
            }
        )
    return proof
