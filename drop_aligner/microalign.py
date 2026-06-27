from __future__ import annotations

import math
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .auto_verifier import (
    DEFAULT_AUTO_VERIFIER_PATH,
    load_auto_verifier_payload,
    predict_auto_verifier,
    predict_auto_verifier_candidates,
)
from .candidate_chooser import choose_learned_candidate

try:
    from ableton_analysis_adapter import extract_ableton_onset_markers
except Exception:
    extract_ableton_onset_markers = None


MICROALIGN_FEATURE_KEYS = [
    "microaligned_time",
    "micro_confidence",
    "snap_offset_ms",
    "attack_peak_strength",
    "attack_cleanliness",
    "sustained_after_attack",
    "zero_crossing_quality",
    "visual_onset_knee_time",
    "visual_onset_knee_offset_ms",
    "visual_onset_knee_quality",
    "visual_onset_knee_used",
    "input_boundary_quality",
    "input_boundary_used",
    "ableton_asd_time",
    "ableton_asd_offset_ms",
    "ableton_asd_quality",
    "ableton_asd_used",
]

DEFAULT_AUTO_GATE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "models" / "auto_gate_config.json"


def _clip01(value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _normalize(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, 5.0))
    hi = float(np.percentile(x, percentile))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _smooth(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or samples <= 1:
        return x
    kernel = np.ones(samples, dtype=np.float64) / float(samples)
    return np.convolve(x, kernel, mode="same")


def _trailing_mean(values: np.ndarray, samples: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return x
    window = max(1, int(samples))
    indices = np.arange(n, dtype=np.int64)
    starts = np.maximum(0, indices - window)
    counts = np.maximum(1, indices - starts)
    csum = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(x)])
    return (csum[indices] - csum[starts]) / counts


def _forward_mean(values: np.ndarray, samples: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return x
    window = max(1, int(samples))
    indices = np.arange(n, dtype=np.int64)
    ends = np.minimum(n, indices + window)
    counts = np.maximum(1, ends - indices)
    csum = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(x)])
    return (csum[ends] - csum[indices]) / counts


def _local_contrast(values: np.ndarray, sr: int, *, pre_sec: float = 0.060, post_sec: float = 0.024) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    pre = _trailing_mean(x, max(1, int(round(float(pre_sec) * sr))))
    post = _forward_mean(x, max(1, int(round(float(post_sec) * sr))))
    contrast = np.maximum(0.0, post - pre)
    return _normalize(contrast, percentile=97.0)


def _looks_like_drums(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("drums") or name.startswith("drum_") or name.startswith("drum-")


def _load_window(audio_path: str, candidate_time: float, before_sec: float, after_sec: float) -> tuple[np.ndarray, int, float]:
    start = max(0.0, float(candidate_time) - float(before_sec))
    duration = float(before_sec) + float(after_sec)
    try:
        with sf.SoundFile(str(audio_path)) as fh:
            sr = int(fh.samplerate)
            frame_start = max(0, min(int(round(start * sr)), int(fh.frames)))
            frames = max(0, min(int(round(duration * sr)), int(fh.frames) - frame_start))
            fh.seek(frame_start)
            data = fh.read(frames, dtype="float32", always_2d=True)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > 1:
            y = np.mean(arr, axis=1, dtype=np.float32)
        else:
            y = arr.reshape(-1).astype(np.float32, copy=False)
    except Exception:
        y, sr = librosa.load(audio_path, sr=None, mono=True, offset=start, duration=duration)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Could not load microalignment window: {audio_path}")
    y = np.nan_to_num(y, copy=False)
    peak = float(np.percentile(np.abs(y), 99.9))
    if peak > 1e-9:
        y = np.clip(y / peak, -1.0, 1.0).astype(np.float32)
    return y, int(sr), start


def _percussive_window(y: np.ndarray, audio_path: str) -> np.ndarray:
    if _looks_like_drums(audio_path):
        return np.asarray(y, dtype=np.float32)
    try:
        _, percussive = librosa.effects.hpss(np.asarray(y, dtype=np.float32))
        out = np.asarray(percussive, dtype=np.float32)
        if out.size and float(np.max(np.abs(out))) > 1e-9:
            return out
    except Exception:
        pass
    return np.asarray(y, dtype=np.float32)


def _interp_frames_to_samples(values: np.ndarray, frame_samples: np.ndarray, n_samples: int) -> np.ndarray:
    if values.size == 0 or frame_samples.size == 0:
        return np.zeros(n_samples, dtype=np.float64)
    samples = np.arange(n_samples, dtype=np.float64)
    return np.interp(samples, frame_samples.astype(np.float64), values.astype(np.float64), left=0.0, right=0.0)


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-9))


def _median_smooth(values: np.ndarray, kernel_size: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 3:
        return x
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    size = min(size, int(x.size) if int(x.size) % 2 == 1 else int(x.size) - 1)
    if size <= 1:
        return x
    try:
        return np.asarray(signal.medfilt(x, kernel_size=size), dtype=np.float64)
    except Exception:
        return x


def _frame_rise_curves(y: np.ndarray, sr: int, n_samples: int) -> Dict[str, np.ndarray]:
    if n_samples < 64:
        zeros = np.zeros(n_samples, dtype=np.float64)
        return {"rms_rise": zeros, "peak_rise": zeros, "crest_click": zeros}
    frame_length = min(2048, max(128, int(round(0.030 * sr))))
    frame_length = min(frame_length, int(n_samples))
    hop = 32 if sr >= 32000 else 16
    if frame_length <= 0 or hop <= 0:
        zeros = np.zeros(n_samples, dtype=np.float64)
        return {"rms_rise": zeros, "peak_rise": zeros, "crest_click": zeros}
    try:
        frames = librosa.util.frame(np.asarray(y, dtype=np.float64), frame_length=frame_length, hop_length=hop)
    except Exception:
        zeros = np.zeros(n_samples, dtype=np.float64)
        return {"rms_rise": zeros, "peak_rise": zeros, "crest_click": zeros}
    if frames.size == 0 or frames.shape[-1] < 2:
        zeros = np.zeros(n_samples, dtype=np.float64)
        return {"rms_rise": zeros, "peak_rise": zeros, "crest_click": zeros}

    rms = np.sqrt(np.mean(frames * frames, axis=0))
    peak = np.percentile(np.abs(frames), 99.0, axis=0)
    rms_db = _median_smooth(_db(rms), 5)
    peak_db = _median_smooth(_db(peak), 3)
    rms_rise = np.maximum(0.0, np.diff(rms_db, prepend=rms_db[:1]))
    peak_rise = np.maximum(0.0, np.diff(peak_db, prepend=peak_db[:1]))
    crest_click = np.maximum(0.0, peak_rise - (1.65 * rms_rise))
    frame_samples = np.arange(frames.shape[-1], dtype=np.int64) * int(hop) + int(frame_length // 2)
    frame_samples = np.clip(frame_samples, 0, max(0, int(n_samples) - 1))
    return {
        "rms_rise": _interp_frames_to_samples(_normalize(rms_rise, percentile=97.0), frame_samples, n_samples),
        "peak_rise": _interp_frames_to_samples(_normalize(peak_rise, percentile=97.0), frame_samples, n_samples),
        "crest_click": _interp_frames_to_samples(_normalize(crest_click, percentile=97.0), frame_samples, n_samples),
    }


def _spectral_curves(y: np.ndarray, sr: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 256:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)
    n_fft = min(1024, max(256, 2 ** int(math.floor(math.log2(max(256, n_samples // 2))))))
    hop = 32 if sr >= 32000 else 16
    try:
        stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop, center=True)
        mag = np.abs(stft).astype(np.float64)
    except Exception:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)
    if mag.shape[1] < 2:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)

    flux = np.maximum(0.0, np.diff(mag, axis=1, prepend=mag[:, :1])).mean(axis=0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 35.0) & (freqs <= 180.0)
    if not np.any(low_mask):
        low_mask[: max(1, len(freqs) // 12)] = True
    low_energy = np.mean(mag[low_mask, :] ** 2, axis=0)
    low_flux = np.maximum(0.0, np.diff(low_energy, prepend=low_energy[:1]))
    frames = librosa.frames_to_samples(np.arange(mag.shape[1]), hop_length=hop)
    return (
        _interp_frames_to_samples(_normalize(flux), frames, n_samples),
        _interp_frames_to_samples(_normalize(low_flux), frames, n_samples),
    )


def _transient_curves(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    n = len(y)
    abs_y = np.abs(y).astype(np.float64)
    rms_fast = np.sqrt(_smooth(y.astype(np.float64) ** 2, max(3, int(round(0.0025 * sr)))))
    rms_slow = np.sqrt(_smooth(y.astype(np.float64) ** 2, max(5, int(round(0.010 * sr)))))
    envelope = _smooth(abs_y, max(3, int(round(0.0015 * sr))))
    rise = np.maximum(0.0, np.diff(rms_fast, prepend=rms_fast[:1]))
    flux, low_flux = _spectral_curves(y, sr, n)
    frame_rises = _frame_rise_curves(y, sr, n)
    rms_rise = frame_rises["rms_rise"]
    peak_rise = frame_rises["peak_rise"]
    crest_click = frame_rises["crest_click"]

    noise_floor = _trailing_mean(envelope, max(3, int(round(0.070 * sr))))
    denoised_envelope = np.maximum(0.0, envelope - (0.70 * noise_floor))
    impact_rise = np.maximum(0.0, np.diff(denoised_envelope, prepend=denoised_envelope[:1]))
    impact_contrast = _local_contrast(envelope, sr)
    impact_score = (
        (0.27 * _normalize(denoised_envelope))
        + (0.21 * _normalize(impact_rise, percentile=97.0))
        + (0.18 * impact_contrast)
        + (0.14 * rms_rise)
        + (0.10 * peak_rise)
        + (0.07 * low_flux)
        + (0.03 * flux)
    )
    impact_score = np.power(_normalize(impact_score, percentile=97.0), 1.12)

    raw_score = (
        (0.30 * _normalize(envelope))
        + (0.25 * _normalize(rise))
        + (0.25 * flux)
        + (0.20 * low_flux)
    )
    score = _normalize(raw_score)
    return {
        "abs": abs_y,
        "envelope": envelope,
        "noise_floor": noise_floor,
        "denoised_envelope": denoised_envelope,
        "rms_fast": rms_fast,
        "rms_slow": rms_slow,
        "rise": rise,
        "impact_rise": impact_rise,
        "impact_contrast": impact_contrast,
        "impact_score": impact_score,
        "rms_rise": rms_rise,
        "peak_rise": peak_rise,
        "crest_click": crest_click,
        "flux": flux,
        "low_flux": low_flux,
        "raw_score": _normalize(raw_score),
        "score": _normalize(score),
    }


def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    start = max(0, int(start))
    end = min(len(values), max(start, int(end)))
    if end <= start:
        return 0.0
    return float(np.mean(values[start:end]))


def _candidate_peak_indices(score: np.ndarray, env: np.ndarray, sr: int) -> np.ndarray:
    if score.size < 3:
        return np.asarray([], dtype=np.int64)
    distance = max(1, int(round(0.035 * sr)))
    height = max(0.20, float(np.percentile(score, 70.0)))
    peaks, _ = signal.find_peaks(score, distance=distance, height=height, prominence=0.025)
    if peaks.size:
        return peaks.astype(np.int64)
    fallback = int(np.argmax((0.65 * score) + (0.35 * _normalize(env))))
    return np.asarray([fallback], dtype=np.int64)


def _score_peak(idx: int, curves: Mapping[str, np.ndarray], candidate_idx: int, sr: int) -> tuple[float, Dict[str, float]]:
    env = curves["envelope"]
    score = curves["score"]
    impact_score = curves.get("impact_score", score)
    impact_contrast_curve = curves.get("impact_contrast", impact_score)
    denoised_env = curves.get("denoised_envelope", env)
    noise_floor_curve = curves.get("noise_floor", np.zeros_like(env))
    rms_rise_curve = curves.get("rms_rise", np.zeros_like(env))
    peak_rise_curve = curves.get("peak_rise", np.zeros_like(env))
    crest_click_curve = curves.get("crest_click", np.zeros_like(env))
    peak_env = float(env[idx])
    pre_start = max(0, idx - int(round(0.090 * sr)))
    pre_end = max(pre_start + 1, idx - int(round(0.008 * sr)))
    post_start = min(len(env), idx + int(round(0.006 * sr)))
    post_end = min(len(env), idx + int(round(0.120 * sr)))
    pre_floor = float(np.percentile(env[pre_start:pre_end], 35.0)) if pre_end > pre_start else 0.0
    pre_median = float(np.median(env[pre_start:pre_end])) if pre_end > pre_start else 0.0
    post_energy = _window_mean(env, post_start, post_end)
    immediate = _window_mean(env, idx, min(len(env), idx + int(round(0.018 * sr))))
    slope = _window_mean(curves["rise"], max(0, idx - int(round(0.006 * sr))), min(len(env), idx + int(round(0.012 * sr))))
    impact_peak = _clip01(float(impact_score[idx]))
    impact_contrast = _clip01(
        _window_mean(
            impact_contrast_curve,
            max(0, idx - int(round(0.006 * sr))),
            min(len(env), idx + int(round(0.020 * sr))),
        )
    )
    denoised_peak = float(denoised_env[idx]) if idx < len(denoised_env) else peak_env
    noise_floor = float(noise_floor_curve[idx]) if idx < len(noise_floor_curve) else 0.0
    rms_rise_score = _clip01(float(rms_rise_curve[idx])) if idx < len(rms_rise_curve) else 0.0
    peak_rise_score = _clip01(float(peak_rise_curve[idx])) if idx < len(peak_rise_curve) else 0.0
    crest_click_score = _clip01(float(crest_click_curve[idx])) if idx < len(crest_click_curve) else 0.0
    low_flux_score = _clip01(float(curves["low_flux"][idx])) if idx < len(curves["low_flux"]) else 0.0

    peak_strength = _clip01(float(score[idx]))
    attack_cleanliness = _clip01(((peak_env - pre_median) / max(1e-6, peak_env)) * 1.20)
    floor_cleanliness = _clip01(denoised_peak / max(1e-6, peak_env))
    attack_cleanliness = max(attack_cleanliness, 0.82 * floor_cleanliness)
    sustained = _clip01(post_energy / max(1e-6, max(peak_env, immediate)) * 1.25)
    slope_score = _clip01(slope / max(1e-6, float(np.percentile(curves["rise"], 97.0))) if curves["rise"].size else 0.0)
    distance_penalty = _clip01(abs(float(idx - candidate_idx)) / max(1.0, 0.250 * sr))
    pre_click_penalty = 0.45 if sustained < 0.24 and peak_strength < 0.55 else 0.0
    riser_penalty = 0.32 if attack_cleanliness < 0.45 and sustained > 0.65 else 0.0
    isolated_pre_hit_penalty = (
        0.36
        if idx < candidate_idx - int(round(0.020 * sr))
        and (impact_peak < 0.35 or rms_rise_score < 0.24)
        and (impact_contrast < 0.25 or crest_click_score > 0.55)
        else 0.0
    )
    pre_candidate_riser_penalty = (
        0.20
        if idx < candidate_idx - int(round(0.035 * sr)) and attack_cleanliness < 0.50 and sustained > 0.70
        else 0.0
    )
    tail_onset_penalty = (
        0.24
        if idx < candidate_idx - int(round(0.030 * sr))
        and sustained > 0.72
        and low_flux_score < 0.18
        and impact_contrast < 0.66
        else 0.0
    )
    body_rise_score = _clip01(
        max(
            impact_contrast,
            (0.55 * rms_rise_score) + (0.45 * low_flux_score),
            (0.70 * impact_peak) + (0.30 * low_flux_score),
        )
    )

    composite = (
        (0.32 * peak_strength)
        + (0.24 * attack_cleanliness)
        + (0.23 * sustained)
        + (0.11 * body_rise_score)
        + (0.15 * slope_score)
        - (0.45 * distance_penalty)
        - pre_click_penalty
        - riser_penalty
        - isolated_pre_hit_penalty
        - pre_candidate_riser_penalty
        - tail_onset_penalty
    )
    return float(composite), {
        "attack_peak_strength": peak_strength,
        "attack_cleanliness": attack_cleanliness,
        "sustained_after_attack": sustained,
        "denoised_impact_strength": impact_peak,
        "impact_contrast": impact_contrast,
        "rms_rise_score": rms_rise_score,
        "peak_rise_score": peak_rise_score,
        "crest_click_score": crest_click_score,
        "low_flux_score": low_flux_score,
        "body_rise_score": body_rise_score,
        "local_noise_floor": float(noise_floor),
        "slope_score": slope_score,
        "pre_floor": float(pre_floor),
        "peak_env": float(peak_env),
    }


def _find_attack_start(peak_idx: int, curves: Mapping[str, np.ndarray], sr: int) -> int:
    env = curves["envelope"]
    rise = curves["rise"]
    back = max(1, int(round(0.080 * sr)))
    start = max(0, peak_idx - back)
    pre = env[start:peak_idx]
    if pre.size:
        floor = float(np.percentile(pre, 25.0))
        pre_hi = float(np.percentile(pre, 80.0))
    else:
        floor = 0.0
        pre_hi = 0.0
    peak_env = float(env[peak_idx])
    threshold = max(pre_hi, floor + (0.14 * max(0.0, peak_env - floor)))
    sustain_threshold = floor + (0.08 * max(0.0, peak_env - floor))
    hold = max(1, int(round(0.004 * sr)))
    slope_threshold = float(np.percentile(rise[start : peak_idx + 1], 65.0)) if peak_idx > start else 0.0

    best = peak_idx
    for idx in range(start, max(start, peak_idx - hold)):
        if float(env[idx]) < threshold:
            continue
        if _window_mean(env, idx, idx + hold) < sustain_threshold:
            continue
        if _window_mean(rise, idx, idx + hold) < slope_threshold * 0.20:
            continue
        best = idx
        break

    valley_limit = max(start, peak_idx - int(round(0.040 * sr)))
    below = np.where(env[valley_limit:peak_idx] <= threshold * 0.75)[0]
    if below.size:
        best = min(best, int(valley_limit + below[-1] + 1))
    return int(np.clip(best, 0, len(env) - 1))


def _zero_crossing_before(y: np.ndarray, attack_idx: int, sr: int) -> tuple[int, float]:
    if y.size == 0:
        return int(attack_idx), 0.0
    search = max(1, int(round(0.012 * sr)))
    start = max(1, int(attack_idx) - search)
    end = max(start + 1, int(attack_idx) + 1)
    best = int(attack_idx)
    best_abs = abs(float(y[best])) if 0 <= best < len(y) else float("inf")
    crossing_idx: Optional[int] = None
    for idx in range(end - 1, start, -1):
        v0 = float(y[idx - 1])
        v1 = float(y[idx])
        if abs(v1) < best_abs:
            best = idx
            best_abs = abs(v1)
        if (v0 <= 0.0 <= v1) or (v0 >= 0.0 >= v1):
            crossing_idx = idx
            best = idx
            best_abs = abs(v1)
            break
    local_peak = float(np.percentile(np.abs(y[max(0, attack_idx - search) : min(len(y), attack_idx + search)]), 95.0))
    amp_quality = 1.0 - _clip01(best_abs / max(1e-6, local_peak))
    distance_quality = 1.0 - _clip01(abs(float(best - attack_idx)) / max(1.0, float(search)))
    quality = _clip01((0.70 * amp_quality) + (0.30 * distance_quality))
    if crossing_idx is None:
        quality *= 0.75
    return int(best), float(quality)


def _zero_crossings_in_range(y: np.ndarray, start: int, end: int) -> List[int]:
    start = max(1, int(start))
    end = min(len(y), max(start + 1, int(end)))
    out: List[int] = []
    for idx in range(start, end):
        v0 = float(y[idx - 1])
        v1 = float(y[idx])
        if (v0 <= 0.0 <= v1) or (v0 >= 0.0 >= v1):
            out.append(int(idx))
    return out


@lru_cache(maxsize=128)
def _load_asd_marker_seconds(audio_path: str) -> tuple[float, ...]:
    if extract_ableton_onset_markers is None:
        return ()
    try:
        markers = extract_ableton_onset_markers(str(audio_path))
    except Exception:
        return ()
    if markers is None:
        return ()
    return tuple(float(t) for t in markers.candidate_seconds)


def _centerline_departure_before_peak(
    y: np.ndarray,
    curves: Mapping[str, np.ndarray],
    candidate_idx: int,
    peak_idx: int,
    attack_idx: int,
    sr: int,
) -> tuple[Optional[int], float]:
    if y.size == 0 or peak_idx <= 0:
        return None, 0.0

    env = curves["envelope"]
    abs_y = np.abs(y).astype(np.float64)
    peak_window = max(1, int(round(0.030 * sr)))
    p0 = max(0, int(peak_idx) - peak_window)
    p1 = min(len(y), int(peak_idx) + peak_window)
    local_peak = float(np.percentile(abs_y[p0:p1], 98.0)) if p1 > p0 else float(np.max(abs_y))
    if local_peak <= 1e-8:
        return None, 0.0

    search_back = max(1, int(round(0.045 * sr)))
    search_start = max(1, int(peak_idx) - search_back)
    search_end = min(len(y), max(search_start + 1, int(attack_idx)))
    pre_floor = float(np.percentile(env[search_start:search_end], 20.0)) if search_end > search_start else 0.0
    peak_env = float(np.percentile(env[p0:p1], 95.0)) if p1 > p0 else float(env[int(peak_idx)])
    quiet_limit = pre_floor + (0.055 * max(0.0, peak_env - pre_floor))
    future = max(1, int(round(0.018 * sr)))

    def quality_for(idx: int, *, candidate_bonus: float = 0.0) -> float:
        if idx >= peak_idx:
            return 0.0
        if (float(peak_idx - idx) / float(sr)) > 0.045:
            return 0.0
        if float(env[idx]) > quiet_limit:
            return 0.0
        f1 = min(len(y), idx + future)
        if f1 <= idx + 1:
            return 0.0
        future_peak = float(np.max(abs_y[idx:f1]))
        future_env = float(np.percentile(env[idx:f1], 80.0))
        if future_peak < 0.28 * local_peak or future_env < quiet_limit * 1.8:
            return 0.0
        center_quality = 1.0 - _clip01(abs(float(y[idx])) / max(1e-8, 0.12 * local_peak))
        quiet_quality = 1.0 - _clip01(float(env[idx]) / max(1e-8, quiet_limit))
        future_quality = _clip01(future_peak / max(1e-8, local_peak))
        distance_quality = 1.0 - _clip01(abs(float(idx - candidate_idx)) / max(1.0, 0.012 * sr))
        return _clip01(
            (0.30 * center_quality)
            + (0.25 * quiet_quality)
            + (0.25 * future_quality)
            + (0.15 * distance_quality)
            + candidate_bonus
        )

    candidate_radius = max(1, int(round(0.0025 * sr)))
    candidate_crossings = _zero_crossings_in_range(y, candidate_idx - candidate_radius, candidate_idx + candidate_radius + 1)
    best_idx: Optional[int] = None
    best_quality = 0.0
    for idx in candidate_crossings:
        quality = quality_for(idx, candidate_bonus=0.10)
        if quality > best_quality:
            best_idx = int(idx)
            best_quality = float(quality)
    if best_idx is not None and best_quality >= 0.55:
        return best_idx, best_quality

    crossings = _zero_crossings_in_range(y, search_start, search_end)
    for idx in crossings:
        quality = quality_for(idx)
        if quality <= 0.0:
            continue
        if best_idx is None or quality > best_quality or (quality >= best_quality - 0.04 and abs(idx - candidate_idx) < abs(best_idx - candidate_idx)):
            best_idx = int(idx)
            best_quality = float(quality)
    if best_idx is None or best_quality < 0.50:
        return None, 0.0
    return best_idx, best_quality


def _visual_onset_knee_before_impact(
    y: np.ndarray,
    curves: Mapping[str, np.ndarray],
    candidate_idx: int,
    peak_idx: int,
    attack_idx: int,
    sr: int,
) -> tuple[Optional[int], float]:
    if y.size == 0 or peak_idx <= 0:
        return None, 0.0

    env = curves["envelope"]
    abs_y = np.abs(y).astype(np.float64)
    n = len(y)
    peak_window = max(1, int(round(0.030 * sr)))
    p0 = max(0, int(peak_idx) - peak_window)
    p1 = min(n, int(peak_idx) + peak_window)
    local_peak = float(np.percentile(abs_y[p0:p1], 98.0)) if p1 > p0 else float(np.max(abs_y))
    if local_peak <= 1e-8:
        return None, 0.0

    peak_env = float(np.percentile(env[p0:p1], 95.0)) if p1 > p0 else float(env[int(peak_idx)])
    anchor_idx = int(np.clip(min(int(candidate_idx), int(attack_idx), int(peak_idx)), 0, n - 1))
    pre_start = max(0, anchor_idx - int(round(0.025 * sr)))
    pre_end = max(pre_start + 1, min(int(candidate_idx) + int(round(0.002 * sr)), int(attack_idx)))
    pre_floor = float(np.percentile(env[pre_start:pre_end], 25.0)) if pre_end > pre_start else 0.0
    pre_hi = float(np.percentile(env[pre_start:pre_end], 90.0)) if pre_end > pre_start else pre_floor

    search_start = max(1, anchor_idx - int(round(0.015 * sr)))
    search_end = min(n, max(search_start + 1, min(int(peak_idx), int(attack_idx) + int(round(0.012 * sr)))))
    future = max(1, int(round(0.006 * sr)))
    high_abs = max(pre_hi * 8.0, 0.16 * local_peak)
    impact_idx: Optional[int] = None
    for idx in range(search_start, search_end):
        if float(abs_y[idx]) < high_abs:
            continue
        f1 = min(n, idx + future)
        if f1 <= idx + 1:
            continue
        if float(np.max(abs_y[idx:f1])) < 0.35 * local_peak:
            continue
        impact_idx = int(idx)
        break

    if impact_idx is None:
        high_env = pre_floor + (0.28 * max(0.0, peak_env - pre_floor))
        for idx in range(search_start, search_end):
            if float(env[idx]) < high_env:
                continue
            f1 = min(n, idx + future)
            if f1 <= idx + 1:
                continue
            if float(np.max(abs_y[idx:f1])) < 0.35 * local_peak:
                continue
            impact_idx = int(idx)
            break

    if impact_idx is None:
        return None, 0.0

    lookback = max(1, int(round(0.0045 * sr)))
    max_gap = max(1, int(round(0.0011 * sr)))
    start = max(search_start, impact_idx - lookback)
    end = max(start + 1, impact_idx)
    crossings = _zero_crossings_in_range(y, start, end)
    if not crossings:
        crossings = list(range(max(start, impact_idx - max_gap), impact_idx))

    best_idx: Optional[int] = None
    best_quality = 0.0
    for crossing in crossings:
        idx = int(crossing)
        if idx > 0 and abs_y[idx - 1] <= abs_y[idx]:
            idx -= 1
        gap = impact_idx - idx
        if gap <= 0 or gap > max_gap:
            continue
        f1 = min(n, idx + future)
        if f1 <= idx + 1:
            continue
        future_peak = float(np.max(abs_y[idx:f1]))
        if future_peak < 0.32 * local_peak:
            continue
        pre0 = max(0, idx - int(round(0.0018 * sr)))
        pre_quiet = float(np.percentile(env[pre0:idx], 75.0)) if idx > pre0 else float(env[idx])
        center_quality = 1.0 - _clip01(abs(float(y[idx])) / max(1e-8, 0.045 * local_peak))
        quiet_quality = 1.0 - _clip01(max(0.0, pre_quiet - pre_floor) / max(1e-8, 0.16 * max(0.0, peak_env - pre_floor)))
        future_quality = _clip01(future_peak / max(1e-8, 0.50 * local_peak))
        gap_sec = float(gap) / float(sr)
        gap_quality = 1.0 - _clip01(abs(gap_sec - 0.00025) / 0.0011)
        quality = _clip01(
            (0.30 * center_quality)
            + (0.18 * quiet_quality)
            + (0.22 * future_quality)
            + (0.30 * gap_quality)
        )
        if best_idx is None or quality > best_quality or (quality >= best_quality - 0.03 and idx > best_idx):
            best_idx = int(idx)
            best_quality = float(quality)

    if best_idx is None or best_quality < 0.58:
        return None, 0.0
    return best_idx, best_quality


def _ableton_asd_anchor_before_peak(
    y: np.ndarray,
    curves: Mapping[str, np.ndarray],
    *,
    window_start: float,
    sr: int,
    candidate_idx: int,
    peak_idx: int,
    attack_idx: int,
    audio_path: str,
    asd_marker_times: Optional[Sequence[float]] = None,
) -> tuple[Optional[int], float, Optional[float]]:
    markers = tuple(float(t) for t in asd_marker_times) if asd_marker_times is not None else _load_asd_marker_seconds(str(audio_path))
    if not markers or y.size == 0 or peak_idx <= 0:
        return None, 0.0, None

    env = curves["envelope"]
    abs_y = np.abs(y).astype(np.float64)
    peak_window = max(1, int(round(0.030 * sr)))
    p0 = max(0, int(peak_idx) - peak_window)
    p1 = min(len(y), int(peak_idx) + peak_window)
    local_peak = float(np.percentile(abs_y[p0:p1], 98.0)) if p1 > p0 else float(np.max(abs_y))
    if local_peak <= 1e-8:
        return None, 0.0, None

    search_start_idx = max(1, min(int(candidate_idx), int(peak_idx) - int(round(0.045 * sr))))
    search_start_time = float(window_start) + (float(search_start_idx) / float(sr)) - 0.004
    search_end_time = float(window_start) + (float(min(int(peak_idx), int(attack_idx) + int(round(0.006 * sr)))) / float(sr))
    if search_end_time <= search_start_time:
        return None, 0.0, None

    pre_start = max(0, int(peak_idx) - int(round(0.045 * sr)))
    pre_end = min(len(y), max(pre_start + 1, int(attack_idx)))
    pre_floor = float(np.percentile(env[pre_start:pre_end], 20.0)) if pre_end > pre_start else 0.0
    peak_env = float(np.percentile(env[p0:p1], 95.0)) if p1 > p0 else float(env[int(peak_idx)])
    quiet_limit = pre_floor + (0.090 * max(0.0, peak_env - pre_floor))
    future = max(1, int(round(0.018 * sr)))

    best_idx: Optional[int] = None
    best_time: Optional[float] = None
    best_quality = 0.0
    denom = max(1.0, float(attack_idx - candidate_idx))
    for marker_time in markers:
        if marker_time < search_start_time or marker_time > search_end_time:
            continue
        idx = int(round((float(marker_time) - float(window_start)) * float(sr)))
        if idx < 0 or idx >= len(y) or idx >= peak_idx:
            continue
        if float(env[idx]) > quiet_limit:
            continue
        f1 = min(len(y), idx + future)
        if f1 <= idx + 1:
            continue
        future_peak = float(np.max(abs_y[idx:f1]))
        future_env = float(np.percentile(env[idx:f1], 80.0))
        if future_peak < 0.24 * local_peak or future_env < quiet_limit * 1.6:
            continue

        quiet_quality = 1.0 - _clip01(float(env[idx]) / max(1e-8, quiet_limit))
        future_quality = _clip01(future_peak / max(1e-8, local_peak))
        candidate_distance = abs(float(idx - candidate_idx)) / max(1.0, 0.018 * sr)
        candidate_quality = 1.0 - _clip01(candidate_distance)
        pre_attack_position = _clip01(float(idx - candidate_idx) / denom)
        pre_attack_quality = 1.0 - abs(pre_attack_position - 0.72) / 0.72
        pre_attack_quality = _clip01(pre_attack_quality)
        quality = _clip01(
            (0.32 * quiet_quality)
            + (0.28 * future_quality)
            + (0.18 * candidate_quality)
            + (0.17 * pre_attack_quality)
            + 0.05
        )
        if best_idx is None or quality > best_quality or (quality >= best_quality - 0.04 and idx > best_idx):
            best_idx = int(idx)
            best_time = float(marker_time)
            best_quality = float(quality)

    if best_idx is None or best_quality < 0.48:
        return None, 0.0, None
    return best_idx, best_quality, best_time


def _input_boundary_anchor_when_bracketed(
    *,
    candidate_idx: int,
    attack_idx: int,
    zero_idx: int,
    zero_quality: float,
    center_idx: Optional[int],
    center_quality: float,
    knee_idx: Optional[int],
    knee_quality: float,
    sr: int,
) -> tuple[Optional[int], float]:
    if center_idx is None or knee_idx is None:
        return None, 0.0
    if center_quality < 0.80 or knee_quality < 0.80:
        return None, 0.0

    candidate_idx = int(candidate_idx)
    center_idx = int(center_idx)
    knee_idx = int(knee_idx)
    low = min(center_idx, knee_idx)
    high = max(center_idx, knee_idx)
    bracket_width = high - low
    if bracket_width <= 0 or bracket_width > max(1, int(round(0.0060 * sr))):
        return None, 0.0

    bracket_slack = max(1, int(round(0.00035 * sr)))
    if candidate_idx < low - bracket_slack or candidate_idx > high + bracket_slack:
        return None, 0.0

    nearest_visual = min(abs(candidate_idx - center_idx), abs(candidate_idx - knee_idx))
    if nearest_visual > max(1, int(round(0.0022 * sr))):
        return None, 0.0

    attack_support = abs(candidate_idx - int(attack_idx)) <= max(1, int(round(0.0025 * sr)))
    zero_support = zero_quality >= 0.55 and abs(candidate_idx - int(zero_idx)) <= max(1, int(round(0.0025 * sr)))
    if not attack_support and not zero_support:
        return None, 0.0

    bracket_quality = 1.0 - _clip01(float(bracket_width) / max(1.0, 0.0060 * sr))
    proximity_quality = 1.0 - _clip01(float(nearest_visual) / max(1.0, 0.0022 * sr))
    source_quality = min(float(center_quality), float(knee_quality))
    support_quality = max(0.82 if attack_support else 0.0, float(zero_quality) if zero_support else 0.0)
    quality = _clip01(
        (0.35 * source_quality)
        + (0.24 * bracket_quality)
        + (0.21 * proximity_quality)
        + (0.20 * support_quality)
    )
    if quality < 0.72:
        return None, 0.0
    return candidate_idx, float(quality)


def _impact_body_anchor_after_tail(
    y: np.ndarray,
    curves: Mapping[str, np.ndarray],
    *,
    candidate_idx: int,
    aligned_idx: int,
    peak_idx: int,
    attack_idx: int,
    sr: int,
    selected_quality: float,
) -> tuple[Optional[int], float]:
    if y.size == 0:
        return None, 0.0

    n = len(y)
    aligned_idx = int(np.clip(aligned_idx, 0, n - 1))
    candidate_idx = int(np.clip(candidate_idx, 0, n - 1))
    peak_idx = int(np.clip(peak_idx, 0, n - 1))
    attack_idx = int(np.clip(attack_idx, 0, n - 1))
    min_shift = max(1, int(round(0.0035 * sr)))
    search_start = min(n - 1, aligned_idx + min_shift)
    if search_start >= n - 1:
        return None, 0.0

    env = curves["envelope"]
    impact_score = curves.get("impact_score", curves["score"])
    rms_rise = curves.get("rms_rise", np.zeros_like(env))
    peak_rise = curves.get("peak_rise", np.zeros_like(env))
    low_flux = curves.get("low_flux", np.zeros_like(env))
    crest_click = curves.get("crest_click", np.zeros_like(env))
    denoised_env = curves.get("denoised_envelope", env)

    search_forward = max(1, int(round(0.120 * sr)))
    body_window = max(1, int(round(0.048 * sr)))
    search_end = min(
        n,
        aligned_idx + search_forward,
        max(
            search_start + max(1, int(round(0.012 * sr))),
            peak_idx + max(1, int(round(0.012 * sr))),
            attack_idx + body_window,
            candidate_idx + max(1, int(round(0.055 * sr))),
        ),
    )
    if search_end <= search_start + 1:
        return None, 0.0

    body_curve = (
        (0.28 * impact_score)
        + (0.23 * rms_rise)
        + (0.22 * low_flux)
        + (0.10 * peak_rise)
        + (0.17 * _normalize(denoised_env, percentile=97.0))
    )
    body_curve = np.clip(body_curve, 0.0, 1.0)
    region = body_curve[search_start:search_end]
    if region.size == 0:
        return None, 0.0
    local_body_max = float(np.max(region))
    if local_body_max < 0.42:
        return None, 0.0

    pre_start = max(0, aligned_idx - int(round(0.070 * sr)))
    pre_end = max(pre_start + 1, aligned_idx)
    pre_floor = float(np.percentile(env[pre_start:pre_end], 25.0)) if pre_end > pre_start else 0.0
    pre_tail = float(np.percentile(env[pre_start:pre_end], 82.0)) if pre_end > pre_start else pre_floor
    peak_span_start = max(0, min(peak_idx, search_end - 1) - int(round(0.018 * sr)))
    peak_span_end = min(n, max(peak_idx + int(round(0.018 * sr)), search_start + 1, search_end))
    peak_env = float(np.percentile(env[peak_span_start:peak_span_end], 94.0)) if peak_span_end > peak_span_start else float(env[peak_idx])
    body_env_threshold = max(pre_tail * 1.14, pre_floor + (0.22 * max(0.0, peak_env - pre_floor)))
    sustain = max(1, int(round(0.010 * sr)))
    selected_body = float(body_curve[aligned_idx]) if 0 <= aligned_idx < len(body_curve) else 0.0
    selected_env = float(env[aligned_idx]) if 0 <= aligned_idx < len(env) else 0.0
    threshold = max(0.46, 0.58 * local_body_max)
    if selected_quality >= 0.90:
        threshold = max(threshold, min(0.86 * local_body_max, selected_body + 0.10))

    def refine_body_entry(trigger_idx: int) -> int:
        refine_end = min(n, max(trigger_idx + 1, trigger_idx + int(round(0.032 * sr))))
        if refine_end <= trigger_idx + 1:
            return int(trigger_idx)
        local_peak_env = float(np.percentile(env[trigger_idx:refine_end], 92.0))
        hard_env_threshold = max(
            body_env_threshold,
            pre_tail * 1.04,
            pre_floor + (0.30 * max(0.0, local_peak_env - pre_floor)),
        )
        hold = max(1, int(round(0.0045 * sr)))
        for body_idx in range(trigger_idx, refine_end):
            if float(env[body_idx]) < hard_env_threshold:
                continue
            hold_end = min(n, body_idx + hold)
            if hold_end <= body_idx + 1:
                continue
            if float(np.percentile(env[body_idx:hold_end], 60.0)) < hard_env_threshold * 0.78:
                continue
            return int(body_idx)
        return int(trigger_idx)

    best_idx: Optional[int] = None
    best_quality = 0.0
    for idx in range(search_start, search_end):
        trigger_body_value = float(body_curve[idx])
        if trigger_body_value < threshold:
            continue
        if float(crest_click[idx]) > 0.76 and float(rms_rise[idx]) < 0.34:
            continue
        body_idx = refine_body_entry(int(idx))
        if body_idx <= aligned_idx:
            continue
        if float(crest_click[body_idx]) > 0.76 and float(rms_rise[body_idx]) < 0.34:
            continue
        body_value = max(trigger_body_value, float(body_curve[body_idx]))
        e1 = min(n, body_idx + sustain)
        if e1 <= body_idx + 1:
            continue
        future_env = float(np.percentile(env[body_idx:e1], 70.0))
        future_body = float(np.mean(body_curve[body_idx:e1]))
        if future_env < body_env_threshold and future_body < max(0.35, body_value * 0.54):
            continue

        distance_improvement = _clip01(
            (abs(float(aligned_idx - candidate_idx)) - abs(float(body_idx - candidate_idx)))
            / max(1.0, 0.055 * sr)
        )
        env_gain = _clip01((future_env - selected_env) / max(1e-6, peak_env - pre_floor))
        body_gain = _clip01((body_value - selected_body) / max(1e-6, local_body_max))
        low_body = _clip01(
            max(
                (0.55 * float(low_flux[idx])) + (0.45 * float(rms_rise[idx])),
                (0.55 * float(low_flux[body_idx])) + (0.45 * float(rms_rise[body_idx])),
            )
        )
        source_margin = 1.0 - _clip01(float(selected_quality) - 0.76)
        quality = _clip01(
            (0.30 * (body_value / max(1e-6, local_body_max)))
            + (0.22 * env_gain)
            + (0.20 * low_body)
            + (0.15 * body_gain)
            + (0.08 * distance_improvement)
            + (0.05 * source_margin)
        )
        if quality < 0.58:
            continue
        if quality >= 0.62:
            return int(body_idx), float(quality)
        if best_idx is None or quality > best_quality or (quality >= best_quality - 0.04 and body_idx < best_idx):
            best_idx = int(body_idx)
            best_quality = float(quality)
            if quality >= 0.74:
                break

    if best_idx is None:
        return None, 0.0

    shift_sec = float(best_idx - aligned_idx) / float(sr)
    if shift_sec < 0.0035 or shift_sec > 0.120:
        return None, 0.0
    if abs(float(best_idx - candidate_idx)) > abs(float(aligned_idx - candidate_idx)) + (0.030 * sr):
        return None, 0.0
    return best_idx, best_quality


def microalign_marker(
    audio_path: str,
    candidate_time: float,
    search_before_ms: float = 250,
    search_after_ms: float = 500,
    prefer_zero_crossing: bool = True,
    asd_marker_times: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    before_sec = max(0.001, float(search_before_ms) * 0.001)
    after_sec = max(0.001, float(search_after_ms) * 0.001)
    candidate = float(candidate_time)
    y_raw, sr, window_start = _load_window(audio_path, candidate, before_sec, after_sec)
    y = _percussive_window(y_raw, audio_path)
    curves = _transient_curves(y, sr)
    candidate_idx = int(np.clip(round((candidate - window_start) * sr), 0, len(y) - 1))
    peaks = _candidate_peak_indices(curves["score"], curves["envelope"], sr)

    best_idx = candidate_idx
    best_score = -float("inf")
    best_features: Dict[str, float] = {
        "attack_peak_strength": 0.0,
        "attack_cleanliness": 0.0,
        "sustained_after_attack": 0.0,
        "slope_score": 0.0,
    }
    for peak_idx in peaks:
        score, features = _score_peak(int(peak_idx), curves, candidate_idx, sr)
        if score > best_score:
            best_idx = int(peak_idx)
            best_score = float(score)
            best_features = features

    attack_idx = _find_attack_start(best_idx, curves, sr)
    zero_idx, zero_quality = _zero_crossing_before(y_raw, attack_idx, sr)
    knee_idx, knee_quality = _visual_onset_knee_before_impact(y_raw, curves, candidate_idx, best_idx, attack_idx, sr)
    center_idx, center_quality = _centerline_departure_before_peak(y_raw, curves, candidate_idx, best_idx, attack_idx, sr)
    asd_idx, asd_quality, asd_time = _ableton_asd_anchor_before_peak(
        y_raw,
        curves,
        window_start=window_start,
        sr=sr,
        candidate_idx=candidate_idx,
        peak_idx=best_idx,
        attack_idx=attack_idx,
        audio_path=audio_path,
        asd_marker_times=asd_marker_times,
    )
    knee_eligible = bool(knee_idx is not None and knee_quality >= 0.58)
    asd_eligible = bool(asd_idx is not None and asd_quality >= 0.52)
    centerline_eligible = bool(center_idx is not None and center_quality >= 0.55)
    selected_micro_source = "zero" if prefer_zero_crossing and zero_quality >= 0.35 else "attack"
    input_boundary_idx, input_boundary_quality = _input_boundary_anchor_when_bracketed(
        candidate_idx=candidate_idx,
        attack_idx=attack_idx,
        zero_idx=zero_idx,
        zero_quality=zero_quality,
        center_idx=center_idx,
        center_quality=center_quality,
        knee_idx=knee_idx,
        knee_quality=knee_quality,
        sr=sr,
    )
    input_boundary_eligible = bool(input_boundary_idx is not None)
    asd_matches_knee = bool(
        knee_eligible
        and asd_eligible
        and knee_idx is not None
        and asd_idx is not None
        and abs(int(asd_idx) - int(knee_idx)) <= max(1, int(round(0.00025 * sr)))
    )
    centerline_matches_knee = bool(
        centerline_eligible
        and knee_eligible
        and center_idx is not None
        and knee_idx is not None
        and abs(int(center_idx) - int(knee_idx)) <= max(1, int(round(0.006 * sr)))
    )
    if asd_matches_knee:
        aligned_idx = int(asd_idx)
        selected_micro_source = "asd"
    elif input_boundary_eligible:
        aligned_idx = int(input_boundary_idx)
        selected_micro_source = "input_boundary"
    elif centerline_matches_knee and center_quality >= knee_quality - 0.08:
        aligned_idx = int(center_idx)
        selected_micro_source = "centerline"
    elif knee_eligible:
        aligned_idx = int(knee_idx)
        selected_micro_source = "knee"
    elif asd_eligible:
        aligned_idx = int(asd_idx)
        selected_micro_source = "asd"
    elif centerline_eligible:
        aligned_idx = int(center_idx)
        selected_micro_source = "centerline"
    else:
        aligned_idx = zero_idx if prefer_zero_crossing and zero_quality >= 0.35 else attack_idx

    selected_quality = 0.0
    if selected_micro_source == "knee":
        selected_quality = float(knee_quality)
    elif selected_micro_source == "asd":
        selected_quality = float(asd_quality)
    elif selected_micro_source == "centerline":
        selected_quality = float(center_quality)
    elif selected_micro_source == "input_boundary":
        selected_quality = float(input_boundary_quality)
    elif selected_micro_source == "zero":
        selected_quality = float(zero_quality)
    impact_body_idx, impact_body_quality = _impact_body_anchor_after_tail(
        y_raw,
        curves,
        candidate_idx=candidate_idx,
        aligned_idx=int(aligned_idx),
        peak_idx=best_idx,
        attack_idx=attack_idx,
        sr=sr,
        selected_quality=selected_quality,
    )
    tail_bypass_ms = 0.0
    high_quality_start = bool(
        selected_micro_source in {"knee", "asd", "centerline", "input_boundary"}
        and selected_quality >= 0.74
    )
    allow_tail_bypass = bool(
        impact_body_idx is not None
        and impact_body_quality >= 0.66
        and (
            not high_quality_start
            or (
                selected_quality < 0.82
                and impact_body_quality >= selected_quality + 0.08
            )
        )
    )
    if allow_tail_bypass:
        tail_bypass_ms = (float(impact_body_idx - int(aligned_idx)) / float(sr)) * 1000.0
        aligned_idx = int(impact_body_idx)
        selected_micro_source = "impact_body"
    knee_used = selected_micro_source == "knee"
    asd_used = selected_micro_source == "asd"
    centerline_used = selected_micro_source == "centerline"
    input_boundary_used = selected_micro_source == "input_boundary"
    impact_body_used = selected_micro_source == "impact_body"

    peak_time = window_start + (float(best_idx) / float(sr))
    attack_time = window_start + (float(attack_idx) / float(sr))
    zero_time = window_start + (float(zero_idx) / float(sr))
    aligned_time = window_start + (float(aligned_idx) / float(sr))
    knee_time = None if knee_idx is None else window_start + (float(knee_idx) / float(sr))
    impact_body_time = None if impact_body_idx is None else window_start + (float(impact_body_idx) / float(sr))
    offset_ms = (aligned_time - candidate) * 1000.0

    offset_quality = 1.0 - _clip01(abs(offset_ms) / 180.0)
    impact_score_curve = curves.get("impact_score", curves["score"])
    impact_contrast_curve = curves.get("impact_contrast", impact_score_curve)
    rms_rise_curve = curves.get("rms_rise", np.zeros_like(impact_score_curve))
    peak_rise_curve = curves.get("peak_rise", np.zeros_like(impact_score_curve))
    crest_click_curve = curves.get("crest_click", np.zeros_like(impact_score_curve))
    aligned_impact_strength = (
        float(impact_score_curve[aligned_idx]) if 0 <= int(aligned_idx) < len(impact_score_curve) else 0.0
    )
    aligned_impact_contrast = (
        float(impact_contrast_curve[aligned_idx]) if 0 <= int(aligned_idx) < len(impact_contrast_curve) else 0.0
    )
    rms_end = min(len(rms_rise_curve), int(aligned_idx) + max(1, int(round(0.014 * sr))))
    peak_end = min(len(peak_rise_curve), int(aligned_idx) + max(1, int(round(0.085 * sr))))
    aligned_rms_rise = (
        float(np.max(rms_rise_curve[int(aligned_idx) : rms_end]))
        if 0 <= int(aligned_idx) < len(rms_rise_curve) and rms_end > int(aligned_idx)
        else 0.0
    )
    aligned_peak_rise = (
        float(np.max(peak_rise_curve[int(aligned_idx) : peak_end]))
        if 0 <= int(aligned_idx) < len(peak_rise_curve) and peak_end > int(aligned_idx)
        else 0.0
    )
    aligned_crest_click = (
        float(np.max(crest_click_curve[int(aligned_idx) : peak_end]))
        if 0 <= int(aligned_idx) < len(crest_click_curve) and peak_end > int(aligned_idx)
        else 0.0
    )
    impact_strength = _clip01(max(aligned_impact_strength, 0.72 * best_features.get("denoised_impact_strength", 0.0)))
    impact_contrast = _clip01(max(aligned_impact_contrast, 0.72 * best_features.get("impact_contrast", 0.0)))
    rms_rise_score = _clip01(max(aligned_rms_rise, 0.72 * best_features.get("rms_rise_score", 0.0)))
    peak_rise_score = _clip01(max(aligned_peak_rise, 0.72 * best_features.get("peak_rise_score", 0.0)))
    crest_click_score = _clip01(max(aligned_crest_click, 0.72 * best_features.get("crest_click_score", 0.0)))
    legacy_micro_confidence = _clip01(
        (0.25 * best_features["attack_peak_strength"])
        + (0.25 * best_features["attack_cleanliness"])
        + (0.25 * best_features["sustained_after_attack"])
        + (0.15 * zero_quality)
        + (0.10 * offset_quality)
    )
    impact_boundary_confidence = _clip01(
        (0.31 * impact_contrast)
        + (0.27 * impact_strength)
        + (0.18 * zero_quality)
        + (0.12 * offset_quality)
        + (0.05 * best_features["attack_peak_strength"])
        + (0.03 * best_features["attack_cleanliness"])
        + (0.04 * best_features["sustained_after_attack"])
    )
    boundary_quality = 0.0
    if knee_used:
        boundary_quality = max(boundary_quality, float(knee_quality))
    if asd_used:
        boundary_quality = max(boundary_quality, float(asd_quality))
    if centerline_used:
        boundary_quality = max(boundary_quality, float(center_quality))
    if input_boundary_used:
        boundary_quality = max(boundary_quality, float(input_boundary_quality))
    if impact_body_used:
        boundary_quality = max(boundary_quality, float(impact_body_quality))
    if boundary_quality > 0.0:
        boundary_confidence = _clip01(
            (0.30 * best_features["attack_cleanliness"])
            + (0.22 * impact_strength)
            + (0.22 * boundary_quality)
            + (0.10 * zero_quality)
            + (0.10 * offset_quality)
            + (0.06 * best_features["attack_peak_strength"])
        )
        impact_boundary_confidence = max(impact_boundary_confidence, boundary_confidence)
    if crest_click_score > 0.65 and rms_rise_score < 0.30:
        impact_boundary_confidence *= 0.76
    micro_confidence = legacy_micro_confidence
    if abs(offset_ms) > 250.0:
        micro_confidence *= 0.55
        impact_boundary_confidence *= 0.55

    if impact_body_used and micro_confidence >= 0.75 and abs(offset_ms) <= 120.0:
        reason = "tail bypass to sustained drum/bass impact body"
    elif knee_used and micro_confidence >= 0.75 and abs(offset_ms) <= 120.0:
        reason = "sample-level centerline knee immediately before sustained attack"
    elif asd_used and micro_confidence >= 0.75 and abs(offset_ms) <= 120.0:
        reason = "Ableton ASD transient marker at quiet pre-attack boundary"
    elif centerline_used and micro_confidence >= 0.75 and abs(offset_ms) <= 120.0:
        reason = "quiet center-line departure before sustained attack"
    elif input_boundary_used and micro_confidence >= 0.75 and abs(offset_ms) <= 120.0:
        reason = "input boundary kept inside tight visual onset bracket"
    elif micro_confidence >= 0.80 and abs(offset_ms) <= 120.0:
        reason = "strong clean attack with sustained post-attack energy"
    elif abs(offset_ms) > 120.0:
        reason = "large microalignment offset; review recommended"
    elif micro_confidence < 0.55:
        reason = "weak or ambiguous transient boundary; review recommended"
    else:
        reason = "plausible transient boundary; review if musical confidence is low"

    return {
        "input_candidate_time": float(candidate),
        "microaligned_time": float(aligned_time),
        "attack_start_time": float(attack_time),
        "peak_time": float(peak_time),
        "zero_crossing_time": float(zero_time),
        "centerline_boundary_time": None if center_idx is None else float(window_start + (float(center_idx) / float(sr))),
        "centerline_boundary_quality": float(center_quality),
        "centerline_boundary_used": bool(centerline_used),
        "visual_onset_knee_time": 0.0 if knee_time is None else float(knee_time),
        "visual_onset_knee_offset_ms": 0.0 if knee_time is None else float((float(knee_time) - candidate) * 1000.0),
        "visual_onset_knee_quality": float(knee_quality),
        "visual_onset_knee_used": float(1.0 if knee_used else 0.0),
        "input_boundary_quality": float(input_boundary_quality),
        "input_boundary_used": float(1.0 if input_boundary_used else 0.0),
        "ableton_asd_time": 0.0 if asd_time is None else float(asd_time),
        "ableton_asd_offset_ms": 0.0 if asd_time is None else float((float(asd_time) - candidate) * 1000.0),
        "ableton_asd_quality": float(asd_quality),
        "ableton_asd_used": float(1.0 if asd_used else 0.0),
        "impact_body_time": 0.0 if impact_body_time is None else float(impact_body_time),
        "impact_body_offset_ms": 0.0 if impact_body_time is None else float((float(impact_body_time) - candidate) * 1000.0),
        "impact_body_quality": float(impact_body_quality),
        "impact_body_used": float(1.0 if impact_body_used else 0.0),
        "tail_bypass_ms": float(tail_bypass_ms),
        "snap_offset_ms": float(offset_ms),
        "micro_confidence": float(micro_confidence),
        "reason": reason,
        "attack_peak_strength": float(best_features["attack_peak_strength"]),
        "attack_cleanliness": float(best_features["attack_cleanliness"]),
        "sustained_after_attack": float(best_features["sustained_after_attack"]),
        "denoised_impact_strength": float(impact_strength),
        "impact_contrast": float(impact_contrast),
        "impact_boundary_confidence": float(impact_boundary_confidence),
        "rms_rise_score": float(rms_rise_score),
        "peak_rise_score": float(peak_rise_score),
        "crest_click_score": float(crest_click_score),
        "peak_denoised_impact_strength": float(best_features.get("denoised_impact_strength", 0.0)),
        "peak_impact_contrast": float(best_features.get("impact_contrast", 0.0)),
        "local_noise_floor": float(best_features.get("local_noise_floor", 0.0)),
        "zero_crossing_quality": float(zero_quality),
        "review_needed": bool(micro_confidence < 0.80 or abs(offset_ms) > 120.0),
    }


def candidate_timestamp(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "snapped_sec", "time_sec", "coarse_timestamp"):
        value = candidate.get(key)
        if value is None:
            continue
        out = _finite_float(value, default=float("nan"))
        if math.isfinite(out):
            return out
    return None


def _lock_clock_microalign(candidate: Mapping[str, Any], micro: Mapping[str, Any], timestamp: float) -> Dict[str, Any]:
    clock = candidate.get("bpm_clock")
    source_names = candidate.get("multistem_source_names")
    has_clock_source = isinstance(source_names, Sequence) and not isinstance(source_names, (str, bytes)) and any(
        str(name).startswith("bpm_clock") for name in source_names
    )
    if not isinstance(clock, Mapping) and not has_clock_source:
        return dict(micro)
    if isinstance(clock, Mapping) and clock.get("on_one") is False:
        return dict(micro)

    locked = dict(micro)
    lock_time = float(timestamp)
    if isinstance(clock, Mapping):
        clock_time = _finite_float(clock.get("nearest_one_time"), default=float("nan"))
        if math.isfinite(clock_time) and abs(clock_time - float(timestamp)) <= 0.080:
            lock_time = float(clock_time)
    snapped = _finite_float(locked.get("microaligned_time"), default=float(timestamp))
    if abs(float(snapped) - float(lock_time)) <= 0.040:
        return locked

    visual_sources = (
        ("input_boundary_used", "input_boundary_quality"),
        ("centerline_boundary_used", "centerline_boundary_quality"),
        ("visual_onset_knee_used", "visual_onset_knee_quality"),
        ("ableton_asd_used", "ableton_asd_quality"),
        ("impact_body_used", "impact_body_quality"),
    )
    visual_quality = 0.0
    for used_key, quality_key in visual_sources:
        used = micro.get(used_key)
        is_used = bool(used is True or _finite_float(used, default=0.0) > 0.0)
        if is_used:
            visual_quality = max(visual_quality, _finite_float(micro.get(quality_key), default=0.0))
    impact_confidence = _finite_float(micro.get("impact_boundary_confidence"), default=0.0)
    visual_offset = abs(float(snapped) - float(lock_time))
    if visual_quality >= 0.72 and impact_confidence >= 0.66 and visual_offset <= 0.160:
        locked["clock_lock_skipped"] = True
        locked["clock_lock_skip_reason"] = "kept strong visual waveform onset instead of BPM-clock grid"
        locked["clock_lock_candidate_time"] = float(lock_time)
        locked["clock_lock_visual_offset_ms"] = float((float(snapped) - float(lock_time)) * 1000.0)
        locked["micro_confidence"] = max(0.82, _finite_float(locked.get("micro_confidence"), default=0.0))
        return locked

    original_offset = _finite_float(locked.get("snap_offset_ms"), default=(snapped - float(timestamp)) * 1000.0)
    locked["clock_locked"] = True
    locked["clock_lock_reason"] = "kept BPM-clock bar candidate on musical ONE"
    locked["clock_locked_from_time"] = float(snapped)
    locked["clock_locked_original_snap_offset_ms"] = float(original_offset)
    locked["microaligned_time"] = float(lock_time)
    locked["snap_offset_ms"] = float((float(lock_time) - float(timestamp)) * 1000.0)
    locked["micro_confidence"] = max(0.82, _finite_float(locked.get("micro_confidence"), default=0.0) * 0.92)
    return locked


def microalign_candidate_dicts(audio_path: str, candidates: Sequence[Mapping[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for candidate in list(candidates)[: max(0, int(limit))]:
        item = dict(candidate)
        existing_micro = item.get("microalign")
        if isinstance(existing_micro, Mapping) and existing_micro.get("microaligned_time") is not None and existing_micro.get("ok", True):
            micro = dict(existing_micro)
            micro["ok"] = True
            timestamp = candidate_timestamp(item)
            if timestamp is not None:
                micro = _lock_clock_microalign(item, micro, float(timestamp))
            item["microalign"] = micro
            for key in MICROALIGN_FEATURE_KEYS:
                if key in micro:
                    item[key] = _finite_float(micro.get(key))
            out.append(item)
            continue
        timestamp = candidate_timestamp(item)
        if timestamp is None:
            item["microalign"] = {"ok": False, "error": "missing candidate timestamp"}
            out.append(item)
            continue
        try:
            micro = microalign_marker(audio_path, timestamp)
            micro["ok"] = True
        except Exception as exc:
            micro = {"ok": False, "error": str(exc) or exc.__class__.__name__, "input_candidate_time": float(timestamp)}
        micro = _lock_clock_microalign(item, micro, float(timestamp))
        item["microalign"] = micro
        for key in MICROALIGN_FEATURE_KEYS:
            item[key] = _finite_float(micro.get(key))
        out.append(item)
    return out


def _candidate_margin(candidates: Sequence[Mapping[str, Any]]) -> float:
    scores = sorted(
        [_finite_float(candidate.get("confidence_score", candidate.get("score"))) for candidate in candidates],
        reverse=True,
    )
    if len(scores) < 2:
        return 1.0 if scores else 0.0
    return max(0.0, float(scores[0] - scores[1]))


def _rank_agreement(candidate: Mapping[str, Any]) -> bool:
    handcrafted = int(_finite_float(candidate.get("handcrafted_rank", candidate.get("rank")), 0.0))
    model = int(_finite_float(candidate.get("model_rank"), 0.0))
    rank = int(_finite_float(candidate.get("rank"), 0.0))
    return handcrafted == 1 and (model in {0, 1}) and rank <= 2


def _drum_metric(candidate: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        nested = candidate.get("drumprint")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    return _finite_float(value)


def _groove_metric(candidate: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        nested = candidate.get("full_groove")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    return _finite_float(value)


def _has_groove_metrics(candidate: Mapping[str, Any]) -> bool:
    if any(key in candidate and candidate[key] not in (None, "") for key in ("sustained_full_groove_score", "immediate_groove_start_score")):
        return True
    nested = candidate.get("full_groove")
    return isinstance(nested, Mapping) and any(
        nested.get(key) not in (None, "")
        for key in ("sustained_full_groove_score", "immediate_groove_start_score")
    )


def _micro_metric(candidate: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        nested = candidate.get("microalign")
        if isinstance(nested, Mapping):
            value = nested.get(key)
    return _finite_float(value)


def _candidate_margin_for_gate(candidate: Mapping[str, Any], candidates: Optional[Sequence[Mapping[str, Any]]]) -> Optional[float]:
    for key in ("candidate_margin", "confidence_rank_gap", "score_margin"):
        if key in candidate and candidate[key] is not None:
            return _finite_float(candidate[key])
    if candidates:
        return _candidate_margin(candidates)
    return None


def _model_disagreement(candidate: Mapping[str, Any]) -> bool:
    handcrafted = int(_finite_float(candidate.get("handcrafted_rank", candidate.get("rank")), 0.0))
    model = int(_finite_float(candidate.get("model_rank"), 0.0))
    if model <= 0:
        return False
    if handcrafted <= 1 and model > 3:
        return True
    if model <= 1 and handcrafted > 4:
        return True
    return abs(model - handcrafted) >= 4


@lru_cache(maxsize=1)
def _load_auto_gate_config() -> Dict[str, Any]:
    path = DEFAULT_AUTO_GATE_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _attach_auto_verifier(
    selected_candidate: Dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_time: Optional[float],
) -> None:
    if not DEFAULT_AUTO_VERIFIER_PATH.exists():
        return
    try:
        payload = load_auto_verifier_payload(str(DEFAULT_AUTO_VERIFIER_PATH))
        if payload is None:
            return
        prediction = predict_auto_verifier(payload, candidates, selected_time=selected_time)
    except Exception:
        return
    if not isinstance(prediction, Mapping):
        return
    selected_candidate["auto_verifier_p_within_25ms"] = float(prediction.get("p_within_25ms", 0.0) or 0.0)
    selected_candidate["auto_verifier_predicted_abs_error_sec"] = float(
        prediction.get("predicted_abs_error_sec", 999.0) or 999.0
    )
    selected_candidate["auto_verifier_model_path"] = str(prediction.get("model_path", ""))
    selected_candidate["auto_verifier_model_type"] = str(prediction.get("model_type", ""))


def _best_auto_verifier_candidate(candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not DEFAULT_AUTO_VERIFIER_PATH.exists():
        return None
    try:
        payload = load_auto_verifier_payload(str(DEFAULT_AUTO_VERIFIER_PATH))
        if payload is None:
            return None
        predictions = predict_auto_verifier_candidates(payload, candidates)
    except Exception:
        return None
    if not predictions:
        return None
    ranked = sorted(
        predictions,
        key=lambda row: (
            -float(row.get("p_within_25ms", 0.0)),
            float(row.get("predicted_abs_error_sec", 999.0)),
            int(row.get("index", 999999)),
        ),
    )
    best = ranked[0]
    candidate = best.get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    out = dict(candidate)
    out["auto_verifier_p_within_25ms"] = float(best.get("p_within_25ms", 0.0) or 0.0)
    out["auto_verifier_predicted_abs_error_sec"] = float(best.get("predicted_abs_error_sec", 999.0) or 999.0)
    out["auto_verifier_model_path"] = str(best.get("model_path", ""))
    out["auto_verifier_model_type"] = str(best.get("model_type", ""))
    out["selected_by"] = "auto_verifier"
    return out


def _auto_verifier_gate(
    candidate: Mapping[str, Any],
    mode: str,
    *,
    candidate_margin: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    probability_value = candidate.get("auto_verifier_p_within_25ms")
    predicted_error_value = candidate.get("auto_verifier_predicted_abs_error_sec")
    if probability_value is None or predicted_error_value is None:
        return None
    gate_config = _load_auto_gate_config()
    mode_key = {"conservative": "safe", "normal": "balanced", "aggressive": "aggressive"}.get(str(mode), "balanced")
    mode_config = gate_config.get(mode_key) if isinstance(gate_config.get(mode_key), Mapping) else {}
    thresholds = mode_config.get("thresholds") if isinstance(mode_config.get("thresholds"), Mapping) else {}
    if not thresholds:
        return None

    probability = _clip01(_finite_float(probability_value))
    predicted_error_ms = max(0.0, _finite_float(predicted_error_value, 999.0) * 1000.0)
    micro_conf = _micro_metric(candidate, "micro_confidence")
    offset = abs(_micro_metric(candidate, "snap_offset_ms"))
    fake = _drum_metric(candidate, "fake_hit_penalty")
    margin = candidate_margin
    if margin is None:
        margin = _finite_float(candidate.get("candidate_margin"), _finite_float(candidate.get("probability_margin"), 0.0))

    min_probability = _finite_float(thresholds.get("min_p_within_25ms"), 0.99)
    max_predicted_error_ms = _finite_float(thresholds.get("max_predicted_error_ms"), 8.0)
    min_micro = _finite_float(thresholds.get("min_micro_confidence"), 0.0)
    max_offset_ms = _finite_float(thresholds.get("max_snap_offset_ms"), 220.0)
    max_fake = _finite_float(thresholds.get("max_fake_hit_penalty"), 1.0)
    min_margin = _finite_float(thresholds.get("min_candidate_margin"), 0.0)

    risk_flags: List[str] = []
    if probability < min_probability:
        risk_flags.append(f"verifier probability {probability:.3f} below {min_probability:.3f}")
    if predicted_error_ms > max_predicted_error_ms:
        risk_flags.append(f"verifier predicted error {predicted_error_ms:.1f}ms above {max_predicted_error_ms:.0f}ms")
    if micro_conf < min_micro:
        risk_flags.append(f"verifier micro confidence {micro_conf:.3f} below {min_micro:.2f}")
    if offset > max_offset_ms:
        risk_flags.append(f"verifier snap offset {offset:.1f}ms above {max_offset_ms:.0f}ms")
    if fake > max_fake:
        risk_flags.append(f"verifier fake-hit penalty {fake:.3f} above {max_fake:.2f}")
    if margin is not None and float(margin) < min_margin:
        risk_flags.append(f"verifier candidate margin {float(margin):.3f} below {min_margin:.2f}")

    return {
        "auto_accept": bool(not risk_flags),
        "mode": mode_key,
        "risk_flags": risk_flags,
        "p_within_25ms": float(probability),
        "predicted_abs_error_sec": float(predicted_error_ms / 1000.0),
        "thresholds": dict(thresholds),
    }


def _learned_candidate_gate(
    candidate: Mapping[str, Any],
    mode: str,
    *,
    tier: str,
    model_disagree: bool,
) -> Optional[Dict[str, Any]]:
    if str(candidate.get("selected_by") or "") != "candidate_chooser":
        return None
    probability_value = candidate.get("candidate_chooser_probability")
    if probability_value is None:
        probability_value = candidate.get("selection_probability")
    if probability_value is None:
        return None

    probability = _clip01(_finite_float(probability_value))
    confidence = _clip01(_finite_float(candidate.get("candidate_chooser_confidence")))
    probability_margin = _clip01(
        _finite_float(candidate.get("candidate_chooser_probability_margin", candidate.get("probability_margin")))
    )
    predicted_error = max(0.0, _finite_float(candidate.get("candidate_chooser_predicted_abs_error_sec"), 999.0))
    micro_conf = _micro_metric(candidate, "micro_confidence")
    offset = abs(_micro_metric(candidate, "snap_offset_ms"))
    fake = _drum_metric(candidate, "fake_hit_penalty")
    drum_score = _drum_metric(candidate, "drumprint_pattern_score")
    groove_score = _groove_metric(candidate, "sustained_full_groove_score")
    immediate_groove = _groove_metric(candidate, "immediate_groove_start_score")

    learned_thresholds = {
        "conservative": {
            "probability": 0.94,
            "confidence": 0.92,
            "margin": 0.20,
            "predicted_error": 0.020,
            "micro": 0.96,
            "offset": 20.0,
            "fake": 0.25,
            "disagree_probability": 1.01,
        },
        "normal": {
            "probability": 0.85,
            "confidence": 0.82,
            "margin": 0.10,
            "predicted_error": 0.035,
            "micro": 0.93,
            "offset": 45.0,
            "fake": 0.35,
            "disagree_probability": 0.96,
        },
        "aggressive": {
            "probability": 0.80,
            "confidence": 0.75,
            "margin": 0.07,
            "predicted_error": 0.070,
            "micro": 0.88,
            "offset": 90.0,
            "fake": 0.60,
            "disagree_probability": 0.94,
        },
    }[mode]
    override_map = candidate.get("candidate_chooser_auto_gate")
    if isinstance(override_map, Mapping):
        mode_override = override_map.get(mode)
        if isinstance(mode_override, Mapping):
            for key in ("probability", "confidence", "margin", "predicted_error", "micro", "offset", "fake", "disagree_probability"):
                if key in mode_override and mode_override.get(key) is not None:
                    learned_thresholds[key] = _finite_float(mode_override.get(key), learned_thresholds[key])

    risk_flags: List[str] = []
    if tier not in {"HIGH", "MEDIUM", "LOW"}:
        risk_flags.append(f"confidence tier unknown: {tier}")
    if probability < learned_thresholds["probability"]:
        risk_flags.append(f"learned probability {probability:.3f} below {learned_thresholds['probability']:.2f}")
    if confidence < learned_thresholds["confidence"]:
        risk_flags.append(f"learned confidence {confidence:.3f} below {learned_thresholds['confidence']:.2f}")
    if probability_margin < learned_thresholds["margin"]:
        risk_flags.append(f"learned margin {probability_margin:.3f} below {learned_thresholds['margin']:.2f}")
    if predicted_error > learned_thresholds["predicted_error"]:
        risk_flags.append(f"predicted error {predicted_error * 1000.0:.1f}ms above {learned_thresholds['predicted_error'] * 1000.0:.0f}ms")
    if micro_conf < learned_thresholds["micro"]:
        risk_flags.append(f"micro confidence {micro_conf:.3f} below {learned_thresholds['micro']:.2f}")
    if offset > learned_thresholds["offset"]:
        risk_flags.append(f"snap offset {offset:.1f}ms above {learned_thresholds['offset']:.0f}ms")
    if fake > learned_thresholds["fake"]:
        risk_flags.append(f"fake-hit penalty {fake:.3f} above {learned_thresholds['fake']:.2f}")
    if model_disagree and probability < learned_thresholds["disagree_probability"]:
        risk_flags.append("model and handcrafted ranks strongly disagree")

    verifier_gate = _auto_verifier_gate(candidate, mode)
    if verifier_gate is not None and not bool(verifier_gate.get("auto_accept")):
        risk_flags.extend(str(flag) for flag in verifier_gate.get("risk_flags", []))

    auto_accept = len(risk_flags) == 0
    reason = "learned auto-accept gate passed" if auto_accept else "review recommended: " + "; ".join(risk_flags[:3])
    return {
        "auto_accept": bool(auto_accept),
        "reason": reason,
        "risk_flags": risk_flags,
        "mode": mode,
        "confidence_tier": tier,
        "micro_confidence": float(micro_conf),
        "snap_offset_ms": float(_micro_metric(candidate, "snap_offset_ms")),
        "fake_hit_penalty": float(fake),
        "drumprint_pattern_score": float(drum_score),
        "sustained_full_groove_score": float(groove_score),
        "immediate_groove_start_score": float(immediate_groove),
        "candidate_chooser_probability": float(probability),
        "candidate_chooser_probability_margin": float(probability_margin),
        "candidate_chooser_confidence": float(confidence),
        "candidate_chooser_predicted_abs_error_sec": float(predicted_error),
        "auto_verifier": dict(verifier_gate) if isinstance(verifier_gate, Mapping) else None,
        "learned_gate": True,
    }


def _auto_verifier_selected_gate(
    candidate: Mapping[str, Any],
    mode: str,
    *,
    tier: str,
    candidate_margin: Optional[float],
) -> Optional[Dict[str, Any]]:
    if str(candidate.get("selected_by") or "") != "auto_verifier":
        return None
    verifier_gate = _auto_verifier_gate(candidate, mode, candidate_margin=candidate_margin)
    if verifier_gate is None:
        return None
    risk_flags = [str(flag) for flag in verifier_gate.get("risk_flags", [])]
    if tier not in {"HIGH", "MEDIUM", "LOW"}:
        risk_flags.append(f"confidence tier unknown: {tier}")
    auto_accept = len(risk_flags) == 0
    return {
        "auto_accept": bool(auto_accept),
        "reason": "auto-verifier gate passed" if auto_accept else "review recommended: " + "; ".join(risk_flags[:3]),
        "risk_flags": risk_flags,
        "mode": mode,
        "confidence_tier": tier,
        "micro_confidence": float(_micro_metric(candidate, "micro_confidence")),
        "snap_offset_ms": float(_micro_metric(candidate, "snap_offset_ms")),
        "candidate_margin": None if candidate_margin is None else float(candidate_margin),
        "fake_hit_penalty": float(_drum_metric(candidate, "fake_hit_penalty")),
        "auto_verifier": dict(verifier_gate),
        "auto_verifier_gate": True,
    }


def should_auto_accept(
    candidate: Mapping[str, Any],
    mode: str = "conservative",
    *,
    candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    confidence_tier: Optional[str] = None,
) -> Dict[str, Any]:
    mode = str(mode or "conservative").strip().lower()
    if mode not in {"conservative", "normal", "aggressive"}:
        mode = "conservative"

    tier = str(confidence_tier or candidate.get("confidence_tier") or "UNKNOWN").strip().upper()
    micro_conf = _micro_metric(candidate, "micro_confidence")
    offset = abs(_micro_metric(candidate, "snap_offset_ms"))
    fake = _drum_metric(candidate, "fake_hit_penalty")
    drum_score = _drum_metric(candidate, "drumprint_pattern_score")
    groove_score = _groove_metric(candidate, "sustained_full_groove_score")
    immediate_groove = _groove_metric(candidate, "immediate_groove_start_score")
    has_groove = _has_groove_metrics(candidate)
    margin = _candidate_margin_for_gate(candidate, candidates)
    agreement = _rank_agreement(candidate)
    model_disagree = _model_disagreement(candidate)
    risk_flags: List[str] = []

    verifier_selected_gate = _auto_verifier_selected_gate(candidate, mode, tier=tier, candidate_margin=margin)
    if verifier_selected_gate is not None:
        return verifier_selected_gate

    learned_gate = _learned_candidate_gate(candidate, mode, tier=tier, model_disagree=model_disagree)
    if learned_gate is not None:
        return learned_gate

    if mode == "conservative":
        thresholds = {
            "micro": 0.88,
            "offset": 60.0,
            "fake": 0.25,
            "drum": 0.35,
            "groove": 0.45,
            "immediate_groove": 0.45,
            "margin": 0.06,
        }
        tier_ok = tier == "HIGH"
    elif mode == "normal":
        thresholds = {
            "micro": 0.82,
            "offset": 90.0,
            "fake": 0.35,
            "drum": 0.45,
            "groove": 0.38,
            "immediate_groove": 0.35,
            "margin": 0.045,
        }
        tier_ok = tier == "HIGH" or (tier == "MEDIUM" and (agreement or (margin is not None and margin >= thresholds["margin"])))
    else:
        thresholds = {
            "micro": 0.90 if tier == "LOW" else 0.76,
            "offset": 120.0,
            "fake": 0.15 if tier == "LOW" else 0.45,
            "drum": 0.70 if tier == "LOW" else 0.40,
            "groove": 0.68 if tier == "LOW" else 0.30,
            "immediate_groove": 0.70 if tier == "LOW" else 0.30,
            "margin": 0.10 if tier == "LOW" else 0.035,
        }
        tier_ok = tier in {"HIGH", "MEDIUM"} or tier == "LOW"

    if not tier_ok:
        risk_flags.append(f"confidence tier not allowed: {tier}")
    if micro_conf < thresholds["micro"]:
        risk_flags.append(f"micro confidence {micro_conf:.3f} below {thresholds['micro']:.2f}")
    if offset > thresholds["offset"]:
        risk_flags.append(f"snap offset {offset:.1f}ms above {thresholds['offset']:.0f}ms")
    if fake > thresholds["fake"]:
        risk_flags.append(f"fake-hit penalty {fake:.3f} above {thresholds['fake']:.2f}")
    if drum_score < thresholds["drum"]:
        risk_flags.append(f"DrumPrint score {drum_score:.3f} below {thresholds['drum']:.2f}")
    if has_groove and groove_score < thresholds["groove"]:
        risk_flags.append(f"full-groove score {groove_score:.3f} below {thresholds['groove']:.2f}")
    if has_groove and immediate_groove < thresholds["immediate_groove"]:
        risk_flags.append(f"immediate groove {immediate_groove:.3f} below {thresholds['immediate_groove']:.2f}")
    if margin is None:
        risk_flags.append("candidate score margin unknown")
    elif margin < thresholds["margin"]:
        risk_flags.append(f"candidate score margin {margin:.3f} below {thresholds['margin']:.3f}")
    if model_disagree:
        risk_flags.append("model and handcrafted ranks strongly disagree")
    verifier_gate = _auto_verifier_gate(candidate, mode, candidate_margin=margin)
    if verifier_gate is not None and not bool(verifier_gate.get("auto_accept")):
        risk_flags.extend(str(flag) for flag in verifier_gate.get("risk_flags", []))

    auto_accept = len(risk_flags) == 0
    if auto_accept:
        reason = f"{mode} auto-accept gate passed"
    elif mode == "aggressive" and tier == "LOW":
        reason = "do not auto-accept LOW track unless all aggressive gates pass"
    else:
        reason = "review recommended: " + "; ".join(risk_flags[:3])

    return {
        "auto_accept": bool(auto_accept),
        "reason": reason,
        "risk_flags": risk_flags,
        "mode": mode,
        "confidence_tier": tier,
        "micro_confidence": float(micro_conf),
        "snap_offset_ms": float(_micro_metric(candidate, "snap_offset_ms")),
        "candidate_margin": None if margin is None else float(margin),
        "fake_hit_penalty": float(fake),
        "drumprint_pattern_score": float(drum_score),
        "sustained_full_groove_score": float(groove_score),
        "immediate_groove_start_score": float(immediate_groove),
        "auto_verifier": dict(verifier_gate) if isinstance(verifier_gate, Mapping) else None,
    }


def choose_microaligned_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    confidence_tier: str = "UNKNOWN",
    mode: str = "normal",
    chooser_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    mode = str(mode or "normal").strip().lower()
    if mode not in {"conservative", "normal", "aggressive"}:
        mode = "normal"
    tier = str(confidence_tier or "UNKNOWN").strip().upper()
    margin = _candidate_margin(candidates)

    try:
        learned = choose_learned_candidate(candidates, model_path=chooser_model_path)
    except Exception:
        learned = None
    verifier_candidate = _best_auto_verifier_candidate(candidates)
    if learned is not None:
        candidate = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
        micro = candidate.get("microalign") if isinstance(candidate, Mapping) else None
        if isinstance(candidate, Mapping) and isinstance(micro, Mapping) and micro.get("microaligned_time") is not None:
            micro_conf = _finite_float(micro.get("micro_confidence"))
            offset = abs(_finite_float(micro.get("snap_offset_ms")))
            fake = _drum_metric(candidate, "fake_hit_penalty")
            if micro_conf >= 0.50 and offset <= 260.0 and fake <= 0.90:
                selected_candidate = dict(candidate)
                selected_candidate["candidate_chooser_predicted_abs_error_sec"] = float(learned.get("predicted_abs_error_sec", 0.0))
                selected_candidate["candidate_chooser_score"] = float(learned.get("chooser_score", 0.0))
                selected_candidate["candidate_chooser_margin_sec"] = float(learned.get("prediction_margin_sec", 0.0))
                selected_candidate["candidate_chooser_confidence"] = float(learned.get("selection_confidence", 0.0))
                selected_candidate["candidate_chooser_model_type"] = str(learned.get("model_type", ""))
                if "selection_probability" in learned:
                    selected_candidate["candidate_chooser_probability"] = float(learned.get("selection_probability", 0.0))
                    selected_candidate["candidate_chooser_probability_margin"] = float(learned.get("probability_margin", 0.0))
                selected_candidate["candidate_chooser_model_path"] = str(learned.get("model_path", ""))
                selected_candidate["candidate_chooser_selector_rows"] = int(learned.get("selector_correction_rows", 0) or 0)
                selected_candidate["candidate_chooser_bad_candidate_sets"] = int(learned.get("bad_candidate_sets", 0) or 0)
                if isinstance(learned.get("auto_gate_thresholds"), Mapping):
                    selected_candidate["candidate_chooser_auto_gate"] = dict(learned.get("auto_gate_thresholds") or {})
                selected_candidate["selected_by"] = "candidate_chooser"
                suggested_time = float(micro.get("microaligned_time"))
                _attach_auto_verifier(selected_candidate, candidates, selected_time=suggested_time)
                selected_p = _clip01(_finite_float(selected_candidate.get("auto_verifier_p_within_25ms")))
                verifier_p = _clip01(_finite_float((verifier_candidate or {}).get("auto_verifier_p_within_25ms")))
                verifier_error = _finite_float((verifier_candidate or {}).get("auto_verifier_predicted_abs_error_sec"), 999.0)
                verifier_micro = (verifier_candidate or {}).get("microalign") if isinstance(verifier_candidate, Mapping) else None
                if (
                    isinstance(verifier_candidate, Mapping)
                    and isinstance(verifier_micro, Mapping)
                    and verifier_micro.get("microaligned_time") is not None
                    and verifier_p >= 0.55
                    and verifier_error <= 0.050
                    and (verifier_p >= selected_p + 0.12 or selected_p < 0.45)
                ):
                    selected_candidate = dict(verifier_candidate)
                    micro = dict(verifier_micro)
                    suggested_time = float(micro.get("microaligned_time"))
                auto_accept = {
                    gate_mode: should_auto_accept(selected_candidate, gate_mode, candidates=candidates, confidence_tier=tier)
                    for gate_mode in ("conservative", "normal", "aggressive")
                }
                predicted_ms = float(learned.get("predicted_abs_error_sec", 0.0)) * 1000.0
                if str(selected_candidate.get("selected_by") or "") == "auto_verifier":
                    reason = (
                        "auto verifier selected MicroSnap marker "
                        f"(P<=25ms {float(selected_candidate.get('auto_verifier_p_within_25ms', 0.0)):.3f})"
                    )
                else:
                    reason = (
                        f"learned candidate chooser selected MicroSnap marker "
                        f"(predicted error {predicted_ms:.1f} ms)"
                    )
                return {
                    "auto_place": bool((auto_accept.get("normal") or {}).get("auto_accept")),
                    "review_needed": not bool((auto_accept.get("normal") or {}).get("auto_accept")),
                    "risk": bool(tier == "LOW" or not bool((auto_accept.get("normal") or {}).get("auto_accept"))),
                    "mode": mode,
                    "confidence_tier": tier,
                    "candidate": selected_candidate,
                    "microalign": dict(micro),
                    "suggested_time": suggested_time,
                    "score": float(learned.get("chooser_score", 0.0)),
                    "reason": reason,
                    "auto_accept": auto_accept,
                    "candidate_chooser": {
                        "predicted_abs_error_sec": float(learned.get("predicted_abs_error_sec", 0.0)),
                        "prediction_margin_sec": float(learned.get("prediction_margin_sec", 0.0)),
                        "selection_probability": float(learned.get("selection_probability", learned.get("chooser_score", 0.0))),
                        "probability_margin": float(learned.get("probability_margin", 0.0)),
                        "selection_confidence": float(learned.get("selection_confidence", 0.0)),
                        "model_type": str(learned.get("model_type", "")),
                        "model_path": str(learned.get("model_path", "")),
                        "training_rows": int(learned.get("training_rows", 0) or 0),
                        "correction_rows": int(learned.get("correction_rows", 0) or 0),
                        "selector_correction_rows": int(learned.get("selector_correction_rows", 0) or 0),
                        "bad_candidate_sets": int(learned.get("bad_candidate_sets", 0) or 0),
                        "auto_gate_thresholds": dict(learned.get("auto_gate_thresholds") or {})
                        if isinstance(learned.get("auto_gate_thresholds"), Mapping)
                        else {},
                        "auto_verifier": {
                            "p_within_25ms": selected_candidate.get("auto_verifier_p_within_25ms"),
                            "predicted_abs_error_sec": selected_candidate.get("auto_verifier_predicted_abs_error_sec"),
                            "model_path": selected_candidate.get("auto_verifier_model_path"),
                            "model_type": selected_candidate.get("auto_verifier_model_type"),
                        },
                    },
                }

    verifier_micro = (verifier_candidate or {}).get("microalign") if isinstance(verifier_candidate, Mapping) else None
    verifier_p = _clip01(_finite_float((verifier_candidate or {}).get("auto_verifier_p_within_25ms")))
    verifier_error = _finite_float((verifier_candidate or {}).get("auto_verifier_predicted_abs_error_sec"), 999.0)
    if (
        isinstance(verifier_candidate, Mapping)
        and isinstance(verifier_micro, Mapping)
        and verifier_micro.get("microaligned_time") is not None
        and verifier_p >= 0.55
        and verifier_error <= 0.050
    ):
        selected_candidate = dict(verifier_candidate)
        suggested_time = float(verifier_micro.get("microaligned_time"))
        auto_accept = {
            gate_mode: should_auto_accept(selected_candidate, gate_mode, candidates=candidates, confidence_tier=tier)
            for gate_mode in ("conservative", "normal", "aggressive")
        }
        return {
            "auto_place": bool((auto_accept.get("normal") or {}).get("auto_accept")),
            "review_needed": not bool((auto_accept.get("normal") or {}).get("auto_accept")),
            "risk": bool(tier == "LOW" or not bool((auto_accept.get("normal") or {}).get("auto_accept"))),
            "mode": mode,
            "confidence_tier": tier,
            "candidate": selected_candidate,
            "microalign": dict(verifier_micro),
            "suggested_time": suggested_time,
            "score": float(verifier_p),
            "reason": f"auto verifier selected MicroSnap marker (P<=25ms {verifier_p:.3f})",
            "auto_accept": auto_accept,
            "candidate_chooser": {
                "auto_verifier": {
                    "p_within_25ms": selected_candidate.get("auto_verifier_p_within_25ms"),
                    "predicted_abs_error_sec": selected_candidate.get("auto_verifier_predicted_abs_error_sec"),
                    "model_path": selected_candidate.get("auto_verifier_model_path"),
                    "model_type": selected_candidate.get("auto_verifier_model_type"),
                }
            },
        }

    thresholds = {
        "conservative": {"micro": 0.80, "offset": 120.0, "fake": 0.35},
        "normal": {"micro": 0.74, "offset": 150.0, "fake": 0.45},
        "aggressive": {"micro": 0.65, "offset": 220.0, "fake": 0.65},
    }[mode]

    eligible: List[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    fallback: List[tuple[float, Mapping[str, Any], Mapping[str, Any], List[str]]] = []
    rejected: List[str] = []
    for candidate in candidates:
        micro = candidate.get("microalign")
        if not isinstance(micro, Mapping) or not micro.get("ok", True):
            rejected.append("microalignment failed")
            continue
        micro_conf = _finite_float(micro.get("micro_confidence"))
        offset = abs(_finite_float(micro.get("snap_offset_ms")))
        fake = _drum_metric(candidate, "fake_hit_penalty")
        drum_score = _drum_metric(candidate, "drumprint_pattern_score")
        groove_score = _groove_metric(candidate, "sustained_full_groove_score")
        immediate_groove = _groove_metric(candidate, "immediate_groove_start_score")
        base_score = _finite_float(candidate.get("confidence_score", candidate.get("score")))
        agreement = _rank_agreement(candidate)

        if mode == "conservative":
            tier_ok = tier == "HIGH" or (tier == "MEDIUM" and (agreement or margin >= 0.08))
        elif mode == "normal":
            tier_ok = tier in {"HIGH", "MEDIUM"} or (tier == "LOW" and agreement and drum_score >= 0.55)
        else:
            tier_ok = True

        score = (
            (0.35 * micro_conf)
            + (0.22 * base_score)
            + (0.16 * _clip01(drum_score))
            + (0.10 * _clip01(groove_score))
            + (0.06 * _clip01(immediate_groove))
            + (0.07 * (1.0 if agreement else 0.0))
            + (0.04 * _clip01(margin / 0.12))
            - (0.10 * _clip01(fake))
            - (0.08 * _clip01(offset / thresholds["offset"]))
        )
        fallback_score = (
            (0.50 * micro_conf)
            + (0.30 * base_score)
            + (0.10 * _clip01(drum_score))
            + (0.05 * _clip01(groove_score))
            + (0.05 * _clip01(immediate_groove))
            - (0.10 * _clip01(fake))
            - (0.06 * _clip01(offset / max(1.0, thresholds["offset"])))
        )
        risk_reasons: List[str] = []
        if not tier_ok:
            risk_reasons.append(f"tier gate failed ({tier})")
        if micro_conf < thresholds["micro"]:
            risk_reasons.append(f"micro confidence below {thresholds['micro']:.2f}")
        if offset > thresholds["offset"]:
            risk_reasons.append(f"snap offset above {thresholds['offset']:.0f}ms")
        if fake > thresholds["fake"]:
            risk_reasons.append("fake-hit penalty too high")

        # The review UI still needs a useful suggestion on LOW-confidence tracks.
        # Keep these candidates out of auto-placement, but return the best one as
        # a human-approved suggestion if MicroSnap found a plausible boundary.
        if micro_conf >= 0.60 and offset <= max(220.0, thresholds["offset"]) and fake <= 0.70:
            fallback.append((float(fallback_score), candidate, micro, risk_reasons))

        if risk_reasons:
            rejected.extend(risk_reasons)
            continue

        eligible.append((float(score), candidate, micro))

    if not eligible:
        if fallback:
            fallback.sort(key=lambda row: (-row[0], _finite_float(row[1].get("rank"), 999.0)))
            score, candidate, micro, risk_reasons = fallback[0]
            selected_candidate = dict(candidate)
            suggested_time = float(micro.get("microaligned_time"))
            _attach_auto_verifier(selected_candidate, candidates, selected_time=suggested_time)
            auto_accept = {
                gate_mode: should_auto_accept(selected_candidate, gate_mode, candidates=candidates, confidence_tier=tier)
                for gate_mode in ("conservative", "normal", "aggressive")
            }
            reason_bits = risk_reasons or ["auto-placement gate failed"]
            return {
                "auto_place": False,
                "review_needed": True,
                "risk": True,
                "reason": "review recommended: " + "; ".join(reason_bits[:3]),
                "mode": mode,
                "confidence_tier": tier,
                "candidate": selected_candidate,
                "microalign": dict(micro),
                "suggested_time": suggested_time,
                "score": float(score),
                "auto_accept": auto_accept,
            }
        reason = "review recommended: " + (rejected[0] if rejected else "no eligible microaligned candidate")
        return {
            "auto_place": False,
            "review_needed": True,
            "reason": reason,
            "mode": mode,
            "confidence_tier": tier,
            "candidate": None,
            "microalign": None,
        }

    eligible.sort(key=lambda row: (-row[0], _finite_float(row[1].get("rank"), 999.0)))
    score, candidate, micro = eligible[0]
    risky = bool(mode == "aggressive" or tier == "LOW")
    selected_candidate = dict(candidate)
    suggested_time = float(micro.get("microaligned_time"))
    _attach_auto_verifier(selected_candidate, candidates, selected_time=suggested_time)
    auto_accept = {
        gate_mode: should_auto_accept(selected_candidate, gate_mode, candidates=candidates, confidence_tier=tier)
        for gate_mode in ("conservative", "normal", "aggressive")
    }
    return {
        "auto_place": True,
        "review_needed": risky,
        "risk": risky,
        "mode": mode,
        "confidence_tier": tier,
        "candidate": selected_candidate,
        "microalign": dict(micro),
        "suggested_time": suggested_time,
        "score": float(score),
        "reason": str(micro.get("reason") or "microaligned candidate selected"),
        "auto_accept": auto_accept,
    }
