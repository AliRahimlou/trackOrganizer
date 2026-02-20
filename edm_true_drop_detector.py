#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
True drop detector for EDM DJ prep.

Pipeline:
1) Decode WAV/FLAC/AIFF/MP3 with ffmpeg
2) Tempo + beat grid estimation (120-160 BPM focus)
3) Bar segmentation (4/4)
4) Per-bar features: RMS, low band, high band, flux, onset, kick periodicity
5) Buildup detection + drop candidate scoring
6) True-drop confirmation with sustained energy and fake-drop rejection
7) Phrase-aligned beat refinement + transient anchor selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import math
import subprocess

import numpy as np


@dataclass
class DropDetectionResult:
    drop_time_sec: float
    bar_index: int
    bar_number: int
    beat_index: int
    bpm: float
    confidence: float
    ableton_cue_beat_time: float
    debug: Dict[str, float]


@dataclass
class DetectorConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_sec: float = 0.05
    feature_smooth_bars: int = 2
    bpm_min: float = 120.0
    bpm_max: float = 160.0
    min_bars_after_drop: int = 8
    min_bars_before_drop: int = 4
    peak_search_beats: int = 4
    lock_bpm_to_hint: bool = True
    local_refine_back_beats: int = 1
    local_refine_fwd_beats: int = 3
    drop_zone_radius_bars: int = 4
    structural_backtrack_bars: int = 2
    structural_confirm_bars: int = 4
    refine_downbeat_with_transient: bool = False
    max_first_drop_ratio: float = 0.55


def _decode_audio_ffmpeg(path: str, sr: int) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except Exception:
        return None
    if not proc.stdout:
        return None
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=False)
    if y.size == 0:
        return None
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    w = int(max(1, win))
    if w % 2 == 0:
        w += 1
    kern = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kern, mode="same").astype(np.float32, copy=False)


def _robust_norm(x: np.ndarray, q_lo: float = 20.0, q_hi: float = 80.0) -> np.ndarray:
    lo = float(np.percentile(x, q_lo))
    hi = float(np.percentile(x, q_hi))
    span = hi - lo
    if span <= 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _stft_features(y: np.ndarray, sr: int, n_fft: int, hop: int):
    if len(y) < n_fft + hop:
        return None

    n_frames = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    low_mask = (freqs >= 20.0) & (freqs <= 120.0)
    high_mask = (freqs >= 2000.0) & (freqs <= min(10000.0, float(sr) * 0.5))
    broad_mask = (freqs >= 20.0) & (freqs <= min(10000.0, float(sr) * 0.5))

    rms = np.empty(n_frames, dtype=np.float32)
    low_e = np.empty(n_frames, dtype=np.float32)
    high_e = np.empty(n_frames, dtype=np.float32)
    broad_e = np.empty(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)

    prev_mag = None
    prev_rms = 0.0
    for i in range(n_frames):
        frame = y[i * hop:i * hop + n_fft]
        rms_i = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        rms[i] = rms_i
        onset[i] = max(0.0, rms_i - prev_rms)
        prev_rms = rms_i

        mag = np.abs(np.fft.rfft(frame * win)).astype(np.float32, copy=False)
        pwr = mag * mag
        low_i = float(np.sum(pwr[low_mask])) if np.any(low_mask) else 0.0
        high_i = float(np.sum(pwr[high_mask])) if np.any(high_mask) else 0.0
        broad_i = float(np.sum(pwr[broad_mask])) if np.any(broad_mask) else float(np.sum(pwr))
        low_e[i] = low_i
        high_e[i] = high_i
        broad_e[i] = broad_i
        if prev_mag is not None:
            d = mag - prev_mag
            flux[i] = float(np.sum(d[d > 0.0]))
        prev_mag = mag

    low_ratio = low_e / (broad_e + 1e-12)
    high_ratio = high_e / (broad_e + 1e-12)
    low_transient = np.maximum(0.0, np.diff(low_ratio, prepend=low_ratio[0])).astype(np.float32, copy=False)
    high_flux = np.maximum(0.0, np.diff(high_ratio, prepend=high_ratio[0])).astype(np.float32, copy=False)

    return {
        "rms": rms,
        "low_energy": low_e.astype(np.float32, copy=False),
        "high_energy": high_e.astype(np.float32, copy=False),
        "broad_energy": broad_e.astype(np.float32, copy=False),
        "low_ratio": low_ratio.astype(np.float32, copy=False),
        "high_ratio": high_ratio.astype(np.float32, copy=False),
        "flux": flux,
        "onset": onset.astype(np.float32, copy=False),
        "low_transient": low_transient,
        "high_flux": high_flux,
    }


def _estimate_tempo_bpm(onset_env: np.ndarray, hop_sec: float, bpm_min: float, bpm_max: float, bpm_hint: Optional[float]) -> float:
    x = onset_env.astype(np.float32, copy=False)
    x = x - float(np.mean(x))
    if len(x) < 128:
        return float(np.clip(bpm_hint or 140.0, bpm_min, bpm_max))

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lag_min = max(1, int(round((60.0 / bpm_max) / hop_sec)))
    lag_max = max(lag_min + 1, int(round((60.0 / bpm_min) / hop_sec)))
    lag_max = min(lag_max, len(ac) - 1)

    best_lag = lag_min
    best_score = -1.0
    for lag in range(lag_min, lag_max + 1):
        sc = float(ac[lag])
        half = lag // 2
        dbl = lag * 2
        if half >= lag_min:
            sc += 0.35 * float(ac[half])
        if dbl <= lag_max:
            sc += 0.15 * float(ac[dbl])

        bpm = 60.0 / (lag * hop_sec)
        if bpm_hint is not None:
            sc += max(0.0, 1.0 - abs(bpm - bpm_hint) / 12.0) * 0.05 * float(ac[lag_min])

        if sc > best_score:
            best_score = sc
            best_lag = lag

    bpm = 60.0 / (best_lag * hop_sec)
    return float(np.clip(bpm, bpm_min, bpm_max))


def _estimate_beat_phase(onset_env: np.ndarray, beat_period_frames: int) -> int:
    best_phase = 0
    best_score = -1.0
    for phase in range(max(1, beat_period_frames)):
        vals = onset_env[phase::beat_period_frames]
        if len(vals) == 0:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_phase = phase
    return int(best_phase)


def _build_beats(duration_sec: float, beat_sec: float, beat_phase_sec: float) -> np.ndarray:
    if beat_sec <= 1e-9 or duration_sec <= 0.0:
        return np.array([], dtype=np.float32)
    start = max(0.0, float(beat_phase_sec))
    if start >= duration_sec:
        return np.array([], dtype=np.float32)
    n = int(math.ceil((duration_sec - start) / beat_sec))
    out = start + (np.arange(n, dtype=np.float32) * float(beat_sec))
    return out[out < duration_sec].astype(np.float32, copy=False)


def _choose_downbeat_offset(beat_times: np.ndarray, low_transient: np.ndarray, onset: np.ndarray, hop_sec: float) -> int:
    if len(beat_times) < 8:
        return 0
    scores = []
    for off in range(4):
        vals = []
        for t in beat_times[off::4]:
            i = int(round(float(t) / hop_sec))
            i = min(max(1, i), len(onset) - 2)
            vals.append(0.65 * float(low_transient[i]) + 0.35 * float(onset[i]))
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return int(np.argmax(np.asarray(scores, dtype=np.float32)))


def _bar_downbeats(beat_times: np.ndarray, downbeat_off: int) -> np.ndarray:
    db = beat_times[downbeat_off::4]
    return db if len(db) >= 2 else np.array([], dtype=np.float32)


def _aggregate_bar_feature(frame_values: np.ndarray, hop_sec: float, downbeats: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(downbeats) - 1):
        a = int(round(float(downbeats[i]) / hop_sec))
        b = int(round(float(downbeats[i + 1]) / hop_sec))
        a = min(max(0, a), len(frame_values))
        b = min(max(a + 1, b), len(frame_values))
        out.append(float(np.mean(frame_values[a:b])))
    return np.asarray(out, dtype=np.float32)


def _count_peaks_bar(frame_values: np.ndarray, hop_sec: float, downbeats: np.ndarray, q: float = 80.0) -> np.ndarray:
    thr = float(np.percentile(frame_values, q))
    peaks = np.zeros_like(frame_values, dtype=np.float32)
    for i in range(1, len(frame_values) - 1):
        v = float(frame_values[i])
        if v >= thr and v >= float(frame_values[i - 1]) and v >= float(frame_values[i + 1]):
            peaks[i] = 1.0
    return _aggregate_bar_feature(peaks, hop_sec, downbeats)


def _kick_periodicity_per_bar(low_transient: np.ndarray, hop_sec: float, downbeats: np.ndarray, beat_sec: float) -> np.ndarray:
    beat_hop = max(1, int(round(beat_sec / hop_sec)))
    tol = max(1, int(round(0.12 * beat_hop)))
    per = []
    for i in range(len(downbeats) - 1):
        t0 = float(downbeats[i])
        amps = []
        for k in range(4):
            t = t0 + k * beat_sec
            ci = int(round(t / hop_sec))
            a = max(1, ci - tol)
            b = min(len(low_transient) - 2, ci + tol)
            amps.append(float(np.max(low_transient[a:b + 1])) if b > a else float(low_transient[min(max(ci, 0), len(low_transient) - 1)]))
        arr = np.asarray(amps, dtype=np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        periodic = mean / (std + 1e-6)
        per.append(periodic)
    return np.asarray(per, dtype=np.float32)


def _sample_near_beat(x: np.ndarray, beat_times: np.ndarray, beat_idx: int, hop_sec: float, tol_frames: int = 2) -> float:
    if beat_idx < 0 or beat_idx >= len(beat_times):
        return 0.0
    ci = int(round(float(beat_times[beat_idx]) / hop_sec))
    a = max(0, ci - tol_frames)
    b = min(len(x) - 1, ci + tol_frames)
    if b < a:
        return 0.0
    return float(np.max(x[a:b + 1]))


def _mean_between_times(x: np.ndarray, hop_sec: float, t0: float, t1: float) -> float:
    i0 = max(0, int(round(t0 / hop_sec)))
    i1 = min(len(x), int(round(t1 / hop_sec)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(x[i0:i1]))


def _refine_local_downbeat(
    drop_time_sec: float,
    beat_times: np.ndarray,
    beat_sec: float,
    hop_sec: float,
    rms: np.ndarray,
    low_ratio: np.ndarray,
    onset: np.ndarray,
    low_transient: np.ndarray,
    back_beats: int = 1,
    fwd_beats: int = 3,
) -> float:
    if len(beat_times) < 40:
        return drop_time_sec

    k = int(np.argmin(np.abs(beat_times - float(drop_time_sec))))
    best_score = -1e9
    best_t = float(drop_time_sec)
    rms_floor = float(np.percentile(rms, 15.0))
    for shift in range(-int(max(0, back_beats)), int(max(0, fwd_beats)) + 1):
        b = k + shift
        if b < 8 or (b + 32) >= len(beat_times):
            continue
        t = float(beat_times[b])
        pre_t0 = float(beat_times[b - 8])      # 2 bars pre
        pre_t1 = t
        post_t1 = float(beat_times[b + 32])    # 8 bars post

        pre_r = _mean_between_times(rms, hop_sec, pre_t0, pre_t1)
        post_r = _mean_between_times(rms, hop_sec, t, post_t1)
        pre_l = _mean_between_times(low_ratio, hop_sec, pre_t0, pre_t1)
        post_l = _mean_between_times(low_ratio, hop_sec, t, post_t1)

        impact = _sample_near_beat(low_transient, beat_times, b, hop_sec, tol_frames=2)
        impact_on = _sample_near_beat(onset, beat_times, b, hop_sec, tol_frames=2)
        nxt = _sample_near_beat(low_transient, beat_times, b + 1, hop_sec, tol_frames=2)
        nxt2 = _sample_near_beat(low_transient, beat_times, b + 2, hop_sec, tol_frames=2)
        accent = (impact + 0.65 * impact_on) - (0.35 * nxt + 0.20 * nxt2)

        seq = [_sample_near_beat(low_transient, beat_times, b + j, hop_sec, tol_frames=2) for j in range(16)]
        seq = np.asarray(seq, dtype=np.float32)
        per = float(np.mean(seq) / (float(np.std(seq)) + 1e-6)) if len(seq) else 0.0

        onset_seq = [_sample_near_beat(onset, beat_times, b + j, hop_sec, tol_frames=1) for j in range(8)]
        onset_seq = np.asarray(onset_seq, dtype=np.float32)
        onset_cons = float(np.mean(onset_seq)) if len(onset_seq) else 0.0

        early_r_min = _mean_between_times(rms, hop_sec, t, float(beat_times[b + 8]))
        no_dip = 1.0 if early_r_min > max(rms_floor * 1.25, pre_r * 0.70) else 0.0

        score = (
            0.34 * (post_r - pre_r) +
            0.30 * (post_l - pre_l) +
            0.18 * accent +
            0.10 * per +
            0.06 * onset_cons +
            0.02 * no_dip
        )
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def _snap_to_first_structural_kick_bar(
    zone_center_bar: int,
    bar_rms: np.ndarray,
    bar_low: np.ndarray,
    bar_kick_period: np.ndarray,
    cfg: DetectorConfig,
) -> int:
    n_bars = int(len(bar_rms))
    if n_bars <= (cfg.min_bars_before_drop + cfg.structural_confirm_bars + 1):
        return int(max(0, min(zone_center_bar, n_bars - 1)))

    n_rms = _robust_norm(bar_rms, 15.0, 90.0)
    n_low = _robust_norm(bar_low, 15.0, 90.0)

    zone_start = max(cfg.min_bars_before_drop, zone_center_bar - cfg.drop_zone_radius_bars)
    zone_end = min(n_bars - cfg.structural_confirm_bars - 1, zone_center_bar + cfg.drop_zone_radius_bars)
    if zone_end < zone_start:
        return int(max(0, min(zone_center_bar, n_bars - 1)))

    search_start = max(zone_start, zone_center_bar - cfg.structural_backtrack_bars)
    search_end = zone_end
    confirm = max(4, int(cfg.structural_confirm_bars))

    kick_floor = max(0.40, float(np.percentile(bar_kick_period, 35.0)))
    noise_floor = float(np.percentile(bar_rms, 15.0))

    # Step 2 (requested): first bar near the spike that starts sustained kick+sub phrase.
    for i in range(search_start, search_end + 1):
        prev = slice(i - 4, i)
        post = slice(i, i + confirm)
        if i - 4 < 0 or (i + confirm) > n_bars:
            continue

        prev_r = float(np.mean(bar_rms[prev]))
        prev_l = float(np.mean(n_low[prev]))
        prev_k = float(np.mean(bar_kick_period[prev]))

        post_r = float(np.mean(bar_rms[post]))
        post_l = float(np.mean(n_low[post]))
        post_k = float(np.mean(bar_kick_period[post]))

        post2_min = float(np.min(bar_rms[i:i + 2]))
        no_dip = post2_min >= max(prev_r * 1.05, noise_floor * 1.15)
        elevated_4 = (post_r - prev_r) >= max(0.06 * post_r, 0.010)
        low_dom_4 = (post_l - prev_l) >= 0.09 and post_l >= max(prev_l + 0.07, prev_l * 1.20)
        kick_stable = (
            post_k >= kick_floor and
            float(np.std(bar_kick_period[post]) / (post_k + 1e-6)) <= 0.30 and
            post_k >= (prev_k * 0.85)
        )
        if no_dip and elevated_4 and low_dom_4 and kick_stable:
            return int(i)

    # Fallback: best structural bar in the zone, but still choose earliest near-best.
    cands: List[Tuple[float, int]] = []
    for i in range(zone_start, zone_end + 1):
        if i - 4 < 0 or (i + confirm) > n_bars:
            continue
        prev = slice(i - 4, i)
        post = slice(i, i + confirm)

        prev_r = float(np.mean(bar_rms[prev]))
        prev_l = float(np.mean(n_low[prev]))
        prev_k = float(np.mean(bar_kick_period[prev]))
        post_r = float(np.mean(bar_rms[post]))
        post_l = float(np.mean(n_low[post]))
        post_k = float(np.mean(bar_kick_period[post]))

        post2_min = float(np.min(bar_rms[i:i + 2]))
        no_dip = post2_min >= max(prev_r * 1.03, noise_floor * 1.10)
        kick_cons = float(np.std(bar_kick_period[post]) / (post_k + 1e-6))
        score = (
            0.42 * (post_l - prev_l) +
            0.34 * (post_r - prev_r) +
            0.18 * (post_k - prev_k) +
            0.06 * (1.0 - min(1.0, kick_cons)) +
            (0.03 if no_dip else -0.03)
        )
        cands.append((score, i))

    if not cands:
        return int(max(0, min(zone_center_bar, n_bars - 1)))

    cands.sort(key=lambda x: x[0], reverse=True)
    best_score = float(cands[0][0])
    near = sorted([x for x in cands if x[0] >= (best_score - 0.04)], key=lambda x: x[1])
    return int(near[0][1] if near else cands[0][1])


def _refine_structural_downbeat_time(
    downbeat_sec: float,
    beat_sec: float,
    hop_sec: float,
    low_transient: np.ndarray,
    onset: np.ndarray,
) -> float:
    if len(low_transient) < 6 or len(onset) < 6:
        return float(downbeat_sec)

    dur = float(len(low_transient) * hop_sec)
    left = max(0.0, float(downbeat_sec) - 0.45 * beat_sec)
    right = min(dur, float(downbeat_sec) + 0.35 * beat_sec)
    i0 = int(round(left / hop_sec))
    i1 = int(round(right / hop_sec))
    if i1 <= i0 + 2:
        return float(downbeat_sec)

    lt = low_transient[i0:i1 + 1].astype(np.float32, copy=False)
    on = onset[i0:i1 + 1].astype(np.float32, copy=False)
    n_lt = _robust_norm(lt, 25.0, 90.0)
    n_on = _robust_norm(on, 25.0, 90.0)
    comp = (0.62 * n_lt) + (0.38 * n_on)
    thr = max(0.34, float(np.percentile(comp, 72.0)))

    best_idx = None
    best_score = -1.0
    for j in range(1, len(comp) - 1):
        if comp[j] < thr:
            continue
        if comp[j] < comp[j - 1] or comp[j] < comp[j + 1]:
            continue
        # Prefer earliest strong peak so we land at first kick of the sustained bar.
        if best_idx is None:
            best_idx = j
            best_score = float(comp[j])
            break
        if float(comp[j]) > best_score:
            best_idx = j
            best_score = float(comp[j])

    if best_idx is None:
        best_idx = int(np.argmax(comp))

    refined = float((i0 + best_idx) * hop_sec)
    if abs(refined - float(downbeat_sec)) > (0.50 * beat_sec):
        return float(downbeat_sec)
    return refined


def _trend_score(series: np.ndarray, i: int, lookback: int = 4) -> float:
    if i < lookback:
        return 0.0
    y = series[i - lookback:i].astype(np.float32, copy=False)
    x = np.arange(len(y), dtype=np.float32)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    den = float(np.sum((x - xm) ** 2))
    if den <= 1e-12:
        return 0.0
    slope = float(np.sum((x - xm) * (y - ym)) / den)
    return max(0.0, slope)


def _smooth_bars(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) <= 1:
        return x.astype(np.float32, copy=False)
    w = int(max(1, win))
    kern = np.ones(w, dtype=np.float32) / float(w)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    xp = np.pad(x.astype(np.float32, copy=False), (pad_l, pad_r), mode="edge")
    y = np.convolve(xp, kern, mode="valid")
    return y.astype(np.float32, copy=False)


def _evaluate_drop_for_downbeats(
    downbeats: np.ndarray,
    feats: Dict[str, np.ndarray],
    cfg: DetectorConfig,
    beat_sec: float,
    downbeat_off: int,
    duration_sec: float,
) -> Optional[Dict[str, float]]:
    # Bars are [downbeats[i], downbeats[i+1]) so feature arrays are len(downbeats)-1.
    if len(downbeats) < (cfg.min_bars_before_drop + cfg.min_bars_after_drop + 2):
        return None

    bar_rms = _aggregate_bar_feature(feats["rms"], cfg.hop_sec, downbeats)
    bar_low = _aggregate_bar_feature(np.log1p(feats["low_energy"]), cfg.hop_sec, downbeats)
    bar_flux = _aggregate_bar_feature(feats["flux"], cfg.hop_sec, downbeats)
    bar_high = _aggregate_bar_feature(np.log1p(feats["high_energy"]), cfg.hop_sec, downbeats)
    bar_sub = _aggregate_bar_feature(feats["low_ratio"], cfg.hop_sec, downbeats)
    bar_kick = _kick_periodicity_per_bar(feats["low_transient"], cfg.hop_sec, downbeats, beat_sec)

    # Required by spec: smooth all per-bar features over 2 bars.
    bar_rms = _smooth_bars(bar_rms, cfg.feature_smooth_bars)
    bar_low = _smooth_bars(bar_low, cfg.feature_smooth_bars)
    bar_flux = _smooth_bars(bar_flux, cfg.feature_smooth_bars)
    bar_high = _smooth_bars(bar_high, cfg.feature_smooth_bars)
    bar_sub = _smooth_bars(bar_sub, cfg.feature_smooth_bars)
    bar_kick = _smooth_bars(bar_kick, cfg.feature_smooth_bars)

    n_bars = int(len(bar_rms))
    if n_bars < (cfg.min_bars_before_drop + cfg.min_bars_after_drop + 1):
        return None

    n_rms = _robust_norm(bar_rms, 15.0, 90.0)
    n_low = _robust_norm(bar_low, 15.0, 90.0)
    n_flux = _robust_norm(bar_flux, 15.0, 90.0)
    n_high = _robust_norm(bar_high, 15.0, 90.0)
    n_sub = _robust_norm(bar_sub, 15.0, 90.0)
    n_kick = _robust_norm(bar_kick, 15.0, 90.0)

    # -------------------------
    # STAGE 1: Drop zone detect
    # -------------------------
    max_drop_sec = max(8.0, float(duration_sec) * float(cfg.max_first_drop_ratio))
    stage1_items: List[Tuple[int, float]] = []
    for i in range(cfg.min_bars_before_drop, n_bars - cfg.min_bars_after_drop):
        if float(downbeats[i]) > max_drop_sec:
            continue
        prev = slice(i - 4, i)
        d_low = float(n_low[i] - np.mean(n_low[prev]))
        d_rms = float(n_rms[i] - np.mean(n_rms[prev]))
        d_flux = float(n_flux[i] - np.mean(n_flux[prev]))
        s1 = (0.46 * d_low) + (0.34 * d_rms) + (0.20 * d_flux)
        stage1_items.append((i, s1))
    if not stage1_items:
        return None

    stage1_vals = np.asarray([v for _, v in stage1_items], dtype=np.float32)
    stage1_thr = max(0.11, float(np.percentile(stage1_vals, 72.0)))
    zone_center = None
    for i, s1 in stage1_items:
        prev = slice(i - 4, i)
        d_low = float(n_low[i] - np.mean(n_low[prev]))
        d_rms = float(n_rms[i] - np.mean(n_rms[prev]))
        pre_k = float(np.mean(n_kick[prev]))
        post4_k = float(np.mean(n_kick[i:i + 4])) if (i + 4) <= n_bars else pre_k
        kick_enter = post4_k >= max(0.18, pre_k + 0.04)
        if s1 >= stage1_thr and d_low >= 0.08 and d_rms >= 0.05 and kick_enter:
            zone_center = i
            break
    if zone_center is None:
        zone_center = int(stage1_items[int(np.argmax(stage1_vals))][0])

    zone_start = max(cfg.min_bars_before_drop, zone_center - cfg.drop_zone_radius_bars)
    zone_end = min(n_bars - cfg.min_bars_after_drop, zone_center + cfg.drop_zone_radius_bars)
    if zone_end < zone_start:
        return None

    # --------------------------------------------
    # STAGE 2: Structural correction inside window
    # --------------------------------------------
    rms_floor = float(np.percentile(n_rms, 20.0))
    low_floor = float(np.percentile(n_low, 20.0))
    kick_floor = max(0.18, float(np.percentile(n_kick, 45.0)))

    cand_rows: List[Dict[str, float]] = []
    for i in range(zone_start, zone_end + 1):
        if i < 4 or (i + 8) > n_bars:
            continue
        if float(downbeats[i]) > max_drop_sec:
            continue
        prev = slice(i - 4, i)
        post4 = slice(i, i + 4)
        post8 = slice(i, i + 8)

        pre_r = float(np.mean(n_rms[prev]))
        pre_l = float(np.mean(n_low[prev]))
        pre_k = float(np.mean(n_kick[prev]))
        pre8_l = float(np.mean(n_low[i - 8:i - 4])) if i >= 8 else pre_l

        post4_r = float(np.mean(n_rms[post4]))
        post4_l = float(np.mean(n_low[post4]))
        post4_k = float(np.mean(n_kick[post4]))
        post8_r = float(np.mean(n_rms[post8]))
        post8_l = float(np.mean(n_low[post8]))

        low_jump = float(n_low[i] - pre_l)
        rms_jump = float(n_rms[i] - pre_r)
        flux_jump = float(n_flux[i] - float(np.mean(n_flux[prev])))
        buildup = 0.56 * _trend_score(n_high, i, lookback=4) + 0.44 * _trend_score(n_flux, i, lookback=4)

        # Immediate structural snap conditions (anti-late: don't wait for long stabilization).
        has_pre8 = i >= 8
        pre_drop_dip = pre_l <= (pre8_l + 0.03)
        preceded_low = float(n_low[i - 1]) <= (post4_l - 0.10)
        kick_stable = (post4_k >= max(kick_floor, pre_k + 0.03)) and (float(np.std(n_kick[post4])) <= 0.30)
        elevated_4 = (post4_l >= (pre_l + 0.10)) and (post4_r >= (pre_r + 0.08))
        no_dip_2 = float(np.min(n_rms[i:i + 2])) >= max(rms_floor + 0.03, pre_r * 0.70)
        immediate_ok = bool(has_pre8 and pre_drop_dip and preceded_low and kick_stable and elevated_4 and no_dip_2)

        # Validate after snapping (anti-late rule).
        sustain_8 = bool((post8_l >= (pre_l + 0.10)) and (post8_r >= (pre_r + 0.08)))
        fake_drop = bool((float(n_rms[i] - np.mean(n_rms[i:i + 2])) > 0.20) and (float(np.mean(n_low[i:i + 2])) < (pre_l + 0.05)))
        valid = bool(immediate_ok and sustain_8 and (not fake_drop))

        structural = (
            0.28 * low_jump +
            0.22 * rms_jump +
            0.14 * flux_jump +
            0.16 * (post4_k - pre_k) +
            0.14 * (post8_l - pre_l) +
            0.06 * buildup
        )

        cand_rows.append({
            "bar_i": float(i),
            "stage1": float(next((v for bi, v in stage1_items if bi == i), 0.0)),
            "struct": float(structural),
            "valid": 1.0 if valid else 0.0,
            "immediate_ok": 1.0 if immediate_ok else 0.0,
            "sustain8": 1.0 if sustain_8 else 0.0,
            "fake": 1.0 if fake_drop else 0.0,
            "low_jump": float(low_jump),
            "rms_jump": float(rms_jump),
            "kick_delta": float(post4_k - pre_k),
            "buildup": float(buildup),
        })

    if not cand_rows:
        return None

    # First structurally valid bar in window.
    valid_rows = [r for r in cand_rows if r["valid"] > 0.5]
    if valid_rows:
        pick = valid_rows[0]
    else:
        # Fallback: prefer sustained candidates, then choose near-best composite earliest.
        fallback_pool = [r for r in cand_rows if r["sustain8"] > 0.5]
        if not fallback_pool:
            fallback_pool = cand_rows
        comp = np.asarray([
            (0.46 * float(r["struct"])) +
            (0.38 * float(r["stage1"])) +
            (0.16 * float(r["low_jump"]))
            for r in fallback_pool
        ], dtype=np.float32)
        best = float(np.max(comp))
        near = [fallback_pool[j] for j in range(len(fallback_pool)) if comp[j] >= (best - 0.04)]
        near.sort(key=lambda r: r["bar_i"])
        pick = near[0] if near else fallback_pool[int(np.argmax(comp))]

    bar_i = int(pick["bar_i"])
    bar_i = max(0, min(bar_i, len(downbeats) - 2))
    drop_sec = float(downbeats[bar_i])

    # Confidence uses zone sharpness + structural quality + validity.
    s1_arr = np.asarray([r["stage1"] for r in cand_rows], dtype=np.float32)
    st_arr = np.asarray([r["struct"] for r in cand_rows], dtype=np.float32)
    n_s1 = _robust_norm(s1_arr, 20.0, 85.0)
    n_st = _robust_norm(st_arr, 20.0, 85.0)
    p_idx = max(0, min(len(cand_rows) - 1, next((j for j, r in enumerate(cand_rows) if int(r["bar_i"]) == bar_i), 0)))
    conf = float(np.clip(
        0.32 * float(n_s1[p_idx]) +
        0.26 * float(n_st[p_idx]) +
        0.26 * float(pick["valid"]) +
        0.10 * float(pick["sustain8"]) +
        0.06 * float(pick["immediate_ok"]) -
        0.10 * float(pick["fake"]),
        0.0,
        1.0,
    ))

    return {
        "drop_sec": float(drop_sec),
        "bar_i": float(bar_i),
        "confidence": conf,
        "valid": float(pick["valid"]),
        "downbeat_off": float(downbeat_off),
        "zone_center_bar": float(zone_center),
        "stage1": float(pick["stage1"]),
        "struct": float(pick["struct"]),
        "sustain8": float(pick["sustain8"]),
        "fake": float(pick["fake"]),
    }


def _fallback_shift_forward_bar(
    feats: Dict[str, np.ndarray],
    cfg: DetectorConfig,
    beat_times: np.ndarray,
    beat_sec: float,
    downbeat_off: int,
    bar_i: int,
) -> int:
    downbeats = _bar_downbeats(beat_times, downbeat_off)
    if len(downbeats) < 4:
        return int(bar_i)
    if bar_i < 0 or bar_i >= (len(downbeats) - 1):
        return int(max(0, min(bar_i, len(downbeats) - 2)))

    bar_rms = _smooth_bars(_aggregate_bar_feature(feats["rms"], cfg.hop_sec, downbeats), cfg.feature_smooth_bars)
    bar_low = _smooth_bars(_aggregate_bar_feature(np.log1p(feats["low_energy"]), cfg.hop_sec, downbeats), cfg.feature_smooth_bars)
    bar_kick = _smooth_bars(_kick_periodicity_per_bar(feats["low_transient"], cfg.hop_sec, downbeats, beat_sec), cfg.feature_smooth_bars)
    n_rms = _robust_norm(bar_rms, 15.0, 90.0)
    n_low = _robust_norm(bar_low, 15.0, 90.0)
    n_kick = _robust_norm(bar_kick, 15.0, 90.0)

    i = int(max(0, min(bar_i, len(n_rms) - 3)))
    # Only apply fallback-shift when the picked bar still looks like a transient/transition
    # with weak kick establishment; otherwise keep the first structural bar.
    if float(n_kick[i]) >= 0.45:
        return int(i)

    curr = 0.55 * float(n_low[i]) + 0.45 * float(n_rms[i])
    for step in (1, 2):
        j = i + step
        if j >= (len(n_rms) - 2):
            break
        nxt = 0.55 * float(n_low[j]) + 0.45 * float(n_rms[j])
        nxt2 = 0.55 * float(np.mean(n_low[j:j + 2])) + 0.45 * float(np.mean(n_rms[j:j + 2]))
        if nxt >= (curr + 0.10) and nxt2 >= (curr + 0.08):
            return int(j)
    return int(i)


def _nudge_forward_to_drop_attack(
    drop_sec: float,
    beat_sec: float,
    hop_sec: float,
    low_transient: np.ndarray,
    onset: np.ndarray,
) -> float:
    if len(low_transient) < 6 or len(onset) < 6:
        return float(drop_sec)

    dur = float(len(low_transient) * hop_sec)
    i0 = int(round(float(drop_sec) / hop_sec))
    i1 = int(round(min(dur, float(drop_sec) + 0.75 * beat_sec) / hop_sec))
    i0 = max(1, min(i0, len(low_transient) - 3))
    i1 = max(i0 + 2, min(i1, len(low_transient) - 2))
    if i1 <= i0 + 2:
        return float(drop_sec)

    lt = low_transient[i0:i1 + 1].astype(np.float32, copy=False)
    on = onset[i0:i1 + 1].astype(np.float32, copy=False)
    comp = (0.64 * _robust_norm(lt, 25.0, 90.0)) + (0.36 * _robust_norm(on, 25.0, 90.0))
    if len(comp) < 3:
        return float(drop_sec)

    base = float(max(comp[0], np.mean(comp[:min(3, len(comp))])))
    thr = max(0.38, float(np.percentile(comp, 70.0)))

    pick = None
    for j in range(1, len(comp) - 1):
        if comp[j] < thr:
            continue
        if comp[j] < comp[j - 1] or comp[j] < comp[j + 1]:
            continue
        if comp[j] >= (base + 0.08):
            pick = j
            break

    if pick is None:
        j = int(np.argmax(comp))
        if float(comp[j]) >= (base + 0.14):
            pick = j

    # If the strongest rise sits at the window end, allow endpoint selection.
    if pick is None and len(comp) >= 2:
        j = len(comp) - 1
        if comp[j] >= thr and comp[j] > comp[j - 1] and float(comp[j]) >= (base + 0.10):
            pick = j

    if pick is None:
        return float(drop_sec)

    refined = float((i0 + pick) * hop_sec)
    delta = refined - float(drop_sec)
    if delta <= 0.05:
        return float(drop_sec)
    if delta > (0.80 * beat_sec):
        return float(drop_sec)
    return refined


def detect_true_drop(
    path: str,
    bpm_hint: Optional[float] = None,
    bpm_min: float = 120.0,
    bpm_max: float = 160.0,
    config: Optional[DetectorConfig] = None,
) -> Optional[DropDetectionResult]:
    cfg = config or DetectorConfig(bpm_min=bpm_min, bpm_max=bpm_max)
    y = _decode_audio_ffmpeg(path, cfg.sr)
    if y is None or len(y) < cfg.sr * 8:
        return None

    hop = max(1, int(round(cfg.hop_sec * cfg.sr)))
    feats = _stft_features(y, cfg.sr, cfg.n_fft, hop)
    if feats is None:
        return None

    # Smooth frame features.
    beat_guess = float(np.clip(bpm_hint or 140.0, cfg.bpm_min, cfg.bpm_max))
    bars_to_frames = max(3, int(round((4 * (60.0 / beat_guess)) / cfg.hop_sec)))
    feats["rms"] = _moving_average(feats["rms"], bars_to_frames // 2)
    feats["low_energy"] = _moving_average(feats["low_energy"], bars_to_frames // 2)
    feats["high_energy"] = _moving_average(feats["high_energy"], bars_to_frames // 2)
    feats["low_ratio"] = _moving_average(feats["low_ratio"], bars_to_frames // 2)
    feats["high_ratio"] = _moving_average(feats["high_ratio"], bars_to_frames // 2)
    feats["flux"] = _moving_average(feats["flux"], max(3, bars_to_frames // 4))
    feats["onset"] = _moving_average(feats["onset"], max(3, bars_to_frames // 4))
    feats["low_transient"] = _moving_average(feats["low_transient"], max(3, bars_to_frames // 4))
    feats["high_flux"] = _moving_average(feats["high_flux"], max(3, bars_to_frames // 4))

    if bpm_hint is not None and cfg.lock_bpm_to_hint:
        bpm = float(np.clip(float(bpm_hint), cfg.bpm_min, cfg.bpm_max))
    else:
        bpm = _estimate_tempo_bpm(feats["onset"], cfg.hop_sec, cfg.bpm_min, cfg.bpm_max, bpm_hint)
    beat_sec = 60.0 / max(1e-6, bpm)
    beat_period_frames = max(1, int(round(beat_sec / cfg.hop_sec)))
    phase = _estimate_beat_phase(feats["onset"], beat_period_frames)
    duration_sec = float(len(feats["onset"])) * float(cfg.hop_sec)
    beat_times = _build_beats(duration_sec, beat_sec, float(phase) * float(cfg.hop_sec))
    if len(beat_times) < 16:
        return None

    # Evaluate all 4 bar-phase offsets and choose the musically best early drop.
    offset_results: List[Dict[str, float]] = []
    for off in range(4):
        downbeats = _bar_downbeats(beat_times, off)
        res = _evaluate_drop_for_downbeats(
            downbeats=downbeats,
            feats=feats,
            cfg=cfg,
            beat_sec=beat_sec,
            downbeat_off=off,
            duration_sec=duration_sec,
        )
        if res is not None:
            offset_results.append(res)
    if not offset_results:
        return None

    valid_results = [r for r in offset_results if r.get("valid", 0.0) > 0.5]
    if valid_results:
        top_conf = max(float(r["confidence"]) for r in valid_results)
        # First true drop: strongly prefer earliest valid bar unless confidence is drastically worse.
        near_valid = [r for r in valid_results if float(r["confidence"]) >= max(0.40, top_conf - 0.30)]
        near_valid.sort(key=lambda r: float(r["drop_sec"]))
        picked = near_valid[0] if near_valid else min(valid_results, key=lambda r: float(r["drop_sec"]))
    else:
        s1 = np.asarray([float(r.get("stage1", 0.0)) for r in offset_results], dtype=np.float32)
        st = np.asarray([float(r.get("struct", 0.0)) for r in offset_results], dtype=np.float32)
        n_s1 = _robust_norm(s1, 20.0, 85.0)
        n_st = _robust_norm(st, 20.0, 85.0)
        off_score = np.asarray([
            (0.57 * float(n_s1[i])) +
            (0.43 * float(n_st[i])) +
            (0.08 * float(offset_results[i].get("sustain8", 0.0))) -
            (0.08 * float(offset_results[i].get("fake", 0.0)))
            for i in range(len(offset_results))
        ], dtype=np.float32)
        best = float(np.max(off_score))
        near = [offset_results[i] for i in range(len(offset_results)) if off_score[i] >= (best - 0.08)]
        near.sort(key=lambda r: float(r["drop_sec"]))
        picked = near[0] if near else offset_results[int(np.argmax(off_score))]

    drop_sec = float(picked["drop_sec"])
    bar_i = int(picked["bar_i"])
    bar_downbeat_sec = float(drop_sec)

    # If no strictly valid structural candidate exists, avoid pre-drop transitional bars.
    if float(picked.get("valid", 0.0)) < 0.5:
        shifted_bar = _fallback_shift_forward_bar(
            feats=feats,
            cfg=cfg,
            beat_times=beat_times,
            beat_sec=beat_sec,
            downbeat_off=int(picked.get("downbeat_off", 0.0)),
            bar_i=bar_i,
        )
        if shifted_bar != bar_i:
            downbeats_pick = _bar_downbeats(beat_times, int(picked.get("downbeat_off", 0.0)))
            if len(downbeats_pick) >= 2:
                bar_i = max(0, min(int(shifted_bar), len(downbeats_pick) - 2))
                drop_sec = float(downbeats_pick[bar_i])
                bar_downbeat_sec = float(drop_sec)

    if cfg.refine_downbeat_with_transient:
        drop_sec = _refine_structural_downbeat_time(
            downbeat_sec=drop_sec,
            beat_sec=beat_sec,
            hop_sec=cfg.hop_sec,
            low_transient=feats["low_transient"],
            onset=feats["onset"],
        )

    # For low-confidence/fallback picks, avoid early anchors by nudging forward to
    # the first strong attack inside the current beat neighborhood.
    if float(picked.get("valid", 0.0)) < 0.5 or float(picked.get("confidence", 0.0)) < 0.80:
        drop_sec = _nudge_forward_to_drop_attack(
            drop_sec=drop_sec,
            beat_sec=beat_sec,
            hop_sec=cfg.hop_sec,
            low_transient=feats["low_transient"],
            onset=feats["onset"],
        )

    beat_pos = (drop_sec - float(beat_times[0])) / beat_sec
    beat_index = max(0, int(round(beat_pos)))
    conf = float(np.clip(float(picked["confidence"]), 0.0, 1.0))

    return DropDetectionResult(
        drop_time_sec=drop_sec,
        bar_index=bar_i,
        bar_number=bar_i + 1,
        beat_index=beat_index,
        bpm=float(bpm),
        confidence=conf,
        ableton_cue_beat_time=float(max(0.0, beat_pos)),
        debug={
            "stage1_score": float(picked.get("stage1", 0.0)),
            "struct_score": float(picked.get("struct", 0.0)),
            "valid": float(picked.get("valid", 0.0)),
            "sustain8": float(picked.get("sustain8", 0.0)),
            "fake": float(picked.get("fake", 0.0)),
            "zone_center_bar": float(picked.get("zone_center_bar", float(bar_i)) + 1.0),
            "snap_bar": float(bar_i + 1),
            "downbeat_offset": float(picked.get("downbeat_off", 0.0)),
            "bar_downbeat_sec": float(bar_downbeat_sec),
            "tempo_bpm": float(bpm),
        },
    )


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Detect first true drop in EDM track.")
    ap.add_argument("audio", help="Path to WAV/FLAC/AIFF/MP3")
    ap.add_argument("--bpm-hint", type=float, default=None)
    ap.add_argument("--bpm-min", type=float, default=120.0)
    ap.add_argument("--bpm-max", type=float, default=160.0)
    args = ap.parse_args()

    res = detect_true_drop(
        args.audio,
        bpm_hint=args.bpm_hint,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
    )
    if res is None:
        print(json.dumps({"drop_time_sec": None, "confidence": 0.0}, indent=2))
        return
    print(json.dumps({
        "drop_time_sec": res.drop_time_sec,
        "bar_number": res.bar_number,
        "beat_index": res.beat_index,
        "bpm": res.bpm,
        "confidence": res.confidence,
        "ableton_cue_beat_time": res.ableton_cue_beat_time,
        "debug": res.debug,
    }, indent=2))


if __name__ == "__main__":
    _main()
