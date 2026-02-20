#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np


def decode_audio_ffmpeg(path: str, sr: int = 22050) -> Optional[np.ndarray]:
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
    if peak > 1e-9:
        y = y / peak
    return y


def _robust_scale(v: np.ndarray) -> np.ndarray:
    if v is None or len(v) == 0:
        return np.asarray([], dtype=np.float32)
    q25 = float(np.percentile(v, 25.0))
    q50 = float(np.percentile(v, 50.0))
    q75 = float(np.percentile(v, 75.0))
    iqr = max(1e-6, q75 - q25)
    z = (v - q50) / iqr
    z = np.clip(z, -3.0, 3.0)
    z = (z + 3.0) / 6.0
    return z.astype(np.float32, copy=False)


def _resample(v: np.ndarray, n: int) -> np.ndarray:
    if len(v) == n:
        return v.astype(np.float32, copy=False)
    if len(v) <= 1:
        return np.zeros(n, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(v), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return np.interp(x_new, x_old, v).astype(np.float32, copy=False)


def _frame_features(y: np.ndarray, sr: int, hop_sec: float = 0.05, n_fft: int = 1024) -> Optional[Dict[str, np.ndarray]]:
    hop = max(1, int(round(hop_sec * sr)))
    if len(y) < n_fft + hop:
        return None
    n = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    low_mask = (freqs >= 20.0) & (freqs <= 120.0)
    high_mask = (freqs >= 2000.0) & (freqs <= min(10000.0, float(sr) * 0.5))

    rms = np.empty(n, dtype=np.float32)
    low = np.empty(n, dtype=np.float32)
    high = np.empty(n, dtype=np.float32)
    onset = np.zeros(n, dtype=np.float32)
    prev_r = 0.0
    for i in range(n):
        seg = y[i * hop:i * hop + n_fft]
        if len(seg) < n_fft:
            break
        r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        rms[i] = r
        onset[i] = max(0.0, r - prev_r)
        prev_r = r

        mag = np.abs(np.fft.rfft(seg * win)).astype(np.float32, copy=False)
        p = mag * mag
        low[i] = float(np.sum(p[low_mask])) if np.any(low_mask) else 0.0
        high[i] = float(np.sum(p[high_mask])) if np.any(high_mask) else 0.0

    return {
        "rms": rms,
        "low": low,
        "high": high,
        "onset": onset,
    }


def extract_drop_signature_from_audio(
    y: np.ndarray,
    sr: int,
    center_sec: float,
    bpm: int,
    pre_beats: float = 8.0,
    post_beats: float = 16.0,
    n_bins: int = 48,
) -> Optional[np.ndarray]:
    if y is None or len(y) < 1000 or bpm <= 0:
        return None
    beat_sec = 60.0 / float(bpm)
    pre_sec = pre_beats * beat_sec
    post_sec = post_beats * beat_sec
    dur = float(len(y)) / float(sr)
    t0 = max(0.0, float(center_sec) - pre_sec)
    t1 = min(dur, float(center_sec) + post_sec)
    if (t1 - t0) < max(3.0, 6.0 * beat_sec):
        return None

    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))
    i0 = max(0, min(i0, len(y)))
    i1 = max(i0 + 1, min(i1, len(y)))
    seg = y[i0:i1]
    feats = _frame_features(seg, sr=sr, hop_sec=0.05, n_fft=1024)
    if feats is None:
        return None

    rms = _robust_scale(feats["rms"])
    low = _robust_scale(np.log1p(feats["low"]))
    onset = _robust_scale(feats["onset"])
    low_ratio = _robust_scale(feats["low"] / (feats["low"] + feats["high"] + 1e-9))

    v = np.concatenate([
        _resample(rms, n_bins),
        _resample(low, n_bins),
        _resample(onset, n_bins),
        _resample(low_ratio, n_bins),
    ]).astype(np.float32, copy=False)
    nrm = float(np.linalg.norm(v))
    if nrm <= 1e-9:
        return None
    return (v / nrm).astype(np.float32, copy=False)


def extract_drop_signature_from_path(path: str, center_sec: float, bpm: int, sr: int = 22050) -> Optional[np.ndarray]:
    y = decode_audio_ffmpeg(path, sr=sr)
    if y is None:
        return None
    return extract_drop_signature_from_audio(y=y, sr=sr, center_sec=center_sec, bpm=bpm)


def signature_matrix_from_model(model: Dict[str, object]) -> Optional[np.ndarray]:
    rows = model.get("signature_rows")
    if not isinstance(rows, list) or not rows:
        return None
    sigs = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        s = r.get("sig")
        if not isinstance(s, list) or not s:
            continue
        arr = np.asarray(s, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n <= 1e-9:
            continue
        sigs.append(arr / n)
    if not sigs:
        return None
    return np.vstack(sigs).astype(np.float32, copy=False)


def suggest_shift_by_signature(
    y: np.ndarray,
    sr: int,
    bpm: int,
    anchor_sec: float,
    signature_matrix: np.ndarray,
    max_shift_beats: float = 8.0,
    step_beats: float = 0.5,
    top_k: int = 20,
) -> Tuple[Optional[float], float, str]:
    if signature_matrix is None or len(signature_matrix) == 0 or bpm <= 0:
        return None, 0.0, ""

    beat_sec = 60.0 / float(bpm)
    shifts = np.arange(-max_shift_beats, max_shift_beats + 1e-6, step_beats, dtype=np.float32)
    if 0.0 not in shifts:
        shifts = np.sort(np.append(shifts, 0.0))

    best_shift = None
    best_score = -1e9
    zero_score = None
    for sb in shifts:
        csec = float(anchor_sec + (float(sb) * beat_sec))
        sig = extract_drop_signature_from_audio(y=y, sr=sr, center_sec=csec, bpm=bpm)
        if sig is None:
            continue
        sims = np.dot(signature_matrix, sig)
        if sims.size == 0:
            continue
        k = max(1, min(int(top_k), int(len(sims))))
        top = np.partition(sims, -k)[-k:]
        score = float(np.mean(top))
        if abs(float(sb)) <= 1e-6:
            zero_score = score
        if score > best_score:
            best_score = score
            best_shift = float(sb)

    if best_shift is None:
        return None, 0.0, ""
    if zero_score is None:
        zero_score = best_score

    improve = float(best_score - zero_score)
    if abs(best_shift) < 0.20:
        return None, 0.0, ""
    if improve < 0.015:
        return None, 0.0, ""

    conf = 0.60 + min(0.35, max(0.0, improve * 4.0))
    conf = max(0.0, min(1.0, conf))
    return float(best_shift), float(conf), f"signature(top={best_score:.3f},base={zero_score:.3f})"

