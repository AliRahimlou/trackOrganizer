#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import DEFAULT_HOP, DEFAULT_MELS, DEFAULT_SR, FEATURE_VERSION
from .utils import audio_id, quantile_clip_norm


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError("librosa is required: pip install librosa soundfile") from e
    return librosa


def load_audio(audio_path: str, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    librosa = _require_librosa()
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        raise RuntimeError(f"Empty audio: {audio_path}")
    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = y / peak
    return y.astype(np.float32, copy=False), int(sr_loaded)


def _stft_power(y: np.ndarray, sr: int, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    librosa = _require_librosa()
    st = librosa.stft(y=y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(st).astype(np.float32, copy=False)
    pwr = (mag * mag).astype(np.float32, copy=False)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32, copy=False)
    return pwr, freqs


def _align_len(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    m = min(len(a) for a in arrs)
    return tuple(a[:m] for a in arrs)


def _flux_from_power(power: np.ndarray) -> np.ndarray:
    if power.shape[1] <= 1:
        return np.zeros(power.shape[1], dtype=np.float32)
    d = np.diff(power, axis=1, prepend=power[:, :1])
    d = np.maximum(d, 0.0)
    return np.sum(d, axis=0).astype(np.float32, copy=False)


def extract_features(
    audio_path: str,
    sr: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_MELS,
) -> Dict[str, np.ndarray]:
    librosa = _require_librosa()
    y, sr_loaded = load_audio(audio_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_loaded,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20.0,
        fmax=min(12000.0, float(sr_loaded) * 0.5),
        power=2.0,
    ).astype(np.float32, copy=False)
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32, copy=False)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0].astype(np.float32, copy=False)
    onset = librosa.onset.onset_strength(y=y, sr=sr_loaded, hop_length=hop_length).astype(np.float32, copy=False)

    p1, f1 = _stft_power(y=y, sr=sr_loaded, n_fft=1024, hop=hop_length)
    p2, f2 = _stft_power(y=y, sr=sr_loaded, n_fft=4096, hop=hop_length)

    broad1 = np.sum(p1[(f1 >= 20.0) & (f1 <= min(10000.0, float(sr_loaded) * 0.5)), :], axis=0).astype(np.float32, copy=False)
    low1 = np.sum(p1[(f1 >= 20.0) & (f1 <= 150.0), :], axis=0).astype(np.float32, copy=False)
    high1 = np.sum(p1[(f1 >= 2000.0) & (f1 <= min(10000.0, float(sr_loaded) * 0.5)), :], axis=0).astype(np.float32, copy=False)

    low_ratio = (low1 / (broad1 + 1e-9)).astype(np.float32, copy=False)
    high_ratio = (high1 / (broad1 + 1e-9)).astype(np.float32, copy=False)

    flux_1024 = _flux_from_power(p1)
    flux_4096 = _flux_from_power(p2)

    # HPSS percussive ratio
    y_h, y_p = librosa.effects.hpss(y)
    rms_h = librosa.feature.rms(y=y_h, frame_length=2048, hop_length=hop_length)[0].astype(np.float32, copy=False)
    rms_p = librosa.feature.rms(y=y_p, frame_length=2048, hop_length=hop_length)[0].astype(np.float32, copy=False)
    perc_ratio = (rms_p / (rms_h + rms_p + 1e-9)).astype(np.float32, copy=False)

    # Spectral contrast + novelty cue
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr_loaded, hop_length=hop_length, n_fft=2048)
    contrast_mean = np.mean(contrast, axis=0).astype(np.float32, copy=False)
    novelty = np.maximum(0.0, np.diff(contrast_mean, prepend=contrast_mean[:1])).astype(np.float32, copy=False)

    # Beat/downbeat priors
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_loaded, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_loaded, hop_length=hop_length).astype(np.float32, copy=False)

    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr_loaded, hop_length=hop_length).astype(np.float32, copy=False)

    mel_db, rms, onset, low1, high1, broad1, low_ratio, high_ratio, flux_1024, flux_4096, perc_ratio, novelty, contrast_mean = _align_len(
        mel_db,
        rms,
        onset,
        low1,
        high1,
        broad1,
        low_ratio,
        high_ratio,
        flux_1024,
        flux_4096,
        perc_ratio,
        novelty,
        contrast_mean,
    )
    frame_times = frame_times[: len(rms)]

    return {
        "mel_db": mel_db,
        "rms": rms,
        "onset": onset,
        "low_energy": low1,
        "high_energy": high1,
        "broad_energy": broad1,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "flux_1024": flux_1024,
        "flux_4096": flux_4096,
        "perc_ratio": perc_ratio,
        "novelty": novelty,
        "contrast": contrast_mean,
        "frame_times": frame_times,
        "beat_times": beat_times,
        "tempo_est": np.asarray([float(tempo)], dtype=np.float32),
        "sr": np.asarray([int(sr_loaded)], dtype=np.int32),
        "hop_length": np.asarray([int(hop_length)], dtype=np.int32),
        "n_mels": np.asarray([int(n_mels)], dtype=np.int32),
        "version": np.asarray([int(FEATURE_VERSION)], dtype=np.int32),
    }


def save_feature_cache(path: str, feat: Dict[str, np.ndarray], meta: Optional[Dict[str, object]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = dict(feat)
    if meta:
        for k, v in meta.items():
            if isinstance(v, str):
                data[f"meta_{k}"] = np.asarray([v], dtype=object)
            elif isinstance(v, (int, np.integer)):
                data[f"meta_{k}"] = np.asarray([int(v)], dtype=np.int64)
            elif isinstance(v, (float, np.floating)):
                data[f"meta_{k}"] = np.asarray([float(v)], dtype=np.float32)
            else:
                data[f"meta_{k}"] = np.asarray([str(v)], dtype=object)
    np.savez_compressed(path, **data)


def load_feature_cache(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def feature_cache_path(cache_dir: str, audio_path: str) -> str:
    aid = audio_id(audio_path)
    return os.path.abspath(os.path.join(cache_dir, f"{aid}.npz"))


def channel_matrix(feat: Dict[str, np.ndarray]) -> np.ndarray:
    mel = feat["mel_db"].T.astype(np.float32, copy=False)
    extra = [
        quantile_clip_norm(feat["rms"])[None, :],
        quantile_clip_norm(feat["onset"])[None, :],
        quantile_clip_norm(feat["low_ratio"])[None, :],
        quantile_clip_norm(feat["high_ratio"])[None, :],
        quantile_clip_norm(feat["flux_1024"])[None, :],
        quantile_clip_norm(feat["flux_4096"])[None, :],
        quantile_clip_norm(feat["perc_ratio"])[None, :],
        quantile_clip_norm(feat["novelty"])[None, :],
        quantile_clip_norm(feat["contrast"])[None, :],
    ]
    c = np.concatenate([mel] + extra, axis=0).astype(np.float32, copy=False)
    return c


def interpolate_context(channels: np.ndarray, frame_times: np.ndarray, center_sec: float, left_sec: float, right_sec: float, out_frames: int) -> np.ndarray:
    c, t = channels.shape
    if t == 0 or out_frames <= 1:
        return np.zeros((c, max(1, out_frames)), dtype=np.float32)

    a = float(center_sec) - float(left_sec)
    b = float(center_sec) + float(right_sec)
    x_new = np.linspace(a, b, num=out_frames, dtype=np.float32)
    ft = frame_times.astype(np.float32, copy=False)
    ch = channels.astype(np.float32, copy=False)

    # Vectorized linear interpolation for all channels at once.
    idx = np.searchsorted(ft, x_new, side="left").astype(np.int32, copy=False)
    out = np.empty((c, out_frames), dtype=np.float32)

    left_mask = idx <= 0
    right_mask = idx >= t
    mid_mask = ~(left_mask | right_mask)

    if np.any(left_mask):
        out[:, left_mask] = ch[:, [0]]
    if np.any(right_mask):
        out[:, right_mask] = ch[:, [-1]]
    if np.any(mid_mask):
        j1 = idx[mid_mask]
        j0 = j1 - 1
        x0 = ft[j0]
        x1 = ft[j1]
        denom = np.maximum(x1 - x0, 1e-9)
        w = ((x_new[mid_mask] - x0) / denom).astype(np.float32, copy=False)
        y0 = ch[:, j0]
        y1 = ch[:, j1]
        out[:, mid_mask] = y0 + (y1 - y0) * w[None, :]
    return out
