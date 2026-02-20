#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "librosa is required for audio feature extraction. "
            "Install with: pip install librosa soundfile"
        ) from e
    return librosa


def audio_id(audio_path: str) -> str:
    return hashlib.sha1(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:16]


def load_audio(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    librosa = _require_librosa()
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        raise RuntimeError(f"Decoded empty audio: {audio_path}")
    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = y / peak
    return y.astype(np.float32, copy=False), int(sr_loaded)


def _align_len(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    m = min(len(a) for a in arrs)
    return tuple(a[:m] for a in arrs)


def extract_feature_dict(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 96,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    librosa = _require_librosa()
    if fmax is None:
        fmax = min(float(sr) * 0.5, 12000.0)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    ).astype(np.float32, copy=False)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32, copy=False)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0].astype(np.float32, copy=False)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32, copy=False)

    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft).astype(np.float32, copy=False)
    power = (mag * mag).astype(np.float32, copy=False)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 20.0) & (freqs <= 150.0)
    if not np.any(low_mask):
        low_energy = np.zeros(power.shape[1], dtype=np.float32)
    else:
        low_energy = np.sum(power[low_mask, :], axis=0).astype(np.float32, copy=False)

    mel_db_t = mel_db.T.astype(np.float32, copy=False)
    mel_db_t, rms, onset, low_energy = _align_len(mel_db_t, rms, onset, low_energy)
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length).astype(np.float32, copy=False)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).astype(np.float32, copy=False)

    out: Dict[str, np.ndarray] = {
        "mel_db": mel_db_t,
        "rms": rms,
        "onset": onset,
        "low_energy": low_energy,
        "frame_times": frame_times,
        "beat_times": beat_times,
        "tempo_est": np.asarray([float(tempo)], dtype=np.float32),
    }
    return out


def extract_features_from_path(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 96,
) -> Dict[str, np.ndarray]:
    y, sr_loaded = load_audio(audio_path, sr=sr)
    return extract_feature_dict(
        y=y,
        sr=sr_loaded,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
    )


def save_feature_cache(
    cache_path: str,
    features: Dict[str, np.ndarray],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    data = dict(features)
    if meta:
        for k, v in meta.items():
            if isinstance(v, str):
                data[f"meta_{k}"] = np.asarray([v], dtype=object)
            else:
                data[f"meta_{k}"] = np.asarray([v])
    np.savez_compressed(cache_path, **data)


def load_feature_cache(cache_path: str) -> Dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    return out

