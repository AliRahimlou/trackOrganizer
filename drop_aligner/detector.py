from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy import signal

from .drumprint import (
    DRUMPRINT_FEATURE_KEYS,
    DrumprintAnalysis,
    build_drumprint_analysis,
    empty_drumprint_features,
    score_candidate_drumprint,
)
from .groove import FULL_GROOVE_FEATURE_KEYS, empty_full_groove_features
from .microalign import MICROALIGN_FEATURE_KEYS, microalign_marker
from .ranker import load_ranker_payload, predict_candidate_distances
from .region_model import load_region_model_payload, predict_region_errors


@dataclass
class DropDetectorConfig:
    sample_rate: Optional[int] = None
    n_fft: int = 2048
    hop_length: int = 512
    low_freq_hz: float = 150.0
    hpss: bool = True
    min_drop_time_sec: float = 4.0
    max_drop_time_ratio: float = 0.70
    search_window_bars: float = 8.0
    pre_bars: float = 2.0
    post_bars: float = 4.0
    candidate_peak_distance_beats: float = 0.50
    candidate_prominence: float = 0.12
    min_score: float = 0.42
    first_drop_near_best_ratio: float = 0.90
    min_post_density: float = 0.18
    min_energy_contrast: float = 0.05
    min_low_end_jump: float = -0.05
    snap_back_ms: float = 100.0
    snap_forward_ms: float = 120.0
    zero_crossing_back_ms: float = 10.0
    use_ranker_model: bool = True
    ranker_model_path: Optional[str] = None
    ranker_top_n: int = 10
    use_region_model: bool = True
    region_model_path: Optional[str] = None
    region_model_top_n: int = 80
    region_model_full_track: bool = True
    use_drumprint: Optional[bool] = None
    use_microalign: bool = False
    microalign_confidence_threshold: float = 0.80
    microalign_max_offset_ms: float = 120.0
    full_groove_pre_bars: float = 8.0
    full_groove_post_bars: float = 4.0
    edm_transition_scan: bool = True
    edm_transition_pre_bars: float = 4.0
    edm_transition_post_bars: float = 4.0
    edm_transition_peak_distance_bars: float = 2.0


@dataclass
class FeatureBundle:
    audio_path: str
    y: np.ndarray
    sr: int
    duration_sec: float
    bpm: float
    beat_sec: float
    frame_times: np.ndarray
    onset: np.ndarray
    low_energy: np.ndarray
    low_jump_curve: np.ndarray
    rms: np.ndarray
    contrast: np.ndarray
    spectral_flux: np.ndarray
    combined_attack: np.ndarray


@dataclass
class DropCandidate:
    time_sec: float
    snapped_sec: float
    score: float
    transient_strength: float
    low_end_jump: float
    post_drop_density: float
    pre_post_energy_ratio: float
    energy_contrast: float
    rhythmic_consistency: float
    snap_offset_sec: float = 0.0
    rank: int = 0
    handcrafted_rank: int = 0
    model_rank: int = 0
    model_score: Optional[float] = None
    region_model_rank: int = 0
    region_model_score: Optional[float] = None
    region_model_confidence: float = 0.0
    selected_by: str = ""
    selected: bool = False
    selection_reason: str = ""
    rejected: bool = False
    rejection_reason: str = ""
    drumprint: Dict[str, object] = field(default_factory=dict)
    full_groove: Dict[str, object] = field(default_factory=dict)
    microalign: Dict[str, object] = field(default_factory=dict)
    debug: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        reason = self.selection_reason
        if not reason and self.rejected:
            reason = f"rejected:{self.rejection_reason or 'failed_drop_filters'}"
        elif not reason:
            reason = "ranked_candidate"
        drumprint = dict(self.drumprint or empty_drumprint_features())
        full_groove = dict(self.full_groove or empty_full_groove_features())
        microalign = dict(self.microalign or {})
        payload = {
            "rank": int(self.rank),
            "handcrafted_rank": int(self.handcrafted_rank),
            "model_rank": int(self.model_rank),
            "model_score": None if self.model_score is None else float(self.model_score),
            "region_model_rank": int(self.region_model_rank),
            "region_model_score": None if self.region_model_score is None else float(self.region_model_score),
            "region_model_confidence": float(self.region_model_confidence),
            "selected_by": self.selected_by,
            "timestamp": float(self.snapped_sec),
            "coarse_timestamp": float(self.time_sec),
            "confidence_score": float(self.score),
            "score": float(self.score),
            "transient_strength": float(self.transient_strength),
            "low_end_jump": float(self.low_end_jump),
            "post_drop_density": float(self.post_drop_density),
            "pre_post_energy_ratio": float(self.pre_post_energy_ratio),
            "energy_contrast": float(self.energy_contrast),
            "rhythmic_consistency": float(self.rhythmic_consistency),
            "snap_offset": float(self.snap_offset_sec),
            "snap_offset_sec": float(self.snap_offset_sec),
            "selected": bool(self.selected),
            "rejected": bool(self.rejected),
            "reason": reason,
            "rejection_reason": self.rejection_reason,
            "drumprint": drumprint,
            "full_groove": full_groove,
            "microalign": microalign,
            "debug": dict(self.debug),
        }
        for key in DRUMPRINT_FEATURE_KEYS:
            payload[key] = float(drumprint.get(key, 0.0) or 0.0)
        for key in FULL_GROOVE_FEATURE_KEYS:
            payload[key] = float(full_groove.get(key, 0.0) or 0.0)
        for key in MICROALIGN_FEATURE_KEYS:
            payload[key] = float(microalign.get(key, 0.0) or 0.0)
        return payload


@dataclass
class DropDetectionResult:
    audio_path: str
    drop_sec: float
    coarse_sec: float
    bpm: float
    confidence: float
    sample_rate: int
    candidates: List[DropCandidate]
    features_summary: Dict[str, object]
    selected_by: str = "handcrafted"
    confidence_tier: str = "MEDIUM"

    def to_dict(self) -> Dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "drop_sec": float(self.drop_sec),
            "coarse_sec": float(self.coarse_sec),
            "bpm": float(self.bpm),
            "confidence": float(self.confidence),
            "confidence_tier": self.confidence_tier,
            "sample_rate": int(self.sample_rate),
            "selected_by": self.selected_by,
            "features_summary": dict(self.features_summary),
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def top_candidate_dicts(self, limit: int = 10) -> List[Dict[str, object]]:
        return [candidate.to_dict() for candidate in self.candidates[: max(0, int(limit))]]

    def selected_candidate_dict(self) -> Optional[Dict[str, object]]:
        for candidate in self.candidates:
            if candidate.selected:
                return candidate.to_dict()
        return None


def _safe_norm(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if not math.isfinite(a) or not math.isfinite(b) or b <= a + 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames))
    if frames <= 1 or x.size == 0:
        return x.astype(np.float64, copy=False)
    kernel = np.ones(frames, dtype=np.float64) / float(frames)
    return np.convolve(x.astype(np.float64, copy=False), kernel, mode="same")


def _window_indices(times: np.ndarray, start: float, end: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    return max(0, i0), min(len(times), i1)


def _mean_window(x: np.ndarray, times: np.ndarray, start: float, end: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, start, end)
    if i1 <= i0:
        return float(fallback)
    return float(np.mean(x[i0:i1]))


def _max_window(x: np.ndarray, times: np.ndarray, center: float, radius: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, center - radius, center + radius)
    if i1 <= i0:
        return float(fallback)
    return float(np.max(x[i0:i1]))


def _argmax_time_window(
    x: np.ndarray,
    times: np.ndarray,
    center: float,
    radius: float,
    fallback: Optional[float] = None,
) -> float:
    i0, i1 = _window_indices(times, center - radius, center + radius)
    if i1 <= i0:
        return float(center if fallback is None else fallback)
    idx = i0 + int(np.argmax(x[i0:i1]))
    return float(times[int(np.clip(idx, 0, len(times) - 1))])


def _first_strong_time_window(
    x: np.ndarray,
    times: np.ndarray,
    center: float,
    back_radius: float,
    forward_radius: float,
    fallback: Optional[float] = None,
) -> float:
    i0, i1 = _window_indices(times, center - back_radius, center + forward_radius)
    if i1 <= i0:
        return float(center if fallback is None else fallback)
    local = np.asarray(x[i0:i1], dtype=np.float64)
    if local.size == 0:
        return float(center if fallback is None else fallback)
    high = float(np.max(local))
    if high <= 1e-9:
        return float(center if fallback is None else fallback)
    threshold = max(float(np.percentile(local, 70.0)), high * 0.58)
    for offset, value in enumerate(local):
        if float(value) >= threshold:
            return float(times[i0 + offset])
    return float(times[i0 + int(np.argmax(local))])


def _load_audio(audio_path: str, cfg: DropDetectorConfig) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")
    peak = float(np.percentile(np.abs(y), 99.9))
    if peak > 1e-9:
        y = np.clip(y / peak, -1.0, 1.0).astype(np.float32)
    y = np.nan_to_num(y, copy=False).astype(np.float32, copy=False)
    return y, int(sr)


def extract_features(audio_path: str, cfg: Optional[DropDetectorConfig] = None, bpm: Optional[float] = None) -> FeatureBundle:
    cfg = cfg or DropDetectorConfig()
    y, sr = _load_audio(audio_path, cfg)
    analysis_y = y
    if cfg.hpss:
        try:
            _, percussive = librosa.effects.hpss(y)
            analysis_y = np.asarray(percussive, dtype=np.float32)
        except Exception:
            analysis_y = y

    onset = librosa.onset.onset_strength(
        y=analysis_y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        aggregate=np.median,
    )
    onset = _safe_norm(_smooth(onset, 3))

    stft = librosa.stft(y=analysis_y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=True)
    mag = np.abs(stft).astype(np.float64)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
    low_mask = freqs <= float(cfg.low_freq_hz)
    if not np.any(low_mask):
        low_mask[: max(1, len(freqs) // 16)] = True
    low_energy = np.mean(mag[low_mask, :] ** 2, axis=0)
    low_energy = _safe_norm(_smooth(np.log1p(low_energy), 5))
    low_jump_curve = _safe_norm(np.maximum(0.0, np.diff(low_energy, prepend=low_energy[:1])))

    rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length, center=True)[0]
    rms = _safe_norm(_smooth(np.log1p(rms), 5))

    try:
        contrast_raw = librosa.feature.spectral_contrast(S=mag + 1e-9, sr=sr)
        contrast = _safe_norm(_smooth(np.mean(contrast_raw, axis=0), 5))
    except Exception:
        contrast = np.zeros_like(rms)

    log_mag = np.log1p(mag)
    spectral_flux = np.sqrt(np.sum(np.maximum(0.0, np.diff(log_mag, axis=1, prepend=log_mag[:, :1])) ** 2, axis=0))
    spectral_flux = _safe_norm(_smooth(spectral_flux, 3))

    n = min(len(onset), len(low_energy), len(rms), len(contrast), len(spectral_flux))
    onset = onset[:n]
    low_energy = low_energy[:n]
    low_jump_curve = low_jump_curve[:n]
    rms = rms[:n]
    contrast = contrast[:n]
    spectral_flux = spectral_flux[:n]
    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=cfg.hop_length)

    if bpm is None or bpm <= 0:
        try:
            tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=cfg.hop_length)
            bpm = float(np.ravel(tempo)[0])
        except Exception:
            bpm = 128.0
    if not math.isfinite(float(bpm)) or float(bpm) <= 0:
        bpm = 128.0
    bpm = float(np.clip(float(bpm), 60.0, 220.0))

    combined_attack = _safe_norm((0.55 * onset) + (0.45 * low_jump_curve))
    duration_sec = float(len(y) / float(sr))
    return FeatureBundle(
        audio_path=str(audio_path),
        y=y,
        sr=sr,
        duration_sec=duration_sec,
        bpm=bpm,
        beat_sec=60.0 / bpm,
        frame_times=frame_times,
        onset=onset,
        low_energy=low_energy,
        low_jump_curve=low_jump_curve,
        rms=rms,
        contrast=contrast,
        spectral_flux=spectral_flux,
        combined_attack=combined_attack,
    )


def _region_centers(features: FeatureBundle, cfg: DropDetectorConfig, external_cues: Optional[Sequence[float]]) -> List[float]:
    times = features.frame_times
    beat = features.beat_sec
    bar = 4.0 * beat
    pre = max(beat, float(cfg.pre_bars) * bar)
    post = max(beat, float(cfg.post_bars) * bar)
    region_score = np.zeros_like(times, dtype=np.float64)

    step = max(1, int(round(beat / max(1e-6, times[1] - times[0]))) if len(times) > 1 else 1)
    for idx in range(0, len(times), step):
        t = float(times[idx])
        pre_r = _mean_window(features.rms, times, t - pre, t)
        post_r = _mean_window(features.rms, times, t, t + post)
        pre_l = _mean_window(features.low_energy, times, t - pre, t)
        post_l = _mean_window(features.low_energy, times, t, t + post)
        post_o = _mean_window(features.onset, times, t, t + post)
        score = (0.40 * max(0.0, post_r - pre_r)) + (0.40 * max(0.0, post_l - pre_l)) + (0.20 * post_o)
        region_score[idx : min(len(times), idx + step)] = score

    max_time = max(0.0, features.duration_sec * float(cfg.max_drop_time_ratio))
    distance = max(1, int(round((2.0 * bar) / max(1e-6, times[1] - times[0]))) if len(times) > 1 else 1)
    peaks, _ = signal.find_peaks(region_score, distance=distance, prominence=max(0.02, float(np.std(region_score)) * 0.35))

    centers = [float(times[p]) for p in peaks if cfg.min_drop_time_sec <= float(times[p]) <= max_time]
    if external_cues:
        for cue in external_cues:
            try:
                cue_f = float(cue)
            except Exception:
                continue
            if cfg.min_drop_time_sec <= cue_f <= max_time:
                centers.append(cue_f)
    if not centers:
        fallback_start = max(float(cfg.min_drop_time_sec), 8.0 * bar)
        fallback_end = max(fallback_start + bar, max_time)
        centers = list(np.arange(fallback_start, fallback_end, 4.0 * bar))

    deduped: List[float] = []
    for center in sorted(centers):
        if not deduped or abs(center - deduped[-1]) > bar:
            deduped.append(center)
    return deduped


def _candidate_peak_times(features: FeatureBundle, cfg: DropDetectorConfig, centers: Sequence[float]) -> List[float]:
    times = features.frame_times
    attack = features.combined_attack
    beat = features.beat_sec
    bar = 4.0 * beat
    windows: List[Tuple[float, float]] = []
    for c in centers:
        windows.append((max(float(cfg.min_drop_time_sec), float(c) - cfg.search_window_bars * bar), float(c) + cfg.search_window_bars * bar))
    if not windows:
        return []

    distance = max(1, int(round((cfg.candidate_peak_distance_beats * beat) / max(1e-6, times[1] - times[0]))) if len(times) > 1 else 1)
    peaks, _ = signal.find_peaks(attack, distance=distance, prominence=float(cfg.candidate_prominence))
    out: List[float] = []
    for p in peaks:
        t = float(times[p])
        if any(start <= t <= end for start, end in windows):
            out.append(t)
    return sorted(set(round(t, 6) for t in out))


def _full_track_candidate_peak_times(
    features: FeatureBundle,
    cfg: DropDetectorConfig,
    *,
    external_cues: Optional[Sequence[float]] = None,
) -> List[float]:
    times = features.frame_times
    attack = features.combined_attack
    if times.size == 0 or attack.size == 0:
        return []
    beat = features.beat_sec
    max_time = max(float(cfg.min_drop_time_sec), features.duration_sec * float(cfg.max_drop_time_ratio))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    distance = max(1, int(round((cfg.candidate_peak_distance_beats * beat) / max(1e-6, hop))))
    prominence = max(0.035, min(float(cfg.candidate_prominence), 0.08))
    height = float(np.percentile(attack, 55.0)) if attack.size else 0.0
    peaks, _ = signal.find_peaks(attack, distance=distance, prominence=prominence, height=height)
    out: List[float] = [
        float(times[p])
        for p in peaks
        if float(cfg.min_drop_time_sec) <= float(times[p]) <= max_time
    ]

    if external_cues:
        for cue in external_cues:
            try:
                cue_f = float(cue)
            except Exception:
                continue
            if float(cfg.min_drop_time_sec) <= cue_f <= max_time:
                out.append(cue_f)

    if not out:
        return []
    return sorted(set(round(t, 6) for t in out))


def _edm_transition_candidate_times(
    features: FeatureBundle,
    cfg: DropDetectorConfig,
    *,
    external_cues: Optional[Sequence[float]] = None,
) -> List[float]:
    if not cfg.edm_transition_scan:
        return []
    times = features.frame_times
    if times.size < 8:
        return []

    beat = features.beat_sec
    bar = 4.0 * beat
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    pre = max(2.0 * bar, float(cfg.edm_transition_pre_bars) * bar)
    post = max(2.0 * bar, float(cfg.edm_transition_post_bars) * bar)
    immediate = max(beat, bar)
    max_time = max(float(cfg.min_drop_time_sec), features.duration_sec * float(cfg.max_drop_time_ratio))
    step = max(1, int(round(beat / max(1e-6, hop))))

    scan_times: List[float] = []
    scores: List[float] = []
    for idx in range(0, len(times), step):
        t = float(times[idx])
        if t < float(cfg.min_drop_time_sec) or t > max_time:
            continue
        if t + min(post, 2.0 * bar) > features.duration_sec:
            continue

        early_start = max(0.0, t - pre)
        early_end = max(early_start, t - (1.5 * bar))
        early_rms = _mean_window(features.rms, times, early_start, early_end)
        early_low = _mean_window(features.low_energy, times, early_start, early_end)
        early_flux = _mean_window(features.spectral_flux, times, early_start, early_end)

        immediate_rms = _mean_window(features.rms, times, t - immediate, t, fallback=early_rms)
        immediate_low = _mean_window(features.low_energy, times, t - immediate, t, fallback=early_low)
        immediate_flux = _mean_window(features.spectral_flux, times, t - immediate, t, fallback=early_flux)

        post_rms = _mean_window(features.rms, times, t, t + post)
        post_low = _mean_window(features.low_energy, times, t, t + post)
        post_onset = _mean_window(features.onset, times, t, t + post)
        impact_flux = _max_window(features.spectral_flux, times, t, max(0.12, 0.35 * beat))
        impact_attack = _max_window(features.combined_attack, times, t, max(0.12, 0.40 * beat))

        low_vacancy = float(np.clip(1.0 - (0.62 * immediate_low + 0.38 * immediate_rms), 0.0, 1.0))
        ramp = float(
            np.clip(
                (0.42 * max(0.0, immediate_flux - early_flux))
                + (0.33 * max(0.0, immediate_rms - early_rms))
                + (0.25 * max(0.0, immediate_low - early_low)),
                0.0,
                1.0,
            )
        )
        impact = float(
            np.clip(
                (0.34 * max(0.0, post_rms - immediate_rms))
                + (0.34 * max(0.0, post_low - immediate_low))
                + (0.16 * post_onset)
                + (0.10 * impact_flux)
                + (0.06 * impact_attack),
                0.0,
                1.0,
            )
        )
        first_beat_low = _max_window(features.low_energy, times, t, max(0.12, 0.20 * beat))
        beat_pulses = [
            _max_window(features.low_jump_curve, times, t + (i * beat), max(0.10, 0.18 * beat))
            for i in range(4)
        ]
        kick_reentry = float(np.clip((0.38 * first_beat_low) + (0.42 * np.mean(beat_pulses)) + (0.20 * max(beat_pulses)), 0.0, 1.0))

        score = float(
            np.clip(
                (0.26 * low_vacancy)
                + (0.14 * ramp)
                + (0.32 * impact)
                + (0.28 * kick_reentry),
                0.0,
                1.0,
            )
        )
        scan_times.append(t)
        scores.append(score)

    if not scores:
        return []

    score_curve = np.asarray(scores, dtype=np.float64)
    scan = np.asarray(scan_times, dtype=np.float64)
    distance = max(1, int(round(float(cfg.edm_transition_peak_distance_bars) * bar / max(1e-6, beat))))
    height = max(float(np.percentile(score_curve, 72.0)), float(np.max(score_curve)) * 0.58, 0.24)
    prominence = max(0.015, float(np.std(score_curve)) * 0.20)
    peaks, _props = signal.find_peaks(score_curve, distance=distance, height=height, prominence=prominence)

    reentry_curve = _safe_norm((0.70 * features.low_jump_curve) + (0.30 * features.low_energy))
    out: List[float] = []
    for peak in peaks:
        t = float(scan[int(peak)])
        attack_time = _first_strong_time_window(
            reentry_curve,
            times,
            t,
            max(0.24, 0.85 * beat),
            max(0.12, 0.30 * beat),
            fallback=t,
        )
        if float(cfg.min_drop_time_sec) <= attack_time <= max_time:
            out.append(float(attack_time))
        elif float(cfg.min_drop_time_sec) <= t <= max_time:
            out.append(t)

    if external_cues:
        for cue in external_cues:
            try:
                cue_f = float(cue)
            except Exception:
                continue
            if float(cfg.min_drop_time_sec) <= cue_f <= max_time:
                out.append(cue_f)

    return sorted(set(round(t, 6) for t in out))


def _rhythmic_consistency(features: FeatureBundle, start: float, end: float) -> float:
    times = features.frame_times
    i0, i1 = _window_indices(times, start, end)
    x = features.low_jump_curve[i0:i1]
    if x.size < 8:
        return 0.0
    x = x - float(np.mean(x))
    if float(np.std(x)) <= 1e-8:
        return 0.0
    corr = signal.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    beat_lag = max(1, int(round(features.beat_sec / max(1e-6, hop))))
    search = corr[max(1, beat_lag - 2) : min(len(corr), beat_lag + 3)]
    if search.size == 0 or float(corr[0]) <= 1e-9:
        return 0.0
    return float(np.clip(np.max(search) / corr[0], 0.0, 1.0))


def _onset_density(features: FeatureBundle, start: float, end: float) -> float:
    i0, i1 = _window_indices(features.frame_times, start, end)
    onset = features.onset[i0:i1]
    if onset.size < 2:
        return 0.0
    hop = float(features.frame_times[1] - features.frame_times[0]) if len(features.frame_times) > 1 else 0.01
    distance = max(1, int(round(0.35 * features.beat_sec / max(1e-6, hop))))
    peaks, _ = signal.find_peaks(onset, distance=distance, height=0.30)
    expected_hits = max(1.0, (float(end) - float(start)) / max(1e-6, features.beat_sec))
    return float(np.clip(len(peaks) / expected_hits, 0.0, 1.0))


def _post_bar_groove_stability(features: FeatureBundle, start: float, bars: float) -> float:
    beat = features.beat_sec
    bar = 4.0 * beat
    bar_count = max(2, int(round(float(bars))))
    activities: List[float] = []
    for idx in range(bar_count):
        a = float(start) + (idx * bar)
        b = a + bar
        low = _mean_window(features.low_energy, features.frame_times, a, b)
        rms = _mean_window(features.rms, features.frame_times, a, b)
        density = _onset_density(features, a, b)
        activities.append(float(np.clip((0.38 * low) + (0.32 * rms) + (0.30 * density), 0.0, 1.0)))
    if not activities:
        return 0.0
    arr = np.asarray(activities, dtype=np.float64)
    mean_activity = float(np.mean(arr))
    if mean_activity <= 1e-9:
        return 0.0
    cv = float(np.std(arr) / max(1e-6, mean_activity))
    repeatability = float(np.clip(1.0 - (cv / 0.85), 0.0, 1.0))
    continuity_threshold = max(0.15, 0.45 * mean_activity)
    continuity = float(np.mean(arr >= continuity_threshold))
    return float(np.clip((0.55 * mean_activity) + (0.25 * repeatability) + (0.20 * continuity), 0.0, 1.0))


def _full_groove_transition_features(
    features: FeatureBundle,
    t: float,
    cfg: DropDetectorConfig,
    *,
    low_end_jump: float,
) -> Dict[str, float]:
    beat = features.beat_sec
    bar = 4.0 * beat
    pre = max(beat, float(cfg.full_groove_pre_bars) * bar)
    post = max(bar, float(cfg.full_groove_post_bars) * bar)
    immediate = max(beat, 1.0 * beat)

    pre_low = _mean_window(features.low_energy, features.frame_times, t - pre, t)
    post_low = _mean_window(features.low_energy, features.frame_times, t, t + post)
    pre_rms = _mean_window(features.rms, features.frame_times, t - pre, t)
    post_rms = _mean_window(features.rms, features.frame_times, t, t + post)
    pre_onset = _mean_window(features.onset, features.frame_times, t - pre, t)
    post_onset = _mean_window(features.onset, features.frame_times, t, t + post)
    immediate_low = _mean_window(features.low_energy, features.frame_times, t - immediate, t, fallback=pre_low)
    immediate_rms = _mean_window(features.rms, features.frame_times, t - immediate, t, fallback=pre_rms)
    first_beat_low = _mean_window(features.low_energy, features.frame_times, t, t + beat, fallback=post_low)
    first_beat_rms = _mean_window(features.rms, features.frame_times, t, t + beat, fallback=post_rms)
    first_beat_density = _onset_density(features, t, t + beat)
    post_density = _onset_density(features, t, t + post)

    drum_onset_spike = _max_window(features.onset, features.frame_times, t, max(0.08, 0.20 * beat))
    rms_jump = float(np.clip((post_rms - pre_rms + 0.25) / 1.25, 0.0, 1.0))
    spectral_flux_peak = _max_window(features.spectral_flux, features.frame_times, t, max(0.08, 0.25 * beat))
    first_beat_activity = float(np.clip((0.40 * first_beat_low) + (0.35 * first_beat_rms) + (0.25 * first_beat_density), 0.0, 1.0))
    post_activity = float(np.clip((0.40 * post_low) + (0.35 * post_rms) + (0.25 * post_density), 0.0, 1.0))
    immediate_groove_start_score = float(
        np.clip(first_beat_activity / max(1e-6, post_activity * 0.85), 0.0, 1.0)
    )
    pre_drop_contrast = float(
        np.clip(
            (
                (0.40 * max(0.0, post_rms - immediate_rms))
                + (0.35 * max(0.0, post_low - immediate_low))
                + (0.15 * max(0.0, post_onset - pre_onset))
                + (0.10 * max(0.0, post_rms - pre_rms))
            )
            * 1.35,
            0.0,
            1.0,
        )
    )
    groove_stability = _post_bar_groove_stability(features, t, bars=float(cfg.full_groove_post_bars))
    early_start = max(0.0, t - pre)
    early_end = max(early_start, t - (1.5 * bar))
    early_low = _mean_window(features.low_energy, features.frame_times, early_start, early_end, fallback=pre_low)
    early_rms = _mean_window(features.rms, features.frame_times, early_start, early_end, fallback=pre_rms)
    early_flux = _mean_window(features.spectral_flux, features.frame_times, early_start, early_end)
    immediate_flux = _mean_window(features.spectral_flux, features.frame_times, t - bar, t, fallback=early_flux)
    buildup_low_energy_score = float(
        np.clip(
            (0.52 * max(0.0, post_low - immediate_low))
            + (0.30 * max(0.0, post_rms - immediate_rms))
            + (0.18 * (1.0 - np.clip((0.62 * immediate_low) + (0.38 * immediate_rms), 0.0, 1.0))),
            0.0,
            1.0,
        )
    )
    buildup_ramp_score = float(
        np.clip(
            (0.40 * max(0.0, immediate_flux - early_flux))
            + (0.34 * max(0.0, immediate_rms - early_rms))
            + (0.26 * max(0.0, immediate_low - early_low)),
            0.0,
            1.0,
        )
    )
    drop_impact_score = float(
        np.clip(
            (0.30 * max(0.0, post_rms - immediate_rms))
            + (0.30 * max(0.0, post_low - immediate_low))
            + (0.18 * spectral_flux_peak)
            + (0.14 * drum_onset_spike)
            + (0.08 * first_beat_activity),
            0.0,
            1.0,
        )
    )
    beat_pulses = [
        _max_window(features.low_jump_curve, features.frame_times, t + (idx * beat), max(0.10, 0.18 * beat))
        for idx in range(4)
    ]
    kick_reentry_score = float(
        np.clip(
            (0.34 * _max_window(features.low_energy, features.frame_times, t, max(0.12, 0.20 * beat)))
            + (0.38 * float(np.mean(beat_pulses) if beat_pulses else 0.0))
            + (0.18 * float(max(beat_pulses) if beat_pulses else 0.0))
            + (0.10 * first_beat_density),
            0.0,
            1.0,
        )
    )
    buildup_drop_score = float(
        np.clip(
            (0.24 * buildup_low_energy_score)
            + (0.13 * buildup_ramp_score)
            + (0.28 * drop_impact_score)
            + (0.25 * kick_reentry_score)
            + (0.10 * groove_stability),
            0.0,
            1.0,
        )
    )
    sustained_full_groove_score = float(
        np.clip(
            (0.19 * float(low_end_jump))
            + (0.14 * drum_onset_spike)
            + (0.11 * rms_jump)
            + (0.11 * spectral_flux_peak)
            + (0.08 * pre_drop_contrast)
            + (0.09 * immediate_groove_start_score)
            + (0.12 * groove_stability)
            + (0.16 * buildup_drop_score),
            0.0,
            1.0,
        )
    )
    return {
        "drum_onset_spike": float(drum_onset_spike),
        "rms_jump": float(rms_jump),
        "spectral_flux_peak": float(spectral_flux_peak),
        "pre_drop_contrast": float(pre_drop_contrast),
        "immediate_groove_start_score": float(immediate_groove_start_score),
        "groove_stability": float(groove_stability),
        "buildup_low_energy_score": float(buildup_low_energy_score),
        "buildup_ramp_score": float(buildup_ramp_score),
        "drop_impact_score": float(drop_impact_score),
        "kick_reentry_score": float(kick_reentry_score),
        "buildup_drop_score": float(buildup_drop_score),
        "sustained_full_groove_score": float(sustained_full_groove_score),
    }


def score_candidates(
    features: FeatureBundle,
    peak_times: Sequence[float],
    cfg: Optional[DropDetectorConfig] = None,
    *,
    drumprint_analysis: Optional[DrumprintAnalysis] = None,
    drumprint_status: str = "disabled",
) -> List[DropCandidate]:
    cfg = cfg or DropDetectorConfig()
    beat = features.beat_sec
    bar = 4.0 * beat
    pre = max(beat, float(cfg.pre_bars) * bar)
    post = max(beat, float(cfg.post_bars) * bar)
    candidates: List[DropCandidate] = []
    max_time = features.duration_sec * float(cfg.max_drop_time_ratio)

    for t in peak_times:
        t = float(t)
        if t < float(cfg.min_drop_time_sec) or t > max_time:
            continue

        transient_strength = _max_window(features.combined_attack, features.frame_times, t, 0.15 * beat)
        pre_low = _mean_window(features.low_energy, features.frame_times, t - pre, t)
        post_low = _mean_window(features.low_energy, features.frame_times, t, t + post)
        low_end_jump = float(np.clip((post_low - pre_low + 0.25) / 1.25, 0.0, 1.0))

        pre_rms = _mean_window(features.rms, features.frame_times, t - pre, t)
        post_rms = _mean_window(features.rms, features.frame_times, t, t + post)
        pre_post_energy_ratio = float(post_rms / max(1e-6, pre_rms))
        energy_contrast = float(np.clip((post_rms - pre_rms + 0.25) / 1.25, 0.0, 1.0))

        post_onset = _mean_window(features.onset, features.frame_times, t, t + post)
        density_peaks, _ = signal.find_peaks(
            features.onset[_window_indices(features.frame_times, t, t + post)[0] : _window_indices(features.frame_times, t, t + post)[1]],
            distance=max(1, int(round(0.35 * beat / max(1e-6, features.frame_times[1] - features.frame_times[0]))) if len(features.frame_times) > 1 else 1),
            height=0.30,
        )
        expected_hits = max(1.0, post / max(1e-6, beat))
        post_drop_density = float(np.clip((len(density_peaks) / expected_hits) * 0.65 + post_onset * 0.35, 0.0, 1.0))

        rhythmic_consistency = _rhythmic_consistency(features, t, t + post)
        full_groove = _full_groove_transition_features(
            features,
            t,
            cfg,
            low_end_jump=float(low_end_jump),
        )
        sustained_full_groove_score = float(full_groove.get("sustained_full_groove_score", 0.0) or 0.0)
        buildup_drop_score = float(full_groove.get("buildup_drop_score", 0.0) or 0.0)
        drumprint: Dict[str, object]
        if drumprint_analysis is not None:
            drumprint = score_candidate_drumprint(
                drumprint_analysis,
                t,
                transient_strength=float(transient_strength),
            )
        else:
            drumprint = empty_drumprint_features(enabled=drumprint_status != "disabled", status=drumprint_status)

        if drumprint_analysis is not None and drumprint.get("status") == "ok":
            drumprint_pattern_score = float(drumprint.get("drumprint_pattern_score", 0.0) or 0.0)
            fake_hit_penalty = float(drumprint.get("fake_hit_penalty", 0.0) or 0.0)
            pre_post_energy_ratio_score = float(np.clip(pre_post_energy_ratio / 2.5, 0.0, 1.0))
            score = (
                (0.22 * transient_strength)
                + (0.18 * low_end_jump)
                + (0.15 * post_drop_density)
                + (0.12 * pre_post_energy_ratio_score)
                + (0.10 * rhythmic_consistency)
                + (0.23 * drumprint_pattern_score)
                - (0.15 * fake_hit_penalty)
            )
            score = float(np.clip((0.72 * score) + (0.20 * sustained_full_groove_score) + (0.08 * buildup_drop_score), 0.0, 1.0))
        else:
            pre_post_energy_ratio_score = float(np.clip(pre_post_energy_ratio / 2.5, 0.0, 1.0))
            base_score = (
                (0.35 * transient_strength)
                + (0.25 * low_end_jump)
                + (0.20 * post_drop_density)
                + (0.15 * energy_contrast)
                + (0.05 * rhythmic_consistency)
            )
            score = float(np.clip((0.64 * base_score) + (0.26 * sustained_full_groove_score) + (0.10 * buildup_drop_score), 0.0, 1.0))

        rejected = False
        reasons: List[str] = []
        if transient_strength < 0.25:
            rejected = True
            reasons.append("weak_transient")
        if post_drop_density < float(cfg.min_post_density):
            rejected = True
            reasons.append("low_post_density")
        if (post_rms - pre_rms) < float(cfg.min_energy_contrast):
            rejected = True
            reasons.append("low_energy_contrast")
        if (post_low - pre_low) < float(cfg.min_low_end_jump):
            rejected = True
            reasons.append("no_low_end_jump")
        if t < float(cfg.min_drop_time_sec):
            rejected = True
            reasons.append("too_early")

        candidates.append(
            DropCandidate(
                time_sec=t,
                snapped_sec=t,
                score=float(score),
                transient_strength=float(transient_strength),
                low_end_jump=float(low_end_jump),
                post_drop_density=float(post_drop_density),
                pre_post_energy_ratio=float(pre_post_energy_ratio),
                energy_contrast=float(energy_contrast),
                rhythmic_consistency=float(rhythmic_consistency),
                rejected=bool(rejected),
                rejection_reason=",".join(reasons),
                drumprint=drumprint,
                full_groove=full_groove,
                debug={
                    "pre_low": float(pre_low),
                    "post_low": float(post_low),
                    "pre_rms": float(pre_rms),
                    "post_rms": float(post_rms),
                    "pre_post_energy_ratio_score": float(pre_post_energy_ratio_score),
                    "sustained_full_groove_score": float(sustained_full_groove_score),
                },
            )
        )
    return candidates


def _looks_like_drums_stem(audio_path: str) -> bool:
    name = Path(audio_path).name.lower()
    return name.startswith("drums") or name.startswith("drum_") or name.startswith("drum-")


def _should_use_drumprint(audio_path: str, cfg: DropDetectorConfig) -> bool:
    if cfg.use_drumprint is not None:
        return bool(cfg.use_drumprint)
    return _looks_like_drums_stem(audio_path)


def _chosen_drumprint_summary(chosen: DropCandidate) -> Dict[str, object]:
    drumprint = dict(chosen.drumprint or {})
    out: Dict[str, object] = {}
    for key in DRUMPRINT_FEATURE_KEYS:
        out[f"chosen_{key}"] = float(drumprint.get(key, 0.0) or 0.0)
    out["chosen_drumprint_status"] = str(drumprint.get("status", "missing"))
    out["chosen_drumprint_enabled"] = bool(drumprint.get("enabled", False))
    return out


def _chosen_full_groove_summary(chosen: DropCandidate) -> Dict[str, object]:
    full_groove = dict(chosen.full_groove or {})
    out: Dict[str, object] = {}
    for key in FULL_GROOVE_FEATURE_KEYS:
        out[f"chosen_{key}"] = float(full_groove.get(key, 0.0) or 0.0)
    return out


def _chosen_microalign_summary(chosen: DropCandidate) -> Dict[str, object]:
    micro = dict(chosen.microalign or {})
    out: Dict[str, object] = {}
    for key in MICROALIGN_FEATURE_KEYS:
        out[f"chosen_{key}"] = float(micro.get(key, 0.0) or 0.0)
    out["chosen_microalign_reason"] = str(micro.get("reason", ""))
    out["chosen_microalign_review_needed"] = bool(micro.get("review_needed", False))
    return out


def snap_to_attack(y: np.ndarray, sr: int, peak_sec: float, cfg: Optional[DropDetectorConfig] = None) -> float:
    cfg = cfg or DropDetectorConfig()
    y = np.asarray(y, dtype=np.float32)
    peak_idx = int(round(float(peak_sec) * float(sr)))
    peak_idx = int(np.clip(peak_idx, 0, max(0, len(y) - 1)))
    back = max(1, int(round(float(cfg.snap_back_ms) * 0.001 * sr)))
    fwd = max(1, int(round(float(cfg.snap_forward_ms) * 0.001 * sr)))
    i0 = max(0, peak_idx - back)
    i1 = min(len(y), peak_idx + fwd)
    if i1 <= i0 + 4:
        return float(peak_idx / float(sr))

    env_win = max(1, int(round(0.0025 * sr)))
    kernel = np.ones(env_win, dtype=np.float32) / float(env_win)
    env = np.convolve(np.abs(y), kernel, mode="same")
    local = env[i0:i1]
    pre = env[max(0, i0 - back) : i0] if i0 > 0 else local[: max(1, len(local) // 4)]
    base = float(np.percentile(pre, 60.0)) if pre.size else float(np.percentile(local, 20.0))
    high = float(np.percentile(local, 98.0))
    threshold = base + (0.10 * max(0.0, high - base))
    hold = max(1, int(round(0.003 * sr)))

    attack_idx = peak_idx
    for idx in range(i0, max(i0, i1 - hold)):
        seg = env[idx : idx + hold]
        if seg.size < hold:
            break
        if float(np.mean(seg)) >= threshold and float(np.max(seg)) > base:
            attack_idx = idx
            break

    z_back = max(1, int(round(float(cfg.zero_crossing_back_ms) * 0.001 * sr)))
    z0 = max(1, attack_idx - z_back)
    best = attack_idx
    best_abs = abs(float(y[best])) if len(y) else 0.0
    for idx in range(attack_idx, z0, -1):
        v0 = float(y[idx - 1])
        v1 = float(y[idx])
        if abs(v1) < best_abs:
            best = idx
            best_abs = abs(v1)
        if (v0 <= 0.0 <= v1) or (v0 >= 0.0 >= v1):
            best = idx
            break
    return float(best / float(sr))


def _choose_true_first_drop(candidates: Sequence[DropCandidate], cfg: DropDetectorConfig) -> DropCandidate:
    valid = [c for c in candidates if not c.rejected and c.score >= float(cfg.min_score)]
    if not valid:
        valid = [c for c in candidates if not c.rejected]
    if not valid:
        if not candidates:
            raise RuntimeError("No drop candidates were found.")
        return max(candidates, key=lambda c: c.score)

    def reentry_score(candidate: DropCandidate) -> float:
        full_groove = candidate.full_groove or {}
        kick = float(full_groove.get("kick_reentry_score", 0.0) or 0.0)
        buildup = float(full_groove.get("buildup_drop_score", 0.0) or 0.0)
        sustained = float(full_groove.get("sustained_full_groove_score", 0.0) or 0.0)
        impact = float(full_groove.get("drop_impact_score", 0.0) or 0.0)
        return float(np.clip((0.38 * kick) + (0.26 * buildup) + (0.22 * sustained) + (0.14 * impact), 0.0, 1.0))

    # If any candidate clearly contains the EDM drop signature, choose the first
    # near-best candidate from that pool instead of letting buildup transients win.
    strong_reentry = [
        c
        for c in valid
        if float((c.full_groove or {}).get("kick_reentry_score", 0.0) or 0.0) >= 0.82
        and reentry_score(c) >= 0.46
    ]
    pool = strong_reentry or valid
    best = max(valid, key=lambda c: c.score)
    if strong_reentry:
        best = max(pool, key=lambda c: (c.score, reentry_score(c)))
        threshold_ratio = min(float(cfg.first_drop_near_best_ratio), 0.86)
    else:
        threshold_ratio = float(cfg.first_drop_near_best_ratio)
    threshold = max(float(cfg.min_score), float(best.score) * threshold_ratio)
    first_plausible = [c for c in pool if c.score >= threshold]
    first_plausible.sort(key=lambda c: (c.time_sec, -c.score))
    return first_plausible[0] if first_plausible else best


def _snap_and_rank_candidates(candidates: Sequence[DropCandidate], features: FeatureBundle, cfg: DropDetectorConfig) -> List[DropCandidate]:
    for candidate in candidates:
        snapped = snap_to_attack(features.y, features.sr, candidate.time_sec, cfg)
        candidate.snapped_sec = float(snapped)
        candidate.snap_offset_sec = float(snapped - candidate.time_sec)
    ranked = sorted(candidates, key=lambda c: (-c.score, c.time_sec))
    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank
        candidate.handcrafted_rank = rank
        candidate.model_rank = 0
        candidate.model_score = None
    return ranked


def _apply_microalignment(
    ranked_candidates: Sequence[DropCandidate],
    features: FeatureBundle,
    cfg: DropDetectorConfig,
    *,
    limit: int = 10,
) -> List[DropCandidate]:
    if not cfg.use_microalign:
        return list(ranked_candidates)

    out = list(ranked_candidates)
    for candidate in out[: max(1, int(limit))]:
        try:
            micro = microalign_marker(features.audio_path, candidate.snapped_sec)
            micro["ok"] = True
        except Exception as exc:
            micro = {
                "ok": False,
                "error": str(exc) or exc.__class__.__name__,
                "input_candidate_time": float(candidate.snapped_sec),
                "microaligned_time": float(candidate.snapped_sec),
                "micro_confidence": 0.0,
                "snap_offset_ms": 0.0,
                "reason": "microalignment failed; original marker kept",
                "review_needed": True,
            }
        candidate.microalign = micro
        confidence = float(micro.get("micro_confidence", 0.0) or 0.0)
        offset_ms = abs(float(micro.get("snap_offset_ms", 0.0) or 0.0))
        if confidence >= float(cfg.microalign_confidence_threshold) and offset_ms <= float(cfg.microalign_max_offset_ms):
            previous = float(candidate.snapped_sec)
            candidate.snapped_sec = float(micro.get("microaligned_time", candidate.snapped_sec))
            candidate.snap_offset_sec = float(candidate.snapped_sec - candidate.time_sec)
            candidate.debug["pre_microalign_snapped_sec"] = previous
            candidate.debug["microalign_applied"] = 1.0
        else:
            candidate.debug["microalign_applied"] = 0.0

    merged: List[DropCandidate] = []
    for candidate in out:
        micro = candidate.microalign or {}
        can_merge = bool(micro.get("ok")) and not bool(micro.get("review_needed"))
        if not can_merge:
            merged.append(candidate)
            continue
        duplicate_index = next(
            (
                idx
                for idx, existing in enumerate(merged)
                if abs(float(existing.snapped_sec) - float(candidate.snapped_sec)) <= 0.008
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        existing_conf = float((existing.microalign or {}).get("micro_confidence", 0.0) or 0.0)
        candidate_conf = float(micro.get("micro_confidence", 0.0) or 0.0)
        if (candidate.score, candidate_conf) > (existing.score, existing_conf):
            merged[duplicate_index] = candidate

    return sorted(merged, key=lambda c: (-c.score, c.time_sec))


def _choose_model_ranked_drop(
    ranked_candidates: Sequence[DropCandidate],
    cfg: DropDetectorConfig,
) -> Tuple[Optional[DropCandidate], Dict[str, object]]:
    if not cfg.use_ranker_model:
        return None, {"selected_by": "handcrafted", "ranker_status": "disabled"}

    try:
        payload = load_ranker_payload(cfg.ranker_model_path)
    except Exception as exc:
        return None, {"selected_by": "handcrafted", "ranker_status": "load_failed", "ranker_error": str(exc)}

    if payload is None:
        return None, {"selected_by": "handcrafted", "ranker_status": "missing"}

    top_n = max(1, int(cfg.ranker_top_n))
    top = list(ranked_candidates[:top_n])
    eligible = [candidate for candidate in top if not candidate.rejected] or top
    if not eligible:
        return None, {"selected_by": "handcrafted", "ranker_status": "no_candidates", "ranker_model_path": payload.get("path")}

    try:
        distances = predict_candidate_distances(payload, eligible)
    except Exception as exc:
        return None, {"selected_by": "handcrafted", "ranker_status": "predict_failed", "ranker_error": str(exc)}

    for candidate, distance in zip(eligible, distances):
        candidate.model_score = float(distance)

    model_ranked = sorted(eligible, key=lambda c: (float("inf") if c.model_score is None else c.model_score, c.handcrafted_rank, c.time_sec))
    for model_rank, candidate in enumerate(model_ranked, start=1):
        candidate.model_rank = model_rank

    chosen = model_ranked[0]
    return chosen, {
        "selected_by": "model",
        "ranker_status": "used",
        "ranker_model_path": str(payload.get("path", "")),
        "ranker_model_type": str(payload.get("model_type", type(payload.get("model")).__name__)),
        "ranker_training_rows": int(payload.get("training_rows", 0) or 0),
        "ranker_correction_rows": int(payload.get("correction_rows", 0) or 0),
        "chosen_model_score": float(chosen.model_score if chosen.model_score is not None else 0.0),
    }


def _load_region_payload(cfg: DropDetectorConfig) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    if not cfg.use_region_model:
        return None, {"region_model_status": "disabled"}
    try:
        payload = load_region_model_payload(cfg.region_model_path)
    except Exception as exc:
        return None, {"region_model_status": "load_failed", "region_model_error": str(exc)}
    if payload is None:
        return None, {"region_model_status": "missing"}
    return payload, {
        "region_model_status": "loaded",
        "region_model_path": str(payload.get("path", "")),
        "region_model_type": str(payload.get("model_type", type(payload.get("model")).__name__)),
        "region_model_training_rows": int(payload.get("training_rows", 0) or 0),
        "region_model_correction_rows": int(payload.get("correction_rows", 0) or 0),
    }


def _apply_region_model_ranker(
    ranked_candidates: Sequence[DropCandidate],
    features: FeatureBundle,
    payload: Mapping[str, object],
    cfg: DropDetectorConfig,
) -> Tuple[List[DropCandidate], Dict[str, object]]:
    candidates = list(ranked_candidates)
    if not candidates:
        return candidates, {"region_model_status": "no_candidates"}
    try:
        predictions = predict_region_errors(payload, candidates, duration_sec=features.duration_sec)
    except Exception as exc:
        return candidates, {"region_model_status": "predict_failed", "region_model_error": str(exc)}
    if not predictions:
        return candidates, {"region_model_status": "no_predictions"}

    by_index = {int(row["index"]): row for row in predictions}
    predicted_order = sorted(
        predictions,
        key=lambda row: (
            float(row["predicted_abs_error_sec"]),
            int(getattr(row["candidate"], "handcrafted_rank", 999) or 999),
            float(row["time_sec"]),
        ),
    )
    for region_rank, row in enumerate(predicted_order, start=1):
        candidate = row["candidate"]
        if not isinstance(candidate, DropCandidate):
            continue
        candidate.region_model_rank = int(region_rank)
        candidate.region_model_score = float(row["predicted_abs_error_sec"])
        candidate.region_model_confidence = float(row.get("region_score", 0.0) or 0.0)

    region_ids = {id(row["candidate"]) for row in predicted_order}
    ordered = [row["candidate"] for row in predicted_order if isinstance(row["candidate"], DropCandidate)]
    ordered.extend(candidate for candidate in candidates if id(candidate) not in region_ids)
    top_n = max(1, int(cfg.region_model_top_n))
    chosen = ordered[0]
    second_score = None
    if len(ordered) > 1 and ordered[1].region_model_score is not None:
        second_score = float(ordered[1].region_model_score)
    gap = None
    if chosen.region_model_score is not None and second_score is not None:
        gap = max(0.0, second_score - float(chosen.region_model_score))
    return ordered[:top_n] + ordered[top_n:], {
        "region_model_status": "used",
        "region_model_candidate_count": int(len(candidates)),
        "region_model_ranked_count": int(len(predicted_order)),
        "region_model_top_n": int(top_n),
        "region_model_chosen_time": float(chosen.snapped_sec),
        "region_model_chosen_coarse_time": float(chosen.time_sec),
        "region_model_chosen_predicted_error_sec": None
        if chosen.region_model_score is None
        else float(chosen.region_model_score),
        "region_model_score_gap": None if gap is None else float(gap),
    }


def _final_ranked_candidates(ranked_candidates: Sequence[DropCandidate], selected_by: str) -> List[DropCandidate]:
    if selected_by == "model":
        top_with_model = [c for c in ranked_candidates if c.model_rank > 0]
        model_order = sorted(top_with_model, key=lambda c: (c.model_rank, c.handcrafted_rank))
        model_ids = {id(c) for c in model_order}
        remainder = [c for c in ranked_candidates if id(c) not in model_ids]
        ordered = model_order + remainder
        for rank, candidate in enumerate(ordered, start=1):
            candidate.rank = rank
        return ordered
    if selected_by == "region_model":
        top_with_region = [c for c in ranked_candidates if c.region_model_rank > 0]
        region_order = sorted(top_with_region, key=lambda c: (c.region_model_rank, c.handcrafted_rank))
        region_ids = {id(c) for c in region_order}
        remainder = [c for c in ranked_candidates if id(c) not in region_ids]
        ordered = region_order + remainder
        for rank, candidate in enumerate(ordered, start=1):
            candidate.rank = rank
        return ordered
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate.rank = rank
    return list(ranked_candidates)


def _annotate_candidate_reasons(
    candidates: Sequence[DropCandidate],
    chosen: DropCandidate,
    cfg: DropDetectorConfig,
    selected_by: str,
) -> None:
    valid = [c for c in candidates if not c.rejected]
    best_valid_score = max((c.score for c in valid), default=max((c.score for c in candidates), default=0.0))
    near_best = max(float(cfg.min_score), float(best_valid_score) * float(cfg.first_drop_near_best_ratio))

    for candidate in candidates:
        candidate.selected_by = selected_by
        if candidate is chosen:
            candidate.selected = True
            if selected_by == "model":
                candidate.selection_reason = "selected_by_model:lowest_predicted_user_delta"
            elif selected_by == "region_model":
                candidate.selection_reason = "selected_by_region_model:lowest_predicted_drop_region_error"
            elif candidate.handcrafted_rank == 1:
                candidate.selection_reason = "selected:highest_scoring_valid_sustained_impact"
            else:
                candidate.selection_reason = "selected:first_plausible_drop_near_best_score"
            continue

        candidate.selected = False
        if candidate.rejected:
            candidate.selection_reason = f"rejected:{candidate.rejection_reason or 'failed_drop_filters'}"
        elif selected_by == "model" and candidate.model_rank > 0:
            candidate.selection_reason = "not_selected:model_predicted_larger_user_delta"
        elif selected_by == "region_model" and candidate.region_model_rank > 0:
            candidate.selection_reason = "not_selected:region_model_predicted_larger_drop_region_error"
        elif candidate.score < float(cfg.min_score):
            candidate.selection_reason = "not_selected:below_minimum_confidence"
        elif candidate.score >= near_best and candidate.time_sec > chosen.time_sec:
            candidate.selection_reason = "not_selected:later_than_first_plausible_drop"
        elif candidate.score >= near_best and candidate.time_sec < chosen.time_sec:
            candidate.selection_reason = "not_selected:earlier_candidate_lacked_final_drop_priority"
        elif candidate.score < chosen.score:
            candidate.selection_reason = "not_selected:lower_composite_score"
        else:
            candidate.selection_reason = "not_selected:valid_but_not_first_true_drop"


def _score_gap(ranked_candidates: Sequence[DropCandidate]) -> float:
    valid = [candidate for candidate in ranked_candidates if not candidate.rejected]
    if len(valid) < 2:
        return 1.0 if valid else 0.0
    return float(max(0.0, valid[0].score - valid[1].score))


def _model_gap(ranked_candidates: Sequence[DropCandidate]) -> Optional[float]:
    model_ranked = [candidate for candidate in ranked_candidates if candidate.model_rank > 0 and candidate.model_score is not None]
    if len(model_ranked) < 2:
        return None
    model_ranked.sort(key=lambda c: c.model_rank)
    return float(max(0.0, float(model_ranked[1].model_score) - float(model_ranked[0].model_score)))


def _candidate_density(candidates: Sequence[DropCandidate], chosen: DropCandidate, radius_sec: float = 1.5) -> int:
    center = float(chosen.snapped_sec)
    return int(sum(1 for candidate in candidates[:10] if abs(float(candidate.snapped_sec) - center) <= float(radius_sec)))


def _calibrate_confidence_tier(
    *,
    chosen: DropCandidate,
    ranked_candidates: Sequence[DropCandidate],
    selected_by: str,
) -> Tuple[str, Dict[str, object]]:
    top_score = float(chosen.score)
    gap = _score_gap(ranked_candidates)
    model_gap = _model_gap(ranked_candidates)
    model_distance = chosen.model_score if chosen.model_score is not None else None
    region_distance = chosen.region_model_score if chosen.region_model_score is not None else None
    density = _candidate_density(ranked_candidates, chosen)
    handcrafted_model_agree = bool(
        chosen.handcrafted_rank == 1
        and (selected_by != "model" or chosen.model_rank == 1)
    )

    calibration = 0
    if top_score >= 0.76:
        calibration += 2
    elif top_score >= 0.66:
        calibration += 1
    elif top_score < 0.52:
        calibration -= 2
    elif top_score < 0.60:
        calibration -= 1

    if gap >= 0.08:
        calibration += 2
    elif gap >= 0.04:
        calibration += 1
    elif gap < 0.02:
        calibration -= 1

    if model_distance is not None:
        model_distance = float(model_distance)
        if model_distance <= 0.025:
            calibration += 2
        elif model_distance <= 0.075:
            calibration += 1
        elif model_distance > 0.50:
            calibration -= 2
        elif model_distance > 0.20:
            calibration -= 1

    if model_gap is not None:
        if model_gap >= 0.050:
            calibration += 1
        elif model_gap < 0.010:
            calibration -= 1

    if region_distance is not None:
        region_distance = float(region_distance)
        if region_distance <= 0.050:
            calibration += 2
        elif region_distance <= 0.150:
            calibration += 1
        elif region_distance > 1.00:
            calibration -= 2
        elif region_distance > 0.40:
            calibration -= 1

    calibration += 1 if handcrafted_model_agree else -1

    if density <= 2:
        calibration += 1
    elif density >= 5:
        calibration -= 1

    if calibration >= 4:
        tier = "HIGH"
    elif calibration >= 1:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return tier, {
        "confidence_tier": tier,
        "confidence_calibration_score": int(calibration),
        "confidence_top_candidate_score": top_score,
        "confidence_rank_gap": float(gap),
        "confidence_model_gap": None if model_gap is None else float(model_gap),
        "confidence_model_predicted_distance": None if model_distance is None else float(model_distance),
        "confidence_region_predicted_distance": None if region_distance is None else float(region_distance),
        "confidence_handcrafted_model_agree": bool(handcrafted_model_agree),
        "confidence_candidate_density": int(density),
    }


def detect_drop(
    audio_path: str,
    *,
    bpm: Optional[float] = None,
    external_cues: Optional[Sequence[float]] = None,
    config: Optional[DropDetectorConfig] = None,
    analysis_json: Optional[str] = None,
) -> DropDetectionResult:
    cfg = config or DropDetectorConfig()
    features = extract_features(audio_path, cfg, bpm=bpm)
    region_payload, region_summary = _load_region_payload(cfg)
    centers = _region_centers(features, cfg, external_cues)
    if region_payload is not None and cfg.region_model_full_track:
        peak_times = _full_track_candidate_peak_times(features, cfg, external_cues=external_cues)
        if not peak_times:
            peak_times = _candidate_peak_times(features, cfg, centers)
    else:
        peak_times = _candidate_peak_times(features, cfg, centers)
    transition_peak_times = _edm_transition_candidate_times(features, cfg, external_cues=external_cues)
    if transition_peak_times:
        peak_times = sorted(set(round(float(t), 6) for t in list(peak_times) + transition_peak_times))
    drumprint_analysis: Optional[DrumprintAnalysis] = None
    drumprint_summary: Dict[str, object] = {
        "drumprint_enabled": False,
        "drumprint_status": "disabled",
    }
    drumprint_status = "disabled"
    if _should_use_drumprint(audio_path, cfg):
        drumprint_summary["drumprint_enabled"] = True
        try:
            drumprint_analysis = build_drumprint_analysis(features.y, features.sr, features.bpm)
            drumprint_summary = drumprint_analysis.summary()
            drumprint_status = "ok"
        except Exception as exc:
            drumprint_status = "failed"
            drumprint_summary = {
                "drumprint_enabled": True,
                "drumprint_status": "failed",
                "drumprint_error": str(exc) or exc.__class__.__name__,
            }

    candidates = score_candidates(
        features,
        peak_times,
        cfg,
        drumprint_analysis=drumprint_analysis,
        drumprint_status=drumprint_status,
    )
    handcrafted_ranked = _snap_and_rank_candidates(candidates, features, cfg)
    handcrafted_ranked = _apply_microalignment(handcrafted_ranked, features, cfg, limit=max(10, int(cfg.ranker_top_n)))
    if region_payload is not None:
        handcrafted_ranked, region_rank_summary = _apply_region_model_ranker(
            handcrafted_ranked,
            features,
            region_payload,
            cfg,
        )
        region_summary.update(region_rank_summary)
    candidates = list(handcrafted_ranked)
    handcrafted_chosen = _choose_true_first_drop(candidates, cfg)
    model_chosen, ranker_summary = _choose_model_ranked_drop(handcrafted_ranked, cfg)
    region_chosen = next((candidate for candidate in handcrafted_ranked if candidate.region_model_rank == 1), None)
    selected_by = str(ranker_summary.get("selected_by", "handcrafted"))
    if model_chosen is not None:
        chosen = model_chosen
    elif region_chosen is not None:
        selected_by = "region_model"
        chosen = region_chosen
    else:
        chosen = handcrafted_chosen
    _annotate_candidate_reasons(candidates, chosen, cfg, selected_by)
    ranked = _final_ranked_candidates(handcrafted_ranked, selected_by)
    confidence_tier, confidence_summary = _calibrate_confidence_tier(
        chosen=chosen,
        ranked_candidates=ranked,
        selected_by=selected_by,
    )

    confidence = float(np.clip((chosen.score - cfg.min_score) / max(1e-6, 1.0 - cfg.min_score), 0.0, 1.0))
    result_candidates = list(ranked[:50])
    if chosen not in result_candidates:
        result_candidates.append(chosen)
        result_candidates.sort(key=lambda c: (-c.score, c.time_sec))
    result = DropDetectionResult(
        audio_path=str(audio_path),
        drop_sec=float(chosen.snapped_sec),
        coarse_sec=float(chosen.time_sec),
        bpm=float(features.bpm),
        confidence=confidence,
        sample_rate=int(features.sr),
        candidates=result_candidates,
        selected_by=selected_by,
        confidence_tier=confidence_tier,
        features_summary={
            "duration_sec": float(features.duration_sec),
            "bpm": float(features.bpm),
            "region_count": float(len(centers)),
            "full_track_candidate_scan": bool(region_payload is not None and cfg.region_model_full_track),
            "candidate_count": float(len(candidates)),
            "selected_by": selected_by,
            "confidence_tier": confidence_tier,
            "chosen_score": float(chosen.score),
            "chosen_transient_strength": float(chosen.transient_strength),
            "chosen_low_end_jump": float(chosen.low_end_jump),
            "chosen_post_drop_density": float(chosen.post_drop_density),
            "chosen_pre_post_energy_ratio": float(chosen.pre_post_energy_ratio),
            "chosen_energy_contrast": float(chosen.energy_contrast),
            "chosen_rhythmic_consistency": float(chosen.rhythmic_consistency),
            "chosen_snap_offset_sec": float(chosen.snap_offset_sec),
            "chosen_handcrafted_rank": float(chosen.handcrafted_rank),
            "chosen_model_rank": float(chosen.model_rank),
            "chosen_model_score": float(chosen.model_score) if chosen.model_score is not None else -1.0,
            "chosen_region_model_rank": float(chosen.region_model_rank),
            "chosen_region_model_score": float(chosen.region_model_score) if chosen.region_model_score is not None else -1.0,
            "chosen_region_model_confidence": float(chosen.region_model_confidence),
            **drumprint_summary,
            **_chosen_drumprint_summary(chosen),
            **_chosen_full_groove_summary(chosen),
            **_chosen_microalign_summary(chosen),
            **confidence_summary,
            **region_summary,
            **ranker_summary,
        },
    )
    if analysis_json:
        Path(analysis_json).parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=True)
    return result
