#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production-oriented first-downbeat / 1.1.1 detector.

Design goals:
- Drums stem is the source of truth.
- Prefer the first musically meaningful phrase/drop entry, not the first transient.
- Preserve a clean interface for existing Ableton ALS generation code.
- Keep the scoring model inspectable and tunable.

The detector is intentionally hybrid:
- It performs its own multi-stage candidate scoring over a beat grid.
- It optionally uses the existing ``edm_true_drop_detector`` as a prior/fallback.
- It refines the chosen drums anchor at native sample rate.
- It transfers the same musical start to inst/vocals with constrained local search.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from ableton_analysis_adapter import AbletonAnalysisMarkers, extract_ableton_onset_markers
except Exception:
    AbletonAnalysisMarkers = None  # type: ignore[assignment]
    extract_ableton_onset_markers = None

from edm_true_drop_detector import (
    DetectorConfig,
    _aggregate_bar_feature,
    _bar_downbeats,
    _build_beats,
    _count_peaks_bar,
    _decode_audio_ffmpeg,
    _estimate_beat_phase,
    _estimate_tempo_bpm,
    _kick_periodicity_per_bar,
    _moving_average,
    _robust_norm,
    _sample_near_beat,
    _smooth_bars,
    _stft_features,
    _trend_score,
    detect_true_drop,
)

try:
    from alsdrop.als_io import extract_labels_from_als
except Exception:
    extract_labels_from_als = None


LOG = logging.getLogger("first_downbeat_detector")
STEM_FILE_RE = re.compile(r"^(drums|inst|vocals)_(\d{2,3})_([0-9]{1,2}[aAbB])(?:_(\d{1,2}))?[-_].+\.(?:wav|flac|aiff|aif|mp3)$")
MARKER_RANK_MODEL_ENV = "DOWNBEAT_MARKER_RANK_MODEL"
DEFAULT_MARKER_RANK_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downbeat_marker_rank_model.json")
ROUGH_REGION_MODEL_ENV = "DOWNBEAT_ROUGH_REGION_MODEL"
DEFAULT_ROUGH_REGION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rough_region_rank_model.json")
MARKER_MODEL_FEATURES: Tuple[str, ...] = (
    "delta_beats_from_rough",
    "marker_support",
    "attack_edge_norm",
    "attack_started_norm",
    "inside_body_norm",
    "sustain_after_norm",
    "density_after_norm",
    "low_growth_norm",
    "pre_post_ratio_norm",
    "dense_run_after_norm",
    "isolated_preroll_norm",
    "gap_prev_norm",
    "gap_next_norm",
    "event_followthrough_norm",
    "cluster_index_norm",
    "event_score_norm",
    "event_marker_count_norm",
    "event_plausible_count_norm",
)
ROUGH_REGION_MODEL_FEATURES: Tuple[str, ...] = (
    "delta_beats_from_reference",
    "candidate_confidence",
    "candidate_score",
    "rough_followthrough",
    "norm_phrase",
    "norm_density",
    "norm_sustain",
    "norm_lowend",
    "norm_contrast",
    "norm_grid",
    "norm_onset",
    "norm_repeat",
    "norm_preroll",
    "norm_weak",
    "norm_fake",
)


@dataclass
class MarkerRankModel:
    path: str
    feature_order: Tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float
    label_name: str = "candidate_is_manual_match_25ms"
    score_scale: float = 1.0

    def predict_proba(self, feature_values: Dict[str, float]) -> float:
        vec = np.asarray([float(feature_values.get(name, 0.0)) for name in self.feature_order], dtype=np.float64)
        scales = np.where(self.scales == 0.0, 1.0, self.scales)
        z = ((vec - self.means) / scales).dot(self.weights) + float(self.bias)
        z *= float(self.score_scale)
        return float(_sigmoid(float(z)))


@dataclass
class ScoringWeights:
    onset: float = 0.19
    lowend: float = 0.18
    contrast: float = 0.15
    density: float = 0.12
    grid: float = 0.11
    phrase: float = 0.10
    repeat: float = 0.07
    sustain: float = 0.06
    ableton: float = 0.16
    legacy: float = 0.05
    preroll: float = 0.11
    weak: float = 0.08
    fake: float = 0.07


@dataclass
class DetectorOptions:
    analysis_sr: int = 22050
    n_fft: int = 2048
    hop_sec: float = 0.02
    bpm_min: float = 70.0
    bpm_max: float = 180.0
    max_first_downbeat_ratio: float = 0.65
    min_bars_after_candidate: int = 4
    earliest_near_best_margin: float = 0.08
    earliest_near_best_margin_ableton: float = 0.22
    drums_refine_back_beats: float = 0.20
    drums_refine_fwd_beats: float = 0.25
    secondary_search_back_ms: int = 90
    secondary_search_fwd_ms: int = 140
    vocals_search_back_ms: int = 60
    vocals_search_fwd_ms: int = 120
    secondary_min_improvement: float = 0.06
    vocals_min_improvement: float = 0.10
    zero_crossing_back_ms: int = 5
    legacy_high_confidence: float = 0.90
    legacy_guidance_window_beats: float = 6.0
    legacy_override_late_beats: float = 16.0
    legacy_override_score_margin: float = 0.22
    use_ableton_asd_candidates: bool = True
    ableton_candidate_window_beats: float = 0.75
    rough_region_score_margin: float = 0.10
    rough_region_conf_margin: float = 0.05
    rough_region_min_density: float = 0.45
    rough_region_near_best_margin: float = 0.08
    rough_region_min_phrase: float = 0.35
    rough_region_min_followthrough: float = 0.38
    rough_region_ref_density_margin: float = 0.12
    rough_region_ref_lowend_margin: float = 0.12
    ableton_snap_search_beats: float = 2.5
    ableton_snap_max_late_beats: float = 0.85
    ableton_snap_accept_near_custom_beats: float = 0.02
    ableton_cluster_max_spacing_beats: float = 0.18
    ableton_cluster_window_beats: float = 0.90
    ableton_cluster_auto_accept_min_markers: int = 2
    ableton_cluster_earliest_min_support: float = 0.55
    ableton_cluster_max_post_attack_drift_beats: float = 0.20
    ableton_cluster_max_early_pull_beats: float = 0.25
    ableton_event_prefer_earlier_margin: float = 0.08
    ableton_event_min_sustain_norm: float = 0.30
    ableton_event_min_density_norm: float = 0.25
    ableton_marker_prefer_earlier_margin: float = 0.05
    ableton_self_check_margin: float = 0.06
    ableton_attack_start_min_score: float = 0.12
    ableton_previous_marker_promotion_bonus: float = 0.12
    ableton_later_in_cluster_penalty: float = 0.18
    ableton_attack_started_penalty: float = 0.24
    ableton_inside_body_penalty: float = 0.18
    ableton_attack_start_reward: float = 0.16
    ableton_snap_early_bias: float = 0.06
    ableton_snap_min_support: float = 0.55
    ableton_snap_late_section_penalty: float = 0.22
    ableton_snap_near_best_margin: float = 0.035
    ableton_override_earlier_beats: float = 8.0
    ableton_override_min_support: float = 0.85
    ableton_override_conf_tolerance: float = 0.03
    ableton_exact_anchor_prefer_within_beats: float = 0.35
    ableton_exact_anchor_min_event_score: float = 0.24
    ableton_exact_anchor_min_support: float = 0.60
    ableton_attack_start_snap_beats: float = 0.08
    ableton_attack_start_min_event_score: float = 0.45
    ableton_attack_start_min_support: float = 0.72
    ableton_attack_start_min_edge: float = 0.80
    ableton_attack_start_max_inside_body: float = 0.20
    ableton_attack_start_min_followthrough: float = 0.45
    rough_region_model_path: Optional[str] = None
    rough_region_model_blend: float = 0.45
    marker_rank_model_path: Optional[str] = None
    marker_rank_model_blend: float = 0.35
    weights: ScoringWeights = field(default_factory=ScoringWeights)


@dataclass
class StemAudio:
    role: str
    path: str
    native_sr: int
    native_y: np.ndarray
    analysis_sr: int
    analysis_y: np.ndarray
    analysis_offset_samples: int

    @property
    def analysis_offset_seconds(self) -> float:
        return float(self.analysis_offset_samples) / float(self.analysis_sr)

    @property
    def duration_seconds(self) -> float:
        if self.native_sr <= 0:
            return 0.0
        return float(len(self.native_y)) / float(self.native_sr)


@dataclass
class CandidateScore:
    bpm: float
    downbeat_offset: int
    bar_index: int
    beat_index: int
    time_rel_sec: float
    time_abs_sec: float
    raw_onset: float
    raw_lowend: float
    raw_contrast: float
    raw_density: float
    raw_grid: float
    raw_phrase: float
    raw_repeat: float
    raw_sustain: float
    raw_ableton: float
    raw_legacy: float
    raw_preroll: float
    raw_weak: float
    raw_fake: float
    norm_onset: float = 0.0
    norm_lowend: float = 0.0
    norm_contrast: float = 0.0
    norm_density: float = 0.0
    norm_grid: float = 0.0
    norm_phrase: float = 0.0
    norm_repeat: float = 0.0
    norm_sustain: float = 0.0
    norm_ableton: float = 0.0
    norm_legacy: float = 0.0
    norm_preroll: float = 0.0
    norm_weak: float = 0.0
    norm_fake: float = 0.0
    score: float = 0.0
    valid: bool = False
    chosen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoughRegionStage:
    candidate_count: int
    reference_candidate_seconds: Optional[float]
    reference_anchor_seconds: Optional[float]
    reference_confidence: Optional[float]
    rough_candidate_seconds: Optional[float]
    rough_anchor_seconds: Optional[float]
    rough_confidence: Optional[float]
    reason: Optional[str]
    legacy_prior_seconds: Optional[float]
    legacy_prior_confidence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalEventStage:
    source: str
    available: bool
    reason: Optional[str]
    rough_anchor_seconds: Optional[float]
    selected_event_id: Optional[int]
    selected_event_score: Optional[float]
    selected_event_reason: Optional[str]
    cluster_start_seconds: Optional[float]
    cluster_end_seconds: Optional[float]
    cluster_marker_count: int
    cluster_plausible_marker_count: int
    chosen_marker_seconds: Optional[float]
    chosen_candidate_seconds: Optional[float]
    chosen_marker_reason: Optional[str]
    first_plausible_marker_seconds: Optional[float]
    initial_preferred_marker_seconds: Optional[float]
    earliest_near_best_marker_seconds: Optional[float]
    previous_marker_seconds: Optional[float]
    previous_marker_exists: bool
    previous_marker_won: bool
    previous_marker_rejection_reason: Optional[str]
    accepted_as_final: Optional[bool]
    accept_reason: Optional[str]
    rejected_as_final_reason: Optional[str]
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfCheckStage:
    passed: Optional[bool]
    reason: Optional[str]
    chosen_marker_seconds: Optional[float]
    first_plausible_marker_seconds: Optional[float]
    final_is_first_plausible_marker: bool
    late_relative_to_rough: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StemAlignment:
    role: str
    sample_rate: int
    coarse_sample: int
    anchor_sample: int
    anchor_seconds: float
    beat_position: float
    confidence: float
    analysis_offset_samples: int
    search_shift_samples: int = 0
    safe_cut_sample: Optional[int] = None
    inherited_from_drums: bool = False

    def to_output_dict(self) -> Dict[str, Any]:
        base = {
            "sample_rate": int(self.sample_rate),
            "confidence": float(self.confidence),
            "beat_position": float(self.beat_position),
            "analysis_offset_samples": int(self.analysis_offset_samples),
            "coarse_sample": int(self.coarse_sample),
            "search_shift_samples": int(self.search_shift_samples),
            "safe_cut_sample": int(self.safe_cut_sample) if self.safe_cut_sample is not None else None,
            "inherited_from_drums": bool(self.inherited_from_drums),
        }
        if self.role == "drums":
            base.update(
                {
                    "downbeat_sample": int(self.anchor_sample),
                    "downbeat_seconds": float(self.anchor_seconds),
                }
            )
        else:
            base.update(
                {
                    "aligned_sample": int(self.anchor_sample),
                    "aligned_seconds": float(self.anchor_seconds),
                }
            )
        return base


@dataclass
class SongDownbeatResult:
    bpm: float
    drums: StemAlignment
    inst: Optional[StemAlignment]
    vocals: Optional[StemAlignment]
    candidates: List[CandidateScore]
    chosen_reason: str
    custom_candidates: Optional[List[CandidateScore]] = None
    candidate_strategy: str = "custom"
    custom_reference_candidate: Optional[CandidateScore] = None
    rough_custom_candidate: Optional[CandidateScore] = None
    rough_custom_reason: Optional[str] = None
    ableton_snap_debug: Optional[Dict[str, Any]] = None
    rough_region_stage: Optional[RoughRegionStage] = None
    local_event_stage: Optional[LocalEventStage] = None
    self_check_stage: Optional[SelfCheckStage] = None
    candidate_csv: Optional[str] = None
    plots: List[str] = field(default_factory=list)
    legacy_prior_seconds: Optional[float] = None
    legacy_prior_confidence: Optional[float] = None
    legacy_prior_source: Optional[str] = None
    ableton_markers: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        chosen = next((c for c in self.candidates if c.chosen), None)
        return {
            "bpm": float(self.bpm),
            "drums": self.drums.to_output_dict(),
            "inst": self.inst.to_output_dict() if self.inst else None,
            "vocals": self.vocals.to_output_dict() if self.vocals else None,
            "debug": {
                "candidate_samples": [int(round(c.time_abs_sec * self.drums.sample_rate)) for c in self.candidates],
                "candidate_seconds": [float(c.time_abs_sec) for c in self.candidates],
                "candidate_scores": [float(c.score) for c in self.candidates],
                "custom_candidate_seconds": [float(c.time_abs_sec) for c in (self.custom_candidates or [])],
                "custom_candidate_scores": [float(c.score) for c in (self.custom_candidates or [])],
                "custom_candidates": [c.to_dict() for c in (self.custom_candidates or [])],
                "chosen_reason": str(self.chosen_reason),
                "chosen_candidate": chosen.to_dict() if chosen else None,
                "candidate_strategy": str(self.candidate_strategy),
                "custom_reference_candidate": self.custom_reference_candidate.to_dict() if self.custom_reference_candidate else None,
                "rough_custom_candidate": self.rough_custom_candidate.to_dict() if self.rough_custom_candidate else None,
                "rough_custom_reason": str(self.rough_custom_reason) if self.rough_custom_reason else None,
                "ableton_snap": dict(self.ableton_snap_debug or {}) if self.ableton_snap_debug else None,
                "stages": {
                    "rough_region": self.rough_region_stage.to_dict() if self.rough_region_stage else None,
                    "local_event": self.local_event_stage.to_dict() if self.local_event_stage else None,
                    "self_check": self.self_check_stage.to_dict() if self.self_check_stage else None,
                },
                "legacy_prior_seconds": float(self.legacy_prior_seconds) if self.legacy_prior_seconds is not None else None,
                "legacy_prior_confidence": float(self.legacy_prior_confidence) if self.legacy_prior_confidence is not None else None,
                "legacy_prior_source": str(self.legacy_prior_source) if self.legacy_prior_source else None,
                "ableton_markers": dict(self.ableton_markers or {}) if self.ableton_markers else None,
                "candidate_csv": self.candidate_csv,
                "plots": list(self.plots),
            },
        }


@dataclass
class DiscoveredTrack:
    track_dir: str
    drums_path: str
    inst_path: Optional[str]
    vocals_path: Optional[str]
    bpm: Optional[float]
    camelot_key: Optional[str]
    ch1_als_path: Optional[str]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _moving_average_samples(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32, copy=False)
    kern = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32, copy=False), kern, mode="same").astype(np.float32, copy=False)


def _safe_percentile(x: np.ndarray, q: float, fallback: float = 0.0) -> float:
    if x.size == 0:
        return float(fallback)
    return float(np.percentile(x, q))


def _safe_mean(x: np.ndarray, fallback: float = 0.0) -> float:
    if x.size == 0:
        return float(fallback)
    return float(np.mean(x))


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _rank_norm(values: Sequence[float], invert: bool = False) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [1.0]
    if float(np.max(arr) - np.min(arr)) <= 1e-8:
        base = np.full(arr.size, 0.5, dtype=np.float32)
        if invert:
            base = 1.0 - base
        return [float(v) for v in base.tolist()]
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(arr.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float32)
    if invert:
        ranks = 1.0 - ranks
    return [float(v) for v in ranks.tolist()]


def _probe_audio_info(path: str) -> Tuple[int, Optional[float]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return 44100, None

    sr = 44100
    duration = None
    try:
        streams = data.get("streams") or []
        if streams and isinstance(streams[0], dict):
            sr = int(float(streams[0].get("sample_rate") or 44100))
    except Exception:
        sr = 44100
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except Exception:
        duration = None
    return int(max(1, sr)), duration


def _decode_audio_ffmpeg_mono(path: str, sr: int) -> np.ndarray:
    y = _decode_audio_ffmpeg(path, sr)
    if y is None:
        raise RuntimeError(f"ffmpeg decode failed for {path}")
    return y.astype(np.float32, copy=False)


def _safe_normalize(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32, copy=False)
    peak = float(np.percentile(np.abs(y), 99.8))
    if peak <= 1e-8:
        return y.astype(np.float32, copy=False)
    out = y.astype(np.float32, copy=False) / peak
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _trim_leading_silence_for_analysis(y: np.ndarray, sr: int) -> int:
    if y.size == 0 or sr <= 0:
        return 0
    env = _moving_average_samples(np.abs(y).astype(np.float32, copy=False), max(1, int(round(0.020 * sr))))
    head = env[: min(len(env), int(round(2.0 * sr)))]
    noise = _safe_percentile(head, 25.0, fallback=0.0)
    peak = _safe_percentile(env, 99.7, fallback=0.0)
    if peak <= 1e-6:
        return 0
    thr = max(noise * 3.0, noise + (0.05 * max(0.0, peak - noise)), 2.5e-4)
    hold = max(1, int(round(0.035 * sr)))
    for idx in range(0, max(1, len(env) - hold)):
        seg = env[idx : idx + hold]
        if seg.size < hold:
            break
        if float(np.median(seg)) >= thr:
            return int(max(0, idx - int(round(0.020 * sr))))
    return 0


def _local_peak_near_time(x: np.ndarray, sec: float, hop_sec: float, radius_sec: float) -> float:
    if x.size == 0 or hop_sec <= 0.0:
        return 0.0
    ci = int(round(float(sec) / hop_sec))
    rad = max(1, int(round(radius_sec / hop_sec)))
    a = max(0, ci - rad)
    b = min(len(x), ci + rad + 1)
    if b <= a:
        return 0.0
    return float(np.max(x[a:b]))


def _local_mean(x: np.ndarray, sec: float, width_sec: float, hop_sec: float) -> float:
    if x.size == 0 or hop_sec <= 0.0:
        return 0.0
    i0 = max(0, int(round((float(sec) - width_sec) / hop_sec)))
    i1 = min(len(x), int(round((float(sec) + width_sec) / hop_sec)))
    if i1 <= i0:
        return 0.0
    return _safe_mean(x[i0:i1], fallback=0.0)


def _pattern_repeat_score(low_transient: np.ndarray, onset: np.ndarray, candidate_sec: float, beat_sec: float, hop_sec: float) -> float:
    if beat_sec <= 1e-6 or low_transient.size == 0 or onset.size == 0:
        return 0.0
    vals_a: List[float] = []
    vals_b: List[float] = []
    for k in range(8):
        ta = candidate_sec + (k * beat_sec)
        tb = candidate_sec + ((k + 8) * beat_sec)
        vals_a.append(_local_peak_near_time(low_transient, ta, hop_sec, 0.12 * beat_sec) + (0.45 * _local_peak_near_time(onset, ta, hop_sec, 0.10 * beat_sec)))
        vals_b.append(_local_peak_near_time(low_transient, tb, hop_sec, 0.12 * beat_sec) + (0.45 * _local_peak_near_time(onset, tb, hop_sec, 0.10 * beat_sec)))
    a = np.asarray(vals_a, dtype=np.float32)
    b = np.asarray(vals_b, dtype=np.float32)
    if np.std(a) <= 1e-6 or np.std(b) <= 1e-6:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(corr):
        return 0.0
    return float(max(0.0, corr))


def _estimate_phase_from_markers(marker_times: np.ndarray, beat_sec: float, bins: int = 720) -> float:
    if beat_sec <= 1e-6 or marker_times.size == 0:
        return 0.0
    markers = marker_times[np.isfinite(marker_times)]
    markers = markers[markers >= 0.0]
    if markers.size == 0:
        return 0.0
    markers = markers[: min(int(markers.size), 2048)].astype(np.float64, copy=False)
    phases = np.linspace(0.0, float(beat_sec), num=max(24, int(bins)), endpoint=False, dtype=np.float64)
    tol = max(1e-6, 0.07 * float(beat_sec))
    best_phase = 0.0
    best_score = -1.0
    for phase in phases:
        rem = np.mod(markers - phase, float(beat_sec))
        dist = np.minimum(rem, float(beat_sec) - rem)
        score = float(np.sum(np.exp(-dist / tol)))
        if score > best_score:
            best_score = score
            best_phase = float(phase)
    return float(best_phase)


def _find_zero_crossing_before(y: np.ndarray, idx: int, max_back: int) -> int:
    if y.size == 0:
        return idx
    idx = max(1, min(int(idx), len(y) - 1))
    start = max(1, idx - max(1, int(max_back)))
    best = idx
    best_abs = abs(float(y[idx]))
    for i in range(idx, start - 1, -1):
        v0 = float(y[i - 1])
        v1 = float(y[i])
        av = abs(v1)
        if av < best_abs:
            best = i
            best_abs = av
        if (v0 <= 0.0 <= v1) or (v0 >= 0.0 >= v1):
            return i
    return best


class FirstDownbeatDetector:
    def __init__(self, options: Optional[DetectorOptions] = None):
        self.options = options or DetectorOptions()
        self._rough_region_model: Optional[MarkerRankModel] = None
        self._rough_region_model_loaded = False
        self._marker_rank_model: Optional[MarkerRankModel] = None
        self._marker_rank_model_loaded = False

    def _resolve_rough_region_model_path(self) -> Optional[str]:
        explicit = (self.options.rough_region_model_path or "").strip() if self.options.rough_region_model_path else ""
        if explicit:
            return explicit
        env_path = (os.environ.get(ROUGH_REGION_MODEL_ENV) or "").strip()
        if env_path:
            return env_path
        if os.path.exists(DEFAULT_ROUGH_REGION_MODEL_PATH):
            return DEFAULT_ROUGH_REGION_MODEL_PATH
        return None

    def _get_rough_region_model(self) -> Optional[MarkerRankModel]:
        if self._rough_region_model_loaded:
            return self._rough_region_model
        self._rough_region_model_loaded = True
        model_path = self._resolve_rough_region_model_path()
        if not model_path:
            return None
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            feature_order = tuple(str(v) for v in (payload.get("feature_order") or []))
            means = np.asarray(payload.get("means") or [], dtype=np.float64)
            scales = np.asarray(payload.get("scales") or [], dtype=np.float64)
            weights = np.asarray(payload.get("weights") or [], dtype=np.float64)
            bias = float(payload.get("bias") or 0.0)
            if not feature_order or means.size != len(feature_order) or scales.size != len(feature_order) or weights.size != len(feature_order):
                raise ValueError("invalid feature shape")
            self._rough_region_model = MarkerRankModel(
                path=os.path.abspath(model_path),
                feature_order=feature_order,
                means=means,
                scales=scales,
                weights=weights,
                bias=bias,
                label_name=str(payload.get("label_name") or "candidate_is_manual_match_1beat"),
                score_scale=float(payload.get("score_scale") or 1.0),
            )
        except Exception as exc:
            LOG.warning("Could not load rough-region model from %s: %s", model_path, exc)
            self._rough_region_model = None
        return self._rough_region_model

    def _resolve_marker_rank_model_path(self) -> Optional[str]:
        explicit = (self.options.marker_rank_model_path or "").strip() if self.options.marker_rank_model_path else ""
        if explicit:
            return explicit
        env_path = (os.environ.get(MARKER_RANK_MODEL_ENV) or "").strip()
        if env_path:
            return env_path
        if os.path.exists(DEFAULT_MARKER_RANK_MODEL_PATH):
            return DEFAULT_MARKER_RANK_MODEL_PATH
        return None

    def _get_marker_rank_model(self) -> Optional[MarkerRankModel]:
        if self._marker_rank_model_loaded:
            return self._marker_rank_model
        self._marker_rank_model_loaded = True
        model_path = self._resolve_marker_rank_model_path()
        if not model_path:
            return None
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            feature_order = tuple(str(v) for v in (payload.get("feature_order") or []))
            means = np.asarray(payload.get("means") or [], dtype=np.float64)
            scales = np.asarray(payload.get("scales") or [], dtype=np.float64)
            weights = np.asarray(payload.get("weights") or [], dtype=np.float64)
            bias = float(payload.get("bias") or 0.0)
            if not feature_order or means.size != len(feature_order) or scales.size != len(feature_order) or weights.size != len(feature_order):
                raise ValueError("invalid feature shape")
            self._marker_rank_model = MarkerRankModel(
                path=os.path.abspath(model_path),
                feature_order=feature_order,
                means=means,
                scales=scales,
                weights=weights,
                bias=bias,
                label_name=str(payload.get("label_name") or "candidate_is_manual_match_25ms"),
                score_scale=float(payload.get("score_scale") or 1.0),
            )
        except Exception as exc:
            LOG.warning("Could not load marker rank model from %s: %s", model_path, exc)
            self._marker_rank_model = None
        return self._marker_rank_model

    def _marker_model_feature_values(
        self,
        *,
        entry: Dict[str, Any],
        event_score: float,
        event_marker_count: int,
        event_plausible_count: int,
    ) -> Dict[str, float]:
        cluster_index = int(entry.get("cluster_index") or 0)
        denom = max(1, int(event_marker_count) - 1)
        return {
            "delta_beats_from_rough": float(entry.get("delta_beats_from_rough") or 0.0),
            "marker_support": float(entry.get("marker_support") or 0.0),
            "attack_edge_norm": float(entry.get("attack_edge_norm") or 0.0),
            "attack_started_norm": float(entry.get("attack_started_norm") or 0.0),
            "inside_body_norm": float(entry.get("inside_body_norm") or 0.0),
            "sustain_after_norm": float(entry.get("sustain_after_norm") or 0.0),
            "density_after_norm": float(entry.get("density_after_norm") or 0.0),
            "low_growth_norm": float(entry.get("low_growth_norm") or 0.0),
            "pre_post_ratio_norm": float(entry.get("pre_post_ratio_norm") or 0.0),
            "dense_run_after_norm": float(entry.get("dense_run_after_norm") or 0.0),
            "isolated_preroll_norm": float(entry.get("isolated_preroll_norm") or 0.0),
            "gap_prev_norm": float(entry.get("gap_prev_norm") or 0.0),
            "gap_next_norm": float(entry.get("gap_next_norm") or 0.0),
            "event_followthrough_norm": float(entry.get("event_followthrough_norm") or 0.0),
            "cluster_index_norm": float(cluster_index) / float(denom),
            "event_score_norm": float(event_score),
            "event_marker_count_norm": min(1.0, float(max(0, event_marker_count - 1)) / 3.0),
            "event_plausible_count_norm": min(1.0, float(event_plausible_count) / float(max(1, event_marker_count))),
        }

    def _apply_marker_rank_model(
        self,
        *,
        cluster_entries: Sequence[Dict[str, Any]],
        event_score: float,
        event_marker_count: int,
        event_plausible_count: int,
        debug: Dict[str, Any],
    ) -> None:
        model = self._get_marker_rank_model()
        if model is None:
            return
        blend = float(np.clip(float(self.options.marker_rank_model_blend), 0.0, 1.0))
        if blend <= 0.0:
            return
        debug["marker_rank_model_path"] = model.path
        debug["marker_rank_model_label"] = model.label_name
        debug["marker_rank_model_blend"] = blend
        for entry in cluster_entries:
            feature_values = self._marker_model_feature_values(
                entry=entry,
                event_score=event_score,
                event_marker_count=event_marker_count,
                event_plausible_count=event_plausible_count,
            )
            model_prob = float(model.predict_proba(feature_values))
            entry["model_rank_probability"] = model_prob
            base_rank = float(entry.get("marker_rank_score") or entry.get("cluster_local_score") or entry.get("snap_score") or 0.0)
            entry["marker_rank_score"] = ((1.0 - blend) * base_rank) + (blend * model_prob)

    def _build_rough_region_stage(
        self,
        *,
        candidates: Sequence[CandidateScore],
        reference_candidate: Optional[CandidateScore],
        reference_anchor_seconds: Optional[float],
        rough_candidate: Optional[CandidateScore],
        rough_anchor_seconds: Optional[float],
        rough_reason: Optional[str],
        legacy_prior_seconds: Optional[float],
        legacy_prior_confidence: Optional[float],
    ) -> RoughRegionStage:
        return RoughRegionStage(
            candidate_count=int(len(candidates)),
            reference_candidate_seconds=float(reference_candidate.time_abs_sec) if reference_candidate is not None else None,
            reference_anchor_seconds=float(reference_anchor_seconds) if reference_anchor_seconds is not None else None,
            reference_confidence=float(self._candidate_confidence(reference_candidate)) if reference_candidate is not None else None,
            rough_candidate_seconds=float(rough_candidate.time_abs_sec) if rough_candidate is not None else None,
            rough_anchor_seconds=float(rough_anchor_seconds) if rough_anchor_seconds is not None else None,
            rough_confidence=float(self._candidate_confidence(rough_candidate)) if rough_candidate is not None else None,
            reason=str(rough_reason) if rough_reason else None,
            legacy_prior_seconds=float(legacy_prior_seconds) if legacy_prior_seconds is not None else None,
            legacy_prior_confidence=float(legacy_prior_confidence) if legacy_prior_confidence is not None else None,
        )

    def _build_local_event_stage(self, ableton_snap_debug: Optional[Dict[str, Any]]) -> Optional[LocalEventStage]:
        if not ableton_snap_debug:
            return None
        return LocalEventStage(
            source="ableton_asd",
            available=bool(ableton_snap_debug.get("used")),
            reason=str(ableton_snap_debug.get("reason")) if ableton_snap_debug.get("reason") else None,
            rough_anchor_seconds=_safe_float(ableton_snap_debug.get("rough_anchor_seconds")),
            selected_event_id=_safe_int(ableton_snap_debug.get("selected_event_id")),
            selected_event_score=_safe_float(ableton_snap_debug.get("selected_event_score")),
            selected_event_reason=str(ableton_snap_debug.get("selected_event_reason")) if ableton_snap_debug.get("selected_event_reason") else None,
            cluster_start_seconds=_safe_float(ableton_snap_debug.get("cluster_start_seconds")),
            cluster_end_seconds=_safe_float(ableton_snap_debug.get("cluster_end_seconds")),
            cluster_marker_count=int(ableton_snap_debug.get("cluster_marker_count") or 0),
            cluster_plausible_marker_count=int(ableton_snap_debug.get("cluster_plausible_marker_count") or 0),
            chosen_marker_seconds=_safe_float(ableton_snap_debug.get("chosen_marker_seconds")),
            chosen_candidate_seconds=_safe_float(ableton_snap_debug.get("chosen_candidate_seconds")),
            chosen_marker_reason=str(ableton_snap_debug.get("chosen_marker_reason")) if ableton_snap_debug.get("chosen_marker_reason") else None,
            first_plausible_marker_seconds=_safe_float(ableton_snap_debug.get("first_plausible_marker_seconds")),
            initial_preferred_marker_seconds=_safe_float(ableton_snap_debug.get("initial_preferred_marker_seconds")),
            earliest_near_best_marker_seconds=_safe_float(ableton_snap_debug.get("earliest_near_best_marker_seconds")),
            previous_marker_seconds=_safe_float(ableton_snap_debug.get("previous_marker_seconds")),
            previous_marker_exists=bool(ableton_snap_debug.get("previous_marker_exists")),
            previous_marker_won=bool(ableton_snap_debug.get("previous_marker_won")),
            previous_marker_rejection_reason=str(ableton_snap_debug.get("previous_marker_rejection_reason")) if ableton_snap_debug.get("previous_marker_rejection_reason") else None,
            accepted_as_final=(bool(ableton_snap_debug.get("accepted_as_final")) if ableton_snap_debug.get("accepted_as_final") is not None else None),
            accept_reason=str(ableton_snap_debug.get("accept_reason")) if ableton_snap_debug.get("accept_reason") else None,
            rejected_as_final_reason=str(ableton_snap_debug.get("rejected_as_final_reason")) if ableton_snap_debug.get("rejected_as_final_reason") else None,
            events=list(ableton_snap_debug.get("events") or []),
        )

    def _build_self_check_stage(self, ableton_snap_debug: Optional[Dict[str, Any]]) -> Optional[SelfCheckStage]:
        if not ableton_snap_debug:
            return None
        return SelfCheckStage(
            passed=(bool(ableton_snap_debug.get("self_check_passed")) if ableton_snap_debug.get("self_check_passed") is not None else None),
            reason=str(ableton_snap_debug.get("self_check_reason")) if ableton_snap_debug.get("self_check_reason") else None,
            chosen_marker_seconds=_safe_float(ableton_snap_debug.get("chosen_marker_seconds")),
            first_plausible_marker_seconds=_safe_float(ableton_snap_debug.get("first_plausible_marker_seconds")),
            final_is_first_plausible_marker=bool(ableton_snap_debug.get("final_is_first_plausible_marker")),
            late_relative_to_rough=(bool(ableton_snap_debug.get("late_relative_to_rough")) if ableton_snap_debug.get("late_relative_to_rough") is not None else None),
        )

    def detect(
        self,
        drums_path: str,
        inst_path: Optional[str] = None,
        vocals_path: Optional[str] = None,
        bpm: Optional[float] = None,
        debug_dir: Optional[str] = None,
        generate_plots: bool = True,
        manual_anchor_sec: Optional[float] = None,
        legacy_prior_seconds: Optional[float] = None,
        legacy_prior_confidence: Optional[float] = None,
        legacy_prior_source: Optional[str] = None,
    ) -> SongDownbeatResult:
        drums = self._load_stem("drums", drums_path)
        inst = self._load_stem("inst", inst_path) if inst_path else None
        vocals = self._load_stem("vocals", vocals_path) if vocals_path else None

        legacy_sec = None
        legacy_conf = None
        legacy_source = None
        try:
            legacy = detect_true_drop(
                drums_path,
                bpm_hint=float(bpm) if bpm else None,
                bpm_min=float(self.options.bpm_min),
                bpm_max=float(self.options.bpm_max),
                config=DetectorConfig(
                    sr=self.options.analysis_sr,
                    n_fft=self.options.n_fft,
                    hop_sec=max(self.options.hop_sec, 0.02),
                    bpm_min=float(self.options.bpm_min),
                    bpm_max=float(self.options.bpm_max),
                    max_first_drop_ratio=float(self.options.max_first_downbeat_ratio),
                ),
            )
        except Exception as exc:
            LOG.debug("Legacy detector prior failed: %s", exc)
            legacy = None
        if legacy is not None:
            legacy_sec = float(legacy.drop_time_sec)
            legacy_conf = float(legacy.confidence)
            legacy_source = "legacy_detector"
        if legacy_prior_seconds is not None:
            legacy_sec = float(legacy_prior_seconds)
            if legacy_prior_confidence is None:
                legacy_conf = max(float(legacy_conf or 0.0), float(self.options.legacy_high_confidence))
            else:
                legacy_conf = float(legacy_prior_confidence)
            legacy_source = str(legacy_prior_source or "external_prior")

        ableton_markers = None
        if bool(self.options.use_ableton_asd_candidates) and extract_ableton_onset_markers is not None:
            try:
                ableton_markers = extract_ableton_onset_markers(drums_path)
            except Exception as exc:
                LOG.debug("Ableton analysis adapter failed for %s: %s", drums_path, exc)
                ableton_markers = None

        analysis = self._prepare_analysis(drums, bpm_hint=bpm)
        custom_candidates, custom_chosen = self._score_drums_candidates(
            drums=drums,
            analysis=analysis,
            bpm_hint=bpm,
            legacy_sec=legacy_sec,
            legacy_conf=legacy_conf,
            ableton_markers=None,
        )
        custom_prior_reason = ""
        if legacy_source and legacy_source != "legacy_detector":
            custom_chosen, custom_prior_reason = self._apply_explicit_prior_guidance(
                candidates=custom_candidates,
                chosen=custom_chosen,
                prior_sec=legacy_sec,
                prior_conf=legacy_conf,
                prior_source=legacy_source,
            )
        custom_chosen, custom_legacy_reason = self._apply_legacy_guidance(
            candidates=custom_candidates,
            chosen=custom_chosen,
            legacy_sec=legacy_sec,
            legacy_conf=legacy_conf,
        )
        rough_custom_candidate, rough_custom_reason = self._select_rough_custom_candidate(
            candidates=custom_candidates,
            reference=custom_chosen,
            bpm_hint=bpm,
        )
        custom_reference_anchor_sec = None
        if custom_chosen is not None:
            custom_reference_alignment = self._refine_drums_alignment(
                drums=drums,
                coarse_sec=float(custom_chosen.time_abs_sec),
                beat_sec=60.0 / max(1e-6, float(custom_chosen.bpm)),
                beat_position=self._beat_position_from_seconds(analysis, float(custom_chosen.time_abs_sec), float(custom_chosen.bpm)),
                confidence=self._candidate_confidence(custom_chosen),
                chosen=custom_chosen,
            )
            custom_reference_anchor_sec = float(custom_reference_alignment.anchor_seconds)
        rough_custom_anchor_sec = None
        if rough_custom_candidate is not None:
            rough_alignment = self._refine_drums_alignment(
                drums=drums,
                coarse_sec=float(rough_custom_candidate.time_abs_sec),
                beat_sec=60.0 / max(1e-6, float(rough_custom_candidate.bpm)),
                beat_position=self._beat_position_from_seconds(analysis, float(rough_custom_candidate.time_abs_sec), float(rough_custom_candidate.bpm)),
                confidence=self._candidate_confidence(rough_custom_candidate),
                chosen=rough_custom_candidate,
            )
            rough_custom_anchor_sec = float(rough_alignment.anchor_seconds)
        rough_region_stage = self._build_rough_region_stage(
            candidates=custom_candidates,
            reference_candidate=custom_chosen,
            reference_anchor_seconds=custom_reference_anchor_sec,
            rough_candidate=rough_custom_candidate,
            rough_anchor_seconds=rough_custom_anchor_sec,
            rough_reason=rough_custom_reason,
            legacy_prior_seconds=legacy_sec,
            legacy_prior_confidence=legacy_conf,
        )
        candidates = custom_candidates
        chosen = custom_chosen
        legacy_reason = " ".join(part for part in [custom_prior_reason, custom_legacy_reason] if part)
        candidate_strategy = "custom"
        arbitration_reason = ""
        ableton_snap_debug: Optional[Dict[str, Any]] = None
        local_event_stage: Optional[LocalEventStage] = None
        self_check_stage: Optional[SelfCheckStage] = None

        if ableton_markers is not None:
            ableton_candidates, ableton_chosen = self._score_drums_candidates(
                drums=drums,
                analysis=analysis,
                bpm_hint=bpm,
                legacy_sec=legacy_sec,
                legacy_conf=legacy_conf,
                ableton_markers=ableton_markers,
            )
            ableton_prior_reason = ""
            if legacy_source and legacy_source != "legacy_detector":
                ableton_chosen, ableton_prior_reason = self._apply_explicit_prior_guidance(
                    candidates=ableton_candidates,
                    chosen=ableton_chosen,
                    prior_sec=legacy_sec,
                    prior_conf=legacy_conf,
                    prior_source=legacy_source,
                )
            ableton_chosen, ableton_legacy_reason = self._apply_legacy_guidance(
                candidates=ableton_candidates,
                chosen=ableton_chosen,
                legacy_sec=legacy_sec,
                legacy_conf=legacy_conf,
            )
            ableton_first_drop = self._choose_first_drop_event_candidate(ableton_candidates)

            snap_choice, ableton_snap_debug = self._snap_to_rough_ableton_marker(
                rough_candidate=rough_custom_candidate,
                rough_anchor_sec=rough_custom_anchor_sec,
                ableton_candidates=ableton_candidates,
                ableton_markers=ableton_markers,
                analysis=analysis,
            )
            if snap_choice is not None:
                snap_accept = False
                snap_accept_reason = ""
                beat_ref = 60.0 / max(1e-6, float(snap_choice.bpm))
                custom_conf = self._candidate_confidence(chosen) if chosen is not None else 0.0
                snap_conf = self._candidate_confidence(snap_choice)
                snap_to_custom_beats = (
                    abs(float(snap_choice.time_abs_sec) - float(custom_reference_anchor_sec)) / max(1e-6, beat_ref)
                    if custom_reference_anchor_sec is not None
                    else None
                )
                self_check_passed = bool(ableton_snap_debug.get("self_check_passed", True))
                chosen_late_from_rough = float(max(0.0, float(ableton_snap_debug.get("chosen_marker_delta_beats_from_rough") or 0.0)))
                earlier_rough_beats = (
                    (float(custom_chosen.time_abs_sec) - float(rough_custom_candidate.time_abs_sec)) / max(1e-6, beat_ref)
                    if custom_chosen is not None and rough_custom_candidate is not None
                    else 0.0
                )
                chosen_delta_from_rough = float(ableton_snap_debug.get("chosen_marker_delta_beats_from_rough") or 0.0)
                cluster_marker_count = int(ableton_snap_debug.get("cluster_marker_count") or 0)
                selected_event_score = _safe_float(ableton_snap_debug.get("selected_event_score"))
                chosen_marker_support = _safe_float(ableton_snap_debug.get("chosen_marker_support"))
                chosen_attack_edge_norm = _safe_float(ableton_snap_debug.get("chosen_attack_edge_norm"))
                chosen_inside_body_norm = _safe_float(ableton_snap_debug.get("chosen_inside_body_norm"))
                chosen_event_followthrough_norm = _safe_float(ableton_snap_debug.get("chosen_event_followthrough_norm"))
                chosen_strong_late_attack_start = bool(ableton_snap_debug.get("chosen_strong_late_attack_start", False))
                first_plausible_auto_accept = (
                    self_check_passed
                    and
                    bool(ableton_snap_debug.get("final_is_first_plausible_marker"))
                    and cluster_marker_count >= int(self.options.ableton_cluster_auto_accept_min_markers)
                    and chosen_late_from_rough <= float(self.options.ableton_cluster_max_post_attack_drift_beats)
                    and chosen_delta_from_rough >= -float(self.options.ableton_cluster_max_early_pull_beats)
                )
                strong_single_marker_exact_anchor = (
                    self_check_passed
                    and bool(ableton_snap_debug.get("final_is_first_plausible_marker"))
                    and cluster_marker_count == 1
                    and chosen_late_from_rough <= float(self.options.ableton_snap_max_late_beats)
                    and chosen_delta_from_rough >= -float(self.options.ableton_cluster_max_early_pull_beats)
                    and float(selected_event_score or 0.0) >= 0.50
                    and float(chosen_marker_support or 0.0) >= 0.60
                    and float(chosen_attack_edge_norm or 0.0) >= 0.55
                    and float(chosen_event_followthrough_norm or 0.0) >= 0.55
                    and bool(chosen_strong_late_attack_start)
                )
                same_region_exact_anchor = (
                    snap_to_custom_beats is not None
                    and snap_to_custom_beats <= float(self.options.ableton_exact_anchor_prefer_within_beats)
                    and float(selected_event_score or 0.0) >= float(self.options.ableton_exact_anchor_min_event_score)
                    and float(chosen_marker_support or 0.0) >= float(self.options.ableton_exact_anchor_min_support)
                    and cluster_marker_count >= int(self.options.ableton_cluster_auto_accept_min_markers)
                    and chosen_late_from_rough <= max(0.35, float(self.options.ableton_cluster_max_post_attack_drift_beats))
                    and chosen_delta_from_rough >= -float(self.options.ableton_cluster_max_early_pull_beats)
                    and (
                        self_check_passed
                        or bool(ableton_snap_debug.get("final_is_first_plausible_marker"))
                        or cluster_marker_count >= int(self.options.ableton_cluster_auto_accept_min_markers)
                    )
                )
                early_attack_start_promotion = False
                early_attack_start_gap_beats = None
                if custom_reference_anchor_sec is not None:
                    early_attack_start_gap_beats = (
                        float(custom_reference_anchor_sec) - float(snap_choice.time_abs_sec)
                    ) / max(1e-6, beat_ref)
                    early_attack_start_promotion = (
                        early_attack_start_gap_beats > 0.0
                        and early_attack_start_gap_beats <= float(self.options.ableton_attack_start_snap_beats)
                        and self_check_passed
                        and float(selected_event_score or 0.0) >= float(self.options.ableton_attack_start_min_event_score)
                        and float(chosen_marker_support or 0.0) >= float(self.options.ableton_attack_start_min_support)
                        and float(chosen_attack_edge_norm or 0.0) >= float(self.options.ableton_attack_start_min_edge)
                        and float(chosen_inside_body_norm if chosen_inside_body_norm is not None else 1.0) <= float(self.options.ableton_attack_start_max_inside_body)
                        and float(chosen_event_followthrough_norm or 0.0) >= float(self.options.ableton_attack_start_min_followthrough)
                    )
                ableton_snap_debug["early_attack_start_gap_beats"] = float(early_attack_start_gap_beats) if early_attack_start_gap_beats is not None else None
                if (
                    first_plausible_auto_accept
                ):
                    snap_accept = True
                    snap_accept_reason = (
                        "Ableton cluster snap accepted because it selected the first plausible marker in a multi-marker local attack cluster."
                    )
                elif strong_single_marker_exact_anchor:
                    snap_accept = True
                    snap_accept_reason = (
                        "Ableton single-marker snap accepted because it selected a strong local attack-start marker inside the rough first-drop event."
                    )
                elif early_attack_start_promotion:
                    snap_accept = True
                    snap_accept_reason = (
                        "Promoted an earlier Ableton marker because it behaves like the attack-start boundary of the same local drop event."
                    )
                elif same_region_exact_anchor:
                    snap_accept = True
                    snap_accept_reason = (
                        "Ableton `.asd` marker accepted as the primary exact anchor inside the rough first-drop event."
                    )
                elif snap_to_custom_beats is not None and snap_to_custom_beats <= float(self.options.ableton_snap_accept_near_custom_beats):
                    snap_accept = True
                    snap_accept_reason = (
                        f"Ableton snap accepted as the exact anchor because it stayed within "
                        f"{snap_to_custom_beats:.2f} beats of the refined custom anchor."
                    )
                elif (
                    self_check_passed
                    and
                    earlier_rough_beats >= float(self.options.ableton_override_earlier_beats)
                    and float(ableton_snap_debug.get('chosen_marker_support') or 0.0) >= float(self.options.ableton_override_min_support)
                    and (snap_conf + float(self.options.ableton_override_conf_tolerance)) >= custom_conf
                ):
                    snap_accept = True
                    snap_accept_reason = (
                        f"Ableton snap overrode a much later custom region because the rough custom estimate "
                        f"was {earlier_rough_beats:.2f} beats earlier and the local Ableton marker stayed plausible."
                    )

                if snap_accept and ableton_first_drop is not None and snap_choice is not None and ableton_first_drop is not snap_choice:
                    beat_ref = 60.0 / max(1e-6, float(ableton_first_drop.bpm))
                    first_earlier_beats = (float(snap_choice.time_abs_sec) - float(ableton_first_drop.time_abs_sec)) / max(1e-6, beat_ref)
                    first_conf = self._candidate_confidence(ableton_first_drop)
                    snap_conf = self._candidate_confidence(snap_choice)
                    first_bang = self._drop_bang_score(ableton_first_drop)
                    first_followthrough = self._drop_followthrough(ableton_first_drop)
                    first_drop_can_override_snap = (
                        first_earlier_beats >= max(6.0, float(self.options.ableton_override_earlier_beats) * 0.75)
                        and float(ableton_first_drop.score) >= 0.62
                        and first_bang >= 0.78
                        and first_followthrough >= 0.50
                        and (first_conf + 0.16) >= snap_conf
                    )
                    if first_drop_can_override_snap:
                        snap_choice = ableton_first_drop
                        snap_accept_reason = (
                            f"Ableton first-drop override replaced a later accepted local snap with the earliest strong "
                            f"drop bang at {ableton_first_drop.time_abs_sec:.3f}s."
                        )
                        ableton_snap_debug["global_first_drop_override_seconds"] = float(ableton_first_drop.time_abs_sec)
                        ableton_snap_debug["chosen_marker_seconds"] = float(ableton_first_drop.time_abs_sec)
                        ableton_snap_debug["chosen_candidate_seconds"] = float(ableton_first_drop.time_abs_sec)
                        ableton_snap_debug["chosen_marker_reason"] = "Global first-drop event override selected this earlier strong bang."

                ableton_snap_debug["accepted_as_final"] = bool(snap_accept)
                ableton_snap_debug["accept_reason"] = snap_accept_reason or None
                if snap_accept:
                    for cand in ableton_candidates:
                        cand.chosen = False
                    snap_choice.chosen = True
                    candidates = ableton_candidates
                    chosen = snap_choice
                    legacy_reason = " ".join(part for part in [ableton_prior_reason, custom_prior_reason] if part)
                    candidate_strategy = "ableton_asd"
                    arbitration_reason = " ".join(
                        part for part in [snap_accept_reason, str(ableton_snap_debug.get("chosen_marker_reason") or "")] if part
                    )
                else:
                    ableton_snap_debug["rejected_as_final_reason"] = (
                        "The Ableton snap candidate was not close enough to the refined custom anchor, did not come from a "
                        "multi-marker first-plausible cluster, failed its local self-check, or did not clearly fix a later custom-region error."
                    )
                    snap_choice = None
            if snap_choice is None:
                choose_ableton = False
                if chosen is None and ableton_chosen is not None:
                    choose_ableton = True
                elif chosen is not None and ableton_chosen is not None:
                    if legacy_sec is not None and legacy_conf is not None and float(legacy_conf) >= float(self.options.legacy_high_confidence):
                        beat_ref = 60.0 / max(1e-6, float(ableton_chosen.bpm))
                        custom_dist = abs(float(chosen.time_abs_sec) - float(legacy_sec)) / max(1e-6, beat_ref)
                        ableton_dist = abs(float(ableton_chosen.time_abs_sec) - float(legacy_sec)) / max(1e-6, beat_ref)
                        if ableton_dist + 0.25 < custom_dist:
                            choose_ableton = True
                    if not choose_ableton:
                        custom_conf = self._candidate_confidence(chosen)
                        ableton_conf = self._candidate_confidence(ableton_chosen)
                        if custom_conf < 0.45 and ableton_conf > (custom_conf + 0.08):
                            choose_ableton = True
                    if not choose_ableton and ableton_first_drop is not None:
                        beat_ref = 60.0 / max(1e-6, float(ableton_first_drop.bpm))
                        earlier_beats = (float(chosen.time_abs_sec) - float(ableton_first_drop.time_abs_sec)) / max(1e-6, beat_ref)
                        custom_conf = self._candidate_confidence(chosen)
                        first_conf = self._candidate_confidence(ableton_first_drop)
                        first_bang = self._drop_bang_score(ableton_first_drop)
                        first_followthrough = self._drop_followthrough(ableton_first_drop)
                        first_drop_is_strong = (
                            float(ableton_first_drop.score) >= 0.62
                            and first_bang >= 0.78
                            and first_followthrough >= 0.50
                            and (
                                float(ableton_first_drop.norm_ableton) >= 0.72
                                or float(ableton_first_drop.norm_onset) >= 0.70
                                or float(ableton_first_drop.norm_lowend) >= 0.70
                            )
                        )
                        first_drop_competes = (
                            (first_conf + 0.16) >= custom_conf
                            or (float(ableton_first_drop.score) + 0.12) >= float(chosen.score)
                        )
                        if (
                            earlier_beats >= max(6.0, float(self.options.ableton_override_earlier_beats) * 0.75)
                            and first_drop_is_strong
                            and first_drop_competes
                        ):
                            ableton_chosen = ableton_first_drop
                            choose_ableton = True
                            arbitration_reason = (
                                f"Ableton first-drop override: kept the earliest ready drop event at "
                                f"{ableton_first_drop.time_abs_sec:.3f}s and selected its strongest bang over "
                                f"a later pick at {chosen.time_abs_sec:.3f}s."
                            )
                        elif first_drop_is_strong:
                            later_beats = -earlier_beats
                            chosen_followthrough = self._drop_followthrough(chosen)
                            chosen_is_tail = (
                                custom_conf < 0.70
                                or chosen_followthrough < 0.56
                                or float(chosen.norm_density) < 0.45
                                or float(chosen.norm_weak) > 0.55
                                or float(chosen.norm_fake) > 0.55
                            )
                            first_drop_is_better = (
                                first_conf >= (custom_conf + 0.10)
                                or float(ableton_first_drop.score) >= (float(chosen.score) + 0.04)
                            )
                            if (
                                2.0 <= later_beats <= 12.0
                                and chosen_is_tail
                                and first_drop_is_better
                            ):
                                ableton_chosen = ableton_first_drop
                                choose_ableton = True
                                arbitration_reason = (
                                    f"Ableton first-drop promotion: skipped an earlier weak/tail candidate at "
                                    f"{chosen.time_abs_sec:.3f}s and moved to the first strong drop bang at "
                                    f"{ableton_first_drop.time_abs_sec:.3f}s."
                                )
                    if not choose_ableton:
                        beat_ref = 60.0 / max(1e-6, float(ableton_chosen.bpm))
                        earlier_beats = (float(chosen.time_abs_sec) - float(ableton_chosen.time_abs_sec)) / max(1e-6, beat_ref)
                        custom_conf = self._candidate_confidence(chosen)
                        ableton_conf = self._candidate_confidence(ableton_chosen)
                        if (
                            earlier_beats >= float(self.options.ableton_override_earlier_beats)
                            and bool(ableton_chosen.valid)
                            and float(ableton_chosen.norm_ableton) >= float(self.options.ableton_override_min_support)
                            and (ableton_conf + float(self.options.ableton_override_conf_tolerance)) >= custom_conf
                        ):
                            choose_ableton = True
                            arbitration_reason = (
                                f"Ableton-guided override: kept an earlier Ableton-backed candidate "
                                f"({ableton_chosen.time_abs_sec:.3f}s) over a much later custom pick "
                                f"({chosen.time_abs_sec:.3f}s)."
                            )

                if choose_ableton:
                    for cand in ableton_candidates:
                        cand.chosen = False
                    if ableton_chosen is not None:
                        ableton_chosen.chosen = True
                    candidates = ableton_candidates
                    chosen = ableton_chosen
                    legacy_reason = " ".join(part for part in [ableton_prior_reason, ableton_legacy_reason] if part)
                    candidate_strategy = "ableton_asd"

            local_event_stage = self._build_local_event_stage(ableton_snap_debug)
            self_check_stage = self._build_self_check_stage(ableton_snap_debug)

        if chosen is None:
            if legacy_sec is None:
                raise RuntimeError("Could not determine a plausible downbeat candidate.")
            LOG.warning("Candidate scoring failed; falling back to legacy detector at %.3fs", legacy_sec)
            beat_sec = 60.0 / float(analysis["bpm"])
            coarse_sec = float(legacy_sec)
            chosen_reason = "Fallback to legacy detector because no valid multi-stage candidate was produced."
            candidate_strategy = "legacy_fallback"
        else:
            beat_sec = 60.0 / float(chosen.bpm)
            coarse_sec = float(chosen.time_abs_sec)
            chosen_reason = self._build_reason_string(chosen)
            if arbitration_reason:
                chosen_reason = f"{arbitration_reason} {chosen_reason}"
            if legacy_reason:
                chosen_reason = f"{legacy_reason} {chosen_reason}"

        drums_alignment = self._refine_drums_alignment(
            drums=drums,
            coarse_sec=coarse_sec,
            beat_sec=beat_sec,
            beat_position=self._beat_position_from_seconds(analysis, coarse_sec, chosen.bpm if chosen else float(analysis["bpm"])),
            confidence=float(chosen.score if chosen else legacy_conf or 0.0),
            chosen=chosen,
        )

        inst_alignment = self._align_secondary_stem(
            stem=inst,
            expected_sec=drums_alignment.anchor_seconds,
            beat_position=drums_alignment.beat_position,
            bpm=float(chosen.bpm if chosen else analysis["bpm"]),
            drums_conf=drums_alignment.confidence,
        )
        vocals_alignment = self._align_secondary_stem(
            stem=vocals,
            expected_sec=drums_alignment.anchor_seconds,
            beat_position=drums_alignment.beat_position,
            bpm=float(chosen.bpm if chosen else analysis["bpm"]),
            drums_conf=drums_alignment.confidence,
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        candidate_csv = self._write_candidate_csv(debug_dir, candidates) if debug_dir else None
        plots = self._write_debug_plots(
            debug_dir,
            drums,
            analysis,
            candidates,
            drums_alignment,
            generate_plots,
            ableton_snap_debug=ableton_snap_debug,
            manual_anchor_sec=manual_anchor_sec,
        ) if debug_dir else []

        result = SongDownbeatResult(
            bpm=float(chosen.bpm if chosen else analysis["bpm"]),
            drums=drums_alignment,
            inst=inst_alignment,
            vocals=vocals_alignment,
            candidates=candidates,
            custom_candidates=list(custom_candidates),
            chosen_reason=chosen_reason,
            candidate_strategy=candidate_strategy,
            custom_reference_candidate=custom_chosen,
            rough_custom_candidate=rough_custom_candidate,
            rough_custom_reason=rough_custom_reason,
            ableton_snap_debug=ableton_snap_debug,
            rough_region_stage=rough_region_stage,
            local_event_stage=local_event_stage,
            self_check_stage=self_check_stage,
            candidate_csv=candidate_csv,
            plots=plots,
            legacy_prior_seconds=legacy_sec,
            legacy_prior_confidence=legacy_conf,
            legacy_prior_source=legacy_source,
            ableton_markers=ableton_markers.to_dict() if ableton_markers is not None else None,
        )

        if debug_dir:
            out_path = os.path.join(debug_dir, "downbeat_debug.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, ensure_ascii=True, default=_json_default)

        return result

    def _load_stem(self, role: str, path: Optional[str]) -> StemAudio:
        if not path:
            raise ValueError(f"Missing path for {role}")
        sample_rate, _ = _probe_audio_info(path)
        native_sr = int(max(1, sample_rate))
        analysis_sr = int(min(native_sr, int(self.options.analysis_sr))) if native_sr > 0 else int(self.options.analysis_sr)

        native_y = _decode_audio_ffmpeg_mono(path, native_sr)
        analysis_y = _decode_audio_ffmpeg_mono(path, analysis_sr)
        analysis_y = _safe_normalize(analysis_y)
        analysis_offset = _trim_leading_silence_for_analysis(analysis_y, analysis_sr)
        if analysis_offset > 0 and analysis_offset < len(analysis_y):
            analysis_y = analysis_y[analysis_offset:]
        else:
            analysis_offset = 0

        return StemAudio(
            role=role,
            path=path,
            native_sr=native_sr,
            native_y=native_y.astype(np.float32, copy=False),
            analysis_sr=analysis_sr,
            analysis_y=analysis_y.astype(np.float32, copy=False),
            analysis_offset_samples=int(analysis_offset),
        )

    def _prepare_analysis(self, drums: StemAudio, bpm_hint: Optional[float]) -> Dict[str, Any]:
        hop = max(1, int(round(float(self.options.hop_sec) * float(drums.analysis_sr))))
        feats = _stft_features(drums.analysis_y, drums.analysis_sr, int(self.options.n_fft), hop)
        if feats is None:
            raise RuntimeError("Drums stem is too short for analysis.")

        onset_env = feats["onset"].astype(np.float32, copy=False)
        est_bpm = float(
            np.clip(
                float(bpm_hint)
                if bpm_hint
                else _estimate_tempo_bpm(onset_env, float(self.options.hop_sec), float(self.options.bpm_min), float(self.options.bpm_max), None),
                float(self.options.bpm_min),
                float(self.options.bpm_max),
            )
        )
        guess_beat_sec = 60.0 / max(1e-6, est_bpm)
        frames_per_bar = max(4, int(round((4.0 * guess_beat_sec) / float(self.options.hop_sec))))

        feats["rms"] = _moving_average(feats["rms"], max(3, frames_per_bar // 2))
        feats["low_energy"] = _moving_average(feats["low_energy"], max(3, frames_per_bar // 2))
        feats["high_energy"] = _moving_average(feats["high_energy"], max(3, frames_per_bar // 2))
        feats["low_ratio"] = _moving_average(feats["low_ratio"], max(3, frames_per_bar // 2))
        feats["high_ratio"] = _moving_average(feats["high_ratio"], max(3, frames_per_bar // 2))
        feats["flux"] = _moving_average(feats["flux"], max(3, frames_per_bar // 4))
        feats["onset"] = _moving_average(feats["onset"], max(3, frames_per_bar // 4))
        feats["low_transient"] = _moving_average(feats["low_transient"], max(3, frames_per_bar // 4))
        feats["high_flux"] = _moving_average(feats["high_flux"], max(3, frames_per_bar // 4))
        feats["frame_times"] = (np.arange(len(feats["onset"]), dtype=np.float32) * float(self.options.hop_sec)).astype(np.float32, copy=False)

        bpm_candidates: List[float] = []
        if bpm_hint is not None and bpm_hint > 0.0:
            bpm_candidates = [float(np.clip(float(bpm_hint), float(self.options.bpm_min), float(self.options.bpm_max)))]
        else:
            raw = [est_bpm, est_bpm * 0.5, est_bpm * 2.0]
            for v in raw:
                if float(self.options.bpm_min) <= float(v) <= float(self.options.bpm_max):
                    bpm_candidates.append(float(v))
            bpm_candidates = sorted(set(round(v, 3) for v in bpm_candidates))
            if not bpm_candidates:
                bpm_candidates = [est_bpm]

        return {
            "feats": feats,
            "hop_sec": float(self.options.hop_sec),
            "duration_sec": float(len(drums.analysis_y)) / float(drums.analysis_sr),
            "offset_sec": float(drums.analysis_offset_seconds),
            "bpm": est_bpm,
            "bpm_candidates": bpm_candidates,
        }

    def _score_drums_candidates(
        self,
        drums: StemAudio,
        analysis: Dict[str, Any],
        bpm_hint: Optional[float],
        legacy_sec: Optional[float],
        legacy_conf: Optional[float],
        ableton_markers: Optional["AbletonAnalysisMarkers"],
    ) -> Tuple[List[CandidateScore], Optional[CandidateScore]]:
        all_candidates: List[CandidateScore] = []
        feats = analysis["feats"]
        hop_sec = float(analysis["hop_sec"])
        duration_sec = float(analysis["duration_sec"])
        offset_sec = float(analysis["offset_sec"])
        ableton_times = None
        if ableton_markers is not None:
            vals = sorted(set(float(v) for v in ableton_markers.candidate_seconds if v is not None))
            if vals:
                ableton_times = np.asarray(vals, dtype=np.float32)

        beat_offset_scores: Dict[Tuple[float, int], float] = {}
        best_by_bpm: List[CandidateScore] = []

        for bpm in analysis["bpm_candidates"]:
            beat_sec = 60.0 / max(1e-6, float(bpm))
            beat_period_frames = max(1, int(round(beat_sec / hop_sec)))
            if ableton_times is not None and ableton_times.size > 0:
                phase_sec = _estimate_phase_from_markers(ableton_times - offset_sec, beat_sec)
            else:
                phase_sec = float(_estimate_beat_phase(feats["onset"], beat_period_frames)) * hop_sec
            beat_times = _build_beats(duration_sec, beat_sec, phase_sec)
            if len(beat_times) < 8:
                continue

            for off in range(4):
                downbeats = _bar_downbeats(beat_times, off)
                if len(downbeats) < 3:
                    continue
                offset_attack = []
                for k in range(off, len(beat_times), 4):
                    offset_attack.append(
                        _sample_near_beat(feats["low_transient"], beat_times, k, hop_sec, tol_frames=2)
                        + (0.55 * _sample_near_beat(feats["onset"], beat_times, k, hop_sec, tol_frames=2))
                    )
                beat_offset_scores[(float(bpm), off)] = _safe_mean(np.asarray(offset_attack, dtype=np.float32), fallback=0.0) if offset_attack else 0.0

        if not beat_offset_scores:
            return [], None

        beat_offset_norm: Dict[Tuple[float, int], float] = {}
        score_keys = list(beat_offset_scores.keys())
        score_vals = _rank_norm([beat_offset_scores[k] for k in score_keys], invert=False)
        for idx, key in enumerate(score_keys):
            beat_offset_norm[key] = float(score_vals[idx])

        for bpm in analysis["bpm_candidates"]:
            beat_sec = 60.0 / max(1e-6, float(bpm))
            beat_period_frames = max(1, int(round(beat_sec / hop_sec)))
            if ableton_times is not None and ableton_times.size > 0:
                phase_sec = _estimate_phase_from_markers(ableton_times - offset_sec, beat_sec)
            else:
                phase_sec = float(_estimate_beat_phase(feats["onset"], beat_period_frames)) * hop_sec
            beat_times = _build_beats(duration_sec, beat_sec, phase_sec)
            if len(beat_times) < 8:
                continue
            duration_cap = max(8.0, float(drums.duration_seconds) * float(self.options.max_first_downbeat_ratio))
            bpm_candidates: List[CandidateScore] = []

            for off in range(4):
                downbeats = _bar_downbeats(beat_times, off)
                if len(downbeats) < 3:
                    continue

                bar_rms = _smooth_bars(_aggregate_bar_feature(feats["rms"], hop_sec, downbeats), 2)
                bar_low = _smooth_bars(_aggregate_bar_feature(np.log1p(feats["low_energy"]), hop_sec, downbeats), 2)
                bar_flux = _smooth_bars(_aggregate_bar_feature(feats["flux"], hop_sec, downbeats), 2)
                bar_high = _smooth_bars(_aggregate_bar_feature(np.log1p(feats["high_energy"]), hop_sec, downbeats), 2)
                bar_sub = _smooth_bars(_aggregate_bar_feature(feats["low_ratio"], hop_sec, downbeats), 2)
                bar_density = _smooth_bars(_count_peaks_bar(feats["onset"], hop_sec, downbeats, q=78.0), 2)
                bar_kick = _smooth_bars(_kick_periodicity_per_bar(feats["low_transient"], hop_sec, downbeats, beat_sec), 2)

                if len(bar_rms) < 2:
                    continue

                n_rms = _robust_norm(bar_rms, 15.0, 90.0)
                n_low = _robust_norm(bar_low, 15.0, 90.0)
                n_flux = _robust_norm(bar_flux, 15.0, 90.0)
                n_high = _robust_norm(bar_high, 15.0, 90.0)
                n_sub = _robust_norm(bar_sub, 15.0, 90.0)
                n_density = _robust_norm(bar_density, 15.0, 90.0)
                n_kick = _robust_norm(bar_kick, 15.0, 90.0)

                for i in range(0, len(bar_rms)):
                    grid_rel = float(downbeats[i])
                    grid_abs = grid_rel + offset_sec
                    candidate_rel = float(grid_rel)
                    candidate_abs = float(grid_abs)
                    ableton_support = 0.0
                    if ableton_times is not None and ableton_times.size > 0:
                        max_snap = float(self.options.ableton_candidate_window_beats) * beat_sec
                        lo = int(np.searchsorted(ableton_times, grid_abs - max_snap, side="left"))
                        hi = int(np.searchsorted(ableton_times, grid_abs + max_snap, side="right"))
                        nearby = ableton_times[lo:hi]
                        if nearby.size == 0:
                            continue
                        snapped_abs = None
                        snapped_diff = None
                        best_local = -1e18
                        for marker_abs in nearby:
                            marker_abs_f = float(marker_abs)
                            marker_rel = max(0.0, marker_abs_f - offset_sec)
                            marker_low = _local_peak_near_time(feats["low_transient"], marker_rel, hop_sec, 0.18 * beat_sec)
                            marker_onset = _local_peak_near_time(feats["onset"], marker_rel, hop_sec, 0.14 * beat_sec)
                            marker_dist = abs(marker_abs_f - grid_abs)
                            marker_score = (
                                (0.70 * marker_low)
                                + (0.30 * marker_onset)
                                - (0.18 * (marker_dist / max(1e-6, 0.35 * beat_sec)))
                            )
                            if marker_score > best_local:
                                best_local = marker_score
                                snapped_abs = marker_abs_f
                                snapped_diff = marker_dist
                        if snapped_abs is None or snapped_diff is None:
                            continue
                        candidate_abs = float(snapped_abs)
                        candidate_rel = max(0.0, float(candidate_abs) - offset_sec)
                        ableton_support = math.exp(-snapped_diff / max(1e-6, 0.18 * beat_sec))
                    if candidate_abs > duration_cap:
                        continue
                    if (i + int(self.options.min_bars_after_candidate)) >= len(bar_rms):
                        continue

                    pre0 = max(0, i - 4)
                    prev = slice(pre0, i)
                    post2 = slice(i, min(len(bar_rms), i + 2))
                    post4 = slice(i, min(len(bar_rms), i + 4))
                    post8 = slice(i, min(len(bar_rms), i + 8))

                    if i > pre0:
                        pre_r = _safe_mean(n_rms[prev], fallback=0.0)
                        pre_l = _safe_mean(n_low[prev], fallback=0.0)
                        pre_f = _safe_mean(n_flux[prev], fallback=0.0)
                        pre_k = _safe_mean(n_kick[prev], fallback=0.0)
                    else:
                        pre_r = _safe_percentile(n_rms, 20.0, fallback=0.0)
                        pre_l = _safe_percentile(n_low, 20.0, fallback=0.0)
                        pre_f = _safe_percentile(n_flux, 20.0, fallback=0.0)
                        pre_k = _safe_percentile(n_kick, 20.0, fallback=0.0)

                    post2_r = _safe_mean(n_rms[post2], fallback=0.0)
                    post4_r = _safe_mean(n_rms[post4], fallback=0.0)
                    post8_r = _safe_mean(n_rms[post8], fallback=0.0)
                    post2_l = _safe_mean(n_low[post2], fallback=0.0)
                    post4_l = _safe_mean(n_low[post4], fallback=0.0)
                    post8_l = _safe_mean(n_low[post8], fallback=0.0)
                    post4_d = _safe_mean(n_density[post4], fallback=0.0)
                    post4_k = _safe_mean(n_kick[post4], fallback=0.0)

                    onset_peak = _local_peak_near_time(feats["onset"], candidate_rel, hop_sec, 0.14 * beat_sec)
                    low_peak = _local_peak_near_time(feats["low_transient"], candidate_rel, hop_sec, 0.18 * beat_sec)
                    attack = (0.62 * low_peak) + (0.38 * onset_peak)
                    low_jump = (post2_l - pre_l) + (0.55 * (post4_l - pre_l))
                    contrast = (post4_r - pre_r) + (0.35 * (post4_l - pre_l))
                    density = post4_d + (0.25 * post4_k)
                    grid = (0.65 * beat_offset_norm.get((float(bpm), off), 0.0)) + (0.35 * post4_k)
                    lookback = max(1, min(4, i))
                    buildup = (0.55 * _trend_score(n_high, i, lookback=lookback)) + (0.45 * _trend_score(n_flux, i, lookback=lookback))
                    repeat = _pattern_repeat_score(feats["low_transient"], feats["onset"], candidate_rel, beat_sec, hop_sec)
                    phrase = (0.35 * buildup) + (0.35 * max(0.0, post4_l - pre_l)) + (0.15 * max(0.0, post4_r - pre_r)) + (0.15 * repeat)
                    sustain = (0.50 * max(0.0, post8_l - pre_l)) + (0.50 * max(0.0, post8_r - pre_r))
                    legacy = 0.0
                    if legacy_sec is not None and legacy_conf is not None and legacy_conf > 0.0:
                        legacy = float(legacy_conf) * math.exp(-abs(candidate_abs - float(legacy_sec)) / max(0.10, 0.55 * beat_sec))

                    preroll = 0.0
                    if candidate_abs <= (offset_sec + (0.35 * beat_sec)):
                        preroll += max(0.0, 0.70 - density) + max(0.0, 0.60 - contrast)
                    if post4_d <= 0.15 and low_jump <= 0.0:
                        preroll += 0.35

                    weak = max(0.0, float(n_rms[i]) - post4_r) + max(0.0, float(n_low[i]) - post4_l)
                    fake = max(0.0, float(n_rms[i]) - post2_r) + max(0.0, float(n_low[i]) - post2_l)

                    cand = CandidateScore(
                        bpm=float(bpm),
                        downbeat_offset=int(off),
                        bar_index=int(i),
                        beat_index=int(off + (4 * i)),
                        time_rel_sec=float(candidate_rel),
                        time_abs_sec=float(candidate_abs),
                        raw_onset=float(attack),
                        raw_lowend=float(low_jump),
                        raw_contrast=float(contrast),
                        raw_density=float(density),
                        raw_grid=float(grid),
                        raw_phrase=float(phrase),
                        raw_repeat=float(repeat),
                        raw_sustain=float(sustain),
                        raw_ableton=float(ableton_support),
                        raw_legacy=float(legacy),
                        raw_preroll=float(preroll),
                        raw_weak=float(weak),
                        raw_fake=float(fake),
                    )
                    bpm_candidates.append(cand)
                    all_candidates.append(cand)

            if bpm_candidates:
                best_by_bpm.append(max(bpm_candidates, key=lambda c: (c.raw_density + c.raw_contrast + c.raw_phrase, -c.time_abs_sec)))

        if not all_candidates:
            return [], None

        self._normalize_and_score_candidates(all_candidates)
        chosen = self._choose_candidate(all_candidates, bpm_hint)
        if chosen is not None:
            chosen.chosen = True
        return sorted(all_candidates, key=lambda c: c.time_abs_sec), chosen

    def _normalize_and_score_candidates(self, candidates: List[CandidateScore]) -> None:
        fields = {
            "norm_onset": [c.raw_onset for c in candidates],
            "norm_lowend": [c.raw_lowend for c in candidates],
            "norm_contrast": [c.raw_contrast for c in candidates],
            "norm_density": [c.raw_density for c in candidates],
            "norm_grid": [c.raw_grid for c in candidates],
            "norm_phrase": [c.raw_phrase for c in candidates],
            "norm_repeat": [c.raw_repeat for c in candidates],
            "norm_sustain": [c.raw_sustain for c in candidates],
            "norm_ableton": [c.raw_ableton for c in candidates],
            "norm_legacy": [c.raw_legacy for c in candidates],
            "norm_preroll": [c.raw_preroll for c in candidates],
            "norm_weak": [c.raw_weak for c in candidates],
            "norm_fake": [c.raw_fake for c in candidates],
        }

        for name, vals in fields.items():
            invert = name in {"norm_preroll", "norm_weak", "norm_fake"}
            ranked = _rank_norm(vals, invert=False)
            for idx, cand in enumerate(candidates):
                setattr(cand, name, float(ranked[idx]))
            if invert:
                ranked_inv = _rank_norm(vals, invert=True)
                for idx, cand in enumerate(candidates):
                    setattr(cand, name, float(1.0 - ranked_inv[idx]))

        w = self.options.weights
        for cand in candidates:
            cand.score = float(
                (w.onset * cand.norm_onset)
                + (w.lowend * cand.norm_lowend)
                + (w.contrast * cand.norm_contrast)
                + (w.density * cand.norm_density)
                + (w.grid * cand.norm_grid)
                + (w.phrase * cand.norm_phrase)
                + (w.repeat * cand.norm_repeat)
                + (w.sustain * cand.norm_sustain)
                + (w.ableton * cand.norm_ableton)
                + (w.legacy * cand.norm_legacy)
                - (w.preroll * cand.norm_preroll)
                - (w.weak * cand.norm_weak)
                - (w.fake * cand.norm_fake)
            )
            followthrough = (0.45 * cand.norm_density) + (0.35 * cand.norm_sustain) + (0.20 * cand.norm_lowend)
            penalties = max(cand.norm_preroll, cand.norm_weak, cand.norm_fake)
            cand.valid = bool(
                cand.norm_grid >= 0.20
                and followthrough >= 0.38
                and penalties <= 0.82
                and (
                    cand.norm_onset >= 0.30
                    or cand.norm_lowend >= 0.35
                    or cand.norm_phrase >= 0.55
                    or cand.norm_ableton >= 0.85
                )
            )
            if cand.valid and int(cand.bar_index) <= 1:
                if cand.norm_density < 0.35 and (cand.norm_weak > 0.55 or cand.norm_fake > 0.65):
                    cand.valid = False
            if cand.valid:
                followthrough = self._drop_followthrough(cand)
                bar_index = int(cand.bar_index)
                if bar_index < 4:
                    if float(cand.score) < 0.74 and float(cand.norm_ableton) < 0.86:
                        cand.valid = False
                    elif float(cand.norm_density) < 0.55 or followthrough < 0.58:
                        cand.valid = False
                elif bar_index < 8:
                    if float(cand.norm_density) < 0.34 and float(cand.norm_ableton) < 0.88:
                        cand.valid = False

    def _choose_candidate(self, candidates: List[CandidateScore], bpm_hint: Optional[float]) -> Optional[CandidateScore]:
        if not candidates:
            return None
        pool = [c for c in candidates if c.valid]
        if not pool:
            pool = list(candidates)

        if bpm_hint is not None:
            hinted = [c for c in pool if abs(float(c.bpm) - float(bpm_hint)) <= 0.01]
            if hinted:
                pool = hinted

        best_score = max(float(c.score) for c in pool)
        use_ableton_margin = any(float(c.raw_ableton) > 0.0 for c in pool)
        margin = (
            float(self.options.earliest_near_best_margin_ableton)
            if use_ableton_margin
            else float(self.options.earliest_near_best_margin)
        )
        near = [c for c in pool if float(c.score) >= (best_score - margin)]
        near.sort(key=lambda c: (float(c.time_abs_sec), -float(c.score)))
        return near[0] if near else max(pool, key=lambda c: float(c.score))

    def _drop_followthrough(self, candidate: CandidateScore) -> float:
        return float(
            (0.40 * float(candidate.norm_density))
            + (0.28 * float(candidate.norm_sustain))
            + (0.20 * float(candidate.norm_lowend))
            + (0.12 * float(candidate.norm_contrast))
        )

    def _drop_bang_score(self, candidate: CandidateScore) -> float:
        return float(
            float(candidate.score)
            + (0.12 * float(candidate.norm_onset))
            + (0.10 * float(candidate.norm_lowend))
            + (0.08 * float(candidate.norm_ableton))
            + (0.06 * float(candidate.norm_density))
            - (0.10 * float(candidate.norm_weak))
            - (0.08 * float(candidate.norm_fake))
        )

    def _is_drop_ready_candidate(self, candidate: CandidateScore) -> bool:
        followthrough = self._drop_followthrough(candidate)
        bang = max(float(candidate.norm_onset), float(candidate.norm_lowend), float(candidate.norm_ableton))
        penalty = max(float(candidate.norm_weak), float(candidate.norm_fake))
        bar_index = int(candidate.bar_index)

        if float(candidate.score) < 0.54 and not (float(candidate.norm_ableton) >= 0.86 and bang >= 0.58):
            return False
        if bang < 0.38:
            return False
        if followthrough < 0.34 and float(candidate.norm_ableton) < 0.82:
            return False
        if penalty >= 0.88 and float(candidate.norm_ableton) < 0.90:
            return False

        # Early bars often contain tails, pickups, or intro hits. They must show
        # real drum-body density before they can define the first drop event.
        if bar_index < 4:
            if float(candidate.score) < 0.74 and float(candidate.norm_ableton) < 0.86:
                return False
            if float(candidate.norm_density) < 0.55 or followthrough < 0.58:
                return False
        elif bar_index < 8:
            if float(candidate.norm_density) < 0.34 and float(candidate.norm_ableton) < 0.88:
                return False

        return True

    def _is_strong_drop_seed(self, candidate: CandidateScore) -> bool:
        followthrough = self._drop_followthrough(candidate)
        bang = max(float(candidate.norm_onset), float(candidate.norm_lowend), float(candidate.norm_ableton))
        if not self._is_drop_ready_candidate(candidate):
            return False
        if float(candidate.score) < 0.58:
            return False
        if bang < 0.74:
            return False
        if followthrough < 0.56:
            return False
        if float(candidate.norm_density) < 0.40 and float(candidate.norm_ableton) < 0.90:
            return False
        return True

    def _choose_first_drop_event_candidate(self, pool: Sequence[CandidateScore]) -> Optional[CandidateScore]:
        ready = [c for c in pool if bool(c.valid) and self._is_drop_ready_candidate(c)]
        if not ready:
            return None
        ready.sort(key=lambda c: (float(c.time_abs_sec), -self._drop_bang_score(c)))

        seeds = [c for c in ready if self._is_strong_drop_seed(c)]
        if not seeds:
            return None
        first = seeds[0]
        beat_sec = 60.0 / max(1e-6, float(first.bpm))
        max_gap_sec = 1.35 * beat_sec
        max_event_span_sec = 4.25 * beat_sec
        event = [first]
        last_t = float(first.time_abs_sec)
        for cand in [c for c in ready if float(c.time_abs_sec) > float(first.time_abs_sec)]:
            t = float(cand.time_abs_sec)
            if (t - last_t) > max_gap_sec:
                break
            if (t - float(first.time_abs_sec)) > max_event_span_sec:
                break
            event.append(cand)
            last_t = t

        return max(
            event,
            key=lambda c: (
                self._drop_bang_score(c),
                float(c.norm_onset),
                float(c.norm_lowend),
                -float(c.time_abs_sec),
            ),
        )

    def _rough_region_score(self, candidate: CandidateScore) -> float:
        followthrough = (
            (0.34 * float(candidate.norm_density))
            + (0.24 * float(candidate.norm_sustain))
            + (0.22 * float(candidate.norm_lowend))
            + (0.20 * float(candidate.norm_repeat))
        )
        structure = (
            (0.30 * float(candidate.norm_phrase))
            + (0.24 * followthrough)
            + (0.18 * float(candidate.norm_contrast))
            + (0.12 * float(candidate.norm_grid))
            + (0.10 * self._candidate_confidence(candidate))
            + (0.06 * float(candidate.norm_onset))
        )
        penalties = (
            (0.18 * float(candidate.norm_preroll))
            + (0.12 * float(candidate.norm_weak))
            + (0.10 * float(candidate.norm_fake))
        )
        return float(structure - penalties)

    def _rough_region_feature_values(
        self,
        *,
        candidate: CandidateScore,
        reference: CandidateScore,
    ) -> Dict[str, float]:
        beat_sec = 60.0 / max(1e-6, float(reference.bpm))
        delta_beats = (float(reference.time_abs_sec) - float(candidate.time_abs_sec)) / max(1e-6, beat_sec)
        rough_followthrough = (
            (0.34 * float(candidate.norm_density))
            + (0.24 * float(candidate.norm_sustain))
            + (0.22 * float(candidate.norm_lowend))
            + (0.20 * float(candidate.norm_repeat))
        )
        return {
            "delta_beats_from_reference": float(delta_beats),
            "candidate_confidence": float(self._candidate_confidence(candidate)),
            "candidate_score": float(candidate.score),
            "rough_followthrough": float(rough_followthrough),
            "norm_phrase": float(candidate.norm_phrase),
            "norm_density": float(candidate.norm_density),
            "norm_sustain": float(candidate.norm_sustain),
            "norm_lowend": float(candidate.norm_lowend),
            "norm_contrast": float(candidate.norm_contrast),
            "norm_grid": float(candidate.norm_grid),
            "norm_onset": float(candidate.norm_onset),
            "norm_repeat": float(candidate.norm_repeat),
            "norm_preroll": float(candidate.norm_preroll),
            "norm_weak": float(candidate.norm_weak),
            "norm_fake": float(candidate.norm_fake),
        }

    def _select_rough_custom_candidate(
        self,
        candidates: Sequence[CandidateScore],
        reference: Optional[CandidateScore],
        bpm_hint: Optional[float],
    ) -> Tuple[Optional[CandidateScore], str]:
        if reference is None:
            return None, "No custom reference candidate was available for rough first-drop estimation."

        pool = [c for c in candidates if c.valid]
        if not pool:
            pool = list(candidates)
        if bpm_hint is not None:
            hinted = [c for c in pool if abs(float(c.bpm) - float(bpm_hint)) <= 0.01]
            if hinted:
                pool = hinted
        if not pool:
            return reference, "Used the custom reference candidate as the rough first-drop estimate because no candidate pool was available."

        ref_conf = self._candidate_confidence(reference)
        score_margin = float(self.options.rough_region_score_margin)
        conf_margin = float(self.options.rough_region_conf_margin)
        min_density = max(float(self.options.rough_region_min_density), float(reference.norm_density) - 0.30)
        min_phrase = max(float(self.options.rough_region_min_phrase), float(reference.norm_phrase) - 0.35)
        min_followthrough = float(self.options.rough_region_min_followthrough)
        ref_density_floor = float(reference.norm_density) - float(self.options.rough_region_ref_density_margin)
        ref_lowend_floor = float(reference.norm_lowend) - float(self.options.rough_region_ref_lowend_margin)
        model = self._get_rough_region_model()
        blend = float(np.clip(float(self.options.rough_region_model_blend), 0.0, 1.0))
        candidate_rows: List[Tuple[CandidateScore, float, float, Optional[float], float]] = []
        for c in pool:
            if float(c.time_abs_sec) > float(reference.time_abs_sec):
                continue
            if float(c.score) < (float(reference.score) - score_margin):
                continue
            conf = self._candidate_confidence(c)
            if conf < (ref_conf - conf_margin):
                continue
            followthrough = (
                (0.34 * float(c.norm_density))
                + (0.24 * float(c.norm_sustain))
                + (0.22 * float(c.norm_lowend))
                + (0.20 * float(c.norm_repeat))
            )
            if float(c.norm_density) < min_density:
                continue
            if c is not reference and float(c.norm_density) < ref_density_floor:
                continue
            if c is not reference and float(c.norm_lowend) < ref_lowend_floor:
                continue
            if float(c.norm_phrase) < min_phrase:
                continue
            if float(followthrough) < min_followthrough:
                continue
            heuristic_score = self._rough_region_score(c)
            model_prob = None
            final_score = float(heuristic_score)
            if model is not None and blend > 0.0:
                model_prob = float(model.predict_proba(self._rough_region_feature_values(candidate=c, reference=reference)))
                final_score = ((1.0 - blend) * float(heuristic_score)) + (blend * float(model_prob))
            candidate_rows.append((c, heuristic_score, followthrough, model_prob, final_score))

        if not candidate_rows:
            return reference, "Used the custom reference candidate as the rough first-drop estimate because no earlier structural custom candidate passed the rough-region filters."

        best_rough = max(float(final_score) for _, _, _, _, final_score in candidate_rows)
        near_best_margin = float(self.options.rough_region_near_best_margin)
        rough_pool = [
            (c, heuristic_score, followthrough, model_prob, final_score)
            for c, heuristic_score, followthrough, model_prob, final_score in candidate_rows
            if float(final_score) >= (best_rough - near_best_margin)
        ]
        rough_pool.sort(
            key=lambda item: (
                float(item[0].time_abs_sec),
                -float(item[4]),
                -float(item[1]),
                -float(item[2]),
                -self._candidate_confidence(item[0]),
            )
        )
        rough, rough_score, followthrough, model_prob, final_score = rough_pool[0]
        delta_sec = float(reference.time_abs_sec) - float(rough.time_abs_sec)
        delta_beats = delta_sec / max(1e-6, 60.0 / max(1e-6, float(reference.bpm)))
        if rough is reference or abs(delta_sec) <= 1e-6:
            return rough, "Used the custom reference candidate itself as the rough first-drop estimate."
        return (
            rough,
            f"Promoted an earlier rough first-drop region at {rough.time_abs_sec:.3f}s because it was the earliest near-best "
            f"structural section-entry candidate (rough_score={rough_score:.2f}, followthrough={followthrough:.2f}, "
            f"final_score={final_score:.2f}{f', model_prob={model_prob:.2f}' if model_prob is not None else ''}) "
            f"while staying {delta_beats:.2f} beats before the custom reference.",
        )

    def _snap_to_rough_ableton_marker(
        self,
        *,
        rough_candidate: Optional[CandidateScore],
        rough_anchor_sec: Optional[float],
        ableton_candidates: Sequence[CandidateScore],
        ableton_markers: Optional["AbletonAnalysisMarkers"],
        analysis: Dict[str, Any],
    ) -> Tuple[Optional[CandidateScore], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "used": False,
            "reason": "",
            "rough_candidate_seconds": float(rough_candidate.time_abs_sec) if rough_candidate is not None else None,
            "rough_anchor_seconds": float(rough_anchor_sec) if rough_anchor_sec is not None else None,
            "search_window_beats": float(self.options.ableton_snap_search_beats),
            "max_late_drift_beats": float(self.options.ableton_snap_max_late_beats),
            "considered_markers": [],
            "events": [],
            "selected_event_id": None,
            "selected_event_score": None,
            "selected_event_reason": None,
            "marker_rank_model_path": None,
            "marker_rank_model_label": None,
            "marker_rank_model_blend": None,
            "chosen_marker_seconds": None,
            "chosen_candidate_seconds": None,
            "chosen_marker_reason": None,
            "late_relative_to_rough": None,
            "cluster_start_seconds": None,
            "cluster_end_seconds": None,
            "cluster_markers_seconds": [],
            "cluster_marker_count": 0,
            "cluster_plausible_marker_count": 0,
            "initial_preferred_marker_seconds": None,
            "first_plausible_marker_seconds": None,
            "previous_marker_seconds": None,
            "previous_marker_exists": False,
            "previous_marker_won": False,
            "previous_marker_rejection_reason": None,
            "previous_marker_score_gap": None,
            "earliest_near_best_marker_seconds": None,
            "final_is_first_plausible_marker": False,
            "self_check_passed": None,
            "self_check_reason": None,
            "rejected_earlier_markers": [],
        }
        if rough_candidate is None:
            debug["reason"] = "No rough custom first-drop estimate was available."
            return None, debug
        if ableton_markers is None or not ableton_markers.candidate_seconds:
            debug["reason"] = "No Ableton `.asd` markers were available."
            return None, debug
        if not ableton_candidates:
            debug["reason"] = "No Ableton-backed beat-grid candidates were available for snap selection."
            return None, debug

        beat_sec = 60.0 / max(1e-6, float(rough_candidate.bpm))
        rough_sec = float(rough_anchor_sec) if rough_anchor_sec is not None else float(rough_candidate.time_abs_sec)
        search_window_sec = float(self.options.ableton_snap_search_beats) * beat_sec
        max_late_beats = float(self.options.ableton_snap_max_late_beats)
        min_support = float(self.options.ableton_snap_min_support)
        cluster_gap_sec = float(self.options.ableton_cluster_max_spacing_beats) * beat_sec
        cluster_span_sec = float(self.options.ableton_cluster_window_beats) * beat_sec

        marker_times = sorted(set(float(v) for v in ableton_markers.candidate_seconds if v is not None))
        nearby_markers = [sec for sec in marker_times if abs(float(sec) - rough_sec) <= search_window_sec]
        debug["nearby_raw_markers_seconds"] = nearby_markers
        if not nearby_markers:
            debug["reason"] = (
                f"No Ableton markers fell within +/-{self.options.ableton_snap_search_beats:.2f} beats "
                f"of the rough custom first-drop estimate at {rough_sec:.3f}s."
            )
            return None, debug

        best_by_time: Dict[int, CandidateScore] = {}
        for cand in ableton_candidates:
            key = int(round(float(cand.time_abs_sec) * 1000.0))
            prev = best_by_time.get(key)
            if prev is None or self._candidate_confidence(cand) > self._candidate_confidence(prev) or float(cand.score) > float(prev.score):
                best_by_time[key] = cand
        unique_candidates = [
            cand
            for cand in best_by_time.values()
            if abs(float(cand.time_abs_sec) - rough_sec) <= search_window_sec
        ]
        if not unique_candidates:
            debug["reason"] = "No Ableton-backed exact-anchor candidates fell inside the rough-region snap window."
            debug["considered_markers"] = []
            return None, debug

        feats = analysis["feats"]
        hop_sec = float(analysis["hop_sec"])
        offset_sec = float(analysis["offset_sec"])
        rms_signal = np.asarray(feats["rms"], dtype=np.float32)
        onset_signal = np.asarray(feats["onset"], dtype=np.float32)
        low_signal = np.asarray(feats["low_transient"], dtype=np.float32)
        attack_signal = np.maximum(
            onset_signal,
            low_signal,
        )

        def _window_mean(x: np.ndarray, start_sec: float, end_sec: float) -> float:
            if x.size == 0:
                return 0.0
            i0 = max(0, int(math.floor(start_sec / hop_sec)))
            i1 = min(x.size, int(math.ceil(end_sec / hop_sec)))
            if i1 <= i0:
                return 0.0
            return _safe_mean(x[i0:i1], fallback=0.0)

        def _window_max(x: np.ndarray, start_sec: float, end_sec: float) -> float:
            if x.size == 0:
                return 0.0
            i0 = max(0, int(math.floor(start_sec / hop_sec)))
            i1 = min(x.size, int(math.ceil(end_sec / hop_sec)))
            if i1 <= i0:
                return 0.0
            return float(np.max(x[i0:i1]))

        ordered_candidates = sorted(unique_candidates, key=lambda c: float(c.time_abs_sec))
        ordered_times = [float(c.time_abs_sec) for c in ordered_candidates]
        raw_attack: List[float] = []
        raw_low: List[float] = []
        raw_attack_edge: List[float] = []
        raw_attack_started: List[float] = []
        raw_inside_body: List[float] = []
        raw_pre_rms: List[float] = []
        raw_post_rms: List[float] = []
        raw_sustain_after: List[float] = []
        raw_density_after: List[float] = []
        raw_low_growth: List[float] = []
        raw_pre_post_ratio: List[float] = []
        raw_dense_run_after: List[float] = []
        raw_isolated_preroll: List[float] = []
        raw_gap_prev: List[float] = []
        raw_gap_next: List[float] = []
        for cand in ordered_candidates:
            marker_rel = max(0.0, float(cand.time_abs_sec) - offset_sec)
            onset_peak = _local_peak_near_time(onset_signal, marker_rel, hop_sec, 0.14 * beat_sec)
            low_peak = _local_peak_near_time(low_signal, marker_rel, hop_sec, 0.18 * beat_sec)
            raw_attack.append((0.60 * low_peak) + (0.40 * onset_peak))
            raw_low.append(low_peak)
            pre_energy = _window_mean(rms_signal, marker_rel - (0.16 * beat_sec), marker_rel - (0.02 * beat_sec))
            post_energy = _window_mean(rms_signal, marker_rel, marker_rel + (0.16 * beat_sec))
            sustain_after = _window_mean(rms_signal, marker_rel + (0.06 * beat_sec), marker_rel + (0.52 * beat_sec))
            density_after = _window_mean(onset_signal, marker_rel + (0.02 * beat_sec), marker_rel + (0.55 * beat_sec))
            low_before = _window_mean(low_signal, marker_rel - (0.22 * beat_sec), marker_rel - (0.02 * beat_sec))
            low_after = _window_mean(low_signal, marker_rel, marker_rel + (0.36 * beat_sec))
            pre_attack = _window_max(attack_signal, marker_rel - (0.12 * beat_sec), marker_rel - (0.01 * beat_sec))
            post_attack = _window_max(attack_signal, marker_rel, marker_rel + (0.12 * beat_sec))
            raw_attack_edge.append(max(0.0, post_energy - pre_energy) + (0.65 * max(0.0, post_attack - pre_attack)))
            raw_attack_started.append(max(0.0, pre_attack - (0.70 * post_attack)))
            raw_inside_body.append(max(0.0, pre_energy - (0.80 * post_energy)) + max(0.0, pre_attack - post_attack))
            raw_pre_rms.append(pre_energy)
            raw_post_rms.append(post_energy)
            raw_sustain_after.append(sustain_after)
            raw_density_after.append(density_after)
            raw_low_growth.append(max(0.0, low_after - low_before))
            raw_pre_post_ratio.append(post_energy / max(1e-6, pre_energy))

        for idx, marker_sec in enumerate(ordered_times):
            prev_gap_beats = float("inf") if idx <= 0 else float(marker_sec - ordered_times[idx - 1]) / max(1e-6, beat_sec)
            next_gap_beats = float("inf") if idx >= (len(ordered_times) - 1) else float(ordered_times[idx + 1] - marker_sec) / max(1e-6, beat_sec)
            dense_run_count = 0
            for ahead_sec in ordered_times[idx + 1 :]:
                if float(ahead_sec - marker_sec) <= (0.55 * beat_sec):
                    dense_run_count += 1
                else:
                    break
            dense_run_after = min(1.0, float(dense_run_count) / 3.0)
            isolated_preroll = (
                max(0.0, next_gap_beats - 0.30)
                + max(0.0, 0.20 - raw_sustain_after[idx])
                + max(0.0, 0.12 - raw_density_after[idx])
                + max(0.0, 0.18 - dense_run_after)
            )
            raw_gap_prev.append(prev_gap_beats)
            raw_gap_next.append(next_gap_beats)
            raw_dense_run_after.append(dense_run_after)
            raw_isolated_preroll.append(isolated_preroll)

        attack_norm = _rank_norm(raw_attack, invert=False)
        low_norm = _rank_norm(raw_low, invert=False)
        attack_edge_norm = _rank_norm(raw_attack_edge, invert=False)
        attack_started_norm = _rank_norm(raw_attack_started, invert=False)
        inside_body_norm = _rank_norm(raw_inside_body, invert=False)
        sustain_after_norm = _rank_norm(raw_sustain_after, invert=False)
        density_after_norm = _rank_norm(raw_density_after, invert=False)
        low_growth_norm = _rank_norm(raw_low_growth, invert=False)
        pre_post_ratio_norm = _rank_norm(raw_pre_post_ratio, invert=False)
        dense_run_norm = _rank_norm(raw_dense_run_after, invert=False)
        isolated_preroll_norm = _rank_norm(raw_isolated_preroll, invert=False)
        gap_prev_norm = _rank_norm(raw_gap_prev, invert=False)
        gap_next_norm = _rank_norm(raw_gap_next, invert=False)

        considered: List[Dict[str, Any]] = []
        plausible: List[Dict[str, Any]] = []
        for idx, cand in enumerate(ordered_candidates):
            marker_sec = float(cand.time_abs_sec)
            delta_beats = (marker_sec - rough_sec) / max(1e-6, beat_sec)
            early_beats = max(0.0, -delta_beats)
            late_beats = max(0.0, delta_beats)
            late_excess = max(0.0, late_beats - max_late_beats)
            marker_support = float(max(attack_norm[idx], low_norm[idx], float(cand.norm_ableton)))

            candidate_conf = self._candidate_confidence(cand)
            structural = (
                (0.52 * candidate_conf)
                + (0.15 * float(cand.norm_phrase))
                + (0.12 * float(cand.norm_density))
                + (0.10 * float(cand.norm_grid))
                + (0.06 * float(cand.norm_lowend))
                + (0.05 * float(cand.norm_contrast))
            )
            early_bonus = float(self.options.ableton_snap_early_bias) * min(1.5, early_beats)
            distance_penalty = (0.05 * early_beats) + (0.10 * late_beats)
            late_penalty = float(self.options.ableton_snap_late_section_penalty) * late_excess
            event_followthrough_norm = (
                (0.50 * sustain_after_norm[idx])
                + (0.30 * density_after_norm[idx])
                + (0.20 * dense_run_norm[idx])
            )
            snap_score = (
                structural
                + (0.10 * marker_support)
                + (0.08 * attack_norm[idx])
                + (0.05 * low_norm[idx])
                + (0.10 * event_followthrough_norm)
                + (0.05 * pre_post_ratio_norm[idx])
                + early_bonus
                - distance_penalty
                - late_penalty
                - (0.08 * isolated_preroll_norm[idx])
            )

            rejection_reason = ""
            plausible_marker = True
            if not bool(cand.valid):
                plausible_marker = False
                rejection_reason = "Matched Ableton-backed candidate failed the musical validity filter."
            elif marker_support < min_support:
                plausible_marker = False
                rejection_reason = f"Marker support {marker_support:.2f} fell below the minimum {min_support:.2f}."
            elif late_beats > (max_late_beats + 1.0):
                plausible_marker = False
                rejection_reason = (
                    f"Marker drifted {late_beats:.2f} beats late relative to the rough first-drop estimate, "
                    "which is too far for the exact-anchor snap stage."
                )

            entry = {
                "marker_seconds": float(marker_sec),
                "delta_beats_from_rough": float(delta_beats),
                "matched_candidate_seconds": float(cand.time_abs_sec),
                "matched_candidate_confidence": float(candidate_conf),
                "matched_candidate_score": float(cand.score),
                "marker_support": float(marker_support),
                "attack_support": float(attack_norm[idx]),
                "low_support": float(low_norm[idx]),
                "attack_edge_norm": float(attack_edge_norm[idx]),
                "attack_started_norm": float(attack_started_norm[idx]),
                "inside_body_norm": float(inside_body_norm[idx]),
                "pre_rms": float(raw_pre_rms[idx]),
                "post_rms": float(raw_post_rms[idx]),
                "sustain_after": float(raw_sustain_after[idx]),
                "density_after": float(raw_density_after[idx]),
                "low_growth": float(raw_low_growth[idx]),
                "pre_post_ratio": float(raw_pre_post_ratio[idx]),
                "gap_prev_beats": float(raw_gap_prev[idx]),
                "gap_next_beats": float(raw_gap_next[idx]),
                "dense_run_after": float(raw_dense_run_after[idx]),
                "isolated_preroll_raw": float(raw_isolated_preroll[idx]),
                "sustain_after_norm": float(sustain_after_norm[idx]),
                "density_after_norm": float(density_after_norm[idx]),
                "low_growth_norm": float(low_growth_norm[idx]),
                "pre_post_ratio_norm": float(pre_post_ratio_norm[idx]),
                "dense_run_after_norm": float(dense_run_norm[idx]),
                "isolated_preroll_norm": float(isolated_preroll_norm[idx]),
                "gap_prev_norm": float(gap_prev_norm[idx]),
                "gap_next_norm": float(gap_next_norm[idx]),
                "event_followthrough_norm": float(event_followthrough_norm),
                "late_relative_to_rough": bool(late_beats > max_late_beats),
                "snap_score": float(snap_score),
                "attack_start_score": None,
                "cluster_local_score": None,
                "marker_rank_score": None,
                "event_id": None,
                "is_selected_event": False,
                "plausible": bool(plausible_marker),
                "rejection_reason": rejection_reason or None,
                "chosen": False,
            }
            considered.append(entry)
            if plausible_marker:
                plausible.append({"entry": entry, "candidate": cand})

        if not plausible:
            debug["reason"] = "No plausible Ableton marker survived the rough-region snap filter."
            debug["considered_markers"] = considered
            return None, debug

        cluster_ranges: List[Tuple[int, int]] = []
        start_idx = 0
        for idx in range(1, len(considered) + 1):
            split = idx >= len(considered)
            if not split:
                gap_sec = float(considered[idx]["marker_seconds"]) - float(considered[idx - 1]["marker_seconds"])
                span_sec = float(considered[idx]["marker_seconds"]) - float(considered[start_idx]["marker_seconds"])
                if gap_sec > cluster_gap_sec or span_sec > cluster_span_sec:
                    split = True
            if split:
                cluster_ranges.append((start_idx, idx))
                start_idx = idx

        cluster_infos: List[Dict[str, Any]] = []
        for cluster_id, (lo, hi) in enumerate(cluster_ranges):
            entries = considered[lo:hi]
            start_sec = float(entries[0]["marker_seconds"])
            end_sec = float(entries[-1]["marker_seconds"])
            if start_sec <= rough_sec <= end_sec:
                dist_beats = 0.0
            else:
                dist_beats = min(abs(start_sec - rough_sec), abs(end_sec - rough_sec)) / max(1e-6, beat_sec)
            for idx in range(lo, hi):
                considered[idx]["cluster_id"] = int(cluster_id)
                considered[idx]["cluster_index"] = int(idx - lo)
            cluster_infos.append(
                {
                    "cluster_id": int(cluster_id),
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "distance_beats": float(dist_beats),
                    "entries": entries,
                }
            )

        cluster_min_support = float(self.options.ableton_cluster_earliest_min_support)
        max_post_attack_drift = float(self.options.ableton_cluster_max_post_attack_drift_beats)
        max_early_pull = float(self.options.ableton_cluster_max_early_pull_beats)
        prev_bonus = float(self.options.ableton_previous_marker_promotion_bonus)
        later_penalty_weight = float(self.options.ableton_later_in_cluster_penalty)
        attack_start_reward = float(self.options.ableton_attack_start_reward)
        attack_started_penalty = float(self.options.ableton_attack_started_penalty)
        inside_body_penalty = float(self.options.ableton_inside_body_penalty)
        attack_start_min_score = float(self.options.ableton_attack_start_min_score)
        marker_prefer_margin = max(float(self.options.ableton_marker_prefer_earlier_margin), float(self.options.ableton_snap_near_best_margin))
        event_prefer_margin = float(self.options.ableton_event_prefer_earlier_margin)
        min_followthrough = max(0.18, min(float(self.options.ableton_event_min_sustain_norm), float(self.options.ableton_event_min_density_norm)))

        event_rows: List[Dict[str, Any]] = []
        for cluster_info in cluster_infos:
            cluster_entries = list(cluster_info["entries"])
            plausible_cluster = [entry for entry in cluster_entries if bool(entry["plausible"])]
            if not plausible_cluster:
                plausible_cluster = list(cluster_entries)
            first_plausible_entry: Optional[Dict[str, Any]] = None
            for idx_in_cluster, entry in enumerate(cluster_entries):
                pos_norm = float(idx_in_cluster) / float(max(1, len(cluster_entries) - 1))
                later_penalty = later_penalty_weight * pos_norm
                attack_start_score = (
                    (0.42 * float(entry["attack_edge_norm"]))
                    + (0.18 * float(entry["marker_support"]))
                    + (0.15 * float(entry["pre_post_ratio_norm"]))
                    + (0.15 * float(entry["event_followthrough_norm"]))
                    + (0.10 * float(entry["matched_candidate_confidence"]))
                    - (0.36 * float(entry["attack_started_norm"]))
                    - (0.28 * float(entry["inside_body_norm"]))
                )
                cluster_local_score = (
                    float(entry["snap_score"])
                    + (attack_start_reward * float(entry["attack_edge_norm"]))
                    + (0.16 * float(entry["event_followthrough_norm"]))
                    + (0.08 * float(entry["low_growth_norm"]))
                    - (attack_started_penalty * float(entry["attack_started_norm"]))
                    - (inside_body_penalty * float(entry["inside_body_norm"]))
                    - (0.10 * float(entry["isolated_preroll_norm"]))
                    - later_penalty
                )
                marker_rank_score = (
                    float(cluster_local_score)
                    + (0.12 * (1.0 - pos_norm))
                    + (0.08 * float(entry["dense_run_after_norm"]))
                    + (0.07 * float(entry["sustain_after_norm"]))
                    + (0.05 * float(entry["density_after_norm"]))
                )
                strong_late_attack_start = bool(
                    max(0.0, float(entry["delta_beats_from_rough"])) <= float(self.options.ableton_snap_max_late_beats)
                    and float(entry["marker_support"]) >= max(0.60, cluster_min_support + 0.05)
                    and float(attack_start_score) >= max(0.30, attack_start_min_score + 0.18)
                    and float(entry["event_followthrough_norm"]) >= max(0.55, min_followthrough + 0.25)
                )
                entry["event_id"] = int(cluster_info["cluster_id"])
                entry["attack_start_score"] = float(attack_start_score)
                entry["cluster_local_score"] = float(cluster_local_score)
                entry["marker_rank_score"] = float(marker_rank_score)
                entry["strong_late_attack_start"] = bool(strong_late_attack_start)
                entry["start_plausible"] = bool(
                    bool(entry["plausible"])
                    and float(entry["marker_support"]) >= cluster_min_support
                    and (
                        max(0.0, float(entry["delta_beats_from_rough"])) <= max_post_attack_drift
                        or bool(strong_late_attack_start)
                    )
                    and max(0.0, -float(entry["delta_beats_from_rough"])) <= max_early_pull
                    and float(entry["attack_start_score"]) >= attack_start_min_score
                    and float(entry["event_followthrough_norm"]) >= min_followthrough
                )
                if first_plausible_entry is None and bool(entry["start_plausible"]):
                    first_plausible_entry = entry

            anchor_entry = first_plausible_entry or max(cluster_entries, key=lambda entry: float(entry.get("marker_rank_score") or -1e9))
            anchor_sec = float(anchor_entry["marker_seconds"])
            delta_beats = (anchor_sec - rough_sec) / max(1e-6, beat_sec)
            early_beats = max(0.0, -delta_beats)
            late_beats = max(0.0, delta_beats)
            cluster_size_norm = min(1.0, float(max(0, len(cluster_entries) - 1)) / 2.0)
            event_start_score = float(anchor_entry.get("attack_start_score") or 0.0)
            event_followthrough = max(float(entry["event_followthrough_norm"]) for entry in cluster_entries)
            event_sustain = max(float(entry["sustain_after_norm"]) for entry in cluster_entries)
            event_density = max(float(entry["density_after_norm"]) for entry in cluster_entries)
            event_low_growth = max(float(entry["low_growth_norm"]) for entry in cluster_entries)
            event_preroll = float(anchor_entry["isolated_preroll_norm"])
            proximity_bonus = max(0.0, 1.0 - (abs(delta_beats) / max(1e-6, float(self.options.ableton_snap_search_beats))))
            event_score = (
                (0.30 * event_start_score)
                + (0.22 * event_followthrough)
                + (0.16 * event_density)
                + (0.10 * event_sustain)
                + (0.08 * event_low_growth)
                + (0.06 * cluster_size_norm)
                + (0.08 * proximity_bonus)
                - (0.07 * early_beats)
                - (0.15 * late_beats)
                - (0.12 * event_preroll)
            )
            cluster_info["anchor_entry"] = anchor_entry
            cluster_info["anchor_seconds"] = float(anchor_sec)
            cluster_info["event_score"] = float(event_score)
            cluster_info["first_plausible_entry"] = first_plausible_entry
            cluster_info["plausible_count"] = int(sum(1 for entry in cluster_entries if bool(entry["start_plausible"])))
            cluster_info["event_reason"] = (
                f"start={event_start_score:.2f}, followthrough={event_followthrough:.2f}, density={event_density:.2f}, "
                f"sustain={event_sustain:.2f}, low_growth={event_low_growth:.2f}, early={early_beats:.2f}, late={late_beats:.2f}, "
                f"preroll={event_preroll:.2f}"
            )
            event_rows.append(
                {
                    "event_id": int(cluster_info["cluster_id"]),
                    "start_seconds": float(cluster_info["start_seconds"]),
                    "end_seconds": float(cluster_info["end_seconds"]),
                    "marker_count": int(len(cluster_entries)),
                    "plausible_marker_count": int(cluster_info["plausible_count"]),
                    "anchor_seconds": float(anchor_sec),
                    "first_plausible_marker_seconds": float(first_plausible_entry["marker_seconds"]) if first_plausible_entry is not None else None,
                    "distance_beats": float(cluster_info["distance_beats"]),
                    "event_score": float(event_score),
                    "event_reason": str(cluster_info["event_reason"]),
                    "marker_seconds": [float(entry["marker_seconds"]) for entry in cluster_entries],
                }
            )

        best_event_score = max(float(info["event_score"]) for info in cluster_infos)
        near_best_events = [info for info in cluster_infos if float(info["event_score"]) >= (best_event_score - event_prefer_margin)]
        target_cluster = min(
            near_best_events,
            key=lambda info: (
                float(info.get("anchor_seconds") or info["start_seconds"]),
                float(info["distance_beats"]),
                -float(info["event_score"]),
            ),
        )
        cluster_entries = list(target_cluster["entries"])
        plausible_cluster = [entry for entry in cluster_entries if bool(entry["plausible"])]
        if not plausible_cluster:
            plausible_cluster = list(cluster_entries)
        first_plausible_entry = target_cluster.get("first_plausible_entry")

        debug["events"] = event_rows
        debug["selected_event_id"] = int(target_cluster["cluster_id"])
        debug["selected_event_score"] = float(target_cluster["event_score"])
        debug["selected_event_reason"] = str(target_cluster["event_reason"])
        debug["cluster_start_seconds"] = float(target_cluster["start_seconds"])
        debug["cluster_end_seconds"] = float(target_cluster["end_seconds"])
        debug["cluster_markers_seconds"] = [float(entry["marker_seconds"]) for entry in cluster_entries]
        debug["cluster_marker_count"] = int(len(cluster_entries))
        debug["cluster_plausible_marker_count"] = int(sum(1 for entry in cluster_entries if bool(entry.get("start_plausible"))))
        debug["previous_marker_promotion_bonus"] = float(self.options.ableton_previous_marker_promotion_bonus)
        for entry in cluster_entries:
            entry["is_selected_event"] = True
        self._apply_marker_rank_model(
            cluster_entries=cluster_entries,
            event_score=float(target_cluster["event_score"]),
            event_marker_count=int(len(cluster_entries)),
            event_plausible_count=int(sum(1 for entry in cluster_entries if bool(entry.get("start_plausible")))),
            debug=debug,
        )

        initial_preferred_entry = max(plausible_cluster, key=lambda entry: float(entry.get("marker_rank_score") or entry["snap_score"]))
        best_marker_score = float(initial_preferred_entry.get("marker_rank_score") or initial_preferred_entry["snap_score"])
        near_best_markers = [
            entry
            for entry in cluster_entries
            if bool(entry["plausible"])
            and float(entry.get("marker_rank_score") or entry["snap_score"]) >= (best_marker_score - marker_prefer_margin)
        ]
        if not near_best_markers:
            near_best_markers = list(cluster_entries)
        start_pool = [entry for entry in near_best_markers if bool(entry.get("start_plausible"))]
        earliest_near_best_entry = min(
            start_pool or near_best_markers,
            key=lambda entry: (float(entry["marker_seconds"]), -float(entry.get("marker_rank_score") or entry["snap_score"])),
        )
        debug["initial_preferred_marker_seconds"] = float(initial_preferred_entry["marker_seconds"])
        debug["earliest_near_best_marker_seconds"] = float(earliest_near_best_entry["marker_seconds"])
        debug["first_plausible_marker_seconds"] = float(first_plausible_entry["marker_seconds"]) if first_plausible_entry is not None else None

        previous_entry = None
        try:
            chosen_idx_seed = cluster_entries.index(earliest_near_best_entry)
        except ValueError:
            chosen_idx_seed = -1
        if chosen_idx_seed > 0:
            previous_entry = cluster_entries[chosen_idx_seed - 1]
        debug["previous_marker_exists"] = bool(previous_entry is not None)
        debug["previous_marker_seconds"] = float(previous_entry["marker_seconds"]) if previous_entry is not None else None

        chosen_entry = earliest_near_best_entry
        selection_reason = "Selected the earliest near-best Ableton marker inside the chosen local drop event."
        if previous_entry is not None and bool(previous_entry.get("start_plausible")):
            prev_score = float(previous_entry.get("marker_rank_score") or previous_entry["snap_score"])
            current_score = float(chosen_entry.get("marker_rank_score") or chosen_entry["snap_score"])
            score_gap = current_score - prev_score
            debug["previous_marker_score_gap"] = float(score_gap)
            previous_better_boundary = (
                float(previous_entry["attack_started_norm"]) + 0.08 < float(chosen_entry["attack_started_norm"])
                or float(previous_entry["inside_body_norm"]) + 0.08 < float(chosen_entry["inside_body_norm"])
                or float(previous_entry["attack_edge_norm"]) > float(chosen_entry["attack_edge_norm"]) + 0.05
            )
            if (prev_score + prev_bonus) >= current_score or previous_better_boundary:
                chosen_entry = previous_entry
                debug["previous_marker_won"] = True
                selection_reason = (
                    "Promoted the previous plausible Ableton marker in the same local event because the later marker "
                    "looked deeper inside the attack/body."
                )
            else:
                debug["previous_marker_rejection_reason"] = (
                    "A plausible previous marker existed, but the later marker kept materially stronger local event-start evidence."
                )
        elif previous_entry is not None and not bool(previous_entry.get("start_plausible")):
            debug["previous_marker_rejection_reason"] = (
                "The previous marker stayed in the same local event but did not qualify as a plausible attack start."
            )
        else:
            debug["previous_marker_rejection_reason"] = (
                "No immediate previous plausible marker existed inside the selected local event."
            )

        self_check_margin = float(self.options.ableton_self_check_margin)
        self_check_passed = True
        self_check_reason = "Chosen marker passed the local sustained-event self-check."
        if first_plausible_entry is not None and chosen_entry is not first_plausible_entry:
            first_score = float(first_plausible_entry.get("marker_rank_score") or first_plausible_entry["snap_score"])
            chosen_score = float(chosen_entry.get("marker_rank_score") or chosen_entry["snap_score"])
            later_body = (
                float(chosen_entry["attack_started_norm"]) > float(first_plausible_entry["attack_started_norm"]) + 0.08
                or float(chosen_entry["inside_body_norm"]) > float(first_plausible_entry["inside_body_norm"]) + 0.08
            )
            if later_body or (first_score + self_check_margin) >= chosen_score:
                chosen_entry = first_plausible_entry
                selection_reason = (
                    "Self-check promoted the first plausible marker in the selected local event over a later body marker."
                )
                self_check_reason = "A later marker looked too far inside the event body, so the first plausible marker won."
            else:
                self_check_passed = False
                self_check_reason = (
                    "Chosen marker remained later than the first plausible event-start marker because it had materially "
                    "stronger local event evidence."
                )
        elif not bool(chosen_entry.get("start_plausible")):
            self_check_passed = False
            self_check_reason = "Chosen marker did not pass the local attack-start plausibility check."
        elif float(chosen_entry["event_followthrough_norm"]) < min_followthrough:
            self_check_passed = False
            self_check_reason = "Chosen marker did not show enough sustained energy/density after the attack."

        chosen_idx = considered.index(chosen_entry)
        chosen_candidate = ordered_candidates[chosen_idx]
        chosen_entry["chosen"] = True
        debug["final_is_first_plausible_marker"] = bool(first_plausible_entry is not None and chosen_entry is first_plausible_entry)
        debug["self_check_passed"] = bool(self_check_passed)
        debug["self_check_reason"] = str(self_check_reason)

        for entry in considered:
            if entry["chosen"]:
                continue
            if entry["rejection_reason"]:
                continue
            if int(entry.get("event_id") or -1) != int(target_cluster["cluster_id"]):
                entry["rejection_reason"] = "Marker belonged to a nearby event that scored below the chosen local drop event."
            elif float(entry["marker_seconds"]) < float(chosen_entry["marker_seconds"]):
                entry["rejection_reason"] = "Earlier marker in the selected event lost on attack-start or followthrough evidence."
            else:
                entry["rejection_reason"] = "Lower local event marker rank than the chosen Ableton marker."

        debug["used"] = True
        debug["reason"] = "Snapped the rough custom first-drop region to a scored local Ableton event."
        debug["considered_markers"] = considered
        debug["chosen_marker_seconds"] = float(chosen_entry["marker_seconds"])
        debug["chosen_candidate_seconds"] = float(chosen_candidate.time_abs_sec)
        debug["chosen_marker_support"] = float(chosen_entry["marker_support"])
        debug["chosen_candidate_confidence"] = float(chosen_entry["matched_candidate_confidence"])
        debug["chosen_marker_delta_beats_from_rough"] = float(chosen_entry["delta_beats_from_rough"])
        debug["chosen_attack_edge_norm"] = float(chosen_entry["attack_edge_norm"])
        debug["chosen_attack_started_norm"] = float(chosen_entry["attack_started_norm"])
        debug["chosen_inside_body_norm"] = float(chosen_entry["inside_body_norm"])
        debug["chosen_event_followthrough_norm"] = float(chosen_entry["event_followthrough_norm"])
        debug["chosen_strong_late_attack_start"] = bool(chosen_entry.get("strong_late_attack_start", False))
        debug["chosen_marker_reason"] = (
            f"{selection_reason} Chose Ableton marker {chosen_entry['marker_seconds']:.3f}s from event "
            f"{int(target_cluster['cluster_id'])} near rough region {rough_sec:.3f}s with marker rank "
            f"{float(chosen_entry.get('marker_rank_score') or 0.0):.2f}."
        )
        debug["late_relative_to_rough"] = bool(chosen_entry["late_relative_to_rough"])
        debug["rejected_earlier_markers"] = [
            {
                "marker_seconds": float(entry["marker_seconds"]),
                "delta_beats_from_rough": float(entry["delta_beats_from_rough"]),
                "marker_rank_score": float(entry.get("marker_rank_score") or 0.0),
                "rejection_reason": str(entry["rejection_reason"]),
            }
            for entry in considered
            if float(entry["marker_seconds"]) < float(chosen_entry["marker_seconds"]) and not bool(entry["chosen"])
        ]
        return chosen_candidate, debug

    def _build_reason_string(self, chosen: CandidateScore) -> str:
        return (
            f"Selected earliest near-best downbeat at {chosen.time_abs_sec:.3f}s: "
            f"offset={chosen.downbeat_offset}, bpm={chosen.bpm:.2f}, "
            f"onset={chosen.norm_onset:.2f}, lowend={chosen.norm_lowend:.2f}, "
            f"density={chosen.norm_density:.2f}, phrase={chosen.norm_phrase:.2f}, "
            f"grid={chosen.norm_grid:.2f}, ableton={chosen.norm_ableton:.2f}, "
            f"weak_penalty={chosen.norm_weak:.2f}, fake_penalty={chosen.norm_fake:.2f}."
        )

    def _apply_legacy_guidance(
        self,
        candidates: Sequence[CandidateScore],
        chosen: Optional[CandidateScore],
        legacy_sec: Optional[float],
        legacy_conf: Optional[float],
    ) -> Tuple[Optional[CandidateScore], str]:
        if chosen is None or legacy_sec is None or legacy_conf is None:
            return chosen, ""
        if float(legacy_conf) < float(self.options.legacy_high_confidence):
            return chosen, ""

        beat_sec = 60.0 / max(1e-6, float(chosen.bpm))
        valid = [c for c in candidates if c.valid]
        if not valid:
            return chosen, ""

        window_sec = float(self.options.legacy_guidance_window_beats) * beat_sec
        near_legacy = [c for c in valid if abs(float(c.time_abs_sec) - float(legacy_sec)) <= window_sec]
        if not near_legacy:
            return chosen, ""

        legacy_best_score = max(float(c.score) for c in near_legacy)
        use_ableton_margin = any(float(c.raw_ableton) > 0.0 for c in near_legacy)
        margin = (
            float(self.options.earliest_near_best_margin_ableton)
            if use_ableton_margin
            else float(self.options.earliest_near_best_margin)
        )
        legacy_pool = [c for c in near_legacy if float(c.score) >= (legacy_best_score - margin)]
        legacy_pool.sort(
            key=lambda c: (
                float(c.time_abs_sec),
                -float(c.score),
                abs(float(c.time_abs_sec) - float(legacy_sec)),
            )
        )
        legacy_pick = legacy_pool[0]
        chosen_late_beats = (float(chosen.time_abs_sec) - float(legacy_sec)) / max(1e-9, beat_sec)
        score_gap = float(chosen.score) - float(legacy_pick.score)

        if chosen_late_beats > float(self.options.legacy_override_late_beats) and score_gap <= float(self.options.legacy_override_score_margin):
            if chosen is not legacy_pick:
                chosen.chosen = False
                legacy_pick.chosen = True
            return (
                legacy_pick,
                f"Legacy-guided override: legacy prior {legacy_sec:.3f}s conf={legacy_conf:.2f} kept an earlier valid candidate "
                f"({legacy_pick.time_abs_sec:.3f}s) over a much later pick ({chosen.time_abs_sec:.3f}s).",
            )
        return chosen, ""

    def _apply_explicit_prior_guidance(
        self,
        candidates: Sequence[CandidateScore],
        chosen: Optional[CandidateScore],
        prior_sec: Optional[float],
        prior_conf: Optional[float],
        prior_source: Optional[str],
    ) -> Tuple[Optional[CandidateScore], str]:
        source = str(prior_source or "").strip()
        if chosen is None or prior_sec is None or prior_conf is None or not source or source == "legacy_detector":
            return chosen, ""
        if float(prior_conf) < float(self.options.legacy_high_confidence):
            return chosen, ""

        beat_sec = 60.0 / max(1e-6, float(chosen.bpm))
        valid = [c for c in candidates if c.valid]
        if not valid:
            valid = list(candidates)
        if not valid:
            return chosen, ""

        window_beats = max(float(self.options.legacy_guidance_window_beats), float(self.options.ableton_snap_search_beats) + 0.50)
        window_sec = window_beats * beat_sec
        near_prior = [c for c in valid if abs(float(c.time_abs_sec) - float(prior_sec)) <= window_sec]
        if not near_prior:
            return chosen, ""

        prior_best_score = max(float(c.score) for c in near_prior)
        use_ableton_margin = any(float(c.raw_ableton) > 0.0 for c in near_prior)
        margin = (
            float(self.options.earliest_near_best_margin_ableton)
            if use_ableton_margin
            else float(self.options.earliest_near_best_margin)
        )
        prior_pool = [c for c in near_prior if float(c.score) >= (prior_best_score - margin)]
        prior_pool.sort(
            key=lambda c: (
                abs(float(c.time_abs_sec) - float(prior_sec)),
                float(c.time_abs_sec),
                -float(c.score),
            )
        )
        prior_pick = prior_pool[0]
        chosen_gap_beats = abs(float(chosen.time_abs_sec) - float(prior_sec)) / max(1e-9, beat_sec)
        prior_gap_beats = abs(float(prior_pick.time_abs_sec) - float(prior_sec)) / max(1e-9, beat_sec)
        score_gap = float(chosen.score) - float(prior_pick.score)

        if prior_pick is chosen or chosen_gap_beats <= 0.35:
            return chosen, ""
        if prior_gap_beats > (chosen_gap_beats - 0.10):
            return chosen, ""
        if score_gap > 0.30 and chosen_gap_beats <= window_beats:
            return chosen, ""

        if chosen is not prior_pick:
            chosen.chosen = False
            prior_pick.chosen = True
        return (
            prior_pick,
            f"{source} prior {prior_sec:.3f}s shifted selection toward the first-drop cue region ({prior_pick.time_abs_sec:.3f}s).",
        )

    def _beat_position_from_seconds(self, analysis: Dict[str, Any], sec: float, bpm: float) -> float:
        offset_sec = float(analysis["offset_sec"])
        beat_sec = 60.0 / max(1e-6, float(bpm))
        rel = max(0.0, float(sec) - offset_sec)
        return float(rel / beat_sec)

    def _refine_drums_alignment(
        self,
        drums: StemAudio,
        coarse_sec: float,
        beat_sec: float,
        beat_position: float,
        confidence: float,
        chosen: Optional[CandidateScore],
    ) -> StemAlignment:
        coarse_sample = int(round(float(coarse_sec) * float(drums.native_sr)))
        y = drums.native_y.astype(np.float32, copy=False)
        env = _moving_average_samples(np.abs(y), max(1, int(round(0.0025 * drums.native_sr))))
        diff = np.maximum(0.0, np.diff(env, prepend=env[:1]))

        back = int(round(float(self.options.drums_refine_back_beats) * beat_sec * drums.native_sr))
        fwd = int(round(float(self.options.drums_refine_fwd_beats) * beat_sec * drums.native_sr))
        i0 = max(1, coarse_sample - max(1, back))
        i1 = min(len(y) - 2, coarse_sample + max(2, fwd))
        if i1 <= i0:
            i0 = max(1, coarse_sample - int(round(0.020 * drums.native_sr)))
            i1 = min(len(y) - 2, coarse_sample + int(round(0.040 * drums.native_sr)))

        local_env = env[i0 : i1 + 1]
        local_diff = diff[i0 : i1 + 1]
        base = _safe_percentile(local_env[: max(8, len(local_env) // 5)], 60.0, fallback=_safe_mean(local_env, fallback=0.0))
        peak = _safe_percentile(local_env, 99.0, fallback=base)
        env_thr = base + (0.10 * max(0.0, peak - base))
        diff_thr = _safe_percentile(local_diff, 80.0, fallback=0.0)
        hold = max(1, int(round(0.004 * drums.native_sr)))

        best_idx = None
        best_score = -1e9
        search_start = max(1, coarse_sample - int(round(0.012 * drums.native_sr)))
        for idx in range(search_start, i1 - hold):
            if idx < i0:
                continue
            env_here = float(env[idx])
            diff_here = float(diff[idx])
            sustain = _safe_mean(env[idx : idx + hold], fallback=0.0)
            if sustain < env_thr:
                continue
            if diff_here < diff_thr and env_here < env_thr:
                continue
            dist_pen = abs(idx - coarse_sample) / max(1.0, 0.20 * beat_sec * drums.native_sr)
            score = (0.60 * diff_here) + (0.30 * env_here) + (0.10 * sustain) - (0.18 * dist_pen)
            if best_idx is None or score > best_score:
                best_idx = idx
                best_score = score
                if idx >= coarse_sample and score >= (0.90 * best_score):
                    break

        if best_idx is None:
            best_idx = coarse_sample

        max_delta = int(round(0.30 * beat_sec * drums.native_sr))
        if abs(best_idx - coarse_sample) > max_delta:
            best_idx = coarse_sample

        safe_cut = _find_zero_crossing_before(y, best_idx, int(round((self.options.zero_crossing_back_ms / 1000.0) * drums.native_sr)))
        score_conf = float(confidence)
        if chosen is not None:
            score_conf = self._candidate_confidence(chosen)

        return StemAlignment(
            role="drums",
            sample_rate=int(drums.native_sr),
            coarse_sample=int(coarse_sample),
            anchor_sample=int(best_idx),
            anchor_seconds=float(best_idx) / float(drums.native_sr),
            beat_position=float(beat_position),
            confidence=float(score_conf),
            analysis_offset_samples=int(round(drums.analysis_offset_seconds * drums.native_sr)),
            search_shift_samples=int(best_idx - coarse_sample),
            safe_cut_sample=int(safe_cut),
            inherited_from_drums=False,
        )

    def _candidate_confidence(self, chosen: CandidateScore) -> float:
        strength = (
            0.26 * chosen.norm_onset
            + 0.22 * chosen.norm_lowend
            + 0.18 * chosen.norm_density
            + 0.12 * chosen.norm_phrase
            + 0.10 * chosen.norm_grid
            + 0.08 * chosen.norm_ableton
            + 0.06 * chosen.norm_sustain
            + 0.06 * chosen.norm_legacy
        )
        penalty = (0.45 * chosen.norm_preroll) + (0.30 * chosen.norm_weak) + (0.25 * chosen.norm_fake)
        conf = float(np.clip(_sigmoid((strength - penalty - 0.15) * 5.0), 0.0, 1.0))
        if not chosen.valid:
            conf *= 0.80
        return float(conf)

    def _align_secondary_stem(
        self,
        stem: Optional[StemAudio],
        expected_sec: float,
        beat_position: float,
        bpm: float,
        drums_conf: float,
    ) -> Optional[StemAlignment]:
        if stem is None:
            return None

        y = stem.native_y.astype(np.float32, copy=False)
        sr = stem.native_sr
        expected_idx = int(round(float(expected_sec) * float(sr)))
        expected_idx = max(1, min(expected_idx, max(1, len(y) - 2)))

        env = _moving_average_samples(np.abs(y), max(1, int(round(0.003 * sr))))
        diff = np.maximum(0.0, np.diff(env, prepend=env[:1]))
        beat_sec = 60.0 / max(1e-6, float(bpm))

        if stem.role == "vocals":
            back_ms = int(self.options.vocals_search_back_ms)
            fwd_ms = int(self.options.vocals_search_fwd_ms)
            min_improvement = float(self.options.vocals_min_improvement)
            earlier_bias = 0.35
        else:
            back_ms = int(self.options.secondary_search_back_ms)
            fwd_ms = int(self.options.secondary_search_fwd_ms)
            min_improvement = float(self.options.secondary_min_improvement)
            earlier_bias = 0.20

        search_back = int(round(min(back_ms / 1000.0, 0.22 * beat_sec) * sr))
        search_fwd = int(round(min(fwd_ms / 1000.0, 0.28 * beat_sec) * sr))
        i0 = max(1, expected_idx - max(1, search_back))
        i1 = min(len(y) - 2, expected_idx + max(2, search_fwd))
        stride = max(1, int(round(0.0015 * sr)))
        hold = max(1, int(round(0.020 * sr)))

        def score_at(idx: int) -> float:
            a0 = max(1, idx - hold)
            a1 = idx
            b0 = idx
            b1 = min(len(env), idx + hold)
            pre = _safe_mean(env[a0:a1], fallback=0.0) if a1 > a0 else 0.0
            post = _safe_mean(env[b0:b1], fallback=0.0) if b1 > b0 else 0.0
            attack = float(np.max(diff[max(i0, idx - max(1, stride)) : min(i1, idx + max(2, stride))]))
            sustain = _safe_mean(env[b0 : min(len(env), idx + (2 * hold))], fallback=post) if b1 > b0 else post
            dist = abs(idx - expected_idx) / max(1.0, float(max(search_back, search_fwd)))
            early_pen = (max(0, expected_idx - idx) / max(1.0, float(max(search_back, search_fwd)))) * earlier_bias
            return (0.48 * attack) + (0.32 * max(0.0, post - pre)) + (0.20 * sustain) - (0.16 * dist) - early_pen

        best_idx = expected_idx
        best_score = score_at(expected_idx)
        for idx in range(i0, i1 + 1, stride):
            sc = score_at(idx)
            if sc > best_score:
                best_score = sc
                best_idx = idx

        if (best_score - score_at(expected_idx)) < min_improvement:
            best_idx = expected_idx
            best_score = score_at(expected_idx)

        fine_idx = best_idx
        local_hold = max(1, int(round(0.006 * sr)))
        local_start = max(i0, best_idx - max(1, int(round(0.010 * sr))))
        local_end = min(i1, best_idx + max(2, int(round(0.014 * sr))))
        local_best = best_score
        for idx in range(local_start, local_end):
            sustain = _safe_mean(env[idx : min(len(env), idx + local_hold)], fallback=0.0)
            attack = float(diff[idx])
            dist = abs(idx - best_idx) / max(1.0, float(local_end - local_start + 1))
            sc = (0.62 * attack) + (0.28 * sustain) - (0.10 * dist)
            if sc > local_best:
                local_best = sc
                fine_idx = idx

        if abs(fine_idx - expected_idx) > int(round(0.30 * beat_sec * sr)):
            fine_idx = expected_idx

        conf = float(np.clip((0.65 * drums_conf) + (0.35 * _sigmoid(best_score * 3.0 - 0.4)), 0.0, 1.0))
        safe_cut = _find_zero_crossing_before(y, fine_idx, int(round((self.options.zero_crossing_back_ms / 1000.0) * sr)))

        return StemAlignment(
            role=stem.role,
            sample_rate=int(sr),
            coarse_sample=int(expected_idx),
            anchor_sample=int(fine_idx),
            anchor_seconds=float(fine_idx) / float(sr),
            beat_position=float(beat_position),
            confidence=conf,
            analysis_offset_samples=int(round(stem.analysis_offset_seconds * sr)),
            search_shift_samples=int(fine_idx - expected_idx),
            safe_cut_sample=int(safe_cut),
            inherited_from_drums=bool(fine_idx == expected_idx),
        )

    def _write_candidate_csv(self, debug_dir: Optional[str], candidates: Sequence[CandidateScore]) -> Optional[str]:
        if not debug_dir:
            return None
        path = os.path.join(debug_dir, "candidate_scores.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "chosen",
                    "valid",
                    "bpm",
                    "downbeat_offset",
                    "bar_index",
                    "beat_index",
                    "time_abs_sec",
                    "score",
                    "raw_onset",
                    "raw_lowend",
                    "raw_contrast",
                    "raw_density",
                    "raw_grid",
                    "raw_phrase",
                    "raw_repeat",
                    "raw_sustain",
                    "raw_ableton",
                    "raw_legacy",
                    "raw_preroll",
                    "raw_weak",
                    "raw_fake",
                    "norm_onset",
                    "norm_lowend",
                    "norm_contrast",
                    "norm_density",
                    "norm_grid",
                    "norm_phrase",
                    "norm_repeat",
                    "norm_sustain",
                    "norm_ableton",
                    "norm_legacy",
                    "norm_preroll",
                    "norm_weak",
                    "norm_fake",
                ],
            )
            writer.writeheader()
            for cand in sorted(candidates, key=lambda c: (c.time_abs_sec, -c.score)):
                writer.writerow(
                    {
                        "chosen": int(cand.chosen),
                        "valid": int(cand.valid),
                        "bpm": f"{cand.bpm:.3f}",
                        "downbeat_offset": int(cand.downbeat_offset),
                        "bar_index": int(cand.bar_index),
                        "beat_index": int(cand.beat_index),
                        "time_abs_sec": f"{cand.time_abs_sec:.6f}",
                        "score": f"{cand.score:.6f}",
                        "raw_onset": f"{cand.raw_onset:.6f}",
                        "raw_lowend": f"{cand.raw_lowend:.6f}",
                        "raw_contrast": f"{cand.raw_contrast:.6f}",
                        "raw_density": f"{cand.raw_density:.6f}",
                        "raw_grid": f"{cand.raw_grid:.6f}",
                        "raw_phrase": f"{cand.raw_phrase:.6f}",
                        "raw_repeat": f"{cand.raw_repeat:.6f}",
                        "raw_sustain": f"{cand.raw_sustain:.6f}",
                        "raw_ableton": f"{cand.raw_ableton:.6f}",
                        "raw_legacy": f"{cand.raw_legacy:.6f}",
                        "raw_preroll": f"{cand.raw_preroll:.6f}",
                        "raw_weak": f"{cand.raw_weak:.6f}",
                        "raw_fake": f"{cand.raw_fake:.6f}",
                        "norm_onset": f"{cand.norm_onset:.6f}",
                        "norm_lowend": f"{cand.norm_lowend:.6f}",
                        "norm_contrast": f"{cand.norm_contrast:.6f}",
                        "norm_density": f"{cand.norm_density:.6f}",
                        "norm_grid": f"{cand.norm_grid:.6f}",
                        "norm_phrase": f"{cand.norm_phrase:.6f}",
                        "norm_repeat": f"{cand.norm_repeat:.6f}",
                        "norm_sustain": f"{cand.norm_sustain:.6f}",
                        "norm_ableton": f"{cand.norm_ableton:.6f}",
                        "norm_legacy": f"{cand.norm_legacy:.6f}",
                        "norm_preroll": f"{cand.norm_preroll:.6f}",
                        "norm_weak": f"{cand.norm_weak:.6f}",
                        "norm_fake": f"{cand.norm_fake:.6f}",
                    }
                )
        return path

    def _write_debug_plots(
        self,
        debug_dir: Optional[str],
        drums: StemAudio,
        analysis: Dict[str, Any],
        candidates: Sequence[CandidateScore],
        drums_alignment: StemAlignment,
        generate_plots: bool,
        ableton_snap_debug: Optional[Dict[str, Any]] = None,
        manual_anchor_sec: Optional[float] = None,
    ) -> List[str]:
        if not debug_dir or not generate_plots:
            return []
        if plt is None:
            try:
                note_path = os.path.join(debug_dir, "debug_plots_unavailable.txt")
                with open(note_path, "w", encoding="utf-8") as fh:
                    fh.write("matplotlib is unavailable in the current Python environment, so debug plots were skipped.\n")
            except Exception:
                pass
            return []

        plot_paths: List[str] = []
        try:
            env = _moving_average_samples(np.abs(drums.native_y), max(1, int(round(0.004 * drums.native_sr))))
            t_native = np.arange(len(env), dtype=np.float32) / float(drums.native_sr)

            fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
            ax[0].plot(t_native, env, color="#204b57", linewidth=0.8, label="drums envelope")
            for cand in candidates:
                ax[0].axvline(cand.time_abs_sec, color="#c8d9df", linewidth=0.8, alpha=0.55)
            ax[0].axvline(drums_alignment.anchor_seconds, color="#d13f30", linewidth=1.6, label="chosen 1.1.1")
            if manual_anchor_sec is not None:
                ax[0].axvline(float(manual_anchor_sec), color="#2f8f4f", linewidth=1.2, linestyle="--", label="manual 1.1.1")
            ax[0].set_title("Drums Envelope And Candidate Downbeats")
            ax[0].set_ylabel("Envelope")
            ax[0].legend(loc="upper right")

            feats = analysis["feats"]
            frame_times = feats["frame_times"] + float(analysis["offset_sec"])
            low_t = _robust_norm(feats["low_transient"], 20.0, 90.0)
            onset = _robust_norm(feats["onset"], 20.0, 90.0)
            ax[1].plot(frame_times, onset, color="#3f7f5f", linewidth=0.9, label="onset")
            ax[1].plot(frame_times, low_t, color="#3769a7", linewidth=0.9, label="low transient")
            for cand in candidates:
                ax[1].axvline(cand.time_abs_sec, color="#d0d0d0", linewidth=0.7, alpha=0.45)
            ax[1].axvline(drums_alignment.anchor_seconds, color="#d13f30", linewidth=1.5)
            if manual_anchor_sec is not None:
                ax[1].axvline(float(manual_anchor_sec), color="#2f8f4f", linewidth=1.1, linestyle="--")
            ax[1].set_title("Onset / Low-End Features")
            ax[1].set_xlabel("Seconds")
            ax[1].set_ylabel("Normalized feature")
            ax[1].legend(loc="upper right")

            fig.tight_layout()
            path = os.path.join(debug_dir, "drums_downbeat_debug.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(path)

            snap = ableton_snap_debug or {}
            chosen_marker = _safe_float(snap.get("chosen_marker_seconds"))
            rough_anchor = _safe_float(snap.get("rough_anchor_seconds"))
            cluster_start = _safe_float(snap.get("cluster_start_seconds"))
            cluster_end = _safe_float(snap.get("cluster_end_seconds"))
            considered = list(snap.get("considered_markers") or [])
            if considered and chosen_marker is not None:
                beat_sec = 60.0 / max(1e-6, float(analysis["bpm"]))
                window_left = min(
                    [v for v in [rough_anchor, cluster_start, chosen_marker, manual_anchor_sec] if v is not None]
                ) - (0.40 * beat_sec)
                window_right = max(
                    [v for v in [rough_anchor, cluster_end, chosen_marker, manual_anchor_sec] if v is not None]
                ) + (0.65 * beat_sec)
                window_left = max(0.0, window_left)
                window_right = min(float(drums.duration_seconds), window_right)
                if window_right > window_left:
                    fig2, ax2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                    mask = (t_native >= window_left) & (t_native <= window_right)
                    ax2[0].plot(t_native[mask], env[mask], color="#1d4250", linewidth=0.9, label="drums envelope")
                    if cluster_start is not None and cluster_end is not None:
                        ax2[0].axvspan(float(cluster_start), float(cluster_end), color="#f2d9a6", alpha=0.22, label="selected local event")
                    for entry in considered:
                        marker_sec = _safe_float(entry.get("marker_seconds"))
                        if marker_sec is None or marker_sec < window_left or marker_sec > window_right:
                            continue
                        line_color = "#b9b9b9"
                        line_width = 0.9
                        alpha = 0.55
                        if bool(entry.get("chosen")):
                            line_color = "#d13f30"
                            line_width = 1.8
                            alpha = 0.95
                        elif bool(entry.get("start_plausible")):
                            line_color = "#4d84c4"
                            line_width = 1.2
                            alpha = 0.85
                        ax2[0].axvline(marker_sec, color=line_color, linewidth=line_width, alpha=alpha)
                        marker_rank = _safe_float(entry.get("marker_rank_score"))
                        if marker_rank is not None:
                            ax2[0].text(
                                marker_sec,
                                1.02 * float(np.max(env[mask])) if np.any(mask) else 0.0,
                                f"{marker_rank:.2f}",
                                rotation=90,
                                fontsize=7,
                                color=line_color,
                                ha="center",
                                va="bottom",
                                clip_on=False,
                            )
                    if rough_anchor is not None:
                        ax2[0].axvline(float(rough_anchor), color="#f29f05", linewidth=1.3, linestyle="--", label="rough drop region")
                    ax2[0].axvline(float(drums_alignment.anchor_seconds), color="#d13f30", linewidth=1.6, label="final 1.1.1")
                    if manual_anchor_sec is not None:
                        ax2[0].axvline(float(manual_anchor_sec), color="#2f8f4f", linewidth=1.2, linestyle="--", label="manual 1.1.1")
                    ax2[0].set_ylabel("Envelope")
                    ax2[0].set_title("Local Ableton Event And Marker Ranking")
                    ax2[0].legend(loc="upper right")

                    frame_mask = (frame_times >= window_left) & (frame_times <= window_right)
                    ax2[1].plot(frame_times[frame_mask], onset[frame_mask], color="#3f7f5f", linewidth=0.9, label="onset")
                    ax2[1].plot(frame_times[frame_mask], low_t[frame_mask], color="#3769a7", linewidth=0.9, label="low transient")
                    if cluster_start is not None and cluster_end is not None:
                        ax2[1].axvspan(float(cluster_start), float(cluster_end), color="#f2d9a6", alpha=0.22)
                    for entry in considered:
                        marker_sec = _safe_float(entry.get("marker_seconds"))
                        if marker_sec is None or marker_sec < window_left or marker_sec > window_right:
                            continue
                        line_color = "#b9b9b9"
                        line_width = 0.9
                        alpha = 0.55
                        if bool(entry.get("chosen")):
                            line_color = "#d13f30"
                            line_width = 1.8
                            alpha = 0.95
                        elif bool(entry.get("start_plausible")):
                            line_color = "#4d84c4"
                            line_width = 1.2
                            alpha = 0.85
                        ax2[1].axvline(marker_sec, color=line_color, linewidth=line_width, alpha=alpha)
                    if rough_anchor is not None:
                        ax2[1].axvline(float(rough_anchor), color="#f29f05", linewidth=1.3, linestyle="--")
                    ax2[1].axvline(float(drums_alignment.anchor_seconds), color="#d13f30", linewidth=1.5)
                    if manual_anchor_sec is not None:
                        ax2[1].axvline(float(manual_anchor_sec), color="#2f8f4f", linewidth=1.1, linestyle="--")
                    ax2[1].set_xlabel("Seconds")
                    ax2[1].set_ylabel("Normalized feature")
                    ax2[1].legend(loc="upper right")

                    fig2.tight_layout()
                    path2 = os.path.join(debug_dir, "drums_local_event_debug.png")
                    fig2.savefig(path2, dpi=160)
                    plt.close(fig2)
                    plot_paths.append(path2)
        except Exception as exc:
            LOG.warning("Could not write debug plots: %s", exc)
        return plot_paths


def detect_song_first_downbeat(
    drums_path: str,
    inst_path: Optional[str] = None,
    vocals_path: Optional[str] = None,
    bpm: Optional[float] = None,
    out_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
    generate_plots: bool = True,
    manual_anchor_sec: Optional[float] = None,
    legacy_prior_seconds: Optional[float] = None,
    legacy_prior_confidence: Optional[float] = None,
    legacy_prior_source: Optional[str] = None,
    options: Optional[DetectorOptions] = None,
) -> Dict[str, Any]:
    detector = FirstDownbeatDetector(options=options)
    result = detector.detect(
        drums_path=drums_path,
        inst_path=inst_path,
        vocals_path=vocals_path,
        bpm=bpm,
        debug_dir=debug_dir,
        generate_plots=generate_plots,
        manual_anchor_sec=manual_anchor_sec,
        legacy_prior_seconds=legacy_prior_seconds,
        legacy_prior_confidence=legacy_prior_confidence,
        legacy_prior_source=legacy_prior_source,
    )
    data = result.to_dict()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=True, default=_json_default)
    return data


def build_als_anchor_map(result: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    drums = result.get("drums") or {}
    inst = result.get("inst") or {}
    vocals = result.get("vocals") or {}
    if "downbeat_seconds" in drums:
        out["drums"] = float(drums["downbeat_seconds"])
    if "aligned_seconds" in inst:
        out["inst"] = float(inst["aligned_seconds"])
    if "aligned_seconds" in vocals:
        out["vocals"] = float(vocals["aligned_seconds"])
    return out


def _find_track_ch1_als(track_dir: str) -> Optional[str]:
    preferred = os.path.join(track_dir, "CH1 Project", "CH1.als")
    if os.path.exists(preferred):
        return preferred
    fallback = os.path.join(track_dir, "CH1.als")
    if os.path.exists(fallback):
        return fallback
    return None


def discover_track_folder(track_dir: str) -> Optional[DiscoveredTrack]:
    drums = None
    inst = None
    vocals = None
    bpm = None
    camelot = None
    ch1 = _find_track_ch1_als(track_dir)

    try:
        entries = sorted(os.listdir(track_dir))
    except Exception:
        return None

    for name in entries:
        low = name.lower()
        if low.startswith("._") or low == ".ds_store" or low.endswith(".asd"):
            continue
        full = os.path.join(track_dir, name)
        if os.path.isdir(full):
            continue
        if low == "ch1.als":
            if ch1 is None:
                ch1 = full
            continue
        m = STEM_FILE_RE.match(name)
        if not m:
            continue
        role = str(m.group(1)).lower()
        bpm_val = float(m.group(2))
        key_val = str(m.group(3)).upper()
        bpm = bpm if bpm is not None else bpm_val
        camelot = camelot if camelot is not None else key_val
        if role == "drums" and drums is None:
            drums = full
        elif role == "inst" and inst is None:
            inst = full
        elif role == "vocals" and vocals is None:
            vocals = full

    if not drums:
        return None

    return DiscoveredTrack(
        track_dir=os.path.abspath(track_dir),
        drums_path=os.path.abspath(drums),
        inst_path=os.path.abspath(inst) if inst else None,
        vocals_path=os.path.abspath(vocals) if vocals else None,
        bpm=bpm,
        camelot_key=camelot,
        ch1_als_path=os.path.abspath(ch1) if ch1 else None,
    )


def discover_library_tracks(root_dir: str) -> List[DiscoveredTrack]:
    out: List[DiscoveredTrack] = []
    for cur_root, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith(".")]
        if "__MACOSX" in cur_root or "/__MACOSX/" in cur_root:
            continue
        if not filenames:
            continue
        track = discover_track_folder(cur_root)
        if track is not None:
            out.append(track)
    out.sort(key=lambda t: (t.bpm or 0.0, t.camelot_key or "", os.path.basename(t.track_dir).lower()))
    return out


def _extract_zip_to_temp(zip_path: str) -> Tuple[str, tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="downbeat_lib_")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir.name)
    return temp_dir.name, temp_dir


def _manual_downbeat_from_ch1(ch1_path: Optional[str]) -> Optional[float]:
    if not ch1_path or not os.path.exists(ch1_path) or extract_labels_from_als is None:
        return None
    try:
        labels = extract_labels_from_als(ch1_path, resolve_paths=True)
    except Exception:
        return None
    drums_rows = []
    for row in labels:
        audio_path = str(getattr(row, "audio_path", "") or "")
        base = os.path.basename(audio_path).lower()
        if base.startswith("drums_"):
            drums_rows.append(row)
    if not drums_rows:
        return None
    drums_rows.sort(key=lambda r: float(getattr(r, "target_sec", 0.0)))
    try:
        return float(getattr(drums_rows[0], "target_sec"))
    except Exception:
        return None


def detect_track_folder(
    track_dir: str,
    out_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
    generate_plots: bool = True,
    legacy_prior_seconds: Optional[float] = None,
    legacy_prior_confidence: Optional[float] = None,
    legacy_prior_source: Optional[str] = None,
    options: Optional[DetectorOptions] = None,
) -> Dict[str, Any]:
    track = discover_track_folder(track_dir)
    if track is None:
        raise ValueError(f"Could not find drums/inst/vocals triplet in {track_dir}")
    manual = _manual_downbeat_from_ch1(track.ch1_als_path)
    result = detect_song_first_downbeat(
        drums_path=track.drums_path,
        inst_path=track.inst_path,
        vocals_path=track.vocals_path,
        bpm=track.bpm,
        out_path=out_path,
        debug_dir=debug_dir,
        generate_plots=generate_plots,
        manual_anchor_sec=manual,
        legacy_prior_seconds=legacy_prior_seconds,
        legacy_prior_confidence=legacy_prior_confidence,
        legacy_prior_source=legacy_prior_source,
        options=options,
    )
    result["track_dir"] = track.track_dir
    result["camelot_key"] = track.camelot_key
    result["ch1_als_path"] = track.ch1_als_path
    if manual is not None:
        drums = result.get("drums") or {}
        pred = drums.get("downbeat_seconds")
        bpm = float(result.get("bpm") or track.bpm or 0.0)
        beat_sec = 60.0 / bpm if bpm > 0.0 else None
        result["debug"]["manual_ch1_downbeat_seconds"] = float(manual)
        if pred is not None:
            err_sec = float(pred) - float(manual)
            result["debug"]["manual_ch1_abs_error_ms"] = abs(err_sec) * 1000.0
            result["debug"]["manual_ch1_beat_error"] = (err_sec / beat_sec) if beat_sec else None
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=True, default=_json_default)
    return result


def detect_library(
    library_path: str,
    out_path: Optional[str] = None,
    debug_root: Optional[str] = None,
    generate_plots: bool = False,
    options: Optional[DetectorOptions] = None,
) -> Dict[str, Any]:
    extracted_temp = None
    root_dir = library_path
    if zipfile.is_zipfile(library_path):
        root_dir, extracted_temp = _extract_zip_to_temp(library_path)

    try:
        tracks = discover_library_tracks(root_dir)
        items: List[Dict[str, Any]] = []
        for idx, track in enumerate(tracks):
            item_debug_dir = None
            if debug_root:
                safe_name = os.path.basename(track.track_dir).replace("/", "_")
                item_debug_dir = os.path.join(debug_root, f"{idx:03d}_{safe_name}")
                os.makedirs(item_debug_dir, exist_ok=True)
            item = detect_track_folder(
                track_dir=track.track_dir,
                out_path=None,
                debug_dir=item_debug_dir,
                generate_plots=generate_plots,
                options=options,
            )
            items.append(item)

        manual_errors = [float((item.get("debug") or {}).get("manual_ch1_abs_error_ms")) for item in items if (item.get("debug") or {}).get("manual_ch1_abs_error_ms") is not None]
        payload = {
            "library_path": os.path.abspath(library_path),
            "resolved_root": os.path.abspath(root_dir),
            "tracks": len(items),
            "with_manual_ch1": len(manual_errors),
            "mean_manual_ch1_abs_error_ms": _safe_mean(np.asarray(manual_errors, dtype=np.float32), fallback=0.0) if manual_errors else None,
            "items": items,
        }
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=True, default=_json_default)
        return payload
    finally:
        if extracted_temp is not None:
            extracted_temp.cleanup()


def _read_manifest_rows(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return list(data)
        raise ValueError("JSON evaluation manifest must be a list of row objects.")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def evaluate_downbeat_manifest(
    manifest_path: str,
    out_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
    generate_plots: bool = False,
    options: Optional[DetectorOptions] = None,
) -> Dict[str, Any]:
    rows = _read_manifest_rows(manifest_path)
    detector = FirstDownbeatDetector(options=options)
    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        drums = row.get("drums") or row.get("drums_path") or row.get("audio") or row.get("audio_path")
        if not drums:
            raise ValueError(f"Row {idx} missing drums/audio path")
        inst = row.get("inst") or row.get("inst_path")
        vocals = row.get("vocals") or row.get("vocals_path")
        bpm = row.get("bpm")
        truth = row.get("ground_truth_seconds") or row.get("downbeat_seconds") or row.get("label_seconds")
        if truth is None:
            raise ValueError(f"Row {idx} missing ground truth seconds")
        bpm_f = float(bpm) if bpm not in (None, "") else None
        truth_f = float(truth)

        item_debug_dir = None
        if debug_dir:
            item_debug_dir = os.path.join(debug_dir, f"item_{idx:04d}")
            os.makedirs(item_debug_dir, exist_ok=True)

        result = detector.detect(
            drums_path=str(drums),
            inst_path=str(inst) if inst else None,
            vocals_path=str(vocals) if vocals else None,
            bpm=bpm_f,
            debug_dir=item_debug_dir,
            generate_plots=generate_plots,
        )
        pred_sec = float(result.drums.anchor_seconds)
        pred_bpm = float(result.bpm)
        beat_sec = 60.0 / max(1e-6, pred_bpm)
        err_sec = float(pred_sec - truth_f)
        results.append(
            {
                "drums": str(drums),
                "bpm": pred_bpm,
                "predicted_seconds": pred_sec,
                "ground_truth_seconds": truth_f,
                "abs_error_ms": abs(err_sec) * 1000.0,
                "beat_error": err_sec / beat_sec,
                "confidence": float(result.drums.confidence),
            }
        )

    abs_ms = np.asarray([r["abs_error_ms"] for r in results], dtype=np.float32) if results else np.asarray([], dtype=np.float32)
    abs_beats = np.asarray([abs(float(r["beat_error"])) for r in results], dtype=np.float32) if results else np.asarray([], dtype=np.float32)
    summary = {
        "tracks": int(len(results)),
        "mean_abs_error_ms": _safe_mean(abs_ms, fallback=0.0) if abs_ms.size else None,
        "median_abs_error_ms": float(np.median(abs_ms)) if abs_ms.size else None,
        "p90_abs_error_ms": float(np.percentile(abs_ms, 90.0)) if abs_ms.size else None,
        "mean_abs_beat_error": _safe_mean(abs_beats, fallback=0.0) if abs_beats.size else None,
        "within_10ms": int(np.sum(abs_ms <= 10.0)) if abs_ms.size else 0,
        "within_25ms": int(np.sum(abs_ms <= 25.0)) if abs_ms.size else 0,
        "within_50ms": int(np.sum(abs_ms <= 50.0)) if abs_ms.size else 0,
    }
    payload = {"summary": summary, "items": results}

    if out_path:
        if out_path.lower().endswith(".csv"):
            with open(out_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["drums", "bpm", "predicted_seconds", "ground_truth_seconds", "abs_error_ms", "beat_error", "confidence"],
                )
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        else:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=True, default=_json_default)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Detect the first musical downbeat / 1.1.1 from stems.")
    ap.add_argument("--drums", help="Path to drums stem")
    ap.add_argument("--inst", help="Path to inst stem", default=None)
    ap.add_argument("--vocals", help="Path to vocals stem", default=None)
    ap.add_argument("--track-dir", help="Track folder containing drums_/inst_/vocals_ stems and optional CH1.als", default=None)
    ap.add_argument("--library", help="Root library folder or zip containing many track folders", default=None)
    ap.add_argument("--bpm", type=float, default=None, help="Known BPM")
    ap.add_argument("--out", default=None, help="Path to output JSON")
    ap.add_argument("--debug-dir", default=None, help="Directory for debug artifacts")
    ap.add_argument("--no-plots", action="store_true", help="Disable matplotlib plot generation")
    ap.add_argument("--eval-manifest", default=None, help="CSV/JSON/JSONL manifest for evaluation mode")
    ap.add_argument("--eval-out", default=None, help="Path to write evaluation JSON/CSV")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(message)s")

    if args.eval_manifest:
        payload = evaluate_downbeat_manifest(
            manifest_path=args.eval_manifest,
            out_path=args.eval_out,
            debug_dir=args.debug_dir,
            generate_plots=not bool(args.no_plots),
        )
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=True, default=_json_default))
        return 0

    if args.library:
        payload = detect_library(
            library_path=args.library,
            out_path=args.out,
            debug_root=args.debug_dir,
            generate_plots=not bool(args.no_plots),
        )
        summary = {
            "tracks": int(payload.get("tracks") or 0),
            "with_manual_ch1": int(payload.get("with_manual_ch1") or 0),
            "mean_manual_ch1_abs_error_ms": payload.get("mean_manual_ch1_abs_error_ms"),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True, default=_json_default))
        return 0

    if args.track_dir:
        payload = detect_track_folder(
            track_dir=args.track_dir,
            out_path=args.out,
            debug_dir=args.debug_dir,
            generate_plots=not bool(args.no_plots),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default))
        return 0

    if not args.drums:
        ap.error("--drums is required unless --eval-manifest, --track-dir, or --library is used")

    result = detect_song_first_downbeat(
        drums_path=args.drums,
        inst_path=args.inst,
        vocals_path=args.vocals,
        bpm=args.bpm,
        out_path=args.out,
        debug_dir=args.debug_dir,
        generate_plots=not bool(args.no_plots),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
