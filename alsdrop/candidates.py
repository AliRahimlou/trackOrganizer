#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import CANDIDATE_VERSION
from .utils import audio_id


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError("librosa is required: pip install librosa soundfile") from e
    return librosa


def _madmom_downbeats(audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor  # type: ignore
    except Exception:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    try:
        proc = RNNDownBeatProcessor()
        act = proc(audio_path)
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
        db = tracker(act)
    except Exception:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    if db is None or len(db) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    times = np.asarray([float(row[0]) for row in db], dtype=np.float32)
    conf = np.full_like(times, 0.90, dtype=np.float32)
    return times, conf


def _norm01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    span = max(1e-9, hi - lo)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _tempo_from_beats(beat_times: np.ndarray) -> float:
    if beat_times.size < 4:
        return 0.0
    d = np.diff(beat_times)
    d = d[(d > 0.1) & (d < 1.5)]
    if d.size == 0:
        return 0.0
    beat_sec = float(np.median(d))
    if beat_sec <= 1e-6:
        return 0.0
    bpm = 60.0 / beat_sec
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 200.0:
        bpm *= 0.5
    return float(bpm)


def _interp_at(x: np.ndarray, frame_times: np.ndarray, t: np.ndarray) -> np.ndarray:
    if x.size == 0 or frame_times.size == 0 or t.size == 0:
        return np.zeros(t.shape[0], dtype=np.float32)
    return np.interp(
        t.astype(np.float64, copy=False),
        frame_times.astype(np.float64, copy=False),
        x.astype(np.float64, copy=False),
        left=float(x[0]),
        right=float(x[-1]),
    ).astype(np.float32, copy=False)


def _bar_candidates_from_beats(beat_times: np.ndarray, feature_dict: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if beat_times.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    if beat_times.size < 8:
        return beat_times.astype(np.float32, copy=False), np.full(beat_times.shape[0], 0.35, dtype=np.float32)

    frame_times = np.asarray([], dtype=np.float32)
    onset = np.asarray([], dtype=np.float32)
    low_ratio = np.asarray([], dtype=np.float32)
    if isinstance(feature_dict, dict):
        frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
        onset = feature_dict.get("onset", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
        low_ratio = feature_dict.get("low_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)

    onset_n = _norm01(onset) if onset.size else onset
    low_n = _norm01(low_ratio) if low_ratio.size else low_ratio

    all_times: List[float] = []
    all_conf: List[float] = []
    phase_scores: List[float] = []
    phase_times_rows: List[np.ndarray] = []
    for phase in range(4):
        phase_times = beat_times[phase::4]
        if phase_times.size == 0:
            phase_scores.append(0.0)
            phase_times_rows.append(np.asarray([], dtype=np.float32))
            continue
        phase_times_rows.append(phase_times.astype(np.float32, copy=False))
        if frame_times.size and onset_n.size and low_n.size:
            o = _interp_at(onset_n, frame_times, phase_times)
            l = _interp_at(low_n, frame_times, phase_times)
            sc = float(np.median((0.62 * o) + (0.38 * l))) if o.size else 0.0
        else:
            sc = 0.20
        phase_scores.append(float(sc))

    p = np.asarray(phase_scores, dtype=np.float32)
    p = _norm01(p) if p.size else p
    # Keep all phases for recall, but confidence-weight them by phase quality.
    for phase in range(4):
        phase_times = phase_times_rows[phase] if phase < len(phase_times_rows) else np.asarray([], dtype=np.float32)
        if phase_times.size == 0:
            continue
        conf = 0.35 + (0.55 * float(p[phase] if phase < len(p) else 0.0))
        for t in phase_times.tolist():
            all_times.append(float(t))
            all_conf.append(float(conf))

    return np.asarray(all_times, dtype=np.float32), np.asarray(all_conf, dtype=np.float32)


def _librosa_bar_downbeats(
    audio_path: Optional[str],
    feature_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    librosa = _require_librosa()
    beat_times = np.asarray([], dtype=np.float32)

    if audio_path and os.path.isfile(audio_path):
        try:
            sr = int(feature_dict.get("sr", np.asarray([22050], dtype=np.int32))[0])
            hop = int(feature_dict.get("hop_length", np.asarray([256], dtype=np.int32))[0])
            y, _sr = librosa.load(audio_path, sr=sr, mono=True)
            if y is not None and len(y) > 8:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
                tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
                if beat_frames is not None and len(beat_frames):
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).astype(np.float32, copy=False)
        except Exception:
            beat_times = np.asarray([], dtype=np.float32)

    if beat_times.size == 0:
        beat_times = feature_dict.get("beat_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)

    bars, conf = _bar_candidates_from_beats(beat_times, feature_dict=feature_dict)
    return bars, conf, beat_times


def _peak_candidates(feature_dict: Dict[str, np.ndarray], max_peaks: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    if frame_times.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    rms = feature_dict.get("rms", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    low_abs = feature_dict.get("low_energy", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    low_ratio = feature_dict.get("low_ratio", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    flux = feature_dict.get("flux_1024", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)

    m = min(len(frame_times), len(rms), len(onset), len(low_abs), len(low_ratio), len(flux))
    if m <= 4:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    frame_times = frame_times[:m]
    rms = rms[:m]
    onset = onset[:m]
    low_abs = low_abs[:m]
    low_ratio = low_ratio[:m]
    flux = flux[:m]

    def _nz(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo = float(np.percentile(x, 10.0))
        hi = float(np.percentile(x, 90.0))
        span = max(1e-9, hi - lo)
        return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)

    # Blend emphasizes structural low-end entry + onset/flux transitions.
    score = (
        0.35 * _nz(low_ratio)
        + 0.25 * _nz(low_abs)
        + 0.18 * _nz(onset)
        + 0.12 * _nz(flux)
        + 0.10 * _nz(rms)
    ).astype(np.float32, copy=False)

    q = float(np.percentile(score, 92.0))
    idx: List[int] = []
    for i in range(2, m - 2):
        s = float(score[i])
        if s < q:
            continue
        if s >= float(score[i - 1]) and s >= float(score[i + 1]):
            idx.append(i)

    if not idx:
        idx = np.argsort(score)[::-1][: min(max_peaks, m)].tolist()
    if len(idx) > max_peaks:
        idx = idx[:max_peaks]

    idx = sorted(set(int(i) for i in idx if 0 <= i < m))
    return frame_times[np.asarray(idx, dtype=np.int32)], score[np.asarray(idx, dtype=np.int32)]


def _merge_candidates(
    groups: Sequence[Tuple[np.ndarray, np.ndarray, float]],
    dedupe_sec: float = 0.070,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float]] = []
    for times, conf, weight in groups:
        if times.size == 0:
            continue
        for i, t in enumerate(times.tolist()):
            c = float(conf[i]) if i < len(conf) else 0.3
            rows.append((float(t), float(c) * float(weight)))

    if not rows:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    rows.sort(key=lambda x: x[0])
    merged_t: List[float] = [rows[0][0]]
    merged_c: List[float] = [rows[0][1]]

    for t, c in rows[1:]:
        if abs(float(t) - float(merged_t[-1])) <= float(dedupe_sec):
            # Keep weighted midpoint and best confidence.
            merged_t[-1] = float((merged_t[-1] + float(t)) * 0.5)
            merged_c[-1] = float(max(merged_c[-1], float(c)))
        else:
            merged_t.append(float(t))
            merged_c.append(float(c))

    return np.asarray(merged_t, dtype=np.float32), np.asarray(merged_c, dtype=np.float32)


def _apply_region_guard(
    times: np.ndarray,
    conf: np.ndarray,
    duration_sec: float,
    start_guard_sec: float = 5.0,
    end_guard_sec: float = 10.0,
    allow_early_if_conf: float = 0.93,
) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, conf
    dur = max(0.0, float(duration_sec))
    if dur <= 0.0:
        return times, conf

    start_guard = min(10.0, max(float(start_guard_sec), 0.06 * dur))
    end_guard = max(0.0, dur - float(end_guard_sec))

    keep: List[int] = []
    for i, t in enumerate(times.tolist()):
        c = float(conf[i]) if i < len(conf) else 0.0
        if t < 0.5:
            continue
        if t < start_guard and c < float(allow_early_if_conf):
            continue
        if t > end_guard:
            continue
        keep.append(i)

    if not keep:
        return times, conf
    kk = np.asarray(sorted(set(keep)), dtype=np.int32)
    return times[kk], conf[kk]


def _trim_with_coverage(times: np.ndarray, conf: np.ndarray, max_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    if times.size <= int(max_candidates):
        return times, conf
    n = int(max_candidates)
    n_top = max(1, int(round(0.65 * n)))
    n_cov = max(0, n - n_top)

    top_idx = np.argsort(conf)[::-1][:n_top]
    if n_cov > 0:
        cov_idx = np.linspace(0, times.size - 1, num=n_cov, dtype=np.int32)
        keep = np.unique(np.concatenate([top_idx.astype(np.int32), cov_idx], axis=0))
    else:
        keep = np.unique(top_idx.astype(np.int32))
    if keep.size > n:
        keep = keep[:n]
    keep = np.sort(keep)
    return times[keep], conf[keep]


@dataclass
class CandidateSet:
    times: np.ndarray
    confidence: np.ndarray
    downbeats: np.ndarray
    tempo_bpm: float
    version: int = CANDIDATE_VERSION

    def to_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times.astype(np.float32, copy=False),
            confidence=self.confidence.astype(np.float32, copy=False),
            downbeats=self.downbeats.astype(np.float32, copy=False),
            tempo_bpm=np.asarray([float(self.tempo_bpm)], dtype=np.float32),
            version=np.asarray([int(self.version)], dtype=np.int32),
        )

    @staticmethod
    def from_npz(path: str) -> "CandidateSet":
        with np.load(path, allow_pickle=True) as z:
            times = z["times"].astype(np.float32, copy=False)
            conf = z["confidence"].astype(np.float32, copy=False)
            downbeats = z["downbeats"].astype(np.float32, copy=False) if "downbeats" in z else times.copy()
            tempo = float(z["tempo_bpm"][0]) if "tempo_bpm" in z and len(z["tempo_bpm"]) else 0.0
            ver = int(z["version"][0]) if "version" in z and len(z["version"]) else 0
        return CandidateSet(times=times, confidence=conf, downbeats=downbeats, tempo_bpm=tempo, version=ver)


def candidate_cache_path(cache_dir: str, audio_path: str) -> str:
    return os.path.abspath(os.path.join(cache_dir, f"{audio_id(audio_path)}.npz"))


def generate_candidates(
    feature_dict: Dict[str, np.ndarray],
    audio_path: Optional[str] = None,
    use_madmom: bool = True,
    max_candidates: int = 1200,
) -> CandidateSet:
    librosa = _require_librosa()

    frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    duration_sec = float(frame_times[-1]) if frame_times.size else 0.0

    # Generator B: librosa beats -> bar downbeats
    bar_times, bar_conf, beat_times = _librosa_bar_downbeats(audio_path=audio_path, feature_dict=feature_dict)

    if beat_times.size == 0 and frame_times.size > 1:
        # Fallback from onset envelope if beat tracker data missing in cache.
        onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
        # Rebuild pseudo beats at local onset peaks.
        idx = np.argsort(onset)[::-1][: min(256, len(onset))]
        beat_times = np.sort(frame_times[idx]).astype(np.float32, copy=False)
        bar_times, bar_conf = _bar_candidates_from_beats(beat_times, feature_dict=feature_dict)
    peak_times, peak_conf = _peak_candidates(feature_dict)

    # Generator A: madmom downbeats
    mad_times = np.asarray([], dtype=np.float32)
    mad_conf = np.asarray([], dtype=np.float32)
    if use_madmom and audio_path:
        mad_times, mad_conf = _madmom_downbeats(audio_path)

    cand_times, cand_conf = _merge_candidates(
        [
            (mad_times, mad_conf, 1.00),   # downbeat tracker A
            (bar_times, bar_conf, 0.95),   # beat->bar tracker B
            (peak_times, peak_conf, 0.45), # structural backup
        ]
    )

    if cand_times.size == 0 and frame_times.size:
        # Last-resort uniform grid every ~2 beats.
        tempo = _tempo_from_beats(beat_times)
        beat_sec = (60.0 / tempo) if tempo > 0 else 0.5
        step = max(0.5, beat_sec * 2.0)
        times = np.arange(float(frame_times[0]), float(frame_times[-1]) + 1e-9, step, dtype=np.float32)
        cand_times = times
        cand_conf = np.full(times.shape[0], 0.20, dtype=np.float32)

    cand_times, cand_conf = _apply_region_guard(
        cand_times.astype(np.float32, copy=False),
        cand_conf.astype(np.float32, copy=False),
        duration_sec=float(duration_sec),
        start_guard_sec=5.0,
        end_guard_sec=10.0,
    )
    cand_times, cand_conf = _trim_with_coverage(cand_times, cand_conf, int(max_candidates))

    # Downbeat list for snapping.
    downbeats, down_conf = _merge_candidates(
        [
            (mad_times, mad_conf, 1.00),
            (bar_times, bar_conf, 0.95),
        ]
    )
    if downbeats.size == 0:
        downbeats = cand_times.copy()
        down_conf = np.ones_like(downbeats, dtype=np.float32) * 0.20
    downbeats, down_conf = _apply_region_guard(
        downbeats.astype(np.float32, copy=False),
        down_conf.astype(np.float32, copy=False),
        duration_sec=float(duration_sec),
        start_guard_sec=5.0,
        end_guard_sec=10.0,
    )
    if downbeats.size == 0:
        downbeats = cand_times.copy()

    tempo = _tempo_from_beats(beat_times)
    if tempo <= 0 and frame_times.size > 1:
        # librosa beat fallback from available onset envelope could fail earlier.
        try:
            onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
            tempo_arr = librosa.beat.tempo(onset_envelope=onset, sr=int(feature_dict.get("sr", np.asarray([22050]))[0]), hop_length=int(feature_dict.get("hop_length", np.asarray([256]))[0]))
            tempo = float(tempo_arr[0]) if tempo_arr is not None and len(tempo_arr) else 0.0
        except Exception:
            tempo = 0.0

    return CandidateSet(
        times=cand_times.astype(np.float32, copy=False),
        confidence=cand_conf.astype(np.float32, copy=False),
        downbeats=downbeats.astype(np.float32, copy=False),
        tempo_bpm=float(tempo),
        version=CANDIDATE_VERSION,
    )


def snap_to_nearest_downbeat(pred_sec: float, downbeats: Sequence[float]) -> float:
    if not downbeats:
        return float(pred_sec)
    return float(min(downbeats, key=lambda t: abs(float(t) - float(pred_sec))))


def bar_number_from_sec(sec: float, bpm: float) -> int:
    if bpm <= 0:
        return 1
    beat_sec = 60.0 / float(bpm)
    if beat_sec <= 1e-9:
        return 1
    return int(np.floor(float(sec) / (4.0 * beat_sec))) + 1
