#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def nearest_frame_idx(frame_times: np.ndarray, sec: float) -> int:
    if frame_times.size == 0:
        return 0
    return int(np.argmin(np.abs(frame_times - float(sec))))


def _peak_indices(x: np.ndarray, q: float = 85.0) -> np.ndarray:
    if x.size < 3:
        return np.asarray([], dtype=np.int32)
    thr = float(np.percentile(x, q))
    out = []
    for i in range(1, len(x) - 1):
        if x[i] >= thr and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            out.append(i)
    return np.asarray(out, dtype=np.int32)


def madmom_downbeats(audio_path: str) -> np.ndarray:
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor  # type: ignore
    except Exception:
        return np.asarray([], dtype=np.float32)

    try:
        proc = RNNDownBeatProcessor()
        act = proc(audio_path)
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
        db = tracker(act)
    except Exception:
        return np.asarray([], dtype=np.float32)
    if db is None or len(db) == 0:
        return np.asarray([], dtype=np.float32)
    times = np.asarray([float(t[0]) for t in db], dtype=np.float32)
    return times


def generate_candidate_indices(
    feature_dict: Dict[str, np.ndarray],
    audio_path: Optional[str] = None,
    use_madmom: bool = True,
    max_candidates: int = 600,
) -> np.ndarray:
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)
    rms = feature_dict["rms"].astype(np.float32, copy=False)
    onset = feature_dict["onset"].astype(np.float32, copy=False)
    low = feature_dict["low_energy"].astype(np.float32, copy=False)
    beat_times = feature_dict.get("beat_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)

    idx: List[int] = []
    for t in beat_times:
        idx.append(nearest_frame_idx(frame_times, float(t)))

    if use_madmom and audio_path:
        for t in madmom_downbeats(audio_path):
            idx.append(nearest_frame_idx(frame_times, float(t)))

    for arr, q in ((onset, 88.0), (rms, 90.0), (low, 90.0)):
        peaks = _peak_indices(arr, q=q)
        idx.extend(int(i) for i in peaks.tolist())

    if not idx:
        idx = list(range(0, len(frame_times), max(1, len(frame_times) // 256)))

    uniq = np.unique(np.asarray(idx, dtype=np.int32))
    uniq = uniq[(uniq >= 0) & (uniq < len(frame_times))]
    if len(uniq) > max_candidates:
        # Keep broad coverage while preserving structural points.
        order = np.linspace(0, len(uniq) - 1, num=max_candidates, dtype=np.int32)
        uniq = uniq[order]
    return uniq.astype(np.int32, copy=False)


def sample_training_indices(
    feature_dict: Dict[str, np.ndarray],
    target_sec: float,
    rng: np.random.Generator,
    pos_radius_frames: int = 1,
    n_random_neg: int = 64,
    n_hard_neg: int = 48,
) -> List[Tuple[int, int]]:
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)
    rms = feature_dict["rms"].astype(np.float32, copy=False)
    onset = feature_dict["onset"].astype(np.float32, copy=False)
    low = feature_dict["low_energy"].astype(np.float32, copy=False)

    if frame_times.size == 0:
        return []
    pos_idx = nearest_frame_idx(frame_times, float(target_sec))

    out: List[Tuple[int, int]] = []
    for i in range(max(0, pos_idx - pos_radius_frames), min(len(frame_times), pos_idx + pos_radius_frames + 1)):
        out.append((int(i), 1))

    cand = generate_candidate_indices(feature_dict, use_madmom=False)
    neg_cand = [int(i) for i in cand.tolist() if abs(int(i) - pos_idx) > max(2, pos_radius_frames + 1)]
    if neg_cand:
        rng.shuffle(neg_cand)
        neg_cand = neg_cand[:n_random_neg]
        out.extend((i, 0) for i in neg_cand)

    hard_score = (0.45 * rms) + (0.35 * onset) + (0.20 * low / (np.max(low) + 1e-6))
    hard_idx = np.argsort(hard_score)[::-1]
    hard = []
    for i in hard_idx.tolist():
        if abs(int(i) - pos_idx) <= max(3, pos_radius_frames + 1):
            continue
        hard.append(int(i))
        if len(hard) >= n_hard_neg:
            break
    out.extend((i, 0) for i in hard)

    # Random negatives over full track.
    n_add = max(0, n_random_neg - len(neg_cand))
    if n_add > 0 and len(frame_times) > 1:
        picks = rng.integers(low=0, high=len(frame_times), size=n_add)
        for i in picks.tolist():
            if abs(int(i) - pos_idx) <= max(2, pos_radius_frames + 1):
                continue
            out.append((int(i), 0))

    # De-dup by index with positive precedence.
    by: Dict[int, int] = {}
    for i, y in out:
        if i not in by:
            by[i] = y
        else:
            by[i] = max(by[i], y)
    rows = sorted([(i, by[i]) for i in by], key=lambda t: t[0])
    return rows


def snap_to_nearest_downbeat(pred_sec: float, downbeat_times: Sequence[float]) -> float:
    if not downbeat_times:
        return float(pred_sec)
    best = min(downbeat_times, key=lambda t: abs(float(t) - float(pred_sec)))
    return float(best)

