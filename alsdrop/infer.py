#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio_features import extract_features
from .candidates import CandidateSet, bar_number_from_sec, generate_candidates, snap_to_nearest_downbeat
from .constants import DEFAULT_HOP, DEFAULT_MELS, DEFAULT_SR
from .model import ContextConfig, build_candidate_contexts, build_model, sigmoid_np, temperature_scale_logits
from .utils import as_float, parent_dir


_TORCH_MODULE = None
_MODEL_CACHE_LOCK = threading.Lock()
_PREDICT_RUNTIME_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


def _require_torch():
    global _TORCH_MODULE
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for inference. Install with: pip install torch") from e
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available() and getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.benchmark = True
    _TORCH_MODULE = torch
    return torch


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError("librosa is required: pip install librosa soundfile") from e
    return librosa


def _device_auto(torch, device_arg: str) -> str:
    if device_arg != "auto":
        return str(device_arg)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_predict_runtime(model_path: str, device: str) -> Tuple[object, Dict[str, object]]:
    torch = _require_torch()
    resolved_model_path = os.path.abspath(model_path)
    dev = _device_auto(torch, device)
    cache_key = (resolved_model_path, dev)

    with _MODEL_CACHE_LOCK:
        cached = _PREDICT_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return torch, cached

    ckpt = torch.load(resolved_model_path, map_location="cpu")
    cfg = dict(ckpt.get("config") or {})

    sr = int(cfg.get("sr", DEFAULT_SR))
    hop = int(cfg.get("hop", DEFAULT_HOP))
    n_mels = int(cfg.get("n_mels", DEFAULT_MELS))
    context = ContextConfig(
        long_left_sec=float(cfg.get("long_left_sec", 12.0)),
        long_right_sec=float(cfg.get("long_right_sec", 6.0)),
        short_left_sec=float(cfg.get("short_left_sec", 2.0)),
        short_right_sec=float(cfg.get("short_right_sec", 2.0)),
        long_frames=int(cfg.get("long_frames", 384)),
        short_frames=int(cfg.get("short_frames", 128)),
    )

    in_channels = int(cfg.get("in_channels", n_mels + 9))
    meta_dim = int(cfg.get("meta_dim", 4))
    width = int(cfg.get("width", 128))
    dropout = float(cfg.get("dropout", 0.15))
    max_offset_sec = float(cfg.get("max_offset_sec", 0.150))

    model = build_model(
        in_channels=in_channels,
        meta_dim=meta_dim,
        width=width,
        dropout=dropout,
        max_offset_sec=max_offset_sec,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()

    runtime = {
        "cfg": cfg,
        "sr": sr,
        "hop": hop,
        "n_mels": n_mels,
        "context": context,
        "model": model,
        "device": dev,
        "drop_region_prior": dict(ckpt.get("drop_region_prior") or {}),
        "guardrails": dict(ckpt.get("guardrails") or {}),
        "temperature": float(ckpt.get("temperature", 1.0)),
        # A single cached model instance may be shared across worker threads.
        # Serialize the forward pass; CPU feature extraction still runs in parallel.
        "predict_lock": threading.Lock(),
    }

    with _MODEL_CACHE_LOCK:
        existing = _PREDICT_RUNTIME_CACHE.get(cache_key)
        if existing is not None:
            return torch, existing
        _PREDICT_RUNTIME_CACHE[cache_key] = runtime
    return torch, runtime


def _candidate_meta(
    feature_dict: Dict[str, np.ndarray],
    candidate_times: np.ndarray,
    candidate_conf: np.ndarray,
    tempo_bpm: float,
) -> np.ndarray:
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)
    dur = float(frame_times[-1]) if frame_times.size else 1.0
    onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    low_ratio = feature_dict.get("low_ratio", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    n = int(candidate_times.shape[0])
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out

    out[:, 0] = candidate_conf[:n].astype(np.float32, copy=False) if candidate_conf.size else 0.0
    out[:, 1] = candidate_times.astype(np.float32, copy=False) / max(1e-6, dur)
    if frame_times.size and low_ratio.size:
        out[:, 2] = np.interp(candidate_times, frame_times, low_ratio, left=low_ratio[0], right=low_ratio[-1]).astype(np.float32, copy=False)
    if frame_times.size and onset.size:
        out[:, 3] = np.interp(candidate_times, frame_times, onset, left=onset[0], right=onset[-1]).astype(np.float32, copy=False)

    # tempo hint included by mild scaling in conf channel
    if tempo_bpm > 0:
        out[:, 0] = np.clip(out[:, 0] + (float(tempo_bpm) / 220.0) * 0.05, 0.0, 1.0)
    return out


def _norm01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    span = max(1e-9, hi - lo)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _slice_mean(series: np.ndarray, times: np.ndarray, a: float, b: float) -> float:
    if series.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, float(a), side="left"))
    i1 = int(np.searchsorted(times, float(b), side="right"))
    i0 = max(0, min(i0, len(series)))
    i1 = max(0, min(i1, len(series)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(series[i0:i1]))


def _safe_fallback_candidate(
    feature_dict: Dict[str, np.ndarray],
    candidate_times: np.ndarray,
    candidate_conf: np.ndarray,
    downbeats: np.ndarray,
    duration_sec: float,
    region: Tuple[float, float],
) -> Tuple[float, float]:
    frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    onset = _norm01(feature_dict.get("onset", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    low = _norm01(feature_dict.get("low_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    novelty = _norm01(feature_dict.get("novelty", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    cand_conf_n = _norm01(candidate_conf.astype(np.float32, copy=False)) if candidate_conf.size else candidate_conf

    if candidate_times.size == 0:
        return float(0.0), 0.0

    lo_t, hi_t = float(region[0]), float(region[1])
    rows: List[Dict[str, float]] = []
    for i, t in enumerate(candidate_times.tolist()):
        tt = float(t)
        if tt < 0.5:
            continue
        if tt > max(0.0, float(duration_sec) - 1.0):
            continue
        if tt < lo_t or tt > hi_t:
            continue

        pre8 = _slice_mean(onset, frame_times, tt - 16.0, tt - 8.0)
        pre4 = _slice_mean(onset, frame_times, tt - 8.0, tt - 2.0)
        pre2_on = _slice_mean(onset, frame_times, tt - 2.0, tt)
        post2_on = _slice_mean(onset, frame_times, tt, tt + 2.0)

        pre2_low = _slice_mean(low, frame_times, tt - 2.0, tt)
        post2_low = _slice_mean(low, frame_times, tt, tt + 2.0)
        post8_low = _slice_mean(low, frame_times, tt + 2.0, tt + 8.0)
        nov_jump = _slice_mean(novelty, frame_times, tt - 1.0, tt + 1.5)

        buildup = max(0.0, pre4 - pre8)
        jump_on = max(0.0, post2_on - pre2_on)
        jump_low = max(0.0, post2_low - pre2_low)
        sustain = max(0.0, post8_low - pre2_low)
        conf = float(cand_conf_n[i]) if i < len(cand_conf_n) else 0.0

        rows.append(
            {
                "t": tt,
                "jump_low": float(jump_low),
                "jump_on": float(jump_on),
                "sustain": float(sustain),
                "buildup": float(buildup),
                "nov_jump": float(max(0.0, nov_jump)),
                "cand_conf": float(conf),
            }
        )

    if not rows:
        return float(candidate_times[0]), 0.0

    def _rank01(vals: np.ndarray) -> np.ndarray:
        n = int(vals.size)
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        order = np.argsort(vals, kind="mergesort")
        out = np.zeros(n, dtype=np.float32)
        out[order] = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        return out

    arr_jump_low = np.asarray([r["jump_low"] for r in rows], dtype=np.float32)
    arr_jump_on = np.asarray([r["jump_on"] for r in rows], dtype=np.float32)
    arr_sustain = np.asarray([r["sustain"] for r in rows], dtype=np.float32)
    arr_buildup = np.asarray([r["buildup"] for r in rows], dtype=np.float32)
    arr_nov = np.asarray([r["nov_jump"] for r in rows], dtype=np.float32)
    arr_conf = np.asarray([r["cand_conf"] for r in rows], dtype=np.float32)

    rk_low = _rank01(arr_jump_low)
    rk_on = _rank01(arr_jump_on)
    rk_sus = _rank01(arr_sustain)
    rk_bui = _rank01(arr_buildup)
    rk_nov = _rank01(arr_nov)
    rk_conf = _rank01(arr_conf)

    best_idx = 0
    best_score = -1e9
    best_hits = 0
    best_core_rank = 0.0
    core_thr = 0.72
    for i, r in enumerate(rows):
        core_hits = int(rk_low[i] >= core_thr) + int(rk_on[i] >= core_thr) + int(rk_sus[i] >= core_thr) + int(rk_bui[i] >= core_thr)
        core_rank_mean = float((rk_low[i] + rk_on[i] + rk_sus[i] + rk_bui[i]) / 4.0)
        novelty_boost = float(rk_nov[i])
        conf_boost = float(rk_conf[i])

        # Relative candidate scoring emphasizes structural drop signatures.
        score = (
            0.30 * float(rk_low[i])
            + 0.24 * float(rk_on[i])
            + 0.22 * float(rk_sus[i])
            + 0.14 * float(rk_bui[i])
            + 0.06 * novelty_boost
            + 0.04 * conf_boost
        )
        # Enforce "top percentile on >=2 core signals" as a strong preference.
        if core_hits < 2:
            score -= 0.14
        if score > best_score:
            best_score = float(score)
            best_idx = int(i)
            best_hits = int(core_hits)
            best_core_rank = float(core_rank_mean)

    best_t = float(rows[best_idx]["t"])
    best_s = float(best_score)

    if downbeats.size:
        best_t = snap_to_nearest_downbeat(float(best_t), downbeats.tolist())
    conf = float(1.0 / (1.0 + np.exp(-5.0 * (best_s - 0.58))))
    if best_hits < 2:
        conf *= 0.40
    elif best_core_rank < 0.65:
        conf *= 0.70
    return float(best_t), float(np.clip(conf, 0.0, 1.0))


def _region_window(duration_sec: float, prior: Dict[str, float]) -> Tuple[float, float]:
    dur = max(1e-6, float(duration_sec))
    p05 = float(prior.get("norm_p05", 0.06))
    p95 = float(prior.get("norm_p95", 0.90))
    lo_norm = max(0.0, min(1.0, p05 - 0.06))
    hi_norm = max(0.0, min(1.0, p95 + 0.06))
    lo = max(0.5, lo_norm * dur)
    hi = min(max(lo + 0.5, dur - 1.0), hi_norm * dur)
    # Always enforce coarse anti-intro/anti-tail limits.
    lo = max(lo, min(10.0, 0.04 * dur))
    hi = min(hi, dur - 10.0 if dur > 20.0 else dur - 1.0)
    return float(lo), float(max(lo + 0.5, hi))


def _dedupe_times(times: np.ndarray, conf: np.ndarray, tol_sec: float = 0.070) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times.astype(np.float32, copy=False), conf.astype(np.float32, copy=False)
    order = np.argsort(times)
    t = times[order].astype(np.float32, copy=False)
    c = conf[order].astype(np.float32, copy=False) if conf.size else np.zeros(t.shape[0], dtype=np.float32)
    out_t: List[float] = [float(t[0])]
    out_c: List[float] = [float(c[0])]
    for i in range(1, int(t.shape[0])):
        ti = float(t[i])
        ci = float(c[i])
        if abs(ti - out_t[-1]) <= float(tol_sec):
            out_t[-1] = 0.5 * (out_t[-1] + ti)
            out_c[-1] = max(out_c[-1], ci)
        else:
            out_t.append(ti)
            out_c.append(ci)
    return np.asarray(out_t, dtype=np.float32), np.asarray(out_c, dtype=np.float32)


def _union_candidate_sets(sets: Sequence[CandidateSet], max_candidates: int = 1400) -> CandidateSet:
    times_rows: List[np.ndarray] = []
    conf_rows: List[np.ndarray] = []
    down_rows: List[np.ndarray] = []
    tempo_vals: List[float] = []
    for s in sets:
        if not isinstance(s, CandidateSet):
            continue
        if s.times.size:
            times_rows.append(s.times.astype(np.float32, copy=False))
            conf_rows.append(s.confidence.astype(np.float32, copy=False) if s.confidence.size else np.zeros(s.times.shape[0], dtype=np.float32))
        if s.downbeats.size:
            down_rows.append(s.downbeats.astype(np.float32, copy=False))
        if float(s.tempo_bpm) > 0:
            tempo_vals.append(float(s.tempo_bpm))

    if not times_rows:
        return CandidateSet(
            times=np.asarray([], dtype=np.float32),
            confidence=np.asarray([], dtype=np.float32),
            downbeats=np.asarray([], dtype=np.float32),
            tempo_bpm=0.0,
        )

    times = np.concatenate(times_rows, axis=0).astype(np.float32, copy=False)
    conf = np.concatenate(conf_rows, axis=0).astype(np.float32, copy=False)
    times, conf = _dedupe_times(times, conf, tol_sec=0.070)

    if times.size > int(max_candidates):
        idx_top = np.argsort(conf)[::-1][: int(max_candidates)]
        idx_top = np.sort(idx_top.astype(np.int32))
        times = times[idx_top]
        conf = conf[idx_top]

    if down_rows:
        down = np.concatenate(down_rows, axis=0).astype(np.float32, copy=False)
        down, _ = _dedupe_times(down, np.ones(down.shape[0], dtype=np.float32), tol_sec=0.070)
    else:
        down = times.copy()
    tempo = float(np.median(np.asarray(tempo_vals, dtype=np.float32))) if tempo_vals else 0.0
    return CandidateSet(times=times, confidence=conf, downbeats=down, tempo_bpm=tempo)


def _deterministic_structural_candidates(
    feature_dict: Dict[str, np.ndarray],
    region: Tuple[float, float],
    max_candidates: int = 256,
) -> CandidateSet:
    frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    if frame_times.size < 4:
        return CandidateSet(
            times=np.asarray([], dtype=np.float32),
            confidence=np.asarray([], dtype=np.float32),
            downbeats=np.asarray([], dtype=np.float32),
            tempo_bpm=0.0,
        )

    duration_sec = float(frame_times[-1])
    onset = _norm01(feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False))
    low = _norm01(feature_dict.get("low_ratio", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False))
    novelty = _norm01(feature_dict.get("novelty", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False))
    m = min(len(frame_times), len(onset), len(low), len(novelty))
    if m < 4:
        return CandidateSet(
            times=np.asarray([], dtype=np.float32),
            confidence=np.asarray([], dtype=np.float32),
            downbeats=np.asarray([], dtype=np.float32),
            tempo_bpm=0.0,
        )
    frame_times = frame_times[:m]
    onset = onset[:m]
    low = low[:m]
    novelty = novelty[:m]

    tempo = float(as_float((feature_dict.get("tempo_est", np.asarray([0.0], dtype=np.float32))[0]), 0.0))
    if tempo <= 0:
        beat_times = feature_dict.get("beat_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
        if beat_times.size > 3:
            d = np.diff(beat_times)
            d = d[(d > 0.15) & (d < 1.2)]
            if d.size:
                tempo = 60.0 / float(np.median(d))
    if tempo <= 0:
        tempo = 128.0
    beat_sec = max(0.28, min(0.75, 60.0 / float(tempo)))
    bar_sec = 4.0 * beat_sec

    lo_t = max(0.5, float(region[0]))
    hi_t = min(max(lo_t + 1.0, float(region[1])), duration_sec - 0.5)
    if hi_t <= lo_t:
        return CandidateSet(
            times=np.asarray([], dtype=np.float32),
            confidence=np.asarray([], dtype=np.float32),
            downbeats=np.asarray([], dtype=np.float32),
            tempo_bpm=float(tempo),
        )

    times = np.arange(lo_t, hi_t + 1e-9, max(0.25, bar_sec), dtype=np.float32)
    if times.size == 0:
        return CandidateSet(
            times=np.asarray([], dtype=np.float32),
            confidence=np.asarray([], dtype=np.float32),
            downbeats=np.asarray([], dtype=np.float32),
            tempo_bpm=float(tempo),
        )

    score_rows: List[float] = []
    for t in times.tolist():
        pre_on = _slice_mean(onset, frame_times, t - 8.0, t - 1.0)
        post_on = _slice_mean(onset, frame_times, t, t + 2.0)
        pre_low = _slice_mean(low, frame_times, t - 2.0, t)
        post_low = _slice_mean(low, frame_times, t, t + 2.0)
        sus_low = _slice_mean(low, frame_times, t + 2.0, t + 8.0)
        nov = _slice_mean(novelty, frame_times, t - 0.8, t + 1.2)
        buildup = max(0.0, post_on - pre_on)
        impact = max(0.0, post_low - pre_low)
        sustain = max(0.0, sus_low - pre_low)
        sc = (0.40 * impact) + (0.30 * buildup) + (0.22 * sustain) + (0.08 * max(0.0, nov))
        score_rows.append(float(sc))

    score = _norm01(np.asarray(score_rows, dtype=np.float32))
    if score.size > int(max_candidates):
        idx = np.argsort(score)[::-1][: int(max_candidates)]
        idx = np.sort(idx.astype(np.int32))
        times = times[idx]
        score = score[idx]
    return CandidateSet(
        times=times.astype(np.float32, copy=False),
        confidence=score.astype(np.float32, copy=False),
        downbeats=times.astype(np.float32, copy=False),
        tempo_bpm=float(tempo),
    )


def _candidate_cascade(
    feature_dict: Dict[str, np.ndarray],
    audio_path: str,
    use_madmom: bool,
    region: Tuple[float, float],
) -> Tuple[CandidateSet, Dict[str, object]]:
    primary = generate_candidates(feature_dict=feature_dict, audio_path=audio_path, use_madmom=bool(use_madmom))
    secondary = generate_candidates(feature_dict=feature_dict, audio_path=audio_path, use_madmom=False)
    tertiary = _deterministic_structural_candidates(feature_dict=feature_dict, region=region, max_candidates=256)

    stage = "primary_madmom" if bool(use_madmom) else "secondary_librosa"
    if primary.times.size == 0 and secondary.times.size > 0:
        stage = "secondary_librosa"
    if primary.times.size == 0 and secondary.times.size == 0 and tertiary.times.size > 0:
        stage = "tertiary_structural"

    merged = _union_candidate_sets([primary, secondary, tertiary], max_candidates=1400)
    info = {
        "stage": stage,
        "primary_n": int(primary.times.size),
        "secondary_n": int(secondary.times.size),
        "tertiary_n": int(tertiary.times.size),
        "merged_n": int(merged.times.size),
    }
    return merged, info


def _micro_align_anchor(
    audio_path: str,
    anchor_sec: float,
    pre_ms: float = 120.0,
    post_ms: float = 180.0,
    threshold_k: float = 1.25,
    sr: int = 22050,
    hop_length: int = 128,
) -> Tuple[float, Dict[str, float]]:
    librosa = _require_librosa()
    pre = max(0.0, float(pre_ms) / 1000.0)
    post = max(0.0, float(post_ms) / 1000.0)
    left = max(0.0, float(anchor_sec) - pre)
    right = float(anchor_sec) + post
    duration = max(0.05, right - left)
    try:
        y, sr_loaded = librosa.load(audio_path, sr=int(sr), mono=True, offset=float(left), duration=float(duration))
    except Exception:
        return float(anchor_sec), {"used": 0.0, "reason": 1.0}
    if y is None or len(y) < 256:
        return float(anchor_sec), {"used": 0.0, "reason": 2.0}

    n_fft = 1024
    hop = max(64, min(128, int(hop_length)))
    onset = librosa.onset.onset_strength(y=y, sr=sr_loaded, hop_length=hop).astype(np.float32, copy=False)
    st = librosa.stft(y=y, n_fft=n_fft, hop_length=hop)
    pwr = (np.abs(st) ** 2).astype(np.float32, copy=False)
    freqs = librosa.fft_frequencies(sr=sr_loaded, n_fft=n_fft).astype(np.float32, copy=False)
    low_mask = (freqs >= 20.0) & (freqs <= 150.0)
    low_band = np.sum(pwr[low_mask, :], axis=0).astype(np.float32, copy=False) if np.any(low_mask) else np.zeros(pwr.shape[1], dtype=np.float32)
    low_flux = np.maximum(0.0, np.diff(low_band, prepend=low_band[:1])).astype(np.float32, copy=False)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop)[0].astype(np.float32, copy=False)
    novelty = np.maximum(0.0, np.diff(cent, prepend=cent[:1])).astype(np.float32, copy=False)

    n = min(len(onset), len(low_flux), len(novelty))
    if n < 4:
        return float(anchor_sec), {"used": 0.0, "reason": 3.0}
    onset = onset[:n]
    low_flux = low_flux[:n]
    novelty = novelty[:n]

    def _z(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32, copy=False)
        m = float(np.mean(x))
        s = float(np.std(x))
        if s <= 1e-9:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - m) / s).astype(np.float32, copy=False)

    s_on = _z(onset)
    s_low = _z(low_flux)
    s_nov = _z(novelty)
    score = (0.55 * s_on) + (0.35 * s_low) + (0.10 * s_nov)

    frame_times = librosa.frames_to_time(np.arange(n), sr=sr_loaded, hop_length=hop).astype(np.float32, copy=False)
    abs_times = (frame_times + float(left)).astype(np.float32, copy=False)
    mu = float(np.mean(score))
    sd = float(np.std(score))
    thr = mu + (float(threshold_k) * sd)

    after = np.where(abs_times >= float(anchor_sec))[0]
    pick = None
    for idx in after.tolist():
        if float(score[idx]) >= thr:
            pick = int(idx)
            break
    if pick is None:
        pick = int(np.argmax(score))
    aligned = float(abs_times[pick])
    aligned = min(float(right), max(float(left), aligned))
    return aligned, {
        "used": 1.0,
        "threshold": float(thr),
        "score_at_pick": float(score[pick]),
        "shift_sec": float(aligned - float(anchor_sec)),
    }


def _predict_internal(
    audio_path: str,
    model_path: str,
    device: str,
    use_madmom: bool,
    bpm_override: Optional[float],
    review_threshold: float,
    return_candidates: bool,
    top_k: int = 5,
    micro_align: bool = False,
    micro_window_pre_ms: float = 120.0,
    micro_window_post_ms: float = 180.0,
    micro_threshold_k: float = 1.25,
) -> Dict[str, object]:
    torch, runtime = _load_predict_runtime(model_path, device)
    cfg = dict(runtime.get("cfg") or {})
    sr = int(runtime.get("sr") or DEFAULT_SR)
    hop = int(runtime.get("hop") or DEFAULT_HOP)
    n_mels = int(runtime.get("n_mels") or DEFAULT_MELS)
    context = runtime.get("context") or ContextConfig()
    model = runtime["model"]
    dev = str(runtime.get("device") or _device_auto(torch, device))

    feat = extract_features(audio_path=audio_path, sr=sr, hop_length=hop, n_mels=n_mels)
    duration_sec = float(feat.get("frame_times", np.asarray([], dtype=np.float32))[-1]) if "frame_times" in feat and len(feat["frame_times"]) else 0.0
    prior = dict(runtime.get("drop_region_prior") or {})
    region_lo, region_hi = _region_window(duration_sec, prior=prior)
    cset, cascade_info = _candidate_cascade(
        feature_dict=feat,
        audio_path=audio_path,
        use_madmom=bool(use_madmom),
        region=(float(region_lo), float(region_hi)),
    )
    if cset.times.size == 0:
        raise RuntimeError("No candidate downbeats generated for this track")

    long_ctx, short_ctx = build_candidate_contexts(feat, cset.times.tolist(), cfg=context)
    meta = _candidate_meta(feat, cset.times, cset.confidence, cset.tempo_bpm)

    xb_long = torch.from_numpy(long_ctx).to(device=dev, dtype=torch.float32)
    xb_short = torch.from_numpy(short_ctx).to(device=dev, dtype=torch.float32)
    xb_meta = torch.from_numpy(meta).to(device=dev, dtype=torch.float32)

    predict_lock = runtime["predict_lock"]
    with predict_lock:
        with torch.inference_mode():
            logits, offset = model(xb_long, xb_short, xb_meta)
            logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            off_np = offset.detach().cpu().numpy().astype(np.float32, copy=False)

    temperature = float(runtime.get("temperature") or 1.0)
    probs = sigmoid_np(temperature_scale_logits(logits_np, temperature)).astype(np.float32, copy=False)
    order = np.argsort(probs)[::-1]
    j = int(order[0])
    top1 = float(probs[j])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = float(top1 - top2)
    cand_t = float(cset.times[j])
    refined = float(cand_t + off_np[j])

    # Musical guardrails: keep refinement local to downbeat.
    if cset.tempo_bpm > 0:
        beat_sec = 60.0 / float(cset.tempo_bpm)
    else:
        beat_sec = 0.5
    if abs(refined - cand_t) > (0.22 * beat_sec):
        refined = cand_t

    guardrails = dict(runtime.get("guardrails") or {})
    min_conf = float(guardrails.get("min_confidence", max(0.55, float(review_threshold))))
    min_margin = float(guardrails.get("min_margin", 0.05))
    region_valid = bool(region_lo <= float(cand_t) <= region_hi)

    accept_model = bool((top1 >= min_conf) and (margin >= min_margin) and region_valid)

    selected_by = "model"
    fallback_sec = None
    fallback_conf = None
    if not accept_model:
        fb_sec, fb_conf = _safe_fallback_candidate(
            feature_dict=feat,
            candidate_times=cset.times,
            candidate_conf=cset.confidence,
            downbeats=cset.downbeats,
            duration_sec=float(duration_sec),
            region=(float(region_lo), float(region_hi)),
        )
        if fb_sec > 0:
            fallback_sec = float(fb_sec)
            fallback_conf = float(fb_conf)
            cand_t = float(fb_sec)
            refined = float(fb_sec)
            selected_by = "safe_fallback"
        else:
            selected_by = "model_rejected"

    snapped = snap_to_nearest_downbeat(refined, cset.downbeats.tolist())
    predicted = float(snapped)
    micro_meta: Dict[str, float] = {"used": 0.0}
    if bool(micro_align):
        predicted, micro_meta = _micro_align_anchor(
            audio_path=audio_path,
            anchor_sec=float(snapped),
            pre_ms=float(micro_window_pre_ms),
            post_ms=float(micro_window_post_ms),
            threshold_k=float(micro_threshold_k),
            sr=int(sr),
            hop_length=128,
        )

    bpm_used = as_float(bpm_override)
    if bpm_used is None or bpm_used <= 0:
        bpm_used = as_float(cset.tempo_bpm, 0.0)
    if bpm_used is None or bpm_used <= 0:
        bpm_used = as_float(feat.get("tempo_est", np.asarray([128.0], dtype=np.float32))[0], 128.0)
    if bpm_used <= 0:
        bpm_used = 128.0

    top_order = order[: max(1, int(top_k))]
    top = [
        {
            "sec": float(cset.times[i]),
            "prob": float(probs[i]),
            "offset_sec": float(off_np[i]),
            "refined_sec": float(cset.times[i] + off_np[i]),
        }
        for i in top_order.tolist()
    ]

    effective_conf = float(top1 if selected_by == "model" else (fallback_conf if fallback_conf is not None else 0.0))
    fallback_review_threshold = max(float(review_threshold), 0.68)
    if selected_by == "model":
        needs_review = bool(effective_conf < float(review_threshold))
    else:
        needs_review = bool((fallback_conf is None) or (float(fallback_conf) < float(fallback_review_threshold)))
    res = {
        "audio_path": os.path.abspath(audio_path),
        "predicted_sec": float(predicted),
        "candidate_sec": float(cand_t),
        "refined_sec": float(refined),
        "downbeat_sec": float(snapped),
        "ableton_cue_sec": float(predicted),
        "confidence": float(effective_conf),
        "model_confidence": float(top1),
        "score_margin": float(margin),
        "guardrail_accept": bool(accept_model),
        "region_valid": bool(region_valid),
        "selected_by": str(selected_by),
        "fallback_sec": float(fallback_sec) if fallback_sec is not None else None,
        "fallback_confidence": float(fallback_conf) if fallback_conf is not None else None,
        "needs_manual_review": bool(needs_review),
        "bar_number": int(bar_number_from_sec(predicted, float(bpm_used))),
        "bpm_used": float(bpm_used),
        "tempo_est": float(cset.tempo_bpm),
        "n_candidates": int(cset.times.shape[0]),
        "candidate_cascade": cascade_info,
        "temperature": float(temperature),
        "guardrail_thresholds": {"min_confidence": float(min_conf), "min_margin": float(min_margin)},
        "fallback_review_threshold": float(fallback_review_threshold),
        "region_window_sec": [float(region_lo), float(region_hi)],
        "top_candidates": top,
        "model_path": os.path.abspath(model_path),
        "micro_align": {
            "enabled": bool(micro_align),
            "window_pre_ms": float(micro_window_pre_ms),
            "window_post_ms": float(micro_window_post_ms),
            "threshold_k": float(micro_threshold_k),
            "meta": micro_meta,
        },
    }
    if bool(return_candidates):
        res["candidate_times"] = [float(x) for x in cset.times.tolist()]
        res["candidate_confidences"] = [float(x) for x in cset.confidence.tolist()]
        res["downbeats"] = [float(x) for x in cset.downbeats.tolist()]
    return res


def run_predict(
    audio_path: str,
    model_path: str,
    out_json: Optional[str] = None,
    device: str = "auto",
    use_madmom: bool = True,
    bpm_override: Optional[float] = None,
    review_threshold: float = 0.55,
    return_candidates: bool = False,
    micro_align: bool = False,
    micro_window_pre_ms: float = 120.0,
    micro_window_post_ms: float = 180.0,
    micro_threshold_k: float = 1.25,
) -> Dict[str, object]:
    res = _predict_internal(
        audio_path=audio_path,
        model_path=model_path,
        device=device,
        use_madmom=use_madmom,
        bpm_override=bpm_override,
        review_threshold=review_threshold,
        return_candidates=bool(return_candidates),
        micro_align=bool(micro_align),
        micro_window_pre_ms=float(micro_window_pre_ms),
        micro_window_post_ms=float(micro_window_post_ms),
        micro_threshold_k=float(micro_threshold_k),
    )
    if out_json:
        parent_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        res["out_json"] = os.path.abspath(out_json)
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Predict ALS drop anchor (1.1.1) for an audio file")
    ap.add_argument("--audio", required=True, help="Input WAV/FLAC")
    ap.add_argument("--model", default="alsdrop/models/model.pt", help="Trained model checkpoint")
    ap.add_argument("--out", default="", help="Output JSON path")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--no-madmom", action="store_true", help="Disable madmom downbeat proposals")
    ap.add_argument("--bpm", type=float, default=0.0, help="Optional BPM override")
    ap.add_argument("--review-threshold", type=float, default=0.55)
    ap.add_argument("--micro-align", action="store_true", help="Apply final high-resolution transient micro-alignment")
    ap.add_argument("--micro-window-pre-ms", type=float, default=120.0, help="Micro-align search window before downbeat (ms)")
    ap.add_argument("--micro-window-post-ms", type=float, default=180.0, help="Micro-align search window after downbeat (ms)")
    ap.add_argument("--micro-threshold-k", type=float, default=1.25, help="Threshold k for first strong transient (mean + k*std)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_predict(
        audio_path=args.audio,
        model_path=args.model,
        out_json=args.out.strip() or None,
        device=args.device,
        use_madmom=not bool(args.no_madmom),
        bpm_override=args.bpm if args.bpm > 0 else None,
        review_threshold=float(args.review_threshold),
        micro_align=bool(args.micro_align),
        micro_window_pre_ms=float(args.micro_window_pre_ms),
        micro_window_post_ms=float(args.micro_window_post_ms),
        micro_threshold_k=float(args.micro_threshold_k),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
