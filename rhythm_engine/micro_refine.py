from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from numeric_backend import array_backend_status, moving_average

from .types import RhythmEngineConfig, RhythmEstimate


def _moving_average(x: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames))
    return moving_average(x, frames, dtype=np.float64)


def _refine_one_time(y: np.ndarray, sr: int, time_sec: float, *, radius_sec: float) -> float:
    center = int(round(float(time_sec) * float(sr)))
    radius = max(1, int(round(float(radius_sec) * float(sr))))
    i0 = max(0, center - radius)
    i1 = min(len(y), center + radius + 1)
    if i1 <= i0 + 4:
        return float(time_sec)

    env_frames = max(1, int(round(0.0025 * float(sr))))
    env = _moving_average(np.abs(y).astype(np.float64, copy=False), env_frames)
    onset = np.maximum(0.0, np.diff(env, prepend=env[:1]))
    local = onset[i0:i1]
    if local.size == 0 or float(np.max(local)) <= 1e-10:
        return float(time_sec)

    peak = i0 + int(np.argmax(local))
    pre = env[max(0, i0 - radius) : i0]
    base = float(np.percentile(pre, 65.0)) if pre.size else float(np.percentile(env[i0:i1], 20.0))
    high = float(np.percentile(env[max(i0, peak - env_frames) : min(len(env), peak + (4 * env_frames) + 1)], 95.0))
    threshold = base + (0.12 * max(0.0, high - base))

    attack = peak
    for idx in range(peak, i0, -1):
        if float(env[idx]) <= threshold:
            attack = min(peak, idx + 1)
            break

    z_back = max(1, int(round(0.006 * float(sr))))
    best = attack
    best_abs = abs(float(y[best])) if 0 <= best < len(y) else 0.0
    for idx in range(attack, max(1, attack - z_back), -1):
        v0 = float(y[idx - 1])
        v1 = float(y[idx])
        if abs(v1) < best_abs:
            best = idx
            best_abs = abs(v1)
        if (v0 <= 0.0 <= v1) or (v0 >= 0.0 >= v1):
            best = idx
            break
    return float(best / float(sr))


def _refine_times(y: np.ndarray, sr: int, times: Iterable[float], *, radius_sec: float) -> Tuple[float, ...]:
    return tuple(_refine_one_time(y, sr, float(t), radius_sec=radius_sec) for t in times)


def _refinement_audio_path(audio_path: str, cfg: RhythmEngineConfig) -> tuple[str, str]:
    if not cfg.micro_refine_stem_aware:
        return str(audio_path), "input"
    try:
        from drop_aligner.multistem import find_stem_group

        group = find_stem_group(audio_path)
        for role in ("drums", "bass", "full", "instrumental"):
            path = group.roles.get(role)
            if path:
                return str(path), f"stem:{role}"
    except Exception:
        pass
    return str(audio_path), "input"


def refine_estimate_to_attacks(audio_path: str, estimate: RhythmEstimate, config: RhythmEngineConfig | None = None) -> RhythmEstimate:
    cfg = config or RhythmEngineConfig()
    if not estimate.available:
        return estimate
    try:
        import librosa
    except Exception as exc:
        metadata = dict(estimate.metadata)
        metadata["micro_refine_status"] = "unavailable"
        metadata["micro_refine_error"] = str(exc) or exc.__class__.__name__
        return estimate.with_updates(metadata=metadata)

    try:
        refine_path, refine_source = _refinement_audio_path(audio_path, cfg)
        y, sr = librosa.load(refine_path, sr=int(cfg.micro_refine_sample_rate), mono=True)
        if y.size == 0:
            raise ValueError("empty_audio")
        y = np.asarray(y, dtype=np.float32)
        radius_sec = max(0.001, float(cfg.micro_refine_window_ms) * 0.001)
        refined_beats = _refine_times(y, int(sr), estimate.beats, radius_sec=radius_sec)
        refined_downbeats = _refine_times(y, int(sr), estimate.downbeats, radius_sec=radius_sec)
        offsets = [float(new - old) for new, old in zip(refined_beats, estimate.beats)]
        metadata = dict(estimate.metadata)
        backend = array_backend_status(int(y.size))
        metadata["micro_refine_status"] = "ok"
        metadata["micro_refine_audio_path"] = str(refine_path)
        metadata["micro_refine_source"] = str(refine_source)
        metadata["micro_refine_window_ms"] = float(cfg.micro_refine_window_ms)
        metadata["micro_refine_sample_rate"] = int(sr)
        metadata["micro_refine_array_backend"] = str(backend.get("active", "numpy"))
        metadata["micro_refine_array_backend_reason"] = str(backend.get("reason", ""))
        metadata["micro_refine_median_abs_offset_ms"] = float(np.median(np.abs(offsets)) * 1000.0) if offsets else 0.0
        return RhythmEstimate(
            provider=f"{estimate.provider}+micro_refine",
            beats=refined_beats,
            downbeats=refined_downbeats,
            bpm=estimate.bpm,
            confidence=estimate.confidence,
            duration_sec=estimate.duration_sec,
            sample_rate=int(sr),
            metadata=metadata,
        )
    except Exception as exc:
        metadata = dict(estimate.metadata)
        metadata["micro_refine_status"] = "failed"
        metadata["micro_refine_error"] = str(exc) or exc.__class__.__name__
        return estimate.with_updates(metadata=metadata)
