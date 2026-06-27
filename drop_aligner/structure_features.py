from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .beatgrid import BeatGrid, resolve_beatgrid
from .detector import DropDetectorConfig, FeatureBundle, extract_features
from .multistem import find_stem_group, infer_bpm_from_path


STRUCTURE_FEATURE_VERSION = 5


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _window(values: np.ndarray, times: np.ndarray, start: float, end: float) -> np.ndarray:
    if values.size == 0 or times.size == 0:
        return np.asarray([], dtype=np.float64)
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    if i1 <= i0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(values[i0:i1], dtype=np.float64)


def _mean(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    segment = _window(values, times, start, end)
    return 0.0 if segment.size == 0 else float(np.mean(segment))


def _max(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    segment = _window(values, times, start, end)
    return 0.0 if segment.size == 0 else float(np.max(segment))


def _density(values: np.ndarray, times: np.ndarray, start: float, end: float, *, threshold: float = 0.32) -> float:
    segment = _window(values, times, start, end)
    if segment.size < 2:
        return 0.0
    count = int(np.sum(segment >= float(threshold)))
    return _clip01(count / max(1.0, segment.size * 0.30))


def _occupancy(values: np.ndarray, times: np.ndarray, start: float, end: float, *, threshold: float = 0.32) -> float:
    segment = _window(values, times, start, end)
    if segment.size < 2:
        return 0.0
    return _clip01(float(np.mean(segment >= float(threshold))))


def _role_bar_features(features: FeatureBundle, start: float, end: float) -> Dict[str, float]:
    return {
        "low_energy": _mean(features.low_energy, features.frame_times, start, end),
        "rms": _mean(features.rms, features.frame_times, start, end),
        "rms_occupancy": _occupancy(features.rms, features.frame_times, start, end, threshold=0.32),
        "rms_mid_occupancy": _occupancy(features.rms, features.frame_times, start, end, threshold=0.24),
        "low_occupancy": _occupancy(features.low_energy, features.frame_times, start, end, threshold=0.16),
        "spectral_flux": _mean(features.spectral_flux, features.frame_times, start, end),
        "onset": _mean(features.onset, features.frame_times, start, end),
        "combined_attack": _mean(features.combined_attack, features.frame_times, start, end),
        "low_jump_peak": _max(features.low_jump_curve, features.frame_times, start, end),
        "attack_peak": _max(features.combined_attack, features.frame_times, start, end),
        "onset_density": _density(features.onset, features.frame_times, start, end),
    }


def _cache_root() -> Path:
    root = Path(tempfile.gettempdir()) / "track_organizer_structure_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _file_signature(path: str) -> Dict[str, Any]:
    p = Path(path)
    try:
        stat = p.stat()
        return {"path": str(p.resolve()), "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
    except OSError:
        return {"path": str(p), "mtime_ns": 0, "size": 0}


def _cache_key(audio_path: str, *, sample_rate: int) -> str:
    group = find_stem_group(audio_path)
    payload = {
        "version": STRUCTURE_FEATURE_VERSION,
        "sample_rate": int(sample_rate),
        "roles": {role: _file_signature(path) for role, path in sorted(group.roles.items())},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_path(audio_path: str, *, sample_rate: int) -> Path:
    return _cache_root() / f"{_cache_key(audio_path, sample_rate=sample_rate)}.json"


def _load_cache(audio_path: str, *, sample_rate: int) -> Optional[Dict[str, Any]]:
    path = _cache_path(audio_path, sample_rate=sample_rate)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if int(payload.get("version", 0) or 0) != STRUCTURE_FEATURE_VERSION:
            return None
        payload["cache_hit"] = True
        return payload
    except Exception:
        return None


def _write_cache(audio_path: str, *, sample_rate: int, payload: Mapping[str, Any]) -> None:
    path = _cache_path(audio_path, sample_rate=sample_rate)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = dict(payload)
    data["cache_hit"] = False
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"), ensure_ascii=True)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _load_stem_features(audio_path: str, *, sample_rate: int) -> tuple[Dict[str, FeatureBundle], Dict[str, Any]]:
    group = find_stem_group(audio_path)
    bpm = infer_bpm_from_path(audio_path)
    cfg = DropDetectorConfig(sample_rate=int(sample_rate), use_drumprint=False, hpss=False)
    features_by_role: Dict[str, FeatureBundle] = {}
    errors: Dict[str, str] = {}
    for role, path in group.roles.items():
        if role == "full":
            continue
        try:
            features = extract_features(path, cfg, bpm=bpm)
            features_by_role[role] = features
            if bpm is None:
                bpm = float(features.bpm)
        except Exception as exc:
            errors[role] = str(exc) or exc.__class__.__name__
    if not features_by_role:
        features_by_role["drums"] = extract_features(audio_path, cfg, bpm=bpm)
    return features_by_role, {"stem_group": group.to_dict(), "errors": errors}


def compute_bar_feature_map(audio_path: str, *, sample_rate: int = 16000, use_cache: bool = True) -> Dict[str, Any]:
    if use_cache:
        cached = _load_cache(audio_path, sample_rate=int(sample_rate))
        if cached is not None:
            return cached

    features_by_role, group_info = _load_stem_features(audio_path, sample_rate=int(sample_rate))
    bpm = infer_bpm_from_path(audio_path)
    grid: BeatGrid = resolve_beatgrid(features_by_role, bpm=bpm)
    duration = max(float(features.duration_sec) for features in features_by_role.values())
    bar_count = max(0, int(math.floor((duration - float(grid.bar_zero_sec)) / max(1e-6, float(grid.bar_sec)))))

    bars: list[Dict[str, Any]] = []
    previous_roles: Dict[str, Dict[str, float]] = {}
    for index in range(bar_count):
        start = float(grid.bar_zero_sec) + (index * float(grid.bar_sec))
        end = start + float(grid.bar_sec)
        roles: Dict[str, Dict[str, float]] = {}
        for role, features in features_by_role.items():
            roles[role] = _role_bar_features(features, start, end)

        drums = roles.get("drums", {})
        bass = roles.get("bass", {})
        inst = roles.get("instrumental", {})
        vocals = roles.get("vocals", {})
        bass_low = max(float(bass.get("low_energy", 0.0) or 0.0), float(drums.get("low_energy", 0.0) or 0.0), 0.55 * float(inst.get("low_energy", 0.0) or 0.0))
        drum_density = max(float(drums.get("onset_density", 0.0) or 0.0), float(drums.get("attack_peak", 0.0) or 0.0))
        inst_energy = max(float(inst.get("rms", 0.0) or 0.0), float(inst.get("spectral_flux", 0.0) or 0.0))
        vocal_presence = max(float(vocals.get("rms", 0.0) or 0.0), float(vocals.get("onset", 0.0) or 0.0))
        aggregate_energy = _clip01((0.34 * bass_low) + (0.28 * float(drums.get("rms", 0.0) or 0.0)) + (0.24 * inst_energy) + (0.14 * drum_density))
        groove_energy = _clip01((0.36 * bass_low) + (0.32 * drum_density) + (0.20 * float(drums.get("rms", 0.0) or 0.0)) + (0.12 * inst_energy))

        prev_inst = previous_roles.get("instrumental", {})
        prev_vocals = previous_roles.get("vocals", {})
        inst_reentry = _clip01(inst_energy - max(float(prev_inst.get("rms", 0.0) or 0.0), float(prev_inst.get("spectral_flux", 0.0) or 0.0)) + 0.20)
        vocal_dropout = _clip01(max(float(prev_vocals.get("rms", 0.0) or 0.0), float(prev_vocals.get("onset", 0.0) or 0.0)) - vocal_presence + 0.20)

        bars.append(
            {
                "bar": int(index + 1),
                "start_sec": float(start),
                "end_sec": float(end),
                "roles": roles,
                "aggregate_energy": float(aggregate_energy),
                "groove_energy": float(groove_energy),
                "bass_low_energy": float(bass_low),
                "drum_density": float(drum_density),
                "instrumental_energy": float(inst_energy),
                "instrumental_reentry": float(inst_reentry),
                "vocal_presence": float(vocal_presence),
                "vocal_dropout": float(vocal_dropout),
                "timbre_novelty": 0.0,
            }
        )
        previous_roles = roles

    vectors = np.asarray(
        [
            [
                bar["aggregate_energy"],
                bar["groove_energy"],
                bar["bass_low_energy"],
                bar["drum_density"],
                bar["instrumental_energy"],
                bar["vocal_presence"],
            ]
            for bar in bars
        ],
        dtype=np.float64,
    )
    if vectors.shape[0] >= 3:
        for idx, bar in enumerate(bars):
            pre = vectors[max(0, idx - 4) : idx]
            post = vectors[idx + 1 : min(vectors.shape[0], idx + 5)]
            if pre.size and post.size:
                novelty = float(np.linalg.norm(np.mean(post, axis=0) - np.mean(pre, axis=0)) / math.sqrt(vectors.shape[1]))
                bar["timbre_novelty"] = _clip01(novelty * 1.8)

    payload = {
        "ok": True,
        "version": STRUCTURE_FEATURE_VERSION,
        "audio_path": str(audio_path),
        "sample_rate": int(sample_rate),
        "duration_sec": float(duration),
        "beatgrid": grid.to_dict(),
        "stem_group": group_info.get("stem_group"),
        "stem_errors": group_info.get("errors") or {},
        "bar_count": int(len(bars)),
        "bars": bars,
        "cache_hit": False,
    }
    if use_cache:
        _write_cache(audio_path, sample_rate=int(sample_rate), payload=payload)
    return payload
