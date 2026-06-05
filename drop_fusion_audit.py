#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    np = None  # type: ignore[assignment]


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"}
ROLE_NAMES = ("drums", "instrumental", "vocals", "bass", "full", "unknown")
REKORDBOX_XML_DEFAULT = "/Users/alirahimlou/Documents/rekordbox.xml"
ANALYSIS_VERSION = "drop_fusion_audit_v1"


@dataclass(frozen=True)
class StemInfo:
    role: str
    path: str


@dataclass
class FeatureBundle:
    role: str
    path: str
    sample_rate: int
    duration_sec: float
    bpm: Optional[float]
    beat_sec: float
    frame_times: Any
    rms: Any
    peak: Any
    low_energy: Any
    low_jump: Any
    spectral_flux: Any
    combined_attack: Any


def _json_default(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _clip01(value: Any) -> float:
    return float(max(0.0, min(1.0, _finite_float(value, 0.0))))


def _role_from_name(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name.lower()
    if name.startswith(("drums_", "drum_", "drums-", "drum-")):
        return "drums"
    if name.startswith(("inst_", "instrumental_", "other_", "inst-", "instrumental-", "other-")):
        return "instrumental"
    if name.startswith(("vocals_", "vocal_", "vocals-", "vocal-", "acapella_", "acapella-")) or "acapella" in name:
        return "vocals"
    if name.startswith(("bass_", "bass-")):
        return "bass"
    if name.startswith(("full_", "mix_", "master_", "original_")):
        return "full"
    return "unknown"


def _infer_bpm(path_or_name: str | Path) -> Optional[float]:
    path = Path(path_or_name)
    candidates = [path.name, *[parent.name for parent in path.parents]]
    for text in candidates:
        match = re.search(r"(?:^|[_\-/ ])(\d{2,3})(?:[_\-/ ]|$)", str(text))
        if not match:
            continue
        bpm = int(match.group(1))
        if 60 <= bpm <= 220:
            return float(bpm)
    return None


def _track_query_name(target: str | Path) -> str:
    path = Path(target)
    return path.stem if path.is_file() else path.name


def discover_stems(target: str | Path) -> Dict[str, StemInfo]:
    target_path = Path(target).expanduser().resolve()
    root = target_path if target_path.is_dir() else target_path.parent
    stems: Dict[str, StemInfo] = {}

    def add(path: Path) -> None:
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        role = _role_from_name(path)
        existing = stems.get(role)
        if existing is None or path == target_path:
            stems[role] = StemInfo(role=role, path=str(path.resolve()))

    if target_path.is_file():
        add(target_path)
    if root.exists():
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            add(path)
    if "drums" not in stems and target_path.is_file():
        stems["drums"] = StemInfo(role="drums", path=str(target_path))
    return stems


def _decode_audio_mono(path: str, sample_rate: int) -> Tuple[Any, int]:
    if np is None:
        raise RuntimeError("numpy is required for audio analysis")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {err}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio from {path}")
    y = np.nan_to_num(y, copy=False)
    peak = float(np.percentile(np.abs(y), 99.9))
    if peak > 1e-8:
        y = np.clip(y / peak, -1.0, 1.0).astype(np.float32)
    return y, int(sample_rate)


def _robust_norm(values: Any, lo: float = 5.0, hi: float = 95.0) -> Any:
    if np is None:
        return values
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if not math.isfinite(a) or not math.isfinite(b) or b <= a + 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _smooth(values: Any, frames: int) -> Any:
    if np is None:
        return values
    x = np.asarray(values, dtype=np.float64)
    frames = max(1, int(frames))
    if frames <= 1 or x.size == 0:
        return x
    kernel = np.ones(frames, dtype=np.float64) / float(frames)
    return np.convolve(x, kernel, mode="same")


def _frame_audio(y: Any, frame_length: int, hop_length: int) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for audio analysis")
    n = int(y.size)
    if n < frame_length:
        padded = np.zeros(int(frame_length), dtype=np.float32)
        padded[:n] = y
        y = padded
        n = int(y.size)
    frame_count = 1 + int((n - frame_length) // hop_length)
    shape = (frame_count, int(frame_length))
    strides = (int(hop_length) * y.strides[0], y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def extract_features(path: str, *, bpm: Optional[float], sample_rate: int) -> FeatureBundle:
    if np is None:
        raise RuntimeError("numpy is required for audio analysis")
    y, sr = _decode_audio_mono(path, sample_rate=sample_rate)
    duration_sec = float(len(y)) / float(sr)
    bpm_value = float(bpm or _infer_bpm(path) or 120.0)
    beat_sec = 60.0 / max(1.0, bpm_value)

    frame_length = 1024 if sr <= 24000 else 2048
    hop_length = max(128, frame_length // 4)
    frames = _frame_audio(y, frame_length=frame_length, hop_length=hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    framed = frames * window[None, :]

    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1))
    peak = np.percentile(np.abs(frames), 98.0, axis=1)
    mag = np.abs(np.fft.rfft(framed, axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / float(sr))
    low_mask = (freqs >= 35.0) & (freqs <= 180.0)
    if not np.any(low_mask):
        low_mask[: max(1, len(freqs) // 32)] = True
    low_energy = np.mean(mag[:, low_mask] ** 2, axis=1)

    mag_norm = mag / np.maximum(1e-9, np.max(mag, axis=1, keepdims=True))
    spectral_flux = np.maximum(0.0, np.diff(mag_norm, axis=0, prepend=mag_norm[:1])).mean(axis=1)
    low_jump = np.maximum(0.0, np.diff(_robust_norm(low_energy), prepend=_robust_norm(low_energy)[:1]))
    rms_rise = np.maximum(0.0, np.diff(_robust_norm(rms), prepend=_robust_norm(rms)[:1]))
    peak_rise = np.maximum(0.0, np.diff(_robust_norm(peak), prepend=_robust_norm(peak)[:1]))

    spectral_flux = _robust_norm(_smooth(spectral_flux, 2))
    low_energy = _robust_norm(_smooth(low_energy, 3))
    low_jump = _robust_norm(_smooth(low_jump, 2), hi=97.0)
    rms = _robust_norm(_smooth(rms, 2))
    peak = _robust_norm(_smooth(peak, 2))
    combined_attack = _robust_norm(
        (0.34 * _robust_norm(peak_rise, hi=97.0))
        + (0.30 * spectral_flux)
        + (0.22 * low_jump)
        + (0.14 * _robust_norm(rms_rise, hi=97.0)),
        hi=97.0,
    )
    frame_times = (np.arange(len(rms), dtype=np.float64) * float(hop_length) + (0.5 * frame_length)) / float(sr)
    return FeatureBundle(
        role=_role_from_name(path),
        path=str(path),
        sample_rate=int(sr),
        duration_sec=float(duration_sec),
        bpm=float(bpm_value),
        beat_sec=float(beat_sec),
        frame_times=frame_times,
        rms=rms,
        peak=peak,
        low_energy=low_energy,
        low_jump=low_jump,
        spectral_flux=spectral_flux,
        combined_attack=combined_attack,
    )


def _window_indices(times: Any, start: float, end: float) -> Tuple[int, int]:
    if np is None:
        return 0, 0
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    return max(0, i0), min(len(times), i1)


def _mean_window(values: Any, times: Any, start: float, end: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, start, end)
    if i1 <= i0:
        return float(fallback)
    return float(np.mean(values[i0:i1]))  # type: ignore[union-attr]


def _max_window(values: Any, times: Any, center: float, radius: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, float(center) - float(radius), float(center) + float(radius))
    if i1 <= i0:
        return float(fallback)
    return float(np.max(values[i0:i1]))  # type: ignore[union-attr]


def stem_evidence(features: FeatureBundle, time_sec: float) -> Dict[str, float]:
    t = float(time_sec)
    beat = float(features.beat_sec)
    bar = 4.0 * beat
    times = features.frame_times
    pre_long = max(beat, 2.0 * bar)
    pre_short = max(beat, 1.0 * bar)
    post_long = max(beat, 4.0 * bar)
    post_short = max(beat, 1.0 * bar)
    pre_rms = _mean_window(features.rms, times, t - pre_long, t)
    post_rms = _mean_window(features.rms, times, t, t + post_long)
    pre_short_rms = _mean_window(features.rms, times, t - pre_short, t, fallback=pre_rms)
    post_short_rms = _mean_window(features.rms, times, t, t + post_short, fallback=post_rms)
    pre_low = _mean_window(features.low_energy, times, t - pre_long, t)
    post_low = _mean_window(features.low_energy, times, t, t + post_long)
    attack = _max_window(features.combined_attack, times, t, max(0.08, 0.22 * beat))
    low_hit = _max_window(features.low_jump, times, t, max(0.08, 0.24 * beat))
    flux_peak = _max_window(features.spectral_flux, times, t, max(0.08, 0.30 * beat))
    energy_jump = _clip01((post_rms - pre_rms + 0.18) / 0.95)
    short_energy_jump = _clip01((post_short_rms - pre_short_rms + 0.15) / 0.85)
    low_jump = _clip01(max(post_low - pre_low, low_hit) + 0.08)
    dropout = _clip01((pre_short_rms - post_short_rms + 0.10) / 0.70)
    reentry = _clip01((post_short_rms - pre_short_rms + 0.10) / 0.70)
    post_activity = _clip01((0.42 * post_rms) + (0.34 * _mean_window(features.combined_attack, times, t, t + post_long)) + (0.24 * post_low))
    return {
        "attack": float(attack),
        "low_hit": float(low_hit),
        "flux_peak": float(flux_peak),
        "energy_jump": float(energy_jump),
        "short_energy_jump": float(short_energy_jump),
        "low_jump": float(low_jump),
        "dropout": float(dropout),
        "reentry": float(reentry),
        "post_activity": float(post_activity),
        "pre_rms": float(pre_rms),
        "post_rms": float(post_rms),
        "pre_low": float(pre_low),
        "post_low": float(post_low),
    }


def _local_peak_indices(values: Any, *, distance: int, threshold: float, limit: int) -> List[int]:
    if np is None:
        return []
    x = np.asarray(values, dtype=np.float64)
    if x.size < 3:
        return []
    distance = max(1, int(distance))
    threshold = float(threshold)
    raw: List[Tuple[float, int]] = []
    for idx in range(1, len(x) - 1):
        value = float(x[idx])
        if value < threshold or value < float(x[idx - 1]) or value < float(x[idx + 1]):
            continue
        lo = max(0, idx - max(2, distance // 2))
        hi = min(len(x), idx + max(2, distance // 2) + 1)
        local_floor = float(np.percentile(x[lo:hi], 30.0))
        if value - local_floor < 0.025:
            continue
        raw.append((value, idx))
    raw.sort(key=lambda item: (-item[0], item[1]))
    kept: List[int] = []
    for _value, idx in raw:
        if any(abs(idx - prev) < distance for prev in kept):
            continue
        kept.append(int(idx))
        if len(kept) >= int(limit):
            break
    return kept


def feature_seed_candidates(features: FeatureBundle, *, per_role_limit: int) -> List[Dict[str, Any]]:
    if np is None:
        return []
    times = features.frame_times
    if len(times) < 3:
        return []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.02
    beat = float(features.beat_sec)
    role = features.role
    min_time = max(0.75, 2.0 * beat)
    max_time = max(min_time, float(features.duration_sec) * 0.72)

    curves: List[Tuple[str, Any, float, float]] = []
    if role == "drums":
        curves.append(("drums_attack", (0.60 * features.combined_attack) + (0.40 * features.low_jump), 0.70, 0.75))
        curves.append(("drums_low_jump", features.low_jump, 0.70, 1.0))
    elif role == "instrumental":
        curves.append(("instrumental_flux", (0.55 * features.spectral_flux) + (0.45 * features.rms), 0.72, 0.75))
        curves.append(("instrumental_reentry", features.rms, 0.76, 4.0))
    elif role == "vocals":
        curves.append(("vocals_transition", (0.55 * features.spectral_flux) + (0.45 * features.rms), 0.76, 2.0))
    else:
        curves.append((f"{role}_attack", features.combined_attack, 0.73, 1.0))

    out: List[Dict[str, Any]] = []
    for source, curve, percentile, distance_beats in curves:
        values = np.asarray(curve, dtype=np.float64)
        threshold = max(float(np.percentile(values, percentile * 100.0)), float(np.max(values)) * 0.32)
        distance = max(1, int(round((distance_beats * beat) / max(1e-6, hop))))
        for idx in _local_peak_indices(values, distance=distance, threshold=threshold, limit=per_role_limit):
            sec = float(times[idx])
            if sec < min_time or sec > max_time:
                continue
            out.append(
                {
                    "seed_sec": sec,
                    "source": source,
                    "role": role,
                    "source_score": _clip01(values[idx] / max(1e-6, float(np.max(values)))),
                    "snap_window_beats": 0.55,
                }
            )
    return out


def bar_clock_seed_candidates(features_by_role: Mapping[str, FeatureBundle], *, bpm: Optional[float], limit: int = 96) -> List[Dict[str, Any]]:
    if not bpm or bpm <= 0.0 or not features_by_role:
        return []
    beat = 60.0 / float(bpm)
    bar = 4.0 * beat
    duration = min(float(features.duration_sec) for features in features_by_role.values())
    max_time = max(0.0, duration * 0.72)
    out: List[Dict[str, Any]] = []
    for bar_number in range(1, min(int(limit), int(max_time / max(1e-6, bar)) + 2)):
        sec = float((bar_number - 1) * bar)
        if sec < max(0.75, 2.0 * beat) or sec > max_time:
            continue
        phrase = 1.0 if (bar_number - 1) % 32 == 0 else (0.72 if (bar_number - 1) % 16 == 0 else (0.42 if (bar_number - 1) % 8 == 0 else 0.0))
        if phrase <= 0.0:
            continue
        role_scores = {
            role: max(stem_evidence(features, sec).get("attack", 0.0), stem_evidence(features, sec).get("energy_jump", 0.0))
            for role, features in features_by_role.items()
            if role in {"drums", "instrumental", "vocals", "bass"}
        }
        audio_score = max(role_scores.values(), default=0.0)
        combined = _clip01((0.55 * phrase) + (0.45 * audio_score))
        if combined < 0.28:
            continue
        out.append(
            {
                "seed_sec": sec,
                "source": "bpm_phrase_bar",
                "role": "clock",
                "source_score": combined,
                "clock_bar": int(bar_number),
                "phrase_strength": float(phrase),
                "clock_role_scores": role_scores,
                "snap_window_beats": 0.80,
            }
        )
    return out


def _parse_mik_cue_payload(payload: Any) -> List[float]:
    if payload is None:
        return []
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8", "ignore")
    else:
        raw = str(payload).encode("utf-8", "ignore")
    candidates: List[bytes] = [raw]
    try:
        candidates.append(base64.b64decode(raw, validate=False))
    except Exception:
        pass
    for candidate in candidates:
        try:
            text = candidate.decode("utf-8", "ignore")
        except Exception:
            continue
        if '"cues"' not in text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        cues = obj.get("cues") if isinstance(obj, Mapping) else None
        if not isinstance(cues, list):
            continue
        out: List[float] = []
        seen: set[float] = set()
        for cue in cues:
            if not isinstance(cue, Mapping):
                continue
            raw_ms = _optional_float(cue.get("time"))
            if raw_ms is None:
                continue
            sec = round(max(0.0, raw_ms / 1000.0), 6)
            if sec in seen:
                continue
            seen.add(sec)
            out.append(float(sec))
        return sorted(out)
    return []


def source_audio_mik_cues(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    try:
        import mutagen  # type: ignore
        from mutagen.id3 import ID3  # type: ignore
    except Exception:
        return [], "mutagen_unavailable"

    for audio_path in paths:
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            audio = mutagen.File(audio_path)
        except Exception:
            continue
        tags = getattr(audio, "tags", None)
        if tags is None:
            continue
        payloads: List[Any] = []
        if isinstance(tags, ID3):
            for frame in tags.getall("GEOB"):
                if str(getattr(frame, "desc", "")).strip().lower() == "cuepoints":
                    payloads.append(getattr(frame, "data", None))
        if hasattr(tags, "keys"):
            for key in tags.keys():
                if str(key).strip().lower() != "cuepoints":
                    continue
                value = tags.get(key)
                if isinstance(value, list):
                    payloads.extend(value)
                else:
                    payloads.append(value)
        for payload in payloads:
            seconds = _parse_mik_cue_payload(payload)
            if seconds:
                return [
                    {
                        "seed_sec": float(sec),
                        "source": "source_mik_cue",
                        "role": "cue",
                        "source_score": 0.78,
                        "cue_index": index,
                        "snap_window_beats": 2.0,
                    }
                    for index, sec in enumerate(seconds, start=1)
                ], f"source_mik_tags:{Path(audio_path).name}"
    return [], ""


def rekordbox_cues(
    *,
    xml_path: str,
    track_dir: str,
    source_audio_path: Optional[str],
    stem_paths: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    xml = Path(xml_path).expanduser()
    if not xml.exists():
        return [], {"available": False, "reason": "rekordbox_xml_missing", "xml_path": str(xml)}
    try:
        from rekordbox_mik_prior import lookup_first_drop_cue, lookup_track_cues
    except Exception as exc:
        return [], {"available": False, "reason": f"import_failed:{exc.__class__.__name__}", "xml_path": str(xml)}

    out: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"available": True, "xml_path": str(xml), "match_key": "", "first_drop": None}
    try:
        cues, match_key = lookup_track_cues(
            xml_path=str(xml),
            track_dir=track_dir,
            source_audio_path=source_audio_path,
            stem_paths=stem_paths,
        )
        meta["match_key"] = str(match_key or "")
        for index, sec in enumerate(cues, start=1):
            out.append(
                {
                    "seed_sec": float(sec),
                    "source": "rekordbox_mik_cue",
                    "role": "cue",
                    "source_score": 0.72,
                    "cue_index": index,
                    "snap_window_beats": 2.0,
                }
            )
    except Exception as exc:
        meta["cue_error"] = f"{exc.__class__.__name__}: {exc}"
    try:
        cue, conf, reason = lookup_first_drop_cue(
            xml_path=str(xml),
            track_dir=track_dir,
            source_audio_path=source_audio_path,
            stem_paths=stem_paths,
            confidence=0.98,
        )
        if cue is not None:
            meta["first_drop"] = {"sec": float(cue.start_sec), "name": cue.name, "num": cue.num, "confidence": float(conf), "reason": reason}
            out.append(
                {
                    "seed_sec": float(cue.start_sec),
                    "source": "rekordbox_first_drop_prior",
                    "role": "cue",
                    "source_score": float(conf),
                    "cue_name": cue.name,
                    "cue_num": cue.num,
                    "snap_window_beats": 2.25,
                }
            )
    except Exception as exc:
        meta["first_drop_error"] = f"{exc.__class__.__name__}: {exc}"
    return out, meta


def ableton_markers_for_stems(stems: Mapping[str, StemInfo]) -> Dict[str, Dict[str, Any]]:
    try:
        from ableton_analysis_adapter import extract_ableton_onset_markers
    except Exception as exc:
        return {"_meta": {"available": False, "reason": f"import_failed:{exc.__class__.__name__}"}}
    out: Dict[str, Dict[str, Any]] = {"_meta": {"available": True}}
    for role, stem in stems.items():
        try:
            markers = extract_ableton_onset_markers(stem.path)
        except Exception as exc:
            out[role] = {"available": False, "reason": f"{exc.__class__.__name__}: {exc}", "path": stem.path}
            continue
        if markers is None:
            out[role] = {"available": False, "reason": "no_sdk_or_asd_markers", "path": stem.path}
            continue
        out[role] = {
            "available": True,
            "path": stem.path,
            "source": markers.source,
            "count": int(len(markers.candidate_seconds)),
            "seconds": [float(sec) for sec in markers.candidate_seconds],
            "metadata": dict(markers.metadata),
        }
    return out


def _value(node: Optional[ET.Element]) -> Optional[str]:
    if node is None:
        return None
    if "Value" in node.attrib:
        return node.attrib.get("Value")
    return node.text


def _marker_float(marker: ET.Element, name: str) -> Optional[float]:
    if name in marker.attrib:
        return _optional_float(marker.attrib.get(name))
    return _optional_float(_value(marker.find(name)))


def _clip_file_refs(clip: ET.Element) -> List[str]:
    refs: List[str] = []
    for ref in clip.findall(".//FileRef"):
        for child_name in ("Path", "RelativePath"):
            raw = _value(ref.find(child_name))
            if raw and str(raw).strip():
                refs.append(str(raw).strip())
    return refs


def _clip_names(clip: ET.Element) -> List[str]:
    values: List[str] = []
    for path in ("Name", "EffectiveName", "UserName", ".//SampleRef/FileRef/Path", ".//SampleRef/FileRef/RelativePath"):
        raw = _value(clip.find(path))
        if raw:
            values.append(str(raw))
    values.extend(_clip_file_refs(clip))
    return values


def _clip_role(clip: ET.Element) -> str:
    for value in _clip_names(clip):
        role = _role_from_name(value)
        if role != "unknown":
            return role
    return "unknown"


def current_als_anchor_seeds(als_path: Optional[str], stems: Mapping[str, StemInfo]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not als_path:
        return [], {"available": False, "reason": "not_requested"}
    path = Path(als_path).expanduser()
    if not path.exists():
        return [], {"available": False, "reason": "als_missing", "als_path": str(path)}
    stem_basenames = {Path(stem.path).stem.lower(): role for role, stem in stems.items()}
    try:
        with gzip.open(path, "rb") as fh:
            root = ET.fromstring(fh.read())
    except Exception as exc:
        return [], {"available": False, "reason": f"parse_failed:{exc.__class__.__name__}", "als_path": str(path)}

    out: List[Dict[str, Any]] = []
    clips_seen = 0
    clips_matched = 0
    for clip in root.iter("AudioClip"):
        clips_seen += 1
        names = _clip_names(clip)
        role = _clip_role(clip)
        matched = False
        for value in names:
            low = Path(str(value)).stem.lower()
            if low in stem_basenames:
                role = stem_basenames[low]
                matched = True
                break
        if stem_basenames and not matched:
            continue
        if not matched and role == "unknown":
            continue
        clips_matched += 1
        markers = clip.find("WarpMarkers")
        if markers is None:
            continue
        for marker in markers.findall("WarpMarker"):
            sec = _marker_float(marker, "SecTime")
            beat = _marker_float(marker, "BeatTime")
            if sec is None or beat is None:
                continue
            if sec <= 0.0 or abs(float(beat)) > 1e-6:
                continue
            out.append(
                {
                    "seed_sec": float(sec),
                    "source": "current_als_111",
                    "role": role,
                    "source_score": 0.60,
                    "clip_names": names[:3],
                    "snap_window_beats": 0.60,
                }
            )
    return out, {"available": True, "als_path": str(path), "clips_seen": clips_seen, "clips_matched": clips_matched, "anchor_count": len(out)}


def _nearest_marker(markers: Sequence[float], target: float, *, before_slack: float, after_slack: float) -> Optional[Tuple[float, float]]:
    best: Optional[Tuple[float, float]] = None
    for marker in markers:
        delta = float(marker) - float(target)
        if delta < -float(before_slack) or delta > float(after_slack):
            continue
        score_delta = abs(delta)
        if best is None or score_delta < abs(best[1]) or (score_delta <= abs(best[1]) + 1e-9 and marker < best[0]):
            best = (float(marker), float(delta))
    return best


def refine_seed(
    seed: Mapping[str, Any],
    *,
    features_by_role: Mapping[str, FeatureBundle],
    ableton_markers: Mapping[str, Dict[str, Any]],
    bpm: Optional[float],
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy is required for audio analysis")
    source = str(seed.get("source") or "")
    seed_sec = float(seed.get("seed_sec", 0.0) or 0.0)
    beat_sec = 60.0 / float(bpm) if bpm and bpm > 0 else next((f.beat_sec for f in features_by_role.values()), 0.5)
    window_beats = float(seed.get("snap_window_beats", 0.60) or 0.60)
    if source in {"source_mik_cue", "rekordbox_mik_cue", "rekordbox_first_drop_prior"}:
        window_beats = max(window_beats, 2.0)
    radius = max(0.10, window_beats * beat_sec)

    primary = features_by_role.get("drums") or features_by_role.get("bass") or features_by_role.get("instrumental") or next(iter(features_by_role.values()))
    times = primary.frame_times
    curve = (0.58 * primary.combined_attack) + (0.30 * primary.low_jump) + (0.12 * primary.spectral_flux)
    i0, i1 = _window_indices(times, seed_sec - radius, seed_sec + radius)
    refined = float(seed_sec)
    peak_score = 0.0
    if i1 > i0:
        local = np.asarray(curve[i0:i1], dtype=np.float64)
        best_local = int(np.argmax(local))
        best_value = float(local[best_local])
        good = np.where(local >= max(0.10, 0.88 * best_value))[0]
        if good.size:
            # Prefer the first comparable attack inside the event, not the later body/tail.
            chosen_local = int(good[0])
        else:
            chosen_local = best_local
        refined = float(times[i0 + chosen_local])
        peak_score = _clip01(best_value)

    marker_roles = ["drums", "bass", "instrumental", "vocals"]
    marker_hit: Optional[Tuple[str, float, float]] = None
    for role in marker_roles:
        info = ableton_markers.get(role)
        seconds = info.get("seconds") if isinstance(info, Mapping) else None
        if not isinstance(seconds, Sequence):
            continue
        hit = _nearest_marker([float(sec) for sec in seconds], refined, before_slack=max(0.045, 0.12 * beat_sec), after_slack=max(0.025, 0.06 * beat_sec))
        if hit is None:
            continue
        if marker_hit is None or abs(hit[1]) < abs(marker_hit[2]):
            marker_hit = (role, float(hit[0]), float(hit[1]))
    snapped = refined
    if marker_hit is not None:
        snapped = float(marker_hit[1])
    return {
        "seed_sec": float(seed_sec),
        "refined_sec": float(refined),
        "snapped_sec": float(snapped),
        "source": source,
        "role": str(seed.get("role") or "unknown"),
        "source_score": float(seed.get("source_score", 0.0) or 0.0),
        "peak_score": float(peak_score),
        "ableton_snap": None
        if marker_hit is None
        else {"role": marker_hit[0], "sec": float(marker_hit[1]), "offset_ms": float(marker_hit[2] * 1000.0)},
        "raw": dict(seed),
    }


def _cluster_refined(refined: Sequence[Mapping[str, Any]], *, radius_sec: float) -> List[List[Dict[str, Any]]]:
    clusters: List[List[Dict[str, Any]]] = []
    for item in sorted((dict(row) for row in refined), key=lambda row: float(row.get("snapped_sec", row.get("refined_sec", 0.0)))):
        t = float(item.get("snapped_sec", item.get("refined_sec", 0.0)) or 0.0)
        best: Optional[List[Dict[str, Any]]] = None
        best_dist = float("inf")
        for cluster in clusters:
            center = sum(float(row.get("snapped_sec", row.get("refined_sec", 0.0)) or 0.0) for row in cluster) / max(1, len(cluster))
            dist = abs(t - center)
            if dist <= radius_sec and dist < best_dist:
                best = cluster
                best_dist = dist
        if best is None:
            clusters.append([item])
        else:
            best.append(item)
    return clusters


def _source_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for row in rows:
        source = str(row.get("source") or "")
        if source and source not in seen:
            seen.add(source)
            out.append(source)
    return out


def _roles(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for row in rows:
        role = str(row.get("role") or "")
        if role and role not in seen:
            seen.add(role)
            out.append(role)
    return out


def score_candidate(
    *,
    time_sec: float,
    rows: Sequence[Mapping[str, Any]],
    features_by_role: Mapping[str, FeatureBundle],
    bpm: Optional[float],
) -> Dict[str, Any]:
    evidence = {
        role: stem_evidence(features, time_sec)
        for role, features in features_by_role.items()
        if 0.0 <= time_sec <= float(features.duration_sec)
    }
    drums = evidence.get("drums", {})
    bass = evidence.get("bass", {})
    inst = evidence.get("instrumental", {})
    vocals = evidence.get("vocals", {})

    drums_attack = _clip01(drums.get("attack", 0.0))
    drums_low = max(_clip01(drums.get("low_jump", 0.0)), _clip01(bass.get("low_jump", 0.0)))
    drums_energy = _clip01(drums.get("energy_jump", 0.0))
    drums_post = _clip01(drums.get("post_activity", 0.0))
    drums_bass_impact = _clip01((0.36 * drums_attack) + (0.27 * drums_low) + (0.20 * drums_energy) + (0.17 * drums_post))

    inst_jump = max(_clip01(inst.get("energy_jump", 0.0)), _clip01(inst.get("short_energy_jump", 0.0)), 0.85 * _clip01(inst.get("flux_peak", 0.0)))
    vocal_transition = max(_clip01(vocals.get("dropout", 0.0)), 0.70 * _clip01(vocals.get("reentry", 0.0)), 0.50 * _clip01(vocals.get("flux_peak", 0.0)))
    source_names = _source_names(rows)
    roles = _roles(rows)
    marker_snap = any(isinstance(row.get("ableton_snap"), Mapping) for row in rows)
    cue_sources = {"source_mik_cue", "rekordbox_mik_cue", "rekordbox_first_drop_prior"}
    cue_used = any(str(row.get("source") or "") in cue_sources for row in rows)
    als_used = any(str(row.get("source") or "") == "current_als_111" for row in rows)
    phrase_strength = max(_finite_float((row.get("raw") or {}).get("phrase_strength"), 0.0) if isinstance(row.get("raw"), Mapping) else 0.0 for row in rows)
    source_strength = max((_finite_float(row.get("source_score"), 0.0) for row in rows), default=0.0)
    role_agreement = _clip01(len({role for role in roles if role not in {"unknown", "cue", "clock"}}) / 3.0)
    source_agreement = _clip01((len(source_names) - 1) / 4.0)
    ableton_bonus = 0.11 if marker_snap else 0.0
    cue_bonus = 0.08 if cue_used else 0.0
    als_bonus = 0.02 if als_used else 0.0

    fake_hit_penalty = 0.16 if drums_attack >= 0.62 and drums_post < 0.26 else 0.0
    vocal_only_penalty = 0.18 if vocal_transition >= 0.56 and drums_bass_impact < 0.25 else 0.0
    weak_body_penalty = 0.12 if drums_bass_impact < 0.22 and inst_jump < 0.22 else 0.0
    penalties = fake_hit_penalty + vocal_only_penalty + weak_body_penalty

    score = _clip01(
        (0.43 * drums_bass_impact)
        + (0.20 * inst_jump)
        + (0.08 * vocal_transition)
        + (0.10 * role_agreement)
        + (0.07 * source_agreement)
        + (0.05 * source_strength)
        + (0.04 * phrase_strength)
        + ableton_bonus
        + cue_bonus
        + als_bonus
        - penalties
    )
    return {
        "time_sec": float(time_sec),
        "score": float(score),
        "components": {
            "drums_bass_impact": float(drums_bass_impact),
            "drums_attack": float(drums_attack),
            "drums_low_jump": float(drums_low),
            "drums_energy_jump": float(drums_energy),
            "drums_post_activity": float(drums_post),
            "instrumental_jump": float(inst_jump),
            "vocal_transition": float(vocal_transition),
            "role_agreement": float(role_agreement),
            "source_agreement": float(source_agreement),
            "source_strength": float(source_strength),
            "phrase_strength": float(phrase_strength),
            "ableton_bonus": float(ableton_bonus),
            "cue_bonus": float(cue_bonus),
            "als_bonus": float(als_bonus),
            "fake_hit_penalty": float(fake_hit_penalty),
            "vocal_only_penalty": float(vocal_only_penalty),
            "weak_body_penalty": float(weak_body_penalty),
            "penalties": float(penalties),
        },
        "evidence": evidence,
        "sources": source_names,
        "roles": roles,
        "marker_snap": bool(marker_snap),
        "cue_used": bool(cue_used),
        "current_als_used": bool(als_used),
    }


def _choose_suggestion(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"available": False, "reason": "no_candidates"}
    ranked = sorted(candidates, key=lambda row: (-float(row.get("score", 0.0)), float(row.get("time_sec", 0.0))))
    best = ranked[0]
    best_score = float(best.get("score", 0.0))
    second = float(ranked[1].get("score", 0.0)) if len(ranked) > 1 else 0.0
    margin = max(0.0, best_score - second)
    near_best = [
        row
        for row in sorted(candidates, key=lambda row: float(row.get("time_sec", 0.0)))
        if float(row.get("score", 0.0)) >= max(0.0, best_score - 0.065)
        and float(((row.get("components") or {}) if isinstance(row.get("components"), Mapping) else {}).get("drums_bass_impact", 0.0)) >= 0.26
    ]
    chosen = near_best[0] if near_best else best
    confidence = _clip01((0.78 * float(chosen.get("score", 0.0))) + (0.55 * margin))
    safe_to_write = bool(
        confidence >= 0.64
        and float(chosen.get("score", 0.0)) >= 0.58
        and margin >= 0.025
        and float(((chosen.get("components") or {}) if isinstance(chosen.get("components"), Mapping) else {}).get("drums_bass_impact", 0.0)) >= 0.32
        and len(list(chosen.get("sources") or [])) >= 2
        and (bool(chosen.get("marker_snap")) or bool(chosen.get("cue_used")) or float(((chosen.get("components") or {}) if isinstance(chosen.get("components"), Mapping) else {}).get("source_agreement", 0.0)) >= 0.25)
    )
    return {
        "available": True,
        "time_sec": float(chosen.get("time_sec", 0.0)),
        "score": float(chosen.get("score", 0.0)),
        "confidence": float(confidence),
        "margin_from_runner_up": float(margin),
        "selected_policy": "earliest_near_best_sustained_impact" if chosen is not best else "best_score",
        "safe_to_write": safe_to_write,
        "write_policy": "review_only" if not safe_to_write else "high_confidence_candidate_available_but_not_written_by_audit",
        "sources": list(chosen.get("sources") or []),
        "roles": list(chosen.get("roles") or []),
        "components": dict(chosen.get("components") or {}),
    }


def build_audit(
    target: str,
    *,
    als_path: Optional[str],
    current_als_seeds: Optional[Sequence[Mapping[str, Any]]] = None,
    current_als_meta: Optional[Mapping[str, Any]] = None,
    source_audio_path: Optional[str],
    rekordbox_xml_path: str,
    sample_rate: int,
    per_role_limit: int,
    candidate_limit: int,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy is not installed. Use a Python runtime with numpy available.")

    target_path = Path(target).expanduser().resolve()
    stems = discover_stems(target_path)
    if not stems:
        raise RuntimeError(f"No audio stems found for {target_path}")
    bpm = _infer_bpm(target_path)
    if bpm is None:
        for stem in stems.values():
            bpm = _infer_bpm(stem.path)
            if bpm is not None:
                break
    stem_paths = [stem.path for stem in stems.values()]

    features_by_role: Dict[str, FeatureBundle] = {}
    feature_errors: Dict[str, str] = {}
    for role, stem in stems.items():
        if role == "full":
            continue
        try:
            features = extract_features(stem.path, bpm=bpm, sample_rate=int(sample_rate))
            features_by_role[role] = features
            if bpm is None and features.bpm:
                bpm = float(features.bpm)
        except Exception as exc:
            feature_errors[role] = f"{exc.__class__.__name__}: {exc}"
    if not features_by_role:
        raise RuntimeError(f"No stems could be analyzed: {feature_errors}")

    ableton = ableton_markers_for_stems(stems)
    seeds: List[Dict[str, Any]] = []
    for role, features in features_by_role.items():
        seeds.extend(feature_seed_candidates(features, per_role_limit=int(per_role_limit)))
    seeds.extend(bar_clock_seed_candidates(features_by_role, bpm=bpm, limit=96))

    mik_paths: List[str] = []
    if source_audio_path:
        mik_paths.append(str(Path(source_audio_path).expanduser()))
    mik_paths.extend(stem_paths)
    source_mik, source_mik_status = source_audio_mik_cues(mik_paths)
    seeds.extend(source_mik)

    rbx, rbx_meta = rekordbox_cues(
        xml_path=rekordbox_xml_path,
        track_dir=str(target_path if target_path.is_dir() else target_path.parent),
        source_audio_path=source_audio_path,
        stem_paths=stem_paths,
    )
    seeds.extend(rbx)

    if current_als_seeds is not None:
        als_seeds = [dict(seed) for seed in current_als_seeds]
        als_meta = dict(current_als_meta or {"available": True, "source": "precomputed"})
    else:
        als_seeds, als_meta = current_als_anchor_seeds(als_path, stems)
    seeds.extend(als_seeds)

    refined: List[Dict[str, Any]] = []
    seen_seed_keys: set[Tuple[str, int]] = set()
    for seed in seeds:
        sec = _optional_float(seed.get("seed_sec"))
        if sec is None or sec <= 0.0:
            continue
        key = (str(seed.get("source") or ""), int(round(float(sec) * 1000.0)))
        if key in seen_seed_keys:
            continue
        seen_seed_keys.add(key)
        try:
            refined.append(refine_seed(seed, features_by_role=features_by_role, ableton_markers=ableton, bpm=bpm))
        except Exception as exc:
            refined.append({"seed_sec": float(sec), "source": str(seed.get("source") or ""), "error": f"{exc.__class__.__name__}: {exc}", "raw": dict(seed)})

    beat_sec = 60.0 / float(bpm) if bpm and bpm > 0 else next(iter(features_by_role.values())).beat_sec
    cluster_radius = max(0.070, min(0.120, 0.16 * beat_sec))
    clusters = _cluster_refined([row for row in refined if _optional_float(row.get("snapped_sec")) is not None], radius_sec=cluster_radius)
    candidates: List[Dict[str, Any]] = []
    for cluster in clusters:
        scored_rows = sorted(cluster, key=lambda row: (-_finite_float(row.get("source_score"), 0.0), abs(_finite_float(row.get("snapped_sec"), 0.0) - _finite_float(row.get("seed_sec"), 0.0))))
        time_sec = float(scored_rows[0].get("snapped_sec", scored_rows[0].get("refined_sec", 0.0)) or 0.0)
        score = score_candidate(time_sec=time_sec, rows=cluster, features_by_role=features_by_role, bpm=bpm)
        score["cluster_size"] = int(len(cluster))
        score["cluster_radius_ms"] = float(cluster_radius * 1000.0)
        score["source_rows"] = cluster[:12]
        candidates.append(score)
    candidates.sort(key=lambda row: (-float(row.get("score", 0.0)), float(row.get("time_sec", 0.0))))
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = int(index)
    candidates = candidates[: max(1, int(candidate_limit))]
    suggestion = _choose_suggestion(candidates)

    return {
        "ok": True,
        "analysis_version": ANALYSIS_VERSION,
        "target": str(target_path),
        "query_name": _track_query_name(target_path),
        "bpm": None if bpm is None else float(bpm),
        "beat_sec": float(beat_sec),
        "sample_rate": int(sample_rate),
        "stems": {role: stem.path for role, stem in stems.items()},
        "feature_errors": feature_errors,
        "source_summary": {
            "seed_count": int(len(seeds)),
            "refined_count": int(len(refined)),
            "cluster_count": int(len(clusters)),
            "source_mik_status": source_mik_status,
            "source_mik_count": int(len(source_mik)),
            "rekordbox": rbx_meta,
            "rekordbox_seed_count": int(len(rbx)),
            "current_als": als_meta,
            "current_als_seed_count": int(len(als_seeds)),
            "ableton": {
                role: {
                    key: value
                    for key, value in info.items()
                    if key != "seconds"
                }
                for role, info in ableton.items()
            },
        },
        "suggestion": suggestion,
        "candidates": candidates,
        "refined_seed_errors": [row for row in refined if row.get("error")],
    }


def default_output_path(target: str | Path) -> Path:
    path = Path(target).expanduser().resolve()
    if path.is_dir():
        return path / "drop_fusion_audit.json"
    return path.with_name(path.stem + "_drop_fusion_audit.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fused drop candidates without modifying Ableton files.")
    parser.add_argument("target", help="Track folder or one stem/audio file to audit.")
    parser.add_argument("--als", dest="als_path", help="Optional ALS file to read current 1.1.1 anchors from.")
    parser.add_argument("--source-audio", help="Optional original full/source audio file for Mixed In Key cue tags.")
    parser.add_argument("--rekordbox-xml", default=os.environ.get("REKORDBOX_XML_PATH", REKORDBOX_XML_DEFAULT), help="Rekordbox XML path for Mixed In Key cue priors.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Analysis sample rate used through ffmpeg.")
    parser.add_argument("--per-role-limit", type=int, default=28, help="Maximum feature seeds per role/source curve.")
    parser.add_argument("--candidate-limit", type=int, default=40, help="Maximum ranked candidates to keep in the report.")
    parser.add_argument("--out", help="Output JSON path. Defaults beside the target.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        audit = build_audit(
            args.target,
            als_path=args.als_path,
            source_audio_path=args.source_audio,
            rekordbox_xml_path=args.rekordbox_xml,
            sample_rate=int(args.sample_rate),
            per_role_limit=int(args.per_role_limit),
            candidate_limit=int(args.candidate_limit),
        )
    except Exception as exc:
        print(f"[drop_fusion_audit] ERROR: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser().resolve() if args.out else default_output_path(args.target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")

    suggestion = audit.get("suggestion") if isinstance(audit.get("suggestion"), Mapping) else {}
    if suggestion.get("available"):
        print(
            "suggestion "
            f"{float(suggestion.get('time_sec', 0.0)):.3f}s "
            f"score={float(suggestion.get('score', 0.0)):.3f} "
            f"confidence={float(suggestion.get('confidence', 0.0)):.3f} "
            f"safe_to_write={bool(suggestion.get('safe_to_write'))}"
        )
        print("sources " + ", ".join(str(item) for item in suggestion.get("sources", [])))
    else:
        print("suggestion unavailable")
    print(f"report {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
