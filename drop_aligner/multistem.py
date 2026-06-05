from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from .beatgrid import BeatGrid, resolve_beatgrid
from .candidate_chooser import candidate_effective_time
from .detector import DropDetectorConfig, FeatureBundle, extract_features
from .musical_clock import bpm_clock_for_time, phrase_strength_for_bar
from .microalign import choose_microaligned_candidate, microalign_candidate_dicts
from .pipeline import run_drop_candidate_pipeline


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"}
BPM_RE = re.compile(r"^(?:drums|drum|inst|instrumental|vocals|vocal|bass|other)_(\d{2,3})_", re.IGNORECASE)
DEFAULT_MULTISTEM_CHOOSER_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_multistem_candidate_chooser.pkl"
DEFAULT_MULTISTEM_GROUPWISE_PATH = Path(__file__).resolve().parents[1] / "models" / "drop_multistem_groupwise_ranker.pkl"
DEFAULT_MULTISTEM_CHOOSER_REPORT = Path(__file__).resolve().parents[1] / "models" / "multistem_candidate_chooser_report.json"
DEFAULT_MULTISTEM_GROUPWISE_REPORT = Path(__file__).resolve().parents[1] / "models" / "multistem_groupwise_ranker_report.json"


@dataclass(frozen=True)
class StemGroup:
    root: str
    primary: str
    roles: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"root": self.root, "primary": self.primary, "roles": dict(self.roles)}


def classify_stem(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.startswith(("drums_", "drum_", "drums-", "drum-")):
        return "drums"
    if name.startswith(("bass_", "bass-")):
        return "bass"
    if name.startswith(("vocals_", "vocal_", "vocals-", "vocal-", "acapella_", "acapella-")) or "acapella" in name:
        return "vocals"
    if name.startswith(("inst_", "instrumental_", "other_", "inst-", "instrumental-", "other-")):
        return "instrumental"
    if name.startswith(("full_", "mix_", "master_", "original_")):
        return "full"
    return "unknown"


def infer_bpm_from_path(path: str | Path) -> Optional[float]:
    path = Path(path)
    match = BPM_RE.match(path.name)
    if match:
        return float(match.group(1))
    for parent in [path.parent, *path.parents]:
        try:
            bpm = int(parent.name)
        except ValueError:
            continue
        if 60 <= bpm <= 220:
            return float(bpm)
    return None


def find_stem_group(audio_path: str | Path) -> StemGroup:
    primary = Path(audio_path).expanduser().resolve()
    parent = primary.parent
    roles: Dict[str, str] = {}
    if primary.exists() and primary.suffix.lower() in AUDIO_EXTENSIONS:
        role = classify_stem(primary)
        if role != "unknown":
            roles[role] = str(primary)

    siblings = sorted(
        (
            path
            for path in parent.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    for path in siblings:
        role = classify_stem(path)
        if role == "unknown":
            continue
        existing = roles.get(role)
        if existing is None or path == primary or path.suffix.lower() == primary.suffix.lower():
            roles[role] = str(path.resolve())
    if "drums" not in roles and primary.exists():
        roles["drums"] = str(primary)
    return StemGroup(root=str(parent), primary=str(primary), roles=roles)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_finite_float(value), 0.0, 1.0))


def _window_indices(times: np.ndarray, start: float, end: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(times, max(0.0, float(start)), side="left"))
    i1 = int(np.searchsorted(times, max(0.0, float(end)), side="right"))
    return max(0, i0), min(len(times), i1)


def _mean_window(values: np.ndarray, times: np.ndarray, start: float, end: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, start, end)
    if i1 <= i0:
        return float(fallback)
    return float(np.mean(values[i0:i1]))


def _max_window(values: np.ndarray, times: np.ndarray, center: float, radius: float, fallback: float = 0.0) -> float:
    i0, i1 = _window_indices(times, center - radius, center + radius)
    if i1 <= i0:
        return float(fallback)
    return float(np.max(values[i0:i1]))


def _load_features(path: str, role: str, bpm: Optional[float], sample_rate: int) -> FeatureBundle:
    cfg = DropDetectorConfig(
        sample_rate=int(sample_rate),
        hpss=role == "drums",
        use_ranker_model=False,
        use_region_model=False,
        use_drumprint=False,
        use_microalign=False,
    )
    return extract_features(path, cfg, bpm=bpm)


def _read_report_metric(path: Path, *keys: str) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
        for key in keys:
            if not isinstance(data, Mapping):
                return None
            data = data.get(key)
        value = float(data)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _default_multistem_chooser_path() -> Optional[str]:
    if DEFAULT_MULTISTEM_GROUPWISE_PATH.exists():
        groupwise_100 = _read_report_metric(
            DEFAULT_MULTISTEM_GROUPWISE_REPORT,
            "validation_metrics",
            "groupwise",
            "selected_within_100ms_percent",
        )
        chooser_100 = _read_report_metric(
            DEFAULT_MULTISTEM_CHOOSER_REPORT,
            "validation_metrics",
            "selected_within_100ms_percent",
        )
        groupwise_over_1s = _read_report_metric(
            DEFAULT_MULTISTEM_GROUPWISE_REPORT,
            "validation_metrics",
            "groupwise",
            "selected_over_1s",
        )
        chooser_over_1s = _read_report_metric(
            DEFAULT_MULTISTEM_CHOOSER_REPORT,
            "validation_metrics",
            "selected_over_1s",
        )
        groupwise_auto_accepted = _read_report_metric(
            DEFAULT_MULTISTEM_GROUPWISE_REPORT,
            "validation_metrics",
            "auto_gate_shadow",
            "normal",
            "accepted",
        )
        chooser_auto_accepted = _read_report_metric(
            DEFAULT_MULTISTEM_CHOOSER_REPORT,
            "validation_metrics",
            "auto_gate_shadow",
            "normal",
            "accepted",
        )
        groupwise_is_better = (
            groupwise_100 is not None
            and (chooser_100 is None or groupwise_100 >= chooser_100)
            and (groupwise_over_1s is None or chooser_over_1s is None or groupwise_over_1s <= chooser_over_1s)
            and (
                groupwise_auto_accepted is None
                or chooser_auto_accepted is None
                or groupwise_auto_accepted >= chooser_auto_accepted
            )
        )
        if groupwise_is_better:
            return str(DEFAULT_MULTISTEM_GROUPWISE_PATH)
    if DEFAULT_MULTISTEM_CHOOSER_PATH.exists():
        return str(DEFAULT_MULTISTEM_CHOOSER_PATH)
    if DEFAULT_MULTISTEM_GROUPWISE_PATH.exists():
        return str(DEFAULT_MULTISTEM_GROUPWISE_PATH)
    return None


def _stem_evidence(features: FeatureBundle, time_sec: float) -> Dict[str, float]:
    t = float(time_sec)
    beat = float(features.beat_sec)
    bar = 4.0 * beat
    times = features.frame_times
    pre_short = max(beat, 1.0 * bar)
    pre_long = max(beat, 2.0 * bar)
    post_short = max(beat, 1.0 * bar)
    post_long = max(beat, 4.0 * bar)
    pre_rms = _mean_window(features.rms, times, t - pre_long, t)
    post_rms = _mean_window(features.rms, times, t, t + post_long)
    pre_short_rms = _mean_window(features.rms, times, t - pre_short, t, fallback=pre_rms)
    post_short_rms = _mean_window(features.rms, times, t, t + post_short, fallback=post_rms)
    pre_low = _mean_window(features.low_energy, times, t - pre_long, t)
    post_low = _mean_window(features.low_energy, times, t, t + post_long)
    onset_peak = _max_window(features.onset, times, t, max(0.07, 0.20 * beat))
    attack_peak = _max_window(features.combined_attack, times, t, max(0.07, 0.20 * beat))
    flux_peak = _max_window(features.spectral_flux, times, t, max(0.08, 0.30 * beat))
    density = _mean_window(features.onset, times, t, t + post_long)
    energy_jump = _clip01((post_rms - pre_rms + 0.25) / 1.25)
    short_energy_jump = _clip01((post_short_rms - pre_short_rms + 0.25) / 1.25)
    low_jump = _clip01((post_low - pre_low + 0.25) / 1.25)
    dropout = _clip01((pre_short_rms - post_short_rms + 0.12) / 0.70)
    reentry = _clip01((post_short_rms - pre_short_rms + 0.12) / 0.70)
    post_activity = _clip01((0.45 * post_rms) + (0.35 * density) + (0.20 * post_low))
    return {
        "attack": float(attack_peak),
        "onset_peak": float(onset_peak),
        "flux_peak": float(flux_peak),
        "energy_jump": float(energy_jump),
        "short_energy_jump": float(short_energy_jump),
        "low_jump": float(low_jump),
        "dropout": float(dropout),
        "reentry": float(reentry),
        "post_activity": float(post_activity),
        "pre_rms": float(pre_rms),
        "post_rms": float(post_rms),
    }


def _feature_peak_candidates(
    features: FeatureBundle,
    *,
    role: str,
    limit: int,
) -> List[Dict[str, Any]]:
    times = features.frame_times
    if times.size < 4:
        return []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    beat = float(features.beat_sec)
    max_time = max(0.0, float(features.duration_sec) * 0.72)
    if role == "instrumental":
        curve = (0.45 * features.combined_attack) + (0.35 * features.spectral_flux) + (0.20 * features.rms)
    elif role == "vocals":
        curve = (0.55 * features.combined_attack) + (0.30 * features.spectral_flux) + (0.15 * features.rms)
    elif role == "bass":
        curve = (0.55 * features.low_jump_curve) + (0.25 * features.low_energy) + (0.20 * features.rms)
    else:
        curve = features.combined_attack
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size == 0 or float(np.max(curve)) <= 1e-9:
        return []
    distance = max(1, int(round((0.50 * beat) / max(1e-6, hop))))
    height = max(float(np.percentile(curve, 60.0)), float(np.max(curve)) * 0.24)
    peaks, _ = signal.find_peaks(curve, distance=distance, height=height, prominence=max(0.015, float(np.std(curve)) * 0.25))
    min_time = max(0.75, 2.0 * beat)
    valid = [
        int(idx)
        for idx in peaks
        if min_time <= float(times[idx]) <= max_time
    ]
    ranked = sorted(valid, key=lambda idx: float(curve[idx]), reverse=True)[: max(0, int(limit))]
    out: List[Dict[str, Any]] = []
    denom = max(1e-6, float(np.max(curve)))
    for idx in ranked:
        out.append(
            {
                "time_sec": float(times[idx]),
                "role": role,
                "source": f"{role}_peak",
                "source_score": float(np.clip(curve[idx] / denom, 0.0, 1.0)),
            }
        )
    return out


def _transition_candidates(
    features: FeatureBundle,
    *,
    role: str,
    kind: str,
    limit: int,
) -> List[Dict[str, Any]]:
    times = features.frame_times
    if times.size < 8:
        return []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    beat = float(features.beat_sec)
    pre_frames = max(2, int(round((2.0 * beat) / max(1e-6, hop))))
    post_frames = max(2, int(round((1.5 * beat) / max(1e-6, hop))))
    rms = np.asarray(features.rms, dtype=np.float64)
    curve = np.zeros_like(rms)
    for idx in range(pre_frames, max(pre_frames, len(rms) - post_frames)):
        pre = float(np.mean(rms[idx - pre_frames : idx]))
        post = float(np.mean(rms[idx : idx + post_frames]))
        if kind == "dropout":
            curve[idx] = max(0.0, pre - post)
        else:
            curve[idx] = max(0.0, post - pre)
    if np.max(curve) <= 1e-9:
        return []
    distance = max(1, int(round((2.0 * beat) / max(1e-6, hop))))
    height = max(float(np.percentile(curve, 72.0)), float(np.max(curve)) * 0.30)
    peaks, _ = signal.find_peaks(curve, distance=distance, height=height, prominence=max(0.01, float(np.std(curve)) * 0.30))
    min_time = max(0.75, 2.0 * beat)
    max_time = max(0.0, float(features.duration_sec) * 0.72)
    valid = [int(idx) for idx in peaks if min_time <= float(times[idx]) <= max_time]
    ranked = sorted(valid, key=lambda idx: float(curve[idx]), reverse=True)[: max(0, int(limit))]
    return [
        {
            "time_sec": float(times[idx]),
            "role": role,
            "source": f"{role}_{kind}",
            "source_score": float(np.clip(curve[idx] / max(1e-6, np.max(curve)), 0.0, 1.0)),
        }
        for idx in ranked
    ]


def _windowed_means(values: np.ndarray, pre_frames: int, post_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return x, x
    indices = np.arange(n, dtype=np.int64)
    csum = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(x)])

    pre_start = np.maximum(0, indices - max(1, int(pre_frames)))
    pre_count = np.maximum(1, indices - pre_start)
    pre = (csum[indices] - csum[pre_start]) / pre_count

    post_end = np.minimum(n, indices + max(1, int(post_frames)))
    post_count = np.maximum(1, post_end - indices)
    post = (csum[post_end] - csum[indices]) / post_count
    return pre, post


def _curve_candidates(
    features: FeatureBundle,
    *,
    role: str,
    source: str,
    curve: np.ndarray,
    limit: int,
    distance_beats: float = 0.75,
    percentile: float = 68.0,
    prominence_scale: float = 0.22,
) -> List[Dict[str, Any]]:
    times = features.frame_times
    values = np.asarray(curve, dtype=np.float64)
    n = min(int(times.size), int(values.size))
    if n < 4:
        return []
    times = times[:n]
    values = values[:n]
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 1e-9:
        return []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    beat = float(features.beat_sec)
    max_time = max(0.0, float(features.duration_sec) * 0.72)
    min_time = max(0.75, 2.0 * beat)
    distance = max(1, int(round((float(distance_beats) * beat) / max(1e-6, hop))))
    height = max(float(np.percentile(values, float(percentile))), max_value * 0.20)
    peaks, _props = signal.find_peaks(
        values,
        distance=distance,
        height=height,
        prominence=max(0.008, float(np.std(values)) * float(prominence_scale)),
    )
    valid = [int(idx) for idx in peaks if min_time <= float(times[idx]) <= max_time]
    ranked = sorted(valid, key=lambda idx: float(values[idx]), reverse=True)[: max(0, int(limit))]
    denom = max(1e-6, max_value)
    return [
        {
            "time_sec": float(times[idx]),
            "role": role,
            "source": source,
            "source_score": float(np.clip(values[idx] / denom, 0.0, 1.0)),
        }
        for idx in ranked
    ]


def _bar_transition_candidates(
    features: FeatureBundle,
    *,
    role: str,
    limit: int,
) -> List[Dict[str, Any]]:
    times = features.frame_times
    if times.size < 8:
        return []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    beat = float(features.beat_sec)
    bar = 4.0 * beat
    pre_frames = max(2, int(round((2.0 * bar) / max(1e-6, hop))))
    post_frames = max(2, int(round((4.0 * bar) / max(1e-6, hop))))
    short_frames = max(2, int(round((1.0 * beat) / max(1e-6, hop))))

    pre_rms, post_rms = _windowed_means(features.rms, pre_frames, post_frames)
    pre_low, post_low = _windowed_means(features.low_energy, pre_frames, post_frames)
    _pre_onset, post_onset = _windowed_means(features.onset, short_frames, max(short_frames, int(round((2.0 * beat) / max(1e-6, hop)))))
    _pre_flux, post_flux = _windowed_means(features.spectral_flux, short_frames, max(short_frames, int(round((2.0 * beat) / max(1e-6, hop)))))

    rms_reentry = np.maximum(0.0, post_rms - pre_rms)
    low_reentry = np.maximum(0.0, post_low - pre_low)
    rms_dropout = np.maximum(0.0, pre_rms - post_rms)
    low_dropout = np.maximum(0.0, pre_low - post_low)

    if role == "vocals":
        curve = (0.45 * rms_dropout) + (0.25 * low_dropout) + (0.18 * rms_reentry) + (0.12 * post_flux)
    elif role == "bass":
        curve = (0.54 * low_reentry) + (0.25 * rms_reentry) + (0.13 * post_low) + (0.08 * post_onset)
    elif role == "instrumental":
        curve = (0.40 * rms_reentry) + (0.24 * post_flux) + (0.20 * low_reentry) + (0.16 * post_rms)
    else:
        curve = (0.34 * rms_reentry) + (0.24 * low_reentry) + (0.23 * post_onset) + (0.19 * post_flux)

    return _curve_candidates(
        features,
        role=role,
        source=f"{role}_bar_transition",
        curve=curve,
        limit=limit,
        distance_beats=8.0,
        percentile=72.0,
        prominence_scale=0.25,
    )


def _bpm_clock_bar_sources(
    features_by_role: Mapping[str, FeatureBundle],
    *,
    bpm: Optional[float],
    clock_zero_sec: float = 0.0,
    limit: int = 96,
) -> List[Dict[str, Any]]:
    if not bpm or bpm <= 0 or not features_by_role:
        return []
    beat = 60.0 / float(bpm)
    bar = 4.0 * beat
    duration = max(float(features.duration_sec) for features in features_by_role.values())
    max_time = max(0.0, duration * 0.72)
    max_bar = min(int(limit), max(0, int(math.floor(max_time / max(1e-6, bar))) + 1))
    out: List[Dict[str, Any]] = []
    for bar_number in range(1, max_bar + 1):
        t = float(clock_zero_sec) + float((bar_number - 1) * bar)
        if t < max(0.75, 2.0 * beat) or t > max_time:
            continue
        clock = bpm_clock_for_time(t, bpm, clock_zero_sec=float(clock_zero_sec or 0.0))
        phrase_score, phrase_label = phrase_strength_for_bar(bar_number)
        phrase_score = float(phrase_score or 0.0)
        per_role: Dict[str, float] = {}
        for role, features in features_by_role.items():
            if role not in {"drums", "bass", "instrumental", "vocals"}:
                continue
            evidence = _stem_evidence(features, t)
            if role == "drums":
                role_score = max(
                    _clip01(evidence.get("attack", 0.0)),
                    0.65 * _clip01(evidence.get("low_jump", 0.0)),
                    0.60 * _clip01(evidence.get("post_activity", 0.0)),
                )
            elif role == "bass":
                role_score = max(
                    _clip01(evidence.get("low_jump", 0.0)),
                    0.70 * _clip01(evidence.get("post_activity", 0.0)),
                )
            elif role == "instrumental":
                role_score = max(
                    _clip01(evidence.get("energy_jump", 0.0)),
                    _clip01(evidence.get("short_energy_jump", 0.0)),
                    0.80 * _clip01(evidence.get("flux_peak", 0.0)),
                    0.65 * _clip01(evidence.get("post_activity", 0.0)),
                )
            else:
                role_score = max(
                    _clip01(evidence.get("dropout", 0.0)),
                    0.75 * _clip01(evidence.get("reentry", 0.0)),
                    0.45 * _clip01(evidence.get("flux_peak", 0.0)),
                )
            per_role[role] = float(role_score)
        if not per_role:
            continue
        audio_score = max(per_role.values())
        mean_score = float(np.mean(list(per_role.values())))
        combined = float(np.clip((0.55 * audio_score) + (0.25 * mean_score) + (0.20 * phrase_score), 0.0, 1.0))
        source = "bpm_clock_phrase_bar" if phrase_score >= 0.48 else "bpm_clock_bar"
        if combined < 0.22 and phrase_score < 0.86:
            continue
        out.append(
            {
                "time_sec": t,
                "role": "unknown",
                "source": source,
                "source_score": combined,
                "bpm_clock": clock or {},
                "clock_bar": int(bar_number),
                "phrase_strength": phrase_score,
                "phrase": str(phrase_label or ""),
                "clock_audio_score": float(audio_score),
                "clock_role_scores": dict(per_role),
            }
        )
    return out


def _saved_candidate_sources(candidates: Sequence[Mapping[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for candidate in list(candidates)[: max(0, int(limit))]:
        t = candidate_effective_time(candidate)
        if t is None:
            continue
        micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
        out.append(
            {
                "time_sec": float(t),
                "role": "saved",
                "source": "saved_candidate",
                "source_score": float(candidate.get("score", candidate.get("confidence_score", 0.0)) or 0.0),
                "saved_selected": bool(candidate.get("selected")),
                "saved_rejected": bool(candidate.get("rejected")),
                "saved_rank": int(_finite_float(candidate.get("rank", candidate.get("handcrafted_rank", 999)), 999)),
                "saved_model_rank": int(_finite_float(candidate.get("model_rank", 999), 999)),
                "saved_score": float(candidate.get("score", candidate.get("confidence_score", 0.0)) or 0.0),
                "saved_micro_confidence": float(candidate.get("micro_confidence", micro.get("micro_confidence", 0.0)) or 0.0),
            }
        )
    return out


def _source_names(item: Mapping[str, Any]) -> List[str]:
    sources = item.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        name = str(source.get("source") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _dedupe_sources(sources: Sequence[Mapping[str, Any]], radius_sec: float = 0.055) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for source in sorted(sources, key=lambda row: float(row.get("time_sec", 0.0))):
        t = float(source.get("time_sec", 0.0))
        match: Optional[Dict[str, Any]] = None
        for item in grouped:
            if abs(float(item["time_sec"]) - t) <= float(radius_sec):
                match = item
                break
        if match is None:
            grouped.append(
                {
                    "time_sec": t,
                    "roles": {str(source.get("role", "unknown"))},
                    "sources": [dict(source)],
                    "source_score": float(source.get("source_score", 0.0) or 0.0),
                }
            )
            continue
        match["sources"].append(dict(source))
        match["roles"].add(str(source.get("role", "unknown")))
        source_name = str(source.get("source") or "")
        source_clock = source.get("bpm_clock") if isinstance(source.get("bpm_clock"), Mapping) else {}
        prefer_clock_time = (
            source_name.startswith("bpm_clock")
            and bool(source_clock.get("on_one"))
            and float(source.get("phrase_strength", 0.0) or 0.0) >= 0.48
        )
        if prefer_clock_time or float(source.get("source_score", 0.0) or 0.0) > float(match.get("source_score", 0.0) or 0.0):
            match["time_sec"] = t
            match["source_score"] = max(float(match.get("source_score", 0.0) or 0.0), float(source.get("source_score", 0.0) or 0.0))
    for item in grouped:
        item["roles"] = sorted(role for role in item["roles"] if role)
    return grouped


def _append_unique_candidate(
    out: List[Dict[str, Any]],
    candidate: Mapping[str, Any],
    times: List[float],
    *,
    radius_sec: float = 0.055,
) -> bool:
    t = _finite_float(candidate.get("timestamp"), 0.0)
    if t <= 0.0:
        return False
    if any(abs(t - existing) <= float(radius_sec) for existing in times):
        return False
    out.append(dict(candidate))
    times.append(float(t))
    return True


def _diversified_candidates(candidates: Sequence[Mapping[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    max_count = max(0, int(limit))
    if max_count <= 0:
        return []
    ranked = [dict(candidate) for candidate in candidates]
    kept: List[Dict[str, Any]] = []
    kept_times: List[float] = []
    front_count = min(max_count, max(12, max_count // 3))
    for candidate in ranked[:front_count]:
        _append_unique_candidate(kept, candidate, kept_times)
        if len(kept) >= max_count:
            return kept

    priority_sources = (
        "saved_candidate",
        "drums_bar_transition",
        "bass_bar_transition",
        "instrumental_bar_transition",
        "vocals_bar_transition",
        "drums_low_jump",
        "bass_low_jump",
        "instrumental_flux",
        "vocals_dropout",
        "vocals_reentry",
        "bpm_clock_phrase_bar",
        "bpm_clock_bar",
        "drums_peak",
        "instrumental_peak",
        "bass_peak",
        "vocals_peak",
    )
    per_source = 3 if max_count >= 50 else 2
    for source_name in priority_sources:
        added = 0
        source_cap = min(24, max(8, max_count // 4)) if source_name.startswith("bpm_clock") else per_source
        for candidate in ranked:
            names = candidate.get("multistem_source_names")
            if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or source_name not in {str(name) for name in names}:
                continue
            if _append_unique_candidate(kept, candidate, kept_times):
                added += 1
                if len(kept) >= max_count:
                    return kept
                if added >= source_cap:
                    break

    for candidate in ranked:
        _append_unique_candidate(kept, candidate, kept_times)
        if len(kept) >= max_count:
            break
    return kept


def _format_candidate(
    *,
    time_sec: float,
    rank: int,
    score: float,
    roles: Sequence[str],
    evidence: Mapping[str, Any],
    source_score: float,
) -> Dict[str, Any]:
    drums = evidence.get("drums", {})
    bass = evidence.get("bass", {})
    inst = evidence.get("instrumental", {})
    vocals = evidence.get("vocals", {})
    drums_attack = _clip01(drums.get("attack", 0.0))
    bass_low = max(_clip01(bass.get("low_jump", 0.0)), _clip01(drums.get("low_jump", 0.0)))
    inst_jump = max(_clip01(inst.get("energy_jump", 0.0)), _clip01(inst.get("flux_peak", 0.0)))
    vocal_score = max(_clip01(vocals.get("dropout", 0.0)), 0.65 * _clip01(vocals.get("reentry", 0.0)))
    post_activity = max(
        _clip01(drums.get("post_activity", 0.0)),
        _clip01(inst.get("post_activity", 0.0)),
        _clip01(bass.get("post_activity", 0.0)),
    )
    reason_roles = ",".join(roles) if roles else "evidence"
    return {
        "rank": int(rank),
        "handcrafted_rank": int(rank),
        "model_rank": 0,
        "model_score": None,
        "selected_by": "multistem_candidate",
        "timestamp": float(time_sec),
        "coarse_timestamp": float(time_sec),
        "confidence_score": float(score),
        "score": float(score),
        "transient_strength": float(drums_attack),
        "low_end_jump": float(bass_low),
        "post_drop_density": float(post_activity),
        "pre_post_energy_ratio": float(1.0 + (2.0 * max(inst_jump, bass_low))),
        "energy_contrast": float(max(inst_jump, bass_low, vocal_score)),
        "rhythmic_consistency": float(_clip01(drums.get("post_activity", 0.0))),
        "snap_offset": 0.0,
        "snap_offset_sec": 0.0,
        "selected": False,
        "rejected": False,
        "reason": f"multistem_candidate:{reason_roles}",
        "rejection_reason": "",
        "multistem_agreement": float(min(1.0, len(set(roles) - {"saved", "unknown", "clock"}) / 3.0)),
        "multistem_roles": list(roles),
        "multistem_source_score": float(source_score),
        "drums_transient_score": float(drums_attack),
        "bass_low_jump_score": float(bass_low),
        "inst_energy_jump_score": float(inst_jump),
        "vocal_transition_score": float(vocal_score),
        "vocal_dropout_score": float(_clip01(vocals.get("dropout", 0.0))),
        "vocal_reentry_score": float(_clip01(vocals.get("reentry", 0.0))),
        "debug": {
            "multistem_score": float(score),
            "drums_attack": float(drums_attack),
            "bass_low_jump": float(bass_low),
            "inst_jump": float(inst_jump),
            "vocal_transition": float(vocal_score),
            "post_activity": float(post_activity),
            "source_score": float(source_score),
        },
    }


def generate_multistem_candidates(
    audio_path: str,
    *,
    saved_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    limit: int = 50,
    per_stem_limit: int = 24,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    group = find_stem_group(audio_path)
    bpm = infer_bpm_from_path(audio_path)
    features_by_role: Dict[str, FeatureBundle] = {}
    for role, path in group.roles.items():
        if role == "full":
            continue
        try:
            features_by_role[role] = _load_features(path, role, bpm=bpm, sample_rate=int(sample_rate))
            if bpm is None:
                bpm = float(features_by_role[role].bpm)
        except Exception:
            continue
    try:
        beatgrid: Optional[BeatGrid] = resolve_beatgrid(features_by_role, bpm=bpm) if features_by_role else None
    except Exception:
        beatgrid = None

    sources: List[Dict[str, Any]] = _saved_candidate_sources(saved_candidates or [], limit=20)
    for role, features in features_by_role.items():
        if role not in {"drums", "bass", "instrumental", "vocals"}:
            continue
        sources.extend(_feature_peak_candidates(features, role=role, limit=per_stem_limit))
        sources.extend(_bar_transition_candidates(features, role=role, limit=max(6, per_stem_limit // 2)))
        sources.extend(
            _curve_candidates(
                features,
                role=role,
                source=f"{role}_low_jump",
                curve=features.low_jump_curve,
                limit=max(4, per_stem_limit // 3),
                distance_beats=0.75,
                percentile=68.0,
            )
        )
        if role in {"drums", "instrumental", "vocals"}:
            sources.extend(
                _curve_candidates(
                    features,
                    role=role,
                    source=f"{role}_flux",
                    curve=features.spectral_flux,
                    limit=max(4, per_stem_limit // 3),
                    distance_beats=0.75,
                    percentile=70.0,
                )
            )
    vocals_features = features_by_role.get("vocals")
    if vocals_features is not None:
        sources.extend(_transition_candidates(vocals_features, role="vocals", kind="dropout", limit=16))
        sources.extend(_transition_candidates(vocals_features, role="vocals", kind="reentry", limit=10))
    inst_features = features_by_role.get("instrumental")
    if inst_features is not None:
        sources.extend(_transition_candidates(inst_features, role="instrumental", kind="reentry", limit=12))
    sources.extend(
        _bpm_clock_bar_sources(
            features_by_role,
            bpm=bpm,
            clock_zero_sec=float((beatgrid or {}).bar_zero_sec) if beatgrid is not None else 0.0,
            limit=96,
        )
    )

    grouped_sources = _dedupe_sources(sources)
    candidates: List[Dict[str, Any]] = []
    for item in grouped_sources:
        t = float(item["time_sec"])
        evidence = {
            role: _stem_evidence(features, t)
            for role, features in features_by_role.items()
            if 0.0 < t < float(features.duration_sec)
        }
        drums = evidence.get("drums", {})
        bass = evidence.get("bass", {})
        inst = evidence.get("instrumental", {})
        vocals = evidence.get("vocals", {})
        drums_attack = _clip01(drums.get("attack", 0.0))
        bass_low = max(_clip01(bass.get("low_jump", 0.0)), _clip01(drums.get("low_jump", 0.0)))
        inst_jump = max(_clip01(inst.get("energy_jump", 0.0)), _clip01(inst.get("flux_peak", 0.0)))
        vocal_transition = max(_clip01(vocals.get("dropout", 0.0)), 0.65 * _clip01(vocals.get("reentry", 0.0)))
        roles = list(item.get("roles") or [])
        agreement = float(min(1.0, len(set(roles) - {"saved", "unknown", "clock"}) / 3.0))
        saved_prior = 0.08 if "saved" in set(roles) else 0.0
        post_activity = max(
            _clip01(drums.get("post_activity", 0.0)),
            _clip01(inst.get("post_activity", 0.0)),
            _clip01(bass.get("post_activity", 0.0)),
        )
        source_score = _clip01(item.get("source_score", 0.0))
        source_names = _source_names(item)
        score = float(
            np.clip(
                (0.26 * drums_attack)
                + (0.21 * inst_jump)
                + (0.19 * bass_low)
                + (0.13 * vocal_transition)
                + (0.12 * agreement)
                + (0.06 * post_activity)
                + (0.03 * source_score)
                + saved_prior,
                0.0,
                1.0,
            )
        )
        candidate = _format_candidate(
            time_sec=t,
            rank=0,
            score=score,
            roles=roles,
            evidence=evidence,
            source_score=source_score,
        )
        candidate["multistem_source_names"] = source_names
        candidate["multistem_sources"] = source_names
        clock_sources = [
            source
            for source in item.get("sources", [])
            if isinstance(source, Mapping) and str(source.get("source") or "").startswith("bpm_clock")
        ]
        if clock_sources:
            best_clock = max(clock_sources, key=lambda source: float(source.get("source_score", 0.0) or 0.0))
            candidate["bpm_clock"] = dict(best_clock.get("bpm_clock") or {})
            candidate["clock_bar"] = int(best_clock.get("clock_bar", 0) or 0)
            candidate["clock_phrase"] = str(best_clock.get("phrase") or "")
            candidate["clock_phrase_strength"] = float(best_clock.get("phrase_strength", 0.0) or 0.0)
            candidate["clock_audio_score"] = float(best_clock.get("clock_audio_score", 0.0) or 0.0)
            candidate["reason"] = f"{candidate.get('reason')};{best_clock.get('source')}:{candidate['clock_phrase'] or 'bar'}"
        saved_sources = [
            source
            for source in item.get("sources", [])
            if isinstance(source, Mapping) and str(source.get("source") or "") == "saved_candidate"
        ]
        if saved_sources:
            candidate["saved_source_count"] = int(len(saved_sources))
            candidate["saved_selected_used"] = 1.0 if any(bool(source.get("saved_selected")) for source in saved_sources) else 0.0
            candidate["saved_rejected_used"] = 1.0 if any(bool(source.get("saved_rejected")) for source in saved_sources) else 0.0
            candidate["saved_best_rank"] = float(min(_finite_float(source.get("saved_rank"), 999.0) for source in saved_sources))
            candidate["saved_best_model_rank"] = float(min(_finite_float(source.get("saved_model_rank"), 999.0) for source in saved_sources))
            candidate["saved_best_score"] = float(max(_finite_float(source.get("saved_score"), 0.0) for source in saved_sources))
            candidate["saved_best_micro_confidence"] = float(max(_finite_float(source.get("saved_micro_confidence"), 0.0) for source in saved_sources))
        candidates.append(candidate)

    candidates.sort(key=lambda row: (-float(row.get("score", 0.0)), float(row.get("timestamp", 0.0))))
    kept = _diversified_candidates(candidates, limit=max(0, int(limit)))
    kept_times = [float(candidate.get("timestamp", 0.0) or 0.0) for candidate in kept]
    for candidate in candidates:
        roles = set(candidate.get("multistem_roles") or [])
        if "saved" not in roles:
            continue
        t = float(candidate.get("timestamp", 0.0) or 0.0)
        if any(abs(t - existing) <= 0.010 for existing in kept_times):
            continue
        kept.append(candidate)
        kept_times.append(t)
    for rank, candidate in enumerate(kept, start=1):
        candidate["rank"] = int(rank)
        candidate["handcrafted_rank"] = int(rank)
    return {
        "stem_group": group.to_dict(),
        "bpm": bpm,
        "beatgrid": beatgrid.to_dict() if beatgrid is not None else None,
        "candidate_count": int(len(candidates)),
        "source_count": int(len(sources)),
        "candidates": kept,
    }


def choose_multistem_candidate(
    audio_path: str,
    *,
    saved_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    confidence_tier: str = "UNKNOWN",
    mode: str = "normal",
    chooser_model_path: Optional[str] = None,
    expanded_limit: int = 120,
    microalign_limit: int = 50,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    generated = generate_multistem_candidates(
        audio_path,
        saved_candidates=saved_candidates,
        limit=max(expanded_limit, microalign_limit),
        sample_rate=sample_rate,
    )
    expanded = list(generated.get("candidates") or [])
    aligned = microalign_candidate_dicts(audio_path, expanded, limit=max(1, int(microalign_limit)))
    pipeline = run_drop_candidate_pipeline(
        aligned,
        cluster_radius_sec=0.085,
        limit=max(1, int(microalign_limit)),
    )
    ranked = list(pipeline.get("candidates") or aligned)
    model_path = chooser_model_path
    if model_path is None:
        model_path = _default_multistem_chooser_path()
    suggestion = choose_microaligned_candidate(
        ranked,
        confidence_tier=str(confidence_tier or "UNKNOWN"),
        mode=str(mode or "normal"),
        chooser_model_path=model_path,
    )
    return {
        "ok": True,
        "stem_group": generated.get("stem_group"),
        "bpm": generated.get("bpm"),
        "source_count": generated.get("source_count"),
        "candidate_count": generated.get("candidate_count"),
        "pipeline": pipeline.get("summary"),
        "candidates": ranked,
        "suggestion": suggestion,
    }
