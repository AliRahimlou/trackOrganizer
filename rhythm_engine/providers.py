from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .types import RhythmEngineConfig, RhythmEstimate


ProviderFn = Callable[[str, RhythmEngineConfig], RhythmEstimate]


def _estimate_bpm_from_beats(beats: Sequence[float]) -> Optional[float]:
    arr = np.asarray(list(beats), dtype=np.float64)
    if arr.size < 3:
        return None
    intervals = np.diff(np.sort(arr))
    intervals = intervals[(intervals > 0.12) & (intervals < 3.0)]
    if intervals.size == 0:
        return None
    bpm = 60.0 / float(np.median(intervals))
    return float(bpm) if math.isfinite(bpm) and bpm > 0 else None


def _heuristic_downbeats(beats: Sequence[float], strength_times: np.ndarray, strength: np.ndarray, *, beats_per_bar: int = 4) -> List[float]:
    beat_arr = np.asarray(list(beats), dtype=np.float64)
    if beat_arr.size < max(8, int(beats_per_bar) * 2):
        return []
    if strength_times.size == 0 or strength.size == 0:
        return [float(t) for t in beat_arr[:: max(1, int(beats_per_bar))]]

    limit = min(strength_times.size, strength.size)
    times = np.asarray(strength_times[:limit], dtype=np.float64)
    values = np.asarray(strength[:limit], dtype=np.float64)
    phase_scores: List[tuple[float, int]] = []
    for phase in range(max(1, int(beats_per_bar))):
        idx = np.arange(phase, len(beat_arr), max(1, int(beats_per_bar)))
        if idx.size == 0:
            continue
        sampled = np.interp(beat_arr[idx], times, values, left=0.0, right=0.0)
        boundary_bonus = 0.025 * max(0.0, 1.0 - (phase / max(1, int(beats_per_bar) - 1)))
        phase_scores.append((float(np.mean(sampled) + boundary_bonus), phase))
    if not phase_scores:
        return []
    phase_scores.sort(key=lambda row: (-row[0], row[1]))
    phase = int(phase_scores[0][1])
    return [float(t) for t in beat_arr[phase:: max(1, int(beats_per_bar))]]


def librosa_provider(audio_path: str, config: RhythmEngineConfig) -> RhythmEstimate:
    try:
        import librosa
    except Exception as exc:
        return RhythmEstimate.unavailable("librosa", str(exc) or exc.__class__.__name__)

    try:
        y, sr = librosa.load(audio_path, sr=int(config.sample_rate), mono=True)
        if y.size == 0:
            return RhythmEstimate.failed("librosa", "empty_audio")
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sr,
            hop_length=512,
            start_bpm=120.0,
            units="frames",
        )
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        frame_times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=512)
        downbeats = _heuristic_downbeats(beats, frame_times, onset, beats_per_bar=config.beats_per_bar)
        bpm = float(np.ravel(tempo)[0]) if np.size(tempo) else _estimate_bpm_from_beats(beats)
        if bpm is None or bpm < config.min_bpm or bpm > config.max_bpm:
            bpm = _estimate_bpm_from_beats(beats)
        confidence = min(0.72, 0.18 + (0.015 * len(beats)))
        return RhythmEstimate(
            provider="librosa",
            beats=tuple(float(t) for t in beats),
            downbeats=tuple(downbeats),
            bpm=bpm,
            confidence=confidence,
            duration_sec=float(len(y) / float(sr)),
            sample_rate=int(sr),
            metadata={"beat_count": int(len(beats)), "downbeat_heuristic": True},
        )
    except Exception as exc:
        return RhythmEstimate.failed("librosa", str(exc) or exc.__class__.__name__)


def track_organizer_provider(audio_path: str, config: RhythmEngineConfig) -> RhythmEstimate:
    try:
        from drop_aligner.beatgrid import resolve_beatgrid
        from drop_aligner.detector import DropDetectorConfig, extract_features
    except Exception as exc:
        return RhythmEstimate.unavailable("track_organizer", str(exc) or exc.__class__.__name__)

    try:
        cfg = DropDetectorConfig(
            sample_rate=int(config.sample_rate),
            min_drop_time_sec=0.0,
            min_score=0.0,
            use_ranker_model=False,
            use_region_model=False,
            use_drumprint=False,
        )
        features = extract_features(audio_path, cfg)
        grid = resolve_beatgrid({"full": features}, bpm=features.bpm)
        beat = float(grid.beat_sec)
        if beat <= 0:
            return RhythmEstimate.failed("track_organizer", "invalid_beat_period")
        beats = np.arange(float(grid.bar_zero_sec), float(features.duration_sec) + (0.5 * beat), beat, dtype=np.float64)
        beats = beats[beats >= 0.0]
        downbeats = np.arange(float(grid.bar_zero_sec), float(features.duration_sec) + (0.5 * grid.bar_sec), float(grid.bar_sec), dtype=np.float64)
        downbeats = downbeats[downbeats >= 0.0]
        confidence = max(0.18, min(0.86, float(grid.downbeat_confidence)))
        return RhythmEstimate(
            provider="track_organizer",
            beats=tuple(float(t) for t in beats),
            downbeats=tuple(float(t) for t in downbeats),
            bpm=float(grid.bpm),
            confidence=confidence,
            duration_sec=float(features.duration_sec),
            sample_rate=int(features.sr),
            metadata={"beatgrid": grid.to_dict(), "beat_count": int(len(beats))},
        )
    except Exception as exc:
        return RhythmEstimate.failed("track_organizer", str(exc) or exc.__class__.__name__)


def madmom_provider(audio_path: str, config: RhythmEngineConfig) -> RhythmEstimate:
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor  # type: ignore
    except Exception as exc:
        return RhythmEstimate.unavailable("madmom", str(exc) or exc.__class__.__name__)

    try:
        proc = RNNDownBeatProcessor()
        activations = proc(audio_path)
        tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=[int(config.beats_per_bar)],
            min_bpm=float(config.min_bpm),
            max_bpm=float(config.max_bpm),
            fps=100,
        )
        seq = np.asarray(tracker(activations), dtype=np.float64)
        if seq.ndim != 2 or seq.shape[0] == 0:
            return RhythmEstimate.failed("madmom", "empty_sequence")
        beats = seq[:, 0]
        downbeats = seq[seq[:, 1] == 1][:, 0]
        return RhythmEstimate(
            provider="madmom",
            beats=tuple(float(t) for t in beats),
            downbeats=tuple(float(t) for t in downbeats),
            bpm=_estimate_bpm_from_beats(beats),
            confidence=0.82,
            metadata={"activation_fps": 100, "beat_count": int(len(beats))},
        )
    except Exception as exc:
        return RhythmEstimate.failed("madmom", str(exc) or exc.__class__.__name__)


def beat_this_provider(audio_path: str, config: RhythmEngineConfig) -> RhythmEstimate:
    try:
        from beat_this.inference import File2Beats  # type: ignore
    except Exception as exc:
        return RhythmEstimate.unavailable("beat_this", str(exc) or exc.__class__.__name__)

    try:
        device = None if config.beat_this_device == "auto" else str(config.beat_this_device)
        kwargs = {
            "checkpoint_path": str(config.beat_this_checkpoint),
            "dbn": bool(config.use_beat_this_dbn),
        }
        if device is not None:
            kwargs["device"] = device
        file2beats = File2Beats(**kwargs)
        beats, downbeats = file2beats(audio_path)
        beat_arr = np.asarray(beats, dtype=np.float64)
        downbeat_arr = np.asarray(downbeats, dtype=np.float64)
        return RhythmEstimate(
            provider="beat_this",
            beats=tuple(float(t) for t in beat_arr),
            downbeats=tuple(float(t) for t in downbeat_arr),
            bpm=_estimate_bpm_from_beats(beat_arr),
            confidence=0.92,
            metadata={
                "checkpoint": str(config.beat_this_checkpoint),
                "dbn": bool(config.use_beat_this_dbn),
                "beat_count": int(len(beat_arr)),
            },
        )
    except Exception as exc:
        return RhythmEstimate.failed("beat_this", str(exc) or exc.__class__.__name__)


def stem_ensemble_provider(audio_path: str, config: RhythmEngineConfig) -> RhythmEstimate:
    try:
        from drop_aligner.multistem import find_stem_group
    except Exception as exc:
        return RhythmEstimate.unavailable("stem_ensemble", str(exc) or exc.__class__.__name__)

    try:
        group = find_stem_group(audio_path)
    except Exception as exc:
        return RhythmEstimate.failed("stem_ensemble", str(exc) or exc.__class__.__name__)

    roles = dict(group.roles)
    musical_roles = {role: path for role, path in roles.items() if role in {"drums", "bass", "instrumental", "vocals", "full"}}
    if len(musical_roles) <= 1:
        return RhythmEstimate.unavailable(
            "stem_ensemble",
            "no_sibling_stems",
            metadata={"stem_group": group.to_dict()},
        )

    role_weights = {
        "drums": 1.25,
        "bass": 0.85,
        "instrumental": 0.66,
        "full": 0.62,
        "vocals": 0.42,
    }
    role_estimates: List[RhythmEstimate] = []
    for role in ("drums", "bass", "instrumental", "full", "vocals"):
        path = musical_roles.get(role)
        if not path:
            continue
        for provider_fn, provider_name in ((track_organizer_provider, "track_organizer"), (librosa_provider, "librosa")):
            estimate = provider_fn(path, config)
            metadata = dict(estimate.metadata)
            metadata["stem_role"] = role
            metadata["stem_path"] = path
            if not estimate.available:
                role_estimates.append(
                    RhythmEstimate(
                        provider=f"stem_ensemble:{role}:{provider_name}",
                        status=estimate.status,
                        reason=estimate.reason,
                        metadata=metadata,
                    )
                )
                continue
            role_weight = float(role_weights.get(role, 0.5))
            role_estimates.append(
                RhythmEstimate(
                    provider=f"stem_ensemble:{role}:{provider_name}",
                    beats=estimate.beats,
                    downbeats=estimate.downbeats,
                    bpm=estimate.bpm,
                    confidence=min(1.0, float(estimate.confidence) * role_weight),
                    duration_sec=estimate.duration_sec,
                    sample_rate=estimate.sample_rate,
                    metadata=metadata,
                )
            )

    usable = available_estimates(role_estimates, config)
    if not usable:
        return RhythmEstimate.failed(
            "stem_ensemble",
            "no_available_stem_estimates",
            metadata={
                "stem_group": group.to_dict(),
                "role_estimates": [estimate.to_dict() for estimate in role_estimates],
            },
        )

    from .fusion import fuse_estimates

    fused = fuse_estimates(usable, config)
    metadata = dict(fused.metadata)
    metadata["stem_group"] = group.to_dict()
    metadata["role_estimates"] = [estimate.to_dict() for estimate in role_estimates]
    metadata["stem_roles"] = sorted(musical_roles)
    return RhythmEstimate(
        provider="stem_ensemble",
        beats=fused.beats,
        downbeats=fused.downbeats,
        bpm=fused.bpm,
        confidence=min(0.94, max(0.18, float(fused.confidence))),
        duration_sec=fused.duration_sec,
        sample_rate=fused.sample_rate,
        metadata=metadata,
    )


PROVIDERS: Dict[str, ProviderFn] = {
    "beat_this": beat_this_provider,
    "librosa": librosa_provider,
    "madmom": madmom_provider,
    "stem_ensemble": stem_ensemble_provider,
    "track_organizer": track_organizer_provider,
}


def run_providers(audio_path: str, config: RhythmEngineConfig) -> List[RhythmEstimate]:
    estimates: List[RhythmEstimate] = []
    seen: set[str] = set()
    for name in config.providers:
        key = str(name).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        provider = PROVIDERS.get(key)
        if provider is None:
            estimates.append(RhythmEstimate.unavailable(key, "unknown_provider"))
            continue
        estimates.append(provider(audio_path, config))
    return estimates


def available_estimates(estimates: Iterable[RhythmEstimate], config: RhythmEngineConfig) -> List[RhythmEstimate]:
    return [
        estimate
        for estimate in estimates
        if estimate.available and float(estimate.confidence) >= float(config.min_provider_confidence)
    ]
