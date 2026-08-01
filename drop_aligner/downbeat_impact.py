from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_WINDOW_BEFORE_MS = 8.0
DEFAULT_WINDOW_AFTER_MS = 90.0
DEFAULT_ZERO_CROSSING_MS = 1.5
DEFAULT_CREST_WINDOW_MS = 60.0
DEFAULT_CREST_RMS_WINDOW_MS = 2.5
MIN_ATTACK_EVIDENCE = 0.12


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: np.ndarray, start: int, end: int) -> float:
    lo = max(0, int(start))
    hi = min(int(values.size), int(end))
    if hi <= lo:
        return 0.0
    return float(np.mean(values[lo:hi], dtype=np.float64))


def _nearest_zero_crossing(
    samples: np.ndarray,
    attack_index: int,
    *,
    radius: int,
    lower: int,
    upper: int,
) -> int:
    if samples.ndim != 2 or samples.shape[0] < 2:
        return int(attack_index)
    attack = max(1, min(int(samples.shape[0] - 1), int(attack_index)))
    channel_delta = np.abs(samples[attack] - samples[attack - 1])
    channel = int(np.argmax(channel_delta)) if channel_delta.size else 0
    mono = np.asarray(samples[:, channel], dtype=np.float64)
    start = max(1, int(lower), attack - max(0, int(radius)))
    end = min(int(upper), int(mono.size - 1), attack + max(0, int(radius)))
    best: Optional[tuple[int, float, int]] = None
    for index in range(start, end + 1):
        previous = float(mono[index - 1])
        current = float(mono[index])
        crossed = previous == 0.0 or current == 0.0 or math.copysign(1.0, previous) != math.copysign(1.0, current)
        if not crossed:
            continue
        candidate = (abs(index - attack), abs(previous) + abs(current), index)
        if best is None or candidate < best:
            best = candidate
    return int(best[2]) if best is not None else int(attack)


def _missing_boom_body_crest(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "reason": str(reason),
        "diagnostic_only": True,
        "moves_marker": False,
    }


def _measure_boom_body_crest(
    samples: np.ndarray,
    *,
    sample_rate: int,
    read_start: int,
    attack_sample: int,
    downbeat_sample: int,
    beat_samples: int,
    crest_window_ms: float,
    crest_rms_window_ms: float,
) -> Dict[str, Any]:
    """Measure the first body's bounded energy crest without moving its edge.

    The useful analogue of an antinode in recorded music is the peak of a
    short-time amplitude envelope, not a single phase-dependent PCM sample.
    The envelope is restricted to the approved one's first quarter-beat and to
    ``crest_window_ms`` after the already-selected attack.
    """

    if samples.ndim != 2 or samples.shape[0] < 2:
        return _missing_boom_body_crest("invalid_audio_window")
    attack_local = int(attack_sample) - int(read_start)
    if attack_local < 0 or attack_local >= int(samples.shape[0] - 1):
        return _missing_boom_body_crest("attack_outside_audio_window")

    requested_samples = max(
        1,
        int(round((max(0.0, float(crest_window_ms)) / 1000.0) * int(sample_rate))),
    )
    quarter_beat_end = int(downbeat_sample) + max(1, int(math.floor(0.25 * int(beat_samples))))
    window_start_sample = int(attack_sample)
    window_end_sample = min(
        int(read_start) + int(samples.shape[0]),
        int(attack_sample) + int(requested_samples),
        int(quarter_beat_end),
    )
    rms_window_samples = max(
        4,
        int(round((max(0.25, float(crest_rms_window_ms)) / 1000.0) * int(sample_rate))),
    )
    if window_end_sample - window_start_sample < rms_window_samples:
        return _missing_boom_body_crest("insufficient_post_attack_audio")

    power = np.mean(np.square(samples.astype(np.float64, copy=False)), axis=1)
    window_start_local = int(window_start_sample - read_start)
    window_end_local = int(window_end_sample - read_start)
    body_power = np.asarray(power[window_start_local:window_end_local], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(body_power, dtype=np.float64)))
    envelope_power = (
        cumulative[rms_window_samples:] - cumulative[:-rms_window_samples]
    ) / float(rms_window_samples)
    if envelope_power.size == 0:
        return _missing_boom_body_crest("empty_crest_envelope")

    maximum_power = float(np.max(envelope_power))
    if not math.isfinite(maximum_power) or maximum_power <= 0.0:
        return _missing_boom_body_crest("silent_post_attack_window")
    # Choose the first point on a flat-topped crest instead of its arbitrary
    # loudest sample. This remains a diagnostic and never relocates the edge.
    crest_candidates = np.flatnonzero(envelope_power >= (0.98 * maximum_power))
    crest_window_offset = int(crest_candidates[0]) if crest_candidates.size else int(np.argmax(envelope_power))
    crest_window_start_sample = int(window_start_sample + crest_window_offset)
    crest_sample = int(crest_window_start_sample + (rms_window_samples // 2))

    crest_start_local = int(crest_window_start_sample - read_start)
    crest_end_local = min(int(samples.shape[0]), crest_start_local + rms_window_samples)
    crest_pcm = np.asarray(samples[crest_start_local:crest_end_local], dtype=np.float64)
    per_frame_peak = np.max(np.abs(crest_pcm), axis=1)
    instantaneous_offset = int(np.argmax(per_frame_peak)) if per_frame_peak.size else 0
    instantaneous_peak_sample = int(crest_window_start_sample + instantaneous_offset)
    instantaneous_peak_abs = (
        float(per_frame_peak[instantaneous_offset]) if per_frame_peak.size else 0.0
    )

    pre_edge_samples = max(rms_window_samples, int(round(0.006 * int(sample_rate))))
    pre_edge_power = _mean(power, attack_local - pre_edge_samples, attack_local)
    sustain_samples = max(rms_window_samples, int(round(0.030 * int(sample_rate))))
    sustain_end_local = min(window_end_local, window_start_local + sustain_samples)
    body_sustain_power = _mean(power, window_start_local, sustain_end_local)
    floor = 1e-12
    crest_to_pre_db = 10.0 * math.log10(max(maximum_power, floor) / max(pre_edge_power, floor))
    normalized_contrast = (maximum_power - pre_edge_power) / max(
        maximum_power + pre_edge_power,
        floor,
    )

    return {
        "ok": True,
        "reason": "bounded_body_envelope_crest_measured",
        "diagnostic_only": True,
        "moves_marker": False,
        "provenance": "bounded-post-attack-rms-envelope-v1",
        "attack_sample": int(attack_sample),
        "crest_sample": int(crest_sample),
        "crest_time_sec": float(crest_sample) / float(sample_rate),
        "crest_offset_ms": float(crest_sample - attack_sample) * 1000.0 / float(sample_rate),
        "crest_power": float(maximum_power),
        "crest_rms": float(math.sqrt(max(0.0, maximum_power))),
        "pre_edge_power": float(pre_edge_power),
        "pre_edge_rms": float(math.sqrt(max(0.0, pre_edge_power))),
        "crest_to_pre_db": float(crest_to_pre_db),
        "normalized_contrast": float(np.clip(normalized_contrast, -1.0, 1.0)),
        "instantaneous_peak_sample": int(instantaneous_peak_sample),
        "instantaneous_peak_time_sec": float(instantaneous_peak_sample) / float(sample_rate),
        "instantaneous_peak_abs": float(instantaneous_peak_abs),
        "window_start_sample": int(window_start_sample),
        "window_end_sample": int(window_end_sample),
        "window_ms": float(window_end_sample - window_start_sample) * 1000.0 / float(sample_rate),
        "requested_window_ms": float(crest_window_ms),
        "rms_window_samples": int(rms_window_samples),
        "rms_window_ms": float(rms_window_samples) * 1000.0 / float(sample_rate),
        "body_sustain_power": float(body_sustain_power),
        "body_sustain_rms": float(math.sqrt(max(0.0, body_sustain_power))),
        "body_sustain_window_ms": float(sustain_end_local - window_start_local)
        * 1000.0
        / float(sample_rate),
        "quarter_beat_guard_sample": int(quarter_beat_end),
    }


def refine_approved_downbeat_attack(
    audio_path: str,
    downbeat_sec: float,
    bpm: float,
    *,
    window_before_ms: float = DEFAULT_WINDOW_BEFORE_MS,
    window_after_ms: float = DEFAULT_WINDOW_AFTER_MS,
    zero_crossing_radius_ms: float = DEFAULT_ZERO_CROSSING_MS,
    crest_window_ms: float = DEFAULT_CREST_WINDOW_MS,
    crest_rms_window_ms: float = DEFAULT_CREST_RMS_WINDOW_MS,
) -> Dict[str, Any]:
    """Bind an already-approved musical one to its first bounded body attack.

    This is deliberately a second-stage operation. It cannot choose another
    beat or another bar: the search ends no later than one quarter of a beat
    after the approved downbeat. The returned ``attack_*`` value is the ALS
    marker candidate; ``zero_crossing_*`` is diagnostic playback information.
    """

    downbeat = _finite_float(downbeat_sec)
    bpm_value = _finite_float(bpm)
    path = Path(audio_path).expanduser()
    if downbeat is None or downbeat < 0.0:
        return {"ok": False, "refined": False, "reason": "invalid_downbeat"}
    if bpm_value is None or bpm_value <= 0.0:
        return {"ok": False, "refined": False, "reason": "invalid_bpm"}
    if not path.is_file():
        return {"ok": False, "refined": False, "reason": "missing_audio"}

    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - dependency failure is runtime-specific
        return {
            "ok": False,
            "refined": False,
            "reason": "soundfile_unavailable",
            "error": str(exc) or exc.__class__.__name__,
        }

    try:
        with sf.SoundFile(str(path)) as source:
            sample_rate = int(source.samplerate)
            frame_count = int(source.frames)
            channel_count = int(source.channels)
            if sample_rate <= 0 or frame_count <= 2 or channel_count <= 0:
                return {"ok": False, "refined": False, "reason": "invalid_audio_geometry"}

            downbeat_sample = int(round(float(downbeat) * float(sample_rate)))
            beat_samples = max(1, int(round((60.0 / float(bpm_value)) * float(sample_rate))))
            if downbeat_sample < 1 or downbeat_sample >= frame_count - 2:
                return {"ok": False, "refined": False, "reason": "downbeat_outside_audio"}

            before_samples = max(0, int(round((float(window_before_ms) / 1000.0) * sample_rate)))
            after_samples = max(1, int(round((float(window_after_ms) / 1000.0) * sample_rate)))
            beat_guard_end = downbeat_sample + max(1, int(math.floor(0.25 * beat_samples)))
            search_start = max(1, downbeat_sample - before_samples)
            search_end = min(frame_count - 2, beat_guard_end, downbeat_sample + after_samples)
            short_samples = max(4, int(round(0.0025 * sample_rate)))
            history_samples = max(short_samples, int(round(0.006 * sample_rate)))
            crest_read_samples = max(
                short_samples,
                int(round((max(0.0, float(crest_window_ms)) / 1000.0) * sample_rate)),
            )
            read_start = max(0, search_start - history_samples - 2)
            read_end = min(frame_count, search_end + crest_read_samples + short_samples + 2)
            if search_end <= search_start or read_end <= read_start + short_samples:
                return {"ok": False, "refined": False, "reason": "empty_attack_window"}
            source.seek(read_start)
            samples = source.read(
                max(1, read_end - read_start),
                dtype="float32",
                always_2d=True,
            )
    except Exception as exc:
        return {
            "ok": False,
            "refined": False,
            "reason": "audio_read_failed",
            "error": str(exc) or exc.__class__.__name__,
        }

    if samples.size == 0 or samples.shape[0] < short_samples + history_samples:
        return {"ok": False, "refined": False, "reason": "insufficient_audio_window"}

    power = np.mean(np.square(samples.astype(np.float64, copy=False)), axis=1)
    step = max(1, int(math.floor(sample_rate / 12000.0)))
    scored: List[tuple[float, int, float, float]] = []
    for absolute_sample in range(search_start, search_end + 1, step):
        local = absolute_sample - read_start
        before_power = _mean(power, local - history_samples, local)
        after_power = _mean(power, local, local + short_samples)
        rise = max(0.0, after_power - before_power)
        score = rise / max(1e-10, after_power + before_power)
        scored.append((float(score), int(absolute_sample), float(before_power), float(after_power)))
    if not scored:
        return {"ok": False, "refined": False, "reason": "no_attack_candidates"}

    maximum_score = max(row[0] for row in scored)
    near_maximum = [row for row in scored if row[0] >= maximum_score * 0.96]
    selected = min(near_maximum or scored, key=lambda row: row[1])
    refined = bool(selected[0] >= MIN_ATTACK_EVIDENCE)
    attack_sample = int(downbeat_sample)
    if refined:
        selected_local = int(selected[1] - read_start)
        upper_local = min(int(samples.shape[0] - 1), int(search_end - read_start))
        edge_end = min(upper_local, selected_local + short_samples)
        derivatives: List[tuple[float, int]] = []
        for local in range(max(1, selected_local), edge_end + 1):
            derivative = float(np.mean(np.abs(samples[local] - samples[local - 1]), dtype=np.float64))
            derivatives.append((derivative, local))
        derivative_peak = max((row[0] for row in derivatives), default=0.0)
        edge = next(
            (local for value, local in derivatives if derivative_peak > 0.0 and value >= 0.80 * derivative_peak),
            selected_local,
        )
        attack_sample = max(search_start, min(search_end, int(read_start + edge)))

    attack_local = int(attack_sample - read_start)
    zero_crossing_radius = max(0, int(round((float(zero_crossing_radius_ms) / 1000.0) * sample_rate)))
    zero_crossing_local = _nearest_zero_crossing(
        samples,
        attack_local,
        radius=zero_crossing_radius,
        lower=search_start - read_start,
        upper=search_end - read_start,
    )
    zero_crossing_sample = int(read_start + zero_crossing_local)
    boom_body_crest = (
        _measure_boom_body_crest(
            samples,
            sample_rate=int(sample_rate),
            read_start=int(read_start),
            attack_sample=int(attack_sample),
            downbeat_sample=int(downbeat_sample),
            beat_samples=int(beat_samples),
            crest_window_ms=float(crest_window_ms),
            crest_rms_window_ms=float(crest_rms_window_ms),
        )
        if refined
        else _missing_boom_body_crest("attack_not_refined")
    )

    alternate_separation = max(1, int(round(0.005 * sample_rate)))
    alternates: List[Dict[str, Any]] = []
    for score, sample, _before_power, _after_power in sorted(scored, key=lambda row: (-row[0], row[1])):
        if abs(int(sample) - int(attack_sample)) < alternate_separation:
            continue
        if any(abs(int(item["sample"]) - int(sample)) < alternate_separation for item in alternates):
            continue
        alternates.append(
            {
                "sample": int(sample),
                "time_sec": float(sample) / float(sample_rate),
                "confidence": float(np.clip(score, 0.0, 1.0)),
            }
        )
        if len(alternates) >= 3:
            break

    return {
        "ok": True,
        "refined": bool(refined),
        "reason": "bounded_attack_found" if refined else "insufficient_attack_evidence",
        "provenance": "bounded-drums-stem-power-onset-v1",
        "audio_path": str(path),
        "sample_rate": int(sample_rate),
        "channel_count": int(channel_count),
        "frame_count": int(frame_count),
        "bpm": float(bpm_value),
        "beat_samples": int(beat_samples),
        "downbeat_sample": int(downbeat_sample),
        "downbeat_time_sec": float(downbeat_sample) / float(sample_rate),
        "attack_sample": int(attack_sample),
        "attack_time_sec": float(attack_sample) / float(sample_rate),
        "attack_delta_ms": float(attack_sample - downbeat_sample) * 1000.0 / float(sample_rate),
        "attack_confidence": float(np.clip(selected[0], 0.0, 1.0)),
        "attack_before_power": float(selected[2]),
        "attack_after_power": float(selected[3]),
        "zero_crossing_sample": int(zero_crossing_sample),
        "zero_crossing_time_sec": float(zero_crossing_sample) / float(sample_rate),
        "search_start_sample": int(search_start),
        "search_end_sample": int(search_end),
        "window_before_ms": float(window_before_ms),
        "window_after_ms": float(window_after_ms),
        "beat_guard_fraction": 0.25,
        "boom_body_crest": boom_body_crest,
        "alternate_candidates": alternates,
    }
