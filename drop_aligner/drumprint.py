from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy import ndimage, signal


DrumprintHash = Tuple[int, int, int]
DrumprintBag = Counter[DrumprintHash]

DRUMPRINT_FEATURE_KEYS = [
    "fingerprint_density",
    "fingerprint_novelty",
    "post_drop_pattern_stability",
    "kick_pattern_repeat_score",
    "self_similarity_boundary_score",
    "later_drop_match_score",
    "fake_hit_penalty",
    "drumprint_pattern_score",
]


@dataclass
class DrumprintConfig:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 96
    fmin_hz: float = 40.0
    fmax_hz: float = 12000.0
    peak_neighborhood_freq_bins: int = 2
    peak_neighborhood_time_frames: int = 3
    peak_percentile: float = 82.0
    peak_window_sec: float = 0.50
    top_peaks_per_window: int = 10
    min_pair_delta_sec: float = 0.06
    max_pair_delta_sec: float = 2.50
    pair_delta_bin_sec: float = 0.08
    fanout: int = 6
    analysis_bars: float = 8.0
    later_match_bars: float = 4.0


@dataclass
class Peak:
    time: float
    freq_bin: int
    magnitude: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "time": float(self.time),
            "freq_bin": int(self.freq_bin),
            "magnitude": float(self.magnitude),
        }


@dataclass
class HashEvent:
    anchor_time: float
    key: DrumprintHash
    magnitude: float


@dataclass
class DrumprintAnalysis:
    duration_sec: float
    bpm: float
    beat_sec: float
    bar_sec: float
    segment_sec: float
    times: np.ndarray
    segment_starts: np.ndarray
    segment_hashes: List[DrumprintBag]
    segment_density: np.ndarray
    novelty_curve: np.ndarray
    boundary_curve: np.ndarray
    low_curve: np.ndarray
    low_peak_times: np.ndarray
    peaks: List[Peak]
    hash_events: List[HashEvent]

    def summary(self) -> Dict[str, object]:
        return {
            "drumprint_enabled": True,
            "drumprint_status": "ok",
            "drumprint_peak_count": int(len(self.peaks)),
            "drumprint_hash_count": int(len(self.hash_events)),
            "drumprint_segment_sec": float(self.segment_sec),
        }


def empty_drumprint_features(*, enabled: bool = False, status: str = "disabled", error: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {
        "enabled": bool(enabled),
        "status": str(status),
    }
    if error:
        out["error"] = str(error)
    for key in DRUMPRINT_FEATURE_KEYS:
        out[key] = 0.0
    return out


def _clip01(value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def _normalize_curve(values: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    high = float(np.percentile(x, percentile))
    if not math.isfinite(high) or high <= 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip(x / high, 0.0, 1.0)


def _weighted_jaccard(a: Mapping[DrumprintHash, int], b: Mapping[DrumprintHash, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    inter = 0.0
    union = 0.0
    for key in keys:
        av = float(a.get(key, 0))
        bv = float(b.get(key, 0))
        inter += min(av, bv)
        union += max(av, bv)
    if union <= 1e-12:
        return 0.0
    return _clip01(inter / union)


def _merge_bags(bags: Sequence[DrumprintBag], start_idx: int, end_idx: int) -> DrumprintBag:
    start = max(0, int(start_idx))
    end = min(len(bags), max(start, int(end_idx)))
    out: DrumprintBag = Counter()
    for bag in bags[start:end]:
        out.update(bag)
    return out


def _segment_range(analysis: DrumprintAnalysis, start_sec: float, end_sec: float) -> Tuple[int, int]:
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    i0 = int(math.floor(start / max(1e-6, analysis.segment_sec)))
    i1 = int(math.ceil(end / max(1e-6, analysis.segment_sec)))
    return max(0, i0), min(len(analysis.segment_hashes), max(i0, i1))


def _safe_bpm(bpm: Optional[float]) -> float:
    try:
        out = float(bpm if bpm is not None else 128.0)
    except (TypeError, ValueError):
        out = 128.0
    if not math.isfinite(out) or out <= 0.0:
        out = 128.0
    return float(np.clip(out, 60.0, 220.0))


def _percussive_signal(y: np.ndarray) -> np.ndarray:
    try:
        _, percussive = librosa.effects.hpss(np.asarray(y, dtype=np.float32))
        out = np.asarray(percussive, dtype=np.float32)
        if out.size and np.max(np.abs(out)) > 1e-9:
            return out
    except Exception:
        pass
    return np.asarray(y, dtype=np.float32)


def _band_weighted_log_mel(y: np.ndarray, sr: int, cfg: DrumprintConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmax = min(float(cfg.fmax_hz), float(sr) * 0.49)
    if fmax <= float(cfg.fmin_hz) + 20.0:
        raise ValueError(f"Sample rate too low for drumprint bands: sr={sr}")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=int(sr),
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        n_mels=int(cfg.n_mels),
        fmin=float(cfg.fmin_hz),
        fmax=float(fmax),
        power=2.0,
    )
    if mel.size == 0 or mel.shape[1] == 0:
        raise ValueError("Empty mel spectrogram")

    log_mel = librosa.power_to_db(mel + 1e-12, ref=np.max)
    baseline = np.percentile(log_mel, 55.0, axis=1, keepdims=True)
    spec = np.maximum(0.0, log_mel - baseline)

    freqs = librosa.mel_frequencies(n_mels=int(cfg.n_mels), fmin=float(cfg.fmin_hz), fmax=float(fmax))
    weights = np.ones_like(freqs, dtype=np.float64)
    weights[(freqs >= 40.0) & (freqs <= 180.0)] = 1.20
    weights[(freqs > 180.0) & (freqs <= 3000.0)] = 1.05
    weights[(freqs > 3000.0) & (freqs <= fmax)] = 0.90
    spec = spec * weights[:, None]

    times = librosa.frames_to_time(np.arange(spec.shape[1]), sr=int(sr), hop_length=int(cfg.hop_length))
    return spec.astype(np.float64, copy=False), np.asarray(freqs, dtype=np.float64), np.asarray(times, dtype=np.float64)


def _find_peaks(spec: np.ndarray, times: np.ndarray, cfg: DrumprintConfig) -> List[Peak]:
    if spec.size == 0 or times.size == 0:
        return []

    size = (
        max(1, int(cfg.peak_neighborhood_freq_bins) * 2 + 1),
        max(1, int(cfg.peak_neighborhood_time_frames) * 2 + 1),
    )
    local_max = ndimage.maximum_filter(spec, size=size, mode="nearest")
    threshold = float(np.percentile(spec, float(cfg.peak_percentile)))
    mask = (spec >= local_max) & (spec > max(1e-9, threshold))
    if not np.any(mask):
        return []

    hop_sec = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    frames_per_window = max(1, int(round(float(cfg.peak_window_sec) / max(1e-6, hop_sec))))
    peaks: List[Peak] = []
    for start in range(0, spec.shape[1], frames_per_window):
        end = min(spec.shape[1], start + frames_per_window)
        freq_idx, frame_rel = np.where(mask[:, start:end])
        if freq_idx.size == 0:
            continue
        frame_idx = frame_rel + start
        magnitudes = spec[freq_idx, frame_idx]
        order = np.argsort(magnitudes)[::-1][: max(1, int(cfg.top_peaks_per_window))]
        for pos in order:
            peaks.append(
                Peak(
                    time=float(times[int(frame_idx[pos])]),
                    freq_bin=int(freq_idx[pos]),
                    magnitude=float(magnitudes[pos]),
                )
            )
    peaks.sort(key=lambda peak: (peak.time, peak.freq_bin))
    return peaks


def _hash_peak_pairs(peaks: Sequence[Peak], cfg: DrumprintConfig) -> List[HashEvent]:
    events: List[HashEvent] = []
    if len(peaks) < 2:
        return events

    max_dt = float(cfg.max_pair_delta_sec)
    min_dt = float(cfg.min_pair_delta_sec)
    dt_bin_sec = max(1e-3, float(cfg.pair_delta_bin_sec))
    fanout = max(1, int(cfg.fanout))

    for i, anchor in enumerate(peaks):
        targets: List[Peak] = []
        j = i + 1
        while j < len(peaks):
            delta = float(peaks[j].time - anchor.time)
            if delta > max_dt:
                break
            if delta >= min_dt:
                targets.append(peaks[j])
            j += 1
        if not targets:
            continue
        targets.sort(key=lambda peak: peak.magnitude, reverse=True)
        for target in targets[:fanout]:
            delta_t = float(target.time - anchor.time)
            delta_bin = int(round(delta_t / dt_bin_sec))
            key = (int(anchor.freq_bin), int(target.freq_bin), int(delta_bin))
            events.append(
                HashEvent(
                    anchor_time=float(anchor.time),
                    key=key,
                    magnitude=float((anchor.magnitude + target.magnitude) * 0.5),
                )
            )
    events.sort(key=lambda event: event.anchor_time)
    return events


def _segment_hashes(duration_sec: float, segment_sec: float, events: Sequence[HashEvent]) -> Tuple[np.ndarray, List[DrumprintBag], np.ndarray]:
    segment_sec = max(0.1, float(segment_sec))
    n_segments = max(1, int(math.ceil(max(0.0, float(duration_sec)) / segment_sec)) + 1)
    starts = np.arange(n_segments, dtype=np.float64) * segment_sec
    bags: List[DrumprintBag] = [Counter() for _ in range(n_segments)]
    density_raw = np.zeros(n_segments, dtype=np.float64)

    for event in events:
        idx = int(math.floor(max(0.0, float(event.anchor_time)) / segment_sec))
        if 0 <= idx < n_segments:
            bags[idx].update([event.key])
            density_raw[idx] += 1.0

    return starts, bags, _normalize_curve(density_raw)


def _self_similarity_curves(segment_hashes: Sequence[DrumprintBag], segments_per_bar: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(segment_hashes)
    boundary = np.zeros(n, dtype=np.float64)
    novelty = np.zeros(n, dtype=np.float64)
    flank = max(2, int(round(2.0 * max(1, segments_per_bar))))

    for idx in range(n):
        pre = _merge_bags(segment_hashes, idx - flank, idx)
        post = _merge_bags(segment_hashes, idx, idx + flank)
        diff = 1.0 - _weighted_jaccard(pre, post)
        if pre and post:
            boundary[idx] = _clip01(diff)
            novelty[idx] = _clip01(diff)

    return novelty, boundary


def _low_band_curve(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    mask = (freqs >= 40.0) & (freqs <= 180.0)
    if not np.any(mask):
        mask[: max(1, len(freqs) // 8)] = True
    low = np.mean(spec[mask, :], axis=0)
    if low.size >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        low = np.convolve(low, kernel, mode="same")
    return _normalize_curve(low)


def _low_peak_times(low_curve: np.ndarray, times: np.ndarray, beat_sec: float) -> np.ndarray:
    if low_curve.size < 3 or times.size < 3:
        return np.asarray([], dtype=np.float64)
    hop = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    distance = max(1, int(round(0.45 * float(beat_sec) / max(1e-6, hop))))
    height = max(0.35, float(np.percentile(low_curve, 70.0)))
    peaks, _ = signal.find_peaks(low_curve, distance=distance, height=height)
    return np.asarray(times[peaks], dtype=np.float64)


def build_drumprint_analysis(
    y: np.ndarray,
    sr: int,
    bpm: Optional[float],
    cfg: Optional[DrumprintConfig] = None,
) -> DrumprintAnalysis:
    cfg = cfg or DrumprintConfig()
    bpm_value = _safe_bpm(bpm)
    beat_sec = 60.0 / bpm_value
    bar_sec = 4.0 * beat_sec
    segment_sec = max(0.25, bar_sec / 2.0)
    duration_sec = float(len(y) / float(sr)) if sr else 0.0
    if duration_sec <= 0.0:
        raise ValueError("Empty audio")

    percussive = _percussive_signal(y)
    spec, freqs, times = _band_weighted_log_mel(percussive, int(sr), cfg)
    peaks = _find_peaks(spec, times, cfg)
    hash_events = _hash_peak_pairs(peaks, cfg)
    segment_starts, segment_hashes, segment_density = _segment_hashes(duration_sec, segment_sec, hash_events)
    segments_per_bar = max(1, int(round(bar_sec / max(1e-6, segment_sec))))
    novelty_curve, boundary_curve = _self_similarity_curves(segment_hashes, segments_per_bar)
    low_curve = _low_band_curve(spec, freqs)
    low_peaks = _low_peak_times(low_curve, times, beat_sec)

    return DrumprintAnalysis(
        duration_sec=duration_sec,
        bpm=bpm_value,
        beat_sec=beat_sec,
        bar_sec=bar_sec,
        segment_sec=segment_sec,
        times=times,
        segment_starts=segment_starts,
        segment_hashes=segment_hashes,
        segment_density=segment_density,
        novelty_curve=novelty_curve,
        boundary_curve=boundary_curve,
        low_curve=low_curve,
        low_peak_times=low_peaks,
        peaks=peaks,
        hash_events=hash_events,
    )


def _mean_density(analysis: DrumprintAnalysis, start_sec: float, end_sec: float) -> float:
    i0, i1 = _segment_range(analysis, start_sec, end_sec)
    if i1 <= i0:
        return 0.0
    return _clip01(float(np.mean(analysis.segment_density[i0:i1])))


def _aligned_segment_index(analysis: DrumprintAnalysis, timestamp: float) -> int:
    t = max(0.0, float(timestamp))
    segment = max(1e-6, analysis.segment_sec)
    idx = int(math.floor(t / segment))
    next_start = float(idx + 1) * segment
    tolerance = min(0.16, max(0.08, 0.22 * analysis.beat_sec))
    if (next_start - t) <= tolerance:
        idx += 1
    return idx


def _boundary_at(analysis: DrumprintAnalysis, timestamp: float) -> float:
    idx = _aligned_segment_index(analysis, timestamp)
    if idx < 0 or idx >= len(analysis.boundary_curve):
        return 0.0
    return _clip01(float(analysis.boundary_curve[idx]))


def _post_stability(analysis: DrumprintAnalysis, timestamp: float, post_end: float) -> float:
    i0, i1 = _segment_range(analysis, timestamp, post_end)
    if i1 <= i0 + 1:
        return 0.0
    segments_per_bar = max(1, int(round(analysis.bar_sec / max(1e-6, analysis.segment_sec))))

    sims: List[float] = []
    for idx in range(i0, i1 - segments_per_bar):
        sim = _weighted_jaccard(analysis.segment_hashes[idx], analysis.segment_hashes[idx + segments_per_bar])
        if analysis.segment_hashes[idx] and analysis.segment_hashes[idx + segments_per_bar]:
            sims.append(sim)

    first = _merge_bags(analysis.segment_hashes, i0, min(i1, i0 + int(round(4.0 * segments_per_bar))))
    second = _merge_bags(analysis.segment_hashes, i0 + int(round(4.0 * segments_per_bar)), min(i1, i0 + int(round(8.0 * segments_per_bar))))
    half_repeat = _weighted_jaccard(first, second)
    local_repeat = float(np.mean(sims)) if sims else 0.0
    if not sims and not (first and second):
        return 0.0
    return _clip01((0.65 * local_repeat) + (0.35 * half_repeat))


def _kick_repeat_score(analysis: DrumprintAnalysis, timestamp: float, post_end: float) -> float:
    start = max(0.0, float(timestamp))
    end = min(float(post_end), analysis.duration_sec)
    if end <= start + analysis.beat_sec:
        return 0.0

    expected_beats = max(1, int(math.floor((end - start) / max(1e-6, analysis.beat_sec))))
    peaks = analysis.low_peak_times[(analysis.low_peak_times >= start) & (analysis.low_peak_times <= end)]
    if peaks.size == 0:
        occupancy = 0.0
    else:
        tolerance = min(0.16, 0.22 * analysis.beat_sec)
        slots = set()
        for peak in peaks:
            slot = int(round((float(peak) - start) / max(1e-6, analysis.beat_sec)))
            if 0 <= slot <= expected_beats:
                beat_time = start + (slot * analysis.beat_sec)
                if abs(float(peak) - beat_time) <= tolerance:
                    slots.add(slot)
        occupancy = _clip01(len(slots) / float(max(1, expected_beats)))

    i0 = int(np.searchsorted(analysis.times, start, side="left"))
    i1 = int(np.searchsorted(analysis.times, end, side="right"))
    x = analysis.low_curve[i0:i1]
    autocorr_score = 0.0
    if x.size >= 8 and analysis.times.size > 1:
        x = x - float(np.mean(x))
        if float(np.std(x)) > 1e-8:
            corr = signal.correlate(x, x, mode="full")
            corr = corr[corr.size // 2 :]
            hop = float(np.median(np.diff(analysis.times)))
            beat_lag = max(1, int(round(analysis.beat_sec / max(1e-6, hop))))
            search = corr[max(1, beat_lag - 2) : min(len(corr), beat_lag + 3)]
            if search.size and float(corr[0]) > 1e-9:
                autocorr_score = _clip01(float(np.max(search) / corr[0]))

    return _clip01((0.55 * occupancy) + (0.45 * autocorr_score))


def _later_match_score(analysis: DrumprintAnalysis, timestamp: float) -> float:
    query_len = max(analysis.bar_sec, float(DrumprintConfig().later_match_bars) * analysis.bar_sec)
    query_start = float(timestamp)
    query_end = min(analysis.duration_sec, query_start + query_len)
    if query_end <= query_start + analysis.bar_sec:
        return 0.0

    q0, q1 = _segment_range(analysis, query_start, query_end)
    query = _merge_bags(analysis.segment_hashes, q0, q1)
    if not query:
        return 0.0

    step = max(analysis.segment_sec, analysis.bar_sec / 2.0)
    later_start = query_end
    latest = analysis.duration_sec - (query_end - query_start)
    if latest <= later_start:
        return 0.0

    best = 0.0
    cursor = later_start
    checks = 0
    while cursor <= latest and checks < 512:
        i0, i1 = _segment_range(analysis, cursor, cursor + (query_end - query_start))
        candidate = _merge_bags(analysis.segment_hashes, i0, i1)
        best = max(best, _weighted_jaccard(query, candidate))
        cursor += step
        checks += 1
    return _clip01(best)


def score_candidate_drumprint(
    analysis: DrumprintAnalysis,
    timestamp: float,
    *,
    transient_strength: float = 0.0,
) -> Dict[str, object]:
    t = max(0.0, float(timestamp))
    bars = max(1.0, float(DrumprintConfig().analysis_bars))
    pre_start = max(0.0, t - (bars * analysis.bar_sec))
    post_end = min(analysis.duration_sec, t + (bars * analysis.bar_sec))
    post_early_end = min(analysis.duration_sec, t + (2.0 * analysis.bar_sec))

    pre0, pre1 = _segment_range(analysis, pre_start, t)
    post0, post1 = _segment_range(analysis, t, post_end)
    pre_bag = _merge_bags(analysis.segment_hashes, pre0, pre1)
    post_bag = _merge_bags(analysis.segment_hashes, post0, post1)

    fingerprint_density = _mean_density(analysis, t, post_end)
    fingerprint_novelty = _clip01(1.0 - _weighted_jaccard(pre_bag, post_bag)) if pre_bag and post_bag else 0.0
    post_stability = _post_stability(analysis, t, post_end)
    kick_repeat = _kick_repeat_score(analysis, t, post_end)
    boundary_sample = _boundary_at(analysis, t)
    boundary_score = _clip01((0.60 * boundary_sample) + (0.40 * fingerprint_novelty * (0.5 + 0.5 * post_stability)))
    later_match = _later_match_score(analysis, t)

    short_density = _mean_density(analysis, max(0.0, t - 0.25), min(analysis.duration_sec, t + 1.0))
    early_density = _mean_density(analysis, t, post_early_end)
    candidate_segment = _aligned_segment_index(analysis, t)
    if 0 <= candidate_segment < len(analysis.segment_density):
        candidate_segment_density = float(analysis.segment_density[candidate_segment])
    else:
        candidate_segment_density = 0.0
    early0, early1 = _segment_range(analysis, t, post_early_end)
    early_max_density = float(np.max(analysis.segment_density[early0:early1])) if early1 > early0 else candidate_segment_density
    delayed_start_penalty = _clip01(max(0.0, early_max_density - candidate_segment_density) / 0.65)
    attack_level = _clip01(max(float(transient_strength), short_density))
    weak_sustain = _clip01((0.48 - fingerprint_density) / 0.48)
    weak_early = _clip01((0.40 - early_density) / 0.40)
    weak_stability = _clip01((0.38 - post_stability) / 0.38)
    fake_hit_penalty = _clip01(
        ((0.45 + 0.55 * attack_level) * ((0.40 * weak_sustain) + (0.20 * weak_early) + (0.20 * weak_stability)))
        + (0.35 * delayed_start_penalty * (0.35 + 0.65 * attack_level))
    )

    pattern_score = _clip01(
        (0.18 * fingerprint_density)
        + (0.18 * fingerprint_novelty)
        + (0.22 * post_stability)
        + (0.15 * kick_repeat)
        + (0.17 * boundary_score)
        + (0.10 * later_match)
        - (0.10 * fake_hit_penalty)
    )

    return {
        "enabled": True,
        "status": "ok",
        "fingerprint_density": float(fingerprint_density),
        "fingerprint_novelty": float(fingerprint_novelty),
        "post_drop_pattern_stability": float(post_stability),
        "kick_pattern_repeat_score": float(kick_repeat),
        "self_similarity_boundary_score": float(boundary_score),
        "later_drop_match_score": float(later_match),
        "fake_hit_penalty": float(fake_hit_penalty),
        "drumprint_pattern_score": float(pattern_score),
    }
