"""Whole-track energy sectioning of the drums stem.

Implements Ali's rule for locating the first drop by eye: read the drums
waveform as sections (intro / buildup / drop / breakdown / outro) and put
1.1.1 at the FIRST occurrence of the BIGGEST class of energy boost.  An early
small lift is not the drop when a later section hits materially harder; the
first section entry whose boost reaches the track's top boost class is.

This module is deliberately independent of the visual-first candidate cascade
so it can act as a global prior over it: it only answers "which moment of the
track does the drop section start at", in seconds on the raw drums timeline.
Sample-accurate placement inside that section stays the job of the existing
Stage A/Stage B machinery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ANALYSIS_SR = 16000
LOW_BAND_HZ = 150.0
FRAME_SEC = 0.10
# A boost event must lift sustained energy by at least this factor over the
# preceding context to count as a section entry at all.
MIN_BOOST_RATIO = 1.6
# Events whose post-entry energy reaches this fraction of the strongest
# event's post-entry energy belong to the top boost class.
TOP_CLASS_FRACTION = 0.85
# Refine the coarse event time to the steepest short-window rise nearby.
EDGE_REFINE_SPAN_SEC = 4.0
EDGE_RISE_WINDOW_SEC = 0.8
PRE_WINDOW_SEC = 3.2
POST_WINDOW_SEC = 3.2
MIN_EVENT_GAP_SEC = 4.0
# Ignore boosts inside the first moments of the file: a track that starts in
# a drop is handled by the opening-drop path, not by section comparison.
MIN_EVENT_TIME_SEC = 1.0


@dataclass
class BoostEvent:
    time_sec: float
    boost_ratio: float
    post_energy: float
    pre_energy: float
    top_class: bool = False
    refined_time_sec: Optional[float] = None


@dataclass
class EnergySections:
    ok: bool
    reason: str = ""
    duration_sec: float = 0.0
    events: List[BoostEvent] = field(default_factory=list)
    chosen_time_sec: Optional[float] = None
    chosen_event_index: Optional[int] = None
    max_post_energy: float = 0.0


def _mono_low_envelope(path: str) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != ANALYSIS_SR:
        step = sr / ANALYSIS_SR
        idx = (np.arange(int(mono.size / step)) * step).astype(np.int64)
        mono = mono[np.minimum(idx, mono.size - 1)]
        sr = ANALYSIS_SR
    sos = butter(4, LOW_BAND_HZ, btype="lowpass", fs=sr, output="sos")
    low = sosfilt(sos, mono)
    frame = max(1, int(FRAME_SEC * sr))
    n = mono.size // frame
    if n == 0:
        return np.zeros(0, dtype=np.float64), float(mono.size) / sr
    low_e = np.sqrt(np.mean(np.square(low[: n * frame].reshape(n, frame)), axis=1))
    broad_e = np.sqrt(np.mean(np.square(mono[: n * frame].reshape(n, frame)), axis=1))
    # Weight the low band strongly: the drop is where the bass and drums hit
    # together, and the drums stem's low band is the cleanest witness.
    env = (0.7 * low_e + 0.3 * broad_e).astype(np.float64)
    return env, float(mono.size) / sr


def analyze_energy_sections(drums_path: str) -> EnergySections:
    path = Path(drums_path)
    if not path.is_file():
        return EnergySections(ok=False, reason="drums_missing")
    try:
        env, duration = _mono_low_envelope(str(path))
    except Exception as exc:
        return EnergySections(ok=False, reason=f"read_error:{exc.__class__.__name__}")
    if env.size < 20:
        return EnergySections(ok=False, reason="too_short", duration_sec=duration)

    pre_n = max(1, int(PRE_WINDOW_SEC / FRAME_SEC))
    post_n = max(1, int(POST_WINDOW_SEC / FRAME_SEC))
    floor = max(1e-6, float(np.percentile(env, 90)) * 0.02)

    events: List[BoostEvent] = []
    last_time = -1e9
    for i in range(pre_n, env.size - post_n):
        pre = float(np.median(env[i - pre_n : i]))
        post = float(np.median(env[i : i + post_n]))
        ratio = post / max(pre, floor)
        if ratio < MIN_BOOST_RATIO or post < floor * 4:
            continue
        t = i * FRAME_SEC
        if t < MIN_EVENT_TIME_SEC:
            continue
        if t - last_time < MIN_EVENT_GAP_SEC:
            # Keep the strongest representative of a contiguous rise.
            if events and post > events[-1].post_energy:
                events[-1] = BoostEvent(t, ratio, post, pre)
                last_time = t
            continue
        events.append(BoostEvent(t, ratio, post, pre))
        last_time = t

    if not events:
        return EnergySections(ok=False, reason="no_boost_events", duration_sec=duration)

    max_post = max(event.post_energy for event in events)
    chosen_index: Optional[int] = None
    for index, event in enumerate(events):
        event.refined_time_sec = _refine_entry_edge(env, event.time_sec)
        event.top_class = event.post_energy >= TOP_CLASS_FRACTION * max_post
        if event.top_class and chosen_index is None:
            chosen_index = index

    chosen = events[chosen_index] if chosen_index is not None else None
    return EnergySections(
        ok=True,
        duration_sec=duration,
        events=events,
        chosen_time_sec=chosen.refined_time_sec if chosen else None,
        chosen_event_index=chosen_index,
        max_post_energy=max_post,
    )


def choose_first_top_boost(events: Sequence[BoostEvent], max_post: float, fraction: float) -> Optional[BoostEvent]:
    """First event whose post-entry energy reaches `fraction` of the maximum."""
    for event in events:
        if event.post_energy >= fraction * max_post:
            return event
    return None


def _refine_entry_edge(env: np.ndarray, coarse_time_sec: float) -> float:
    """Snap a coarse section-entry time to the steepest nearby short rise."""
    span = int(EDGE_REFINE_SPAN_SEC / FRAME_SEC)
    rise = max(2, int(EDGE_RISE_WINDOW_SEC / FRAME_SEC))
    center = int(coarse_time_sec / FRAME_SEC)
    lo = max(rise, center - span)
    hi = min(env.size - rise, center + span)
    if hi <= lo:
        return coarse_time_sec
    best_time = coarse_time_sec
    best_ratio = 0.0
    for i in range(lo, hi):
        before = float(np.mean(env[i - rise : i]))
        after = float(np.mean(env[i : i + rise]))
        ratio = after / max(before, 1e-9)
        if ratio > best_ratio:
            best_ratio = ratio
            best_time = i * FRAME_SEC
    return best_time
