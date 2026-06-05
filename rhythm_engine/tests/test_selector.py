from __future__ import annotations

import numpy as np

from rhythm_engine.selector import score_estimate_against_arrays
from rhythm_engine.types import RhythmEstimate


def _pulse_curve(times: np.ndarray, events: tuple[float, ...], *, width: float = 0.018) -> np.ndarray:
    y = np.zeros_like(times, dtype=np.float64)
    for event in events:
        y += np.exp(-0.5 * ((times - float(event)) / width) ** 2)
    return y


def test_selector_prefers_full_beat_grid_over_half_and_double_time() -> None:
    times = np.linspace(0.0, 8.0, 1601)
    true_beats = tuple(1.0 + (0.5 * idx) for idx in range(12))
    onset = _pulse_curve(times, true_beats)
    low = _pulse_curve(times, true_beats[::2], width=0.022)
    rms = 0.25 + (0.45 * onset)
    flux = onset.copy()
    base = RhythmEstimate(provider="base", beats=true_beats, downbeats=true_beats[::4], bpm=120.0, confidence=0.7)
    half = RhythmEstimate(provider="half", beats=true_beats[::2], downbeats=true_beats[::8], bpm=60.0, confidence=0.9)
    double_beats = tuple(sorted([*true_beats, *(t + 0.25 for t in true_beats[:-1])]))
    double = RhythmEstimate(provider="double", beats=double_beats, downbeats=double_beats[::4], bpm=240.0, confidence=0.9)

    scores = {
        estimate.provider: score_estimate_against_arrays(
            estimate,
            frame_times=times,
            onset=onset,
            low_jump=low,
            rms=rms,
            spectral_flux=flux,
            duration_sec=8.0,
        ).final_score
        for estimate in (base, half, double)
    }

    assert scores["base"] > scores["half"]
    assert scores["base"] > scores["double"]


def test_selector_scores_correct_downbeat_phase_higher() -> None:
    times = np.linspace(0.0, 8.0, 1601)
    beats = tuple(1.0 + (0.5 * idx) for idx in range(12))
    true_downbeats = beats[1::4]
    wrong_downbeats = beats[0::4]
    onset = _pulse_curve(times, beats)
    low = _pulse_curve(times, beats, width=0.025)
    rms = 0.20 + (0.20 * onset)
    flux = _pulse_curve(times, true_downbeats, width=0.030) + (0.35 * onset)
    correct = RhythmEstimate(provider="correct", beats=beats, downbeats=true_downbeats, bpm=120.0, confidence=0.7)
    wrong = RhythmEstimate(provider="wrong", beats=beats, downbeats=wrong_downbeats, bpm=120.0, confidence=0.7)

    correct_score = score_estimate_against_arrays(
        correct,
        frame_times=times,
        onset=onset,
        low_jump=low,
        rms=rms,
        spectral_flux=flux,
        duration_sec=8.0,
    )
    wrong_score = score_estimate_against_arrays(
        wrong,
        frame_times=times,
        onset=onset,
        low_jump=low,
        rms=rms,
        spectral_flux=flux,
        duration_sec=8.0,
    )

    assert correct_score.downbeat_support > wrong_score.downbeat_support
    assert correct_score.final_score > wrong_score.final_score
