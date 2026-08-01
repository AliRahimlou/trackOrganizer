from __future__ import annotations

import numpy as np

from drop_aligner.beatgrid import find_visual_drop_drum_downbeat, resolve_beatgrid
from drop_aligner.detector import FeatureBundle


def _pulse(times: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return float(amplitude) * np.exp(-0.5 * ((times - float(center)) / float(width)) ** 2)


def _synthetic_drums_bundle(*, bpm: float, downbeat_sec: float, duration_sec: float = 32.0) -> FeatureBundle:
    beat_sec = 60.0 / float(bpm)
    bar_sec = 4.0 * beat_sec
    times = np.arange(0.0, float(duration_sec), 0.010, dtype=np.float64)
    low = np.zeros_like(times)
    low_jump = np.zeros_like(times)
    rms = np.zeros_like(times) + 0.020
    onset = np.zeros_like(times)
    attack = np.zeros_like(times)
    flux = np.zeros_like(times)

    for bar_start in np.arange(float(downbeat_sec), float(duration_sec), bar_sec):
        # Kick/low-end defines the musical one; beat 3 is present but weaker.
        for hit_time, amplitude in ((bar_start, 0.85), (bar_start + (2.0 * beat_sec), 0.45)):
            low += _pulse(times, hit_time, 0.028, amplitude)
            low_jump += _pulse(times, hit_time, 0.016, amplitude)
            rms += _pulse(times, hit_time, 0.055, amplitude * 0.55)
            onset += _pulse(times, hit_time, 0.018, amplitude * 0.45)
            attack += _pulse(times, hit_time, 0.018, amplitude * 0.45)

        # The backbeat is intentionally louder than the kick, which used to
        # tempt the structural grid into choosing beat 2 as bar zero.
        for hit_time in (bar_start + beat_sec, bar_start + (3.0 * beat_sec)):
            flux += _pulse(times, hit_time, 0.018, 1.00)
            onset += _pulse(times, hit_time, 0.015, 0.90)
            attack += _pulse(times, hit_time, 0.015, 0.92)
            rms += _pulse(times, hit_time, 0.035, 0.35)

    for hit_time in np.arange(float(downbeat_sec) + beat_sec, 8.0, bar_sec):
        attack += _pulse(times, hit_time, 0.012, 0.70)
        flux += _pulse(times, hit_time, 0.012, 0.70)

    for array in (low, low_jump, rms, onset, attack, flux):
        array[:] = np.clip(array, 0.0, 1.0)

    return FeatureBundle(
        audio_path="synthetic-drums-bass",
        y=np.zeros(1, dtype=np.float32),
        sr=44100,
        duration_sec=float(duration_sec),
        bpm=float(bpm),
        beat_sec=float(beat_sec),
        frame_times=times,
        onset=onset,
        low_energy=low,
        low_jump_curve=low_jump,
        rms=rms,
        contrast=np.zeros_like(times),
        spectral_flux=flux,
        combined_attack=attack,
    )


def test_resolve_beatgrid_uses_kick_downbeat_over_louder_snare_backbeat() -> None:
    features = _synthetic_drums_bundle(bpm=120.0, downbeat_sec=0.370)

    grid = resolve_beatgrid({"drums": features}, bpm=120.0)

    np.testing.assert_allclose([grid.bar_zero_sec], [0.370], atol=0.020)
    np.testing.assert_allclose([grid.first_low_downbeat_sec], [0.370], atol=0.020)
    assert grid.source_role == "drums:dj_drum_pattern"
    assert grid.downbeat_confidence >= 0.80


def test_visual_drop_drum_downbeat_refinement_pulls_back_from_snare_to_drop_one() -> None:
    features = _synthetic_drums_bundle(bpm=120.0, downbeat_sec=8.000, duration_sec=24.0)

    refined = find_visual_drop_drum_downbeat(
        features,
        visual_time_sec=8.520,
        current_marker_sec=8.520,
        bpm=120.0,
        clock_zero_sec=0.0,
    )

    assert refined is not None
    np.testing.assert_allclose([refined["time_sec"]], [8.000], atol=0.030)
    assert refined["kick_peak"] >= 0.60
    assert refined["backbeat_snare"] >= 0.30
    assert refined["confidence"] >= 0.30
