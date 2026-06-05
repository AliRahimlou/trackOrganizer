from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rhythm_engine.clicktrack import build_click_signal, render_click_track
from rhythm_engine.types import RhythmEstimate


def test_build_click_signal_marks_downbeats_louder() -> None:
    y = build_click_signal(
        duration_sec=2.0,
        sample_rate=1000,
        beats=(0.5, 1.0, 1.5),
        downbeats=(1.0,),
    )

    beat_peak = float(np.max(np.abs(y[500:560])))
    downbeat_peak = float(np.max(np.abs(y[1000:1080])))
    assert downbeat_peak > beat_peak * 1.5


def test_render_click_track_writes_overlay_wav(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    out = tmp_path / "click.wav"
    wavfile.write(audio, 8000, np.zeros(16000, dtype=np.float32))
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.5, 1.0, 1.5),
        downbeats=(1.0,),
        sample_rate=8000,
        confidence=1.0,
    )

    written = render_click_track(str(audio), estimate, str(out), overlay=True, sample_rate=8000)
    sr, y = wavfile.read(written)

    assert sr == 8000
    assert out.exists()
    assert float(np.max(np.abs(y))) > 0.1
