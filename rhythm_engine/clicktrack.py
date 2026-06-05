from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import wavfile

from .types import RhythmEstimate


def _sine_click(sr: int, *, frequency: float, duration_sec: float, amplitude: float) -> np.ndarray:
    n = max(1, int(round(float(duration_sec) * float(sr))))
    env = np.exp(-np.linspace(0.0, 8.5, n))
    wave = np.sin(2.0 * np.pi * float(frequency) * np.arange(n) / float(sr))
    return (wave * env * float(amplitude)).astype(np.float32)


def build_click_signal(
    *,
    duration_sec: float,
    sample_rate: int,
    beats: Sequence[float],
    downbeats: Sequence[float] = (),
    beat_gain: float = 0.34,
    downbeat_gain: float = 0.72,
) -> np.ndarray:
    sr = int(sample_rate)
    y = np.zeros(max(1, int(round(float(duration_sec) * float(sr)))), dtype=np.float32)
    downbeat_set = {round(float(t), 4) for t in downbeats}
    beat_click = _sine_click(sr, frequency=1320.0, duration_sec=0.035, amplitude=float(beat_gain))
    downbeat_click = _sine_click(sr, frequency=2200.0, duration_sec=0.055, amplitude=float(downbeat_gain))

    for time_sec in beats:
        click = downbeat_click if round(float(time_sec), 4) in downbeat_set else beat_click
        start = int(round(float(time_sec) * float(sr)))
        if start < 0 or start >= len(y):
            continue
        end = min(len(y), start + len(click))
        y[start:end] += click[: end - start]
    return np.clip(y, -1.0, 1.0)


def render_click_track(
    audio_path: str,
    estimate: RhythmEstimate,
    output_path: str,
    *,
    overlay: bool = True,
    gain: float = 0.80,
    sample_rate: int | None = None,
) -> str:
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError(f"librosa is required to render click tracks: {exc}") from exc

    sr = int(sample_rate or estimate.sample_rate or 44100)
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    duration = float(len(y) / float(sr))
    clicks = build_click_signal(
        duration_sec=duration,
        sample_rate=int(sr),
        beats=estimate.beats,
        downbeats=estimate.downbeats,
    )
    out = (y * float(gain)) + clicks if overlay else clicks
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sr), np.asarray(out, dtype=np.float32))
    return str(path)


def render_hypothesis_click_tracks(
    audio_path: str,
    estimates: Iterable[RhythmEstimate],
    output_dir: str,
    *,
    limit: int = 6,
    overlay: bool = True,
) -> list[str]:
    out: list[str] = []
    root = Path(output_dir).expanduser()
    for idx, estimate in enumerate(list(estimates)[: max(0, int(limit))], start=1):
        safe_provider = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in estimate.provider)
        path = root / f"{idx:02d}_{safe_provider}_click.wav"
        out.append(render_click_track(audio_path, estimate, str(path), overlay=overlay))
    return out
