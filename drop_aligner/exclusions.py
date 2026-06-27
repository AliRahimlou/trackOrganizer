from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf


EXCLUDED_DIR_NAMES = frozenset({"notToBeOrganized"})
ACAPELLA_TERMS = frozenset(
    {
        "acapella",
        "a cappella",
        "a capella",
        "a-cappella",
        "a-capella",
        "accapella",
        "vocal only",
        "vocals only",
    }
)
NEAR_EMPTY_DRUMS_PEAK_FLOOR = 0.025
NEAR_EMPTY_DRUMS_WINDOW_RMS_P99_FLOOR = 0.004
NEAR_EMPTY_DRUMS_WINDOW_SIZE = 1024
NEAR_EMPTY_DRUMS_BLOCK_FRAMES = 65_536


def is_excluded_path(value: object, *, excluded_dir_names: Iterable[str] = EXCLUDED_DIR_NAMES) -> bool:
    if value in (None, ""):
        return False
    excluded = {str(name).lower() for name in excluded_dir_names if str(name)}
    if not excluded:
        return False
    try:
        parts = Path(str(value)).expanduser().parts
    except Exception:
        parts = Path(str(value)).parts
    return any(part.lower() in excluded for part in parts)


def row_has_excluded_path(
    row: Mapping[str, object],
    keys: Sequence[str] = ("filename", "track", "audio_path", "output_als", "candidates_json", "debug_png"),
) -> bool:
    return any(is_excluded_path(row.get(key)) for key in keys)


def is_acapella_path(value: object, *, terms: Iterable[str] = ACAPELLA_TERMS) -> bool:
    if value in (None, ""):
        return False
    text = str(value).replace("_", " ").replace("-", " ").lower()
    return any(str(term).lower() in text for term in terms if str(term))


def row_is_acapella(
    row: Mapping[str, object],
    keys: Sequence[str] = ("filename", "track", "audio_path", "drums_path", "output_als", "candidates_json"),
) -> bool:
    values = [row.get(key) for key in keys]
    track = row.get("track")
    if isinstance(track, Mapping):
        values.extend(track.get(key) for key in ("folder", "name", "title", "filename", "src"))
    return any(is_acapella_path(value) for value in values)


def drums_stem_signal_stats(
    audio_path: str | Path,
    *,
    window_size: int = NEAR_EMPTY_DRUMS_WINDOW_SIZE,
    block_frames: int = NEAR_EMPTY_DRUMS_BLOCK_FRAMES,
) -> Dict[str, Any]:
    """Return absolute signal stats used to veto blank/near-empty drums stems."""

    path = Path(audio_path).expanduser()
    safe_window = max(64, int(window_size))
    safe_block = max(safe_window, int(block_frames))
    windows: list[np.ndarray] = []
    peak_abs = 0.0
    square_sum = 0.0
    sample_count = 0
    tail = np.zeros(0, dtype=np.float32)
    with sf.SoundFile(str(path)) as fh:
        sample_rate = int(fh.samplerate)
        frames = int(fh.frames)
        channels = int(fh.channels)
        while True:
            data = fh.read(safe_block, dtype="float32", always_2d=True)
            if len(data) <= 0:
                break
            mono_abs = np.max(np.abs(np.asarray(data, dtype=np.float32)), axis=1).astype(np.float32, copy=False)
            if mono_abs.size <= 0:
                continue
            peak_abs = max(peak_abs, float(np.max(mono_abs)))
            square_sum += float(np.sum(np.square(mono_abs, dtype=np.float32), dtype=np.float64))
            sample_count += int(mono_abs.size)
            mono_abs = np.concatenate([tail, mono_abs]) if tail.size else mono_abs
            full = (int(mono_abs.size) // safe_window) * safe_window
            if full > 0:
                framed = mono_abs[:full].reshape(-1, safe_window)
                windows.append(np.sqrt(np.mean(np.square(framed, dtype=np.float32), axis=1)))
            tail = mono_abs[full:]
    if tail.size:
        windows.append(np.asarray([np.sqrt(np.mean(np.square(tail, dtype=np.float32)))], dtype=np.float32))
    window_rms = np.concatenate(windows) if windows else np.zeros(1, dtype=np.float32)
    rms_mean = float(np.sqrt(square_sum / max(1, sample_count)))
    return {
        "audio_path": str(path),
        "sample_rate": int(sample_rate) if "sample_rate" in locals() else 0,
        "frames": int(frames) if "frames" in locals() else 0,
        "channels": int(channels) if "channels" in locals() else 0,
        "duration_sec": float(frames / sample_rate) if "sample_rate" in locals() and sample_rate else 0.0,
        "peak_abs": float(peak_abs),
        "rms_mean": float(rms_mean),
        "window_rms_p95": float(np.percentile(window_rms, 95.0)),
        "window_rms_p99": float(np.percentile(window_rms, 99.0)),
        "window_rms_max": float(np.max(window_rms)),
        "near_empty_peak_floor": float(NEAR_EMPTY_DRUMS_PEAK_FLOOR),
        "near_empty_window_rms_p99_floor": float(NEAR_EMPTY_DRUMS_WINDOW_RMS_P99_FLOOR),
    }


def is_near_empty_drums_stem(audio_path: str | Path, *, stats: Optional[Mapping[str, Any]] = None) -> bool:
    """Return true when a drums stem is effectively blank for visual drop alignment."""

    measured = dict(stats) if isinstance(stats, Mapping) else drums_stem_signal_stats(audio_path)
    try:
        peak_abs = float(measured.get("peak_abs", 0.0) or 0.0)
        window_rms_p99 = float(measured.get("window_rms_p99", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        peak_abs < NEAR_EMPTY_DRUMS_PEAK_FLOOR
        and window_rms_p99 < NEAR_EMPTY_DRUMS_WINDOW_RMS_P99_FLOOR
    )
