#!/usr/bin/env python3

"""
Adapter for harvesting Ableton Live's own transient candidates from `.asd` files.

Live 12 `.asd` files are not officially documented, and the public `AbletonParsing`
library currently supports saved warp markers rather than the onset/transient arrays
we want. This adapter therefore uses a pragmatic extractor tuned for Live 12 files:

- locate onset-related sections such as `OnSets`, `OnsetArray`, and `OnsetEvent`
- search nearby for the longest monotonic `uint32` sample-position run
- treat that run as Ableton's transient candidate list

This is intentionally narrow in scope: it is not a full `.asd` parser. It is a
batch-friendly way to reuse Live's own onset analysis as candidate markers for the
existing first-downbeat chooser.
"""

from __future__ import annotations

import logging
import os
import subprocess
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import soundfile as sf
except Exception:
    sf = None


LOG = logging.getLogger("ableton_analysis_adapter")
ONSET_SEED_TOKENS: Tuple[bytes, ...] = (b"OnSets", b"OnsetArray", b"OnsetEvent")
SDK_WARP_SCHEMA = "track_organizer_ableton_warp_markers_v1"
SDK_WARP_SIDECAR_SUFFIX = ".ableton_warp_markers.json"


@dataclass
class AbletonAnalysisMarkers:
    audio_path: str
    source: str
    candidate_samples: List[int]
    candidate_seconds: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "source": self.source,
            "candidate_samples": list(self.candidate_samples),
            "candidate_seconds": list(self.candidate_seconds),
            "metadata": dict(self.metadata),
        }


def _probe_audio_info(audio_path: str) -> Tuple[int, int]:
    if sf is not None:
        info = sf.info(audio_path)
        return int(info.samplerate), int(info.frames)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,duration_ts",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    vals = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(vals) < 2:
        raise RuntimeError(f"Could not probe sample rate / frame count for {audio_path}")
    return int(float(vals[0])), int(float(vals[1]))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        LOG.debug("Could not read Ableton warp export %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _stable_export_name(audio_path: str) -> str:
    return hashlib.sha1(os.path.abspath(audio_path).encode("utf-8", errors="ignore")).hexdigest()[:20] + ".json"


def _candidate_sdk_export_paths(
    audio_path: str,
    *,
    export_path: Optional[str] = None,
    export_dir: Optional[str] = None,
) -> List[Path]:
    audio_abs = os.path.abspath(audio_path)
    paths: List[Path] = []
    if export_path:
        paths.append(Path(export_path).expanduser())
    paths.append(Path(audio_abs + SDK_WARP_SIDECAR_SUFFIX))

    root = export_dir or os.environ.get("ABLETON_WARP_EXPORT_DIR") or ""
    if root:
        base = Path(root).expanduser()
        stable = _stable_export_name(audio_abs)
        paths.append(base / stable)
        paths.append(base / "track-organizer-warp-markers" / stable)

    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _marker_seconds_from_sdk_export(
    payload: Mapping[str, Any],
    *,
    sample_rate: int,
    frames: int,
) -> Tuple[List[int], List[float], str]:
    raw_markers = payload.get("warpMarkers")
    if not isinstance(raw_markers, list):
        return [], [], "missing"

    sample_times: List[float] = []
    for marker in raw_markers:
        if not isinstance(marker, Mapping):
            continue
        try:
            value = float(marker.get("sampleTime"))
        except (TypeError, ValueError):
            continue
        if value >= 0.0:
            sample_times.append(value)
    if not sample_times:
        return [], [], "missing"

    duration_sec = float(frames) / max(1.0, float(sample_rate))
    max_value = max(sample_times)
    # The beta docs expose `sampleTime` but do not specify units. Live's UI/ALS
    # uses seconds for sample time; if values are far beyond the track duration
    # but inside frame-count range, treat them as sample frames.
    if max_value > duration_sec + max(4.0, duration_sec * 0.25) and max_value <= float(frames) * 1.10:
        unit = "sample_frames"
        seconds = [value / float(sample_rate) for value in sample_times]
    else:
        unit = "seconds"
        seconds = sample_times

    cleaned: List[float] = []
    prev: Optional[float] = None
    for sec in sorted(seconds):
        if sec < 0.0 or sec > duration_sec + 1.0:
            continue
        if prev is None or abs(sec - prev) > 1e-4:
            cleaned.append(float(sec))
            prev = float(sec)

    samples = [int(round(sec * float(sample_rate))) for sec in cleaned]
    return samples, cleaned, unit


def extract_ableton_warp_markers(
    audio_path: str,
    *,
    export_path: Optional[str] = None,
    export_dir: Optional[str] = None,
) -> Optional[AbletonAnalysisMarkers]:
    """Load Ableton SDK-exported warp markers for an audio file.

    The public Extensions SDK exposes `AudioClip.warpMarkers`. The extension in
    `tools/ableton-warp-probe` writes those markers to a sidecar JSON file that
    this loader can use as official Live evidence before falling back to binary
    `.asd` scraping.
    """

    audio_abs = os.path.abspath(audio_path)
    try:
        sample_rate, frames = _probe_audio_info(audio_abs)
    except Exception as exc:
        LOG.debug("Could not probe audio info for %s: %s", audio_abs, exc)
        return None

    for candidate in _candidate_sdk_export_paths(audio_abs, export_path=export_path, export_dir=export_dir):
        if not candidate.exists():
            continue
        payload = _read_json(candidate)
        if not payload:
            continue
        samples, seconds, unit = _marker_seconds_from_sdk_export(payload, sample_rate=sample_rate, frames=frames)
        if not seconds:
            continue
        return AbletonAnalysisMarkers(
            audio_path=audio_abs,
            source="sdk_warp_markers",
            candidate_samples=samples,
            candidate_seconds=seconds,
            metadata={
                "export_path": str(candidate),
                "parser": "ableton_extensions_sdk_warp_markers_v1",
                "schema": str(payload.get("schema") or ""),
                "expected_schema": SDK_WARP_SCHEMA,
                "sample_time_unit": unit,
                "candidate_count": int(len(seconds)),
                "sample_rate": int(sample_rate),
                "frames": int(frames),
                "first_seconds": float(seconds[0]),
                "last_seconds": float(seconds[-1]),
                "clip": payload.get("clip") if isinstance(payload.get("clip"), Mapping) else {},
            },
        )
    return None


def _collect_seed_offsets(raw: bytes) -> List[int]:
    out: List[int] = []
    for token in ONSET_SEED_TOKENS:
        start = 0
        while True:
            idx = raw.find(token, start)
            if idx < 0:
                break
            out.append(int(idx))
            start = idx + 1
    return sorted(set(out))


def _score_run(
    *,
    run_offset: int,
    values: Sequence[int],
    seed_offset: int,
    max_sample: int,
) -> Tuple[int, float, int, int]:
    positives = [v for v in values if v > 0]
    if not positives:
        return (0, float("inf"), 0, 0)
    span = positives[-1] - positives[0]
    coverage = float(span) / max(1.0, float(max_sample))
    seed_dist = abs(run_offset - seed_offset)
    return (
        len(positives),
        seed_dist,
        int(round(coverage * 1000.0)),
        -positives[0],
    )


def _find_best_uint32_run(
    raw: bytes,
    *,
    max_sample: int,
    search_before: int,
    search_after: int,
    min_run: int,
) -> Optional[Tuple[int, List[int], int]]:
    seeds = _collect_seed_offsets(raw)
    if not seeds:
        return None

    best_run: Optional[Tuple[int, List[int], int]] = None
    best_score: Optional[Tuple[int, float, int, int]] = None
    for seed in seeds:
        lo = max(0, seed - int(search_before))
        hi = min(len(raw) - 4, seed + int(search_after))
        off = lo - (lo % 4)
        while off < hi:
            first = int.from_bytes(raw[off : off + 4], "little", signed=False)
            if first > max_sample:
                off += 4
                continue

            vals: List[int] = []
            cur = off
            while cur < hi:
                value = int.from_bytes(raw[cur : cur + 4], "little", signed=False)
                if vals and (value < vals[-1] or value > max_sample):
                    break
                vals.append(value)
                cur += 4

            positives = [v for v in vals if v > 0]
            if len(positives) >= int(min_run):
                score = _score_run(run_offset=off, values=vals, seed_offset=seed, max_sample=max_sample)
                if best_score is None or score > best_score:
                    best_run = (off, positives, seed)
                    best_score = score
                off = cur
            else:
                off += 4

    return best_run


def extract_ableton_onset_markers(
    audio_path: str,
    *,
    asd_path: Optional[str] = None,
    search_before_bytes: int = 4096,
    search_after_bytes: int = 131072,
    min_run: int = 32,
) -> Optional[AbletonAnalysisMarkers]:
    audio_abs = os.path.abspath(audio_path)

    sdk_markers = extract_ableton_warp_markers(audio_abs)
    if sdk_markers is not None:
        return sdk_markers

    asd_abs = os.path.abspath(asd_path or (audio_path + ".asd"))
    if not os.path.exists(asd_abs):
        return None

    try:
        sample_rate, frames = _probe_audio_info(audio_abs)
    except Exception as exc:
        LOG.debug("Could not probe audio info for %s: %s", audio_abs, exc)
        return None

    try:
        raw = open(asd_abs, "rb").read()
    except Exception as exc:
        LOG.debug("Could not read Ableton analysis file %s: %s", asd_abs, exc)
        return None

    run = _find_best_uint32_run(
        raw,
        max_sample=int(frames),
        search_before=int(search_before_bytes),
        search_after=int(search_after_bytes),
        min_run=int(min_run),
    )
    if run is None:
        return None

    run_offset, samples, seed = run
    deduped: List[int] = []
    prev = None
    for sample in samples:
        if prev is None or sample != prev:
            deduped.append(int(sample))
            prev = int(sample)
    if not deduped:
        return None

    seconds = [float(sample) / float(sample_rate) for sample in deduped]
    return AbletonAnalysisMarkers(
        audio_path=audio_abs,
        source="asd",
        candidate_samples=deduped,
        candidate_seconds=seconds,
        metadata={
            "asd_path": asd_abs,
            "parser": "live12_onsets_monotonic_uint32_v1",
            "seed_tokens": [token.decode("ascii", errors="ignore") for token in ONSET_SEED_TOKENS],
            "seed_offset": int(seed),
            "run_offset": int(run_offset),
            "candidate_count": int(len(deduped)),
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "search_before_bytes": int(search_before_bytes),
            "search_after_bytes": int(search_after_bytes),
            "min_run": int(min_run),
            "first_seconds": float(seconds[0]),
            "last_seconds": float(seconds[-1]),
        },
    )
