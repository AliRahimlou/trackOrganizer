from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw


CACHE_VERSION = 6
RAW_SAMPLE_LIMIT = 96_000
RAW_SAMPLES_PER_PIXEL = 16.0
MAX_TILE_WIDTH = 48_000
PEAK_FULL_READ_LIMIT = 12_000_000
PEAK_STREAM_BLOCK_FRAMES = 1_000_000
PNG_VERSION = 5
RMS_MAXIMIZER_TARGET = 0.78
RMS_MAXIMIZER_CEILING = 0.985
RMS_MAXIMIZER_KNEE = 0.72
RMS_MAXIMIZER_MIN_GAIN = 0.85
RMS_MAXIMIZER_MAX_GAIN = 7.5


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clip01(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _audio_key(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    raw = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _json_safe(values: np.ndarray, digits: int = 6) -> List[float]:
    if values.size == 0:
        return []
    rounded = np.round(np.asarray(values, dtype=np.float32), digits)
    return [float(v) for v in rounded]


def _mono_preserve_peak(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    indices = np.argmax(np.abs(arr), axis=1)
    return arr[np.arange(arr.shape[0]), indices].astype(np.float32, copy=False)


def _frame_min_max(data: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.min(arr)), float(np.max(arr))


def _frame_rms(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float32), dtype=np.float64)))


def _rms_bins_from_mono(data: np.ndarray, bins: int) -> np.ndarray:
    mono = np.asarray(data, dtype=np.float32)
    safe_bins = max(1, int(bins))
    rms = np.zeros(safe_bins, dtype=np.float32)
    if mono.size == 0:
        return rms
    edges = np.floor(np.linspace(0, int(mono.size), safe_bins + 1)).astype(np.int64)
    for idx in range(safe_bins):
        seg = mono[int(edges[idx]) : int(edges[idx + 1])]
        rms[idx] = _frame_rms(seg)
    return rms


def _rms_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([max(0.0, float(v)) for v in values], dtype=np.float32)
    if arr.size == 0:
        return {
            "rms_peak": 1.0,
            "rms_percentile_10": 0.0,
            "rms_percentile_95": 1.0,
            "rms_percentile_99": 1.0,
            "rms_visual_ceiling": 1.0,
            "visual_maximizer_target": RMS_MAXIMIZER_TARGET,
            "visual_maximizer_makeup_gain": 1.0,
            "visual_maximizer_ceiling": RMS_MAXIMIZER_CEILING,
            "visual_maximizer_knee": RMS_MAXIMIZER_KNEE,
        }
    peak = float(np.max(arr))
    p10 = float(np.percentile(arr, 10.0))
    p95 = float(np.percentile(arr, 95.0))
    p99 = float(np.percentile(arr, 99.0))
    ceiling = max(1e-5, min(max(p95 * 1.35, p99 * 0.82), max(peak, 1e-5)))
    reference = max(p95, p99 * 0.74, peak * 0.58, 1e-5)
    makeup = float(np.clip(RMS_MAXIMIZER_TARGET / reference, RMS_MAXIMIZER_MIN_GAIN, RMS_MAXIMIZER_MAX_GAIN))
    return {
        "rms_peak": float(peak if peak > 1e-9 else 1.0),
        "rms_percentile_10": float(p10),
        "rms_percentile_95": float(p95 if p95 > 1e-9 else peak if peak > 1e-9 else 1.0),
        "rms_percentile_99": float(p99 if p99 > 1e-9 else peak if peak > 1e-9 else 1.0),
        "rms_visual_ceiling": float(ceiling),
        "visual_maximizer_target": RMS_MAXIMIZER_TARGET,
        "visual_maximizer_makeup_gain": makeup,
        "visual_maximizer_ceiling": RMS_MAXIMIZER_CEILING,
        "visual_maximizer_knee": RMS_MAXIMIZER_KNEE,
    }


def _resolution_level(samples_per_pixel: float) -> int:
    spp = max(1.0, float(samples_per_pixel))
    return int(2 ** round(math.log2(spp))) if spp > 1.0 else 1


class WaveformCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def info(self, audio_path: str) -> Dict[str, Any]:
        path = Path(audio_path).expanduser()
        with sf.SoundFile(str(path)) as fh:
            sample_rate = int(fh.samplerate)
            frames = int(fh.frames)
            channels = int(fh.channels)
        return {
            "sample_rate": sample_rate,
            "total_samples": frames,
            "duration": float(frames / sample_rate) if sample_rate else 0.0,
            "channels": channels,
            "audio_key": _audio_key(path),
        }

    def _cache_path(
        self,
        path: Path,
        *,
        start_sample: int,
        end_sample: int,
        width: int,
        mode: str,
        level: int,
    ) -> Path:
        key = _audio_key(path)
        shard = self.cache_dir / key[:2] / key[2:4]
        shard.mkdir(parents=True, exist_ok=True)
        raw = f"v{CACHE_VERSION}|{key}|{start_sample}|{end_sample}|{width}|{mode}|{level}"
        name = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return shard / f"{name}.json.gz"

    def _read_frames(self, path: Path, start_sample: int, end_sample: int) -> np.ndarray:
        frames = max(0, int(end_sample) - int(start_sample))
        if frames <= 0:
            return np.zeros(0, dtype=np.float32)
        with sf.SoundFile(str(path)) as fh:
            safe_start = max(0, min(int(start_sample), int(fh.frames)))
            fh.seek(safe_start)
            data = fh.read(max(0, min(frames, int(fh.frames) - safe_start)), dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32)

    def _raw_tile(self, path: Path, start_sample: int, end_sample: int, width: int) -> Dict[str, Any]:
        data = self._read_frames(path, start_sample, end_sample)
        mono = _mono_preserve_peak(data)
        rms = _rms_bins_from_mono(mono, width)
        return {
            "mode": "samples",
            "sample_start": int(start_sample),
            "samples": _json_safe(np.clip(mono, -1.0, 1.0), digits=7),
            "rms": _json_safe(np.clip(rms, 0.0, 1.0), digits=7),
        }

    def _peak_tile(self, path: Path, start_sample: int, end_sample: int, width: int) -> Dict[str, Any]:
        total = max(0, int(end_sample) - int(start_sample))
        bins = max(1, int(width))
        mins = np.zeros(bins, dtype=np.float32)
        maxs = np.zeros(bins, dtype=np.float32)
        rms = np.zeros(bins, dtype=np.float32)
        if total <= 0:
            return {"mode": "peaks", "mins": _json_safe(mins), "maxs": _json_safe(maxs), "rms": _json_safe(rms)}

        edges = np.floor(np.linspace(0, total, bins + 1)).astype(np.int64)
        if total <= PEAK_FULL_READ_LIMIT:
            data = self._read_frames(path, start_sample, end_sample)
            data = _mono_preserve_peak(data)
            for idx in range(bins):
                seg = data[int(edges[idx]) : int(edges[idx + 1])]
                mins[idx], maxs[idx] = _frame_min_max(seg)
                rms[idx] = _frame_rms(seg)
        else:
            mins.fill(np.inf)
            maxs.fill(-np.inf)
            sums = np.zeros(bins, dtype=np.float64)
            counts = np.zeros(bins, dtype=np.int64)
            absolute_edges = int(start_sample) + edges
            with sf.SoundFile(str(path)) as fh:
                frame_count = int(fh.frames)
                cursor = max(0, min(int(start_sample), frame_count))
                final = max(cursor, min(int(end_sample), frame_count))
                fh.seek(cursor)
                while cursor < final:
                    count = min(PEAK_STREAM_BLOCK_FRAMES, final - cursor)
                    data = fh.read(count, dtype="float32", always_2d=True)
                    mono = _mono_preserve_peak(data)
                    if mono.size == 0:
                        break
                    block_start = cursor
                    block_end = cursor + int(mono.size)
                    first_bin = max(0, min(bins - 1, int(np.searchsorted(absolute_edges, block_start, side="right") - 1)))
                    last_bin = max(0, min(bins - 1, int(np.searchsorted(absolute_edges, block_end - 1, side="right") - 1)))
                    for idx in range(first_bin, last_bin + 1):
                        local_start = max(block_start, int(absolute_edges[idx])) - block_start
                        local_end = min(block_end, int(absolute_edges[idx + 1])) - block_start
                        if local_end <= local_start:
                            continue
                        seg = mono[int(local_start) : int(local_end)]
                        lo, hi = _frame_min_max(seg)
                        mins[idx] = min(float(mins[idx]), lo)
                        maxs[idx] = max(float(maxs[idx]), hi)
                        sums[idx] += float(np.sum(np.square(seg, dtype=np.float32), dtype=np.float64))
                        counts[idx] += int(seg.size)
                    cursor = block_end
            mins = np.where(np.isfinite(mins), mins, 0.0).astype(np.float32, copy=False)
            maxs = np.where(np.isfinite(maxs), maxs, 0.0).astype(np.float32, copy=False)
            rms = np.sqrt(np.divide(sums, np.maximum(counts, 1), dtype=np.float64)).astype(np.float32, copy=False)
        return {
            "mode": "peaks",
            "mins": _json_safe(np.clip(mins, -1.0, 1.0)),
            "maxs": _json_safe(np.clip(maxs, -1.0, 1.0)),
            "rms": _json_safe(np.clip(rms, 0.0, 1.0), digits=7),
        }

    def tile(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        width: int,
        force_mode: str = "auto",
    ) -> Dict[str, Any]:
        path = Path(audio_path).expanduser()
        info = self.info(str(path))
        sr = int(info["sample_rate"])
        total_samples = int(info["total_samples"])
        start = max(0, min(total_samples, int(round(float(start_sec) * sr))))
        end = max(start + 1, min(total_samples, int(round(float(end_sec) * sr))))
        safe_width = max(1, min(MAX_TILE_WIDTH, int(width or 1)))
        samples_per_pixel = max(1.0, (end - start) / safe_width)
        use_raw = (
            force_mode == "samples"
            or (
                force_mode == "auto"
                and (end - start) <= RAW_SAMPLE_LIMIT
                and samples_per_pixel <= RAW_SAMPLES_PER_PIXEL
            )
        )
        mode = "samples" if use_raw else "peaks"
        level = 1 if use_raw else _resolution_level(samples_per_pixel)
        cache_path = self._cache_path(path, start_sample=start, end_sample=end, width=safe_width, mode=mode, level=level)
        if cache_path.exists():
            try:
                with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
                    cached = json.load(fh)
                if isinstance(cached, dict):
                    cached["cache_hit"] = True
                    return cached
            except Exception:
                cache_path.unlink(missing_ok=True)

        payload = self._raw_tile(path, start, end, safe_width) if use_raw else self._peak_tile(path, start, end, safe_width)
        amplitude_values: List[float] = []
        if payload["mode"] == "samples":
            amplitude_values = list(payload.get("samples", []))
        else:
            amplitude_values = list(payload.get("mins", [])) + list(payload.get("maxs", []))
        peak = max((abs(float(v)) for v in amplitude_values), default=1.0)
        rms_stats = _rms_stats(payload.get("rms", []))
        out = {
            "ok": True,
            "audio_key": info["audio_key"],
            "sample_rate": sr,
            "total_samples": total_samples,
            "duration": float(info["duration"]),
            "start_sec": float(start / sr),
            "end_sec": float(end / sr),
            "start_sample": int(start),
            "end_sample": int(end),
            "width": int(safe_width),
            "samples_per_pixel": float(samples_per_pixel),
            "resolution_level": int(level),
            "amplitude_peak": float(peak if peak > 1e-9 else 1.0),
            **rms_stats,
            "cache_hit": False,
            **payload,
        }
        with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=True)
        return out

    def render_png(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        width: int,
        height: int,
        markers: Sequence[Mapping[str, Any]] = (),
    ) -> bytes:
        width = max(400, min(8000, int(width or 2400)))
        height = max(180, min(4000, int(height or 720)))
        tile = self.tile(audio_path, start_sec=start_sec, end_sec=end_sec, width=width, force_mode="auto")
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        mid = height // 2
        draw.line([(0, mid), (width, mid)], fill="#d7dce2", width=max(1, height // 360))
        gain = 0.92 / max(float(tile.get("amplitude_peak", 1.0) or 1.0), 1e-9)

        rms_values = [max(0.0, float(v)) for v in tile.get("rms", [])]
        if len(rms_values) >= 2:
            def rms_amp(value: float) -> float:
                makeup = float(tile.get("visual_maximizer_makeup_gain", 1.0) or 1.0)
                ceiling = max(0.5, min(1.0, float(tile.get("visual_maximizer_ceiling", RMS_MAXIMIZER_CEILING) or RMS_MAXIMIZER_CEILING)))
                knee = max(0.2, min(ceiling - 0.02, float(tile.get("visual_maximizer_knee", RMS_MAXIMIZER_KNEE) or RMS_MAXIMIZER_KNEE)))
                lifted = max(0.0, float(value)) * makeup
                if lifted <= knee:
                    limited = lifted
                else:
                    limiter_range = max(0.001, ceiling - knee)
                    limited = knee + limiter_range * math.tanh((lifted - knee) / limiter_range)
                return float(np.clip((limited / ceiling) ** 0.82, 0.0, 1.0))

            tile_span = max(1e-9, float(tile.get("end_sec", end_sec)) - float(tile.get("start_sec", start_sec)))
            bin_span = tile_span / max(1, len(rms_values))
            target = max(0.006, min(0.120, tile_span / 500.0))
            radius = max(1, int(round(target / max(bin_span, 1e-9) / 2.0)))
            if radius > 1:
                source = np.asarray(rms_values, dtype=np.float64)
                prefix = np.concatenate([np.asarray([0.0]), np.cumsum(source)])
                smoothed: List[float] = []
                for index, value in enumerate(source):
                    i0 = max(0, index - radius)
                    i1 = min(len(source), index + radius + 1)
                    averaged = float((prefix[i1] - prefix[i0]) / max(1, i1 - i0))
                    smoothed.append(max(averaged, float(value) * 0.18))
                rms_values = smoothed

            span = max(1, len(rms_values) - 1)
            upper = [
                (
                    int(round(i * (width - 1) / span)),
                    int(round(mid - rms_amp(value) * (height * 0.42))),
                )
                for i, value in enumerate(rms_values)
            ]
            lower = [
                (
                    int(round(i * (width - 1) / span)),
                    int(round(mid + rms_amp(value) * (height * 0.42))),
                )
                for i, value in reversed(list(enumerate(rms_values)))
            ]
            draw.polygon(upper + lower, fill="#e8eef3")
            draw.line(upper, fill="#263746", width=max(1, height // 360))
            draw.line(list(reversed(lower)), fill="#263746", width=max(1, height // 360))
        elif tile.get("mode") == "samples":
            samples = [float(v) for v in tile.get("samples", [])]
            if len(samples) >= 2:
                span = max(1, len(samples) - 1)
                points = [
                    (int(round(i * (width - 1) / span)), int(round(mid - _clip01(sample * gain) * (height * 0.46))))
                    for i, sample in enumerate(samples)
                ]
                draw.line(points, fill="#263746", width=max(1, height // 320))
                if width / span >= 8:
                    radius = max(2, height // 180)
                    for x, y in points:
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#263746")
        else:
            mins = [float(v) for v in tile.get("mins", [])]
            maxs = [float(v) for v in tile.get("maxs", [])]
            for x, (lo, hi) in enumerate(zip(mins, maxs)):
                y1 = int(round(mid - _clip01(hi * gain) * (height * 0.46)))
                y2 = int(round(mid - _clip01(lo * gain) * (height * 0.46)))
                draw.line([(x, y1), (x, y2)], fill="#263746")

        start = float(tile.get("start_sec", start_sec))
        end = float(tile.get("end_sec", end_sec))
        span_sec = max(1e-9, end - start)
        colors = {
            "ai": "#c2352b",
            "candidate": "#d88716",
            "micro": "#005f73",
            "knee": "#008f7a",
            "asd": "#7a4cff",
            "attack": "#6a3d9a",
            "zero": "#111820",
            "user": "#177245",
        }
        for marker in markers:
            t = _float_or_none(marker.get("time"))
            if t is None or t < start or t > end:
                continue
            x = int(round((t - start) / span_sec * (width - 1)))
            kind = str(marker.get("kind", "user"))
            color = colors.get(kind, "#111820")
            draw.line([(x, 0), (x, height)], fill=color, width=max(2, width // 1400))
            label = str(marker.get("label") or kind.upper())
            draw.rectangle((x + 4, 6, x + 4 + max(28, len(label) * 8), 26), fill="#ffffff", outline=color)
            draw.text((x + 8, 10), label, fill=color)

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
