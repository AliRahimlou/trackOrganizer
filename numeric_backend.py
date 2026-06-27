from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np


_DISABLE_VALUES = {"0", "false", "no", "off", "numpy", "cpu"}
_FORCE_CUPY_VALUES = {"cupy", "gpu", "cuda"}
_DEFAULT_MIN_GPU_ELEMENTS = 200_000


def _backend_mode() -> str:
    value = os.getenv("TRACKORGANIZER_ARRAY_BACKEND", os.getenv("TRACK_ORGANIZER_ARRAY_BACKEND", "auto"))
    return str(value or "auto").strip().lower()


def _min_gpu_elements() -> int:
    raw = os.getenv("TRACKORGANIZER_CUPY_MIN_ELEMENTS", os.getenv("TRACK_ORGANIZER_CUPY_MIN_ELEMENTS", ""))
    try:
        return max(0, int(raw)) if raw else _DEFAULT_MIN_GPU_ELEMENTS
    except ValueError:
        return _DEFAULT_MIN_GPU_ELEMENTS


@lru_cache(maxsize=1)
def _load_cupy() -> Tuple[Optional[Any], str]:
    mode = _backend_mode()
    if mode in _DISABLE_VALUES:
        return None, "disabled"
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as exc:
        return None, f"cupy_unavailable:{exc.__class__.__name__}"
    try:
        devices = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return None, f"cuda_unavailable:{exc.__class__.__name__}"
    if devices <= 0:
        return None, "cuda_unavailable:no_devices"
    return cp, f"cupy:{devices}_device{'s' if devices != 1 else ''}"


def reset_array_backend_cache() -> None:
    _load_cupy.cache_clear()


def _cupy_for_size(size: int) -> Tuple[Optional[Any], str]:
    cp, reason = _load_cupy()
    if cp is None:
        return None, reason
    mode = _backend_mode()
    if mode in _FORCE_CUPY_VALUES:
        return cp, reason
    if int(size) >= _min_gpu_elements():
        return cp, reason
    return None, f"numpy:below_cupy_threshold:{_min_gpu_elements()}"


def array_backend_status(size: int = 0) -> dict[str, object]:
    cp, reason = _cupy_for_size(int(size))
    return {
        "requested": _backend_mode(),
        "active": "cupy" if cp is not None else "numpy",
        "reason": reason,
        "min_gpu_elements": int(_min_gpu_elements()),
    }


def _gpu_scalar(value: Any, cp: Any) -> float:
    return float(cp.asnumpy(value).item())


def percentile_normalize(
    values: Any,
    *,
    lo_percentile: float = 5.0,
    hi_percentile: float = 95.0,
    dtype: Any = np.float64,
) -> np.ndarray:
    x_np = np.asarray(values, dtype=dtype)
    if x_np.size == 0:
        return x_np
    cp, _ = _cupy_for_size(int(x_np.size))
    if cp is None:
        lo = float(np.percentile(x_np, float(lo_percentile)))
        hi = float(np.percentile(x_np, float(hi_percentile)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            return np.zeros_like(x_np, dtype=dtype)
        return np.clip((x_np - lo) / (hi - lo), 0.0, 1.0)

    x_gpu = cp.asarray(x_np)
    lo = _gpu_scalar(cp.percentile(x_gpu, float(lo_percentile)), cp)
    hi = _gpu_scalar(cp.percentile(x_gpu, float(hi_percentile)), cp)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.zeros_like(x_np, dtype=dtype)
    out = cp.clip((x_gpu - lo) / (hi - lo), 0.0, 1.0)
    return cp.asnumpy(out)


def moving_average(values: Any, frames: int, *, dtype: Any = np.float64) -> np.ndarray:
    safe_frames = max(1, int(frames))
    x_np = np.asarray(values, dtype=dtype)
    if safe_frames <= 1 or x_np.size == 0:
        return x_np.astype(dtype, copy=False)
    cp, _ = _cupy_for_size(int(x_np.size))
    if cp is None:
        kernel = np.ones(safe_frames, dtype=dtype) / float(safe_frames)
        return np.convolve(x_np.astype(dtype, copy=False), kernel, mode="same")

    x_gpu = cp.asarray(x_np)
    kernel = cp.ones(safe_frames, dtype=dtype) / float(safe_frames)
    return cp.asnumpy(cp.convolve(x_gpu.astype(dtype, copy=False), kernel, mode="same"))


def trailing_mean(values: Any, samples: int, *, dtype: Any = np.float64) -> np.ndarray:
    x_np = np.asarray(values, dtype=dtype)
    n = int(x_np.size)
    if n == 0:
        return x_np
    window = max(1, int(samples))
    cp, _ = _cupy_for_size(n)
    if cp is None:
        indices = np.arange(n, dtype=np.int64)
        starts = np.maximum(0, indices - window)
        counts = np.maximum(1, indices - starts)
        csum = np.concatenate([np.asarray([0.0], dtype=dtype), np.cumsum(x_np)])
        return (csum[indices] - csum[starts]) / counts

    x_gpu = cp.asarray(x_np)
    indices = cp.arange(n, dtype=cp.int64)
    starts = cp.maximum(0, indices - window)
    counts = cp.maximum(1, indices - starts)
    csum = cp.concatenate([cp.asarray([0.0], dtype=dtype), cp.cumsum(x_gpu)])
    return cp.asnumpy((csum[indices] - csum[starts]) / counts)


def forward_mean(values: Any, samples: int, *, dtype: Any = np.float64) -> np.ndarray:
    x_np = np.asarray(values, dtype=dtype)
    n = int(x_np.size)
    if n == 0:
        return x_np
    window = max(1, int(samples))
    cp, _ = _cupy_for_size(n)
    if cp is None:
        indices = np.arange(n, dtype=np.int64)
        ends = np.minimum(n, indices + window)
        counts = np.maximum(1, ends - indices)
        csum = np.concatenate([np.asarray([0.0], dtype=dtype), np.cumsum(x_np)])
        return (csum[ends] - csum[indices]) / counts

    x_gpu = cp.asarray(x_np)
    indices = cp.arange(n, dtype=cp.int64)
    ends = cp.minimum(n, indices + window)
    counts = cp.maximum(1, ends - indices)
    csum = cp.concatenate([cp.asarray([0.0], dtype=dtype), cp.cumsum(x_gpu)])
    return cp.asnumpy((csum[ends] - csum[indices]) / counts)
