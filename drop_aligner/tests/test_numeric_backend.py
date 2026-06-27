from __future__ import annotations

import numpy as np

import numeric_backend


def test_moving_average_matches_numpy_when_cpu_forced(monkeypatch) -> None:
    monkeypatch.setenv("TRACKORGANIZER_ARRAY_BACKEND", "numpy")
    numeric_backend.reset_array_backend_cache()
    x = np.asarray([0.0, 1.0, 3.0, 7.0, 11.0], dtype=np.float64)

    actual = numeric_backend.moving_average(x, 3)
    expected = np.convolve(x, np.ones(3, dtype=np.float64) / 3.0, mode="same")

    assert np.allclose(actual, expected)
    assert numeric_backend.array_backend_status(x.size)["active"] == "numpy"


def test_windowed_means_match_existing_cpu_formulas(monkeypatch) -> None:
    monkeypatch.setenv("TRACKORGANIZER_ARRAY_BACKEND", "numpy")
    numeric_backend.reset_array_backend_cache()
    x = np.asarray([2.0, 4.0, 8.0, 16.0], dtype=np.float64)

    assert np.allclose(numeric_backend.trailing_mean(x, 2), np.asarray([0.0, 2.0, 3.0, 6.0]))
    assert np.allclose(numeric_backend.forward_mean(x, 2), np.asarray([3.0, 6.0, 12.0, 16.0]))


def test_forced_cupy_without_install_falls_back_to_numpy(monkeypatch) -> None:
    monkeypatch.setenv("TRACKORGANIZER_ARRAY_BACKEND", "cupy")
    numeric_backend.reset_array_backend_cache()
    monkeypatch.setattr(numeric_backend, "_load_cupy", lambda: (None, "cupy_unavailable:test"))
    x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    normalized = numeric_backend.percentile_normalize(x)
    status = numeric_backend.array_backend_status(x.size)

    assert normalized.shape == x.shape
    assert status["active"] == "numpy"
    assert str(status["reason"]).startswith(("cupy_unavailable", "cuda_unavailable"))
