from __future__ import annotations

from typing import Any, Dict

import pytest

import drop_aligner.als_anchor as als_anchor
import drop_aligner.beatgrid as beatgrid
import drop_aligner.detector as detector


class _FakeFeatures:
    duration_sec = 180.0


def _patch_stack(monkeypatch, *, local_one, old_conf, new_conf, local_conf=0.5, local_score=0.4):
    monkeypatch.setattr(detector, "extract_features", lambda *_a, **_k: _FakeFeatures())
    monkeypatch.setattr(
        beatgrid,
        "find_visual_drop_drum_downbeat",
        lambda *_a, **_k: {"time_sec": local_one, "confidence": local_conf, "score": local_score},
    )
    monkeypatch.setattr(beatgrid, "_pattern_arrays", lambda *_a, **_k: {"times": None})

    def fake_score(_features, *, bpm, bar_zero, arrays=None) -> Dict[str, float]:
        import math

        bar = 4.0 * 60.0 / bpm
        new_phase = math.fmod(local_one, bar)
        near_new = abs(bar_zero - new_phase) < 1e-6
        return {"confidence": new_conf if near_new else old_conf, "score": 0.0}

    monkeypatch.setattr(beatgrid, "_score_drum_pattern_bar_zero", fake_score)


def _attempt(**overrides) -> Any:
    kwargs = dict(
        marker=56.402,
        bpm_value=120.0,
        grid_zero=0.25,
        beat_sec=0.5,
        bar_sec=2.0,
        max_after_sec=0.090,
        sample_tolerance_sec=1.0 / 44100.0,
    )
    kwargs.update(overrides)
    return als_anchor._attempt_local_grid_phase_recovery("/tmp/fake.flac", **kwargs)


def test_recovery_accepts_better_global_phase(monkeypatch) -> None:
    _patch_stack(monkeypatch, local_one=56.40, old_conf=0.20, new_conf=0.55)
    result = _attempt()
    assert result is not None
    assert result["grid_downbeat_sec"] == pytest.approx(56.40)
    assert result["evidence"]["pattern_margin"] == pytest.approx(0.35)


def test_recovery_micro_drift_accepts_impact_near_global_one(monkeypatch) -> None:
    # Global grid one at 56.25 (grid_zero 0.25 + 28 bars * 2.0); local impact
    # 30ms later with both phases explaining the track: anchor the impact.
    _patch_stack(monkeypatch, local_one=56.28, old_conf=0.90, new_conf=0.88, local_conf=0.95)
    result = _attempt(marker=56.30)
    assert result is not None
    assert result["evidence"]["mode"] == "micro_drift"
    assert result["grid_downbeat_sec"] == pytest.approx(56.28)


def test_recovery_micro_drift_refuses_low_local_confidence(monkeypatch) -> None:
    _patch_stack(monkeypatch, local_one=56.28, old_conf=0.90, new_conf=0.88, local_conf=0.40)
    assert _attempt(marker=56.30) is None


def test_recovery_refuses_when_neither_branch_qualifies(monkeypatch) -> None:
    # Same-phase margin too small AND drift too large for the micro branch
    # (local one 56.40 is 150ms from the nearest global one at 56.25).
    _patch_stack(monkeypatch, local_one=56.40, old_conf=0.54, new_conf=0.55)
    assert _attempt() is None


def test_recovery_refuses_weak_new_phase(monkeypatch) -> None:
    _patch_stack(monkeypatch, local_one=56.40, old_conf=0.10, new_conf=0.30)
    assert _attempt() is None


def test_recovery_refuses_marker_far_from_local_one(monkeypatch) -> None:
    # Local one is 300ms (>0.5 beat at 120bpm) after the marker: not the same event.
    _patch_stack(monkeypatch, local_one=56.702, old_conf=0.10, new_conf=0.80)
    assert _attempt() is None


def test_recovery_allows_early_eye_mark_within_half_beat(monkeypatch) -> None:
    # Marker 73ms before the local one: same event, Stage B re-anchors on PCM.
    _patch_stack(monkeypatch, local_one=56.475, old_conf=0.20, new_conf=0.60)
    result = _attempt()
    assert result is not None
    assert result["evidence"]["mode"] == "phase_error"
    assert result["grid_downbeat_sec"] == 56.475


def test_recovery_disabled_by_env_flag(monkeypatch) -> None:
    _patch_stack(monkeypatch, local_one=56.40, old_conf=0.20, new_conf=0.55)
    monkeypatch.setattr(als_anchor, "LOCAL_PHASE_RECOVERY_ENABLED", False)
    assert _attempt() is None
