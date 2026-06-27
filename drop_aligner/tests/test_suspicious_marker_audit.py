from __future__ import annotations

from pathlib import Path

import audit_visual_first_suspicious_markers as audit


class FakeWaveformCache:
    def __init__(self, _cache_dir: Path) -> None:
        pass

    def info(self, _audio_path: str) -> dict:
        return {"duration": 120.0}

    def tile(self, _audio_path: str, *, start_sec: float, end_sec: float, width: int, force_mode: str = "auto") -> dict:
        if float(start_sec) == 0.0:
            density = [0.0] * 60
            density[10:15] = [0.92] * 5
            density[30:36] = [0.84] * 6
            placeable = [False] * 60
            placeable[10] = True
            placeable[30] = True
            return {
                "start_sec": 0.0,
                "end_sec": 120.0,
                "boom_body_density": density,
                "boom_body_mask": [value > 0.3 for value in density],
                "boom_placeable_mask": placeable,
                "boom_placeable_count": 2,
            }
        density = [0.0] * 1200
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "boom_body_density": density,
            "boom_body_mask": [False] * 1200,
            "boom_placeable_mask": [False] * 1200,
            "boom_placeable_count": 0,
        }


class FakeQuantizedLocalWaveformCache(FakeWaveformCache):
    def tile(self, _audio_path: str, *, start_sec: float, end_sec: float, width: int, force_mode: str = "auto") -> dict:
        if float(start_sec) == 0.0:
            return super().tile(_audio_path, start_sec=start_sec, end_sec=end_sec, width=width, force_mode=force_mode)
        density = [0.03] * 1200
        placeable = [False] * 1200
        placeable[604] = True
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "boom_body_density": density,
            "boom_body_mask": [False] * 1200,
            "boom_placeable_mask": placeable,
            "boom_placeable_count": 1,
        }


class FakeShortSpikeAlternativeWaveformCache(FakeWaveformCache):
    def tile(self, _audio_path: str, *, start_sec: float, end_sec: float, width: int, force_mode: str = "auto") -> dict:
        if float(start_sec) == 0.0:
            density = [0.0] * 600
            body = [False] * 600
            placeable = [False] * 600
            density[100:110] = [0.24] * 10
            body[100:110] = [True] * 10
            placeable[100] = True
            density[400:402] = [1.0, 0.96]
            placeable[400] = True
            return {
                "start_sec": 0.0,
                "end_sec": 120.0,
                "boom_body_density": density,
                "boom_body_mask": body,
                "boom_placeable_mask": placeable,
                "boom_placeable_count": 2,
            }
        density = [0.20] * 1200
        body = [False] * 1200
        body[598:604] = [True] * 6
        placeable = [False] * 1200
        placeable[600] = True
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "boom_body_density": density,
            "boom_body_mask": body,
            "boom_placeable_mask": placeable,
            "boom_placeable_count": 1,
        }


class FakeSustainedAlternativeWaveformCache(FakeShortSpikeAlternativeWaveformCache):
    def tile(self, _audio_path: str, *, start_sec: float, end_sec: float, width: int, force_mode: str = "auto") -> dict:
        if float(start_sec) != 0.0:
            return super().tile(_audio_path, start_sec=start_sec, end_sec=end_sec, width=width, force_mode=force_mode)
        density = [0.0] * 600
        body = [False] * 600
        placeable = [False] * 600
        density[100:110] = [0.24] * 10
        body[100:110] = [True] * 10
        placeable[100] = True
        density[400:430] = [0.92] * 30
        body[400:430] = [True] * 30
        placeable[400] = True
        return {
            "start_sec": 0.0,
            "end_sec": 120.0,
            "boom_body_density": density,
            "boom_body_mask": body,
            "boom_placeable_mask": placeable,
            "boom_placeable_count": 2,
        }


def _row() -> dict:
    return {
        "track": {"folder": "demo"},
        "drums_path": "/tmp/drums_120_1A_5-demo.flac",
        "marker": 60.0,
        "selected_by": "visual_gui_boom_front_edge_contract",
        "boom_proof": {
            "passes": True,
            "marker_sec": 60.0,
            "nearest": {"edge_time": 60.0, "offset_sec": 0.0, "abs_offset_sec": 0.0},
            "nearest_profile": {"profile_score": 0.66, "passes_profile": True},
            "reasons": [],
        },
        "gui_mask_proof": {"passes": True, "reasons": [], "marker_signal_present": True},
    }


def test_suspicious_audit_allows_exact_boom_proof_relief(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeWaveformCache)
    monkeypatch.setattr(audit, "visual_gui_mask_proof", lambda *args, **kwargs: {"passes": False, "reasons": ["raw"]})
    monkeypatch.setattr(
        audit,
        "accept_gui_boom_mask_with_front_edge_proof",
        lambda gui, proof, **_kwargs: {
            **gui,
            "passes": True,
            "accepted_by_boom_front_edge_proof": True,
            "marker_signal_present": True,
        },
    )

    result = audit._audit_row({"index": 1, "row": _row(), "cache_dir": "/tmp/cache", "width": 60})

    assert result["fail_flags"] == ""
    assert "local_mask_relied_on_exact_boom_proof" in result["warn_flags"]


def test_suspicious_audit_suppresses_small_exact_proof_quantization_warning(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeQuantizedLocalWaveformCache)
    monkeypatch.setattr(audit, "visual_gui_mask_proof", lambda *args, **kwargs: {"passes": False, "reasons": ["raw"]})
    monkeypatch.setattr(
        audit,
        "accept_gui_boom_mask_with_front_edge_proof",
        lambda gui, proof, **_kwargs: {
            **gui,
            "passes": True,
            "accepted_by_boom_front_edge_proof": True,
            "marker_signal_present": True,
            "nearest_placeable_offset_sec": 0.020,
            "front_edge_offset_sec": 0.018,
        },
    )

    result = audit._audit_row({"index": 1, "row": _row(), "cache_dir": "/tmp/cache", "width": 60})

    assert result["fail_flags"] == ""
    assert "local_mask_relied_on_exact_boom_proof" not in result["warn_flags"]


def test_suspicious_audit_uses_strong_selected_proof_for_later_comparison(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeWaveformCache)
    monkeypatch.setattr(audit, "visual_gui_mask_proof", lambda *args, **kwargs: {"passes": False, "reasons": ["raw"]})
    monkeypatch.setattr(
        audit,
        "accept_gui_boom_mask_with_front_edge_proof",
        lambda gui, proof, **_kwargs: {
            **gui,
            "passes": True,
            "accepted_by_boom_front_edge_proof": True,
            "marker_signal_present": True,
        },
    )
    row = _row()
    row["marker"] = 20.0
    row["boom_proof"] = {
        "passes": True,
        "marker_sec": 20.0,
        "nearest": {
            "edge_time": 20.0,
            "offset_sec": 0.0,
            "abs_offset_sec": 0.0,
            "candidate_score": 0.95,
        },
        "nearest_profile": {
            "profile_score": 0.93,
            "passes_profile": True,
            "metrics": {
                "score": 0.92,
                "darkness": 0.94,
                "body_score": 0.91,
                "post8_height": 0.90,
            },
        },
        "reasons": [],
    }

    result = audit._audit_row({"index": 1, "row": row, "cache_dir": "/tmp/cache", "width": 60})

    assert result["fail_flags"] == ""
    assert float(result["selected_score"]) >= 0.90
    assert float(result["selected_gui_score"]) < 0.01
    assert "later_much_stronger_boom_front_edge" not in result["warn_flags"]


def test_suspicious_audit_ignores_peak_only_later_spike(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeShortSpikeAlternativeWaveformCache)
    monkeypatch.setattr(
        audit,
        "visual_gui_mask_proof",
        lambda *args, **kwargs: {"passes": True, "reasons": [], "marker_signal_present": True},
    )
    monkeypatch.setattr(audit, "accept_gui_boom_mask_with_front_edge_proof", lambda gui, proof, **_kwargs: gui)
    row = _row()
    row["marker"] = 20.0
    row["boom_proof"] = {
        "passes": True,
        "marker_sec": 20.0,
        "nearest": {"edge_time": 20.0, "offset_sec": 0.0, "abs_offset_sec": 0.0},
        "nearest_profile": {"profile_score": 0.20, "passes_profile": True},
        "reasons": [],
    }

    result = audit._audit_row({"index": 1, "row": row, "cache_dir": "/tmp/cache", "width": 600})

    assert result["fail_flags"] == ""
    assert "later_much_stronger_boom_front_edge" not in result["warn_flags"]


def test_suspicious_audit_warns_on_sustained_later_body(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeSustainedAlternativeWaveformCache)
    monkeypatch.setattr(
        audit,
        "visual_gui_mask_proof",
        lambda *args, **kwargs: {"passes": True, "reasons": [], "marker_signal_present": True},
    )
    monkeypatch.setattr(audit, "accept_gui_boom_mask_with_front_edge_proof", lambda gui, proof, **_kwargs: gui)
    row = _row()
    row["marker"] = 20.0
    row["boom_proof"] = {
        "passes": True,
        "marker_sec": 20.0,
        "nearest": {"edge_time": 20.0, "offset_sec": 0.0, "abs_offset_sec": 0.0},
        "nearest_profile": {"profile_score": 0.20, "passes_profile": True},
        "reasons": [],
    }

    result = audit._audit_row({"index": 1, "row": row, "cache_dir": "/tmp/cache", "width": 600})

    assert result["fail_flags"] == ""
    assert "later_much_stronger_boom_front_edge" in result["warn_flags"]
    assert float(result["best_later_sustain_score"]) > 0.50


def test_suspicious_audit_rejects_unrescued_gui_mask(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeWaveformCache)
    monkeypatch.setattr(audit, "visual_gui_mask_proof", lambda *args, **kwargs: {"passes": False, "reasons": ["raw"]})
    monkeypatch.setattr(audit, "accept_gui_boom_mask_with_front_edge_proof", lambda gui, proof, **_kwargs: gui)

    result = audit._audit_row({"index": 1, "row": _row(), "cache_dir": "/tmp/cache", "width": 60})

    assert "recomputed_gui_mask_not_passing" in result["fail_flags"]
    assert "no_local_placeable_boom_front_edge" in result["fail_flags"]


def test_suspicious_audit_rejects_stale_persisted_boom_proof(monkeypatch) -> None:
    monkeypatch.setattr(audit, "WaveformCache", FakeWaveformCache)
    monkeypatch.setattr(
        audit,
        "visual_gui_mask_proof",
        lambda *args, **kwargs: {"passes": True, "reasons": [], "marker_signal_present": True},
    )
    monkeypatch.setattr(audit, "accept_gui_boom_mask_with_front_edge_proof", lambda gui, proof, **_kwargs: gui)
    row = _row()
    row["boom_proof"] = {
        "passes": True,
        "marker_sec": 60.421,
        "nearest": {"edge_time": 60.0, "offset_sec": 0.421, "abs_offset_sec": 0.421},
        "reasons": [],
    }

    result = audit._audit_row({"index": 1, "row": row, "cache_dir": "/tmp/cache", "width": 60})

    assert "persisted_boom_proof_not_fresh" in result["fail_flags"]
