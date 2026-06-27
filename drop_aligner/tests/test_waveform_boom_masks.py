from __future__ import annotations

import gzip
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from drop_aligner import waveform as waveform_module
from drop_aligner.waveform import (
    WaveformCache,
    accept_gui_boom_mask_with_front_edge_proof,
    enforce_strict_gui_boom_mask_contract,
    gui_boom_mask_proof_for_tile,
    gui_boom_mask_strict_contract_issue,
)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def _dark_pixel_count(payload: bytes) -> int:
    image = Image.open(BytesIO(payload)).convert("RGB")
    pixels = np.asarray(image)
    dark = (pixels[:, :, 0] < 90) & (pixels[:, :, 1] < 105) & (pixels[:, :, 2] < 120)
    return int(np.count_nonzero(dark))


def test_rms_bins_match_reference_loop_for_uneven_edges() -> None:
    mono = np.asarray([0.0, 0.25, -0.50, 1.0, -0.75, 0.125, 0.5], dtype=np.float32)
    bins = 4
    edges = np.floor(np.linspace(0, int(mono.size), bins + 1)).astype(np.int64)
    expected = []
    for idx in range(bins):
        seg = mono[int(edges[idx]) : int(edges[idx + 1])]
        expected.append(float(np.sqrt(np.mean(np.square(seg, dtype=np.float32), dtype=np.float64))) if seg.size else 0.0)

    actual = waveform_module._rms_bins_from_mono(mono, bins)

    assert np.allclose(actual, np.asarray(expected, dtype=np.float32))


def test_moving_average_matches_prefix_loop_at_boundaries() -> None:
    values = [0.0, 1.0, 3.0, 9.0, 27.0]
    radius = 2
    expected = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        expected.append(sum(values[start:end]) / max(1, end - start))

    actual = waveform_module._moving_average(values, radius)

    assert np.allclose(actual, expected)


def test_gzip_json_memory_cache_reuses_payload_without_reparse(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "tile.json.gz"
    with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
        json.dump({"ok": True, "value": 1}, fh)
    waveform_module._GZIP_JSON_MEMORY_CACHE.clear()
    load_count = 0
    original_load = waveform_module.json.load

    def counted_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(waveform_module.json, "load", counted_load)

    first = waveform_module._read_gzip_json_with_memory_cache(cache_path)
    first["value"] = 99
    second = waveform_module._read_gzip_json_with_memory_cache(cache_path)

    assert load_count == 1
    assert second["value"] == 1


def test_boom_masks_hide_silent_tile_from_placement(tmp_path: Path) -> None:
    audio = np.zeros(16000 * 3, dtype=np.float32)
    path = tmp_path / "silent.wav"
    _write_wav(path, audio)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=3.0, width=300)

    assert tile["boom_placeable_count"] == 0
    assert not any(tile["boom_body_mask"])
    assert not any(tile["boom_relevant_mask"])
    assert not any(tile["boom_placeable_mask"])
    assert max(tile["boom_body_density"]) == 0.0


def test_boom_masks_mark_only_hard_body_transition_as_placeable(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    hard_body = 0.75 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    audio = np.concatenate([np.zeros(sr, dtype=np.float32), hard_body, np.zeros(sr, dtype=np.float32)])
    path = tmp_path / "drop.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=3.0, width=300)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert tile["boom_placeable_count"] > 0
    assert any(tile["boom_body_mask"])
    assert any(tile["boom_relevant_mask"])
    assert placeable[0] <= 105
    assert placeable[-1] <= 110
    assert not any(tile["boom_placeable_mask"][:80])
    assert not any(tile["boom_placeable_mask"][115:])

    proof = gui_boom_mask_proof_for_tile(tile, 1.0)
    assert proof["passes"] is True
    assert proof["marker_immediate_body_present"] is True
    assert proof["marker_post_body_occupancy_500ms"] > 0.50
    assert proof["marker_post_density_mean_500ms"] > 0.10


def test_boom_masks_allow_impact_that_leads_into_sustained_body(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    impact = 0.78 * np.sin(2 * np.pi * 90 * t[: int(sr * 0.080)]).astype(np.float32)
    hard_body = 0.72 * np.sin(2 * np.pi * 85 * t[: int(sr * 0.900)]).astype(np.float32)
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[sr : sr + len(impact)] = impact
    body_start = int(sr * 1.52)
    audio[body_start : body_start + len(hard_body)] = hard_body
    path = tmp_path / "impact_then_body.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=3.0, width=600)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert tile["boom_placeable_count"] > 0
    assert any(195 <= index <= 218 for index in placeable)
    assert not any(tile["boom_placeable_mask"][:160])


def test_boom_masks_allow_sparse_pulse_train_front_edge(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 5, dtype=np.float32)
    pulse_len = int(sr * 0.045)
    t = np.arange(pulse_len, dtype=np.float32) / sr
    pulse = 0.68 * np.sin(2 * np.pi * 72 * t).astype(np.float32)
    for start_sec in (1.0, 1.34, 1.68, 2.02):
        start = int(sr * start_sec)
        audio[start : start + pulse_len] = pulse
    path = tmp_path / "sparse_pulse_train.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=5.0, width=1000)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert tile["boom_placeable_count"] > 0
    assert any(195 <= index <= 215 for index in placeable)
    assert not any(tile["boom_placeable_mask"][:170])
    assert not any(tile["boom_placeable_mask"][250:320])


def test_boom_masks_allow_lower_relative_sparse_pulse_train_front_edge(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 6, dtype=np.float32)
    pulse_len = int(sr * 0.028)
    t = np.arange(pulse_len, dtype=np.float32) / sr
    pulse = 0.34 * np.sin(2 * np.pi * 74 * t).astype(np.float32)
    for start_sec in (2.10, 2.44, 2.78, 3.12, 3.46):
        start = int(sr * start_sec)
        audio[start : start + pulse_len] = pulse
    path = tmp_path / "lower_sparse_pulse_train.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=6.0, width=1200)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert tile["boom_placeable_count"] > 0
    assert any(410 <= index <= 430 for index in placeable)
    assert not any(tile["boom_placeable_mask"][:380])
    assert not any(tile["boom_placeable_mask"][470:560])


def test_boom_masks_keep_sparse_pulse_front_edge_when_zoomed_in(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 12, dtype=np.float32)
    pulse_len = int(sr * 0.004)
    t = np.arange(pulse_len, dtype=np.float32) / sr
    pulse = 0.84 * np.sin(2 * np.pi * 74 * t).astype(np.float32)
    for start_sec in (6.00, 6.32, 6.64, 6.96, 7.28):
        start = int(sr * start_sec)
        audio[start : start + pulse_len] = pulse
    path = tmp_path / "zoom_stable_sparse_pulse_train.wav"
    _write_wav(path, audio, sr=sr)

    cache = WaveformCache(tmp_path / "cache")
    wide_tile = cache.tile(str(path), start_sec=0.0, end_sec=12.0, width=1200)
    tight_tile = cache.tile(str(path), start_sec=3.0, end_sec=9.0, width=1200)
    wide_placeable = [index for index, value in enumerate(wide_tile["boom_placeable_mask"]) if value]
    tight_placeable = [index for index, value in enumerate(tight_tile["boom_placeable_mask"]) if value]

    assert wide_tile["boom_placeable_count"] > 0
    assert tight_tile["boom_placeable_count"] > 0
    assert any(590 <= index <= 610 for index in wide_placeable)
    assert any(590 <= index <= 620 for index in tight_placeable)
    assert any(tight_tile["boom_relevant_mask"][620:760])


def test_boom_masks_reject_isolated_sparse_hit(tmp_path: Path) -> None:
    sr = 16000
    audio = np.zeros(sr * 5, dtype=np.float32)
    pulse_len = int(sr * 0.045)
    t = np.arange(pulse_len, dtype=np.float32) / sr
    pulse = 0.68 * np.sin(2 * np.pi * 72 * t).astype(np.float32)
    start = sr
    audio[start : start + pulse_len] = pulse
    path = tmp_path / "isolated_sparse_hit.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=5.0, width=1000)

    assert tile["boom_placeable_count"] == 0
    assert not any(tile["boom_placeable_mask"])


def test_boom_masks_reject_quiet_intro_when_louder_drop_exists_elsewhere(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    quiet_intro_hit = 0.08 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    hard_drop = 0.75 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    audio = np.zeros(sr * 10, dtype=np.float32)
    audio[sr : sr * 2] = quiet_intro_hit
    audio[sr * 6 : sr * 7] = hard_drop
    path = tmp_path / "quiet_then_drop.wav"
    _write_wav(path, audio, sr=sr)

    cache = WaveformCache(tmp_path / "cache")
    quiet_tile = cache.tile(str(path), start_sec=0.0, end_sec=2.5, width=500)
    drop_tile = cache.tile(str(path), start_sec=5.5, end_sec=7.5, width=500)
    full_tile = cache.tile(str(path), start_sec=0.0, end_sec=10.0, width=500)

    assert quiet_tile["boom_placeable_count"] == 0
    assert not any(quiet_tile["boom_relevant_mask"])
    assert not any(quiet_tile["boom_placeable_mask"])
    assert drop_tile["boom_placeable_count"] > 0
    assert full_tile["boom_placeable_count"] > 0

    drop_placeable = [index for index, value in enumerate(drop_tile["boom_placeable_mask"]) if value]
    full_placeable = [index for index, value in enumerate(full_tile["boom_placeable_mask"]) if value]
    assert drop_placeable[0] <= 130
    assert drop_placeable[-1] <= 146
    assert full_placeable[0] >= 295
    assert full_placeable[-1] <= 305


def test_boom_masks_cap_overstrict_floor_for_audible_dense_body(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    intro = 0.04 * np.sin(2 * np.pi * 70 * t).astype(np.float32)
    dense = 0.38 * np.sin(2 * np.pi * 76 * t).astype(np.float32)
    dense += 0.19 * np.sin(2 * np.pi * 152 * t).astype(np.float32)
    dense = np.clip(dense, -0.78, 0.78).astype(np.float32)
    audio = np.concatenate([intro, dense, dense * 0.94, np.zeros(sr, dtype=np.float32)])
    path = tmp_path / "audible_dense_body.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=4.0, width=800)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert max(tile["boom_body_density"]) > 0.0
    assert tile["boom_placeable_count"] > 0
    assert any(190 <= index <= 225 for index in placeable)


def test_boom_masks_cap_overstrict_floor_for_quiet_track_relative_body(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    intro = np.zeros(sr, dtype=np.float32)
    quiet_body = 0.075 * np.sin(2 * np.pi * 76 * t).astype(np.float32)
    quiet_body += 0.028 * np.sin(2 * np.pi * 152 * t).astype(np.float32)
    quiet_body = np.clip(quiet_body, -0.110, 0.110).astype(np.float32)
    audio = np.concatenate([intro, quiet_body, quiet_body * 0.94, np.zeros(sr, dtype=np.float32)])
    path = tmp_path / "quiet_relative_body.wav"
    _write_wav(path, audio, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=4.0, width=800)
    placeable = [index for index, value in enumerate(tile["boom_placeable_mask"]) if value]

    assert max(tile["boom_body_density"]) > 0.0
    assert tile["boom_placeable_count"] > 0
    assert any(190 <= index <= 230 for index in placeable)


def test_boom_masks_do_not_floor_cap_quiet_texture(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr * 4, dtype=np.float32) / sr
    texture = 0.045 * np.sin(2 * np.pi * 55 * t).astype(np.float32)
    path = tmp_path / "quiet_texture.wav"
    _write_wav(path, texture, sr=sr)

    tile = WaveformCache(tmp_path / "cache").tile(str(path), start_sec=0.0, end_sec=4.0, width=800)

    assert tile["boom_placeable_count"] == 0
    assert not any(tile["boom_relevant_mask"])
    assert not any(tile["boom_placeable_mask"])


def test_export_png_uses_boom_masks_so_quiet_regions_do_not_render_dark(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    quiet_intro_hit = 0.08 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    hard_drop = 0.75 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    audio = np.zeros(sr * 10, dtype=np.float32)
    audio[sr : sr * 2] = quiet_intro_hit
    audio[sr * 6 : sr * 7] = hard_drop
    path = tmp_path / "quiet_then_drop_png.wav"
    _write_wav(path, audio, sr=sr)
    cache = WaveformCache(tmp_path / "cache")

    quiet_png = cache.render_png(str(path), start_sec=0.0, end_sec=2.5, width=500, height=220)
    drop_png = cache.render_png(str(path), start_sec=5.5, end_sec=7.5, width=500, height=220)

    assert _dark_pixel_count(quiet_png) == 0
    assert _dark_pixel_count(drop_png) > 100


def test_export_png_treats_all_false_server_boom_masks_as_authoritative(tmp_path: Path) -> None:
    cache = WaveformCache(tmp_path / "cache")
    rms = [0.9] * 360
    tile = {
        "ok": True,
        "mode": "peaks",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "width": len(rms),
        "sample_rate": 16000,
        "duration": 3.0,
        "amplitude_peak": 1.0,
        "rms": rms,
        "boom_body_density": [1.0] * len(rms),
        "boom_body_mask": [False] * len(rms),
        "boom_relevant_mask": [False] * len(rms),
        "boom_placeable_mask": [False] * len(rms),
        "boom_placeable_count": 0,
    }

    def fake_tile(*_args, **_kwargs):
        return tile

    cache.tile = fake_tile  # type: ignore[method-assign]
    png = cache.render_png("fake.wav", start_sec=0.0, end_sec=3.0, width=420, height=220)

    assert _dark_pixel_count(png) == 0


def test_export_png_uses_relevant_mask_not_body_mask_for_dark_regions(tmp_path: Path) -> None:
    cache = WaveformCache(tmp_path / "cache")
    rms = [0.9] * 360
    tile = {
        "ok": True,
        "mode": "peaks",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "width": len(rms),
        "sample_rate": 16000,
        "duration": 3.0,
        "amplitude_peak": 1.0,
        "rms": rms,
        "boom_body_density": [1.0] * len(rms),
        "boom_body_mask": [True] * len(rms),
        "boom_relevant_mask": [False] * len(rms),
        "boom_placeable_mask": [False] * len(rms),
        "boom_placeable_count": 0,
    }

    def fake_tile(*_args, **_kwargs):
        return tile

    cache.tile = fake_tile  # type: ignore[method-assign]
    png = cache.render_png("fake.wav", start_sec=0.0, end_sec=3.0, width=420, height=220)

    assert _dark_pixel_count(png) == 0


def test_boom_proof_cannot_rescue_gui_tile_with_no_placeable_front_edge() -> None:
    gui_mask = {
        "passes": False,
        "reasons": ["marker_not_on_gui_boom_front_edge_mask", "gui_tile_has_no_placeable_boom_front_edge"],
        "placeable_count": 0,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.0, "edge_time": 12.0},
        "nearest_profile": {"passes_profile": True, "profile_score": 0.72},
    }

    proof = accept_gui_boom_mask_with_front_edge_proof(gui_mask, boom_proof)

    assert proof["passes"] is False
    assert proof.get("accepted_by_boom_front_edge_proof") is None
    assert "gui_tile_has_no_placeable_boom_front_edge" in proof["reasons"]


def test_boom_proof_can_rescue_small_offset_when_gui_has_placeable_front_edge() -> None:
    gui_mask = {
        "passes": False,
        "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
        "placeable_count": 3,
        "nearest_placeable_offset_sec": 0.006,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.006, "edge_time": 12.0},
        "nearest_profile": {"passes_profile": True, "profile_score": 0.72},
    }

    proof = accept_gui_boom_mask_with_front_edge_proof(gui_mask, boom_proof)

    assert proof["passes"] is True
    assert proof["accepted_by_boom_front_edge_proof"] is True
    assert proof["raw_reasons"] == ["marker_not_on_gui_boom_front_edge_mask"]


def test_gui_mask_proof_uses_time_radius_for_sparse_relevant_signal() -> None:
    length = 1200
    tile = {
        "start_sec": 1.6439909297052153,
        "end_sec": 7.6439909297052155,
        "boom_placeable_mask": [False] * length,
        "boom_relevant_mask": [False] * length,
        "boom_body_mask": [False] * length,
        "boom_body_density": [0.0] * length,
        "rms": [0.0] * length,
        "rms_peak": 0.4,
        "global_rms_peak": 0.4,
        "boom_placeable_count": 1,
    }
    marker = 4.6439909297052155
    marker_index = 600
    relevant_index = marker_index - 4
    tile["boom_relevant_mask"][relevant_index] = True
    tile["rms"][relevant_index] = 0.32
    tile["boom_placeable_mask"][340] = True

    proof = gui_boom_mask_proof_for_tile(tile, marker, radius_bins=2)

    assert proof["passes"] is False
    assert proof["marker_relevant_mask"] is True
    assert proof["marker_signal_present"] is True
    assert proof["relevant_radius_bins"] >= 7
    assert "marker_not_on_gui_boom_relevant_mask" not in proof["reasons"]
    assert "marker_not_on_gui_boom_front_edge_mask" in proof["reasons"]


def test_boom_proof_cannot_rescue_marker_without_visible_signal() -> None:
    gui_mask = {
        "passes": False,
        "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
        "placeable_count": 3,
        "nearest_placeable_offset_sec": 0.006,
        "marker_relevant_mask": True,
        "marker_signal_present": False,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.006, "edge_time": 12.0},
        "nearest_profile": {"passes_profile": True, "profile_score": 0.72},
    }

    proof = accept_gui_boom_mask_with_front_edge_proof(gui_mask, boom_proof)

    assert proof["passes"] is False
    assert proof.get("accepted_by_boom_front_edge_proof") is None
    assert "marker_has_no_visible_signal" in proof["reasons"]


def test_boom_proof_cannot_rescue_far_gui_front_edge() -> None:
    gui_mask = {
        "passes": False,
        "reasons": ["marker_not_on_gui_boom_front_edge_mask"],
        "placeable_count": 3,
        "nearest_placeable_offset_sec": 0.180,
        "marker_relevant_mask": True,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.006, "edge_time": 12.0},
        "nearest_profile": {"passes_profile": True, "profile_score": 0.72},
    }

    proof = accept_gui_boom_mask_with_front_edge_proof(gui_mask, boom_proof)

    assert proof["passes"] is False
    assert proof.get("accepted_by_boom_front_edge_proof") is None


def test_boom_proof_cannot_rescue_marker_outside_relevant_gui_mask() -> None:
    gui_mask = {
        "passes": False,
        "reasons": ["marker_not_on_gui_boom_relevant_mask", "marker_not_on_gui_boom_front_edge_mask"],
        "placeable_count": 3,
        "nearest_placeable_offset_sec": 0.006,
        "marker_relevant_mask": False,
        "marker_signal_present": True,
    }
    boom_proof = {
        "passes": True,
        "nearest": {"offset_sec": 0.006, "edge_time": 12.0},
        "nearest_profile": {"passes_profile": True, "profile_score": 0.72},
    }

    proof = accept_gui_boom_mask_with_front_edge_proof(gui_mask, boom_proof)

    assert proof["passes"] is False
    assert proof.get("accepted_by_boom_front_edge_proof") is None
    assert "marker_not_on_relevant_boom_mask" in proof["reasons"]


def test_strict_gui_contract_rejects_nominal_pass_without_immediate_body() -> None:
    gui_mask = {
        "passes": True,
        "reasons": [],
        "placeable_count": 1,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": False,
        "marker_post_body_occupancy_250ms": 0.0,
        "marker_post_body_occupancy_500ms": 0.0,
        "marker_post_density_mean_250ms": 0.0,
        "marker_post_density_mean_500ms": 0.0,
        "marker_post_density_max_250ms": 0.0,
        "marker_post_density_max_500ms": 0.0,
        "marker_post_relevant_occupancy_250ms": 0.0,
        "marker_post_relevant_occupancy_500ms": 0.0,
    }

    assert gui_boom_mask_strict_contract_issue(gui_mask) == "marker_has_no_immediate_drop_body"

    proof = enforce_strict_gui_boom_mask_contract(gui_mask)

    assert proof["passes"] is False
    assert proof["strict_gui_contract_failed"] is True
    assert "marker_has_no_immediate_drop_body" in proof["reasons"]


def test_strict_gui_contract_allows_explicit_sparse_front_relief() -> None:
    gui_mask = {
        "passes": True,
        "reasons": [],
        "placeable_count": 1,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": False,
        "marker_post_body_occupancy_250ms": 0.0,
        "marker_post_body_occupancy_500ms": 0.0,
        "marker_post_density_mean_250ms": 0.0,
        "marker_post_density_mean_500ms": 0.0,
        "marker_post_density_max_250ms": 0.0,
        "marker_post_density_max_500ms": 0.0,
        "accepted_by_staccato_front_body_proof": True,
    }

    assert gui_boom_mask_strict_contract_issue(gui_mask) is None
    assert enforce_strict_gui_boom_mask_contract(gui_mask)["passes"] is True


def test_strict_gui_contract_allows_valid_transition_start_relief() -> None:
    gui_mask = {
        "passes": True,
        "reasons": [],
        "raw_reasons": ["marker_has_no_immediate_drop_body"],
        "placeable_count": 12,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": False,
        "marker_post_body_occupancy_250ms": 0.04,
        "marker_post_body_occupancy_500ms": 0.02,
        "marker_post_density_mean_250ms": 0.0,
        "marker_post_density_mean_500ms": 0.0,
        "marker_post_density_max_250ms": 0.0,
        "marker_post_density_max_500ms": 0.0,
        "marker_post_relevant_occupancy_250ms": 0.66,
        "marker_post_relevant_occupancy_500ms": 0.65,
        "marker_post_rms_mean_250ms": 0.202,
        "marker_post_rms_mean_500ms": 0.186,
        "accepted_by_low_density_transition_body_start": True,
    }

    assert gui_boom_mask_strict_contract_issue(gui_mask) is None
    assert enforce_strict_gui_boom_mask_contract(gui_mask)["passes"] is True


def test_strict_gui_contract_rejects_weak_transition_start_relief() -> None:
    gui_mask = {
        "passes": True,
        "reasons": [],
        "raw_reasons": ["marker_has_no_immediate_drop_body"],
        "placeable_count": 12,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": True,
        "marker_signal_present": True,
        "marker_immediate_body_present": False,
        "marker_post_body_occupancy_250ms": 0.0,
        "marker_post_body_occupancy_500ms": 0.0,
        "marker_post_density_mean_250ms": 0.0,
        "marker_post_density_mean_500ms": 0.0,
        "marker_post_density_max_250ms": 0.0,
        "marker_post_density_max_500ms": 0.0,
        "marker_post_relevant_occupancy_250ms": 0.20,
        "marker_post_relevant_occupancy_500ms": 0.20,
        "marker_post_rms_mean_250ms": 0.06,
        "marker_post_rms_mean_500ms": 0.06,
        "accepted_by_low_density_transition_body_start": True,
    }

    assert gui_boom_mask_strict_contract_issue(gui_mask) == "marker_has_no_immediate_drop_body"
    assert enforce_strict_gui_boom_mask_contract(gui_mask)["passes"] is False


def test_strict_gui_contract_allows_opening_body_start_mask_miss_relief() -> None:
    gui_mask = {
        "passes": True,
        "raw_reasons": [
            "marker_not_on_gui_boom_relevant_mask",
            "marker_not_on_gui_boom_front_edge_mask",
        ],
        "reasons": [],
        "placeable_count": 1,
        "nearest_placeable_offset_sec": 0.0,
        "marker_relevant_mask": False,
        "marker_signal_present": True,
        "marker_immediate_body_present": True,
        "accepted_by_opening_body_start_proof": True,
        "opening_body_start_body_score": 0.820,
        "opening_body_start_darkness": 0.900,
        "opening_body_start_post8": 0.620,
        "opening_body_start_bass": 0.700,
        "opening_body_start_drum_cont": 0.680,
        "opening_body_start_clock_bar": 4,
    }

    assert gui_boom_mask_strict_contract_issue(gui_mask) is None
    assert enforce_strict_gui_boom_mask_contract(gui_mask)["passes"] is True
