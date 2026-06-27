from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.boom_profile import boom_proof_front_edge_freshness
from drop_aligner.visual_first import _accept_gui_contract_by_actual_body_proof
from drop_aligner.visual_first import _gui_mask_proof as visual_gui_mask_proof
from drop_aligner.visual_first import visual_first_marker
from drop_aligner.waveform import (
    WaveformCache,
    accept_gui_boom_mask_with_front_edge_proof,
)


DEFAULT_REPORT = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_contract_hardened_v5_freshproof_report.json"
)
DEFAULT_CACHE_DIR = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/.waveform_cache"
)
DEFAULT_OUT_DIR = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/suspicious_marker_audit"
)

STRICT_FRONT_EDGE_SEC = 0.080
MIN_SELECTED_SCORE = 0.380
MIN_POST_BODY = 0.340
MIN_POST_PEAK = 0.430
BOOM_BODY_VISIBLE_DENSITY = 0.220
BOOM_BODY_STRONG_DENSITY = 0.360
MIN_SUSTAINED_POST_LONG = 0.240
MIN_SUSTAINED_FRACTION = 0.120
MIN_SUSTAINED_BODY_RUN_SEC = 0.120
MIN_SUSTAINED_SCORE = 0.300
STRONGER_SECTION_ABS_MARGIN = 0.220
STRONGER_SECTION_RATIO = 1.36
ADVISORY_LATER_ABS_MARGIN = 0.300
ADVISORY_LATER_RATIO = 1.48
LOCAL_PROOF_RELIEF_QUANTIZATION_SEC = 0.040
LOCAL_PROOF_RELIEF_HARD_FLAGS = {
    "no_local_placeable_boom_front_edge",
    "marker_not_on_local_boom_front_edge",
    "marker_inside_body_without_front_edge",
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _percentile(values: Sequence[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return float(finite[0])
    position = (len(finite) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(finite[lower])
    frac = position - lower
    return float((finite[lower] * (1.0 - frac)) + (finite[upper] * frac))


def _segments(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    active: Optional[int] = None
    for index, value in enumerate(list(mask) + [False]):
        if bool(value) and active is None:
            active = int(index)
        elif not bool(value) and active is not None:
            segments.append((int(active), int(index)))
            active = None
    return segments


def _nearest_true_index(mask: Sequence[bool], target: int) -> Optional[int]:
    indexes = [idx for idx, value in enumerate(mask) if bool(value)]
    if not indexes:
        return None
    return int(min(indexes, key=lambda idx: abs(idx - int(target))))


def _segment_containing(segments: Sequence[Tuple[int, int]], index: int) -> Optional[Tuple[int, int]]:
    for start, end in segments:
        if int(start) <= int(index) < int(end):
            return int(start), int(end)
    return None


def _merge_nearby_segments(segments: Sequence[Tuple[int, int]], max_gap_bins: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in segments):
        if not merged or start - merged[-1][1] > max(0, int(max_gap_bins)):
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _longest_true_run(mask: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _bpm_from_path(path: str) -> Optional[float]:
    match = re.search(r"(?:^|/)drums_(\d+(?:\.\d+)?)_", str(path))
    if not match:
        return None
    bpm = _finite_float(match.group(1), default=0.0)
    return bpm if bpm > 0.0 else None


def _safe_read_json(path: Any) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_indices(raw: str) -> List[int]:
    indices: List[int] = []
    seen: set[int] = set()
    for part in (item.strip() for item in str(raw or "").split(",")):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for value in values:
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            indices.append(value)
    return indices


def _select_indexed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    indices: Sequence[int],
    offset: int,
    limit: int,
) -> List[Tuple[int, Mapping[str, Any]]]:
    indexed = list(enumerate(rows, start=1))
    if indices:
        wanted = set(int(index) for index in indices)
        indexed = [(index, row) for index, row in indexed if index in wanted]
    elif offset:
        indexed = [(index, row) for index, row in indexed if index > int(offset)]
    if limit:
        indexed = indexed[: int(limit)]
    return indexed


def _time_for_index(start_sec: float, bin_span: float, index: int) -> float:
    return float(start_sec + (max(0, int(index)) * float(bin_span)))


def _score_front_edge(
    density: Sequence[float],
    edge_index: int,
    *,
    bin_span: float,
    body_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    n = len(density)
    edge = max(0, min(n - 1, int(edge_index))) if n else 0
    post_bins = max(1, int(round(0.420 / max(float(bin_span), 1e-9))))
    peak_bins = max(post_bins, int(round(0.650 / max(float(bin_span), 1e-9))))
    long_bins = max(peak_bins, int(round(1.850 / max(float(bin_span), 1e-9))))
    pre_bins = max(1, int(round(0.180 / max(float(bin_span), 1e-9))))
    post = [float(value) for value in density[edge : min(n, edge + post_bins)]]
    peak_window = [float(value) for value in density[edge : min(n, edge + peak_bins)]]
    long_window = [float(value) for value in density[edge : min(n, edge + long_bins)]]
    pre = [float(value) for value in density[max(0, edge - pre_bins) : edge]]
    post_body = _percentile(post, 72.0)
    post_peak = max(peak_window, default=0.0)
    post_mean = sum(post) / max(1, len(post))
    post_long = _percentile(long_window, 70.0)
    pre_body = _percentile(pre, 58.0) if pre else 0.0
    contrast = max(0.0, post_body - pre_body)
    body_window: List[bool]
    if body_mask and len(body_mask) >= n:
        body_window = [bool(value) for value in body_mask[edge : min(n, edge + long_bins)]]
    else:
        body_window = [float(value) >= BOOM_BODY_VISIBLE_DENSITY for value in long_window]
    strong_window = [float(value) >= BOOM_BODY_STRONG_DENSITY for value in long_window]
    post_sustain_fraction = sum(1 for value in body_window if value) / max(1, len(body_window))
    post_strong_fraction = sum(1 for value in strong_window if value) / max(1, len(strong_window))
    body_run_sec = float(_longest_true_run(body_window)) * max(float(bin_span), 1e-9)
    sustain_score = max(
        0.0,
        min(
            1.0,
            (0.40 * post_sustain_fraction)
            + (0.26 * min(1.0, body_run_sec / 0.650))
            + (0.22 * post_long)
            + (0.12 * post_strong_fraction),
        ),
    )
    score = max(
        0.0,
        min(
            1.0,
            (0.34 * post_body)
            + (0.18 * post_peak)
            + (0.18 * post_long)
            + (0.12 * post_mean)
            + (0.12 * sustain_score)
            + (0.06 * contrast),
        ),
    )
    return {
        "score": float(score),
        "post_body": float(post_body),
        "post_peak": float(post_peak),
        "post_mean": float(post_mean),
        "post_long": float(post_long),
        "post_sustain_fraction": float(post_sustain_fraction),
        "post_strong_fraction": float(post_strong_fraction),
        "body_run_sec": float(body_run_sec),
        "sustain_score": float(sustain_score),
        "pre_body": float(pre_body),
        "contrast": float(contrast),
    }


def _front_edge_has_sustained_drop_body(front: Mapping[str, float]) -> bool:
    post_long = _finite_float(front.get("post_long"))
    post_sustain = _finite_float(front.get("post_sustain_fraction"))
    post_strong = _finite_float(front.get("post_strong_fraction"))
    body_run_sec = _finite_float(front.get("body_run_sec"))
    sustain_score = _finite_float(front.get("sustain_score"))
    score = _finite_float(front.get("score"))
    return bool(
        score >= MIN_SUSTAINED_SCORE
        and post_long >= MIN_SUSTAINED_POST_LONG
        and (
            post_sustain >= MIN_SUSTAINED_FRACTION
            or post_strong >= (MIN_SUSTAINED_FRACTION * 0.60)
            or body_run_sec >= MIN_SUSTAINED_BODY_RUN_SEC
            or sustain_score >= 0.360
        )
    )


def _selected_reference_score(
    selected_metrics: Mapping[str, float],
    selected: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> float:
    """Use the strongest available visual proof for selected-vs-alternative checks."""

    scores = [
        _finite_float(selected_metrics.get("score")),
        _finite_float(selected.get("score")),
        _finite_float(selected.get("confidence_score")),
    ]
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    scores.extend(
        [
            _finite_float(visual.get("boom_section_darkness")),
            _finite_float(visual.get("bar_height")),
            _finite_float(visual.get("max_post8_height")),
        ]
    )
    nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
    nearest_profile = proof.get("nearest_profile") if isinstance(proof.get("nearest_profile"), Mapping) else {}
    profile_metrics = nearest_profile.get("metrics") if isinstance(nearest_profile.get("metrics"), Mapping) else {}
    scores.extend(
        [
            _finite_float(nearest.get("candidate_score")),
            _finite_float(nearest_profile.get("profile_score")),
            _finite_float(profile_metrics.get("score")),
            _finite_float(profile_metrics.get("darkness")),
            _finite_float(profile_metrics.get("body_score")),
            _finite_float(profile_metrics.get("post8_height")),
        ]
    )
    return max((float(score) for score in scores if math.isfinite(float(score))), default=0.0)


def _audit_row(job: Mapping[str, Any]) -> Dict[str, Any]:
    row = job["row"]
    cache_dir = Path(str(job["cache_dir"]))
    width = int(job["width"])
    index = int(job["index"])
    if bool(job.get("rerun_detector")):
        drums_path_for_rerun = str(row.get("drums_path") or row.get("filename") or row.get("audio_path") or "")
        if drums_path_for_rerun:
            try:
                payload = visual_first_marker(
                    drums_path_for_rerun,
                    sample_rate=int(job.get("sample_rate") or 44100),
                    use_cache=bool(job.get("use_cache", True)),
                )
            except Exception as exc:
                payload = {"ok": False, "error": str(exc) or exc.__class__.__name__}
            selected = (
                payload.get("selected_candidate")
                if isinstance(payload.get("selected_candidate"), Mapping)
                else {}
            )
            rerun_marker = _finite_float(
                payload.get("marker")
                or selected.get("timestamp")
                or selected.get("time_sec")
                or selected.get("snapped_sec"),
                default=float("nan"),
            )
            row = {
                **dict(row),
                "marker": rerun_marker,
                "selected_by": str(selected.get("selected_by") or payload.get("selected_by") or ""),
                "boom_proof": payload.get("boom_proof") if isinstance(payload.get("boom_proof"), Mapping) else selected.get("boom_proof"),
                "gui_mask_proof": (
                    payload.get("gui_mask_proof")
                    if isinstance(payload.get("gui_mask_proof"), Mapping)
                    else selected.get("gui_mask_proof")
                ),
                "selected_candidate": dict(selected) if isinstance(selected, Mapping) else {},
                "detector_ok": bool(payload.get("ok")),
                "detector_error": str(payload.get("error") or ""),
            }
    marker = _finite_float(row.get("marker"), default=float("nan"))
    drums_path = str(row.get("drums_path") or "")
    base: Dict[str, Any] = {
        "index": index,
        "track": str(row.get("track") or ""),
        "drums_path": drums_path,
        "marker": marker if math.isfinite(marker) else "",
        "selected_by": str(row.get("selected_by") or ""),
        "fail_flags": "",
        "warn_flags": "",
        "selected_score": "",
        "selected_gui_score": "",
        "selected_post_body": "",
        "selected_post_peak": "",
        "selected_post_long": "",
        "selected_sustain_score": "",
        "selected_body_run_ms": "",
        "selected_contrast": "",
        "nearest_front_edge_offset_ms": "",
        "best_earlier_time": "",
        "best_earlier_score": "",
        "best_earlier_sustain_score": "",
        "best_later_time": "",
        "best_later_score": "",
        "best_later_sustain_score": "",
        "detector_ok": row.get("detector_ok", ""),
        "detector_error": row.get("detector_error", ""),
    }
    try:
        if not drums_path or not math.isfinite(marker):
            return {**base, "fail_flags": "missing_marker_or_drums_path"}
        candidate_payload = _safe_read_json(row.get("candidates_json"))
        selected = (
            candidate_payload.get("selected_candidate")
            if isinstance(candidate_payload.get("selected_candidate"), Mapping)
            else {}
        )
        if not selected and isinstance(row.get("selected_candidate"), Mapping):
            selected = row.get("selected_candidate")  # type: ignore[assignment]
        cache = WaveformCache(cache_dir)
        info = cache.info(drums_path)
        duration = _finite_float(info.get("duration"), default=0.0)
        if duration <= 0.0:
            return {**base, "fail_flags": "missing_audio_duration"}
        half_view = 3.0
        local_start = max(0.0, marker - half_view)
        local_end = min(duration, marker + half_view)
        if local_end - local_start < 0.250:
            local_end = min(duration, local_start + 0.250)
        local_tile = cache.tile(drums_path, start_sec=local_start, end_sec=local_end, width=1200)
        local_density = [
            max(0.0, min(1.0, _finite_float(value)))
            for value in local_tile.get("boom_body_density", [])
        ]
        local_placeable = [bool(value) for value in local_tile.get("boom_placeable_mask", [])]
        local_body_mask = [bool(value) for value in local_tile.get("boom_body_mask", [])]
        local_n = len(local_density)
        if local_n < 2 or len(local_placeable) != local_n:
            return {**base, "fail_flags": "missing_local_boom_mask"}
        local_tile_start = _finite_float(local_tile.get("start_sec"), default=local_start)
        local_tile_end = _finite_float(local_tile.get("end_sec"), default=local_end)
        local_span = max(1e-9, local_tile_end - local_tile_start)
        local_bin_span = local_span / max(1, local_n)
        local_marker_index = max(
            0,
            min(local_n - 1, int(math.floor(((marker - local_tile_start) / local_span) * local_n))),
        )
        local_nearest_placeable = _nearest_true_index(local_placeable, local_marker_index)
        fail_flags: List[str] = []
        warn_flags: List[str] = []
        local_raw_flags: List[str] = []
        if local_nearest_placeable is None:
            local_raw_flags.append("no_local_placeable_boom_front_edge")
            selected_index = local_marker_index
            nearest_offset_sec = float("inf")
        else:
            selected_index = int(local_nearest_placeable)
            nearest_offset_sec = (int(local_nearest_placeable) - int(local_marker_index)) * float(local_bin_span)
            if abs(nearest_offset_sec) > STRICT_FRONT_EDGE_SEC:
                local_raw_flags.append("marker_not_on_local_boom_front_edge")
        selected_metrics = _score_front_edge(
            local_density,
            selected_index,
            bin_span=local_bin_span,
            body_mask=local_body_mask,
        )
        if selected_metrics["score"] < MIN_SELECTED_SCORE:
            local_raw_flags.append("weak_selected_boom_score")
        if selected_metrics["post_body"] < MIN_POST_BODY:
            local_raw_flags.append("weak_selected_post_body")
        if selected_metrics["post_peak"] < MIN_POST_PEAK:
            local_raw_flags.append("weak_selected_post_peak")

        local_body_segments = _segments(local_body_mask)
        marker_body_segment = _segment_containing(local_body_segments, local_marker_index)
        if marker_body_segment and local_nearest_placeable is None:
            body_start_sec = _time_for_index(local_tile_start, local_bin_span, marker_body_segment[0])
            if marker - body_start_sec > 0.250:
                local_raw_flags.append("marker_inside_body_without_front_edge")
        proof = row.get("boom_proof") if isinstance(row.get("boom_proof"), Mapping) else {}
        selected_reference_score = _selected_reference_score(selected_metrics, selected, proof)

        tile = cache.tile(drums_path, start_sec=0.0, end_sec=duration, width=width)
        density = [max(0.0, min(1.0, _finite_float(value))) for value in tile.get("boom_body_density", [])]
        placeable = [bool(value) for value in tile.get("boom_placeable_mask", [])]
        body_mask = [bool(value) for value in tile.get("boom_body_mask", [])]
        n = len(density)
        if n < 2 or len(placeable) != n:
            return {**base, "fail_flags": "missing_full_track_boom_mask"}
        if len(body_mask) != n:
            body_mask = [value >= BOOM_BODY_VISIBLE_DENSITY for value in density]
        start_sec = _finite_float(tile.get("start_sec"), default=0.0)
        end_sec = _finite_float(tile.get("end_sec"), default=duration)
        span = max(1e-9, end_sec - start_sec)
        bin_span = span / max(1, n)

        placeable_segments = _merge_nearby_segments(
            _segments(placeable),
            max_gap_bins=max(1, int(round(0.120 / max(bin_span, 1e-9)))),
        )
        fronts: List[Dict[str, float]] = []
        for front_start, _front_end in placeable_segments:
            metrics = _score_front_edge(density, front_start, bin_span=bin_span, body_mask=body_mask)
            fronts.append(
                {
                    **metrics,
                    "index": float(front_start),
                    "time": _time_for_index(start_sec, bin_span, front_start),
                }
            )
        bpm = _bpm_from_path(drums_path)
        bar_sec = 240.0 / bpm if bpm else 2.0
        sustained_fronts = [front for front in fronts if _front_edge_has_sustained_drop_body(front)]
        earlier = [
            front
            for front in sustained_fronts
            if front["time"] < marker - max(0.750, 0.50 * bar_sec)
            and front["time"] >= max(4.0, 1.50 * bar_sec)
        ]
        later = [front for front in sustained_fronts if front["time"] > marker + max(0.750, 0.50 * bar_sec)]
        best_earlier = max(earlier, key=lambda item: item["score"], default=None)
        best_later = max(later, key=lambda item: item["score"], default=None)
        selected_score = float(selected_reference_score)
        if best_earlier and (
            best_earlier["score"] >= selected_score + STRONGER_SECTION_ABS_MARGIN
            and best_earlier["score"] >= selected_score * STRONGER_SECTION_RATIO
        ):
            warn_flags.append("earlier_stronger_boom_front_edge")
        if best_later and (
            best_later["score"] >= selected_score + ADVISORY_LATER_ABS_MARGIN
            and best_later["score"] >= selected_score * ADVISORY_LATER_RATIO
        ):
            warn_flags.append("later_much_stronger_boom_front_edge")
        proof_nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
        proof_offset = abs(_finite_float(proof_nearest.get("offset_sec"), default=999999.0))
        freshness = boom_proof_front_edge_freshness(proof)
        if proof_offset > STRICT_FRONT_EDGE_SEC:
            fail_flags.append("persisted_proof_front_edge_offset")
        if bool(proof.get("passes")) is not True:
            fail_flags.append("persisted_boom_proof_not_passing")
        if not bool(freshness.get("fresh")):
            fail_flags.append("persisted_boom_proof_not_fresh")
        gui_proof = row.get("gui_mask_proof") if isinstance(row.get("gui_mask_proof"), Mapping) else {}
        if bool(gui_proof.get("passes")) is not True:
            fail_flags.append("persisted_gui_mask_not_passing")
        elif "marker_signal_present" not in gui_proof:
            fail_flags.append("persisted_gui_mask_missing_marker_signal")
        elif gui_proof.get("marker_signal_present") is not True:
            fail_flags.append("persisted_gui_mask_marker_has_no_signal")
        recomputed_gui = visual_gui_mask_proof(
            drums_path,
            marker,
            cache_dir=cache_dir,
        )
        selected_by = str(row.get("selected_by") or selected.get("selected_by") or "")
        repair_gui_lag_sec = 0.300 if selected_by.startswith("visual_final_contract_") else 0.040
        repair_gui_profile = 0.620 if selected_by.startswith("visual_final_contract_") else 0.560
        recomputed_gui = accept_gui_boom_mask_with_front_edge_proof(
            recomputed_gui,
            proof,
            near_offset_sec=repair_gui_lag_sec,
            near_profile_score=repair_gui_profile,
        )
        recomputed_gui = _accept_gui_contract_by_actual_body_proof(recomputed_gui, proof, selected)
        if "marker_signal_present" not in recomputed_gui:
            fail_flags.append("recomputed_gui_mask_missing_marker_signal")
        elif recomputed_gui.get("marker_signal_present") is not True:
            fail_flags.append("recomputed_gui_mask_marker_has_no_signal")
        if not bool(recomputed_gui.get("passes")):
            fail_flags.append("recomputed_gui_mask_not_passing")
            fail_flags.extend(local_raw_flags)
        elif local_raw_flags and bool(recomputed_gui.get("accepted_by_boom_front_edge_proof")):
            relief_nearest_offset = abs(
                _finite_float(recomputed_gui.get("nearest_placeable_offset_sec"), default=999999.0)
            )
            relief_proof_offset = abs(_finite_float(recomputed_gui.get("front_edge_offset_sec"), default=999999.0))
            hard_local_flags = LOCAL_PROOF_RELIEF_HARD_FLAGS.intersection(local_raw_flags)
            mask_quantization_only = bool(
                not hard_local_flags
                and relief_nearest_offset <= LOCAL_PROOF_RELIEF_QUANTIZATION_SEC
                and relief_proof_offset <= LOCAL_PROOF_RELIEF_QUANTIZATION_SEC
            )
            if not mask_quantization_only:
                warn_flags.append("local_mask_relied_on_exact_boom_proof")
        return {
            **base,
            "fail_flags": ";".join(sorted(set(fail_flags))),
            "warn_flags": ";".join(sorted(set(warn_flags))),
            "selected_score": f"{selected_score:.6f}",
            "selected_gui_score": f"{selected_metrics['score']:.6f}",
            "selected_post_body": f"{selected_metrics['post_body']:.6f}",
            "selected_post_peak": f"{selected_metrics['post_peak']:.6f}",
            "selected_post_long": f"{selected_metrics['post_long']:.6f}",
            "selected_sustain_score": f"{selected_metrics['sustain_score']:.6f}",
            "selected_body_run_ms": f"{selected_metrics['body_run_sec'] * 1000.0:.3f}",
            "selected_contrast": f"{selected_metrics['contrast']:.6f}",
            "nearest_front_edge_offset_ms": (
                "" if not math.isfinite(nearest_offset_sec) else f"{nearest_offset_sec * 1000.0:.3f}"
            ),
            "best_earlier_time": "" if not best_earlier else f"{best_earlier['time']:.6f}",
            "best_earlier_score": "" if not best_earlier else f"{best_earlier['score']:.6f}",
            "best_earlier_sustain_score": "" if not best_earlier else f"{best_earlier['sustain_score']:.6f}",
            "best_later_time": "" if not best_later else f"{best_later['time']:.6f}",
            "best_later_score": "" if not best_later else f"{best_later['score']:.6f}",
            "best_later_sustain_score": "" if not best_later else f"{best_later['sustain_score']:.6f}",
        }
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        return {**base, "fail_flags": "audit_error", "warn_flags": str(exc)}


def _timeout_result(job: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    row = job["row"]
    index = int(job["index"])
    return {
        "index": index,
        "track": str(row.get("track") or ""),
        "drums_path": str(row.get("drums_path") or row.get("filename") or row.get("audio_path") or ""),
        "marker": "",
        "selected_by": "",
        "fail_flags": str(reason),
        "warn_flags": "",
        "selected_score": "",
        "selected_gui_score": "",
        "selected_post_body": "",
        "selected_post_peak": "",
        "selected_post_long": "",
        "selected_sustain_score": "",
        "selected_body_run_ms": "",
        "selected_contrast": "",
        "nearest_front_edge_offset_ms": "",
        "best_earlier_time": "",
        "best_earlier_score": "",
        "best_earlier_sustain_score": "",
        "best_later_time": "",
        "best_later_score": "",
        "best_later_sustain_score": "",
        "detector_ok": False,
        "detector_error": str(reason),
    }


def _audit_row_queue(job: Mapping[str, Any], queue: Any) -> None:
    try:
        queue.put(_audit_row(job))
    except BaseException as exc:  # pragma: no cover - child process boundary
        queue.put(_timeout_result(job, f"audit_error:{str(exc) or exc.__class__.__name__}"))


def _audit_row_with_timeout(job: Mapping[str, Any], timeout_sec: float) -> Dict[str, Any]:
    if float(timeout_sec) <= 0.0:
        return _audit_row(job)
    ctx = mp.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_audit_row_queue, args=(dict(job), queue))
    proc.start()
    proc.join(float(timeout_sec))
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():  # pragma: no cover - defensive hard kill
            proc.kill()
            proc.join(1.0)
        return _timeout_result(job, f"audit_timeout:{float(timeout_sec):.3f}s")
    if not queue.empty():
        return queue.get()
    if proc.exitcode == 0:
        return _timeout_result(job, "audit_error:missing_child_result")
    return _timeout_result(job, f"audit_error:child_exit_{proc.exitcode}")


def _processed_rows(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rows = report.get("processed_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    rows = report.get("tracks")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def run_audit(
    report_path: Path,
    *,
    cache_dir: Path,
    out_dir: Path,
    width: int,
    workers: int,
    indices: Sequence[int] = (),
    offset: int = 0,
    limit: int = 0,
    rerun_detector: bool = False,
    sample_rate: int = 44100,
    use_cache: bool = True,
    per_row_timeout_sec: float = 0.0,
    progress_every: int = 25,
) -> Dict[str, Any]:
    report = json.loads(report_path.expanduser().read_text(encoding="utf-8"))
    rows = _processed_rows(report if isinstance(report, Mapping) else {})
    indexed_rows = _select_indexed_rows(rows, indices=indices, offset=int(offset), limit=int(limit))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "index": index,
            "row": row,
            "cache_dir": str(cache_dir.expanduser()),
            "width": int(width),
            "rerun_detector": bool(rerun_detector),
            "sample_rate": int(sample_rate),
            "use_cache": bool(use_cache),
        }
        for index, row in indexed_rows
    ]
    results: List[Dict[str, Any]] = []
    progress_step = max(1, int(progress_every))
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            results.append(_audit_row_with_timeout(job, float(per_row_timeout_sec)))
            if index % progress_step == 0 or index == len(jobs):
                print(f"audited {index}/{len(jobs)}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_audit_row, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index % progress_step == 0 or index == len(futures):
                    print(f"audited {index}/{len(futures)}", file=sys.stderr, flush=True)
    results.sort(key=lambda row: int(row.get("index") or 0))
    fail_rows = [row for row in results if str(row.get("fail_flags") or "")]
    warn_rows = [row for row in results if str(row.get("warn_flags") or "")]
    csv_path = out_dir / f"{report_path.stem}_suspicious_markers.csv"
    fail_csv_path = out_dir / f"{report_path.stem}_suspicious_marker_failures.csv"
    fieldnames = [
        "index",
        "track",
        "marker",
        "selected_by",
        "fail_flags",
        "warn_flags",
        "selected_score",
        "selected_gui_score",
        "selected_post_body",
        "selected_post_peak",
        "selected_post_long",
        "selected_sustain_score",
        "selected_body_run_ms",
        "selected_contrast",
        "nearest_front_edge_offset_ms",
        "best_earlier_time",
        "best_earlier_score",
        "best_earlier_sustain_score",
        "best_later_time",
        "best_later_score",
        "best_later_sustain_score",
        "detector_ok",
        "detector_error",
        "drums_path",
    ]
    for path, selected in ((csv_path, results), (fail_csv_path, fail_rows)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
    summary = {
        "source_report": str(report_path.expanduser()),
        "audited_count": len(results),
        "selected_row_count": len(jobs),
        "rerun_detector": bool(rerun_detector),
        "failure_count": len(fail_rows),
        "warning_count": len(warn_rows),
        "all_fail_closed_checks_passed": len(fail_rows) == 0,
        "csv": str(csv_path),
        "failure_csv": str(fail_csv_path),
        "failure_flag_counts": _count_flags(row.get("fail_flags") for row in fail_rows),
        "warning_flag_counts": _count_flags(row.get("warn_flags") for row in warn_rows),
    }
    json_path = out_dir / f"{report_path.stem}_suspicious_marker_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(json_path)
    return summary


def _count_flags(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        for flag in str(value or "").split(";"):
            flag = flag.strip()
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent full-track suspicious-marker audit for visual-first production reports.",
    )
    parser.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--indices", default="", help="1-based row indices or ranges, e.g. 1,254,800-820.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerun-detector", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--per-row-timeout-sec", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=True)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    parser.add_argument("--require-no-failures", action="store_true")
    args = parser.parse_args(argv)
    summary = run_audit(
        Path(args.report),
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        width=int(args.width),
        workers=int(args.workers),
        indices=_parse_indices(args.indices),
        offset=int(args.offset),
        limit=int(args.limit),
        rerun_detector=bool(args.rerun_detector),
        sample_rate=int(args.sample_rate),
        use_cache=bool(args.use_cache),
        per_row_timeout_sec=float(args.per_row_timeout_sec),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_no_failures and int(summary.get("failure_count") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
