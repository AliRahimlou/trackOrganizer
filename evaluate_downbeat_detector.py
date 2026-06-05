#!/usr/bin/env python3

"""
Batch evaluator for the first-downbeat / 1.1.1 detector.

This script walks a stem library, extracts the human-labeled drums anchor from
the existing Ableton project for each track, runs the detector, and compares
prediction vs. ground truth. It is intentionally conservative about extraction:

- Track discovery is reused from ``first_downbeat_detector.discover_library_tracks``.
- Ground truth comes from the drums clip's BeatTime=0 marker via
  ``alsdrop.als_io.extract_labels_from_als``.
- ``CH1 Project/CH1.als`` is preferred when present because it reflects the
  manually corrected project; root-level ``CH1.als`` is the fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from alsdrop.als_io import extract_labels_from_als
from first_downbeat_detector import DetectorOptions, detect_track_folder, discover_library_tracks, discover_track_folder


LOG = logging.getLogger("evaluate_downbeat_detector")


@dataclass
class GroundTruthAnchor:
    seconds: float
    als_path: str
    clip_name: Optional[str]
    target_source: Optional[str]
    marker_count: Optional[int]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "track"


def _pct(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (100.0 * float(count)) / float(total)


def _pass_bucket(abs_ms: Optional[float], abs_beats: Optional[float]) -> Optional[str]:
    if abs_ms is None or abs_beats is None:
        return None
    if abs_ms <= 10.0:
        return "<=10ms"
    if abs_ms <= 25.0:
        return "<=25ms"
    if abs_ms <= 50.0:
        return "<=50ms"
    if abs_beats <= 0.125:
        return "<=0.125beat"
    if abs_beats <= 0.25:
        return "<=0.25beat"
    return "fail"


def _confidence_bucket(conf: Optional[float]) -> str:
    if conf is None:
        return "unknown"
    if conf < 0.45:
        return "<0.45"
    if conf < 0.60:
        return "0.45-0.59"
    if conf < 0.75:
        return "0.60-0.74"
    if conf < 0.90:
        return "0.75-0.89"
    return ">=0.90"


def _bpm_bucket(bpm: Optional[float]) -> str:
    if bpm is None or bpm <= 0:
        return "unknown"
    if bpm < 90.0:
        return "<90"
    if bpm < 110.0:
        return "90-109"
    if bpm < 130.0:
        return "110-129"
    if bpm < 150.0:
        return "130-149"
    return ">=150"


def _summarize_error_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("abs_error_ms") is not None]
    total = len(valid)
    if total == 0:
        return {
            "tracks": 0,
            "mean_abs_error_ms": None,
            "median_abs_error_ms": None,
            "p90_abs_error_ms": None,
            "p95_abs_error_ms": None,
            "within_10ms": 0,
            "within_25ms": 0,
            "within_50ms": 0,
            "within_0p125beat": 0,
            "within_0p25beat": 0,
            "within_10ms_pct": None,
            "within_25ms_pct": None,
            "within_50ms_pct": None,
            "within_0p125beat_pct": None,
            "within_0p25beat_pct": None,
        }

    abs_ms = np.asarray([float(r["abs_error_ms"]) for r in valid], dtype=np.float64)
    abs_beats = np.asarray([float(r["abs_beat_error"]) for r in valid], dtype=np.float64)
    within_10 = int(np.sum(abs_ms <= 10.0))
    within_25 = int(np.sum(abs_ms <= 25.0))
    within_50 = int(np.sum(abs_ms <= 50.0))
    within_0125 = int(np.sum(abs_beats <= 0.125))
    within_025 = int(np.sum(abs_beats <= 0.25))
    return {
        "tracks": total,
        "mean_abs_error_ms": float(np.mean(abs_ms)),
        "median_abs_error_ms": float(np.median(abs_ms)),
        "p90_abs_error_ms": float(np.percentile(abs_ms, 90.0)),
        "p95_abs_error_ms": float(np.percentile(abs_ms, 95.0)),
        "mean_abs_beat_error": float(np.mean(abs_beats)),
        "median_abs_beat_error": float(np.median(abs_beats)),
        "within_10ms": within_10,
        "within_25ms": within_25,
        "within_50ms": within_50,
        "within_0p125beat": within_0125,
        "within_0p25beat": within_025,
        "within_10ms_pct": _pct(within_10, total),
        "within_25ms_pct": _pct(within_25, total),
        "within_50ms_pct": _pct(within_50, total),
        "within_0p125beat_pct": _pct(within_0125, total),
        "within_0p25beat_pct": _pct(within_025, total),
    }


def _extract_ground_truth(ch1_path: Optional[str]) -> Tuple[Optional[GroundTruthAnchor], Optional[str]]:
    if not ch1_path:
        return None, "missing_ch1"
    if not os.path.exists(ch1_path):
        return None, "missing_ch1"

    try:
        labels = extract_labels_from_als(ch1_path, resolve_paths=True)
    except Exception as exc:
        return None, f"als_parse_error: {exc}"

    drums_rows = []
    for row in labels:
        audio_path = str(getattr(row, "audio_path", "") or "")
        base = os.path.basename(audio_path).lower()
        clip_name = str((getattr(row, "metadata", {}) or {}).get("clip_name") or "").lower()
        if base.startswith("drums_") or clip_name.startswith("drums_"):
            drums_rows.append(row)

    if not drums_rows:
        return None, "no_drums_clip_label"

    drums_rows.sort(key=lambda r: float(getattr(r, "target_sec", math.inf)))
    row = drums_rows[0]
    target_sec = _safe_float(getattr(row, "target_sec", None))
    if target_sec is None:
        return None, "no_target_sec"

    meta = getattr(row, "metadata", {}) or {}
    return (
        GroundTruthAnchor(
            seconds=float(target_sec),
            als_path=os.path.abspath(ch1_path),
            clip_name=str(meta.get("clip_name")) if meta.get("clip_name") else None,
            target_source=str(meta.get("target_source")) if meta.get("target_source") else None,
            marker_count=_safe_int(meta.get("marker_count")),
        ),
        None,
    )


def _candidate_analysis(
    manual_sec: float,
    bpm: Optional[float],
    candidate_seconds: Sequence[Any],
    candidate_scores: Sequence[Any],
) -> Dict[str, Any]:
    secs = [_safe_float(v) for v in candidate_seconds]
    scores = [_safe_float(v) for v in candidate_scores]
    pairs = [(sec, score if score is not None else float("-inf")) for sec, score in zip(secs, scores) if sec is not None]
    if not pairs:
        return {
            "candidate_count": 0,
            "nearest_candidate_seconds": None,
            "nearest_candidate_abs_error_ms": None,
            "nearest_candidate_abs_beat_error": None,
            "manual_in_candidates_0p125beat": False,
            "manual_in_candidates_0p25beat": False,
            "manual_in_candidates_0p50beat": False,
            "nearest_candidate_rank": None,
        }

    nearest_sec, nearest_score = min(pairs, key=lambda item: abs(float(item[0]) - float(manual_sec)))
    beat_sec = (60.0 / float(bpm)) if bpm and bpm > 0.0 else None
    err_sec = float(nearest_sec) - float(manual_sec)
    abs_beats = abs(err_sec / beat_sec) if beat_sec else None
    rank = 1 + int(sum(1 for _, score in pairs if float(score) > float(nearest_score)))
    return {
        "candidate_count": int(len(pairs)),
        "nearest_candidate_seconds": float(nearest_sec),
        "nearest_candidate_abs_error_ms": abs(err_sec) * 1000.0,
        "nearest_candidate_abs_beat_error": abs_beats,
        "manual_in_candidates_0p125beat": bool(abs_beats is not None and abs_beats <= 0.125),
        "manual_in_candidates_0p25beat": bool(abs_beats is not None and abs_beats <= 0.25),
        "manual_in_candidates_0p50beat": bool(abs_beats is not None and abs_beats <= 0.50),
        "nearest_candidate_rank": int(rank),
    }


def _marker_analysis(
    manual_sec: float,
    bpm: Optional[float],
    marker_seconds: Sequence[Any],
    *,
    prefix: str,
) -> Dict[str, Any]:
    secs = [_safe_float(v) for v in marker_seconds]
    vals = [sec for sec in secs if sec is not None]
    if not vals:
        return {
            f"{prefix}_count": 0,
            f"nearest_{prefix}_seconds": None,
            f"nearest_{prefix}_abs_error_ms": None,
            f"nearest_{prefix}_abs_beat_error": None,
            f"manual_in_{prefix}_0p125beat": False,
            f"manual_in_{prefix}_0p25beat": False,
            f"manual_in_{prefix}_0p50beat": False,
        }

    nearest_sec = min(vals, key=lambda sec: abs(float(sec) - float(manual_sec)))
    beat_sec = (60.0 / float(bpm)) if bpm and bpm > 0.0 else None
    err_sec = float(nearest_sec) - float(manual_sec)
    abs_beats = abs(err_sec / beat_sec) if beat_sec else None
    return {
        f"{prefix}_count": int(len(vals)),
        f"nearest_{prefix}_seconds": float(nearest_sec),
        f"nearest_{prefix}_abs_error_ms": abs(err_sec) * 1000.0,
        f"nearest_{prefix}_abs_beat_error": abs_beats,
        f"manual_in_{prefix}_0p125beat": bool(abs_beats is not None and abs_beats <= 0.125),
        f"manual_in_{prefix}_0p25beat": bool(abs_beats is not None and abs_beats <= 0.25),
        f"manual_in_{prefix}_0p50beat": bool(abs_beats is not None and abs_beats <= 0.50),
    }


def _failure_mode(
    row: Dict[str, Any],
    expected_bpm: Optional[float],
) -> str:
    if row.get("skip_reason"):
        return str(row["skip_reason"])
    if row.get("detector_error"):
        return "detector_error"
    abs_ms = _safe_float(row.get("abs_error_ms"))
    abs_beats = _safe_float(row.get("abs_beat_error"))
    if abs_ms is None or abs_beats is None:
        return "unknown"
    if abs_ms <= 50.0 or abs_beats <= 0.125:
        return "pass"

    detector_bpm = _safe_float(row.get("detector_bpm"))
    if expected_bpm and detector_bpm and abs(float(detector_bpm) - float(expected_bpm)) > 0.5:
        return "bpm_mismatch"
    if row.get("manual_in_candidates_0p125beat"):
        return "ranking"
    if row.get("manual_in_candidates_0p50beat"):
        return "coarse_candidate_nearby"
    return "candidate_missing"


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "track_rel",
        "track_dir",
        "manual_als_path",
        "drums_path",
        "inst_path",
        "vocals_path",
        "expected_bpm",
        "detector_bpm",
        "predicted_seconds",
        "manual_seconds",
        "signed_error_ms",
        "abs_error_ms",
        "beat_error",
        "abs_beat_error",
        "confidence",
        "fallback_would_trigger",
        "pass_bucket",
        "pass_10ms",
        "pass_25ms",
        "pass_50ms",
        "pass_0p125beat",
        "pass_0p25beat",
        "candidate_count",
        "candidate_strategy",
        "custom_reference_seconds",
        "rough_custom_seconds",
        "rough_region_reason",
        "rough_custom_abs_error_ms",
        "snap_chosen_marker_seconds",
        "snap_first_plausible_marker_seconds",
        "snap_selected_event_score",
        "snap_selected_event_reason",
        "snap_accept_reason",
        "snap_self_check_passed",
        "snap_self_check_reason",
        "ableton_markers_available",
        "ableton_marker_count",
        "nearest_ableton_marker_seconds",
        "nearest_ableton_marker_abs_error_ms",
        "nearest_ableton_marker_abs_beat_error",
        "manual_in_ableton_marker_0p125beat",
        "manual_in_ableton_marker_0p25beat",
        "manual_in_ableton_marker_0p50beat",
        "nearest_candidate_seconds",
        "nearest_candidate_abs_error_ms",
        "nearest_candidate_abs_beat_error",
        "manual_in_candidates_0p125beat",
        "manual_in_candidates_0p25beat",
        "manual_in_candidates_0p50beat",
        "nearest_candidate_rank",
        "legacy_prior_seconds",
        "legacy_prior_confidence",
        "failure_mode",
        "skip_reason",
        "detector_error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_dynamic_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_training_rows(
    *,
    track_rel: str,
    result: Dict[str, Any],
    manual_sec: float,
    detector_bpm: Optional[float],
) -> List[Dict[str, Any]]:
    debug = result.get("debug") or {}
    snap = (debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}
    considered = list(snap.get("considered_markers") or [])
    if not considered:
        return []

    beat_sec = (60.0 / float(detector_bpm)) if detector_bpm and detector_bpm > 0.0 else None
    predicted_sec = _safe_float((result.get("drums") or {}).get("downbeat_seconds"))
    chosen_marker_sec = _safe_float(snap.get("chosen_marker_seconds"))
    rough_anchor_sec = _safe_float(snap.get("rough_anchor_seconds"))
    first_plausible_sec = _safe_float(snap.get("first_plausible_marker_seconds"))
    event_map = {
        int(event["event_id"]): event
        for event in list(snap.get("events") or [])
        if isinstance(event, dict) and _safe_int(event.get("event_id")) is not None
    }
    rows: List[Dict[str, Any]] = []
    for entry in considered:
        marker_sec = _safe_float(entry.get("marker_seconds"))
        if marker_sec is None:
            continue
        err_sec = float(marker_sec) - float(manual_sec)
        beat_error = (err_sec / beat_sec) if beat_sec else None
        event_id = _safe_int(entry.get("event_id"))
        event_row = event_map.get(int(event_id)) if event_id is not None else None
        row: Dict[str, Any] = {
            "track_rel": track_rel,
            "candidate_strategy": debug.get("candidate_strategy"),
            "manual_seconds": float(manual_sec),
            "predicted_seconds": predicted_sec,
            "rough_anchor_seconds": rough_anchor_sec,
            "first_plausible_marker_seconds": first_plausible_sec,
            "chosen_marker_seconds": chosen_marker_sec,
            "candidate_marker_seconds": float(marker_sec),
            "candidate_signed_error_ms": float(err_sec) * 1000.0,
            "candidate_abs_error_ms": abs(float(err_sec)) * 1000.0,
            "candidate_beat_error": beat_error,
            "candidate_abs_beat_error": abs(float(beat_error)) if beat_error is not None else None,
            "candidate_is_final_prediction": bool(predicted_sec is not None and abs(float(predicted_sec) - float(marker_sec)) <= 1e-3),
            "candidate_is_snap_choice": bool(chosen_marker_sec is not None and abs(float(chosen_marker_sec) - float(marker_sec)) <= 1e-3),
            "candidate_is_first_plausible": bool(first_plausible_sec is not None and abs(float(first_plausible_sec) - float(marker_sec)) <= 1e-3),
            "candidate_is_manual_match_25ms": bool(abs(float(err_sec)) * 1000.0 <= 25.0),
            "candidate_is_manual_match_0p125beat": bool(beat_error is not None and abs(float(beat_error)) <= 0.125),
            "event_id": event_id,
            "event_score": _safe_float(event_row.get("event_score")) if event_row else None,
            "event_anchor_seconds": _safe_float(event_row.get("anchor_seconds")) if event_row else None,
            "event_first_plausible_marker_seconds": _safe_float(event_row.get("first_plausible_marker_seconds")) if event_row else None,
            "event_marker_count": _safe_int(event_row.get("marker_count")) if event_row else None,
            "event_plausible_marker_count": _safe_int(event_row.get("plausible_marker_count")) if event_row else None,
            "event_selected": bool(entry.get("is_selected_event")),
            "self_check_passed": bool(snap.get("self_check_passed")) if snap.get("self_check_passed") is not None else None,
            "self_check_reason": snap.get("self_check_reason"),
            "rough_region_reason": debug.get("rough_custom_reason"),
            "selected_event_reason": snap.get("selected_event_reason"),
            "chosen_marker_reason": snap.get("chosen_marker_reason"),
            "accept_reason": snap.get("accept_reason"),
        }
        for key, value in entry.items():
            if key in row:
                continue
            row[f"feature_{key}"] = value
        rows.append(row)
    return rows


def _rough_candidate_training_rows(
    *,
    track_rel: str,
    result: Dict[str, Any],
    manual_sec: float,
    detector_bpm: Optional[float],
) -> List[Dict[str, Any]]:
    debug = result.get("debug") or {}
    custom_candidates = list(debug.get("custom_candidates") or [])
    if not custom_candidates:
        return []
    reference = (debug.get("custom_reference_candidate") or {}) if isinstance(debug.get("custom_reference_candidate"), dict) else {}
    rough = (debug.get("rough_custom_candidate") or {}) if isinstance(debug.get("rough_custom_candidate"), dict) else {}
    reference_sec = _safe_float(reference.get("time_abs_sec"))
    rough_sec = _safe_float(rough.get("time_abs_sec"))
    predicted_sec = _safe_float((result.get("drums") or {}).get("downbeat_seconds"))
    beat_sec = (60.0 / float(detector_bpm)) if detector_bpm and detector_bpm > 0.0 else None
    rows: List[Dict[str, Any]] = []
    for cand in custom_candidates:
        cand_sec = _safe_float(cand.get("time_abs_sec"))
        if cand_sec is None:
            continue
        err_sec = float(cand_sec) - float(manual_sec)
        beat_error = (err_sec / beat_sec) if beat_sec else None
        strength = (
            0.26 * float(_safe_float(cand.get("norm_onset")) or 0.0)
            + 0.22 * float(_safe_float(cand.get("norm_lowend")) or 0.0)
            + 0.18 * float(_safe_float(cand.get("norm_density")) or 0.0)
            + 0.12 * float(_safe_float(cand.get("norm_phrase")) or 0.0)
            + 0.10 * float(_safe_float(cand.get("norm_grid")) or 0.0)
            + 0.08 * float(_safe_float(cand.get("norm_ableton")) or 0.0)
            + 0.06 * float(_safe_float(cand.get("norm_sustain")) or 0.0)
            + 0.06 * float(_safe_float(cand.get("norm_legacy")) or 0.0)
        )
        penalty = (
            0.45 * float(_safe_float(cand.get("norm_preroll")) or 0.0)
            + 0.30 * float(_safe_float(cand.get("norm_weak")) or 0.0)
            + 0.25 * float(_safe_float(cand.get("norm_fake")) or 0.0)
        )
        candidate_confidence = 1.0 / (1.0 + math.exp(-((strength - penalty - 0.15) * 5.0)))
        if not bool(cand.get("valid")):
            candidate_confidence *= 0.80
        delta_beats_from_reference = None
        if reference_sec is not None and beat_sec:
            delta_beats_from_reference = (float(reference_sec) - float(cand_sec)) / float(beat_sec)
        rough_followthrough = (
            (0.34 * float(_safe_float(cand.get("norm_density")) or 0.0))
            + (0.24 * float(_safe_float(cand.get("norm_sustain")) or 0.0))
            + (0.22 * float(_safe_float(cand.get("norm_lowend")) or 0.0))
            + (0.20 * float(_safe_float(cand.get("norm_repeat")) or 0.0))
        )
        row: Dict[str, Any] = {
            "track_rel": track_rel,
            "manual_seconds": float(manual_sec),
            "predicted_seconds": predicted_sec,
            "custom_reference_seconds": reference_sec,
            "rough_choice_seconds": rough_sec,
            "rough_region_reason": debug.get("rough_custom_reason"),
            "candidate_seconds": float(cand_sec),
            "candidate_signed_error_ms": float(err_sec) * 1000.0,
            "candidate_abs_error_ms": abs(float(err_sec)) * 1000.0,
            "candidate_beat_error": beat_error,
            "candidate_abs_beat_error": abs(float(beat_error)) if beat_error is not None else None,
            "candidate_is_custom_reference": bool(reference_sec is not None and abs(float(reference_sec) - float(cand_sec)) <= 1e-3),
            "candidate_is_rough_choice": bool(rough_sec is not None and abs(float(rough_sec) - float(cand_sec)) <= 1e-3),
            "candidate_is_final_prediction": bool(predicted_sec is not None and abs(float(predicted_sec) - float(cand_sec)) <= 1e-3),
            "candidate_is_manual_match_0p25beat": bool(beat_error is not None and abs(float(beat_error)) <= 0.25),
            "candidate_is_manual_match_0p50beat": bool(beat_error is not None and abs(float(beat_error)) <= 0.50),
            "candidate_is_manual_match_1beat": bool(beat_error is not None and abs(float(beat_error)) <= 1.0),
            "feature_delta_beats_from_reference": delta_beats_from_reference,
            "feature_candidate_confidence": float(candidate_confidence),
            "feature_candidate_score": _safe_float(cand.get("score")),
            "feature_rough_followthrough": float(rough_followthrough),
            "feature_norm_phrase": _safe_float(cand.get("norm_phrase")),
            "feature_norm_density": _safe_float(cand.get("norm_density")),
            "feature_norm_sustain": _safe_float(cand.get("norm_sustain")),
            "feature_norm_lowend": _safe_float(cand.get("norm_lowend")),
            "feature_norm_contrast": _safe_float(cand.get("norm_contrast")),
            "feature_norm_grid": _safe_float(cand.get("norm_grid")),
            "feature_norm_onset": _safe_float(cand.get("norm_onset")),
            "feature_norm_repeat": _safe_float(cand.get("norm_repeat")),
            "feature_norm_preroll": _safe_float(cand.get("norm_preroll")),
            "feature_norm_weak": _safe_float(cand.get("norm_weak")),
            "feature_norm_fake": _safe_float(cand.get("norm_fake")),
        }
        rows.append(row)
    return rows


def _extract_library_root(library_path: str) -> Tuple[str, Optional[tempfile.TemporaryDirectory[str]]]:
    if zipfile.is_zipfile(library_path):
        temp_dir = tempfile.TemporaryDirectory(prefix="downbeat_eval_")
        with zipfile.ZipFile(library_path) as zf:
            zf.extractall(temp_dir.name)
        return temp_dir.name, temp_dir
    return library_path, None


def _load_track_manifest(manifest_path: str) -> List[str]:
    rows: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            path = str(line).strip()
            if not path or path.startswith("#"):
                continue
            rows.append(os.path.abspath(path))
    return rows


def _discover_tracks_from_manifest(manifest_path: str) -> Tuple[str, List[DiscoveredTrack]]:
    track_dirs = _load_track_manifest(manifest_path)
    if not track_dirs:
        raise ValueError(f"Track manifest is empty: {manifest_path}")
    common_root = os.path.commonpath(track_dirs)
    if os.path.isfile(common_root):
        common_root = os.path.dirname(common_root)
    discovered: List[DiscoveredTrack] = []
    missing: List[str] = []
    for track_dir in track_dirs:
        track = discover_track_folder(track_dir)
        if track is None:
            missing.append(track_dir)
            continue
        discovered.append(track)
    if not discovered:
        raise ValueError(f"No discoverable track folders found in manifest: {manifest_path}")
    if missing:
        LOG.warning("Skipped %d manifest entries that did not resolve to track folders.", len(missing))
    discovered.sort(key=lambda t: (t.bpm or 0.0, t.camelot_key or "", os.path.basename(t.track_dir).lower()))
    return common_root, discovered


def _bucket_summary(rows: Sequence[Dict[str, Any]], bucket_fn) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("abs_error_ms") is None:
            continue
        key = str(bucket_fn(row))
        grouped.setdefault(key, []).append(row)
    return {key: _summarize_error_rows(group) for key, group in sorted(grouped.items())}


def evaluate_library(
    library_path: Optional[str] = None,
    out_csv: Optional[str] = None,
    out_candidate_csv: Optional[str] = None,
    out_rough_candidate_csv: Optional[str] = None,
    out_json: Optional[str] = None,
    debug_dir: Optional[str] = None,
    worst_n: int = 50,
    confidence_threshold: float = 0.45,
    generate_plots: bool = False,
    options: Optional[DetectorOptions] = None,
    max_tracks: Optional[int] = None,
    track_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    if bool(library_path) == bool(track_manifest):
        raise ValueError("Provide exactly one of library_path or track_manifest.")

    temp_dir = None
    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    rough_candidate_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    try:
        if track_manifest:
            root_dir, tracks = _discover_tracks_from_manifest(track_manifest)
        else:
            assert library_path is not None
            root_dir, temp_dir = _extract_library_root(library_path)
            tracks = discover_library_tracks(root_dir)
        if max_tracks is not None and max_tracks > 0:
            tracks = tracks[: int(max_tracks)]
        source_desc = track_manifest if track_manifest else root_dir
        LOG.info("Evaluating %d discovered track folders from %s", len(tracks), source_desc)

        for idx, track in enumerate(tracks, start=1):
            track_rel = os.path.relpath(track.track_dir, root_dir)
            LOG.info("[%d/%d] %s", idx, len(tracks), track_rel)

            gt, gt_error = _extract_ground_truth(track.ch1_als_path)
            row: Dict[str, Any] = {
                "track_rel": track_rel,
                "track_dir": track.track_dir,
                "manual_als_path": track.ch1_als_path,
                "drums_path": track.drums_path,
                "inst_path": track.inst_path,
                "vocals_path": track.vocals_path,
                "expected_bpm": float(track.bpm) if track.bpm is not None else None,
                "manual_seconds": float(gt.seconds) if gt is not None else None,
                "manual_target_source": gt.target_source if gt is not None else None,
                "manual_marker_count": gt.marker_count if gt is not None else None,
                "skip_reason": None,
                "detector_error": None,
            }
            if gt_error is not None:
                row["skip_reason"] = str(gt_error)
                skipped[str(gt_error)] = skipped.get(str(gt_error), 0) + 1
                rows.append(row)
                continue

            try:
                result = detect_track_folder(
                    track_dir=track.track_dir,
                    debug_dir=None,
                    generate_plots=False,
                    options=options,
                )
            except Exception as exc:
                row["detector_error"] = str(exc)
                skipped["detector_error"] = skipped.get("detector_error", 0) + 1
                rows.append(row)
                LOG.exception("Detector failed for %s", track.track_dir)
                continue

            drums = result.get("drums") or {}
            debug = result.get("debug") or {}
            ableton_markers = (debug.get("ableton_markers") or {}) if isinstance(debug.get("ableton_markers"), dict) else {}
            custom_reference_candidate = (debug.get("custom_reference_candidate") or {}) if isinstance(debug.get("custom_reference_candidate"), dict) else {}
            rough_custom_candidate = (debug.get("rough_custom_candidate") or {}) if isinstance(debug.get("rough_custom_candidate"), dict) else {}
            detector_bpm = _safe_float(result.get("bpm"))
            pred_sec = _safe_float(drums.get("downbeat_seconds"))
            conf = _safe_float(drums.get("confidence"))
            beat_sec = (60.0 / detector_bpm) if detector_bpm and detector_bpm > 0.0 else None
            err_sec = (float(pred_sec) - float(gt.seconds)) if pred_sec is not None else None
            beat_error = (err_sec / beat_sec) if (err_sec is not None and beat_sec is not None) else None

            row.update(
                {
                    "detector_bpm": detector_bpm,
                    "predicted_seconds": pred_sec,
                    "confidence": conf,
                    "fallback_would_trigger": bool(conf is not None and float(conf) < float(confidence_threshold)),
                    "signed_error_ms": (float(err_sec) * 1000.0) if err_sec is not None else None,
                    "abs_error_ms": (abs(float(err_sec)) * 1000.0) if err_sec is not None else None,
                    "beat_error": beat_error,
                    "abs_beat_error": abs(float(beat_error)) if beat_error is not None else None,
                    "pass_10ms": bool(err_sec is not None and abs(float(err_sec)) * 1000.0 <= 10.0),
                    "pass_25ms": bool(err_sec is not None and abs(float(err_sec)) * 1000.0 <= 25.0),
                    "pass_50ms": bool(err_sec is not None and abs(float(err_sec)) * 1000.0 <= 50.0),
                    "pass_0p125beat": bool(beat_error is not None and abs(float(beat_error)) <= 0.125),
                    "pass_0p25beat": bool(beat_error is not None and abs(float(beat_error)) <= 0.25),
                    "pass_bucket": _pass_bucket(
                        (abs(float(err_sec)) * 1000.0) if err_sec is not None else None,
                        abs(float(beat_error)) if beat_error is not None else None,
                    ),
                    "legacy_prior_seconds": _safe_float(debug.get("legacy_prior_seconds")),
                    "legacy_prior_confidence": _safe_float(debug.get("legacy_prior_confidence")),
                    "chosen_reason": debug.get("chosen_reason"),
                    "candidate_strategy": debug.get("candidate_strategy"),
                    "custom_reference_seconds": _safe_float(custom_reference_candidate.get("time_abs_sec")),
                    "rough_custom_seconds": _safe_float(rough_custom_candidate.get("time_abs_sec")),
                    "rough_region_reason": debug.get("rough_custom_reason"),
                    "rough_custom_abs_error_ms": (
                        abs(float(_safe_float(rough_custom_candidate.get("time_abs_sec")) or 0.0) - float(gt.seconds)) * 1000.0
                        if _safe_float(rough_custom_candidate.get("time_abs_sec")) is not None
                        else None
                    ),
                    "snap_chosen_marker_seconds": _safe_float(((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("chosen_marker_seconds")),
                    "snap_first_plausible_marker_seconds": _safe_float(((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("first_plausible_marker_seconds")),
                    "snap_selected_event_score": _safe_float(((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("selected_event_score")),
                    "snap_selected_event_reason": (((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("selected_event_reason")),
                    "snap_accept_reason": (((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("accept_reason")),
                    "snap_self_check_passed": ((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("self_check_passed"),
                    "snap_self_check_reason": (((debug.get("ableton_snap") or {}) if isinstance(debug.get("ableton_snap"), dict) else {}).get("self_check_reason")),
                    "ableton_markers_available": bool(ableton_markers.get("candidate_seconds")),
                }
            )
            row.update(
                _candidate_analysis(
                    manual_sec=float(gt.seconds),
                    bpm=detector_bpm or track.bpm,
                    candidate_seconds=debug.get("candidate_seconds") or [],
                    candidate_scores=debug.get("candidate_scores") or [],
                )
            )
            row.update(
                _marker_analysis(
                    manual_sec=float(gt.seconds),
                    bpm=detector_bpm or track.bpm,
                    marker_seconds=ableton_markers.get("candidate_seconds") or [],
                    prefix="ableton_marker",
                )
            )
            row["failure_mode"] = _failure_mode(row, expected_bpm=track.bpm)
            rows.append(row)
            candidate_rows.extend(
                _candidate_training_rows(
                    track_rel=track_rel,
                    result=result,
                    manual_sec=float(gt.seconds),
                    detector_bpm=detector_bpm or track.bpm,
                )
            )
            rough_candidate_rows.extend(
                _rough_candidate_training_rows(
                    track_rel=track_rel,
                    result=result,
                    manual_sec=float(gt.seconds),
                    detector_bpm=detector_bpm or track.bpm,
                )
            )

        measured = [r for r in rows if r.get("abs_error_ms") is not None]
        misses = [r for r in measured if r.get("failure_mode") != "pass"]
        worst = sorted(misses, key=lambda r: float(r["abs_error_ms"]), reverse=True)[: max(0, int(worst_n))]
        if debug_dir and worst:
            os.makedirs(debug_dir, exist_ok=True)
            for rank, row in enumerate(worst, start=1):
                track_rel = str(row["track_rel"])
                item_dir = os.path.join(debug_dir, f"{rank:03d}_{_safe_slug(track_rel)}")
                os.makedirs(item_dir, exist_ok=True)
                with open(os.path.join(item_dir, "eval_row.json"), "w", encoding="utf-8") as fh:
                    json.dump(row, fh, indent=2, ensure_ascii=True, default=_json_default)
                try:
                    payload = detect_track_folder(
                        track_dir=str(row["track_dir"]),
                        out_path=os.path.join(item_dir, "detector_result.json"),
                        debug_dir=item_dir,
                        generate_plots=generate_plots,
                        options=options,
                    )
                    with open(os.path.join(item_dir, "detector_summary.json"), "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2, ensure_ascii=True, default=_json_default)
                except Exception as exc:
                    with open(os.path.join(item_dir, "detector_error.txt"), "w", encoding="utf-8") as fh:
                        fh.write(str(exc).strip() + "\n")

        summary = _summarize_error_rows(measured)
        summary.update(
            {
                "library_path": library_path,
                "track_manifest": os.path.abspath(track_manifest) if track_manifest else None,
                "evaluated_tracks": int(len(measured)),
                "total_discovered_tracks": int(len(tracks)),
                "skipped_tracks": int(sum(skipped.values())),
                "skipped_by_reason": skipped,
                "confidence_threshold": float(confidence_threshold),
                "failure_mode_counts": {
                    key: int(sum(1 for row in measured if row.get("failure_mode") == key))
                    for key in sorted({str(row.get("failure_mode")) for row in measured if row.get("failure_mode")})
                },
                "candidate_strategy_counts": {
                    key: int(sum(1 for row in measured if row.get("candidate_strategy") == key))
                    for key in sorted({str(row.get("candidate_strategy")) for row in measured if row.get("candidate_strategy")})
                },
            }
        )

        by_confidence_bucket = _bucket_summary(measured, lambda row: _confidence_bucket(_safe_float(row.get("confidence"))))
        by_bpm_bucket = _bucket_summary(measured, lambda row: _bpm_bucket(_safe_float(row.get("expected_bpm")) or _safe_float(row.get("detector_bpm"))))

        payload = {
            "summary": summary,
            "candidate_training_rows": int(len(candidate_rows)),
            "rough_candidate_training_rows": int(len(rough_candidate_rows)),
            "by_confidence_bucket": by_confidence_bucket,
            "by_bpm_bucket": by_bpm_bucket,
            "worst_misses": [
                {
                    "track_rel": row["track_rel"],
                    "abs_error_ms": row["abs_error_ms"],
                    "beat_error": row["beat_error"],
                    "confidence": row.get("confidence"),
                    "candidate_strategy": row.get("candidate_strategy"),
                    "failure_mode": row.get("failure_mode"),
                    "predicted_seconds": row.get("predicted_seconds"),
                    "manual_seconds": row.get("manual_seconds"),
                    "rough_custom_abs_error_ms": row.get("rough_custom_abs_error_ms"),
                    "nearest_candidate_abs_error_ms": row.get("nearest_candidate_abs_error_ms"),
                    "nearest_ableton_marker_abs_error_ms": row.get("nearest_ableton_marker_abs_error_ms"),
                    "nearest_candidate_rank": row.get("nearest_candidate_rank"),
                }
                for row in worst
            ],
            "tracks": rows,
            "tuning_notes": {
                "confidence_threshold": (
                    "Use the confidence buckets plus worst misses to decide whether the fallback gate should move "
                    "up or down. If high-confidence misses dominate, the scoring model needs work more than the gate."
                ),
                "candidate_failure_analysis": (
                    "If many failures are tagged as ranking, tune score weights and selection margins. "
                    "If many are tagged as coarse_candidate_nearby or candidate_missing, improve beat-grid generation "
                    "or sample refinement before weight tuning."
                ),
                "training_a_marker_ranker": (
                    "Use --out-candidate-csv to export one row per nearby Ableton marker. Train a simple binary ranker "
                    "on the candidate_is_manual_match_25ms / candidate_is_manual_match_0p125beat labels, then rank "
                    "markers within each track/event by the model score. A no-dependency trainer is included in "
                    "train_downbeat_marker_ranker.py and its JSON output can be loaded by the detector via "
                    "DOWNBEAT_MARKER_RANK_MODEL or DetectorOptions(marker_rank_model_path=...)."
                ),
                "training_a_rough_region_model": (
                    "Use --out-rough-candidate-csv to export one row per custom rough-region candidate. "
                    "Train a simple boundary classifier on candidate_is_manual_match_1beat (or 0p50beat for tighter labels). "
                    "A no-dependency trainer is included in train_rough_region_ranker.py and its JSON output can be loaded "
                    "by the detector via DOWNBEAT_ROUGH_REGION_MODEL or DetectorOptions(rough_region_model_path=...)."
                ),
            },
        }

        if out_csv:
            _write_csv(out_csv, rows)
        if out_candidate_csv:
            _write_dynamic_csv(out_candidate_csv, candidate_rows)
        if out_rough_candidate_csv:
            _write_dynamic_csv(out_rough_candidate_csv, rough_candidate_rows)
        if out_json:
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=True, default=_json_default)
        return payload
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch-evaluate the first downbeat detector against manual CH1 anchors.")
    ap.add_argument("--library", default=None, help="Root stem library folder or zip archive")
    ap.add_argument("--track-manifest", default=None, help="Text file with one absolute track directory per line")
    ap.add_argument("--out-csv", default=None, help="Per-track evaluation CSV")
    ap.add_argument("--out-candidate-csv", default=None, help="Per-nearby-marker training/evaluation CSV")
    ap.add_argument("--out-rough-candidate-csv", default=None, help="Per-custom-candidate rough-region training/evaluation CSV")
    ap.add_argument("--out-json", default=None, help="Evaluation summary JSON")
    ap.add_argument("--debug-dir", default=None, help="Directory for worst-failure detector debug artifacts")
    ap.add_argument("--worst-n", type=int, default=50, help="Number of worst failures to export into --debug-dir")
    ap.add_argument("--confidence-threshold", type=float, default=0.45, help="Fallback gate to evaluate")
    ap.add_argument("--plots", action="store_true", help="Generate plots for worst failures")
    ap.add_argument("--max-tracks", type=int, default=None, help="Optional cap for smoke-testing")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(message)s")

    if bool(args.library) == bool(args.track_manifest):
        raise SystemExit("Provide exactly one of --library or --track-manifest")

    payload = evaluate_library(
        library_path=args.library,
        out_csv=args.out_csv,
        out_candidate_csv=args.out_candidate_csv,
        out_rough_candidate_csv=args.out_rough_candidate_csv,
        out_json=args.out_json,
        debug_dir=args.debug_dir,
        worst_n=args.worst_n,
        confidence_threshold=args.confidence_threshold,
        generate_plots=bool(args.plots),
        options=DetectorOptions(),
        max_tracks=args.max_tracks,
        track_manifest=args.track_manifest,
    )

    console_summary = {
        "total_discovered_tracks": payload["summary"]["total_discovered_tracks"],
        "evaluated_tracks": payload["summary"]["evaluated_tracks"],
        "skipped_tracks": payload["summary"]["skipped_tracks"],
        "mean_abs_error_ms": payload["summary"]["mean_abs_error_ms"],
        "median_abs_error_ms": payload["summary"]["median_abs_error_ms"],
        "p95_abs_error_ms": payload["summary"]["p95_abs_error_ms"],
        "within_25ms_pct": payload["summary"]["within_25ms_pct"],
        "within_0p25beat_pct": payload["summary"]["within_0p25beat_pct"],
    }
    print(json.dumps(console_summary, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
