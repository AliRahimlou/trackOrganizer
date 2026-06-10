#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse


def _reexec_with_local_venv() -> None:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except Exception:
        return
    if current == target:
        return
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_with_local_venv()

from lxml import etree

from drop_aligner.als import modify_als
from drop_aligner.candidate_chooser import choose_learned_candidate, load_candidate_chooser_payload, predict_candidate_errors
from drop_aligner.exclusions import EXCLUDED_DIR_NAMES, row_has_excluded_path
from drop_aligner.historical_markers import HistoricalMarker, is_human_review_source, load_historical_markers
from drop_aligner.learning import closest_candidate_to_pick, log_correction
from drop_aligner.microalign import choose_microaligned_candidate, microalign_candidate_dicts, microalign_marker, should_auto_accept
from drop_aligner.musical_clock import bpm_clock_for_time, feature_grid_for_time, phrase_strength_for_bar
from drop_aligner.multistem import choose_multistem_candidate, find_stem_group
from drop_aligner.structure_map import analyze_track_structure
from drop_aligner.summary_rerank import rerank_summary_with_model
from drop_aligner.visual_first import visual_first_marker
from drop_aligner.waveform import WaveformCache
from review import _run_retrain
from verify_als import verify_als
from project_config import STEMS_ROOT_DIR


DEFAULT_LIBRARY_DIR = STEMS_ROOT_DIR
DEFAULT_SUMMARY_NAME = "drop_batch_summary.csv"
BPM_RE = re.compile(r"^(?:drums|drum|inst|instrumental|vocals|vocal|bass|other|full)_(\d{2,3})_", re.IGNORECASE)
TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}
PREVIEW_BEFORE_SEC = 20.0
PREVIEW_AFTER_SEC = 30.0
MAX_PREVIEW_DURATION_SEC = 120.0
WAVEFORM_STEM_ROLES = (
    ("drums", "Drums"),
    ("instrumental", "Instrumental"),
    ("vocals", "Vocals"),
)
POST_STRUCTURE_CHOOSER_PATH = Path(__file__).resolve().parent / "models" / "drop_post_structure_candidate_chooser.pkl"


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").strip().upper()
    return tier if tier in TIER_ORDER else "UNKNOWN"


def _infer_bpm_from_path(path: str) -> Optional[float]:
    audio = Path(str(path)).expanduser()
    match = BPM_RE.match(audio.name)
    if match:
        return float(match.group(1))
    for parent in [audio.parent, *audio.parents]:
        try:
            bpm = int(parent.name)
        except ValueError:
            continue
        if 60 <= bpm <= 220:
            return float(bpm)
    return None


def _local_lan_ip() -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        return str(ip) if ip and not ip.startswith("127.") else None
    except OSError:
        return None
    finally:
        sock.close()


def _startup_urls(host: str, port: int) -> Dict[str, str]:
    host_text = str(host or "0.0.0.0")
    local_url = f"http://127.0.0.1:{int(port)}"
    if host_text in {"0.0.0.0", "::", ""}:
        lan_ip = _local_lan_ip()
        urls = {"local": local_url}
        if lan_ip:
            urls["phone"] = f"http://{lan_ip}:{int(port)}"
        return urls
    return {"local": f"http://{host_text}:{int(port)}"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_metric(
    selected_candidate: Mapping[str, Any],
    feature_summary: Mapping[str, Any],
    key: str,
) -> float:
    value = selected_candidate.get(key)
    if value is None:
        for nested_name in ("drumprint", "full_groove", "microalign"):
            nested = selected_candidate.get(nested_name)
            if isinstance(nested, Mapping) and nested.get(key) is not None:
                value = nested.get(key)
                break
    if value is None:
        value = feature_summary.get(f"chosen_{key}")
    return _float_or_none(value) or 0.0


def _clip01(value: Any) -> float:
    number = _float_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, float(number)))


def _candidate_marker_time(candidate: Mapping[str, Any]) -> Optional[float]:
    micro = candidate.get("microalign")
    if isinstance(micro, Mapping):
        value = _float_or_none(micro.get("microaligned_time"))
        if value is not None and value > 0:
            return float(value)
    for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec"):
        value = _float_or_none(candidate.get(key))
        if value is not None and value > 0:
            return float(value)
    return None


def _candidate_boundary_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec"):
        value = _float_or_none(candidate.get(key))
        if value is not None and value > 0:
            return float(value)
    return _candidate_marker_time(candidate)


def _candidate_nested_value(candidate: Mapping[str, Any], key: str) -> Optional[float]:
    value = _float_or_none(candidate.get(key))
    if value is not None:
        return float(value)
    for nested_name in ("microalign", "drumprint", "full_groove"):
        nested = candidate.get(nested_name)
        if isinstance(nested, Mapping):
            value = _float_or_none(nested.get(key))
            if value is not None:
                return float(value)
    return None


def _candidate_evidence_score(candidate: Mapping[str, Any]) -> float:
    micro_conf = _candidate_nested_value(candidate, "micro_confidence") or 0.0
    snap_offset_ms = abs(_candidate_nested_value(candidate, "snap_offset_ms") or 0.0)
    fake = _candidate_nested_value(candidate, "fake_hit_penalty") or 0.0
    base = max(
        _candidate_nested_value(candidate, "candidate_chooser_score") or 0.0,
        _candidate_nested_value(candidate, "confidence_score") or 0.0,
        _candidate_nested_value(candidate, "score") or 0.0,
    )
    verifier = _candidate_nested_value(candidate, "auto_verifier_p_within_25ms") or 0.0
    drum = _candidate_nested_value(candidate, "drumprint_pattern_score") or 0.0
    groove = _candidate_nested_value(candidate, "sustained_full_groove_score") or 0.0
    immediate = _candidate_nested_value(candidate, "immediate_groove_start_score") or 0.0
    return (
        (0.28 * _clip01(micro_conf))
        + (0.20 * _clip01(base))
        + (0.18 * _clip01(drum))
        + (0.13 * _clip01(groove))
        + (0.08 * _clip01(immediate))
        + (0.08 * _clip01(verifier))
        - (0.08 * _clip01(fake))
        - (0.05 * _clip01(snap_offset_ms / 220.0))
    )


def _phrase_strength(even_bar: int) -> tuple[float, str]:
    return phrase_strength_for_bar(even_bar)


def _nearest_even_bar(value: float) -> int:
    return max(2, int(((float(value) / 2.0) + 0.5)) * 2)


def _bar_prior_for_time(time_sec: Optional[float], bpm: Optional[float], *, bar_zero_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if time_sec is None:
        return None
    clock_zero = float(bar_zero_sec) if bar_zero_sec is not None else 0.0
    clock = bpm_clock_for_time(time_sec, bpm, clock_zero_sec=clock_zero)
    if not clock:
        return None
    bpm_value = float(clock.get("bpm", 0.0) or 0.0)
    beat_sec = float(clock.get("beat_sec", 0.0) or 0.0)
    bar_sec = float(clock.get("bar_sec", 0.0) or 0.0)
    nearest_bar = int(clock.get("nearest_one_bar", 0) or 0)
    nearest_time = float(clock.get("nearest_one_time", 0.0) or 0.0)
    distance_sec = float(clock.get("one_distance_sec", 0.0) or 0.0)
    distance_beats = float(clock.get("one_distance_beats", 0.0) or 0.0)
    strength = float(clock.get("phrase_strength", 0.0) or 0.0)
    phrase = str(clock.get("phrase") or "off-one")
    if distance_beats <= 0.25:
        proximity = 1.0
    else:
        proximity = max(0.0, 1.0 - ((distance_beats - 0.25) / 1.25))
    alignment_score = strength * proximity
    grid = feature_grid_for_time(time_sec, bpm, bar_zero_sec=bar_zero_sec)
    return {
        "bpm": float(bpm_value),
        "bar_seconds": float(bar_sec),
        "bar_zero_sec": float(clock_zero),
        "bar_index": float(((float(time_sec) - clock_zero) / bar_sec) + 1.0 if bar_sec > 0 else 0.0),
        "nearest_even_bar": int(nearest_bar),
        "nearest_musical_bar": int(nearest_bar),
        "nearest_even_bar_time": float(nearest_time),
        "distance_sec": float(distance_sec),
        "distance_beats": float(distance_beats),
        "phrase": phrase,
        "phrase_strength": float(strength),
        "alignment_score": float(alignment_score),
        "bpm_clock": clock,
        "feature_grid_prior": grid or {},
    }


def _bpm_clock_for_time(time_sec: Optional[float], bpm: Optional[float], *, clock_zero_sec: float = 0.0) -> Optional[Dict[str, Any]]:
    return bpm_clock_for_time(time_sec, bpm, clock_zero_sec=float(clock_zero_sec or 0.0))


def _annotate_bar_prior(candidates: List[Mapping[str, Any]], bpm: Optional[float], *, bar_zero_sec: Optional[float] = None) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        evidence = _candidate_evidence_score(row)
        prior = _bar_prior_for_time(_candidate_marker_time(row), bpm, bar_zero_sec=bar_zero_sec)
        clock = (prior or {}).get("bpm_clock") if isinstance(prior, Mapping) else None
        if not isinstance(clock, Mapping):
            clock = _bpm_clock_for_time(_candidate_marker_time(row), bpm, clock_zero_sec=float(bar_zero_sec or 0.0))
        row["bar_prior_evidence_score"] = float(evidence)
        if clock:
            row["bpm_clock"] = clock
            evidence += (0.030 * float(clock.get("on_beat_score", 0.0) or 0.0)) + (0.060 * float(clock.get("on_one_score", 0.0) or 0.0))
        if prior:
            row["musical_bar_prior"] = prior
            if isinstance(prior.get("feature_grid_prior"), Mapping):
                row["feature_grid_prior"] = dict(prior.get("feature_grid_prior") or {})
            row["bar_prior_adjusted_score"] = float(evidence + (0.14 * float(prior.get("alignment_score", 0.0))))
        else:
            row["bar_prior_adjusted_score"] = float(evidence)
        annotated.append(row)
    return annotated


def _matching_bar_prior_candidate(candidates: List[Dict[str, Any]], time_sec: Optional[float]) -> Optional[Dict[str, Any]]:
    if time_sec is None:
        return None
    matches = [
        candidate
        for candidate in candidates
        if _candidate_marker_time(candidate) is not None and abs(float(_candidate_marker_time(candidate) or 0.0) - float(time_sec)) <= 0.060
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: -float(row.get("bar_prior_adjusted_score", 0.0) or 0.0))
    return matches[0]


def _review_required_auto_accept(reason: str, candidate: Mapping[str, Any], candidates: List[Dict[str, Any]], tier: str) -> Dict[str, Dict[str, Any]]:
    gates: Dict[str, Dict[str, Any]] = {}
    for gate_mode in ("conservative", "normal", "aggressive"):
        gate = should_auto_accept(candidate, mode=gate_mode, candidates=candidates[:10], confidence_tier=tier)
        gate = dict(gate) if isinstance(gate, Mapping) else {"auto_accept": False, "risk_flags": []}
        flags = list(gate.get("risk_flags") or [])
        if reason not in flags:
            flags.insert(0, reason)
        gate["auto_accept"] = False
        gate["risk_flags"] = flags
        gates[gate_mode] = gate
    return gates


def _apply_even_bar_prior(
    candidates: List[Mapping[str, Any]],
    suggestion: Mapping[str, Any],
    *,
    bpm: Optional[float],
    confidence_tier: str,
    bar_zero_sec: Optional[float] = None,
    allow_promotion: bool = True,
) -> Dict[str, Any]:
    annotated = _annotate_bar_prior(candidates, bpm, bar_zero_sec=bar_zero_sec)
    next_suggestion = dict(suggestion or {})
    tier = _normalize_tier(confidence_tier)
    current_candidate = next_suggestion.get("candidate")
    current_time = _float_or_none(next_suggestion.get("suggested_time"))
    if current_time is None and isinstance(current_candidate, Mapping):
        current_time = _candidate_marker_time(current_candidate)
    current_prior = _bar_prior_for_time(current_time, bpm, bar_zero_sec=bar_zero_sec)
    current_match = _matching_bar_prior_candidate(annotated, current_time)
    structure_locked = str(next_suggestion.get("mode") or "") in {"structure", "post_structure"}
    if isinstance(current_candidate, Mapping):
        selected_by = str(current_candidate.get("selected_by") or "")
        structure_role = str(current_candidate.get("structure_role") or "")
        structure_locked = (
            structure_locked
            or selected_by.startswith("structure_map")
            or selected_by == "post_structure_candidate_chooser"
            or structure_role in {"first_drop", "second_drop"}
        )
    if isinstance(current_candidate, Mapping) and (current_prior or current_match):
        merged = dict(current_candidate)
        if current_match:
            merged.update({key: value for key, value in current_match.items() if key.startswith("bar_prior_") or key in {"musical_bar_prior", "bpm_clock"}})
            current_prior = current_match.get("musical_bar_prior") or current_prior
        if current_prior:
            merged["musical_bar_prior"] = current_prior
        next_suggestion["candidate"] = merged
    if current_prior:
        next_suggestion["bar_prior"] = current_prior
    if current_match and isinstance(current_match.get("bpm_clock"), Mapping):
        next_suggestion["bpm_clock"] = current_match.get("bpm_clock")

    eligible: List[Dict[str, Any]] = []
    for candidate in annotated[:140]:
        prior = candidate.get("musical_bar_prior")
        if not isinstance(prior, Mapping):
            continue
        distance_beats = _float_or_none(prior.get("distance_beats"))
        if (distance_beats if distance_beats is not None else 99.0) > 0.85:
            continue
        if float(prior.get("alignment_score", 0.0) or 0.0) < 0.42:
            continue
        if (_candidate_nested_value(candidate, "fake_hit_penalty") or 0.0) > 0.78:
            continue
        eligible.append(candidate)
    high_phrase = [
        candidate
        for candidate in eligible
        if float((candidate.get("musical_bar_prior") or {}).get("phrase_strength", 0.0) or 0.0) >= 0.86
        and int((candidate.get("musical_bar_prior") or {}).get("nearest_musical_bar", 0) or 0) >= 5
    ]
    promotion_pool = high_phrase or eligible
    if promotion_pool:
        strongest_adjusted = max(float(row.get("bar_prior_adjusted_score", 0.0) or 0.0) for row in promotion_pool)
        adjusted_floor = max(0.40, strongest_adjusted - (0.18 if high_phrase else 0.10))
        near_best_phrase = [
            row
            for row in promotion_pool
            if float(row.get("bar_prior_adjusted_score", 0.0) or 0.0) >= adjusted_floor
            and float(row.get("bar_prior_evidence_score", 0.0) or 0.0) >= 0.28
        ]
    else:
        near_best_phrase = []
    near_best_phrase.sort(
        key=lambda row: (
            int((row.get("musical_bar_prior") or {}).get("nearest_musical_bar", 999999) or 999999),
            -float((row.get("musical_bar_prior") or {}).get("phrase_strength", 0.0) or 0.0),
            -float(row.get("bar_prior_adjusted_score", 0.0) or 0.0),
        )
    )
    eligible.sort(key=lambda row: (-float(row.get("bar_prior_adjusted_score", 0.0) or 0.0), _candidate_marker_time(row) or 999999.0))

    best = near_best_phrase[0] if near_best_phrase else (eligible[0] if eligible else None)
    current_evidence = _candidate_evidence_score(current_match or current_candidate) if isinstance((current_match or current_candidate), Mapping) else 0.0
    current_alignment = float((current_prior or {}).get("alignment_score", 0.0) or 0.0)
    current_adjusted = current_evidence + (0.14 * current_alignment)
    summary = {
        "enabled": bool(bpm),
        "bpm": _float_or_none(bpm),
        "candidate_count": len(annotated),
        "promoted": False,
        "promotion_locked": bool(structure_locked or not allow_promotion),
    }
    current_clock = _bpm_clock_for_time(current_time, bpm, clock_zero_sec=float(bar_zero_sec or 0.0))
    if not structure_locked and current_time is not None and current_clock:
        current_clock_bar = int(current_clock.get("nearest_one_bar", 0) or 0)
        current_one_distance_ms = float(current_clock.get("one_distance_ms", 0.0) or 0.0)
        if current_clock_bar >= 8 and current_one_distance_ms > 40.0:
            rescue_pool: List[Dict[str, Any]] = []
            for candidate in eligible:
                prior = candidate.get("musical_bar_prior") if isinstance(candidate.get("musical_bar_prior"), Mapping) else {}
                candidate_bar = int(prior.get("nearest_musical_bar", prior.get("nearest_even_bar", 0)) or 0)
                if candidate_bar <= 0 or candidate_bar < current_clock_bar - 2 or candidate_bar > current_clock_bar:
                    continue
                distance_beats = _float_or_none(prior.get("distance_beats"))
                if (distance_beats if distance_beats is not None else 99.0) > 0.16:
                    continue
                if float(candidate.get("bar_prior_evidence_score", 0.0) or 0.0) < 0.22:
                    continue
                rescue_pool.append(candidate)
            rescue_pool.sort(
                key=lambda row: (
                    -float((row.get("musical_bar_prior") or {}).get("phrase_strength", 0.0) or 0.0),
                    int((row.get("musical_bar_prior") or {}).get("nearest_musical_bar", 999999) or 999999),
                    int(row.get("rank", row.get("handcrafted_rank", 999999)) or 999999),
                )
            )
            rescue = rescue_pool[0] if rescue_pool else None
            rescue_time = _candidate_marker_time(rescue) if rescue else None
            beat_sec = 60.0 / max(1.0, float(bpm or 0.0))
            max_rescue_jump_sec = max(0.35, min(0.72, 1.15 * beat_sec))
            rescue_jump_sec = abs(float(rescue_time) - float(current_time)) if rescue_time is not None else None
            if rescue_jump_sec is not None and rescue_jump_sec > max_rescue_jump_sec:
                summary["skipped_rescue_large_jump_ms"] = round(float(rescue_jump_sec * 1000.0), 3)
            if (
                rescue
                and rescue_time is not None
                and rescue_jump_sec is not None
                and 0.010 < rescue_jump_sec <= max_rescue_jump_sec
            ):
                rescue_prior = rescue.get("musical_bar_prior") if isinstance(rescue.get("musical_bar_prior"), Mapping) else {}
                selected = dict(rescue)
                selected["selected_by"] = "musical_clock_rescue"
                micro = selected.get("microalign")
                next_suggestion.update(
                    {
                        "auto_place": False,
                        "review_needed": True,
                        "risk": True,
                        "candidate": selected,
                        "microalign": dict(micro) if isinstance(micro, Mapping) else None,
                        "suggested_time": float(rescue_time),
                        "score": float(rescue.get("bar_prior_adjusted_score", rescue.get("score", 0.0)) or 0.0),
                        "reason": (
                            f"musical-clock rescue moved off-ONE pick to bar "
                            f"{int(rescue_prior.get('nearest_musical_bar', 0) or 0)}; "
                            f"{next_suggestion.get('reason') or 'audio evidence stayed within threshold'}"
                        ),
                        "bar_prior": dict(rescue_prior),
                        "bpm_clock": rescue.get("bpm_clock"),
                        "auto_accept": _review_required_auto_accept("musical-clock rescue requires review", selected, annotated, tier),
                    }
                )
                summary["rescued_off_one"] = True
                summary["rescued_rank"] = selected.get("rank")
                current_time = float(rescue_time)
                current_prior = rescue_prior

    active_candidate = next_suggestion.get("candidate") if isinstance(next_suggestion.get("candidate"), Mapping) else {}
    active_time = _float_or_none(next_suggestion.get("suggested_time")) or _candidate_marker_time(active_candidate)
    active_snap_offset_ms = abs(_candidate_nested_value(active_candidate, "snap_offset_ms") or 0.0)
    if not structure_locked and active_time is not None and active_snap_offset_ms >= 50.0:
        boundary_alternates: List[Dict[str, Any]] = []
        for candidate in annotated:
            candidate_time = _candidate_marker_time(candidate)
            if candidate_time is None:
                continue
            if abs(float(candidate_time) - float(active_time)) < 0.010 or abs(float(candidate_time) - float(active_time)) > 0.080:
                continue
            selected_by = str(candidate.get("selected_by") or "")
            reason = str(candidate.get("reason") or "")
            roles = {str(role) for role in candidate.get("multistem_roles", [])} if isinstance(candidate.get("multistem_roles"), list) else set()
            if selected_by != "structure_map" and "saved" not in roles and "saved" not in reason:
                continue
            boundary_alternates.append(candidate)
        boundary_alternates.sort(
            key=lambda row: (
                abs(float(_candidate_marker_time(row) or active_time) - float(active_time)),
                -float(row.get("bar_prior_evidence_score", 0.0) or 0.0),
            )
        )
        boundary = boundary_alternates[0] if boundary_alternates else None
        boundary_time = _candidate_marker_time(boundary) if boundary else None
        if boundary and boundary_time is not None:
            selected = dict(boundary)
            selected["selected_by"] = "cluster_boundary_resnap"
            next_suggestion.update(
                {
                    "auto_place": False,
                    "review_needed": True,
                    "risk": True,
                    "candidate": selected,
                    "microalign": selected.get("microalign") if isinstance(selected.get("microalign"), Mapping) else None,
                    "suggested_time": float(boundary_time),
                    "score": float(selected.get("bar_prior_adjusted_score", selected.get("score", 0.0)) or 0.0),
                    "reason": (
                        f"cluster boundary resnap kept nearby structure/saved boundary after MicroSnap moved "
                        f"{active_snap_offset_ms:.1f} ms; {next_suggestion.get('reason') or ''}"
                    ),
                    "bpm_clock": selected.get("bpm_clock"),
                    "auto_accept": _review_required_auto_accept("cluster-boundary resnap requires review", selected, annotated, tier),
                }
            )
            summary["cluster_boundary_resnap"] = True
            summary["cluster_boundary_resnap_rank"] = selected.get("rank")
            current_time = float(boundary_time)

    if best:
        best_prior = best.get("musical_bar_prior") if isinstance(best.get("musical_bar_prior"), Mapping) else {}
        summary.update(
            {
                "best_rank": best.get("rank"),
                "best_nearest_even_bar": best_prior.get("nearest_even_bar"),
                "best_distance_beats": best_prior.get("distance_beats"),
                "best_alignment_score": best_prior.get("alignment_score"),
            }
        )
        best_time = _candidate_marker_time(best)
        same_marker = current_time is not None and best_time is not None and abs(float(best_time) - float(current_time)) <= 0.060
        best_evidence = float(best.get("bar_prior_evidence_score", 0.0) or 0.0)
        best_adjusted = float(best.get("bar_prior_adjusted_score", 0.0) or 0.0)
        current_distance_beats = _float_or_none((current_prior or {}).get("distance_beats")) if isinstance(current_prior, Mapping) else None
        current_weak_bar = current_prior is None or (current_distance_beats if current_distance_beats is not None else 99.0) > 1.10 or current_alignment < 0.30
        comparable_evidence = best_evidence >= current_evidence - 0.10
        clearly_better_adjusted = best_adjusted >= current_adjusted + 0.06 and best_evidence >= current_evidence - 0.16
        if allow_promotion and not structure_locked and not same_marker and best_time is not None and (
            current_time is None or (current_weak_bar and comparable_evidence) or clearly_better_adjusted
        ):
            selected = dict(best)
            selected["selected_by"] = "musical_clock_prior"
            micro = selected.get("microalign")
            reason = (
                f"musical-clock prior promoted candidate near bar {int(best_prior.get('nearest_musical_bar', best_prior.get('nearest_even_bar', 0)) or 0)} "
                f"({float(best_prior.get('distance_beats', 0.0) or 0.0):.2f} beats off); "
                f"{next_suggestion.get('reason') or 'audio evidence stayed within threshold'}"
            )
            next_suggestion.update(
                {
                    "auto_place": False,
                    "review_needed": True,
                    "risk": True,
                    "candidate": selected,
                    "microalign": dict(micro) if isinstance(micro, Mapping) else None,
                    "suggested_time": float(best_time),
                    "score": float(best_adjusted),
                    "reason": reason,
                    "bar_prior": dict(best_prior),
                    "auto_accept": _review_required_auto_accept("bar-prior promotion requires review", selected, annotated, tier),
                }
            )
            summary["promoted"] = True
            summary["promoted_rank"] = selected.get("rank")

    return {
        "candidates": annotated,
        "suggestion": next_suggestion,
        "bar_prior": summary,
    }


def _restricted_structure_microalign(
    audio_path: str,
    candidate: Mapping[str, Any],
    beat_sec: float,
    *,
    bpm: Optional[float] = None,
    clock_zero_sec: float = 0.0,
) -> Dict[str, Any]:
    marker = _candidate_boundary_time(candidate)
    if marker is None or not audio_path:
        return {
            "ok": False,
            "used": False,
            "microaligned_time": marker,
            "reason": "structure marker kept; no marker available for MicroSnap",
        }
    limit = min(0.040, max(0.020, 0.25 * max(1e-6, float(beat_sec))))
    marker_clock = _bpm_clock_for_time(marker, bpm, clock_zero_sec=float(clock_zero_sec or 0.0))
    try:
        micro = microalign_marker(audio_path, float(marker))
        micro = dict(micro)
        micro["ok"] = True
    except Exception as exc:
        return {
            "ok": False,
            "used": False,
            "input_candidate_time": float(marker),
            "microaligned_time": float(marker),
            "snap_offset_ms": 0.0,
            "micro_confidence": 0.0,
            "reason": f"structure marker kept; MicroSnap failed ({str(exc) or exc.__class__.__name__})",
            "review_needed": True,
        }
    micro_time = _float_or_none(micro.get("microaligned_time"))
    offset_sec = abs(float(micro_time) - float(marker)) if micro_time is not None else float("inf")
    micro_conf = _float_or_none(micro.get("micro_confidence")) or 0.0
    micro_clock = _bpm_clock_for_time(micro_time, bpm, clock_zero_sec=float(clock_zero_sec or 0.0))
    same_one = (
        marker_clock is None
        or micro_clock is None
        or int(marker_clock.get("nearest_one_bar", 0) or 0) == int(micro_clock.get("nearest_one_bar", 0) or 0)
    )
    micro_on_one = micro_clock is None or bool(micro_clock.get("on_one"))
    used = bool(micro_time is not None and offset_sec <= limit and micro_conf >= 0.45 and same_one and micro_on_one)
    if marker_clock:
        micro["input_bpm_clock"] = marker_clock
    if micro_clock:
        micro["bpm_clock"] = micro_clock
    if used:
        micro["used"] = True
        micro["review_needed"] = True
        micro["reason"] = str(micro.get("reason") or "MicroSnap refined inside structural downbeat window")
        return micro
    micro["used"] = False
    micro["microaligned_time"] = float(marker)
    micro["snap_offset_ms"] = float((micro_time - float(marker)) * 1000.0) if micro_time is not None else 0.0
    micro["review_needed"] = True
    micro["reason"] = (
        "structure marker kept; MicroSnap candidate was outside the ONE structure window"
        if micro_time is not None
        else "structure marker kept; MicroSnap did not return a refined marker"
    )
    return micro


def _candidate_dedupe_priority(candidate: Mapping[str, Any]) -> float:
    selected_by = str(candidate.get("selected_by") or "")
    source = str(candidate.get("source") or "")
    structure_role = str(candidate.get("structure_role") or "")
    priority = 0.0
    if selected_by in {"historical_human_marker", "historical_review_memory"}:
        priority = max(priority, 130.0)
    if selected_by == "post_structure_candidate_chooser":
        priority = max(priority, 110.0)
    if selected_by == "visual_structure_section_guard":
        priority = max(priority, 109.5)
    if selected_by == "visual_gui_first_fat_block":
        priority = max(priority, 109.0)
    if selected_by == "visual_primary_phrase_prior":
        priority = max(priority, 108.0)
    if selected_by == "candidate_chooser":
        priority = max(priority, 100.0)
    if _candidate_nested_value(candidate, "candidate_chooser_probability") is not None:
        priority = max(priority, 96.0)
    if _candidate_nested_value(candidate, "candidate_chooser_score") is not None:
        priority = max(priority, 94.0)
    if source == "saved_closest_to_review_pick" or selected_by.startswith("saved_"):
        priority = max(priority, 90.0)
    if selected_by == "multistem_candidate":
        priority = max(priority, 86.0)
    if selected_by == "micro_boundary_variant" or source.startswith("micro_boundary"):
        priority = max(priority, 78.0)
    if selected_by.startswith("structure_map") or structure_role in {"first_drop", "second_drop"}:
        priority = max(priority, 66.0)
    if selected_by == "track_zero_grid_variant" or source == "track_zero_grid_variant":
        priority = max(priority, 54.0)
    if selected_by == "model":
        priority = max(priority, 44.0)
    if selected_by == "handcrafted":
        priority = max(priority, 36.0)
    score = _candidate_nested_value(candidate, "score") or _candidate_nested_value(candidate, "confidence_score") or 0.0
    return float(priority + min(0.99, max(0.0, float(score))))


def _is_historical_candidate(candidate: Mapping[str, Any]) -> bool:
    selected_by = str(candidate.get("selected_by") or "")
    source = str(candidate.get("source") or "")
    reason = str(candidate.get("reason") or "")
    return bool(
        selected_by in {"historical_human_marker", "historical_review_memory"}
        or source in {"historical_human_marker", "historical_review_memory"}
        or reason.startswith("historical ")
    )


def _without_historical_candidates(candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping) and not _is_historical_candidate(candidate)
    ]


def _candidate_phrase_prior(candidate: Mapping[str, Any], bpm: Optional[float], *, bar_zero_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    prior = candidate.get("musical_bar_prior") if isinstance(candidate.get("musical_bar_prior"), Mapping) else None
    if prior:
        return dict(prior)
    return _bar_prior_for_time(_candidate_marker_time(candidate), bpm, bar_zero_sec=bar_zero_sec)


def _primary_visual_phrase_candidate(
    candidates: List[Mapping[str, Any]],
    current_candidate: Optional[Mapping[str, Any]],
    *,
    bpm: Optional[float],
    bar_zero_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not bpm:
        return None
    current_time = _candidate_marker_time(current_candidate) if isinstance(current_candidate, Mapping) else None
    current_prior = _bar_prior_for_time(current_time, bpm, bar_zero_sec=bar_zero_sec)
    current_bar = int((current_prior or {}).get("nearest_musical_bar", 0) or 0)
    current_distance = _float_or_none((current_prior or {}).get("distance_beats"))
    current_strength = float((current_prior or {}).get("phrase_strength", 0.0) or 0.0)
    current_distance_value = current_distance if current_distance is not None else 99.0
    current_is_primary = bool(
        current_bar >= 25
        and current_bar <= 49
        and (current_bar - 1) % 32 == 0
        and current_distance_value <= 0.16
        and current_strength >= 0.94
    )
    if current_is_primary:
        return None
    current_evidence = _candidate_evidence_score(current_candidate) if isinstance(current_candidate, Mapping) else 0.0
    current_score = (
        _candidate_nested_value(current_candidate or {}, "score")
        or _candidate_nested_value(current_candidate or {}, "confidence_score")
        or 0.0
    )
    current_selected_by = str((current_candidate or {}).get("selected_by") or "")
    current_visual_components = (
        current_candidate.get("visual_components")
        if isinstance(current_candidate, Mapping) and isinstance(current_candidate.get("visual_components"), Mapping)
        else {}
    )
    current_visual_clock_bar = int(current_visual_components.get("clock_bar", current_bar) or current_bar)
    current_visual_primary_bar = bool(
        current_visual_clock_bar >= 25
        and current_visual_clock_bar <= 49
        and (current_visual_clock_bar - 1) % 32 == 0
    )
    current_visual_body_used = False
    current_micro = 0.0
    if isinstance(current_candidate, Mapping):
        current_micro_map = current_candidate.get("microalign") if isinstance(current_candidate.get("microalign"), Mapping) else {}
        current_micro = (
            _float_or_none(current_micro_map.get("micro_confidence"))
            or _float_or_none(current_candidate.get("micro_confidence"))
            or 0.0
        )
        current_visual_body_used = bool(current_micro_map.get("visual_body_onset_used"))
    if (
        current_selected_by == "visual_gui_first_fat_block"
        and (current_visual_primary_bar or 25 <= current_bar <= 49)
        and float(current_score) >= 0.55
        and (
            current_visual_body_used
            or _float_or_none(current_visual_components.get("post_bass8")) is not None
            and (_float_or_none(current_visual_components.get("post_bass8")) or 0.0) >= 0.45
            or (_float_or_none(current_visual_components.get("post_drum_cont8")) or 0.0) >= 0.86
        )
    ):
        return None
    if (
        9 <= current_bar <= 21
        and current_distance_value <= 0.16
        and current_strength >= 0.75
        and float(current_score) >= 0.68
        and float(current_micro) >= 0.80
    ):
        return None
    current_secondary_phrase = bool(
        current_bar >= 25
        and current_bar <= 49
        and current_distance_value <= 0.16
        and current_strength >= 0.86
    )
    current_early_or_weak = bool(
        current_bar <= 21
        or current_bar == 0
        or current_distance_value > 0.25
        or current_strength < 0.94
    )

    primary_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        marker = _candidate_marker_time(candidate)
        if marker is None:
            continue
        prior = _candidate_phrase_prior(candidate, bpm, bar_zero_sec=bar_zero_sec)
        if not prior:
            continue
        bar = int(prior.get("nearest_musical_bar", 0) or 0)
        distance = _float_or_none(prior.get("distance_beats"))
        strength = float(prior.get("phrase_strength", 0.0) or 0.0)
        if bar < 25 or bar > 49 or (bar - 1) % 32 != 0:
            continue
        if (distance if distance is not None else 99.0) > 0.16 or strength < 0.94:
            continue
        evidence = _candidate_evidence_score(candidate)
        micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
        micro_conf = _float_or_none(micro.get("micro_confidence")) or _float_or_none(candidate.get("micro_confidence")) or 0.0
        candidate_score = _candidate_nested_value(candidate, "score") or _candidate_nested_value(candidate, "confidence_score") or 0.0
        clean_primary_phrase_rescue = bool(
            current_secondary_phrase
            and current_time is not None
            and marker > current_time
            and micro_conf >= 0.90
            and candidate_score >= max(0.68, float(current_score) - 0.12)
            and evidence >= current_evidence - 0.14
        )
        if evidence < max(0.46, current_evidence - 0.18) and micro_conf < 0.80 and not clean_primary_phrase_rescue:
            continue
        if not current_early_or_weak and evidence < current_evidence + 0.06 and not clean_primary_phrase_rescue:
            continue
        row = dict(candidate)
        row["musical_bar_prior"] = dict(prior)
        if isinstance(prior.get("bpm_clock"), Mapping):
            row["bpm_clock"] = dict(prior.get("bpm_clock") or {})
        row["bar_prior_evidence_score"] = float(evidence)
        row["bar_prior_adjusted_score"] = float(evidence + (0.14 * float(prior.get("alignment_score", 0.0) or 0.0)))
        row["_visual_primary_bar"] = bar
        row["_visual_primary_micro"] = float(micro_conf)
        primary_rows.append(row)
    if not primary_rows:
        return None

    primary_rows.sort(
        key=lambda row: (
            -float((row.get("musical_bar_prior") or {}).get("phrase_strength", 0.0) or 0.0),
            -float(row.get("_visual_primary_micro", 0.0) or 0.0),
            -float(row.get("bar_prior_adjusted_score", 0.0) or 0.0),
            int(row.get("rank", row.get("handcrafted_rank", 999999)) or 999999),
        )
    )
    selected = dict(primary_rows[0])
    selected.pop("_visual_primary_bar", None)
    selected.pop("_visual_primary_micro", None)
    selected["selected_by"] = "visual_primary_phrase_prior"
    selected["structure_role"] = "first_drop"
    selected["section_label"] = "first_drop"
    selected["reason"] = (
        f"visual 32-bar phrase prior selected over early/off-phrase marker; "
        f"{selected.get('reason') or 'candidate at primary phrase start'}"
    )
    return selected


def _dedupe_candidate_dicts(candidates: List[Mapping[str, Any]], *, radius_sec: float = 0.010) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        t = _candidate_marker_time(row)
        duplicate_index: Optional[int] = None
        if t is not None:
            for existing_index, existing in enumerate(out):
                existing_time = _candidate_marker_time(existing)
                if existing_time is not None and abs(float(t) - float(existing_time)) <= float(radius_sec):
                    duplicate_index = existing_index
                    break
        if duplicate_index is not None:
            if _candidate_dedupe_priority(row) > _candidate_dedupe_priority(out[duplicate_index]):
                out[duplicate_index] = row
            continue
        out.append(row)
    for rank, row in enumerate(out, start=1):
        row["rank"] = int(rank)
        row["handcrafted_rank"] = int(rank)
    return out


def _boundary_variant_candidates(
    candidates: List[Mapping[str, Any]],
    *,
    max_candidates: int = 80,
    max_offset_sec: float = 0.250,
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    fields = (
        ("attack_start_time", "attack_start", 0.82),
        ("peak_time", "rms_peak", 0.88),
        ("zero_crossing_time", "zero_crossing", 0.76),
        ("visual_onset_knee_time", "visual_onset_knee", 0.84),
        ("centerline_boundary_time", "centerline_boundary", 0.80),
        ("ableton_asd_time", "ableton_asd", 0.86),
    )
    for candidate in list(candidates)[: max(0, int(max_candidates))]:
        marker = _candidate_marker_time(candidate)
        micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
        if marker is None or not isinstance(micro, Mapping):
            continue
        input_time = _float_or_none(micro.get("input_candidate_time")) or _float_or_none(candidate.get("coarse_timestamp")) or marker
        input_boundary_time = _float_or_none(micro.get("input_candidate_time"))
        input_boundary_quality = _float_or_none(micro.get("input_boundary_quality")) or 0.0
        input_boundary_used = bool(float(micro.get("input_boundary_used") or 0.0) > 0.0)
        if input_boundary_time is not None and (input_boundary_used or input_boundary_quality >= 0.72):
            fields_to_emit = (("input_candidate_time", "input_boundary", 0.86),) + fields
        else:
            fields_to_emit = fields
        for field, label, confidence_floor in fields_to_emit:
            value = _float_or_none(micro.get(field))
            if value is None or value <= 0.0:
                continue
            if abs(float(value) - float(marker)) > float(max_offset_sec) and abs(float(value) - float(input_time)) > float(max_offset_sec):
                continue
            row = dict(candidate)
            row["timestamp"] = float(value)
            row["snapped_sec"] = float(value)
            row["microaligned_time"] = float(value)
            row["selected_by"] = "micro_boundary_variant"
            row["boundary_variant"] = label
            row["boundary_variant_source"] = field
            row["reason"] = f"{label} boundary variant from MicroSnap analysis; {candidate.get('reason') or ''}".strip()
            row["source"] = f"micro_boundary_{label}"
            variant_micro = dict(micro)
            variant_micro["boundary_variant"] = label
            variant_micro["boundary_variant_source"] = field
            variant_micro["microaligned_time"] = float(value)
            variant_micro["snap_offset_ms"] = float((float(value) - float(input_time)) * 1000.0)
            variant_micro["micro_confidence"] = max(
                _float_or_none(variant_micro.get("micro_confidence")) or 0.0,
                float(confidence_floor),
            )
            variant_micro["reason"] = f"{label} boundary variant retained for selector"
            row["microalign"] = variant_micro
            row["micro_confidence"] = float(variant_micro["micro_confidence"])
            row["snap_offset_ms"] = float(variant_micro["snap_offset_ms"])
            variants.append(row)
    return variants


def _structure_boundary_variant_candidates(
    audio_path: str,
    candidates: List[Mapping[str, Any]],
    *,
    beat_sec: float,
    bpm: Optional[float],
    clock_zero_sec: float,
    max_structure_candidates: int = 4,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used = 0
    for candidate in candidates:
        if used >= int(max_structure_candidates):
            break
        selected_by = str(candidate.get("selected_by") or "")
        role = str(candidate.get("structure_role") or "")
        if not (selected_by.startswith("structure_map") or bool(role)):
            continue
        if isinstance(candidate.get("microalign"), Mapping):
            continue
        marker = _candidate_marker_time(candidate)
        if marker is None:
            continue
        row = dict(candidate)
        try:
            micro = _restricted_structure_microalign(audio_path, row, beat_sec, bpm=bpm, clock_zero_sec=clock_zero_sec)
        except Exception:
            continue
        row["microalign"] = dict(micro)
        row["micro_confidence"] = _float_or_none(micro.get("micro_confidence")) or 0.0
        row["snap_offset_ms"] = _float_or_none(micro.get("snap_offset_ms")) or 0.0
        out.append(row)
        out.extend(_boundary_variant_candidates([row], max_candidates=1))
        used += 1
    return out


def _track_zero_grid_variant_candidates(
    structure: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    *,
    max_source_candidates: int = 48,
    max_variants: int = 180,
    window_sec: float = 1.05,
    subdivisions_per_beat: int = 16,
) -> List[Dict[str, Any]]:
    beatgrid = structure.get("beatgrid") if isinstance(structure.get("beatgrid"), Mapping) else {}
    audio_path = str(structure.get("audio_path") or "")
    bpm_value = _float_or_none(beatgrid.get("bpm")) or _infer_bpm_from_path(audio_path)
    if bpm_value is None or bpm_value <= 0.0:
        return []
    beat_sec = 60.0 / float(bpm_value)
    step_sec = beat_sec / max(1, int(subdivisions_per_beat))
    if step_sec <= 0.0:
        return []

    out: List[Dict[str, Any]] = []
    seen: List[float] = []
    sources_used = 0

    def priority(candidate: Mapping[str, Any]) -> tuple[int, int, float]:
        selected_by = str(candidate.get("selected_by") or "")
        source = str(candidate.get("source") or "")
        reason = str(candidate.get("reason") or "")
        structure_role = str(candidate.get("structure_role") or "")
        rank = int(_float_or_none(candidate.get("rank")) or 9999)
        marker = _candidate_marker_time(candidate)
        if selected_by == "saved_closest_to_review_pick" or source == "saved_closest_to_review_pick":
            group = 0
        elif (structure_role or selected_by.startswith("structure_map")) and rank <= 12:
            group = 1
        elif selected_by.startswith("saved") or source.startswith("saved") or "saved" in reason:
            group = 2
        elif structure_role or selected_by.startswith("structure_map"):
            group = 3
        else:
            group = 4
        return (group, rank, float(marker if marker is not None else 999999.0))

    for candidate in sorted([candidate for candidate in candidates if isinstance(candidate, Mapping)], key=priority):
        if sources_used >= int(max_source_candidates) or len(out) >= int(max_variants):
            break
        marker = _candidate_marker_time(candidate)
        if marker is None:
            continue
        roles = candidate.get("roles") if isinstance(candidate.get("roles"), Sequence) and not isinstance(candidate.get("roles"), (str, bytes)) else []
        selected_by = str(candidate.get("selected_by") or "")
        source = str(candidate.get("source") or "")
        reason = str(candidate.get("reason") or "")
        structure_role = str(candidate.get("structure_role") or "")
        keep_source = bool(
            structure_role
            or selected_by.startswith("structure_map")
            or selected_by.startswith("saved")
            or source.startswith("saved")
            or "saved" in reason
            or any(str(role) == "saved" for role in roles)
        )
        if not keep_source:
            continue
        sources_used += 1
        effective_window_sec = float(window_sec)
        if structure_role or selected_by.startswith("structure_map"):
            effective_window_sec = min(effective_window_sec, 0.120)
        start_index = int(math.floor((float(marker) - effective_window_sec) / step_sec))
        end_index = int(math.ceil((float(marker) + effective_window_sec) / step_sec))
        for grid_index in range(start_index, end_index + 1):
            if len(out) >= int(max_variants):
                break
            grid_time = float(grid_index) * step_sec
            if grid_time < 0.0:
                continue
            if abs(grid_time - float(marker)) > effective_window_sec:
                continue
            if any(abs(grid_time - existing) <= 0.010 for existing in seen):
                continue
            clock = _bpm_clock_for_time(grid_time, bpm_value, clock_zero_sec=0.0)
            row = dict(candidate)
            row["timestamp"] = float(grid_time)
            row["snapped_sec"] = float(grid_time)
            row["microaligned_time"] = float(grid_time)
            row["selected_by"] = "track_zero_grid_variant"
            row["boundary_variant"] = "track_zero_subbeat"
            row["boundary_variant_source"] = "title_bpm_track_zero_grid"
            row["source"] = "track_zero_grid_variant"
            row["reason"] = f"track-zero BPM grid variant near {selected_by or source or 'candidate'}"
            if clock:
                row["bpm_clock"] = clock
            micro = dict(row.get("microalign") if isinstance(row.get("microalign"), Mapping) else {})
            micro["ok"] = True
            micro["used"] = False
            micro["boundary_variant"] = "track_zero_subbeat"
            micro["boundary_variant_source"] = "title_bpm_track_zero_grid"
            micro["input_candidate_time"] = float(marker)
            micro["microaligned_time"] = float(grid_time)
            micro["snap_offset_ms"] = float((grid_time - float(marker)) * 1000.0)
            micro["micro_confidence"] = max(0.50, _float_or_none(micro.get("micro_confidence")) or 0.0)
            micro["reason"] = "track-zero BPM grid candidate retained for selector"
            row["microalign"] = micro
            row["micro_confidence"] = float(micro["micro_confidence"])
            row["snap_offset_ms"] = float(micro["snap_offset_ms"])
            out.append(row)
            seen.append(grid_time)
    return out


def _merge_structure_candidates(structure: Mapping[str, Any], candidates: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Mapping[str, Any]] = []
    for role in ("first_drop", "second_drop"):
        candidate = structure.get(role)
        if isinstance(candidate, Mapping):
            row = dict(candidate)
            row["structure_role"] = role
            row["section_label"] = role
            merged.append(row)
    merged.extend(list(structure.get("candidates") or [])[:40])
    merged.extend(candidates)
    merged.extend(_boundary_variant_candidates([candidate for candidate in merged if isinstance(candidate, Mapping)]))
    merged.extend(_track_zero_grid_variant_candidates(structure, [candidate for candidate in merged if isinstance(candidate, Mapping)]))
    return _dedupe_candidate_dicts(merged)


def _structure_component(candidate: Mapping[str, Any], key: str) -> Optional[float]:
    components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
    return _float_or_none(components.get(key))


def _primary_structure_first_drop_override(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return None
    selected_time = _candidate_boundary_time(selected)
    if selected_time is None:
        return None

    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_clock = _bpm_clock_for_time(selected_time, bpm_value, clock_zero_sec=clock_zero_sec)
    selected_bar = int((selected_clock or {}).get("nearest_one_bar", 0) or 0)
    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    if selected_bar > 17 or selected_prob >= 0.80:
        return None
    visual_primary = _primary_visual_phrase_candidate(candidates, selected, bpm=bpm_value, bar_zero_sec=clock_zero_sec)
    if visual_primary is not None:
        replacement = dict(learned)
        replacement["candidate"] = visual_primary
        replacement["selection_probability"] = max(selected_prob, 0.42)
        replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.42)
        replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.58)
        replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
        replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
        replacement["post_structure_guard_reason"] = (
            "post-structure guard used 32-bar visual phrase candidate over early phrase probe"
        )
        return replacement

    selected_components = selected.get("structure_components") if isinstance(selected.get("structure_components"), Mapping) else {}
    selected_sustain = float(selected_components.get("sustained_groove", 0.0) or 0.0)
    selected_structure_score = _float_or_none(selected.get("score")) if selected_components else None
    primary_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("structure_role") or "") != "first_drop":
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None or float(marker) <= float(selected_time) + (3.0 if selected_bar < 9 else 8.0):
            continue
        clock = _bpm_clock_for_time(marker, bpm_value, clock_zero_sec=clock_zero_sec)
        clock_bar = int((clock or {}).get("nearest_one_bar", 0) or 0)
        if selected_bar < 9:
            if clock_bar < 9 or clock_bar > 49:
                continue
        elif clock_bar < 25 or clock_bar > 49 or (clock_bar - 1) % 32 != 0:
            continue
        phrase_prior = _structure_component(candidate, "phrase_prior") or 0.0
        sustained = _structure_component(candidate, "sustained_groove") or 0.0
        structure_score = _float_or_none(candidate.get("score")) or 0.0
        if selected_bar < 9:
            if phrase_prior < 0.45 or sustained < max(0.54, selected_sustain + 0.04):
                continue
        else:
            if phrase_prior < 0.94 or sustained < max(0.48, selected_sustain + 0.06):
                continue
        if selected_structure_score is not None and structure_score < max(0.44, selected_structure_score - 0.16):
            continue
        if selected_structure_score is None and structure_score < 0.44:
            continue
        row = dict(candidate)
        row["_primary_override_clock_bar"] = clock_bar
        row["_primary_override_sustained"] = sustained
        row["_primary_override_score"] = structure_score
        primary_rows.append(row)
    if not primary_rows:
        return None

    primary_rows.sort(
        key=lambda row: (
            -float(row.get("_primary_override_sustained", 0.0) or 0.0),
            -float(row.get("_primary_override_score", 0.0) or 0.0),
            int(row.get("_primary_override_clock_bar", 999999) or 999999),
        )
    )
    selected_primary = dict(primary_rows[0])
    selected_primary.pop("_primary_override_clock_bar", None)
    selected_primary.pop("_primary_override_sustained", None)
    selected_primary.pop("_primary_override_score", None)
    replacement = dict(learned)
    replacement["candidate"] = selected_primary
    replacement["selection_probability"] = max(selected_prob, 0.36)
    replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.36)
    replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.54)
    replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["post_structure_guard_reason"] = (
        "post-structure guard used stronger 32-bar visual first-drop over earlier 16-bar probe"
    )
    return replacement


def _visual_first_drop_strength(candidate: Mapping[str, Any], bpm: Optional[float], clock_zero_sec: float) -> float:
    marker = _candidate_boundary_time(candidate)
    prior = _bar_prior_for_time(marker, bpm, bar_zero_sec=clock_zero_sec) if marker is not None else {}
    components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
    block_height = _float_or_none(components.get("block_height")) or 0.0
    sustain = _float_or_none(components.get("sustained_groove")) or 0.0
    low_to_high = _float_or_none(components.get("low_to_high")) or 0.0
    novelty = _float_or_none(components.get("timbre_novelty")) or _float_or_none(components.get("novelty")) or 0.0
    phrase = _float_or_none(components.get("phrase_prior"))
    if phrase is None:
        phrase = _float_or_none((prior or {}).get("phrase_strength")) or 0.0
    score = _float_or_none(candidate.get("score")) or _float_or_none(candidate.get("confidence_score")) or 0.0
    micro = _candidate_nested_value(candidate, "micro_confidence") or 0.0
    evidence = _candidate_evidence_score(candidate)
    return (
        (0.24 * _clip01(evidence))
        + (0.18 * _clip01(score))
        + (0.20 * _clip01(block_height))
        + (0.14 * _clip01(sustain))
        + (0.10 * _clip01(low_to_high))
        + (0.07 * _clip01(novelty))
        + (0.04 * _clip01(micro))
        + (0.03 * _clip01(phrase))
    )


def _stronger_later_visual_block_override(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return None
    selected_time = _candidate_boundary_time(selected)
    if selected_time is None:
        return None
    selected_by = str(selected.get("selected_by") or "")
    selected_source = str(selected.get("source") or "")
    if is_human_review_source(selected_by) or is_human_review_source(selected_source):
        return None

    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_prior = _bar_prior_for_time(selected_time, bpm_value, bar_zero_sec=clock_zero_sec) if selected_time is not None else {}
    selected_bar = int((selected_prior or {}).get("nearest_musical_bar", 0) or 0)
    selected_is_early = bool(float(selected_time) < 12.0 or (selected_bar > 0 and selected_bar <= 7))
    if not selected_is_early:
        return None

    selected_score = max(
        _candidate_nested_value(selected, "candidate_chooser_score") or 0.0,
        _candidate_nested_value(selected, "confidence_score") or 0.0,
        _candidate_nested_value(selected, "score") or 0.0,
    )
    selected_micro = _candidate_nested_value(selected, "micro_confidence") or 0.0
    selected_strength = _visual_first_drop_strength(selected, bpm_value, clock_zero_sec)
    selected_components = selected.get("structure_components") if isinstance(selected.get("structure_components"), Mapping) else {}
    selected_height = _float_or_none(selected_components.get("block_height")) or 0.0
    selected_sustain = _float_or_none(selected_components.get("sustained_groove")) or 0.0

    min_gap = 3.5 if float(selected_time) < 8.0 else 5.5
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None:
            continue
        marker = float(marker)
        if marker <= float(selected_time) + min_gap or marker < 12.0:
            continue
        prior = _candidate_phrase_prior(candidate, bpm_value, bar_zero_sec=clock_zero_sec)
        bar = int((prior or {}).get("nearest_musical_bar", 0) or 0)
        if bar > 0 and bar < 5:
            continue
        distance = _float_or_none((prior or {}).get("distance_beats"))
        phrase_strength = _float_or_none((prior or {}).get("phrase_strength")) or 0.0
        if (distance if distance is not None else 99.0) > 0.22 and phrase_strength < 0.55:
            continue
        components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
        score = max(
            _candidate_nested_value(candidate, "candidate_chooser_score") or 0.0,
            _candidate_nested_value(candidate, "confidence_score") or 0.0,
            _candidate_nested_value(candidate, "score") or 0.0,
        )
        micro = _candidate_nested_value(candidate, "micro_confidence") or 0.0
        height = _float_or_none(components.get("block_height")) or 0.0
        sustain = _float_or_none(components.get("sustained_groove")) or 0.0
        strength = _visual_first_drop_strength(candidate, bpm_value, clock_zero_sec)

        score_lift = score >= selected_score + (0.045 if float(selected_time) < 8.0 else 0.085)
        strength_lift = strength >= selected_strength + (0.030 if float(selected_time) < 8.0 else 0.050)
        height_lift = height >= max(0.46, selected_height + 0.050)
        sustain_lift = sustain >= max(0.44, selected_sustain + 0.040)
        micro_lift = micro >= max(0.90, selected_micro + 0.035)
        strong_saved_or_multistem = bool(
            score >= 0.72
            and micro >= 0.82
            and any(token in str(candidate.get("reason") or "") for token in ("multistem_candidate", "saved"))
        )
        if not (
            score_lift
            or (strength_lift and (height_lift or sustain_lift or micro_lift))
            or (strong_saved_or_multistem and (score >= selected_score - 0.035 or strength >= selected_strength + 0.020))
        ):
            continue
        row = dict(candidate)
        row["musical_bar_prior"] = dict(prior or {})
        row["_later_visual_time"] = marker
        row["_later_visual_score"] = float(score)
        row["_later_visual_strength"] = float(strength)
        row["_later_visual_micro"] = float(micro)
        row["_later_visual_bar"] = int(bar)
        rows.append(row)

    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            float(row.get("_later_visual_time", 999999.0) or 999999.0),
            -float(row.get("_later_visual_score", 0.0) or 0.0),
            -float(row.get("_later_visual_strength", 0.0) or 0.0),
            -float(row.get("_later_visual_micro", 0.0) or 0.0),
        )
    )
    replacement_candidate = dict(rows[0])
    replacement_candidate.pop("_later_visual_time", None)
    replacement_candidate.pop("_later_visual_score", None)
    replacement_candidate.pop("_later_visual_strength", None)
    replacement_candidate.pop("_later_visual_micro", None)
    replacement_candidate.pop("_later_visual_bar", None)

    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    replacement = dict(learned)
    replacement["candidate"] = replacement_candidate
    replacement["selection_probability"] = max(selected_prob, 0.50)
    replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.50)
    replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.62)
    replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["post_structure_guard_reason"] = (
        "visual-first guard replaced an early smaller block with a stronger later visual block"
    )
    return replacement


def _late_learned_first_drop_override(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return None
    selected_time = _candidate_boundary_time(selected)
    if selected_time is None:
        return None

    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    selected_strength = _visual_first_drop_strength(selected, bpm_value, clock_zero_sec)
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("structure_role") or "") != "first_drop":
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None:
            continue
        gap = float(selected_time) - float(marker)
        if gap < 8.0 or float(marker) < 4.0:
            continue
        strength = _visual_first_drop_strength(candidate, bpm_value, clock_zero_sec)
        score = _float_or_none(candidate.get("score")) or _float_or_none(candidate.get("confidence_score")) or 0.0
        components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
        sustain = _float_or_none(components.get("sustained_groove")) or 0.0
        phrase = _float_or_none(components.get("phrase_prior")) or 0.0
        if strength < 0.34 and score < 0.44:
            continue
        if sustain < 0.42 and phrase < 0.55 and strength < 0.42:
            continue
        if selected_prob >= 0.80 and selected_strength > strength + 0.18 and gap < 48.0:
            continue
        row = dict(candidate)
        row["_late_override_gap"] = float(gap)
        row["_late_override_strength"] = float(strength)
        row["_late_override_score"] = float(score)
        rows.append(row)
    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            -float(row.get("_late_override_strength", 0.0) or 0.0),
            -float(row.get("_late_override_score", 0.0) or 0.0),
            float(_candidate_boundary_time(row) or 999999.0),
        )
    )
    first_drop = dict(rows[0])
    first_drop.pop("_late_override_gap", None)
    first_drop.pop("_late_override_strength", None)
    first_drop.pop("_late_override_score", None)
    replacement = dict(learned)
    replacement["candidate"] = first_drop
    replacement["selection_probability"] = max(selected_prob, 0.46)
    replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.46)
    replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.60)
    replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["post_structure_guard_reason"] = (
        "visual-first guard restored the earlier structure first_drop over a later learned pick"
    )
    return replacement


def _far_later_nonvisual_first_drop_override(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return None
    selected_time = _candidate_boundary_time(selected)
    if selected_time is None:
        return None
    selected_role = str(selected.get("structure_role") or "")
    selected_by = str(selected.get("selected_by") or "")
    selected_source = str(selected.get("source") or "")
    selected_reason = str(selected.get("reason") or "")
    if selected_role == "first_drop" or selected_by.startswith("structure_map_first_drop"):
        return None
    if is_human_review_source(selected_by) or is_human_review_source(selected_source):
        return None

    bpm_value = _float_or_none(beatgrid.get("bpm"))
    bar_duration_sec = (240.0 / float(bpm_value)) if bpm_value and float(bpm_value) > 0.0 else None
    min_gap = max(20.0, 8.0 * float(bar_duration_sec or 0.0))

    first_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("structure_role") or "")
        section = str(candidate.get("section_label") or "")
        if role != "first_drop" and section != "first_drop":
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None or float(marker) < 1.0:
            continue
        if float(selected_time) - float(marker) < min_gap:
            continue
        score = _float_or_none(candidate.get("score")) or _float_or_none(candidate.get("confidence_score")) or 0.0
        components = candidate.get("structure_components") if isinstance(candidate.get("structure_components"), Mapping) else {}
        sustain = _float_or_none(components.get("sustained_groove")) or 0.0
        block_height = _float_or_none(components.get("block_height")) or 0.0
        phrase = _float_or_none(components.get("phrase_prior")) or 0.0
        if score < 0.50 and sustain < 0.48 and block_height < 0.42 and phrase < 0.75:
            continue
        first_rows.append(dict(candidate))
    if not first_rows:
        return None

    first_time = min(float(_candidate_boundary_time(row) or selected_time) for row in first_rows)
    cluster: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None or abs(float(marker) - first_time) > 0.25:
            continue
        cluster.append(dict(candidate))
    if not cluster:
        cluster = [dict(row) for row in first_rows if abs(float(_candidate_boundary_time(row) or first_time) - first_time) <= 0.25]

    def cluster_score(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
        marker = float(_candidate_boundary_time(candidate) or first_time)
        score = max(
            _candidate_nested_value(candidate, "candidate_chooser_score") or 0.0,
            _candidate_nested_value(candidate, "confidence_score") or 0.0,
            _candidate_nested_value(candidate, "score") or 0.0,
        )
        evidence = _candidate_evidence_score(candidate)
        role_bonus = 0.08 if str(candidate.get("structure_role") or "") == "first_drop" else 0.0
        saved_bonus = 0.04 if "saved_visual_batch_auto_marker" in str(candidate.get("selected_by") or "") else 0.0
        return (
            float(score) + (0.35 * float(evidence)) + role_bonus + saved_bonus,
            -abs(marker - first_time),
            -float(_float_or_none(candidate.get("rank")) or _float_or_none(candidate.get("handcrafted_rank")) or 9999.0),
        )

    restored = dict(max(cluster, key=cluster_score))
    replacement = dict(learned)
    replacement["candidate"] = restored
    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    replacement["selection_probability"] = max(selected_prob, 0.48)
    replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.48)
    replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.62)
    replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["post_structure_guard_reason"] = (
        "visual-first guard restored the first-drop cluster over a far-later non-visual pick"
        f" ({selected_by or selected_source or selected_reason or 'learned candidate'})"
    )
    return replacement


def _clean_visual_one_candidate_override(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return None
    selected_time = _candidate_boundary_time(selected)
    if selected_time is None:
        return None
    selected_by = str(selected.get("selected_by") or "")
    selected_source = str(selected.get("source") or "")
    selected_reason = str(selected.get("reason") or "")
    selected_original_by = str(selected.get("post_structure_original_selected_by") or "")
    if is_human_review_source(selected_by) or is_human_review_source(selected_source):
        return None

    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_prior = _bar_prior_for_time(selected_time, bpm_value, bar_zero_sec=clock_zero_sec) if selected_time is not None else {}
    selected_distance = _float_or_none((selected_prior or {}).get("distance_beats"))
    selected_phrase = str((selected_prior or {}).get("phrase") or "")
    selected_phrase_strength = _float_or_none((selected_prior or {}).get("phrase_strength")) or 0.0
    selected_bar = int((selected_prior or {}).get("nearest_musical_bar", 0) or 0)
    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    selected_primary_phrase = bool(
        selected_by == "visual_primary_phrase_prior"
        or selected_original_by == "visual_primary_phrase_prior"
        or "visual 32-bar phrase prior" in selected_reason
        or (selected_bar >= 25 and selected_phrase_strength >= 0.94 and selected_prob < 0.75)
    )
    selected_off_or_weak = bool(
        selected_distance is None
        or float(selected_distance) > 0.25
        or selected_phrase in {"off-one", "off-phrase"}
        or selected_phrase_strength < 0.45
    )
    if not selected_primary_phrase and not selected_off_or_weak and selected_prob >= 0.35:
        return None

    bar_duration_sec = (240.0 / float(bpm_value)) if bpm_value and float(bpm_value) > 0.0 else 0.0
    nearby_window = max(4.50, 1.35 * float(bar_duration_sec or 0.0))
    primary_min_gap = max(8.0, 1.75 * float(bar_duration_sec or 0.0))
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None or abs(float(marker) - float(selected_time)) <= 0.010:
            continue
        delta = float(marker) - float(selected_time)
        if selected_primary_phrase:
            if delta >= -0.25 or abs(delta) < primary_min_gap:
                continue
        elif abs(delta) > nearby_window:
            continue
        prior = _candidate_phrase_prior(candidate, bpm_value, bar_zero_sec=clock_zero_sec)
        if not prior:
            continue
        bar = int(prior.get("nearest_musical_bar", prior.get("nearest_even_bar", 0)) or 0)
        if bar < 9 or bar > (21 if selected_primary_phrase else 25):
            continue
        distance = _float_or_none(prior.get("distance_beats"))
        phrase_strength = _float_or_none(prior.get("phrase_strength")) or 0.0
        if (distance if distance is not None else 99.0) > 0.10 or phrase_strength < 0.75:
            continue
        score = max(
            _candidate_nested_value(candidate, "candidate_chooser_score") or 0.0,
            _candidate_nested_value(candidate, "confidence_score") or 0.0,
            _candidate_nested_value(candidate, "score") or 0.0,
        )
        micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
        micro_conf = _float_or_none(micro.get("micro_confidence")) or _float_or_none(candidate.get("micro_confidence")) or 0.0
        snap_ms = abs(_float_or_none(micro.get("snap_offset_ms")) or _float_or_none(candidate.get("snap_offset_ms")) or 0.0)
        if score < 0.68 and micro_conf < 0.88:
            continue
        if snap_ms > 140.0 and micro_conf < 0.92:
            continue
        row = dict(candidate)
        row["musical_bar_prior"] = dict(prior)
        row["_clean_visual_score"] = float(score)
        row["_clean_visual_micro"] = float(micro_conf)
        row["_clean_visual_bar"] = int(bar)
        rows.append(row)
    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            -float(row.get("_clean_visual_score", 0.0) or 0.0),
            -float(row.get("_clean_visual_micro", 0.0) or 0.0),
            -float((row.get("musical_bar_prior") or {}).get("phrase_strength", 0.0) or 0.0),
            int(row.get("_clean_visual_bar", 999999) or 999999),
            int(row.get("rank", row.get("handcrafted_rank", 999999)) or 999999),
        )
    )
    restored = dict(rows[0])
    restored.pop("_clean_visual_score", None)
    restored.pop("_clean_visual_micro", None)
    restored.pop("_clean_visual_bar", None)
    replacement = dict(learned)
    replacement["candidate"] = restored
    replacement["selection_probability"] = max(selected_prob, 0.52)
    replacement["chooser_score"] = max(_clip01(replacement.get("chooser_score")), 0.52)
    replacement["selection_confidence"] = max(_clip01(replacement.get("selection_confidence")), 0.64)
    replacement["probability_margin"] = max(0.0, float(replacement["selection_probability"]) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["post_structure_guard_reason"] = (
        "visual-first guard restored a clean on-ONE visual candidate over "
        f"{'a later primary phrase' if selected_primary_phrase else 'an off-phrase/tail pick'}"
    )
    return replacement


def _apply_post_structure_selection_guard(
    learned: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    beatgrid: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return dict(learned)
    selected_time = _candidate_boundary_time(selected)
    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    selected_prior = _bar_prior_for_time(selected_time, bpm_value, bar_zero_sec=clock_zero_sec) if selected_time is not None else {}
    selected_prob = _clip01(learned.get("selection_probability", learned.get("chooser_score", 0.0)))
    selected_by = str(selected.get("selected_by") or "")
    selected_source = str(selected.get("source") or "")
    selected_reason = str(selected.get("reason") or "")
    selected_role = str(selected.get("structure_role") or "")
    selected_rank = _float_or_none(selected.get("rank")) or _float_or_none(selected.get("handcrafted_rank")) or 9999.0
    selected_distance = _float_or_none((selected_prior or {}).get("distance_beats"))
    selected_phrase = str((selected_prior or {}).get("phrase") or "")
    selected_phrase_strength = _float_or_none((selected_prior or {}).get("phrase_strength")) or 0.0
    selected_off_one = bool((selected_distance if selected_distance is not None else 99.0) > 0.85)
    selected_phrase_start = bool("phrase start" in selected_phrase and selected_phrase_strength >= 0.80)
    selected_generic_phrase_probe = bool(
        selected_prob < 0.22
        and "phrase start" in selected_phrase
        and selected_phrase_strength <= 0.55
    )
    selected_close_off_one = bool(
        selected_phrase == "off-one"
        and selected_distance is not None
        and 0.08 <= float(selected_distance) <= 0.35
    )
    selected_weak_off_phrase = bool(
        selected_phrase == "off-phrase"
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_strength <= 0.25
    )
    selected_one_bar_probe = bool(selected_prob < 0.25 and selected_phrase_start)
    selected_off_phrase_probe = bool(selected_prob < 0.30 and selected_weak_off_phrase)
    selected_near_off_one_probe = bool(
        selected_phrase == "off-one"
        and selected_prob < 0.16
        and selected_distance is not None
        and float(selected_distance) >= 0.75
    )
    selected_deferred_probe = bool(selected_time is not None and float(selected_time) < 16.0 and selected_prob < 0.16)
    bar_duration_sec = (240.0 / float(bpm_value)) if bpm_value and float(bpm_value) > 0.0 else None
    stronger_later_visual_override = _stronger_later_visual_block_override(learned, candidates, beatgrid)
    if stronger_later_visual_override is not None:
        return stronger_later_visual_override
    clean_visual_one_override = _clean_visual_one_candidate_override(learned, candidates, beatgrid)
    if clean_visual_one_override is not None:
        return clean_visual_one_override
    primary_override = _primary_structure_first_drop_override(learned, candidates, beatgrid)
    if primary_override is not None:
        return primary_override
    late_first_drop_override = _late_learned_first_drop_override(learned, candidates, beatgrid)
    if late_first_drop_override is not None:
        return late_first_drop_override
    far_later_first_drop_override = _far_later_nonvisual_first_drop_override(learned, candidates, beatgrid)
    if far_later_first_drop_override is not None:
        return far_later_first_drop_override
    selected_suspicious = bool(
        selected_prob < 0.12
        or (
            selected_source == "saved_closest_to_review_pick"
            and "not_selected" in selected_reason
        )
        or (
            "saved" in selected_reason
            and selected_prob < 0.20
            and selected_distance is not None
            and float(selected_distance) > 0.40
        )
        or (
            selected_off_one
            and (
                selected_prob < 0.45
                or selected_by in {"model", "handcrafted"}
                or selected_source == "saved_closest_to_review_pick"
                or "not_selected:" in selected_reason
            )
        )
    )
    if not (
        selected_suspicious
        or selected_one_bar_probe
        or selected_off_phrase_probe
        or selected_generic_phrase_probe
        or selected_near_off_one_probe
        or selected_deferred_probe
    ):
        return dict(learned)
    if (
        selected_by == "micro_boundary_variant"
        and selected_prob >= 0.040
        and selected_distance is not None
        and 0.08 <= float(selected_distance) <= 0.25
        and (selected_phrase == "off-one" or selected_phrase_strength <= 0.10)
    ):
        return dict(learned)
    if (
        (selected_source == "saved_closest_to_review_pick" or selected_by == "review_auto_place")
        and "not_selected" not in selected_reason
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_strength >= 0.45
    ):
        return dict(learned)
    if (
        selected_by == "candidate_chooser"
        and "saved" in selected_reason
        and selected_rank <= 2.0
        and selected_prob >= 0.10
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_start
        and not (selected_deferred_probe and selected_prob >= 0.13)
    ):
        return dict(learned)
    if (
        selected_by == "candidate_chooser"
        and "saved" in selected_reason
        and selected_rank <= 1.0
        and selected_prob >= 0.020
        and selected_prob <= 0.080
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and 0.60 <= selected_phrase_strength <= 0.70
    ):
        return dict(learned)
    if (
        selected_by == "candidate_chooser"
        and "saved" in selected_reason
        and selected_rank <= 8.0
        and 0.040 <= selected_prob <= 0.080
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_strength >= 0.45
        and (selected_phrase_strength < 0.80 or selected_rank <= 1.0)
    ):
        return dict(learned)
    if (
        selected_by == "model"
        and "not_selected:model_predicted_larger_user_delta" in selected_reason
        and selected_prob >= 0.080
        and selected_close_off_one
    ):
        return dict(learned)
    if selected_by == "track_zero_grid_variant" and selected_prob >= 0.20:
        return dict(learned)
    if (
        selected_source == "saved_closest_to_review_pick"
        and selected_rank <= 2.0
        and selected_prob >= 0.07
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_start
    ):
        return dict(learned)
    if (
        selected_source == "saved_closest_to_review_pick"
        and selected_by == "model"
        and "not_selected:model_predicted_larger_user_delta" in selected_reason
        and selected_rank <= 5.0
        and selected_prob >= 0.15
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_strength >= 0.45
    ):
        return dict(learned)
    if (
        selected_source == "saved_closest_to_review_pick"
        and selected_by == "model"
        and "not_selected:model_predicted_larger_user_delta" in selected_reason
        and selected_prob >= 0.070
        and selected_off_one
        and selected_distance is not None
        and 0.80 <= float(selected_distance) <= 1.20
    ):
        return dict(learned)
    if (
        selected_by == "micro_boundary_variant"
        and selected_source == "micro_boundary_clock_boundary"
        and selected_role == "drop_candidate"
        and selected_prob >= 0.045
        and selected_distance is not None
        and float(selected_distance) <= 0.08
        and selected_phrase_strength >= 0.90
    ):
        return dict(learned)
    try:
        payload = load_candidate_chooser_payload(str(POST_STRUCTURE_CHOOSER_PATH))
        predictions = predict_candidate_errors(payload, candidates) if payload else []
    except Exception:
        predictions = []
    alternatives: List[tuple[float, float, float, float, Dict[str, Any]]] = []
    for prediction in predictions:
        candidate = prediction.get("candidate") if isinstance(prediction.get("candidate"), Mapping) else None
        if not isinstance(candidate, Mapping):
            continue
        marker = _candidate_boundary_time(candidate)
        if marker is None or (selected_time is not None and abs(float(marker) - float(selected_time)) <= 1e-9):
            continue
        probability = _clip01(prediction.get("selection_probability", prediction.get("chooser_score", 0.0)))
        candidate_by = str(candidate.get("selected_by") or "")
        candidate_source = str(candidate.get("source") or "")
        candidate_reason = str(candidate.get("reason") or "")
        candidate_role = str(candidate.get("structure_role") or "")
        candidate_rank = _float_or_none(candidate.get("rank")) or _float_or_none(candidate.get("handcrafted_rank")) or 9999.0
        prior = _bar_prior_for_time(marker, bpm_value, bar_zero_sec=clock_zero_sec) or {}
        distance = _float_or_none(prior.get("distance_beats"))
        phrase_strength = _float_or_none(prior.get("phrase_strength")) or 0.0
        candidate_chooser_phrase_alt = bool(
            selected_suspicious
            and
            candidate_by == "candidate_chooser"
            and (
                selected_prob < 0.050
                or (candidate_rank <= 1.0 and phrase_strength >= 0.90 and selected_prob < 0.070)
            )
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) > 0.50
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and candidate_rank <= 8.0
            and (
                candidate_rank <= 5.0
                or "saved" in candidate_reason
                or candidate_source.startswith("saved")
            )
        )
        candidate_chooser_alt = bool(
            selected_suspicious
            and candidate_by == "candidate_chooser"
            and (
                candidate_chooser_phrase_alt
                or (
                    probability >= max(0.003, selected_prob - 0.08)
                    and distance is not None
                    and float(distance) <= 0.08
                    and phrase_strength >= 0.45
                )
            )
        )
        strong_structure_alt = bool(
            selected_suspicious
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.80
            and probability >= max(0.04, selected_prob * 0.25)
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
        )
        early_top_phrase_alt = bool(
            selected_suspicious
            and selected_prob < 0.110
            and selected_time is not None
            and float(marker) < (float(selected_time) - 4.0)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.75
            and candidate_rank <= 5.0
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
            and (
                selected_source == "saved_closest_to_review_pick"
                or selected_rank >= 20.0
                or "not_selected" in selected_reason
            )
        )
        nearby_strong_phrase_alt = bool(
            selected_suspicious
            and selected_prob < 0.120
            and selected_time is not None
            and float(marker) > (float(selected_time) + 0.25)
            and float(marker) <= (float(selected_time) + 3.0)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.90
            and candidate_rank <= 8.0
            and probability >= max(0.015, selected_prob * 0.15)
            and (
                selected_off_one
                or selected_phrase in {"off-one", "off-phrase"}
                or selected_phrase_strength <= 0.20
            )
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate", "candidate_chooser"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
        )
        saved_review_pick_alt = bool(
            selected_suspicious
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and candidate_rank <= 8.0
            and (
                candidate_source == "saved_closest_to_review_pick"
                or candidate_by == "review_auto_place"
            )
            and (
                selected_prob < 0.20
                or selected_rank >= 20.0
                or "saved" in selected_reason
            )
        )
        nearby_grid_snap_alt = bool(
            selected_suspicious
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.25
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and candidate_rank <= 5.0
            and probability >= max(0.015, selected_prob * 0.25)
            and (
                selected_source == "saved_closest_to_review_pick"
                or "not_selected" in selected_reason
                or (selected_distance is not None and float(selected_distance) > 0.10)
            )
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate", "candidate_chooser"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
        )
        nearby_phrase_boundary_alt = bool(
            selected_suspicious
            and selected_prob < 0.10
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.50
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.75
            and candidate_rank <= 5.0
            and probability >= 0.005
            and (
                selected_source == "saved_closest_to_review_pick"
                or "not_selected" in selected_reason
                or (selected_distance is not None and float(selected_distance) > 0.40)
            )
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate", "candidate_chooser"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
        )
        strong_saved_phrase_anchor_alt = bool(
            (selected_suspicious or selected_off_phrase_probe or selected_generic_phrase_probe)
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) >= 0.50
            and abs(float(marker) - float(selected_time)) <= 35.0
            and not (selected_off_one and float(marker) < (float(selected_time) - 12.0))
            and not (float(marker) > (float(selected_time) + 20.0))
            and not (selected_phrase_start and selected_prob >= 0.25)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.90
            and candidate_rank <= 20.0
            and (
                candidate_source.startswith("saved")
                or "saved" in candidate_reason
                or candidate_by == "candidate_chooser"
            )
            and (
                probability >= max(0.010, selected_prob - 0.020)
                or (
                    "saved" in candidate_reason
                    and (candidate_rank <= 5.0 or phrase_strength >= 0.94)
                )
            )
        )
        strong_micro_phrase_anchor_alt = bool(
            selected_generic_phrase_probe
            and selected_time is not None
            and 2.50 <= abs(float(marker) - float(selected_time)) <= 4.50
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.90
            and candidate_rank <= 2.0
            and probability >= 0.010
            and candidate_source.startswith("micro_boundary")
            and (
                "not_selected:model_predicted_larger_user_delta" in candidate_reason
                or candidate_role
                or candidate_rank <= 1.0
            )
        )
        nearby_off_one_phrase_alt = bool(
            selected_suspicious
            and selected_off_one
            and selected_prob < 0.16
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.85
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.60
            and candidate_rank <= 12.0
            and (
                candidate_by in {"candidate_chooser", "micro_boundary_variant", "model", "multistem_candidate"}
                or candidate_source.startswith("micro_boundary")
                or candidate_source.startswith("saved")
                or candidate_reason
            )
        )
        rank_one_phrase_snap_alt = bool(
            selected_suspicious
            and selected_close_off_one
            and selected_prob < 0.10
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.35
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.60
            and candidate_rank <= 1.0
            and (
                probability >= 0.001
                or (
                    candidate_source == "micro_boundary_clock_boundary"
                    and candidate_role
                    and candidate_role == selected_role
                )
            )
            and (
                candidate_by in {"candidate_chooser", "multistem_candidate", "review_auto_place"}
                or candidate_source.startswith("saved")
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
        )
        off_one_ranked_phrase_alt = bool(
            selected_suspicious
            and selected_off_one
            and selected_prob < 0.10
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.75
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.60
            and candidate_rank <= 2.0
            and probability >= 0.005
            and (
                candidate_by in {"candidate_chooser", "multistem_candidate", "review_auto_place"}
                or candidate_source.startswith("saved")
                or candidate_reason
            )
        )
        nearby_integer_grid_snap_alt = bool(
            (selected_suspicious or selected_near_off_one_probe)
            and (selected_off_one or selected_phrase == "off-one")
            and selected_prob < 0.16
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) <= 0.12
            and candidate_by == "track_zero_grid_variant"
            and distance is not None
            and abs(float(distance) - round(float(distance))) <= 0.020
        )
        later_saved_offphrase_alt = bool(
            selected_suspicious
            and selected_off_one
            and selected_prob < 0.16
            and selected_time is not None
            and 1.50 <= (float(marker) - float(selected_time)) <= 2.50
            and distance is not None
            and float(distance) <= 0.08
            and candidate_rank <= 20.0
            and probability >= 0.005
            and "saved" in candidate_reason
        )
        early_saved_phrase_rescue_alt = bool(
            selected_suspicious
            and selected_prob < 0.020
            and selected_time is not None
            and 4.0 <= (float(selected_time) - float(marker)) <= 12.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.80
            and candidate_rank <= 5.0
            and "saved" in candidate_reason
        )
        early_phrase_anchor_alt = bool(
            selected_suspicious
            and selected_prob < 0.07
            and selected_time is not None
            and (
                (
                    selected_off_one
                    and float(marker) < (float(selected_time) - 4.0)
                    and float(marker) >= (float(selected_time) - 12.0)
                    and phrase_strength >= 0.75
                    and probability >= 0.001
                )
                or (
                    float(selected_time) <= 8.0
                    and float(marker) >= 20.0
                    and "saved" in candidate_reason
                )
            )
            and distance is not None
            and float(distance) <= 0.08
            and candidate_rank <= 8.0
            and (
                candidate_by in {"candidate_chooser", "multistem_candidate", "review_auto_place"}
                or candidate_source.startswith("saved")
            )
        )
        saved_candidate_reason_alt = bool(
            selected_suspicious
            and selected_prob < 0.10
            and selected_time is not None
            and abs(float(marker) - float(selected_time)) >= 4.0
            and candidate_by == "candidate_chooser"
            and "saved" in candidate_reason
            and candidate_rank <= 50.0
            and probability >= 0.015
        )
        one_bar_late_candidate_chooser_alt = bool(
            selected_one_bar_probe
            and selected_time is not None
            and bar_duration_sec is not None
            and abs((float(marker) - float(selected_time)) - float(bar_duration_sec)) <= 0.25
            and candidate_by == "candidate_chooser"
            and candidate_rank <= 8.0
            and distance is not None
            and float(distance) <= 0.08
            and probability >= max(0.008, selected_prob * 0.04)
            and "drums" in candidate_reason
        )
        one_bar_late_phrase_start_alt = bool(
            selected_off_phrase_probe
            and selected_time is not None
            and bar_duration_sec is not None
            and abs((float(marker) - float(selected_time)) - float(bar_duration_sec)) <= 0.25
            and candidate_by == "candidate_chooser"
            and candidate_rank <= 2.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.60
            and probability >= max(0.050, selected_prob * 0.30)
        )
        one_bar_late_saved_model_alt = bool(
            selected_off_phrase_probe
            and selected_time is not None
            and bar_duration_sec is not None
            and abs((float(marker) - float(selected_time)) - float(bar_duration_sec)) <= 0.25
            and candidate_by in {"model", "candidate_chooser"}
            and candidate_rank <= 10.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and (
                probability >= max(0.060, selected_prob * 0.40)
                or (
                    candidate_by == "candidate_chooser"
                    and candidate_rank <= 10.0
                    and "multistem_candidate:saved" in candidate_reason
                )
            )
            and (
                "not_selected:model_predicted_larger_user_delta" in candidate_reason
                or candidate_source == "saved_closest_to_review_pick"
                or "saved" in candidate_reason
            )
        )
        nearby_structure_drop_from_offone_alt = bool(
            selected_suspicious
            and selected_off_one
            and selected_prob < 0.12
            and selected_time is not None
            and 0.80 <= abs(float(marker) - float(selected_time)) <= 1.60
            and candidate_role == "drop_candidate"
            and candidate_source == "micro_boundary_clock_boundary"
            and candidate_rank <= 10.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
        )
        later_track_grid_phrase_alt = bool(
            (selected_suspicious or selected_deferred_probe)
            and selected_prob < 0.04
            and selected_time is not None
            and 20.0 <= (float(marker) - float(selected_time)) <= 35.0
            and candidate_by == "track_zero_grid_variant"
            and candidate_rank <= 20.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.80
        )
        later_structure_drop_alt = bool(
            selected_deferred_probe
            and selected_time is not None
            and 25.0 <= (float(marker) - float(selected_time)) <= 35.0
            and candidate_role == "drop_candidate"
            and candidate_rank <= 5.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and probability >= 0.035
        )
        nearby_micro_saved_offset_alt = bool(
            selected_by == "micro_boundary_variant"
            and selected_source == "micro_boundary_clock_boundary"
            and selected_prob < 0.22
            and selected_phrase_start
            and selected_time is not None
            and float(marker) > float(selected_time)
            and abs(float(marker) - float(selected_time)) <= 0.040
            and candidate_by == "candidate_chooser"
            and candidate_rank <= 50.0
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.60
            and probability >= 0.020
            and "saved" in candidate_reason
        )
        later_nearby_drop_alt = bool(
            selected_suspicious
            and selected_time is not None
            and float(marker) > (float(selected_time) + 2.0)
            and float(marker) <= (float(selected_time) + 8.0)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.45
            and candidate_rank <= 8.0
            and probability >= max(0.050, selected_prob * 0.30)
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate", "candidate_chooser"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
            and (
                selected_source == "saved_closest_to_review_pick"
                or "not_selected" in selected_reason
            )
        )
        deferred_phrase_alt = bool(
            selected_suspicious
            and selected_off_one
            and selected_time is not None
            and float(marker) > (float(selected_time) + 4.0)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.90
            and probability >= max(0.020, selected_prob * 0.10)
            and candidate_rank <= max(30.0, selected_rank - 10.0)
            and (
                candidate_by.startswith("structure_map")
                or candidate_by in {"micro_boundary_variant", "multistem_candidate", "candidate_chooser"}
                or candidate_source.startswith("micro_boundary")
                or candidate_role
            )
            and (
                selected_source == "saved_closest_to_review_pick"
                or "not_selected" in selected_reason
            )
        )
        saved_phrase_alt = bool(
            selected_suspicious
            and selected_prob < 0.020
            and selected_time is not None
            and float(marker) < (float(selected_time) - 0.50)
            and distance is not None
            and float(distance) <= 0.08
            and phrase_strength >= 0.75
            and probability >= max(0.001, selected_prob * 0.10)
            and candidate_rank <= max(12.0, selected_rank - 2.0)
            and (
                candidate_by.startswith("saved")
                or candidate_source.startswith("saved")
                or "saved" in candidate_reason
            )
        )
        if not (
            candidate_chooser_alt
            or nearby_strong_phrase_alt
            or saved_review_pick_alt
            or nearby_grid_snap_alt
            or nearby_phrase_boundary_alt
            or strong_saved_phrase_anchor_alt
            or strong_micro_phrase_anchor_alt
            or nearby_off_one_phrase_alt
            or rank_one_phrase_snap_alt
            or off_one_ranked_phrase_alt
            or nearby_integer_grid_snap_alt
            or later_saved_offphrase_alt
            or early_saved_phrase_rescue_alt
            or early_phrase_anchor_alt
            or saved_candidate_reason_alt
            or one_bar_late_candidate_chooser_alt
            or one_bar_late_phrase_start_alt
            or one_bar_late_saved_model_alt
            or nearby_structure_drop_from_offone_alt
            or later_track_grid_phrase_alt
            or later_structure_drop_alt
            or nearby_micro_saved_offset_alt
            or later_nearby_drop_alt
            or deferred_phrase_alt
            or saved_phrase_alt
            or early_top_phrase_alt
            or strong_structure_alt
        ):
            continue
        if nearby_strong_phrase_alt:
            kind_priority = 5.25
        elif saved_review_pick_alt:
            kind_priority = 5.22
        elif nearby_grid_snap_alt:
            kind_priority = 5.20
        elif nearby_phrase_boundary_alt:
            kind_priority = 5.19
        elif strong_saved_phrase_anchor_alt:
            kind_priority = 5.185
        elif strong_micro_phrase_anchor_alt:
            kind_priority = 5.182
        elif nearby_off_one_phrase_alt:
            kind_priority = 5.17
        elif rank_one_phrase_snap_alt:
            kind_priority = 5.18
        elif off_one_ranked_phrase_alt:
            kind_priority = 5.16
        elif nearby_integer_grid_snap_alt:
            kind_priority = 5.15
        elif nearby_micro_saved_offset_alt:
            kind_priority = 5.14
        elif nearby_structure_drop_from_offone_alt:
            kind_priority = 5.12
        elif one_bar_late_candidate_chooser_alt:
            kind_priority = 5.10
        elif one_bar_late_phrase_start_alt:
            kind_priority = 5.08
        elif one_bar_late_saved_model_alt:
            kind_priority = 5.06
        elif later_saved_offphrase_alt:
            kind_priority = 5.03
        elif later_structure_drop_alt:
            kind_priority = 5.02
        elif early_saved_phrase_rescue_alt:
            kind_priority = 5.21
        elif candidate_chooser_alt:
            kind_priority = 5.0
        elif saved_candidate_reason_alt:
            kind_priority = 4.95
        elif early_phrase_anchor_alt:
            kind_priority = 4.90
        elif later_track_grid_phrase_alt:
            kind_priority = 4.85
        elif later_nearby_drop_alt:
            kind_priority = 4.75
        elif early_top_phrase_alt:
            kind_priority = 4.5
        elif deferred_phrase_alt:
            kind_priority = 4.25
        elif saved_phrase_alt:
            kind_priority = 4.0
        else:
            kind_priority = 2.0
        if candidate_by == "candidate_chooser" and phrase_strength >= 0.80:
            kind_priority += 0.5
        alternatives.append((kind_priority, -float(candidate_rank), probability, -abs(float(distance or 0.0)), dict(prediction)))
    if not alternatives:
        return dict(learned)
    alternatives.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    replacement = dict(alternatives[0][4])
    replacement_prob = _clip01(replacement.get("selection_probability", replacement.get("chooser_score", 0.0)))
    replacement["probability_margin"] = max(0.0, float(replacement_prob) - float(selected_prob))
    replacement["prediction_margin_sec"] = float(replacement["probability_margin"])
    replacement["selection_confidence"] = max(
        _clip01(replacement.get("selection_confidence")),
        _clip01((0.70 * replacement_prob) + (0.30 * _clip01(float(replacement["probability_margin"]) / 0.30))),
    )
    replacement["post_structure_guard_reason"] = (
        f"post-structure guard replaced weak/off-one candidate "
        f"(P {selected_prob:.3f})"
    )
    far_later_first_drop_override = _far_later_nonvisual_first_drop_override(replacement, candidates, beatgrid)
    if far_later_first_drop_override is not None:
        return far_later_first_drop_override
    return replacement


def _post_structure_chooser_suggestion(
    merged_candidates: List[Mapping[str, Any]],
    structure: Mapping[str, Any],
    *,
    beatgrid: Mapping[str, Any],
    confidence_tier: str,
) -> Optional[Dict[str, Any]]:
    if not POST_STRUCTURE_CHOOSER_PATH.exists():
        return None
    try:
        learned = choose_learned_candidate(merged_candidates, model_path=str(POST_STRUCTURE_CHOOSER_PATH))
    except Exception:
        return None
    if not isinstance(learned, Mapping):
        return None
    learned = _apply_post_structure_selection_guard(learned, merged_candidates, beatgrid)
    candidate = learned.get("candidate") if isinstance(learned.get("candidate"), Mapping) else None
    if not isinstance(candidate, Mapping):
        return None
    marker = _candidate_boundary_time(candidate)
    if marker is None:
        return None
    selected = dict(candidate)
    selected["post_structure_original_selected_by"] = str(selected.get("selected_by") or "")
    selected["selected_by"] = "post_structure_candidate_chooser"
    selected["candidate_chooser_predicted_abs_error_sec"] = float(learned.get("predicted_abs_error_sec", 0.0) or 0.0)
    selected["candidate_chooser_score"] = float(learned.get("chooser_score", 0.0) or 0.0)
    selected["candidate_chooser_confidence"] = float(learned.get("selection_confidence", 0.0) or 0.0)
    selected["candidate_chooser_margin_sec"] = float(learned.get("prediction_margin_sec", 0.0) or 0.0)
    if learned.get("selection_probability") is not None:
        selected["candidate_chooser_probability"] = float(learned.get("selection_probability", 0.0) or 0.0)
    if learned.get("probability_margin") is not None:
        selected["candidate_chooser_probability_margin"] = float(learned.get("probability_margin", 0.0) or 0.0)
    selected["candidate_chooser_model_path"] = str(learned.get("model_path", POST_STRUCTURE_CHOOSER_PATH))
    selected["timestamp"] = float(marker)
    selected["snapped_sec"] = float(marker)
    selected["microaligned_time"] = float(marker)
    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    clock = _bpm_clock_for_time(marker, bpm_value, clock_zero_sec=clock_zero_sec)
    grid = feature_grid_for_time(marker, bpm_value, bar_zero_sec=beatgrid.get("bar_zero_sec"))
    if clock:
        selected["bpm_clock"] = clock
    if grid:
        selected["feature_grid_prior"] = grid
    micro = selected.get("microalign") if isinstance(selected.get("microalign"), Mapping) else None
    if micro is None:
        micro = {
            "ok": True,
            "used": False,
            "microaligned_time": float(marker),
            "snap_offset_ms": 0.0,
            "micro_confidence": 0.0,
            "reason": "post-structure chooser kept candidate boundary; MicroSnap not applied",
        }
    probability = float(learned.get("selection_probability", learned.get("chooser_score", 0.0)) or 0.0)
    margin = float(learned.get("probability_margin", learned.get("prediction_margin_sec", 0.0)) or 0.0)
    reason = (
        f"post-structure chooser selected merged candidate "
        f"(P {probability:.3f}, margin {margin:.3f}); "
        f"{selected.get('reason') or 'merged structure/audio evidence'}"
    )
    guard_reason = str(learned.get("post_structure_guard_reason") or "")
    if guard_reason:
        reason = f"{reason}; {guard_reason}"
    return {
        "auto_place": False,
        "review_needed": True,
        "risk": True,
        "mode": "post_structure",
        "confidence_tier": _normalize_tier(confidence_tier),
        "candidate": selected,
        "microalign": dict(micro),
        "suggested_time": float(marker),
        "score": probability,
        "reason": reason,
        "structure_map": {
            "first_drop": structure.get("first_drop"),
            "second_drop": structure.get("second_drop"),
            "sections": structure.get("sections") or [],
            "beatgrid": beatgrid,
        },
        "auto_accept": _review_required_auto_accept("post-structure chooser requires review", selected, merged_candidates, _normalize_tier(confidence_tier)),
    }


def _apply_final_visual_block_guard(
    bar_prior_result: Mapping[str, Any],
    beatgrid: Mapping[str, Any],
    *,
    confidence_tier: str,
) -> Dict[str, Any]:
    suggestion = dict(bar_prior_result.get("suggestion") if isinstance(bar_prior_result.get("suggestion"), Mapping) else {})
    selected = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else None
    if not isinstance(selected, Mapping):
        return dict(bar_prior_result)
    candidates = [dict(candidate) for candidate in bar_prior_result.get("candidates", []) or [] if isinstance(candidate, Mapping)]
    if not candidates:
        return dict(bar_prior_result)
    learned = {
        "candidate": dict(selected),
        "selection_probability": _float_or_none(suggestion.get("score"))
        or _candidate_nested_value(selected, "candidate_chooser_probability")
        or _candidate_nested_value(selected, "candidate_chooser_score")
        or _candidate_nested_value(selected, "score")
        or 0.0,
        "chooser_score": _float_or_none(suggestion.get("score"))
        or _candidate_nested_value(selected, "candidate_chooser_score")
        or _candidate_nested_value(selected, "score")
        or 0.0,
        "selection_confidence": _candidate_nested_value(selected, "candidate_chooser_confidence") or 0.0,
    }
    override = _stronger_later_visual_block_override(learned, candidates, beatgrid)
    if override is None:
        return dict(bar_prior_result)
    replacement = dict(override.get("candidate") if isinstance(override.get("candidate"), Mapping) else {})
    marker = _candidate_boundary_time(replacement)
    if marker is None:
        return dict(bar_prior_result)

    original_selected_by = str(replacement.get("selected_by") or "")
    replacement["post_structure_original_selected_by"] = original_selected_by
    replacement["selected_by"] = "post_structure_candidate_chooser"
    replacement["timestamp"] = float(marker)
    replacement["snapped_sec"] = float(marker)
    replacement["microaligned_time"] = float(marker)
    bpm_value = _float_or_none(beatgrid.get("bpm"))
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    clock = _bpm_clock_for_time(marker, bpm_value, clock_zero_sec=clock_zero_sec)
    grid = feature_grid_for_time(marker, bpm_value, bar_zero_sec=beatgrid.get("bar_zero_sec"))
    if clock:
        replacement["bpm_clock"] = clock
    if grid:
        replacement["feature_grid_prior"] = grid

    next_suggestion = dict(suggestion)
    next_suggestion["candidate"] = replacement
    next_suggestion["suggested_time"] = float(marker)
    next_suggestion["score"] = max(
        _clip01(next_suggestion.get("score")),
        _clip01(override.get("selection_probability", override.get("chooser_score", 0.0))),
    )
    reason = str(next_suggestion.get("reason") or "")
    guard_reason = str(override.get("post_structure_guard_reason") or "visual-first guard replaced early smaller block")
    next_suggestion["reason"] = f"{reason}; {guard_reason}" if reason else guard_reason
    next_suggestion["auto_accept"] = _review_required_auto_accept(
        "post-structure chooser requires review",
        replacement,
        candidates,
        _normalize_tier(confidence_tier),
    )

    out = dict(bar_prior_result)
    out["suggestion"] = next_suggestion
    out["candidates"] = _dedupe_candidate_dicts([replacement] + candidates)
    return out


def _apply_structure_map_prior(
    audio_path: str,
    candidates: List[Mapping[str, Any]],
    suggestion: Mapping[str, Any],
    *,
    confidence_tier: str,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    try:
        structure = analyze_track_structure(audio_path, sample_rate=int(sample_rate), use_cache=True)
    except Exception as exc:
        return {
            "candidates": [dict(candidate) for candidate in candidates],
            "suggestion": dict(suggestion or {}),
            "structure_map": {"ok": False, "error": str(exc) or exc.__class__.__name__},
        }

    base_candidates = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
    suggestion_candidate = suggestion.get("candidate") if isinstance(suggestion, Mapping) else None
    if isinstance(suggestion_candidate, Mapping):
        base_candidates.append(dict(suggestion_candidate))
    merged_candidates = _merge_structure_candidates(structure, base_candidates)
    first = structure.get("first_drop") if isinstance(structure.get("first_drop"), Mapping) else None
    if not first:
        return {
            "candidates": merged_candidates,
            "suggestion": dict(suggestion or {}),
            "structure_map": structure,
        }

    beatgrid = structure.get("beatgrid") if isinstance(structure.get("beatgrid"), Mapping) else {}
    bpm_value = _float_or_none(beatgrid.get("bpm"))
    beat_sec = _float_or_none(beatgrid.get("beat_sec")) or 60.0 / max(1.0, bpm_value or 128.0)
    clock_zero_sec = _float_or_none(beatgrid.get("bar_zero_sec")) or 0.0
    merged_candidates = _dedupe_candidate_dicts(
        merged_candidates
        + _structure_boundary_variant_candidates(
            audio_path,
            merged_candidates,
            beat_sec=beat_sec,
            bpm=bpm_value,
            clock_zero_sec=clock_zero_sec,
        )
    )
    post_structure = _post_structure_chooser_suggestion(
        merged_candidates,
        structure,
        beatgrid=beatgrid,
        confidence_tier=confidence_tier,
    )
    if post_structure is not None:
        post_candidate = post_structure.get("candidate") if isinstance(post_structure.get("candidate"), Mapping) else None
        if post_candidate:
            merged_candidates = _dedupe_candidate_dicts([dict(post_candidate)] + merged_candidates)
        return {
            "candidates": merged_candidates,
            "suggestion": post_structure,
            "structure_map": structure,
        }
    base_suggestion = dict(suggestion or {})
    base_candidate = base_suggestion.get("candidate") if isinstance(base_suggestion.get("candidate"), Mapping) else {}
    base_time = _float_or_none(base_suggestion.get("suggested_time")) or _candidate_marker_time(base_candidate)
    first_time = _candidate_marker_time(first)
    base_micro = _candidate_nested_value(base_candidate, "micro_confidence") or 0.0
    base_score = max(
        _candidate_nested_value(base_candidate, "candidate_chooser_score") or 0.0,
        _candidate_nested_value(base_candidate, "confidence_score") or 0.0,
        _candidate_nested_value(base_candidate, "score") or 0.0,
    )
    base_chooser_score = _candidate_nested_value(base_candidate, "candidate_chooser_score") or 0.0
    base_selected_by = str(base_candidate.get("selected_by") or "")
    base_clock = _bpm_clock_for_time(base_time, bpm_value, clock_zero_sec=clock_zero_sec)
    base_clock_bar = int((base_clock or {}).get("nearest_one_bar", 0) or 0)
    base_is_early_anchor = bool(base_clock_bar > 0 and base_clock_bar <= 4)
    first_clock = _bpm_clock_for_time(first_time, bpm_value, clock_zero_sec=clock_zero_sec)
    first_clock_bar = int((first_clock or {}).get("nearest_one_bar", 0) or 0)
    structure_is_far_later = bool(
        base_time is not None and first_time is not None and float(first_time) > float(base_time) + (16.0 * 4.0 * float(beat_sec))
    )
    same_clock_bar_as_base = bool(base_clock_bar > 0 and first_clock_bar > 0 and base_clock_bar == first_clock_bar)
    base_is_reliable_learned = bool(
        base_selected_by == "candidate_chooser"
        and base_chooser_score >= 0.55
        and base_micro >= 0.78
        and base_score >= 0.62
    )
    legacy_structure_guard = bool(
        (
            base_is_early_anchor
            or structure_is_far_later
            or same_clock_bar_as_base
        )
        and base_micro >= 0.82
        and base_score >= 0.58
    )
    if (
        base_time is not None
        and first_time is not None
        and (base_is_reliable_learned or legacy_structure_guard)
    ):
        guarded = dict(base_suggestion)
        guarded["structure_map"] = {
            "first_drop": structure.get("first_drop"),
            "second_drop": structure.get("second_drop"),
            "sections": structure.get("sections") or [],
            "beatgrid": beatgrid,
        }
        guarded["reason"] = (
            f"{guarded.get('reason') or 'candidate chooser kept'}; "
            "structure map was advisory because the learned candidate was high-confidence"
        )
        return {
            "candidates": merged_candidates,
            "suggestion": guarded,
            "structure_map": structure,
        }
    selected = dict(first)
    selected["selected_by"] = "structure_map_first_drop"
    selected["structure_map_version"] = int(structure.get("version", 0) or 0)
    micro = _restricted_structure_microalign(audio_path, selected, beat_sec, bpm=bpm_value, clock_zero_sec=clock_zero_sec)
    suggested_time = _float_or_none(micro.get("microaligned_time")) or _candidate_marker_time(selected)
    if suggested_time is not None:
        selected["timestamp"] = float(suggested_time)
        selected["snapped_sec"] = float(suggested_time)
        selected["microaligned_time"] = float(suggested_time)
        selected_clock = _bpm_clock_for_time(suggested_time, bpm_value, clock_zero_sec=clock_zero_sec)
        selected_grid = feature_grid_for_time(suggested_time, bpm_value, bar_zero_sec=beatgrid.get("bar_zero_sec"))
        if selected_clock:
            selected["bpm_clock"] = selected_clock
        if selected_grid:
            selected["feature_grid_prior"] = selected_grid
    selected["microalign"] = dict(micro)
    components = selected.get("structure_components") if isinstance(selected.get("structure_components"), Mapping) else {}
    clock_bar = int((selected.get("bpm_clock") or {}).get("nearest_one_bar", selected.get("structure_clock_bar", 0)) or 0)
    grid_bar = int((selected.get("feature_grid_prior") or {}).get("nearest_grid_bar", selected.get("structure_bar", 0)) or 0)
    clock_grid_label = " | ".join(
        part for part in (f"clock b{clock_bar}" if clock_bar > 0 else "", f"grid {grid_bar}" if grid_bar > 0 else "") if part
    ) or f"bar {int(selected.get('structure_bar', 0) or 0)}"
    reason = (
        f"structure map selected first drop at {clock_grid_label} "
        f"({components.get('phrase', 'phrase')}; low->high {float(components.get('low_to_high', 0.0) or 0.0):.2f}, "
        f"sustain {float(components.get('sustained_groove', 0.0) or 0.0):.2f})"
    )
    merged_candidates = _dedupe_candidate_dicts([selected] + _boundary_variant_candidates([selected], max_candidates=1) + merged_candidates)
    suggestion_out = {
        "auto_place": False,
        "review_needed": True,
        "risk": True,
        "mode": "structure",
        "confidence_tier": _normalize_tier(confidence_tier),
        "candidate": selected,
        "microalign": dict(micro),
        "suggested_time": None if suggested_time is None else float(suggested_time),
        "score": float(selected.get("score", 0.0) or 0.0),
        "reason": reason,
        "structure_map": {
            "first_drop": structure.get("first_drop"),
            "second_drop": structure.get("second_drop"),
            "sections": structure.get("sections") or [],
            "beatgrid": beatgrid,
        },
        "auto_accept": _review_required_auto_accept("structure-map selection requires review", selected, merged_candidates, _normalize_tier(confidence_tier)),
    }
    return {
        "candidates": merged_candidates,
        "suggestion": suggestion_out,
        "structure_map": structure,
    }


def _read_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path).expanduser()
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _review_memory_keys(path: str) -> List[str]:
    text = str(path or "").strip()
    if not text:
        return []
    keys = [text]
    audio = Path(text).expanduser()
    try:
        keys.append(str(audio.resolve()))
    except Exception:
        pass
    keys.append(audio.name)
    out: List[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _load_review_memory(path: str) -> Dict[str, Dict[str, Any]]:
    log_path = Path(path).expanduser()
    memory: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(log_path):
        track = str(row.get("track") or row.get("filename") or row.get("audio_path") or "")
        user_pick = _float_or_none(row.get("user_pick"))
        if not track or user_pick is None:
            continue
        if not is_human_review_source(row.get("reviewed_from")):
            continue
        entry = dict(row)
        entry["user_pick"] = float(user_pick)
        entry["source"] = "historical_review_memory"
        for key in _review_memory_keys(track):
            memory[key] = entry
    return memory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
    try:
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _stable_id(row: Mapping[str, str]) -> str:
    raw = "|".join([row.get("filename", ""), row.get("output_als", ""), row.get("candidates_json", "")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    minutes = int(seconds // 60)
    sec = seconds - (minutes * 60)
    return f"{minutes}:{sec:06.3f}"


def _default_template_path() -> Path:
    root = Path(__file__).resolve().parent
    for rel in ("alsFiles/128.als", "stuff/CH1.als", "alsFiles/CH1.als"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return root / "alsFiles" / "128.als"


def _default_summary_path() -> Path:
    return DEFAULT_LIBRARY_DIR / DEFAULT_SUMMARY_NAME


def _template_stats(path: Path) -> Dict[str, int]:
    with gzip.open(path, "rb") as fh:
        root = etree.fromstring(fh.read())
    return {
        "audio_clips": int(len(root.xpath('.//*[local-name()="AudioClip"]'))),
        "warp_markers": int(len(root.xpath('.//*[local-name()="WarpMarker"]'))),
    }


def _validate_template_for_regeneration(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    try:
        stats = _template_stats(path)
    except Exception as exc:
        raise ValueError(f"Template is not valid gzip XML: {path} ({exc})") from exc
    if stats["audio_clips"] < 1 or stats["warp_markers"] < 1:
        raise ValueError(
            "Template cannot regenerate corrected ALS files because it has "
            f"{stats['audio_clips']} AudioClips and {stats['warp_markers']} WarpMarkers: {path}"
        )


class ReviewApp:
    def __init__(
        self,
        *,
        summary_csv: str,
        template: str,
        correction_log: str,
        auto_retrain_every: int,
        review_low_only: bool,
        review_medium_and_low: bool,
        regenerate_als_on_correction: bool,
        visual_first: bool = False,
        review_queue: str = "",
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.summary_csv = Path(summary_csv).expanduser().resolve()
        self.template = Path(template).expanduser().resolve()
        self.correction_log = str(Path(correction_log).expanduser())
        self.section_label_log = str(Path(self.correction_log).with_name("drop_section_labels.jsonl"))
        self.structure_label_log = str(Path(self.correction_log).with_name("track_structure_labels.jsonl"))
        self.review_memory = _load_review_memory(self.correction_log)
        self.auto_retrain_every = max(0, int(auto_retrain_every))
        self.review_low_only = bool(review_low_only)
        self.review_medium_and_low = bool(review_medium_and_low)
        self.regenerate_als_on_correction = bool(regenerate_als_on_correction)
        self.visual_first = bool(visual_first)
        self.review_queue = Path(review_queue).expanduser().resolve() if review_queue else None
        self.queue_order = self._load_queue_order(self.review_queue) if self.review_queue else {}
        self.state_path = self.summary_csv.parent / "review_state.json"
        self.preview_dir = self.summary_csv.parent / ".review_previews"
        self.waveform_cache = WaveformCache(self.summary_csv.parent / ".waveform_cache")
        self.lock = threading.RLock()
        historical_logs = [
            self.correction_log,
            self.root / "models" / "multistem_training_corrections.jsonl",
            self.root / "drop_corrections.before_notToBeOrganized_filter_20260501_022657.jsonl",
        ]
        self.historical_markers = load_historical_markers(
            correction_logs=historical_logs,
            marker_db_path=self.root / "drop_marker_db.json",
        )
        self.items = self._load_items()
        if self.regenerate_als_on_correction:
            _validate_template_for_regeneration(self.template)
        self.item_by_id = {item["id"]: item for item in self.items}
        self.state = self._load_state()
        self._sync_state_items()

    def _load_queue_order(self, queue_path: Optional[Path]) -> Dict[str, int]:
        if queue_path is None or not queue_path.exists():
            return {}
        order: Dict[str, int] = {}
        with open(queue_path, "r", encoding="utf-8", newline="") as fh:
            for index, row in enumerate(csv.DictReader(fh)):
                track = str(row.get("track") or row.get("filename") or "")
                if track:
                    order[track] = int(index)
        return order

    def _load_items(self) -> List[Dict[str, Any]]:
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Batch summary not found: {self.summary_csv}")
        out: List[Dict[str, Any]] = []
        with open(self.summary_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row_has_excluded_path(row):
                    continue
                status = str(row.get("status", "")).strip().lower()
                if status and status not in {"processed", "skipped", "partial"}:
                    continue
                als_valid = str(row.get("als_valid", "")).strip().lower()
                if als_valid and als_valid != "true":
                    continue
                ai_pick = _float_or_none(row.get("detected_drop_time"))
                payload = _read_json(row.get("candidates_json", ""))
                if ai_pick is None:
                    ai_pick = _float_or_none(payload.get("final_ai_pick"))
                if ai_pick is None:
                    continue

                feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
                tier = _normalize_tier(row.get("confidence_tier") or payload.get("confidence_tier") or feature_summary.get("confidence_tier"))
                if self.review_low_only and tier != "LOW":
                    continue
                if self.review_medium_and_low and tier not in {"LOW", "MEDIUM"}:
                    continue
                if self.queue_order and row.get("filename", "") not in self.queue_order:
                    continue

                candidates = payload.get("top_10_candidates") if isinstance(payload.get("top_10_candidates"), list) else []
                if self.visual_first:
                    candidates = _without_historical_candidates(candidates)
                selected_candidate = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
                if self.visual_first and _is_historical_candidate(selected_candidate):
                    selected_candidate = {}
                drumprint_pattern_score = _candidate_metric(selected_candidate, feature_summary, "drumprint_pattern_score")
                fake_hit_penalty = _candidate_metric(selected_candidate, feature_summary, "fake_hit_penalty")
                post_stability = _candidate_metric(selected_candidate, feature_summary, "post_drop_pattern_stability")
                later_match = _candidate_metric(selected_candidate, feature_summary, "later_drop_match_score")
                full_groove_score = _candidate_metric(selected_candidate, feature_summary, "sustained_full_groove_score")
                immediate_groove = _candidate_metric(selected_candidate, feature_summary, "immediate_groove_start_score")
                groove_stability = _candidate_metric(selected_candidate, feature_summary, "groove_stability")
                pre_drop_contrast = _candidate_metric(selected_candidate, feature_summary, "pre_drop_contrast")
                gate_candidate = dict(selected_candidate)
                gate_candidate["confidence_tier"] = tier
                auto_accept = {
                    mode: should_auto_accept(gate_candidate, mode=mode, candidates=candidates[:10], confidence_tier=tier)
                    for mode in ("conservative", "normal", "aggressive")
                }
                item = {
                    "id": _stable_id(row),
                    "filename": row.get("filename", ""),
                    "track_name": Path(row.get("filename", "")).name,
                    "audio_path": row.get("filename", ""),
                    "ai_pick": float(ai_pick),
                    "ai_pick_label": _format_time(float(ai_pick)),
                    "confidence": _float_or_none(row.get("confidence")) or _float_or_none(payload.get("confidence")) or 0.0,
                    "confidence_tier": tier,
                    "selected_by": row.get("selected_by") or payload.get("selected_by") or "",
                    "sustained_full_groove_score": _float_or_none(row.get("sustained_full_groove_score")) or full_groove_score,
                    "immediate_groove_start_score": _float_or_none(row.get("immediate_groove_start_score")) or immediate_groove,
                    "groove_stability": _float_or_none(row.get("groove_stability")) or groove_stability,
                    "pre_drop_contrast": _float_or_none(row.get("pre_drop_contrast")) or pre_drop_contrast,
                    "drumprint_pattern_score": _float_or_none(row.get("drumprint_pattern_score")) or drumprint_pattern_score,
                    "fake_hit_penalty": _float_or_none(row.get("fake_hit_penalty")) or fake_hit_penalty,
                    "post_drop_pattern_stability": post_stability,
                    "later_drop_match_score": later_match,
                    "output_als": row.get("output_als", ""),
                    "candidates_json": row.get("candidates_json", ""),
                    "debug_png": row.get("debug_png", ""),
                    "status": row.get("status", ""),
                    "top_10_candidates": candidates[:10],
                    "selected_candidate": dict(selected_candidate),
                    "feature_summary": dict(feature_summary),
                    "auto_accept": auto_accept,
                    "bpm": _infer_bpm_from_path(row.get("filename", "")) or _float_or_none(payload.get("bpm")) or _float_or_none(feature_summary.get("bpm")) or 128.0,
                    "duration_sec": _float_or_none(feature_summary.get("duration_sec")) or 0.0,
                    "als_valid": row.get("als_valid", ""),
                    "als_validation_error": row.get("als_validation_error", ""),
                    "row_status": row.get("status", ""),
                    "review": {},
                }
                if self.visual_first:
                    self._prime_item_with_visual_candidate(item, scan=False)
                else:
                    self._prime_item_with_historical_marker(item)
                out.append(item)
        if self.queue_order:
            out.sort(key=lambda item: (self.queue_order.get(item["filename"], 999999), item["track_name"].lower()))
        else:
            out.sort(key=lambda item: (TIER_ORDER.get(item["confidence_tier"], 3), item["track_name"].lower()))
        return out

    def _prime_item_with_historical_marker(self, item: Dict[str, Any]) -> None:
        result: Dict[str, Any] = {}
        memory = self._review_memory_for_item(item)
        if memory:
            result = self._historical_review_auto_place(item, memory)
        if not result:
            historical = self.historical_markers.find(str(item.get("audio_path", "")), bpm=item.get("bpm"))
            if historical:
                result = self._historical_auto_place(
                    item,
                    historical,
                    list(item.get("top_10_candidates") or []),
                    confidence_tier=str(item.get("confidence_tier", "UNKNOWN")),
                )
        if not result:
            visual_primary = _primary_visual_phrase_candidate(
                list(item.get("top_10_candidates") or []),
                item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else None,
                bpm=_float_or_none(item.get("bpm")),
            )
            if visual_primary:
                candidates = _dedupe_candidate_dicts([visual_primary] + [dict(row) for row in item.get("top_10_candidates") or []])
                suggestion = {
                    "candidate": dict(candidates[0]),
                    "suggested_time": _candidate_marker_time(candidates[0]),
                    "auto_accept": _review_required_auto_accept(
                        "visual 32-bar phrase prior requires review",
                        candidates[0],
                        candidates,
                        str(item.get("confidence_tier", "UNKNOWN")),
                    ),
                }
                result = {
                    "source": "visual_primary_phrase_prior",
                    "candidates": candidates,
                    "suggestion": suggestion,
                }
        if not result:
            return

        candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
        suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), Mapping) else {}
        selected = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else None
        if candidates:
            item["top_10_candidates"] = [dict(candidate) for candidate in candidates[:10] if isinstance(candidate, Mapping)]
            selected_time = _candidate_marker_time(selected) if isinstance(selected, Mapping) else None
            if selected_time is not None:
                for candidate in item["top_10_candidates"]:
                    candidate_time = _candidate_marker_time(candidate)
                    if candidate_time is not None and abs(float(candidate_time) - float(selected_time)) <= 0.010:
                        selected = dict(candidate)
                        break
        if selected:
            item["selected_candidate"] = dict(selected)
            item["selected_by"] = str(selected.get("selected_by") or result.get("source") or item.get("selected_by") or "")
        elif result.get("source"):
            item["selected_by"] = str(result.get("source"))
        if isinstance(suggestion.get("auto_accept"), Mapping):
            item["auto_accept"] = dict(suggestion.get("auto_accept") or {})
        item["initial_candidate_source"] = str(result.get("source") or "")
        suggested_time = _float_or_none(suggestion.get("suggested_time"))
        if suggested_time is not None:
            item["initial_candidate_time"] = float(suggested_time)

    def _apply_visual_first_result(self, item: Dict[str, Any], visual: Mapping[str, Any]) -> bool:
        if not visual.get("ok"):
            return False
        selected = visual.get("selected_candidate") if isinstance(visual.get("selected_candidate"), Mapping) else {}
        marker = _float_or_none(visual.get("marker")) or _candidate_marker_time(selected)
        if marker is None:
            return False
        visual_candidates = [dict(row) for row in visual.get("candidates") or [] if isinstance(row, Mapping)]
        candidates = _dedupe_candidate_dicts(
            [dict(selected)] + visual_candidates + _without_historical_candidates(item.get("top_10_candidates") or [])
        )
        if not candidates:
            return False
        item["top_10_candidates"] = candidates[:10]
        item["selected_candidate"] = dict(candidates[0])
        item["selected_by"] = str(candidates[0].get("selected_by") or "visual_gui_first_fat_block")
        item["ai_pick"] = float(marker)
        item["ai_pick_label"] = _format_time(float(marker))
        item["initial_candidate_time"] = float(marker)
        item["initial_candidate_source"] = item["selected_by"]
        item["visual_first_scan"] = {
            "version": visual.get("version"),
            "raw_visual_time": visual.get("raw_visual_time"),
            "marker": float(marker),
            "source": item["selected_by"],
        }
        item["auto_accept"] = _review_required_auto_accept(
            "visual-first waveform marker requires review",
            candidates[0],
            candidates,
            str(item.get("confidence_tier", "UNKNOWN")),
        )
        return True

    def _apply_review_memory_result(self, item: Dict[str, Any], result: Mapping[str, Any]) -> bool:
        suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), Mapping) else {}
        selected = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else {}
        marker = _float_or_none(suggestion.get("suggested_time")) or _candidate_marker_time(selected)
        if marker is None:
            return False
        candidates = [dict(row) for row in result.get("candidates") or [] if isinstance(row, Mapping)]
        if not candidates:
            candidates = [dict(selected)]
        item["top_10_candidates"] = _dedupe_candidate_dicts(candidates)[:10]
        item["selected_candidate"] = dict(selected)
        item["selected_by"] = "historical_review_memory"
        item["ai_pick"] = float(marker)
        item["ai_pick_label"] = _format_time(float(marker))
        item["initial_candidate_time"] = float(marker)
        item["initial_candidate_source"] = "historical_review_memory"
        if isinstance(suggestion.get("auto_accept"), Mapping):
            item["auto_accept"] = dict(suggestion.get("auto_accept") or {})
        return True

    def _prime_item_with_visual_candidate(self, item: Dict[str, Any], *, scan: bool = False) -> None:
        item["top_10_candidates"] = _without_historical_candidates(item.get("top_10_candidates") or [])
        selected = item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else {}
        if _is_historical_candidate(selected):
            item["selected_candidate"] = {}
            item["selected_by"] = ""
        if scan and not item.get("visual_first_scanned"):
            item["visual_first_scanned"] = True
            memory = self._review_memory_for_item(item)
            if memory:
                remembered = self._historical_review_auto_place(item, memory)
                if remembered and self._apply_review_memory_result(item, remembered):
                    return
            audio_path = str(item.get("audio_path") or "")
            if audio_path:
                try:
                    visual = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
                except Exception as exc:
                    item["visual_first_scan_error"] = str(exc) or exc.__class__.__name__
                else:
                    if self._apply_visual_first_result(item, visual):
                        item.pop("visual_first_scan_error", None)
                        return
                    item["visual_first_scan_error"] = str(visual.get("error") or "visual_first_no_marker")
        visual_primary = _primary_visual_phrase_candidate(
            list(item.get("top_10_candidates") or []),
            item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else None,
            bpm=_float_or_none(item.get("bpm")),
        )
        if not visual_primary:
            selected = item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else {}
            selected_time = _candidate_marker_time(selected)
            if selected_time is not None and not _is_historical_candidate(selected):
                item["initial_candidate_time"] = float(selected_time)
            return
        candidates = _dedupe_candidate_dicts([visual_primary] + [dict(row) for row in item.get("top_10_candidates") or []])
        item["top_10_candidates"] = candidates[:10]
        item["selected_candidate"] = dict(candidates[0])
        item["selected_by"] = str(candidates[0].get("selected_by") or "visual_primary_phrase_prior")
        marker = _candidate_marker_time(candidates[0])
        if marker is not None:
            item["initial_candidate_time"] = float(marker)
        item["auto_accept"] = _review_required_auto_accept(
            "visual-first candidate requires review",
            candidates[0],
            candidates,
            str(item.get("confidence_tier", "UNKNOWN")),
        )
        item["initial_candidate_source"] = str(item["selected_by"])

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    state = json.load(fh)
                if isinstance(state, dict):
                    state.setdefault("items", {})
                    state.setdefault("current_index", 0)
                    state.setdefault("new_reviews_since_retrain", 0)
                    return state
            except Exception:
                pass
        return {"items": {}, "current_index": 0, "new_reviews_since_retrain": 0}

    def _save_state(self) -> None:
        _write_json(self.state_path, self.state)

    def _sync_state_items(self) -> None:
        items = self.state.setdefault("items", {})
        active_ids = {item["id"] for item in self.items}
        for item_id in list(items.keys()):
            if item_id not in active_ids:
                items.pop(item_id, None)
        for item in self.items:
            item_state = items.setdefault(
                item["id"],
                {
                    "reviewed": False,
                    "skipped": False,
                    "approved": False,
                    "corrected": False,
                    "user_pick": None,
                    "timestamp_reviewed": "",
                    "regenerated_als_path": "",
                    "als_valid": "",
                    "als_validation_error": "",
                },
            )
            item["review"] = item_state
        self.state["current_index"] = self._first_active_index(int(self.state.get("current_index", 0)))
        self._save_state()

    def _completed(self, item: Mapping[str, Any]) -> bool:
        review = item.get("review", {})
        return bool(review.get("reviewed") or review.get("skipped"))

    def _first_active_index(self, start: int = 0) -> int:
        if not self.items:
            return 0
        start = max(0, min(start, len(self.items) - 1))
        for idx in range(start, len(self.items)):
            if not self._completed(self.items[idx]):
                return idx
        for idx in range(0, start):
            if not self._completed(self.items[idx]):
                return idx
        return start

    def _current_item(self) -> Optional[Dict[str, Any]]:
        if not self.items:
            return None
        idx = max(0, min(int(self.state.get("current_index", 0)), len(self.items) - 1))
        return self.items[idx]

    def _item_public(self, item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        if self.visual_first and isinstance(item, dict):
            self._prime_item_with_visual_candidate(item, scan=True)
        preview = self._preview_info(item)
        out = dict(item)
        try:
            info = self.waveform_cache.info(str(item.get("audio_path", "")))
        except Exception:
            info = {
                "sample_rate": 44100,
                "total_samples": 0,
                "duration": float(item.get("duration_sec") or preview["duration"]),
                "channels": 0,
            }
        out["audio_url"] = f"/media/audio?id={item['id']}&view=window"
        out["audio_url_full"] = ""
        out["audio_url_window"] = f"/media/audio?id={item['id']}&view=window"
        out["debug_url"] = f"/media/debug?id={item['id']}" if item.get("debug_png") else ""
        out["preview_offset_sec"] = preview["offset"]
        out["preview_duration_sec"] = preview["duration"]
        out["sample_rate"] = int(info.get("sample_rate") or 44100)
        out["total_samples"] = int(info.get("total_samples") or 0)
        out["full_duration_sec"] = float(info.get("duration") or item.get("duration_sec") or preview["duration"])
        out["stem_waveforms"] = self._stem_waveform_sources(item)
        out["review"] = dict(item.get("review", {}))
        return out

    def state_payload(self) -> Dict[str, Any]:
        with self.lock:
            reviewed = sum(1 for item in self.items if item["review"].get("reviewed"))
            skipped = sum(1 for item in self.items if item["review"].get("skipped"))
            corrected = sum(1 for item in self.items if item["review"].get("corrected"))
            approved = sum(1 for item in self.items if item["review"].get("approved"))
            current = self._current_item()
            return {
                "summary_csv": str(self.summary_csv),
                "template": str(self.template),
                "correction_log": self.correction_log,
                "state_path": str(self.state_path),
                "current_index": int(self.state.get("current_index", 0)),
                "total": len(self.items),
                "counts": {
                    "reviewed": reviewed,
                    "skipped": skipped,
                    "corrected": corrected,
                    "approved": approved,
                    "remaining": max(0, len(self.items) - reviewed - skipped),
                },
                "auto_retrain_every": self.auto_retrain_every,
                "new_reviews_since_retrain": int(self.state.get("new_reviews_since_retrain", 0)),
                "regenerate_als_on_correction": self.regenerate_als_on_correction,
                "visual_first": self.visual_first,
                "current": self._item_public(current),
            }

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self._item_public(self.item_by_id.get(item_id))

    def navigate(self, direction: str) -> Dict[str, Any]:
        with self.lock:
            if not self.items:
                return self.state_payload()
            idx = int(self.state.get("current_index", 0))
            if direction == "previous":
                idx = max(0, idx - 1)
            elif direction == "next":
                idx = min(len(self.items) - 1, idx + 1)
            else:
                try:
                    idx = int(direction)
                except ValueError:
                    idx = self._first_active_index(idx)
            self.state["current_index"] = idx
            self._save_state()
        return self.state_payload()

    def _advance(self) -> None:
        if not self.items:
            return
        current = int(self.state.get("current_index", 0))
        self.state["current_index"] = self._first_active_index(current + 1)

    def _preview_info(
        self,
        item: Mapping[str, Any],
        view: str = "window",
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Dict[str, float]:
        if start is not None or end is not None:
            duration_hint = _float_or_none(item.get("duration_sec")) or 0.0
            if duration_hint <= 0:
                try:
                    duration_hint = float(self.waveform_cache.info(str(item.get("audio_path", ""))).get("duration") or 0.0)
                except Exception:
                    duration_hint = 0.0
            safe_start = max(0.0, float(start or 0.0))
            safe_end = float(end if end is not None else safe_start + PREVIEW_BEFORE_SEC + PREVIEW_AFTER_SEC)
            if duration_hint > 0:
                safe_start = min(safe_start, duration_hint)
                safe_end = min(max(safe_end, safe_start), duration_hint)
            duration = max(0.05, min(MAX_PREVIEW_DURATION_SEC, safe_end - safe_start))
            return {"offset": float(safe_start), "duration": float(duration)}
        ai_pick = float(item.get("ai_pick", 0.0))
        offset = max(0.0, ai_pick - PREVIEW_BEFORE_SEC)
        return {"offset": offset, "duration": PREVIEW_BEFORE_SEC + PREVIEW_AFTER_SEC}

    def _preview_path(
        self,
        item: Mapping[str, Any],
        view: str = "window",
        suffix: str = "wav",
        *,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
        source_path: str = "",
    ) -> Path:
        raw = f"{item['id']}|{item.get('ai_pick')}|{view}|{offset}|{duration}|{source_path}"
        key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return self.preview_dir / f"{key}.{suffix}"

    def _stem_audio_source(self, item: Mapping[str, Any], role: str = "drums") -> Path:
        requested = str(role or "drums").strip().lower()
        if requested in {"inst", "instrument", "instruments"}:
            requested = "instrumental"
        if requested in {"vocal", "vox"}:
            requested = "vocals"
        if requested in {"primary", "main", "drums"}:
            return Path(str(item.get("audio_path", ""))).expanduser()
        for source in self._stem_waveform_sources(item):
            if str(source.get("role", "")).lower() == requested and source.get("audio_path"):
                return Path(str(source.get("audio_path", ""))).expanduser()
        return Path(str(item.get("audio_path", ""))).expanduser()

    def media_audio_path(
        self,
        item_id: str,
        view: str = "window",
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        role: str = "drums",
    ) -> tuple[Optional[Path], str]:
        item = self.item_by_id.get(item_id)
        if not item:
            return None, "application/octet-stream"
        source = self._stem_audio_source(item, role=role)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and source.exists():
            view = "range" if start is not None or end is not None else "window"
            info = self._preview_info(item, view=view, start=start, end=end)
            preview = self._preview_path(
                item,
                view=f"{view}:{role or 'drums'}",
                suffix="wav",
                offset=round(info["offset"], 3),
                duration=round(info["duration"], 3),
                source_path=str(source),
            )
            if not preview.exists():
                preview.parent.mkdir(parents=True, exist_ok=True)
                cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
                cmd += ["-ss", f"{info['offset']:.3f}", "-t", f"{info['duration']:.3f}"]
                cmd += ["-i", str(source), "-ac", "2", "-ar", "44100", "-vn"]
                cmd.append(str(preview))
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    preview.unlink(missing_ok=True)
            if preview.exists():
                return preview, "audio/wav"
        return None, "application/octet-stream"

    def waveform_tile(self, item_id: str, start: float, end: float, width: int) -> Dict[str, Any]:
        item = self.item_by_id.get(item_id)
        if not item:
            return {"ok": False, "error": "unknown item"}
        return self.waveform_cache.tile(
            str(item.get("audio_path", "")),
            start_sec=float(start),
            end_sec=float(end),
            width=int(width),
        )

    def _stem_waveform_sources(self, item: Mapping[str, Any]) -> List[Dict[str, Any]]:
        audio_path = str(item.get("audio_path", ""))
        roles: Dict[str, str] = {}
        if audio_path:
            try:
                roles = find_stem_group(audio_path).roles
            except Exception:
                roles = {}
        if not roles and audio_path:
            roles = {"drums": audio_path}

        sources: List[Dict[str, Any]] = []
        for role, label in WAVEFORM_STEM_ROLES:
            path_text = roles.get(role)
            if not path_text:
                continue
            path = Path(str(path_text)).expanduser()
            sources.append(
                {
                    "role": role,
                    "label": label,
                    "audio_path": str(path),
                    "filename": path.name,
                    "available": bool(path.exists()),
                }
            )
        if not any(source.get("available") for source in sources) and audio_path:
            path = Path(audio_path).expanduser()
            sources.insert(
                0,
                {
                    "role": "drums",
                    "label": "Drums",
                    "audio_path": str(path),
                    "filename": path.name,
                    "available": bool(path.exists()),
                },
            )
        return sources

    def waveform_stems_tile(self, item_id: str, start: float, end: float, width: int) -> Dict[str, Any]:
        item = self.item_by_id.get(item_id)
        if not item:
            return {"ok": False, "error": "unknown item"}

        tiles: List[Dict[str, Any]] = []
        for source in self._stem_waveform_sources(item):
            tile_base = dict(source)
            if not source.get("available"):
                tile_base.update({"ok": False, "error": "stem file missing"})
                tiles.append(tile_base)
                continue
            try:
                tile = self.waveform_cache.tile(
                    str(source.get("audio_path", "")),
                    start_sec=float(start),
                    end_sec=float(end),
                    width=int(width),
                )
                tile.update(tile_base)
                tiles.append(tile)
            except Exception as exc:
                tile_base.update({"ok": False, "error": str(exc) or exc.__class__.__name__})
                tiles.append(tile_base)

        usable = [tile for tile in tiles if tile.get("ok") is not False and tile.get("duration") is not None]
        primary = usable[0] if usable else {}
        return {
            "ok": bool(usable),
            "tiles": tiles,
            "duration": float(primary.get("duration") or item.get("duration_sec") or 0.0),
            "sample_rate": int(primary.get("sample_rate") or 44100),
            "primary_role": str(primary.get("role") or "drums"),
            "error": "" if usable else "no stem waveform tiles available",
        }

    def _selected_micro(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        selected = item.get("selected_candidate") if isinstance(item.get("selected_candidate"), Mapping) else {}
        micro = selected.get("microalign") if isinstance(selected.get("microalign"), Mapping) else {}
        if micro:
            return dict(micro)
        features = item.get("feature_summary") if isinstance(item.get("feature_summary"), Mapping) else {}
        out: Dict[str, Any] = {}
        for key in ("microaligned_time", "attack_start_time", "zero_crossing_time", "visual_onset_knee_time", "ableton_asd_time"):
            value = _float_or_none(features.get(f"chosen_{key}"))
            if value is not None and value > 0.0:
                out[key] = value
        return out

    def _png_markers(
        self,
        item: Mapping[str, Any],
        *,
        user_pick: Optional[float] = None,
        refined_pick: Optional[float] = None,
        attack_time: Optional[float] = None,
        zero_time: Optional[float] = None,
        knee_time: Optional[float] = None,
        asd_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        markers: List[Dict[str, Any]] = []
        for cand in item.get("top_10_candidates") or []:
            if not isinstance(cand, Mapping):
                continue
            timestamp = _float_or_none(cand.get("timestamp"))
            if timestamp is not None:
                markers.append({"kind": "candidate", "time": timestamp, "label": f"#{cand.get('rank', '')}"})
        ai = _float_or_none(item.get("ai_pick"))
        if ai is not None:
            markers.append({"kind": "ai", "time": ai, "label": "AI"})
        micro = self._selected_micro(item)
        micro_time = refined_pick if refined_pick is not None else _float_or_none(micro.get("microaligned_time"))
        if micro_time is not None:
            markers.append({"kind": "micro", "time": micro_time, "label": "MICRO"})
        attack = attack_time if attack_time is not None else _float_or_none(micro.get("attack_start_time"))
        if attack is not None:
            markers.append({"kind": "attack", "time": attack, "label": "ATTACK"})
        zero = zero_time if zero_time is not None else _float_or_none(micro.get("zero_crossing_time"))
        if zero is not None:
            markers.append({"kind": "zero", "time": zero, "label": "ZERO"})
        knee = knee_time if knee_time is not None else _float_or_none(micro.get("visual_onset_knee_time"))
        if knee is not None and knee > 0.0:
            markers.append({"kind": "knee", "time": knee, "label": "KNEE"})
        asd = asd_time if asd_time is not None else _float_or_none(micro.get("ableton_asd_time"))
        if asd is not None and asd > 0.0:
            markers.append({"kind": "asd", "time": asd, "label": "ASD"})
        if user_pick is not None:
            markers.append({"kind": "user", "time": user_pick, "label": "YOU"})
        return markers

    def waveform_png(
        self,
        item_id: str,
        *,
        start: float,
        end: float,
        width: int,
        height: int,
        user_pick: Optional[float] = None,
        refined_pick: Optional[float] = None,
        attack_time: Optional[float] = None,
        zero_time: Optional[float] = None,
        knee_time: Optional[float] = None,
        asd_time: Optional[float] = None,
    ) -> Optional[bytes]:
        item = self.item_by_id.get(item_id)
        if not item:
            return None
        return self.waveform_cache.render_png(
            str(item.get("audio_path", "")),
            start_sec=float(start),
            end_sec=float(end),
            width=int(width),
            height=int(height),
            markers=self._png_markers(
                item,
                user_pick=user_pick,
                refined_pick=refined_pick,
                attack_time=attack_time,
                zero_time=zero_time,
                knee_time=knee_time,
                asd_time=asd_time,
            ),
        )

    def media_debug_path(self, item_id: str) -> Optional[Path]:
        item = self.item_by_id.get(item_id)
        if not item:
            return None
        path = Path(str(item.get("debug_png", ""))).expanduser()
        return path if path.exists() else None

    def _candidate_context(
        self,
        item: Mapping[str, Any],
        *,
        top_candidates: Optional[Any] = None,
        selected_candidate: Optional[Any] = None,
        selected_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        candidates = top_candidates if isinstance(top_candidates, list) else item.get("top_10_candidates") or []
        candidates = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        selected = dict(selected_candidate) if isinstance(selected_candidate, Mapping) else dict(item.get("selected_candidate") or {})
        selected_by_value = selected_by or selected.get("selected_by") or item.get("selected_by") or ""
        return {
            "top_candidates": candidates,
            "selected_candidate": selected,
            "selected_by": str(selected_by_value),
        }

    def _remember_review(
        self,
        item: Mapping[str, Any],
        user_pick: float,
        *,
        reviewed_from: str,
        top_candidates: Optional[Any] = None,
        selected_candidate: Optional[Any] = None,
        selected_by: Optional[str] = None,
    ) -> None:
        context = self._candidate_context(
            item,
            top_candidates=top_candidates,
            selected_candidate=selected_candidate,
            selected_by=selected_by,
        )
        track = str(item.get("audio_path", ""))
        entry = {
            "source": "historical_review_memory",
            "track": track,
            "ai_pick": float(item.get("ai_pick", 0.0) or 0.0),
            "user_pick": float(user_pick),
            "reviewed_from": str(reviewed_from),
            "selected_by": context["selected_by"],
            "selected_candidate": context["selected_candidate"],
            "top_10_candidates": context["top_candidates"],
            "confidence_tier": str(item.get("confidence_tier", "UNKNOWN")),
            "timestamp": _now_iso(),
        }
        for key in _review_memory_keys(track):
            self.review_memory[key] = dict(entry)

    def _review_memory_for_item(self, item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        for key in _review_memory_keys(str(item.get("audio_path", ""))):
            entry = self.review_memory.get(key)
            user_pick = _float_or_none((entry or {}).get("user_pick")) if isinstance(entry, Mapping) else None
            if user_pick is not None:
                return dict(entry)
        return None

    def _historical_review_auto_place(self, item: Mapping[str, Any], memory: Mapping[str, Any]) -> Dict[str, Any]:
        user_pick = _float_or_none(memory.get("user_pick"))
        if user_pick is None:
            return {}
        bpm = _float_or_none(item.get("bpm")) or _infer_bpm_from_path(str(item.get("audio_path", "")))
        candidate = dict(memory.get("selected_candidate") if isinstance(memory.get("selected_candidate"), Mapping) else {})
        candidate.update(
            {
                "timestamp": float(user_pick),
                "snapped_sec": float(user_pick),
                "microaligned_time": float(user_pick),
                "selected": True,
                "selected_by": "historical_review_memory",
                "reason": "historical approved/corrected marker from correction log",
                "confidence_score": 1.0,
                "score": 1.0,
            }
        )
        micro = dict(candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {})
        micro.update(
            {
                "microaligned_time": float(user_pick),
                "input_candidate_time": float(user_pick),
                "snap_offset_ms": 0.0,
                "reason": "historical approved/corrected marker from correction log",
                "review_needed": False,
            }
        )
        candidate["microalign"] = micro
        clock = _bpm_clock_for_time(user_pick, bpm)
        if clock:
            candidate["bpm_clock"] = clock
        bar_prior = _bar_prior_for_time(user_pick, bpm)
        if bar_prior:
            candidate["musical_bar_prior"] = bar_prior
        candidates = [candidate]
        for row in list(memory.get("top_10_candidates") if isinstance(memory.get("top_10_candidates"), list) else []) + list(
            item.get("top_10_candidates") or []
        ):
            if isinstance(row, Mapping):
                candidates.append(dict(row))
        candidates = _dedupe_candidate_dicts(candidates)
        auto_accept = {
            gate_mode: {
                "auto_accept": True,
                "risk_flags": [],
                "reason": "historical review memory exact marker",
                "mode": gate_mode,
            }
            for gate_mode in ("conservative", "normal", "aggressive")
        }
        suggestion = {
            "auto_place": True,
            "review_needed": False,
            "risk": False,
            "mode": "historical_review_memory",
            "confidence_tier": str(memory.get("confidence_tier") or item.get("confidence_tier", "UNKNOWN")),
            "candidate": candidate,
            "microalign": candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else None,
            "suggested_time": float(user_pick),
            "score": 1.0,
            "reason": "previous human-approved marker reused exactly",
            "bpm_clock": clock or {},
            "bar_prior": bar_prior or {},
            "auto_accept": auto_accept,
        }
        return {
            "ok": True,
            "source": "historical_review_memory",
            "source_info": {
                "historical_review": {
                    "reviewed_from": memory.get("reviewed_from", ""),
                    "selected_by": memory.get("selected_by", ""),
                    "timestamp": memory.get("timestamp", ""),
                    "correction_log": self.correction_log,
                }
            },
            "candidates": candidates,
            "suggestion": suggestion,
        }

    def _update_candidate_json_user_pick(
        self,
        item: Mapping[str, Any],
        user_pick: float,
        *,
        reviewed_from: str = "web_review",
        top_candidates: Optional[Any] = None,
        selected_candidate: Optional[Any] = None,
        selected_by: Optional[str] = None,
    ) -> None:
        path = Path(str(item.get("candidates_json", ""))).expanduser()
        if not path.exists():
            return
        payload = _read_json(str(path))
        context = self._candidate_context(
            item,
            top_candidates=top_candidates,
            selected_candidate=selected_candidate,
            selected_by=selected_by,
        )
        payload["user_pick"] = float(user_pick)
        payload["corrected_drop_time"] = float(user_pick)
        payload["reviewed_from"] = str(reviewed_from)
        if context["top_candidates"]:
            payload["top_10_candidates"] = context["top_candidates"]
        if context["selected_candidate"]:
            payload["selected_candidate"] = context["selected_candidate"]
        if context["selected_by"]:
            payload["selected_by"] = context["selected_by"]
        payload["closest_candidate_to_user_pick"] = closest_candidate_to_pick(
            context["top_candidates"] or payload.get("top_10_candidates", []),
            float(user_pick),
        )
        payload["reviewed_at"] = _now_iso()
        _write_json(path, payload)

    def regenerate_als(
        self,
        item_id: str,
        marker_time: float,
        *,
        reviewed_from: str = "web_review",
        top_candidates: Optional[Any] = None,
        selected_candidate: Optional[Any] = None,
        selected_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            output = Path(str(item.get("output_als", ""))).expanduser()
            if output.exists():
                backup = output.with_name(output.stem + ".previous" + output.suffix)
                shutil.copy2(output, backup)
            try:
                modify_als(
                    template_path=str(self.template),
                    audio_path=str(item.get("audio_path", "")),
                    drop_sec=float(marker_time),
                    bpm=float(item.get("bpm") or 128.0),
                    output_path=str(output),
                    strict_stems=True,
                )
                self._update_candidate_json_user_pick(
                    item,
                    float(marker_time),
                    reviewed_from=reviewed_from,
                    top_candidates=top_candidates,
                    selected_candidate=selected_candidate,
                    selected_by=selected_by,
                )
                report = verify_als(str(output), candidates_json=str(item.get("candidates_json", "")) or None)
                review = item["review"]
                review["regenerated_als_path"] = str(output)
                review["als_valid"] = bool(report.get("valid"))
                review["als_validation_error"] = "; ".join(str(err) for err in report.get("errors", []))
                self._save_state()
                return {"ok": bool(report.get("valid")), "output_als": str(output), "verification": report}
            except Exception as exc:
                return {"ok": False, "error": str(exc) or exc.__class__.__name__}

    def _log_review(
        self,
        item: Mapping[str, Any],
        user_pick: float,
        reviewed_from: str = "web_review",
        *,
        top_candidates: Optional[Any] = None,
        selected_candidate: Optional[Any] = None,
        selected_by: Optional[str] = None,
    ) -> None:
        context = self._candidate_context(
            item,
            top_candidates=top_candidates,
            selected_candidate=selected_candidate,
            selected_by=selected_by,
        )
        log_correction(
            track=str(item.get("audio_path", "")),
            ai_pick=float(item.get("ai_pick", 0.0)),
            user_pick=float(user_pick),
            features=dict(item.get("feature_summary") or {}),
            top_candidates=context["top_candidates"],
            selected_candidate=context["selected_candidate"],
            selected_by=context["selected_by"],
            confidence_tier=str(item.get("confidence_tier", "UNKNOWN")),
            reviewed_from=reviewed_from,
            log_path=self.correction_log,
        )
        self._remember_review(
            item,
            float(user_pick),
            reviewed_from=reviewed_from,
            top_candidates=top_candidates,
            selected_candidate=selected_candidate,
            selected_by=selected_by,
        )

    def refine_marker(self, item_id: str, marker_time: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            marker = float(marker_time if marker_time is not None else item.get("ai_pick", 0.0))
            micro = microalign_marker(str(item.get("audio_path", "")), marker)
            candidate = dict(item.get("selected_candidate") or {})
            candidate["microalign"] = micro
            candidate["confidence_tier"] = str(item.get("confidence_tier", "UNKNOWN"))
            gates = {
                mode: should_auto_accept(
                    candidate,
                    mode=mode,
                    candidates=item.get("top_10_candidates") or [],
                    confidence_tier=str(item.get("confidence_tier", "UNKNOWN")),
                )
                for mode in ("conservative", "normal", "aggressive")
            }
            return {"ok": True, "microalign": micro, "auto_accept": gates}

    def _historical_auto_place(
        self,
        item: Mapping[str, Any],
        marker: HistoricalMarker,
        base_candidates: List[Mapping[str, Any]],
        *,
        confidence_tier: str,
    ) -> Dict[str, Any]:
        bpm = _float_or_none(item.get("bpm")) or _float_or_none(marker.bpm) or _infer_bpm_from_path(str(item.get("audio_path", "")))
        marker_time = float(marker.user_pick)
        clock = _bpm_clock_for_time(marker_time, bpm)
        prior = _bar_prior_for_time(marker_time, bpm)
        candidate: Dict[str, Any] = {
            "rank": 1,
            "handcrafted_rank": 1,
            "historical_rank": 1,
            "timestamp": marker_time,
            "snapped_sec": marker_time,
            "time_sec": marker_time,
            "microaligned_time": marker_time,
            "confidence_score": 1.0,
            "score": 1.0,
            "selected": True,
            "selected_by": "historical_human_marker",
            "structure_role": "first_drop",
            "section_label": "first_drop",
            "reason": f"historical user-approved marker from {marker.source}",
        }
        if clock:
            candidate["bpm_clock"] = clock
        if prior:
            candidate["musical_bar_prior"] = prior
            if isinstance(prior.get("feature_grid_prior"), Mapping):
                candidate["feature_grid_prior"] = dict(prior.get("feature_grid_prior") or {})
        candidates = _dedupe_candidate_dicts([candidate] + [dict(row) for row in base_candidates], radius_sec=0.040)
        selected = dict(candidates[0]) if candidates else candidate
        gates = {
            mode: {
                "auto_accept": True,
                "risk_flags": [],
                "reason": "historical human marker already approved",
                "confidence_tier": _normalize_tier(confidence_tier),
            }
            for mode in ("conservative", "normal", "aggressive")
        }
        suggestion = {
            "auto_place": True,
            "review_needed": False,
            "risk": False,
            "mode": "historical",
            "confidence_tier": _normalize_tier(confidence_tier),
            "candidate": selected,
            "microalign": {
                "ok": True,
                "used": False,
                "input_candidate_time": marker_time,
                "microaligned_time": marker_time,
                "snap_offset_ms": 0.0,
                "micro_confidence": 1.0,
                "reason": "historical marker kept exactly; MicroSnap not applied",
                "review_needed": False,
            },
            "suggested_time": marker_time,
            "score": 1.0,
            "reason": f"historical human-approved marker matched this track exactly ({marker.reviewed_from or marker.source})",
            "bpm_clock": clock or {},
            "bar_prior": prior or {},
            "auto_accept": gates,
        }
        return {
            "ok": True,
            "source": "historical_human_marker",
            "source_info": {
                "historical_marker": marker.as_dict(),
                "historical_marker_count": len(self.historical_markers),
            },
            "candidates": candidates,
            "suggestion": suggestion,
        }

    def auto_place(self, item_id: str, mode: str = "normal") -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            item_snapshot = dict(item)

        mode_text = str(mode or "").strip().lower().replace("-", "_")
        visual_only = mode_text in {"visual_only", "gui_visual", "visual_gui", "visual_first_only"}
        if visual_only:
            audio_path = str(item_snapshot.get("audio_path", ""))
            try:
                visual = visual_first_marker(audio_path, sample_rate=16000, use_cache=True)
            except Exception as exc:
                return {"ok": False, "error": str(exc) or exc.__class__.__name__, "source": "visual_gui_only"}
            if not visual.get("ok"):
                return {"ok": False, "error": visual.get("error") or "visual_only_failed", "source": "visual_gui_only"}
            selected = dict(visual.get("selected_candidate") if isinstance(visual.get("selected_candidate"), Mapping) else {})
            marker = _float_or_none(visual.get("marker")) or _candidate_marker_time(selected)
            micro = selected.get("microalign") if isinstance(selected.get("microalign"), Mapping) else {}
            candidates = [dict(row) for row in visual.get("candidates") or [] if isinstance(row, Mapping)]
            gates = _review_required_auto_accept(
                "visual-only marker requires review",
                selected,
                candidates,
                _normalize_tier(str(item_snapshot.get("confidence_tier", "UNKNOWN"))),
            )
            return {
                "ok": True,
                "source": "visual_gui_only",
                "source_info": {
                    "visual_first": {
                        "version": visual.get("version"),
                        "raw_visual_time": visual.get("raw_visual_time"),
                        "feature_map": visual.get("feature_map") or {},
                    }
                },
                "candidates": candidates,
                "suggestion": {
                    "auto_place": False,
                    "review_needed": True,
                    "risk": True,
                    "mode": "visual_only",
                    "confidence_tier": _normalize_tier(str(item_snapshot.get("confidence_tier", "UNKNOWN"))),
                    "candidate": selected,
                    "microalign": dict(micro) if isinstance(micro, Mapping) else None,
                    "suggested_time": None if marker is None else float(marker),
                    "score": float(selected.get("score", 0.0) or 0.0),
                    "reason": selected.get("reason") or "visual-only selected first fat waveform block",
                    "auto_accept": gates,
                },
            }

        history_enabled = mode_text not in {"raw", "detector", "no_history", "no_historical"}
        if history_enabled:
            memory = self._review_memory_for_item(item_snapshot)
            if memory:
                remembered = self._historical_review_auto_place(item_snapshot, memory)
                if remembered:
                    return remembered

        audio_path = str(item_snapshot.get("audio_path", ""))
        source = "saved_candidates"
        source_info: Dict[str, Any] = {}
        base_candidates = item_snapshot.get("top_10_candidates") or []
        if not history_enabled:
            base_candidates = [
                dict(candidate)
                for candidate in base_candidates
                if isinstance(candidate, Mapping) and not _is_historical_candidate(candidate)
            ]
        effective_confidence_tier = str(item_snapshot.get("confidence_tier", "UNKNOWN"))
        if history_enabled:
            historical = self.historical_markers.find(audio_path, bpm=item_snapshot.get("bpm"))
            if historical:
                return self._historical_auto_place(
                    item_snapshot,
                    historical,
                    base_candidates,
                    confidence_tier=effective_confidence_tier,
                )
        if audio_path:
            try:
                result = choose_multistem_candidate(
                    audio_path,
                    saved_candidates=base_candidates,
                    confidence_tier=effective_confidence_tier,
                    mode=mode,
                    expanded_limit=120,
                    microalign_limit=50,
                    sample_rate=16000,
                )
                candidate_pool = list(result.get("candidates", []) or [])
                candidate_pool.extend(dict(candidate) for candidate in base_candidates if isinstance(candidate, Mapping))
                structure_prior = _apply_structure_map_prior(
                    audio_path,
                    candidate_pool,
                    result.get("suggestion", {}) if isinstance(result.get("suggestion"), Mapping) else {},
                    confidence_tier=effective_confidence_tier,
                    sample_rate=16000,
                )
                structure_map = structure_prior.get("structure_map") if isinstance(structure_prior.get("structure_map"), Mapping) else {}
                beatgrid = structure_map.get("beatgrid") if isinstance(structure_map.get("beatgrid"), Mapping) else {}
                bar_prior = _apply_even_bar_prior(
                    list(structure_prior.get("candidates", []) or []),
                    structure_prior.get("suggestion", {}) if isinstance(structure_prior.get("suggestion"), Mapping) else {},
                    bpm=_float_or_none(item_snapshot.get("bpm")) or _float_or_none(result.get("bpm")),
                    confidence_tier=effective_confidence_tier,
                    bar_zero_sec=_float_or_none(beatgrid.get("bar_zero_sec")),
                    allow_promotion=False,
                )
                bar_prior = _apply_final_visual_block_guard(
                    bar_prior,
                    beatgrid,
                    confidence_tier=effective_confidence_tier,
                )
                return {
                    "ok": True,
                    "source": "structure_map",
                    "source_info": {
                        "stem_group": result.get("stem_group"),
                        "source_count": int(result.get("source_count", 0) or 0),
                        "candidate_count": int(result.get("candidate_count", 0) or 0),
                        "pipeline": result.get("pipeline") or {},
                        "bar_prior": bar_prior.get("bar_prior") or {},
                        "structure_map": {
                            "ok": bool(structure_map.get("ok")),
                            "bar_count": int(structure_map.get("bar_count", 0) or 0),
                            "feature_cache_hit": bool(structure_map.get("feature_cache_hit")),
                            "beatgrid": structure_map.get("beatgrid") or {},
                            "first_drop": structure_map.get("first_drop"),
                            "second_drop": structure_map.get("second_drop"),
                            "sections": structure_map.get("sections") or [],
                        },
                    },
                    "structure_map": structure_map,
                    "candidates": bar_prior.get("candidates", []),
                    "suggestion": bar_prior.get("suggestion", {}),
                }
            except Exception as exc:
                source_info = {
                    "multistem_fallback_reason": str(exc) or exc.__class__.__name__,
                }

        candidates = microalign_candidate_dicts(
            audio_path,
            base_candidates,
            limit=10,
        )
        suggestion = choose_microaligned_candidate(
            candidates,
            confidence_tier=effective_confidence_tier,
            mode=mode,
        )
        structure_prior = _apply_structure_map_prior(
            audio_path,
            list(candidates),
            suggestion,
            confidence_tier=effective_confidence_tier,
            sample_rate=16000,
        )
        structure_map = structure_prior.get("structure_map") if isinstance(structure_prior.get("structure_map"), Mapping) else {}
        beatgrid = structure_map.get("beatgrid") if isinstance(structure_map.get("beatgrid"), Mapping) else {}
        bar_prior = _apply_even_bar_prior(
            list(structure_prior.get("candidates", []) or []),
            structure_prior.get("suggestion", {}) if isinstance(structure_prior.get("suggestion"), Mapping) else {},
            bpm=_float_or_none(item_snapshot.get("bpm")),
            confidence_tier=effective_confidence_tier,
            bar_zero_sec=_float_or_none(beatgrid.get("bar_zero_sec")),
            allow_promotion=False,
        )
        bar_prior = _apply_final_visual_block_guard(
            bar_prior,
            beatgrid,
            confidence_tier=effective_confidence_tier,
        )
        source_info["bar_prior"] = bar_prior.get("bar_prior") or {}
        source_info["structure_map"] = {
            "ok": bool(structure_map.get("ok")),
            "bar_count": int(structure_map.get("bar_count", 0) or 0),
            "feature_cache_hit": bool(structure_map.get("feature_cache_hit")),
            "beatgrid": structure_map.get("beatgrid") or {},
            "first_drop": structure_map.get("first_drop"),
            "second_drop": structure_map.get("second_drop"),
            "sections": structure_map.get("sections") or [],
        }
        return {
            "ok": True,
            "source": "structure_map" if structure_map.get("ok") else source,
            "source_info": source_info,
            "structure_map": structure_map,
            "candidates": bar_prior.get("candidates", []),
            "suggestion": bar_prior.get("suggestion", {}),
        }

    def approve(self, item_id: str) -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            user_pick = float(item.get("ai_pick", 0.0))
            self._log_review(item, user_pick)
            review = item["review"]
            review.update(
                {
                    "reviewed": True,
                    "skipped": False,
                    "approved": True,
                    "corrected": False,
                    "user_pick": user_pick,
                    "timestamp_reviewed": _now_iso(),
                }
            )
            self._after_logged_review()
        return {"ok": True, "state": self.state_payload()}

    def correct(
        self,
        item_id: str,
        user_pick: float,
        reviewed_from: str = "web_review",
        *,
        picked_candidate: Optional[Any] = None,
        top_candidates: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            user_pick = float(user_pick)
            selected_candidate: Optional[Dict[str, Any]] = None
            selected_by: Optional[str] = None
            if isinstance(picked_candidate, Mapping):
                selected_candidate = dict(picked_candidate)
                selected_candidate["selected"] = True
                selected_candidate["selected_by"] = "user_candidate_pick"
                selected_candidate["reason"] = "selected_by_user_candidate_pick"
                selected_candidate["timestamp"] = float(user_pick)
                selected_candidate["snapped_sec"] = float(user_pick)
                selected_by = "user_candidate_pick"
                reviewed_from = "web_candidate_pick" if reviewed_from == "web_manual_marker" else reviewed_from

            self._log_review(
                item,
                user_pick,
                reviewed_from=reviewed_from,
                top_candidates=top_candidates,
                selected_candidate=selected_candidate,
                selected_by=selected_by,
            )
            regen_report: Optional[Dict[str, Any]] = None
            if self.regenerate_als_on_correction:
                regen_report = self.regenerate_als(
                    item_id,
                    user_pick,
                    reviewed_from=reviewed_from,
                    top_candidates=top_candidates,
                    selected_candidate=selected_candidate,
                    selected_by=selected_by,
                )
            else:
                self._update_candidate_json_user_pick(
                    item,
                    user_pick,
                    reviewed_from=reviewed_from,
                    top_candidates=top_candidates,
                    selected_candidate=selected_candidate,
                    selected_by=selected_by,
                )
            review = item["review"]
            review.update(
                {
                    "reviewed": True,
                    "skipped": False,
                    "approved": False,
                    "corrected": True,
                    "user_pick": user_pick,
                    "timestamp_reviewed": _now_iso(),
                }
            )
            self._after_logged_review()
        return {"ok": True, "regeneration": regen_report, "state": self.state_payload()}

    def label_section(
        self,
        item_id: str,
        label: str,
        marker_time: Optional[float] = None,
        *,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        candidate: Optional[Any] = None,
        top_candidates: Optional[Any] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        allowed = {
            "intro",
            "build",
            "drop",
            "breakdown",
            "second_build",
            "first_drop",
            "second_drop",
            "fake_drop",
            "outro",
            "wrong_candidate",
        }
        label_value = str(label or "").strip().lower()
        if label_value not in allowed:
            return {"ok": False, "error": f"unknown section label: {label}"}
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            marker = _float_or_none(marker_time)
            start = _float_or_none(start_sec)
            end = _float_or_none(end_sec)
            selected = dict(candidate) if isinstance(candidate, Mapping) else {}
            if marker is not None:
                selected.setdefault("timestamp", float(marker))
                selected.setdefault("snapped_sec", float(marker))
            selected["section_label"] = label_value
            selected["selected_by"] = str(selected.get("selected_by") or f"web_section_{label_value}")
            bpm = _float_or_none(item.get("bpm"))
            clock = _bpm_clock_for_time(marker, bpm)
            feature_grid = selected.get("feature_grid_prior") if isinstance(selected.get("feature_grid_prior"), Mapping) else {}
            if not feature_grid and isinstance(selected.get("musical_bar_prior"), Mapping):
                maybe_grid = selected["musical_bar_prior"].get("feature_grid_prior")
                feature_grid = maybe_grid if isinstance(maybe_grid, Mapping) else {}
            event = {
                "type": label_value,
                "time_sec": None if marker is None else float(marker),
                "clock_bar": None if not clock else int(clock.get("nearest_one_bar", 0) or 0),
                "grid_bar": None if not feature_grid else int(feature_grid.get("nearest_grid_bar", 0) or 0),
                "beat_in_bar": None if not clock else int(clock.get("beat_in_bar", 0) or 0),
                "one_distance_ms": None if not clock else float(clock.get("one_distance_ms", 0.0) or 0.0),
                "verdict": "rejected_as_real_drop" if label_value in {"fake_drop", "wrong_candidate"} else "approved",
                "candidate_rank": selected.get("rank") or selected.get("picked_candidate_rank"),
                "candidate_time": _candidate_marker_time(selected),
            }
            span = None
            if start is not None or end is not None:
                span = {
                    "label": label_value,
                    "start_sec": None if start is None else float(start),
                    "end_sec": None if end is None else float(end),
                }
            payload = {
                "schema_version": 1,
                "timestamp_logged": _now_iso(),
                "track": str(item.get("audio_path", "")),
                "item_id": str(item.get("id", item_id)),
                "label": label_value,
                "marker_time": None if marker is None else float(marker),
                "ai_pick": float(item.get("ai_pick", 0.0) or 0.0),
                "bpm": float(bpm or 0.0),
                "bpm_clock": clock or {},
                "feature_grid": feature_grid or {},
                "confidence_tier": str(item.get("confidence_tier", "UNKNOWN")),
                "candidate": selected,
                "top_candidates": top_candidates if isinstance(top_candidates, list) else item.get("top_10_candidates") or [],
                "context": context if isinstance(context, Mapping) else {},
            }
            with open(self.section_label_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            structure_payload = {
                "schema_version": 1,
                "timestamp_logged": payload["timestamp_logged"],
                "track": payload["track"],
                "item_id": payload["item_id"],
                "bpm": payload["bpm"],
                "beatgrid": {
                    "clock_zero_sec": 0.0,
                    "feature_grid": feature_grid or {},
                },
                "events": [event],
                "sections": [span] if span else [],
                "candidate_feedback": [event] if label_value in {"fake_drop", "wrong_candidate"} else [],
                "candidate": selected,
                "context": payload["context"],
            }
            with open(self.structure_label_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(structure_payload, ensure_ascii=True, sort_keys=True) + "\n")
            review = item["review"]
            labels = list(review.get("section_labels") or [])
            labels.append(
                {
                    "label": label_value,
                    "marker_time": marker,
                    "start_sec": start,
                    "end_sec": end,
                    "clock_bar": event.get("clock_bar"),
                    "grid_bar": event.get("grid_bar"),
                    "timestamp_reviewed": _now_iso(),
                }
            )
            review["section_labels"] = labels
            structure_review = dict(review.get("structure_review") or {})
            events = list(structure_review.get("events") or [])
            events.append(event)
            structure_review["events"] = events
            if span:
                spans = list(structure_review.get("spans") or [])
                spans.append(span)
                structure_review["spans"] = spans
            structure_review["status"] = "in_progress"
            review["structure_review"] = structure_review
            self._save_state()
            return {
                "ok": True,
                "label": label_value,
                "log_path": self.section_label_log,
                "structure_log_path": self.structure_label_log,
                "state": self.state_payload(),
            }

    def skip(self, item_id: str) -> Dict[str, Any]:
        with self.lock:
            item = self.item_by_id.get(item_id)
            if not item:
                return {"ok": False, "error": "unknown item"}
            item["review"].update(
                {
                    "reviewed": False,
                    "skipped": True,
                    "approved": False,
                    "corrected": False,
                    "timestamp_reviewed": _now_iso(),
                }
            )
            self._advance()
            self._save_state()
        return {"ok": True, "state": self.state_payload()}

    def _after_logged_review(self) -> None:
        self.state["new_reviews_since_retrain"] = int(self.state.get("new_reviews_since_retrain", 0)) + 1
        self._advance()
        self._save_state()
        if self.auto_retrain_every and int(self.state.get("new_reviews_since_retrain", 0)) >= self.auto_retrain_every:
            self.state["retrain_due"] = True
            self.state["retrain_due_since"] = _now_iso()
            self._save_state()

    def run_retrain(self) -> Dict[str, Any]:
        code = _run_retrain(self.correction_log)
        report_path = Path("models/promotion_report.json")
        report = _read_json(str(report_path)) if report_path.exists() else {}
        rerank_report: Dict[str, Any] = {}
        if code == 0:
            try:
                rerank_report = rerank_summary_with_model(self.summary_csv, model_path="models/drop_ranker.pkl", backup=True)
                self.items = self._load_items()
                self.item_by_id = {item["id"]: item for item in self.items}
                self._sync_state_items()
            except Exception as exc:
                rerank_report = {"ok": False, "error": str(exc) or exc.__class__.__name__}
        self.state["last_retrain"] = {"return_code": int(code), "timestamp": _now_iso(), "promotion_report": report}
        if rerank_report:
            self.state["last_retrain"]["rerank_report"] = rerank_report
        self.state["new_reviews_since_retrain"] = 0
        self.state["retrain_due"] = False
        self.state.pop("retrain_due_since", None)
        self._save_state()
        return {"ok": code == 0, "return_code": int(code), "promotion_report": report, "rerank_report": rerank_report}


def _make_handler(app: ReviewApp):
    client_disconnect_errors = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

    class Handler(BaseHTTPRequestHandler):
        def handle(self) -> None:
            try:
                super().handle()
            except client_disconnect_errors:
                return

        def log_message(self, fmt: str, *args: Any) -> None:
            if self.path.startswith("/api/waveform_tile") or self.path.startswith("/api/waveform_stems_tile"):
                return
            print(f"{self.address_string()} - {fmt % args}")

        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            data = json.dumps(dict(payload), ensure_ascii=True).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)
            except client_disconnect_errors:
                return

        def _send_file(self, path: Path, mime: Optional[str] = None) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                with open(path, "rb") as fh:
                    shutil.copyfileobj(fh, self.wfile)
            except client_disconnect_errors:
                return

        def _send_bytes(self, data: bytes, mime: str) -> None:
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)
            except client_disconnect_errors:
                return

        def _read_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if path == "/":
                self._send_file(app.root / "static" / "index.html", "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._send_file(app.root / path.lstrip("/"))
            elif path == "/api/state":
                self._send_json(app.state_payload())
            elif path.startswith("/api/track"):
                item_id = (qs.get("id") or [""])[0]
                item = app.get_item(item_id) if item_id else app.state_payload().get("current")
                self._send_json({"ok": item is not None, "item": item})
            elif path == "/api/waveform_tile":
                item_id = (qs.get("id") or [""])[0]
                start = float((qs.get("start") or ["0"])[0])
                end = float((qs.get("end") or ["1"])[0])
                width = int(float((qs.get("width") or ["1000"])[0]))
                self._send_json(app.waveform_tile(item_id, start, end, width))
            elif path == "/api/waveform_stems_tile":
                item_id = (qs.get("id") or [""])[0]
                start = float((qs.get("start") or ["0"])[0])
                end = float((qs.get("end") or ["1"])[0])
                width = int(float((qs.get("width") or ["1000"])[0]))
                self._send_json(app.waveform_stems_tile(item_id, start, end, width))
            elif path == "/media/audio":
                item_id = (qs.get("id") or [""])[0]
                view = (qs.get("view") or ["window"])[0]
                role = (qs.get("role") or ["drums"])[0]
                start = _float_or_none((qs.get("start") or [""])[0])
                end = _float_or_none((qs.get("end") or [""])[0])
                media_path, mime = app.media_audio_path(item_id, view=view, start=start, end=end, role=role)
                if media_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(media_path, mime)
            elif path == "/media/debug":
                item_id = (qs.get("id") or [""])[0]
                media_path = app.media_debug_path(item_id)
                if media_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(media_path, "image/png")
            elif path == "/media/waveform_png":
                item_id = (qs.get("id") or [""])[0]
                user_pick = _float_or_none((qs.get("user_pick") or [""])[0])
                refined_pick = _float_or_none((qs.get("refined_pick") or [""])[0])
                attack_time = _float_or_none((qs.get("attack_time") or [""])[0])
                zero_time = _float_or_none((qs.get("zero_time") or [""])[0])
                knee_time = _float_or_none((qs.get("knee_time") or [""])[0])
                asd_time = _float_or_none((qs.get("asd_time") or [""])[0])
                data = app.waveform_png(
                    item_id,
                    start=float((qs.get("start") or ["0"])[0]),
                    end=float((qs.get("end") or ["1"])[0]),
                    width=int(float((qs.get("width") or ["2400"])[0])),
                    height=int(float((qs.get("height") or ["720"])[0])),
                    user_pick=user_pick,
                    refined_pick=refined_pick,
                    attack_time=attack_time,
                    zero_time=zero_time,
                    knee_time=knee_time,
                    asd_time=asd_time,
                )
                if data is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_bytes(data, "image/png")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_body()
                path = urlparse(self.path).path
                if path == "/api/approve":
                    self._send_json(app.approve(str(payload.get("id", ""))))
                elif path == "/api/correct":
                    self._send_json(
                        app.correct(
                            str(payload.get("id", "")),
                            float(payload.get("user_pick")),
                            str(payload.get("reviewed_from") or "web_review"),
                            picked_candidate=payload.get("picked_candidate"),
                            top_candidates=payload.get("top_10_candidates"),
                        )
                    )
                elif path == "/api/refine_marker":
                    marker_time = payload.get("marker_time")
                    self._send_json(
                        app.refine_marker(
                            str(payload.get("id", "")),
                            None if marker_time in (None, "") else float(marker_time),
                        )
                    )
                elif path == "/api/auto_place":
                    self._send_json(app.auto_place(str(payload.get("id", "")), str(payload.get("mode", "normal"))))
                elif path == "/api/label_section":
                    self._send_json(
                        app.label_section(
                            str(payload.get("id", "")),
                            str(payload.get("label", "")),
                            _float_or_none(payload.get("marker_time")),
                            start_sec=_float_or_none(payload.get("start_sec")),
                            end_sec=_float_or_none(payload.get("end_sec")),
                            candidate=payload.get("candidate"),
                            top_candidates=payload.get("top_10_candidates"),
                            context=payload.get("context"),
                        )
                    )
                elif path == "/api/skip":
                    self._send_json(app.skip(str(payload.get("id", ""))))
                elif path == "/api/retrain":
                    self._send_json(app.run_retrain())
                elif path == "/api/regenerate_als":
                    self._send_json(app.regenerate_als(str(payload.get("id", "")), float(payload.get("marker_time"))))
                elif path == "/api/navigate":
                    self._send_json(app.navigate(str(payload.get("direction", "next"))))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc) or exc.__class__.__name__}, status=500)

    return Handler


def _run_default_batch(args: argparse.Namespace, summary_csv: Path, template: Path) -> None:
    library = Path(args.library).expanduser().resolve()
    if not library.exists() or not library.is_dir():
        raise FileNotFoundError(f"Default library folder not found: {library}")

    batch_script = Path(__file__).resolve().parent / "batch.py"
    cmd = [
        sys.executable,
        str(batch_script),
        str(library),
        "--template",
        str(template),
        "--recursive",
        "--stem-role",
        "drums",
        "--strict-stem-set",
        "--exclude-dir-name",
        next(iter(EXCLUDED_DIR_NAMES)),
        "--workers",
        str(max(1, int(args.batch_workers))),
        "--no-hpss",
        "--summary",
        str(summary_csv),
    ]
    if args.debug_candidates:
        cmd.append("--debug-candidates")
    if args.use_drumprint is True:
        cmd.append("--use-drumprint")
    elif args.use_drumprint is False:
        cmd.append("--no-drumprint")
    if args.use_microalign:
        cmd.append("--microalign")
    if args.force_batch:
        cmd.append("--force")

    print(f"Batch summary not found, creating it now: {summary_csv}")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local browser review UI for drop-aligned ALS results.")
    parser.add_argument(
        "summary_csv",
        nargs="?",
        help=f"drop_batch_summary.csv from batch.py. Defaults to {_default_summary_path()}",
    )
    parser.add_argument(
        "--template",
        default=str(_default_template_path()),
        help=f"Ableton .als template for corrected ALS regeneration. Defaults to {_default_template_path()}",
    )
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY_DIR), help="Library folder used for auto-batch mode")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address. Default exposes the review app to your phone on the same Wi-Fi.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--correction-log", default="drop_corrections.jsonl")
    parser.add_argument("--auto-retrain-every", type=int, default=25)
    parser.add_argument("--auto-batch", action=argparse.BooleanOptionalAction, default=True, help="Run default batch if the summary CSV is missing")
    parser.add_argument("--refresh-batch", action="store_true", help="Recreate the summary CSV before starting review")
    parser.add_argument("--force-batch", action="store_true", help="Pass --force to batch.py when auto/refresh batching")
    parser.add_argument("--batch-workers", type=int, default=8, help="Worker count used by auto/refresh batching")
    parser.add_argument("--debug-candidates", action="store_true", help="Create debug PNGs when auto/refresh batching")
    parser.set_defaults(use_drumprint=None)
    parser.add_argument("--use-drumprint", dest="use_drumprint", action="store_true", help="Force DrumPrint pattern scoring when auto/refresh batching")
    parser.add_argument("--no-drumprint", dest="use_drumprint", action="store_false", help="Disable DrumPrint pattern scoring when auto/refresh batching")
    parser.set_defaults(use_microalign=False)
    parser.add_argument("--microalign", dest="use_microalign", action="store_true", help="Enable sample-level MicroSnap marker refinement when auto/refresh batching")
    parser.add_argument("--no-microalign", dest="use_microalign", action="store_false", help="Disable sample-level MicroSnap marker refinement when auto/refresh batching")
    parser.add_argument("--review-low-only", action="store_true")
    parser.add_argument("--review-medium-and-low", action="store_true")
    parser.add_argument("--queue", default="", help="Optional active_learning_queue.csv; review only these tracks in queue order")
    parser.add_argument("--regenerate-als-on-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-first", action="store_true", help="Open each track on the full waveform so manual visual placement is the primary flow")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_csv = Path(args.summary_csv).expanduser() if args.summary_csv else _default_summary_path()
    template = Path(args.template).expanduser()

    if args.refresh_batch or not summary_csv.exists():
        if not args.auto_batch and not args.refresh_batch:
            print(f"FAIL: Batch summary not found: {summary_csv}", file=sys.stderr)
            print("Run with --auto-batch, or run batch.py first.", file=sys.stderr)
            return 1
        try:
            _run_default_batch(args, summary_csv, template)
        except subprocess.CalledProcessError as exc:
            print(f"FAIL: batch.py exited with status {exc.returncode}", file=sys.stderr)
            return int(exc.returncode or 1)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1

    try:
        app = ReviewApp(
            summary_csv=str(summary_csv),
            template=str(template),
            correction_log=args.correction_log,
            auto_retrain_every=int(args.auto_retrain_every),
            review_low_only=bool(args.review_low_only),
            review_medium_and_low=bool(args.review_medium_and_low),
            regenerate_als_on_correction=bool(args.regenerate_als_on_correction),
            visual_first=bool(args.visual_first),
            review_queue=str(args.queue or ""),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        if "Batch summary not found" in str(exc):
            print(
                "\nRun batch.py first, then pass the generated summary CSV to web_review.py.\n"
                "For Ali's default STEMS library, you can now just run:\n"
                "  python3 web_review.py\n\n"
                "Example:\n"
                '  python3 batch.py "~/Desktop/MUSIC/STEMS/128" --template "alsFiles/128.als" '
                "--recursive --debug-candidates --workers 4\n"
                '  python3 web_review.py "~/Desktop/MUSIC/STEMS/128/drop_batch_summary.csv" '
                '--template "alsFiles/128.als" --open-browser --regenerate-als-on-correction',
                file=sys.stderr,
            )
        elif "Template cannot regenerate" in str(exc):
            print(
                "\nUse an ALS template that already contains AudioClip and WarpMarker nodes. "
                "In this repo, alsFiles/128.als and stuff/CH1.als look usable.",
                file=sys.stderr,
            )
        return 1
    urls = _startup_urls(str(args.host), int(args.port))
    open_url = urls.get("local") or f"http://127.0.0.1:{int(args.port)}"
    try:
        server = ThreadingHTTPServer((args.host, int(args.port)), _make_handler(app))
    except OSError as exc:
        if getattr(exc, "errno", None) in {48, 98} or "Address already in use" in str(exc):
            print(f"Review server already appears to be running.")
            for label, url in urls.items():
                print(f"{label.title()}: {url}")
            if args.open_browser:
                webbrowser.open(open_url)
            return 0
        raise
    print(f"Loaded {len(app.items)} review items")
    print(f"State: {app.state_path}")
    for label, url in urls.items():
        print(f"{label.title()}: {url}")
    if args.open_browser:
        webbrowser.open(open_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web review server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
