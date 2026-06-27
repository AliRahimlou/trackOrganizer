#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import json
import math
import os
import signal
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import buildSetAndGenerateAls as builder
from audit_visual_first_human_review_memory import (
    MANUAL_REVIEW_SOURCES,
    build_review_indexes,
    find_review_for_row,
    load_review_markers,
    review_marker_gate,
)
from audit_visual_first_library_coverage import compare_rows_to_eligible
from drop_aligner.als import modify_als
from drop_aligner.boom_profile import boom_proof_front_edge_freshness
from drop_aligner.exclusions import drums_stem_signal_stats, is_near_empty_drums_stem, row_is_acapella
from drop_aligner.visual_first import (
    VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
    _accept_gui_contract_by_actual_body_proof,
    _accept_gui_contract_by_opening_body_start_proof,
    _accept_gui_contract_by_sparse_groove_front_edge_proof,
    _accept_gui_contract_by_sustained_body_section,
    _accept_gui_contract_by_staccato_front_body_proof,
    _gui_proof_has_exact_sparse_front_edge,
    _gui_mask_proof as visual_gui_mask_proof,
    _gui_mask_proof_needs_front_edge_repair,
    visual_first_marker,
)
from drop_aligner.waveform import accept_gui_boom_mask_with_front_edge_proof, gui_boom_mask_strict_contract_issue
from project_config import BASE_ALS_TEMPLATE, DEFAULT_ALS_TEMPLATE, GENERATED_SET_DIR, STEMS_ROOT_DIR
from verify_als import verify_als


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}
ROLE_ORDER = {"drums": 1, "inst": 2, "vocals": 3}
DISALLOWED_PASS_SOURCES = {
    "historical_human_marker",
    "historical_review_memory",
    "manual_review_marker",
    "review_auto_place",
    "saved_closest_to_review_pick",
    "visual_drop_v2",
    "visual_drop_v2_candidate",
    "visual_first_rms_body_fallback",
    "web_accept_blue_marker",
    "web_save_placed_marker",
}
DISALLOWED_PASS_SOURCE_PREFIXES = ("historical_", "saved_")
DEFAULT_WAVEFORM_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "waveform_cache"
VALIDATED_HUMAN_OVERRIDE_SELECTED_BY = "visual_validated_human_review_override"
DEFAULT_PER_TRACK_TIMEOUT_SEC = 600.0


class _TrackTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def _time_limit(timeout_sec: float, label: str):
    timeout = float(timeout_sec or 0.0)
    if timeout <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise _TrackTimeoutError(f"{label}_timeout:{timeout:.3f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _run_visual_first_marker_with_timeout(
    audio_path: Path,
    *,
    sample_rate: int,
    use_cache: bool,
    timeout_sec: float,
) -> Dict[str, Any]:
    with _time_limit(float(timeout_sec or 0.0), "visual_first_marker"):
        return visual_first_marker(str(audio_path), sample_rate=sample_rate, use_cache=use_cache)


def _timeout_row_for_task(task: Mapping[str, Any], timeout_sec: float, *, reason: str) -> Dict[str, Any]:
    track = dict(task.get("track") if isinstance(task.get("track"), Mapping) else {})
    drums_path = ""
    try:
        src = Path(str(track.get("src") or ""))
        found = _find_role_audio(src.parent, "drums", track) if src else None
        drums_path = str(found) if found is not None else ""
    except Exception:
        drums_path = ""
    return {
        "status": "error",
        "track": track,
        "drums_path": drums_path,
        "error": f"{reason}:{float(timeout_sec):.3f}s",
    }


def _terminate_process_pool(pool: ProcessPoolExecutor) -> None:
    processes = getattr(pool, "_processes", None)
    if isinstance(processes, Mapping):
        for process in list(processes.values()):
            try:
                process.terminate()
            except Exception:
                pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _scene_name(track: Mapping[str, Any]) -> str:
    energy = track.get("energy")
    folder_name = builder.swap_artist_title(str(track["folder"])) if builder.RENAME_TRACKS else str(track["folder"])
    if energy is not None:
        return f"{track['bpm']}_{track['key']}_{energy}-{folder_name}"
    return f"{track['bpm']}_{track['key']}-{folder_name}"


def _find_role_audio(folder: Path, role: str, track: Mapping[str, Any]) -> Optional[Path]:
    bpm = int(track["bpm"])
    key = str(track["key"]).upper()
    energy = track.get("energy")
    matches: List[Path] = []
    try:
        children = list(folder.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_file() or child.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        match = builder.STEM_RE.match(child.name)
        if not match:
            continue
        child_role, child_bpm, child_key, child_energy = match.groups()
        if child_role.lower() != role:
            continue
        if int(child_bpm) != bpm or child_key.upper() != key:
            continue
        if energy is not None and int(child_energy) != int(energy):
            continue
        matches.append(child)
    if not matches and role == "drums":
        matches = [
            child
            for child in children
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS and child.name.lower().startswith("drums_")
        ]
    return sorted(matches, key=lambda path: path.name.lower())[0] if matches else None


def _fresh_paths(drums_path: Path, run_stamp: str) -> Tuple[Path, Path]:
    stem = f"{drums_path.stem}_VISUAL_FIRST_FRESH_{run_stamp}"
    return (
        drums_path.with_name(f"{stem}_DROP_ALIGNED.als"),
        drums_path.with_name(f"{stem}_drop_candidates.json"),
    )


def _candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "microaligned_time", "snapped_sec", "time_sec"):
        try:
            value = float(candidate.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return None


def _selected_by(result: Mapping[str, Any]) -> str:
    selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
    return str(selected.get("selected_by") or result.get("source") or "visual_first_marker")


def _audit_status_flags(payload: Mapping[str, Any]) -> Tuple[str, List[str]]:
    audit = payload.get("visual_audit") if isinstance(payload.get("visual_audit"), Mapping) else {}
    status = str(audit.get("status") or payload.get("audit_status") or "").strip().lower()
    raw_flags = audit.get("flag_codes") if isinstance(audit.get("flag_codes"), list) else payload.get("audit_flags")
    flags = [str(flag) for flag in raw_flags or [] if str(flag)]
    return status, flags


def _boom_proof(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = payload.get("boom_proof")
    return proof if isinstance(proof, Mapping) else {}


def _gui_mask_proof(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    proof = payload.get("gui_mask_proof")
    if isinstance(proof, Mapping):
        return proof
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    proof = selected.get("gui_mask_proof")
    return proof if isinstance(proof, Mapping) else {}


def _production_gate_reasons(
    *,
    audit_status: str,
    audit_flags: Sequence[str],
    selected_by: str,
    boom_proof: Mapping[str, Any],
    gui_mask_proof: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    if str(audit_status or "").strip().lower() != "pass":
        reasons.append(f"audit_status={audit_status or 'missing'}")
    if audit_flags:
        reasons.append("audit_flags=" + ";".join(str(flag) for flag in audit_flags))
    selected_source = str(selected_by or "").strip()
    if selected_source in DISALLOWED_PASS_SOURCES or any(
        selected_source.startswith(prefix) for prefix in DISALLOWED_PASS_SOURCE_PREFIXES
    ):
        reasons.append(f"unsafe_source={selected_source}")
    if not bool(boom_proof.get("passes")):
        proof_reasons = [str(reason) for reason in boom_proof.get("reasons") or [] if str(reason)]
        reasons.append("boom_proof=hold" + (":" + ";".join(proof_reasons) if proof_reasons else ":missing"))
    else:
        freshness = boom_proof_front_edge_freshness(boom_proof)
        if not bool(freshness.get("fresh")):
            reasons.append(f"boom_proof=stale_front_edge:{freshness.get('reason') or 'unknown'}")
    if not bool(gui_mask_proof.get("passes")):
        gui_reasons = [str(reason) for reason in gui_mask_proof.get("reasons") or [] if str(reason)]
        reasons.append("gui_mask=hold" + (":" + ";".join(gui_reasons) if gui_reasons else ":missing"))
    elif _gui_mask_proof_needs_front_edge_repair(
        gui_mask_proof,
        trust_reasonless_relief_offset=True,
    ):
        reasons.append("gui_mask=wide_exact_boom_front_edge_mismatch")
    elif "marker_relevant_mask" not in gui_mask_proof:
        reasons.append("gui_mask=stale_missing_marker_relevant_mask")
    elif gui_mask_proof.get("marker_relevant_mask") is not True:
        strict_issue = gui_boom_mask_strict_contract_issue(gui_mask_proof)
        if strict_issue:
            reasons.append(f"gui_mask=strict_contract_hold:{strict_issue}")
    elif "marker_signal_present" not in gui_mask_proof:
        reasons.append("gui_mask=stale_missing_marker_signal")
    elif gui_mask_proof.get("marker_signal_present") is not True:
        reasons.append("gui_mask=blank_marker:no_visible_signal")
    else:
        strict_issue = gui_boom_mask_strict_contract_issue(gui_mask_proof)
        if strict_issue and not (
            strict_issue == "marker_has_no_immediate_drop_body"
            and bool(gui_mask_proof.get("accepted_by_exact_sparse_front_edge_proof"))
        ):
            reasons.append(f"gui_mask=strict_contract_hold:{strict_issue}")
    return reasons


def _result_production_gate_reasons(result: Mapping[str, Any]) -> List[str]:
    audit_status, audit_flags = _audit_status_flags(result)
    selected_by = _selected_by(result)
    return _production_gate_reasons(
        audit_status=audit_status,
        audit_flags=audit_flags,
        selected_by=selected_by,
        boom_proof=_boom_proof(result),
        gui_mask_proof=_gui_mask_proof(result),
    )


def _write_visual_candidate_json(
    path: Path,
    *,
    audio_path: Path,
    marker: float,
    result: Mapping[str, Any],
    track: Mapping[str, Any],
    output_als: Path,
) -> None:
    selected = dict(result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {})
    selected["timestamp"] = float(marker)
    selected["snapped_sec"] = float(marker)
    selected["microaligned_time"] = float(marker)
    selected["selected"] = True
    selected.setdefault("selected_by", _selected_by(result))
    gui_mask_proof = result.get("gui_mask_proof") if isinstance(result.get("gui_mask_proof"), Mapping) else {}
    if gui_mask_proof and not isinstance(selected.get("gui_mask_proof"), Mapping):
        selected["gui_mask_proof"] = dict(gui_mask_proof)
    payload = {
        "source": "visual_first_fresh_library_set",
        "reviewed_from": "visual_detector_prep",
        "audio_path": str(audio_path),
        "track_folder": str(track.get("folder") or ""),
        "bpm": float(track.get("bpm") or 0.0),
        "key": str(track.get("key") or ""),
        "energy": track.get("energy"),
        "final_ai_pick": float(marker),
        "drop_sec": float(marker),
        "selected_by": str(selected.get("selected_by") or _selected_by(result)),
        "selected_candidate": selected,
        "top_10_candidates": [dict(row) for row in result.get("candidates") or [] if isinstance(row, Mapping)][:10],
        "visual_audit": result.get("visual_audit") if isinstance(result.get("visual_audit"), Mapping) else {},
        "boom_proof": result.get("boom_proof") if isinstance(result.get("boom_proof"), Mapping) else {},
        "gui_mask_proof": dict(gui_mask_proof),
        "boom_candidates": [dict(row) for row in result.get("boom_candidates") or [] if isinstance(row, Mapping)][:10],
        "feature_map": result.get("feature_map") if isinstance(result.get("feature_map"), Mapping) else {},
        "output_als": str(output_als),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    human_override = result.get("human_review_override") if isinstance(result.get("human_review_override"), Mapping) else {}
    if human_override:
        payload["human_review_override"] = dict(human_override)
        selected["human_review_override"] = dict(human_override)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


def _verify_anchor(path: Path, marker: float, candidates_json: Path) -> Dict[str, Any]:
    report = verify_als(str(path), candidates_json=str(candidates_json), tolerance_sec=0.002)
    if report.get("valid"):
        return report
    return report


def _load_existing_fresh_row(output_als: Path, candidates_json: Path, track: Mapping[str, Any], drums_path: Path) -> Optional[Dict[str, Any]]:
    if not output_als.exists() or not candidates_json.exists():
        return None
    try:
        payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    marker = None
    for key in ("final_ai_pick", "drop_sec"):
        try:
            marker = float(payload.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(marker):
            break
    if marker is None:
        return None
    audit_status, audit_flags = _audit_status_flags(payload)
    selected_by = str(payload.get("selected_by") or "visual_first_existing_fresh")
    boom_proof = _boom_proof(payload)
    gui_mask_proof = _gui_mask_proof(payload)
    if _production_gate_reasons(
        audit_status=audit_status,
        audit_flags=audit_flags,
        selected_by=selected_by,
        boom_proof=boom_proof,
        gui_mask_proof=gui_mask_proof,
    ):
        return None
    verification = _verify_anchor(output_als, float(marker), candidates_json)
    if not verification.get("valid"):
        return None
    return {
        "status": "processed",
        "track": dict(track),
        "drums_path": str(drums_path),
        "output_als": str(output_als),
        "candidates_json": str(candidates_json),
        "marker": float(marker),
        "raw_visual_time": payload.get("raw_visual_time"),
        "selected_by": selected_by,
        "audit_status": audit_status,
        "audit_flags": audit_flags,
        "boom_proof": dict(boom_proof),
        "gui_mask_proof": dict(gui_mask_proof),
        "verification": {
            "valid": True,
            "drop_marker_time": verification.get("drop_marker_time"),
            "drop_marker_count": verification.get("drop_marker_count"),
            "errors": verification.get("errors") or [],
        },
    }


def _process_track(task: Mapping[str, Any]) -> Dict[str, Any]:
    track = dict(task["track"])
    src_ch1 = Path(str(track["src"]))
    folder = src_ch1.parent
    template = Path(str(task["template"]))
    run_stamp = str(task["run_stamp"])
    sample_rate = int(task["sample_rate"])
    use_cache = bool(task["use_cache"])
    strict_stems = bool(task["strict_stems"])
    force = bool(task["force"])
    dry_run = bool(task["dry_run"])
    per_track_timeout_sec = float(task.get("per_track_timeout_sec") or 0.0)
    waveform_cache_dir = Path(str(task.get("waveform_cache_dir") or DEFAULT_WAVEFORM_CACHE_DIR)).expanduser()
    try:
        drums_path = _find_role_audio(folder, "drums", track)
        if drums_path is None:
            return {"status": "error", "track": track, "error": "missing_drums_stem", "folder": str(folder)}
        signal_stats = drums_stem_signal_stats(drums_path)
        if is_near_empty_drums_stem(drums_path, stats=signal_stats):
            return {
                "status": "excluded_near_empty_drums",
                "track": track,
                "drums_path": str(drums_path),
                "error": "near_empty_drums_stem",
                "signal_stats": signal_stats,
            }
        if strict_stems:
            missing = [
                role
                for role in ("drums", "inst", "vocals")
                if _find_role_audio(folder, role, track) is None
            ]
            if missing:
                return {
                    "status": "error",
                    "track": track,
                    "drums_path": str(drums_path),
                    "error": f"missing_required_stems:{','.join(missing)}",
                }
        output_als, candidates_json = _fresh_paths(drums_path, run_stamp)
        if output_als.exists() and not force:
            existing = _load_existing_fresh_row(output_als, candidates_json, track, drums_path)
            if existing is not None:
                return existing
        if dry_run:
            return {
                "status": "dry_run",
                "track": track,
                "drums_path": str(drums_path),
                "output_als": str(output_als),
                "candidates_json": str(candidates_json),
            }

        try:
            result = _run_visual_first_marker_with_timeout(
                drums_path,
                sample_rate=sample_rate,
                use_cache=use_cache,
                timeout_sec=per_track_timeout_sec,
            )
        except _TrackTimeoutError as exc:
            return {
                "status": "error",
                "track": track,
                "drums_path": str(drums_path),
                "error": str(exc),
            }
        if not result.get("ok"):
            return {
                "status": "error",
                "track": track,
                "drums_path": str(drums_path),
                "error": str(result.get("error") or "visual_first_marker_failed"),
                "result": result,
            }
        selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
        marker = result.get("marker")
        if marker is None:
            marker = _candidate_time(selected)
        if marker is None:
            return {
                "status": "error",
                "track": track,
                "drums_path": str(drums_path),
                "error": "visual_first_marker_missing_time",
                "result": result,
            }
        marker = float(marker)
        result = dict(result)
        selected_for_gui = dict(result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {})
        boom_proof = _boom_proof(result)
        detector_gui_proof = dict(_gui_mask_proof(result))
        try:
            gui_proof = visual_gui_mask_proof(
                str(drums_path),
                marker,
                cache_dir=waveform_cache_dir,
            )
        except Exception as exc:
            gui_proof = {
                "passes": False,
                "reasons": [f"gui_mask_error:{str(exc) or exc.__class__.__name__}"],
                "placeable_count": 0,
                "mask_index": None,
            }
        selected_source = str(selected_for_gui.get("selected_by") or _selected_by(result))
        repair_gui_lag_sec = 0.300 if selected_source.startswith("visual_final_contract_") else 0.040
        repair_gui_profile = 0.620 if selected_source.startswith("visual_final_contract_") else 0.560
        gui_proof = accept_gui_boom_mask_with_front_edge_proof(
            gui_proof,
            boom_proof,
            near_offset_sec=repair_gui_lag_sec,
            near_profile_score=repair_gui_profile,
        )
        gui_proof = _accept_gui_contract_by_staccato_front_body_proof(gui_proof, boom_proof, selected_for_gui)
        gui_proof = _accept_gui_contract_by_sparse_groove_front_edge_proof(gui_proof, boom_proof, selected_for_gui)
        gui_proof = _accept_gui_contract_by_sustained_body_section(gui_proof, boom_proof, selected_for_gui)
        gui_proof = _accept_gui_contract_by_actual_body_proof(gui_proof, boom_proof, selected_for_gui)
        gui_proof = _accept_gui_contract_by_opening_body_start_proof(gui_proof, boom_proof, selected_for_gui)
        if _gui_proof_has_exact_sparse_front_edge(gui_proof, boom_proof, selected_for_gui):
            raw_reasons = [
                str(reason)
                for reason in list(gui_proof.get("raw_reasons") or []) + list(gui_proof.get("reasons") or [])
                if str(reason)
            ]
            gui_proof = dict(gui_proof)
            gui_proof["accepted_by_exact_sparse_front_edge_proof"] = True
            gui_proof["passes"] = True
            gui_proof["reasons"] = []
            if raw_reasons:
                gui_proof["raw_reasons"] = sorted(set(raw_reasons))
        if (
            (not bool(gui_proof.get("passes")) or gui_boom_mask_strict_contract_issue(gui_proof))
            and bool(detector_gui_proof.get("passes"))
            and not gui_boom_mask_strict_contract_issue(detector_gui_proof)
            and detector_gui_proof.get("marker_signal_present") is True
        ):
            gui_proof = dict(detector_gui_proof)
            gui_proof["builder_preserved_detector_gui_mask_proof"] = True
        selected_for_gui["gui_mask_proof"] = dict(gui_proof)
        result["selected_candidate"] = selected_for_gui
        result["gui_mask_proof"] = dict(gui_proof)
        audit_status, audit_flags = _audit_status_flags(result)
        gui_mask_proof = _gui_mask_proof(result)
        gate_reasons = _result_production_gate_reasons(result)
        if gate_reasons:
            return {
                "status": "hold",
                "track": track,
                "drums_path": str(drums_path),
                "output_als": str(output_als),
                "candidates_json": str(candidates_json),
                "marker": marker,
                "raw_visual_time": result.get("raw_visual_time"),
                "selected_by": _selected_by(result),
                "audit_status": audit_status,
                "audit_flags": audit_flags,
                "boom_proof": dict(boom_proof),
                "gui_mask_proof": dict(gui_mask_proof),
                "error": "visual_production_gate_hold:" + "|".join(gate_reasons),
                "result": result,
            }
        _write_visual_candidate_json(
            candidates_json,
            audio_path=drums_path,
            marker=marker,
            result=result,
            track=track,
            output_als=output_als,
        )
        modify_als(
            template_path=str(template),
            audio_path=str(drums_path),
            drop_sec=marker,
            bpm=float(track["bpm"]),
            output_path=str(output_als),
            strict_stems=strict_stems,
        )
        verification = _verify_anchor(output_als, marker, candidates_json)
        if not verification.get("valid"):
            return {
                "status": "error",
                "track": track,
                "drums_path": str(drums_path),
                "output_als": str(output_als),
                "candidates_json": str(candidates_json),
                "marker": marker,
                "selected_by": _selected_by(result),
                "error": "als_verification_failed",
                "verification": verification,
            }
        return {
            "status": "processed",
            "track": track,
            "drums_path": str(drums_path),
            "output_als": str(output_als),
            "candidates_json": str(candidates_json),
            "marker": marker,
            "raw_visual_time": result.get("raw_visual_time"),
            "selected_by": _selected_by(result),
            "audit_status": audit_status,
            "audit_flags": audit_flags,
            "boom_proof": dict(boom_proof),
            "gui_mask_proof": dict(gui_mask_proof),
            "verification": {
                "valid": bool(verification.get("valid")),
                "drop_marker_time": verification.get("drop_marker_time"),
                "drop_marker_count": verification.get("drop_marker_count"),
                "errors": verification.get("errors") or [],
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "track": track,
            "folder": str(folder),
            "error": str(exc) or exc.__class__.__name__,
            "traceback": traceback.format_exc(),
        }


def _write_playlist(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["#", "BPM", "Key", "Energy", "TrackFolder", "MarkerSec", "SelectedBy", "FreshAlignedAls", "DrumsPath"])
        for idx, row in enumerate(rows, start=1):
            track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
            writer.writerow(
                [
                    idx,
                    track.get("bpm", ""),
                    track.get("key", ""),
                    track.get("energy", ""),
                    track.get("folder", ""),
                    row.get("marker", ""),
                    row.get("selected_by", ""),
                    row.get("output_als", ""),
                    row.get("drums_path", ""),
                ]
            )


def _write_failure_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["#", "BPM", "Key", "Energy", "TrackFolder", "Error", "DrumsPath", "SourceCH1"])
        for idx, row in enumerate(rows, start=1):
            track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
            writer.writerow(
                [
                    idx,
                    track.get("bpm", ""),
                    track.get("key", ""),
                    track.get("energy", ""),
                    track.get("folder", ""),
                    row.get("error", ""),
                    row.get("drums_path", ""),
                    track.get("src", ""),
                ]
            )


def _row_passes_visual_audit(row: Mapping[str, Any]) -> bool:
    status = str(row.get("audit_status") or "").strip().lower()
    flags = [str(flag) for flag in row.get("audit_flags") or [] if str(flag)]
    selected_by = str(row.get("selected_by") or "").strip()
    boom_proof = _boom_proof(row)
    gui_mask_proof = _gui_mask_proof(row)
    return not _production_gate_reasons(
        audit_status=status,
        audit_flags=flags,
        selected_by=selected_by,
        boom_proof=boom_proof,
        gui_mask_proof=gui_mask_proof,
    )


def _write_audit_hold_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "#",
                "BPM",
                "Key",
                "Energy",
                "TrackFolder",
                "MarkerSec",
                "AuditStatus",
                "AuditFlags",
                "BoomProof",
                "BoomProofReasons",
                "GuiMaskProof",
                "GuiMaskReasons",
                "SelectedBy",
                "DrumsPath",
                "CandidatesJson",
                "FreshAlignedAls",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
            boom_proof = row.get("boom_proof") if isinstance(row.get("boom_proof"), Mapping) else {}
            gui_mask_proof = row.get("gui_mask_proof") if isinstance(row.get("gui_mask_proof"), Mapping) else {}
            writer.writerow(
                [
                    idx,
                    track.get("bpm", ""),
                    track.get("key", ""),
                    track.get("energy", ""),
                    track.get("folder", ""),
                    row.get("marker", ""),
                    row.get("audit_status", ""),
                    ";".join(str(flag) for flag in row.get("audit_flags") or []),
                    "pass" if bool(boom_proof.get("passes")) else "hold",
                    ";".join(str(reason) for reason in boom_proof.get("reasons") or []),
                    "pass" if bool(gui_mask_proof.get("passes")) else "hold",
                    ";".join(str(reason) for reason in gui_mask_proof.get("reasons") or []),
                    row.get("selected_by", ""),
                    row.get("drums_path", ""),
                    row.get("candidates_json", ""),
                    row.get("output_als", ""),
                ]
            )


def _build_combined_set(base_als: Path, output: Path, processed_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    root = ET.fromstring(builder.read_als_text(base_als))
    tracks_parent = builder.tracks_parent(root)
    if tracks_parent is None:
        raise RuntimeError(f"Base ALS has no <Tracks> node: {base_als}")
    first_for: Dict[int, ET.Element] = {}
    for track_node in list(tracks_parent):
        idx = builder.ch_index_from_label(builder.track_label(track_node))
        if idx and idx not in first_for:
            first_for[idx] = track_node
    keep = set(first_for.values())
    for track_node in list(tracks_parent):
        if track_node not in keep:
            tracks_parent.remove(track_node)
    scenes = builder.scenes_node(root)
    for scene in list(scenes.findall("./Scene")):
        scenes.remove(scene)
    for track_node in list(tracks_parent):
        for csl_fn in (builder.main_csl, builder.freeze_csl):
            clip_slots = csl_fn(track_node)
            if clip_slots is None:
                continue
            for slot in list(clip_slots.findall("./ClipSlot")):
                clip_slots.remove(slot)
    ch_map = builder.ch_tracks_map(root)
    if not ch_map:
        raise RuntimeError(f"Base ALS has no CH1/CH2/CH3 tracks: {base_als}")
    all_tracks = [track for track in list(tracks_parent) if track.tag in ("AudioTrack", "MidiTrack", "GroupTrack")]
    row_keys: List[Tuple[int, int, int, int, str]] = []
    for idx, row in enumerate(processed_rows, start=1):
        track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
        source = Path(str(row.get("output_als") or ""))
        scene_name = _scene_name(track)
        insert_idx = len(row_keys)
        builder.insert_scene(root, all_tracks, ch_map, source, scene_name, insert_idx)
        row_keys.append(builder.scene_sort_key_from_track(dict(track)))
        if idx % 100 == 0 or idx == len(processed_rows):
            print(f"[SET] inserted {idx}/{len(processed_rows)} rows", flush=True)
    builder.write_als_gz(output, root)
    return _verify_combined_set(output, processed_rows)


def _current_anchor_sec(clip: ET.Element, *, allow_opening_anchor: bool = False) -> Optional[float]:
    vals: List[float] = []
    for marker in clip.findall("./WarpMarkers/WarpMarker"):
        try:
            sec = float(marker.get("SecTime"))
            beat = float(marker.get("BeatTime"))
        except Exception:
            continue
        if abs(beat) > 0.002:
            continue
        if sec > 1.0 or (allow_opening_anchor and sec >= -0.002):
            vals.append(sec)
    return min(vals) if vals else None


def _slot_audio_clip(slot: ET.Element) -> Optional[ET.Element]:
    for path in ("./ClipSlot/Value/AudioClip", "./Value/AudioClip", ".//AudioClip"):
        clip = slot.find(path)
        if clip is not None:
            return clip
    return None


def _xml_value(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return str(node.get("Value") or node.text or "").strip()


def _clip_file_ref_paths(clip: ET.Element, *, als_path: Path) -> List[str]:
    refs: List[str] = []
    seen: set[str] = set()
    for file_ref in clip.findall(".//FileRef"):
        for name in ("Path", "RelativePath"):
            raw = _xml_value(file_ref.find(name))
            if not raw:
                continue
            candidates = [Path(raw).expanduser()]
            if not candidates[0].is_absolute():
                candidates.append((als_path.parent / candidates[0]).expanduser())
            for candidate in candidates:
                try:
                    text = str(candidate.resolve())
                except Exception:
                    text = str(candidate)
                if text and text not in seen:
                    seen.add(text)
                    refs.append(text)
    return refs


def _expected_role_audio_paths(row: Mapping[str, Any]) -> Dict[int, str]:
    track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
    source_text = str(track.get("src") or "")
    drums_text = str(row.get("drums_path") or "")
    folder = Path(source_text).expanduser().parent if source_text else Path(drums_text).expanduser().parent
    expected: Dict[int, str] = {}
    for ch, role in ((1, "drums"), (2, "inst"), (3, "vocals")):
        audio: Optional[Path]
        if role == "drums" and drums_text:
            audio = Path(drums_text).expanduser()
        elif track and folder:
            audio = _find_role_audio(folder, role, track)
        else:
            audio = None
        if audio is None:
            continue
        try:
            expected[ch] = str(audio.resolve())
        except Exception:
            expected[ch] = str(audio)
    return expected


def _verify_combined_set(
    path: Path,
    processed_rows: Sequence[Mapping[str, Any]] = (),
    *,
    tolerance_sec: float = 0.002,
    sample_limit: int = 25,
) -> Dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    root = ET.fromstring(raw)
    scenes = builder.scenes_node(root).findall("./Scene")
    ch_map = builder.ch_tracks_map(root)
    slot_counts: Dict[str, int] = {}
    anchored_rows = 0
    complete_triplet_rows = 0
    ref_track = ch_map.get(1)
    if ref_track is None:
        ref_track = ch_map.get(2)
    if ref_track is None:
        ref_track = ch_map.get(3)
    ref_slots = builder.track_slot_list(ref_track) if ref_track is not None else []
    row_count = max([len(ref_slots or []), len(scenes)] + [len(builder.track_slot_list(track) or []) for track in ch_map.values()])
    expected_row_count = len(processed_rows)
    anchor_match_rows = 0
    anchor_mismatch_count = 0
    row_errors: List[Dict[str, Any]] = []
    file_ref_checked_rows = 0
    file_ref_match_rows = 0
    file_ref_mismatch_count = 0
    file_ref_errors: List[Dict[str, Any]] = []
    for ch, track in sorted(ch_map.items()):
        slot_counts[f"CH{ch}"] = len(builder.track_slot_list(track) or [])

    def add_row_error(row_idx: int, reason: str, **extra: Any) -> None:
        nonlocal anchor_mismatch_count
        anchor_mismatch_count += 1
        if len(row_errors) >= max(0, int(sample_limit)):
            return
        track = processed_rows[row_idx].get("track") if row_idx < len(processed_rows) and isinstance(processed_rows[row_idx], Mapping) else {}
        row_errors.append(
            {
                "row": int(row_idx),
                "reason": str(reason),
                "expected_marker": processed_rows[row_idx].get("marker") if row_idx < len(processed_rows) and isinstance(processed_rows[row_idx], Mapping) else None,
                "folder": track.get("folder") if isinstance(track, Mapping) else None,
                "bpm": track.get("bpm") if isinstance(track, Mapping) else None,
                "key": track.get("key") if isinstance(track, Mapping) else None,
                **extra,
            }
        )

    def add_file_ref_error(row_idx: int, reason: str, **extra: Any) -> None:
        nonlocal file_ref_mismatch_count
        file_ref_mismatch_count += 1
        if len(file_ref_errors) >= max(0, int(sample_limit)):
            return
        track = processed_rows[row_idx].get("track") if row_idx < len(processed_rows) and isinstance(processed_rows[row_idx], Mapping) else {}
        file_ref_errors.append(
            {
                "row": int(row_idx),
                "reason": str(reason),
                "expected_marker": processed_rows[row_idx].get("marker") if row_idx < len(processed_rows) and isinstance(processed_rows[row_idx], Mapping) else None,
                "folder": track.get("folder") if isinstance(track, Mapping) else None,
                "bpm": track.get("bpm") if isinstance(track, Mapping) else None,
                "key": track.get("key") if isinstance(track, Mapping) else None,
                **extra,
            }
        )

    for row_idx in range(row_count):
        expected_marker = None
        if row_idx < len(processed_rows):
            try:
                expected_marker = float(processed_rows[row_idx].get("marker"))
            except (TypeError, ValueError, AttributeError):
                expected_marker = None
        allow_opening_anchor = expected_marker is not None and math.isfinite(expected_marker) and expected_marker <= 1.0
        row_clips: List[ET.Element] = []
        row_clips_by_ch: Dict[int, ET.Element] = {}
        for ch in (1, 2, 3):
            track = ch_map.get(ch)
            slots = builder.track_slot_list(track) if track is not None else []
            if row_idx >= len(slots or []):
                continue
            clip = _slot_audio_clip(slots[row_idx])
            if clip is not None:
                row_clips.append(clip)
                row_clips_by_ch[ch] = clip
        if len(row_clips) >= 3:
            complete_triplet_rows += 1
        if not row_clips:
            add_row_error(row_idx, "missing_all_channel_clips")
            continue
        if len(row_clips) < 3:
            add_row_error(row_idx, "incomplete_triplet", clip_count=len(row_clips))
            continue
        anchors = [_current_anchor_sec(clip, allow_opening_anchor=allow_opening_anchor) for clip in row_clips]
        if all(anchor is not None for anchor in anchors):
            anchored_rows += 1
        else:
            add_row_error(row_idx, "missing_anchor", anchors=anchors)
            continue
        if expected_marker is None or not math.isfinite(float(expected_marker)):
            add_row_error(row_idx, "missing_expected_marker", anchors=anchors)
            continue
        deltas = [abs(float(anchor) - float(expected_marker)) for anchor in anchors if anchor is not None]
        if deltas and max(deltas) <= float(tolerance_sec):
            anchor_match_rows += 1
        else:
            add_row_error(row_idx, "anchor_marker_mismatch", anchors=anchors, max_delta_sec=max(deltas) if deltas else None)

        expected_paths = _expected_role_audio_paths(processed_rows[row_idx]) if row_idx < len(processed_rows) else {}
        file_ref_checked_rows += 1
        missing_expected = [ch for ch in (1, 2, 3) if ch not in expected_paths]
        if missing_expected:
            add_file_ref_error(row_idx, "missing_expected_role_audio", missing_channels=missing_expected)
            continue
        mismatches: List[Dict[str, Any]] = []
        for ch, clip in sorted(row_clips_by_ch.items()):
            expected_path = expected_paths.get(ch)
            refs = _clip_file_ref_paths(clip, als_path=path)
            if not expected_path or expected_path not in refs:
                mismatches.append(
                    {
                        "ch": int(ch),
                        "expected": expected_path,
                        "refs": refs[:4],
                    }
                )
        if mismatches:
            add_file_ref_error(row_idx, "file_ref_mismatch", mismatches=mismatches)
        else:
            file_ref_match_rows += 1

    row_count_matches_report = row_count == expected_row_count == len(scenes)
    all_rows_anchored = anchored_rows == complete_triplet_rows == row_count
    all_file_refs_match_expected = (
        row_count_matches_report
        and file_ref_checked_rows == expected_row_count
        and file_ref_match_rows == expected_row_count
        and file_ref_mismatch_count == 0
    )
    all_rows_match_expected = (
        row_count_matches_report
        and all_rows_anchored
        and anchor_match_rows == expected_row_count
        and anchor_mismatch_count == 0
        and all_file_refs_match_expected
    )
    return {
        "valid_xml": True,
        "audio_tracks": len(root.findall(".//AudioTrack")),
        "audio_clips": len(root.findall(".//AudioClip")),
        "scenes": len(scenes),
        "slot_counts": slot_counts,
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "row_count_matches_report": row_count_matches_report,
        "complete_triplet_rows": complete_triplet_rows,
        "anchored_rows": anchored_rows,
        "anchor_match_rows": anchor_match_rows,
        "anchor_mismatch_count": anchor_mismatch_count,
        "anchor_mismatch_samples": row_errors,
        "all_rows_anchored": all_rows_anchored,
        "file_ref_checked_rows": file_ref_checked_rows,
        "file_ref_match_rows": file_ref_match_rows,
        "file_ref_mismatch_count": file_ref_mismatch_count,
        "file_ref_mismatch_samples": file_ref_errors,
        "all_file_refs_match_expected": all_file_refs_match_expected,
        "all_rows_match_expected": all_rows_match_expected,
    }


def _load_existing_report(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("processed_rows") if isinstance(data, Mapping) else None
    return [dict(row) for row in rows or [] if isinstance(row, Mapping)]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _default_human_correction_logs(out_dir: Path) -> List[Path]:
    repo = Path(__file__).resolve().parent
    candidates: List[Path] = [
        repo / "drop_corrections.jsonl",
        out_dir / "drop_corrections.jsonl",
    ]
    candidates.extend(sorted(repo.glob("drop_corrections.before*.jsonl")))
    backups = repo / "backups"
    if backups.exists():
        candidates.extend(sorted(backups.glob("*/drop_corrections.jsonl")))
    seen: set[str] = set()
    logs: List[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.expanduser().resolve())
        except Exception:
            key = str(candidate.expanduser())
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        logs.append(candidate.expanduser())
    return logs


def _human_correction_logs_from_args(args: argparse.Namespace, out_dir: Path) -> List[Path]:
    if bool(getattr(args, "no_human_review_overrides", False)):
        return []
    raw_logs = getattr(args, "human_correction_log", None)
    if raw_logs:
        return [Path(path).expanduser() for path in raw_logs]
    return _default_human_correction_logs(out_dir)


def _load_candidate_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _apply_human_override_to_row(
    row: Dict[str, Any],
    *,
    review: Any,
    gate: Mapping[str, Any],
    template: Path,
    strict_stems: bool,
) -> Dict[str, Any]:
    marker = float(review.user_pick)
    drums_path_raw = str(row.get("drums_path") or "").strip()
    output_als_raw = str(row.get("output_als") or "").strip()
    candidates_json_raw = str(row.get("candidates_json") or "").strip()
    drums_path = Path(drums_path_raw).expanduser() if drums_path_raw else None
    output_als = Path(output_als_raw).expanduser() if output_als_raw else None
    candidates_json = Path(candidates_json_raw).expanduser() if candidates_json_raw else None
    track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
    if drums_path is None:
        return {"applied": False, "error": "missing_drums_path"}
    if output_als is None:
        return {"applied": False, "error": "missing_output_als"}
    if candidates_json is None:
        return {"applied": False, "error": "missing_candidates_json"}
    if not track:
        return {"applied": False, "error": "missing_track"}
    payload = _load_candidate_payload(candidates_json)
    selected = dict(payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {})
    selected.update(
        {
            "timestamp": marker,
            "snapped_sec": marker,
            "microaligned_time": marker,
            "selected": True,
            "selected_by": VALIDATED_HUMAN_OVERRIDE_SELECTED_BY,
            "reason": "validated manual review marker passed current boom/front-edge GUI contract",
        }
    )
    override_info = {
        "selected_by": VALIDATED_HUMAN_OVERRIDE_SELECTED_BY,
        "review_marker": marker,
        "previous_marker": row.get("marker"),
        "previous_selected_by": row.get("selected_by"),
        "reviewed_from": review.reviewed_from,
        "review_timestamp": review.timestamp,
        "review_selected_by": review.selected_by,
        "review_source_path": review.source_path,
        "review_line_number": int(review.line_number),
    }
    result = {
        "ok": True,
        "marker": marker,
        "raw_visual_time": payload.get("raw_visual_time", row.get("raw_visual_time")),
        "source": VALIDATED_HUMAN_OVERRIDE_SELECTED_BY,
        "selected_candidate": selected,
        "candidates": [dict(item) for item in payload.get("top_10_candidates") or [] if isinstance(item, Mapping)],
        "visual_audit": {
            "status": "pass",
            "flag_codes": [],
            "human_review_override": True,
        },
        "boom_proof": dict(gate.get("boom_proof") if isinstance(gate.get("boom_proof"), Mapping) else {}),
        "gui_mask_proof": dict(gate.get("gui_proof") if isinstance(gate.get("gui_proof"), Mapping) else {}),
        "boom_candidates": [dict(item) for item in payload.get("boom_candidates") or [] if isinstance(item, Mapping)],
        "feature_map": payload.get("feature_map") if isinstance(payload.get("feature_map"), Mapping) else {},
        "human_review_override": override_info,
    }
    _write_visual_candidate_json(
        candidates_json,
        audio_path=drums_path,
        marker=marker,
        result=result,
        track=track,
        output_als=output_als,
    )
    modify_als(
        template_path=str(template),
        audio_path=str(drums_path),
        drop_sec=marker,
        bpm=float(track.get("bpm") or 0.0),
        output_path=str(output_als),
        strict_stems=bool(strict_stems),
    )
    verification = _verify_anchor(output_als, marker, candidates_json)
    if not verification.get("valid"):
        return {
            "applied": False,
            "error": "human_override_als_verification_failed",
            "verification": verification,
        }
    row.update(
        {
            "marker": marker,
            "raw_visual_time": result.get("raw_visual_time"),
            "selected_by": VALIDATED_HUMAN_OVERRIDE_SELECTED_BY,
            "audit_status": "pass",
            "audit_flags": [],
            "boom_proof": dict(result["boom_proof"]),
            "gui_mask_proof": dict(result["gui_mask_proof"]),
            "human_review_override": override_info,
            "verification": {
                "valid": bool(verification.get("valid")),
                "drop_marker_time": verification.get("drop_marker_time"),
                "drop_marker_count": verification.get("drop_marker_count"),
                "errors": verification.get("errors") or [],
            },
        }
    )
    return {"applied": True, "marker": marker, "verification": verification}


def _apply_validated_human_overrides(
    processed_rows: List[Dict[str, Any]],
    *,
    correction_logs: Sequence[Path],
    template: Path,
    strict_stems: bool,
    waveform_cache_dir: Path,
    tolerance_sec: float = 0.050,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "enabled": bool(correction_logs),
        "correction_logs": [str(path) for path in correction_logs],
        "review_marker_count": 0,
        "matched_rows": 0,
        "manual_match_count": 0,
        "already_matching_count": 0,
        "stale_manual_mismatch_count": 0,
        "applied_count": 0,
        "failure_count": 0,
        "applied_rows": [],
        "stale_rows": [],
        "failure_rows": [],
    }
    if not correction_logs:
        return summary
    markers = load_review_markers(correction_logs)
    summary["review_marker_count"] = len(markers)
    exact, stem, slug = build_review_indexes(markers)
    for index, row in enumerate(processed_rows, 1):
        review = find_review_for_row(row, exact, stem, slug)
        if review is None:
            continue
        summary["matched_rows"] += 1
        if str(review.reviewed_from or "").strip().lower() not in MANUAL_REVIEW_SOURCES:
            continue
        summary["manual_match_count"] += 1
        current_marker = _float_or_none(row.get("marker"))
        if current_marker is None:
            continue
        delta = abs(float(current_marker) - float(review.user_pick))
        if delta <= float(tolerance_sec):
            summary["already_matching_count"] += 1
            continue
        gate = review_marker_gate(row, float(review.user_pick), cache_dir=waveform_cache_dir)
        base_detail = {
            "index": int(index),
            "track": str((row.get("track") or {}).get("folder") if isinstance(row.get("track"), Mapping) else row.get("track") or ""),
            "drums_path": str(row.get("drums_path") or ""),
            "previous_marker": float(current_marker),
            "review_marker": float(review.user_pick),
            "abs_delta_sec": float(delta),
            "reviewed_from": review.reviewed_from,
            "review_timestamp": review.timestamp,
            "review_source_path": review.source_path,
            "review_line_number": int(review.line_number),
        }
        if not bool(gate.get("passes")):
            detail = dict(base_detail)
            detail["gate_reasons"] = ";".join(str(reason) for reason in gate.get("reasons") or [] if str(reason))
            summary["stale_manual_mismatch_count"] += 1
            summary["stale_rows"].append(detail)
            continue
        outcome = _apply_human_override_to_row(
            row,
            review=review,
            gate=gate,
            template=template,
            strict_stems=bool(strict_stems),
        )
        detail = dict(base_detail)
        if bool(outcome.get("applied")):
            detail["selected_by"] = VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
            summary["applied_count"] += 1
            summary["applied_rows"].append(detail)
        else:
            detail["error"] = str(outcome.get("error") or "human_override_failed")
            detail["verification"] = outcome.get("verification")
            summary["failure_count"] += 1
            summary["failure_rows"].append(detail)
    return summary


def _write_human_override_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for status, key in (
        ("applied", "applied_rows"),
        ("stale", "stale_rows"),
        ("failure", "failure_rows"),
    ):
        for row in summary.get(key) or []:
            if isinstance(row, Mapping):
                rows.append({"status": status, **dict(row)})
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "index",
        "track",
        "drums_path",
        "previous_marker",
        "review_marker",
        "abs_delta_sec",
        "selected_by",
        "reviewed_from",
        "review_timestamp",
        "review_source_path",
        "review_line_number",
        "gate_reasons",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run(args: argparse.Namespace) -> int:
    run_stamp = args.run_stamp or _stamp()
    stems = Path(args.stems).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    base_als = Path(args.base_als).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.out).expanduser().resolve() if args.out else out_dir / f"VISUAL_FIRST_FRESH_ALL_TRACKS_{run_stamp}.als"
    report_path = Path(args.report).expanduser().resolve() if args.report else output.with_name(f"{output.stem}_report.json")
    waveform_cache_dir = Path(getattr(args, "waveform_cache_dir", "") or DEFAULT_WAVEFORM_CACHE_DIR).expanduser().resolve()
    playlist_path = output.with_suffix(".csv")
    failure_csv_path = output.with_name(f"{output.stem}_failures.csv")
    audit_hold_csv_path = output.with_name(f"{output.stem}_audit_holds.csv")
    human_override_csv_path = output.with_name(f"{output.stem}_human_review_overrides.csv")
    coverage_report_path = output.with_name(f"{output.stem}_coverage.json")
    human_correction_logs = _human_correction_logs_from_args(args, out_dir)

    if not stems.exists():
        raise FileNotFoundError(stems)
    if not template.exists():
        raise FileNotFoundError(template)
    if not base_als.exists():
        raise FileNotFoundError(base_als)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite output ALS without --force: {output}")

    all_tracks = builder.sort_by_bpm_key_energy(builder.collect_tracks(str(stems)))
    acapella_tracks = [track for track in all_tracks if row_is_acapella({"track": track})]
    non_acapella_tracks = [track for track in all_tracks if track not in acapella_tracks]
    missing_drums_tracks = [
        track
        for track in non_acapella_tracks
        if _find_role_audio(Path(str(track["src"])).parent, "drums", track) is None
    ]
    missing_drums_ids = {str(track.get("src") or track.get("folder") or "") for track in missing_drums_tracks}
    tracks = [
        track
        for track in non_acapella_tracks
        if str(track.get("src") or track.get("folder") or "") not in missing_drums_ids
    ]
    if int(args.limit or 0) > 0:
        tracks = tracks[: int(args.limit)]
    print(
        f"[POOL] {len(tracks)} eligible track folder(s) from {stems} "
        f"({len(acapella_tracks)} acapella/excluded, {len(missing_drums_tracks)} missing drums stem)",
        flush=True,
    )
    print(f"[RUN] stamp={run_stamp} workers={args.workers} strict_stems={args.strict_stems}", flush=True)

    processed_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    excluded_near_empty_rows: List[Dict[str, Any]] = []
    if args.reuse_report:
        loaded_rows = _load_existing_report(Path(args.reuse_report).expanduser().resolve())
        processed_rows = [row for row in loaded_rows if not row_is_acapella(row)]
        print(
            f"[REUSE] loaded {len(processed_rows)} eligible processed row(s) from {args.reuse_report} "
            f"({len(loaded_rows) - len(processed_rows)} acapella/excluded)",
            flush=True,
        )
    else:
        tasks = [
            {
                "track": track,
                "template": str(template),
                "run_stamp": run_stamp,
                "sample_rate": int(args.sample_rate),
                "use_cache": bool(args.use_cache),
                "strict_stems": bool(args.strict_stems),
                "force": bool(args.force),
                "dry_run": bool(args.dry_run),
                "per_track_timeout_sec": float(args.per_track_timeout_sec),
                "waveform_cache_dir": str(waveform_cache_dir),
            }
            for track in tracks
        ]
        total = len(tasks)
        if int(args.workers) <= 1:
            iterator = enumerate((_process_track(task) for task in tasks), start=1)
            for idx, row in iterator:
                if row.get("status") == "processed":
                    processed_rows.append(row)
                elif row.get("status") == "dry_run":
                    processed_rows.append(row)
                elif row.get("status") == "excluded_near_empty_drums":
                    excluded_near_empty_rows.append(row)
                else:
                    failure_rows.append(row)
                    track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
                    print(
                        "[FAIL] "
                        f"{track.get('bpm', '')}/{track.get('key', '')}/{track.get('folder', '')} "
                        f"status={row.get('status', '')} error={row.get('error', '')} "
                        f"selected_by={row.get('selected_by', '')} marker={row.get('marker', '')}",
                        flush=True,
                    )
                if idx % 25 == 0 or idx == total:
                    print(
                        f"[ALIGN] {idx}/{total} processed={len(processed_rows)} "
                        f"near_empty={len(excluded_near_empty_rows)} failures={len(failure_rows)}",
                        flush=True,
                    )
        else:
            timeout_sec = max(0.0, float(args.per_track_timeout_sec or 0.0))
            timeout_wait = timeout_sec if timeout_sec > 0.0 else None
            task_index = 0
            completed = 0
            stop_after_timeout = False
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                future_map: Dict[Any, Mapping[str, Any]] = {}

                def submit_next() -> None:
                    nonlocal task_index
                    if task_index >= total:
                        return
                    task = tasks[task_index]
                    task_index += 1
                    future_map[pool.submit(_process_track, task)] = task

                for _ in range(min(int(args.workers), total)):
                    submit_next()

                while future_map:
                    done, _pending = wait(
                        set(future_map.keys()),
                        timeout=timeout_wait,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        timed_out_tasks = list(future_map.values())
                        _terminate_process_pool(pool)
                        for task in timed_out_tasks:
                            row = _timeout_row_for_task(
                                task,
                                timeout_sec,
                                reason="visual_first_worker_timeout",
                            )
                            failure_rows.append(row)
                            completed += 1
                            track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
                            print(
                                "[FAIL] "
                                f"{track.get('bpm', '')}/{track.get('key', '')}/{track.get('folder', '')} "
                                f"status={row.get('status', '')} error={row.get('error', '')} "
                                f"selected_by={row.get('selected_by', '')} marker={row.get('marker', '')}",
                                flush=True,
                            )
                        if completed % 25 == 0 or completed == total:
                            print(
                                f"[ALIGN] {completed}/{total} processed={len(processed_rows)} "
                                f"near_empty={len(excluded_near_empty_rows)} failures={len(failure_rows)}",
                                flush=True,
                            )
                        stop_after_timeout = True
                        break
                    for future in done:
                        task = future_map.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = _timeout_row_for_task(
                                task,
                                timeout_sec,
                                reason=f"visual_first_worker_error:{str(exc) or exc.__class__.__name__}",
                            )
                        completed += 1
                        if row.get("status") == "processed":
                            processed_rows.append(row)
                        elif row.get("status") == "dry_run":
                            processed_rows.append(row)
                        elif row.get("status") == "excluded_near_empty_drums":
                            excluded_near_empty_rows.append(row)
                        else:
                            failure_rows.append(row)
                            track = row.get("track") if isinstance(row.get("track"), Mapping) else {}
                            print(
                                "[FAIL] "
                                f"{track.get('bpm', '')}/{track.get('key', '')}/{track.get('folder', '')} "
                                f"status={row.get('status', '')} error={row.get('error', '')} "
                                f"selected_by={row.get('selected_by', '')} marker={row.get('marker', '')}",
                                flush=True,
                            )
                        if completed % 25 == 0 or completed == total:
                            print(
                                f"[ALIGN] {completed}/{total} processed={len(processed_rows)} "
                                f"near_empty={len(excluded_near_empty_rows)} failures={len(failure_rows)}",
                                flush=True,
                            )
                        submit_next()
                    if stop_after_timeout:
                        break
        processed_rows.sort(key=lambda row: builder.scene_sort_key_from_track(dict(row["track"])))
        failure_rows.sort(key=lambda row: builder.scene_sort_key_from_track(dict(row["track"])) if isinstance(row.get("track"), Mapping) else (999, 999, 999, 999, ""))
        excluded_near_empty_rows.sort(
            key=lambda row: builder.scene_sort_key_from_track(dict(row["track"]))
            if isinstance(row.get("track"), Mapping)
            else (999, 999, 999, 999, "")
        )

    if args.dry_run:
        human_override_summary = {
            "enabled": bool(human_correction_logs),
            "skipped": True,
            "reason": "dry_run",
            "correction_logs": [str(path) for path in human_correction_logs],
            "review_marker_count": 0,
            "matched_rows": 0,
            "manual_match_count": 0,
            "already_matching_count": 0,
            "stale_manual_mismatch_count": 0,
            "applied_count": 0,
            "failure_count": 0,
            "applied_rows": [],
            "stale_rows": [],
            "failure_rows": [],
        }
    else:
        human_override_summary = _apply_validated_human_overrides(
            processed_rows,
            correction_logs=human_correction_logs,
            template=template,
            strict_stems=bool(args.strict_stems),
            waveform_cache_dir=waveform_cache_dir,
        )
    _write_human_override_csv(human_override_csv_path, human_override_summary)
    _write_playlist(playlist_path, processed_rows)
    _write_failure_csv(failure_csv_path, failure_rows)
    audit_hold_rows = [row for row in processed_rows if not _row_passes_visual_audit(row)]
    _write_audit_hold_csv(audit_hold_csv_path, audit_hold_rows)
    if int(args.limit or 0) > 0:
        coverage_report: Dict[str, Any] = {
            "coverage_checked": False,
            "all_covered": False,
            "reason": "coverage skipped for --limit",
        }
    else:
        try:
            coverage_report = compare_rows_to_eligible(
                processed_rows,
                stems=stems,
                source=str(report_path if args.reuse_report else playlist_path),
            )
        except Exception as exc:
            coverage_report = {
                "coverage_checked": True,
                "all_covered": False,
                "error": str(exc) or exc.__class__.__name__,
            }
    coverage_report_path.write_text(json.dumps(coverage_report, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(f"[LOG] playlist -> {playlist_path}", flush=True)
    print(f"[LOG] failures -> {failure_csv_path}", flush=True)
    print(f"[LOG] audit holds -> {audit_hold_csv_path}", flush=True)
    print(f"[LOG] human review overrides -> {human_override_csv_path}", flush=True)
    print(f"[LOG] coverage -> {coverage_report_path}", flush=True)

    verification: Optional[Dict[str, Any]] = None
    coverage_blocking = bool(coverage_report.get("coverage_checked")) and not bool(coverage_report.get("all_covered"))
    human_override_failed = int(human_override_summary.get("failure_count") or 0) > 0
    combined_verification_failed = False
    if args.dry_run:
        print(f"[DRY RUN] set not written; planned output -> {output}", flush=True)
    else:
        if failure_rows and not args.allow_partial:
            print("[HOLD] alignment failures found; rerun with --allow-partial to write a partial set", flush=True)
        elif coverage_blocking and not args.allow_partial:
            print(
                "[HOLD] coverage check failed; full-library production builds require every eligible drums-stem track. "
                "Use --allow-partial only for a draft/test ALS.",
                flush=True,
            )
        elif human_override_failed and not args.allow_partial:
            print(
                "[HOLD] validated human review overrides could not be written or verified. "
                "Production builds cannot ignore still-valid manual marker evidence.",
                flush=True,
            )
        elif audit_hold_rows and not args.allow_unsafe_audit:
            print(
                "[HOLD] visual audit holds found; production builds require audit_status=pass for every row. "
                "Use build_visual_first_production_queue.py to create a review queue, or rerun with "
                "--allow-unsafe-audit only for a draft/test ALS.",
                flush=True,
            )
        else:
            verification = _build_combined_set(base_als, output, processed_rows)
            combined_verification_failed = not bool(verification.get("all_rows_match_expected"))
            print(f"[DONE] wrote ALS -> {output}", flush=True)
            print(f"[VERIFY] {json.dumps(verification, ensure_ascii=True)}", flush=True)
            if combined_verification_failed:
                print(
                    "[HOLD] combined ALS verification failed; row anchors do not match the validated marker report.",
                    flush=True,
                )

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_stamp": run_stamp,
        "stems": str(stems),
        "template": str(template),
        "base_als": str(base_als),
        "waveform_cache_dir": str(waveform_cache_dir),
        "output_als": str(output),
        "playlist_csv": str(playlist_path),
        "failure_csv": str(failure_csv_path),
        "audit_hold_csv": str(audit_hold_csv_path),
        "human_review_override_csv": str(human_override_csv_path),
        "coverage_json": str(coverage_report_path),
        "track_count": len(tracks),
        "input_track_count": len(all_tracks),
        "excluded_track_count": len(all_tracks) - len(tracks),
        "excluded_acapella_count": len(acapella_tracks),
        "excluded_missing_drums_count": len(missing_drums_tracks),
        "excluded_near_empty_drums_count": len(excluded_near_empty_rows),
        "excluded_missing_drums_rows": [
            {
                "bpm": track.get("bpm"),
                "key": track.get("key"),
                "energy": track.get("energy"),
                "folder": track.get("folder"),
                "src": track.get("src"),
            }
            for track in missing_drums_tracks
        ],
        "excluded_near_empty_drums_rows": excluded_near_empty_rows,
        "processed_count": len(processed_rows),
        "failure_count": len(failure_rows),
        "human_review_override_count": int(human_override_summary.get("applied_count") or 0),
        "human_review_override_failure_count": int(human_override_summary.get("failure_count") or 0),
        "human_review_override_stale_manual_mismatch_count": int(human_override_summary.get("stale_manual_mismatch_count") or 0),
        "audit_hold_count": len(audit_hold_rows),
        "coverage_checked": bool(coverage_report.get("coverage_checked")),
        "coverage_all_covered": bool(coverage_report.get("all_covered")),
        "coverage_eligible_track_count": int(coverage_report.get("eligible_track_count") or 0),
        "coverage_missing_from_summary_count": int(coverage_report.get("missing_from_summary_count") or 0),
        "coverage_extra_in_summary_count": int(coverage_report.get("extra_in_summary_count") or 0),
        "dry_run": bool(args.dry_run),
        "allow_partial": bool(args.allow_partial),
        "allow_unsafe_audit": bool(args.allow_unsafe_audit),
        "verification": verification,
        "coverage_report": coverage_report,
        "human_review_override_summary": human_override_summary,
        "processed_rows": processed_rows,
        "failure_rows": failure_rows,
        "excluded_near_empty_rows": excluded_near_empty_rows,
        "audit_hold_rows": audit_hold_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(f"[LOG] report -> {report_path}", flush=True)
    if failure_rows and not args.allow_partial:
        return 1
    if coverage_blocking and not args.allow_partial:
        return 1
    if human_override_failed and not args.allow_partial:
        return 1
    if audit_hold_rows and not args.allow_unsafe_audit:
        return 1
    if combined_verification_failed:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freshly regenerate visual-first DROP_ALIGNED ALS sources for the whole STEMS library, "
            "then build a new combined all-tracks ALS from those fresh sources."
        )
    )
    parser.add_argument("--stems", default=str(STEMS_ROOT_DIR), help="STEMS root with /BPM/Key/Track folders.")
    parser.add_argument("--template", default=str(DEFAULT_ALS_TEMPLATE), help="Per-track three-stem ALS template.")
    parser.add_argument("--base-als", default=str(BASE_ALS_TEMPLATE), help="Blank combined-set base ALS with CH tracks.")
    parser.add_argument("--out-dir", default=str(GENERATED_SET_DIR / "VisualFirstFresh"), help="Directory for the combined ALS.")
    parser.add_argument("--out", default="", help="Explicit output ALS path.")
    parser.add_argument("--report", default="", help="Explicit report JSON path.")
    parser.add_argument("--run-stamp", default="", help="Stable stamp for generated per-track files.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)), help="Parallel visual-first workers.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        help="Visual-first analysis sample rate. Defaults to GUI-grade resolution for sample-level drop placement.",
    )
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-stems", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per-track-timeout-sec",
        type=float,
        default=DEFAULT_PER_TRACK_TIMEOUT_SEC,
        help="Maximum seconds for one visual-first detector call. Use 0 to disable.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite same-stamp per-track fresh ALS/JSON files and output ALS.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N sorted tracks.")
    parser.add_argument("--allow-partial", action="store_true", help="Write combined ALS even if some tracks fail fresh alignment.")
    parser.add_argument(
        "--allow-unsafe-audit",
        action="store_true",
        help="Draft/test only: write a combined ALS even when visual audit rows are review/replace/fallback.",
    )
    parser.add_argument("--reuse-report", default="", help="Skip alignment and build from processed_rows in an existing report JSON.")
    parser.add_argument(
        "--waveform-cache-dir",
        default=str(DEFAULT_WAVEFORM_CACHE_DIR),
        help="Waveform cache directory for production GUI Boom-mask proof.",
    )
    parser.add_argument(
        "--human-correction-log",
        action="append",
        default=None,
        help=(
            "JSONL correction log to use for validated manual-review overrides. "
            "Can be repeated; defaults to discovered repo/out-dir correction logs."
        ),
    )
    parser.add_argument(
        "--no-human-review-overrides",
        action="store_true",
        help="Disable validated human-review override promotion for this build.",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
