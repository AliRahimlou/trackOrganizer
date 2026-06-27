#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _reexec_with_local_venv() -> None:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_with_local_venv()

from drop_aligner.boom_profile import (  # noqa: E402
    boom_proof_front_edge_freshness,
    load_boom_profile,
    marker_boom_proof,
)
from drop_aligner.exclusions import row_has_excluded_path, row_is_acapella  # noqa: E402
from drop_aligner.production_failure_inventory import (  # noqa: E402
    build_failure_inventory,
    build_regression_seed_payload,
    write_jsonl,
)
from drop_aligner.structure_features import compute_bar_feature_map  # noqa: E402
from drop_aligner.visual_first import (  # noqa: E402
    EXTENDED_VISUAL_MAX_CLOCK_BAR,
    VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
    _accept_boom_proof_by_sparse_impact,
    _accept_gui_contract_by_actual_body_proof,
    _accept_gui_contract_by_staccato_front_body_proof,
    _candidate_has_minimum_actual_visual_body,
    _gui_mask_proof as visual_gui_mask_proof,
    _gui_mask_proof_needs_front_edge_repair,
    boom_body_section_candidates,
)
from drop_aligner.waveform import (  # noqa: E402
    accept_gui_boom_mask_with_front_edge_proof,
    enforce_strict_gui_boom_mask_contract,
    gui_boom_mask_proof_for_tile,
    gui_boom_mask_strict_contract_issue,
)
from audit_visual_first_human_review_memory import audit_human_review_memory  # noqa: E402
from audit_visual_first_library_coverage import compare_rows_to_eligible  # noqa: E402
from build_fresh_visual_first_library_set import _verify_combined_set  # noqa: E402
from project_config import STEMS_ROOT_DIR  # noqa: E402


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
VALIDATED_HUMAN_OVERRIDE_SELECTED_BY = "visual_validated_human_review_override"
DEFAULT_OUT_DIR = Path("models")
DEFAULT_REPORT_ROOT = Path.home() / "Desktop" / "MUSIC" / "GeneratedSet" / "VisualFirstFresh"
CURRENT_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_TRACKS_*_report.json"
LEGACY_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_DRUMS_*_report.json"
MAX_GRID_ONE_DISTANCE_MS = 90.0
DERIVED_REPORT_NAME_TOKENS = (
    "_detector_pass_only_report",
    "_production_validation",
)


def _unsafe_selected_source(selected_by: str) -> bool:
    source = str(selected_by or "").strip()
    return source in DISALLOWED_PASS_SOURCES or any(
        source.startswith(prefix) for prefix in DISALLOWED_PASS_SOURCE_PREFIXES
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def _safe_read_json(path: str | Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_report(path: Path) -> List[Dict[str, Any]]:
    payload = _safe_read_json(path)
    rows = payload.get("processed_rows") if isinstance(payload.get("processed_rows"), list) else []
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    return out


def _source_reports(root: Path, pattern: str) -> List[Path]:
    return sorted(
        (
            path
            for path in root.glob(pattern)
            if path.is_file()
            and not any(token in path.name for token in DERIVED_REPORT_NAME_TOKENS)
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _default_human_review_logs(report_path: Path, out_dir: Path) -> List[Path]:
    repo = Path(__file__).resolve().parent
    candidates = [
        repo / "drop_corrections.jsonl",
        report_path.parent / "drop_corrections.jsonl",
        out_dir / "drop_corrections.jsonl",
    ]
    logs: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = candidate.expanduser()
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        logs.append(path)
    return logs


def _human_review_logs_from_args(args: argparse.Namespace, report_path: Path, out_dir: Path) -> List[Path]:
    raw_logs = getattr(args, "human_correction_log", None)
    if raw_logs:
        return [Path(path).expanduser() for path in raw_logs]
    return _default_human_review_logs(report_path, out_dir)


def _latest_report(root: Path | None = None) -> Path:
    search_root = (root or DEFAULT_REPORT_ROOT).expanduser()
    current_reports = _source_reports(search_root, CURRENT_REPORT_GLOB)
    if current_reports:
        return current_reports[-1]

    legacy_reports = _source_reports(search_root, LEGACY_REPORT_GLOB)
    if legacy_reports:
        return legacy_reports[-1]

    raise FileNotFoundError(
        "No fresh visual-first production report found in "
        f"{search_root} using {CURRENT_REPORT_GLOB} or {LEGACY_REPORT_GLOB}"
    )


def _audit(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = payload.get("visual_audit") if isinstance(payload.get("visual_audit"), Mapping) else {}
    if audit:
        return audit
    status = str(row.get("audit_status") or "")
    flags = row.get("audit_flags") if isinstance(row.get("audit_flags"), list) else []
    return {"status": status, "flag_codes": flags}


def _selected_candidate(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    return selected if selected else {}


def _selected_by(row: Mapping[str, Any], selected: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    return str(selected.get("selected_by") or payload.get("selected_by") or row.get("selected_by") or "")


def _is_validated_human_override(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> bool:
    selected_by = _selected_by(row, selected, payload)
    return bool(
        selected_by == VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
        or isinstance(row.get("human_review_override"), Mapping)
        or isinstance(payload.get("human_review_override"), Mapping)
        or isinstance(selected.get("human_review_override"), Mapping)
    )


def _persisted_proofs(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    selected: Mapping[str, Any],
    key: str,
) -> List[Dict[str, Any]]:
    proofs: List[Dict[str, Any]] = []
    for source in (payload, selected, row):
        proof = source.get(key) if isinstance(source, Mapping) else {}
        if isinstance(proof, Mapping):
            proofs.append(dict(proof))
    return proofs


def _best_persisted_proof(proofs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    for proof in proofs:
        if bool(proof.get("passes")):
            return dict(proof)
    for proof in proofs:
        if proof:
            return dict(proof)
    return {}


def _proof_hold_reason(prefix: str, proof: Mapping[str, Any]) -> str:
    proof_reasons = [str(reason) for reason in proof.get("reasons") or [] if str(reason)]
    return prefix + "=hold" + (":" + ";".join(proof_reasons) if proof_reasons else ":missing")


def _gui_signal_hold_reason(prefix: str, proof: Mapping[str, Any]) -> Optional[str]:
    if "marker_relevant_mask" not in proof:
        return f"{prefix}=hold:stale_missing_marker_relevant_mask"
    if proof.get("marker_relevant_mask") is not True:
        return f"{prefix}=hold:marker_not_on_relevant_boom_mask"
    if "marker_signal_present" not in proof:
        return f"{prefix}=hold:stale_missing_marker_signal"
    if proof.get("marker_signal_present") is not True:
        return f"{prefix}=hold:blank_marker_no_visible_signal"
    return None


def _persisted_boom_freshness_reason(proof: Mapping[str, Any]) -> str:
    freshness = boom_proof_front_edge_freshness(proof)
    if bool(freshness.get("fresh")):
        return ""
    return "persisted_boom_proof=stale_front_edge:" + str(freshness.get("reason") or "unknown")


def _finite_optional(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _actual_visual_body_contract(selected: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(selected, Mapping) or not selected:
        return {
            "checked": True,
            "passes": False,
            "reasons": ["missing_selected_candidate"],
            "persisted_flag": None,
        }
    visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    persisted_flag = visual.get("actual_visual_body_contract_pass") if isinstance(visual, Mapping) else None
    body_keys = (
        "boom_section_darkness",
        "bar_height",
        "max_post8_height",
        "post8_height",
        "post4_height",
    )
    body_values = [
        value
        for value in (_finite_optional(visual.get(key)) for key in body_keys)
        if value is not None
    ]
    support_keys = (
        "post_bass8",
        "post_drum8",
        "post_drum_cont4",
        "post_drum_cont8",
        "jump4",
        "jump8",
        "phrase_prior",
    )
    support_values = [
        value
        for value in (_finite_optional(visual.get(key)) for key in support_keys)
        if value is not None
    ]
    if not body_values and not support_values:
        return {
            "checked": True,
            "passes": False,
            "reasons": ["missing_visual_body_metrics"],
            "persisted_flag": persisted_flag,
        }
    passes = bool(_candidate_has_minimum_actual_visual_body(selected))
    reasons = [] if passes else ["actual_visual_body_below_floor"]
    return {
        "checked": True,
        "passes": passes,
        "reasons": reasons,
        "persisted_flag": persisted_flag,
        "body_peak": max(body_values) if body_values else None,
        "support_peak": max(support_values) if support_values else None,
    }


def _grid_on_one_contract(
    selected: Mapping[str, Any],
    proof: Mapping[str, Any],
    *additional_proofs: Mapping[str, Any],
    max_one_distance_ms: float = MAX_GRID_ONE_DISTANCE_MS,
) -> Dict[str, Any]:
    values: List[float] = []
    selected_clock = selected.get("bpm_clock") if isinstance(selected.get("bpm_clock"), Mapping) else {}
    selected_visual = selected.get("visual_components") if isinstance(selected.get("visual_components"), Mapping) else {}
    proof_sources: List[Mapping[str, Any]] = [proof, *additional_proofs]
    metric_sources: List[Mapping[str, Any]] = [selected_clock, selected_visual]
    for proof_source in proof_sources:
        if not isinstance(proof_source, Mapping):
            continue
        nearest = proof_source.get("nearest") if isinstance(proof_source.get("nearest"), Mapping) else {}
        nearest_profile = proof_source.get("nearest_profile") if isinstance(proof_source.get("nearest_profile"), Mapping) else {}
        profile_metrics = (
            nearest_profile.get("metrics") if isinstance(nearest_profile.get("metrics"), Mapping) else {}
        )
        metric_sources.extend([nearest, profile_metrics])
    for source in metric_sources:
        value = _finite_optional(source.get("one_distance_ms") if isinstance(source, Mapping) else None)
        if value is not None:
            values.append(abs(float(value)))
    if values:
        distance = min(values)
        return {
            "checked": True,
            "passes": distance <= float(max_one_distance_ms),
            "one_distance_ms": distance,
            "max_one_distance_ms": float(max_one_distance_ms),
            "source": "one_distance_ms",
            "reasons": []
            if distance <= float(max_one_distance_ms)
            else [f"one_distance_ms {distance:.1f} above {float(max_one_distance_ms):.1f}"],
        }
    if selected_clock.get("on_one") is True:
        return {
            "checked": True,
            "passes": True,
            "one_distance_ms": None,
            "max_one_distance_ms": float(max_one_distance_ms),
            "source": "on_one_boolean",
            "reasons": [],
        }
    return {
        "checked": True,
        "passes": False,
        "one_distance_ms": None,
        "max_one_distance_ms": float(max_one_distance_ms),
        "source": "missing",
        "reasons": ["missing_grid_one_evidence"],
    }


def _marker(row: Mapping[str, Any], payload: Mapping[str, Any], selected: Mapping[str, Any]) -> Optional[float]:
    for value in (
        row.get("marker"),
        payload.get("drop_sec"),
        payload.get("final_ai_pick"),
        selected.get("timestamp"),
        selected.get("microaligned_time"),
        selected.get("snapped_sec"),
    ):
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return float(out)
    return None


def _gui_mask_proof_for_tile(tile: Mapping[str, Any], marker: float, *, radius_bins: int = 2) -> Dict[str, Any]:
    return gui_boom_mask_proof_for_tile(tile, marker, radius_bins=radius_bins)


def _gui_mask_proof(
    audio_path: str,
    marker: float,
    *,
    cache_dir: str | Path,
    view_sec: float = 6.0,
    width: int = 1200,
) -> Dict[str, Any]:
    return visual_gui_mask_proof(
        audio_path,
        marker,
        cache_dir=cache_dir,
        view_sec=view_sec,
        width=width,
    )


def _validate_one(payload: Mapping[str, Any]) -> Dict[str, Any]:
    row = payload["row"] if isinstance(payload.get("row"), Mapping) else {}
    row_index = int(payload.get("index") or 0)
    profile_path = payload.get("profile_path") or None
    sample_rate = int(payload.get("sample_rate") or 44100)
    rerun_detector = bool(payload.get("rerun_detector"))
    require_rerun_matches_report = bool(payload.get("require_rerun_matches_report"))
    rerun_report_tolerance_sec = float(payload.get("rerun_report_tolerance_sec") or 0.002)
    skip_gui_mask = bool(payload.get("skip_gui_mask"))
    skip_persisted_proof = bool(payload.get("skip_persisted_proof"))
    waveform_cache_dir = str(payload.get("waveform_cache_dir") or (DEFAULT_REPORT_ROOT / ".waveform_cache"))
    started = time.time()
    candidates_json = str(row.get("candidates_json") or "")
    report_candidate_payload = _safe_read_json(candidates_json) if candidates_json else {}
    report_selected = _selected_candidate(row, report_candidate_payload)
    report_marker = _marker(row, report_candidate_payload, report_selected)
    validated_human_override = _is_validated_human_override(row, report_candidate_payload, report_selected)
    persisted_boom_proofs = _persisted_proofs(row, report_candidate_payload, report_selected, "boom_proof")
    persisted_gui_proofs = _persisted_proofs(row, report_candidate_payload, report_selected, "gui_mask_proof")
    persisted_boom_proof = _best_persisted_proof(persisted_boom_proofs)
    persisted_gui_proof = _best_persisted_proof(persisted_gui_proofs)
    candidate_payload = dict(report_candidate_payload)
    drums_path = str(row.get("drums_path") or candidate_payload.get("audio_path") or "")
    if rerun_detector and drums_path:
        try:
            from drop_aligner.visual_first import visual_first_marker

            detector_result = visual_first_marker(drums_path, sample_rate=sample_rate, use_cache=True)
            if isinstance(detector_result, Mapping) and detector_result.get("ok"):
                candidate_payload = dict(detector_result)
            else:
                error = ""
                if isinstance(detector_result, Mapping):
                    error = str(detector_result.get("error") or detector_result.get("status") or "")
                candidate_payload = {
                    "audio_path": drums_path,
                    "visual_audit": {
                        "status": "error",
                        "flag_codes": [f"detector_not_ok:{error or 'unknown'}"],
                    },
                }
        except Exception as exc:
            candidate_payload = {
                "audio_path": drums_path,
                "visual_audit": {"status": "error", "flag_codes": [f"detector_error:{str(exc) or exc.__class__.__name__}"]},
            }
    detector_selected = _selected_candidate(row, candidate_payload)
    detector_marker = _marker({}, candidate_payload, detector_selected) if rerun_detector else None
    if validated_human_override:
        selected = report_selected
        marker = report_marker
        selected_by = _selected_by(row, selected, report_candidate_payload)
        audit = _audit(row, report_candidate_payload)
    else:
        selected = detector_selected
        marker = detector_marker if rerun_detector else _marker(row, candidate_payload, selected)
        selected_by = _selected_by(row, selected, candidate_payload)
        audit = _audit(row, candidate_payload)
    drums_path = str(row.get("drums_path") or candidate_payload.get("audio_path") or drums_path)
    reasons: List[str] = []
    if marker is None:
        reasons.append("missing_marker")
    if not drums_path:
        reasons.append("missing_drums_path")
    if _unsafe_selected_source(selected_by):
        reasons.append(f"unsafe_source:{selected_by}")
    audit_status = str(audit.get("status") or "").strip().lower()
    audit_flags = [str(flag) for flag in audit.get("flag_codes") or [] if str(flag)]
    if audit_status != "pass":
        reasons.append(f"audit_status={audit_status or 'missing'}")
    if audit_flags:
        reasons.append("audit_flags=" + ";".join(audit_flags))
    actual_visual_body_contract = _actual_visual_body_contract(selected)
    if not bool(actual_visual_body_contract.get("passes")):
        body_reasons = [str(reason) for reason in actual_visual_body_contract.get("reasons") or [] if str(reason)]
        reasons.append(
            "actual_visual_body_contract=hold"
            + (":" + ";".join(body_reasons) if body_reasons else ":missing")
        )
    if not skip_persisted_proof:
        if not bool(persisted_boom_proof.get("passes")):
            reasons.append(_proof_hold_reason("persisted_boom_proof", persisted_boom_proof))
        else:
            stale_reason = _persisted_boom_freshness_reason(persisted_boom_proof)
            if stale_reason:
                reasons.append(stale_reason)
        if not bool(persisted_gui_proof.get("passes")):
            reasons.append(_proof_hold_reason("persisted_gui_mask_proof", persisted_gui_proof))
        elif _gui_mask_proof_needs_front_edge_repair(persisted_gui_proof):
            reasons.append("persisted_gui_mask_proof=wide_exact_boom_front_edge_mismatch")
        elif gui_boom_mask_strict_contract_issue(persisted_gui_proof):
            reasons.append(
                "persisted_gui_mask_proof=strict_contract_hold:"
                + str(gui_boom_mask_strict_contract_issue(persisted_gui_proof))
            )
        else:
            gui_signal_reason = _gui_signal_hold_reason("persisted_gui_mask_proof", persisted_gui_proof)
            if gui_signal_reason:
                reasons.append(gui_signal_reason)
    rerun_report_delta_sec = None
    if rerun_detector and require_rerun_matches_report:
        if validated_human_override:
            if detector_marker is not None and report_marker is not None:
                rerun_report_delta_sec = abs(float(detector_marker) - float(report_marker))
        elif marker is None:
            reasons.append("rerun_marker_missing")
        elif report_marker is None:
            reasons.append("report_marker_missing_for_rerun_compare")
        else:
            rerun_report_delta_sec = abs(float(marker) - float(report_marker))
            if rerun_report_delta_sec > rerun_report_tolerance_sec:
                reasons.append(
                    f"rerun_marker_mismatch:{rerun_report_delta_sec:.9f}s"
                    f">tolerance:{rerun_report_tolerance_sec:.9f}s"
                )

    proof: Dict[str, Any] = {}
    if marker is not None and drums_path:
        try:
            feature_map = compute_bar_feature_map(drums_path, sample_rate=sample_rate, use_cache=True)
            max_clock_bar = max(
                81,
                min(EXTENDED_VISUAL_MAX_CLOCK_BAR, int(feature_map.get("bar_count", 0) or EXTENDED_VISUAL_MAX_CLOCK_BAR)),
            )
            payload_boom_candidates = [
                dict(candidate)
                for candidate in candidate_payload.get("boom_candidates") or []
                if isinstance(candidate, Mapping)
            ]
            boom_candidates = (
                payload_boom_candidates
                if payload_boom_candidates
                else boom_body_section_candidates(feature_map, max_clock_bar=max_clock_bar)
            )
            beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
            profile = load_boom_profile(profile_path)
            proof_candidates: List[Dict[str, Any]] = []
            if isinstance(selected, Mapping) and selected:
                proof_candidates.append(dict(selected))
            seen_times = {
                (
                    str(candidate.get("selected_by") or candidate.get("source") or ""),
                    round(float(_safe_float(candidate.get("timestamp"), default=-1.0)), 3),
                )
                for candidate in proof_candidates
                if isinstance(candidate, Mapping) and candidate.get("timestamp") is not None
            }
            for candidate in boom_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_time = candidate.get("timestamp")
                if candidate_time is None:
                    candidate_time = candidate.get("time_sec")
                if candidate_time is None:
                    candidate_time = candidate.get("snapped_sec")
                key = None
                if candidate_time is not None:
                    key = (
                        str(candidate.get("selected_by") or candidate.get("source") or ""),
                        round(float(_safe_float(candidate_time, default=-1.0)), 3),
                    )
                    if key in seen_times:
                        continue
                    seen_times.add(key)
                proof_candidates.append(dict(candidate))
            proof = marker_boom_proof(
                float(marker),
                proof_candidates,
                selected_candidate=selected,
                profile=profile,
                beatgrid=beatgrid,
            )
            proof = _accept_boom_proof_by_sparse_impact(proof, selected)
        except Exception as exc:
            proof = {"passes": False, "reasons": [f"boom_proof_error:{str(exc) or exc.__class__.__name__}"]}
    else:
        proof = {"passes": False, "reasons": ["missing_marker_or_track"]}

    if not bool(proof.get("passes")):
        if not rerun_detector and bool(persisted_boom_proof.get("passes")):
            proof = dict(persisted_boom_proof)
            proof["accepted_by_persisted_no_rerun_proof"] = True
        else:
            proof_reasons = [str(reason) for reason in proof.get("reasons") or []]
            reasons.append("boom_proof=hold" + (":" + ";".join(proof_reasons) if proof_reasons else ":missing"))
    grid_on_one_contract = _grid_on_one_contract(selected, proof, persisted_boom_proof)
    if not bool(grid_on_one_contract.get("passes")):
        grid_reasons = [str(reason) for reason in grid_on_one_contract.get("reasons") or [] if str(reason)]
        reasons.append(
            "grid_on_one_contract=hold"
            + (":" + ";".join(grid_reasons) if grid_reasons else ":missing")
        )
    gui_mask: Dict[str, Any] = {"checked": False, "passes": False, "reasons": ["skipped"]}
    if not skip_gui_mask:
        if marker is not None and drums_path:
            try:
                gui_mask = {
                    "checked": True,
                    **_gui_mask_proof(drums_path, float(marker), cache_dir=waveform_cache_dir),
                }
            except Exception as exc:
                gui_mask = {
                    "checked": True,
                    "passes": False,
                    "reasons": [f"gui_mask_error:{str(exc) or exc.__class__.__name__}"],
                }
        else:
            gui_mask = {"checked": True, "passes": False, "reasons": ["missing_marker_or_track"]}
        repair_gui_lag_sec = 0.300 if str(selected_by or "").startswith("visual_final_contract_") else 0.040
        repair_gui_profile = 0.620 if str(selected_by or "").startswith("visual_final_contract_") else 0.560
        gui_mask = accept_gui_boom_mask_with_front_edge_proof(
            gui_mask,
            proof,
            near_offset_sec=repair_gui_lag_sec,
            near_profile_score=repair_gui_profile,
        )
        gui_mask = _accept_gui_contract_by_staccato_front_body_proof(gui_mask, proof, selected)
        gui_mask = _accept_gui_contract_by_actual_body_proof(gui_mask, proof, selected)
        gui_mask = enforce_strict_gui_boom_mask_contract(gui_mask)
        gui_signal_reason = _gui_signal_hold_reason("gui_mask", gui_mask) if bool(gui_mask.get("passes")) else None
        if gui_signal_reason:
            gui_mask = {
                **dict(gui_mask),
                "passes": False,
                "reasons": [*list(gui_mask.get("reasons") or []), gui_signal_reason.split(":", 1)[-1]],
            }
        if not bool(gui_mask.get("passes")):
            gui_reasons = [str(reason) for reason in gui_mask.get("reasons") or [] if str(reason)]
            reasons.append("gui_mask=hold" + (":" + ";".join(gui_reasons) if gui_reasons else ":missing"))
        elif _gui_mask_proof_needs_front_edge_repair(gui_mask):
            reasons.append("gui_mask=wide_exact_boom_front_edge_mismatch")
    nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
    nearest_profile = proof.get("nearest_profile") if isinstance(proof.get("nearest_profile"), Mapping) else {}
    suggested_marker = None
    if nearest and bool(nearest_profile.get("passes_profile")):
        suggested_marker = nearest.get("edge_time")
    return {
        "index": row_index,
        "passes": not reasons,
        "reasons": reasons,
        "track": drums_path,
        "marker": marker,
        "report_marker": report_marker,
        "rerun_report_delta_sec": rerun_report_delta_sec,
        "rerun_report_match_exempt": bool(validated_human_override),
        "validated_human_override": bool(validated_human_override),
        "selected_by": selected_by,
        "audit_status": audit_status,
        "audit_flags": audit_flags,
        "actual_visual_body_contract_pass": bool(actual_visual_body_contract.get("passes")),
        "actual_visual_body_contract": actual_visual_body_contract,
        "grid_on_one_contract_pass": bool(grid_on_one_contract.get("passes")),
        "grid_on_one_contract": grid_on_one_contract,
        "boom_proof": proof,
        "persisted_boom_proof": persisted_boom_proof,
        "persisted_boom_proof_count": len(persisted_boom_proofs),
        "boom_nearest_edge": nearest.get("edge_time"),
        "boom_edge_offset_sec": nearest.get("offset_sec"),
        "boom_clock_bar": nearest.get("clock_bar"),
        "boom_profile_score": nearest_profile.get("profile_score"),
        "gui_mask": gui_mask,
        "persisted_gui_mask_proof": persisted_gui_proof,
        "persisted_gui_mask_proof_count": len(persisted_gui_proofs),
        "suggested_marker_sec": suggested_marker,
        "candidates_json": candidates_json,
        "output_als": str(row.get("output_als") or ""),
        "elapsed_sec": time.time() - started,
    }


def _reason_family(reason: str) -> str:
    text = str(reason or "")
    if text.startswith("audit_status="):
        return text
    if text.startswith("audit_flags="):
        flags = text.split("=", 1)[1]
        return "audit_flags=" + ";".join(sorted(flag for flag in flags.split(";") if flag))
    if text.startswith("unsafe_source:"):
        return text
    if text.startswith("actual_visual_body_contract=hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        if "missing_selected_candidate" in detail:
            return "actual_visual_body_contract:missing_selected"
        if "missing_visual_body_metrics" in detail:
            return "actual_visual_body_contract:missing_metrics"
        if "actual_visual_body_below_floor" in detail:
            return "actual_visual_body_contract:below_floor"
        return "actual_visual_body_contract:hold"
    if text.startswith("grid_on_one_contract=hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        if "missing_grid_one_evidence" in detail:
            return "grid_on_one:missing_evidence"
        if "one_distance_ms" in detail:
            return "grid_on_one:off_one"
        return "grid_on_one:hold"
    if text.startswith("persisted_boom_proof=hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        if detail == "missing":
            return "persisted_boom_proof:missing"
        return "persisted_boom_proof:hold"
    if text.startswith("persisted_boom_proof=stale_front_edge"):
        return "persisted_boom_proof:stale_front_edge"
    if text.startswith("persisted_gui_mask_proof=hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        if detail == "missing":
            return "persisted_gui_mask_proof:missing"
        if "marker_relevant_mask" in detail or "relevant_boom_mask" in detail:
            return "persisted_gui_mask_proof:not_relevant"
        if "marker_not_on_gui_boom_front_edge_mask" in detail:
            return "persisted_gui_mask_proof:not_placeable"
        return "persisted_gui_mask_proof:hold"
    if text.startswith("persisted_gui_mask_proof=strict_contract_hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        return "persisted_gui_mask_proof:strict_contract:" + (detail or "unknown")
    if text.startswith("gui_mask=hold"):
        detail = text.split(":", 1)[1] if ":" in text else ""
        if "marker_relevant_mask" in detail or "relevant_boom_mask" in detail:
            return "gui_mask:not_relevant"
        if "marker_not_on_gui_boom_front_edge_mask" in detail:
            return "gui_mask:not_placeable"
        if "no_placeable_boom_front_edge" in detail:
            return "gui_mask:no_placeable_front_edge"
        if "marker_has_no_immediate_drop_body" in detail:
            return "gui_mask:no_immediate_drop_body"
        if "missing_gui_boom_placeable_mask" in detail:
            return "gui_mask:missing_mask"
        if "marker_outside_gui_waveform_tile" in detail:
            return "gui_mask:outside_tile"
        return "gui_mask:hold"
    if not text.startswith("boom_proof=hold"):
        return text or "unknown"
    detail = text.split(":", 1)[1] if ":" in text else ""
    if "no_boom_body_section_candidate" in detail:
        return "boom:no_body_section_candidate"
    if "earlier_dominant_boom_available" in detail:
        return "boom:earlier_dominant_available"
    if "after boom front edge" in detail:
        return "boom:late_inside_body"
    if "before boom front edge" in detail:
        return "boom:before_body_front_edge"
    if "nearest_boom_edge" in detail:
        return "boom:far_from_body_edge"
    if any(token in detail for token in ("profile_score", "darkness", "post8_height", "simultaneity", "post_bass8", "post_drum8", "post_drum_cont8", "sustain", "contrast")):
        return "boom:profile_below_threshold"
    if "one_distance_ms" in detail:
        return "boom:off_one"
    return "boom:proof_hold"


def _write_failures_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "track",
        "marker",
        "report_marker",
        "rerun_report_delta_sec",
        "selected_by",
        "audit_status",
        "audit_flags",
        "boom_clock_bar",
        "boom_nearest_edge",
        "boom_edge_offset_sec",
        "boom_profile_score",
        "actual_visual_body_contract",
        "persisted_boom_proof",
        "persisted_gui_mask_proof",
        "gui_mask_placeable_count",
        "gui_mask_reasons",
        "suggested_marker_sec",
        "reasons",
        "candidates_json",
        "output_als",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "track": row.get("track", ""),
                    "marker": row.get("marker", ""),
                    "report_marker": row.get("report_marker", ""),
                    "rerun_report_delta_sec": row.get("rerun_report_delta_sec", ""),
                    "selected_by": row.get("selected_by", ""),
                    "audit_status": row.get("audit_status", ""),
                    "audit_flags": ";".join(str(flag) for flag in row.get("audit_flags") or []),
                    "boom_clock_bar": row.get("boom_clock_bar", ""),
                    "boom_nearest_edge": row.get("boom_nearest_edge", ""),
                    "boom_edge_offset_sec": row.get("boom_edge_offset_sec", ""),
                    "boom_profile_score": row.get("boom_profile_score", ""),
                    "actual_visual_body_contract": "pass"
                    if bool(row.get("actual_visual_body_contract_pass"))
                    else "hold",
                    "persisted_boom_proof": "pass"
                    if isinstance(row.get("persisted_boom_proof"), Mapping)
                    and bool(row["persisted_boom_proof"].get("passes"))
                    else "hold",
                    "persisted_gui_mask_proof": "pass"
                    if isinstance(row.get("persisted_gui_mask_proof"), Mapping)
                    and bool(row["persisted_gui_mask_proof"].get("passes"))
                    else "hold",
                    "gui_mask_placeable_count": (row.get("gui_mask") or {}).get("placeable_count", "")
                    if isinstance(row.get("gui_mask"), Mapping)
                    else "",
                    "gui_mask_reasons": ";".join(str(reason) for reason in (row.get("gui_mask") or {}).get("reasons") or [])
                    if isinstance(row.get("gui_mask"), Mapping)
                    else "",
                    "suggested_marker_sec": row.get("suggested_marker_sec", ""),
                    "reasons": ";".join(str(reason) for reason in row.get("reasons") or []),
                    "candidates_json": row.get("candidates_json", ""),
                    "output_als": row.get("output_als", ""),
                }
            )


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = Path(args.report).expanduser().resolve() if args.report else _latest_report().resolve()
    source_payload = _safe_read_json(report_path)
    rows = [dict(row) for row in source_payload.get("processed_rows") or [] if isinstance(row, Mapping)]
    input_count = len(rows)
    coverage_report: Dict[str, Any] = {"coverage_checked": False, "all_covered": False}
    coverage_required = not bool(args.skip_coverage) and not bool(args.limit)
    coverage_pass = True
    if coverage_required:
        try:
            coverage_report = compare_rows_to_eligible(
                rows,
                stems=Path(args.stems).expanduser(),
                source=str(report_path),
                sample_limit=int(args.coverage_sample_limit),
            )
        except Exception as exc:
            coverage_report = {
                "coverage_checked": True,
                "all_covered": False,
                "error": str(exc) or exc.__class__.__name__,
            }
        coverage_pass = bool(coverage_report.get("all_covered"))
    elif bool(args.skip_coverage):
        coverage_report = {"coverage_checked": False, "all_covered": False, "reason": "--skip-coverage"}
    elif bool(args.limit):
        coverage_report = {"coverage_checked": False, "all_covered": False, "reason": "coverage skipped for --limit"}

    combined_als_report: Dict[str, Any] = {"checked": False, "all_rows_match_expected": False}
    combined_als_pass = True
    combined_required = not bool(args.skip_combined_als) and not bool(args.limit)
    if combined_required:
        output_als = str(source_payload.get("output_als") or "").strip()
        if not output_als:
            combined_als_report = {
                "checked": True,
                "all_rows_match_expected": False,
                "error": "missing output_als in source report",
            }
        else:
            try:
                combined_als_report = {
                    "checked": True,
                    "output_als": str(Path(output_als).expanduser()),
                    **_verify_combined_set(Path(output_als).expanduser(), rows),
                }
            except Exception as exc:
                combined_als_report = {
                    "checked": True,
                    "output_als": str(Path(output_als).expanduser()),
                    "all_rows_match_expected": False,
                    "error": str(exc) or exc.__class__.__name__,
                }
        combined_als_pass = bool(combined_als_report.get("all_rows_match_expected"))
    elif bool(args.skip_combined_als):
        combined_als_report = {"checked": False, "all_rows_match_expected": False, "reason": "--skip-combined-als"}
    elif bool(args.limit):
        combined_als_report = {"checked": False, "all_rows_match_expected": False, "reason": "combined ALS skipped for --limit"}

    excluded_rows = [row for row in rows if row_has_excluded_path(row) or row_is_acapella(row)]
    if not bool(args.include_excluded):
        rows = [row for row in rows if not (row_has_excluded_path(row) or row_is_acapella(row))]
    if args.limit:
        rows = rows[: int(args.limit)]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    human_review_audit_report: Dict[str, Any] = {"checked": False, "passed": True}
    human_review_audit_pass = True
    human_review_required = not bool(getattr(args, "skip_human_review_audit", False)) and not bool(args.limit)
    if human_review_required:
        correction_logs = _human_review_logs_from_args(args, report_path, out_dir)
        if correction_logs:
            audit_out_dir_raw = str(getattr(args, "human_review_audit_dir", "") or "").strip()
            audit_out_dir = Path(audit_out_dir_raw).expanduser().resolve() if audit_out_dir_raw else out_dir / "human_review_audit"
            cache_dir_raw = str(getattr(args, "waveform_cache_dir", "") or "").strip()
            cache_dir = Path(cache_dir_raw).expanduser() if cache_dir_raw else report_path.parent / ".waveform_cache"
            try:
                human_review_audit_report = {
                    "checked": True,
                    **audit_human_review_memory(
                        report_path,
                        correction_logs=correction_logs,
                        out_dir=audit_out_dir,
                        cache_dir=cache_dir,
                        manual_tolerance_sec=float(getattr(args, "manual_review_tolerance_sec", 0.050) or 0.050),
                        blue_tolerance_sec=float(getattr(args, "blue_review_tolerance_sec", 0.050) or 0.050),
                    ),
                }
            except Exception as exc:
                human_review_audit_report = {
                    "checked": True,
                    "passed": False,
                    "error": str(exc) or exc.__class__.__name__,
                    "correction_logs": [str(path) for path in correction_logs],
                }
            human_review_audit_pass = bool(human_review_audit_report.get("passed"))
        else:
            human_review_audit_report = {
                "checked": False,
                "passed": True,
                "reason": "no correction logs found",
            }
    elif bool(getattr(args, "skip_human_review_audit", False)):
        human_review_audit_report = {"checked": False, "passed": True, "reason": "--skip-human-review-audit"}
    elif bool(args.limit):
        human_review_audit_report = {"checked": False, "passed": True, "reason": "human review audit skipped for --limit"}

    started = time.time()
    jobs = [
        {
            "index": index,
            "row": row,
            "sample_rate": int(args.sample_rate),
            "profile_path": str(Path(args.profile).expanduser()) if args.profile else None,
            "rerun_detector": bool(args.rerun_detector),
            "require_rerun_matches_report": bool(args.rerun_detector) and not bool(args.skip_rerun_report_match),
            "rerun_report_tolerance_sec": float(args.rerun_report_tolerance_sec),
            "skip_gui_mask": bool(args.skip_gui_mask),
            "skip_persisted_proof": bool(args.skip_persisted_proof),
            "waveform_cache_dir": str(Path(args.waveform_cache_dir).expanduser())
            if args.waveform_cache_dir
            else str(report_path.parent / ".waveform_cache"),
        }
        for index, row in enumerate(rows, start=1)
    ]
    results: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_validate_one(job))
            if args.progress_every and index % int(args.progress_every) == 0:
                print(f"validated {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_validate_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"validated {index}/{len(jobs)}", flush=True)

    failures = [row for row in results if not row.get("passes")]
    passes = [row for row in results if row.get("passes")]
    reason_counts = Counter(reason for row in failures for reason in row.get("reasons") or [])
    reason_family_counts = Counter(_reason_family(reason) for row in failures for reason in row.get("reasons") or [])
    failure_inventory = build_failure_inventory(failures, source_report=str(report_path))
    taxonomy_counts = Counter(str(row.get("taxonomy") or "unknown_hold") for row in failure_inventory)
    source_counts = Counter(str(row.get("selected_by") or "") for row in results)
    gui_mask_checked_count = sum(1 for row in results if isinstance(row.get("gui_mask"), Mapping) and row["gui_mask"].get("checked"))
    gui_mask_pass_count = sum(1 for row in results if isinstance(row.get("gui_mask"), Mapping) and row["gui_mask"].get("passes"))
    gui_mask_hold_count = gui_mask_checked_count - gui_mask_pass_count
    actual_visual_body_checked_count = sum(
        1
        for row in results
        if isinstance(row.get("actual_visual_body_contract"), Mapping)
        and row["actual_visual_body_contract"].get("checked")
    )
    actual_visual_body_pass_count = sum(
        1
        for row in results
        if isinstance(row.get("actual_visual_body_contract"), Mapping)
        and bool(row["actual_visual_body_contract"].get("passes"))
    )
    actual_visual_body_hold_count = actual_visual_body_checked_count - actual_visual_body_pass_count
    persisted_boom_proof_pass_count = sum(
        1
        for row in results
        if isinstance(row.get("persisted_boom_proof"), Mapping)
        and bool(row["persisted_boom_proof"].get("passes"))
    )
    persisted_gui_mask_proof_pass_count = sum(
        1
        for row in results
        if isinstance(row.get("persisted_gui_mask_proof"), Mapping)
        and bool(row["persisted_gui_mask_proof"].get("passes"))
    )
    rerun_checked_count = len(results) if bool(args.rerun_detector) else 0
    rerun_reason_prefixes = (
        "rerun_marker_missing",
        "report_marker_missing_for_rerun_compare",
        "rerun_marker_mismatch:",
    )
    rerun_mismatch_count = (
        sum(
            1
            for row in results
            if any(str(reason).startswith(rerun_reason_prefixes) for reason in row.get("reasons") or [])
        )
        if bool(args.rerun_detector) and not bool(args.skip_rerun_report_match)
        else 0
    )
    rerun_match_count = (
        max(0, rerun_checked_count - rerun_mismatch_count)
        if bool(args.rerun_detector) and not bool(args.skip_rerun_report_match)
        else 0
    )
    human_override_rerun_exempt_count = sum(1 for row in results if bool(row.get("rerun_report_match_exempt")))
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(report_path),
        "profile": str(Path(args.profile).expanduser()) if args.profile else "default/models/boom_profile.json",
        "sample_rate": int(args.sample_rate),
        "rerun_detector": bool(args.rerun_detector),
        "rerun_report_match_required": bool(args.rerun_detector) and not bool(args.skip_rerun_report_match),
        "rerun_checked_count": int(rerun_checked_count),
        "rerun_match_count": int(rerun_match_count),
        "rerun_mismatch_count": int(rerun_mismatch_count),
        "human_override_rerun_exempt_count": int(human_override_rerun_exempt_count),
        "rerun_report_tolerance_sec": float(args.rerun_report_tolerance_sec),
        "input_count": int(input_count),
        "excluded_count": int(len(excluded_rows)) if not bool(args.include_excluded) else 0,
        "coverage_checked": bool(coverage_report.get("coverage_checked")),
        "coverage_all_covered": bool(coverage_report.get("all_covered")),
        "coverage_eligible_track_count": int(coverage_report.get("eligible_track_count") or 0),
        "coverage_missing_from_summary_count": int(coverage_report.get("missing_from_summary_count") or 0),
        "coverage_extra_in_summary_count": int(coverage_report.get("extra_in_summary_count") or 0),
        "combined_als_checked": bool(combined_als_report.get("checked")),
        "combined_als_all_rows_match_expected": bool(combined_als_report.get("all_rows_match_expected")),
        "combined_als_row_count": int(combined_als_report.get("row_count") or 0),
        "combined_als_expected_row_count": int(combined_als_report.get("expected_row_count") or 0),
        "combined_als_anchor_mismatch_count": int(combined_als_report.get("anchor_mismatch_count") or 0),
        "combined_als_file_ref_mismatch_count": int(combined_als_report.get("file_ref_mismatch_count") or 0),
        "combined_als_file_ref_match_rows": int(combined_als_report.get("file_ref_match_rows") or 0),
        "combined_als_all_file_refs_match_expected": bool(combined_als_report.get("all_file_refs_match_expected")),
        "human_review_audit_checked": bool(human_review_audit_report.get("checked")),
        "human_review_audit_passed": bool(human_review_audit_report.get("passed")),
        "human_review_marker_count": int(human_review_audit_report.get("review_marker_count") or 0),
        "human_review_matched_rows": int(human_review_audit_report.get("matched_review_rows") or 0),
        "human_review_manual_rows": int(human_review_audit_report.get("manual_review_rows") or 0),
        "human_review_blue_approval_rows": int(human_review_audit_report.get("blue_approval_rows") or 0),
        "human_review_hard_mismatch_count": int(human_review_audit_report.get("hard_mismatch_count") or 0),
        "human_review_validated_hard_mismatch_count": int(
            human_review_audit_report.get("validated_hard_mismatch_count") or 0
        ),
        "human_review_stale_manual_mismatch_count": int(human_review_audit_report.get("stale_manual_mismatch_count") or 0),
        "human_review_advisory_mismatch_count": int(human_review_audit_report.get("advisory_mismatch_count") or 0),
        "gui_mask_checked": not bool(args.skip_gui_mask),
        "gui_mask_checked_count": int(gui_mask_checked_count),
        "gui_mask_pass_count": int(gui_mask_pass_count),
        "gui_mask_hold_count": int(gui_mask_hold_count),
        "actual_visual_body_checked_count": int(actual_visual_body_checked_count),
        "actual_visual_body_pass_count": int(actual_visual_body_pass_count),
        "actual_visual_body_hold_count": int(actual_visual_body_hold_count),
        "persisted_proof_checked": not bool(args.skip_persisted_proof),
        "persisted_boom_proof_pass_count": int(persisted_boom_proof_pass_count),
        "persisted_gui_mask_proof_pass_count": int(persisted_gui_mask_proof_pass_count),
        "validated_count": len(results),
        "pass_count": len(passes),
        "hold_count": len(failures),
        "all_passed": len(failures) == 0
        and len(results) > 0
        and coverage_pass
        and combined_als_pass
        and human_review_audit_pass,
        "reason_counts": dict(reason_counts),
        "reason_family_counts": dict(reason_family_counts),
        "failure_taxonomy_counts": dict(taxonomy_counts),
        "selected_by_counts": dict(source_counts),
        "elapsed_sec": time.time() - started,
    }
    failures_csv = out_dir / f"{report_path.stem}_production_validation_failures.csv"
    failure_inventory_jsonl = out_dir / f"{report_path.stem}_production_failure_inventory.jsonl"
    regression_seed_json = out_dir / f"{report_path.stem}_production_regression_seeds.json"
    output_json = out_dir / f"{report_path.stem}_production_validation.json"
    _write_failures_csv(failures_csv, failures)
    write_jsonl(failure_inventory_jsonl, failure_inventory)
    regression_seed_payload = build_regression_seed_payload(failure_inventory)
    regression_seed_json.write_text(
        json.dumps(regression_seed_payload, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )
    output_json.write_text(
        json.dumps(
            {
                **summary,
                "failures_csv": str(failures_csv),
                "failure_inventory_jsonl": str(failure_inventory_jsonl),
                "regression_seed_json": str(regression_seed_json),
                "failure_inventory": failure_inventory,
                "regression_seeds": regression_seed_payload,
                "coverage_report": coverage_report,
                "combined_als_report": combined_als_report,
                "human_review_audit_report": human_review_audit_report,
                "results": results,
            },
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **summary,
                "failures_csv": str(failures_csv),
                "failure_inventory_jsonl": str(failure_inventory_jsonl),
                "regression_seed_json": str(regression_seed_json),
                "output_json": str(output_json),
            },
            indent=2,
        )
    )
    if bool(args.require_all_pass) and not bool(summary.get("all_passed")):
        return {**summary, "exit_code": 1}
    return {**summary, "exit_code": 0}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for visual-first all-library production ALS reports."
    )
    parser.add_argument("report", nargs="?", default="", help="Fresh visual-first report JSON. Defaults to latest VisualFirstFresh report.")
    parser.add_argument("--profile", default="", help="Boom profile JSON. Defaults to models/boom_profile.json when present.")
    parser.add_argument("--out-dir", default="", help="Output directory for validation report/CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Validate first N rows; 0 means all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        help="Detector rerun sample rate. Defaults to the same GUI-grade rate used by fresh production builds.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--rerun-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate current visual_first_marker output and require it to match saved production report "
            "markers. Enabled by default; use --no-rerun-detector only for a fast persisted-payload audit."
        ),
    )
    parser.add_argument(
        "--skip-rerun-report-match",
        action="store_true",
        help="With --rerun-detector, do not require current detector markers to match the saved production report markers.",
    )
    parser.add_argument("--rerun-report-tolerance-sec", type=float, default=0.002)
    parser.add_argument("--require-all-pass", action="store_true", help="Exit nonzero unless every row passes.")
    parser.add_argument("--include-excluded", action="store_true", help="Include excluded/acapella rows in validation.")
    parser.add_argument("--stems", default=str(STEMS_ROOT_DIR), help="STEMS root for full-library coverage verification.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip full-library coverage verification.")
    parser.add_argument("--skip-combined-als", action="store_true", help="Skip combined ALS row/anchor verification.")
    parser.add_argument(
        "--skip-human-review-audit",
        action="store_true",
        help="Skip correction-log/manual-review mismatch verification.",
    )
    parser.add_argument(
        "--human-correction-log",
        action="append",
        default=None,
        help=(
            "JSONL correction log to use for human-review mismatch verification. "
            "Can be repeated; defaults to repo and report-directory correction logs."
        ),
    )
    parser.add_argument(
        "--human-review-audit-dir",
        default="",
        help="Output directory for human-review audit JSON/CSVs. Defaults to <out-dir>/human_review_audit.",
    )
    parser.add_argument("--manual-review-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--blue-review-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--skip-gui-mask", action="store_true", help="Skip WebGUI waveform Boom mask verification.")
    parser.add_argument("--skip-persisted-proof", action="store_true", help="Skip candidate JSON/report persisted proof verification.")
    parser.add_argument("--waveform-cache-dir", default="", help="Waveform cache directory for GUI mask verification.")
    parser.add_argument("--coverage-sample-limit", type=int, default=25, help="Coverage mismatch samples to store in the JSON report.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = validate(parse_args())
    raise SystemExit(int(result.get("exit_code", 0) or 0))
