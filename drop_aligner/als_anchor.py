from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional

from .downbeat_impact import refine_approved_downbeat_attack
from .visual_first import (
    VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
    _gui_mask_proof_needs_front_edge_repair,
    visual_first_marker,
)
from .waveform import gui_boom_mask_strict_contract_issue


MAX_BPM_MISMATCH_RATIO = 0.0075
SELF_CALIBRATED_CLOCK_SOURCES = {
    "gui_boom_front_edge_calibrated_grid",
    "visual_reclaimed_body_phase_calibrated",
    "visual_boom_grid_phase_calibration",
    "visual_boom_phase_calibrated",
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reject(reason: str, **details: Any) -> Dict[str, Any]:
    return {
        "ok": False,
        "accepted": False,
        "reason": str(reason),
        **details,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_visual_first_als_anchor(
    audio_path: str,
    bpm: float,
    *,
    sample_rate: int = VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
    use_cache: bool = True,
    visual_result: Optional[Mapping[str, Any]] = None,
    impact_refiner: Callable[..., Mapping[str, Any]] = refine_approved_downbeat_attack,
) -> Dict[str, Any]:
    """Produce a fail-closed ALS anchor from independent grid and body proof.

    Stage A comes from the visual-first detector and chooses the drop section.
    Its timestamp is checked against the unchanged feature-map grid.  Stage B
    then searches only the small DJ-mode window around that approved bar one
    and returns the first sample-level drum-body attack.  A visual candidate is
    never allowed to translate the grid that proves it is on one.
    """

    bpm_value = _finite_float(bpm)
    if bpm_value is None or bpm_value <= 0.0:
        return _reject("invalid_bpm")

    if visual_result is None:
        try:
            result: Mapping[str, Any] = visual_first_marker(
                str(audio_path),
                sample_rate=int(sample_rate),
                use_cache=bool(use_cache),
            )
        except Exception as exc:
            return _reject(
                "visual_first_failed",
                error=str(exc) or exc.__class__.__name__,
            )
    else:
        result = visual_result

    if not bool(result.get("ok")) or bool(result.get("final_contract_failed")):
        return _reject(
            "visual_contract_failed",
            visual_error=str(result.get("error") or ""),
        )

    selected = _mapping(result.get("selected_candidate"))
    audit = _mapping(result.get("visual_audit"))
    boom = _mapping(result.get("boom_proof"))
    gui = _mapping(result.get("gui_mask_proof"))
    if str(audit.get("status") or "").lower() != "pass" or list(audit.get("flag_codes") or []):
        return _reject("visual_audit_not_passed")
    if not bool(boom.get("passes")):
        return _reject("boom_proof_not_passed")
    if not bool(gui.get("passes")):
        return _reject("gui_mask_not_passed")
    if _gui_mask_proof_needs_front_edge_repair(
        gui,
        trust_reasonless_relief_offset=True,
    ):
        return _reject("wide_exact_gui_front_edge_mismatch")
    strict_issue = gui_boom_mask_strict_contract_issue(gui)
    if strict_issue:
        return _reject("strict_gui_mask_not_passed", strict_gui_issue=str(strict_issue))
    if gui.get("marker_signal_present") is not True:
        return _reject("blank_waveform_at_marker")
    if gui.get("marker_relevant_mask") is not True:
        return _reject("marker_outside_relevant_body")

    marker = _finite_float(result.get("marker"))
    if marker is None or marker < 0.0:
        return _reject("missing_visual_marker")

    selected_clock = _mapping(selected.get("bpm_clock"))
    selected_clock_source = str(selected_clock.get("source") or "")
    selected_by = str(selected.get("selected_by") or "")
    selected_visual = _mapping(selected.get("visual_components"))
    if selected_clock_source in SELF_CALIBRATED_CLOCK_SOURCES or bool(
        selected_clock.get("visual_phase_calibration_marker_sec") is not None
        or (
            "calibrat" in selected_clock_source.lower()
            and (
                "visual" in selected_clock_source.lower()
                or "gui" in selected_clock_source.lower()
            )
        )
        or "phase_calibrat" in selected_by.lower()
        or selected_visual.get("visual_boom_grid_phase_calibrated")
        or selected_visual.get("visual_body_reclaim_phase_calibrated")
    ):
        return _reject(
            "visual_candidate_self_calibrated_grid",
            clock_source=selected_clock_source,
        )

    feature_map = _mapping(result.get("feature_map"))
    beatgrid = _mapping(feature_map.get("beatgrid"))
    grid_bpm = _finite_float(beatgrid.get("bpm"))
    grid_zero = _finite_float(beatgrid.get("bar_zero_sec"))
    if grid_bpm is None or grid_bpm <= 0.0 or grid_zero is None:
        return _reject("missing_independent_beatgrid")
    bpm_mismatch = abs(float(grid_bpm) - float(bpm_value)) / max(float(bpm_value), 1e-9)
    if bpm_mismatch > MAX_BPM_MISMATCH_RATIO:
        return _reject(
            "title_bpm_mismatch",
            title_bpm=float(bpm_value),
            grid_bpm=float(grid_bpm),
            mismatch_ratio=float(bpm_mismatch),
        )

    beat_sec = 60.0 / float(bpm_value)
    bar_sec = 4.0 * float(beat_sec)
    nearest_bar_index = int(round((float(marker) - float(grid_zero)) / float(bar_sec)))
    grid_downbeat_sec = float(grid_zero) + (float(nearest_bar_index) * float(bar_sec))
    if grid_downbeat_sec < 0.0:
        return _reject("nearest_grid_one_before_audio")
    visual_delta_sec = float(marker) - float(grid_downbeat_sec)
    max_after_sec = min(0.090, 0.25 * float(beat_sec))
    visual_sample_tolerance_sec = 1.0 / float(max(1, int(sample_rate)))
    if (
        visual_delta_sec < -0.008 - visual_sample_tolerance_sec
        or visual_delta_sec > max_after_sec + visual_sample_tolerance_sec
    ):
        return _reject(
            "visual_marker_not_on_independent_one",
            marker_sec=float(marker),
            grid_downbeat_sec=float(grid_downbeat_sec),
            delta_ms=float(visual_delta_sec) * 1000.0,
        )

    try:
        attack = dict(
            impact_refiner(
                str(audio_path),
                float(grid_downbeat_sec),
                float(bpm_value),
            )
        )
    except Exception as exc:
        return _reject(
            "bounded_attack_refinement_failed",
            error=str(exc) or exc.__class__.__name__,
        )
    if not bool(attack.get("ok")) or not bool(attack.get("refined")):
        return _reject(
            "bounded_attack_not_proven",
            attack_reason=str(attack.get("reason") or ""),
            attack=dict(attack),
        )

    impact_sec = _finite_float(attack.get("attack_time_sec"))
    impact_sample = attack.get("attack_sample")
    grid_sample = attack.get("downbeat_sample")
    attack_rate = attack.get("sample_rate")
    if impact_sec is None or impact_sec < 0.0:
        return _reject("missing_attack_time", attack=dict(attack))
    impact_delta_sec = float(impact_sec) - float(grid_downbeat_sec)
    attack_sample_tolerance_sec = 1.0 / float(max(1, int(attack_rate or sample_rate)))
    if (
        impact_delta_sec < -0.008 - attack_sample_tolerance_sec
        or impact_delta_sec > max_after_sec + attack_sample_tolerance_sec
    ):
        return _reject(
            "bounded_attack_left_one_window",
            grid_downbeat_sec=float(grid_downbeat_sec),
            impact_sec=float(impact_sec),
            delta_ms=float(impact_delta_sec) * 1000.0,
        )

    attack_confidence = _finite_float(attack.get("attack_confidence")) or 0.0
    confidence = min(0.99, max(0.65, 0.60 + (0.35 * attack_confidence)))
    phase_translation_samples = None
    try:
        phase_translation_samples = int(impact_sample) - int(grid_sample)
    except (TypeError, ValueError):
        pass
    boom_body_crest = dict(_mapping(attack.get("boom_body_crest")))
    if boom_body_crest:
        # Renderer-facing contract: this point describes the first body's
        # energy shape. It is never an alternate ALS/drop marker.
        boom_body_crest["diagnostic_only"] = True
        boom_body_crest["moves_marker"] = False
        crest_sec = _finite_float(boom_body_crest.get("crest_time_sec"))
        crest_is_bounded = bool(
            crest_sec is not None
            and float(impact_sec) <= float(crest_sec)
            and float(crest_sec) <= float(grid_downbeat_sec) + (0.25 * float(beat_sec))
            and float(crest_sec) - float(impact_sec) <= 0.0605
        )
        if bool(boom_body_crest.get("ok")) and not crest_is_bounded:
            boom_body_crest["ok"] = False
            boom_body_crest["reason"] = "body_crest_outside_diagnostic_window"
        attack["boom_body_crest"] = dict(boom_body_crest)

    return {
        "ok": True,
        "accepted": True,
        "reason": "visual_grid_one_with_bounded_drum_attack",
        "source": "visual_first_bounded_attack_v1",
        "drop_sec": float(impact_sec),
        "confidence": float(confidence),
        "title_bpm": float(bpm_value),
        "grid_bpm": float(grid_bpm),
        "grid_source": str(beatgrid.get("source_role") or ""),
        "grid_bar_zero_sec": float(grid_zero),
        "grid_bar_index": int(nearest_bar_index),
        "grid_downbeat_sec": float(grid_downbeat_sec),
        "grid_downbeat_sample": int(grid_sample) if grid_sample is not None else None,
        "impact_sec": float(impact_sec),
        "impact_sample": int(impact_sample) if impact_sample is not None else None,
        "phase_translation_samples": phase_translation_samples,
        "sample_rate": int(attack_rate) if attack_rate is not None else None,
        "beat_position": 1,
        "boom_body_crest": boom_body_crest,
        "visual_marker_sec": float(marker),
        "visual_to_grid_delta_ms": float(visual_delta_sec) * 1000.0,
        "impact_delta_ms": float(impact_delta_sec) * 1000.0,
        "selected_by": selected_by,
        "clock_source": selected_clock_source,
        "evidence": {
            "visual_audit_pass": True,
            "boom_proof_pass": True,
            "gui_mask_pass": True,
            "marker_signal_present": True,
            "marker_relevant_mask": True,
            "independent_grid_one": True,
            "bounded_attack_refined": True,
            "bounded_body_crest_measured": bool(boom_body_crest.get("ok")),
        },
        "attack": attack,
    }
