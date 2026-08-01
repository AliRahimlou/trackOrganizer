from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_BOOM_PROFILE_PATH = Path(__file__).resolve().parents[1] / "models" / "boom_profile.json"
DEFAULT_PROOF_FRONT_EDGE_MAX_OFFSET_SEC = 0.080

DEFAULT_PROFILE: Dict[str, Any] = {
    "version": 1,
    "source": "default_dynamic_boom_profile",
    "thresholds": {
        "min_profile_score": 0.54,
        "min_darkness": 0.42,
        "min_post8_height": 0.40,
        "min_simultaneity": 0.36,
        "min_post_bass8": 0.20,
        "min_post_drum8": 0.48,
        "min_post_drum_cont8": 0.36,
        "min_contrast": 0.060,
        "min_sustain": 0.18,
        "max_one_distance_ms": 90.0,
        "max_front_edge_offset_sec": 0.450,
        "max_pre_front_edge_offset_sec": 0.020,
        "max_nearest_edge_sec": 1.250,
        "min_front_edge_correction_sec": 0.080,
        "max_front_edge_correction_sec": 12.0,
        "min_front_edge_correction_contrast": 0.160,
    },
}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def boom_proof_front_edge_freshness(
    proof: Mapping[str, Any],
    *,
    max_offset_sec: float = DEFAULT_PROOF_FRONT_EDGE_MAX_OFFSET_SEC,
) -> Dict[str, Any]:
    if not isinstance(proof, Mapping) or not bool(proof.get("passes")):
        return {"fresh": False, "reason": "boom_proof_not_passing"}
    nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
    if not nearest:
        return {"fresh": False, "reason": "missing_boom_nearest_edge"}
    offset = finite_float(nearest.get("offset_sec"), default=float("nan"))
    if not math.isfinite(offset):
        return {"fresh": False, "reason": "missing_boom_edge_offset"}
    abs_offset = abs(float(offset))
    max_offset = max(0.0, float(max_offset_sec))
    if abs_offset > max_offset:
        return {
            "fresh": False,
            "reason": f"boom_edge_offset {abs_offset:.3f}s above {max_offset:.3f}s",
            "offset_sec": float(offset),
            "max_offset_sec": float(max_offset),
            "edge_time": nearest.get("edge_time"),
            "marker_sec": proof.get("marker_sec"),
        }
    return {
        "fresh": True,
        "reason": "",
        "offset_sec": float(offset),
        "max_offset_sec": float(max_offset),
        "edge_time": nearest.get("edge_time"),
        "marker_sec": proof.get("marker_sec"),
    }


def clip01(value: Any) -> float:
    return max(0.0, min(1.0, finite_float(value)))


def load_boom_profile(path: str | Path | None = None) -> Dict[str, Any]:
    profile_path = Path(path).expanduser() if path else DEFAULT_BOOM_PROFILE_PATH
    if not profile_path.exists():
        return dict(DEFAULT_PROFILE)
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_PROFILE)
    if not isinstance(payload, Mapping):
        return dict(DEFAULT_PROFILE)
    merged = dict(DEFAULT_PROFILE)
    merged.update(dict(payload))
    thresholds = dict(DEFAULT_PROFILE.get("thresholds") or {})
    if isinstance(payload.get("thresholds"), Mapping):
        thresholds.update(dict(payload["thresholds"]))
    merged["thresholds"] = thresholds
    return merged


def candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "snapped_sec", "time_sec", "microaligned_time", "visual_raw_chunk_time"):
        if key not in candidate:
            continue
        value = finite_float(candidate.get(key), default=float("nan"))
        if math.isfinite(value):
            return float(value)
    return None


def visual_components(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    visual = candidate.get("visual_components")
    return visual if isinstance(visual, Mapping) else {}


def candidate_gui_front_edge_proven(candidate: Mapping[str, Any]) -> bool:
    visual = visual_components(candidate)
    gui_proof = candidate.get("gui_mask_proof") if isinstance(candidate.get("gui_mask_proof"), Mapping) else {}
    return bool(
        visual.get("gui_mask_contract_pass")
        or gui_proof.get("passes")
        or str(candidate.get("selected_by") or "") == "visual_gui_boom_front_edge_contract"
    )


def candidate_boom_metrics(candidate: Mapping[str, Any]) -> Dict[str, float]:
    visual = visual_components(candidate)
    score = finite_float(candidate.get("score"), finite_float(candidate.get("confidence_score")))
    post4 = clip01(visual.get("post4_height"))
    post8 = clip01(visual.get("post8_height"))
    darkness = max(
        score,
        post4,
        post8,
        clip01(visual.get("boom_section_darkness")),
        clip01(visual.get("bar_height")),
    )
    post_bass8 = clip01(visual.get("post_bass8"))
    post_drum8 = clip01(visual.get("post_drum8"))
    post_drum_cont4 = clip01(visual.get("post_drum_cont4"))
    post_drum_cont8 = clip01(visual.get("post_drum_cont8"))
    simultaneity = max(
        clip01(visual.get("boom_section_simultaneity")),
        clip01((0.40 * post_bass8) + (0.36 * post_drum8) + (0.24 * post_drum_cont8)),
    )
    contrast = max(
        finite_float(visual.get("local_reentry_gap")),
        finite_float(visual.get("jump4")),
        finite_float(visual.get("jump8")),
        0.0,
    )
    explicit_sustain = visual.get("sustained")
    if explicit_sustain is None:
        body_like = 1.0 if bool(visual.get("body_like")) else 0.0
        height_floor = min(post4, post8)
        height_stability = clip01(1.0 - (abs(post4 - post8) * 4.0))
        sustain = clip01(
            (0.82 * post_drum_cont8)
            + (0.10 * post_drum_cont4)
            + (0.06 * height_floor)
            + (0.04 * height_stability)
            + (0.02 * body_like)
        )
    else:
        sustain = clip01(explicit_sustain)
    pre4 = clip01(visual.get("pre4_height"))
    pre_space = max(clip01(visual.get("pre_space")), clip01(1.0 - pre4))
    phrase_prior = clip01(visual.get("phrase_prior"))
    one_distance_ms = abs(finite_float(visual.get("one_distance_ms"), 999999.0))
    body_score = clip01(
        (0.25 * darkness)
        + (0.18 * post8)
        + (0.18 * simultaneity)
        + (0.12 * post_bass8)
        + (0.12 * post_drum8)
        + (0.08 * post_drum_cont8)
        + (0.04 * sustain)
        + (0.03 * pre_space)
    )
    transition_score = clip01((0.46 * contrast) + (0.25 * pre_space) + (0.17 * phrase_prior) + (0.12 * sustain))
    profile_score = clip01((0.68 * body_score) + (0.24 * transition_score) + (0.08 * phrase_prior))
    return {
        "score": float(score),
        "profile_score": float(profile_score),
        "body_score": float(body_score),
        "transition_score": float(transition_score),
        "darkness": float(darkness),
        "post4_height": float(post4),
        "post8_height": float(post8),
        "post_bass8": float(post_bass8),
        "post_drum8": float(post_drum8),
        "post_drum_cont4": float(post_drum_cont4),
        "post_drum_cont8": float(post_drum_cont8),
        "simultaneity": float(simultaneity),
        "contrast": float(contrast),
        "sustain": float(sustain),
        "pre_space": float(pre_space),
        "phrase_prior": float(phrase_prior),
        "one_distance_ms": float(one_distance_ms),
    }


def candidate_has_gui_proven_boom_body(candidate: Mapping[str, Any]) -> bool:
    if not candidate_gui_front_edge_proven(candidate):
        return False
    visual = visual_components(candidate)
    metrics = candidate_boom_metrics(candidate)
    transition_like = bool(
        metrics["contrast"] >= 0.080
        or metrics["pre_space"] >= 0.500
        or bool(visual.get("local_reentry"))
        or bool(visual.get("phrase_body_shift"))
        or bool(visual.get("phrase_dropout_reentry"))
        or bool(visual.get("opening_body_start"))
    )
    return bool(
        metrics["profile_score"] >= 0.500
        and metrics["body_score"] >= 0.560
        and metrics["darkness"] >= 0.500
        and metrics["post8_height"] >= 0.480
        and metrics["post_drum8"] >= 0.850
        and metrics["post_drum_cont8"] >= 0.420
        and (
            metrics["post_bass8"] >= 0.180
            or metrics["simultaneity"] >= 0.540
        )
        and transition_like
    )


def _grid_phase_neutralized_candidate(candidate: Mapping[str, Any], marker: float) -> Dict[str, Any]:
    """Compatibility seam that preserves the independent grid unchanged."""
    out = dict(candidate)
    visual = dict(visual_components(out))
    visual["visual_body_proof_only"] = True
    visual["grid_phase_neutralization_rejected"] = True
    out["visual_components"] = visual
    return out


def score_candidate_against_profile(
    candidate: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    profile_data = profile if isinstance(profile, Mapping) else load_boom_profile()
    thresholds = profile_data.get("thresholds") if isinstance(profile_data.get("thresholds"), Mapping) else {}
    metrics = candidate_boom_metrics(candidate)
    reasons: List[str] = []
    deficits: Dict[str, float] = {}

    def require(metric: str, threshold_key: str) -> None:
        threshold = finite_float(thresholds.get(threshold_key), finite_float(DEFAULT_PROFILE["thresholds"][threshold_key]))
        value = float(metrics.get(metric, 0.0))
        if value < threshold:
            deficits[metric] = float(threshold - value)
            reasons.append(f"{metric} {metrics.get(metric, 0.0):.3f} below {threshold:.3f}")

    require("profile_score", "min_profile_score")
    require("darkness", "min_darkness")
    require("post8_height", "min_post8_height")
    require("simultaneity", "min_simultaneity")
    require("post_bass8", "min_post_bass8")
    require("post_drum8", "min_post_drum8")
    require("post_drum_cont8", "min_post_drum_cont8")
    require("sustain", "min_sustain")

    visual = visual_components(candidate)
    has_transition_exemption = bool(
        visual.get("opening_body_start")
        or visual.get("phrase_body_shift")
        or visual.get("phrase_dropout_reentry")
        or visual.get("local_reentry")
    )
    min_contrast = finite_float(thresholds.get("min_contrast"), finite_float(DEFAULT_PROFILE["thresholds"]["min_contrast"]))
    if metrics["contrast"] < min_contrast and not has_transition_exemption:
        reasons.append(f"contrast {metrics['contrast']:.3f} below {min_contrast:.3f}")

    max_one_distance = finite_float(
        thresholds.get("max_one_distance_ms"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_one_distance_ms"]),
    )
    if metrics["one_distance_ms"] > max_one_distance:
        reasons.append(f"one_distance_ms {metrics['one_distance_ms']:.1f} above {max_one_distance:.1f}")

    hard_one_distance_miss = any(str(reason).startswith("one_distance_ms") for reason in reasons)
    compensated_reasons: List[str] = []
    if reasons and not hard_one_distance_miss:
        strong_body = bool(
            metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) - 0.060
            and metrics["body_score"] >= 0.640
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"]) - 0.080
            and metrics["post8_height"]
            >= finite_float(thresholds.get("min_post8_height"), DEFAULT_PROFILE["thresholds"]["min_post8_height"]) - 0.080
            and metrics["simultaneity"]
            >= finite_float(thresholds.get("min_simultaneity"), DEFAULT_PROFILE["thresholds"]["min_simultaneity"]) - 0.070
            and metrics["post_bass8"]
            >= finite_float(thresholds.get("min_post_bass8"), DEFAULT_PROFILE["thresholds"]["min_post_bass8"]) - 0.060
            and metrics["post_drum8"]
            >= finite_float(thresholds.get("min_post_drum8"), DEFAULT_PROFILE["thresholds"]["min_post_drum8"]) - 0.090
            and metrics["post_drum_cont8"]
            >= finite_float(thresholds.get("min_post_drum_cont8"), DEFAULT_PROFILE["thresholds"]["min_post_drum_cont8"]) - 0.180
            and metrics["sustain"] >= 0.500
        )
        strong_transition = bool(
            metrics["contrast"]
            >= finite_float(thresholds.get("min_contrast"), DEFAULT_PROFILE["thresholds"]["min_contrast"]) - 0.080
            or metrics["pre_space"] >= 0.540
            or metrics["phrase_prior"] >= 0.860
            or has_transition_exemption
        )
        groove_body = bool(
            metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) - 0.075
            and metrics["body_score"] >= 0.600
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"]) - 0.120
            and metrics["post8_height"]
            >= finite_float(thresholds.get("min_post8_height"), DEFAULT_PROFILE["thresholds"]["min_post8_height"]) - 0.110
            and metrics["post_drum8"] >= 0.900
            and metrics["post_drum_cont8"] >= 0.520
            and metrics["sustain"] >= 0.500
            and (
                metrics["post_bass8"]
                >= finite_float(thresholds.get("min_post_bass8"), DEFAULT_PROFILE["thresholds"]["min_post_bass8"]) - 0.120
                or metrics["simultaneity"]
                >= finite_float(thresholds.get("min_simultaneity"), DEFAULT_PROFILE["thresholds"]["min_simultaneity"]) - 0.080
            )
            and (
                metrics["contrast"] >= 0.040
                or metrics["pre_space"] >= 0.400
                or metrics["phrase_prior"] >= 0.480
                or has_transition_exemption
            )
        )
        sparse_transition_body = bool(
            metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) - 0.035
            and metrics["body_score"] >= 0.560
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"]) - 0.130
            and metrics["post8_height"]
            >= finite_float(thresholds.get("min_post8_height"), DEFAULT_PROFILE["thresholds"]["min_post8_height"]) - 0.080
            and metrics["post_bass8"]
            >= finite_float(thresholds.get("min_post_bass8"), DEFAULT_PROFILE["thresholds"]["min_post_bass8"]) - 0.070
            and metrics["post_drum8"] >= 0.900
            and metrics["post_drum_cont8"] >= 0.280
            and metrics["simultaneity"]
            >= finite_float(thresholds.get("min_simultaneity"), DEFAULT_PROFILE["thresholds"]["min_simultaneity"]) - 0.100
            and (
                metrics["sustain"] >= 0.850
                or (
                    metrics["body_score"] >= 0.600
                    and metrics["contrast"] >= 0.280
                    and metrics["pre_space"] >= 0.500
                )
            )
            and (
                metrics["contrast"] >= 0.140
                or metrics["pre_space"] >= 0.520
                or metrics["phrase_prior"] >= 0.860
                or has_transition_exemption
            )
        )
        sparse_phrase_pulse_body = bool(
            metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) - 0.035
            and metrics["body_score"] >= 0.560
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"]) - 0.130
            and metrics["post8_height"] >= 0.500
            and metrics["post_bass8"] >= 0.380
            and metrics["post_drum8"] >= 0.950
            and metrics["post_drum_cont8"] >= 0.180
            and metrics["simultaneity"]
            >= finite_float(thresholds.get("min_simultaneity"), DEFAULT_PROFILE["thresholds"]["min_simultaneity"]) - 0.060
            and metrics["sustain"] >= 0.850
            and metrics["pre_space"] >= 0.520
            and metrics["phrase_prior"] >= 0.820
        )
        continuous_bass_body = bool(
            metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) + 0.080
            and metrics["body_score"] >= 0.700
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"])
            and metrics["post8_height"]
            >= finite_float(thresholds.get("min_post8_height"), DEFAULT_PROFILE["thresholds"]["min_post8_height"])
            and metrics["post_bass8"]
            >= max(0.500, finite_float(thresholds.get("min_post_bass8"), DEFAULT_PROFILE["thresholds"]["min_post_bass8"]) + 0.180)
            and metrics["post_drum8"] >= 0.820
            and metrics["post_drum_cont4"] >= 0.820
            and metrics["post_drum_cont8"]
            >= max(0.860, finite_float(thresholds.get("min_post_drum_cont8"), DEFAULT_PROFILE["thresholds"]["min_post_drum_cont8"]) + 0.160)
            and metrics["simultaneity"]
            >= max(0.700, finite_float(thresholds.get("min_simultaneity"), DEFAULT_PROFILE["thresholds"]["min_simultaneity"]) + 0.060)
            and metrics["sustain"] >= 0.850
            and metrics["contrast"]
            >= finite_float(thresholds.get("min_contrast"), DEFAULT_PROFILE["thresholds"]["min_contrast"])
            and metrics["pre_space"] >= 0.350
            and (metrics["phrase_prior"] >= 0.800 or has_transition_exemption)
        )
        sparse_gui_transition_body = bool(
            candidate_gui_front_edge_proven(candidate)
            and metrics["profile_score"]
            >= finite_float(thresholds.get("min_profile_score"), DEFAULT_PROFILE["thresholds"]["min_profile_score"]) - 0.060
            and metrics["body_score"] >= 0.560
            and metrics["darkness"]
            >= finite_float(thresholds.get("min_darkness"), DEFAULT_PROFILE["thresholds"]["min_darkness"]) - 0.180
            and metrics["post8_height"]
            >= finite_float(thresholds.get("min_post8_height"), DEFAULT_PROFILE["thresholds"]["min_post8_height"]) - 0.120
            and metrics["post_drum8"] >= 0.920
            and metrics["post_drum_cont8"] >= 0.400
            and (
                metrics["post_bass8"] >= 0.150
                or metrics["simultaneity"] >= 0.495
            )
            and metrics["sustain"] >= 0.850
            and (
                metrics["contrast"] >= 0.240
                or metrics["pre_space"] >= 0.780
                or has_transition_exemption
            )
        )
        gross_misses = [
            metric
            for metric, deficit in deficits.items()
            if (
                (metric == "profile_score" and deficit > 0.080)
                or (metric == "darkness" and deficit > 0.110)
                or (metric == "post8_height" and deficit > 0.120)
                or (metric == "simultaneity" and deficit > 0.100)
                or (metric == "post_bass8" and deficit > 0.090)
                or (metric == "post_drum8" and deficit > 0.120)
                or (metric == "post_drum_cont8" and deficit > 0.220)
                or (metric == "sustain" and metrics["sustain"] < 0.500)
            )
        ]
        if (
            (strong_body and strong_transition and not gross_misses)
            or groove_body
            or sparse_transition_body
            or sparse_phrase_pulse_body
            or continuous_bass_body
            or sparse_gui_transition_body
        ):
            compensated_reasons = list(reasons)
            reasons = []

    return {
        "passes_profile": not reasons,
        "profile_score": float(metrics["profile_score"]),
        "metrics": metrics,
        "reasons": reasons,
        "compensated_reasons": compensated_reasons,
        "profile_source": str(profile_data.get("source") or ""),
    }


def _apply_track_relative_boom_profile(
    row: Dict[str, Any],
    scored: Sequence[Mapping[str, Any]],
) -> None:
    profile = row.get("profile") if isinstance(row.get("profile"), Mapping) else {}
    if bool(profile.get("passes_profile")):
        return
    reasons = [str(reason) for reason in profile.get("reasons") or []]
    if any(reason.startswith("one_distance_ms") for reason in reasons):
        return

    profiles = [item.get("profile") for item in scored if isinstance(item.get("profile"), Mapping)]
    if any(bool(profile_row.get("passes_profile")) for profile_row in profiles):
        return

    metrics = profile.get("metrics") if isinstance(profile.get("metrics"), Mapping) else {}
    if not metrics:
        return
    all_metrics = [
        profile_row.get("metrics")
        for profile_row in profiles
        if isinstance(profile_row.get("metrics"), Mapping)
    ]
    best_score = max((finite_float(item.get("candidate_score"), 0.0) for item in scored), default=0.0)
    best_darkness = max((finite_float(m.get("darkness"), 0.0) for m in all_metrics), default=0.0)
    best_post8 = max((finite_float(m.get("post8_height"), 0.0) for m in all_metrics), default=0.0)
    best_bass = max((finite_float(m.get("post_bass8"), 0.0) for m in all_metrics), default=0.0)
    best_drum_cont = max((finite_float(m.get("post_drum_cont8"), 0.0) for m in all_metrics), default=0.0)

    candidate_score = finite_float(row.get("candidate_score"), 0.0)
    profile_score = finite_float(profile.get("profile_score"), finite_float(metrics.get("profile_score"), 0.0))
    darkness = finite_float(metrics.get("darkness"), 0.0)
    post8 = finite_float(metrics.get("post8_height"), 0.0)
    bass = finite_float(metrics.get("post_bass8"), 0.0)
    drum = finite_float(metrics.get("post_drum8"), 0.0)
    drum_cont = finite_float(metrics.get("post_drum_cont8"), 0.0)
    simultaneity = finite_float(metrics.get("simultaneity"), 0.0)
    sustain = finite_float(metrics.get("sustain"), 0.0)
    contrast = finite_float(metrics.get("contrast"), 0.0)
    pre_space = finite_float(metrics.get("pre_space"), 0.0)

    relative_pass = bool(
        candidate_score >= max(0.500, best_score * 0.78)
        and profile_score >= 0.460
        and darkness >= max(0.520, best_darkness * 0.82)
        and post8 >= max(0.480, best_post8 * 0.82)
        and bass >= max(0.180, best_bass * 0.68)
        and drum >= 0.900
        and drum_cont >= max(0.450, best_drum_cont * 0.72)
        and simultaneity >= 0.500
        and sustain >= 0.600
        and (contrast >= 0.030 or pre_space >= 0.400)
    )
    if not relative_pass:
        return

    updated = dict(profile)
    updated["passes_profile"] = True
    updated["relative_boom_profile"] = True
    updated["compensated_reasons"] = [*list(updated.get("compensated_reasons") or []), *reasons]
    updated["reasons"] = []
    row["profile"] = updated


def _bar_sec_from_candidate(candidate: Mapping[str, Any], beatgrid: Optional[Mapping[str, Any]] = None) -> float:
    clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
    for source in (clock, beatgrid if isinstance(beatgrid, Mapping) else {}):
        bar_sec = finite_float(source.get("bar_sec"), 0.0)
        if bar_sec > 0.0:
            return float(bar_sec)
        beat_sec = finite_float(source.get("beat_sec"), 0.0)
        if beat_sec > 0.0:
            return float(beat_sec * 4.0)
        bpm = finite_float(source.get("bpm"), 0.0)
        if bpm > 0.0:
            return float(240.0 / bpm)
    return 2.0


def _one_distance_ms_from_marker(
    marker_sec: float,
    candidate: Mapping[str, Any],
    beatgrid: Optional[Mapping[str, Any]] = None,
) -> float:
    bar_sec = _bar_sec_from_candidate(candidate, beatgrid)
    if bar_sec <= 0.0:
        return 999999.0
    clock = candidate.get("bpm_clock") if isinstance(candidate.get("bpm_clock"), Mapping) else {}
    zero = finite_float(
        (beatgrid or {}).get("bar_zero_sec") if isinstance(beatgrid, Mapping) else None,
        finite_float(clock.get("clock_zero_sec"), 0.0),
    )
    phase = (float(marker_sec) - float(zero)) % float(bar_sec)
    distance = min(phase, float(bar_sec) - phase)
    return abs(float(distance) * 1000.0)


def _section_end_time(candidate: Mapping[str, Any], edge_time: float, beatgrid: Optional[Mapping[str, Any]] = None) -> float:
    visual = visual_components(candidate)
    bar_sec = _bar_sec_from_candidate(candidate, beatgrid)
    edge_bar = int(finite_float(visual.get("feature_bar"), 0))
    end_bar = int(finite_float(visual.get("boom_section_end_bar"), edge_bar))
    if edge_bar > 0 and end_bar >= edge_bar:
        return float(edge_time + ((end_bar - edge_bar + 1) * bar_sec))
    return float(edge_time + (4.0 * bar_sec))


def marker_boom_proof(
    marker_sec: Any,
    boom_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_candidate: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    beatgrid: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    marker = finite_float(marker_sec, default=float("nan"))
    if not math.isfinite(marker):
        return {"passes": False, "reasons": ["missing_marker_time"], "candidate_count": len(boom_candidates or [])}

    profile_data = profile if isinstance(profile, Mapping) else load_boom_profile()
    thresholds = profile_data.get("thresholds") if isinstance(profile_data.get("thresholds"), Mapping) else {}
    candidates = [dict(row) for row in boom_candidates or [] if isinstance(row, Mapping)]
    ignored_earlier_edges: set[float] = set()
    if isinstance(selected_candidate, Mapping):
        ignored_earlier_edges = {
            round(finite_float(value, default=float("nan")), 3)
            for value in selected_candidate.get("boom_proof_ignore_earlier_edges", []) or []
            if math.isfinite(finite_float(value, default=float("nan")))
        }
    if isinstance(selected_candidate, Mapping):
        selected_time = candidate_time(selected_candidate)
        selected_visual = visual_components(selected_candidate)
        selected_clock = selected_candidate.get("bpm_clock") if isinstance(selected_candidate.get("bpm_clock"), Mapping) else {}
        selected_one_distance = abs(
            finite_float(
                selected_visual.get("one_distance_ms", selected_clock.get("one_distance_ms")),
                999999.0,
            )
        )
        authoritative_zero = (
            finite_float(beatgrid.get("bar_zero_sec"), float("nan"))
            if isinstance(beatgrid, Mapping)
            else float("nan")
        )
        if math.isfinite(authoritative_zero):
            selected_one_distance = max(
                float(selected_one_distance),
                _one_distance_ms_from_marker(float(marker), selected_candidate, beatgrid),
            )
        max_selected_one_distance = finite_float(
            thresholds.get("max_one_distance_ms"),
            finite_float(DEFAULT_PROFILE["thresholds"]["max_one_distance_ms"]),
        )
        selected_gui_body = candidate_has_gui_proven_boom_body(selected_candidate)
        try:
            selected_profile = score_candidate_against_profile(selected_candidate, profile_data)
        except Exception:
            selected_profile = {}
        selected_profile_passes = bool(selected_profile.get("passes_profile"))
        if (
            selected_time is not None
            and abs(float(selected_time) - float(marker)) <= 0.250
            and selected_one_distance <= max_selected_one_distance
            and selected_visual
            and (
                selected_gui_body
                or selected_profile_passes
                or not any(
                    candidate_time(row) is not None
                    and abs(float(candidate_time(row)) - float(selected_time)) <= 0.050
                    for row in candidates
                )
            )
        ):
            selected_copy = dict(selected_candidate)
            selected_copy["__selected_marker_candidate"] = True
            candidates.append(selected_copy)
    if not candidates and isinstance(selected_candidate, Mapping):
        visual = visual_components(selected_candidate)
        if bool(visual.get("boom_body_section")):
            selected_copy = dict(selected_candidate)
            selected_copy["__selected_marker_candidate"] = True
            candidates = [selected_copy]
    if not candidates:
        return {
            "passes": False,
            "reasons": ["no_boom_body_section_candidate"],
            "marker_sec": float(marker),
            "candidate_count": 0,
        }

    max_nearest = finite_float(
        thresholds.get("max_nearest_edge_sec"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_nearest_edge_sec"]),
    )
    max_front_edge = finite_float(
        thresholds.get("max_front_edge_offset_sec"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_front_edge_offset_sec"]),
    )
    max_pre_front_edge = finite_float(
        thresholds.get("max_pre_front_edge_offset_sec"),
        min(0.020, max_front_edge),
    )

    scored: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        edge_time = candidate_time(candidate)
        if edge_time is None:
            continue
        profile_score = score_candidate_against_profile(candidate, profile_data)
        section_end = _section_end_time(candidate, float(edge_time), beatgrid)
        visual = visual_components(candidate)
        score = finite_float(candidate.get("score"), finite_float(candidate.get("confidence_score")))
        scored.append(
            {
                "index": idx,
                "rank": int(finite_float(candidate.get("rank"), idx + 1)),
                "edge_time": float(edge_time),
                "section_end_time": float(section_end),
                "offset_sec": float(marker - edge_time),
                "abs_offset_sec": float(abs(marker - edge_time)),
                "contains_marker": bool(edge_time - 0.100 <= marker <= section_end + 0.100),
                "candidate_score": float(score),
                "clock_bar": int(finite_float(visual.get("clock_bar"), 0)),
                "feature_bar": int(finite_float(visual.get("feature_bar"), 0)),
                "profile": profile_score,
                "candidate": candidate,
                "selected_marker_candidate": bool(candidate.get("__selected_marker_candidate")),
            }
        )
    if not scored:
        return {
            "passes": False,
            "reasons": ["boom_candidates_missing_times"],
            "marker_sec": float(marker),
            "candidate_count": len(candidates),
        }

    for row in scored:
        _apply_track_relative_boom_profile(row, scored)
        candidate = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
        if (
            bool(row.get("selected_marker_candidate"))
            and candidate_has_gui_proven_boom_body(candidate)
            and isinstance(row.get("profile"), Mapping)
            and not bool(row["profile"].get("passes_profile"))
        ):
            updated_profile = dict(row["profile"])
            updated_profile["passes_profile"] = True
            updated_profile["gui_front_edge_profile_relief"] = True
            updated_profile["compensated_reasons"] = [
                *list(updated_profile.get("compensated_reasons") or []),
                *[str(reason) for reason in updated_profile.get("reasons") or []],
            ]
            updated_profile["reasons"] = []
            row["profile"] = updated_profile

    nearest = min(scored, key=lambda row: (row["abs_offset_sec"], 0 if row.get("selected_marker_candidate") else 1))
    reasons: List[str] = []
    if nearest["abs_offset_sec"] > max_nearest and not nearest["contains_marker"]:
        reasons.append(f"nearest_boom_edge {nearest['abs_offset_sec']:.3f}s away")
    if nearest["offset_sec"] > max_front_edge:
        reasons.append(f"marker {nearest['offset_sec']:.3f}s after boom front edge")
    if nearest["offset_sec"] < -max_pre_front_edge:
        reasons.append(f"marker {abs(nearest['offset_sec']):.3f}s before boom front edge")
    if not bool(nearest["profile"].get("passes_profile")):
        reasons.extend(str(reason) for reason in nearest["profile"].get("reasons") or [])
    if isinstance(beatgrid, Mapping):
        authoritative_zero = finite_float(beatgrid.get("bar_zero_sec"), default=float("nan"))
        if math.isfinite(authoritative_zero):
            distance_candidate = (
                selected_candidate
                if isinstance(selected_candidate, Mapping)
                else nearest.get("candidate")
                if isinstance(nearest.get("candidate"), Mapping)
                else {}
            )
            independent_one_distance = _one_distance_ms_from_marker(
                float(marker),
                distance_candidate,
                beatgrid,
            )
            independent_max_one = finite_float(
                thresholds.get("max_one_distance_ms"),
                finite_float(DEFAULT_PROFILE["thresholds"]["max_one_distance_ms"]),
            )
            if independent_one_distance > independent_max_one:
                reasons.append(
                    f"one_distance_ms {independent_one_distance:.1f} above {independent_max_one:.1f} "
                    "on authoritative beatgrid"
                )

    def earlier_row_is_same_coarse_section(row: Mapping[str, Any]) -> bool:
        return bool(
            nearest.get("selected_marker_candidate")
            and row.get("contains_marker")
            and finite_float(row.get("edge_time"), 0.0) < marker
            and finite_float(row.get("section_end_time"), 0.0) >= marker - 0.100
        )

    def row_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
        row_profile = row.get("profile") if isinstance(row.get("profile"), Mapping) else {}
        metrics = row_profile.get("metrics") if isinstance(row_profile.get("metrics"), Mapping) else {}
        return metrics if isinstance(metrics, Mapping) else {}

    reference_rows = [
        row
        for row in scored
        if row.get("contains_marker")
        or row.get("selected_marker_candidate")
        or row["abs_offset_sec"] <= max_nearest
    ]
    if not reference_rows:
        reference_rows = [nearest]
    reference_score = max(float(row.get("candidate_score", 0.0) or 0.0) for row in reference_rows)
    reference_bass = max(float(row_metrics(row).get("post_bass8", 0.0) or 0.0) for row in reference_rows)
    reference_darkness = max(float(row_metrics(row).get("darkness", 0.0) or 0.0) for row in reference_rows)
    reference_body = max(float(row_metrics(row).get("body_score", 0.0) or 0.0) for row in reference_rows)
    reference_drum_cont = max(float(row_metrics(row).get("post_drum_cont8", 0.0) or 0.0) for row in reference_rows)
    reference_phrase = max(float(row_metrics(row).get("phrase_prior", 0.0) or 0.0) for row in reference_rows)
    earlier_viable: List[Dict[str, Any]] = []
    for row in scored:
        if row["edge_time"] >= nearest["edge_time"] - 0.250:
            continue
        if round(float(row["edge_time"]), 3) in ignored_earlier_edges:
            continue
        if earlier_row_is_same_coarse_section(row):
            continue
        if row["edge_time"] < 8.0 and row["clock_bar"] < 5:
            continue
        if not bool(row["profile"].get("passes_profile")):
            continue
        if reference_score > 0.0 and row["candidate_score"] < reference_score * 0.94:
            continue
        metrics = row_metrics(row)
        row_bass = float(metrics.get("post_bass8", 0.0) or 0.0)
        row_darkness = float(metrics.get("darkness", 0.0) or 0.0)
        row_body = float(metrics.get("body_score", 0.0) or 0.0)
        row_phrase = float(metrics.get("phrase_prior", 0.0) or 0.0)
        row_contrast = float(metrics.get("contrast", 0.0) or 0.0)
        row_drum_cont = float(metrics.get("post_drum_cont8", 0.0) or 0.0)
        row_visual = visual_components(row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {})
        bar_sec = _bar_sec_from_candidate(row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}, beatgrid)
        row_is_file_start_opening = bool(
            row["edge_time"] <= max(8.0, 4.0 * float(bar_sec))
            and (
                row["clock_bar"] <= 5
                or bool(row_visual.get("opening_drop_profile"))
                or bool(row_visual.get("opening_body_start"))
            )
        )
        opening_before_stronger_later_body = bool(
            row_is_file_start_opening
            and marker - row["edge_time"] > max(16.0, 8.0 * float(bar_sec))
            and reference_body >= row_body + 0.030
            and reference_drum_cont >= row_drum_cont + 0.200
            and reference_darkness >= row_darkness - 0.040
        )
        if opening_before_stronger_later_body:
            continue
        row_is_prephrase_intro = bool(
            not row_is_file_start_opening
            and row["clock_bar"] <= 8
            and row["edge_time"] <= max(16.0, 8.0 * float(bar_sec))
            and row_phrase < 0.600
            and reference_phrase >= 0.780
            and not bool(row_visual.get("opening_drop_profile"))
        )
        if row_is_prephrase_intro:
            continue
        first_drop_has_strong_transition = bool(
            (row_phrase >= 0.860 or bool(row_visual.get("phrase_body_shift")))
            and (
                row_contrast >= 0.300
                or bool(row_visual.get("local_reentry"))
                or bool(row_visual.get("phrase_dropout_reentry"))
            )
        )
        bass_tolerance = 0.140 if first_drop_has_strong_transition else 0.080
        darkness_tolerance = 0.125 if first_drop_has_strong_transition else 0.100
        body_tolerance = 0.105 if first_drop_has_strong_transition else 0.080
        if reference_bass >= 0.260 and row_bass < reference_bass - bass_tolerance:
            continue
        if reference_darkness >= 0.500 and row_darkness < reference_darkness - darkness_tolerance:
            continue
        if reference_body >= 0.600 and row_body < reference_body - body_tolerance:
            continue
        earlier_viable.append(row)
    if earlier_viable:
        earliest = min(earlier_viable, key=lambda row: row["edge_time"])
        reasons.append(
            f"earlier_dominant_boom_available at {earliest['edge_time']:.3f}s b{earliest['clock_bar']}"
        )

    return {
        "passes": not reasons,
        "marker_sec": float(marker),
        "candidate_count": len(candidates),
        "nearest": {
            key: nearest[key]
            for key in (
                "index",
                "rank",
                "edge_time",
                "section_end_time",
                "offset_sec",
                "abs_offset_sec",
                "contains_marker",
                "candidate_score",
                "clock_bar",
                "feature_bar",
                "selected_marker_candidate",
            )
        },
        "nearest_profile": nearest["profile"],
        "reasons": reasons,
        "profile_source": str(profile_data.get("source") or ""),
    }


def boom_front_edge_correction(
    marker_sec: Any,
    boom_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_candidate: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    beatgrid: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a proven Boom front-edge replacement for markers inside/near a body.

    This is intentionally conservative: the Boom section must pass the learned
    profile, the current marker must be measurably early/late, and huge jumps are
    left as validator holds instead of silently changing sections.
    """

    candidates = [dict(row) for row in boom_candidates or [] if isinstance(row, Mapping)]
    proof = marker_boom_proof(
        marker_sec,
        candidates,
        selected_candidate=None,
        profile=profile,
        beatgrid=beatgrid,
    )
    nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
    nearest_profile = proof.get("nearest_profile") if isinstance(proof.get("nearest_profile"), Mapping) else {}
    if not nearest or not bool(nearest_profile.get("passes_profile")):
        return None
    if any("earlier_dominant_boom_available" in str(reason) for reason in proof.get("reasons") or []):
        return None

    index = int(finite_float(nearest.get("index"), -1))
    if index < 0 or index >= len(candidates):
        return None
    candidate = dict(candidates[index])
    visual = visual_components(candidate)
    selected_visual = visual_components(selected_candidate) if isinstance(selected_candidate, Mapping) else {}
    metrics = nearest_profile.get("metrics") if isinstance(nearest_profile.get("metrics"), Mapping) else {}
    edge_time = finite_float(nearest.get("edge_time"), float("nan"))
    offset = finite_float(nearest.get("offset_sec"), 0.0)
    if not math.isfinite(edge_time):
        return None

    profile_data = profile if isinstance(profile, Mapping) else load_boom_profile()
    thresholds = profile_data.get("thresholds") if isinstance(profile_data.get("thresholds"), Mapping) else {}
    min_offset = finite_float(
        thresholds.get("min_front_edge_correction_sec"),
        finite_float(DEFAULT_PROFILE["thresholds"]["min_front_edge_correction_sec"]),
    )
    max_pre_front_edge = finite_float(
        thresholds.get("max_pre_front_edge_offset_sec"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_pre_front_edge_offset_sec"]),
    )
    max_offset = finite_float(
        thresholds.get("max_front_edge_correction_sec"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_front_edge_correction_sec"]),
    )
    min_correction_contrast = finite_float(
        thresholds.get("min_front_edge_correction_contrast"),
        finite_float(thresholds.get("min_contrast"), DEFAULT_PROFILE["thresholds"]["min_front_edge_correction_contrast"]),
    )
    bar_sec = _bar_sec_from_candidate(candidate, beatgrid)
    dynamic_max = max(1.250, min(max_offset, max(4.0 * bar_sec, 2.0)))
    contains_marker = bool(nearest.get("contains_marker"))
    section_span = max(0.0, finite_float(nearest.get("section_end_time"), edge_time) - float(edge_time))
    long_sustained_section = bool(
        contains_marker
        and offset > 0.0
        and section_span >= max(16.0, 8.0 * bar_sec)
        and float(edge_time) >= 8.0
    )
    if long_sustained_section:
        dynamic_max = max(dynamic_max, min(section_span, max(64.0, 12.0 * bar_sec)))
    elif contains_marker and offset > 0.0 and bar_sec >= 2.0:
        dynamic_max = max(
            dynamic_max,
            min(max(8.0 * bar_sec, dynamic_max), max(section_span, dynamic_max), 30.0),
        )
    elif bar_sec < 2.50:
        fast_front_edge_max = finite_float(
            thresholds.get("max_front_edge_offset_sec"),
            finite_float(DEFAULT_PROFILE["thresholds"]["max_front_edge_offset_sec"]),
        )
        if offset > 0.0 and contains_marker:
            dynamic_max = max(
                fast_front_edge_max,
                min(dynamic_max, max(1.35 * bar_sec, min(section_span, 2.25 * bar_sec))),
            )
        else:
            dynamic_max = max(0.250, min(dynamic_max, fast_front_edge_max))
    abs_offset = abs(float(offset))
    dynamic_min = max(min_offset, 0.60 * bar_sec)
    if offset < 0.0:
        dynamic_min = min(dynamic_min, max(0.0, max_pre_front_edge))
    selected_clock = selected_candidate.get("bpm_clock") if isinstance(selected_candidate, Mapping) and isinstance(selected_candidate.get("bpm_clock"), Mapping) else {}
    selected_one_distance = abs(
        finite_float(
            selected_visual.get("one_distance_ms", selected_clock.get("one_distance_ms")),
            0.0,
        )
    )
    selected_one_distance = max(
        selected_one_distance,
        _one_distance_ms_from_marker(finite_float(marker_sec), candidate, beatgrid),
    )
    max_one_distance = finite_float(
        thresholds.get("max_one_distance_ms"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_one_distance_ms"]),
    )
    selected_profile = (
        score_candidate_against_profile(selected_candidate, profile_data)
        if isinstance(selected_candidate, Mapping)
        else {}
    )
    selected_metrics = (
        selected_profile.get("metrics")
        if isinstance(selected_profile.get("metrics"), Mapping)
        else {}
    )
    selected_visible_body = bool(
        selected_visual
        and (
            bool(selected_profile.get("passes_profile"))
            or (
                finite_float(selected_metrics.get("profile_score"), 0.0) >= 0.560
                and finite_float(selected_metrics.get("body_score"), 0.0) >= 0.600
                and finite_float(selected_metrics.get("darkness"), 0.0) >= 0.540
                and finite_float(selected_metrics.get("post8_height"), 0.0) >= 0.500
                and finite_float(selected_metrics.get("post_drum8"), 0.0) >= 0.860
                and finite_float(selected_metrics.get("post_drum_cont8"), 0.0) >= 0.520
                and (
                    finite_float(selected_metrics.get("post_bass8"), 0.0) >= 0.180
                    or finite_float(selected_metrics.get("simultaneity"), 0.0) >= 0.560
                )
                and (
                    finite_float(selected_metrics.get("contrast"), 0.0) >= 0.120
                    or clip01(selected_visual.get("phrase_prior")) >= 0.780
                    or bool(selected_visual.get("local_reentry"))
                    or bool(selected_visual.get("phrase_body_shift"))
                    or bool(selected_visual.get("phrase_dropout_reentry"))
                    or bool(selected_visual.get("opening_body_start"))
                )
            )
        )
    )
    high_dark_body_target = bool(
        offset > 0.0
        and contains_marker
        and finite_float(nearest_profile.get("profile_score"), 0.0) >= 0.580
        and finite_float(metrics.get("body_score"), 0.0) >= 0.680
        and finite_float(metrics.get("darkness"), 0.0) >= 0.780
        and finite_float(metrics.get("post_bass8"), 0.0) >= 0.340
        and finite_float(metrics.get("post_drum8"), 0.0) >= 0.940
        and finite_float(metrics.get("post_drum_cont8"), 0.0) >= 0.520
        and (
            finite_float(metrics.get("contrast"), 0.0) >= 0.190
            or bool(visual.get("local_reentry"))
            or bool(visual.get("opening_body_start"))
            or bool(visual.get("phrase_body_shift"))
        )
    )
    if high_dark_body_target:
        dynamic_max = max(dynamic_max, min(max(section_span, 0.0), max(8.0, 4.0 * bar_sec)))
    target_passes_profile = bool(nearest_profile.get("passes_profile"))
    strict_before_body_target = bool(
        offset < 0.0
        and not (selected_visible_body and selected_one_distance <= max_one_distance)
        and float(edge_time) >= 8.0
        and finite_float(nearest_profile.get("profile_score"), 0.0) >= 0.600
        and finite_float(metrics.get("body_score"), 0.0) >= 0.680
        and finite_float(metrics.get("darkness"), 0.0) >= 0.680
        and finite_float(metrics.get("simultaneity"), 0.0) >= 0.680
        and finite_float(metrics.get("post_bass8"), 0.0) >= 0.340
        and finite_float(metrics.get("post_drum8"), 0.0) >= 0.940
        and finite_float(metrics.get("post_drum_cont8"), 0.0) >= 0.640
        and (
            finite_float(metrics.get("contrast"), 0.0) >= 0.160
            or bool(visual.get("local_reentry"))
            or bool(visual.get("phrase_body_shift"))
            or bool(visual.get("phrase_dropout_reentry"))
        )
    )
    compensated_before_body_target = bool(
        offset < 0.0
        and selected_one_distance > max_one_distance
        and target_passes_profile
        and float(edge_time) >= 8.0
        and finite_float(nearest_profile.get("profile_score"), 0.0) >= 0.570
        and finite_float(metrics.get("body_score"), 0.0) >= 0.650
        and finite_float(metrics.get("darkness"), 0.0) >= 0.640
        and finite_float(metrics.get("simultaneity"), 0.0) >= 0.640
        and finite_float(metrics.get("post_bass8"), 0.0) >= 0.300
        and finite_float(metrics.get("post_drum8"), 0.0) >= 0.940
        and finite_float(metrics.get("post_drum_cont8"), 0.0) >= 0.680
        and (
            finite_float(metrics.get("contrast"), 0.0) >= 0.280
            or bool(visual.get("phrase_body_shift"))
            or bool(visual.get("phrase_dropout_reentry"))
        )
    )
    strong_before_body_target = bool(strict_before_body_target or compensated_before_body_target)
    if strong_before_body_target:
        dynamic_max = max(dynamic_max, min(10.0, max(4.50 * bar_sec, 6.0)))
    if offset < 0.0 and selected_one_distance > max_one_distance:
        dynamic_min = min(dynamic_min, max(min_offset, 0.20 * bar_sec))
    if offset > 0.0 and contains_marker and bar_sec >= 2.0 and selected_one_distance > max_one_distance:
        dynamic_min = min(dynamic_min, max(min_offset, 0.20 * bar_sec))
    if offset > 0.0 and contains_marker and abs_offset >= max(0.80, 0.45 * bar_sec):
        dynamic_min = min(dynamic_min, max(min_offset, 0.20 * bar_sec))
    if abs_offset < dynamic_min or abs_offset > dynamic_max:
        return None

    if offset > 0.0 and not contains_marker:
        return None
    if offset < 0.0:
        max_nearest = finite_float(
            thresholds.get("max_nearest_edge_sec"),
            finite_float(DEFAULT_PROFILE["thresholds"]["max_nearest_edge_sec"]),
        )
        before_body_max = max(0.250, min(max_nearest, 2.0 * bar_sec))
        if strong_before_body_target:
            before_body_max = max(before_body_max, min(10.0, max(4.50 * bar_sec, 6.0)))
        if bar_sec >= 2.50:
            before_body_max = max(before_body_max, min(dynamic_max, 2.10 * bar_sec))
        if abs_offset > before_body_max:
            return None
    contrast = finite_float(metrics.get("contrast"), 0.0)
    candidate_phrase = clip01(visual.get("phrase_prior"))
    selected_phrase = clip01(selected_visual.get("phrase_prior"))
    candidate_clock_bar = int(finite_float(visual.get("clock_bar"), finite_float(candidate.get("clock_bar"), 0.0)))
    selected_clock_bar = int(finite_float(selected_visual.get("clock_bar"), finite_float(selected_candidate.get("clock_bar") if isinstance(selected_candidate, Mapping) else 0.0, 0.0)))
    section_start_bar = int(finite_float(visual.get("boom_section_start_bar"), candidate_clock_bar))
    section_end_bar = int(finite_float(visual.get("boom_section_end_bar"), candidate_clock_bar))
    selected_inside_boom_section = bool(
        candidate_clock_bar > 0
        and selected_clock_bar > 0
        and (
            (section_start_bar > 0 and section_end_bar >= section_start_bar and section_start_bar <= selected_clock_bar <= section_end_bar + 1)
            or selected_clock_bar <= candidate_clock_bar + 8
        )
    )
    same_section_front_edge = bool(
        offset > 0.0
        and contains_marker
        and bar_sec >= 1.45
        and abs_offset >= max(0.80, 0.45 * bar_sec)
        and candidate_clock_bar > 0
        and selected_clock_bar > 0
        and selected_inside_boom_section
        and contrast >= min_correction_contrast
        and metrics.get("body_score", 0.0) >= 0.660
        and (
            bool(visual.get("opening_body_start"))
            or bool(visual.get("local_reentry"))
            or bool(visual.get("phrase_dropout_reentry"))
        )
    )
    sustained_section_front_edge = bool(
        offset > 0.0
        and contains_marker
        and selected_inside_boom_section
        and float(edge_time) >= 8.0
        and section_span >= max(16.0, 8.0 * bar_sec)
        and abs_offset >= max(4.0, 2.0 * bar_sec)
        and finite_float(nearest_profile.get("profile_score"), 0.0) >= 0.600
        and finite_float(metrics.get("body_score"), 0.0) >= 0.700
        and finite_float(metrics.get("darkness"), 0.0) >= 0.700
        and finite_float(metrics.get("simultaneity"), 0.0) >= 0.700
        and finite_float(metrics.get("post_bass8"), 0.0) >= 0.400
        and finite_float(metrics.get("post_drum_cont8"), 0.0) >= 0.720
    )
    high_dark_body_front_edge = bool(
        offset > 0.0
        and contains_marker
        and selected_inside_boom_section
        and abs_offset >= max(1.0, 0.55 * bar_sec)
        and abs_offset <= min(max(section_span, 0.0), max(8.0, 4.0 * bar_sec))
        and finite_float(nearest_profile.get("profile_score"), 0.0) >= 0.580
        and finite_float(metrics.get("body_score"), 0.0) >= 0.680
        and finite_float(metrics.get("darkness"), 0.0) >= 0.780
        and finite_float(metrics.get("post_bass8"), 0.0) >= 0.340
        and finite_float(metrics.get("post_drum8"), 0.0) >= 0.940
        and finite_float(metrics.get("post_drum_cont8"), 0.0) >= 0.520
        and (
            finite_float(metrics.get("contrast"), 0.0) >= 0.190
            or bool(visual.get("local_reentry"))
            or bool(visual.get("opening_body_start"))
            or bool(visual.get("phrase_body_shift"))
        )
    )
    if (
        same_section_front_edge
        and not sustained_section_front_edge
        and not high_dark_body_front_edge
        and abs_offset > max(8.0, 4.0 * bar_sec)
        and (
            finite_float(metrics.get("body_score"), 0.0) < 0.700
            or finite_float(metrics.get("simultaneity"), 0.0) < 0.700
            or finite_float(metrics.get("post_drum_cont8"), 0.0) < 0.650
        )
    ):
        return None
    if (
        offset > 0.0
        and not sustained_section_front_edge
        and not high_dark_body_front_edge
        and abs_offset > max(3.0, 1.50 * bar_sec)
        and (
            finite_float(metrics.get("body_score"), 0.0) < 0.700
            or finite_float(metrics.get("simultaneity"), 0.0) < 0.680
            or finite_float(metrics.get("post_drum_cont8"), 0.0) < 0.650
        )
    ):
        return None
    if (
        selected_phrase >= 0.66
        and candidate_phrase < selected_phrase - 0.30
        and not same_section_front_edge
        and not sustained_section_front_edge
        and not high_dark_body_front_edge
        and not strong_before_body_target
    ):
        return None
    if (
        offset > 0.0
        and abs_offset <= 1.25 * bar_sec
        and candidate_phrase < 0.48
        and not bool(visual.get("phrase_body_shift"))
        and not same_section_front_edge
        and not sustained_section_front_edge
        and not high_dark_body_front_edge
    ):
        return None
    transition_reset = bool(
        visual.get("phrase_body_shift")
        or visual.get("phrase_dropout_reentry")
        or visual.get("opening_body_start")
        or sustained_section_front_edge
        or high_dark_body_front_edge
        or strong_before_body_target
        or (
            visual.get("local_reentry")
            and selected_one_distance > max_one_distance
            and finite_float(metrics.get("body_score"), 0.0) >= 0.620
        )
    )
    if contrast < min_correction_contrast and not transition_reset:
        return None

    corrected_candidate = dict(candidate)
    corrected_candidate["timestamp"] = float(edge_time)
    corrected_candidate["snapped_sec"] = float(edge_time)
    corrected_candidate["time_sec"] = float(edge_time)
    corrected_candidate["microaligned_time"] = float(edge_time)
    corrected_candidate["visual_raw_chunk_time"] = float(edge_time)
    corrected_proof = marker_boom_proof(
        float(edge_time),
        candidates,
        selected_candidate=corrected_candidate,
        profile=profile_data,
        beatgrid=beatgrid,
    )
    if not bool(corrected_proof.get("passes")):
        return None

    direction = "late_inside_body" if offset > 0.0 else "before_body_front_edge"
    return {
        "marker": float(edge_time),
        "previous_marker": finite_float(marker_sec),
        "offset_sec": float(offset),
        "direction": direction,
        "candidate": corrected_candidate,
        "proof": corrected_proof,
        "profile": nearest_profile,
        "reason": f"Boom proof moved {direction} marker by {offset:.3f}s to the body front edge",
    }


def earliest_dominant_boom_replacement(
    marker_sec: Any,
    boom_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_candidate: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    beatgrid: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    candidates = [dict(row) for row in boom_candidates or [] if isinstance(row, Mapping)]
    if not candidates:
        return None
    proof = marker_boom_proof(
        marker_sec,
        candidates,
        selected_candidate=selected_candidate,
        profile=profile,
        beatgrid=beatgrid,
    )
    if not any("earlier_dominant_boom_available" in str(reason) for reason in proof.get("reasons") or []):
        return None
    marker = finite_float(marker_sec, default=float("nan"))
    if not math.isfinite(marker):
        return None
    nearest = proof.get("nearest") if isinstance(proof.get("nearest"), Mapping) else {}
    current_score = finite_float(nearest.get("candidate_score"), 0.0)
    profile_data = profile if isinstance(profile, Mapping) else load_boom_profile()
    selected_bar_sec = _bar_sec_from_candidate(selected_candidate, beatgrid) if isinstance(selected_candidate, Mapping) else _bar_sec_from_candidate(candidates[0], beatgrid)
    if selected_bar_sec < 1.35:
        return None
    fast_tempo = bool(selected_bar_sec < 2.0)
    selected_visual = visual_components(selected_candidate) if isinstance(selected_candidate, Mapping) else {}
    selected_clock = (
        selected_candidate.get("bpm_clock")
        if isinstance(selected_candidate, Mapping) and isinstance(selected_candidate.get("bpm_clock"), Mapping)
        else {}
    )
    selected_one_distance = abs(
        finite_float(
            selected_visual.get("one_distance_ms", selected_clock.get("one_distance_ms")),
            0.0,
        )
    )
    profile_for_thresholds = profile if isinstance(profile, Mapping) else load_boom_profile()
    profile_thresholds = (
        profile_for_thresholds.get("thresholds")
        if isinstance(profile_for_thresholds.get("thresholds"), Mapping)
        else {}
    )
    max_one_distance = finite_float(
        profile_thresholds.get("max_one_distance_ms"),
        finite_float(DEFAULT_PROFILE["thresholds"]["max_one_distance_ms"]),
    )
    selected_off_one = bool(selected_one_distance > max_one_distance)

    reference_candidates: List[Mapping[str, Any]] = []
    if isinstance(selected_candidate, Mapping) and visual_components(selected_candidate):
        reference_candidates.append(selected_candidate)
    for candidate in candidates:
        edge_time = candidate_time(candidate)
        if edge_time is None:
            continue
        if abs(float(edge_time) - marker) <= max(1.250, selected_bar_sec):
            reference_candidates.append(candidate)
            continue
        section_end = _section_end_time(candidate, float(edge_time), beatgrid)
        if float(edge_time) - 0.100 <= marker <= section_end + 0.100:
            reference_candidates.append(candidate)
    if not reference_candidates:
        reference_candidates = [nearest.get("candidate")] if isinstance(nearest.get("candidate"), Mapping) else []
    reference_scores = [
        finite_float(row.get("score"), finite_float(row.get("confidence_score")))
        for row in reference_candidates
        if isinstance(row, Mapping)
    ]
    reference_metrics = [
        candidate_boom_metrics(row)
        for row in reference_candidates
        if isinstance(row, Mapping)
    ]
    reference_score = max(reference_scores or [current_score])
    reference_bass = max((float(row.get("post_bass8", 0.0) or 0.0) for row in reference_metrics), default=0.0)
    reference_darkness = max((float(row.get("darkness", 0.0) or 0.0) for row in reference_metrics), default=0.0)
    reference_body = max((float(row.get("body_score", 0.0) or 0.0) for row in reference_metrics), default=0.0)
    reference_phrase = max((float(row.get("phrase_prior", 0.0) or 0.0) for row in reference_metrics), default=0.0)

    viable: List[Dict[str, Any]] = []
    for candidate in candidates:
        edge_time = candidate_time(candidate)
        if edge_time is None or float(edge_time) >= marker - 0.250:
            continue
        visual = visual_components(candidate)
        clock_bar = int(finite_float(visual.get("clock_bar"), 0.0))
        if clock_bar < 5 and float(edge_time) < 8.0:
            continue
        candidate_metrics = candidate_boom_metrics(candidate)
        candidate_phrase = float(candidate_metrics.get("phrase_prior", 0.0) or 0.0)
        bar_sec = _bar_sec_from_candidate(candidate, beatgrid)
        file_start_opening = bool(
            float(edge_time) <= max(4.0, 2.0 * float(bar_sec))
            and bool(visual.get("opening_drop_profile"))
        )
        selected_source = str(selected_candidate.get("selected_by") or "") if isinstance(selected_candidate, Mapping) else ""
        selected_metrics_for_opening = (
            candidate_boom_metrics(selected_candidate)
            if isinstance(selected_candidate, Mapping)
            else {}
        )
        selected_visual_for_opening = visual_components(selected_candidate) if isinstance(selected_candidate, Mapping) else {}
        selected_locked_later_body = bool(
            selected_source
            in {
                "visual_candidate_selector",
                "visual_boom_grid_phase_calibration",
                "visual_boom_grid_one_snap",
                "visual_selector_locked_front_edge",
                "visual_drop_v2",
            }
            and finite_float(selected_candidate.get("score"), finite_float(selected_candidate.get("confidence_score"), 0.0))
            >= 0.620
            and finite_float(selected_metrics_for_opening.get("post8_height"), 0.0) >= 0.560
            and finite_float(selected_metrics_for_opening.get("post_bass8"), 0.0) >= 0.300
            and finite_float(selected_metrics_for_opening.get("post_drum8"), 0.0) >= 0.900
            and finite_float(selected_metrics_for_opening.get("post_drum_cont8"), 0.0) >= 0.540
            and (
                finite_float(selected_metrics_for_opening.get("contrast"), 0.0) >= 0.220
                or bool(selected_visual_for_opening.get("phrase_body_shift"))
                or bool(selected_visual_for_opening.get("phrase_dropout_reentry"))
            )
        )
        if (
            not file_start_opening
            and clock_bar <= 4
            and float(edge_time) <= max(4.0, 2.0 * float(bar_sec))
            and selected_locked_later_body
        ):
            continue
        if (
            not file_start_opening
            and clock_bar <= 8
            and float(edge_time) <= max(16.0, 8.0 * float(bar_sec))
            and candidate_phrase < 0.600
            and reference_phrase >= 0.780
        ):
            continue
        candidate_score = finite_float(candidate.get("score"), finite_float(candidate.get("confidence_score")))
        if current_score > 0.0 and candidate_score < current_score * 0.80:
            continue
        ratio_gate = 0.94
        if fast_tempo and selected_off_one:
            ratio_gate = 0.88
        if reference_score > 0.0 and candidate_score < reference_score * ratio_gate:
            continue
        candidate_proof = marker_boom_proof(
            float(edge_time),
            candidates,
            selected_candidate=candidate,
            profile=profile_data,
            beatgrid=beatgrid,
        )
        if not bool(candidate_proof.get("passes")):
            continue
        profile_score = (
            candidate_proof.get("nearest_profile")
            if isinstance(candidate_proof.get("nearest_profile"), Mapping)
            else {}
        )
        metrics = profile_score.get("metrics") if isinstance(profile_score.get("metrics"), Mapping) else {}
        min_body_score = 0.600 if selected_off_one else 0.640
        if finite_float(metrics.get("body_score"), 0.0) < min_body_score:
            continue
        candidate_bass = finite_float(metrics.get("post_bass8"), 0.0)
        candidate_darkness = finite_float(metrics.get("darkness"), 0.0)
        candidate_body = finite_float(metrics.get("body_score"), 0.0)
        candidate_contrast = finite_float(metrics.get("contrast"), 0.0)
        bass_tolerance = 0.080 if (fast_tempo and selected_off_one) else (0.050 if fast_tempo else 0.080)
        darkness_tolerance = 0.080 if (fast_tempo and selected_off_one) else (0.060 if fast_tempo else 0.100)
        body_tolerance = 0.065 if (fast_tempo and selected_off_one) else (0.050 if fast_tempo else 0.080)
        first_drop_has_strong_transition = bool(
            (candidate_phrase >= 0.860 or bool(visual.get("phrase_body_shift")))
            and (
                candidate_contrast >= 0.300
                or bool(visual.get("local_reentry"))
                or bool(visual.get("phrase_dropout_reentry"))
            )
        )
        if first_drop_has_strong_transition:
            bass_tolerance = max(bass_tolerance, 0.140)
            darkness_tolerance = max(darkness_tolerance, 0.125)
            body_tolerance = max(body_tolerance, 0.105)
        if reference_bass >= 0.260 and candidate_bass < reference_bass - bass_tolerance:
            continue
        if reference_darkness >= 0.500 and candidate_darkness < reference_darkness - darkness_tolerance:
            continue
        if reference_body >= 0.600 and candidate_body < reference_body - body_tolerance:
            continue
        viable.append(
            {
                "candidate": candidate,
                "edge_time": float(edge_time),
                "proof": candidate_proof,
                "profile": profile_score,
                "score": float(candidate_score),
                "clock_bar": clock_bar,
            }
        )

    if not viable:
        return None
    viable.sort(key=lambda row: (row["edge_time"], -row["score"]))
    best = viable[0]
    return {
        "marker": float(best["edge_time"]),
        "previous_marker": float(marker),
        "offset_sec": float(marker - best["edge_time"]),
        "direction": "earlier_dominant_boom",
        "candidate": dict(best["candidate"]),
        "proof": dict(best["proof"]),
        "profile": dict(best["profile"]),
        "reason": (
            "Boom proof moved marker to the earliest proven dominant body edge "
            f"at {best['edge_time']:.3f}s b{best['clock_bar']}"
        ),
    }
