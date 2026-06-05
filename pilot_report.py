#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.exclusions import row_has_excluded_path


ACCEPT_TOLERANCE_SEC = 1e-3
PERCENT_LABELS = ("5ms", "10ms", "25ms", "50ms", "100ms")


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> Any:
    with open(path.expanduser(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_json_optional(path: Path) -> tuple[Optional[Any], str]:
    try:
        if not path.expanduser().exists():
            return None, f"missing: {path}"
        return _read_json(path), ""
    except Exception as exc:
        return None, str(exc) or exc.__class__.__name__


def _read_batch_rows(path: Path) -> List[Dict[str, str]]:
    if not path.expanduser().exists():
        return []
    rows: List[Dict[str, str]] = []
    with open(path.expanduser(), "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row_has_excluded_path(row):
                rows.append(dict(row))
    return rows


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.expanduser().exists():
        return
    with open(path.expanduser(), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and not row_has_excluded_path(row):
                yield row


def _ai_pick(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    return _float_or_none(row.get("user_pick"))


def _status_counts(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown").strip().lower() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _confidence_counts(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    for row in rows:
        tier = str(row.get("confidence_tier") or "").strip().upper()
        if tier not in counts:
            tier = "UNKNOWN"
        counts[tier] += 1
    return counts


def summarize_library(batch_summary: Path) -> Dict[str, Any]:
    rows = _read_batch_rows(batch_summary)
    valid_rows = [row for row in rows if str(row.get("als_valid") or "").strip().lower() == "true"]
    invalid_rows = [row for row in rows if str(row.get("als_valid") or "").strip().lower() == "false"]
    checked = len(valid_rows) + len(invalid_rows)
    failed = [row for row in rows if str(row.get("status") or "").strip().lower() == "error"]
    return {
        "batch_summary": str(batch_summary),
        "exists": batch_summary.expanduser().exists(),
        "total_tracks_processed": int(len(rows)),
        "successful_als_files": int(len(valid_rows)),
        "failed_tracks": int(len(failed)),
        "als_validation_checked": int(checked),
        "als_validation_pass_rate": float((100.0 * len(valid_rows) / checked) if checked else 0.0),
        "status_counts": _status_counts(rows),
        "confidence_counts": _confidence_counts(rows),
        "failure_examples": [
            {
                "track": row.get("filename", ""),
                "status": row.get("status", ""),
                "error": row.get("error", "") or row.get("als_validation_error", ""),
            }
            for row in failed[:10]
        ],
    }


def _review_state_summary(batch_summary: Path) -> Dict[str, Any]:
    state_path = batch_summary.expanduser().parent / "review_state.json"
    payload, error = _read_json_optional(state_path)
    if not isinstance(payload, Mapping):
        return {"exists": False, "path": str(state_path), "error": error, "skipped": 0, "state_reviewed": 0}
    items = payload.get("items")
    if not isinstance(items, Mapping):
        return {"exists": True, "path": str(state_path), "skipped": 0, "state_reviewed": 0}
    values = [item for item in items.values() if isinstance(item, Mapping)]
    return {
        "exists": True,
        "path": str(state_path),
        "skipped": int(sum(1 for item in values if item.get("skipped"))),
        "state_reviewed": int(sum(1 for item in values if item.get("reviewed"))),
        "state_approved": int(sum(1 for item in values if item.get("approved"))),
        "state_corrected": int(sum(1 for item in values if item.get("corrected"))),
    }


def summarize_review(corrections: Path, batch_summary: Path) -> Dict[str, Any]:
    rows = list(_iter_jsonl(corrections))
    reviewed: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    corrected: List[Dict[str, Any]] = []
    deltas: List[float] = []
    corrected_deltas: List[float] = []

    for row in rows:
        user = _user_pick(row)
        ai = _ai_pick(row)
        if user is None or ai is None:
            continue
        delta = _float_or_none(row.get("delta"))
        if delta is None:
            delta = float(user - ai)
        item = {
            "track": row.get("track", ""),
            "ai_pick": float(ai),
            "user_pick": float(user),
            "delta_sec": float(delta),
            "abs_delta_sec": abs(float(delta)),
            "confidence_tier": row.get("confidence_tier", "UNKNOWN"),
            "selected_by": row.get("selected_by", ""),
        }
        reviewed.append(item)
        deltas.append(float(delta))
        if abs(float(delta)) <= ACCEPT_TOLERANCE_SEC:
            accepted.append(item)
        else:
            corrected.append(item)
            corrected_deltas.append(float(delta))

    state = _review_state_summary(batch_summary)
    worst = sorted(corrected, key=lambda item: item["abs_delta_sec"], reverse=True)[:10]
    return {
        "corrections": str(corrections),
        "exists": corrections.expanduser().exists(),
        "total_reviewed": int(len(reviewed)),
        "accepted_ai_picks": int(len(accepted)),
        "corrected_picks": int(len(corrected)),
        "skipped": int(state.get("skipped", 0)),
        "review_state": state,
        "average_correction_delta_sec": float(mean(corrected_deltas)) if corrected_deltas else 0.0,
        "median_correction_delta_sec": float(median(corrected_deltas)) if corrected_deltas else 0.0,
        "average_abs_correction_delta_sec": float(mean(abs(v) for v in corrected_deltas)) if corrected_deltas else 0.0,
        "median_abs_correction_delta_sec": float(median(abs(v) for v in corrected_deltas)) if corrected_deltas else 0.0,
        "average_review_delta_sec": float(mean(deltas)) if deltas else 0.0,
        "median_review_delta_sec": float(median(deltas)) if deltas else 0.0,
        "worst_corrections": worst,
    }


def _sec_to_ms(value: Any) -> float:
    sec = _float_or_none(value)
    return float(sec * 1000.0) if sec is not None else 0.0


def summarize_evaluation(evaluation: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(evaluation)
    if not isinstance(payload, Mapping):
        return {"evaluation": str(evaluation), "exists": False, "error": error}

    tiers = payload.get("accuracy_by_confidence_tier")
    tier_rows: Dict[str, Dict[str, Any]] = {}
    if isinstance(tiers, Mapping):
        for tier, metrics in tiers.items():
            if not isinstance(metrics, Mapping):
                continue
            tier_rows[str(tier)] = {
                "count": int(metrics.get("count", 0) or 0),
                "mean_absolute_error_ms": _sec_to_ms(metrics.get("mean_absolute_error_sec")),
                "median_absolute_error_ms": _sec_to_ms(metrics.get("median_absolute_error_sec")),
                "model_selected_closest_candidate_percent": float(metrics.get("model_selected_closest_candidate_percent", 0.0) or 0.0),
                "handcrafted_selected_closest_candidate_percent": float(metrics.get("handcrafted_selected_closest_candidate_percent", 0.0) or 0.0),
                "percent_within": dict(metrics.get("percent_within", {}) if isinstance(metrics.get("percent_within"), Mapping) else {}),
            }

    return {
        "evaluation": str(evaluation),
        "exists": True,
        "total_reviewed_tracks": int(payload.get("total_reviewed_tracks", 0) or 0),
        "accepted_picks": int(payload.get("accepted_picks", 0) or 0),
        "corrected_picks": int(payload.get("corrected_picks", 0) or 0),
        "mean_absolute_error_ms": _sec_to_ms(payload.get("mean_absolute_error_sec")),
        "median_absolute_error_ms": _sec_to_ms(payload.get("median_absolute_error_sec")),
        "percent_within": dict(payload.get("percent_within", {}) if isinstance(payload.get("percent_within"), Mapping) else {}),
        "accuracy_by_confidence_tier": tier_rows,
        "worst_misses": list(payload.get("worst_20_misses", [])[:10] if isinstance(payload.get("worst_20_misses"), list) else []),
    }


def summarize_drumprint_ablation(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"drumprint_ablation": str(path), "exists": False, "error": error}

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    fake_hit = payload.get("fake_hit_analysis") if isinstance(payload.get("fake_hit_analysis"), Mapping) else {}
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), Mapping) else {}
    referenced = int(metrics.get("referenced_tracks", 0) or 0)

    if referenced < 25:
        decision = "needs more data"
    elif recommendation.get("recommend_drumprint_default"):
        decision = "keep DrumPrint default"
    else:
        decision = "keep optional"

    return {
        "drumprint_ablation": str(path),
        "exists": True,
        "referenced_tracks": referenced,
        "improved_count": int(metrics.get("tracks_improved_by_drumprint", 0) or 0),
        "worsened_count": int(metrics.get("tracks_worsened_by_drumprint", 0) or 0),
        "unchanged_count": int(metrics.get("tracks_unchanged", 0) or 0),
        "fake_hit_rescues": int(fake_hit.get("fake_hit_rescues", 0) or 0),
        "fake_hit_regressions": int(fake_hit.get("fake_hit_regressions", 0) or 0),
        "best_improvements": list(metrics.get("best_drumprint_improvements", [])[:10] if isinstance(metrics.get("best_drumprint_improvements"), list) else []),
        "worst_regressions": list(metrics.get("worst_drumprint_regressions", [])[:10] if isinstance(metrics.get("worst_drumprint_regressions"), list) else []),
        "recommendation": decision,
        "ablation_recommendation": dict(recommendation),
    }


def summarize_microalign_ablation(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"microalign_ablation": str(path), "exists": False, "error": error}

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    no_metrics = metrics.get("no_microalign") if isinstance(metrics.get("no_microalign"), Mapping) else {}
    micro_metrics = metrics.get("microalign") if isinstance(metrics.get("microalign"), Mapping) else {}
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), Mapping) else {}
    eligible = metrics.get("auto_accept_eligible") if isinstance(metrics.get("auto_accept_eligible"), Mapping) else {}
    referenced = int(metrics.get("referenced_tracks", 0) or 0)

    if referenced < 25:
        decision = "needs more data"
    elif recommendation.get("recommend_microalign_default"):
        decision = "enable MicroSnap default"
    else:
        decision = "keep optional"

    return {
        "microalign_ablation": str(path),
        "exists": True,
        "referenced_tracks": referenced,
        "improved_count": int(metrics.get("tracks_improved_by_microalign", 0) or 0),
        "worsened_count": int(metrics.get("tracks_worsened_by_microalign", 0) or 0),
        "unchanged_count": int(metrics.get("tracks_unchanged", 0) or 0),
        "median_error_before_ms": float(no_metrics.get("median_absolute_error_ms", 0.0) or 0.0),
        "median_error_after_ms": float(micro_metrics.get("median_absolute_error_ms", 0.0) or 0.0),
        "within_10ms_before": float((no_metrics.get("percent_within") or {}).get("10ms", 0.0) or 0.0),
        "within_10ms_after": float((micro_metrics.get("percent_within") or {}).get("10ms", 0.0) or 0.0),
        "within_25ms_before": float((no_metrics.get("percent_within") or {}).get("25ms", 0.0) or 0.0),
        "within_25ms_after": float((micro_metrics.get("percent_within") or {}).get("25ms", 0.0) or 0.0),
        "average_snap_offset_ms": float(metrics.get("average_snap_offset_ms", 0.0) or 0.0),
        "median_snap_offset_ms": float(metrics.get("median_snap_offset_ms", 0.0) or 0.0),
        "percent_snap_offsets_over_100ms": float(metrics.get("percent_snap_offsets_over_100ms", 0.0) or 0.0),
        "auto_accept_eligible_count": int(eligible.get("conservative", 0) or 0),
        "auto_accept_eligible": dict(eligible),
        "best_improvements": list(metrics.get("best_microalign_improvements", [])[:10] if isinstance(metrics.get("best_microalign_improvements"), list) else []),
        "worst_regressions": list(metrics.get("worst_microalign_regressions", [])[:10] if isinstance(metrics.get("worst_microalign_regressions"), list) else []),
        "recommendation": decision,
        "ablation_recommendation": dict(recommendation),
    }


def summarize_oracle_diagnostics(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"oracle_diagnostics": str(path), "exists": False, "error": error}
    oracle = payload.get("oracle_percent") if isinstance(payload.get("oracle_percent"), Mapping) else {}
    selected = payload.get("selected_percent") if isinstance(payload.get("selected_percent"), Mapping) else {}
    return {
        "oracle_diagnostics": str(path),
        "exists": True,
        "reviewed_tracks": int(payload.get("reviewed_tracks", 0) or 0),
        "oracle_at_10_within_25ms_percent": float(oracle.get("oracle_at_10_within_25ms_percent", 0.0) or 0.0),
        "oracle_at_25_within_25ms_percent": float(oracle.get("oracle_at_25_within_25ms_percent", 0.0) or 0.0),
        "oracle_at_50_within_25ms_percent": float(oracle.get("oracle_at_50_within_25ms_percent", 0.0) or 0.0),
        "selected_within_25ms_percent": float(selected.get("selected_within_25ms_percent", 0.0) or 0.0),
        "selected_within_100ms_percent": float(selected.get("selected_within_100ms_percent", 0.0) or 0.0),
        "average_rank_of_closest_candidate": payload.get("average_rank_of_closest_candidate"),
        "median_rank_of_closest_candidate": payload.get("median_rank_of_closest_candidate"),
        "counts": dict(payload.get("counts", {}) if isinstance(payload.get("counts"), Mapping) else {}),
    }


def summarize_gate_rejections(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"gate_rejection_report": str(path), "exists": False, "error": error}
    return {
        "gate_rejection_report": str(path),
        "exists": True,
        "total_rows": int(payload.get("total_rows", 0) or 0),
        "auto_accepted": int(payload.get("auto_accepted", 0) or 0),
        "total_rejected": int(payload.get("total_rejected", 0) or 0),
        "rejection_reasons_ranked": list(payload.get("rejection_reasons_ranked", [])[:10] if isinstance(payload.get("rejection_reasons_ranked"), list) else []),
    }


def summarize_auto_verifier(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"auto_verifier_report": str(path), "exists": False, "error": error}
    validation = payload.get("validation_metrics") if isinstance(payload.get("validation_metrics"), Mapping) else {}
    candidate = validation.get("candidate_classifier") if isinstance(validation.get("candidate_classifier"), Mapping) else {}
    thresholds = validation.get("threshold_coverage") if isinstance(validation.get("threshold_coverage"), Mapping) else {}
    return {
        "auto_verifier_report": str(path),
        "exists": True,
        "training_rows": int(payload.get("training_rows", 0) or 0),
        "correction_rows": int(payload.get("correction_rows", 0) or 0),
        "validation_selected_count": int(validation.get("selected_count", 0) or 0),
        "validation_selected_actual_within_25ms_percent": validation.get("selected_actual_within_25ms_percent"),
        "candidate_roc_auc": candidate.get("roc_auc"),
        "candidate_average_precision": candidate.get("average_precision"),
        "threshold_coverage": dict(thresholds),
    }


def summarize_auto_gate_config(path: Path) -> Dict[str, Any]:
    payload, error = _read_json_optional(path)
    if not isinstance(payload, Mapping):
        return {"auto_gate_config": str(path), "exists": False, "error": error}
    return {
        "auto_gate_config": str(path),
        "exists": True,
        "calibration_selected_tracks": int(payload.get("calibration_selected_tracks", 0) or 0),
        "safe": dict(payload.get("safe", {}) if isinstance(payload.get("safe"), Mapping) else {}),
        "balanced": dict(payload.get("balanced", {}) if isinstance(payload.get("balanced"), Mapping) else {}),
        "aggressive": dict(payload.get("aggressive", {}) if isinstance(payload.get("aggressive"), Mapping) else {}),
    }


def build_action_recommendations(
    library: Mapping[str, Any],
    review: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    drumprint: Mapping[str, Any],
    microalign: Mapping[str, Any],
) -> List[str]:
    actions: List[str] = []
    reviewed = int(review.get("total_reviewed", 0) or 0)
    corrected = int(review.get("corrected_picks", 0) or 0)
    low_count = int((library.get("confidence_counts") or {}).get("LOW", 0))
    als_pass = float(library.get("als_validation_pass_rate", 0.0) or 0.0)

    if als_pass and als_pass < 100.0:
        actions.append("Fix ALS validation failures before scaling to the full library.")
    if low_count:
        actions.append("Review more LOW confidence tracks.")
    if reviewed < 25:
        actions.append(f"Review at least {25 - reviewed} more tracks before deciding defaults.")
    elif corrected < 10:
        actions.append("Add more corrected examples before trusting the trained ranker on edge cases.")
    if not evaluation.get("exists"):
        actions.append("Run evaluate_ranker.py after retraining so model quality is visible.")
    else:
        p25 = float((evaluation.get("percent_within") or {}).get("25ms", 0.0) or 0.0)
        if p25 < 80.0:
            actions.append("Do not promote the model yet; 25ms accuracy is still below 80%.")
    if not drumprint.get("exists"):
        actions.append("Run compare_drumprint.py to prove DrumPrint before making it default.")
    else:
        rescues = int(drumprint.get("fake_hit_rescues", 0) or 0)
        regressions = int(drumprint.get("fake_hit_regressions", 0) or 0)
        if rescues > regressions:
            actions.append("DrumPrint appears helpful for fake pre-drop hits.")
        if drumprint.get("recommendation") == "keep DrumPrint default":
            actions.append("DrumPrint passes the pilot gate; enable it by default for drums stems.")
        elif drumprint.get("recommendation") == "keep optional":
            actions.append("Keep DrumPrint optional until regressions are understood.")
        else:
            actions.append("Add more golden tracks for dubstep/techno and rerun DrumPrint A/B.")
    if not microalign.get("exists"):
        actions.append("Run compare_microalign.py to prove MicroSnap before enabling conservative auto-review.")
    else:
        if microalign.get("recommendation") == "enable MicroSnap default":
            actions.append("MicroSnap passes the pilot gate; enable it by default and keep auto-review conservative.")
        elif int(microalign.get("auto_accept_eligible_count", 0) or 0):
            actions.append("Keep MicroSnap optional, but conservative auto-review has safe candidates to inspect.")
        else:
            actions.append("Keep MicroSnap suggestions manual until A/B metrics improve.")
    if not actions:
        actions.append("Pilot looks healthy; scale gradually and keep reviewing LOW confidence misses.")
    return actions


def _fmt(value: Any, digits: int = 2) -> str:
    num = _float_or_none(value)
    if num is None:
        return str(value) if value not in (None, "") else "-"
    return f"{num:.{digits}f}"


def _ms(value_sec: Any) -> str:
    return f"{_sec_to_ms(value_sec):.1f} ms"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def render_markdown(report: Mapping[str, Any]) -> str:
    library = report["library_processing"]
    review = report["human_review"]
    evaluation = report["model_performance"]
    drumprint = report["drumprint_ab"]
    microalign = report["microsnap_ab"]
    oracle = report.get("oracle_diagnostics", {})
    gate = report.get("gate_rejections", {})
    verifier = report.get("auto_verifier", {})
    gate_config = report.get("auto_gate_config", {})
    actions = report["action_recommendations"]
    confidence = library.get("confidence_counts", {})

    lines: List[str] = [
        "# Pilot Report",
        "",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "## Library Processing",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Total tracks processed", library.get("total_tracks_processed", 0)],
                ["Successful ALS files", library.get("successful_als_files", 0)],
                ["Failed tracks", library.get("failed_tracks", 0)],
                ["ALS validation pass rate", f"{float(library.get('als_validation_pass_rate', 0.0)):.2f}%"],
            ],
        ),
        "",
        _table(
            ["Confidence", "Count"],
            [[tier, confidence.get(tier, 0)] for tier in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")],
        ),
        "",
        "## Human Review",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Total reviewed", review.get("total_reviewed", 0)],
                ["Accepted AI picks", review.get("accepted_ai_picks", 0)],
                ["Corrected picks", review.get("corrected_picks", 0)],
                ["Skipped", review.get("skipped", 0)],
                ["Average correction delta", _ms(review.get("average_correction_delta_sec"))],
                ["Median correction delta", _ms(review.get("median_correction_delta_sec"))],
                ["Median absolute correction delta", _ms(review.get("median_abs_correction_delta_sec"))],
            ],
        ),
        "",
    ]

    worst_corrections = review.get("worst_corrections", [])
    if worst_corrections:
        lines.extend(
            [
                "### Worst Corrections",
                "",
                _table(
                    ["Track", "AI", "User", "Delta"],
                    [
                        [
                            Path(str(row.get("track", ""))).name,
                            _fmt(row.get("ai_pick"), 3),
                            _fmt(row.get("user_pick"), 3),
                            _ms(row.get("delta_sec")),
                        ]
                        for row in worst_corrections[:8]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Model Performance",
            "",
            _table(
                ["Metric", "Value"],
                [
                    ["Mean absolute error", f"{float(evaluation.get('mean_absolute_error_ms', 0.0)):.2f} ms"],
                    ["Median absolute error", f"{float(evaluation.get('median_absolute_error_ms', 0.0)):.2f} ms"],
                    ["Accepted picks", evaluation.get("accepted_picks", 0)],
                    ["Corrected picks", evaluation.get("corrected_picks", 0)],
                ],
            ),
            "",
            _table(
                ["Within", "Percent"],
                [[label, f"{float((evaluation.get('percent_within') or {}).get(label, 0.0)):.2f}%"] for label in PERCENT_LABELS],
            ),
            "",
        ]
    )

    tiers = evaluation.get("accuracy_by_confidence_tier", {})
    if tiers:
        lines.extend(
            [
                "### Accuracy By Confidence Tier",
                "",
                _table(
                    ["Tier", "Count", "Mean Error", "Median Error", "Model Closest"],
                    [
                        [
                            tier,
                            (tiers.get(tier) or {}).get("count", 0),
                            f"{float((tiers.get(tier) or {}).get('mean_absolute_error_ms', 0.0)):.2f} ms",
                            f"{float((tiers.get(tier) or {}).get('median_absolute_error_ms', 0.0)):.2f} ms",
                            f"{float((tiers.get(tier) or {}).get('model_selected_closest_candidate_percent', 0.0)):.2f}%",
                        ]
                        for tier in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
                    ],
                ),
                "",
            ]
        )

    if oracle.get("exists"):
        lines.extend(
            [
                "## Oracle Diagnostics",
                "",
                _table(
                    ["Metric", "Value"],
                    [
                        ["Reviewed tracks", oracle.get("reviewed_tracks", 0)],
                        ["Oracle@10 within 25ms", f"{float(oracle.get('oracle_at_10_within_25ms_percent', 0.0)):.2f}%"],
                        ["Oracle@25 within 25ms", f"{float(oracle.get('oracle_at_25_within_25ms_percent', 0.0)):.2f}%"],
                        ["Oracle@50 within 25ms", f"{float(oracle.get('oracle_at_50_within_25ms_percent', 0.0)):.2f}%"],
                        ["Selected within 25ms", f"{float(oracle.get('selected_within_25ms_percent', 0.0)):.2f}%"],
                        ["Selected within 100ms", f"{float(oracle.get('selected_within_100ms_percent', 0.0)):.2f}%"],
                        ["Median closest rank", _fmt(oracle.get("median_rank_of_closest_candidate"), 2)],
                    ],
                ),
                "",
            ]
        )

    if gate.get("exists"):
        lines.extend(
            [
                "## Gate Rejections",
                "",
                _table(
                    ["Metric", "Value"],
                    [
                        ["Rows analyzed", gate.get("total_rows", 0)],
                        ["Auto accepted", gate.get("auto_accepted", 0)],
                        ["Held for review", gate.get("total_rejected", 0)],
                    ],
                ),
                "",
            ]
        )
        reasons = gate.get("rejection_reasons_ranked", [])
        if reasons:
            lines.extend(
                [
                    _table(["Reason", "Count"], [[item[0], item[1]] for item in reasons[:8] if isinstance(item, (list, tuple)) and len(item) >= 2]),
                    "",
                ]
            )

    if verifier.get("exists"):
        lines.extend(
            [
                "## Auto Verifier",
                "",
                _table(
                    ["Metric", "Value"],
                    [
                        ["Training rows", verifier.get("training_rows", 0)],
                        ["Correction rows", verifier.get("correction_rows", 0)],
                        ["Validation selected tracks", verifier.get("validation_selected_count", 0)],
                        ["Selected actual within 25ms", f"{float(verifier.get('validation_selected_actual_within_25ms_percent') or 0.0):.2f}%"],
                        ["Candidate ROC AUC", _fmt(verifier.get("candidate_roc_auc"), 4)],
                        ["Candidate average precision", _fmt(verifier.get("candidate_average_precision"), 4)],
                    ],
                ),
                "",
            ]
        )

    if gate_config.get("exists"):
        safe = gate_config.get("safe") if isinstance(gate_config.get("safe"), Mapping) else {}
        balanced = gate_config.get("balanced") if isinstance(gate_config.get("balanced"), Mapping) else {}
        aggressive = gate_config.get("aggressive") if isinstance(gate_config.get("aggressive"), Mapping) else {}
        lines.extend(
            [
                "## Tuned Auto Gate",
                "",
                _table(
                    ["Mode", "Accepted", "Coverage", "False Saves", "Max Error"],
                    [
                        ["safe", safe.get("accepted", 0), f"{float(safe.get('coverage_percent') or 0.0):.2f}%", safe.get("false_auto_save_count", 0), f"{float(safe.get('max_actual_error_ms') or 0.0):.2f} ms"],
                        ["balanced", balanced.get("accepted", 0), f"{float(balanced.get('coverage_percent') or 0.0):.2f}%", balanced.get("false_auto_save_count", 0), f"{float(balanced.get('max_actual_error_ms') or 0.0):.2f} ms"],
                        ["aggressive", aggressive.get("accepted", 0), f"{float(aggressive.get('coverage_percent') or 0.0):.2f}%", aggressive.get("false_auto_save_count", 0), f"{float(aggressive.get('max_actual_error_ms') or 0.0):.2f} ms"],
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## DrumPrint A/B",
            "",
            _table(
                ["Metric", "Value"],
                [
                    ["Referenced tracks", drumprint.get("referenced_tracks", 0)],
                    ["Improved", drumprint.get("improved_count", 0)],
                    ["Worsened", drumprint.get("worsened_count", 0)],
                    ["Unchanged", drumprint.get("unchanged_count", 0)],
                    ["Fake-hit rescues", drumprint.get("fake_hit_rescues", 0)],
                    ["Fake-hit regressions", drumprint.get("fake_hit_regressions", 0)],
                    ["Recommendation", drumprint.get("recommendation", "needs more data")],
                ],
            ),
            "",
        ]
    )

    best = drumprint.get("best_improvements", [])
    worst = drumprint.get("worst_regressions", [])
    if best:
        lines.extend(
            [
                "### Best DrumPrint Improvements",
                "",
                _table(
                    ["Track", "Improvement", "No DrumPrint", "DrumPrint"],
                    [
                        [
                            Path(str(row.get("track", ""))).name,
                            f"{float(row.get('improvement_ms', 0.0)):.1f} ms",
                            _fmt((row.get("no_drumprint") or {}).get("time"), 3),
                            _fmt((row.get("drumprint") or {}).get("time"), 3),
                        ]
                        for row in best[:8]
                    ],
                ),
                "",
            ]
        )
    if worst:
        lines.extend(
            [
                "### Worst DrumPrint Regressions",
                "",
                _table(
                    ["Track", "Regression", "No DrumPrint", "DrumPrint"],
                    [
                        [
                            Path(str(row.get("track", ""))).name,
                            f"{abs(float(row.get('improvement_ms', 0.0))):.1f} ms",
                            _fmt((row.get("no_drumprint") or {}).get("time"), 3),
                            _fmt((row.get("drumprint") or {}).get("time"), 3),
                        ]
                        for row in worst[:8]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## MicroSnap A/B",
            "",
            _table(
                ["Metric", "Value"],
                [
                    ["Referenced tracks", microalign.get("referenced_tracks", 0)],
                    ["Improved", microalign.get("improved_count", 0)],
                    ["Worsened", microalign.get("worsened_count", 0)],
                    ["Unchanged", microalign.get("unchanged_count", 0)],
                    ["Median error before", f"{float(microalign.get('median_error_before_ms', 0.0)):.2f} ms"],
                    ["Median error after", f"{float(microalign.get('median_error_after_ms', 0.0)):.2f} ms"],
                    ["Within 10ms before/after", f"{float(microalign.get('within_10ms_before', 0.0)):.2f}% / {float(microalign.get('within_10ms_after', 0.0)):.2f}%"],
                    ["Within 25ms before/after", f"{float(microalign.get('within_25ms_before', 0.0)):.2f}% / {float(microalign.get('within_25ms_after', 0.0)):.2f}%"],
                    ["Auto-accept eligible", microalign.get("auto_accept_eligible_count", 0)],
                    ["Median snap offset", f"{float(microalign.get('median_snap_offset_ms', 0.0)):.2f} ms"],
                    ["Snap offsets over 100ms", f"{float(microalign.get('percent_snap_offsets_over_100ms', 0.0)):.2f}%"],
                    ["Recommendation", microalign.get("recommendation", "needs more data")],
                ],
            ),
            "",
        ]
    )

    micro_best = microalign.get("best_improvements", [])
    micro_worst = microalign.get("worst_regressions", [])
    if micro_best:
        lines.extend(
            [
                "### Best MicroSnap Improvements",
                "",
                _table(
                    ["Track", "Improvement", "No MicroSnap", "MicroSnap"],
                    [
                        [
                            Path(str(row.get("track", ""))).name,
                            f"{float(row.get('improvement_ms', 0.0)):.1f} ms",
                            _fmt((row.get("no_microalign") or {}).get("time"), 3),
                            _fmt((row.get("microalign") or {}).get("time"), 3),
                        ]
                        for row in micro_best[:8]
                    ],
                ),
                "",
            ]
        )
    if micro_worst:
        lines.extend(
            [
                "### Worst MicroSnap Regressions",
                "",
                _table(
                    ["Track", "Regression", "No MicroSnap", "MicroSnap"],
                    [
                        [
                            Path(str(row.get("track", ""))).name,
                            f"{abs(float(row.get('improvement_ms', 0.0))):.1f} ms",
                            _fmt((row.get("no_microalign") or {}).get("time"), 3),
                            _fmt((row.get("microalign") or {}).get("time"), 3),
                        ]
                        for row in micro_worst[:8]
                    ],
                ),
                "",
            ]
        )

    lines.extend(["## Action Recommendations", ""])
    lines.extend(f"- {action}" for action in actions)
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: Path, md_path: Path) -> tuple[str, str]:
    json_path = json_path.expanduser()
    md_path = md_path.expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, indent=2, ensure_ascii=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return str(json_path), str(md_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a drop-aligner pilot run without training or modifying ALS files.")
    parser.add_argument("--batch-summary", default="drop_batch_summary.csv")
    parser.add_argument("--corrections", default="drop_corrections.jsonl")
    parser.add_argument("--evaluation", default="models/evaluation_report.json")
    parser.add_argument("--drumprint-ablation", default="models/drumprint_eval/drumprint_ablation_report.json")
    parser.add_argument("--microalign-ablation", default="models/microalign_eval/microalign_ablation_report.json")
    parser.add_argument("--oracle-diagnostics", default="models/oracle_diagnostics.json")
    parser.add_argument("--gate-rejection-report", default="models/gate_rejection_report.json")
    parser.add_argument("--auto-verifier-report", default="models/auto_verifier_report.json")
    parser.add_argument("--auto-gate-config", default="models/auto_gate_config.json")
    parser.add_argument("--output-json", default="pilot_report.json")
    parser.add_argument("--output-md", default="pilot_report.md")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    batch_summary = Path(args.batch_summary)
    corrections = Path(args.corrections)
    evaluation = Path(args.evaluation)
    drumprint_ablation = Path(args.drumprint_ablation)
    microalign_ablation = Path(args.microalign_ablation)
    oracle_diagnostics = Path(args.oracle_diagnostics)
    gate_rejection_report = Path(args.gate_rejection_report)
    auto_verifier_report = Path(args.auto_verifier_report)
    auto_gate_config = Path(args.auto_gate_config)

    library = summarize_library(batch_summary)
    review = summarize_review(corrections, batch_summary)
    model = summarize_evaluation(evaluation)
    drumprint = summarize_drumprint_ablation(drumprint_ablation)
    microalign = summarize_microalign_ablation(microalign_ablation)
    oracle = summarize_oracle_diagnostics(oracle_diagnostics)
    gate = summarize_gate_rejections(gate_rejection_report)
    verifier = summarize_auto_verifier(auto_verifier_report)
    tuned_gate = summarize_auto_gate_config(auto_gate_config)
    actions = build_action_recommendations(library, review, model, drumprint, microalign)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "batch_summary": str(batch_summary),
            "corrections": str(corrections),
            "evaluation": str(evaluation),
            "drumprint_ablation": str(drumprint_ablation),
            "microalign_ablation": str(microalign_ablation),
            "oracle_diagnostics": str(oracle_diagnostics),
            "gate_rejection_report": str(gate_rejection_report),
            "auto_verifier_report": str(auto_verifier_report),
            "auto_gate_config": str(auto_gate_config),
        },
        "library_processing": library,
        "human_review": review,
        "model_performance": model,
        "drumprint_ab": drumprint,
        "microsnap_ab": microalign,
        "oracle_diagnostics": oracle,
        "gate_rejections": gate,
        "auto_verifier": verifier,
        "auto_gate_config": tuned_gate,
        "action_recommendations": actions,
    }
    json_path, md_path = write_report(report, Path(args.output_json), Path(args.output_md))

    print("Pilot report written")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Reviewed: {review.get('total_reviewed', 0)}")
    print(f"Accepted: {review.get('accepted_ai_picks', 0)}")
    print(f"Corrected: {review.get('corrected_picks', 0)}")
    print(f"ALS validation pass rate: {float(library.get('als_validation_pass_rate', 0.0)):.2f}%")
    print(f"DrumPrint recommendation: {drumprint.get('recommendation', 'needs more data')}")
    print(f"MicroSnap recommendation: {microalign.get('recommendation', 'needs more data')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
