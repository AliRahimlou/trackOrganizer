from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


KNOWN_VISUAL_REGRESSIONS: List[Dict[str, Any]] = [
    {
        "id": "oracles_video_manual_111",
        "drums_path": (
            "/Users/alirahimlou/Desktop/MUSIC/STEMS/140/4A/"
            "The Widdler - Oracles/drums_140_4A_6-The Widdler - Oracles.flac"
        ),
        "expected_marker_sec": 54.857142857142854,
        "tolerance_sec": 0.002,
        "taxonomy": "known_manual_video_regression",
        "reason": "Manual video review placed the true drop at 54.857142857s, not the later detector hit.",
        "source": "Screen Recording 2026-06-07 at 11.26.15 AM.mov",
        "needs_manual_confirmation": False,
    }
]


TAXONOMY_PRIORITY = [
    "manual_review_mismatch",
    "blank_waveform_marker",
    "off_grid_or_not_on_one",
    "late_inside_body_or_tail",
    "pre_hit_or_before_body",
    "later_than_first_true_drop",
    "fake_or_weak_first_drop",
    "weak_or_missing_visual_body",
    "detector_drift",
    "unsafe_or_stale_source",
    "stale_or_incomplete_proof",
    "als_mapping_error",
    "missing_marker_or_source",
    "unknown_hold",
]


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _split_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    raw = row.get("reasons")
    if isinstance(raw, str):
        reasons.extend(part.strip() for part in raw.split(";") if part.strip())
    elif isinstance(raw, Iterable):
        reasons.extend(str(part).strip() for part in raw if str(part).strip())
    fail_flags = str(row.get("fail_flags") or "")
    warn_flags = str(row.get("warn_flags") or "")
    for value in (fail_flags, warn_flags):
        reasons.extend(part.strip() for part in value.split(";") if part.strip())
    audit_flags = row.get("audit_flags")
    if isinstance(audit_flags, Iterable) and not isinstance(audit_flags, (str, bytes)):
        reasons.extend(f"audit_flag:{flag}" for flag in audit_flags if str(flag))
    return reasons


def classify_failure(row: Mapping[str, Any]) -> List[str]:
    """Map low-level validation reasons into DJ-actionable failure families."""

    text = " | ".join(_split_reasons(row)).lower()
    labels: List[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    if "human_review" in text or "manual_review" in text or "validated_hard_mismatch" in text:
        add("manual_review_mismatch")
    if (
        "blank_marker" in text
        or "no_visible_signal" in text
        or "marker_has_no_signal" in text
        or "marker_signal_present" in text and "false" in text
    ):
        add("blank_waveform_marker")
    if (
        "one_distance_ms" in text
        or "boom:off_one" in text
        or "off_one" in text
        or "grid_on_one_contract" in text
        or "missing_grid_one_evidence" in text
        or "beat two" in text
        or "beat three" in text
        or "beat four" in text
    ):
        add("off_grid_or_not_on_one")
    if (
        "late_inside_body" in text
        or "after boom front edge" in text
        or "stale_front_edge" in text
        or "wide_exact_boom_front_edge_mismatch" in text
        or "inside_body_without_front_edge" in text
        or "persisted_proof_front_edge_offset" in text
    ):
        add("late_inside_body_or_tail")
    if "before boom front edge" in text or "pre_hit" in text or "before_body_front_edge" in text:
        add("pre_hit_or_before_body")
    if "earlier_dominant_boom_available" in text or "earlier_stronger_boom_front_edge" in text:
        add("later_than_first_true_drop")
    if "later_much_stronger_boom_front_edge" in text or "fake" in text:
        add("fake_or_weak_first_drop")
    if (
        "actual_visual_body" in text
        or "no_immediate_drop_body" in text
        or "profile_below_threshold" in text
        or "weak_selected" in text
        or "no_body_section_candidate" in text
    ):
        add("weak_or_missing_visual_body")
    if "rerun_marker_mismatch" in text or "detector_not_ok" in text or "detector_error" in text:
        add("detector_drift")
    if "unsafe_source" in text or "historical_" in text or "saved_" in text:
        add("unsafe_or_stale_source")
    if (
        "persisted_" in text
        or "stale_missing" in text
        or "missing_gui" in text
        or "missing_mask" in text
        or "not_relevant" in text
    ):
        add("stale_or_incomplete_proof")
    if "combined_als" in text or "anchor_mismatch" in text or "file_ref_mismatch" in text:
        add("als_mapping_error")
    if "missing_marker" in text or "missing_drums_path" in text or "missing_marker_or_drums_path" in text:
        add("missing_marker_or_source")

    return labels or ["unknown_hold"]


def primary_taxonomy(labels: Sequence[str]) -> str:
    label_set = set(labels)
    for label in TAXONOMY_PRIORITY:
        if label in label_set:
            return label
    return labels[0] if labels else "unknown_hold"


def build_failure_inventory(
    failures: Sequence[Mapping[str, Any]],
    *,
    source_report: str = "",
) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for fallback_index, row in enumerate(failures, start=1):
        labels = classify_failure(row)
        boom = row.get("boom_proof") if isinstance(row.get("boom_proof"), Mapping) else {}
        gui = row.get("gui_mask") if isinstance(row.get("gui_mask"), Mapping) else {}
        nearest = boom.get("nearest") if isinstance(boom.get("nearest"), Mapping) else {}
        item = {
            "index": int(row.get("index") or fallback_index),
            "source_report": source_report,
            "taxonomy": primary_taxonomy(labels),
            "taxonomies": labels,
            "reasons": _split_reasons(row),
            "track": row.get("track", ""),
            "drums_path": row.get("track") or row.get("drums_path") or "",
            "marker_sec": _finite_float(row.get("marker")),
            "report_marker_sec": _finite_float(row.get("report_marker")),
            "suggested_marker_sec": _finite_float(row.get("suggested_marker_sec")),
            "selected_by": row.get("selected_by", ""),
            "audit_status": row.get("audit_status", ""),
            "audit_flags": row.get("audit_flags") or [],
            "boom_nearest_edge_sec": _finite_float(row.get("boom_nearest_edge") or nearest.get("edge_time")),
            "boom_edge_offset_sec": _finite_float(row.get("boom_edge_offset_sec") or nearest.get("offset_sec")),
            "boom_profile_score": _finite_float(row.get("boom_profile_score")),
            "gui_placeable_count": gui.get("placeable_count"),
            "gui_reasons": gui.get("reasons") if isinstance(gui.get("reasons"), list) else [],
            "candidates_json": row.get("candidates_json", ""),
            "output_als": row.get("output_als", ""),
        }
        inventory.append(item)
    return inventory


def build_regression_seed_payload(
    inventory: Sequence[Mapping[str, Any]],
    *,
    max_proposed: int = 200,
) -> Dict[str, Any]:
    proposed: List[Dict[str, Any]] = []
    for item in inventory:
        suggested = _finite_float(item.get("suggested_marker_sec") or item.get("boom_nearest_edge_sec"))
        marker = _finite_float(item.get("marker_sec"))
        drums_path = str(item.get("drums_path") or "").strip()
        if suggested is None or not drums_path:
            continue
        if marker is not None and abs(marker - suggested) < 0.002:
            continue
        proposed.append(
            {
                "id": f"validation_row_{int(item.get('index') or len(proposed) + 1)}",
                "drums_path": drums_path,
                "expected_marker_sec": suggested,
                "current_marker_sec": marker,
                "tolerance_sec": 0.025,
                "taxonomy": item.get("taxonomy") or "unknown_hold",
                "taxonomies": list(item.get("taxonomies") or []),
                "reason": ";".join(str(reason) for reason in item.get("reasons") or []),
                "source": item.get("source_report") or "production_validation",
                "needs_manual_confirmation": True,
                "candidates_json": item.get("candidates_json", ""),
            }
        )
        if len(proposed) >= max(0, int(max_proposed)):
            break

    return {
        "known_regressions": list(KNOWN_VISUAL_REGRESSIONS),
        "proposed_regression_seeds": proposed,
        "proposed_count": len(proposed),
        "note": (
            "Known regressions are ground truth. Proposed seeds are detector/validator suggestions "
            "and must be manually confirmed before becoming hard expected markers."
        ),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
