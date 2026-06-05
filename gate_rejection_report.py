#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_JSON = Path("models/gate_rejection_report.json")
DEFAULT_CSV = Path("models/gate_rejection_report.csv")


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _read_summary(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, Mapping[str, str]]]:
    if not path.exists():
        return [], {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows, {str(row.get("filename", "")): row for row in rows if row.get("filename")}


def _latest_shadow_report() -> Optional[Path]:
    candidates = sorted(Path("eval_reports").glob("*shadow*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _rows_from_shadow(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return [], str(path)
    return [dict(row) for row in rows if isinstance(row, Mapping)], str(path)


def _rows_from_summary(summary_rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    for summary in summary_rows:
        candidates_path = Path(str(summary.get("candidates_json", ""))).expanduser()
        if not candidates_path.exists():
            continue
        try:
            payload = _read_json(candidates_path)
        except Exception:
            continue
        reanalysis = payload.get("multistem_reanalysis") if isinstance(payload.get("multistem_reanalysis"), Mapping) else {}
        gate_map = reanalysis.get("auto_accept") if isinstance(reanalysis.get("auto_accept"), Mapping) else {}
        mode = str(reanalysis.get("mode") or "normal")
        gate = gate_map.get(mode) if isinstance(gate_map.get(mode), Mapping) else {}
        chooser = reanalysis.get("candidate_chooser") if isinstance(reanalysis.get("candidate_chooser"), Mapping) else {}
        rows.append(
            {
                "status": "from_candidates_json",
                "filename": str(summary.get("filename", "")),
                "candidates_json": str(summary.get("candidates_json", "")),
                "suggested_time": reanalysis.get("suggested_time"),
                "diagnostic_reason": str(reanalysis.get("diagnostic_reason", "")),
                "suggestion_reason": str(reanalysis.get("reason", "")),
                "auto_accept_passed": bool(reanalysis.get("auto_accept_passed", False)),
                "gate_reason": str(gate.get("reason", "")),
                "risk_flags": list(gate.get("risk_flags") or []) if isinstance(gate.get("risk_flags"), list) else [],
                "source_count": reanalysis.get("source_count"),
                "candidate_count": reanalysis.get("candidate_count"),
                "pipeline_output_count": (reanalysis.get("pipeline") or {}).get("output_count") if isinstance(reanalysis.get("pipeline"), Mapping) else None,
                "pipeline_deduped_count": (reanalysis.get("pipeline") or {}).get("deduped_count") if isinstance(reanalysis.get("pipeline"), Mapping) else None,
                "model_path": chooser.get("model_path"),
                "model_type": chooser.get("model_type"),
                "selection_probability": chooser.get("selection_probability"),
                "probability_margin": chooser.get("probability_margin"),
                "selection_confidence": chooser.get("selection_confidence"),
                "predicted_abs_error_sec": chooser.get("predicted_abs_error_sec"),
                "micro_confidence": gate.get("micro_confidence"),
                "snap_offset_ms": gate.get("snap_offset_ms"),
            }
        )
    return rows, "candidates_json"


def _reason_categories(row: Mapping[str, Any]) -> List[str]:
    flags = row.get("risk_flags")
    if isinstance(flags, list) and flags:
        text_items = [str(flag).lower() for flag in flags]
    else:
        text_items = [str(row.get("gate_reason") or "").lower()]
    diagnostic = str(row.get("diagnostic_reason") or "").lower()
    if diagnostic:
        text_items.append(diagnostic)
    text = " | ".join(text_items)
    categories: List[str] = []
    if "no candidate sources generated" in text:
        categories.append("no candidate sources generated")
    if "no candidates survived" in text:
        categories.append("no generated candidates survived scoring")
    if "drop pipeline returned no ranked candidates" in text:
        categories.append("drop pipeline produced no ranked candidates")
    if "no suggestion candidate" in text or "no eligible microaligned candidate" in text:
        categories.append("no eligible MicroSnap suggestion")
    if "verifier probability" in text:
        categories.append("low auto-verifier probability")
    if "verifier predicted error" in text:
        categories.append("auto-verifier predicted error too high")
    if "learned probability" in text or "probability " in text:
        categories.append("low model probability")
    if "learned confidence" in text:
        categories.append("low learned confidence")
    if "margin" in text:
        categories.append("low candidate margin")
    if "predicted error" in text:
        categories.append("predicted error too high")
    if "micro confidence" in text:
        categories.append("low MicroSnap confidence")
    if "snap offset" in text:
        categories.append("large MicroSnap snap offset")
    if "fake" in text:
        categories.append("high fake-hit penalty")
    if "drumprint" in text or "drum print" in text:
        categories.append("weak DrumPrint score")
    if "disagree" in text:
        categories.append("model/handcrafted disagreement")
    if "groove" in text:
        categories.append("unstable post-drop groove")
    if "low-end" in text or "low end" in text or "bass" in text:
        categories.append("no strong low-end confirmation")
    if "missing" in text and "drumprint" in text:
        categories.append("missing DrumPrint data")
    if "missing" in text and "micro" in text:
        categories.append("missing MicroSnap data")
    if "als" in text:
        categories.append("ALS verification issue")
    if not categories:
        categories.append("other review gate")
    return categories


def _near_pass_score(row: Mapping[str, Any]) -> float:
    if bool(row.get("auto_accept_passed")):
        return 1.0
    probability = _safe_float(row.get("selection_probability"))
    confidence = _safe_float(row.get("selection_confidence"))
    predicted_error = _safe_float(row.get("predicted_abs_error_sec"))
    micro = _safe_float(row.get("micro_confidence"))
    snap_offset = abs(_safe_float(row.get("snap_offset_ms")) or 0.0)
    margin = _safe_float(row.get("probability_margin"))
    deficits = []
    if probability is not None:
        deficits.append(max(0.0, 0.80 - probability) / 0.80)
    if confidence is not None:
        deficits.append(max(0.0, 0.77 - confidence) / 0.77)
    if predicted_error is not None:
        deficits.append(max(0.0, predicted_error - 0.025) / 0.100)
    if micro is not None:
        deficits.append(max(0.0, 0.93 - micro) / 0.93)
    if margin is not None:
        deficits.append(max(0.0, 0.10 - margin) / 0.10)
    deficits.append(max(0.0, snap_offset - 45.0) / 220.0)
    if not deficits:
        return 0.0
    return _clip01(1.0 - (sum(_clip01(value) for value in deficits) / float(len(deficits))))


def _review_value_score(row: Mapping[str, Any], categories: Sequence[str]) -> float:
    near_pass = _near_pass_score(row)
    score = 0.60 * near_pass
    if "no candidate sources generated" in categories:
        score += 0.20
    if "no generated candidates survived scoring" in categories:
        score += 0.18
    if "drop pipeline produced no ranked candidates" in categories:
        score += 0.16
    if "no eligible MicroSnap suggestion" in categories:
        score += 0.12
    if "model/handcrafted disagreement" in categories:
        score += 0.15
    if "high fake-hit penalty" in categories:
        score += 0.12
    if "predicted error too high" in categories:
        score += 0.08
    if "low MicroSnap confidence" in categories:
        score += 0.05
    if int(_safe_float(row.get("pipeline_deduped_count")) or 0) > 0:
        score += 0.04
    return _clip01(score)


def _summary_enrich(row: Dict[str, Any], summary_by_track: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    summary = summary_by_track.get(str(row.get("filename", "")), {})
    out = dict(row)
    out["confidence_tier"] = str(summary.get("confidence_tier", ""))
    out["selected_by"] = str(summary.get("selected_by", ""))
    out["detected_drop_time"] = str(summary.get("detected_drop_time", ""))
    out["als_valid"] = str(summary.get("als_valid", ""))
    return out


def build_gate_rejection_report(
    *,
    summary: str,
    shadow_report: Optional[str],
    output_json: str,
    output_csv: str,
) -> Dict[str, Any]:
    summary_path = Path(summary).expanduser()
    summary_rows, summary_by_track = _read_summary(summary_path)
    source_path: Optional[Path] = Path(shadow_report).expanduser() if shadow_report else _latest_shadow_report()
    if source_path and source_path.exists():
        source_rows, source_name = _rows_from_shadow(source_path)
    else:
        source_rows, source_name = _rows_from_summary(summary_rows)

    rows: List[Dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    gate_reason_counts: Counter[str] = Counter()
    rejected = 0
    accepted = 0
    for raw in source_rows:
        row = _summary_enrich(dict(raw), summary_by_track)
        passed = bool(row.get("auto_accept_passed", False))
        if passed:
            accepted += 1
            categories = ["auto accepted"]
        else:
            rejected += 1
            categories = _reason_categories(row)
        for category in categories:
            category_counts[category] += 1
        gate_reason = str(row.get("gate_reason") or "")
        if gate_reason:
            gate_reason_counts[gate_reason] += 1
        near_pass = _near_pass_score(row)
        value_score = _review_value_score(row, categories)
        row["auto_gate_near_pass_score"] = float(near_pass)
        row["expected_value_score"] = float(value_score)
        row["rejection_categories"] = "; ".join(categories)
        row["top_failure_reason"] = categories[0] if categories else ""
        rows.append(row)

    rejected_rows = [row for row in rows if not bool(row.get("auto_accept_passed", False))]
    near_pass_rows = sorted(rejected_rows, key=lambda row: (-float(row.get("auto_gate_near_pass_score", 0.0)), str(row.get("filename", ""))))[:100]
    review_targets = sorted(rejected_rows, key=lambda row: (-float(row.get("expected_value_score", 0.0)), str(row.get("filename", ""))))[:100]
    report = {
        "summary": str(summary_path),
        "source": source_name,
        "total_rows": int(len(rows)),
        "auto_accepted": int(accepted),
        "total_rejected": int(rejected),
        "rejection_reasons_ranked": category_counts.most_common(),
        "gate_reasons_ranked": gate_reason_counts.most_common(50),
        "rows": rows,
        "top_100_near_pass_tracks": near_pass_rows,
        "top_100_highest_value_review_targets": review_targets,
    }

    json_path = Path(output_json).expanduser()
    csv_path = Path(output_csv).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    fieldnames = [
        "filename",
        "candidates_json",
        "status",
        "auto_accept_passed",
        "top_failure_reason",
        "rejection_categories",
        "auto_gate_near_pass_score",
        "expected_value_score",
        "gate_reason",
        "diagnostic_reason",
        "suggestion_reason",
        "auto_verifier_p_within_25ms",
        "auto_verifier_predicted_abs_error_sec",
        "auto_verifier_mode",
        "selection_probability",
        "probability_margin",
        "selection_confidence",
        "predicted_abs_error_sec",
        "micro_confidence",
        "snap_offset_ms",
        "candidate_count",
        "pipeline_output_count",
        "pipeline_deduped_count",
        "confidence_tier",
        "selected_by",
        "detected_drop_time",
        "suggested_time",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-float(item.get("expected_value_score", 0.0)), str(item.get("filename", "")))):
            writer.writerow(row)
    report["output_json"] = str(json_path)
    report["output_csv"] = str(csv_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain why tracks are held out of autonomous auto-save.")
    parser.add_argument("summary", nargs="?", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv path")
    parser.add_argument("--shadow-report", help="Shadow report from reanalyze_remaining_hard_cases.py. Defaults to latest eval_reports/*shadow*.json")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON), help="Output JSON report")
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV), help="Output CSV report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_gate_rejection_report(
        summary=str(args.summary),
        shadow_report=str(args.shadow_report) if args.shadow_report else None,
        output_json=str(args.output_json),
        output_csv=str(args.output_csv),
    )
    compact = {
        "total_rows": report.get("total_rows"),
        "auto_accepted": report.get("auto_accepted"),
        "total_rejected": report.get("total_rejected"),
        "top_rejection_reasons": report.get("rejection_reasons_ranked", [])[:10],
        "source": report.get("source"),
        "output_json": report.get("output_json"),
        "output_csv": report.get("output_csv"),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
