#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from gate_rejection_report import DEFAULT_SUMMARY, build_gate_rejection_report


DEFAULT_GATE_REPORT = Path("models/gate_rejection_report.json")
DEFAULT_ORACLE_REPORT = Path("models/oracle_diagnostics.json")
DEFAULT_OUTPUT = Path("models/active_review_queue.csv")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out)


def _oracle_cases(path: Path) -> Dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    out: Dict[str, Mapping[str, Any]] = {}
    for key in (
        "tracks_where_no_candidate_within_100ms",
        "tracks_where_correct_candidate_existed_but_model_chose_wrong",
        "tracks_where_candidate_was_correct_but_gate_rejected",
    ):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping) and row.get("track"):
                out[str(row.get("track"))] = row
    return out


def _queue_reason(row: Mapping[str, Any], oracle_row: Optional[Mapping[str, Any]]) -> str:
    if oracle_row:
        case_type = str(oracle_row.get("case_type") or "")
        if case_type == "candidate_generation_miss":
            return "candidate generation miss"
        if case_type == "ranking_miss":
            return "ranking miss with correct candidate present"
        if case_type == "gate_rejected_correct_selection":
            return "gate rejected correct selection"
    category = str(row.get("top_failure_reason") or "")
    if category:
        return category
    return "held by auto-save gate"


def _queue_score(row: Mapping[str, Any], oracle_row: Optional[Mapping[str, Any]]) -> float:
    score = _safe_float(row.get("expected_value_score"), 0.0)
    near_pass = _safe_float(row.get("auto_gate_near_pass_score"), 0.0)
    if oracle_row:
        case_type = str(oracle_row.get("case_type") or "")
        if case_type == "gate_rejected_correct_selection":
            score += 0.20
        elif case_type == "ranking_miss":
            score += 0.18
        elif case_type == "candidate_generation_miss":
            score += 0.10
    if near_pass >= 0.85:
        score += 0.12
    return max(0.0, min(1.0, score))


def build_active_learning_queue(
    *,
    summary: str,
    gate_report: str,
    oracle_report: str,
    output: str,
    shadow_report: Optional[str] = None,
) -> Dict[str, Any]:
    gate_path = Path(gate_report).expanduser()
    if not gate_path.exists():
        build_gate_rejection_report(
            summary=str(summary),
            shadow_report=shadow_report,
            output_json=str(gate_path),
            output_csv=str(gate_path.with_suffix(".csv")),
        )
    gate_payload = _read_json(gate_path)
    gate_rows = gate_payload.get("rows") if isinstance(gate_payload.get("rows"), list) else []
    oracle_by_track = _oracle_cases(Path(oracle_report).expanduser())
    rows: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for raw in gate_rows:
        if not isinstance(raw, Mapping):
            continue
        if bool(raw.get("auto_accept_passed", False)):
            continue
        track = str(raw.get("filename") or "")
        oracle_row = oracle_by_track.get(track)
        queue_score = _queue_score(raw, oracle_row)
        reason = _queue_reason(raw, oracle_row)
        reason_counts[reason] += 1
        rows.append(
            {
                "track": track,
                "candidates_json": str(raw.get("candidates_json") or ""),
                "reason_to_review": reason,
                "expected_value_score": f"{queue_score:.6f}",
                "top_failure_reason": str(raw.get("top_failure_reason") or ""),
                "confidence_tier": str(raw.get("confidence_tier") or ""),
                "auto_gate_near_pass_score": f"{_safe_float(raw.get('auto_gate_near_pass_score'), 0.0):.6f}",
                "model_disagreement_score": "1.000000" if "disagreement" in str(raw.get("rejection_categories") or "").lower() else "0.000000",
                "fake_hit_risk": "1.000000" if "fake-hit" in str(raw.get("rejection_categories") or "").lower() else "0.000000",
                "oracle_case_type": str(oracle_row.get("case_type") if oracle_row else ""),
                "selection_probability": raw.get("selection_probability", ""),
                "probability_margin": raw.get("probability_margin", ""),
                "selection_confidence": raw.get("selection_confidence", ""),
                "predicted_abs_error_sec": raw.get("predicted_abs_error_sec", ""),
                "micro_confidence": raw.get("micro_confidence", ""),
                "snap_offset_ms": raw.get("snap_offset_ms", ""),
                "suggested_time": raw.get("suggested_time", ""),
            }
        )
    rows.sort(key=lambda row: (-_safe_float(row.get("expected_value_score"), 0.0), str(row.get("track", ""))))

    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "candidates_json",
        "reason_to_review",
        "expected_value_score",
        "top_failure_reason",
        "confidence_tier",
        "auto_gate_near_pass_score",
        "model_disagreement_score",
        "fake_hit_risk",
        "oracle_case_type",
        "selection_probability",
        "probability_margin",
        "selection_confidence",
        "predicted_abs_error_sec",
        "micro_confidence",
        "snap_offset_ms",
        "suggested_time",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "summary": str(Path(summary).expanduser()),
        "gate_report": str(gate_path),
        "oracle_report": str(Path(oracle_report).expanduser()),
        "output": str(out_path),
        "queued_tracks": int(len(rows)),
        "top_reasons": dict(reason_counts.most_common(20)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank held tracks by expected training value for the next review pass.")
    parser.add_argument("summary", nargs="?", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv path")
    parser.add_argument("--gate-report", default=str(DEFAULT_GATE_REPORT), help="Gate rejection JSON report")
    parser.add_argument("--oracle-report", default=str(DEFAULT_ORACLE_REPORT), help="Oracle diagnostics JSON report")
    parser.add_argument("--shadow-report", help="Optional shadow report used if the gate report must be built")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output active learning queue CSV")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_active_learning_queue(
        summary=str(args.summary),
        gate_report=str(args.gate_report),
        oracle_report=str(args.oracle_report),
        output=str(args.output),
        shadow_report=str(args.shadow_report) if args.shadow_report else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
