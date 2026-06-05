#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.ranker import candidate_timestamp, load_ranker_payload, predict_candidate_distances
from train_ranker import ACCEPT_TOLERANCE_SEC


THRESHOLDS_SEC = [0.005, 0.010, 0.025, 0.050, 0.100]
CSV_COLUMNS = [
    "track",
    "ai_pick",
    "user_pick",
    "delta",
    "accepted_pick",
    "confidence_tier",
    "handcrafted_timestamp",
    "handcrafted_rank",
    "handcrafted_abs_error",
    "model_timestamp",
    "model_rank",
    "model_score",
    "model_abs_error",
    "closest_timestamp",
    "closest_rank",
    "closest_abs_error",
    "model_selected_closest",
    "handcrafted_selected_closest",
]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    return _float_or_none(row.get("user_pick"))


def _ai_pick(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _candidate_rows(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    source: List[Any] = []
    candidates = row.get("top_10_candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        source.extend(candidates)
    else:
        candidates = row.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            source.extend(candidates)
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        source.append(selected)

    out: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in source:
        if not isinstance(candidate, Mapping):
            continue
        t = candidate_timestamp(candidate)
        key = f"{t:.6f}" if t is not None else json.dumps(dict(candidate), sort_keys=True, default=str)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _confidence_tier(row: Mapping[str, Any]) -> str:
    tier = str(row.get("confidence_tier") or "").strip().upper()
    if not tier:
        features = row.get("features")
        if isinstance(features, Mapping):
            tier = str(features.get("confidence_tier") or "").strip().upper()
    return tier if tier in {"HIGH", "MEDIUM", "LOW"} else "UNKNOWN"


def _rank_value(candidate: Mapping[str, Any], key: str, fallback_key: str = "rank") -> int:
    value = candidate.get(key, candidate.get(fallback_key))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_identity(candidate: Mapping[str, Any]) -> Tuple[int, float]:
    rank = _rank_value(candidate, "handcrafted_rank")
    timestamp = candidate_timestamp(candidate)
    return rank, float(timestamp if timestamp is not None else -1.0)


def _best_handcrafted(candidates: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (_rank_value(c, "handcrafted_rank"), _rank_value(c, "rank")))[0]


def _closest_candidate(candidates: Sequence[Mapping[str, Any]], user_pick: float) -> Optional[Mapping[str, Any]]:
    best = None
    best_error = float("inf")
    for candidate in candidates:
        timestamp = candidate_timestamp(candidate)
        if timestamp is None:
            continue
        error = abs(float(timestamp) - float(user_pick))
        if error < best_error:
            best = candidate
            best_error = error
    return best


def _with_model_predictions(payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(candidate) for candidate in candidates]
    predictions = predict_candidate_distances(payload, out)
    for candidate, prediction in zip(out, predictions):
        candidate["evaluation_model_score"] = float(prediction)
    return out


def _best_model_candidate(candidates: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            _float_or_none(candidate.get("evaluation_model_score")) if _float_or_none(candidate.get("evaluation_model_score")) is not None else float("inf"),
            _rank_value(candidate, "handcrafted_rank"),
            candidate_timestamp(candidate) or float("inf"),
        ),
    )[0]


def _candidate_error(candidate: Optional[Mapping[str, Any]], user_pick: float) -> Optional[float]:
    if not candidate:
        return None
    timestamp = candidate_timestamp(candidate)
    if timestamp is None:
        return None
    return abs(float(timestamp) - float(user_pick))


def _candidate_time(candidate: Optional[Mapping[str, Any]]) -> str:
    if not candidate:
        return ""
    timestamp = candidate_timestamp(candidate)
    return "" if timestamp is None else f"{float(timestamp):.9f}".rstrip("0").rstrip(".")


def _candidate_rank(candidate: Optional[Mapping[str, Any]], key: str = "handcrafted_rank") -> str:
    if not candidate:
        return ""
    rank = _rank_value(candidate, key)
    return str(rank) if rank else ""


def _same_candidate(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> bool:
    if not a or not b:
        return False
    return _candidate_identity(a) == _candidate_identity(b)


def _evaluate_row(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    user_pick = _user_pick(row)
    ai_pick = _ai_pick(row)
    if user_pick is None or ai_pick is None:
        return None
    candidates = _candidate_rows(row)
    if not candidates:
        return None

    predicted_candidates = _with_model_predictions(payload, candidates)
    handcrafted = _best_handcrafted(candidates)
    model = _best_model_candidate(predicted_candidates)
    closest = _closest_candidate(candidates, float(user_pick))

    handcrafted_error = _candidate_error(handcrafted, float(user_pick))
    model_error = _candidate_error(model, float(user_pick))
    closest_error = _candidate_error(closest, float(user_pick))
    if model_error is None:
        return None

    accepted = abs(float(user_pick) - float(ai_pick)) <= ACCEPT_TOLERANCE_SEC
    return {
        "track": str(row.get("track", "")),
        "ai_pick": float(ai_pick),
        "user_pick": float(user_pick),
        "delta": float(user_pick) - float(ai_pick),
        "accepted_pick": bool(accepted),
        "confidence_tier": _confidence_tier(row),
        "handcrafted_timestamp": _candidate_time(handcrafted),
        "handcrafted_rank": _candidate_rank(handcrafted, "handcrafted_rank"),
        "handcrafted_abs_error": "" if handcrafted_error is None else float(handcrafted_error),
        "model_timestamp": _candidate_time(model),
        "model_rank": _candidate_rank(model, "handcrafted_rank"),
        "model_score": "" if not model else _float_or_none(model.get("evaluation_model_score")),
        "model_abs_error": float(model_error),
        "closest_timestamp": _candidate_time(closest),
        "closest_rank": _candidate_rank(closest, "handcrafted_rank"),
        "closest_abs_error": "" if closest_error is None else float(closest_error),
        "model_selected_closest": bool(_same_candidate(model, closest)),
        "handcrafted_selected_closest": bool(_same_candidate(handcrafted, closest)),
    }


def _percent_within(errors: Sequence[float], threshold: float) -> float:
    if not errors:
        return 0.0
    return float(100.0 * np.mean(np.asarray(errors, dtype=np.float64) <= float(threshold)))


def _rank_accuracy(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(100.0 * np.mean([bool(row.get(key)) for row in rows]))


def _tier_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tier in ("LOW", "MEDIUM", "HIGH", "UNKNOWN"):
        tier_rows = [row for row in rows if row.get("confidence_tier") == tier]
        errors = [float(row["model_abs_error"]) for row in tier_rows]
        deltas = [float(row["delta"]) for row in tier_rows]
        out[tier] = {
            "count": int(len(tier_rows)),
            "accepted_picks": int(sum(1 for row in tier_rows if row.get("accepted_pick"))),
            "corrected_picks": int(sum(1 for row in tier_rows if not row.get("accepted_pick"))),
            "mean_absolute_error_sec": float(np.mean(errors)) if errors else 0.0,
            "median_absolute_error_sec": float(np.median(errors)) if errors else 0.0,
            "average_delta_sec": float(np.mean(deltas)) if deltas else 0.0,
            "average_abs_delta_sec": float(np.mean(np.abs(deltas))) if deltas else 0.0,
            "model_selected_closest_candidate_percent": _rank_accuracy(tier_rows, "model_selected_closest"),
            "handcrafted_selected_closest_candidate_percent": _rank_accuracy(tier_rows, "handcrafted_selected_closest"),
            "percent_within": {
                "5ms": _percent_within(errors, 0.005),
                "10ms": _percent_within(errors, 0.010),
                "25ms": _percent_within(errors, 0.025),
                "50ms": _percent_within(errors, 0.050),
                "100ms": _percent_within(errors, 0.100),
            },
        }
    return out


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
    return str(out)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return str(out)


def evaluate_ranker(
    *,
    corrections: str,
    model: str,
    report_json: str = "models/evaluation_report.json",
    report_csv: str = "models/evaluation_report.csv",
) -> Dict[str, Any]:
    payload = load_ranker_payload(model)
    if payload is None:
        raise FileNotFoundError(f"Model not found: {model}")

    evaluated_rows: List[Dict[str, Any]] = []
    for row in _iter_jsonl(Path(corrections).expanduser()):
        if row_has_excluded_path(row):
            continue
        evaluated = _evaluate_row(row, payload)
        if evaluated is not None:
            evaluated_rows.append(evaluated)

    errors = [float(row["model_abs_error"]) for row in evaluated_rows]
    handcrafted_errors = [
        float(row["handcrafted_abs_error"])
        for row in evaluated_rows
        if row.get("handcrafted_abs_error") not in ("", None)
    ]
    accepted = [row for row in evaluated_rows if row.get("accepted_pick")]
    corrected = [row for row in evaluated_rows if not row.get("accepted_pick")]
    worst = sorted(evaluated_rows, key=lambda row: float(row["model_abs_error"]), reverse=True)[:20]
    tier_metrics = _tier_metrics(evaluated_rows)

    summary = {
        "corrections": str(Path(corrections).expanduser()),
        "model": str(Path(model).expanduser()),
        "total_reviewed_tracks": int(len(evaluated_rows)),
        "accepted_picks": int(len(accepted)),
        "corrected_picks": int(len(corrected)),
        "median_absolute_error_sec": float(np.median(errors)) if errors else 0.0,
        "mean_absolute_error_sec": float(np.mean(errors)) if errors else 0.0,
        "handcrafted_median_absolute_error_sec": float(np.median(handcrafted_errors)) if handcrafted_errors else 0.0,
        "handcrafted_mean_absolute_error_sec": float(np.mean(handcrafted_errors)) if handcrafted_errors else 0.0,
        "percent_within": {
            "5ms": _percent_within(errors, 0.005),
            "10ms": _percent_within(errors, 0.010),
            "25ms": _percent_within(errors, 0.025),
            "50ms": _percent_within(errors, 0.050),
            "100ms": _percent_within(errors, 0.100),
        },
        "handcrafted_rank_accuracy_percent": _rank_accuracy(evaluated_rows, "handcrafted_selected_closest"),
        "model_rank_accuracy_percent": _rank_accuracy(evaluated_rows, "model_selected_closest"),
        "model_selected_closest_candidate_percent": _rank_accuracy(evaluated_rows, "model_selected_closest"),
        "accuracy_by_confidence_tier": tier_metrics,
        "average_delta_by_confidence_tier": {
            tier: metrics["average_delta_sec"]
            for tier, metrics in tier_metrics.items()
        },
        "worst_20_misses": worst,
    }
    json_path = _write_json(report_json, summary)
    csv_path = _write_csv(report_csv, evaluated_rows)
    summary["evaluation_report_json"] = json_path
    summary["evaluation_report_csv"] = csv_path
    return summary


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("\nRanker Evaluation")
    print("=================")
    print(f"Reviewed tracks: {summary['total_reviewed_tracks']}")
    print(f"Accepted picks:  {summary['accepted_picks']}")
    print(f"Corrected picks: {summary['corrected_picks']}")
    print(f"Median abs err:  {summary['median_absolute_error_sec']:.6f}s")
    print(f"Mean abs err:    {summary['mean_absolute_error_sec']:.6f}s")
    print("Within:")
    for label, value in summary["percent_within"].items():
        print(f"  {label:>5}: {value:6.2f}%")
    print(f"Handcrafted closest-candidate accuracy: {summary['handcrafted_rank_accuracy_percent']:.2f}%")
    print(f"Model closest-candidate accuracy:       {summary['model_rank_accuracy_percent']:.2f}%")
    print("\nBy confidence tier:")
    for tier in ("LOW", "MEDIUM", "HIGH", "UNKNOWN"):
        metrics = summary["accuracy_by_confidence_tier"].get(tier, {})
        print(
            f"  {tier:>7}: n={metrics.get('count', 0):>3} "
            f"model_acc={metrics.get('model_selected_closest_candidate_percent', 0.0):6.2f}% "
            f"mean_err={metrics.get('mean_absolute_error_sec', 0.0):.6f}s "
            f"avg_delta={metrics.get('average_delta_sec', 0.0):+.6f}s"
        )
    print("\nWorst misses:")
    for row in summary["worst_20_misses"][:20]:
        print(
            f"  {float(row['model_abs_error']):9.3f}s  "
            f"model={row.get('model_timestamp', '')} user={row.get('user_pick', '')}  "
            f"{row.get('track', '')}"
        )
    print(f"\nJSON: {summary['evaluation_report_json']}")
    print(f"CSV:  {summary['evaluation_report_csv']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Ali's learned drop candidate ranker against correction logs.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL path")
    parser.add_argument("--model", default="models/drop_ranker.pkl", help="Trained ranker pickle path")
    parser.add_argument("--report-json", default="models/evaluation_report.json", help="Output JSON report path")
    parser.add_argument("--report-csv", default="models/evaluation_report.csv", help="Output CSV report path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_ranker(
        corrections=args.corrections,
        model=args.model,
        report_json=args.report_json,
        report_csv=args.report_csv,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
