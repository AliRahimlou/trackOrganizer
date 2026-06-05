#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.learning import log_correction


TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}
PRODUCTION_MODEL = Path("models/drop_ranker.pkl")
CANDIDATE_MODEL = Path("models/drop_ranker_candidate.pkl")
PREVIOUS_MODEL = Path("models/drop_ranker_previous.pkl")
PROMOTION_REPORT = Path("models/promotion_report.json")
CANDIDATE_TRAINING_REPORT = Path("models/training_report_candidate.json")
PRODUCTION_EVAL_JSON = Path("models/evaluation_report_production.json")
PRODUCTION_EVAL_CSV = Path("models/evaluation_report_production.csv")
CANDIDATE_EVAL_JSON = Path("models/evaluation_report_candidate.json")
CANDIDATE_EVAL_CSV = Path("models/evaluation_report_candidate.csv")


def parse_timestamp(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            seconds = (minutes * 60.0) + seconds
        elif len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            seconds = (hours * 3600.0) + (minutes * 60.0) + seconds
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if seconds < 0:
        raise ValueError(f"Timestamp must be positive: {value!r}")
    return float(seconds)


def format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    minutes = int(seconds // 60)
    sec = seconds - (minutes * 60)
    return f"{minutes}:{sec:06.3f}"


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_metric(candidate: Mapping[str, Any], feature_summary: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        for nested_name in ("drumprint", "full_groove", "microalign"):
            nested = candidate.get(nested_name)
            if isinstance(nested, Mapping):
                value = nested.get(key)
                if value is not None:
                    break
    if value is None:
        value = feature_summary.get(f"chosen_{key}")
    return _float_or_none(value) or 0.0


def _load_csv(path: str) -> List[Dict[str, str]]:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Batch summary not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_candidate_payload(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    candidate_path = Path(path).expanduser()
    if not candidate_path.exists():
        return {}
    with open(candidate_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").strip().upper()
    return tier if tier in TIER_ORDER else "UNKNOWN"


def _row_confidence_tier(row: Mapping[str, str], payload: Optional[Mapping[str, Any]] = None) -> str:
    if row.get("confidence_tier"):
        return _normalize_tier(row.get("confidence_tier"))
    if payload:
        if payload.get("confidence_tier"):
            return _normalize_tier(payload.get("confidence_tier"))
        features = payload.get("feature_summary")
        if isinstance(features, Mapping):
            return _normalize_tier(features.get("confidence_tier"))
    return "UNKNOWN"


def _filter_and_sort_rows(rows: Sequence[Dict[str, str]], *, low_only: bool, medium_and_low: bool) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for row in rows:
        if row_has_excluded_path(row):
            continue
        payload = _load_candidate_payload(row.get("candidates_json", "")) if not row.get("confidence_tier") else {}
        tier = _row_confidence_tier(row, payload)
        if low_only and tier != "LOW":
            continue
        if medium_and_low and tier not in {"LOW", "MEDIUM"}:
            continue
        item = dict(row)
        item["confidence_tier"] = tier
        prepared.append(item)
    return sorted(prepared, key=lambda row: (TIER_ORDER.get(row.get("confidence_tier", "UNKNOWN"), 3), row.get("filename", "")))


def _open_png(path: str) -> None:
    if not path:
        print("  debug PNG: missing from CSV")
        return
    png = Path(path).expanduser()
    if not png.exists():
        print(f"  debug PNG missing: {png}")
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(png)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.name == "nt":
            os.startfile(str(png))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(png)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"  could not open debug PNG: {exc}")


def _top_candidates(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = payload.get("top_10_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _selected_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_candidate")
    return selected if isinstance(selected, Mapping) else {}


def _print_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    print("  top candidates:")
    if not candidates:
        print("    no candidate JSON available")
        return
    for candidate in candidates[:10]:
        rank = candidate.get("rank", "")
        handcrafted_rank = candidate.get("handcrafted_rank", "")
        model_rank = candidate.get("model_rank", "")
        timestamp = _float_or_none(candidate.get("timestamp"))
        score = _float_or_none(candidate.get("confidence_score", candidate.get("score")))
        model_score = _float_or_none(candidate.get("model_score"))
        full_groove = _candidate_metric(candidate, {}, "sustained_full_groove_score")
        immediate_groove = _candidate_metric(candidate, {}, "immediate_groove_start_score")
        groove_stability = _candidate_metric(candidate, {}, "groove_stability")
        pre_drop_contrast = _candidate_metric(candidate, {}, "pre_drop_contrast")
        drumprint_score = _candidate_metric(candidate, {}, "drumprint_pattern_score")
        fake_hit_penalty = _candidate_metric(candidate, {}, "fake_hit_penalty")
        later_match = _candidate_metric(candidate, {}, "later_drop_match_score")
        selected = "*" if candidate.get("selected") else " "
        reason = str(candidate.get("reason", ""))
        print(
            f"   {selected}#{rank:>2} t={format_time(timestamp):>9} "
            f"score={score if score is not None else '':>7} "
            f"h={handcrafted_rank!s:>2} m={model_rank!s:>2} "
            f"model={model_score if model_score is not None else '':>7} "
            f"groove={full_groove:.3f} imm={immediate_groove:.3f} stable={groove_stability:.3f} contrast={pre_drop_contrast:.3f} "
            f"dp={drumprint_score:.3f} fake={fake_hit_penalty:.3f} later={later_match:.3f} "
            f"{reason}"
        )


def _print_detection_summary(candidate: Mapping[str, Any], features: Mapping[str, Any]) -> None:
    print(f"Full groove score: {_candidate_metric(candidate, features, 'sustained_full_groove_score'):.3f}")
    print(f"Immediate groove: {_candidate_metric(candidate, features, 'immediate_groove_start_score'):.3f}")
    print(f"Groove stability: {_candidate_metric(candidate, features, 'groove_stability'):.3f}")
    print(f"Pre-drop contrast: {_candidate_metric(candidate, features, 'pre_drop_contrast'):.3f}")
    print(f"DrumPrint score: {_candidate_metric(candidate, features, 'drumprint_pattern_score'):.3f}")
    print(f"Pattern stability: {_candidate_metric(candidate, features, 'post_drop_pattern_stability'):.3f}")
    print(f"Fake hit penalty: {_candidate_metric(candidate, features, 'fake_hit_penalty'):.3f}")
    print(f"Later match score: {_candidate_metric(candidate, features, 'later_drop_match_score'):.3f}")


def _row_ai_pick(row: Mapping[str, str], payload: Mapping[str, Any]) -> Optional[float]:
    value = _float_or_none(row.get("detected_drop_time"))
    if value is not None:
        return value
    return _float_or_none(payload.get("final_ai_pick"))


def _review_row(row: Mapping[str, str], corrections_path: str) -> str:
    payload = _load_candidate_payload(row.get("candidates_json", ""))
    ai_pick = _row_ai_pick(row, payload)
    if ai_pick is None:
        print(f"\nSkipping row without detected drop time: {row.get('filename', '')}")
        return "skip"

    track = row.get("filename") or str(payload.get("track", ""))
    confidence = row.get("confidence") or str(payload.get("confidence", ""))
    confidence_tier = _row_confidence_tier(row, payload)
    selected_by = row.get("selected_by") or str(payload.get("selected_by", ""))
    candidates = _top_candidates(payload)
    selected_candidate = _selected_candidate(payload)
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}

    print("\n" + "=" * 88)
    print(f"track: {track}")
    print(f"AI pick: {format_time(ai_pick)} ({ai_pick:.6f}s)")
    print(f"confidence: {confidence}")
    print(f"confidence_tier: {confidence_tier}")
    print(f"selected_by: {selected_by}")
    _print_detection_summary(selected_candidate, features)
    _print_candidates(candidates)
    _open_png(row.get("debug_png", ""))

    while True:
        answer = input("Enter=accept, timestamp=correct, s=skip, q=quit > ").strip()
        if answer == "":
            user_pick = float(ai_pick)
            action = "accepted"
        elif answer.lower() == "s":
            return "skip"
        elif answer.lower() == "q":
            return "quit"
        else:
            try:
                user_pick = parse_timestamp(answer)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            action = "corrected"

        log_correction(
            track=track,
            ai_pick=float(ai_pick),
            user_pick=float(user_pick),
            features=dict(features),
            top_candidates=candidates,
            selected_candidate=selected_candidate,
            selected_by=selected_by,
            confidence_tier=confidence_tier,
            log_path=corrections_path,
        )
        delta = user_pick - float(ai_pick)
        print(f"  wrote {action}: user_pick={format_time(user_pick)} delta={delta:+.6f}s")
        return action


def _metric(summary: Optional[Mapping[str, Any]], key: str, default: float = 0.0) -> float:
    if not summary:
        return float(default)
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _percent_25ms(summary: Optional[Mapping[str, Any]]) -> float:
    if not summary:
        return 0.0
    within = summary.get("percent_within")
    if not isinstance(within, Mapping):
        return 0.0
    try:
        return float(within.get("25ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _low_tier_accuracy(summary: Optional[Mapping[str, Any]]) -> float:
    if not summary:
        return 0.0
    tiers = summary.get("accuracy_by_confidence_tier")
    if not isinstance(tiers, Mapping):
        return 0.0
    low = tiers.get("LOW")
    if not isinstance(low, Mapping):
        return 0.0
    try:
        return float(low.get("model_selected_closest_candidate_percent", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _low_tier_count(summary: Optional[Mapping[str, Any]]) -> int:
    if not summary:
        return 0
    tiers = summary.get("accuracy_by_confidence_tier")
    if not isinstance(tiers, Mapping):
        return 0
    low = tiers.get("LOW")
    if not isinstance(low, Mapping):
        return 0
    try:
        return int(low.get("count", 0))
    except (TypeError, ValueError):
        return 0


def _passes_promotion_gate(old_metrics: Optional[Mapping[str, Any]], candidate_metrics: Mapping[str, Any]) -> tuple[bool, str]:
    if old_metrics is None:
        return True, "promoted: no existing production model"

    old_mae = _metric(old_metrics, "mean_absolute_error_sec")
    candidate_mae = _metric(candidate_metrics, "mean_absolute_error_sec")
    old_p25 = _percent_25ms(old_metrics)
    candidate_p25 = _percent_25ms(candidate_metrics)
    old_low = _low_tier_accuracy(old_metrics)
    candidate_low = _low_tier_accuracy(candidate_metrics)
    low_rows = max(_low_tier_count(old_metrics), _low_tier_count(candidate_metrics))
    low_regression = max(0.0, old_low - candidate_low)
    one_row_low_tolerance = (100.0 / low_rows) + 1e-9 if low_rows > 0 else 0.0

    mae_improved = candidate_mae < old_mae
    p25_improved = candidate_p25 > old_p25
    material_mae_improvement = old_mae > 0.0 and candidate_mae <= old_mae * 0.90
    low_tolerated = bool(
        low_rows > 0
        and low_regression <= one_row_low_tolerance
        and material_mae_improvement
        and p25_improved
    )
    low_not_regressed = True if low_rows == 0 else candidate_low >= old_low or low_tolerated

    if (mae_improved or p25_improved) and low_not_regressed:
        reasons = []
        if mae_improved:
            reasons.append(f"mean absolute error improved {old_mae:.6f}s -> {candidate_mae:.6f}s")
        if p25_improved:
            reasons.append(f"25ms accuracy improved {old_p25:.2f}% -> {candidate_p25:.2f}%")
        if low_rows > 0:
            if candidate_low >= old_low:
                reasons.append(f"LOW-tier accuracy did not regress {old_low:.2f}% -> {candidate_low:.2f}%")
            else:
                reasons.append(
                    f"LOW-tier accuracy regression tolerated {old_low:.2f}% -> {candidate_low:.2f}% "
                    f"because MAE improved materially"
                )
        else:
            reasons.append("no LOW-tier rows in evaluation set")
        return True, "promoted: " + "; ".join(reasons)

    reasons = []
    if not (mae_improved or p25_improved):
        reasons.append(
            f"no improvement: MAE {old_mae:.6f}s -> {candidate_mae:.6f}s, "
            f"25ms {old_p25:.2f}% -> {candidate_p25:.2f}%"
        )
    if not low_not_regressed:
        reasons.append(f"LOW-tier accuracy regressed {old_low:.2f}% -> {candidate_low:.2f}%")
    return False, "not promoted: " + "; ".join(reasons)


def _write_promotion_report(report: Mapping[str, Any], path: Path = PROMOTION_REPORT) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, indent=2, ensure_ascii=True)
    return str(path)


def _promote_candidate_model() -> None:
    PRODUCTION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if PRODUCTION_MODEL.exists():
        shutil.copy2(PRODUCTION_MODEL, PREVIOUS_MODEL)
    os.replace(CANDIDATE_MODEL, PRODUCTION_MODEL)


def _run_retrain(corrections_path: str) -> int:
    from evaluate_ranker import evaluate_ranker
    from train_candidate_chooser import train_candidate_chooser
    from train_ranker import train_ranker

    print("\nTraining candidate ranker:")
    print(f"{sys.executable} train_ranker.py --corrections {corrections_path} --output {CANDIDATE_MODEL}")

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corrections": corrections_path,
        "production_model": str(PRODUCTION_MODEL),
        "candidate_model": str(CANDIDATE_MODEL),
        "previous_model": str(PREVIOUS_MODEL),
        "promoted": False,
        "reason": "",
        "old_metrics": None,
        "candidate_metrics": None,
    }

    try:
        training_summary = train_ranker(
            corrections=corrections_path,
            output=str(CANDIDATE_MODEL),
            report=str(CANDIDATE_TRAINING_REPORT),
        )
        report["candidate_training"] = training_summary
    except Exception as exc:
        report["reason"] = f"candidate training failed: {exc}"
        report_path = _write_promotion_report(report)
        print(f"Promotion report: {report_path}")
        return 1

    candidate_metrics: Optional[Mapping[str, Any]] = None
    old_metrics: Optional[Mapping[str, Any]] = None
    try:
        candidate_metrics = evaluate_ranker(
            corrections=corrections_path,
            model=str(CANDIDATE_MODEL),
            report_json=str(CANDIDATE_EVAL_JSON),
            report_csv=str(CANDIDATE_EVAL_CSV),
        )
        report["candidate_metrics"] = candidate_metrics
    except Exception as exc:
        report["reason"] = f"candidate evaluation failed: {exc}"
        report_path = _write_promotion_report(report)
        print(f"Promotion report: {report_path}")
        return 1

    if PRODUCTION_MODEL.exists():
        try:
            old_metrics = evaluate_ranker(
                corrections=corrections_path,
                model=str(PRODUCTION_MODEL),
                report_json=str(PRODUCTION_EVAL_JSON),
                report_csv=str(PRODUCTION_EVAL_CSV),
            )
            report["old_metrics"] = old_metrics
        except Exception as exc:
            report["reason"] = f"production evaluation failed: {exc}"
            report_path = _write_promotion_report(report)
            print(f"Promotion report: {report_path}")
            return 1

    promoted, reason = _passes_promotion_gate(old_metrics, candidate_metrics)
    report["promoted"] = bool(promoted)
    report["reason"] = reason
    report["gate"] = {
        "mean_absolute_error_improved": old_metrics is None
        or _metric(candidate_metrics, "mean_absolute_error_sec") < _metric(old_metrics, "mean_absolute_error_sec"),
        "percent_within_25ms_improved": old_metrics is None
        or _percent_25ms(candidate_metrics) > _percent_25ms(old_metrics),
        "low_tier_accuracy_gate_passed": promoted,
        "low_tier_accuracy_not_strictly_regressed": old_metrics is None
        or max(_low_tier_count(old_metrics), _low_tier_count(candidate_metrics)) == 0
        or _low_tier_accuracy(candidate_metrics) >= _low_tier_accuracy(old_metrics),
    }

    if promoted:
        try:
            _promote_candidate_model()
        except Exception as exc:
            report["promoted"] = False
            report["reason"] = f"promotion file operation failed: {exc}"
            report_path = _write_promotion_report(report)
            print(f"Promotion report: {report_path}")
            return 1

    try:
        chooser_summary = train_candidate_chooser(
            corrections=corrections_path,
            output="models/drop_candidate_chooser.pkl",
            report="models/candidate_chooser_report.json",
        )
        report["candidate_chooser_training"] = chooser_summary
    except Exception as exc:
        report["candidate_chooser_training"] = {"ok": False, "error": str(exc) or exc.__class__.__name__}

    report_path = _write_promotion_report(report)
    print("\nPromotion gate:")
    print(f"  promoted: {report['promoted']}")
    print(f"  reason: {report['reason']}")
    print(f"  report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review batch drop candidates and create correction training data.")
    parser.add_argument("summary_csv", help="drop_batch_summary.csv from batch.py")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Correction JSONL output path")
    parser.add_argument("--retrain", action="store_true", help="Run train_ranker.py after review completes")
    parser.add_argument("--review-low-only", action="store_true", help="Only review LOW confidence tracks")
    parser.add_argument("--review-medium-and-low", action="store_true", help="Only review LOW and MEDIUM confidence tracks")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _filter_and_sort_rows(
        _load_csv(args.summary_csv),
        low_only=bool(args.review_low_only),
        medium_and_low=bool(args.review_medium_and_low),
    )
    counts = {"accepted": 0, "corrected": 0, "skip": 0}

    for row in rows:
        status = _review_row(row, args.corrections)
        if status == "quit":
            break
        counts[status] = counts.get(status, 0) + 1

    print("\nReview summary:")
    print(json.dumps({"corrections": args.corrections, "counts": counts}, indent=2, ensure_ascii=True))

    if args.retrain:
        return _run_retrain(args.corrections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
