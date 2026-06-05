#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from drop_aligner.candidate_chooser import candidate_effective_time
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.microalign import microalign_candidate_dicts
from drop_aligner.multistem import generate_multistem_candidates
from drop_aligner.pipeline import run_drop_candidate_pipeline
from project_config import DROP_BATCH_SUMMARY


DEFAULT_SUMMARY = DROP_BATCH_SUMMARY
DEFAULT_CORRECTIONS = Path("models/multistem_training_corrections.jsonl")
FALLBACK_CORRECTIONS = Path("drop_corrections.jsonl")
DEFAULT_JSON = Path("models/oracle_diagnostics.json")
DEFAULT_CSV = Path("models/oracle_diagnostics.csv")
THRESHOLDS_SEC = (0.005, 0.010, 0.025, 0.050, 0.100)
TOP_K = (1, 3, 5, 10, 25, 50)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
        t = candidate_effective_time(candidate)
        key = f"{t:.6f}" if t is not None else json.dumps(dict(candidate), sort_keys=True, default=str)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _selected_time(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_ai_pick", "ai_pick", "detected_drop_time", "user_pick"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        return candidate_effective_time(selected)
    return None


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or row.get("filename") or "")


def _read_summary(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {str(row.get("filename", "")): row for row in reader if row.get("filename")}


def _pct(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return 100.0 * float(count) / float(total)


def _fmt_threshold(threshold: float) -> str:
    return f"{int(round(threshold * 1000.0))}ms"


def _candidate_times(candidates: Sequence[Mapping[str, Any]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for index, candidate in enumerate(candidates, start=1):
        value = candidate_effective_time(candidate)
        if value is not None:
            out.append((index, float(value)))
    return out


def _oracle_hit(candidate_times: Sequence[Tuple[int, float]], user_pick: float, *, top_k: int, threshold: float) -> bool:
    for _rank, time_sec in list(candidate_times)[: int(top_k)]:
        if abs(float(time_sec) - float(user_pick)) <= float(threshold):
            return True
    return False


def _nearest(candidate_times: Sequence[Tuple[int, float]], user_pick: float) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not candidate_times:
        return None, None, None
    rank, time_sec = min(candidate_times, key=lambda item: (abs(float(item[1]) - float(user_pick)), int(item[0])))
    return int(rank), float(time_sec), abs(float(time_sec) - float(user_pick))


def build_oracle_diagnostics(
    *,
    corrections: str,
    batch_summary: str,
    output_json: str,
    output_csv: str,
    regenerate_candidates: bool = False,
    expanded_limit: int = 120,
    microalign_limit: int = 50,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    corrections_path = Path(corrections).expanduser()
    summary_path = Path(batch_summary).expanduser()
    summary_by_track = _read_summary(summary_path)

    rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    oracle_counts: Dict[str, int] = {
        f"oracle_at_{k}_within_{_fmt_threshold(threshold)}": 0
        for k in TOP_K
        for threshold in THRESHOLDS_SEC
    }
    selected_counts: Dict[str, int] = {f"selected_within_{_fmt_threshold(threshold)}": 0 for threshold in THRESHOLDS_SEC}
    closest_ranks: List[int] = []
    closest_errors: List[float] = []
    selected_errors: List[float] = []

    for row_index, row in enumerate(_iter_jsonl(corrections_path), start=1):
        if row_has_excluded_path(row):
            counts["excluded"] += 1
            continue
        track = _track(row)
        user_pick = _safe_float(row.get("user_pick"))
        if not track or user_pick is None:
            counts["missing_track_or_user_pick"] += 1
            continue
        candidates = _candidate_rows(row)
        if regenerate_candidates:
            try:
                generated = generate_multistem_candidates(
                    track,
                    saved_candidates=candidates,
                    limit=max(int(expanded_limit), int(microalign_limit)),
                    sample_rate=int(sample_rate),
                )
                regenerated = list(generated.get("candidates") or [])
                if int(microalign_limit) > 0:
                    regenerated = microalign_candidate_dicts(track, regenerated, limit=int(microalign_limit))
                pipeline = run_drop_candidate_pipeline(
                    regenerated,
                    cluster_radius_sec=0.085,
                    limit=int(microalign_limit) if int(microalign_limit) > 0 else None,
                )
                candidates = list(pipeline.get("candidates") or regenerated)
            except Exception:
                counts["regenerate_failed"] += 1
        candidate_times = _candidate_times(candidates)
        if not candidate_times:
            counts["no_candidate_times"] += 1
            continue

        selected = _selected_time(row)
        selected_error = None if selected is None else abs(float(selected) - float(user_pick))
        closest_rank, closest_time, closest_error = _nearest(candidate_times, float(user_pick))
        if closest_rank is not None:
            closest_ranks.append(int(closest_rank))
        if closest_error is not None:
            closest_errors.append(float(closest_error))
        if selected_error is not None:
            selected_errors.append(float(selected_error))

        for threshold in THRESHOLDS_SEC:
            if selected_error is not None and selected_error <= threshold:
                selected_counts[f"selected_within_{_fmt_threshold(threshold)}"] += 1
            for k in TOP_K:
                key = f"oracle_at_{k}_within_{_fmt_threshold(threshold)}"
                if _oracle_hit(candidate_times, float(user_pick), top_k=int(k), threshold=float(threshold)):
                    oracle_counts[key] += 1

        no_candidate_within_100 = closest_error is None or closest_error > 0.100
        correct_exists_25 = closest_error is not None and closest_error <= 0.025
        selected_wrong_25 = selected_error is None or selected_error > 0.025
        selected_correct_25 = selected_error is not None and selected_error <= 0.025
        reanalysis = row.get("multistem_reanalysis") if isinstance(row.get("multistem_reanalysis"), Mapping) else {}
        gate_rejected = reanalysis and not bool(reanalysis.get("auto_accept_passed", False))

        if no_candidate_within_100:
            case_type = "candidate_generation_miss"
        elif correct_exists_25 and selected_wrong_25:
            case_type = "ranking_miss"
        elif selected_correct_25 and gate_rejected:
            case_type = "gate_rejected_correct_selection"
        elif selected_correct_25:
            case_type = "selected_correct"
        else:
            case_type = "review_needed"
        counts[case_type] += 1

        summary = summary_by_track.get(track, {})
        rows.append(
            {
                "row_index": int(row_index),
                "track": track,
                "confidence_tier": str(row.get("confidence_tier") or summary.get("confidence_tier") or ""),
                "reviewed_from": str(row.get("reviewed_from") or ""),
                "selected_by": str(row.get("selected_by") or summary.get("selected_by") or ""),
                "user_pick": float(user_pick),
                "selected_time": selected,
                "selected_error_ms": None if selected_error is None else float(selected_error * 1000.0),
                "closest_candidate_rank": closest_rank,
                "closest_candidate_time": closest_time,
                "closest_candidate_error_ms": None if closest_error is None else float(closest_error * 1000.0),
                "candidate_count": int(len(candidate_times)),
                "case_type": case_type,
                "no_candidate_within_100ms": bool(no_candidate_within_100),
                "correct_candidate_within_25ms": bool(correct_exists_25),
                "selected_wrong_over_25ms": bool(selected_wrong_25),
                "gate_rejected_correct_selection": bool(selected_correct_25 and gate_rejected),
            }
        )

    total = int(len(rows))
    oracle_percent = {key + "_percent": _pct(value, total) for key, value in oracle_counts.items()}
    selected_percent = {key + "_percent": _pct(value, total) for key, value in selected_counts.items()}
    closest_rank_values = np.asarray(closest_ranks, dtype=np.float64)
    closest_error_values = np.asarray(closest_errors, dtype=np.float64)
    selected_error_values = np.asarray(selected_errors, dtype=np.float64)
    candidate_counts = np.asarray([int(row.get("candidate_count", 0) or 0) for row in rows], dtype=np.float64)
    max_candidate_count = int(np.max(candidate_counts)) if candidate_counts.size else 0
    report: Dict[str, Any] = {
        "corrections": str(corrections_path),
        "batch_summary": str(summary_path),
        "regenerate_candidates": bool(regenerate_candidates),
        "expanded_limit": int(expanded_limit),
        "microalign_limit": int(microalign_limit),
        "reviewed_tracks": total,
        "counts": dict(counts),
        "oracle_counts": oracle_counts,
        "oracle_percent": oracle_percent,
        "selected_counts": selected_counts,
        "selected_percent": selected_percent,
        "average_rank_of_closest_candidate": None if closest_rank_values.size == 0 else float(np.mean(closest_rank_values)),
        "median_rank_of_closest_candidate": None if closest_rank_values.size == 0 else float(np.median(closest_rank_values)),
        "max_candidate_count": int(max_candidate_count),
        "median_candidate_count": None if candidate_counts.size == 0 else float(np.median(candidate_counts)),
        "oracle_k_capped_warning": (
            f"candidate pools only contain up to {max_candidate_count} candidates; regenerate with --regenerate-candidates for true oracle@{max(TOP_K)}"
            if max_candidate_count < max(TOP_K)
            else ""
        ),
        "median_closest_candidate_error_ms": None if closest_error_values.size == 0 else float(np.median(closest_error_values) * 1000.0),
        "median_selected_error_ms": None if selected_error_values.size == 0 else float(np.median(selected_error_values) * 1000.0),
        "tracks_where_no_candidate_within_100ms": [row for row in rows if row["no_candidate_within_100ms"]],
        "tracks_where_correct_candidate_existed_but_model_chose_wrong": [
            row for row in rows if row["case_type"] == "ranking_miss"
        ],
        "tracks_where_candidate_was_correct_but_gate_rejected": [
            row for row in rows if row["case_type"] == "gate_rejected_correct_selection"
        ],
    }

    json_path = Path(output_json).expanduser()
    csv_path = Path(output_csv).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    fieldnames = [
        "row_index",
        "track",
        "confidence_tier",
        "reviewed_from",
        "selected_by",
        "user_pick",
        "selected_time",
        "selected_error_ms",
        "closest_candidate_rank",
        "closest_candidate_time",
        "closest_candidate_error_ms",
        "candidate_count",
        "case_type",
        "no_candidate_within_100ms",
        "correct_candidate_within_25ms",
        "selected_wrong_over_25ms",
        "gate_rejected_correct_selection",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    report["output_json"] = str(json_path)
    report["output_csv"] = str(csv_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    default_corrections = DEFAULT_CORRECTIONS if DEFAULT_CORRECTIONS.exists() else FALLBACK_CORRECTIONS
    parser = argparse.ArgumentParser(description="Measure oracle@K for reviewed drop-candidate pools.")
    parser.add_argument("--corrections", default=str(default_corrections), help="Correction JSONL, preferably expanded multistem training JSONL")
    parser.add_argument("--batch-summary", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv path")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON), help="Output JSON report")
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV), help="Output per-track CSV report")
    parser.add_argument("--regenerate-candidates", action="store_true", help="Regenerate expanded multistem candidates so oracle@25/@50 is not capped by stored top candidates")
    parser.add_argument("--expanded-limit", type=int, default=120)
    parser.add_argument("--microalign-limit", type=int, default=50)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_oracle_diagnostics(
        corrections=str(args.corrections),
        batch_summary=str(args.batch_summary),
        output_json=str(args.output_json),
        output_csv=str(args.output_csv),
        regenerate_candidates=bool(args.regenerate_candidates),
        expanded_limit=int(args.expanded_limit),
        microalign_limit=int(args.microalign_limit),
        sample_rate=int(args.analysis_sr),
    )
    compact = {
        "reviewed_tracks": report.get("reviewed_tracks"),
        "oracle_at_10_within_25ms_percent": report.get("oracle_percent", {}).get("oracle_at_10_within_25ms_percent"),
        "oracle_at_25_within_25ms_percent": report.get("oracle_percent", {}).get("oracle_at_25_within_25ms_percent"),
        "selected_within_25ms_percent": report.get("selected_percent", {}).get("selected_within_25ms_percent"),
        "counts": report.get("counts"),
        "oracle_k_capped_warning": report.get("oracle_k_capped_warning"),
        "output_json": report.get("output_json"),
        "output_csv": report.get("output_csv"),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
