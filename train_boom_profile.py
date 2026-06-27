#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
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
    DEFAULT_BOOM_PROFILE_PATH,
    DEFAULT_PROFILE,
    candidate_boom_metrics,
    candidate_time,
)
from drop_aligner.historical_markers import HistoricalMarker, load_historical_markers  # noqa: E402
from drop_aligner.structure_features import compute_bar_feature_map  # noqa: E402
from drop_aligner.visual_first import EXTENDED_VISUAL_MAX_CLOCK_BAR, boom_body_section_candidates  # noqa: E402


DEFAULT_CORRECTIONS = (
    "models/multistem_training_corrections.jsonl",
    "drop_corrections.jsonl",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(clean[lo])
    weight = pos - lo
    return float((clean[lo] * (1.0 - weight)) + (clean[hi] * weight))


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": min(clean),
        "p10": _quantile(clean, 0.10),
        "p25": _quantile(clean, 0.25),
        "median": statistics.median(clean),
        "p75": _quantile(clean, 0.75),
        "p90": _quantile(clean, 0.90),
        "max": max(clean),
    }


def _threshold_from_positive(values: Sequence[float], *, floor: float, margin: float = 0.025) -> float:
    q10 = _quantile(values, 0.10)
    if q10 is None:
        return float(floor)
    return float(max(float(floor), min(0.98, q10 - float(margin))))


def _max_from_positive(values: Sequence[float], *, default: float, cap: float, margin: float = 0.030) -> float:
    q90 = _quantile(values, 0.90)
    if q90 is None:
        return float(default)
    return float(min(float(cap), max(float(default), q90 + float(margin))))


def _markers_from_logs(paths: Sequence[str], *, track_contains: str = "") -> List[HistoricalMarker]:
    index = load_historical_markers(correction_logs=[Path(path).expanduser() for path in paths])
    markers = list(index.by_path.values())
    needle = track_contains.lower().strip()
    dedup: Dict[str, HistoricalMarker] = {}
    for marker in markers:
        if needle and needle not in marker.track.lower():
            continue
        if not Path(marker.track).expanduser().exists():
            continue
        dedup[str(Path(marker.track).expanduser()).lower()] = marker
    return list(dedup.values())


def _extract_training_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
    track = str(payload["track"])
    expected = float(payload["user_pick"])
    sample_rate = int(payload["sample_rate"])
    started = time.time()
    try:
        feature_map = compute_bar_feature_map(track, sample_rate=sample_rate, use_cache=True)
        beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
        bar_sec = _finite_float(beatgrid.get("bar_sec"), _finite_float(beatgrid.get("beat_sec"), 0.5) * 4.0)
        max_clock_bar = max(
            81,
            min(EXTENDED_VISUAL_MAX_CLOCK_BAR, int(feature_map.get("bar_count", 0) or EXTENDED_VISUAL_MAX_CLOCK_BAR)),
        )
        candidates = boom_body_section_candidates(feature_map, max_clock_bar=max_clock_bar)
    except Exception as exc:
        return {
            "ok": False,
            "track": track,
            "user_pick": expected,
            "error": str(exc) or exc.__class__.__name__,
            "elapsed_sec": time.time() - started,
        }
    timed = []
    for candidate in candidates:
        edge_time = candidate_time(candidate)
        if edge_time is None:
            continue
        timed.append((abs(float(edge_time) - expected), float(edge_time), candidate))
    if not timed:
        return {
            "ok": False,
            "track": track,
            "user_pick": expected,
            "error": "no_boom_candidates",
            "candidate_count": len(candidates),
            "elapsed_sec": time.time() - started,
        }
    timed.sort(key=lambda row: row[0])
    near_limit = max(1.250, min(6.0, float(bar_sec) * 1.25))
    nearest_delta, nearest_time, nearest = timed[0]
    positive = bool(nearest_delta <= near_limit)
    metrics = candidate_boom_metrics(nearest)
    visual = nearest.get("visual_components") if isinstance(nearest.get("visual_components"), Mapping) else {}
    negatives = []
    for delta, edge_time, candidate in timed[1:]:
        if delta <= near_limit * 1.8:
            continue
        negative_metrics = candidate_boom_metrics(candidate)
        negative_visual = candidate.get("visual_components") if isinstance(candidate.get("visual_components"), Mapping) else {}
        negatives.append(
            {
                "delta_sec": float(delta),
                "edge_time": float(edge_time),
                "clock_bar": int(_finite_float(negative_visual.get("clock_bar"), 0)),
                "metrics": negative_metrics,
            }
        )
        if len(negatives) >= 4:
            break
    return {
        "ok": True,
        "positive": positive,
        "track": track,
        "user_pick": expected,
        "reviewed_from": str(payload.get("reviewed_from") or ""),
        "source_path": str(payload.get("source_path") or ""),
        "nearest_edge_time": float(nearest_time),
        "nearest_delta_sec": float(nearest_delta),
        "near_limit_sec": float(near_limit),
        "candidate_count": len(candidates),
        "clock_bar": int(_finite_float(visual.get("clock_bar"), 0)),
        "metrics": metrics,
        "negatives": negatives,
        "elapsed_sec": time.time() - started,
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    markers = _markers_from_logs(args.corrections, track_contains=str(args.track_contains or ""))
    markers.sort(key=lambda marker: (str(marker.track).lower(), str(marker.timestamp)))
    if args.limit:
        markers = markers[: int(args.limit)]
    jobs = [
        {
            "track": marker.track,
            "user_pick": float(marker.user_pick),
            "reviewed_from": marker.reviewed_from,
            "source_path": marker.source_path,
            "sample_rate": int(args.sample_rate),
        }
        for marker in markers
    ]
    results: List[Dict[str, Any]] = []
    started = time.time()
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_extract_training_row(job))
            if args.progress_every and index % int(args.progress_every) == 0:
                print(f"profiled {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_training_row, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"profiled {index}/{len(jobs)}", flush=True)

    positives = [row for row in results if row.get("ok") and row.get("positive")]
    misses = [row for row in results if not row.get("ok") or not row.get("positive")]
    negative_rows = [neg for row in positives for neg in row.get("negatives") or []]
    metric_keys = sorted({key for row in positives for key in (row.get("metrics") or {}).keys()})
    positive_metrics = {
        key: _stats([_finite_float((row.get("metrics") or {}).get(key), float("nan")) for row in positives])
        for key in metric_keys
    }
    negative_metrics = {
        key: _stats([_finite_float((row.get("metrics") or {}).get(key), float("nan")) for row in negative_rows])
        for key in metric_keys
    }
    profile_scores = [_finite_float((row.get("metrics") or {}).get("profile_score"), float("nan")) for row in positives]
    darkness = [_finite_float((row.get("metrics") or {}).get("darkness"), float("nan")) for row in positives]
    post8 = [_finite_float((row.get("metrics") or {}).get("post8_height"), float("nan")) for row in positives]
    simultaneity = [_finite_float((row.get("metrics") or {}).get("simultaneity"), float("nan")) for row in positives]
    post_bass8 = [_finite_float((row.get("metrics") or {}).get("post_bass8"), float("nan")) for row in positives]
    post_drum8 = [_finite_float((row.get("metrics") or {}).get("post_drum8"), float("nan")) for row in positives]
    post_drum_cont8 = [_finite_float((row.get("metrics") or {}).get("post_drum_cont8"), float("nan")) for row in positives]
    contrast = [_finite_float((row.get("metrics") or {}).get("contrast"), float("nan")) for row in positives]
    sustain = [_finite_float((row.get("metrics") or {}).get("sustain"), float("nan")) for row in positives]
    one_distance = [_finite_float((row.get("metrics") or {}).get("one_distance_ms"), float("nan")) for row in positives]
    front_offsets = [abs(_finite_float(row.get("nearest_delta_sec"), float("nan"))) for row in positives]

    thresholds = dict(DEFAULT_PROFILE["thresholds"])
    if len(positives) >= int(args.min_positives):
        thresholds.update(
            {
                "min_profile_score": _threshold_from_positive(profile_scores, floor=0.50, margin=0.020),
                "min_darkness": _threshold_from_positive(darkness, floor=0.34, margin=0.030),
                "min_post8_height": _threshold_from_positive(post8, floor=0.32, margin=0.030),
                "min_simultaneity": _threshold_from_positive(simultaneity, floor=0.28, margin=0.030),
                "min_post_bass8": _threshold_from_positive(post_bass8, floor=0.12, margin=0.025),
                "min_post_drum8": _threshold_from_positive(post_drum8, floor=0.34, margin=0.035),
                "min_post_drum_cont8": _threshold_from_positive(post_drum_cont8, floor=0.26, margin=0.035),
                "min_contrast": _threshold_from_positive(contrast, floor=0.030, margin=0.020),
                "min_sustain": _threshold_from_positive(sustain, floor=0.10, margin=0.020),
                "max_one_distance_ms": _max_from_positive(one_distance, default=70.0, cap=140.0, margin=8.0),
                "max_front_edge_offset_sec": _max_from_positive(front_offsets, default=0.350, cap=0.900, margin=0.050),
                "max_nearest_edge_sec": _max_from_positive(front_offsets, default=0.900, cap=1.750, margin=0.180),
            }
        )

    profile = {
        "version": 1,
        "source": "trained_from_human_reviewed_drop_markers",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": int(args.sample_rate),
        "corrections": list(args.corrections),
        "track_contains": str(args.track_contains or ""),
        "training_counts": {
            "markers_loaded": len(markers),
            "rows_profiled": len(results),
            "positive_count": len(positives),
            "miss_count": len(misses),
            "negative_candidate_count": len(negative_rows),
            "min_positives_for_threshold_update": int(args.min_positives),
        },
        "thresholds": thresholds,
        "positive_metric_stats": positive_metrics,
        "negative_metric_stats": negative_metrics,
        "misses": misses[: int(args.keep_misses)],
        "elapsed_sec": time.time() - started,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(json.dumps({key: profile[key] for key in ("created_at", "training_counts", "thresholds", "elapsed_sec")}, indent=2))
    print(f"Wrote boom profile: {output}")
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn dynamic boom-body proof thresholds from human-reviewed drop markers."
    )
    parser.add_argument("--corrections", action="append", default=list(DEFAULT_CORRECTIONS), help="Human correction JSONL. Repeatable.")
    parser.add_argument("--output", default=str(DEFAULT_BOOM_PROFILE_PATH), help="Output boom profile JSON.")
    parser.add_argument("--track-contains", default="", help="Only train on tracks whose path contains this text.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum reviewed tracks to profile; 0 means all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-positives", type=int, default=40)
    parser.add_argument("--keep-misses", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
