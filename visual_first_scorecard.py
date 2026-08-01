#!/usr/bin/env python3
"""Score the current visual-first pipeline against human-verified drop picks.

Runs the exact production Stage A + Stage B path (``visual_first_marker`` ->
``build_visual_first_als_anchor``) for every truth track and reports the
millisecond error CDF, hold rate, and splits by tempo band / selected_by.
This is the headline accuracy instrument for the drop finder: run it before
and after any detector change.

Truth rows are JSONL with at least {"track": <drums path>, "user_pick": <sec>}
(the format of models/post_reset_human_review_truth_59.jsonl and
models/multistem_training_corrections.jsonl).  Later files and later lines win
when the same drums path appears more than once, so pass the most trusted log
last.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import signal
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_TRUTH_LOGS = (
    Path("models/multistem_training_corrections.jsonl"),
    Path("models/post_reset_human_review_truth_59.jsonl"),
)
DEFAULT_THRESHOLDS_MS = (2.0, 5.0, 10.0, 25.0, 50.0, 90.0)
DEFAULT_TIMEOUT_SEC = 600.0
TEMPO_BANDS = ((0, 100), (100, 140), (140, 175), (175, 10_000))

# Mirrors builder.STEM_RE closely enough to pull the title BPM out of a stem
# filename without importing the full ALS builder in every worker process.
STEM_NAME_RE = re.compile(r"^(drums|inst|vocals)_(\d+)_(\d+[ABab])_(\d+)-", re.IGNORECASE)


def _load_truth(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    truth: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            print(f"[scorecard] truth log missing, skipping: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                track = str(row.get("track") or "").strip()
                pick = row.get("user_pick")
                try:
                    pick_sec = float(pick)
                except (TypeError, ValueError):
                    continue
                if not track or not math.isfinite(pick_sec) or pick_sec < 0.0:
                    continue
                truth[track] = {
                    "drums_path": track,
                    "user_pick": pick_sec,
                    "reviewed_from": str(row.get("reviewed_from") or ""),
                    "truth_source": f"{path.name}:{line_no}",
                }
    return truth


def _bpm_from_path(drums_path: str) -> Optional[float]:
    name = Path(drums_path).name
    match = STEM_NAME_RE.match(name)
    if match:
        return float(int(match.group(2)))
    # Fall back to the STEMS/<BPM>/<KEY>/ folder layout.
    parts = Path(drums_path).parts
    for index in range(len(parts) - 1):
        if parts[index].upper() == "STEMS" and index + 1 < len(parts):
            try:
                return float(int(parts[index + 1]))
            except ValueError:
                return None
    return None


class _TrackTimeout(TimeoutError):
    pass


def _score_one(task: Mapping[str, Any]) -> Dict[str, Any]:
    drums_path = str(task["drums_path"])
    user_pick = float(task["user_pick"])
    use_cache = bool(task["use_cache"])
    timeout_sec = float(task["timeout_sec"])

    out: Dict[str, Any] = {
        "drums_path": drums_path,
        "user_pick_sec": user_pick,
        "reviewed_from": task.get("reviewed_from") or "",
        "truth_source": task.get("truth_source") or "",
        "bpm": None,
        "status": "error",
        "hold_reason": "",
        "selected_by": "",
        "stage_a_marker_sec": None,
        "drop_sec": None,
        "error_ms": None,
        "abs_error_ms": None,
    }

    bpm = _bpm_from_path(drums_path)
    if bpm is None or bpm <= 0.0:
        out["hold_reason"] = "no_bpm_in_path"
        return out
    out["bpm"] = bpm

    if not Path(drums_path).is_file():
        out["status"] = "missing_stem"
        out["hold_reason"] = "drums_stem_not_found"
        return out

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _TrackTimeout()

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        from drop_aligner.als_anchor import build_visual_first_als_anchor

        anchor = build_visual_first_als_anchor(drums_path, bpm, use_cache=use_cache)
    except _TrackTimeout:
        out["hold_reason"] = f"timeout:{timeout_sec:.0f}s"
        return out
    except Exception as exc:  # pragma: no cover - defensive: report, don't die
        out["hold_reason"] = f"exception:{exc.__class__.__name__}:{exc}"
        return out
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    if isinstance(anchor.get("visual_marker_sec"), (int, float)):
        out["stage_a_marker_sec"] = float(anchor["visual_marker_sec"])
    out["selected_by"] = str(anchor.get("selected_by") or "")

    if bool(anchor.get("accepted")) and isinstance(anchor.get("drop_sec"), (int, float)):
        drop_sec = float(anchor["drop_sec"])
        out["status"] = "pass"
        out["drop_sec"] = drop_sec
        out["error_ms"] = (drop_sec - user_pick) * 1000.0
        out["abs_error_ms"] = abs(out["error_ms"])
    else:
        out["status"] = "hold"
        out["hold_reason"] = str(anchor.get("reason") or "rejected")
    return out


def _tempo_band(bpm: Optional[float]) -> str:
    if bpm is None:
        return "unknown"
    for low, high in TEMPO_BANDS:
        if low <= bpm < high:
            return f"{low}-{high}" if high < 10_000 else f"{low}+"
    return "unknown"


def _cdf(rows: Sequence[Mapping[str, Any]], thresholds_ms: Sequence[float]) -> Dict[str, Any]:
    errors = [float(r["abs_error_ms"]) for r in rows if r.get("abs_error_ms") is not None]
    if not errors:
        return {"scored": 0}
    errors.sort()
    summary: Dict[str, Any] = {
        "scored": len(errors),
        "median_abs_ms": round(statistics.median(errors), 3),
        "mean_abs_ms": round(statistics.fmean(errors), 3),
        "p95_abs_ms": round(errors[min(len(errors) - 1, int(0.95 * len(errors)))], 3),
        "max_abs_ms": round(errors[-1], 3),
    }
    for threshold in thresholds_ms:
        hits = sum(1 for err in errors if err <= threshold)
        summary[f"within_{threshold:g}ms"] = round(100.0 * hits / len(errors), 2)
    return summary


def _group_by(rows: Sequence[Mapping[str, Any]], key_fn) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", nargs="+", type=Path, default=list(DEFAULT_TRUTH_LOGS))
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N truth tracks (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true", help="Disable analysis caches for an honest run")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--thresholds-ms", default=",".join(str(t) for t in DEFAULT_THRESHOLDS_MS))
    parser.add_argument("--out-dir", type=Path, default=Path("models/scorecards"))
    args = parser.parse_args()

    thresholds_ms = [float(part) for part in str(args.thresholds_ms).split(",") if part.strip()]
    truth = _load_truth(args.truth)
    if not truth:
        print("[scorecard] no truth rows loaded", file=sys.stderr)
        return 2

    tasks = [
        {
            **row,
            "use_cache": not args.no_cache,
            "timeout_sec": float(args.timeout_sec),
        }
        for row in truth.values()
    ]
    tasks.sort(key=lambda t: str(t["drums_path"]))
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"[scorecard] scoring {len(tasks)} tracks with {args.workers} workers (cache={'off' if args.no_cache else 'on'})")
    results: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for index, task in enumerate(tasks, 1):
            result = _score_one(task)
            results.append(result)
            print(f"[{index}/{len(tasks)}] {result['status']:<12} {Path(result['drums_path']).parent.name}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_score_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(tasks)}] {result['status']:<12} {Path(result['drums_path']).parent.name}")

    scored = [r for r in results if r["status"] == "pass"]
    holds = [r for r in results if r["status"] == "hold"]
    missing = [r for r in results if r["status"] == "missing_stem"]
    errored = [r for r in results if r["status"] == "error"]

    report: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "truth_logs": [str(p) for p in args.truth],
        "track_count": len(results),
        "pass_count": len(scored),
        "hold_count": len(holds),
        "missing_stem_count": len(missing),
        "error_count": len(errored),
        "hold_rate_pct": round(100.0 * len(holds) / len(results), 2) if results else 0.0,
        "overall": _cdf(scored, thresholds_ms),
        "by_tempo_band": {
            band: _cdf(rows, thresholds_ms)
            for band, rows in sorted(_group_by(scored, lambda r: _tempo_band(r.get("bpm"))).items())
        },
        "by_selected_by": {
            source: _cdf(rows, thresholds_ms)
            for source, rows in sorted(_group_by(scored, lambda r: str(r.get("selected_by") or "unknown")).items())
        },
        "hold_reasons": {
            reason: len(rows)
            for reason, rows in sorted(
                _group_by(holds + errored, lambda r: str(r.get("hold_reason") or "unknown")).items(),
                key=lambda item: -len(item[1]),
            )
        },
        "worst_tracks": sorted(
            (
                {
                    "drums_path": r["drums_path"],
                    "abs_error_ms": round(float(r["abs_error_ms"]), 3),
                    "error_ms": round(float(r["error_ms"]), 3),
                    "selected_by": r["selected_by"],
                    "bpm": r["bpm"],
                }
                for r in scored
                if r.get("abs_error_ms") is not None
            ),
            key=lambda row: -row["abs_error_ms"],
        )[:25],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"visual_first_scorecard_{stamp}.json"
    csv_path = args.out_dir / f"visual_first_scorecard_{stamp}.csv"
    json_path.write_text(json.dumps({"report": report, "rows": results}, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "drums_path", "bpm", "status", "user_pick_sec", "stage_a_marker_sec", "drop_sec",
                "error_ms", "abs_error_ms", "selected_by", "hold_reason", "reviewed_from", "truth_source",
            ],
        )
        writer.writeheader()
        for row in sorted(results, key=lambda r: str(r["drums_path"])):
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(json.dumps(report, indent=2))
    print(f"[scorecard] wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
