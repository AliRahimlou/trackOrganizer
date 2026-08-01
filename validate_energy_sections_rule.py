#!/usr/bin/env python3
"""Validate the first-biggest-energy-boost rule against human drop picks.

For every truth track, compare analyze_energy_sections().chosen_time_sec with
the human pick. The rule targets the SECTION, not the sample, so agreement is
measured in bars at the track BPM.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from drop_aligner.energy_sections import analyze_energy_sections
from visual_first_scorecard import DEFAULT_TRUTH_LOGS, _bpm_from_path, _load_truth


SWEEP_FRACTIONS = (0.70, 0.75, 0.78, 0.80, 0.82, 0.85)


def _check(task):
    drums_path, user_pick = task
    bpm = _bpm_from_path(drums_path) or 128.0
    bar_sec = 4.0 * 60.0 / bpm
    result = analyze_energy_sections(drums_path)
    row = {
        "drums_path": drums_path,
        "user_pick": user_pick,
        "bar_sec": bar_sec,
        "ok": result.ok,
        "reason": result.reason,
        "chosen": result.chosen_time_sec,
        "events": len(result.events),
        "top_events": sum(1 for e in result.events if e.top_class),
    }
    if result.ok and result.chosen_time_sec is not None:
        row["delta_bars"] = (result.chosen_time_sec - user_pick) / bar_sec
    if result.ok and result.events:
        from drop_aligner.energy_sections import choose_first_top_boost

        sweep = {}
        for fraction in SWEEP_FRACTIONS:
            event = choose_first_top_boost(result.events, result.max_post_energy, fraction)
            if event is not None and event.refined_time_sec is not None:
                sweep[str(fraction)] = (event.refined_time_sec - user_pick) / bar_sec
        row["sweep_delta_bars"] = sweep
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models/energy_sections_rule_validation.json"))
    args = parser.parse_args()

    truth = _load_truth(list(DEFAULT_TRUTH_LOGS))
    tasks = sorted((row["drums_path"], row["user_pick"]) for row in truth.values() if Path(row["drums_path"]).is_file())
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[rule-check] {len(tasks)} truth tracks")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_check, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}")

    scored = [r for r in rows if r.get("delta_bars") is not None]
    def within(b):
        return sum(1 for r in scored if abs(r["delta_bars"]) <= b)
    n = len(scored)
    summary = {
        "tracks": len(rows),
        "rule_answered": n,
        "no_answer": len(rows) - n,
        "within_half_bar_pct": round(100 * within(0.5) / n, 1) if n else 0,
        "within_1_bar_pct": round(100 * within(1) / n, 1) if n else 0,
        "within_2_bars_pct": round(100 * within(2) / n, 1) if n else 0,
        "within_4_bars_pct": round(100 * within(4) / n, 1) if n else 0,
        "within_8_bars_pct": round(100 * within(8) / n, 1) if n else 0,
        "median_abs_bars": round(sorted(abs(r["delta_bars"]) for r in scored)[n // 2], 3) if n else None,
        "late_gt_2bars": sum(1 for r in scored if r["delta_bars"] > 2),
        "early_lt_minus2bars": sum(1 for r in scored if r["delta_bars"] < -2),
    }
    print(json.dumps(summary, indent=2))

    sweep_summary = {}
    for fraction in SWEEP_FRACTIONS:
        key = str(fraction)
        deltas = [r["sweep_delta_bars"][key] for r in rows if key in (r.get("sweep_delta_bars") or {})]
        if not deltas:
            continue
        n_f = len(deltas)
        sweep_summary[key] = {
            "answered": n_f,
            "within_1_bar_pct": round(100 * sum(1 for d in deltas if abs(d) <= 1) / n_f, 1),
            "within_2_bars_pct": round(100 * sum(1 for d in deltas if abs(d) <= 2) / n_f, 1),
            "median_abs_bars": round(sorted(abs(d) for d in deltas)[n_f // 2], 3),
            "late_gt_2bars": sum(1 for d in deltas if d > 2),
            "early_lt_minus2bars": sum(1 for d in deltas if d < -2),
        }
    print("SWEEP:", json.dumps(sweep_summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "sweep": sweep_summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"[rule-check] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
