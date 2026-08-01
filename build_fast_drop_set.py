#!/usr/bin/env python3
"""Fast whole-library drop set: Ali's manual algorithm, nothing else.

Per track, exactly the manual Ableton workflow:
1. Look at the drums stem's whole waveform and find the FIRST occurrence of the
   BIGGEST class of energy boost (intro/buildup vs drop sectioning).
2. Zoom in: find where the waveform changes from one type to the other and put
   1.1.1 on that impact, snapped to the zero line at native sample rate.
3. Write the per-track ALS with drums/inst/vocals sharing that exact timestamp,
   then insert every track into one combined performance set.

No beatgrid gating, no proof cascade: seconds per track, minutes per library.

Validation mode (--validate-truth) scores the same algorithm against the human
picks in models/*.jsonl instead of writing any ALS.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import soundfile as sf

import buildSetAndGenerateAls as builder
from build_fresh_visual_first_library_set import _build_combined_set, _find_role_audio
from drop_aligner.als import modify_als
from drop_aligner.energy_sections import analyze_energy_sections
from project_config import BASE_ALS_TEMPLATE, DEFAULT_ALS_TEMPLATE, GENERATED_SET_DIR, STEMS_ROOT_DIR

SNAP_WINDOW_SEC = 0.75
SNAP_FRAME_SEC = 0.0025
ZERO_SNAP_MS = 1.5
SELECTED_BY = "fast_first_boost_impact"


def snap_to_impact(drums_path: str, approx_sec: float) -> Optional[Dict[str, float]]:
    """Zoom step: sample-accurate impact start near the section entry.

    Finds the strongest short-window energy step inside +/-SNAP_WINDOW_SEC,
    walks back to where the attack leaves the pre-impact floor, then snaps to
    the nearest zero crossing — 1.1.1 "as close to the zero line as possible".
    """
    try:
        info = sf.info(str(drums_path))
        sr = int(info.samplerate)
        start = max(0, int((approx_sec - SNAP_WINDOW_SEC) * sr))
        frames = int(2 * SNAP_WINDOW_SEC * sr)
        audio, _ = sf.read(str(drums_path), start=start, frames=frames, dtype="float32", always_2d=True)
    except Exception:
        return None
    mono = audio.mean(axis=1)
    frame = max(8, int(SNAP_FRAME_SEC * sr))
    count = mono.size // frame
    if count < 8:
        return None
    energy = np.sqrt(np.mean(np.square(mono[: count * frame].reshape(count, frame)), axis=1))
    # Strongest sustained step: post window well above the rolling pre floor.
    pre_n = max(4, int(0.10 / SNAP_FRAME_SEC / 10))  # ~100ms of pre context
    best_idx, best_ratio = None, 0.0
    for i in range(pre_n, count - 4):
        pre = float(np.median(energy[i - pre_n : i])) + 1e-9
        post = float(np.mean(energy[i : i + 4]))
        ratio = post / pre
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    if best_idx is None or best_ratio < 1.5:
        return None
    # Attack start: first sample in the step frame region that exceeds the floor.
    floor = float(np.median(energy[max(0, best_idx - pre_n) : best_idx])) * 2.0 + 1e-6
    seg_lo = best_idx * frame
    seg_hi = min(mono.size, (best_idx + 2) * frame)
    above = np.flatnonzero(np.abs(mono[seg_lo:seg_hi]) > floor)
    attack = seg_lo + int(above[0]) if above.size else seg_lo
    # Zero-line snap: nearest zero crossing within ZERO_SNAP_MS before the attack.
    back = int(ZERO_SNAP_MS / 1000.0 * sr)
    lo = max(0, attack - back)
    segment = mono[lo : attack + 1]
    zero = None
    for j in range(segment.size - 1, 0, -1):
        if (segment[j - 1] <= 0.0 <= segment[j]) or (segment[j - 1] >= 0.0 >= segment[j]):
            zero = lo + j - 1
            break
    final = zero if zero is not None else attack
    return {
        "drop_sec": (start + final) / sr,
        "step_ratio": float(best_ratio),
        "sample_rate": float(sr),
    }


def find_fast_drop(drums_path: str) -> Dict[str, Any]:
    sections = analyze_energy_sections(str(drums_path))
    if not sections.ok or sections.chosen_time_sec is None:
        return {"ok": False, "reason": sections.reason or "no_section"}
    snap = snap_to_impact(str(drums_path), float(sections.chosen_time_sec))
    if snap is None:
        # Section is known; fall back to its refined entry edge.
        return {
            "ok": True,
            "drop_sec": float(sections.chosen_time_sec),
            "snapped": False,
            "section_sec": float(sections.chosen_time_sec),
            "events": len(sections.events),
        }
    return {
        "ok": True,
        "drop_sec": float(snap["drop_sec"]),
        "snapped": True,
        "step_ratio": snap["step_ratio"],
        "section_sec": float(sections.chosen_time_sec),
        "events": len(sections.events),
    }


def _process_one(task: Mapping[str, Any]) -> Dict[str, Any]:
    track = dict(task["track"])
    template = str(task["template"])
    run_stamp = str(task["run_stamp"])
    write_als = bool(task["write_als"])
    folder = Path(str(track["src"])).parent
    drums = _find_role_audio(folder, "drums", track)
    if drums is None:
        return {"status": "error", "track": track, "error": "missing_drums", "drums_path": ""}
    result = find_fast_drop(str(drums))
    if not result.get("ok"):
        return {
            "status": "hold",
            "track": track,
            "drums_path": str(drums),
            "error": f"fast_no_drop:{result.get('reason')}",
        }
    marker = float(result["drop_sec"])
    row: Dict[str, Any] = {
        "status": "processed",
        "track": track,
        "drums_path": str(drums),
        "marker": marker,
        "selected_by": SELECTED_BY,
        "fast": {key: result.get(key) for key in ("snapped", "step_ratio", "section_sec", "events")},
    }
    if write_als:
        output_als = drums.with_name(f"{drums.stem}_FAST_DROP_{run_stamp}_DROP_ALIGNED.als")
        try:
            modify_als(
                template_path=template,
                audio_path=str(drums),
                drop_sec=marker,
                bpm=float(track["bpm"]),
                output_path=str(output_als),
            )
        except Exception as exc:
            return {
                "status": "error",
                "track": track,
                "drums_path": str(drums),
                "marker": marker,
                "error": f"als_write_failed:{exc.__class__.__name__}",
            }
        row["output_als"] = str(output_als)
    return row


def _validate_truth(args: argparse.Namespace) -> int:
    from visual_first_scorecard import DEFAULT_TRUTH_LOGS, _bpm_from_path, _load_truth

    truth = _load_truth(list(DEFAULT_TRUTH_LOGS))
    tasks = sorted((r["drums_path"], r["user_pick"]) for r in truth.values() if Path(r["drums_path"]).is_file())
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[fast-validate] {len(tasks)} truth tracks")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(find_fast_drop, path): (path, pick) for path, pick in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            path, pick = futures[fut]
            result = fut.result()
            bpm = _bpm_from_path(path) or 128.0
            bar = 4.0 * 60.0 / bpm
            row = {"drums_path": path, "user_pick": pick, "ok": result.get("ok")}
            if result.get("ok"):
                delta = float(result["drop_sec"]) - pick
                row.update(delta_sec=delta, delta_bars=delta / bar, abs_ms=abs(delta) * 1000.0)
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}")
    scored = [r for r in rows if r.get("delta_bars") is not None]
    n = len(scored)
    def pct(cond):
        return round(100.0 * sum(1 for r in scored if cond(r)) / n, 1) if n else 0.0
    summary = {
        "tracks": len(rows),
        "answered": n,
        "within_50ms_pct": pct(lambda r: r["abs_ms"] <= 50),
        "within_90ms_pct": pct(lambda r: r["abs_ms"] <= 90),
        "within_1_bar_pct": pct(lambda r: abs(r["delta_bars"]) <= 1),
        "within_2_bars_pct": pct(lambda r: abs(r["delta_bars"]) <= 2),
        "median_abs_ms": round(sorted(r["abs_ms"] for r in scored)[n // 2], 1) if n else None,
    }
    print(json.dumps(summary, indent=2))
    out = Path("models/fast_drop_validation.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"[fast-validate] wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stems", default=str(STEMS_ROOT_DIR))
    parser.add_argument("--template", default=str(DEFAULT_ALS_TEMPLATE))
    parser.add_argument("--base-als", default=str(BASE_ALS_TEMPLATE))
    parser.add_argument("--out-dir", default=str(Path(GENERATED_SET_DIR) / "FastDrop"))
    parser.add_argument("--run-stamp", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-truth", action="store_true", help="Score vs human picks; write nothing")
    args = parser.parse_args()

    if args.validate_truth:
        return _validate_truth(args)

    stems = Path(args.stems).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"FAST_DROP_ALL_TRACKS_{args.run_stamp}.als"
    report_path = output.with_name(f"{output.stem}_report.json")

    all_tracks = builder.sort_by_bpm_key_energy(builder.collect_tracks(str(stems)))
    tracks = [t for t in all_tracks if _find_role_audio(Path(str(t["src"])).parent, "drums", t) is not None]
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"[fast] {len(tracks)} tracks, {args.workers} workers")

    started = dt.datetime.now()
    processed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    tasks = [
        {"track": t, "template": args.template, "run_stamp": args.run_stamp, "write_als": True}
        for t in tracks
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            (processed if row.get("status") == "processed" else failures).append(row)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} processed={len(processed)} failures={len(failures)}", flush=True)

    print(f"[fast] processed={len(processed)} failures={len(failures)} in {(dt.datetime.now()-started).total_seconds():.0f}s")
    processed.sort(key=lambda r: (int(r["track"]["bpm"]), str(r["track"]["key"]), str(r["track"]["folder"]).lower()))
    verification = _build_combined_set(Path(args.base_als), output, processed)
    print("[VERIFY]", json.dumps(verification))
    report_path.write_text(
        json.dumps({"processed_rows": processed, "failure_rows": failures, "verification": verification}, indent=1, default=str),
        encoding="utf-8",
    )
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bpm", "key", "energy", "track", "drop_sec", "snapped", "drums_path"])
        for row in processed:
            t = row["track"]
            writer.writerow([t["bpm"], t["key"], t.get("energy"), t["folder"], f"{row['marker']:.6f}", row["fast"].get("snapped"), row["drums_path"]])
    print(f"[DONE] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
