#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from drop_aligner.multistem import infer_bpm_from_path  # noqa: E402
from drop_aligner.musical_clock import bpm_clock_for_time  # noqa: E402
from drop_aligner.structure_features import compute_bar_feature_map  # noqa: E402
from drop_aligner.visual_first import (  # noqa: E402
    EXTENDED_VISUAL_MAX_CLOCK_BAR,
    summarize_visual_candidate,
    visual_body_peak_candidates,
    visual_chunk_candidates,
    visual_first_marker,
)
from drop_aligner.waveform import WaveformCache  # noqa: E402
from evaluate_visual_first import DEFAULT_CORRECTIONS, _load_truth_rows  # noqa: E402


DEFAULT_JSONL = "models/visual_candidate_atlas.jsonl"
DEFAULT_CSV = "models/visual_candidate_atlas_summary.csv"
DEFAULT_IMAGE_DIR = "artifacts/visual_candidate_atlas"
VISUAL_KEYS = (
    "clock_bar",
    "phrase_prior",
    "post4_height",
    "post8_height",
    "pre4_height",
    "jump4",
    "jump8",
    "post_bass8",
    "post_drum8",
    "post_inst8",
    "pre_inst4",
    "pre_drum_cont4",
    "post_drum_cont4",
    "post_drum_cont8",
    "local_reentry_gap",
    "frame_body",
    "frame_pre_body",
    "frame_post_body",
    "frame_attack_peak",
    "frame_low_peak",
    "frame_score",
)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    micro = candidate.get("microalign") if isinstance(candidate.get("microalign"), Mapping) else {}
    for source in (micro, candidate):
        for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec", "visual_raw_chunk_time"):
            value = _float_or_none(source.get(key))
            if value is not None:
                return float(value)
    return None


def _candidate_raw_time(candidate: Mapping[str, Any]) -> Optional[float]:
    return _float_or_none(candidate.get("visual_raw_chunk_time")) or _candidate_time(candidate)


def _candidate_visual(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("visual_components")
    return value if isinstance(value, Mapping) else {}


def _candidate_clock_bar(candidate: Mapping[str, Any]) -> int:
    visual = _candidate_visual(candidate)
    try:
        return int(visual.get("clock_bar", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _dedupe_candidates(candidates: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        time_sec = _candidate_raw_time(candidate)
        if time_sec is None:
            continue
        key = int(round(float(time_sec) * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(candidate))
    out.sort(key=lambda row: float(_candidate_raw_time(row) or 1e18))
    return out


def _clock_fields(time_sec: Optional[float], bpm: Optional[float], clock_zero_sec: float) -> Dict[str, Any]:
    clock = bpm_clock_for_time(time_sec, bpm, clock_zero_sec=clock_zero_sec) if time_sec is not None and bpm else None
    if not clock:
        return {"on_one": False, "one_distance_ms": None, "nearest_one_bar": None, "beat_in_bar": None}
    return {
        "on_one": bool(clock.get("on_one")),
        "one_distance_ms": _float_or_none(clock.get("one_distance_ms")),
        "nearest_one_bar": clock.get("nearest_one_bar"),
        "beat_in_bar": clock.get("beat_in_bar"),
    }


def _candidate_row(
    *,
    truth: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_index: int,
    expected: float,
    oracle_index: int,
    detector_time: Optional[float],
    bpm: Optional[float],
    clock_zero_sec: float,
) -> Dict[str, Any]:
    time_sec = _candidate_time(candidate)
    raw_time = _candidate_raw_time(candidate)
    error_ms = None if time_sec is None else abs(float(time_sec) - float(expected)) * 1000.0
    visual = _candidate_visual(candidate)
    clock = _clock_fields(time_sec, bpm, clock_zero_sec)
    row: Dict[str, Any] = {
        "track": truth.get("track"),
        "expected": float(expected),
        "reviewed_from": truth.get("reviewed_from"),
        "truth_selected_by": truth.get("selected_by"),
        "source_path": truth.get("source_path"),
        "line_no": truth.get("line_no"),
        "candidate_index": int(candidate_index),
        "candidate_time": time_sec,
        "candidate_raw_time": raw_time,
        "candidate_error_ms": error_ms,
        "candidate_match_25ms": bool(error_ms is not None and error_ms <= 25.0),
        "candidate_match_50ms": bool(error_ms is not None and error_ms <= 50.0),
        "candidate_match_250ms": bool(error_ms is not None and error_ms <= 250.0),
        "candidate_is_oracle": bool(candidate_index == oracle_index),
        "candidate_is_detector": bool(
            detector_time is not None and time_sec is not None and abs(float(time_sec) - float(detector_time)) <= 0.050
        ),
        "selected_by": str(candidate.get("selected_by") or ""),
        "score": _float_or_none(candidate.get("score")) or _float_or_none(candidate.get("confidence_score")),
        "rank": candidate.get("rank"),
        "reason": str(candidate.get("reason") or ""),
        **clock,
    }
    for key in VISUAL_KEYS:
        row[f"visual_{key}"] = visual.get(key)
    row["visual_local_reentry"] = bool(visual.get("local_reentry"))
    row["visual_phrase_body_shift"] = bool(visual.get("phrase_body_shift"))
    try:
        summary = summarize_visual_candidate(candidate)
        row["drop_strength"] = summary.get("drop_strength")
        row["real_drop_like"] = summary.get("real_drop_like")
        row["instrumental_bass_section_entry"] = summary.get("instrumental_bass_section_entry")
    except Exception:
        row["drop_strength"] = None
        row["real_drop_like"] = None
        row["instrumental_bass_section_entry"] = None
    return row


def _build_one(payload: Mapping[str, Any]) -> Dict[str, Any]:
    track = str(payload["track"])
    expected = float(payload["expected"])
    sample_rate = int(payload["sample_rate"])
    started = time.time()
    try:
        detector = visual_first_marker(track, sample_rate=sample_rate, use_cache=True)
        selected = detector.get("selected_candidate") if isinstance(detector.get("selected_candidate"), Mapping) else {}
        audit = detector.get("visual_audit") if isinstance(detector.get("visual_audit"), Mapping) else {}
        feature_map = compute_bar_feature_map(track, sample_rate=sample_rate, use_cache=True)
        max_clock_bar = max(
            81,
            min(EXTENDED_VISUAL_MAX_CLOCK_BAR, int(feature_map.get("bar_count", 0) or EXTENDED_VISUAL_MAX_CLOCK_BAR)),
        )
        candidates = visual_chunk_candidates(feature_map, max_clock_bar=max_clock_bar)
        candidates.extend(
            visual_body_peak_candidates(
                track,
                feature_map,
                sample_rate=sample_rate,
                max_clock_bar=max_clock_bar,
            )
        )
    except Exception as exc:
        return {
            "truth": dict(payload),
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
            "candidate_rows": [],
            "summary": {
                "track": track,
                "expected": expected,
                "ok": False,
                "error": str(exc) or exc.__class__.__name__,
            },
        }

    rows = _dedupe_candidates([selected, *candidates])
    detector_time = _float_or_none(detector.get("marker")) or _candidate_time(selected)
    feature_beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else {}
    bpm = _float_or_none(feature_beatgrid.get("bpm")) or infer_bpm_from_path(track)
    clock_zero_sec = _float_or_none(feature_beatgrid.get("bar_zero_sec")) or 0.0
    errors = [
        abs(float(_candidate_time(candidate) or 1e18) - expected)
        for candidate in rows
    ]
    oracle_index = int(min(range(len(errors)), key=lambda index: errors[index])) if errors else -1
    candidate_rows = [
        _candidate_row(
            truth=payload,
            candidate=candidate,
            candidate_index=index,
            expected=expected,
            oracle_index=oracle_index,
            detector_time=detector_time,
            bpm=bpm,
            clock_zero_sec=clock_zero_sec,
        )
        for index, candidate in enumerate(rows)
    ]
    detector_error_ms = None if detector_time is None else abs(float(detector_time) - expected) * 1000.0
    oracle_error_ms = None if oracle_index < 0 else candidate_rows[oracle_index].get("candidate_error_ms")
    summary = {
        "track": track,
        "expected": expected,
        "reviewed_from": payload.get("reviewed_from"),
        "truth_selected_by": payload.get("selected_by"),
        "ok": True,
        "detector_time": detector_time,
        "detector_error_ms": detector_error_ms,
        "detector_selected_by": str(selected.get("selected_by") or ""),
        "detector_clock_bar": _candidate_clock_bar(selected),
        "audit_status": str(audit.get("status") or ""),
        "audit_action": str(audit.get("recommended_action") or ""),
        "audit_flags": list(audit.get("flag_codes") or []),
        "audit_preferred_time": (
            _float_or_none(audit.get("preferred_candidate", {}).get("time"))
            if isinstance(audit.get("preferred_candidate"), Mapping)
            else None
        ),
        "candidate_count": int(len(candidate_rows)),
        "oracle_time": candidate_rows[oracle_index].get("candidate_time") if oracle_index >= 0 else None,
        "oracle_error_ms": oracle_error_ms,
        "oracle_selected_by": candidate_rows[oracle_index].get("selected_by") if oracle_index >= 0 else "",
        "oracle_clock_bar": candidate_rows[oracle_index].get("visual_clock_bar") if oracle_index >= 0 else None,
        "oracle_within_50ms": bool(oracle_error_ms is not None and float(oracle_error_ms) <= 50.0),
        "oracle_within_250ms": bool(oracle_error_ms is not None and float(oracle_error_ms) <= 250.0),
        "elapsed_sec": time.time() - started,
    }
    return {"truth": dict(payload), "ok": True, "candidate_rows": candidate_rows, "summary": summary}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _render_images(rows: Sequence[Mapping[str, Any]], *, image_dir: Path, limit: int, width: int, height: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    cache = WaveformCache(Path("artifacts/waveform_cache"))
    rendered = 0
    for row in rows:
        if rendered >= int(limit):
            break
        if not row.get("ok"):
            continue
        track = str(row.get("track") or "")
        if not track:
            continue
        expected = _float_or_none(row.get("expected"))
        detector_time = _float_or_none(row.get("detector_time"))
        oracle_time = _float_or_none(row.get("oracle_time"))
        if expected is None:
            continue
        try:
            info = cache.info(track)
            duration = float(info.get("duration", 0.0) or 0.0)
            markers = [{"time": expected, "kind": "user", "label": "HUMAN"}]
            if detector_time is not None:
                markers.append({"time": detector_time, "kind": "ai", "label": "BLUE"})
            if oracle_time is not None:
                markers.append({"time": oracle_time, "kind": "candidate", "label": "ORACLE"})
            data = cache.render_png(track, start_sec=0.0, end_sec=duration, width=width, height=height, markers=markers)
            name = f"{rendered + 1:04d}_{Path(track).stem[:90].replace('/', '_')}.png"
            (image_dir / name).write_bytes(data)
            rendered += 1
        except Exception:
            continue


def build(args: argparse.Namespace) -> Dict[str, Any]:
    if args.render_from_summary:
        summary_path = Path(args.render_from_summary).expanduser()
        with open(summary_path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        render_source: Sequence[Mapping[str, Any]] = rows
        if args.render_misses_first:
            render_source = sorted(
                rows,
                key=lambda row: float(row.get("detector_error_ms") if row.get("detector_error_ms") not in (None, "") else -1.0),
                reverse=True,
            )
        _render_images(
            render_source,
            image_dir=Path(args.image_dir).expanduser(),
            limit=int(args.image_limit),
            width=int(args.image_width),
            height=int(args.image_height),
        )
        out = {
            "rendered_from_summary": str(summary_path),
            "image_dir": str(Path(args.image_dir).expanduser()),
            "image_limit": int(args.image_limit),
        }
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return out

    truth_rows = _load_truth_rows(
        args.corrections,
        track_contains=str(args.track_contains or ""),
        all_observations=bool(args.all_observations),
    )
    if args.limit:
        truth_rows = truth_rows[: int(args.limit)]
    jobs = [dict(row, sample_rate=int(args.sample_rate)) for row in truth_rows]
    workers = max(1, int(args.workers))
    results: List[Dict[str, Any]] = []
    started = time.time()
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_build_one(job))
            if args.progress_every and index % int(args.progress_every) == 0:
                print(f"built {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_build_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"built {index}/{len(jobs)}", flush=True)

    results.sort(key=lambda row: str(row.get("summary", {}).get("track") or row.get("truth", {}).get("track") or ""))
    candidate_rows = [candidate for result in results for candidate in result.get("candidate_rows", [])]
    summary_rows = [dict(result.get("summary") or {}) for result in results]
    output_jsonl = Path(args.output_jsonl).expanduser()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as fh:
        for row in candidate_rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    _write_csv(Path(args.output_csv).expanduser(), summary_rows)
    if args.render_images:
        render_source = summary_rows
        if args.render_misses_first:
            render_source = sorted(
                summary_rows,
                key=lambda row: float(row.get("detector_error_ms") if row.get("detector_error_ms") not in (None, "") else -1.0),
                reverse=True,
            )
        _render_images(
            render_source,
            image_dir=Path(args.image_dir).expanduser(),
            limit=int(args.image_limit),
            width=int(args.image_width),
            height=int(args.image_height),
        )
    ok_summaries = [row for row in summary_rows if row.get("ok")]
    oracle_50 = sum(1 for row in ok_summaries if bool(row.get("oracle_within_50ms")))
    detector_50 = sum(
        1
        for row in ok_summaries
        if _float_or_none(row.get("detector_error_ms")) is not None
        and float(row.get("detector_error_ms")) <= 50.0
    )
    out = {
        "tracks": int(len(summary_rows)),
        "ok_tracks": int(len(ok_summaries)),
        "candidate_rows": int(len(candidate_rows)),
        "detector_within_50ms": int(detector_50),
        "oracle_candidate_within_50ms": int(oracle_50),
        "oracle_candidate_within_250ms": int(sum(1 for row in ok_summaries if bool(row.get("oracle_within_250ms")))),
        "elapsed_sec": time.time() - started,
        "output_jsonl": str(output_jsonl),
        "output_csv": str(Path(args.output_csv).expanduser()),
    }
    print(json.dumps(out, indent=2, ensure_ascii=True))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual-first candidate atlas from human-reviewed drop markers.")
    parser.add_argument("--corrections", action="append", default=list(DEFAULT_CORRECTIONS), help="Human correction JSONL. Repeatable.")
    parser.add_argument("--track-contains", default="")
    parser.add_argument("--all-observations", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    parser.add_argument("--render-images", action="store_true")
    parser.add_argument("--render-from-summary", default="", help="Render PNGs from an existing atlas summary CSV and exit.")
    parser.add_argument("--render-misses-first", action="store_true")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-limit", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=2400)
    parser.add_argument("--image-height", type=int, default=720)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
