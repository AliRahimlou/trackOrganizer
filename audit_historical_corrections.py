#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.multistem import find_stem_group, infer_bpm_from_path
from drop_aligner.musical_clock import bpm_clock_for_time


DEFAULT_CORRECTIONS = Path("drop_corrections.jsonl")
DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_JSON = Path("models/historical_data_quality_audit.json")
DEFAULT_CSV = Path("models/historical_data_quality_bad_rows.csv")
AUDIO_ROLES = ("drums", "instrumental", "vocals", "bass")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                yield {"_line_no": line_no, "_bad_json": True}
                continue
            if isinstance(row, dict):
                row["_line_no"] = line_no
                yield row


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or row.get("filename") or row.get("audio_path") or "")


def _read_summary(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {str(row.get("filename", "")): row for row in csv.DictReader(fh) if row.get("filename")}


def _duration_sec(path: str) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def _stem_durations(track: str) -> Dict[str, Optional[float]]:
    try:
        group = find_stem_group(track)
    except Exception:
        return {}
    durations: Dict[str, Optional[float]] = {}
    for role, path in group.roles.items():
        if role in AUDIO_ROLES:
            durations[role] = _duration_sec(path)
    return durations


def _summary_bpm(summary_row: Mapping[str, str]) -> Optional[float]:
    for key in ("bpm", "detected_bpm"):
        value = _float_or_none(summary_row.get(key))
        if value is not None:
            return value
    candidates_json = str(summary_row.get("candidates_json") or "")
    if candidates_json:
        try:
            with open(Path(candidates_json).expanduser(), "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for key in ("bpm",):
                value = _float_or_none(payload.get(key))
                if value is not None:
                    return value
            features = payload.get("feature_summary")
            if isinstance(features, Mapping):
                value = _float_or_none(features.get("bpm"))
                if value is not None:
                    return value
        except Exception:
            return None
    return None


def _bad_row(row: Mapping[str, Any], reasons: Sequence[str], extra: Mapping[str, Any]) -> Dict[str, Any]:
    track = _track(row)
    return {
        "line_no": row.get("_line_no", ""),
        "track": track,
        "user_pick": row.get("user_pick", ""),
        "ai_pick": row.get("ai_pick", row.get("final_ai_pick", "")),
        "reviewed_from": row.get("reviewed_from", ""),
        "selected_by": row.get("selected_by", ""),
        "reasons": ";".join(reasons),
        **dict(extra),
    }


def audit_historical_corrections(
    *,
    corrections: str,
    batch_summary: str,
    output_json: str,
    output_csv: str,
    duration_mismatch_sec: float = 1.0,
) -> Dict[str, Any]:
    correction_path = Path(corrections).expanduser()
    summary_path = Path(batch_summary).expanduser()
    summary = _read_summary(summary_path)
    rows = list(_iter_jsonl(correction_path))
    by_track: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[_track(row)].append(row)

    counts: Counter[str] = Counter()
    bad_rows: List[Dict[str, Any]] = []
    duplicate_tracks: List[Dict[str, Any]] = []
    missing_roles_counter: Counter[str] = Counter()

    for track, track_rows in sorted(by_track.items()):
        if not track:
            counts["missing_track"] += len(track_rows)
            continue
        if len(track_rows) > 1:
            picks = sorted(
                {
                    round(float(value), 6)
                    for value in (_float_or_none(row.get("user_pick")) for row in track_rows)
                    if value is not None
                }
            )
            duplicate_tracks.append({"track": track, "rows": len(track_rows), "unique_user_picks": picks})
            counts["duplicate_track_groups"] += 1

    for row in rows:
        reasons: List[str] = []
        extra: Dict[str, Any] = {}
        if row.get("_bad_json"):
            bad_rows.append(_bad_row(row, ["bad_json"], extra))
            counts["bad_json"] += 1
            continue
        track = _track(row)
        user_pick = _float_or_none(row.get("user_pick"))
        ai_pick = _float_or_none(row.get("ai_pick", row.get("final_ai_pick")))
        if not track:
            reasons.append("missing_track")
        path = Path(track).expanduser() if track else Path("")
        exists = bool(track and path.exists())
        extra["file_exists"] = exists
        if not exists:
            reasons.append("missing_audio_file")
        if user_pick is None:
            reasons.append("missing_user_pick")
        if ai_pick is None:
            reasons.append("missing_ai_pick")

        duration = _duration_sec(str(path)) if exists else None
        extra["duration_sec"] = "" if duration is None else round(float(duration), 6)
        if user_pick is not None and duration is not None:
            if user_pick < -0.010 or user_pick > duration + 0.250:
                reasons.append("user_pick_outside_duration")

        parsed_bpm = infer_bpm_from_path(path) if track else None
        summary_row = summary.get(track, {})
        summary_bpm = _summary_bpm(summary_row) if summary_row else None
        extra["path_bpm"] = "" if parsed_bpm is None else parsed_bpm
        extra["summary_bpm"] = "" if summary_bpm is None else summary_bpm
        if parsed_bpm is None:
            reasons.append("bpm_not_parseable_from_path")
        if parsed_bpm is not None and summary_bpm is not None and abs(float(parsed_bpm) - float(summary_bpm)) >= 0.75:
            reasons.append("path_bpm_summary_bpm_mismatch")

        if user_pick is not None and parsed_bpm is not None:
            clock = bpm_clock_for_time(user_pick, parsed_bpm)
            if clock:
                extra["user_clock_bar"] = clock.get("nearest_one_bar")
                extra["user_one_distance_ms"] = round(float(clock.get("one_distance_ms", 0.0) or 0.0), 3)
                if float(clock.get("one_distance_ms", 999.0) or 999.0) > 80.0:
                    reasons.append("approved_marker_far_from_title_bpm_one")

        durations = _stem_durations(track) if exists else {}
        present_roles = set(durations)
        missing_roles = [role for role in ("drums", "instrumental", "vocals") if role not in present_roles]
        for role in missing_roles:
            missing_roles_counter[role] += 1
        if missing_roles:
            reasons.append("missing_core_stem:" + ",".join(missing_roles))
        finite_durations = [float(value) for value in durations.values() if value is not None]
        if finite_durations:
            mismatch = max(finite_durations) - min(finite_durations)
            extra["stem_duration_mismatch_sec"] = round(float(mismatch), 6)
            if mismatch > float(duration_mismatch_sec):
                reasons.append("stem_duration_mismatch")
        if duration is not None and parsed_bpm is not None:
            bar_sec = (60.0 / float(parsed_bpm)) * 4.0
            bar_count = int(math.floor(duration / max(1e-6, bar_sec)))
            extra["title_clock_bar_count"] = bar_count
            if bar_count < 32:
                reasons.append("short_or_truncated_under_32_bars")

        for reason in reasons:
            counts[reason.split(":", 1)[0]] += 1
        if reasons:
            bad_rows.append(_bad_row(row, reasons, extra))

    report: Dict[str, Any] = {
        "corrections": str(correction_path),
        "batch_summary": str(summary_path),
        "rows": int(len(rows)),
        "unique_tracks": int(len(by_track)),
        "counts": dict(counts),
        "duplicate_tracks": duplicate_tracks[:200],
        "missing_core_stems": dict(missing_roles_counter),
        "bad_row_count": int(len(bad_rows)),
        "bad_rows_sample": bad_rows[:100],
    }

    json_path = Path(output_json).expanduser()
    csv_path = Path(output_csv).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    fieldnames = [
        "line_no",
        "track",
        "user_pick",
        "ai_pick",
        "reviewed_from",
        "selected_by",
        "reasons",
        "file_exists",
        "duration_sec",
        "path_bpm",
        "summary_bpm",
        "user_clock_bar",
        "user_one_distance_ms",
        "stem_duration_mismatch_sec",
        "title_clock_bar_count",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in bad_rows:
            writer.writerow(row)
    report["output_json"] = str(json_path)
    report["output_csv"] = str(csv_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only data-quality audit for historical drop corrections.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS))
    parser.add_argument("--batch-summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--duration-mismatch-sec", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_historical_corrections(
        corrections=str(args.corrections),
        batch_summary=str(args.batch_summary),
        output_json=str(args.output_json),
        output_csv=str(args.output_csv),
        duration_mismatch_sec=float(args.duration_mismatch_sec),
    )
    compact = {
        "rows": report["rows"],
        "unique_tracks": report["unique_tracks"],
        "bad_row_count": report["bad_row_count"],
        "counts": report["counts"],
        "output_json": report["output_json"],
        "output_csv": report["output_csv"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
