#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from drop_aligner.als import modify_als
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.legacy_write_guard import add_legacy_detector_write_arg, require_legacy_detector_write_opt_in
from drop_aligner.microalign import choose_microaligned_candidate, microalign_candidate_dicts, should_auto_accept
from verify_als import verify_als


TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _read_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path).expanduser()
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _read_summary(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(Path(path).expanduser(), "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row_has_excluded_path(row):
                rows.append(dict(row))
    return rows


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").strip().upper()
    return tier if tier in TIER_ORDER else "UNKNOWN"


def _candidate_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = payload.get("top_10_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        selected = payload.get("selected_candidate")
        candidates = [selected] if isinstance(selected, Mapping) else []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _filter_rows(rows: Sequence[Dict[str, str]], *, low_only: bool, medium_and_low: bool, max_tracks: Optional[int]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        payload = _read_json(row.get("candidates_json", ""))
        feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
        tier = _normalize_tier(row.get("confidence_tier") or payload.get("confidence_tier") or feature_summary.get("confidence_tier"))
        if low_only and tier != "LOW":
            continue
        if medium_and_low and tier not in {"LOW", "MEDIUM"}:
            continue
        item = dict(row)
        item["confidence_tier"] = tier
        out.append(item)
    out.sort(key=lambda row: (TIER_ORDER.get(row.get("confidence_tier", "UNKNOWN"), 3), row.get("filename", "")))
    if max_tracks is not None:
        out = out[: max(0, int(max_tracks))]
    return out


def _backup_als(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    backup = path.with_name(path.stem + ".previous" + path.suffix)
    shutil.copy2(path, backup)
    return str(backup)


def _regenerate_als(row: Mapping[str, str], template: str, marker_time: float, bpm: float) -> Dict[str, Any]:
    output = Path(str(row.get("output_als") or "")).expanduser()
    if not output:
        return {"ok": False, "error": "missing output_als"}
    backup = _backup_als(output)
    try:
        modify_als(
            template_path=template,
            audio_path=str(row.get("filename", "")),
            drop_sec=float(marker_time),
            bpm=float(bpm or 128.0),
            output_path=str(output),
            strict_stems=True,
        )
        report = verify_als(str(output))
        return {"ok": bool(report.get("valid")), "output_als": str(output), "backup": backup, "verification": report}
    except Exception as exc:
        return {"ok": False, "error": str(exc) or exc.__class__.__name__, "backup": backup}


def _write_auto_log(path: str, row: Mapping[str, Any]) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    return str(out)


def _process_row(
    row: Mapping[str, str],
    *,
    template: str,
    mode: str,
    regenerate_als: bool,
    auto_log: str,
) -> Dict[str, Any]:
    audio_path = str(row.get("filename") or "")
    payload = _read_json(str(row.get("candidates_json") or ""))
    feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    candidates = _candidate_rows(payload)
    tier = _normalize_tier(row.get("confidence_tier") or payload.get("confidence_tier") or feature_summary.get("confidence_tier"))
    bpm = _float_or_none(payload.get("bpm")) or _float_or_none(feature_summary.get("bpm")) or 128.0
    result: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "track": audio_path,
        "mode": mode,
        "confidence_tier": tier,
        "selected_by": row.get("selected_by") or payload.get("selected_by") or "",
        "auto_place": False,
        "suggestion_available": False,
        "review_needed": True,
        "suggested_time": None,
        "reason": "",
        "microaligned_candidates": [],
        "auto_accept_gate": {
            "auto_accept": False,
            "reason": "no MicroSnap suggestion",
            "risk_flags": ["no MicroSnap suggestion"],
            "mode": mode,
            "confidence_tier": tier,
        },
        "regeneration": None,
        "log_type": "auto_marker",
    }
    if not candidates:
        result["reason"] = "no candidate JSON candidates available"
        _write_auto_log(auto_log, result)
        return result

    aligned = microalign_candidate_dicts(audio_path, candidates, limit=10)
    suggestion = choose_microaligned_candidate(aligned, confidence_tier=tier, mode=mode)
    candidate = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else None
    gate = (
        should_auto_accept(candidate, mode=mode, candidates=aligned, confidence_tier=tier)
        if candidate is not None
        else result["auto_accept_gate"]
    )
    result["microaligned_candidates"] = aligned
    result["suggestion_available"] = suggestion.get("suggested_time") is not None
    result["auto_accept_gate"] = gate
    result["auto_place"] = bool(gate.get("auto_accept"))
    result["review_needed"] = not bool(gate.get("auto_accept"))
    result["risk"] = bool(suggestion.get("risk", False)) or not bool(gate.get("auto_accept"))
    result["reason"] = str(gate.get("reason") or suggestion.get("reason", ""))
    result["suggestion"] = suggestion
    if suggestion.get("suggested_time") is not None:
        result["suggested_time"] = float(suggestion["suggested_time"])

    if regenerate_als and result["auto_place"] and result["suggested_time"] is not None:
        result["regeneration"] = _regenerate_als(row, template, float(result["suggested_time"]), float(bpm))

    _write_auto_log(auto_log, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-suggest sample-level 1.1.1 markers without writing human correction labels.")
    parser.add_argument("summary_csv", help="drop_batch_summary.csv from batch.py")
    parser.add_argument("--template", required=True)
    parser.add_argument("--mode", choices=["conservative", "normal", "aggressive"], default="conservative")
    parser.add_argument("--regenerate-als", action="store_true")
    parser.add_argument("--write-auto-log", default="auto_marks.jsonl")
    parser.add_argument("--review-low-only", action="store_true")
    parser.add_argument("--review-medium-and-low", action="store_true")
    parser.add_argument("--max-tracks", type=int)
    add_legacy_detector_write_arg(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    require_legacy_detector_write_opt_in(
        "auto_review.py",
        action="writing legacy auto-review logs or regenerated ALS files",
        explicit=bool(args.allow_legacy_detector_write),
    )
    rows = _filter_rows(
        _read_summary(args.summary_csv),
        low_only=bool(args.review_low_only),
        medium_and_low=bool(args.review_medium_and_low),
        max_tracks=args.max_tracks,
    )

    counts = {"auto_place": 0, "review_needed": 0, "regenerated": 0}
    for index, row in enumerate(rows, start=1):
        result = _process_row(
            row,
            template=str(Path(args.template).expanduser()),
            mode=args.mode,
            regenerate_als=bool(args.regenerate_als),
            auto_log=args.write_auto_log,
        )
        counts["auto_place"] += 1 if result.get("auto_place") else 0
        counts["review_needed"] += 1 if result.get("review_needed") else 0
        regen = result.get("regeneration")
        counts["regenerated"] += 1 if isinstance(regen, Mapping) and regen.get("ok") else 0
        print(
            f"[{index}/{len(rows)}] "
            f"{'AUTO' if result.get('auto_place') else 'REVIEW'} "
            f"{result.get('confidence_tier')} "
            f"{result.get('suggested_time') if result.get('suggested_time') is not None else ''} "
            f"{Path(str(result.get('track', ''))).name}",
            flush=True,
        )

    print(json.dumps({"tracks": len(rows), "counts": counts, "auto_log": args.write_auto_log}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
