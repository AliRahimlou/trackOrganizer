#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.multistem import choose_multistem_candidate
from train_candidate_chooser import train_candidate_chooser
from web_review import _apply_structure_map_prior, _candidate_marker_time, _float_or_none
from project_config import DROP_BATCH_SUMMARY


DEFAULT_CORRECTIONS = Path("drop_corrections.jsonl")
DEFAULT_SUMMARY = DROP_BATCH_SUMMARY
DEFAULT_REVIEW_STATE = DEFAULT_SUMMARY.parent / "review_state.json"
DEFAULT_OUTPUT_CORRECTIONS = Path("models/post_structure_training_corrections.jsonl")
DEFAULT_MODEL = Path("models/drop_post_structure_candidate_chooser.pkl")
DEFAULT_REPORT = Path("models/post_structure_candidate_chooser_report.json")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_line_no"] = line_no
                row["_truth_source"] = str(path)
                yield row


def _read_summary(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.exists():
        return {}
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {str(row.get("filename", "")): row for row in csv.DictReader(fh) if row.get("filename")}


def _read_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _stable_id(row: Mapping[str, Any]) -> str:
    from web_review import _stable_id as stable_id

    return stable_id(row)


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or row.get("filename") or row.get("audio_path") or "")


def _user_pick(row: Mapping[str, Any]) -> Optional[float]:
    return _float_or_none(row.get("user_pick"))


def _review_state_rows(review_state: Path, summary_path: Path) -> List[Dict[str, Any]]:
    if not review_state.exists():
        return []
    try:
        with open(review_state, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return []
    items = state.get("items") if isinstance(state, Mapping) else {}
    if not isinstance(items, Mapping):
        return []
    row_by_id = {_stable_id(row): row for row in _read_summary_rows(summary_path)}
    out: List[Dict[str, Any]] = []
    for item_index, (item_id, review) in enumerate(items.items(), start=1):
        if not isinstance(review, Mapping) or not bool(review.get("reviewed")):
            continue
        user_pick = _float_or_none(review.get("user_pick"))
        summary_row = row_by_id.get(str(item_id))
        if user_pick is None or not summary_row:
            continue
        track = str(summary_row.get("filename") or "")
        if not track:
            continue
        out.append(
            {
                "_line_no": item_index,
                "_truth_source": str(review_state),
                "track": track,
                "user_pick": float(user_pick),
                "reviewed_from": "review_state",
                "selected_by": summary_row.get("selected_by", ""),
                "ai_pick": _float_or_none(summary_row.get("detected_drop_time")),
                "timestamp": review.get("timestamp_reviewed", ""),
            }
        )
    return out


def _latest_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        track = _track(row)
        if track:
            out[track] = row
    return list(out.values())


def _summary_payload(summary_row: Mapping[str, str]) -> Dict[str, Any]:
    path = str(summary_row.get("candidates_json") or "")
    if not path:
        return {}
    try:
        with open(Path(path).expanduser(), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summary_candidates(summary_row: Mapping[str, str], fallback_row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    payload = _summary_payload(summary_row)
    candidates = payload.get("top_10_candidates") if isinstance(payload.get("top_10_candidates"), list) else []
    if not candidates:
        candidates = fallback_row.get("top_10_candidates") if isinstance(fallback_row.get("top_10_candidates"), list) else []
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else None
    closest = payload.get("closest_candidate_to_user_pick") if isinstance(payload.get("closest_candidate_to_user_pick"), Mapping) else None
    out = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    if closest:
        row = dict(closest)
        row.setdefault("source", "saved_closest_to_review_pick")
        row.setdefault("selected_by", "saved_closest_to_review_pick")
        out.append(row)
    if selected:
        out.append(selected)
    return out


def _summary_tier(summary_row: Mapping[str, str], fallback_row: Mapping[str, Any]) -> str:
    from web_review import _normalize_tier

    payload = _summary_payload(summary_row)
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    return _normalize_tier(
        summary_row.get("confidence_tier")
        or payload.get("confidence_tier")
        or features.get("confidence_tier")
        or fallback_row.get("confidence_tier")
    )


def _dedupe(candidates: Sequence[Mapping[str, Any]], radius_sec: float = 0.010) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    times: List[float] = []
    for candidate in candidates:
        row = dict(candidate)
        t = _candidate_marker_time(row)
        if t is not None and any(abs(float(t) - existing) <= float(radius_sec) for existing in times):
            continue
        out.append(row)
        if t is not None:
            times.append(float(t))
    for rank, row in enumerate(out, start=1):
        row["rank"] = int(rank)
        row["handcrafted_rank"] = int(rank)
    return out


def build_post_structure_training_corrections(
    *,
    corrections: str,
    batch_summary: str,
    review_state: str,
    output: str,
    limit: int = 0,
    offset: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
    expanded_limit: int = 120,
    microalign_limit: int = 50,
    sample_rate: int = 16000,
    track_contains: str = "",
) -> Dict[str, Any]:
    corrections_path = Path(corrections).expanduser()
    summary_path = Path(batch_summary).expanduser()
    review_state_path = Path(review_state).expanduser() if review_state else None
    summary = _read_summary(summary_path)
    rows = [row for row in _iter_jsonl(corrections_path) if not row_has_excluded_path(row)]
    if review_state_path:
        rows.extend(_review_state_rows(review_state_path, summary_path))
    rows = [row for row in rows if _track(row) and _user_pick(row) is not None]
    rows = _latest_rows(rows)
    needle = str(track_contains or "").strip().lower()
    if needle:
        rows = [row for row in rows if needle in _track(row).lower()]
    rows.sort(key=lambda row: _track(row).lower())
    total_rows = len(rows)
    shard_count = max(1, int(shard_count))
    shard_index = max(0, min(shard_count - 1, int(shard_index)))
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if offset > 0:
        rows = rows[int(offset) :]
    if limit > 0:
        rows = rows[: int(limit)]

    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    oracle_25 = 0
    oracle_100 = 0
    oracle_250 = 0
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, start=1):
            track = _track(row)
            user_pick = _user_pick(row)
            if user_pick is None or not Path(track).expanduser().exists():
                counts["missing_file_or_pick"] = counts.get("missing_file_or_pick", 0) + 1
                continue
            summary_row = summary.get(track, {})
            try:
                saved_candidates = _summary_candidates(summary_row, row)
                result = choose_multistem_candidate(
                    track,
                    saved_candidates=saved_candidates,
                    confidence_tier=_summary_tier(summary_row, row),
                    mode="normal",
                    expanded_limit=int(expanded_limit),
                    microalign_limit=int(microalign_limit),
                    sample_rate=int(sample_rate),
                )
                candidate_pool = list(result.get("candidates", []) or [])
                candidate_pool.extend(dict(candidate) for candidate in saved_candidates if isinstance(candidate, Mapping))
                structure_prior = _apply_structure_map_prior(
                    track,
                    candidate_pool,
                    result.get("suggestion", {}) if isinstance(result.get("suggestion"), Mapping) else {},
                    confidence_tier=_summary_tier(summary_row, row),
                    sample_rate=int(sample_rate),
                )
                candidates = _dedupe(structure_prior.get("candidates", []) if isinstance(structure_prior.get("candidates"), Sequence) else [])
                suggestion = structure_prior.get("suggestion", {}) if isinstance(structure_prior.get("suggestion"), Mapping) else {}
            except Exception as exc:
                counts[f"failed:{exc.__class__.__name__}"] = counts.get(f"failed:{exc.__class__.__name__}", 0) + 1
                print(f"[{index}/{len(rows)}] failed {Path(track).name}: {exc}", flush=True)
                continue
            if not candidates:
                counts["no_candidates"] = counts.get("no_candidates", 0) + 1
                continue
            errors = []
            for candidate in candidates:
                marker = _candidate_marker_time(candidate)
                if marker is not None:
                    errors.append(abs(float(marker) - float(user_pick)))
            if errors:
                best = min(errors)
                oracle_25 += 1 if best <= 0.025 else 0
                oracle_100 += 1 if best <= 0.100 else 0
                oracle_250 += 1 if best <= 0.250 else 0
            selected_candidate = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else None
            ai_pick = _float_or_none(suggestion.get("suggested_time"))
            if ai_pick is None and isinstance(selected_candidate, Mapping):
                ai_pick = _candidate_marker_time(selected_candidate)
            out = dict(row)
            out["ai_pick"] = float(ai_pick if ai_pick is not None else user_pick)
            out["top_10_candidates"] = candidates
            if isinstance(selected_candidate, Mapping):
                out["selected_candidate"] = dict(selected_candidate)
            out["post_structure_training"] = {
                "source_count": result.get("source_count"),
                "candidate_count": result.get("candidate_count"),
                "merged_candidate_count": len(candidates),
                "structure_map": structure_prior.get("structure_map", {}),
            }
            fh.write(json.dumps(out, ensure_ascii=True) + "\n")
            written += 1
            counts["written"] = written
            print(f"[{index}/{len(rows)}] post-structure candidates: {Path(track).name}", flush=True)
    return {
        "corrections": str(corrections_path),
        "batch_summary": str(summary_path),
        "review_state": "" if review_state_path is None else str(review_state_path),
        "output": str(out_path),
        "total_truth_rows": int(total_rows),
        "rows": int(len(rows)),
        "written_rows": int(written),
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "counts": counts,
        "oracle_within_25ms_percent": None if written <= 0 else 100.0 * oracle_25 / written,
        "oracle_within_100ms_percent": None if written <= 0 else 100.0 * oracle_100 / written,
        "oracle_within_250ms_percent": None if written <= 0 else 100.0 * oracle_250 / written,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a chooser on post-structure merged drop candidates.")
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS))
    parser.add_argument("--batch-summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--review-state", default=str(DEFAULT_REVIEW_STATE))
    parser.add_argument("--output-corrections", default=str(DEFAULT_OUTPUT_CORRECTIONS))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--expanded-limit", type=int, default=120)
    parser.add_argument("--microalign-limit", type=int, default=50)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--track-contains", default="")
    parser.add_argument("--build-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_report = build_post_structure_training_corrections(
        corrections=str(args.corrections),
        batch_summary=str(args.batch_summary),
        review_state=str(args.review_state or ""),
        output=str(args.output_corrections),
        limit=int(args.limit),
        offset=int(args.offset),
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
        expanded_limit=int(args.expanded_limit),
        microalign_limit=int(args.microalign_limit),
        sample_rate=int(args.analysis_sr),
        track_contains=str(args.track_contains or ""),
    )
    result: Dict[str, Any] = {"build": build_report}
    if not args.build_only:
        result["training"] = train_candidate_chooser(
            corrections=str(args.output_corrections),
            output=str(args.model_output),
            report=str(args.report),
            random_state=int(args.random_state),
            valid_frac=float(args.valid_frac),
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
