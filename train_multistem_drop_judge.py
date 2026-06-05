#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.microalign import microalign_candidate_dicts
from drop_aligner.multistem import generate_multistem_candidates
from drop_aligner.pipeline import run_drop_candidate_pipeline
from train_candidate_chooser import train_candidate_chooser


DEFAULT_OUTPUT_CORRECTIONS = Path("models/multistem_training_corrections.jsonl")
DEFAULT_MODEL = Path("models/drop_multistem_candidate_chooser.pkl")
DEFAULT_REPORT = Path("models/multistem_candidate_chooser_report.json")


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
    return out


def _source_candidates(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    source: List[Any] = []
    candidates = row.get("top_10_candidates")
    if isinstance(candidates, list):
        source.extend(candidates)
    else:
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            source.extend(candidates)
    selected = row.get("selected_candidate")
    if isinstance(selected, Mapping):
        source.append(selected)

    out: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in source:
        if not isinstance(candidate, Mapping):
            continue
        t = None
        micro = candidate.get("microalign")
        if isinstance(micro, Mapping):
            t = _safe_float(micro.get("microaligned_time"))
        for key in ("microaligned_time", "timestamp", "snapped_sec", "time_sec"):
            if t is None:
                t = _safe_float(candidate.get(key))
        key = f"{t:.6f}" if t is not None else json.dumps(dict(candidate), sort_keys=True, default=str)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def build_multistem_training_corrections(
    *,
    corrections: str,
    output: str,
    limit: int = 0,
    expanded_limit: int = 120,
    microalign_limit: int = 50,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    source = Path(corrections).expanduser()
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _iter_jsonl(source) if not row_has_excluded_path(row)]
    if int(limit) > 0:
        rows = rows[: int(limit)]

    counts: Dict[str, int] = {}
    written = 0
    oracle_25 = 0
    oracle_100 = 0
    oracle_250 = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, start=1):
            track = str(row.get("track") or row.get("filename") or "")
            user_pick = _safe_float(row.get("user_pick"))
            if not track or user_pick is None:
                counts["missing_track_or_pick"] = counts.get("missing_track_or_pick", 0) + 1
                continue
            try:
                generated = generate_multistem_candidates(
                    track,
                    saved_candidates=_source_candidates(row),
                    limit=int(expanded_limit),
                    sample_rate=int(sample_rate),
                )
                candidates = list(generated.get("candidates") or [])
                if int(microalign_limit) > 0:
                    candidates = microalign_candidate_dicts(track, candidates, limit=int(microalign_limit))
                pipeline = run_drop_candidate_pipeline(
                    candidates,
                    cluster_radius_sec=0.085,
                    limit=int(microalign_limit) if int(microalign_limit) > 0 else None,
                )
                candidates = list(pipeline.get("candidates") or candidates)
            except Exception as exc:
                counts[f"failed:{exc.__class__.__name__}"] = counts.get(f"failed:{exc.__class__.__name__}", 0) + 1
                print(f"[{index}/{len(rows)}] failed: {track}: {exc}", flush=True)
                continue
            if not candidates:
                counts["no_candidates"] = counts.get("no_candidates", 0) + 1
                continue

            errors: List[float] = []
            for candidate in candidates:
                value = candidate.get("microaligned_time") or candidate.get("timestamp") or candidate.get("snapped_sec")
                parsed = _safe_float(value)
                if parsed is not None:
                    errors.append(abs(float(parsed) - float(user_pick)))
            if errors:
                best = min(errors)
                oracle_25 += 1 if best <= 0.025 else 0
                oracle_100 += 1 if best <= 0.100 else 0
                oracle_250 += 1 if best <= 0.250 else 0

            out = dict(row)
            out["top_10_candidates"] = candidates
            out["multistem_training"] = {
                "source_count": int(generated.get("source_count", 0) or 0),
                "candidate_count": int(generated.get("candidate_count", 0) or 0),
                "stem_group": generated.get("stem_group"),
                "pipeline": pipeline.get("summary") if isinstance(pipeline.get("summary"), Mapping) else {},
            }
            fh.write(json.dumps(out, ensure_ascii=True) + "\n")
            written += 1
            counts["written"] = written
            print(f"[{index}/{len(rows)}] training candidates: {track}", flush=True)

    summary = {
        "corrections": str(source),
        "output": str(out_path),
        "input_rows": int(len(rows)),
        "written_rows": int(written),
        "counts": counts,
        "oracle_within_25ms_percent": None if written <= 0 else 100.0 * oracle_25 / written,
        "oracle_within_100ms_percent": None if written <= 0 else 100.0 * oracle_100 / written,
        "oracle_within_250ms_percent": None if written <= 0 else 100.0 * oracle_250 / written,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a candidate chooser on expanded multi-stem candidates.")
    parser.add_argument("--corrections", default="drop_corrections.jsonl", help="Source correction JSONL")
    parser.add_argument("--output-corrections", default=str(DEFAULT_OUTPUT_CORRECTIONS), help="Expanded multistem training JSONL")
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL), help="Output chooser model path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output training report JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Only build this many correction rows")
    parser.add_argument("--expanded-limit", type=int, default=120)
    parser.add_argument("--microalign-limit", type=int, default=50)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--valid-frac", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--build-only", action="store_true", help="Only build the expanded correction JSONL")
    parser.add_argument("--promote", action="store_true", help="Also copy the trained model to models/drop_candidate_chooser.pkl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_report = build_multistem_training_corrections(
        corrections=str(args.corrections),
        output=str(args.output_corrections),
        limit=int(args.limit),
        expanded_limit=int(args.expanded_limit),
        microalign_limit=int(args.microalign_limit),
        sample_rate=int(args.analysis_sr),
    )
    result: Dict[str, Any] = {"build": build_report}
    if not args.build_only:
        train_report = train_candidate_chooser(
            corrections=str(args.output_corrections),
            output=str(args.model_output),
            report=str(args.report),
            random_state=int(args.random_state),
            valid_frac=float(args.valid_frac),
        )
        result["training"] = train_report
        if args.promote:
            default_model = Path("models/drop_candidate_chooser.pkl")
            default_model.parent.mkdir(parents=True, exist_ok=True)
            default_model.write_bytes(Path(args.model_output).read_bytes())
            result["promoted_to"] = str(default_model)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
