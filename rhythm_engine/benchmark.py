from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from .cli import _read_times
from .engine import analyze_rhythm
from .learn_weights import learn_provider_weights_from_payloads
from .types import RhythmEngineConfig


@dataclass(frozen=True)
class BenchmarkItem:
    audio: str
    reference_beats: Optional[str] = None
    reference_downbeats: Optional[str] = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio,
            "reference_beats": self.reference_beats,
            "reference_downbeats": self.reference_downbeats,
            "metadata": dict(self.metadata or {}),
        }


def _item_from_mapping(row: Mapping[str, Any], *, base_dir: Path) -> BenchmarkItem:
    audio = str(row.get("audio") or row.get("audio_path") or row.get("path") or "").strip()
    if not audio:
        raise ValueError("Manifest row is missing audio/audio_path/path")
    beats = row.get("reference_beats") or row.get("beats") or row.get("beats_path")
    downbeats = row.get("reference_downbeats") or row.get("downbeats") or row.get("downbeats_path")

    def resolve(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path)

    audio_path = Path(audio).expanduser()
    if not audio_path.is_absolute():
        audio_path = base_dir / audio_path
    metadata = {
        str(key): value
        for key, value in row.items()
        if key not in {"audio", "audio_path", "path", "reference_beats", "beats", "beats_path", "reference_downbeats", "downbeats", "downbeats_path"}
    }
    return BenchmarkItem(
        audio=str(audio_path),
        reference_beats=resolve(beats),
        reference_downbeats=resolve(downbeats),
        metadata=metadata,
    )


def read_manifest(path: str) -> List[BenchmarkItem]:
    manifest = Path(path).expanduser()
    base_dir = manifest.parent
    if manifest.suffix.lower() == ".jsonl":
        items: List[BenchmarkItem] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL manifest rows must be objects: {manifest}")
            items.append(_item_from_mapping(payload, base_dir=base_dir))
        return items

    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [_item_from_mapping(row, base_dir=base_dir) for row in reader]


def run_benchmark(
    items: Iterable[BenchmarkItem],
    *,
    config: RhythmEngineConfig,
    output_jsonl: str,
    limit: Optional[int] = None,
) -> str:
    path = Path(output_jsonl).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            if limit is not None and count >= int(limit):
                break
            result = analyze_rhythm(
                item.audio,
                config=config,
                reference_beats=_read_times(item.reference_beats),
                reference_downbeats=_read_times(item.reference_downbeats),
            ).to_dict()
            result["benchmark_item"] = item.to_dict()
            fh.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return str(path)


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


COUNT_METRICS = {
    "reference_count",
    "estimated_count",
    "matched_count",
    "false_positive_count_70ms",
    "missed_count_70ms",
}

FLOAT_METRICS = (
    "median_error_ms",
    "mean_error_ms",
    "median_abs_error_ms",
    "mean_abs_error_ms",
    "p90_abs_error_ms",
    "p95_abs_error_ms",
    "max_abs_error_ms",
    "hit_rate_5ms",
    "hit_rate_10ms",
    "hit_rate_20ms",
    "hit_rate_70ms",
    "precision_20ms",
    "recall_20ms",
    "f1_20ms",
    "precision_70ms",
    "recall_70ms",
    "f1_70ms",
    "continuity_10ms",
    "continuity_20ms",
    "continuity_70ms",
)


def _strict_ms_score(report: Mapping[str, Any]) -> float:
    median = _finite_float(report.get("median_abs_error_ms"))
    precision = 0.0 if median is None else 1.0 / (1.0 + max(0.0, median) / 12.0)
    hit10 = _finite_float(report.get("hit_rate_10ms")) or 0.0
    f1_20 = _finite_float(report.get("f1_20ms")) or _finite_float(report.get("hit_rate_20ms")) or 0.0
    f1_70 = _finite_float(report.get("f1_70ms")) or _finite_float(report.get("hit_rate_70ms")) or 0.0
    continuity = _finite_float(report.get("continuity_20ms")) or 0.0
    return float(max(0.0, min(1.0, (0.24 * precision) + (0.24 * hit10) + (0.30 * f1_20) + (0.12 * continuity) + (0.10 * f1_70))))


def _aggregate_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [report for report in reports if isinstance(report, Mapping)]
    if not rows:
        return {"track_count": 0}

    out: dict[str, Any] = {"track_count": int(len(rows))}
    for key in sorted(COUNT_METRICS):
        out[key] = int(sum(int(_finite_float(row.get(key)) or 0.0) for row in rows))

    weights = [max(1.0, _finite_float(row.get("reference_count")) or 1.0) for row in rows]
    for key in FLOAT_METRICS:
        values: list[tuple[float, float]] = []
        for row, weight in zip(rows, weights):
            value = _finite_float(row.get(key))
            if value is not None:
                values.append((value, weight))
        if values:
            total_weight = float(sum(weight for _value, weight in values)) or 1.0
            out[key] = float(sum(value * weight for value, weight in values) / total_weight)

    reference = max(1, int(out.get("reference_count", 0)))
    estimated = max(1, int(out.get("estimated_count", 0)))
    out["false_positive_rate_70ms"] = float(int(out.get("false_positive_count_70ms", 0)) / estimated)
    out["miss_rate_70ms"] = float(int(out.get("missed_count_70ms", 0)) / reference)
    out["strict_ms_score"] = _strict_ms_score(out)
    return out


def _collect_reports(payloads: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, dict[str, list[Mapping[str, Any]]]]]:
    scopes: dict[str, list[Mapping[str, Any]]] = {}
    by_genre: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for payload in payloads:
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        item = payload.get("benchmark_item")
        metadata = item.get("metadata") if isinstance(item, Mapping) and isinstance(item.get("metadata"), Mapping) else {}
        genre = str(metadata.get("genre") or metadata.get("style") or "").strip().lower()

        def add(scope: str, report: Any) -> None:
            if not isinstance(report, Mapping):
                return
            scopes.setdefault(scope, []).append(report)
            if genre:
                by_genre.setdefault(genre, {}).setdefault(scope, []).append(report)

        add("final:beats", evaluation.get("beats"))
        add("final:downbeats", evaluation.get("downbeats"))
        providers = evaluation.get("providers")
        if isinstance(providers, Mapping):
            for name, report in providers.items():
                add(f"provider:{name}:beats", report)
        provider_downbeats = evaluation.get("provider_downbeats")
        if isinstance(provider_downbeats, Mapping):
            for name, report in provider_downbeats.items():
                add(f"provider:{name}:downbeats", report)
    return scopes, by_genre


def summarize_benchmark_payloads(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [payload for payload in payloads if isinstance(payload, Mapping)]
    scopes, by_genre = _collect_reports(rows)
    reports = {scope: _aggregate_reports(scope_reports) for scope, scope_reports in sorted(scopes.items())}
    genre_reports = {
        genre: {scope: _aggregate_reports(scope_reports) for scope, scope_reports in sorted(scope_map.items())}
        for genre, scope_map in sorted(by_genre.items())
    }
    provider_beats = {
        scope: report
        for scope, report in reports.items()
        if scope.startswith("provider:") and scope.endswith(":beats") and int(report.get("track_count", 0) or 0) > 0
    }
    ranked_providers = sorted(
        (
            {
                "scope": scope,
                "track_count": int(report.get("track_count", 0) or 0),
                "strict_ms_score": float(report.get("strict_ms_score", 0.0) or 0.0),
                "median_abs_error_ms": float(report.get("median_abs_error_ms", float("inf"))),
                "f1_20ms": float(report.get("f1_20ms", 0.0) or 0.0),
                "f1_70ms": float(report.get("f1_70ms", 0.0) or 0.0),
            }
            for scope, report in provider_beats.items()
        ),
        key=lambda row: (-row["strict_ms_score"], row["median_abs_error_ms"], row["scope"]),
    )
    final = reports.get("final:beats", {})
    recommendations: list[str] = []
    if final:
        if float(final.get("median_abs_error_ms", 999.0) or 999.0) > 12.0 or float(final.get("hit_rate_10ms", 0.0) or 0.0) < 0.65:
            recommendations.append("prioritize_micro_refine_calibration")
        if float(final.get("f1_70ms", 0.0) or 0.0) < 0.95 or float(final.get("false_positive_rate_70ms", 0.0) or 0.0) > 0.04:
            recommendations.append("sweep_fusion_dedupe_and_min_gap")
        if float(final.get("miss_rate_70ms", 0.0) or 0.0) > 0.04:
            recommendations.append("increase_candidate_recall_or_provider_coverage")
    return {
        "track_count": int(len(rows)),
        "reports": reports,
        "genre_reports": genre_reports,
        "provider_weights": learn_provider_weights_from_payloads(rows),
        "provider_ranking": ranked_providers,
        "recommendations": recommendations,
    }


def summarize_benchmark_jsonl(path: str) -> dict[str, Any]:
    jsonl_path = Path(path).expanduser()
    payloads: list[Mapping[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payloads.append(payload)
    summary = summarize_benchmark_payloads(payloads)
    summary["source"] = str(jsonl_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rhythm_engine over a reference manifest and write JSONL results.")
    parser.add_argument("manifest", help="CSV or JSONL manifest with audio and reference beat/downbeat paths")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", action="append", help="Provider to run. Repeat to set order.")
    parser.add_argument("--provider-weights-json", help="Provider weights JSON")
    parser.add_argument("--no-micro-refine", action="store_true")
    parser.add_argument("--no-grid-repair", action="store_true")
    parser.add_argument("--no-repair-lattice-snap", action="store_true")
    parser.add_argument("--repair-lattice-snap-ratio", type=float, default=0.28)
    parser.add_argument("--fusion-radius-ms", type=float, default=45.0)
    parser.add_argument("--downbeat-fusion-radius-ms", type=float, default=90.0)
    parser.add_argument("--no-fusion-dedupe", action="store_true")
    parser.add_argument("--fusion-min-beat-gap-ratio", type=float, default=0.45)
    parser.add_argument("--fusion-min-downbeat-gap-ratio", type=float, default=0.55)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-output", help="Optional aggregate benchmark summary JSON path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RhythmEngineConfig(
        providers=tuple(args.provider) if args.provider else RhythmEngineConfig().providers,
        provider_weights_json=args.provider_weights_json,
        micro_refine=not bool(args.no_micro_refine),
        repair_steady_grid=not bool(args.no_grid_repair),
        repair_lattice_snap=not bool(args.no_repair_lattice_snap),
        repair_lattice_snap_ratio=float(args.repair_lattice_snap_ratio),
        fusion_radius_ms=float(args.fusion_radius_ms),
        downbeat_fusion_radius_ms=float(args.downbeat_fusion_radius_ms),
        fusion_dedupe_events=not bool(args.no_fusion_dedupe),
        fusion_min_beat_gap_ratio=float(args.fusion_min_beat_gap_ratio),
        fusion_min_downbeat_gap_ratio=float(args.fusion_min_downbeat_gap_ratio),
    )
    output = run_benchmark(read_manifest(args.manifest), config=cfg, output_jsonl=args.output, limit=args.limit)
    payload: dict[str, Any] = {"output": output}
    if args.summary_output:
        summary = summarize_benchmark_jsonl(output)
        summary_path = Path(args.summary_output).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        payload["summary_output"] = str(summary_path)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
