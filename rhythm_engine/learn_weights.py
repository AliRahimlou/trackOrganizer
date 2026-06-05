from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


def _iter_payloads(paths: Iterable[str]) -> Iterable[Mapping[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if text:
                    payload = json.loads(text)
                    if isinstance(payload, Mapping):
                        yield payload
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                yield payload


def _score_report(report: Mapping[str, Any]) -> Optional[float]:
    try:
        median = float(report.get("median_abs_error_ms"))
        p95 = float(report.get("p95_abs_error_ms"))
        hit20 = float(report.get("hit_rate_20ms"))
        hit70 = float(report.get("hit_rate_70ms"))
        continuity = float(report.get("continuity_20ms"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite([median, p95, hit20, hit70, continuity]).all():
        return None
    precision = 1.0 / (1.0 + max(0.0, median) / 12.0)
    tail = 1.0 / (1.0 + max(0.0, p95) / 45.0)
    return float((0.34 * precision) + (0.25 * hit20) + (0.18 * hit70) + (0.13 * continuity) + (0.10 * tail))


def learn_provider_weights_from_payloads(payloads: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {}
    for payload in payloads:
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        providers = evaluation.get("providers")
        if not isinstance(providers, Mapping):
            continue
        for provider, report in providers.items():
            if not isinstance(report, Mapping):
                continue
            score = _score_report(report)
            if score is None:
                continue
            scores.setdefault(str(provider), []).append(float(score))
    if not scores:
        return {}

    raw = {provider: float(np.mean(values)) for provider, values in scores.items() if values}
    mean_score = float(np.mean(list(raw.values()))) if raw else 1.0
    if mean_score <= 1e-9:
        mean_score = 1.0
    return {
        provider: float(np.clip(score / mean_score, 0.20, 3.00))
        for provider, score in sorted(raw.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn rhythm provider fusion weights from reference-evaluated analysis JSON.")
    parser.add_argument("analysis_json", nargs="+", help="Analysis JSON or JSONL files produced by rhythm_engine with references")
    parser.add_argument("--output", required=True, help="Output provider weights JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    weights = learn_provider_weights_from_payloads(_iter_payloads(args.analysis_json))
    payload = {"provider_weights": weights, "training_files": [str(path) for path in args.analysis_json]}
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
