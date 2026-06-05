from __future__ import annotations

from pathlib import Path

import rhythm_engine.benchmark as benchmark_module
from rhythm_engine.benchmark import read_manifest, run_benchmark, summarize_benchmark_jsonl, summarize_benchmark_payloads
from rhythm_engine.types import RhythmAnalysisResult, RhythmEngineConfig, RhythmEstimate


def test_read_manifest_resolves_relative_csv_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "audio,reference_beats,reference_downbeats,genre\n"
        "audio.wav,beats.txt,downbeats.txt,house\n",
        encoding="utf-8",
    )

    items = read_manifest(str(manifest))

    assert len(items) == 1
    assert items[0].audio == str(tmp_path / "audio.wav")
    assert items[0].reference_beats == str(tmp_path / "beats.txt")
    assert items[0].reference_downbeats == str(tmp_path / "downbeats.txt")
    assert items[0].metadata == {"genre": "house"}


def test_run_benchmark_writes_jsonl_results(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    beats = tmp_path / "beats.txt"
    downbeats = tmp_path / "downbeats.txt"
    output = tmp_path / "out.jsonl"
    audio.write_bytes(b"stub")
    beats.write_text("1.0\n1.5\n", encoding="utf-8")
    downbeats.write_text("1.0\n", encoding="utf-8")
    items = [benchmark_module.BenchmarkItem(str(audio), str(beats), str(downbeats), {"id": "x"})]

    def fake_analyze(audio_path: str, *, config: RhythmEngineConfig, reference_beats, reference_downbeats):
        return RhythmAnalysisResult(
            audio_path=audio_path,
            final=RhythmEstimate(provider="fake", beats=(1.0, 1.5), downbeats=(1.0,), confidence=1.0),
            fused=RhythmEstimate(provider="fake", beats=(1.0, 1.5), downbeats=(1.0,), confidence=1.0),
            selected=RhythmEstimate(provider="fake", beats=(1.0, 1.5), downbeats=(1.0,), confidence=1.0),
            providers=tuple(),
            evaluation={"beats": {"median_abs_error_ms": 0.0}},
        )

    monkeypatch.setattr(benchmark_module, "analyze_rhythm", fake_analyze)

    written = run_benchmark(items, config=RhythmEngineConfig(providers=("fake",)), output_jsonl=str(output))

    assert written == str(output)
    text = output.read_text(encoding="utf-8").strip()
    assert '"audio_path":"' in text
    assert '"benchmark_item":' in text


def _eval_report(
    *,
    median: float,
    hit10: float,
    f1_20: float,
    f1_70: float = 1.0,
    reference_count: int = 100,
    estimated_count: int = 100,
) -> dict:
    return {
        "reference_count": reference_count,
        "estimated_count": estimated_count,
        "matched_count": min(reference_count, estimated_count),
        "median_error_ms": median,
        "mean_error_ms": median,
        "median_abs_error_ms": abs(median),
        "mean_abs_error_ms": abs(median),
        "p90_abs_error_ms": abs(median) * 1.5,
        "p95_abs_error_ms": abs(median) * 2.0,
        "hit_rate_5ms": 1.0 if abs(median) <= 5.0 else 0.0,
        "hit_rate_10ms": hit10,
        "hit_rate_20ms": max(hit10, f1_20),
        "hit_rate_70ms": f1_70,
        "precision_20ms": f1_20,
        "recall_20ms": f1_20,
        "f1_20ms": f1_20,
        "precision_70ms": f1_70,
        "recall_70ms": f1_70,
        "f1_70ms": f1_70,
        "false_positive_count_70ms": max(0, estimated_count - min(reference_count, estimated_count)),
        "missed_count_70ms": max(0, reference_count - min(reference_count, estimated_count)),
        "continuity_10ms": hit10,
        "continuity_20ms": f1_20,
        "continuity_70ms": f1_70,
    }


def test_summarize_benchmark_payloads_aggregates_final_provider_and_genre_reports() -> None:
    payloads = [
        {
            "benchmark_item": {"metadata": {"genre": "house"}},
            "evaluation": {
                "beats": _eval_report(median=6.0, hit10=0.90, f1_20=0.96),
                "providers": {
                    "good": _eval_report(median=4.0, hit10=0.96, f1_20=0.98),
                    "bad": _eval_report(median=35.0, hit10=0.10, f1_20=0.20, f1_70=0.60),
                },
            },
        },
        {
            "benchmark_item": {"metadata": {"genre": "house"}},
            "evaluation": {
                "beats": _eval_report(median=8.0, hit10=0.80, f1_20=0.94),
                "providers": {
                    "good": _eval_report(median=5.0, hit10=0.94, f1_20=0.97),
                    "bad": _eval_report(median=40.0, hit10=0.08, f1_20=0.18, f1_70=0.55),
                },
            },
        },
    ]

    summary = summarize_benchmark_payloads(payloads)

    assert summary["track_count"] == 2
    assert summary["reports"]["final:beats"]["track_count"] == 2
    assert summary["reports"]["final:beats"]["median_abs_error_ms"] == 7.0
    assert summary["genre_reports"]["house"]["final:beats"]["track_count"] == 2
    assert summary["provider_weights"]["good"] > summary["provider_weights"]["bad"]
    assert summary["provider_ranking"][0]["scope"] == "provider:good:beats"


def test_summarize_benchmark_jsonl_reads_written_payloads(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        '{"evaluation":{"beats":{"reference_count":1,"estimated_count":1,"matched_count":1,"median_abs_error_ms":3.0,"hit_rate_10ms":1.0,"f1_20ms":1.0,"f1_70ms":1.0,"continuity_20ms":1.0}}}\n',
        encoding="utf-8",
    )

    summary = summarize_benchmark_jsonl(str(path))

    assert summary["source"] == str(path)
    assert summary["reports"]["final:beats"]["strict_ms_score"] > 0.9
