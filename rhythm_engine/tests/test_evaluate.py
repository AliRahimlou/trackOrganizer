from __future__ import annotations

from rhythm_engine.evaluate import evaluate_beat_grid


def test_evaluate_beat_grid_reports_ms_precision_metrics() -> None:
    reference = [1.0, 1.5, 2.0, 2.5]
    estimated = [1.003, 1.492, 2.018, 2.560]

    report = evaluate_beat_grid(reference, estimated)

    assert report.reference_count == 4
    assert report.estimated_count == 4
    assert round(report.median_abs_error_ms, 3) == 13.0
    assert report.hit_rate_5ms == 0.25
    assert report.hit_rate_10ms == 0.50
    assert report.hit_rate_20ms == 0.75
    assert report.hit_rate_70ms == 1.0
    assert report.precision_70ms == 1.0
    assert report.recall_70ms == 1.0
    assert report.f1_70ms == 1.0
    assert report.continuity_20ms == 0.75


def test_evaluate_beat_grid_reports_duplicate_estimate_penalty() -> None:
    reference = [1.0, 2.0]
    estimated = [1.003, 1.004]

    report = evaluate_beat_grid(reference, estimated)

    assert report.matched_count == 1
    assert report.precision_20ms == 0.5
    assert report.recall_20ms == 0.5
    assert report.false_positive_count_70ms == 1
    assert report.missed_count_70ms == 1
