from __future__ import annotations

from rhythm_engine.learn_weights import learn_provider_weights_from_payloads


def _report(median: float, p95: float, hit20: float, hit70: float, continuity: float) -> dict:
    return {
        "median_abs_error_ms": median,
        "p95_abs_error_ms": p95,
        "hit_rate_20ms": hit20,
        "hit_rate_70ms": hit70,
        "continuity_20ms": continuity,
    }


def test_learn_provider_weights_from_payloads_prefers_more_accurate_provider() -> None:
    payloads = [
        {
            "evaluation": {
                "providers": {
                    "good": _report(3.0, 12.0, 0.98, 1.0, 0.95),
                    "bad": _report(55.0, 160.0, 0.10, 0.35, 0.05),
                }
            }
        },
        {
            "evaluation": {
                "providers": {
                    "good": _report(5.0, 15.0, 0.90, 1.0, 0.85),
                    "bad": _report(42.0, 120.0, 0.15, 0.40, 0.10),
                }
            }
        },
    ]

    weights = learn_provider_weights_from_payloads(payloads)

    assert weights["good"] > 1.0
    assert weights["bad"] < 1.0
    assert weights["good"] > weights["bad"]


def test_learn_provider_weights_penalizes_duplicate_and_missed_beats() -> None:
    payloads = [
        {
            "evaluation": {
                "providers": {
                    "clean": {
                        **_report(4.0, 14.0, 0.96, 1.0, 0.92),
                        "f1_20ms": 0.96,
                        "f1_70ms": 1.0,
                        "precision_70ms": 1.0,
                        "recall_70ms": 1.0,
                        "reference_count": 100,
                        "estimated_count": 100,
                        "false_positive_count_70ms": 0,
                        "missed_count_70ms": 0,
                    },
                    "duplicate": {
                        **_report(5.0, 16.0, 0.94, 0.99, 0.88),
                        "f1_20ms": 0.55,
                        "f1_70ms": 0.62,
                        "precision_70ms": 0.58,
                        "recall_70ms": 0.69,
                        "reference_count": 100,
                        "estimated_count": 130,
                        "false_positive_count_70ms": 49,
                        "missed_count_70ms": 31,
                    },
                }
            }
        }
    ]

    weights = learn_provider_weights_from_payloads(payloads)

    assert weights["clean"] > weights["duplicate"]
    assert weights["duplicate"] < 1.0
