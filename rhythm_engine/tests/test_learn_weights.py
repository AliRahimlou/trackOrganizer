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
