from __future__ import annotations

import build_visual_first_production_queue as queue


def _row_with_boom_proof(proof: dict) -> dict:
    return {
        "audit_status": "pass",
        "audit_flags": [],
        "selected_by": "visual_boom_grid_one_snap",
        "boom_proof": proof,
    }


def test_production_queue_accepts_fresh_front_edge_proof() -> None:
    row = _row_with_boom_proof(
        {
            "passes": True,
            "marker_sec": 32.0,
            "nearest": {
                "edge_time": 32.0,
                "offset_sec": 0.0,
                "abs_offset_sec": 0.0,
            },
            "reasons": [],
        }
    )

    assert queue.row_passes_production_gate(row) is True
    assert queue._reason(row) == "held by production gate"


def test_production_queue_rejects_stale_front_edge_proof() -> None:
    row = _row_with_boom_proof(
        {
            "passes": True,
            "marker_sec": 32.421,
            "nearest": {
                "edge_time": 32.0,
                "offset_sec": 0.421,
                "abs_offset_sec": 0.421,
            },
            "reasons": [],
        }
    )

    assert queue.row_passes_production_gate(row) is False
    assert "boom_proof=stale_front_edge" in queue._reason(row)
