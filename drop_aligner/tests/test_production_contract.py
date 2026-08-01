from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import build_fresh_visual_first_library_set as builder
import build_visual_first_production_queue as queue
import validate_visual_first_production as validator
from drop_aligner import production_contract


def test_builder_queue_and_validator_share_the_contract_constants() -> None:
    assert builder.DISALLOWED_PASS_SOURCES is production_contract.DISALLOWED_PASS_SOURCES
    assert queue.DISALLOWED_PASS_SOURCES is production_contract.DISALLOWED_PASS_SOURCES
    assert validator.DISALLOWED_PASS_SOURCES is production_contract.DISALLOWED_PASS_SOURCES
    assert builder.DISALLOWED_PASS_SOURCE_PREFIXES == production_contract.DISALLOWED_PASS_SOURCE_PREFIXES
    assert queue.DISALLOWED_PASS_SOURCE_PREFIXES == production_contract.DISALLOWED_PASS_SOURCE_PREFIXES
    assert validator.DISALLOWED_PASS_SOURCE_PREFIXES == production_contract.DISALLOWED_PASS_SOURCE_PREFIXES
    assert (
        builder.VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
        == validator.VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
        == production_contract.VALIDATED_HUMAN_OVERRIDE_SELECTED_BY
    )
    assert validator.MAX_GRID_ONE_DISTANCE_MS == production_contract.MAX_GRID_ONE_DISTANCE_MS


@pytest.mark.parametrize(
    ("source", "unsafe"),
    [
        ("web_save_placed_marker", True),
        ("historical_anything_at_all", True),
        ("saved_closest_to_review_pick", True),
        ("visual_gui_first_fat_block", False),
        ("", False),
    ],
)
def test_unsafe_selected_source(source: str, unsafe: bool) -> None:
    assert production_contract.unsafe_selected_source(source) is unsafe


def test_human_override_tolerance_is_tighter_than_the_old_dead_zone() -> None:
    signature = inspect.signature(builder._apply_validated_human_overrides)
    default = signature.parameters["tolerance_sec"].default
    assert default == production_contract.HUMAN_OVERRIDE_MATCH_TOLERANCE_SEC
    assert default <= 0.005


def _passing_queue_row() -> dict:
    return {
        "audit_status": "pass",
        "audit_flags": [],
        "selected_by": "visual_gui_first_fat_block",
        "boom_proof": {
            "passes": True,
            "nearest": {"offset_sec": 0.0},
        },
        "gui_mask_proof": {
            "passes": True,
            "marker_signal_present": True,
            "marker_relevant_mask": True,
        },
    }


def test_queue_gate_requires_a_clean_gui_mask_proof() -> None:
    row = _passing_queue_row()
    assert queue.row_passes_production_gate(row) is True

    for broken in (
        {},
        {"passes": False, "reasons": ["front_edge_mismatch"]},
        {"passes": True, "marker_signal_present": False, "marker_relevant_mask": True},
        {"passes": True, "marker_signal_present": True},
    ):
        row = _passing_queue_row()
        row["gui_mask_proof"] = broken
        assert queue.row_passes_production_gate(row) is False, broken
        assert "gui_mask" in queue._reason(row)


def test_recovery_pullback_no_longer_reads_locals_assigned_after_it() -> None:
    """Regression pin for the UnboundLocalError on the analysis-rate recovery
    path: the pullback call inside the recovery block must not pass the outer
    ``bpm``/``clock_zero`` names, which are only assigned after a candidate is
    selected."""
    source = Path(inspect.getfile(production_contract)).parent / "visual_first.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    marker_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "visual_first_marker"
    )
    first_bpm_assignment = min(
        node.lineno
        for node in ast.walk(marker_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "bpm" for target in node.targets)
    )
    recovery_calls = [
        node
        for node in ast.walk(marker_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_same_cluster_stronger_front_edge_pullback_result"
        and node.lineno < first_bpm_assignment
    ]
    assert recovery_calls, "recovery pullback call before bpm assignment not found"
    for call in recovery_calls:
        for keyword in call.keywords:
            if keyword.arg in {"bpm", "clock_zero_sec"} and isinstance(keyword.value, ast.Name):
                assert keyword.value.id not in {"bpm", "clock_zero"}, (
                    f"{keyword.arg} is passed the outer '{keyword.value.id}' local, "
                    "which is unbound when no candidate was selected"
                )
