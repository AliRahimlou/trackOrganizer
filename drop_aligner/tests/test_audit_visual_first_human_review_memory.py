from __future__ import annotations

import json
from pathlib import Path

import pytest

import audit_visual_first_human_review_memory as audit


def _report(path: Path, marker: float = 32.0) -> None:
    drums = "/tmp/drums_128_1A_7-Artist - Track.flac"
    path.write_text(
        json.dumps(
            {
                "processed_rows": [
                    {
                        "track": {"folder": "Artist - Track", "src": "/tmp/CH1.als"},
                        "marker": marker,
                        "drums_path": drums,
                        "selected_by": "visual_gui_first_fat_block",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_log(path: Path, *, reviewed_from: str, user_pick: float) -> None:
    path.write_text(
        json.dumps(
            {
                "track": "/tmp/drums_128_1A_7-Artist - Track.flac",
                "user_pick": user_pick,
                "reviewed_from": reviewed_from,
                "timestamp": "2026-06-20T00:00:00+00:00",
                "selected_by": "user_candidate_pick",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_human_review_audit_passes_matching_manual_marker(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    log = tmp_path / "corrections.jsonl"
    _report(report, marker=32.02)
    _write_log(log, reviewed_from="web_manual_marker", user_pick=32.0)

    result = audit.audit_human_review_memory(report, correction_logs=[log], out_dir=tmp_path / "out")

    assert result["passed"] is True
    assert result["matched_review_rows"] == 1
    assert result["hard_mismatch_count"] == 0
    assert result["manual_review_rows"] == 1


def test_human_review_audit_treats_unproven_manual_mismatch_as_stale(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    log = tmp_path / "corrections.jsonl"
    _report(report, marker=33.0)
    _write_log(log, reviewed_from="web_candidate_pick", user_pick=32.0)

    result = audit.audit_human_review_memory(report, correction_logs=[log], out_dir=tmp_path / "out")

    assert result["passed"] is True
    assert result["hard_mismatch_count"] == 0
    assert result["stale_manual_mismatch_count"] == 1
    assert Path(result["stale_manual_mismatch_csv"]).read_text(encoding="utf-8").count("stale_manual_mismatch") == 1


def test_human_review_audit_rejects_validated_manual_marker_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.json"
    log = tmp_path / "corrections.jsonl"
    _report(report, marker=33.0)
    _write_log(log, reviewed_from="web_manual_marker", user_pick=32.0)

    def passing_gate(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"passes": True, "reasons": [], "boom_passes": True, "gui_passes": True}

    monkeypatch.setattr(audit, "_review_marker_gate", passing_gate)

    result = audit.audit_human_review_memory(report, correction_logs=[log], out_dir=tmp_path / "out")

    assert result["passed"] is False
    assert result["hard_mismatch_count"] == 1
    assert result["validated_hard_mismatch_count"] == 1
    assert Path(result["hard_mismatch_csv"]).read_text(encoding="utf-8").count("validated_hard_mismatch") == 1


def test_human_review_audit_treats_blue_mismatch_as_advisory(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    log = tmp_path / "corrections.jsonl"
    _report(report, marker=33.0)
    _write_log(log, reviewed_from="web_accept_blue_marker", user_pick=32.0)

    result = audit.audit_human_review_memory(report, correction_logs=[log], out_dir=tmp_path / "out")

    assert result["passed"] is True
    assert result["hard_mismatch_count"] == 0
    assert result["advisory_mismatch_count"] == 1
