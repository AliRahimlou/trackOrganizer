from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import apply_folder_drop_candidates_to_set
import batch
import buildSetAndGenerateAls
from drop_aligner.legacy_write_guard import (
    ALLOW_ENV_VAR,
    LegacyDetectorWriteBlocked,
    add_legacy_detector_write_arg,
    legacy_detector_writes_allowed,
    require_legacy_detector_write_opt_in,
)


def test_legacy_write_guard_blocks_by_default(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)

    with pytest.raises(LegacyDetectorWriteBlocked) as exc:
        require_legacy_detector_write_opt_in("old.py", action="writing stale markers")

    assert "legacy non-visual detector write path" in str(exc.value)


def test_legacy_write_guard_allows_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(ALLOW_ENV_VAR, "1")

    assert legacy_detector_writes_allowed() is True
    require_legacy_detector_write_opt_in("old.py", action="writing experimental output")


def test_legacy_write_guard_allows_explicit_parser_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    parser = argparse.ArgumentParser()
    add_legacy_detector_write_arg(parser)
    args = parser.parse_args(["--allow-legacy-detector-write"])

    assert legacy_detector_writes_allowed(explicit=bool(args.allow_legacy_detector_write)) is True
    require_legacy_detector_write_opt_in(
        "old.py",
        action="writing experimental output",
        explicit=bool(args.allow_legacy_detector_write),
    )


def test_batch_legacy_writer_blocks_without_opt_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    template = tmp_path / "template.als"
    summary = tmp_path / "summary.csv"
    template.write_text("", encoding="utf-8")

    with pytest.raises(LegacyDetectorWriteBlocked):
        batch.main([str(tmp_path), "--template", str(template), "--dry-run", "--summary", str(summary)])

    assert not summary.exists()


def test_batch_legacy_writer_allows_explicit_temp_experiment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    template = tmp_path / "template.als"
    summary = tmp_path / "summary.csv"
    template.write_text("", encoding="utf-8")

    result = batch.main(
        [
            str(tmp_path),
            "--template",
            str(template),
            "--dry-run",
            "--summary",
            str(summary),
            "--allow-legacy-detector-write",
        ]
    )

    assert result == 0
    assert summary.exists()


def test_build_set_blocks_local_drop_aligned_source_without_opt_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    monkeypatch.setattr(buildSetAndGenerateAls, "PREFER_DROP_ALIGNED_SOURCES", True)
    source = tmp_path / "CH1.als"
    stale = tmp_path / "track_DROP_ALIGNED.als"
    source.write_text("", encoding="utf-8")
    stale.write_text("", encoding="utf-8")

    with pytest.raises(LegacyDetectorWriteBlocked):
        buildSetAndGenerateAls.preferred_source_als(source)


def test_build_set_allows_local_drop_aligned_source_with_env_opt_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(ALLOW_ENV_VAR, "1")
    monkeypatch.setattr(buildSetAndGenerateAls, "PREFER_DROP_ALIGNED_SOURCES", True)
    source = tmp_path / "CH1.als"
    stale = tmp_path / "track_DROP_ALIGNED.als"
    source.write_text("", encoding="utf-8")
    stale.write_text("", encoding="utf-8")

    assert buildSetAndGenerateAls.preferred_source_als(source) == stale


def test_apply_folder_drop_candidates_blocks_write_mode_without_opt_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)

    with pytest.raises(LegacyDetectorWriteBlocked):
        apply_folder_drop_candidates_to_set.apply_local_candidates(
            str(tmp_path / "in.als"),
            str(tmp_path / "out.als"),
            min_delta_sec=0.04,
            only_missing=False,
            dry_run=False,
        )
