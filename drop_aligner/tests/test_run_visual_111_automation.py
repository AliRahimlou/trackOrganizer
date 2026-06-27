from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_visual_111_automation as visual_auto


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "stems": str(tmp_path / "STEMS"),
        "template": str(tmp_path / "128.als"),
        "base_als": str(tmp_path / "CH1.als"),
        "out_dir": str(tmp_path / "VisualFirstFresh"),
        "out": "",
        "report": "",
        "validation_out_dir": "",
        "run_stamp": "teststamp",
        "workers": 2,
        "sample_rate": 44100,
        "waveform_cache_dir": "",
        "human_correction_log": [],
        "limit": 0,
        "per_track_timeout_sec": 600.0,
        "force": False,
        "dry_run": False,
        "no_human_review_overrides": False,
        "use_cache": True,
        "strict_stems": True,
        "suspicious_audit_width": 8192,
        "suspicious_audit_workers": 0,
        "suspicious_audit_timeout_sec": 45.0,
        "validate_even_on_build_hold": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_visual_111_automation_build_command_is_visual_first_and_fail_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = visual_auto._resolve_run_paths(args)

    cmd = visual_auto._build_fresh_command(args, paths)

    assert cmd[0] == sys.executable
    assert str(visual_auto.REPO_ROOT / "build_fresh_visual_first_library_set.py") in cmd
    assert "--sample-rate" in cmd
    assert cmd[cmd.index("--sample-rate") + 1] == "44100"
    assert cmd[cmd.index("--per-track-timeout-sec") + 1] == "600.0"
    assert "--allow-partial" not in cmd
    assert "--allow-unsafe-audit" not in cmd
    assert "--no-use-cache" not in cmd


def test_visual_111_automation_validation_requires_all_pass(tmp_path: Path) -> None:
    args = _args(tmp_path, waveform_cache_dir=str(tmp_path / "waveform_cache"))
    paths = visual_auto._resolve_run_paths(args)

    cmd = visual_auto._build_validation_command(args, paths)

    assert str(visual_auto.REPO_ROOT / "validate_visual_first_production.py") in cmd
    assert str(paths.report_json) in cmd
    assert "--require-all-pass" in cmd
    assert "--waveform-cache-dir" in cmd


def test_visual_111_automation_suspicious_audit_requires_no_failures(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        waveform_cache_dir=str(tmp_path / "waveform_cache"),
        suspicious_audit_workers=1,
        suspicious_audit_width=4096,
        suspicious_audit_timeout_sec=12.5,
    )
    paths = visual_auto._resolve_run_paths(args)

    cmd = visual_auto._build_suspicious_audit_command(args, paths)

    assert str(visual_auto.REPO_ROOT / "audit_visual_first_suspicious_markers.py") in cmd
    assert str(paths.report_json) in cmd
    assert "--require-no-failures" in cmd
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path / "waveform_cache")
    assert cmd[cmd.index("--workers") + 1] == "1"
    assert cmd[cmd.index("--width") + 1] == "4096"
    assert cmd[cmd.index("--per-row-timeout-sec") + 1] == "12.5"


def test_visual_111_automation_respects_explicit_report_and_safe_debug_flags(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    args = _args(
        tmp_path,
        report=str(report),
        limit=3,
        no_human_review_overrides=True,
        use_cache=False,
        strict_stems=False,
    )
    paths = visual_auto._resolve_run_paths(args)

    cmd = visual_auto._build_fresh_command(args, paths)

    assert paths.report_json == report.resolve()
    assert cmd[cmd.index("--limit") + 1] == "3"
    assert "--allow-partial" not in cmd
    assert "--allow-unsafe-audit" not in cmd
    assert "--no-human-review-overrides" in cmd
    assert "--no-use-cache" in cmd
    assert "--no-strict-stems" in cmd
