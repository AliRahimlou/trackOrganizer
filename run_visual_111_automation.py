#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _reexec_with_local_venv() -> None:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except OSError:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_with_local_venv()

from drop_aligner.visual_first import VISUAL_FIRST_PRODUCTION_SAMPLE_RATE  # noqa: E402
from project_config import BASE_ALS_TEMPLATE, DEFAULT_ALS_TEMPLATE, GENERATED_SET_DIR, STEMS_ROOT_DIR  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPaths:
    run_stamp: str
    out_dir: Path
    output_als: Path
    report_json: Path
    validation_out_dir: Path


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 4) // 2))


def _new_run_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_visual_111")


def _resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    run_stamp = str(args.run_stamp or _new_run_stamp()).strip()
    out_dir = Path(args.out_dir).expanduser().resolve()
    output_als = (
        Path(args.out).expanduser().resolve()
        if str(args.out or "").strip()
        else out_dir / f"VISUAL_FIRST_FRESH_ALL_TRACKS_{run_stamp}.als"
    )
    report_json = (
        Path(args.report).expanduser().resolve()
        if str(args.report or "").strip()
        else output_als.with_name(f"{output_als.stem}_report.json")
    )
    validation_out_dir = (
        Path(args.validation_out_dir).expanduser().resolve()
        if str(args.validation_out_dir or "").strip()
        else out_dir / "validation" / run_stamp
    )
    return RunPaths(
        run_stamp=run_stamp,
        out_dir=out_dir,
        output_als=output_als,
        report_json=report_json,
        validation_out_dir=validation_out_dir,
    )


def _optional_path_arg(cmd: list[str], flag: str, value: str | Path | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([flag, text])


def _build_fresh_command(args: argparse.Namespace, paths: RunPaths) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "build_fresh_visual_first_library_set.py"),
        "--stems",
        str(Path(args.stems).expanduser()),
        "--template",
        str(Path(args.template).expanduser()),
        "--base-als",
        str(Path(args.base_als).expanduser()),
        "--out-dir",
        str(paths.out_dir),
        "--out",
        str(paths.output_als),
        "--report",
        str(paths.report_json),
        "--run-stamp",
        paths.run_stamp,
        "--workers",
        str(int(args.workers)),
        "--sample-rate",
        str(int(args.sample_rate)),
        "--per-track-timeout-sec",
        str(float(args.per_track_timeout_sec)),
    ]
    _optional_path_arg(cmd, "--waveform-cache-dir", args.waveform_cache_dir)
    for log_path in args.human_correction_log or []:
        _optional_path_arg(cmd, "--human-correction-log", log_path)
    if args.limit:
        cmd.extend(["--limit", str(int(args.limit))])
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.no_human_review_overrides:
        cmd.append("--no-human-review-overrides")
    if not args.use_cache:
        cmd.append("--no-use-cache")
    if not args.strict_stems:
        cmd.append("--no-strict-stems")
    return cmd


def _build_validation_command(args: argparse.Namespace, paths: RunPaths) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "validate_visual_first_production.py"),
        str(paths.report_json),
        "--out-dir",
        str(paths.validation_out_dir),
        "--workers",
        str(int(args.workers)),
        "--sample-rate",
        str(int(args.sample_rate)),
        "--stems",
        str(Path(args.stems).expanduser()),
        "--require-all-pass",
    ]
    _optional_path_arg(cmd, "--waveform-cache-dir", args.waveform_cache_dir)
    for log_path in args.human_correction_log or []:
        _optional_path_arg(cmd, "--human-correction-log", log_path)
    return cmd


def _build_suspicious_audit_command(args: argparse.Namespace, paths: RunPaths) -> list[str]:
    cache_dir = (
        Path(args.waveform_cache_dir).expanduser()
        if str(args.waveform_cache_dir or "").strip()
        else paths.report_json.parent / ".waveform_cache"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "audit_visual_first_suspicious_markers.py"),
        str(paths.report_json),
        "--cache-dir",
        str(cache_dir),
        "--out-dir",
        str(paths.validation_out_dir / "suspicious_marker_audit"),
        "--workers",
        str(int(args.suspicious_audit_workers or args.workers)),
        "--width",
        str(int(args.suspicious_audit_width)),
        "--per-row-timeout-sec",
        str(float(args.suspicious_audit_timeout_sec)),
        "--require-no-failures",
    ]
    return cmd


def _run_step(label: str, cmd: Sequence[str]) -> int:
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    return subprocess.run(list(cmd), cwd=str(REPO_ROOT), check=False).returncode


def run(args: argparse.Namespace) -> int:
    paths = _resolve_run_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.validation_out_dir.mkdir(parents=True, exist_ok=True)

    print("[VISUAL 1.1.1] full-drums-waveform automation", flush=True)
    print(f"[RUN] stamp={paths.run_stamp}", flush=True)
    print(f"[RUN] report={paths.report_json}", flush=True)
    print(f"[RUN] output={paths.output_als}", flush=True)

    build_code = _run_step("BUILD", _build_fresh_command(args, paths))
    if build_code != 0 and not args.validate_even_on_build_hold:
        print(
            "[HOLD] visual-first build stopped before validation. "
            "Review the generated failure/audit-hold CSVs from the report directory.",
            flush=True,
        )
        return build_code

    if args.dry_run:
        return build_code

    if not paths.report_json.exists():
        print(f"[HOLD] expected report was not written: {paths.report_json}", flush=True)
        return build_code or 1

    validation_code = _run_step("VALIDATE", _build_validation_command(args, paths))
    if validation_code != 0:
        return validation_code
    suspicious_code = _run_step("SUSPICIOUS_AUDIT", _build_suspicious_audit_command(args, paths))
    if suspicious_code != 0:
        return suspicious_code
    return build_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the visual-first 1.1.1 automation: build fresh waveform-proven "
            "Ableton drop markers, then fail-closed validate the generated set."
        )
    )
    parser.add_argument("--stems", default=str(STEMS_ROOT_DIR), help="STEMS root with /BPM/Key/Track folders.")
    parser.add_argument("--template", default=str(DEFAULT_ALS_TEMPLATE), help="Per-track three-stem ALS template.")
    parser.add_argument("--base-als", default=str(BASE_ALS_TEMPLATE), help="Blank combined-set base ALS.")
    parser.add_argument("--out-dir", default=str(GENERATED_SET_DIR / "VisualFirstFresh"), help="Visual-first output directory.")
    parser.add_argument("--out", default="", help="Explicit combined ALS output path.")
    parser.add_argument("--report", default="", help="Explicit report JSON path.")
    parser.add_argument("--validation-out-dir", default="", help="Validation output directory.")
    parser.add_argument("--run-stamp", default="", help="Stable stamp for generated files.")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Parallel visual-first workers.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=VISUAL_FIRST_PRODUCTION_SAMPLE_RATE,
        help="Analysis sample rate; defaults to GUI-grade visual-first production resolution.",
    )
    parser.add_argument("--waveform-cache-dir", default="", help="Optional WebGUI Boom-mask cache directory.")
    parser.add_argument("--human-correction-log", action="append", default=[], help="Manual review JSONL log to honor.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N sorted tracks.")
    parser.add_argument(
        "--per-track-timeout-sec",
        type=float,
        default=600.0,
        help="Maximum seconds for one visual-first detector call during fresh build. Use 0 to disable.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite same-stamp fresh outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Plan outputs without writing ALS files.")
    parser.add_argument("--no-human-review-overrides", action="store_true", help="Do not promote validated manual review markers.")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-stems", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suspicious-audit-width", type=int, default=8192, help="Full-track Boom-mask audit width.")
    parser.add_argument(
        "--suspicious-audit-workers",
        type=int,
        default=0,
        help="Workers for suspicious-marker audit. Defaults to --workers.",
    )
    parser.add_argument(
        "--suspicious-audit-timeout-sec",
        type=float,
        default=45.0,
        help="Per-row timeout used by the suspicious-marker audit.",
    )
    parser.add_argument(
        "--validate-even-on-build-hold",
        action="store_true",
        help="Run validation when a report exists even if the build exits held.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
