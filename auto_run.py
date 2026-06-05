#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "alsFiles" / "128.als"
MODE_MAP = {
    "safe": "conservative",
    "balanced": "normal",
    "aggressive": "aggressive",
    "audit": "normal",
}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_summary(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_dir():
        candidate = raw / "drop_batch_summary.csv"
        if candidate.exists():
            return candidate.resolve()
    return raw.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "candidates_json",
        "suggested_time",
        "gate_reason",
        "selection_probability",
        "probability_margin",
        "selection_confidence",
        "predicted_abs_error_sec",
        "micro_confidence",
        "snap_offset_ms",
        "status",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _audit_sample(rows: Sequence[Mapping[str, Any]], sample_rate: float) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    for row in rows:
        key = str(row.get("filename") or row.get("candidates_json") or "")
        digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
        if bucket < float(sample_rate):
            selected.append(row)
    return selected


def run_auto(
    *,
    summary: str,
    template: str,
    mode: str,
    dry_run: bool,
    force: bool,
    limit: int,
    expanded_limit: int,
    microalign_limit: int,
    analysis_sr: int,
    audit_sample_rate: float,
) -> Dict[str, Any]:
    summary_path = _resolve_summary(summary)
    template_path = Path(template).expanduser().resolve()
    stamp = _now_stamp()
    report_dir = Path("eval_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    shadow_report = report_dir / f"auto_run_{mode}_{stamp}.json"
    mapped_mode = MODE_MAP[str(mode)]
    cmd = [
        sys.executable,
        "reanalyze_remaining_hard_cases.py",
        "--summary",
        str(summary_path),
        "--template",
        str(template_path),
        "--mode",
        mapped_mode,
        "--apply-auto",
        "--shadow-report",
        str(shadow_report),
        "--expanded-limit",
        str(int(expanded_limit)),
        "--microalign-limit",
        str(int(microalign_limit)),
        "--analysis-sr",
        str(int(analysis_sr)),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")
    if int(limit) > 0:
        cmd.extend(["--limit", str(int(limit))])
    subprocess.run(cmd, check=True)

    shadow = _read_json(shadow_report)
    rows = shadow.get("rows") if isinstance(shadow.get("rows"), list) else []
    accepted = [row for row in rows if isinstance(row, Mapping) and bool(row.get("auto_accept_passed", False))]
    held = [row for row in rows if isinstance(row, Mapping) and not bool(row.get("auto_accept_passed", False))]
    held_csv = report_dir / f"held_for_review_{mode}_{stamp}.csv"
    audit_csv = report_dir / f"audit_sample_{mode}_{stamp}.csv"
    _write_rows(held_csv, held)
    audit_rows = _audit_sample(accepted, audit_sample_rate) if str(mode) == "audit" else []
    if audit_rows:
        _write_rows(audit_csv, audit_rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": str(summary_path),
        "template": str(template_path),
        "mode": str(mode),
        "mapped_reanalysis_mode": mapped_mode,
        "dry_run": bool(dry_run),
        "targets": int(len(rows)),
        "auto_save_eligible": int(len(accepted)),
        "held_for_review": int(len(held)),
        "shadow_report": str(shadow_report),
        "held_for_review_csv": str(held_csv),
        "audit_sample_csv": str(audit_csv) if audit_rows else "",
        "command": cmd,
    }
    report_path = report_dir / f"auto_save_report_{mode}_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    payload["auto_save_report"] = str(report_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autonomous drop auto-save modes with backups and held-review reports.")
    parser.add_argument("library_or_summary", nargs="?", default=str(DEFAULT_SUMMARY), help="Library folder or drop_batch_summary.csv")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Ableton ALS template")
    parser.add_argument("--mode", choices=["safe", "balanced", "aggressive", "audit"], default="safe")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing CSV/JSON/ALS")
    parser.add_argument("--force", action="store_true", help="Reanalyze even if previous multistem reanalysis exists")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expanded-limit", type=int, default=120)
    parser.add_argument("--microalign-limit", type=int, default=50)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--audit-sample-rate", type=float, default=0.10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_auto(
        summary=str(args.library_or_summary),
        template=str(args.template),
        mode=str(args.mode),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        limit=int(args.limit),
        expanded_limit=int(args.expanded_limit),
        microalign_limit=int(args.microalign_limit),
        analysis_sr=int(args.analysis_sr),
        audit_sample_rate=float(args.audit_sample_rate),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
