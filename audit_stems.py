#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.multistem import find_stem_group
from project_config import DROP_BATCH_SUMMARY


DEFAULT_SUMMARY = DROP_BATCH_SUMMARY


def _read_summary(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _status(row: Mapping[str, str]) -> bool:
    status = str(row.get("status", "")).strip().lower()
    if status and status not in {"processed", "skipped", "partial"}:
        return False
    als_valid = str(row.get("als_valid", "")).strip().lower()
    if als_valid and als_valid != "true":
        return False
    return True


def audit(summary: Path, *, limit: int = 0) -> Dict[str, Any]:
    rows = [row for row in _read_summary(summary) if not row_has_excluded_path(row) and _status(row)]
    if limit > 0:
        rows = rows[:limit]
    role_sets: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []
    stem_counts: Counter[str] = Counter()
    for row in rows:
        audio = row.get("filename", "")
        if not audio:
            continue
        group = find_stem_group(audio)
        roles = sorted(group.roles.keys())
        for role in roles:
            stem_counts[role] += 1
        role_sets["+".join(roles) if roles else "none"] += 1
        for role in ("drums", "instrumental", "vocals", "bass"):
            if role not in group.roles:
                missing[role] += 1
        if len(examples) < 20 and ("instrumental" not in group.roles or "vocals" not in group.roles):
            examples.append(
                {
                    "track": audio,
                    "roles": roles,
                    "missing": [role for role in ("drums", "instrumental", "vocals", "bass") if role not in group.roles],
                    "root": group.root,
                }
            )
    return {
        "summary": str(summary),
        "rows": int(len(rows)),
        "stem_counts": dict(stem_counts),
        "role_sets": dict(role_sets),
        "missing_counts": dict(missing),
        "missing_examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit available drums/inst/vocals/bass stems for the drop review set.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv path")
    parser.add_argument("--limit", type=int, default=0, help="Only audit this many rows")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = Path(args.summary).expanduser().resolve()
    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}")
    report = audit(summary, limit=int(args.limit))
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
