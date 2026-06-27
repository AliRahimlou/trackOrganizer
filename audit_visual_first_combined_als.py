from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from build_fresh_visual_first_library_set import _verify_combined_set


DEFAULT_ALS = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_contract_hardened_v5_freshproof.als"
)
DEFAULT_REPORT = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_contract_hardened_v5_freshproof_report.json"
)
DEFAULT_OUT = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "validation_freshproof/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_20260620_contract_hardened_v5_freshproof_combined_als_audit.json"
)


def _load_processed_rows(report_path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(report_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Report is not a JSON object: {report_path}")
    rows = payload.get("processed_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Report has no processed_rows list: {report_path}")
    return [row for row in rows if isinstance(row, Mapping)]


def audit_combined_als(
    als_path: Path,
    report_path: Path,
    *,
    tolerance_sec: float = 0.002,
) -> Dict[str, Any]:
    rows = _load_processed_rows(report_path)
    verification = _verify_combined_set(
        als_path.expanduser(),
        rows,
        tolerance_sec=float(tolerance_sec),
    )
    passed = bool(
        verification.get("valid_xml")
        and verification.get("all_rows_match_expected")
        and verification.get("all_file_refs_match_expected")
        and verification.get("row_count_matches_report")
        and verification.get("all_rows_anchored")
        and int(verification.get("anchor_mismatch_count") or 0) == 0
        and int(verification.get("file_ref_mismatch_count") or 0) == 0
    )
    return {
        "source_als": str(als_path.expanduser()),
        "source_report": str(report_path.expanduser()),
        "tolerance_sec": float(tolerance_sec),
        "processed_rows": len(rows),
        "passed": bool(passed),
        **verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly audit a combined visual-first ALS against its processed marker report.",
    )
    parser.add_argument("--als", default=str(DEFAULT_ALS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tolerance-sec", type=float, default=0.002)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_combined_als(
        Path(args.als),
        Path(args.report),
        tolerance_sec=float(args.tolerance_sec),
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not bool(result.get("passed")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
