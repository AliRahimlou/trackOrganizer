#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _reexec_with_local_venv() -> None:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_with_local_venv()

from drop_aligner.visual_first import visual_first_marker  # noqa: E402
from web_review import ReviewApp, _default_summary_path, _default_template_path  # noqa: E402


DEFAULT_JSON = "models/visual_placement_audit.json"
DEFAULT_CSV = "models/visual_placement_audit_suspects.csv"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _review_status(item: Mapping[str, Any]) -> str:
    review = item.get("review") if isinstance(item.get("review"), Mapping) else {}
    if review.get("skipped"):
        return "skipped"
    if review.get("corrected"):
        return "corrected"
    if review.get("approved"):
        return "approved"
    if review.get("reviewed"):
        return "reviewed"
    return "remaining"


def _timestamp_reviewed(item: Mapping[str, Any]) -> str:
    review = item.get("review") if isinstance(item.get("review"), Mapping) else {}
    return str(review.get("timestamp_reviewed") or "")


def _iter_scope_items(app: ReviewApp, scope: str) -> Iterable[Dict[str, Any]]:
    if scope == "current":
        current = app._current_item()
        if current:
            yield current
        return
    for item in app.items:
        status = _review_status(item)
        if scope == "all":
            yield item
        elif scope == "skipped" and status == "skipped":
            yield item
        elif scope == "remaining" and status == "remaining":
            yield item
        elif scope == "reviewed" and status in {"approved", "corrected", "reviewed"}:
            yield item


def _candidate_time(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "snapped_sec", "time_sec", "microaligned_time"):
        value = _float_or_none(candidate.get(key))
        if value is not None:
            return float(value)
    return None


def _audit_one(job: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.time()
    audio_path = str(job.get("audio_path") or "")
    try:
        visual = visual_first_marker(
            audio_path,
            sample_rate=int(job.get("sample_rate") or 16000),
            use_cache=True,
            rejected_sections=[dict(row) for row in job.get("rejected_sections") or [] if isinstance(row, Mapping)],
        )
    except Exception as exc:
        return {
            **{key: job.get(key) for key in ("id", "audio_path", "track_name", "review_status", "timestamp_reviewed")},
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
            "elapsed_sec": time.time() - started,
            "audit_status": "error",
            "suspect": True,
        }

    selected = visual.get("selected_candidate") if isinstance(visual.get("selected_candidate"), Mapping) else {}
    audit = visual.get("visual_audit") if isinstance(visual.get("visual_audit"), Mapping) else {}
    selected_audit = audit.get("selected") if isinstance(audit.get("selected"), Mapping) else {}
    preferred = audit.get("preferred_candidate") if isinstance(audit.get("preferred_candidate"), Mapping) else None
    status = str(audit.get("status") or "unknown")
    marker = _float_or_none(visual.get("marker")) or _candidate_time(selected)
    preferred_time = _float_or_none(preferred.get("time")) if isinstance(preferred, Mapping) else None
    return {
        **{key: job.get(key) for key in ("id", "audio_path", "track_name", "review_status", "timestamp_reviewed")},
        "ok": bool(visual.get("ok")),
        "error": visual.get("error"),
        "audit_status": status,
        "recommended_action": str(audit.get("recommended_action") or ""),
        "suspect": status != "pass",
        "marker": marker,
        "raw_visual_time": _float_or_none(visual.get("raw_visual_time")),
        "clock_bar": selected_audit.get("clock_bar"),
        "drop_strength": selected_audit.get("drop_strength"),
        "selected_by": str(selected.get("selected_by") or ""),
        "reason": str(selected.get("reason") or ""),
        "flag_codes": list(audit.get("flag_codes") or []),
        "preferred_time": preferred_time,
        "preferred_clock_bar": preferred.get("clock_bar") if isinstance(preferred, Mapping) else None,
        "preferred_drop_strength": preferred.get("drop_strength") if isinstance(preferred, Mapping) else None,
        "candidate_count": int(audit.get("candidate_count") or 0),
        "visual_audit": dict(audit),
        "elapsed_sec": time.time() - started,
    }


def _load_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    app = ReviewApp(
        summary_csv=str(Path(args.summary_csv).expanduser() if args.summary_csv else _default_summary_path()),
        template=str(Path(args.template).expanduser() if args.template else _default_template_path()),
        correction_log=str(args.correction_log),
        auto_retrain_every=0,
        review_low_only=False,
        review_medium_and_low=False,
        regenerate_als_on_correction=False,
        visual_first=True,
        review_queue=str(args.queue or ""),
    )
    items = list(_iter_scope_items(app, str(args.scope)))
    needle = str(args.track_contains or "").strip().lower()
    if needle:
        items = [
            item
            for item in items
            if needle in str(item.get("audio_path") or "").lower()
            or needle in str(item.get("track_name") or "").lower()
        ]
    if args.latest:
        items.sort(key=_timestamp_reviewed, reverse=True)
    if args.limit:
        items = items[: int(args.limit)]
    jobs: List[Dict[str, Any]] = []
    for item in items:
        jobs.append(
            {
                "id": item.get("id"),
                "audio_path": item.get("audio_path"),
                "track_name": item.get("track_name"),
                "review_status": _review_status(item),
                "timestamp_reviewed": _timestamp_reviewed(item),
                "rejected_sections": app._visual_rejections_for_item(item),
                "sample_rate": int(args.sample_rate),
            }
        )
    return jobs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audit_status",
        "recommended_action",
        "marker",
        "preferred_time",
        "clock_bar",
        "preferred_clock_bar",
        "flag_codes",
        "track_name",
        "audio_path",
        "review_status",
        "timestamp_reviewed",
        "selected_by",
        "reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    jobs = _load_jobs(args)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            rows.append(_audit_one(job))
            if args.progress_every and index % int(args.progress_every) == 0:
                print(f"audited {index}/{len(jobs)}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_audit_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if args.progress_every and index % int(args.progress_every) == 0:
                    print(f"audited {index}/{len(jobs)}", file=sys.stderr, flush=True)

    rows.sort(key=lambda row: (0 if row.get("suspect") else 1, str(row.get("track_name") or "")))
    suspects = [row for row in rows if row.get("suspect")]
    summary = {
        "scope": str(args.scope),
        "audited": len(rows),
        "pass": sum(1 for row in rows if row.get("audit_status") == "pass"),
        "review": sum(1 for row in rows if row.get("audit_status") == "review"),
        "replace": sum(1 for row in rows if row.get("audit_status") == "replace"),
        "errors": sum(1 for row in rows if not row.get("ok")),
        "suspects": len(suspects),
        "elapsed_sec": time.time() - started,
    }
    payload = {"summary": summary, "rows": rows}

    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv:
        _write_csv(Path(args.output_csv).expanduser(), suspects)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit visual-first blue marker placements without opening the GUI.")
    parser.add_argument("summary_csv", nargs="?", help="drop_batch_summary.csv. Defaults to web_review.py's default.")
    parser.add_argument("--template", default="", help="ALS template path. Only used to initialize ReviewApp metadata.")
    parser.add_argument("--correction-log", default="drop_corrections.jsonl")
    parser.add_argument("--queue", default="", help="Optional active_learning_queue.csv filter.")
    parser.add_argument("--scope", choices=("skipped", "remaining", "reviewed", "current", "all"), default="skipped")
    parser.add_argument("--latest", action="store_true", help="Audit most recently reviewed/skipped items first.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--track-contains", default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    parser.add_argument("--stdout", action="store_true", help="Print full JSON payload to stdout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    summary = payload["summary"]
    if args.stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            "audited={audited} pass={pass} review={review} replace={replace} "
            "suspects={suspects} errors={errors} elapsed={elapsed_sec:.2f}s".format(**summary)
        )
        if args.output_json:
            print(f"json={Path(args.output_json).expanduser()}")
        if args.output_csv:
            print(f"csv={Path(args.output_csv).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
