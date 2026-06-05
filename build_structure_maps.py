#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.exclusions import EXCLUDED_DIR_NAMES, is_excluded_path, row_has_excluded_path
from drop_aligner.structure_map import analyze_track_structure


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"}
DEFAULT_SUMMARY = Path.home() / "Desktop" / "MUSIC" / "STEMS" / "drop_batch_summary.csv"
DEFAULT_OUTPUT_DIR = Path("models") / "structure_maps"
SUMMARY_COLUMNS = (
    "structure_map_json",
    "structure_ok",
    "structure_error",
    "structure_bar_count",
    "structure_first_drop",
    "structure_second_drop",
    "structure_sections",
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_summary(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _write_summary(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def _stable_id(row: Mapping[str, str]) -> str:
    raw = "|".join([row.get("filename", ""), row.get("output_als", ""), row.get("candidates_json", "")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_reviewed_ids(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return set()
    items = state.get("items") if isinstance(state, Mapping) else {}
    if not isinstance(items, Mapping):
        return set()
    return {
        str(item_id)
        for item_id, item_state in items.items()
        if isinstance(item_state, Mapping) and bool(item_state.get("reviewed") or item_state.get("skipped"))
    }


def _discover_audio(folder: Path, *, recursive: bool, stem_role: str) -> List[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    role = str(stem_role or "").strip().lower()
    out = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if role and not path.name.lower().startswith(role):
            continue
        if is_excluded_path(path, excluded_dir_names=EXCLUDED_DIR_NAMES):
            continue
        out.append(path)
    return sorted(out, key=lambda item: str(item).lower())


def _target_rows_from_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    reviewed_ids: set[str],
    remaining_only: bool,
) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for index, row in enumerate(rows):
        if row_has_excluded_path(row):
            continue
        if remaining_only and _stable_id(row) in reviewed_ids:
            continue
        audio = str(row.get("filename") or "")
        if audio:
            out.append((index, audio))
    return out


def _output_path(audio_path: str, output_dir: Path) -> Path:
    audio = Path(audio_path).expanduser()
    digest = hashlib.sha1(str(audio.resolve() if audio.exists() else audio).encode("utf-8", errors="ignore")).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in audio.stem)[:80]
    return output_dir / f"{digest}_{safe_stem}_structure_map.json"


def _float_string(value: Any, digits: int = 9) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def _candidate_time(candidate: Any) -> str:
    if not isinstance(candidate, Mapping):
        return ""
    for key in ("timestamp", "snapped_sec", "time_sec"):
        if candidate.get(key) not in (None, ""):
            return _float_string(candidate.get(key))
    return ""


def _sections_string(sections: Any) -> str:
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        return ""
    parts = []
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        label = str(section.get("label") or "")
        start = section.get("start_bar")
        end = section.get("end_bar")
        if label and start and end:
            parts.append(f"{label}:{start}-{end}")
    return "; ".join(parts)


def _stem_roles(structure: Mapping[str, Any]) -> str:
    stem_group = structure.get("stem_group") if isinstance(structure.get("stem_group"), Mapping) else {}
    roles = stem_group.get("roles") if isinstance(stem_group.get("roles"), Mapping) else {}
    return ",".join(sorted(str(role) for role in roles.keys()))


def _process_one(task: Mapping[str, Any]) -> Dict[str, Any]:
    audio_path = str(task["audio_path"])
    output_path = Path(str(task["output_path"])).expanduser()
    sample_rate = int(task["sample_rate"])
    force = bool(task["force"])
    use_cache = bool(task["use_cache"])
    if output_path.exists() and not force:
        try:
            with open(output_path, "r", encoding="utf-8") as fh:
                structure = json.load(fh)
            if isinstance(structure, Mapping):
                return _summary_row(audio_path, output_path, structure, status="cached")
        except Exception:
            pass
    try:
        structure = analyze_track_structure(audio_path, sample_rate=sample_rate, use_cache=use_cache)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(structure, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
        tmp.replace(output_path)
        return _summary_row(audio_path, output_path, structure, status="processed")
    except Exception as exc:
        return {
            "status": "error",
            "filename": audio_path,
            "structure_map_json": str(output_path),
            "structure_ok": "false",
            "structure_error": str(exc) or exc.__class__.__name__,
            "structure_bar_count": "",
            "structure_first_drop": "",
            "structure_second_drop": "",
            "structure_sections": "",
            "structure_stem_roles": "",
        }


def _summary_row(audio_path: str, output_path: Path, structure: Mapping[str, Any], *, status: str) -> Dict[str, Any]:
    return {
        "status": status,
        "filename": audio_path,
        "structure_map_json": str(output_path),
        "structure_ok": "true" if bool(structure.get("ok")) else "false",
        "structure_error": str(structure.get("error") or ""),
        "structure_bar_count": int(structure.get("bar_count", 0) or 0),
        "structure_first_drop": _candidate_time(structure.get("first_drop")),
        "structure_second_drop": _candidate_time(structure.get("second_drop")),
        "structure_sections": _sections_string(structure.get("sections")),
        "structure_stem_roles": _stem_roles(structure),
    }


def _update_summary_row(row: Mapping[str, str], structure_row: Mapping[str, Any]) -> Dict[str, str]:
    out = dict(row)
    for key in SUMMARY_COLUMNS:
        out[key] = str(structure_row.get(key, ""))
    return out


def build_structure_maps(
    *,
    library_or_summary: str,
    output_dir: str,
    report_json: str,
    report_csv: str,
    state: str = "",
    recursive: bool = True,
    stem_role: str = "drums",
    remaining_only: bool = False,
    update_summary: bool = False,
    limit: int = 0,
    workers: int = 1,
    sample_rate: int = 16000,
    force: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    source = Path(library_or_summary).expanduser()
    output_root = Path(output_dir).expanduser()
    summary_path: Optional[Path] = None
    summary_rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    targets: List[Tuple[Optional[int], str]] = []
    if source.is_file() and source.name.endswith(".csv"):
        summary_path = source.resolve()
        summary_rows, fieldnames = _read_summary(summary_path)
        state_path = Path(state).expanduser().resolve() if state else summary_path.parent / "review_state.json"
        reviewed_ids = _load_reviewed_ids(state_path)
        targets = [(index, audio) for index, audio in _target_rows_from_summary(summary_rows, reviewed_ids=reviewed_ids, remaining_only=remaining_only)]
    elif source.is_dir():
        targets = [(None, str(path.resolve())) for path in _discover_audio(source, recursive=recursive, stem_role=stem_role)]
    else:
        targets = [(None, str(source.resolve()))]
    if limit > 0:
        targets = targets[: int(limit)]

    started = time.time()
    tasks = [
        {
            "row_index": row_index,
            "audio_path": audio_path,
            "output_path": str(_output_path(audio_path, output_root)),
            "sample_rate": int(sample_rate),
            "force": bool(force),
            "use_cache": bool(use_cache),
        }
        for row_index, audio_path in targets
    ]
    results: List[Dict[str, Any]] = []
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            row = _process_one(task)
            row["row_index"] = task["row_index"]
            results.append(row)
            print(f"[{len(results)}/{len(tasks)}] {row['status']}: {Path(row['filename']).name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            future_map = {pool.submit(_process_one, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                row = future.result()
                row["row_index"] = task["row_index"]
                results.append(row)
                print(f"[{len(results)}/{len(tasks)}] {row['status']}: {Path(row['filename']).name}", flush=True)
    results.sort(key=lambda row: (str(row.get("filename", "")).lower(), int(row.get("row_index") or -1)))

    if update_summary and summary_path is not None:
        by_index = {int(row["row_index"]): row for row in results if row.get("row_index") is not None}
        for index, row in enumerate(summary_rows):
            if index in by_index:
                summary_rows[index] = _update_summary_row(row, by_index[index])
        for name in SUMMARY_COLUMNS:
            if name not in fieldnames:
                fieldnames.append(name)
        _write_summary(summary_path, summary_rows, fieldnames)

    ok_count = sum(1 for row in results if str(row.get("structure_ok")) == "true")
    error_count = sum(1 for row in results if row.get("status") == "error")
    missing_first = sum(1 for row in results if str(row.get("structure_ok")) == "true" and not row.get("structure_first_drop"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "summary": "" if summary_path is None else str(summary_path),
        "output_dir": str(output_root),
        "targets": int(len(tasks)),
        "processed_or_cached": int(len(results) - error_count),
        "ok": int(ok_count),
        "errors": int(error_count),
        "ok_without_first_drop": int(missing_first),
        "elapsed_sec": round(float(time.time() - started), 3),
        "sample_rate": int(sample_rate),
        "remaining_only": bool(remaining_only),
        "update_summary": bool(update_summary),
        "rows": results,
    }
    report_json_path = Path(report_json).expanduser()
    report_csv_path = Path(report_csv).expanduser()
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    fieldnames_out = [
        "status",
        "filename",
        "structure_map_json",
        "structure_ok",
        "structure_error",
        "structure_bar_count",
        "structure_first_drop",
        "structure_second_drop",
        "structure_sections",
        "structure_stem_roles",
    ]
    with open(report_csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_out, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    report["report_json"] = str(report_json_path)
    report["report_csv"] = str(report_csv_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    stamp = _now_stamp()
    parser = argparse.ArgumentParser(description="Build machine-readable per-bar structure maps for library tracks.")
    parser.add_argument("library_or_summary", nargs="?", default=str(DEFAULT_SUMMARY), help="Audio file, folder, or drop_batch_summary.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-json", default=f"eval_reports/structure_maps_{stamp}.json")
    parser.add_argument("--report-csv", default=f"eval_reports/structure_maps_{stamp}.csv")
    parser.add_argument("--state", default="", help="review_state.json path when using --remaining-only")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stem-role", default="drums", help="Folder discovery prefix filter; ignored for summary CSV input")
    parser.add_argument("--remaining-only", action="store_true", help="Skip reviewed/skipped rows from review_state.json")
    parser.add_argument("--update-summary", action="store_true", help="Add structure_map_json and structure fields to the summary CSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_structure_maps(
        library_or_summary=str(args.library_or_summary),
        output_dir=str(args.output_dir),
        report_json=str(args.report_json),
        report_csv=str(args.report_csv),
        state=str(args.state or ""),
        recursive=bool(args.recursive),
        stem_role=str(args.stem_role or ""),
        remaining_only=bool(args.remaining_only),
        update_summary=bool(args.update_summary),
        limit=int(args.limit),
        workers=int(args.workers),
        sample_rate=int(args.analysis_sr),
        force=bool(args.force),
        use_cache=bool(args.use_cache),
    )
    compact = {key: report[key] for key in ("targets", "ok", "errors", "ok_without_first_drop", "elapsed_sec", "report_json", "report_csv")}
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
