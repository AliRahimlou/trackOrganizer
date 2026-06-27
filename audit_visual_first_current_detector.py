#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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

from drop_aligner.waveform import gui_boom_mask_strict_contract_issue  # noqa: E402


DEFAULT_REPORT_ROOT = Path.home() / "Desktop" / "MUSIC" / "GeneratedSet" / "VisualFirstFresh"
CURRENT_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_TRACKS_*_report.json"
LEGACY_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_DRUMS_*_report.json"
DERIVED_REPORT_NAME_TOKENS = (
    "_detector_pass_only_report",
    "_production_validation",
    "_current_detector_audit",
)
DEFAULT_OUT_DIR = Path("models/current_detector_audit")
DEFAULT_STRICT_TOLERANCE_SEC = 0.002
DEFAULT_NEAR_TOLERANCE_SEC = 0.050


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(number) if math.isfinite(number) else default


def _safe_read_json(path: str | Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_reports(root: Path, pattern: str) -> List[Path]:
    return sorted(
        (
            path
            for path in root.glob(pattern)
            if path.is_file()
            and not any(token in path.name for token in DERIVED_REPORT_NAME_TOKENS)
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _latest_report(root: Path | None = None) -> Path:
    search_root = (root or DEFAULT_REPORT_ROOT).expanduser()
    current_reports = _source_reports(search_root, CURRENT_REPORT_GLOB)
    if current_reports:
        return current_reports[-1]
    legacy_reports = _source_reports(search_root, LEGACY_REPORT_GLOB)
    if legacy_reports:
        return legacy_reports[-1]
    raise FileNotFoundError(
        "No fresh visual-first report found in "
        f"{search_root} using {CURRENT_REPORT_GLOB} or {LEGACY_REPORT_GLOB}"
    )


def _selected_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    return selected if selected else {}


def _marker_from_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> Optional[float]:
    for value in (
        row.get("marker"),
        payload.get("marker"),
        payload.get("drop_sec"),
        payload.get("final_ai_pick"),
        selected.get("timestamp"),
        selected.get("time_sec"),
        selected.get("microaligned_time"),
        selected.get("snapped_sec"),
    ):
        out = _safe_float(value)
        if out is not None:
            return out
    return None


def _selected_by(row: Mapping[str, Any], payload: Mapping[str, Any], selected: Mapping[str, Any]) -> str:
    return str(selected.get("selected_by") or payload.get("selected_by") or row.get("selected_by") or "")


def _proof_pass(payload: Mapping[str, Any], selected: Mapping[str, Any], key: str) -> bool:
    for source in (payload, selected):
        proof = source.get(key) if isinstance(source, Mapping) else {}
        if isinstance(proof, Mapping) and bool(proof.get("passes")):
            return True
    return False


def _proof_reasons(payload: Mapping[str, Any], selected: Mapping[str, Any], key: str) -> List[str]:
    for source in (payload, selected):
        proof = source.get(key) if isinstance(source, Mapping) else {}
        if isinstance(proof, Mapping) and proof:
            return [str(reason) for reason in proof.get("reasons") or [] if str(reason)]
    return []


def _proof_dict(payload: Mapping[str, Any], selected: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for source in (payload, selected):
        proof = source.get(key) if isinstance(source, Mapping) else {}
        if isinstance(proof, Mapping) and proof:
            return proof
    return {}


def _staccato_gui_contract_relief(gui_proof: Mapping[str, Any], strict_issue: str) -> bool:
    if not bool(gui_proof.get("accepted_by_staccato_front_body_proof")):
        return False
    if gui_proof.get("marker_signal_present") is not True:
        return False
    allowed_issues = {
        "",
        "marker_has_no_immediate_drop_body",
        "marker_not_on_relevant_boom_mask",
        "marker_not_on_gui_boom_front_edge_mask",
    }
    return str(strict_issue or "") in allowed_issues


def _clock(selected: Mapping[str, Any]) -> Mapping[str, Any]:
    clock = selected.get("bpm_clock") if isinstance(selected.get("bpm_clock"), Mapping) else {}
    return clock if isinstance(clock, Mapping) else {}


def _audit_status(payload: Mapping[str, Any]) -> Tuple[str, List[str]]:
    audit = payload.get("visual_audit") if isinstance(payload.get("visual_audit"), Mapping) else {}
    status = str(audit.get("status") or "").strip().lower()
    flags = [str(flag) for flag in audit.get("flag_codes") or [] if str(flag)]
    return status, flags


def _row_key(index: int, row: Mapping[str, Any]) -> str:
    path = str(row.get("drums_path") or row.get("filename") or row.get("audio_path") or "")
    if path:
        return f"{index}:{path}"
    return str(index)


def _parse_indices(raw: str) -> List[int]:
    indices: List[int] = []
    seen: set[int] = set()
    for part in (item.strip() for item in str(raw or "").split(",")):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for value in values:
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            indices.append(value)
    return indices


def _load_report_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _safe_read_json(path)
    rows = payload.get("processed_rows") if isinstance(payload.get("processed_rows"), list) else []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _select_indexed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    indices: Sequence[int],
    offset: int,
    limit: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = [(index, dict(row)) for index, row in enumerate(rows, start=1)]
    if indices:
        wanted = set(indices)
        indexed = [(index, row) for index, row in indexed if index in wanted]
    elif offset:
        indexed = [(index, row) for index, row in indexed if index > int(offset)]
    if limit:
        indexed = indexed[: int(limit)]
    return indexed


def _completed_rows(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return completed
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        key = str(row.get("audit_key") or "")
        if key:
            completed[key] = row
    return completed


def _classify_delta(
    report_marker: Optional[float],
    current_marker: Optional[float],
    *,
    strict_tolerance_sec: float,
    near_tolerance_sec: float,
) -> Dict[str, Any]:
    if report_marker is None:
        return {
            "delta_sec": None,
            "delta_ms": None,
            "strict_match": False,
            "near_match": False,
            "match_status": "missing_report_marker",
        }
    if current_marker is None:
        return {
            "delta_sec": None,
            "delta_ms": None,
            "strict_match": False,
            "near_match": False,
            "match_status": "missing_current_marker",
        }
    delta = abs(float(current_marker) - float(report_marker))
    strict = delta <= float(strict_tolerance_sec)
    near = delta <= float(near_tolerance_sec)
    return {
        "delta_sec": delta,
        "delta_ms": delta * 1000.0,
        "strict_match": strict,
        "near_match": near,
        "match_status": "strict_match" if strict else "near_match" if near else "mismatch",
    }


def _run_detector(audio_path: str, *, sample_rate: int, use_cache: bool) -> Mapping[str, Any]:
    from drop_aligner.visual_first import visual_first_marker

    result = visual_first_marker(audio_path, sample_rate=int(sample_rate), use_cache=bool(use_cache))
    return result if isinstance(result, Mapping) else {"ok": False, "error": "non_mapping_detector_result"}


def _audit_one(
    task: Mapping[str, Any],
    detector_fn: Any = None,
) -> Dict[str, Any]:
    started = time.time()
    index = int(task["index"])
    row = dict(task["row"])
    sample_rate = int(task.get("sample_rate") or 44100)
    use_cache = bool(task.get("use_cache", True))
    strict_tolerance_sec = float(task.get("strict_tolerance_sec") or DEFAULT_STRICT_TOLERANCE_SEC)
    near_tolerance_sec = float(task.get("near_tolerance_sec") or DEFAULT_NEAR_TOLERANCE_SEC)
    report_selected = _selected_candidate(row)
    report_marker = _marker_from_payload(row, row, report_selected)
    drums_path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    base = {
        "audit_key": _row_key(index, row),
        "index": index,
        "drums_path": drums_path,
        "report_marker": report_marker,
        "report_selected_by": _selected_by(row, row, report_selected),
        "sample_rate": sample_rate,
    }
    if not drums_path:
        return {
            **base,
            **_classify_delta(report_marker, None, strict_tolerance_sec=strict_tolerance_sec, near_tolerance_sec=near_tolerance_sec),
            "status": "error",
            "detector_ok": False,
            "detector_error": "missing_drums_path",
            "elapsed_sec": time.time() - started,
        }
    try:
        detector = detector_fn or _run_detector
        current_payload = detector(drums_path, sample_rate=sample_rate, use_cache=use_cache)
        current_payload = current_payload if isinstance(current_payload, Mapping) else {"ok": False, "error": "non_mapping_detector_result"}
    except Exception as exc:
        current_payload = {"ok": False, "error": str(exc) or exc.__class__.__name__}
    current_selected = _selected_candidate(current_payload)
    current_marker = _marker_from_payload({}, current_payload, current_selected)
    delta = _classify_delta(
        report_marker,
        current_marker,
        strict_tolerance_sec=strict_tolerance_sec,
        near_tolerance_sec=near_tolerance_sec,
    )
    clock = _clock(current_selected)
    audit_status, audit_flags = _audit_status(current_payload)
    detector_ok = bool(current_payload.get("ok"))
    boom_pass = _proof_pass(current_payload, current_selected, "boom_proof")
    gui_pass = _proof_pass(current_payload, current_selected, "gui_mask_proof")
    boom_reasons = _proof_reasons(current_payload, current_selected, "boom_proof")
    gui_reasons = _proof_reasons(current_payload, current_selected, "gui_mask_proof")
    gui_proof = _proof_dict(current_payload, current_selected, "gui_mask_proof")
    gui_strict_issue = gui_boom_mask_strict_contract_issue(gui_proof)
    staccato_gui_relief = _staccato_gui_contract_relief(gui_proof, str(gui_strict_issue or ""))
    effective_gui_strict_issue = "" if staccato_gui_relief else str(gui_strict_issue or "")
    on_one = bool(clock.get("on_one")) if clock else False
    current_selected_by = _selected_by({}, current_payload, current_selected)
    if not detector_ok:
        status = "error"
    elif current_marker is None:
        status = "error"
    elif not boom_pass or not gui_pass or bool(effective_gui_strict_issue) or not on_one:
        status = "proof_hold"
    else:
        status = str(delta["match_status"])
    return {
        **base,
        **delta,
        "status": status,
        "detector_ok": detector_ok,
        "detector_error": str(current_payload.get("error") or ""),
        "current_marker": current_marker,
        "current_selected_by": current_selected_by,
        "current_audit_status": audit_status,
        "current_audit_flags": audit_flags,
        "current_boom_proof_pass": boom_pass,
        "current_boom_proof_reasons": boom_reasons,
        "current_boom_proof_clean": boom_pass and not boom_reasons,
        "current_gui_mask_proof_pass": gui_pass,
        "current_gui_mask_proof_reasons": [] if staccato_gui_relief else gui_reasons,
        "current_gui_mask_proof_clean": gui_pass
        and (not gui_reasons or staccato_gui_relief)
        and not bool(effective_gui_strict_issue),
        "current_gui_strict_contract_issue": effective_gui_strict_issue,
        "current_gui_staccato_contract_relief": bool(staccato_gui_relief),
        "current_gui_marker_immediate_body_present": gui_proof.get("marker_immediate_body_present"),
        "current_gui_marker_signal_present": gui_proof.get("marker_signal_present"),
        "current_gui_marker_post_body_occupancy_250ms": _safe_float(
            gui_proof.get("marker_post_body_occupancy_250ms")
        ),
        "current_gui_marker_post_body_occupancy_500ms": _safe_float(
            gui_proof.get("marker_post_body_occupancy_500ms")
        ),
        "current_gui_marker_post_density_mean_250ms": _safe_float(
            gui_proof.get("marker_post_density_mean_250ms")
        ),
        "current_gui_marker_post_density_mean_500ms": _safe_float(
            gui_proof.get("marker_post_density_mean_500ms")
        ),
        "current_gui_nearest_placeable_offset_sec": _safe_float(
            gui_proof.get("nearest_placeable_offset_sec")
        ),
        "current_on_one": on_one,
        "current_one_distance_ms": _safe_float(clock.get("one_distance_ms")) if clock else None,
        "current_clock_source": str(clock.get("source") or "") if clock else "",
        "elapsed_sec": time.time() - started,
    }


def _timeout_result(task: Mapping[str, Any], timeout_sec: float, elapsed_sec: float) -> Dict[str, Any]:
    index = int(task["index"])
    row = dict(task["row"])
    report_selected = _selected_candidate(row)
    report_marker = _marker_from_payload(row, row, report_selected)
    drums_path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    return {
        "audit_key": _row_key(index, row),
        "index": index,
        "drums_path": drums_path,
        "report_marker": report_marker,
        "report_selected_by": _selected_by(row, row, report_selected),
        "sample_rate": int(task.get("sample_rate") or 44100),
        **_classify_delta(
            report_marker,
            None,
            strict_tolerance_sec=float(task.get("strict_tolerance_sec") or DEFAULT_STRICT_TOLERANCE_SEC),
            near_tolerance_sec=float(task.get("near_tolerance_sec") or DEFAULT_NEAR_TOLERANCE_SEC),
        ),
        "status": "timeout",
        "detector_ok": False,
        "detector_error": f"detector_timeout:{float(timeout_sec):.3f}s",
        "elapsed_sec": float(elapsed_sec),
    }


def _audit_one_child(task: Mapping[str, Any], queue: Any) -> None:
    try:
        queue.put(_audit_one(task))
    except BaseException as exc:  # pragma: no cover - defensive child-process boundary.
        queue.put(
            {
                **_timeout_result(task, 0.0, 0.0),
                "status": "error",
                "detector_error": str(exc) or exc.__class__.__name__,
            }
        )


def _audit_one_with_timeout(task: Mapping[str, Any], timeout_sec: float) -> Dict[str, Any]:
    started = time.time()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_audit_one_child, args=(dict(task), queue))
    process.start()
    process.join(timeout=float(timeout_sec))
    elapsed = time.time() - started
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        return _timeout_result(task, timeout_sec, elapsed)
    try:
        row = queue.get_nowait()
    except Exception:
        return {
            **_timeout_result(task, timeout_sec, elapsed),
            "status": "error",
            "detector_error": "detector_worker_empty_result",
        }
    return row if isinstance(row, dict) else {
        **_timeout_result(task, timeout_sec, elapsed),
        "status": "error",
        "detector_error": "detector_worker_non_mapping_result",
    }


def _write_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "index",
        "status",
        "drums_path",
        "report_marker",
        "current_marker",
        "delta_ms",
        "report_selected_by",
        "current_selected_by",
        "detector_ok",
        "detector_error",
        "current_boom_proof_pass",
        "current_gui_mask_proof_pass",
        "current_gui_strict_contract_issue",
        "current_gui_marker_immediate_body_present",
        "current_gui_marker_signal_present",
        "current_gui_marker_post_body_occupancy_250ms",
        "current_gui_marker_post_body_occupancy_500ms",
        "current_gui_marker_post_density_mean_250ms",
        "current_gui_marker_post_density_mean_500ms",
        "current_gui_nearest_placeable_offset_sec",
        "current_on_one",
        "current_one_distance_ms",
        "elapsed_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_report: Path,
    output_jsonl: Path,
    failures_csv: Path,
    strict_tolerance_sec: float,
    near_tolerance_sec: float,
    total_selected_rows: int,
    total_report_rows: int,
    elapsed_sec: float,
) -> Dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    selected_by_counts = Counter(str(row.get("current_selected_by") or "") for row in rows)
    strict_count = sum(1 for row in rows if bool(row.get("strict_match")))
    near_count = sum(1 for row in rows if bool(row.get("near_match")))
    proof_ready_count = sum(
        1
        for row in rows
        if bool(row.get("detector_ok"))
        and bool(row.get("current_boom_proof_pass"))
        and bool(row.get("current_gui_mask_proof_pass"))
        and not str(row.get("current_gui_strict_contract_issue") or "")
        and bool(row.get("current_on_one"))
    )
    clean_proof_ready_count = sum(
        1
        for row in rows
        if bool(row.get("detector_ok"))
        and bool(row.get("current_boom_proof_clean"))
        and bool(row.get("current_gui_mask_proof_clean"))
        and not str(row.get("current_gui_strict_contract_issue") or "")
        and bool(row.get("current_on_one"))
    )
    boom_repaired_count = sum(
        1
        for row in rows
        if bool(row.get("current_boom_proof_pass")) and bool(row.get("current_boom_proof_reasons"))
    )
    gui_repaired_count = sum(
        1
        for row in rows
        if bool(row.get("current_gui_mask_proof_pass")) and bool(row.get("current_gui_mask_proof_reasons"))
    )
    detector_error_count = sum(1 for row in rows if not bool(row.get("detector_ok")))
    timeout_count = sum(1 for row in rows if str(row.get("status") or "") == "timeout")
    hard_failures = [
        row
        for row in rows
        if str(row.get("status") or "") not in {"strict_match", "near_match"}
    ]
    slowest = sorted(rows, key=lambda row: float(row.get("elapsed_sec") or 0.0), reverse=True)[:10]
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(source_report),
        "output_jsonl": str(output_jsonl),
        "failures_csv": str(failures_csv),
        "strict_tolerance_sec": float(strict_tolerance_sec),
        "near_tolerance_sec": float(near_tolerance_sec),
        "total_report_rows": int(total_report_rows),
        "selected_row_count": int(total_selected_rows),
        "completed_count": len(rows),
        "remaining_selected_count": max(0, int(total_selected_rows) - len(rows)),
        "strict_match_count": int(strict_count),
        "near_match_count": int(near_count),
        "hard_mismatch_count": int(status_counts.get("mismatch", 0)),
        "proof_hold_count": int(status_counts.get("proof_hold", 0)),
        "detector_error_count": int(detector_error_count),
        "timeout_count": int(timeout_count),
        "proof_ready_count": int(proof_ready_count),
        "clean_proof_ready_count": int(clean_proof_ready_count),
        "boom_proof_repaired_pass_count": int(boom_repaired_count),
        "gui_mask_repaired_pass_count": int(gui_repaired_count),
        "all_selected_completed": len(rows) == int(total_selected_rows) and int(total_selected_rows) > 0,
        "all_selected_near_match": len(rows) == int(total_selected_rows) and len(rows) > 0 and near_count == len(rows),
        "all_selected_strict_match": len(rows) == int(total_selected_rows) and len(rows) > 0 and strict_count == len(rows),
        "all_selected_proof_ready": len(rows) == int(total_selected_rows) and len(rows) > 0 and proof_ready_count == len(rows),
        "all_selected_clean_proof_ready": len(rows) == int(total_selected_rows)
        and len(rows) > 0
        and clean_proof_ready_count == len(rows),
        "status_counts": dict(status_counts),
        "selected_by_counts": dict(selected_by_counts),
        "hard_failure_samples": hard_failures[:25],
        "slowest_rows": slowest,
        "elapsed_sec": float(elapsed_sec),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    report_path = Path(args.report).expanduser().resolve() if args.report else _latest_report().resolve()
    rows = _load_report_rows(report_path)
    indices = _parse_indices(args.indices)
    indexed_rows = _select_indexed_rows(rows, indices=indices, offset=int(args.offset), limit=int(args.limit))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{report_path.stem}_current_detector_audit.jsonl"
    summary_path = out_dir / f"{report_path.stem}_current_detector_audit.json"
    failures_csv = out_dir / f"{report_path.stem}_current_detector_audit_failures.csv"
    if bool(args.force) and jsonl_path.exists():
        jsonl_path.unlink()
    completed = _completed_rows(jsonl_path) if bool(args.resume) else {}
    selected_keys = {_row_key(index, row) for index, row in indexed_rows}
    completed = {key: row for key, row in completed.items() if key in selected_keys}
    remaining = [(index, row) for index, row in indexed_rows if _row_key(index, row) not in completed]
    jobs = [
        {
            "index": index,
            "row": row,
            "sample_rate": int(args.sample_rate),
            "use_cache": bool(args.use_cache),
            "strict_tolerance_sec": float(args.strict_tolerance_sec),
            "near_tolerance_sec": float(args.near_tolerance_sec),
        }
        for index, row in remaining
    ]
    results_by_key = dict(completed)
    workers = max(1, int(args.workers))
    per_row_timeout_sec = float(args.per_row_timeout_sec or 0.0)
    if jobs:
        if per_row_timeout_sec > 0.0:
            if workers != 1:
                print("--per-row-timeout-sec uses sequential workers=1 so slow rows can be killed safely", flush=True)
                workers = 1
            for count, job in enumerate(jobs, start=1):
                row = _audit_one_with_timeout(job, per_row_timeout_sec)
                results_by_key[str(row["audit_key"])] = row
                _write_jsonl_row(jsonl_path, row)
                done = len(results_by_key)
                if args.progress_every and (count % int(args.progress_every) == 0 or done == len(indexed_rows)):
                    print(f"audited {done}/{len(indexed_rows)} current detector rows", flush=True)
                if args.max_runtime_sec and (time.time() - started) >= float(args.max_runtime_sec):
                    print("stopping after --max-runtime-sec; rerun with --resume to continue", flush=True)
                    break
        elif workers == 1:
            for count, job in enumerate(jobs, start=1):
                row = _audit_one(job)
                results_by_key[str(row["audit_key"])] = row
                _write_jsonl_row(jsonl_path, row)
                done = len(results_by_key)
                if args.progress_every and (count % int(args.progress_every) == 0 or done == len(indexed_rows)):
                    print(f"audited {done}/{len(indexed_rows)} current detector rows", flush=True)
                if args.max_runtime_sec and (time.time() - started) >= float(args.max_runtime_sec):
                    print("stopping after --max-runtime-sec; rerun with --resume to continue", flush=True)
                    break
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_audit_one, job) for job in jobs]
                for count, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    results_by_key[str(row["audit_key"])] = row
                    _write_jsonl_row(jsonl_path, row)
                    done = len(results_by_key)
                    if args.progress_every and (count % int(args.progress_every) == 0 or done == len(indexed_rows)):
                        print(f"audited {done}/{len(indexed_rows)} current detector rows", flush=True)
                    if args.max_runtime_sec and (time.time() - started) >= float(args.max_runtime_sec):
                        print(
                            "--max-runtime-sec reached after submitted workers finish; rerun with --resume to continue",
                            flush=True,
                        )
                        break
    ordered_results = [
        results_by_key[key]
        for key in (_row_key(index, row) for index, row in indexed_rows)
        if key in results_by_key
    ]
    hard_failures = [
        row
        for row in ordered_results
        if str(row.get("status") or "") not in {"strict_match", "near_match"}
    ]
    _write_csv(failures_csv, hard_failures)
    summary = _summarize(
        ordered_results,
        source_report=report_path,
        output_jsonl=jsonl_path,
        failures_csv=failures_csv,
        strict_tolerance_sec=float(args.strict_tolerance_sec),
        near_tolerance_sec=float(args.near_tolerance_sec),
        total_selected_rows=len(indexed_rows),
        total_report_rows=len(rows),
        elapsed_sec=time.time() - started,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(json.dumps({**summary, "output_json": str(summary_path)}, indent=2, ensure_ascii=True, default=_json_default))
    if bool(args.require_all_near_match) and not (
        bool(summary.get("all_selected_completed"))
        and bool(summary.get("all_selected_near_match"))
        and bool(summary.get("all_selected_proof_ready"))
    ):
        return {**summary, "output_json": str(summary_path), "exit_code": 1}
    if bool(args.require_all_strict_match) and not (
        bool(summary.get("all_selected_completed"))
        and bool(summary.get("all_selected_strict_match"))
        and bool(summary.get("all_selected_proof_ready"))
    ):
        return {**summary, "output_json": str(summary_path), "exit_code": 1}
    return {**summary, "output_json": str(summary_path), "exit_code": 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable audit of today's visual_first_marker output against a saved "
            "visual-first production report."
        )
    )
    parser.add_argument("report", nargs="?", default="", help="Fresh visual-first report JSON; defaults to latest report.")
    parser.add_argument("--out-dir", default="", help="Directory for JSONL cache, summary JSON, and failure CSV.")
    parser.add_argument("--indices", default="", help="1-based row indices or ranges, for example 1,254,800-820.")
    parser.add_argument("--offset", type=int, default=0, help="Skip rows through this 1-based index unless --indices is used.")
    parser.add_argument("--limit", type=int, default=0, help="Audit only this many selected rows; 0 means all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Delete this audit's JSONL cache before running.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-runtime-sec", type=float, default=0.0)
    parser.add_argument(
        "--per-row-timeout-sec",
        type=float,
        default=0.0,
        help="Sequentially kill and record any detector row taking longer than this many seconds.",
    )
    parser.add_argument("--strict-tolerance-sec", type=float, default=DEFAULT_STRICT_TOLERANCE_SEC)
    parser.add_argument("--near-tolerance-sec", type=float, default=DEFAULT_NEAR_TOLERANCE_SEC)
    parser.add_argument(
        "--require-all-near-match",
        action="store_true",
        help="Exit nonzero unless every selected row completes, is proof-ready, and matches within near tolerance.",
    )
    parser.add_argument(
        "--require-all-strict-match",
        action="store_true",
        help="Exit nonzero unless every selected row completes, is proof-ready, and matches within strict tolerance.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = audit(build_parser().parse_args(argv))
    return int(result.get("exit_code", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
