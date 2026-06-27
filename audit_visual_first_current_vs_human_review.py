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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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

from audit_visual_first_human_review_memory import (  # noqa: E402
    ReviewMarker,
    build_review_indexes,
    find_review_for_row,
    load_review_markers,
    review_marker_gate,
)
from drop_aligner.boom_profile import boom_proof_front_edge_freshness, marker_boom_proof  # noqa: E402
from drop_aligner.waveform import (  # noqa: E402
    accept_gui_boom_mask_with_front_edge_proof,
    gui_boom_mask_proof,
    gui_boom_mask_strict_contract_issue,
)


DEFAULT_REPORT_ROOT = Path.home() / "Desktop" / "MUSIC" / "GeneratedSet" / "VisualFirstFresh"
CURRENT_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_TRACKS_*_report.json"
LEGACY_REPORT_GLOB = "VISUAL_FIRST_FRESH_ALL_DRUMS_*_report.json"
DERIVED_REPORT_NAME_TOKENS = (
    "_detector_pass_only_report",
    "_production_validation",
    "_current_detector_audit",
    "_current_vs_human_review",
)
DEFAULT_OUT_DIR = Path("artifacts/current_vs_human_review")
DEFAULT_CACHE_DIR = DEFAULT_REPORT_ROOT / ".waveform_cache"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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
            for path in root.expanduser().glob(pattern)
            if path.is_file() and not any(token in path.name for token in DERIVED_REPORT_NAME_TOKENS)
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _latest_report(root: Optional[Path] = None) -> Path:
    search_root = (root or DEFAULT_REPORT_ROOT).expanduser()
    current = _source_reports(search_root, CURRENT_REPORT_GLOB)
    if current:
        return current[-1]
    legacy = _source_reports(search_root, LEGACY_REPORT_GLOB)
    if legacy:
        return legacy[-1]
    raise FileNotFoundError(f"No visual-first report found in {search_root}")


def _load_report_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _safe_read_json(path)
    rows = payload.get("processed_rows") if isinstance(payload.get("processed_rows"), list) else []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _row_key(index: int, row: Mapping[str, Any]) -> str:
    path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    return f"{int(index)}:{path}" if path else str(int(index))


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
        if isinstance(row, dict) and row.get("audit_key"):
            completed[str(row["audit_key"])] = row
    return completed


def _write_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=True, default=_json_default) + "\n")


def _parse_indices(raw: str) -> List[int]:
    indices: List[int] = []
    seen: set[int] = set()
    for part in (item.strip() for item in str(raw or "").split(",")):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left.strip()), int(right.strip())
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


def _select_indexed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    indices: Sequence[int],
    offset: int,
    limit: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = [(index, dict(row)) for index, row in enumerate(rows, 1)]
    if indices:
        wanted = set(indices)
        indexed = [(index, row) for index, row in indexed if index in wanted]
    elif offset:
        indexed = [(index, row) for index, row in indexed if index > int(offset)]
    if limit:
        indexed = indexed[: int(limit)]
    return indexed


def _default_correction_logs(report_path: Path, out_dir: Path) -> List[Path]:
    repo = Path(__file__).resolve().parent
    candidates: List[Path] = [
        repo / "drop_corrections.jsonl",
        report_path.expanduser().parent / "drop_corrections.jsonl",
        out_dir.expanduser() / "drop_corrections.jsonl",
    ]
    candidates.extend(sorted(repo.glob("drop_corrections.before*.jsonl")))
    backups = repo / "backups"
    if backups.exists():
        candidates.extend(sorted(backups.glob("*/drop_corrections.jsonl")))
    seen: set[str] = set()
    logs: List[Path] = []
    for candidate in candidates:
        path = candidate.expanduser()
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        logs.append(path)
    return logs


def _selected_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), Mapping) else {}
    return selected if isinstance(selected, Mapping) else {}


def _marker_from_payload(payload: Mapping[str, Any], selected: Mapping[str, Any]) -> Optional[float]:
    for value in (
        payload.get("marker"),
        payload.get("drop_sec"),
        payload.get("final_ai_pick"),
        selected.get("timestamp"),
        selected.get("time_sec"),
        selected.get("microaligned_time"),
        selected.get("snapped_sec"),
    ):
        out = _finite_float(value)
        if out is not None:
            return float(out)
    return None


def _proof_dict(payload: Mapping[str, Any], selected: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for source in (payload, selected):
        proof = source.get(key) if isinstance(source, Mapping) else {}
        if isinstance(proof, Mapping) and proof:
            return proof
    return {}


def _review_marker_gate_with_current_context(
    row: Mapping[str, Any],
    marker: float,
    current_payload: Mapping[str, Any],
    *,
    cache_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    boom_candidates = current_payload.get("boom_candidates")
    if not isinstance(boom_candidates, list) or not any(isinstance(item, Mapping) for item in boom_candidates):
        return None
    feature_map = current_payload.get("feature_map") if isinstance(current_payload.get("feature_map"), Mapping) else {}
    beatgrid = feature_map.get("beatgrid") if isinstance(feature_map.get("beatgrid"), Mapping) else None
    profile = feature_map.get("boom_profile") if isinstance(feature_map.get("boom_profile"), Mapping) else None
    reasons: List[str] = []
    boom = marker_boom_proof(
        float(marker),
        [item for item in boom_candidates if isinstance(item, Mapping)],
        profile=profile,
        beatgrid=beatgrid,
    )
    freshness = boom_proof_front_edge_freshness(boom)
    if not bool(boom.get("passes")):
        reasons.append("boom_proof_hold:" + ";".join(str(reason) for reason in boom.get("reasons") or [] if str(reason)))
    elif not bool(freshness.get("fresh")):
        reasons.append(f"boom_proof_stale:{freshness.get('reason') or 'unknown'}")

    gui = {"passes": False, "reasons": ["missing_cache_dir"]}
    drums_path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    if cache_dir is not None and drums_path:
        try:
            gui = gui_boom_mask_proof(drums_path, float(marker), cache_dir=cache_dir)
            gui = accept_gui_boom_mask_with_front_edge_proof(gui, boom)
        except Exception as exc:
            gui = {"passes": False, "reasons": [f"gui_error:{str(exc) or exc.__class__.__name__}"]}
    if not bool(gui.get("passes")):
        reasons.append("gui_mask_hold:" + ";".join(str(reason) for reason in gui.get("reasons") or [] if str(reason)))
    return {
        "passes": bool(boom.get("passes")) and bool(freshness.get("fresh")) and bool(gui.get("passes")),
        "reasons": reasons,
        "boom_passes": bool(boom.get("passes")),
        "boom_fresh": bool(freshness.get("fresh")),
        "gui_passes": bool(gui.get("passes")),
        "boom_proof": boom,
        "gui_proof": gui,
        "source": "current_detector_context",
    }


def _clock(selected: Mapping[str, Any]) -> Mapping[str, Any]:
    clock = selected.get("bpm_clock") if isinstance(selected.get("bpm_clock"), Mapping) else {}
    return clock if isinstance(clock, Mapping) else {}


def _status_for_delta(
    review: ReviewMarker,
    current_marker: Optional[float],
    current_payload: Mapping[str, Any],
    selected: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    manual_tolerance_sec: float,
    blue_tolerance_sec: float,
    gate_blue_markers: bool,
    cache_dir: Optional[Path],
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    if current_marker is None:
        return "missing_current_marker", float("nan"), None
    delta = float(current_marker) - float(review.user_pick)
    tolerance = float(manual_tolerance_sec if review.strength == "manual" else blue_tolerance_sec)
    if abs(delta) <= tolerance:
        return "pass", delta, None
    if review.strength == "manual":
        gate = _review_marker_gate_with_current_context(
            row,
            review.user_pick,
            current_payload,
            cache_dir=cache_dir,
        )
        if gate is None:
            gate = review_marker_gate(row, review.user_pick, cache_dir=cache_dir)
        if bool(gate.get("passes")):
            return "validated_hard_mismatch", delta, gate
        return "stale_manual_mismatch", delta, gate
    if gate_blue_markers:
        gate = _review_marker_gate_with_current_context(
            row,
            review.user_pick,
            current_payload,
            cache_dir=cache_dir,
        )
        if gate is None:
            gate = review_marker_gate(row, review.user_pick, cache_dir=cache_dir)
        if bool(gate.get("passes")):
            return "validated_blue_mismatch", delta, gate
        return "stale_blue_mismatch", delta, gate
    return "advisory_mismatch", delta, None


def _run_detector(audio_path: str, *, sample_rate: int, use_cache: bool) -> Mapping[str, Any]:
    from drop_aligner.visual_first import visual_first_marker

    result = visual_first_marker(audio_path, sample_rate=int(sample_rate), use_cache=bool(use_cache))
    return result if isinstance(result, Mapping) else {"ok": False, "error": "non_mapping_detector_result"}


def _audit_one(
    index: int,
    row: Mapping[str, Any],
    review: ReviewMarker,
    *,
    sample_rate: int,
    use_cache: bool,
    manual_tolerance_sec: float,
    blue_tolerance_sec: float,
    gate_blue_markers: bool,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    started = time.time()
    drums_path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    audit_key = _row_key(index, row)
    current_payload: Mapping[str, Any]
    if not drums_path:
        current_payload = {"ok": False, "error": "missing_drums_path"}
    else:
        try:
            current_payload = _run_detector(drums_path, sample_rate=sample_rate, use_cache=use_cache)
        except Exception as exc:
            current_payload = {"ok": False, "error": str(exc) or exc.__class__.__name__}
    selected = _selected_candidate(current_payload)
    current_marker = _marker_from_payload(current_payload, selected)
    status, delta, review_gate = _status_for_delta(
        review,
        current_marker,
        current_payload,
        selected,
        row,
        manual_tolerance_sec=manual_tolerance_sec,
        blue_tolerance_sec=blue_tolerance_sec,
        gate_blue_markers=gate_blue_markers,
        cache_dir=cache_dir,
    )
    clock = _clock(selected)
    boom = _proof_dict(current_payload, selected, "boom_proof")
    gui = _proof_dict(current_payload, selected, "gui_mask_proof")
    gui_issue = gui_boom_mask_strict_contract_issue(gui)
    detector_ok = bool(current_payload.get("ok"))
    proof_ready = bool(
        detector_ok
        and bool(boom.get("passes"))
        and bool(gui.get("passes"))
        and not str(gui_issue or "")
        and bool(clock.get("on_one"))
    )
    if not detector_ok:
        status = "detector_error"
    elif not proof_ready and status == "pass":
        status = "proof_hold"
    return {
        "audit_key": audit_key,
        "index": int(index),
        "status": status,
        "strength": review.strength,
        "delta_sec": "" if not math.isfinite(delta) else float(delta),
        "abs_delta_sec": "" if not math.isfinite(delta) else abs(float(delta)),
        "current_marker": "" if current_marker is None else float(current_marker),
        "review_marker": float(review.user_pick),
        "current_selected_by": str(selected.get("selected_by") or current_payload.get("selected_by") or ""),
        "current_audit_status": str((current_payload.get("visual_audit") or {}).get("status") or "")
        if isinstance(current_payload.get("visual_audit"), Mapping)
        else "",
        "current_audit_flags": ";".join(
            str(flag)
            for flag in ((current_payload.get("visual_audit") or {}).get("flag_codes") or [])
            if str(flag)
        )
        if isinstance(current_payload.get("visual_audit"), Mapping)
        else "",
        "detector_ok": bool(detector_ok),
        "detector_error": str(current_payload.get("error") or ""),
        "proof_ready": bool(proof_ready),
        "current_boom_proof_pass": bool(boom.get("passes")),
        "current_boom_proof_reasons": ";".join(str(reason) for reason in boom.get("reasons") or [] if str(reason)),
        "current_gui_mask_proof_pass": bool(gui.get("passes")),
        "current_gui_mask_proof_reasons": ";".join(str(reason) for reason in gui.get("reasons") or [] if str(reason)),
        "current_gui_strict_contract_issue": str(gui_issue or ""),
        "current_on_one": bool(clock.get("on_one")) if clock else False,
        "current_one_distance_ms": _finite_float(clock.get("one_distance_ms")) if clock else "",
        "current_clock_source": str(clock.get("source") or "") if clock else "",
        "review_marker_gate_passes": "" if review_gate is None else bool(review_gate.get("passes")),
        "review_marker_gate_reasons": ""
        if review_gate is None
        else ";".join(str(reason) for reason in review_gate.get("reasons") or [] if str(reason)),
        "review_marker_boom_passes": "" if review_gate is None else bool(review_gate.get("boom_passes")),
        "review_marker_gui_passes": "" if review_gate is None else bool(review_gate.get("gui_passes")),
        "reviewed_from": review.reviewed_from,
        "review_timestamp": review.timestamp,
        "review_selected_by": review.selected_by,
        "track": str((row.get("track") or {}).get("folder") if isinstance(row.get("track"), Mapping) else row.get("track") or ""),
        "drums_path": drums_path,
        "review_track": review.track,
        "review_source_path": review.source_path,
        "review_line_number": int(review.line_number),
        "elapsed_sec": time.time() - started,
    }


def _timeout_result(
    index: int,
    row: Mapping[str, Any],
    review: ReviewMarker,
    timeout_sec: float,
    elapsed_sec: float,
) -> Dict[str, Any]:
    drums_path = str(row.get("drums_path") or row.get("audio_path") or row.get("filename") or "")
    return {
        "audit_key": _row_key(index, row),
        "index": int(index),
        "status": "timeout",
        "strength": review.strength,
        "delta_sec": "",
        "abs_delta_sec": "",
        "current_marker": "",
        "review_marker": float(review.user_pick),
        "current_selected_by": "",
        "current_audit_status": "",
        "current_audit_flags": "",
        "detector_ok": False,
        "detector_error": f"detector_timeout:{float(timeout_sec):.3f}s",
        "proof_ready": False,
        "current_boom_proof_pass": False,
        "current_boom_proof_reasons": "",
        "current_gui_mask_proof_pass": False,
        "current_gui_mask_proof_reasons": "",
        "current_gui_strict_contract_issue": "",
        "current_on_one": False,
        "current_one_distance_ms": "",
        "current_clock_source": "",
        "review_marker_gate_passes": "",
        "review_marker_gate_reasons": "",
        "review_marker_boom_passes": "",
        "review_marker_gui_passes": "",
        "reviewed_from": review.reviewed_from,
        "review_timestamp": review.timestamp,
        "review_selected_by": review.selected_by,
        "track": str((row.get("track") or {}).get("folder") if isinstance(row.get("track"), Mapping) else row.get("track") or ""),
        "drums_path": drums_path,
        "review_track": review.track,
        "review_source_path": review.source_path,
        "review_line_number": int(review.line_number),
        "elapsed_sec": float(elapsed_sec),
    }


def _audit_one_child(task: Mapping[str, Any], queue: Any) -> None:
    try:
        queue.put(
            _audit_one(
                int(task["index"]),
                task["row"],
                task["review"],
                sample_rate=int(task["sample_rate"]),
                use_cache=bool(task["use_cache"]),
                manual_tolerance_sec=float(task["manual_tolerance_sec"]),
                blue_tolerance_sec=float(task["blue_tolerance_sec"]),
                gate_blue_markers=bool(task["gate_blue_markers"]),
                cache_dir=Path(task["cache_dir"]).expanduser() if task.get("cache_dir") else None,
            )
        )
    except BaseException as exc:  # pragma: no cover - defensive child-process boundary.
        queue.put(
            {
                **_timeout_result(int(task["index"]), task["row"], task["review"], 0.0, 0.0),
                "status": "detector_error",
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
        return _timeout_result(int(task["index"]), task["row"], task["review"], timeout_sec, elapsed)
    try:
        row = queue.get_nowait()
    except Exception:
        return {
            **_timeout_result(int(task["index"]), task["row"], task["review"], timeout_sec, elapsed),
            "status": "detector_error",
            "detector_error": "detector_worker_empty_result",
        }
    return row if isinstance(row, dict) else {
        **_timeout_result(int(task["index"]), task["row"], task["review"], timeout_sec, elapsed),
        "status": "detector_error",
        "detector_error": "detector_worker_non_mapping_result",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "audit_key",
        "index",
        "status",
        "strength",
        "delta_sec",
        "abs_delta_sec",
        "current_marker",
        "review_marker",
        "current_selected_by",
        "current_audit_status",
        "current_audit_flags",
        "detector_ok",
        "detector_error",
        "proof_ready",
        "current_boom_proof_pass",
        "current_boom_proof_reasons",
        "current_gui_mask_proof_pass",
        "current_gui_mask_proof_reasons",
        "current_gui_strict_contract_issue",
        "current_on_one",
        "current_one_distance_ms",
        "current_clock_source",
        "review_marker_gate_passes",
        "review_marker_gate_reasons",
        "review_marker_boom_passes",
        "review_marker_gui_passes",
        "reviewed_from",
        "review_timestamp",
        "review_selected_by",
        "track",
        "drums_path",
        "review_track",
        "review_source_path",
        "review_line_number",
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
    report_path: Path,
    correction_logs: Sequence[Path],
    total_report_rows: int,
    selected_rows: int,
    matched_review_rows: int,
    elapsed_sec: float,
    jsonl_path: Path,
    csv_path: Path,
    failures_csv: Path,
) -> Dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    selected_by_counts = Counter(str(row.get("current_selected_by") or "") for row in rows)
    proof_ready_count = sum(1 for row in rows if bool(row.get("proof_ready")))
    manual_count = sum(1 for row in rows if row.get("strength") == "manual")
    blue_count = sum(1 for row in rows if row.get("strength") == "blue_approval")
    hard_failure_statuses = {"validated_hard_mismatch", "validated_blue_mismatch", "proof_hold", "detector_error", "timeout"}
    hard_failures = [row for row in rows if str(row.get("status") or "") in hard_failure_statuses]
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(report_path),
        "correction_logs": [str(path) for path in correction_logs],
        "total_report_rows": int(total_report_rows),
        "selected_row_count": int(selected_rows),
        "matched_review_rows": int(matched_review_rows),
        "completed_count": len(rows),
        "proof_ready_count": int(proof_ready_count),
        "manual_review_rows": int(manual_count),
        "blue_approval_rows": int(blue_count),
        "pass_count": int(status_counts.get("pass", 0)),
        "validated_hard_mismatch_count": int(status_counts.get("validated_hard_mismatch", 0)),
        "validated_blue_mismatch_count": int(status_counts.get("validated_blue_mismatch", 0)),
        "stale_manual_mismatch_count": int(status_counts.get("stale_manual_mismatch", 0)),
        "stale_blue_mismatch_count": int(status_counts.get("stale_blue_mismatch", 0)),
        "advisory_mismatch_count": int(status_counts.get("advisory_mismatch", 0)),
        "proof_hold_count": int(status_counts.get("proof_hold", 0)),
        "detector_error_count": int(status_counts.get("detector_error", 0)),
        "timeout_count": int(status_counts.get("timeout", 0)),
        "all_completed": len(rows) == int(matched_review_rows) and int(matched_review_rows) > 0,
        "all_pass_or_stale_or_advisory": len(hard_failures) == 0,
        "status_counts": dict(status_counts),
        "selected_by_counts": dict(selected_by_counts),
        "hard_failure_samples": hard_failures[:25],
        "output_jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "failures_csv": str(failures_csv),
        "elapsed_sec": float(elapsed_sec),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    report_path = Path(args.report).expanduser().resolve() if args.report else _latest_report().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    correction_logs = (
        [Path(path).expanduser() for path in args.correction_log]
        if args.correction_log
        else _default_correction_logs(report_path, out_dir)
    )
    rows = _load_report_rows(report_path)
    review_markers = load_review_markers(correction_logs)
    exact, stem, slug = build_review_indexes(review_markers)
    indices = _parse_indices(args.indices)
    indexed_rows = _select_indexed_rows(rows, indices=indices, offset=int(args.offset), limit=int(args.limit))
    review_rows: List[Tuple[int, Dict[str, Any], ReviewMarker]] = []
    for index, row in indexed_rows:
        review = find_review_for_row(row, exact, stem, slug)
        if review is not None:
            review_rows.append((index, row, review))
    jsonl_path = out_dir / f"{report_path.stem}_current_vs_human_review.jsonl"
    if bool(args.force) and jsonl_path.exists():
        jsonl_path.unlink()
    selected_keys = {_row_key(index, row) for index, row, _review in review_rows}
    completed = _completed_rows(jsonl_path) if bool(args.resume) else {}
    results_by_key = {key: row for key, row in completed.items() if key in selected_keys}
    remaining = [
        (index, row, review)
        for index, row, review in review_rows
        if _row_key(index, row) not in results_by_key
    ]
    per_row_timeout_sec = float(args.per_row_timeout_sec or 0.0)

    def run_task(task: Mapping[str, Any]) -> Dict[str, Any]:
        if per_row_timeout_sec > 0.0:
            return _audit_one_with_timeout(task, per_row_timeout_sec)
        return _audit_one(
            int(task["index"]),
            task["row"],
            task["review"],
            sample_rate=int(task["sample_rate"]),
            use_cache=bool(task["use_cache"]),
            manual_tolerance_sec=float(task["manual_tolerance_sec"]),
            blue_tolerance_sec=float(task["blue_tolerance_sec"]),
            gate_blue_markers=bool(task["gate_blue_markers"]),
            cache_dir=Path(task["cache_dir"]).expanduser() if task.get("cache_dir") else None,
        )

    tasks: List[Dict[str, Any]] = []
    for index, row, review in remaining:
        tasks.append({
            "index": index,
            "row": row,
            "review": review,
            "sample_rate": int(args.sample_rate),
            "use_cache": bool(args.use_cache),
            "manual_tolerance_sec": float(args.manual_tolerance_sec),
            "blue_tolerance_sec": float(args.blue_tolerance_sec),
            "gate_blue_markers": bool(args.gate_blue_markers),
            "cache_dir": str(Path(args.cache_dir).expanduser()) if args.cache_dir else "",
        })

    jobs = max(1, int(args.jobs or 1))
    if jobs == 1 or len(tasks) <= 1:
        for count, task in enumerate(tasks, 1):
            result = run_task(task)
            results_by_key[str(result["audit_key"])] = result
            _write_jsonl_row(jsonl_path, result)
            done = len(results_by_key)
            if args.progress_every and (count % int(args.progress_every) == 0 or done == len(review_rows)):
                print(f"audited {done}/{len(review_rows)} current-vs-review rows", flush=True)
            if args.max_runtime_sec and (time.time() - started) >= float(args.max_runtime_sec):
                print("stopping after --max-runtime-sec", flush=True)
                break
    else:
        completed_this_run = 0
        with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed_this_run += 1
                results_by_key[str(result["audit_key"])] = result
                _write_jsonl_row(jsonl_path, result)
                done = len(results_by_key)
                if args.progress_every and (
                    completed_this_run % int(args.progress_every) == 0 or done == len(review_rows)
                ):
                    print(f"audited {done}/{len(review_rows)} current-vs-review rows", flush=True)
                if args.max_runtime_sec and (time.time() - started) >= float(args.max_runtime_sec):
                    print("stopping after --max-runtime-sec", flush=True)
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
    results = [
        results_by_key[key]
        for key in (_row_key(index, row) for index, row, _review in review_rows)
        if key in results_by_key
    ]
    csv_path = out_dir / f"{report_path.stem}_current_vs_human_review.csv"
    failures_csv = out_dir / f"{report_path.stem}_current_vs_human_review_failures.csv"
    failure_rows = [
        row
        for row in results
        if str(row.get("status") or "")
        in {"validated_hard_mismatch", "validated_blue_mismatch", "proof_hold", "detector_error", "timeout"}
    ]
    _write_csv(csv_path, results)
    _write_csv(failures_csv, failure_rows)
    summary = _summarize(
        results,
        report_path=report_path,
        correction_logs=correction_logs,
        total_report_rows=len(rows),
        selected_rows=len(indexed_rows),
        matched_review_rows=len(review_rows),
        elapsed_sec=time.time() - started,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        failures_csv=failures_csv,
    )
    summary_path = out_dir / f"{report_path.stem}_current_vs_human_review.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")
    print(json.dumps({**summary, "json": str(summary_path)}, indent=2, ensure_ascii=True, default=_json_default))
    if bool(args.require_no_hard_failures) and not bool(summary.get("all_pass_or_stale_or_advisory")):
        return {**summary, "json": str(summary_path), "exit_code": 1}
    return {**summary, "json": str(summary_path), "exit_code": 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit today's visual_first_marker output directly against human review/correction markers.",
    )
    parser.add_argument("report", nargs="?", default="", help="Visual-first report JSON; defaults to latest fresh report.")
    parser.add_argument("--correction-log", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--indices", default="", help="1-based report row indices, e.g. 696,760,809.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manual-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--blue-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--gate-blue-markers", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-runtime-sec", type=float, default=0.0)
    parser.add_argument("--per-row-timeout-sec", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of row audits to run concurrently.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-no-hard-failures", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = audit(build_parser().parse_args(argv))
    return int(result.get("exit_code", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
