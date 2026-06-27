#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from drop_aligner.als import modify_als
from drop_aligner.candidate_chooser import candidate_effective_time
from drop_aligner.exclusions import row_has_excluded_path
from drop_aligner.legacy_write_guard import add_legacy_detector_write_arg, require_legacy_detector_write_opt_in
from drop_aligner.multistem import choose_multistem_candidate
from verify_als import verify_als
from project_config import DEFAULT_ALS_TEMPLATE, DROP_BATCH_SUMMARY


DEFAULT_SUMMARY = DROP_BATCH_SUMMARY
DEFAULT_TEMPLATE = DEFAULT_ALS_TEMPLATE
EXTRA_COLUMNS = ["micro_confidence", "snap_offset_ms", "microaligned_time"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    tmp.replace(path)


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


def _fieldnames(existing: Sequence[str]) -> List[str]:
    out = list(existing)
    for name in EXTRA_COLUMNS:
        if name not in out:
            out.append(name)
    return out


def _stable_id(row: Mapping[str, str]) -> str:
    raw = "|".join([row.get("filename", ""), row.get("output_als", ""), row.get("candidates_json", "")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_reviewed_ids(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        state = _read_json(state_path)
    except Exception:
        return set()
    items = state.get("items")
    if not isinstance(items, Mapping):
        return set()
    return {
        str(item_id)
        for item_id, item_state in items.items()
        if isinstance(item_state, Mapping) and bool(item_state.get("reviewed") or item_state.get("skipped"))
    }


def _iter_target_rows(
    rows: Sequence[Mapping[str, str]],
    reviewed_ids: set[str],
    *,
    remaining_only: bool,
) -> Iterable[Tuple[int, Mapping[str, str]]]:
    for index, row in enumerate(rows):
        if row_has_excluded_path(row):
            continue
        if remaining_only and _stable_id(row) in reviewed_ids:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status and status not in {"processed", "skipped", "partial"}:
            continue
        als_valid = str(row.get("als_valid", "")).strip().lower()
        if als_valid and als_valid != "true":
            continue
        yield index, row


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _format_float(value: Any, digits: int = 9) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}".rstrip("0").rstrip(".")


def _tier(row: Mapping[str, str], payload: Mapping[str, Any]) -> str:
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    value = row.get("confidence_tier") or payload.get("confidence_tier") or features.get("confidence_tier") or "UNKNOWN"
    text = str(value).strip().upper()
    return text if text in {"HIGH", "MEDIUM", "LOW"} else "UNKNOWN"


def _backup_file(path: Path, backup_dir: Path, manifest: List[Dict[str, str]]) -> None:
    if not path.exists():
        return
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    backup = backup_dir / f"{digest}{path.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    manifest.append({"path": str(path), "backup": str(backup)})


def _selected_first(candidates: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]) -> List[Dict[str, Any]]:
    selected_time = candidate_effective_time(selected)
    out: List[Dict[str, Any]] = []
    if selected:
        out.append(dict(selected))
    for candidate in candidates:
        candidate_time = candidate_effective_time(candidate)
        if selected_time is not None and candidate_time is not None and abs(float(candidate_time) - float(selected_time)) <= 0.010:
            continue
        item = dict(candidate)
        item["selected"] = False
        out.append(item)
        if len(out) >= 10:
            break
    for rank, candidate in enumerate(out, start=1):
        candidate["rank"] = int(rank)
    return out


def _promote_candidate(candidate: Mapping[str, Any], suggested_time: float, micro: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(candidate)
    original = candidate_effective_time(out)
    out["pre_multistem_timestamp"] = None if original is None else float(original)
    out["timestamp"] = float(suggested_time)
    out["snapped_sec"] = float(suggested_time)
    out["microaligned_time"] = float(suggested_time)
    out["microalign"] = dict(micro)
    out["micro_confidence"] = float(micro.get("micro_confidence", 0.0) or 0.0)
    out["snap_offset_ms"] = float(micro.get("snap_offset_ms", 0.0) or 0.0)
    out["selected"] = True
    out["selected_by"] = str(out.get("selected_by") or "multistem_candidate")
    out["auto_place_selected_by"] = "multistem_auto_place"
    out["reason"] = "selected_by_multistem_hard_case_agent"
    return out


def _update_feature_summary(payload: Dict[str, Any], selected: Mapping[str, Any], suggested_time: float, mode: str) -> None:
    features = payload.get("feature_summary")
    if not isinstance(features, dict):
        features = {}
        payload["feature_summary"] = features
    features["selected_by"] = "multistem_auto_place"
    features["chosen_microaligned_time"] = float(suggested_time)
    features["chosen_micro_confidence"] = float(selected.get("micro_confidence", 0.0) or 0.0)
    features["chosen_snap_offset_ms"] = float(selected.get("snap_offset_ms", 0.0) or 0.0)
    features["multistem_reanalysis_mode"] = str(mode)


def _shadow_row(row: Mapping[str, str], status: str, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    reanalysis = payload.get("multistem_reanalysis") if isinstance(payload, Mapping) and isinstance(payload.get("multistem_reanalysis"), Mapping) else {}
    gate_map = reanalysis.get("auto_accept") if isinstance(reanalysis.get("auto_accept"), Mapping) else {}
    gate = gate_map.get(str(reanalysis.get("mode") or "normal")) if isinstance(gate_map.get(str(reanalysis.get("mode") or "normal")), Mapping) else {}
    chooser = reanalysis.get("candidate_chooser") if isinstance(reanalysis.get("candidate_chooser"), Mapping) else {}
    verifier = gate.get("auto_verifier") if isinstance(gate.get("auto_verifier"), Mapping) else {}
    return {
        "status": str(status),
        "filename": str(row.get("filename", "")),
        "candidates_json": str(row.get("candidates_json", "")),
        "suggested_time": reanalysis.get("suggested_time"),
        "diagnostic_reason": str(reanalysis.get("diagnostic_reason", "")),
        "suggestion_reason": str(reanalysis.get("reason", "")),
        "stem_roles": sorted((reanalysis.get("stem_group") or {}).get("roles", {}).keys())
        if isinstance(reanalysis.get("stem_group"), Mapping) and isinstance((reanalysis.get("stem_group") or {}).get("roles"), Mapping)
        else [],
        "auto_accept_passed": bool(reanalysis.get("auto_accept_passed", False)),
        "gate_reason": str(gate.get("reason", "")),
        "risk_flags": list(gate.get("risk_flags") or []) if isinstance(gate.get("risk_flags"), list) else [],
        "source_count": int(reanalysis.get("source_count", 0) or 0),
        "candidate_count": int(reanalysis.get("candidate_count", 0) or 0),
        "pipeline_output_count": int((reanalysis.get("pipeline") or {}).get("output_count", 0) or 0)
        if isinstance(reanalysis.get("pipeline"), Mapping)
        else 0,
        "pipeline_deduped_count": int((reanalysis.get("pipeline") or {}).get("deduped_count", 0) or 0)
        if isinstance(reanalysis.get("pipeline"), Mapping)
        else 0,
        "model_path": str(chooser.get("model_path", "")),
        "model_type": str(chooser.get("model_type", "")),
        "selection_probability": chooser.get("selection_probability"),
        "probability_margin": chooser.get("probability_margin"),
        "selection_confidence": chooser.get("selection_confidence"),
        "predicted_abs_error_sec": chooser.get("predicted_abs_error_sec"),
        "micro_confidence": gate.get("micro_confidence"),
        "snap_offset_ms": gate.get("snap_offset_ms"),
        "auto_verifier_p_within_25ms": verifier.get("p_within_25ms"),
        "auto_verifier_predicted_abs_error_sec": verifier.get("predicted_abs_error_sec"),
        "auto_verifier_mode": verifier.get("mode"),
    }


def _diagnose_reanalysis(result: Mapping[str, Any], suggestion: Mapping[str, Any]) -> str:
    source_count = int(result.get("source_count", 0) or 0)
    candidate_count = int(result.get("candidate_count", 0) or 0)
    pipeline = result.get("pipeline") if isinstance(result.get("pipeline"), Mapping) else {}
    output_count = int(pipeline.get("output_count", 0) or 0)
    deduped_count = int(pipeline.get("deduped_count", 0) or 0)
    if source_count <= 0:
        return "no candidate sources generated"
    if candidate_count <= 0:
        return f"{source_count} sources generated, but no candidates survived scoring"
    if output_count <= 0:
        return f"{candidate_count} candidates generated, but the drop pipeline returned no ranked candidates"
    if not isinstance(suggestion.get("candidate"), Mapping):
        return str(suggestion.get("reason") or f"{output_count} ranked candidates, but no suggestion candidate")
    if suggestion.get("suggested_time") in (None, ""):
        return str(suggestion.get("reason") or "suggestion candidate had no suggested time")
    return (
        f"{source_count} sources, {candidate_count} generated candidates, "
        f"{output_count} ranked candidates, {deduped_count} deduped"
    )


def _store_reanalysis_payload(
    payload: Dict[str, Any],
    *,
    mode: str,
    result: Mapping[str, Any],
    suggestion: Mapping[str, Any],
    suggested_time: Optional[float],
) -> None:
    gate_map = suggestion.get("auto_accept") if isinstance(suggestion.get("auto_accept"), Mapping) else {}
    mode_gate = gate_map.get(mode) if isinstance(gate_map.get(mode), Mapping) else {}
    payload["multistem_reanalysis"] = {
        "reanalyzed_at": _now_iso(),
        "mode": str(mode),
        "stem_group": result.get("stem_group"),
        "source_count": int(result.get("source_count", 0) or 0),
        "candidate_count": int(result.get("candidate_count", 0) or 0),
        "pipeline": result.get("pipeline") if isinstance(result.get("pipeline"), Mapping) else {},
        "suggested_time": None if suggested_time is None else float(suggested_time),
        "auto_accept": gate_map,
        "auto_accept_passed": bool(mode_gate.get("auto_accept", False)),
        "candidate_chooser": suggestion.get("candidate_chooser") if isinstance(suggestion.get("candidate_chooser"), Mapping) else None,
        "reason": str(suggestion.get("reason", "")),
        "diagnostic_reason": _diagnose_reanalysis(result, suggestion),
    }


def _regenerate_als(row: Mapping[str, str], payload: Mapping[str, Any], template: Path, marker_time: float) -> Tuple[bool, str]:
    output = Path(str(row.get("output_als", ""))).expanduser()
    if not output:
        return False, "missing output ALS path"
    if output.exists():
        previous = output.with_name(output.stem + ".previous" + output.suffix)
        shutil.copy2(output, previous)
    features = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {}
    bpm = _float_or_none(payload.get("bpm")) or _float_or_none(features.get("bpm")) or 128.0
    modify_als(
        template_path=str(template),
        audio_path=str(row.get("filename", "")),
        drop_sec=float(marker_time),
        bpm=float(bpm),
        output_path=str(output),
        strict_stems=True,
    )
    candidates_path = Path(str(row.get("candidates_json", ""))).expanduser()
    verify_payload = candidates_path.with_suffix(candidates_path.suffix + ".multistem_verify")
    try:
        _write_json(verify_payload, payload)
        report = verify_als(str(output), candidates_json=str(verify_payload))
    finally:
        verify_payload.unlink(missing_ok=True)
    if not report.get("valid"):
        return False, "; ".join(str(err) for err in report.get("errors", []))
    return True, ""


def _apply_one(
    row: Mapping[str, str],
    *,
    template: Path,
    mode: str,
    apply_auto: bool,
    regenerate_als: bool,
    force: bool,
    update_review_candidates: bool,
    expanded_limit: int,
    microalign_limit: int,
    sample_rate: int,
) -> Tuple[Dict[str, str], Optional[Dict[str, Any]], str]:
    updated_row = dict(row)
    audio_path = str(row.get("filename", ""))
    candidates_path = Path(str(row.get("candidates_json", ""))).expanduser()
    if not audio_path or not candidates_path.exists():
        return updated_row, None, "missing_candidates_json"
    payload = _read_json(candidates_path)
    if not force and isinstance(payload.get("multistem_reanalysis"), Mapping):
        return updated_row, None, "already_reanalyzed"
    saved_candidates = payload.get("top_10_candidates") if isinstance(payload.get("top_10_candidates"), list) else []
    try:
        result = choose_multistem_candidate(
            audio_path,
            saved_candidates=saved_candidates,
            confidence_tier=_tier(row, payload),
            mode=mode,
            expanded_limit=int(expanded_limit),
            microalign_limit=int(microalign_limit),
            sample_rate=int(sample_rate),
        )
    except Exception as exc:
        return updated_row, None, f"reanalyze_failed:{str(exc) or exc.__class__.__name__}"

    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
    suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), Mapping) else {}
    candidate = suggestion.get("candidate") if isinstance(suggestion.get("candidate"), Mapping) else {}
    micro = suggestion.get("microalign") if isinstance(suggestion.get("microalign"), Mapping) else {}
    suggested_time = _float_or_none(suggestion.get("suggested_time")) or _float_or_none(micro.get("microaligned_time"))
    gate_map = suggestion.get("auto_accept") if isinstance(suggestion.get("auto_accept"), Mapping) else {}
    mode_gate = gate_map.get(mode) if isinstance(gate_map.get(mode), Mapping) else {}
    auto_pass = bool(mode_gate.get("auto_accept"))

    _store_reanalysis_payload(
        payload,
        mode=mode,
        result=result,
        suggestion=suggestion,
        suggested_time=suggested_time,
    )
    if suggested_time is None or not candidate:
        return updated_row, payload, "no_multistem_suggestion"

    selected = _promote_candidate(candidate, float(suggested_time), micro)

    wrote_payload = False
    if update_review_candidates or apply_auto:
        payload["top_10_candidates"] = _selected_first(candidates, selected)
        wrote_payload = True

    if apply_auto:
        if not auto_pass:
            return updated_row, payload if wrote_payload else None, "review_gate_failed"
        payload["final_ai_pick"] = float(suggested_time)
        payload["drop_sec"] = float(suggested_time)
        payload["selected_by"] = "multistem_auto_place"
        payload["selected_candidate"] = selected
        _update_feature_summary(payload, selected, float(suggested_time), mode)
        updated_row["detected_drop_time"] = _format_float(suggested_time)
        updated_row["selected_by"] = "multistem_auto_place"
        updated_row["microaligned_time"] = _format_float(suggested_time)
        updated_row["micro_confidence"] = _format_float(selected.get("micro_confidence"), digits=6)
        updated_row["snap_offset_ms"] = _format_float(selected.get("snap_offset_ms"), digits=6)
        if regenerate_als:
            ok, error = _regenerate_als(row, payload, template, float(suggested_time))
            updated_row["als_valid"] = "true" if ok else "false"
            updated_row["als_validation_error"] = "" if ok else error
            if not ok:
                return updated_row, payload, "als_validation_failed"
        return updated_row, payload, "auto_updated"

    return updated_row, payload if wrote_payload else None, "review_candidates_updated" if wrote_payload else "analyzed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reanalyze remaining review tracks with multi-stem evidence.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="drop_batch_summary.csv path")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="ALS template used for auto-updates")
    parser.add_argument("--state", help="review_state.json path. Defaults beside summary")
    parser.add_argument("--mode", choices=["conservative", "normal", "aggressive"], default="normal")
    parser.add_argument("--all", dest="remaining_only", action="store_false", help="Include reviewed/skipped rows too")
    parser.set_defaults(remaining_only=True)
    parser.add_argument("--apply-auto", action="store_true", help="Promote only suggestions that pass the selected auto gate")
    parser.add_argument("--no-regenerate-als", dest="regenerate_als", action="store_false", help="Do not regenerate ALS files")
    parser.set_defaults(regenerate_als=True)
    parser.add_argument("--no-update-review-candidates", dest="update_review_candidates", action="store_false", help="Do not rewrite top candidates for review-only rows")
    parser.set_defaults(update_review_candidates=True)
    parser.add_argument("--limit", type=int, default=0, help="Only process this many target rows")
    parser.add_argument("--expanded-limit", type=int, default=120, help="Expanded candidates retained per track")
    parser.add_argument("--microalign-limit", type=int, default=50, help="Top expanded candidates to microalign")
    parser.add_argument("--analysis-sr", type=int, default=16000, help="Sample rate for stem analysis")
    parser.add_argument("--force", action="store_true", help="Reanalyze even when multistem_reanalysis already exists")
    parser.add_argument("--dry-run", action="store_true", help="Analyze but do not write CSV/JSON/ALS")
    parser.add_argument("--shadow-report", help="Write a JSON report of every proposed autonomous decision")
    add_legacy_detector_write_arg(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = Path(args.summary).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    state = Path(args.state).expanduser().resolve() if args.state else summary.parent / "review_state.json"
    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}")
    if bool(args.apply_auto) and bool(args.regenerate_als) and not template.exists():
        raise SystemExit(f"Template not found: {template}")
    if not bool(args.dry_run):
        require_legacy_detector_write_opt_in(
            "reanalyze_remaining_hard_cases.py",
            action="rewriting legacy multistem summary/candidate JSON/ALS output",
            explicit=bool(args.allow_legacy_detector_write),
        )

    rows, fieldnames = _read_summary(summary)
    output_fields = _fieldnames(fieldnames)
    reviewed_ids = _load_reviewed_ids(state)
    targets = list(_iter_target_rows(rows, reviewed_ids, remaining_only=bool(args.remaining_only)))
    if int(args.limit) > 0:
        targets = targets[: int(args.limit)]

    stamp = _now_stamp()
    backup_dir = summary.parent / ".multistem_reanalysis_backups" / stamp
    manifest: List[Dict[str, str]] = []
    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        _backup_file(summary, backup_dir, manifest)

    counts: Dict[str, int] = {}
    shadow_rows: List[Dict[str, Any]] = []
    for number, (row_index, row) in enumerate(targets, start=1):
        updated_row, updated_payload, status = _apply_one(
            row,
            template=template,
            mode=str(args.mode),
            apply_auto=bool(args.apply_auto),
            regenerate_als=bool(args.regenerate_als) and not bool(args.dry_run),
            force=bool(args.force),
            update_review_candidates=bool(args.update_review_candidates),
            expanded_limit=int(args.expanded_limit),
            microalign_limit=int(args.microalign_limit),
            sample_rate=int(args.analysis_sr),
        )
        counts[status] = counts.get(status, 0) + 1
        shadow_rows.append(_shadow_row(row, status, updated_payload))
        if updated_payload is not None and not args.dry_run:
            candidates_path = Path(str(row.get("candidates_json", ""))).expanduser()
            _backup_file(candidates_path, backup_dir, manifest)
            _write_json(candidates_path, updated_payload)
            rows[row_index] = updated_row
        print(f"[{number}/{len(targets)}] {status}: {row.get('filename', '')}", flush=True)

    if not args.dry_run:
        _write_summary(summary, rows, output_fields)
        _write_json(backup_dir / "manifest.json", {"created_at": _now_iso(), "files": manifest})
    if args.shadow_report:
        shadow_path = Path(args.shadow_report).expanduser().resolve()
        shadow_payload = {
            "created_at": _now_iso(),
            "summary": str(summary),
            "state": str(state),
            "targets": int(len(targets)),
            "remaining_only": bool(args.remaining_only),
            "apply_auto": bool(args.apply_auto),
            "dry_run": bool(args.dry_run),
            "counts": counts,
            "rows": shadow_rows,
        }
        _write_json(shadow_path, shadow_payload)

    print(
        json.dumps(
            {
                "summary": str(summary),
                "state": str(state),
                "targets": int(len(targets)),
                "remaining_only": bool(args.remaining_only),
                "apply_auto": bool(args.apply_auto),
                "dry_run": bool(args.dry_run),
                "counts": counts,
                "backup_dir": "" if args.dry_run else str(backup_dir),
                "shadow_report": str(Path(args.shadow_report).expanduser().resolve()) if args.shadow_report else "",
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 1 if counts.get("als_validation_failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
