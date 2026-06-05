from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .groove import FULL_GROOVE_FEATURE_KEYS
from .ranker import candidate_timestamp, load_ranker_payload, predict_candidate_distances


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_float(value: Any, *, digits: int = 9) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=True)
        fh.write("\n")


def _candidate_rank(candidate: Mapping[str, Any], key: str, default: int = 999) -> int:
    value = candidate.get(key, candidate.get("rank", default))
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return out if out > 0 else int(default)


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    return float(_float_or_none(candidate.get("confidence_score", candidate.get("score"))) or 0.0)


def _candidate_metric(candidate: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        for nested_name in ("drumprint", "full_groove", "microalign"):
            nested = candidate.get(nested_name)
            if isinstance(nested, Mapping):
                value = nested.get(key)
                if value is not None:
                    break
    return float(_float_or_none(value) or 0.0)


def _ranked_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]


def _confidence_from_score(score: float, min_score: float = 0.42) -> float:
    return float(np.clip((float(score) - float(min_score)) / max(1e-6, 1.0 - float(min_score)), 0.0, 1.0))


def _confidence_tier(chosen: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]]) -> str:
    score = _candidate_score(chosen)
    model_score = _float_or_none(chosen.get("model_score"))
    model_rank = _candidate_rank(chosen, "model_rank")
    handcrafted_rank = _candidate_rank(chosen, "handcrafted_rank")
    valid_scores = [_candidate_score(candidate) for candidate in ordered if not candidate.get("rejected")]
    gap = 0.0
    if len(valid_scores) >= 2:
        gap = max(0.0, valid_scores[0] - valid_scores[1])

    calibration = 0
    if score >= 0.76:
        calibration += 2
    elif score >= 0.66:
        calibration += 1
    elif score < 0.52:
        calibration -= 2
    elif score < 0.60:
        calibration -= 1

    if gap >= 0.08:
        calibration += 2
    elif gap >= 0.04:
        calibration += 1
    elif gap < 0.02:
        calibration -= 1

    if model_score is not None:
        if model_score <= 0.025:
            calibration += 2
        elif model_score <= 0.075:
            calibration += 1
        elif model_score > 0.50:
            calibration -= 2
        elif model_score > 0.20:
            calibration -= 1

    calibration += 1 if handcrafted_rank == 1 and model_rank == 1 else -1
    if calibration >= 4:
        return "HIGH"
    if calibration >= 1:
        return "MEDIUM"
    return "LOW"


def rerank_candidate_payload(
    payload: Mapping[str, Any],
    *,
    ranker_payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    raw_candidates = payload.get("top_10_candidates") or payload.get("candidates") or []
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return None
    candidates = _ranked_candidates(raw_candidates)
    if not candidates:
        return None

    distances = predict_candidate_distances(ranker_payload, candidates)
    if not distances:
        return None

    for candidate, distance in zip(candidates, distances):
        candidate["model_score"] = float(distance)

    model_order = sorted(
        candidates,
        key=lambda candidate: (
            _float_or_none(candidate.get("model_score")) if _float_or_none(candidate.get("model_score")) is not None else float("inf"),
            _candidate_rank(candidate, "handcrafted_rank"),
            candidate_timestamp(candidate) or float("inf"),
        ),
    )
    for model_rank, candidate in enumerate(model_order, start=1):
        candidate["model_rank"] = int(model_rank)

    model_ids = {id(candidate) for candidate in model_order}
    remainder = [candidate for candidate in candidates if id(candidate) not in model_ids]
    ordered = model_order + remainder
    chosen = ordered[0]
    for rank, candidate in enumerate(ordered, start=1):
        candidate["rank"] = int(rank)
        candidate["selected"] = candidate is chosen
        candidate["selected_by"] = "model"
        if candidate is chosen:
            candidate["reason"] = "selected_by_model:lowest_predicted_user_delta_after_retrain"
        elif _candidate_rank(candidate, "model_rank", 0) > 0:
            candidate["reason"] = "not_selected:model_predicted_larger_user_delta"

    selected_time = candidate_timestamp(chosen)
    if selected_time is None:
        return None

    feature_summary = dict(payload.get("feature_summary") if isinstance(payload.get("feature_summary"), Mapping) else {})
    tier = _confidence_tier(chosen, ordered)
    confidence = _confidence_from_score(_candidate_score(chosen))
    now = datetime.now(timezone.utc).isoformat()
    feature_summary.update(
        {
            "selected_by": "model",
            "confidence_tier": tier,
            "chosen_score": _candidate_score(chosen),
            "chosen_transient_strength": _float_or_none(chosen.get("transient_strength")) or 0.0,
            "chosen_low_end_jump": _float_or_none(chosen.get("low_end_jump")) or 0.0,
            "chosen_post_drop_density": _float_or_none(chosen.get("post_drop_density")) or 0.0,
            "chosen_pre_post_energy_ratio": _float_or_none(chosen.get("pre_post_energy_ratio")) or 0.0,
            "chosen_energy_contrast": _float_or_none(chosen.get("energy_contrast")) or 0.0,
            "chosen_rhythmic_consistency": _float_or_none(chosen.get("rhythmic_consistency")) or 0.0,
            "chosen_snap_offset_sec": _float_or_none(chosen.get("snap_offset_sec", chosen.get("snap_offset"))) or 0.0,
            "chosen_handcrafted_rank": float(_candidate_rank(chosen, "handcrafted_rank", 0)),
            "chosen_model_rank": float(_candidate_rank(chosen, "model_rank", 0)),
            "chosen_model_score": _float_or_none(chosen.get("model_score")) or 0.0,
            "ranker_status": "used_existing_candidate_rerank",
            "ranker_model_path": str(ranker_payload.get("path", "")),
            "ranker_training_rows": int(ranker_payload.get("training_rows", 0) or 0),
            "ranker_correction_rows": int(ranker_payload.get("correction_rows", 0) or 0),
            "reranked_at": now,
        }
    )
    for key in FULL_GROOVE_FEATURE_KEYS:
        feature_summary[f"chosen_{key}"] = _candidate_metric(chosen, key)

    out = dict(payload)
    out["final_ai_pick"] = float(selected_time)
    out["coarse_ai_pick"] = _float_or_none(chosen.get("coarse_timestamp")) or float(selected_time)
    out["confidence"] = float(confidence)
    out["confidence_tier"] = tier
    out["selected_by"] = "model"
    out["selected_candidate"] = dict(chosen)
    out["top_10_candidates"] = ordered[:10]
    out["feature_summary"] = feature_summary
    out["reranked_from_model"] = str(ranker_payload.get("path", ""))
    out["reranked_at"] = now
    return out


def rerank_summary_with_model(
    summary_csv: str | Path,
    *,
    model_path: Optional[str] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    summary_path = Path(summary_csv).expanduser()
    if not summary_path.exists():
        raise FileNotFoundError(f"Batch summary not found: {summary_path}")
    ranker_payload = load_ranker_payload(model_path)
    if ranker_payload is None:
        return {"ok": False, "summary_csv": str(summary_path), "error": "ranker model missing", "changed": 0}

    with open(summary_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"Batch summary has no header: {summary_path}")

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = summary_path.with_name(f"{summary_path.stem}.before_rerank_{stamp}{summary_path.suffix}")
        shutil.copy2(summary_path, backup_path)
    else:
        backup_path = None

    changed = 0
    missing_json = 0
    failed_json = 0
    changed_times = 0
    for row in rows:
        candidate_path_text = str(row.get("candidates_json") or "")
        if not candidate_path_text:
            missing_json += 1
            continue
        candidate_path = Path(candidate_path_text).expanduser()
        if not candidate_path.exists():
            missing_json += 1
            continue
        try:
            old_payload = _read_json(candidate_path)
            new_payload = rerank_candidate_payload(old_payload, ranker_payload=ranker_payload)
            if new_payload is None:
                failed_json += 1
                continue
            old_time = _float_or_none(row.get("detected_drop_time")) or _float_or_none(old_payload.get("final_ai_pick"))
            new_time = _float_or_none(new_payload.get("final_ai_pick"))
            if new_time is None:
                failed_json += 1
                continue
            _write_json(candidate_path, new_payload)
            row["detected_drop_time"] = _format_float(new_time)
            row["confidence"] = _format_float(new_payload.get("confidence"), digits=6)
            row["confidence_tier"] = str(new_payload.get("confidence_tier") or row.get("confidence_tier") or "")
            row["selected_by"] = str(new_payload.get("selected_by") or "model")
            selected_candidate = new_payload.get("selected_candidate") if isinstance(new_payload.get("selected_candidate"), Mapping) else {}
            feature_summary = new_payload.get("feature_summary") if isinstance(new_payload.get("feature_summary"), Mapping) else {}
            for column in FULL_GROOVE_FEATURE_KEYS:
                if column in row:
                    value = _candidate_metric(selected_candidate, column)
                    if not value:
                        value = float(_float_or_none(feature_summary.get(f"chosen_{column}")) or 0.0)
                    row[column] = _format_float(value)
            if old_time is None or abs(float(old_time) - float(new_time)) > 1e-9:
                changed_times += 1
            changed += 1
        except Exception:
            failed_json += 1

    with open(summary_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "ok": True,
        "summary_csv": str(summary_path),
        "model_path": str(ranker_payload.get("path", "")),
        "backup": str(backup_path) if backup_path else "",
        "rows": int(len(rows)),
        "changed": int(changed),
        "changed_times": int(changed_times),
        "missing_json": int(missing_json),
        "failed_json": int(failed_json),
        "ranker_training_rows": int(ranker_payload.get("training_rows", 0) or 0),
        "ranker_correction_rows": int(ranker_payload.get("correction_rows", 0) or 0),
    }
