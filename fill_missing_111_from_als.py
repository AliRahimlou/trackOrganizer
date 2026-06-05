#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import csv
import gzip
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

import trackOrganizerAndAlsGen as tog


ROLE_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.I)
KEY_RE = re.compile(r"^(?:drums|inst|vocals)_\d{2,3}_([0-9]{1,2}[ab])(?:_\d{1,2})?[-_]", re.I)


def _clip_name(clip: ET.Element) -> str:
    n = clip.find("./Name")
    if n is not None:
        return str(n.get("Value") or "").strip()
    return ""


def _clip_role(clip: ET.Element) -> Optional[str]:
    m = ROLE_RE.match(_clip_name(clip))
    if not m:
        return None
    return m.group(1).lower()


def _clip_bpm(clip: ET.Element) -> Optional[int]:
    m = BPM_RE.match(_clip_name(clip))
    if not m:
        return None
    try:
        bpm = int(m.group(1))
    except Exception:
        return None
    return bpm if bpm > 0 else None


def _clip_key(clip: ET.Element) -> Optional[str]:
    m = KEY_RE.match(_clip_name(clip))
    if not m:
        return None
    return m.group(1).upper()


def _track_name(track: ET.Element) -> str:
    n = track.find("./Name/EffectiveName")
    if n is not None and n.get("Value"):
        return str(n.get("Value"))
    n = track.find("./Name/UserName")
    if n is not None and n.get("Value"):
        return str(n.get("Value"))
    return ""


def _slot_audio_clip(slot: ET.Element) -> Optional[ET.Element]:
    for p in ("./ClipSlot/Value/AudioClip", "./Value/AudioClip", ".//AudioClip"):
        c = slot.find(p)
        if c is not None:
            return c
    return None


def _track_slots(track: ET.Element) -> List[ET.Element]:
    return track.findall("./DeviceChain/MainSequencer/ClipSlotList/ClipSlot")


def _resolve_clip_audio_path(clip: ET.Element, als_path: str) -> Optional[str]:
    # Prefer absolute path when present.
    p = clip.find("./SampleRef/FileRef/Path")
    if p is not None:
        v = str(p.get("Value") or "").strip()
        if v and os.path.exists(v):
            return os.path.abspath(v)

    # Fallback to relative path.
    rp = clip.find("./SampleRef/FileRef/RelativePath")
    if rp is not None:
        rv = str(rp.get("Value") or "").strip()
        if rv:
            base_dir = os.path.dirname(os.path.abspath(als_path))
            cand = os.path.abspath(os.path.normpath(os.path.join(base_dir, rv)))
            if os.path.exists(cand):
                return cand
    return None


def _parse_warp_markers(clip: ET.Element) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    wm = clip.find("./WarpMarkers")
    if wm is None:
        return out
    for m in wm.findall("./WarpMarker"):
        try:
            sec = float(m.get("SecTime"))
            beat = float(m.get("BeatTime"))
        except Exception:
            continue
        out.append((sec, beat))
    out.sort(key=lambda x: x[0])
    return out


def _has_drop_anchor_111(clip: ET.Element, min_drop_sec: float = 1.0) -> bool:
    pts = _parse_warp_markers(clip)
    if len(pts) < 3:
        return False
    has_neg = any(b < -1e-6 for _, b in pts)
    zero_secs = [s for s, b in pts if abs(b) <= 1e-6]
    if not zero_secs:
        return False
    if min(zero_secs) <= float(min_drop_sec):
        return False
    return has_neg


def _clip_current_end_sec(clip: ET.Element) -> Optional[float]:
    n = clip.find("./CurrentEnd")
    if n is not None:
        try:
            v = float(n.get("Value"))
            if v > 0:
                return v
        except Exception:
            pass
    n = clip.find("./Loop/OutMarker")
    if n is not None:
        try:
            v = float(n.get("Value"))
            if v > 0:
                return v
        except Exception:
            pass
    return None


def _sec_to_beats(sec: float, bpm: int) -> float:
    return (float(sec) * float(bpm)) / 60.0


def _apply_triplet_warp_markers(clips: Sequence[ET.Element], bpm: int, drop_sec: float, end_sec: float) -> None:
    phi = -_sec_to_beats(drop_sec, bpm)
    points = sorted(set(round(max(0.0, x), 6) for x in [0.0, float(drop_sec), float(end_sec)]))
    if len(points) < 2:
        return

    for clip in clips:
        old = clip.find("./WarpMarkers")
        if old is not None:
            clip.remove(old)
        wm = ET.Element("WarpMarkers")
        for i, sec in enumerate(points):
            beat = _sec_to_beats(sec, bpm) + phi
            wm.append(
                ET.Element(
                    "WarpMarker",
                    {
                        "Id": str(i),
                        "SecTime": f"{sec:.6f}",
                        "BeatTime": f"{beat:.6f}",
                    },
                )
            )
        clip.insert(0, wm)

        iw = clip.find("./IsWarped")
        if iw is None:
            iw = ET.Element("IsWarped", {"Value": "true"})
            clip.insert(0, iw)
        else:
            iw.set("Value", "true")


def _choose_tracks_for_rows(root: ET.Element) -> List[ET.Element]:
    tracks = root.findall(".//AudioTrack")
    if not tracks:
        return []
    by_name = {_track_name(t): t for t in tracks}
    preferred = [by_name[n] for n in ("CH1", "CH2", "CH3") if n in by_name]
    if len(preferred) == 3:
        return preferred
    return tracks[:3]


def _iter_triplet_rows(root: ET.Element) -> List[Tuple[int, Dict[str, ET.Element]]]:
    tracks = _choose_tracks_for_rows(root)
    if len(tracks) < 3:
        return []
    slots_per_track = [_track_slots(t) for t in tracks]
    max_slots = max((len(s) for s in slots_per_track), default=0)
    rows: List[Tuple[int, Dict[str, ET.Element]]] = []
    for i in range(max_slots):
        row: Dict[str, ET.Element] = {}
        for slots in slots_per_track:
            if i >= len(slots):
                continue
            clip = _slot_audio_clip(slots[i])
            if clip is None:
                continue
            role = _clip_role(clip)
            if role in ("drums", "inst", "vocals") and role not in row:
                row[role] = clip
        if {"drums", "inst", "vocals"}.issubset(row.keys()):
            rows.append((i, row))
    return rows


_MODEL_PATH_CACHE: Optional[str] = None
_MODEL_REASON_CACHE: str = ""


def _resolve_alsdrop_model_path() -> Tuple[Optional[str], str]:
    global _MODEL_PATH_CACHE, _MODEL_REASON_CACHE
    if _MODEL_PATH_CACHE is not None:
        return _MODEL_PATH_CACHE, _MODEL_REASON_CACHE
    try:
        p, r = tog._resolve_alsdrop_model_path()
    except Exception:
        p, r = None, "resolve_failed"
    _MODEL_PATH_CACHE = p
    _MODEL_REASON_CACHE = str(r or "")
    return _MODEL_PATH_CACHE, _MODEL_REASON_CACHE


def _apply_stage2_late_push(sec: Optional[float], bpm: int) -> Optional[float]:
    if sec is None:
        return None
    if bool(getattr(tog, "DROP_STAGE2_ENABLE", False)):
        fixed_late = max(0.0, float(getattr(tog, "DROP_STAGE2_FIXED_LATE_SEC", 0.0)))
        if fixed_late > 0.0:
            beat_sec = 60.0 / float(max(1, bpm))
            max_push = min(0.18 * beat_sec, 0.060)
            sec = float(sec) + min(fixed_late, max_push)
    return float(sec)


def _accept_model_prediction(
    pred: Dict[str, Any],
    min_confidence: float,
    min_margin: float,
) -> Tuple[bool, str]:
    conf = float(getattr(tog, "_as_float", lambda x: x)(pred.get("confidence")) or 0.0)
    margin = float(getattr(tog, "_as_float", lambda x: x)(pred.get("score_margin")) or 0.0)
    region_valid = bool(pred.get("region_valid", True))
    if not region_valid:
        return False, "region_invalid"
    if margin < 0.04:
        return False, "margin_too_low"
    if conf >= 0.85:
        return True, "conf_high"
    conf_gate = max(0.72, float(min_confidence))
    margin_gate = max(0.08, float(min_margin))
    if conf >= conf_gate and margin >= margin_gate:
        return True, "conf_margin"
    if conf < conf_gate:
        return False, "low_confidence"
    return False, "low_margin"


def _predict_with_alsdrop(
    drums_audio: str,
    bpm: int,
    *,
    use_madmom: bool,
    verbose_detector: bool,
    review_threshold: float,
    micro_align: bool,
    micro_window_pre_ms: float,
    micro_window_post_ms: float,
    micro_threshold_k: float,
    model_cache: Dict[Tuple[str, int, int, int, int, int], Dict[str, Any]],
) -> Dict[str, Any]:
    key = (
        os.path.abspath(drums_audio),
        int(bpm),
        1 if bool(use_madmom) else 0,
        int(round(float(micro_window_pre_ms))),
        int(round(float(micro_window_post_ms))),
        int(round(float(micro_threshold_k) * 1000.0)),
    )
    if key in model_cache:
        return dict(model_cache[key])

    pred: Dict[str, Any] = {}
    if not os.path.exists(drums_audio):
        pred = {"ok": False, "reason": "missing_audio", "audio_path": os.path.abspath(drums_audio)}
        model_cache[key] = pred
        return dict(pred)

    if tog.alsdrop_run_predict is None:
        pred = {"ok": False, "reason": "alsdrop_unavailable", "audio_path": os.path.abspath(drums_audio)}
        model_cache[key] = pred
        return dict(pred)

    model_path, model_reason = _resolve_alsdrop_model_path()
    if not model_path or not os.path.exists(model_path):
        pred = {"ok": False, "reason": f"model_missing:{model_reason}", "audio_path": os.path.abspath(drums_audio)}
        model_cache[key] = pred
        return dict(pred)

    try:
        if verbose_detector:
            raw = tog.alsdrop_run_predict(
                audio_path=drums_audio,
                model_path=model_path,
                out_json=None,
                device="auto",
                use_madmom=bool(use_madmom),
                bpm_override=float(bpm),
                review_threshold=float(review_threshold),
                micro_align=bool(micro_align),
                micro_window_pre_ms=float(micro_window_pre_ms),
                micro_window_post_ms=float(micro_window_post_ms),
                micro_threshold_k=float(micro_threshold_k),
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                raw = tog.alsdrop_run_predict(
                    audio_path=drums_audio,
                    model_path=model_path,
                    out_json=None,
                    device="auto",
                    use_madmom=bool(use_madmom),
                    bpm_override=float(bpm),
                    review_threshold=float(review_threshold),
                    micro_align=bool(micro_align),
                    micro_window_pre_ms=float(micro_window_pre_ms),
                    micro_window_post_ms=float(micro_window_post_ms),
                    micro_threshold_k=float(micro_threshold_k),
                )
    except Exception as e:
        pred = {"ok": False, "reason": f"infer_exception:{e}", "audio_path": os.path.abspath(drums_audio)}
        model_cache[key] = pred
        return dict(pred)

    sec = getattr(tog, "_as_float", lambda x: x)(raw.get("predicted_sec"))
    conf = float(getattr(tog, "_as_float", lambda x: x)(raw.get("confidence")) or 0.0)
    margin = float(getattr(tog, "_as_float", lambda x: x)(raw.get("score_margin")) or 0.0)
    out = dict(raw)
    out.update(
        {
            "ok": bool(sec is not None and float(sec) >= 0.0),
            "predicted_sec": float(sec) if sec is not None else None,
            "confidence": float(conf),
            "score_margin": float(margin),
            "model_path": os.path.abspath(model_path),
            "model_reason": str(model_reason or ""),
        }
    )
    model_cache[key] = out
    return dict(out)


def _predict_with_legacy(
    drums_audio: str,
    bpm: int,
    *,
    verbose_detector: bool,
    legacy_cache: Dict[Tuple[str, int], Tuple[Optional[float], float, Dict[str, float]]],
) -> Tuple[Optional[float], float, Dict[str, float]]:
    k = (os.path.abspath(drums_audio), int(bpm))
    if k in legacy_cache:
        sec, conf, meta = legacy_cache[k]
        return sec, conf, dict(meta or {})
    try:
        if verbose_detector:
            sec, conf, meta = tog._detect_drop_anchor_sec(drums_audio, bpm)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sec, conf, meta = tog._detect_drop_anchor_sec(drums_audio, bpm)
    except Exception:
        sec, conf, meta = None, 0.0, {}
    sec = _apply_stage2_late_push(sec, bpm)
    legacy_cache[k] = (sec, float(conf or 0.0), dict(meta or {}))
    return sec, float(conf or 0.0), dict(meta or {})


def _duration_seconds_cached(path: str, duration_cache: Dict[str, float]) -> Optional[float]:
    ap = os.path.abspath(path)
    if ap in duration_cache:
        v = float(duration_cache[ap])
        return v if v > 0.0 else None
    sec = tog.get_duration_seconds(ap)
    duration_cache[ap] = float(sec or 0.0)
    if sec and sec > 0:
        return float(sec)
    return None


def _row_end_sec(row: Dict[str, ET.Element], als_path: str, duration_cache: Dict[str, float]) -> Optional[float]:
    vals: List[float] = []
    for role in ("drums", "inst", "vocals"):
        clip = row.get(role)
        if clip is None:
            continue
        p = _resolve_clip_audio_path(clip, als_path)
        if p:
            sec = _duration_seconds_cached(p, duration_cache)
            if sec and sec > 0:
                vals.append(float(sec))
                continue
        sec2 = _clip_current_end_sec(clip)
        if sec2 and sec2 > 0:
            vals.append(float(sec2))
    if not vals:
        return None
    return max(vals)


def _write_review_queue(base_out_als: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    base_abs = os.path.abspath(base_out_als)
    stem, _ = os.path.splitext(base_abs)
    jsonl_path = f"{stem}_review_queue.jsonl"
    csv_path = f"{stem}_review_queue.csv"
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    dedup: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = f"{r.get('slot')}::{r.get('audio_path')}::{r.get('reason')}"
        dedup[k] = r
    out_rows = [dedup[k] for k in sorted(dedup.keys())]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=True))
            f.write("\n")

    fieldnames = [
        "slot",
        "name",
        "audio_path",
        "bpm",
        "predicted_sec",
        "confidence",
        "score_margin",
        "selected_by",
        "needs_manual_review",
        "reason",
        "pass_stage",
        "model_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(
                {
                    "slot": int(r.get("slot") or 0),
                    "name": str(r.get("name") or ""),
                    "audio_path": str(r.get("audio_path") or ""),
                    "bpm": int(r.get("bpm") or 0),
                    "predicted_sec": f"{float(r.get('predicted_sec') or 0.0):.6f}",
                    "confidence": f"{float(r.get('confidence') or 0.0):.6f}",
                    "score_margin": f"{float(r.get('score_margin') or 0.0):.6f}",
                    "selected_by": str(r.get("selected_by") or ""),
                    "needs_manual_review": int(bool(r.get("needs_manual_review", False))),
                    "reason": str(r.get("reason") or ""),
                    "pass_stage": str(r.get("pass_stage") or ""),
                    "model_path": str(r.get("model_path") or ""),
                }
            )
    return jsonl_path, csv_path


def process_als(
    als_in: str,
    als_out: str,
    min_confidence: float,
    min_model_margin: float,
    pass1_min_confidence: float,
    pass1_min_margin: float,
    force: bool,
    dry_run: bool,
    model_only_fast: bool,
    two_pass: bool,
    use_madmom: bool,
    review_threshold: float,
    micro_align: bool,
    micro_window_pre_ms: float,
    micro_window_post_ms: float,
    micro_threshold_k: float,
    verbose_detector: bool,
    min_drop_sec_existing: float,
    max_rows: int,
) -> Tuple[Dict[str, int], str, str]:
    with gzip.open(als_in, "rb") as f:
        root = ET.fromstring(f.read())

    rows = _iter_triplet_rows(root)
    stats = {
        "rows_total": len(rows),
        "rows_missing": 0,
        "rows_updated": 0,
        "rows_updated_pass1": 0,
        "rows_updated_pass2": 0,
        "rows_skipped_conf": 0,
        "rows_skipped_margin": 0,
        "rows_skipped_detect": 0,
        "rows_skipped_missing_audio": 0,
        "rows_skipped_rejected": 0,
        "rows_skipped_end": 0,
        "review_rows": 0,
    }

    model_cache: Dict[Tuple[str, int, int, int, int, int], Dict[str, Any]] = {}
    legacy_cache: Dict[Tuple[str, int], Tuple[Optional[float], float, Dict[str, float]]] = {}
    duration_cache: Dict[str, float] = {}
    review_rows: List[Dict[str, Any]] = []

    processed_missing = 0
    for slot_idx, row in rows:
        drums = row["drums"]
        bpm = _clip_bpm(drums)
        if not bpm or bpm <= 0:
            stats["rows_skipped_detect"] += 1
            print(f"[SKIP] slot={slot_idx} missing BPM in drums name: {_clip_name(drums)}")
            continue

        already = _has_drop_anchor_111(drums, min_drop_sec=min_drop_sec_existing)
        if already and not force:
            continue

        if max_rows > 0 and processed_missing >= max_rows:
            break
        stats["rows_missing"] += 1
        processed_missing += 1
        drums_audio = _resolve_clip_audio_path(drums, als_in)
        if not drums_audio:
            stats["rows_skipped_missing_audio"] += 1
            review_rows.append(
                {
                    "slot": int(slot_idx),
                    "name": _clip_name(drums),
                    "audio_path": "",
                    "bpm": int(bpm),
                    "predicted_sec": 0.0,
                    "confidence": 0.0,
                    "score_margin": 0.0,
                    "selected_by": "",
                    "needs_manual_review": True,
                    "reason": "missing_audio",
                    "pass_stage": "precheck",
                    "model_path": "",
                }
            )
            print(f"[SKIP] slot={slot_idx} missing audio path")
            continue

        chosen_sec: Optional[float] = None
        chosen_conf = 0.0
        chosen_pass = "none"
        model_pred: Dict[str, Any] = _predict_with_alsdrop(
            drums_audio=drums_audio,
            bpm=int(bpm),
            use_madmom=bool(use_madmom),
            verbose_detector=bool(verbose_detector),
            review_threshold=float(review_threshold),
            micro_align=bool(micro_align),
            micro_window_pre_ms=float(micro_window_pre_ms),
            micro_window_post_ms=float(micro_window_post_ms),
            micro_threshold_k=float(micro_threshold_k),
            model_cache=model_cache,
        )

        model_reason = str(model_pred.get("reason") or "")
        model_selected_by = str(model_pred.get("selected_by") or "")
        model_conf = float(getattr(tog, "_as_float", lambda x: x)(model_pred.get("confidence")) or 0.0)
        model_margin = float(getattr(tog, "_as_float", lambda x: x)(model_pred.get("score_margin")) or 0.0)
        model_sec = getattr(tog, "_as_float", lambda x: x)(model_pred.get("predicted_sec"))

        if bool(model_pred.get("ok")) and model_sec is not None:
            # Pass-1 strict model gate.
            pass1_ok, _ = _accept_model_prediction(
                model_pred,
                min_confidence=float(pass1_min_confidence),
                min_margin=float(pass1_min_margin),
            )
            if pass1_ok:
                chosen_sec = _apply_stage2_late_push(float(model_sec), int(bpm))
                chosen_conf = float(model_conf)
                chosen_pass = "pass1_model"
            elif bool(two_pass):
                # Pass-2 relaxed gate.
                pass2_ok, _ = _accept_model_prediction(
                    model_pred,
                    min_confidence=float(min_confidence),
                    min_margin=float(min_model_margin),
                )
                if pass2_ok:
                    chosen_sec = _apply_stage2_late_push(float(model_sec), int(bpm))
                    chosen_conf = float(model_conf)
                    chosen_pass = "pass2_model"

        # Coverage fallback path (only when not strict model-only fast).
        if chosen_sec is None and (not bool(model_only_fast)):
            sec_l, conf_l, _meta_l = _predict_with_legacy(
                drums_audio=drums_audio,
                bpm=int(bpm),
                verbose_detector=bool(verbose_detector),
                legacy_cache=legacy_cache,
            )
            if sec_l is not None and conf_l >= float(min_confidence):
                chosen_sec = float(sec_l)
                chosen_conf = float(conf_l)
                chosen_pass = "pass2_legacy"

        if chosen_sec is None:
            if not os.path.exists(drums_audio):
                stats["rows_skipped_missing_audio"] += 1
                reason = "missing_audio"
            else:
                if model_pred.get("ok") and model_sec is not None:
                    if model_margin < 0.04:
                        stats["rows_skipped_margin"] += 1
                        reason = "low_margin"
                    elif model_conf < float(min_confidence):
                        stats["rows_skipped_conf"] += 1
                        reason = "low_confidence"
                    else:
                        stats["rows_skipped_rejected"] += 1
                        reason = "model_rejected"
                else:
                    stats["rows_skipped_detect"] += 1
                    reason = model_reason or "detect_failed"
            review_rows.append(
                {
                    "slot": int(slot_idx),
                    "name": _clip_name(drums),
                    "audio_path": os.path.abspath(drums_audio),
                    "bpm": int(bpm),
                    "predicted_sec": float(model_sec or 0.0),
                    "confidence": float(model_conf),
                    "score_margin": float(model_margin),
                    "selected_by": str(model_selected_by),
                    "needs_manual_review": True,
                    "reason": str(reason),
                    "pass_stage": "none",
                    "model_path": str(model_pred.get("model_path") or ""),
                }
            )
            print(f"[SKIP] slot={slot_idx} {reason}")
            continue

        end_sec = _row_end_sec(row, als_in, duration_cache)
        if end_sec is None or end_sec <= 0:
            stats["rows_skipped_end"] += 1
            print(f"[SKIP] slot={slot_idx} no end duration")
            continue

        _apply_triplet_warp_markers([row["drums"], row["inst"], row["vocals"]], bpm=bpm, drop_sec=float(chosen_sec), end_sec=end_sec)
        stats["rows_updated"] += 1
        if chosen_pass == "pass1_model":
            stats["rows_updated_pass1"] += 1
        else:
            stats["rows_updated_pass2"] += 1
        if chosen_pass == "pass2_legacy":
            review_rows.append(
                {
                    "slot": int(slot_idx),
                    "name": _clip_name(drums),
                    "audio_path": os.path.abspath(drums_audio),
                    "bpm": int(bpm),
                    "predicted_sec": float(chosen_sec),
                    "confidence": float(chosen_conf),
                    "score_margin": float(model_margin),
                    "selected_by": "legacy_fallback",
                    "needs_manual_review": True,
                    "reason": "legacy_fallback_used",
                    "pass_stage": str(chosen_pass),
                    "model_path": str(model_pred.get("model_path") or ""),
                }
            )
        print(
            f"[OK] slot={slot_idx} bpm={bpm} key={_clip_key(drums) or '?'} "
            f"drop={float(chosen_sec):.3f}s conf={float(chosen_conf):.2f} pass={chosen_pass} "
            f"end={end_sec:.3f}s name={_clip_name(drums)}"
        )

    if not dry_run:
        out_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with gzip.open(als_out, "wb") as f:
            f.write(out_xml)

    stats["review_rows"] = int(len(review_rows))
    jsonl_path, csv_path = _write_review_queue(als_out, review_rows)
    return stats, jsonl_path, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill missing Ableton 1.1.1 anchors (drop markers) for drums/inst/vocals rows in an ALS file."
    )
    ap.add_argument("--als", required=True, help="Input .als path")
    ap.add_argument("--out", default="", help="Output .als path (default: <in>_filled.als)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input ALS")
    ap.add_argument("--force", action="store_true", help="Recompute anchors even when one appears to exist")
    ap.add_argument("--min-confidence", type=float, default=0.70, help="Pass-2 minimum confidence to write markers")
    ap.add_argument(
        "--min-model-margin",
        type=float,
        default=0.08,
        help="Pass-2 minimum model score margin (top1-top2).",
    )
    ap.add_argument(
        "--pass1-min-confidence",
        type=float,
        default=0.80,
        help="Two-pass mode pass-1 strict model confidence.",
    )
    ap.add_argument(
        "--pass1-min-margin",
        type=float,
        default=0.08,
        help="Two-pass mode pass-1 strict model margin.",
    )
    ap.add_argument(
        "--min-drop-sec-existing",
        type=float,
        default=1.0,
        help="Treat existing 1.1.1 as missing if BeatTime=0 is at/before this second",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="Process at most this many missing rows (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="Analyze and print changes, do not write output ALS")
    ap.add_argument(
        "--model-only-fast",
        action="store_true",
        help="Use ALSDrop model only; skip legacy fallback detector for speed.",
    )
    ap.add_argument(
        "--two-pass",
        action="store_true",
        help="Pass 1 strict model fill, then pass 2 coverage (model + legacy fallback).",
    )
    ap.add_argument(
        "--no-madmom",
        action="store_true",
        help="Disable madmom proposals during ALSDrop inference (faster, may reduce accuracy).",
    )
    ap.add_argument("--review-threshold", type=float, default=0.55, help="ALSDrop review threshold.")
    ap.add_argument("--micro-align", action="store_true", help="Apply final high-res transient micro-alignment.")
    ap.add_argument("--micro-window-pre-ms", type=float, default=120.0)
    ap.add_argument("--micro-window-post-ms", type=float, default=180.0)
    ap.add_argument("--micro-threshold-k", type=float, default=1.25)
    ap.add_argument(
        "--verbose-detector",
        action="store_true",
        help="Show internal ALSDrop detector logs for each analyzed row (default: quiet).",
    )
    args = ap.parse_args()

    als_in = os.path.abspath(args.als)
    if not os.path.exists(als_in):
        raise SystemExit(f"Input ALS not found: {als_in}")

    if args.in_place:
        als_out = als_in
    elif args.out.strip():
        als_out = os.path.abspath(args.out.strip())
    else:
        base, ext = os.path.splitext(als_in)
        als_out = f"{base}_filled{ext}"

    if (not args.dry_run) and (not args.in_place):
        os.makedirs(os.path.dirname(als_out), exist_ok=True)

    print(f"[INFO] ALS in:  {als_in}")
    print(f"[INFO] ALS out: {als_out}")
    print(f"[INFO] dry_run={bool(args.dry_run)} force={bool(args.force)} min_conf={float(args.min_confidence):.2f}")
    print(
        f"[INFO] model_only_fast={bool(args.model_only_fast)} two_pass={bool(args.two_pass)} "
        f"madmom={'off' if bool(args.no_madmom) else 'on'} "
        f"min_margin={float(args.min_model_margin):.3f} "
        f"micro_align={bool(args.micro_align)}"
    )
    model_path, model_reason = _resolve_alsdrop_model_path()
    if model_path:
        print(f"[ALSDrop] Using model: {model_path} ({model_reason})")
    else:
        print(f"[ALSDrop][WARN] Model unresolved: {model_reason}")

    if bool(args.no_madmom):
        tog.ALSDROP_USE_MADMOM = False
        print("[WARN] --no-madmom is not recommended for production coverage.")

    stats, review_jsonl, review_csv = process_als(
        als_in=als_in,
        als_out=als_out,
        min_confidence=float(args.min_confidence),
        min_model_margin=float(args.min_model_margin),
        pass1_min_confidence=float(args.pass1_min_confidence),
        pass1_min_margin=float(args.pass1_min_margin),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        model_only_fast=bool(args.model_only_fast),
        two_pass=bool(args.two_pass),
        use_madmom=not bool(args.no_madmom),
        review_threshold=float(args.review_threshold),
        micro_align=bool(args.micro_align),
        micro_window_pre_ms=float(args.micro_window_pre_ms),
        micro_window_post_ms=float(args.micro_window_post_ms),
        micro_threshold_k=float(args.micro_threshold_k),
        verbose_detector=bool(args.verbose_detector),
        min_drop_sec_existing=float(args.min_drop_sec_existing),
        max_rows=int(args.max_rows),
    )
    print(
        "[DONE] "
        f"rows_total={stats['rows_total']} "
        f"rows_missing={stats['rows_missing']} "
        f"rows_updated={stats['rows_updated']} "
        f"updated_pass1={stats['rows_updated_pass1']} "
        f"updated_pass2={stats['rows_updated_pass2']} "
        f"skip_detect={stats['rows_skipped_detect']} "
        f"skip_missing_audio={stats['rows_skipped_missing_audio']} "
        f"skip_conf={stats['rows_skipped_conf']} "
        f"skip_margin={stats['rows_skipped_margin']} "
        f"skip_rejected={stats['rows_skipped_rejected']} "
        f"skip_end={stats['rows_skipped_end']}"
    )
    print(f"[DONE] review_queue_rows={stats['review_rows']}")
    print(f"[DONE] review_queue_jsonl={review_jsonl}")
    print(f"[DONE] review_queue_csv={review_csv}")
    if args.dry_run:
        print("[DONE] Dry run only; no ALS written.")


if __name__ == "__main__":
    main()
