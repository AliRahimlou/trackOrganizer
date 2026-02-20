#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio_features import load_feature_cache
from .utils import as_float, iter_jsonl, write_json, write_jsonl


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError("librosa is required: pip install librosa soundfile") from e
    return librosa


def _duration_sec(path: str) -> float:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(path)
        if info and info.frames and info.samplerate and info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        librosa = _require_librosa()
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def _role_from_audio_path(path: str) -> str:
    b = os.path.basename(path).lower()
    if b.startswith("drums_"):
        return "drums"
    if b.startswith("inst_"):
        return "inst"
    if b.startswith("vocals_"):
        return "vocals"
    return "other"


def _is_monotonic_warp(markers: Sequence[Dict[str, object]]) -> bool:
    if not markers:
        return False
    pts: List[Tuple[float, float]] = []
    for m in markers:
        b = as_float(m.get("beat"))
        s = as_float(m.get("sec"))
        if b is None or s is None:
            return False
        pts.append((float(s), float(b)))
    pts.sort(key=lambda x: x[0])
    for i in range(1, len(pts)):
        if not (pts[i][0] > pts[i - 1][0] + 1e-9):
            return False
        if not (pts[i][1] > pts[i - 1][1] - 1e-6):
            return False
    return True


def _norm01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    span = max(1e-9, hi - lo)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _slice_mean(series: np.ndarray, times: np.ndarray, a: float, b: float, default: float = 0.0) -> float:
    if series.size == 0 or times.size == 0:
        return float(default)
    i0 = int(np.searchsorted(times, float(a), side="left"))
    i1 = int(np.searchsorted(times, float(b), side="right"))
    i0 = max(0, min(i0, len(series)))
    i1 = max(0, min(i1, len(series)))
    if i1 <= i0:
        return float(default)
    return float(np.mean(series[i0:i1]))


def _audio_signature(path: str, target_sec: float, sr: int = 22050, hop: int = 512) -> Optional[Dict[str, float]]:
    librosa = _require_librosa()
    try:
        y, sr_loaded = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return None
    if y is None or len(y) < 128:
        return None

    onset = librosa.onset.onset_strength(y=y, sr=sr_loaded, hop_length=hop).astype(np.float32, copy=False)
    st = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop)).astype(np.float32, copy=False)
    pwr = (st * st).astype(np.float32, copy=False)
    freqs = librosa.fft_frequencies(sr=sr_loaded, n_fft=2048).astype(np.float32, copy=False)
    low = np.sum(pwr[(freqs >= 20.0) & (freqs <= 150.0), :], axis=0).astype(np.float32, copy=False)
    high = np.sum(pwr[(freqs >= 2000.0) & (freqs <= min(10000.0, float(sr_loaded) * 0.5)), :], axis=0).astype(np.float32, copy=False)

    onset_n = _norm01(onset)
    low_n = _norm01(low)
    high_n = _norm01(high)
    t = librosa.frames_to_time(np.arange(len(onset_n)), sr=sr_loaded, hop_length=hop).astype(np.float32, copy=False)

    x = float(target_sec)
    pre16 = _slice_mean(onset_n, t, x - 16.0, x - 8.0)
    pre8 = _slice_mean(onset_n, t, x - 8.0, x - 2.0)
    pre_hf_a = _slice_mean(high_n, t, x - 16.0, x - 8.0)
    pre_hf_b = _slice_mean(high_n, t, x - 8.0, x - 2.0)
    pre2_on = _slice_mean(onset_n, t, x - 2.0, x)
    post2_on = _slice_mean(onset_n, t, x, x + 2.0)
    pre2_low = _slice_mean(low_n, t, x - 2.0, x)
    post2_low = _slice_mean(low_n, t, x, x + 2.0)
    post8_low = _slice_mean(low_n, t, x + 2.0, x + 8.0)

    return {
        "buildup_onset": float(pre8 - pre16),
        "buildup_hf": float(pre_hf_b - pre_hf_a),
        "impact_onset": float(post2_on - pre2_on),
        "impact_low": float(post2_low - pre2_low),
        "sustain_low": float(post8_low - pre2_low),
        "post_low_abs": float(post2_low),
    }


def _audio_signature_from_cache(cache_path: str, target_sec: float) -> Optional[Dict[str, float]]:
    try:
        feat = load_feature_cache(cache_path)
    except Exception:
        return None
    frame_times = feat.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    if frame_times.size < 16:
        return None

    onset = _norm01(feat.get("onset", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    low = _norm01(feat.get("low_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    high = _norm01(feat.get("high_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    x = float(target_sec)

    pre16 = _slice_mean(onset, frame_times, x - 16.0, x - 8.0)
    pre8 = _slice_mean(onset, frame_times, x - 8.0, x - 2.0)
    pre_hf_a = _slice_mean(high, frame_times, x - 16.0, x - 8.0)
    pre_hf_b = _slice_mean(high, frame_times, x - 8.0, x - 2.0)
    pre2_on = _slice_mean(onset, frame_times, x - 2.0, x)
    post2_on = _slice_mean(onset, frame_times, x, x + 2.0)
    pre2_low = _slice_mean(low, frame_times, x - 2.0, x)
    post2_low = _slice_mean(low, frame_times, x, x + 2.0)
    post8_low = _slice_mean(low, frame_times, x + 2.0, x + 8.0)

    return {
        "buildup_onset": float(pre8 - pre16),
        "buildup_hf": float(pre_hf_b - pre_hf_a),
        "impact_onset": float(post2_on - pre2_on),
        "impact_low": float(post2_low - pre2_low),
        "sustain_low": float(post8_low - pre2_low),
        "post_low_abs": float(post2_low),
    }


def _load_manifest(manifest_jsonl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not manifest_jsonl:
        return out
    if not os.path.isfile(manifest_jsonl):
        return out
    for r in iter_jsonl(manifest_jsonl):
        ap = os.path.abspath(str(r.get("audio_path", "")).strip())
        cp = os.path.abspath(str(r.get("cache_path", "")).strip())
        if ap and cp and os.path.isfile(cp):
            out[ap] = cp
    return out


def _tier_row(
    row: Dict[str, object],
    sig: Optional[Dict[str, float]],
    duration_sec: float,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    target_sec = as_float(row.get("target_sec"))
    if target_sec is None:
        return "bronze", ["missing_target"]
    t = float(target_sec)
    if t <= 0.0:
        reasons.append("target_nonpositive")
    if duration_sec <= 0.0:
        reasons.append("missing_duration")
    else:
        if t <= 5.0:
            reasons.append("too_early_lt5s")
        if t >= (duration_sec - 10.0):
            reasons.append("too_late_tail")

    markers = row.get("warp_markers") or []
    if not _is_monotonic_warp(markers if isinstance(markers, list) else []):
        reasons.append("warp_not_monotonic")

    md = row.get("metadata") or {}
    if isinstance(md, dict):
        mc = int(md.get("marker_count") or 0)
        if mc < 2:
            reasons.append("too_few_markers")
    else:
        reasons.append("missing_metadata")

    if sig is None:
        reasons.append("no_audio_signature")
    else:
        buildup_ok = (sig["buildup_onset"] >= 0.008) or (sig["buildup_hf"] >= 0.007)
        impact_ok = (sig["impact_low"] >= 0.040) and (sig["impact_onset"] >= 0.015)
        sustain_ok = (sig["sustain_low"] >= 0.012) and (sig["post_low_abs"] >= 0.09)
        if not buildup_ok:
            reasons.append("weak_buildup")
        if not impact_ok:
            reasons.append("weak_drop_impact")
        if not sustain_ok:
            reasons.append("weak_drop_sustain")

    if not reasons:
        return "gold", []

    # SILVER keeps sane warp + timing but fails strict musical checks.
    silver_block = {"target_nonpositive", "missing_duration", "too_late_tail", "warp_not_monotonic"}
    if any(r in silver_block for r in reasons):
        return "bronze", reasons
    if "too_early_lt5s" in reasons and len(reasons) == 1:
        return "silver", reasons
    if "no_audio_signature" in reasons and len(reasons) <= 2:
        return "silver", reasons
    return "silver", reasons


def run_build_gold_dataset(
    dataset_jsonl: str,
    out_gold: str,
    out_silver: str,
    out_bronze: str,
    out_report_json: str,
    out_report_html: str,
    feature_manifest_jsonl: str = "",
    drums_only: bool = True,
    min_gold_rows: int = 300,
) -> Dict[str, object]:
    rows = list(iter_jsonl(dataset_jsonl))
    if drums_only:
        rows = [r for r in rows if _role_from_audio_path(str(r.get("audio_path", ""))) == "drums"]

    dur_cache: Dict[str, float] = {}
    sig_cache: Dict[Tuple[str, float], Optional[Dict[str, float]]] = {}
    feature_manifest = _load_manifest(feature_manifest_jsonl)

    gold: List[Dict[str, object]] = []
    silver: List[Dict[str, object]] = []
    bronze: List[Dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()

    def _process_rows(src_rows: Sequence[Dict[str, object]], allow_non_drums: bool = False) -> None:
        for row in src_rows:
            ap = os.path.abspath(str(row.get("audio_path", "")).strip())
            if not ap:
                continue
            role = _role_from_audio_path(ap)
            if (not allow_non_drums) and role != "drums":
                continue

            if ap not in dur_cache:
                md = row.get("metadata") or {}
                d = as_float(md.get("duration_sec")) if isinstance(md, dict) else None
                if d is None or d <= 0:
                    d = _duration_sec(ap)
                dur_cache[ap] = float(d or 0.0)
            dur = float(dur_cache.get(ap, 0.0))
            tgt = as_float(row.get("target_sec"))
            key = (ap, float(round(float(tgt or 0.0), 5)))
            if key not in sig_cache and tgt is not None and tgt > 0 and os.path.isfile(ap):
                cache_path = feature_manifest.get(ap)
                if cache_path and os.path.isfile(cache_path):
                    sig_cache[key] = _audio_signature_from_cache(cache_path, float(tgt))
                else:
                    sig_cache[key] = _audio_signature(ap, float(tgt))
            sig = sig_cache.get(key)

            tier, reasons = _tier_row(row, sig=sig, duration_sec=dur)
            rr = dict(row)
            md = dict(rr.get("metadata") or {})
            md["tier"] = tier
            md["tier_reasons"] = list(reasons)
            md["role"] = role
            rr["metadata"] = md

            tier_counts[tier] += 1
            for rs in reasons:
                reason_counts[rs] += 1

            if tier == "gold":
                gold.append(rr)
            elif tier == "silver":
                silver.append(rr)
            else:
                bronze.append(rr)

    _process_rows(rows, allow_non_drums=not bool(drums_only))

    # If user requested drums-only but strict filtering is too aggressive, pull in non-drums GOLD rows.
    if bool(drums_only) and int(min_gold_rows) > 0 and len(gold) < int(min_gold_rows):
        all_rows = list(iter_jsonl(dataset_jsonl))
        non_drums = [r for r in all_rows if _role_from_audio_path(str(r.get("audio_path", ""))) != "drums"]
        before = len(gold)
        _process_rows(non_drums, allow_non_drums=True)
        # Keep only unique audio path rows in each bucket.
        def _dedupe(bucket: List[Dict[str, object]]) -> List[Dict[str, object]]:
            seen = set()
            out: List[Dict[str, object]] = []
            for r in bucket:
                ap = os.path.abspath(str(r.get("audio_path", "")))
                if ap in seen:
                    continue
                seen.add(ap)
                out.append(r)
            return out

        gold[:] = _dedupe(gold)
        silver[:] = _dedupe(silver)
        bronze[:] = _dedupe(bronze)
        after = len(gold)
        if after > before:
            reason_counts["fallback_added_non_drums"] += int(after - before)

    write_jsonl(out_gold, gold)
    write_jsonl(out_silver, silver)
    write_jsonl(out_bronze, bronze)

    report = {
        "ok": True,
        "dataset_in": os.path.abspath(dataset_jsonl),
        "drums_only": bool(drums_only),
        "gold_rows": int(len(gold)),
        "silver_rows": int(len(silver)),
        "bronze_rows": int(len(bronze)),
        "tier_counts": {k: int(v) for k, v in sorted(tier_counts.items())},
        "rejection_reasons": {k: int(v) for k, v in reason_counts.most_common()},
        "feature_manifest_rows": int(len(feature_manifest)),
        "out_gold": os.path.abspath(out_gold),
        "out_silver": os.path.abspath(out_silver),
        "out_bronze": os.path.abspath(out_bronze),
    }
    write_json(out_report_json, report)

    # Minimal HTML report.
    lines = [
        "<html><head><meta charset='utf-8'><title>ALSDrop Tier Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px}",
        "table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #333}</style></head><body>",
        "<h1>ALSDrop Dataset Tier Report</h1>",
        f"<p>Input: <code>{os.path.abspath(dataset_jsonl)}</code></p>",
        f"<p>GOLD: <b>{len(gold)}</b> | SILVER: <b>{len(silver)}</b> | BRONZE: <b>{len(bronze)}</b></p>",
        "<h2>Rejection Reasons</h2><table><tr><th>Reason</th><th>Count</th></tr>",
    ]
    for k, v in reason_counts.most_common():
        lines.append(f"<tr><td>{k}</td><td>{int(v)}</td></tr>")
    lines.extend(["</table>", "</body></html>"])
    os.makedirs(os.path.dirname(os.path.abspath(out_report_html)), exist_ok=True)
    with open(out_report_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build strict GOLD/SILVER/BRONZE datasets for ALSDrop training")
    ap.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    ap.add_argument("--out-gold", default="alsdrop/data/dataset_gold.jsonl")
    ap.add_argument("--out-silver", default="alsdrop/data/dataset_silver.jsonl")
    ap.add_argument("--out-bronze", default="alsdrop/data/dataset_bronze.jsonl")
    ap.add_argument("--report-json", default="alsdrop/outputs/dataset_tier_report.json")
    ap.add_argument("--report-html", default="alsdrop/outputs/dataset_tier_report.html")
    ap.add_argument("--features-manifest", default="", help="Optional feature manifest JSONL for fast signature checks")
    ap.add_argument("--no-drums-only", action="store_true", help="Allow all roles (drums/inst/vocals)")
    ap.add_argument("--min-gold-rows", type=int, default=300)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_build_gold_dataset(
        dataset_jsonl=args.dataset,
        out_gold=args.out_gold,
        out_silver=args.out_silver,
        out_bronze=args.out_bronze,
        out_report_json=args.report_json,
        out_report_html=args.report_html,
        feature_manifest_jsonl=args.features_manifest,
        drums_only=not bool(args.no_drums_only),
        min_gold_rows=int(args.min_gold_rows),
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
