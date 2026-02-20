#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio_features import extract_features
from .candidates import bar_number_from_sec, generate_candidates, snap_to_nearest_downbeat
from .constants import DEFAULT_HOP, DEFAULT_MELS, DEFAULT_SR
from .model import ContextConfig, build_candidate_contexts, build_model, sigmoid_np, temperature_scale_logits
from .utils import as_float, parent_dir


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for inference. Install with: pip install torch") from e
    return torch


def _device_auto(torch, device_arg: str) -> str:
    if device_arg != "auto":
        return str(device_arg)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _candidate_meta(
    feature_dict: Dict[str, np.ndarray],
    candidate_times: np.ndarray,
    candidate_conf: np.ndarray,
    tempo_bpm: float,
) -> np.ndarray:
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)
    dur = float(frame_times[-1]) if frame_times.size else 1.0
    onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    low_ratio = feature_dict.get("low_ratio", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    n = int(candidate_times.shape[0])
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out

    out[:, 0] = candidate_conf[:n].astype(np.float32, copy=False) if candidate_conf.size else 0.0
    out[:, 1] = candidate_times.astype(np.float32, copy=False) / max(1e-6, dur)
    if frame_times.size and low_ratio.size:
        out[:, 2] = np.interp(candidate_times, frame_times, low_ratio, left=low_ratio[0], right=low_ratio[-1]).astype(np.float32, copy=False)
    if frame_times.size and onset.size:
        out[:, 3] = np.interp(candidate_times, frame_times, onset, left=onset[0], right=onset[-1]).astype(np.float32, copy=False)

    # tempo hint included by mild scaling in conf channel
    if tempo_bpm > 0:
        out[:, 0] = np.clip(out[:, 0] + (float(tempo_bpm) / 220.0) * 0.05, 0.0, 1.0)
    return out


def _norm01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    span = max(1e-9, hi - lo)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _slice_mean(series: np.ndarray, times: np.ndarray, a: float, b: float) -> float:
    if series.size == 0 or times.size == 0:
        return 0.0
    i0 = int(np.searchsorted(times, float(a), side="left"))
    i1 = int(np.searchsorted(times, float(b), side="right"))
    i0 = max(0, min(i0, len(series)))
    i1 = max(0, min(i1, len(series)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(series[i0:i1]))


def _safe_fallback_candidate(
    feature_dict: Dict[str, np.ndarray],
    candidate_times: np.ndarray,
    candidate_conf: np.ndarray,
    downbeats: np.ndarray,
    duration_sec: float,
    region: Tuple[float, float],
) -> Tuple[float, float]:
    frame_times = feature_dict.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    onset = _norm01(feature_dict.get("onset", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    low = _norm01(feature_dict.get("low_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    novelty = _norm01(feature_dict.get("novelty", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False))
    cand_conf_n = _norm01(candidate_conf.astype(np.float32, copy=False)) if candidate_conf.size else candidate_conf

    if candidate_times.size == 0:
        return float(0.0), 0.0

    lo_t, hi_t = float(region[0]), float(region[1])
    rows: List[Dict[str, float]] = []
    for i, t in enumerate(candidate_times.tolist()):
        tt = float(t)
        if tt < 0.5:
            continue
        if tt > max(0.0, float(duration_sec) - 1.0):
            continue
        if tt < lo_t or tt > hi_t:
            continue

        pre8 = _slice_mean(onset, frame_times, tt - 16.0, tt - 8.0)
        pre4 = _slice_mean(onset, frame_times, tt - 8.0, tt - 2.0)
        pre2_on = _slice_mean(onset, frame_times, tt - 2.0, tt)
        post2_on = _slice_mean(onset, frame_times, tt, tt + 2.0)

        pre2_low = _slice_mean(low, frame_times, tt - 2.0, tt)
        post2_low = _slice_mean(low, frame_times, tt, tt + 2.0)
        post8_low = _slice_mean(low, frame_times, tt + 2.0, tt + 8.0)
        nov_jump = _slice_mean(novelty, frame_times, tt - 1.0, tt + 1.5)

        buildup = max(0.0, pre4 - pre8)
        jump_on = max(0.0, post2_on - pre2_on)
        jump_low = max(0.0, post2_low - pre2_low)
        sustain = max(0.0, post8_low - pre2_low)
        conf = float(cand_conf_n[i]) if i < len(cand_conf_n) else 0.0

        rows.append(
            {
                "t": tt,
                "jump_low": float(jump_low),
                "jump_on": float(jump_on),
                "sustain": float(sustain),
                "buildup": float(buildup),
                "nov_jump": float(max(0.0, nov_jump)),
                "cand_conf": float(conf),
            }
        )

    if not rows:
        return float(candidate_times[0]), 0.0

    def _rank01(vals: np.ndarray) -> np.ndarray:
        n = int(vals.size)
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        order = np.argsort(vals, kind="mergesort")
        out = np.zeros(n, dtype=np.float32)
        out[order] = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        return out

    arr_jump_low = np.asarray([r["jump_low"] for r in rows], dtype=np.float32)
    arr_jump_on = np.asarray([r["jump_on"] for r in rows], dtype=np.float32)
    arr_sustain = np.asarray([r["sustain"] for r in rows], dtype=np.float32)
    arr_buildup = np.asarray([r["buildup"] for r in rows], dtype=np.float32)
    arr_nov = np.asarray([r["nov_jump"] for r in rows], dtype=np.float32)
    arr_conf = np.asarray([r["cand_conf"] for r in rows], dtype=np.float32)

    rk_low = _rank01(arr_jump_low)
    rk_on = _rank01(arr_jump_on)
    rk_sus = _rank01(arr_sustain)
    rk_bui = _rank01(arr_buildup)
    rk_nov = _rank01(arr_nov)
    rk_conf = _rank01(arr_conf)

    best_idx = 0
    best_score = -1e9
    best_hits = 0
    best_core_rank = 0.0
    core_thr = 0.72
    for i, r in enumerate(rows):
        core_hits = int(rk_low[i] >= core_thr) + int(rk_on[i] >= core_thr) + int(rk_sus[i] >= core_thr) + int(rk_bui[i] >= core_thr)
        core_rank_mean = float((rk_low[i] + rk_on[i] + rk_sus[i] + rk_bui[i]) / 4.0)
        novelty_boost = float(rk_nov[i])
        conf_boost = float(rk_conf[i])

        # Relative candidate scoring emphasizes structural drop signatures.
        score = (
            0.30 * float(rk_low[i])
            + 0.24 * float(rk_on[i])
            + 0.22 * float(rk_sus[i])
            + 0.14 * float(rk_bui[i])
            + 0.06 * novelty_boost
            + 0.04 * conf_boost
        )
        # Enforce "top percentile on >=2 core signals" as a strong preference.
        if core_hits < 2:
            score -= 0.14
        if score > best_score:
            best_score = float(score)
            best_idx = int(i)
            best_hits = int(core_hits)
            best_core_rank = float(core_rank_mean)

    best_t = float(rows[best_idx]["t"])
    best_s = float(best_score)

    if downbeats.size:
        best_t = snap_to_nearest_downbeat(float(best_t), downbeats.tolist())
    conf = float(1.0 / (1.0 + np.exp(-5.0 * (best_s - 0.58))))
    if best_hits < 2:
        conf *= 0.40
    elif best_core_rank < 0.65:
        conf *= 0.70
    return float(best_t), float(np.clip(conf, 0.0, 1.0))


def _region_window(duration_sec: float, prior: Dict[str, float]) -> Tuple[float, float]:
    dur = max(1e-6, float(duration_sec))
    p05 = float(prior.get("norm_p05", 0.06))
    p95 = float(prior.get("norm_p95", 0.90))
    lo_norm = max(0.0, min(1.0, p05 - 0.06))
    hi_norm = max(0.0, min(1.0, p95 + 0.06))
    lo = max(0.5, lo_norm * dur)
    hi = min(max(lo + 0.5, dur - 1.0), hi_norm * dur)
    # Always enforce coarse anti-intro/anti-tail limits.
    lo = max(lo, min(10.0, 0.04 * dur))
    hi = min(hi, dur - 10.0 if dur > 20.0 else dur - 1.0)
    return float(lo), float(max(lo + 0.5, hi))


def _predict_internal(
    audio_path: str,
    model_path: str,
    device: str,
    use_madmom: bool,
    bpm_override: Optional[float],
    review_threshold: float,
    return_candidates: bool,
    top_k: int = 5,
) -> Dict[str, object]:
    torch = _require_torch()
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = dict(ckpt.get("config") or {})

    sr = int(cfg.get("sr", DEFAULT_SR))
    hop = int(cfg.get("hop", DEFAULT_HOP))
    n_mels = int(cfg.get("n_mels", DEFAULT_MELS))
    context = ContextConfig(
        long_left_sec=float(cfg.get("long_left_sec", 12.0)),
        long_right_sec=float(cfg.get("long_right_sec", 6.0)),
        short_left_sec=float(cfg.get("short_left_sec", 2.0)),
        short_right_sec=float(cfg.get("short_right_sec", 2.0)),
        long_frames=int(cfg.get("long_frames", 384)),
        short_frames=int(cfg.get("short_frames", 128)),
    )

    in_channels = int(cfg.get("in_channels", n_mels + 9))
    meta_dim = int(cfg.get("meta_dim", 4))
    width = int(cfg.get("width", 128))
    dropout = float(cfg.get("dropout", 0.15))
    max_offset_sec = float(cfg.get("max_offset_sec", 0.150))

    model = build_model(
        in_channels=in_channels,
        meta_dim=meta_dim,
        width=width,
        dropout=dropout,
        max_offset_sec=max_offset_sec,
    )
    model.load_state_dict(ckpt["state_dict"])

    dev = _device_auto(torch, device)
    model.to(dev)
    model.eval()

    feat = extract_features(audio_path=audio_path, sr=sr, hop_length=hop, n_mels=n_mels)
    cset = generate_candidates(feature_dict=feat, audio_path=audio_path, use_madmom=bool(use_madmom))
    if cset.times.size == 0:
        raise RuntimeError("No candidate downbeats generated for this track")

    long_ctx, short_ctx = build_candidate_contexts(feat, cset.times.tolist(), cfg=context)
    meta = _candidate_meta(feat, cset.times, cset.confidence, cset.tempo_bpm)

    xb_long = torch.from_numpy(long_ctx).to(device=dev, dtype=torch.float32)
    xb_short = torch.from_numpy(short_ctx).to(device=dev, dtype=torch.float32)
    xb_meta = torch.from_numpy(meta).to(device=dev, dtype=torch.float32)

    with torch.no_grad():
        logits, offset = model(xb_long, xb_short, xb_meta)
        logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        off_np = offset.detach().cpu().numpy().astype(np.float32, copy=False)

    temperature = float(ckpt.get("temperature", 1.0))
    probs = sigmoid_np(temperature_scale_logits(logits_np, temperature)).astype(np.float32, copy=False)
    order = np.argsort(probs)[::-1]
    j = int(order[0])
    top1 = float(probs[j])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = float(top1 - top2)
    cand_t = float(cset.times[j])
    refined = float(cand_t + off_np[j])

    # Musical guardrails: keep refinement local to downbeat.
    if cset.tempo_bpm > 0:
        beat_sec = 60.0 / float(cset.tempo_bpm)
    else:
        beat_sec = 0.5
    if abs(refined - cand_t) > (0.22 * beat_sec):
        refined = cand_t

    duration_sec = float(feat.get("frame_times", np.asarray([], dtype=np.float32))[-1]) if "frame_times" in feat and len(feat["frame_times"]) else 0.0
    guardrails = dict(ckpt.get("guardrails") or {})
    prior = dict(ckpt.get("drop_region_prior") or {})
    min_conf = float(guardrails.get("min_confidence", max(0.55, float(review_threshold))))
    min_margin = float(guardrails.get("min_margin", 0.05))
    region_lo, region_hi = _region_window(duration_sec, prior=prior)
    region_valid = bool(region_lo <= float(cand_t) <= region_hi)

    accept_model = bool((top1 >= min_conf) and (margin >= min_margin) and region_valid)

    selected_by = "model"
    fallback_sec = None
    fallback_conf = None
    if not accept_model:
        fb_sec, fb_conf = _safe_fallback_candidate(
            feature_dict=feat,
            candidate_times=cset.times,
            candidate_conf=cset.confidence,
            downbeats=cset.downbeats,
            duration_sec=float(duration_sec),
            region=(float(region_lo), float(region_hi)),
        )
        if fb_sec > 0:
            fallback_sec = float(fb_sec)
            fallback_conf = float(fb_conf)
            cand_t = float(fb_sec)
            refined = float(fb_sec)
            selected_by = "safe_fallback"
        else:
            selected_by = "model_rejected"

    snapped = snap_to_nearest_downbeat(refined, cset.downbeats.tolist())

    bpm_used = as_float(bpm_override)
    if bpm_used is None or bpm_used <= 0:
        bpm_used = as_float(cset.tempo_bpm, 0.0)
    if bpm_used is None or bpm_used <= 0:
        bpm_used = as_float(feat.get("tempo_est", np.asarray([128.0], dtype=np.float32))[0], 128.0)
    if bpm_used <= 0:
        bpm_used = 128.0

    top_order = order[: max(1, int(top_k))]
    top = [
        {
            "sec": float(cset.times[i]),
            "prob": float(probs[i]),
            "offset_sec": float(off_np[i]),
            "refined_sec": float(cset.times[i] + off_np[i]),
        }
        for i in top_order.tolist()
    ]

    effective_conf = float(top1 if selected_by == "model" else (fallback_conf if fallback_conf is not None else 0.0))
    fallback_review_threshold = max(float(review_threshold), 0.68)
    if selected_by == "model":
        needs_review = bool(effective_conf < float(review_threshold))
    else:
        needs_review = bool((fallback_conf is None) or (float(fallback_conf) < float(fallback_review_threshold)))
    res = {
        "audio_path": os.path.abspath(audio_path),
        "predicted_sec": float(snapped),
        "candidate_sec": float(cand_t),
        "refined_sec": float(refined),
        "ableton_cue_sec": float(snapped),
        "confidence": float(effective_conf),
        "model_confidence": float(top1),
        "score_margin": float(margin),
        "guardrail_accept": bool(accept_model),
        "region_valid": bool(region_valid),
        "selected_by": str(selected_by),
        "fallback_sec": float(fallback_sec) if fallback_sec is not None else None,
        "fallback_confidence": float(fallback_conf) if fallback_conf is not None else None,
        "needs_manual_review": bool(needs_review),
        "bar_number": int(bar_number_from_sec(snapped, float(bpm_used))),
        "bpm_used": float(bpm_used),
        "tempo_est": float(cset.tempo_bpm),
        "n_candidates": int(cset.times.shape[0]),
        "temperature": float(temperature),
        "guardrail_thresholds": {"min_confidence": float(min_conf), "min_margin": float(min_margin)},
        "fallback_review_threshold": float(fallback_review_threshold),
        "region_window_sec": [float(region_lo), float(region_hi)],
        "top_candidates": top,
        "model_path": os.path.abspath(model_path),
    }
    if bool(return_candidates):
        res["candidate_times"] = [float(x) for x in cset.times.tolist()]
        res["candidate_confidences"] = [float(x) for x in cset.confidence.tolist()]
        res["downbeats"] = [float(x) for x in cset.downbeats.tolist()]
    return res


def run_predict(
    audio_path: str,
    model_path: str,
    out_json: Optional[str] = None,
    device: str = "auto",
    use_madmom: bool = True,
    bpm_override: Optional[float] = None,
    review_threshold: float = 0.55,
    return_candidates: bool = False,
) -> Dict[str, object]:
    res = _predict_internal(
        audio_path=audio_path,
        model_path=model_path,
        device=device,
        use_madmom=use_madmom,
        bpm_override=bpm_override,
        review_threshold=review_threshold,
        return_candidates=bool(return_candidates),
    )
    if out_json:
        parent_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        res["out_json"] = os.path.abspath(out_json)
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Predict ALS drop anchor (1.1.1) for an audio file")
    ap.add_argument("--audio", required=True, help="Input WAV/FLAC")
    ap.add_argument("--model", default="alsdrop/models/model.pt", help="Trained model checkpoint")
    ap.add_argument("--out", default="", help="Output JSON path")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--no-madmom", action="store_true", help="Disable madmom downbeat proposals")
    ap.add_argument("--bpm", type=float, default=0.0, help="Optional BPM override")
    ap.add_argument("--review-threshold", type=float, default=0.55)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_predict(
        audio_path=args.audio,
        model_path=args.model,
        out_json=args.out.strip() or None,
        device=args.device,
        use_madmom=not bool(args.no_madmom),
        bpm_override=args.bpm if args.bpm > 0 else None,
        review_threshold=float(args.review_threshold),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
