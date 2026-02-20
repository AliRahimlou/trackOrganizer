#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from als_detector.model import build_model, extract_window, stack_channels
from als_detector.utils.audio import extract_features_from_path
from als_detector.utils.candidates import generate_candidate_indices, madmom_downbeats, snap_to_nearest_downbeat
from als_detector.write_als import write_predicted_anchor_to_als


_BPM_IN_NAME_RE = re.compile(r"_(\d{2,3})_[0-9]{1,2}[ab](?:_|-)", re.I)


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for inference. Install with: pip install torch") from e
    return torch


def _predict_anchor_sec(
    model,
    torch,
    feature_dict: Dict[str, np.ndarray],
    audio_path: str,
    window_frames: int,
    device: str,
    use_madmom: bool,
) -> Dict[str, float]:
    cand_idx = generate_candidate_indices(
        feature_dict=feature_dict,
        audio_path=audio_path,
        use_madmom=use_madmom,
    )
    if cand_idx.size == 0:
        raise RuntimeError("No candidate frames generated.")

    channels = stack_channels(feature_dict)
    windows = np.stack([extract_window(channels, int(i), window_frames) for i in cand_idx], axis=0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    j = int(np.argmax(probs))
    pred_raw = float(feature_dict["frame_times"][int(cand_idx[j])])
    prob = float(probs[j])

    downbeats: List[float] = []
    if use_madmom:
        downbeats = [float(t) for t in madmom_downbeats(audio_path).tolist()]
    if not downbeats:
        downbeats = [float(t) for t in feature_dict.get("beat_times", np.asarray([], dtype=np.float32)).tolist()]
    pred_snap = snap_to_nearest_downbeat(pred_raw, downbeats)

    return {
        "pred_sec_raw": pred_raw,
        "pred_sec_snapped": float(pred_snap),
        "confidence": prob,
        "n_candidates": int(len(cand_idx)),
    }


def _bpm_from_filename(path: str) -> float:
    base = os.path.basename(path or "")
    stem, _ = os.path.splitext(base)
    m = _BPM_IN_NAME_RE.search(stem)
    if not m:
        return 0.0
    try:
        v = float(int(m.group(1)))
    except Exception:
        return 0.0
    return v if v > 0 else 0.0


def _normalize_edm_bpm(bpm: float) -> float:
    v = float(bpm)
    if v <= 0:
        return 0.0
    while v < 118.0:
        v *= 2.0
    while v > 190.0:
        v *= 0.5
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer ALS 1.1.1 anchor and optionally write warped ALS.")
    ap.add_argument("--audio", required=True, help="Input audio path (WAV/FLAC/etc)")
    ap.add_argument("--template", required=True, help="Template .als path")
    ap.add_argument("--model", default="als_detector/models/anchor_cnn.pt", help="Trained model checkpoint")
    ap.add_argument("--out", default=None, help="Output .als path. Default: outputs/<audio>_warped.als")
    ap.add_argument("--bpm", type=float, default=0.0, help="Optional BPM override for writing ALS")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--use-madmom", action="store_true", help="Use madmom downbeats for candidate/snap")
    ap.add_argument("--review-threshold", type=float, default=0.55, help="Confidence threshold below which manual review is recommended")
    ap.add_argument("--plot", default="", help="Optional PNG path for inference visualization.")
    args = ap.parse_args()

    torch = _require_torch()
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = ckpt.get("config", {})
    in_channels = int(cfg.get("in_channels", 99))
    window_frames = int(cfg.get("window_frames", 172))
    sr = int(cfg.get("sr", 22050))
    hop_length = int(cfg.get("hop_length", 512))
    n_fft = int(cfg.get("n_fft", 2048))
    n_mels = int(cfg.get("n_mels", 96))

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model = build_model(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])

    feats = extract_features_from_path(
        args.audio,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
    )
    pred = _predict_anchor_sec(
        model=model,
        torch=torch,
        feature_dict=feats,
        audio_path=args.audio,
        window_frames=window_frames,
        device=device,
        use_madmom=bool(args.use_madmom or cfg.get("use_madmom", False)),
    )

    bpm_name = _bpm_from_filename(args.audio)
    bpm_est = float(feats.get("tempo_est", np.asarray([128.0], dtype=np.float32))[0])
    bpm = float(args.bpm) if args.bpm and args.bpm > 0 else 0.0
    if bpm <= 0 and bpm_name > 0:
        bpm = float(bpm_name)
    if bpm <= 0:
        bpm = _normalize_edm_bpm(bpm_est)
    if bpm <= 0:
        bpm = 128.0
    beat_sec = 60.0 / max(1e-9, bpm)
    bar_number = int(np.floor(float(pred["pred_sec_snapped"]) / (beat_sec * 4.0))) + 1

    if args.out:
        out_als = args.out
    else:
        audio_stem = os.path.splitext(os.path.basename(args.audio))[0]
        out_als = os.path.abspath(os.path.join("als_detector", "outputs", f"{audio_stem}_warped.als"))

    out_written = write_predicted_anchor_to_als(
        template_als=args.template,
        out_als=out_als,
        audio_path=args.audio,
        predicted_sec=float(pred["pred_sec_snapped"]),
        bpm=float(bpm),
        apply_to_all_clips=False,
    )

    report = {
        "audio": os.path.abspath(args.audio),
        "template": os.path.abspath(args.template),
        "output_als": out_written,
        "pred_sec_raw": pred["pred_sec_raw"],
        "pred_sec_snapped": pred["pred_sec_snapped"],
        "ableton_cue_sec": pred["pred_sec_snapped"],
        "bar_number": int(max(1, bar_number)),
        "confidence": pred["confidence"],
        "n_candidates": pred["n_candidates"],
        "bpm_used": bpm,
        "needs_manual_review": bool(pred["confidence"] < float(args.review_threshold)),
    }
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import librosa  # type: ignore

            y, sr_plot = librosa.load(args.audio, sr=sr, mono=True)
            t = np.arange(len(y), dtype=np.float32) / float(sr_plot)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            ax1.plot(t, y, linewidth=0.5, color="#445")
            ax1.axvline(float(pred["pred_sec_snapped"]), color="#d22", linestyle="--", linewidth=1.5)
            ax1.set_title("Waveform + Predicted 1.1.1")
            ax1.set_ylabel("amp")

            ft = feats["frame_times"]
            rms = feats["rms"]
            low = feats["low_energy"]
            if len(rms):
                rms_n = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
            else:
                rms_n = rms
            if len(low):
                low_n = (low - np.min(low)) / (np.max(low) - np.min(low) + 1e-9)
            else:
                low_n = low
            ax2.plot(ft, rms_n, color="#2c7", linewidth=1.0, label="RMS")
            ax2.plot(ft, low_n, color="#fa0", linewidth=1.0, label="Low(20-150Hz)")
            ax2.axvline(float(pred["pred_sec_snapped"]), color="#d22", linestyle="--", linewidth=1.5)
            ax2.set_ylabel("norm")
            ax2.set_xlabel("seconds")
            ax2.legend(loc="upper right")
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=130)
            plt.close(fig)
            report["plot_path"] = os.path.abspath(args.plot)
        except Exception as e:
            report["plot_error"] = str(e)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
