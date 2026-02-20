#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

from .als_io import (
    read_als_root,
    rewrite_clip_warp_markers,
    select_target_clips,
    set_clip_audio_path,
    validate_als,
    write_als_root,
)
from .utils import as_float, read_json


def audio_duration_seconds(audio_path: str) -> float:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(audio_path)
        if info and info.frames and info.samplerate and info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        return float(librosa.get_duration(path=audio_path))
    except Exception:
        return 0.0


def run_write_als(
    template_als: str,
    audio_path: str,
    predicted_json: Optional[str],
    out_als: str,
    predicted_sec: Optional[float] = None,
    bpm_override: Optional[float] = None,
    apply_to_all: bool = False,
) -> Dict[str, object]:
    pred_sec = as_float(predicted_sec)
    bpm = as_float(bpm_override)

    if predicted_json:
        p = read_json(predicted_json)
        if pred_sec is None:
            pred_sec = as_float(p.get("predicted_sec"), as_float(p.get("pred_sec"), as_float(p.get("ableton_cue_sec"))))
        if bpm is None:
            bpm = as_float(p.get("bpm_used"), as_float(p.get("bpm"), 128.0))

    if pred_sec is None:
        raise RuntimeError("predicted_sec is required (via --pred-sec or --pred)")
    if bpm is None or bpm <= 0:
        bpm = 128.0

    root = read_als_root(template_als)
    targets = select_target_clips(root, template_als=template_als, audio_path=audio_path, apply_to_all=bool(apply_to_all))
    if not targets:
        raise RuntimeError("No AudioClip nodes found in template ALS.")

    dur = audio_duration_seconds(audio_path)
    if dur <= 0:
        raise RuntimeError(f"Could not determine duration for audio: {audio_path}")

    for clip in targets:
        set_clip_audio_path(clip, audio_path=audio_path, out_als_path=out_als)
        rewrite_clip_warp_markers(
            root=root,
            clip=clip,
            target_sec=float(pred_sec),
            bpm=float(bpm),
            duration_sec=float(dur),
        )

    check = validate_als(root, expected_target_sec=float(pred_sec))
    if check.get("duplicate_list_ids"):
        raise RuntimeError(f"Template has duplicate ListId values: {check['duplicate_list_ids'][:8]}")

    write_als_root(out_als, root)
    return {
        "ok": True,
        "output_als": os.path.abspath(out_als),
        "audio_path": os.path.abspath(audio_path),
        "predicted_sec": float(pred_sec),
        "bpm_used": float(bpm),
        "duration_sec": float(dur),
        "clips_updated": int(len(targets)),
        "validation": check,
    }


def run_validate_als(als_path: str, expected_target_sec: Optional[float] = None) -> Dict[str, object]:
    root = read_als_root(als_path)
    check = validate_als(root, expected_target_sec=expected_target_sec)
    check["als_path"] = os.path.abspath(als_path)
    return check


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Write predicted 1.1.1 anchor into Ableton ALS template")
    ap.add_argument("--template", required=True, help="Template ALS file")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--pred", default="", help="Predicted JSON path from infer step")
    ap.add_argument("--pred-sec", type=float, default=None, help="Direct predicted seconds override")
    ap.add_argument("--bpm", type=float, default=None, help="BPM override")
    ap.add_argument("--out", required=True, help="Output ALS path")
    ap.add_argument("--apply-to-all", action="store_true", help="Apply to all clips in template")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_write_als(
        template_als=args.template,
        audio_path=args.audio,
        predicted_json=args.pred.strip() or None,
        out_als=args.out,
        predicted_sec=args.pred_sec,
        bpm_override=args.bpm,
        apply_to_all=bool(args.apply_to_all),
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
