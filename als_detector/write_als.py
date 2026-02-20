#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from als_detector.utils.als_io import (
    clip_audio_path,
    iter_audio_clips,
    read_als_root,
    replace_warp_markers_with_anchor,
    set_clip_audio_path,
    write_als_root,
)


def _duration_seconds_ffprobe(audio_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        out = (proc.stdout or b"").decode("utf-8", "ignore").strip().splitlines()
        if out:
            d = float(out[0].strip())
            if d > 0:
                return float(d)
    except Exception:
        pass
    return 0.0


def _select_target_clips(root, template_als: str, audio_path: str, apply_to_all: bool) -> List[object]:
    clips = list(iter_audio_clips(root))
    if apply_to_all:
        return clips

    audio_base = os.path.basename(audio_path).lower()
    audio_stem, _ = os.path.splitext(audio_base)

    matched = []
    for c in clips:
        ref = clip_audio_path(c, template_als, resolve=False)
        ref_base = os.path.basename(ref).lower()
        if ref_base == audio_base:
            matched.append(c)
            continue
        ref_stem, _ = os.path.splitext(ref_base)
        if ref_stem and ref_stem == audio_stem:
            matched.append(c)
            continue
        n = c.find("Name")
        nm = (n.get("Value", "") if n is not None else "").lower()
        if audio_stem and audio_stem in nm:
            matched.append(c)
    if matched:
        return matched
    return clips[:1] if clips else []


def write_predicted_anchor_to_als(
    template_als: str,
    out_als: str,
    audio_path: str,
    predicted_sec: float,
    bpm: float,
    apply_to_all_clips: bool = False,
) -> str:
    root = read_als_root(template_als)
    targets = _select_target_clips(root, template_als=template_als, audio_path=audio_path, apply_to_all=apply_to_all_clips)
    if not targets:
        raise RuntimeError("No AudioClip nodes found in template ALS.")

    duration_sec = _duration_seconds_ffprobe(audio_path)
    if duration_sec <= 0:
        raise RuntimeError(f"Could not determine audio duration for: {audio_path}")

    for clip in targets:
        set_clip_audio_path(clip, audio_path=audio_path, out_als_path=out_als)
        replace_warp_markers_with_anchor(
            clip,
            anchor_sec=float(predicted_sec),
            bpm=float(bpm),
            end_sec=float(duration_sec),
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_als)), exist_ok=True)
    write_als_root(out_als, root)
    return os.path.abspath(out_als)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write predicted 1.1.1 anchor into Ableton ALS template.")
    ap.add_argument("--template", required=True, help="Template .als file")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--pred-sec", required=True, type=float, help="Predicted 1.1.1 anchor in seconds")
    ap.add_argument("--bpm", required=True, type=float, help="Track tempo BPM")
    ap.add_argument("--out", required=True, help="Output .als path")
    ap.add_argument("--apply-to-all-clips", action="store_true", help="Apply anchor to every AudioClip in template")
    args = ap.parse_args()

    out = write_predicted_anchor_to_als(
        template_als=args.template,
        out_als=args.out,
        audio_path=args.audio,
        predicted_sec=float(args.pred_sec),
        bpm=float(args.bpm),
        apply_to_all_clips=bool(args.apply_to_all_clips),
    )
    print(f"[OK] Wrote ALS: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
