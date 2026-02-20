#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from als_detector.utils.audio import audio_id, extract_features_from_path, save_feature_cache


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _median_bpm(rows: List[Dict[str, object]]) -> float:
    vals = []
    for r in rows:
        v = r.get("bpm")
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            vals.append(fv)
    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract cached audio features for ALS label dataset.")
    ap.add_argument("--dataset", default="als_detector/data/labels.jsonl", help="Input labels JSONL.")
    ap.add_argument("--cache-dir", default="als_detector/data/features", help="Feature cache directory.")
    ap.add_argument("--manifest", default="als_detector/data/features_manifest.jsonl", help="Output feature manifest JSONL.")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--n-mels", type=int, default=96)
    ap.add_argument("--force", action="store_true", help="Recompute existing cache files.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of unique audio files to process.")
    args = ap.parse_args()

    labels = _load_jsonl(args.dataset)
    by_audio: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in labels:
        p = str(r.get("audio_path", "")).strip()
        if not p:
            continue
        by_audio[p].append(r)
    audio_items = sorted(by_audio.items())
    if args.limit and args.limit > 0:
        audio_items = audio_items[: int(args.limit)]

    os.makedirs(os.path.abspath(args.cache_dir), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)

    rows_out: List[Dict[str, object]] = []
    ok = 0
    fail = 0
    for audio_path, group in audio_items:
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio: {audio_path}")
            fail += 1
            continue
        aid = audio_id(audio_path)
        cache_path = os.path.abspath(os.path.join(args.cache_dir, f"{aid}.npz"))
        if (not args.force) and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=True) as z:
                frame_times = z["frame_times"]
                tempo_est = float(z["tempo_est"][0]) if "tempo_est" in z else 0.0
            duration_sec = float(frame_times[-1]) if len(frame_times) else 0.0
            rows_out.append(
                {
                    "audio_path": os.path.abspath(audio_path),
                    "cache_path": cache_path,
                    "audio_id": aid,
                    "duration_sec": duration_sec,
                    "tempo_est": tempo_est,
                    "bpm_hint": _median_bpm(group),
                }
            )
            ok += 1
            continue

        try:
            feats = extract_features_from_path(
                audio_path,
                sr=args.sr,
                hop_length=args.hop_length,
                n_fft=args.n_fft,
                n_mels=args.n_mels,
            )
            duration_sec = float(feats["frame_times"][-1]) if len(feats["frame_times"]) else 0.0
            tempo_est = float(feats["tempo_est"][0]) if len(feats["tempo_est"]) else 0.0
            save_feature_cache(
                cache_path=cache_path,
                features=feats,
                meta={
                    "audio_path": os.path.abspath(audio_path),
                    "audio_id": aid,
                    "sr": args.sr,
                    "hop_length": args.hop_length,
                    "n_fft": args.n_fft,
                    "n_mels": args.n_mels,
                },
            )
        except Exception as e:
            print(f"[WARN] Feature extraction failed for {audio_path}: {e}")
            fail += 1
            continue

        rows_out.append(
            {
                "audio_path": os.path.abspath(audio_path),
                "cache_path": cache_path,
                "audio_id": aid,
                "duration_sec": duration_sec,
                "tempo_est": tempo_est,
                "bpm_hint": _median_bpm(group),
            }
        )
        ok += 1

    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"[OK] Feature cache complete: {ok} ok, {fail} failed")
    print(f"[OK] Manifest: {os.path.abspath(args.manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
