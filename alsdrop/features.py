#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import os
from typing import Dict, List, Optional, Tuple

from .audio_features import extract_features, feature_cache_path, save_feature_cache
from .constants import DEFAULT_HOP, DEFAULT_MELS, DEFAULT_SR
from .utils import as_float, iter_jsonl, write_jsonl


def _median(vals: List[float], default: float = 0.0) -> float:
    if not vals:
        return float(default)
    s = sorted(float(v) for v in vals)
    return float(s[len(s) // 2])


def _process_audio_item(
    args: Tuple[int, str, str, float, int, int, int, bool]
) -> Tuple[int, bool, Optional[Dict[str, object]], bool]:
    idx, ap, cpath, bpm_hint, sr, hop, n_mels, force = args
    if not os.path.isfile(ap):
        return idx, False, None, True

    if (not force) and os.path.isfile(cpath):
        return (
            idx,
            True,
            {
                "audio_path": ap,
                "cache_path": cpath,
                "bpm_hint": float(bpm_hint),
            },
            False,
        )

    try:
        feat = extract_features(audio_path=ap, sr=int(sr), hop_length=int(hop), n_mels=int(n_mels))
        save_feature_cache(
            path=cpath,
            feat=feat,
            meta={
                "audio_path": ap,
                "sr": int(sr),
                "hop": int(hop),
                "n_mels": int(n_mels),
            },
        )
    except Exception:
        return idx, False, None, True

    return (
        idx,
        True,
        {
            "audio_path": ap,
            "cache_path": cpath,
            "bpm_hint": float(bpm_hint),
        },
        False,
    )


def _default_workers() -> int:
    cpu = int(os.cpu_count() or 1)
    # Leave headroom for OS/audio decode I/O.
    return max(1, min(12, cpu - 2))


def run_features(
    dataset_jsonl: str,
    cache_dir: str,
    manifest_out: str,
    sr: int = DEFAULT_SR,
    hop: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_MELS,
    force: bool = False,
    limit: int = 0,
    workers: int = 0,
) -> Dict[str, object]:
    rows = list(iter_jsonl(dataset_jsonl))
    by_audio: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        ap = str(r.get("audio_path", "")).strip()
        if not ap:
            continue
        by_audio.setdefault(os.path.abspath(ap), []).append(r)

    items = sorted(by_audio.items())
    if limit and limit > 0:
        items = items[: int(limit)]

    os.makedirs(os.path.abspath(cache_dir), exist_ok=True)
    manifest_rows: List[Optional[Dict[str, object]]] = [None] * len(items)

    ok = 0
    fail = 0
    tasks: List[Tuple[int, str, str, float, int, int, int, bool]] = []
    for idx, (ap, group) in enumerate(items):
        bpm_hint = _median([float(v) for v in [as_float(g.get("bpm_hint")) for g in group] if v and v > 0], default=0.0)
        tasks.append((idx, ap, feature_cache_path(cache_dir, ap), float(bpm_hint), int(sr), int(hop), int(n_mels), bool(force)))

    total = len(tasks)
    use_workers = int(workers) if int(workers) > 0 else _default_workers()
    use_workers = max(1, min(use_workers, total or 1))

    if use_workers == 1:
        for i, t in enumerate(tasks, start=1):
            idx, is_ok, row, is_fail = _process_audio_item(t)
            if is_ok and row is not None:
                manifest_rows[idx] = row
                ok += 1
            elif is_fail:
                fail += 1
            if (i % 10 == 0) or (i == total):
                print(f"[features] {i}/{total} processed (ok={ok}, fail={fail})")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=use_workers) as ex:
            futs = [ex.submit(_process_audio_item, t) for t in tasks]
            for i, f in enumerate(concurrent.futures.as_completed(futs), start=1):
                idx, is_ok, row, is_fail = f.result()
                if is_ok and row is not None:
                    manifest_rows[idx] = row
                    ok += 1
                elif is_fail:
                    fail += 1
                if (i % 10 == 0) or (i == total):
                    print(f"[features] {i}/{total} processed (ok={ok}, fail={fail}, workers={use_workers})")

    out_rows = [r for r in manifest_rows if r is not None]
    write_jsonl(manifest_out, out_rows)
    return {
        "ok": True,
        "tracks_ok": int(ok),
        "tracks_failed": int(fail),
        "manifest": os.path.abspath(manifest_out),
        "cache_dir": os.path.abspath(cache_dir),
        "workers": int(use_workers),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract and cache ALSDrop audio features")
    ap.add_argument("--dataset", default="alsdrop/data/dataset.jsonl", help="Dataset JSONL from extract step")
    ap.add_argument("--cache_dir", default="alsdrop/data/features", help="Feature cache directory")
    ap.add_argument("--manifest", default="alsdrop/data/features_manifest.jsonl", help="Output feature manifest JSONL")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP)
    ap.add_argument("--n_mels", type=int, default=DEFAULT_MELS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_features(
        dataset_jsonl=args.dataset,
        cache_dir=args.cache_dir,
        manifest_out=args.manifest,
        sr=int(args.sr),
        hop=int(args.hop),
        n_mels=int(args.n_mels),
        force=bool(args.force),
        limit=int(args.limit),
        workers=int(args.workers),
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
