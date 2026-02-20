#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np

from .als_io import dedupe_labels, extract_labels_from_als
from .constants import DATASET_VERSION
from .report import write_dataset_report
from .utils import as_float, write_jsonl


def _collect_als_paths(als_dir: Optional[str], explicit: List[str], globs: List[str]) -> List[str]:
    out: List[str] = []
    if als_dir:
        root = os.path.abspath(als_dir)
        for p in glob.glob(os.path.join(root, "**", "*.als"), recursive=True):
            if os.path.isfile(p):
                out.append(os.path.abspath(p))
    for p in explicit:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
    for patt in globs:
        for p in glob.glob(patt, recursive=True):
            if os.path.isfile(p) and p.lower().endswith(".als"):
                out.append(os.path.abspath(p))
    return sorted(set(out))


def _audio_duration_sec(path: str) -> float:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(path)
        if info and info.frames and info.samplerate and info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def _sanitize_rows(rows: List[Dict[str, object]], min_tail_sec: float = 5.0) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        ap = str(r.get("audio_path", "")).strip()
        target = as_float(r.get("target_sec"))
        if not ap or target is None:
            continue
        if target <= 0.0:
            continue

        dur = _audio_duration_sec(ap) if os.path.isfile(ap) else 0.0
        if dur > 0.0 and target >= (dur - float(min_tail_sec)):
            continue

        rr = dict(r)
        rr["audio_path"] = os.path.abspath(ap)
        rr["target_sec"] = float(target)
        rr["dataset_version"] = int(DATASET_VERSION)
        md = dict(rr.get("metadata") or {})
        if dur > 0.0:
            md["duration_sec"] = float(dur)
        rr["metadata"] = md
        out.append(rr)
    return out


def run_extract(
    als_dir: Optional[str],
    als_paths: List[str],
    als_globs: List[str],
    out_jsonl: str,
    include_unwarped: bool = False,
    no_resolve_audio: bool = False,
    report_html: Optional[str] = None,
) -> Dict[str, object]:
    paths = _collect_als_paths(als_dir=als_dir, explicit=als_paths, globs=als_globs)
    if not paths:
        raise RuntimeError("No .als files found. Pass --als_dir, --als, or --als-glob.")

    rows: List[Dict[str, object]] = []
    failed = 0
    for p in paths:
        try:
            labels = extract_labels_from_als(
                p,
                resolve_paths=not bool(no_resolve_audio),
                include_unwarped=bool(include_unwarped),
            )
        except Exception:
            failed += 1
            continue
        rows.extend(lbl.to_row() for lbl in labels)

    rows = _sanitize_rows(rows)
    rows = dedupe_labels(rows)
    rows = _sanitize_rows(rows)

    write_jsonl(out_jsonl, rows)

    if report_html:
        write_dataset_report(rows, report_html)

    n_audio = len({str(r.get("audio_path", "")) for r in rows})
    return {
        "ok": True,
        "als_total": len(paths),
        "als_failed": int(failed),
        "rows": len(rows),
        "unique_audio": int(n_audio),
        "out": os.path.abspath(out_jsonl),
        "report": os.path.abspath(report_html) if report_html else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract ALS-supervised 1.1.1 labels from Ableton .als files.")
    ap.add_argument("--als_dir", default="", help="Directory to recursively scan for .als")
    ap.add_argument("--als", action="append", default=[], help="Explicit .als path (repeatable)")
    ap.add_argument("--als-glob", action="append", default=[], help="Glob pattern for .als files (repeatable)")
    ap.add_argument("--out", default="alsdrop/data/dataset.jsonl", help="Output dataset JSONL")
    ap.add_argument("--report", default="", help="Optional dataset report HTML path")
    ap.add_argument("--include-unwarped", action="store_true", help="Include clips with IsWarped=false")
    ap.add_argument("--no-resolve-audio", action="store_true", help="Do not resolve relative audio paths")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    report = args.report.strip()
    if not report:
        report = os.path.join(os.path.dirname(os.path.abspath(args.out)), "dataset_report.html")

    res = run_extract(
        als_dir=args.als_dir.strip() or None,
        als_paths=list(args.als or []),
        als_globs=list(args.als_glob or []),
        out_jsonl=args.out,
        include_unwarped=bool(args.include_unwarped),
        no_resolve_audio=bool(args.no_resolve_audio),
        report_html=report,
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
