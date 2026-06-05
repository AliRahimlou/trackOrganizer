#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from drop_aligner.als import modify_als
from drop_aligner.debug import default_candidate_json, default_debug_plot, write_candidate_debug_json
from drop_aligner.detector import DropDetectorConfig, detect_drop, extract_features
from drop_aligner.exclusions import EXCLUDED_DIR_NAMES, is_excluded_path
from verify_als import verify_als


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}
BPM_RE = re.compile(r"^(?:drums|inst|vocals)_(\d{2,3})_", re.IGNORECASE)
SUMMARY_COLUMNS = [
    "filename",
    "detected_drop_time",
    "confidence",
    "confidence_tier",
    "selected_by",
    "sustained_full_groove_score",
    "immediate_groove_start_score",
    "groove_stability",
    "pre_drop_contrast",
    "drumprint_pattern_score",
    "fake_hit_penalty",
    "micro_confidence",
    "snap_offset_ms",
    "microaligned_time",
    "output_als",
    "candidates_json",
    "debug_png",
    "als_valid",
    "als_validation_error",
    "status",
    "error",
]


def _parse_cues(values: Optional[List[str]]) -> List[float]:
    out: List[float] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def _format_float(value: object, digits: int = 6) -> str:
    if value in (None, ""):
        return ""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def _matches_stem_role(path: Path, stem_role: Optional[str]) -> bool:
    if not stem_role:
        return True
    return path.name.lower().startswith(str(stem_role).lower())


def _discover_audio(
    folder: Path,
    recursive: bool,
    stem_role: Optional[str] = None,
    excluded_dir_names: Iterable[str] = EXCLUDED_DIR_NAMES,
) -> List[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and _matches_stem_role(path, stem_role)
        and not is_excluded_path(path, excluded_dir_names=excluded_dir_names)
    ]
    return sorted(files, key=lambda path: str(path).lower())


def _infer_bpm_from_path(path: Path) -> Optional[float]:
    match = BPM_RE.match(path.name)
    if match:
        return float(match.group(1))
    for parent in [path.parent, *path.parents]:
        try:
            bpm = int(parent.name)
        except ValueError:
            continue
        if 60 <= bpm <= 220:
            return float(bpm)
    return None


def _drop_aligned_als(audio_path: str) -> str:
    audio = Path(audio_path)
    return str(audio.with_name(f"{audio.stem}_DROP_ALIGNED.als"))


def _expected_outputs(audio_path: str) -> Dict[str, str]:
    return {
        "output_als": _drop_aligned_als(audio_path),
        "candidates_json": default_candidate_json(audio_path),
        "debug_png": default_debug_plot(audio_path),
    }


def _is_processed(paths: Dict[str, str], debug_candidates: bool) -> bool:
    required = [paths["output_als"], paths["candidates_json"]]
    if debug_candidates:
        required.append(paths["debug_png"])
    return all(Path(path).exists() for path in required)


def _base_row(audio_path: str, paths: Dict[str, str], debug_candidates: bool) -> Dict[str, str]:
    return {
        "filename": audio_path,
        "detected_drop_time": "",
        "confidence": "",
        "confidence_tier": "",
        "selected_by": "",
        "sustained_full_groove_score": "",
        "immediate_groove_start_score": "",
        "groove_stability": "",
        "pre_drop_contrast": "",
        "drumprint_pattern_score": "",
        "fake_hit_penalty": "",
        "micro_confidence": "",
        "snap_offset_ms": "",
        "microaligned_time": "",
        "output_als": paths["output_als"],
        "candidates_json": paths["candidates_json"],
        "debug_png": paths["debug_png"] if debug_candidates else "",
        "als_valid": "",
        "als_validation_error": "",
        "status": "",
        "error": "",
    }


def _populate_row_from_candidate_json(row: Dict[str, str], candidate_json_path: str) -> None:
    path = Path(candidate_json_path)
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    drop = payload.get("final_ai_pick", payload.get("drop_sec"))
    confidence = payload.get("confidence")
    tier = payload.get("confidence_tier")
    selected_by = payload.get("selected_by")
    selected_candidate = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), dict) else {}
    feature_summary = payload.get("feature_summary") if isinstance(payload.get("feature_summary"), dict) else {}
    if drop is not None:
        row["detected_drop_time"] = f"{float(drop):.9f}".rstrip("0").rstrip(".")
    if confidence is not None:
        row["confidence"] = f"{float(confidence):.6f}".rstrip("0").rstrip(".")
    if tier:
        row["confidence_tier"] = str(tier)
    if selected_by:
        row["selected_by"] = str(selected_by)
    row["sustained_full_groove_score"] = _format_float(
        selected_candidate.get("sustained_full_groove_score", feature_summary.get("chosen_sustained_full_groove_score"))
    )
    row["immediate_groove_start_score"] = _format_float(
        selected_candidate.get("immediate_groove_start_score", feature_summary.get("chosen_immediate_groove_start_score"))
    )
    row["groove_stability"] = _format_float(
        selected_candidate.get("groove_stability", feature_summary.get("chosen_groove_stability"))
    )
    row["pre_drop_contrast"] = _format_float(
        selected_candidate.get("pre_drop_contrast", feature_summary.get("chosen_pre_drop_contrast"))
    )
    row["drumprint_pattern_score"] = _format_float(
        selected_candidate.get("drumprint_pattern_score", feature_summary.get("chosen_drumprint_pattern_score"))
    )
    row["fake_hit_penalty"] = _format_float(
        selected_candidate.get("fake_hit_penalty", feature_summary.get("chosen_fake_hit_penalty"))
    )
    row["micro_confidence"] = _format_float(
        selected_candidate.get("micro_confidence", feature_summary.get("chosen_micro_confidence"))
    )
    row["snap_offset_ms"] = _format_float(
        selected_candidate.get("snap_offset_ms", feature_summary.get("chosen_snap_offset_ms"))
    )
    row["microaligned_time"] = _format_float(
        selected_candidate.get("microaligned_time", feature_summary.get("chosen_microaligned_time")),
        digits=9,
    )


def _verify_output_als(row: Dict[str, str], paths: Dict[str, str]) -> bool:
    report = verify_als(paths["output_als"], candidates_json=paths["candidates_json"])
    row["als_valid"] = "true" if report.get("valid") else "false"
    if not report.get("valid"):
        row["als_validation_error"] = "; ".join(str(err) for err in report.get("errors", []))
        return False
    row["als_validation_error"] = ""
    return True


def _process_one(task: Dict[str, object]) -> Dict[str, str]:
    audio_path = str(task["audio_path"])
    template_path = str(task["template_path"])
    debug_candidates = bool(task["debug_candidates"])
    force = bool(task["force"])
    dry_run = bool(task["dry_run"])
    bpm = task.get("bpm")
    min_drop_sec = float(task["min_drop_sec"])
    cues = list(task.get("cues") or [])
    analysis_sr = task.get("analysis_sr")
    strict_stems = bool(task.get("strict_stems"))
    disable_hpss = bool(task.get("disable_hpss"))
    use_drumprint = task.get("use_drumprint")
    use_microalign = bool(task.get("use_microalign"))
    paths = _expected_outputs(audio_path)
    row = _base_row(audio_path, paths, debug_candidates)

    try:
        if not force and _is_processed(paths, debug_candidates):
            _populate_row_from_candidate_json(row, paths["candidates_json"])
            _verify_output_als(row, paths)
            row["status"] = "skipped"
            return row

        if dry_run:
            row["status"] = "dry_run"
            return row

        cfg = DropDetectorConfig(
            min_drop_time_sec=min_drop_sec,
            sample_rate=int(analysis_sr) if analysis_sr else None,
            hpss=not disable_hpss,
            use_drumprint=bool(use_drumprint) if use_drumprint is not None else None,
            use_microalign=use_microalign,
        )
        result = detect_drop(
            audio_path,
            bpm=float(bpm) if bpm is not None else None,
            external_cues=cues,
            config=cfg,
        )
        row["detected_drop_time"] = f"{float(result.drop_sec):.9f}".rstrip("0").rstrip(".")
        row["confidence"] = f"{float(result.confidence):.6f}".rstrip("0").rstrip(".")
        row["confidence_tier"] = str(result.confidence_tier)
        row["selected_by"] = str(result.selected_by)
        row["sustained_full_groove_score"] = _format_float(result.features_summary.get("chosen_sustained_full_groove_score"))
        row["immediate_groove_start_score"] = _format_float(result.features_summary.get("chosen_immediate_groove_start_score"))
        row["groove_stability"] = _format_float(result.features_summary.get("chosen_groove_stability"))
        row["pre_drop_contrast"] = _format_float(result.features_summary.get("chosen_pre_drop_contrast"))
        row["drumprint_pattern_score"] = _format_float(result.features_summary.get("chosen_drumprint_pattern_score"))
        row["fake_hit_penalty"] = _format_float(result.features_summary.get("chosen_fake_hit_penalty"))
        row["micro_confidence"] = _format_float(result.features_summary.get("chosen_micro_confidence"))
        row["snap_offset_ms"] = _format_float(result.features_summary.get("chosen_snap_offset_ms"))
        row["microaligned_time"] = _format_float(result.features_summary.get("chosen_microaligned_time"), digits=9)

        write_candidate_debug_json(result, paths["candidates_json"])
        modify_als(
            template_path=template_path,
            audio_path=audio_path,
            drop_sec=result.drop_sec,
            bpm=result.bpm,
            output_path=paths["output_als"],
            strict_stems=strict_stems,
        )
        if not _verify_output_als(row, paths):
            row["status"] = "error"
            row["error"] = "als_validation_failed"
            return row

        if debug_candidates:
            try:
                from drop_aligner.plots import write_debug_plot

                features = extract_features(audio_path, cfg, bpm=float(bpm) if bpm is not None else result.bpm)
                write_debug_plot(features, result.candidates, result.drop_sec, paths["debug_png"])
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                row["status"] = "partial"
                row["error"] = f"debug_png_failed: {message}"
                return row

        row["status"] = "processed"
        return row
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        row["status"] = "error"
        row["error"] = message
        return row


def _write_summary(rows: Sequence[Dict[str, str]], output_path: str) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate drop-aligned Ableton ALS files for a folder of tracks.")
    parser.add_argument("folder", help="Folder containing .wav, .flac, .aiff, or .mp3 tracks")
    parser.add_argument("--template", required=True, help="Ableton .als template to modify")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders")
    parser.add_argument("--debug-candidates", action="store_true", help="Write *_drop_debug.png in addition to candidate JSON")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes, e.g. --workers 4")
    parser.add_argument("--force", action="store_true", help="Reprocess tracks even if expected outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and write the summary CSV")
    parser.add_argument("--summary", help="Summary CSV path. Defaults to drop_batch_summary.csv in the input folder")
    parser.add_argument("--bpm", type=float, help="Optional BPM override applied to every track")
    parser.add_argument("--bpm-from-path", action=argparse.BooleanOptionalAction, default=True, help="Infer BPM from stem filename or BPM folder when --bpm is not set")
    parser.add_argument("--analysis-sr", type=int, default=22050, help="Analysis sample rate. Use 0 to preserve source rate")
    parser.add_argument("--no-hpss", action="store_true", help="Skip HPSS separation. Recommended for already-separated drums stems.")
    parser.set_defaults(use_drumprint=None)
    parser.add_argument("--use-drumprint", dest="use_drumprint", action="store_true", help="Force DrumPrint pattern scoring on")
    parser.add_argument("--no-drumprint", dest="use_drumprint", action="store_false", help="Disable DrumPrint pattern scoring")
    parser.set_defaults(use_microalign=False)
    parser.add_argument("--microalign", dest="use_microalign", action="store_true", help="Enable sample-level MicroSnap marker refinement")
    parser.add_argument("--no-microalign", dest="use_microalign", action="store_false", help="Disable sample-level MicroSnap marker refinement")
    parser.add_argument("--stem-role", choices=["drums", "inst", "vocals"], help="Only process stems whose filename starts with this role")
    parser.add_argument("--strict-stem-set", action="store_true", help="Require matching drums/inst/vocals files when writing ALS from a multi-stem template")
    parser.add_argument(
        "--exclude-dir-name",
        action="append",
        default=sorted(EXCLUDED_DIR_NAMES),
        help="Directory name to ignore while scanning. Defaults to notToBeOrganized. Repeat to exclude more.",
    )
    parser.add_argument("--cue", action="append", help="Optional cue time(s) in seconds. Repeat or comma-separate. Used as search regions only.")
    parser.add_argument("--min-drop-sec", type=float, default=4.0, help="Reject candidates before this time")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    if not template.exists() or not template.is_file():
        raise SystemExit(f"Template not found: {template}")

    audio_files = _discover_audio(
        folder,
        bool(args.recursive),
        stem_role=args.stem_role,
        excluded_dir_names=args.exclude_dir_name,
    )
    summary_path = str(Path(args.summary).expanduser()) if args.summary else str(folder / "drop_batch_summary.csv")
    cues = _parse_cues(args.cue)
    use_drumprint = args.use_drumprint
    if use_drumprint is None and args.stem_role == "drums":
        use_drumprint = True

    tasks = [
        {
            "audio_path": str(path.absolute()),
            "template_path": str(template),
            "debug_candidates": bool(args.debug_candidates),
            "force": bool(args.force),
            "dry_run": bool(args.dry_run),
            "bpm": args.bpm if args.bpm is not None else (_infer_bpm_from_path(path) if args.bpm_from_path else None),
            "min_drop_sec": float(args.min_drop_sec),
            "cues": cues,
            "analysis_sr": int(args.analysis_sr) if int(args.analysis_sr) > 0 else None,
            "strict_stems": bool(args.strict_stem_set),
            "disable_hpss": bool(args.no_hpss),
            "use_drumprint": use_drumprint,
            "use_microalign": bool(args.use_microalign),
        }
        for path in audio_files
    ]

    rows: List[Dict[str, str]] = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            row = _process_one(task)
            rows.append(row)
            print(f"[{index}/{len(tasks)}] {row.get('status', '')}: {row.get('filename', '')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_audio = {pool.submit(_process_one, task): str(task["audio_path"]) for task in tasks}
            for index, future in enumerate(as_completed(future_to_audio), start=1):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    message = str(exc) or exc.__class__.__name__
                    audio_path = future_to_audio[future]
                    paths = _expected_outputs(audio_path)
                    row = _base_row(audio_path, paths, bool(args.debug_candidates))
                    row["status"] = "error"
                    row["error"] = message
                    rows.append(row)
                row = rows[-1]
                print(f"[{index}/{len(tasks)}] {row.get('status', '')}: {row.get('filename', '')}", flush=True)
        rows.sort(key=lambda row: row.get("filename", "").lower())

    written_summary = _write_summary(rows, summary_path)
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print(
        json.dumps(
            {
                "folder": str(folder),
                "template": str(template),
                "tracks_found": len(audio_files),
                "summary_csv": written_summary,
                "counts": counts,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
