#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from drop_aligner.als import modify_als
from drop_aligner.detector import DropDetectorConfig, detect_drop, extract_features
from drop_aligner.legacy_write_guard import add_legacy_detector_write_arg, require_legacy_detector_write_opt_in
from drop_aligner.learning import closest_candidate_to_pick, log_correction


def _parse_cues(values: Optional[List[str]]) -> List[float]:
    out: List[float] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect the true first EDM drop and write a drop-aligned Ableton ALS file.")
    parser.add_argument("audio", help="Input FLAC/WAV audio file")
    parser.add_argument("--template", required=True, help="Ableton .als template to modify")
    parser.add_argument("--output", help="Output .als path. Defaults to originalname_DROP_ALIGNED.als beside the audio.")
    parser.add_argument("--bpm", type=float, help="Optional BPM override")
    parser.add_argument("--cue", action="append", help="Optional cue time(s) in seconds. Repeat or comma-separate. Used as search regions only.")
    parser.add_argument("--analysis-json", help="Write JSON analysis output")
    parser.add_argument("--plot", help="Write a debug plot PNG")
    parser.add_argument("--debug-candidates", action="store_true", help="Write top-10 candidate JSON and debug plot beside the audio")
    parser.add_argument("--min-drop-sec", type=float, default=4.0, help="Reject candidates before this time")
    parser.set_defaults(use_drumprint=None)
    parser.add_argument("--use-drumprint", dest="use_drumprint", action="store_true", help="Force DrumPrint pattern scoring on")
    parser.add_argument("--no-drumprint", dest="use_drumprint", action="store_false", help="Disable DrumPrint pattern scoring")
    parser.set_defaults(use_microalign=False)
    parser.add_argument("--microalign", dest="use_microalign", action="store_true", help="Enable sample-level MicroSnap marker refinement")
    parser.add_argument("--no-microalign", dest="use_microalign", action="store_false", help="Disable sample-level MicroSnap marker refinement")
    parser.add_argument("--user-pick", type=float, help="Optional manually corrected pick in seconds; logs correction data")
    parser.add_argument("--correction-log", default="drop_corrections.jsonl", help="Correction JSONL path")
    add_legacy_detector_write_arg(parser)
    return parser


def _default_candidate_json(audio_path: str) -> str:
    audio = Path(audio_path)
    return str(audio.with_name(f"{audio.stem}_drop_candidates.json"))


def _default_debug_plot(audio_path: str) -> str:
    audio = Path(audio_path)
    return str(audio.with_name(f"{audio.stem}_drop_debug.png"))


def _candidate_debug_payload(result, user_pick: Optional[float]) -> Dict[str, Any]:
    top_10 = result.top_candidate_dicts(10)
    payload: Dict[str, Any] = {
        "track": result.audio_path,
        "final_ai_pick": float(result.drop_sec),
        "coarse_ai_pick": float(result.coarse_sec),
        "bpm": float(result.bpm),
        "confidence": float(result.confidence),
        "confidence_tier": result.confidence_tier,
        "selected_by": result.selected_by,
        "feature_summary": dict(result.features_summary),
        "selected_candidate": result.selected_candidate_dict(),
        "top_10_candidates": top_10,
    }
    if user_pick is not None:
        payload["user_pick"] = float(user_pick)
        payload["closest_candidate_to_user_pick"] = closest_candidate_to_pick(top_10, float(user_pick))
    return payload


def _write_candidate_debug_json(result, output_path: str, user_pick: Optional[float]) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_candidate_debug_payload(result, user_pick), fh, indent=2, ensure_ascii=True)
    return str(path)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audio_path = os.path.abspath(args.audio)
    template_path = os.path.abspath(args.template)
    output_path = args.output
    if output_path is None:
        audio = Path(audio_path)
        output_path = str(audio.with_name(f"{audio.stem}_DROP_ALIGNED.als"))
    else:
        output_path = os.path.abspath(output_path)
    require_legacy_detector_write_opt_in(
        "main.py",
        action="writing a legacy detector ALS/candidate/debug output",
        explicit=bool(args.allow_legacy_detector_write),
    )

    cfg = DropDetectorConfig(
        min_drop_time_sec=float(args.min_drop_sec),
        use_drumprint=args.use_drumprint,
        use_microalign=bool(args.use_microalign),
    )
    cues = _parse_cues(args.cue)
    result = detect_drop(
        audio_path,
        bpm=args.bpm,
        external_cues=cues,
        config=cfg,
        analysis_json=args.analysis_json,
    )

    candidate_json = None
    if args.debug_candidates:
        candidate_json = _write_candidate_debug_json(result, _default_candidate_json(audio_path), args.user_pick)

    out_als = modify_als(
        template_path=template_path,
        audio_path=audio_path,
        drop_sec=result.drop_sec,
        bpm=result.bpm,
        output_path=output_path,
    )

    if args.user_pick is not None:
        log_path = log_correction(
            track=audio_path,
            ai_pick=result.drop_sec,
            user_pick=float(args.user_pick),
            features=result.features_summary,
            top_candidates=result.top_candidate_dicts(10),
            selected_candidate=result.selected_candidate_dict(),
            selected_by=result.selected_by,
            confidence_tier=result.confidence_tier,
            log_path=args.correction_log,
        )
    else:
        log_path = None

    plot_path = args.plot or (_default_debug_plot(audio_path) if args.debug_candidates else None)
    plot_written = None
    if plot_path:
        try:
            from drop_aligner.plots import write_debug_plot

            features = extract_features(audio_path, cfg, bpm=args.bpm or result.bpm)
            plot_written = write_debug_plot(features, result.candidates, result.drop_sec, plot_path, user_pick=args.user_pick)
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    summary = {
        "audio": audio_path,
        "template": template_path,
        "output_als": out_als,
        "drop_sec": result.drop_sec,
        "coarse_sec": result.coarse_sec,
        "bpm": result.bpm,
        "confidence": result.confidence,
        "confidence_tier": result.confidence_tier,
        "selected_by": result.selected_by,
        "analysis_json": args.analysis_json,
        "candidate_json": candidate_json,
        "debug_plot": plot_written,
        "correction_log": log_path,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
