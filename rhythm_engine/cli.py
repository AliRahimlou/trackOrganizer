from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .clicktrack import render_click_track, render_hypothesis_click_tracks
from .engine import analyze_rhythm
from .export import write_beatgrid_csv, write_beatgrid_json
from .types import RhythmEngineConfig


def _read_times(path: Optional[str]) -> Optional[List[float]]:
    if not path:
        return None
    out: List[float] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        out.append(float(text.split()[0]))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze beat/downbeat grids with the TrackOrganizer rhythm engine.")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--provider", action="append", help="Provider to run. Repeat to set order.")
    parser.add_argument("--json", dest="json_path", help="Write full JSON result")
    parser.add_argument("--reference-beats", help="Optional newline-separated reference beat seconds")
    parser.add_argument("--reference-downbeats", help="Optional newline-separated reference downbeat seconds")
    parser.add_argument("--fusion-radius-ms", type=float, default=45.0)
    parser.add_argument("--downbeat-fusion-radius-ms", type=float, default=90.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--no-micro-refine", action="store_true", help="Disable sample-level attack refinement")
    parser.add_argument("--no-stem-aware-micro-refine", action="store_true", help="Do not prefer sibling drum/bass stems for micro-refinement")
    parser.add_argument("--beat-this-device", default="auto")
    parser.add_argument("--beat-this-checkpoint", default="final0")
    parser.add_argument("--beat-this-dbn", action="store_true")
    parser.add_argument("--beatnet-model", type=int, default=1, help="BeatNet pretrained model id when BeatNet is installed")
    parser.add_argument("--beatnet-mode", default="offline", help="BeatNet mode: offline, online, realtime, or stream")
    parser.add_argument("--beatnet-inference-model", default="DBN", help="BeatNet inference model, usually DBN offline or PF online")
    parser.add_argument("--beatnet-device", default="auto", help="BeatNet device when supported, e.g. cpu or cuda")
    parser.add_argument("--provider-weights-json", help="JSON mapping of provider names to learned fusion weights")
    parser.add_argument("--no-hypotheses", action="store_true", help="Do not emit half/double-time and phase hypotheses")
    parser.add_argument("--no-hypothesis-selector", action="store_true", help="Use fused grid directly instead of full-song hypothesis selection")
    parser.add_argument("--no-grid-repair", action="store_true", help="Disable steady-grid continuity repair")
    parser.add_argument("--click-wav", help="Write an audio+click overlay for the final grid")
    parser.add_argument("--click-only", action="store_true", help="Write clicks without original audio when rendering click tracks")
    parser.add_argument("--hypothesis-click-dir", help="Write click overlays for top hypotheses into this directory")
    parser.add_argument("--beatgrid-csv", help="Write final beatgrid rows to CSV")
    parser.add_argument("--beatgrid-json", help="Write compact final beatgrid JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RhythmEngineConfig(
        providers=tuple(args.provider) if args.provider else RhythmEngineConfig().providers,
        sample_rate=int(args.sample_rate),
        fusion_radius_ms=float(args.fusion_radius_ms),
        downbeat_fusion_radius_ms=float(args.downbeat_fusion_radius_ms),
        micro_refine=not bool(args.no_micro_refine),
        micro_refine_stem_aware=not bool(args.no_stem_aware_micro_refine),
        beat_this_device=str(args.beat_this_device),
        beat_this_checkpoint=str(args.beat_this_checkpoint),
        use_beat_this_dbn=bool(args.beat_this_dbn),
        beatnet_model=int(args.beatnet_model),
        beatnet_mode=str(args.beatnet_mode),
        beatnet_inference_model=str(args.beatnet_inference_model),
        beatnet_device=str(args.beatnet_device),
        provider_weights_json=args.provider_weights_json,
        preserve_hypotheses=not bool(args.no_hypotheses),
        use_hypothesis_selector=not bool(args.no_hypothesis_selector),
        repair_steady_grid=not bool(args.no_grid_repair),
    )
    result = analyze_rhythm(
        args.audio,
        config=cfg,
        reference_beats=_read_times(args.reference_beats),
        reference_downbeats=_read_times(args.reference_downbeats),
    )
    payload = result.to_dict()
    if args.click_wav:
        payload["final_click_wav"] = render_click_track(
            args.audio,
            result.final,
            args.click_wav,
            overlay=not bool(args.click_only),
        )
    if args.hypothesis_click_dir:
        payload["hypothesis_click_wavs"] = render_hypothesis_click_tracks(
            args.audio,
            result.hypotheses,
            args.hypothesis_click_dir,
            overlay=not bool(args.click_only),
        )
    if args.beatgrid_csv:
        payload["beatgrid_csv"] = write_beatgrid_csv(result.final, args.beatgrid_csv)
    if args.beatgrid_json:
        payload["beatgrid_json"] = write_beatgrid_json(result.final, args.beatgrid_json)
    if args.json_path:
        path = Path(args.json_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
