from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .evaluate import evaluate_beat_grid
from .fusion import fuse_estimates
from .hypotheses import generate_hypotheses
from .grid_repair import repair_steady_grid
from .micro_refine import refine_estimate_to_attacks
from .providers import run_providers
from .quality import assess_grid_quality
from .selector import select_best_hypothesis
from .types import RhythmAnalysisResult, RhythmEngineConfig


def analyze_rhythm(
    audio_path: str,
    *,
    config: Optional[RhythmEngineConfig] = None,
    reference_beats: Optional[Sequence[float]] = None,
    reference_downbeats: Optional[Sequence[float]] = None,
) -> RhythmAnalysisResult:
    cfg = config or RhythmEngineConfig()
    providers = tuple(run_providers(audio_path, cfg))
    hypotheses = tuple(generate_hypotheses(providers, cfg)) if cfg.preserve_hypotheses else tuple()
    fused = fuse_estimates(providers, cfg)
    selector_candidates = tuple([fused, *hypotheses])
    selected = select_best_hypothesis(audio_path, selector_candidates, fallback=fused, config=cfg)
    repaired = repair_steady_grid(selected, cfg)
    final = refine_estimate_to_attacks(audio_path, repaired, cfg) if cfg.micro_refine else repaired
    quality = assess_grid_quality(final, fused=fused).to_dict()

    evaluation: Optional[Mapping[str, object]] = None
    if reference_beats is not None:
        beat_report = evaluate_beat_grid(reference_beats, final.beats).to_dict()
        provider_reports = {
            estimate.provider: evaluate_beat_grid(reference_beats, estimate.beats).to_dict()
            for estimate in providers
            if estimate.available
        }
        hypothesis_reports = {
            estimate.provider: evaluate_beat_grid(reference_beats, estimate.beats).to_dict()
            for estimate in hypotheses
            if estimate.available
        }
        evaluation = {
            "beats": beat_report,
            "providers": provider_reports,
            "hypotheses": hypothesis_reports,
        }
        if reference_downbeats is not None:
            evaluation = {
                **evaluation,
                "downbeats": evaluate_beat_grid(reference_downbeats, final.downbeats).to_dict(),
                "provider_downbeats": {
                    estimate.provider: evaluate_beat_grid(reference_downbeats, estimate.downbeats).to_dict()
                    for estimate in providers
                    if estimate.available and estimate.downbeats
                },
                "hypothesis_downbeats": {
                    estimate.provider: evaluate_beat_grid(reference_downbeats, estimate.downbeats).to_dict()
                    for estimate in hypotheses
                    if estimate.available and estimate.downbeats
                },
            }

    return RhythmAnalysisResult(
        audio_path=str(audio_path),
        final=final,
        fused=fused,
        selected=selected,
        providers=providers,
        hypotheses=hypotheses,
        evaluation=evaluation,
        quality=quality,
        metadata={
            "provider_order": list(cfg.providers),
            "micro_refine": bool(cfg.micro_refine),
            "hypothesis_selector": bool(cfg.use_hypothesis_selector),
            "grid_repair": bool(cfg.repair_steady_grid),
            "repair_lattice_snap": bool(cfg.repair_lattice_snap),
            "repair_lattice_snap_ratio": float(cfg.repair_lattice_snap_ratio),
            "fusion_radius_ms": float(cfg.fusion_radius_ms),
            "downbeat_fusion_radius_ms": float(cfg.downbeat_fusion_radius_ms),
            "fusion_dedupe_events": bool(cfg.fusion_dedupe_events),
            "fusion_min_beat_gap_ratio": float(cfg.fusion_min_beat_gap_ratio),
            "fusion_min_downbeat_gap_ratio": float(cfg.fusion_min_downbeat_gap_ratio),
        },
    )
