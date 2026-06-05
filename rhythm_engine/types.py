from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clean_times(values: Sequence[float]) -> Tuple[float, ...]:
    out = []
    for value in values:
        number = _finite_float(value, default=float("nan"))
        if math.isfinite(number) and number >= 0.0:
            out.append(float(number))
    return tuple(sorted(set(round(value, 9) for value in out)))


@dataclass(frozen=True)
class RhythmEngineConfig:
    providers: Tuple[str, ...] = ("beat_this", "madmom", "stem_ensemble", "track_organizer", "librosa")
    sample_rate: int = 22050
    min_bpm: float = 35.0
    max_bpm: float = 260.0
    beats_per_bar: int = 4
    fusion_radius_ms: float = 45.0
    downbeat_fusion_radius_ms: float = 90.0
    min_provider_confidence: float = 0.05
    micro_refine: bool = True
    micro_refine_stem_aware: bool = True
    micro_refine_sample_rate: int = 44100
    micro_refine_window_ms: float = 35.0
    beat_this_device: str = "auto"
    beat_this_checkpoint: str = "final0"
    use_beat_this_dbn: bool = False
    provider_weights_json: Optional[str] = None
    preserve_hypotheses: bool = True
    max_hypotheses: int = 24
    use_hypothesis_selector: bool = True
    selector_sample_rate: int = 22050
    repair_steady_grid: bool = True
    repair_min_tempo_stability: float = 0.72


@dataclass(frozen=True)
class RhythmEstimate:
    provider: str
    beats: Tuple[float, ...] = field(default_factory=tuple)
    downbeats: Tuple[float, ...] = field(default_factory=tuple)
    bpm: Optional[float] = None
    confidence: float = 0.0
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    status: str = "ok"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "beats", _clean_times(self.beats))
        object.__setattr__(self, "downbeats", _clean_times(self.downbeats))
        bpm = None if self.bpm is None else _finite_float(self.bpm, default=float("nan"))
        object.__setattr__(self, "bpm", bpm if bpm is not None and math.isfinite(bpm) and bpm > 0 else None)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, _finite_float(self.confidence))))

    @property
    def available(self) -> bool:
        return self.status == "ok" and bool(self.beats)

    @classmethod
    def unavailable(cls, provider: str, reason: str, *, metadata: Optional[Mapping[str, Any]] = None) -> "RhythmEstimate":
        return cls(
            provider=str(provider),
            status="unavailable",
            reason=str(reason),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failed(cls, provider: str, reason: str, *, metadata: Optional[Mapping[str, Any]] = None) -> "RhythmEstimate":
        return cls(
            provider=str(provider),
            status="failed",
            reason=str(reason),
            metadata=dict(metadata or {}),
        )

    def with_updates(self, **updates: Any) -> "RhythmEstimate":
        data = self.to_dict()
        data.update(updates)
        return RhythmEstimate(
            provider=str(data["provider"]),
            beats=tuple(float(x) for x in data.get("beats", [])),
            downbeats=tuple(float(x) for x in data.get("downbeats", [])),
            bpm=data.get("bpm"),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            duration_sec=data.get("duration_sec"),
            sample_rate=data.get("sample_rate"),
            status=str(data.get("status", "ok")),
            reason=str(data.get("reason", "")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "reason": self.reason,
            "available": self.available,
            "beats": [float(t) for t in self.beats],
            "downbeats": [float(t) for t in self.downbeats],
            "bpm": None if self.bpm is None else float(self.bpm),
            "confidence": float(self.confidence),
            "duration_sec": None if self.duration_sec is None else float(self.duration_sec),
            "sample_rate": None if self.sample_rate is None else int(self.sample_rate),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RhythmAnalysisResult:
    audio_path: str
    final: RhythmEstimate
    fused: RhythmEstimate
    selected: RhythmEstimate
    providers: Tuple[RhythmEstimate, ...]
    hypotheses: Tuple[RhythmEstimate, ...] = field(default_factory=tuple)
    evaluation: Optional[Mapping[str, Any]] = None
    quality: Optional[Mapping[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "final": self.final.to_dict(),
            "fused": self.fused.to_dict(),
            "selected": self.selected.to_dict(),
            "providers": [estimate.to_dict() for estimate in self.providers],
            "hypotheses": [estimate.to_dict() for estimate in self.hypotheses],
            "evaluation": None if self.evaluation is None else dict(self.evaluation),
            "quality": None if self.quality is None else dict(self.quality),
            "metadata": dict(self.metadata),
        }
