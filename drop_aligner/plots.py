from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np

from .drumprint import build_drumprint_analysis
from .detector import DropCandidate, FeatureBundle


def _downsample_waveform(y: np.ndarray, sr: int, max_points: int = 80000) -> tuple[np.ndarray, np.ndarray]:
    if y.size <= max_points:
        return np.arange(y.size, dtype=np.float64) / float(sr), y
    step = int(np.ceil(y.size / float(max_points)))
    trimmed = y[: (y.size // step) * step]
    blocks = trimmed.reshape(-1, step)
    peak = blocks[np.arange(blocks.shape[0]), np.argmax(np.abs(blocks), axis=1)]
    times = (np.arange(peak.size, dtype=np.float64) * float(step)) / float(sr)
    return times, peak


def _mark_candidates(axes, candidates: Sequence[DropCandidate], drop_sec: float, user_pick: Optional[float]) -> None:
    for candidate in candidates[:10]:
        color = "#8a8a8a" if candidate.rejected else "#f0a202"
        alpha = 0.28 if candidate.rejected else 0.48
        for ax in axes:
            ax.axvline(candidate.snapped_sec, color=color, linewidth=0.8, alpha=alpha)
        axes[0].text(
            candidate.snapped_sec,
            0.92,
            str(candidate.rank),
            color=color,
            fontsize=7,
            ha="center",
            va="top",
            transform=axes[0].get_xaxis_transform(),
        )

    for ax in axes:
        ax.axvline(drop_sec, color="#d62728", linewidth=1.8, alpha=0.95, label="final AI pick")
        if user_pick is not None:
            ax.axvline(float(user_pick), color="#1f77b4", linewidth=1.5, alpha=0.95, linestyle="--", label="user correction")

    selected = next((candidate for candidate in candidates if candidate.selected), None)
    if selected and selected.microalign:
        micro = selected.microalign
        original = micro.get("input_candidate_time")
        attack = micro.get("attack_start_time")
        zero = micro.get("zero_crossing_time")
        micro_time = micro.get("microaligned_time")
        for ax in axes:
            if original is not None:
                ax.axvline(float(original), color="#6a4c93", linewidth=1.0, alpha=0.70, linestyle=":", label="pre-micro marker")
            if attack is not None:
                ax.axvline(float(attack), color="#1982c4", linewidth=1.0, alpha=0.70, linestyle="--", label="micro attack")
            if zero is not None:
                ax.axvline(float(zero), color="#2a9d8f", linewidth=0.9, alpha=0.70, linestyle="-.", label="micro zero crossing")
            if micro_time is not None:
                ax.axvline(float(micro_time), color="#006d77", linewidth=1.3, alpha=0.85, label="microaligned marker")


def write_debug_plot(
    features: FeatureBundle,
    candidates: Sequence[DropCandidate],
    drop_sec: float,
    output_path: str,
    *,
    user_pick: Optional[float] = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    times = features.frame_times
    wave_times, wave = _downsample_waveform(features.y, features.sr)
    drumprint_analysis = None
    if any(bool((candidate.drumprint or {}).get("enabled")) for candidate in candidates):
        try:
            drumprint_analysis = build_drumprint_analysis(features.y, features.sr, features.bpm)
        except Exception:
            drumprint_analysis = None

    axis_count = 5 if drumprint_analysis is not None else 4
    fig, axes = plt.subplots(axis_count, 1, figsize=(15, 10 if axis_count == 5 else 9), sharex=True)

    axes[0].plot(wave_times, wave, color="#303030", linewidth=0.35)
    axes[0].set_ylabel("waveform")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(times, features.rms, color="#2ca02c", linewidth=1.0, label="RMS")
    axes[1].set_ylabel("RMS")

    axes[2].plot(times, features.low_energy, color="#9467bd", linewidth=1.0, label="<150 Hz")
    axes[2].plot(times, features.low_jump_curve, color="#17becf", linewidth=0.8, alpha=0.75, label="low jump")
    axes[2].set_ylabel("low-end")

    axes[3].plot(times, features.onset, color="#ff7f0e", linewidth=1.0, label="onset strength")
    axes[3].plot(times, features.combined_attack, color="#bcbd22", linewidth=0.8, alpha=0.75, label="combined attack")
    axes[3].plot(times, features.spectral_flux, color="#d62728", linewidth=0.7, alpha=0.70, label="spectral flux")
    axes[3].set_ylabel("onset")
    axes[3].set_xlabel("seconds")

    if drumprint_analysis is not None:
        axes[4].plot(
            drumprint_analysis.segment_starts,
            drumprint_analysis.segment_density,
            color="#005f73",
            linewidth=1.0,
            label="fingerprint density",
        )
        axes[4].plot(
            drumprint_analysis.segment_starts,
            drumprint_analysis.novelty_curve,
            color="#9b2226",
            linewidth=0.9,
            alpha=0.85,
            label="drumprint novelty",
        )
        axes[4].plot(
            drumprint_analysis.segment_starts,
            drumprint_analysis.boundary_curve,
            color="#ca6702",
            linewidth=0.9,
            alpha=0.85,
            label="self-sim boundary",
        )
        axes[4].set_ylabel("drumprint")
        axes[4].set_xlabel("seconds")

    _mark_candidates(axes, candidates, drop_sec, user_pick)
    axes[0].set_title(f"{Path(features.audio_path).name} - ranked drop candidates")

    for ax in axes:
        ax.grid(True, linewidth=0.35, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
