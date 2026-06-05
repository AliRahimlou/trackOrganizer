from __future__ import annotations

from typing import Dict


FULL_GROOVE_FEATURE_KEYS = [
    "drum_onset_spike",
    "rms_jump",
    "spectral_flux_peak",
    "pre_drop_contrast",
    "immediate_groove_start_score",
    "groove_stability",
    "buildup_low_energy_score",
    "buildup_ramp_score",
    "drop_impact_score",
    "kick_reentry_score",
    "buildup_drop_score",
    "sustained_full_groove_score",
]


def empty_full_groove_features() -> Dict[str, float]:
    return {key: 0.0 for key in FULL_GROOVE_FEATURE_KEYS}
