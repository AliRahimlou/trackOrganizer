"""Single source of truth for the visual-first production gate contract.

The builder (build_fresh_visual_first_library_set), the review queue
(build_visual_first_production_queue), and the validator
(validate_visual_first_production) must agree on which marker sources are
allowed to auto-pass and on the canonical tolerances.  They used to carry
private copies of these constants; import them from here so they cannot
drift.
"""
from __future__ import annotations

# Marker sources that must never auto-pass the production gate: they replay a
# human or historical pick instead of proving the drop from the audio itself.
DISALLOWED_PASS_SOURCES = frozenset(
    {
        "historical_human_marker",
        "historical_review_memory",
        "manual_review_marker",
        "review_auto_place",
        "saved_closest_to_review_pick",
        "visual_drop_v2",
        "visual_drop_v2_candidate",
        "visual_first_rms_body_fallback",
        "web_accept_blue_marker",
        "web_save_placed_marker",
    }
)
DISALLOWED_PASS_SOURCE_PREFIXES = ("historical_", "saved_")

VALIDATED_HUMAN_OVERRIDE_SELECTED_BY = "visual_validated_human_review_override"

# A human review pick within this window of the current marker still replaces
# it.  The anchor contract is sample accurate (the ALS write is verified at
# 0.002 s), so corrections smaller than the old 0.050 s band must not be
# silently discarded as "already matching".
HUMAN_OVERRIDE_MATCH_TOLERANCE_SEC = 0.005

# Maximum distance between the marker and the beat-one grid line.
MAX_GRID_ONE_DISTANCE_MS = 90.0

# ALS write verification tolerance (seconds).
ALS_ANCHOR_VERIFY_TOLERANCE_SEC = 0.002


def unsafe_selected_source(selected_by: str) -> bool:
    """True when the marker source is not allowed to auto-pass production."""
    source = str(selected_by or "").strip()
    return source in DISALLOWED_PASS_SOURCES or any(
        source.startswith(prefix) for prefix in DISALLOWED_PASS_SOURCE_PREFIXES
    )
