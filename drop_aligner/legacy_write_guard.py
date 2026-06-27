from __future__ import annotations

import os
from argparse import ArgumentParser


ALLOW_ENV_VAR = "TRACK_ORGANIZER_ALLOW_LEGACY_WRITES"
ALLOW_FLAG = "--allow-legacy-detector-write"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class LegacyDetectorWriteBlocked(SystemExit):
    """Raised when an old detector path tries to write production marker data."""


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def legacy_detector_writes_allowed(*, explicit: bool = False) -> bool:
    return bool(explicit) or _truthy(os.environ.get(ALLOW_ENV_VAR))


def add_legacy_detector_write_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        ALLOW_FLAG,
        action="store_true",
        help=(
            "Permit this legacy non-visual detector to write files. "
            "Normal production runs should use the visual-first builder/review path instead."
        ),
    )


def require_legacy_detector_write_opt_in(script_name: str, *, action: str, explicit: bool = False) -> None:
    if legacy_detector_writes_allowed(explicit=explicit):
        return
    raise LegacyDetectorWriteBlocked(
        f"{script_name} blocked {action}. This is a legacy non-visual detector write path; "
        "normal production marker generation must use the visual-first Boom-waveform pipeline. "
        f"For an intentional legacy experiment only, pass {ALLOW_FLAG} or set {ALLOW_ENV_VAR}=1."
    )
