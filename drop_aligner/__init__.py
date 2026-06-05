"""True first-drop alignment tools for Ableton Live projects."""

from .detector import DropCandidate, DropDetectionResult, DropDetectorConfig, detect_drop
from .als import modify_als, save_als
from .learning import log_correction

__all__ = [
    "DropCandidate",
    "DropDetectionResult",
    "DropDetectorConfig",
    "detect_drop",
    "modify_als",
    "save_als",
    "log_correction",
]
