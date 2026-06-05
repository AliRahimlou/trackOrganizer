from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXCLUDED_DIR_NAMES = frozenset({"notToBeOrganized"})


def is_excluded_path(value: object, *, excluded_dir_names: Iterable[str] = EXCLUDED_DIR_NAMES) -> bool:
    if value in (None, ""):
        return False
    excluded = {str(name).lower() for name in excluded_dir_names if str(name)}
    if not excluded:
        return False
    try:
        parts = Path(str(value)).expanduser().parts
    except Exception:
        parts = Path(str(value)).parts
    return any(part.lower() in excluded for part in parts)


def row_has_excluded_path(
    row: Mapping[str, object],
    keys: Sequence[str] = ("filename", "track", "audio_path", "output_als", "candidates_json", "debug_png"),
) -> bool:
    return any(is_excluded_path(row.get(key)) for key in keys)
