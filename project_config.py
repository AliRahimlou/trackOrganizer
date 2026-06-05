from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return str(default)
    return str(value).strip()


def env_path(name: str, default: str | Path) -> Path:
    return Path(env_text(name, str(default))).expanduser()


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip() or str(default))
    except (TypeError, ValueError):
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip() or str(default))
    except (TypeError, ValueError):
        return float(default)


def as_str(path: str | Path | Any) -> str:
    return str(path)


MUSIC_ROOT = env_path("TRACK_ORGANIZER_MUSIC_ROOT", "~/Desktop/MUSIC")
PLAYLISTS_BY_DATE_DIR = env_path("TRACK_ORGANIZER_PLAYLISTS_DIR", MUSIC_ROOT / "PlaylistsByDate")
STEMS_ROOT_DIR = env_path("TRACK_ORGANIZER_STEMS_DIR", MUSIC_ROOT / "STEMS")
STEMS_INBOX_DIR = env_path("TRACK_ORGANIZER_STEMS_INBOX_DIR", STEMS_ROOT_DIR / "toBeOrganized")
GENERATED_SET_DIR = env_path("TRACK_ORGANIZER_GENERATED_SET_DIR", MUSIC_ROOT / "GeneratedSet")
ALS_TEMPLATES_DIR = env_path("TRACK_ORGANIZER_ALS_TEMPLATES_DIR", PROJECT_ROOT / "alsFiles")
DEFAULT_ALS_TEMPLATE = env_path("TRACK_ORGANIZER_DEFAULT_ALS_TEMPLATE", ALS_TEMPLATES_DIR / "128.als")
BASE_ALS_TEMPLATE = env_path("TRACK_ORGANIZER_BASE_ALS", ALS_TEMPLATES_DIR / "CH1.als")
DROP_BATCH_SUMMARY = env_path("TRACK_ORGANIZER_DROP_BATCH_SUMMARY", STEMS_ROOT_DIR / "drop_batch_summary.csv")
REKORDBOX_XML = env_path("REKORDBOX_XML_PATH", "~/Documents/rekordbox.xml")
