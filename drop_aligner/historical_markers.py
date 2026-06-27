from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


STEM_PREFIX_RE = re.compile(
    r"^(?:drums?|inst|instrumental|vocals?|bass|other|full)_(?:\d{2,3})_(?:\d{1,2}[ab])_(?:\d+)-",
    re.IGNORECASE,
)

HUMAN_REVIEW_SOURCES = {
    "web_review",
    "web_manual_marker",
    "web_candidate_pick",
    "web_accept_blue_marker",
    "web_accept_grid_marker",
    "web_accept_knee_marker",
    "web_accept_attack_marker",
    "web_accept_asd_marker",
    "web_accept_micro_marker",
    "web_ai_refined_accept",
}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _path_key(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve()).lower()
    except OSError:
        return str(Path(text).expanduser()).lower()


def exact_key_for_path(path: Any) -> str:
    return Path(str(path or "")).stem.strip().lower()


def slug_for_path(path: Any) -> str:
    stem = exact_key_for_path(path)
    stem = STEM_PREFIX_RE.sub("", stem)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stem.lower()).split())


def is_human_review_source(value: Any) -> bool:
    source = str(value or "").strip().lower()
    return source in HUMAN_REVIEW_SOURCES


@dataclass(frozen=True)
class HistoricalMarker:
    track: str
    user_pick: float
    source: str
    source_path: str = ""
    reviewed_from: str = ""
    timestamp: str = ""
    bpm: Optional[float] = None
    ai_pick: Optional[float] = None
    selected_by: str = ""
    observations: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "user_pick": float(self.user_pick),
            "source": self.source,
            "source_path": self.source_path,
            "reviewed_from": self.reviewed_from,
            "timestamp": self.timestamp,
            "bpm": self.bpm,
            "ai_pick": self.ai_pick,
            "selected_by": self.selected_by,
            "observations": int(self.observations),
        }


class HistoricalMarkerIndex:
    def __init__(self) -> None:
        self.by_path: Dict[str, HistoricalMarker] = {}
        self.by_exact: Dict[str, HistoricalMarker] = {}
        self.by_slug: Dict[str, list[HistoricalMarker]] = {}

    def __len__(self) -> int:
        keys = {marker.track for marker in self.by_path.values()}
        keys.update(marker.track for marker in self.by_exact.values())
        return len(keys)

    def add(self, marker: HistoricalMarker, *, override: bool = True) -> None:
        if marker.user_pick <= 0.0:
            return
        path_key = _path_key(marker.track)
        exact_key = exact_key_for_path(marker.track)
        slug = slug_for_path(marker.track)
        if path_key and (override or path_key not in self.by_path):
            self.by_path[path_key] = marker
        if exact_key and (override or exact_key not in self.by_exact):
            self.by_exact[exact_key] = marker
        if slug:
            bucket = [row for row in self.by_slug.get(slug, []) if exact_key_for_path(row.track) != exact_key]
            bucket.append(marker)
            self.by_slug[slug] = bucket

    def find(self, track: Any, *, bpm: Any = None) -> Optional[HistoricalMarker]:
        path_key = _path_key(track)
        if path_key and path_key in self.by_path:
            return self.by_path[path_key]
        exact_key = exact_key_for_path(track)
        if exact_key and exact_key in self.by_exact:
            return self.by_exact[exact_key]
        slug = slug_for_path(track)
        matches = list(self.by_slug.get(slug, []))
        if not matches:
            return None
        bpm_value = _float_or_none(bpm)
        if bpm_value is not None:
            bpm_matches = [row for row in matches if row.bpm is not None and abs(float(row.bpm) - bpm_value) <= 0.5]
            if len(bpm_matches) == 1:
                return bpm_matches[0]
            if bpm_matches:
                matches = bpm_matches
        rounded = {round(float(row.user_pick), 3) for row in matches}
        return matches[-1] if len(rounded) == 1 else None


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                yield row


def _load_marker_db(index: HistoricalMarkerIndex, marker_db_path: Path, *, min_marker_sec: float) -> None:
    if not marker_db_path.exists():
        return
    try:
        payload = json.loads(marker_db_path.read_text(encoding="utf-8"))
    except Exception:
        return
    exact = payload.get("exact") if isinstance(payload, Mapping) else None
    if not isinstance(exact, Mapping):
        return
    for key, row in exact.items():
        if not isinstance(row, Mapping):
            continue
        marker = _float_or_none(row.get("drop_sec"))
        if marker is None or marker < float(min_marker_sec):
            continue
        track = str(row.get("sample_path") or key)
        index.add(
            HistoricalMarker(
                track=track,
                user_pick=float(marker),
                source="drop_marker_db",
                source_path=str(marker_db_path),
                bpm=_float_or_none(row.get("bpm")),
                observations=int(row.get("obs_used") or row.get("obs") or 1),
            ),
            override=False,
        )


def _load_correction_log(index: HistoricalMarkerIndex, correction_path: Path) -> None:
    if not correction_path.exists():
        return
    for row in _iter_jsonl(correction_path):
        marker = _float_or_none(row.get("user_pick"))
        if marker is None or marker <= 0.0:
            continue
        if not is_human_review_source(row.get("reviewed_from")):
            continue
        track = str(row.get("track") or row.get("filename") or "")
        if not track:
            continue
        index.add(
            HistoricalMarker(
                track=track,
                user_pick=float(marker),
                source="correction_log",
                source_path=str(correction_path),
                reviewed_from=str(row.get("reviewed_from") or ""),
                timestamp=str(row.get("timestamp") or ""),
                bpm=_float_or_none((row.get("features") or {}).get("bpm") if isinstance(row.get("features"), Mapping) else None),
                ai_pick=_float_or_none(row.get("final_ai_pick") if row.get("final_ai_pick") is not None else row.get("ai_pick")),
                selected_by=str(row.get("selected_by") or ""),
            ),
            override=True,
        )


def load_historical_markers(
    *,
    correction_logs: Sequence[str | Path] = (),
    marker_db_path: str | Path | None = None,
    min_marker_db_sec: float = 1.0,
) -> HistoricalMarkerIndex:
    index = HistoricalMarkerIndex()
    if marker_db_path:
        _load_marker_db(index, Path(marker_db_path).expanduser(), min_marker_sec=float(min_marker_db_sec))
    for log_path in correction_logs:
        _load_correction_log(index, Path(log_path).expanduser())
    return index
