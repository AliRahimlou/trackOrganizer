#!/usr/bin/env python3

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse


_STEM_ROLE_PREFIX_RE = re.compile(r"^(?:drums|inst|vocals)[-_ ]*", re.I)
_STEM_META_PREFIX_RE = re.compile(r"^(?:\d{2,3}[_ -]*\d{1,2}[abmd](?:[_ -]*\d{1,2})?|\d{1,3}\s*-)\s*[-_ ]*", re.I)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ENERGY_LABEL_RE = re.compile(r"\benergy\s*([0-9]+)\b", re.I)
_DROP_LABEL_RE = re.compile(r"\b(drop|chorus|hook|main|peak)\b", re.I)
_NON_DROP_LABEL_RE = re.compile(r"\b(intro|outro|break|breakdown|build|buildup|verse)\b", re.I)


@dataclass(frozen=True)
class RekordboxCue:
    start_sec: float
    name: str
    type_code: Optional[str]
    num: Optional[int]


@dataclass(frozen=True)
class RekordboxTrack:
    location: str
    name: str
    artist: str
    bpm: Optional[float]
    cues: Tuple[RekordboxCue, ...]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def _norm_title(text: str) -> str:
    text = _nfc(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(.*?remix.*?\)", "", text)
    text = re.sub(r"\s*\[.*?remix.*?\]", "", text)
    text = re.sub(r"\s*feat\.?.*| ft\.?.*", "", text)
    return text.strip()


def _strip_parenthetical_chunks(text: str) -> str:
    text = _nfc(text)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_stem_prefix(text: str) -> str:
    text = _norm_title(text)
    stripped = _STEM_ROLE_PREFIX_RE.sub("", text)
    stripped = _STEM_META_PREFIX_RE.sub("", stripped)
    stripped = re.sub(r"^[\-_ ]+", "", stripped)
    return stripped.strip()


def _compact_key(text: str) -> str:
    text = _strip_stem_prefix(text)
    text = _strip_parenthetical_chunks(text)
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = _NON_ALNUM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_artist_title(text: str) -> Optional[Tuple[str, str]]:
    text = _strip_stem_prefix(text)
    parts = [part.strip() for part in re.split(r"\s+-\s+", text, maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _key_variants(text: str) -> List[str]:
    variants: List[str] = []
    seen = set()

    def add(value: str) -> None:
        key = _norm_title(value)
        if key and key not in seen:
            seen.add(key)
            variants.append(key)
        compact = _compact_key(value)
        if compact and compact not in seen:
            seen.add(compact)
            variants.append(compact)

    add(text)
    stripped = _strip_stem_prefix(text)
    if stripped and stripped != _norm_title(text):
        add(stripped)
    without_chunks = _strip_parenthetical_chunks(text)
    if without_chunks and _norm_title(without_chunks) != _norm_title(text):
        add(without_chunks)
    if stripped:
        stripped_without_chunks = _strip_parenthetical_chunks(stripped)
        if stripped_without_chunks and _norm_title(stripped_without_chunks) != _norm_title(stripped):
            add(stripped_without_chunks)

    parts = _split_artist_title(text)
    if parts is not None:
        artist, title = parts
        add(f"{artist} - {title}")
        add(f"{title} - {artist}")
        add(title)
        add(_strip_parenthetical_chunks(title))

    return variants


def _tokenize_key(text: str) -> Tuple[str, ...]:
    compact = _compact_key(text)
    if not compact:
        return ()
    return tuple(tok for tok in compact.split() if tok)


def _key_match_score(query_key: str, candidate_key: str) -> Tuple[float, int, float]:
    query_tokens = set(_tokenize_key(query_key))
    candidate_tokens = set(_tokenize_key(candidate_key))
    if not query_tokens or not candidate_tokens:
        return 0.0, 0, 0.0

    shared = len(query_tokens & candidate_tokens)
    if shared <= 0:
        return 0.0, 0, 0.0

    token_containment = float(shared) / float(max(1, min(len(query_tokens), len(candidate_tokens))))
    token_jaccard = float(shared) / float(max(1, len(query_tokens | candidate_tokens)))
    char_ratio = SequenceMatcher(None, _compact_key(query_key), _compact_key(candidate_key)).ratio()
    score = max(
        char_ratio,
        (0.70 * token_containment) + (0.30 * token_jaccard),
    )
    return float(score), int(shared), float(token_containment)


def _lookup_fuzzy_matching_track(
    index: Dict[str, Tuple[RekordboxTrack, ...]],
    *,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
) -> Tuple[Optional[RekordboxTrack], str]:
    ranked: List[Tuple[float, int, float, str, Tuple[RekordboxTrack, ...]]] = []
    query_keys = _candidate_keys(track_dir=track_dir, source_audio_path=source_audio_path, stem_paths=stem_paths)
    for query_key in query_keys:
        for candidate_key, tracks in index.items():
            if not tracks:
                continue
            score, shared, containment = _key_match_score(query_key, candidate_key)
            if shared < 2:
                continue
            if containment < 0.60 and score < 0.84:
                continue
            if shared < 3 and score < 0.90:
                continue
            ranked.append((float(score), int(shared), float(containment), str(candidate_key), tracks))

    if not ranked:
        return None, ""

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    best_score, best_shared, best_containment, best_key, best_tracks = ranked[0]
    if len(ranked) > 1:
        next_score, next_shared, next_containment, _, _ = ranked[1]
        if (best_score - next_score) < 0.05 and best_shared == next_shared and abs(best_containment - next_containment) < 0.05:
            return None, ""

    if len(best_tracks) > 1 and _is_broad_lookup_key(best_key):
        return None, ""
    return best_tracks[0], best_key


def _location_path(location: str) -> str:
    location = str(location or "")
    if location.startswith("file://"):
        return unquote(urlparse(location).path)
    return location


def _basename_key(path_or_location: str) -> str:
    path = _location_path(path_or_location)
    return _norm_title(os.path.splitext(os.path.basename(path))[0])


def _folder_key(path_or_location: str) -> str:
    path = _location_path(path_or_location)
    return _norm_title(os.path.basename(os.path.dirname(path)))


def _parse_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _track_keys(track: RekordboxTrack) -> List[str]:
    keys: List[str] = []
    seen = set()

    def add(value: str) -> None:
        for key in _key_variants(value):
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

    add(_basename_key(track.location))
    add(_folder_key(track.location))
    add(track.name)
    if track.artist and track.name:
        add(f"{track.artist} - {track.name}")
    return keys


def _dedupe_cues(cues: Iterable[RekordboxCue]) -> Tuple[RekordboxCue, ...]:
    by_ms: Dict[int, RekordboxCue] = {}
    for cue in sorted(cues, key=lambda item: (float(item.start_sec), item.num is None, item.num if item.num is not None else 999, item.name)):
        key = int(round(float(cue.start_sec) * 1000.0))
        prev = by_ms.get(key)
        if prev is None:
            by_ms[key] = cue
            continue
        prev_rank = (prev.num is None or prev.num < 0, prev.num if prev.num is not None else 999)
        cue_rank = (cue.num is None or cue.num < 0, cue.num if cue.num is not None else 999)
        if cue_rank < prev_rank:
            by_ms[key] = cue
    ordered = sorted(by_ms.values(), key=lambda item: (float(item.start_sec), item.num if item.num is not None else 999))
    return tuple(ordered)


def _cue_energy(cue: RekordboxCue) -> Optional[int]:
    match = _ENERGY_LABEL_RE.search(cue.name or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _is_explicit_drop_label(cue: RekordboxCue) -> bool:
    name = cue.name or ""
    if not _DROP_LABEL_RE.search(name):
        return False
    return not bool(_NON_DROP_LABEL_RE.search(name))


@lru_cache(maxsize=8)
def load_rekordbox_index(xml_path: str) -> Dict[str, Tuple[RekordboxTrack, ...]]:
    xml_path = os.path.abspath(str(xml_path or ""))
    if not xml_path or not os.path.exists(xml_path):
        return {}

    root = ET.parse(xml_path).getroot()
    index: Dict[str, List[RekordboxTrack]] = {}
    for track_node in root.iter("TRACK"):
        location = _location_path(track_node.get("Location") or track_node.get("LOCATION") or "")
        name = _nfc(track_node.get("Name") or "")
        artist = _nfc(track_node.get("Artist") or "")
        bpm = None
        for attr in ("AverageBpm", "AVERAGEBPM", "BPM", "Tempo", "Bpm"):
            bpm = _parse_float(track_node.get(attr))
            if bpm is not None:
                break

        cues: List[RekordboxCue] = []
        for cue_node in track_node.iter("POSITION_MARK"):
            start_sec = _parse_float(cue_node.get("Start") or cue_node.get("Position"))
            if start_sec is None:
                continue
            cues.append(
                RekordboxCue(
                    start_sec=float(start_sec),
                    name=_nfc(cue_node.get("Name") or ""),
                    type_code=_nfc(cue_node.get("Type") or "") or None,
                    num=_parse_int(cue_node.get("Num")),
                )
            )
        deduped_cues = _dedupe_cues(cues)
        if not deduped_cues:
            continue

        track = RekordboxTrack(
            location=location,
            name=name,
            artist=artist,
            bpm=bpm,
            cues=deduped_cues,
        )
        for key in _track_keys(track):
            index.setdefault(key, []).append(track)

    return {key: tuple(value) for key, value in index.items()}


def _candidate_keys(
    *,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    keys: List[str] = []
    seen = set()

    def add(value: str) -> None:
        for key in _key_variants(value):
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

    add(os.path.basename(track_dir))
    if source_audio_path:
        add(os.path.splitext(os.path.basename(source_audio_path))[0])

    for path in stem_paths or []:
        base = os.path.splitext(os.path.basename(path))[0]
        add(base)
        if "-" in base:
            add(base.split("-", 1)[1])

    return keys


def _is_broad_lookup_key(key: str) -> bool:
    return " - " not in key and len(key.split()) < 4


def _lookup_matching_track(
    index: Dict[str, Tuple[RekordboxTrack, ...]],
    *,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
    allow_fuzzy: bool = True,
) -> Tuple[Optional[RekordboxTrack], str]:
    for key in _candidate_keys(track_dir=track_dir, source_audio_path=source_audio_path, stem_paths=stem_paths):
        tracks = tuple(index.get(key, ()))
        if not tracks:
            continue
        if len(tracks) > 1 and _is_broad_lookup_key(key):
            continue
        return tracks[0], key
    if not allow_fuzzy:
        return None, ""
    return _lookup_fuzzy_matching_track(
        index,
        track_dir=track_dir,
        source_audio_path=source_audio_path,
        stem_paths=stem_paths,
    )


def select_first_drop_cue(
    cues: Sequence[RekordboxCue],
    *,
    preferred_num: int = 1,
    early_ignore_sec: float = 1.0,
) -> Optional[RekordboxCue]:
    ordered = sorted(cues, key=lambda cue: (float(cue.start_sec), cue.num if cue.num is not None else 999))
    usable = [cue for cue in ordered if float(cue.start_sec) >= float(early_ignore_sec)]
    if not usable and ordered:
        usable = [ordered[0]]

    for cue in usable:
        if _is_explicit_drop_label(cue):
            return cue

    energy_rows: List[Tuple[RekordboxCue, int, Optional[int]]] = []
    previous_energy: Optional[int] = None
    for cue in ordered:
        energy = _cue_energy(cue)
        if energy is None:
            continue
        if float(cue.start_sec) >= float(early_ignore_sec):
            energy_rows.append((cue, int(energy), previous_energy))
        previous_energy = int(energy)

    if energy_rows:
        max_energy = max(energy for _cue, energy, _previous in energy_rows)
        for cue, energy, previous in energy_rows:
            strong_jump = previous is not None and energy >= previous + 2
            strong_section = energy >= 7 or energy >= max(6, max_energy - 1)
            if energy >= 6 and (strong_jump or strong_section):
                return cue
        for cue, energy, previous in energy_rows:
            if energy == max_energy and energy >= 6:
                return cue
            if energy >= 5 and (previous is None or energy > previous):
                return cue

    for cue in ordered:
        if cue.num == int(preferred_num) and float(cue.start_sec) >= float(early_ignore_sec):
            return cue

    non_negative = [cue for cue in ordered if cue.num is not None and cue.num >= 0 and float(cue.start_sec) >= float(early_ignore_sec)]
    if len(non_negative) >= 2 and int(non_negative[0].num or -1) == 0:
        return non_negative[1]
    if non_negative:
        return non_negative[0]

    usable = [cue for cue in ordered if float(cue.start_sec) >= float(early_ignore_sec)]
    if usable:
        return usable[0]
    return ordered[0] if ordered else None


def lookup_first_drop_prior(
    *,
    xml_path: str,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
    preferred_num: int = 1,
    confidence: float = 0.98,
) -> Tuple[Optional[float], float, str]:
    cue, cue_confidence, reason = lookup_first_drop_cue(
        xml_path=xml_path,
        track_dir=track_dir,
        source_audio_path=source_audio_path,
        stem_paths=stem_paths,
        preferred_num=preferred_num,
        confidence=confidence,
    )
    if cue is not None:
        return float(cue.start_sec), float(cue_confidence), reason
    return None, 0.0, ""


def lookup_first_drop_cue(
    *,
    xml_path: str,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
    preferred_num: int = 1,
    confidence: float = 0.98,
    allow_fuzzy: bool = True,
) -> Tuple[Optional[RekordboxCue], float, str]:
    index = load_rekordbox_index(xml_path)
    if not index:
        return None, 0.0, ""

    track, key = _lookup_matching_track(
        index,
        track_dir=track_dir,
        source_audio_path=source_audio_path,
        stem_paths=stem_paths,
        allow_fuzzy=allow_fuzzy,
    )
    if track is not None:
        cue = select_first_drop_cue(track.cues, preferred_num=preferred_num)
        if cue is not None:
            cue_name = cue.name or "cue"
            reason = f"rekordbox_mik:{cue_name} num={cue.num if cue.num is not None else 'n/a'} match={key}"
            return cue, float(confidence), reason
    return None, 0.0, ""


def lookup_track_cues(
    *,
    xml_path: str,
    track_dir: str,
    source_audio_path: Optional[str] = None,
    stem_paths: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[float, ...], str]:
    index = load_rekordbox_index(xml_path)
    if not index:
        return (), ""

    track, key = _lookup_matching_track(
        index,
        track_dir=track_dir,
        source_audio_path=source_audio_path,
        stem_paths=stem_paths,
    )
    if track is not None:
        cues = tuple(float(cue.start_sec) for cue in track.cues)
        if cues:
            return cues, key
    return (), ""
