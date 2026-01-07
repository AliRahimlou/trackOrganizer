#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import base64
import shutil
import unicodedata
from datetime import datetime, date
from typing import Optional, Dict, Tuple, List

from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE

# ========= USER SETTINGS =========
mp3_source_folder        = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder   = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate/JujuVocals copy"
destination_folder       = "/Users/alirahimlou/Desktop/MUSIC/STEMS"

DRY_RUN = False

# IMPORTANT: if nothing happens, set this to False
ONLY_PROCESS_TODAY = False

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".mp4")
# =================================

# ========= CAMELOT MAP =========
camelot_wheel = {
    "ABMIN":"1A","G#MIN":"1A","EBMIN":"2A","D#MIN":"2A","BBMIN":"3A","A#MIN":"3A",
    "FMIN":"4A","CMIN":"5A","GMIN":"6A","DMIN":"7A","AMIN":"8A","EMIN":"9A",
    "BMIN":"10A","F#MIN":"11A","GBMIN":"11A","C#MIN":"12A","DBMIN":"12A",
    "BMAJ":"1B","F#MAJ":"2B","GBMAJ":"2B","C#MAJ":"3B","DBMAJ":"3B","ABMAJ":"4B",
    "G#MAJ":"4B","EBMAJ":"5B","D#MAJ":"5B","BBMAJ":"6B","A#MAJ":"6B","FMAJ":"7B",
    "CMAJ":"8B","GMAJ":"9B","DMAJ":"10B","AMAJ":"11B","EMAJ":"12B",
    "ABMINOR":"1A","G#MINOR":"1A","EBMINOR":"2A","D#MINOR":"2A","BBMINOR":"3A","A#MINOR":"3A",
    "FMINOR":"4A","CMINOR":"5A","GMINOR":"6A","DMINOR":"7A","AMINOR":"8A","EMINOR":"9A",
    "BMINOR":"10A","F#MINOR":"11A","GBMINOR":"11A","C#MINOR":"12A","DBMINOR":"12A",
    "BMAJOR":"1B","F#MAJOR":"2B","GBMAJOR":"2B","C#MAJOR":"3B","DBMAJOR":"3B",
    "ABMAJOR":"4B","G#MAJOR":"4B","EBMAJOR":"5B","D#MAJOR":"5B","BBMAJOR":"6B",
    "A#MAJOR":"6B","FMAJOR":"7B","CMAJOR":"8B","GMAJOR":"9B","DMAJOR":"10B",
    "AMAJOR":"11B","EMAJOR":"12B"
}

_CAM_REGEX      = re.compile(r"^\s*0?([1-9]|1[0-2])\s*([abAB])\s*$")
_OPENKEY_REGEX  = re.compile(r"^\s*0?([1-9]|1[0-2])\s*([mdMD])\s*$")
_B64_RE         = re.compile(r'^[A-Za-z0-9+/=]+$')
_MIK_CMT_RE     = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[-–—]\s*(\d{1,2}\s*[AB])\s*[-–—]\s*(\d{1,2})\b")

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def _as_float(x) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _clean_bpm_val(x) -> Optional[int]:
    v = _as_float(x)
    return int(round(v)) if v and v > 0 else None

def normalize_key_text(key: str) -> str:
    k = nfc(key)
    if not k or k.lower() in {"n/a","unknown","all"}:
        return ""
    m = _CAM_REGEX.match(k)
    if m:
        n, let = m.groups()
        return f"{int(n)}{let.upper()}"
    m2 = _OPENKEY_REGEX.match(k)
    if m2:
        n, md = m2.groups()
        return f"{int(n)}{'A' if md.lower()=='m' else 'B'}"
    k = re.sub(r"\s+","",k.upper()).replace("MAJOR","MAJ").replace("MINOR","MIN")
    if re.fullmatch(r"[A-G](#|B)?", k):
        k += "MAJ"
    if re.fullmatch(r"[A-G](#|B)?M", k):
        k = k[:-1] + "MIN"
    return k

def to_camelot(key: str) -> str:
    nk = normalize_key_text(key)
    if not nk:
        return "Unknown Key"
    if _CAM_REGEX.match(nk):
        return nk
    return camelot_wheel.get(nk, "Unknown Key")

def _try_decode_base64(s: str) -> Optional[str]:
    s = s.strip()
    if not s or not _B64_RE.match(s) or len(s) % 4 != 0:
        return None
    try:
        return base64.b64decode(s).decode("utf-8","ignore")
    except Exception:
        return None

def _extract_key_from_blob(val: str) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            for k in ("key","camelot","camelot_key","initial_key"):
                if obj.get(k):
                    return str(obj[k]).strip()
        except Exception:
            pass
    decoded = _try_decode_base64(s)
    if decoded:
        return _extract_key_from_blob(decoded)
    return s

def parse_mik_comment(text: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    if not text:
        return None, None, None
    m = _MIK_CMT_RE.search(text)
    if not m:
        return None, None, None
    bpm = _clean_bpm_val(m.group(1))
    key = to_camelot(m.group(2))
    try:
        energy = int(m.group(3))
        if not (1 <= energy <= 12):
            energy = None
    except Exception:
        energy = None
    return bpm, key, energy

# ---- tag reading (same logic you had) ----
def _get_id3_text(frame) -> Optional[str]:
    try:
        return str(frame.text[0]).strip() if frame and getattr(frame, "text", None) else None
    except Exception:
        return None

def read_id3_bpm(audio: ID3) -> Optional[int]:
    for comm in audio.getall("COMM"):
        for t in getattr(comm, "text", []):
            b, _, _ = parse_mik_comment(str(t))
            if b:
                return b
    bpm = _clean_bpm_val(_get_id3_text(audio.get("TBPM")))
    if bpm:
        return bpm
    for txxx in audio.getall("TXXX"):
        if (txxx.desc or "").strip().lower() in {"bpm","tbpm","tempo"} and txxx.text:
            bpm = _clean_bpm_val(txxx.text[0])
            if bpm:
                return bpm
    return None

def read_id3_key(audio: ID3) -> Optional[str]:
    for comm in audio.getall("COMM"):
        for t in getattr(comm, "text", []):
            _, kk, _ = parse_mik_comment(str(t))
            if kk and kk != "Unknown Key":
                return kk
    raw = _get_id3_text(audio.get("TKEY"))
    k = _extract_key_from_blob(raw) if raw else None
    if k:
        return k
    for txxx in audio.getall("TXXX"):
        if (txxx.desc or "").strip().lower() in {"initialkey","initial key","key","mik key","mixed in key","mixinkey key","tone"} and txxx.text:
            cand = _extract_key_from_blob(str(txxx.text[0]).strip())
            if cand:
                return cand
    return None

def read_id3_energy(audio: ID3) -> Optional[int]:
    for txxx in audio.getall("TXXX"):
        if (txxx.desc or "").strip().lower() in {"energy","mik energy","energy level"} and txxx.text:
            try:
                e = int(str(txxx.text[0]).strip())
                if 1 <= e <= 12:
                    return e
            except Exception:
                pass
    for comm in audio.getall("COMM"):
        for t in getattr(comm, "text", []):
            _, _, e = parse_mik_comment(str(t))
            if e:
                return e
    return None

def read_flac_bpm(audio: FLAC) -> Optional[int]:
    for tag, vals in audio.items():
        if not vals:
            continue
        if tag.upper() in {"COMMENT","COMMENTS","DESCRIPTION"}:
            for v in vals:
                b, _, _ = parse_mik_comment(str(v))
                if b:
                    return b
    for f in ("BPM","TBPM","TEMPO"):
        if f in audio and audio[f]:
            bpm = _clean_bpm_val(audio[f][0])
            if bpm:
                return bpm
    return None

def read_flac_key(audio: FLAC) -> Optional[str]:
    for tag, vals in audio.items():
        if tag.upper() in {"COMMENT","COMMENTS","DESCRIPTION"} and vals:
            for v in vals:
                _, k, _ = parse_mik_comment(str(v))
                if k and k != "Unknown Key":
                    return k
    for tag in ("KEY","INITIALKEY","INITIAL_KEY","TKEY","MIXEDINKEY","MIXED_IN_KEY","MIKKEY"):
        if tag in audio and audio[tag]:
            k = _extract_key_from_blob(audio[tag][0])
            if k:
                return k
    return None

def read_flac_energy(audio: FLAC) -> Optional[int]:
    for tag in ("ENERGY","MIK ENERGY","ENERGY_LEVEL"):
        if tag in audio and audio[tag]:
            try:
                e = int(str(audio[tag][0]).strip())
                if 1 <= e <= 12:
                    return e
            except Exception:
                pass
    for tag, vals in audio.items():
        if tag.upper() in {"COMMENT","COMMENTS","DESCRIPTION"} and vals:
            for v in vals:
                _, _, e = parse_mik_comment(str(v))
                if e:
                    return e
    return None

def get_bpm_key_energy(file_path: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mp3":
            audio = ID3(file_path)
            return read_id3_bpm(audio), read_id3_key(audio), read_id3_energy(audio)
        if ext == ".flac":
            audio = FLAC(file_path)
            return read_flac_bpm(audio), read_flac_key(audio), read_flac_energy(audio)
        if ext == ".wav":
            audio = WAVE(file_path)
            bpm = key = energy = None
            if isinstance(audio.tags, ID3):
                bpm    = read_id3_bpm(audio.tags)
                key    = read_id3_key(audio.tags)
                energy = read_id3_energy(audio.tags)
            return bpm, key, energy
        return None, None, None
    except ID3NoHeaderError:
        return None, None, None
    except Exception as e:
        print(f"[ERROR] Metadata error {file_path}: {e}")
        return None, None, None

# ========= Source library index =========
def _ext_priority(p: str) -> int:
    pl = p.lower()
    if pl.endswith(".mp3"): return 0
    if pl.endswith(".flac"): return 1
    if pl.endswith(".wav"): return 2
    return 3

def index_source_library(mp3_folder: str) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for root, _, files in os.walk(mp3_folder):
        for f in files:
            if not f.lower().endswith(AUDIO_EXTS):
                continue
            base_exact = nfc(os.path.splitext(f)[0])
            path = os.path.join(root, f)
            if base_exact not in best:
                best[base_exact] = path
            else:
                a, b = best[base_exact], path
                ap, bp = _ext_priority(a), _ext_priority(b)
                if bp < ap or (bp == ap and os.path.getmtime(b) > os.path.getmtime(a)):
                    best[base_exact] = b
    return best

# ========= Matching helpers for weird vocal filenames =========
_TRAIL_GARBAGE_RE = re.compile(r"(?i)(?:_pn|_p)?$")

def candidate_track_keys_from_filename(filename_no_ext: str) -> List[str]:
    """
    Generate multiple normalized keys to increase hit rate vs PlaylistsByDate basenames.
    """
    raw = nfc(filename_no_ext)

    cands = []

    # 0) raw as-is
    cands.append(raw)

    # 1) remove trailing _pn
    cands.append(re.sub(r"(?i)_pn$", "", raw).strip())

    # 2) replace underscores with spaces
    cands.append(re.sub(r"_+", " ", raw).strip())

    # 3) replace underscores + remove _pn
    cands.append(re.sub(r"_+", " ", re.sub(r"(?i)_pn$", "", raw)).strip())

    # 4) remove leading track numbers like "01 ", "01-"
    def strip_leading_tracknum(s: str) -> str:
        return re.sub(r"^\s*\d{1,3}\s*[-_ ]\s*", "", s).strip()

    cands = cands + [strip_leading_tracknum(x) for x in list(cands)]

    # 5) remove common acapella markers
    def strip_acapella(s: str) -> str:
        s2 = re.sub(r"(?i)\bacapella\b", "", s)
        s2 = re.sub(r"[\(\)\[\]]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    cands = cands + [strip_acapella(x) for x in list(cands)]

    # dedupe, keep order
    seen = set()
    out = []
    for x in cands:
        x = nfc(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def find_source_for_vocal_file(idx: Dict[str, str], vocal_file_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(vocal_file_path))[0]
    for cand in candidate_track_keys_from_filename(base):
        hit = idx.get(nfc(cand))
        if hit:
            return hit
    return None

# ========= Safety / ops =========
def is_today(ts: float) -> bool:
    return datetime.fromtimestamp(ts).date() == date.today()

def file_is_today(path: str) -> bool:
    if not ONLY_PROCESS_TODAY:
        return True
    try:
        return is_today(os.path.getmtime(path))
    except Exception:
        return False

def prefix_name(original_filename: str, bpm: int, cam: str, energy: Optional[int]) -> str:
    energy_str = str(energy) if energy is not None else "NA"
    prefix = f"{bpm}_{cam}_{energy_str}-"
    if original_filename.startswith(prefix):
        return original_filename
    return nfc(prefix + original_filename).replace("/", "-")

def ensure_dir(path: str):
    if DRY_RUN:
        return
    os.makedirs(path, exist_ok=True)

def move_and_rename_file(src: str, bpm: int, cam: str, energy: Optional[int]):
    dest_dir = os.path.join(destination_folder, str(bpm), cam)
    new_name = prefix_name(os.path.basename(src), bpm, cam, energy)
    dest_path = os.path.join(dest_dir, new_name)

    print(f"[MOVE] {src}")
    print(f"   -> {dest_path}")

    if DRY_RUN:
        return

    os.makedirs(dest_dir, exist_ok=True)

    # If file exists, add a counter
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(new_name)
        i = 2
        while True:
            cand = os.path.join(dest_dir, f"{name} ({i}){ext}")
            if not os.path.exists(cand):
                dest_path = cand
                break
            i += 1

    shutil.move(src, dest_path)

    # move .asd if present
    asd_src = src + ".asd"
    asd_dst = dest_path + ".asd"
    if os.path.exists(asd_src):
        shutil.move(asd_src, asd_dst)

def process_vocals(idx: Dict[str, str]):
    if not os.path.isdir(htdemucs_source_folder):
        print(f"[ERROR] Missing folder: {htdemucs_source_folder}")
        return

    moved = 0
    skipped = 0

    for root, _, files in os.walk(htdemucs_source_folder):
        for f in files:
            if not f.lower().endswith(AUDIO_EXTS):
                continue
            src = os.path.join(root, f)

            if not file_is_today(src):
                skipped += 1
                continue

            src_audio = find_source_for_vocal_file(idx, src)
            if not src_audio:
                print(f"[SKIP] No PlaylistsByDate match for: {f}")
                skipped += 1
                continue

            bpm, key, energy = get_bpm_key_energy(src_audio)
            cam = to_camelot(key or "")
            if bpm is None or cam == "Unknown Key":
                print(f"[SKIP] Missing BPM/Key in source tags: {os.path.basename(src_audio)}")
                skipped += 1
                continue

            move_and_rename_file(src, bpm, cam, energy)
            moved += 1

    print(f"\n[SUMMARY] moved={moved} skipped={skipped}")
    if DRY_RUN:
        print("🟡 DRY RUN enabled — set DRY_RUN=False to actually move/rename.")

# ========= MAIN =========
if __name__ == "__main__":
    if input("Ready? Type 'yes' to begin: ").strip().lower() == "yes":
        idx = index_source_library(mp3_source_folder)
        process_vocals(idx)
        print("🎵 Done.")
    else:
        print("Operation cancelled.")
