#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse
from typing import Optional, Tuple, Dict, Any, List

# --- Mutagen formats ---
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from mutagen.aiff import AIFF
from mutagen.mp4 import MP4, MP4StreamInfoError

# ---------- Camelot mapping (supports lots of synonyms) ----------
CAMELOT_WHEEL: Dict[str, str] = {
    "ABMIN": "1A", "G#MIN": "1A", "EBMIN": "2A", "D#MIN": "2A", "BBMIN": "3A", "A#MIN": "3A",
    "FMIN": "4A", "CMIN": "5A", "GMIN": "6A", "DMIN": "7A", "AMIN": "8A", "EMIN": "9A",
    "BMIN": "10A", "F#MIN": "11A", "GBMIN": "11A", "C#MIN": "12A", "DBMIN": "12A",

    "BMAJ": "1B", "F#MAJ": "2B", "GBMAJ": "2B", "C#MAJ": "3B", "DBMAJ": "3B",
    "ABMAJ": "4B", "G#MAJ": "4B", "EBMAJ": "5B", "D#MAJ": "5B",
    "BBMAJ": "6B", "A#MAJ": "6B", "FMAJ": "7B", "CMAJ": "8B",
    "GMAJ": "9B", "DMAJ": "10B", "AMAJ": "11B", "EMAJ": "12B",

    # extra synonyms
    "ABMINOR": "1A", "G#MINOR": "1A", "EBMINOR": "2A", "D#MINOR": "2A",
    "BBMINOR": "3A", "A#MINOR": "3A", "FMINOR": "4A", "CMINOR": "5A",
    "GMINOR": "6A", "DMINOR": "7A", "AMINOR": "8A", "EMINOR": "9A",
    "BMINOR": "10A", "F#MINOR": "11A", "GBMINOR": "11A", "C#MINOR": "12A", "DBMINOR": "12A",

    "BMAJOR": "1B", "F#MAJOR": "2B", "GBMAJOR": "2B", "C#MAJOR": "3B", "DBMAJOR": "3B",
    "ABMAJOR": "4B", "G#MAJOR": "4B", "EBMAJOR": "5B", "D#MAJOR": "5B",
    "BBMAJOR": "6B", "A#MAJOR": "6B", "FMAJOR": "7B", "CMAJOR": "8B",
    "GMAJOR": "9B", "DMAJOR": "10B", "AMAJOR": "11B", "EMAJOR": "12B",
}

# Acceptable audio file extensions we’ll scan
AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".mp4")


def normalize_key(raw: str) -> str:
    """
    Normalize musical key strings to something we can map to Camelot:
    - Remove spaces, uppercase
    - Convert MAJOR/MINOR -> MAJ/MIN
    - If only note given (e.g., 'A', 'F#'), assume MAJ
    - If note + 'M' (e.g., 'Am'), interpret as MIN
    - Try to handle things like '8A', '10B' (already Camelot: pass through)
    """
    if not raw:
        return ""
    s = raw.strip().upper().replace(" ", "")

    # If it's already a Camelot like "8A" or "11B", let it through
    if re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", s):
        return s  # already Camelot

    # Common DJ tags like "A MINOR", "F# MAJOR"
    s = re.sub(r"MINOR$", "MIN", s)
    s = re.sub(r"MAJOR$", "MAJ", s)

    # Short forms: Am, F#m, A, G#, etc.
    if re.fullmatch(r"^[A-G](#|B)?M$", s):  # e.g., AM, F#M -> treat as MIN
        s = s[:-1] + "MIN"
    elif re.fullmatch(r"^[A-G](#|B)?$", s):  # just the note -> assume MAJ
        s = s + "MAJ"
    elif re.fullmatch(r"^[A-G](#|B)?MIN$", s) or re.fullmatch(r"^[A-G](#|B)?MAJ$", s):
        pass
    else:
        # Some vendors embed Unicode symbols like '♭' or '♯'
        s = s.replace("♭", "B").replace("♯", "#")
        # Retry a quick normalization (best effort)
        if re.fullmatch(r"^[A-G](#|B)?$", s):
            s += "MAJ"
        elif s.endswith("M"):  # Am -> AMIN, etc.
            s = s[:-1] + "MIN"

    return s


def to_camelot(key_str: str) -> str:
    """
    Convert normalized key (or Camelot) to Camelot. If unknown, return 'Unknown'.
    """
    if not key_str:
        return "Unknown"
    # If already Camelot
    if re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", key_str):
        return key_str
    norm = normalize_key(key_str)
    return CAMELOT_WHEEL.get(norm, "Unknown")


def as_float_str(x: Any) -> Optional[str]:
    try:
        v = float(str(x).strip())
        # BPM like 124.0 → '124', keep decimals if present
        return f"{v:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return None


# ------------------------- Readers per format -------------------------

def read_id3_tags(path: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    MP3, WAV (ID3), AIFF (ID3): return (bpm, key, source)
    """
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        return None, None, "id3:none"
    except Exception:
        return None, None, "id3:error"

    bpm = None
    key = None
    source = "id3"

    # BPM (TBPM or TXXX frames)
    if "TBPM" in tags:
        bpm = as_float_str(tags["TBPM"].text[0])
        source += ":TBPM"
    else:
        # Some apps store in TXXX:tempo / bpm
        for k, frame in tags.items():
            if k.startswith("TXXX") and hasattr(frame, "desc"):
                desc = (frame.desc or "").lower()
                if desc in ("tempo", "bpm"):
                    bpm = as_float_str(frame.text[0])
                    source += ":TXXX-bpm"
                    break

    # KEY
    if "TKEY" in tags:
        key = str(tags["TKEY"].text[0]).strip()
        source += ":TKEY"
    else:
        # Look for TXXX key variants used by Rekordbox/others
        for k, frame in tags.items():
            if k.startswith("TXXX") and hasattr(frame, "desc"):
                desc = (frame.desc or "").lower()
                if desc in ("initialkey", "key", "initial key"):
                    key = str(frame.text[0]).strip()
                    source += f":TXXX-{desc}"
                    break

    return bpm, key, source


def read_flac_tags(path: str) -> Tuple[Optional[str], Optional[str], str]:
    try:
        f = FLAC(path)
    except Exception:
        return None, None, "flac:error"

    # Common FLAC tag names
    bpm_candidates = ["BPM", "TEMPO", "TBPM"]
    key_candidates = ["INITIALKEY", "KEY", "TKEY", "INITIAL_KEY"]

    bpm = None
    for k in bpm_candidates:
        if k in f:
            bpm = as_float_str(f[k][0])
            if bpm:
                break

    key = None
    for k in key_candidates:
        if k in f:
            key = f[k][0].strip()
            if key:
                break

    return bpm, key, "flac"


def read_wav_tags(path: str) -> Tuple[Optional[str], Optional[str], str]:
    # WAV can carry ID3 chunks; mutagen.WAVE exposes .tags (ID3)
    try:
        w = WAVE(path)
    except Exception:
        return None, None, "wav:error"

    if w.tags:
        # Reuse the ID3 reader approach by saving temp? Not needed; .tags behaves like ID3
        try:
            # Try TBPM / TKEY directly
            bpm = None
            key = None
            src = "wav:id3"

            if "TBPM" in w.tags:
                bpm = as_float_str(w.tags["TBPM"].text[0])
                src += ":TBPM"

            if "TKEY" in w.tags:
                key = str(w.tags["TKEY"].text[0]).strip()
                src += ":TKEY"

            # TXXX fallback
            if bpm is None:
                for k, frame in w.tags.items():
                    if k.startswith("TXXX") and hasattr(frame, "desc"):
                        if (frame.desc or "").lower() in ("tempo", "bpm"):
                            bpm = as_float_str(frame.text[0])
                            src += ":TXXX-bpm"
                            break
            if key is None:
                for k, frame in w.tags.items():
                    if k.startswith("TXXX") and hasattr(frame, "desc"):
                        if (frame.desc or "").lower() in ("initialkey", "key", "initial key"):
                            key = str(frame.text[0]).strip()
                            src += ":TXXX-key"
                            break

            return bpm, key, src
        except Exception:
            return None, None, "wav:id3-error"

    return None, None, "wav:no-tags"


def read_aiff_tags(path: str) -> Tuple[Optional[str], Optional[str], str]:
    try:
        a = AIFF(path)
    except Exception:
        return None, None, "aiff:error"

    if a.tags:
        # a.tags is ID3-like
        try:
            bpm = None
            key = None
            src = "aiff:id3"

            if "TBPM" in a.tags:
                bpm = as_float_str(a.tags["TBPM"].text[0])
                src += ":TBPM"
            if "TKEY" in a.tags:
                key = str(a.tags["TKEY"].text[0]).strip()
                src += ":TKEY"

            if bpm is None:
                for k, frame in a.tags.items():
                    if k.startswith("TXXX") and hasattr(frame, "desc"):
                        if (frame.desc or "").lower() in ("tempo", "bpm"):
                            bpm = as_float_str(frame.text[0])
                            src += ":TXXX-bpm"
                            break
            if key is None:
                for k, frame in a.tags.items():
                    if k.startswith("TXXX") and hasattr(frame, "desc"):
                        if (frame.desc or "").lower() in ("initialkey", "key", "initial key"):
                            key = str(frame.text[0]).strip()
                            src += ":TXXX-key"
                            break
            return bpm, key, src
        except Exception:
            return None, None, "aiff:id3-error"

    return None, None, "aiff:no-tags"


def read_mp4_tags(path: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    M4A/MP4:
    - BPM: 'tmpo' (int/float)
    - Key: '©key' or custom '----:com.apple.iTunes:initialkey'
    """
    try:
        m = MP4(path)
    except MP4StreamInfoError:
        # Audio-less MP4 or odd file; still try tags
        try:
            m = MP4(path)
        except Exception:
            return None, None, "mp4:error"
    except Exception:
        return None, None, "mp4:error"

    bpm = None
    key = None
    src = "mp4"

    if "tmpo" in m:
        bpm = as_float_str(m["tmpo"][0])
        src += ":tmpo"

    if "©key" in m:
        key = str(m["©key"][0]).strip()
        src += ":©key"
    else:
        # Look for freeform atoms
        for k, v in m.items():
            if isinstance(k, str) and k.startswith("----:"):
                # k example: ----:com.apple.iTunes:initialkey
                if k.lower().endswith("initialkey") or k.lower().endswith(":key"):
                    try:
                        # v is list of bytes
                        if v and isinstance(v[0], (bytes, bytearray)):
                            key = v[0].decode("utf-8", errors="replace").strip()
                            src += ":-----initialkey"
                            break
                    except Exception:
                        pass

    return bpm, key, src


def read_bpm_key(path: str) -> Tuple[Optional[str], Optional[str], str]:
    p = path.lower()
    if p.endswith(".mp3"):
        return read_id3_tags(path)
    if p.endswith(".flac"):
        return read_flac_tags(path)
    if p.endswith(".wav"):
        return read_wav_tags(path)
    if p.endswith(".aiff") or p.endswith(".aif"):
        return read_aiff_tags(path)
    if p.endswith(".m4a") or p.endswith(".mp4"):
        return read_mp4_tags(path)
    # Fallback attempt with ID3 (some odd containers)
    return read_id3_tags(path)


# ------------------------- Search helpers -------------------------

def find_candidates(root: str, query: str, limit: int = 30) -> List[str]:
    """
    Case-insensitive filename substring match under root.
    """
    q = query.lower()
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(AUDIO_EXTS) and q in fn.lower():
                hits.append(os.path.join(dirpath, fn))
                if len(hits) >= limit:
                    return hits
    return hits


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Read BPM and Key (and Camelot) from audio file metadata (Rekordbox/Serato-compatible tags)."
    )
    parser.add_argument("path", nargs="?", help="Path to an audio file. If omitted, enter interactive search.")
    parser.add_argument("--root", default="/Users/alirahimlou/Desktop/MUSIC",
                        help="Root folder for interactive search (default: ~/Desktop/MUSIC)")
    parser.add_argument("--no-camelot", action="store_true", help="Do not show Camelot conversion.")
    args = parser.parse_args()

    path = args.path

    if not path:
        # Interactive search
        root = os.path.expanduser(args.root)
        if not os.path.isdir(root):
            print(f"[ERROR] Search root not found: {root}")
            sys.exit(1)

        query = input("🔎 Enter part of the filename to search: ").strip()
        if not query:
            print("No query, exiting.")
            sys.exit(0)

        hits = find_candidates(root, query)
        if not hits:
            print("No matching files.")
            sys.exit(0)

        print("\nSelect a file:")
        for i, h in enumerate(hits, 1):
            print(f"{i:>2}. {h}")
        try:
            idx = int(input("\nEnter number: ").strip())
            if not (1 <= idx <= len(hits)):
                raise ValueError
            path = hits[idx - 1]
        except Exception:
            print("Invalid selection.")
            sys.exit(1)

    # Validate path
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    bpm, key, source = read_bpm_key(path)

    camelot = None if args.no_camelot else to_camelot(key or "")

    print("\n===== METADATA =====")
    print(f"File    : {path}")
    print(f"Source  : {source}")
    print(f"BPM     : {bpm or 'Unknown'}")
    print(f"Key     : {key or 'Unknown'}")
    if not args.no_camelot:
        print(f"Camelot : {camelot or 'Unknown'}")

    # Compact single-line for scripting
    compact = {
        "file": path,
        "bpm": bpm or "Unknown",
        "key": key or "Unknown",
        "camelot": (camelot or "Unknown") if not args.no_camelot else None,
        "source": source,
    }
    # Pretty minimal JSON-ish output without importing json (keeps it copy/paste simple)
    kvs = [f'"{k}":"{v}"' for k, v in compact.items() if v is not None]
    print("\n{ " + ", ".join(kvs) + " }")


if __name__ == "__main__":
    main()
