#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline (MiK-only cues):
- Index library, read BPM/Key/Energy from tags
- Load Rekordbox (MiK) cues from XML (seconds-based)
- Snap MiK cues to beat grid using BPM
- Write CH1.als: stamp WarpMarkers (SecTime + BeatTime), update LoopEnd/OutMarker, replace clip names/paths

Requires:
  pip install numpy>=2.0 scipy>=1.13 soundfile>=0.12.1 audioread>=3.0.1 librosa>=0.10.2.post1
Optional:
  pip install madmom
"""

import os, re, json, base64, html, shutil, gzip, unicodedata, warnings
from io import BytesIO
from typing import Optional, Dict, Tuple, List
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote

# --- audio tags ---
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE

# --- cue detector deps (kept but unused) ---
import numpy as np
import librosa

_HAS_MADMOM = True
try:
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
except Exception:
    _HAS_MADMOM = False

import trackTime  # your module for duration

# ========= USER SETTINGS =========
mp3_source_folder        = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder   = "/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized"
destination_folder       = "/Users/alirahimlou/Desktop/STEMS2"
ALS_FILES_FOLDER         = "alsFiles"  # folder containing <BPM>.als templates
FLAC_FOLDER              = destination_folder
SKIP_EXISTING            = False   # when False, force-regenerate ALS for every track folder

# MiK Rekordbox XML (seconds-based cues)
REKORDBOX_XMLS = [
    "/Users/alirahimlou/Documents/rekordbox_mikcues_001.xml",
]
IMPORT_RB_MEMORY = True
IMPORT_RB_HOT    = True

# grid snapping for cues → beats
SNAP_NEAR_INT_TOL = 0.125   # snap if within 1/8 beat
GRID_RES           = 24     # else quantize to 1/24 beat

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".mp4")

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

# ========= UTIL =========
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def safe_xml_value(s: str) -> str:
    if not s:
        return s
    out: List[str] = []
    for ch in s:
        code = ord(ch)
        if code > 255:
            out.append(f"&#{code};")
        elif ch == "'":
            out.append("&#39;")
        else:
            out.append(html.escape(ch, quote=True))
    return "".join(out)

_CAM_REGEX      = re.compile(r"^\s*0?([1-9]|1[0-2])\s*([abAB])\s*$")
_OPENKEY_REGEX  = re.compile(r"^\s*0?([1-9]|1[0-2])\s*([mdMD])\s*$")
_B64_RE         = re.compile(r'^[A-Za-z0-9+/=]+$')
_MIK_CMT_RE     = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[-–—]\s*(\d{1,2}\s*[AB])\s*[-–—]\s*(\d{1,2})\b")
_FN_ENERGY_RE   = re.compile(r"^(drums|inst|vocals)_\d+_[0-9]{1,2}[ab]_(\d{1,2})-", re.I)

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

# ========= READ TAGS =========
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

def get_bpm_key_energy(file_path) -> Tuple[Optional[int], Optional[str], Optional[int]]:
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

# ========= LIBRARY INDEX (EXACT after NFC) =========
def _ext_priority(p: str) -> int:
    pl = p.lower()
    if pl.endswith(".mp3"): return 0
    if pl.endswith(".flac"): return 1
    if pl.endswith(".wav"): return 2
    return 3

def index_source_library(mp3_folder) -> Dict[str,str]:
    best: Dict[str,str] = {}
    for root, _, files in os.walk(mp3_folder):
        for f in files:
            if not f.lower().endswith(AUDIO_EXTS):
                continue
            base_exact = nfc(os.path.splitext(f)[0])  # exact title normalized
            path = os.path.join(root, f)
            if base_exact not in best:
                best[base_exact] = path
            else:
                a, b = best[base_exact], path
                ap, bp = _ext_priority(a), _ext_priority(b)
                if bp < ap or (bp == ap and os.path.getmtime(b) > os.path.getmtime(a)):
                    best[base_exact] = b
    return best

def find_source_for_track(idx_exact: Dict[str,str], track_folder_name: str) -> Optional[str]:
    key = nfc(track_folder_name)
    return idx_exact.get(key)

# ========= RBX (MiK Rekordbox XML) =========
def _norm_title(s: str) -> str:
    s = nfc(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(.*?remix.*?\)", "", s)
    s = re.sub(r"\s*\[.*?remix.*?\]", "", s)
    s = re.sub(r"\s*feat\.?.*| ft\.?.*", "", s)
    return s.strip()

def _basename_from_location(loc: str) -> str:
    if not loc: return ""
    if loc.startswith("file://"):
        path = unquote(urlparse(loc).path)
    else:
        path = loc
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]
    return _norm_title(base)

def _gather_mik_rbx(root: ET.Element) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tr in root.iter("TRACK"):
        loc = tr.get("Location") or tr.get("LOCATION") or ""
        key = _basename_from_location(loc)
        if not key:
            continue

        bpm = None
        for att in ("AverageBpm","AVERAGEBPM","BPM","Tempo","Bpm"):
            if tr.get(att):
                try:
                    bpm = float(tr.get(att))
                    break
                except Exception:
                    pass

        secs: List[float] = []
        for pm in tr.iter("POSITION_MARK"):
            pos = pm.get("Start") or pm.get("Position")
            if pos is None:
                continue
            try:
                s = float(pos)
            except Exception:
                continue
            typ = pm.get("Type")
            if typ is None:
                secs.append(s)
            else:
                if typ == "0" and IMPORT_RB_MEMORY:
                    secs.append(s)
                elif typ == "1" and IMPORT_RB_HOT:
                    secs.append(s)

        if secs:
            dedup = sorted(set(round(x, 3) for x in secs))
            cur = out.get(key, {"secs": [], "bpm": None})
            cur["secs"].extend(dedup)
            if bpm and not cur["bpm"]:
                cur["bpm"] = bpm
            out[key] = cur

    for k, v in out.items():
        v["secs"] = sorted(set(v["secs"]))
    return out

def load_rekordbox_mik_maps(paths: List[str]) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    used = [p for p in (nfc(x) for x in paths) if os.path.exists(p)]
    print(f"[RBX] Using {len(used)} MiK Rekordbox XML file(s).")
    for p in used:
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            m = _gather_mik_rbx(root)
            for k, v in m.items():
                cur = merged.get(k, {"bpm": None, "secs": []})
                if v.get("bpm") and not cur.get("bpm"):
                    cur["bpm"] = v["bpm"]
                cur["secs"].extend(v["secs"])
                merged[k] = cur
            print(f"[RBX] Parsed {len(m)} entries from: {p}")
        except Exception as e:
            print(f"[RBX] Could not parse {p}: {e}")
    for k, v in merged.items():
        v["secs"] = sorted(set(v["secs"]))
    print(f"[RBX] Tracks with cues (merged): {sum(1 for v in merged.values() if v['secs'])}")
    return merged

def _candidates_for_folder(folder_abs: str) -> List[str]:
    cands = set()
    folder_name = os.path.basename(folder_abs)
    cands.add(_norm_title(folder_name))
    for f in os.listdir(folder_abs):
        lf = f.lower()
        if not lf.endswith((".flac",".wav",".aiff",".aif",".mp3",".m4a",".mp4")):
            continue
        base = os.path.splitext(f)[0]
        if "-" in base:
            suffix = base.split("-", 1)[1]
            cands.add(_norm_title(suffix))
        cands.add(_norm_title(base))
    return [x for x in cands if x]

def find_mik_cues_for_folder(rbx_map: Dict[str, Dict], folder_abs: str) -> Optional[List[float]]:
    for key in _candidates_for_folder(folder_abs):
        hit = rbx_map.get(key)
        if hit and hit.get("secs"):
            return hit["secs"]
    return None

# ========= cue → beat snapping =========
def _snap_seconds_to_grid(seconds: float, bpm: int) -> Tuple[float, float]:
    """
    Convert seconds to beats using BPM, then:
      - if within SNAP_NEAR_INT_TOL of a whole beat, snap to that downbeat
      - else quantize to 1/GRID_RES beat
    Return (beat_time, sec_time_from_snapped_beat)
    """
    if not bpm or bpm <= 0:
        return seconds, seconds
    beats = seconds * bpm / 60.0
    nearest_int = round(beats)
    if abs(beats - nearest_int) <= SNAP_NEAR_INT_TOL:
        beats_q = float(nearest_int)
    else:
        beats_q = round(beats * GRID_RES) / float(GRID_RES)
    sec_q = beats_q * 60.0 / float(bpm)
    return beats_q, sec_q

# ========= RENAME + ALS =========
def _existing_energy_in_name(name: str) -> Optional[int]:
    m = _FN_ENERGY_RE.match(os.path.basename(name))
    if not m:
        return None
    try:
        e = int(m.group(2))
        if 1 <= e <= 12:
            return e
    except Exception:
        pass
    return None

def rename_stems_in_place(stems_folder: str, bpm: int, cam: str, energy: Optional[int]=None) -> bool:
    added_energy = False
    base = os.path.basename(stems_folder)
    for f in os.listdir(stems_folder):
        if not f.lower().endswith((".flac",".wav",".aiff",".aif",".mp3")):
            continue
        low = f.lower()
        if not (low.startswith("drums") or low.startswith("inst") or low.startswith("vocals")):
            continue
        role = "drums" if low.startswith("drums") else ("inst" if low.startswith("inst") else "vocals")
        _, ext = os.path.splitext(f)

        existing_e = _existing_energy_in_name(f)
        use_e = existing_e if existing_e is not None else (energy if energy is not None else None)
        if existing_e is None and use_e is not None:
            added_energy = True

        prefix = f"{role}_{bpm}_{cam}" + (f"_{use_e}" if use_e is not None else "")
        new_name = f"{prefix}-{base}{ext}"

        src = os.path.join(stems_folder, f)
        dst = os.path.join(stems_folder, new_name)
        if src != dst:
            os.rename(src, dst)
            old_asd, new_asd = src + ".asd", dst + ".asd"
            if os.path.exists(old_asd):
                os.rename(old_asd, new_asd)
            print(f"[RENAME] {f}  →  {new_name}")
    return added_energy

def get_duration_in_beats(track_path: str, bpm: int) -> Optional[str]:
    try:
        sec = trackTime.get_track_duration(track_path)
        return f"{(sec * bpm)/60.0:.6f}"
    except Exception as e:
        print(f"[ERROR] Duration for {track_path}: {e}")
        return None

def select_blank_als(bpm_value: Optional[int]) -> Optional[str]:
    if bpm_value:
        p = os.path.join(ALS_FILES_FOLDER, f"{bpm_value}.als")
        if os.path.exists(p):
            return p
    return None

def collect_track_names_for_folder(folder_abs: str) -> Dict[str, Optional[str]]:
    roles: Dict[str, Optional[str]] = {"drums": None, "inst": None, "vocals": None}
    for f in os.listdir(folder_abs):
        if not f.lower().endswith(".flac"):
            continue
        low = f.lower()
        role = "drums" if low.startswith("drums") else ("inst" if low.startswith("inst") else ("vocals" if low.startswith("vocals") else None))
        if not role:
            continue
        roles[role] = os.path.relpath(os.path.join(folder_abs, f), FLAC_FOLDER)
    return roles

def _replace_warpmarkers_in_audioclip(aclip: ET.Element, cues_sec: List[float], bpm: Optional[int]):
    """
    - compute BeatTime from seconds using BPM
    - snap to grid, then recompute SecTime from snapped BeatTime
    - write BOTH BeatTime and SecTime
    """
    for wm_parent in list(aclip.findall("WarpMarkers")):
        aclip.remove(wm_parent)
    wm_parent = ET.Element("WarpMarkers")

    for i, sec in enumerate(sorted(cues_sec)):
        if bpm and bpm > 0:
            beat_q, sec_q = _snap_seconds_to_grid(sec, bpm)
            attrs = {"Id": str(i), "SecTime": f"{sec_q:.6f}", "BeatTime": f"{beat_q:.6f}"}
        else:
            attrs = {"Id": str(i), "SecTime": f"{sec:.6f}"}
        wm_parent.append(ET.Element("WarpMarker", attrs))

    aclip.insert(0, wm_parent)

def _stamp_cues(root: ET.Element, cues_sec: List[float], bpm: Optional[int]) -> int:
    if not cues_sec:
        return 0
    count = 0
    for aclip in root.iter("AudioClip"):
        _replace_warpmarkers_in_audioclip(aclip, cues_sec, bpm)
        count += len(cues_sec)
    for node in root.iter("IsWarped"):
        node.set("Value", "true")
    return count

# ======== Detector code kept (unused) ========
def _smooth_same(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, k, mode="same")

def _estimate_grid(y: np.ndarray, sr: int, hop: int = 512):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    downbeats = np.array([])
    if _HAS_MADMOM:
        try:
            proc = RNNDownBeatProcessor()
            act = proc(y)
            dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=proc.fps)
            seq = dbn(act)
            downbeats = seq[seq[:,1] == 1][:,0]
        except Exception:
            warnings.warn("madmom failed; using heuristic downbeats.")
    if downbeats.size == 0 and len(beat_times) >= 8:
        beats_per_bar = 4
        env = onset_env
        env_t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        best_phase, best_score = 0, -1e9
        for phase in range(beats_per_bar):
            idx = np.arange(phase, len(beat_times), beats_per_bar)
            vals = np.interp(beat_times[idx], env_t, env)
            sc = float(np.mean(vals)) if len(vals) else -1e9
            if sc > best_score:
                best_score, best_phase = sc, phase
        idx = np.arange(best_phase, len(beat_times), beats_per_bar)
        downbeats = beat_times[idx]
    return beat_times, float(tempo), downbeats

def detect_cues_seconds(audio_path: str, max_hotcues: int = 8) -> Tuple[List[float], Optional[int]]:
    # UNUSED in this MiK-only build
    return [], None

# ======= GLOBAL: MiK cue map
RBX_MAP: Dict[str, Dict] = {}

def modify_als_file(input_path: Optional[str],
                    target_folder: str,
                    track_names: Dict[str, Optional[str]],
                    bpm_value: Optional[int],
                    force: bool=False) -> None:
    """
    Stamp ONLY MiK (RBX) cues into ALS. No detector cues merged.
    """
    output_als = os.path.join(target_folder, "CH1.als")

    if os.path.exists(output_als) and SKIP_EXISTING and not force:
        return

    if input_path and os.path.abspath(input_path) != os.path.abspath(output_als):
        shutil.copy(input_path, output_als)
    elif not os.path.exists(output_als):
        print(f"[WARN] No template for BPM {bpm_value} and no existing CH1.als at {target_folder}; skipping.")
        return

    print(f"\n🎯 Creating/Updating ALS for {target_folder} (BPM {bpm_value})")

    with gzip.open(output_als, "rb") as f:
        als_data = f.read()
    tree = ET.parse(BytesIO(als_data))
    root = tree.getroot()

    # === cues: ONLY Rekordbox (MiK) ===
    rbx_secs = find_mik_cues_for_folder(RBX_MAP, target_folder)
    if rbx_secs:
        _stamp_cues(root, rbx_secs, bpm_value)
        print(f"   ➕ Stamped {len(rbx_secs)} cue(s) from MiK XML.")
    else:
        print("   ⚠️  No MiK cues found for this folder.")

    # === LoopEnd / OutMarker using drums track duration if present
    if track_names.get("drums") and bpm_value:
        flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
        loop_end_beats = get_duration_in_beats(flac_path, bpm_value)
        if loop_end_beats:
            for elem in root.iter():
                if elem.tag in ("LoopEnd","OutMarker"):
                    elem.set("Value", loop_end_beats)

    # Write back structural XML first
    buf = BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    with gzip.open(output_als, "wb") as f_out:
        f_out.write(buf.getvalue())

    # Then textual substitutions for names/paths
    with gzip.open(output_als, "rb") as f:
        als_str = f.read().decode("utf-8")

    for role, rel_path in track_names.items():
        if not rel_path:
            continue
        for role_variant in {role, role.capitalize()}:
            old = f"{role_variant}-Tape B - i won't be ur drug.flac"
            new = safe_xml_value(rel_path)
            als_str = als_str.replace(old, new)
            als_str = als_str.replace(old.replace(" ", "%20"), new.replace(" ", "%20"))

            old_name = old.replace(".flac", "")
            new_name = safe_xml_value(os.path.basename(rel_path).replace(".flac",""))
            for tag in ["MemorizedFirstClipName","UserName","Name","EffectiveName"]:
                pattern = rf'(<{tag}\s+Value="){re.escape(old_name)}(")'
                als_str = re.sub(pattern, rf"\1{new_name}\2", als_str)

    with gzip.open(output_als, "wb") as f_out:
        f_out.write(als_str.encode("utf-8"))
    print(f"✅ ALS saved: {output_als}")

# ========= MOVES =========
def move_folder_to_target(stems_folder_abs: str,
                          bpm: int,
                          cam: str,
                          energy: Optional[int],
                          src_audio_path: Optional[str]) -> Tuple[str, bool, bool, bool]:
    """
    - Moves folder into destination/<bpm>/<cam>/
    - Renames stems (adds energy if missing)
    - Builds/updates CH1.als with ONLY MiK cues
    """
    track_name = os.path.basename(stems_folder_abs)
    bpm_folder = os.path.join(destination_folder, str(bpm))
    cam_folder = os.path.join(bpm_folder, cam)
    os.makedirs(cam_folder, exist_ok=True)

    dest_path = os.path.join(cam_folder, track_name)
    moved = False
    if stems_folder_abs != dest_path:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(stems_folder_abs, dest_path)
        print(f"[MOVE] {track_name}  →  {cam_folder}")
        moved = True

    # No detection — MiK cues only
    energy_added = rename_stems_in_place(dest_path, bpm, cam, energy)

    output_als = os.path.join(dest_path, "CH1.als")
    force_regen = (not SKIP_EXISTING)
    need_als = force_regen or (not os.path.exists(output_als)) or moved or energy_added
    if not need_als:
        return dest_path, moved, energy_added, False

    blank_als   = select_blank_als(bpm)
    track_names = collect_track_names_for_folder(dest_path)

    if blank_als is None and os.path.exists(output_als):
        modify_als_file(output_als, dest_path, track_names, bpm, force=True)
    else:
        modify_als_file(blank_als, dest_path, track_names, bpm, force=force_regen)

    return dest_path, moved, energy_added, True

# ========= FLOWS =========
def process_new_stems_from_toBeOrganized(idx_exact: Dict[str,str]) -> None:
    if not os.path.isdir(htdemucs_source_folder):
        return
    for folder in os.listdir(htdemucs_source_folder):
        stems_abs = os.path.join(htdemucs_source_folder, folder)
        if not os.path.isdir(stems_abs):
            continue

        src_audio = find_source_for_track(idx_exact, folder)
        if not src_audio:
            continue  # silent if no exact match

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key":
            continue

        move_folder_to_target(stems_abs, bpm, cam, energy, src_audio)

def _snapshot_tracks(dest_folder: str):
    items = []
    if not os.path.isdir(dest_folder):
        return items
    for bpm_dir in os.listdir(dest_folder):
        if not bpm_dir.isdigit():
            continue
        bpm_abs = os.path.join(dest_folder, bpm_dir)
        if not os.path.isdir(bpm_abs):
            continue
        for cam_dir in os.listdir(bpm_abs):
            cam_abs = os.path.join(bpm_abs, cam_dir)
            if not os.path.isdir(cam_abs):
                continue
            for track_folder in os.listdir(cam_abs):
                track_abs = os.path.join(cam_abs, track_folder)
                if os.path.isdir(track_abs):
                    items.append((int(bpm_dir), cam_dir, track_folder, track_abs))
    return items

def reconcile_existing_stems(idx_exact: Dict[str,str]) -> None:
    moved = regenerated = 0
    for cur_bpm, cur_cam, track_folder, track_abs in _snapshot_tracks(destination_folder):
        src_audio = find_source_for_track(idx_exact, track_folder)
        if not src_audio:
            continue

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key":
            continue

        _, did_move, added_energy, wrote_als = move_folder_to_target(track_abs, bpm, cam, energy, src_audio)
        if did_move: moved += 1
        if wrote_als: regenerated += 1

    print(f"[INFO] Reconcile complete. Moved: {moved}, ALS updated: {regenerated}")

# ========= MAIN =========
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if input("Ready? Type 'yes' to begin: ").strip().lower() == "yes":
        # Load MiK cues once
        RBX_MAP = load_rekordbox_mik_maps(REKORDBOX_XMLS)

        index_exact = index_source_library(mp3_source_folder)     # exact base-name index (NFC)
        process_new_stems_from_toBeOrganized(index_exact)         # place new folders + build ALS with MiK cues only
        reconcile_existing_stems(index_exact)                     # fix existing folders (force-regens if SKIP_EXISTING=False)
        print("🎵 Done.")
    else:
        print("Operation cancelled.")
