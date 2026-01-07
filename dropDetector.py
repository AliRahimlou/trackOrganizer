#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop-anchored pipeline:
- Index library, read BPM/Key/Energy from tags
- Load Rekordbox (MiK) cues from XML (optional)
- Detect earliest sustained high-energy drop on DRUMS; align bar 1.1.1 there (beat-phase offset)
- Merge cues (RBX + detector), quantize BeatTime (SecTime stays true)
- Stamp WarpMarkers for: zero (Sec=0), all cues, end (overall track length)
- Write CH1.als + replace clip names/paths; LoopEnd/OutMarker = overall length (in beats)

Requires:
  pip install numpy>=2.0 scipy>=1.13 soundfile>=0.12.1 audioread>=3.0.1 librosa>=0.10.2.post1 mutagen
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

# --- analysis ---
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

# Grid / quantization
BEATS_PER_BAR       = 4
SNAP_NEAR_INT_TOL   = 0.125   # snap if within 1/8 beat to an integer beat
GRID_RES            = 24      # else quantize to 1/24 beat

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

# ========= Quantization helpers =========
def _quantize_beats(beats: float) -> float:
    nearest_int = round(beats)
    if abs(beats - nearest_int) <= SNAP_NEAR_INT_TOL:
        return float(nearest_int)
    return round(beats * GRID_RES) / float(GRID_RES)

def _sec_to_beats(seconds: float, bpm: int) -> float:
    return seconds * bpm / 60.0

# ========= Rename + stems =========
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

def _find_drums_stem_path(folder_abs: str) -> Optional[str]:
    # try common drums filenames
    for f in os.listdir(folder_abs):
        low = f.lower()
        if low.startswith("drums") and low.endswith((".flac",".wav",".aiff",".aif",".mp3",".m4a",".mp4")):
            return os.path.join(folder_abs, f)
    # fallback: pick the loudest-looking stem with "drum" in name
    for f in os.listdir(folder_abs):
        if "drum" in f.lower() and f.lower().endswith(AUDIO_EXTS):
            return os.path.join(folder_abs, f)
    return None

# ========= Duration helpers =========
def _safe_duration(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    try:
        return float(trackTime.get_track_duration(path))
    except Exception:
        return None

def get_overall_track_duration(target_folder: str,
                               track_names: Dict[str, Optional[str]],
                               src_audio_path: Optional[str]) -> Optional[float]:
    # Prefer full mix if provided
    d_mix = _safe_duration(src_audio_path)
    if d_mix and d_mix > 1.0:
        return d_mix

    cands: List[float] = []
    for role in ("drums", "inst", "vocals"):
        rel = track_names.get(role)
        abspath = os.path.join(FLAC_FOLDER, rel) if rel else None
        d = _safe_duration(abspath)
        if d: cands.append(d)

    if not cands:
        for f in os.listdir(target_folder):
            if f.lower().endswith(AUDIO_EXTS):
                d = _safe_duration(os.path.join(target_folder, f))
                if d: cands.append(d)
    return max(cands) if cands else None

# ========= Detector (grid + curves + picks) =========
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
        # simple 4/4 phase heuristic
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

def _curves_on_beats(y: np.ndarray, sr: int, beat_times: np.ndarray, hop: int = 512) -> dict:
    S = librosa.stft(y, n_fft=2048, hop_length=hop, window="hann")
    H, P = librosa.decompose.hpss(S)
    mag = np.abs(S); magH = np.abs(H)

    n_frames = mag.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    def _flux(M):
        M = M / (M.sum(axis=0, keepdims=True) + 1e-9)
        d = np.maximum(M[:,1:] - M[:,:-1], 0.0)
        return d.sum(axis=0)

    def _fix_len(curve, target=n_frames):
        L = len(curve)
        if L == target: return curve
        if L < target:  return np.pad(curve, (0, target - L), mode="edge")
        return curve[:target]

    flux_all = _fix_len(_flux(mag))
    flux_perc = _fix_len(_flux(magH))
    rms = librosa.feature.rms(S=mag, frame_length=2048, hop_length=hop, center=True)[0]
    rms = _fix_len(rms)

    S2 = librosa.stft(y, n_fft=4096, hop_length=hop, window="hann")
    mag2 = np.abs(S2)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    bass_band = (freqs >= 20) & (freqs <= 120)
    bass_env = _fix_len(mag2[bass_band].mean(axis=0))

    def to_beats(curve):
        L = min(len(curve), len(frame_times))
        if L == 0:
            return np.zeros_like(beat_times)
        return np.interp(beat_times, frame_times[:L], curve[:L])

    curves = dict(
        flux_all = to_beats(flux_all),
        flux_perc= to_beats(flux_perc),
        rms      = to_beats(rms),
        bass     = to_beats(bass_env),
    )
    # normalize 0..1
    for k in list(curves.keys()):
        c = curves[k]
        rng = np.ptp(c) + 1e-9
        curves[k] = (c - np.min(c)) / rng

    bass_sm   = _smooth_same(curves["bass"], 4)
    bass_ramp = np.gradient(bass_sm)
    bass_ramp = (bass_ramp - np.min(bass_ramp)) / (np.ptp(bass_ramp) + 1e-9)

    perc_sm   = _smooth_same(curves["flux_perc"], 2)
    perc_acc  = np.clip(np.gradient(perc_sm), 0, None)
    perc_acc  = (perc_acc - np.min(perc_acc)) / (np.ptp(perc_acc) + 1e-9)

    energy = 0.45*curves["bass"] + 0.35*curves["flux_perc"] + 0.20*curves["rms"]
    energy = (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)
    energy = _smooth_same(energy, 3)

    drop_score = 0.55*_smooth_same(bass_ramp, 3) + 0.35*perc_acc + 0.15*curves["flux_all"]
    drop_score = (drop_score - float(np.mean(drop_score))) / (float(np.std(drop_score)) + 1e-8)

    curves["bass_ramp"]   = bass_ramp
    curves["perc_accent"] = perc_acc
    curves["energy"]      = energy
    curves["drop_score"]  = drop_score
    return curves

def _pick_drops(beat_times: np.ndarray, downbeats: np.ndarray, curves: dict,
                min_sep_bars=16, beats_per_bar=BEATS_PER_BAR) -> List[int]:
    score = curves["drop_score"].copy()
    n = len(score)
    if n > 32:
        score[:16] -= 2.0
        score[-16:] -= 1.0
    if len(downbeats):
        db_idx = np.searchsorted(beat_times, downbeats)
        mask = np.zeros_like(score)
        for di in db_idx:
            for k in (-1,0,1):
                j = di + k
                if 0 <= j < n: mask[j] += 1.0
        score += 0.6*mask
    used = np.zeros(n, dtype=bool)
    picks = []
    order = np.argsort(-score)
    min_sep = max(1, min_sep_bars*beats_per_bar)
    for idx in order:
        if used[idx]: continue
        if score[idx] < 0.5: break
        picks.append(idx)
        lo = max(0, idx - min_sep); hi = min(n, idx + min_sep + 1)
        used[lo:hi] = True
        if len(picks) == 2: break
    picks.sort()
    return picks

def detect_cues_seconds(audio_path: str, max_hotcues: int = 8) -> Tuple[List[float], Optional[int]]:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"[DETECT] Failed to load {audio_path}: {e}")
        return [], None
    y = librosa.util.normalize(y)

    beat_times, tempo, downbeats = _estimate_grid(y, sr)
    if len(beat_times) < 24:
        return [], None
    curves = _curves_on_beats(y, sr, beat_times)
    drops = _pick_drops(beat_times, downbeats, curves)
    if not drops:
        energy = curves["energy"].copy()
        if len(downbeats):
            db_idx = np.searchsorted(beat_times, downbeats)
            mask = np.zeros_like(energy)
            for di in db_idx:
                for k in (-1,0,1):
                    j = di + k
                    if 0 <= j < len(mask): mask[j] += 1.0
            energy += 0.4*mask
        drops = [int(np.argmax(energy))]

    drop1 = drops[0]
    cues_idx: List[int] = []

    intro_idx = int(np.argmin(np.abs(beat_times - (downbeats[0] if len(downbeats) else 0.0))))
    cues_idx.append(intro_idx)

    pre_drop = max(0, drop1 - 8*BEATS_PER_BAR); cues_idx.append(pre_drop)
    cues_idx.append(drop1)
    post_drop = min(len(beat_times)-1, drop1 + 8*BEATS_PER_BAR); cues_idx.append(post_drop)

    # outro rough
    n = len(beat_times)
    start = n * 2 // 3
    if start < n - 8:
        flux = curves["flux_all"]; bass = curves["bass"]
        best = start; best_score = -1e9; window = 16
        for j in range(start, max(start + 1, n - window)):
            seg = (1.0 - float(np.mean(_smooth_same(flux[j:j+window], 4)))) \
                + (1.0 - float(np.mean(_smooth_same(bass[j:j+window], 4))))
            if seg > best_score:
                best_score = seg; best = j
        cues_idx.append(best)

    # Sort/unique/limit
    cues_idx_sorted = []
    seen = set()
    for i in cues_idx:
        if i not in seen:
            seen.add(i); cues_idx_sorted.append(i)
        if len(cues_idx_sorted) >= max_hotcues:
            break

    cues_sec = [float(beat_times[i]) for i in cues_idx_sorted]

    e90 = float(np.percentile(curves["energy"], 90)) if len(curves["energy"]) else 0.5
    energy12 = max(1, min(12, int(round(1 + 11*e90))))
    return cues_sec, energy12

# ========= "First-high-energy drop" finder on DRUMS =========
def _first_high_energy_drop_sec_on_drums(drums_path: str) -> Optional[float]:
    try:
        y, sr = librosa.load(drums_path, sr=None, mono=True)
    except Exception as e:
        print(f"[DROP] Failed to load drums: {e}")
        return None
    y = librosa.util.normalize(y)

    beat_times, tempo, downbeats = _estimate_grid(y, sr)
    if len(beat_times) < 24:
        return None

    curves = _curves_on_beats(y, sr, beat_times)
    energy = curves["energy"]
    ramp   = curves["bass_ramp"]

    n = len(beat_times)
    if n < 24:
        return None

    start_idx = max(0, int(0.5 * BEATS_PER_BAR))  # skip ~half bar to avoid clicks

    e85 = float(np.percentile(energy, 85)) if len(energy) else 0.7
    e90 = float(np.percentile(energy, 90)) if len(energy) else 0.8
    win_sustain = 8  # ~2 bars
    win_slope   = 4

    best_idx = None
    for i in range(start_idx, n - win_sustain):
        i0 = max(0, i - win_slope)
        slope = float(np.mean(ramp[i0:i+1])) if i > i0 else ramp[i]
        e0 = energy[i]
        sustain = float(np.mean(energy[i:i+win_sustain]))
        strong_now = (e0 >= e90) or (e0 >= e85 and slope > 0.5)
        strong_fw  = sustain >= e85
        if strong_now and strong_fw:
            best_idx = i
            break

    if best_idx is None:
        picks = _pick_drops(beat_times, downbeats, curves, min_sep_bars=16, beats_per_bar=BEATS_PER_BAR)
        if picks:
            best_idx = int(min(picks))

    if best_idx is None:
        top = np.where(energy >= e90)[0]
        if len(top): best_idx = int(top[0])

    if best_idx is None:
        return None

    bt = float(beat_times[best_idx])

    if len(downbeats):
        db_idx = np.searchsorted(downbeats, bt)
        cand_idxs = [db_idx-1, db_idx, db_idx+1]
        best_db = bt
        best_err = 1e9
        beat_sec = 60.0 / max(1.0, float(librosa.beat.tempo(y=y, sr=sr).item() if hasattr(librosa.beat.tempo(y=y, sr=sr), 'item') else librosa.beat.tempo(y=y, sr=sr)))
        for ci in cand_idxs:
            if 0 <= ci < len(downbeats):
                err = abs(downbeats[ci] - bt)
                if err < best_err and err <= beat_sec:
                    best_err = err
                    best_db = downbeats[ci]
        return float(best_db)

    return bt

# ========= Beat-phase / stamping =========
def _compute_phi_to_put_drop_at_bar_1(drop_sec: float, bpm: int) -> float:
    if not bpm or bpm <= 0:
        return 0.0
    beats = _sec_to_beats(drop_sec, bpm)
    k = round(beats / BEATS_PER_BAR) * BEATS_PER_BAR  # nearest bar boundary
    phi = -k
    return float(phi)

def _stamp_markers_with_phase(aclip: ET.Element,
                              bpm: Optional[int],
                              phi_beats: float,
                              zero_sec: Optional[float],
                              end_sec: Optional[float],
                              cue_secs: List[float]) -> int:
    # remove existing markers
    for wm_parent in list(aclip.findall("WarpMarkers")):
        aclip.remove(wm_parent)
    wm_parent = ET.Element("WarpMarkers")

    def _append_marker(idx: int, sec: float, beat_from_sec: Optional[float]):
        if bpm and bpm > 0 and beat_from_sec is not None:
            beats = beat_from_sec + phi_beats
            beats_q = _quantize_beats(beats)
            wm_parent.append(ET.Element("WarpMarker", {
                "Id": str(idx),
                "SecTime": f"{sec:.6f}",
                "BeatTime": f"{beats_q:.6f}",
            }))
        else:
            wm_parent.append(ET.Element("WarpMarker", {
                "Id": str(idx),
                "SecTime": f"{sec:.6f}",
            }))

    items: List[Tuple[float,float]] = []

    if zero_sec is not None:
        items.append((zero_sec, _sec_to_beats(zero_sec, bpm) if bpm else None))

    for s in sorted(set(round(x, 3) for x in cue_secs)):
        items.append((s, _sec_to_beats(s, bpm) if bpm else None))

    if end_sec is not None:
        items.append((end_sec, _sec_to_beats(end_sec, bpm) if bpm else None))

    # sort by SecTime
    items.sort(key=lambda t: t[0])
    for i, (sec, beats0) in enumerate(items):
        _append_marker(i, sec, beats0)

    aclip.insert(0, wm_parent)
    return len(items)

def _stamp_cues(root: ET.Element,
                cues_sec: List[float],
                bpm: Optional[int],
                phi_beats: float = 0.0,
                track_zero_sec: Optional[float] = 0.0,
                track_end_sec: Optional[float] = None) -> int:
    if cues_sec is None:
        cues_sec = []
    total = 0
    for aclip in root.iter("AudioClip"):
        total += _stamp_markers_with_phase(
            aclip,
            bpm=bpm,
            phi_beats=phi_beats,
            zero_sec=track_zero_sec,
            end_sec=track_end_sec,
            cue_secs=cues_sec,
        )
    for node in root.iter("IsWarped"):
        node.set("Value", "true")
    return total

# ======= GLOBAL: MiK cue map
RBX_MAP: Dict[str, Dict] = {}

def _merge_cues_seconds(cues_a: Optional[List[float]], cues_b: Optional[List[float]]) -> List[float]:
    merged = []
    for src in (cues_a or [], cues_b or []):
        merged.extend(src)
    merged = sorted(set(round(x, 3) for x in merged))
    return merged

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

# ======= PATCHED: ONLY update LoopEnd/OutMarker under AudioClip; end = overall max duration =======
def modify_als_file(input_path: Optional[str],
                    target_folder: str,
                    track_names: Dict[str, Optional[str]],
                    bpm_value: Optional[int],
                    cues_sec_merge: Optional[List[float]] = None,
                    src_audio_path: Optional[str] = None,
                    force: bool=False) -> None:
    output_als = os.path.join(target_folder, "CH1.als")

    if os.path.exists(output_als) and SKIP_EXISTING and not force:
        return

    if input_path and os.path.abspath(input_path) != os.path.abspath(output_als):
        shutil.copy(input_path, output_als)
    elif not os.path.exists(output_als):
        print(f"[WARN] No template for BPM {bpm_value} and no existing CH1.als at {target_folder}; skipping.")
        return

    print(f"\n🎯 Creating/Updating ALS for {target_folder} (BPM {bpm_value})")

    # Load ALS (gzip -> XML)
    with gzip.open(output_als, "rb") as f:
        als_data = f.read()
    tree = ET.parse(BytesIO(als_data))
    root = tree.getroot()

    # Overall duration (seconds): prefer full mix, else longest stem, else any audio in folder
    overall_dur_sec = get_overall_track_duration(
        target_folder=target_folder,
        track_names=track_names,
        src_audio_path=src_audio_path
    )

    # Drums path for drop anchor
    drums_rel = track_names.get("drums")
    drums_abs = os.path.join(FLAC_FOLDER, drums_rel) if drums_rel else _find_drums_stem_path(target_folder)

    # Compute phase so first drop = bar 1.1.1
    phi_beats = 0.0
    if bpm_value and drums_abs and os.path.exists(drums_abs):
        drop_sec = _first_high_energy_drop_sec_on_drums(drums_abs)
        if drop_sec is not None:
            phi_beats = _compute_phi_to_put_drop_at_bar_1(drop_sec, bpm_value)

    # Merge cues (RBX + detector)
    rbx_secs = find_mik_cues_for_folder(RBX_MAP, target_folder)
    cues_all = _merge_cues_seconds(rbx_secs, cues_sec_merge)

    # Stamp WarpMarkers: zero, cues, end (in seconds)
    _ = _stamp_cues(
        root,
        cues_sec=cues_all,
        bpm=bpm_value,
        phi_beats=phi_beats,
        track_zero_sec=0.0,
        track_end_sec=overall_dur_sec
    )

    # LoopEnd / OutMarker ONLY on each AudioClip, using overall length in BEATS
    if overall_dur_sec and bpm_value:
        loop_end_beats = f"{(overall_dur_sec * bpm_value)/60.0:.6f}"
        for clip in root.findall(".//AudioClip"):
            loop_node = clip.find("./Loop/LoopEnd")
            if loop_node is not None:
                loop_node.set("Value", loop_end_beats)
            out_node = clip.find("./CurrentStartAndEnd/OutMarker")
            if out_node is not None:
                out_node.set("Value", loop_end_beats)

    # Write back structural XML first
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
    - Renames stems (adds energy if missing; uses detector-fallback if tags missing)
    - Builds/updates CH1.als with merged cues; grid phase aligned so the FIRST high-energy drums hit = 1.1.1
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

    # Detect cues from full mix if we have it
    det_secs: List[float] = []
    energy_fallback: Optional[int] = None
    if src_audio_path and os.path.exists(src_audio_path):
        print(f"[DETECT] Analyzing: {src_audio_path}")
        try:
            det_secs, energy_fallback = detect_cues_seconds(src_audio_path, max_hotcues=8)
            if det_secs:
                print(f"[DETECT] Found {len(det_secs)} cue(s).")
        except Exception as e:
            print(f"[DETECT] Error: {e}")

    if energy is None and energy_fallback is not None:
        energy = int(energy_fallback)

    energy_added = rename_stems_in_place(dest_path, bpm, cam, energy)

    output_als = os.path.join(dest_path, "CH1.als")
    force_regen = (not SKIP_EXISTING)
    need_als = force_regen or (not os.path.exists(output_als)) or moved or energy_added
    if not need_als:
        return dest_path, moved, energy_added, False

    blank_als   = select_blank_als(bpm)
    track_names = collect_track_names_for_folder(dest_path)

    if blank_als is None and os.path.exists(output_als):
        modify_als_file(output_als, dest_path, track_names, bpm,
                        cues_sec_merge=det_secs, src_audio_path=src_audio_path, force=True)
    else:
        modify_als_file(blank_als, dest_path, track_names, bpm,
                        cues_sec_merge=det_secs, src_audio_path=src_audio_path, force=force_regen)

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
        RBX_MAP = load_rekordbox_mik_maps(REKORDBOX_XMLS)
        index_exact = index_source_library(mp3_source_folder)
        process_new_stems_from_toBeOrganized(index_exact)
        reconcile_existing_stems(index_exact)
        print("🎵 Done.")
    else:
        print("Operation cancelled.")
