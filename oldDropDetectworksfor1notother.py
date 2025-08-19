#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, base64, html, shutil, gzip, unicodedata, warnings
from io import BytesIO
from typing import Optional, Dict, Tuple, List
import xml.etree.ElementTree as ET

from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE

import trackTime  # your module for duration

# ===== Optional DSP deps =====
_HAS_LIBROSA = True
try:
    import numpy as np
    import librosa
except Exception:
    _HAS_LIBROSA = False
    np = None

# =================== USER SETTINGS ===================
mp3_source_folder        = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder   = "/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized"
destination_folder       = "/Users/alirahimlou/Desktop/STEMS2"
ALS_FILES_FOLDER         = "alsFiles"
FLAC_FOLDER              = destination_folder
SKIP_EXISTING            = False

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".mp4")

# =================== Detector knobs (auto-tuned inside) ===================
INTRO_SKIP_BARS            = 8          # ignore intro
SEARCH_MAX_BARS            = 96         # search span after intro
BASS_MAX_HZ                = 150        # low band ceiling
SMOOTH_BEATS               = 0.50       # moving-average length (beats)
BASELINE_BARS              = 8          # bars before candidate for baseline
VALLEY_LOOKBACK_BARS       = 8          # must have valley in these bars before candidate
SUSTAIN_BARS               = 8          # require post window length (bars)
SNAP_TO_BEAT               = True
OFFSET_BEATS               = 0.00
DEBUG_DROPS                = True

# =================== Camelot ===================
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

# =================== Utils ===================
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

# =================== Tag readers ===================
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

# =================== Library index ===================
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

def find_source_for_track(idx_exact: Dict[str,str], track_folder_name: str) -> Optional[str]:
    return idx_exact.get(nfc(track_folder_name))

# =================== Small helpers ===================
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-8)

def _moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    ker = np.ones(k)/k
    return np.convolve(x, ker, mode="same")

# =================== DROP DETECTOR ===================
def detect_drop_beats(drums_path: str, bpm: int) -> Optional[Tuple[float, float]]:
    """
    Step-and-sustain detector:
      - Build bar-level low-band dB, bass-share, and kick onsets.
      - For each bar s, compare PRE (s-8..s-1) vs POST (s..s+7).
      - Earliest s with: valley just before, big low-band step, sustained POST,
        bass-share increases, and at least one kick onset near s.
      - Then snap inside that bar to the first strong kick.
    """
    if not _HAS_LIBROSA:
        warnings.warn("librosa not available; skipping drop detection.")
        return None

    try:
        # ---- load ----
        sr, hop, n_fft = 22050, 512, 2048
        y, sr = librosa.load(drums_path, sr=sr, mono=True)
        if len(y) < sr * 5:
            return None

        # ---- frame features ----
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bass_mask = (freqs <= BASS_MAX_HZ)
        low_pow = S[bass_mask, :].sum(axis=0)
        tot_pow = S.sum(axis=0) + 1e-12
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean)

        spb = 60.0 / max(1, bpm)
        t_frames = np.arange(S.shape[1]) * (hop / sr)
        n_beats = int(np.floor(t_frames[-1] / spb))
        if n_beats < 24:
            return None
        beat_times = np.arange(n_beats) * spb

        low_lin_beats  = np.interp(beat_times, t_frames, low_pow)
        full_lin_beats = np.interp(beat_times, t_frames, tot_pow)
        on_beats       = np.interp(beat_times, t_frames, onset_env)

        low_db   = 10*np.log10(np.maximum(low_lin_beats, 1e-12))
        share    = np.clip(low_lin_beats / np.maximum(full_lin_beats, 1e-12), 0.0, 1.0)

        if SMOOTH_BEATS > 0:
            k = max(1, int(round(SMOOTH_BEATS)))
            low_db = _moving_mean(low_db, k)

        n_bars = n_beats // 4
        if n_bars < 24:
            return None

        def R4(x): return x[:n_bars*4].reshape(n_bars, 4)

        bar_low   = R4(low_db).mean(axis=1).astype(float)
        bar_share = R4(share).mean(axis=1).astype(float)
        bar_on    = (R4(on_beats) >= (np.median(on_beats) + 0.5*np.std(on_beats))).sum(axis=1).astype(float)

        start_bar = min(n_bars-2, int(INTRO_SKIP_BARS))
        end_bar   = min(n_bars-1, start_bar + int(SEARCH_MAX_BARS))
        seg = slice(start_bar, end_bar)

        # dynamic scales from track
        p10 = float(np.percentile(bar_low[seg], 10))
        p90 = float(np.percentile(bar_low[seg], 90))
        spread = max(6.0, p90 - p10)          # stabilize
        quiet_thr = float(np.percentile(bar_low[seg], 35))  # what counts as "valley"
        step_thr  = max(5.0, 0.15*spread)     # how big the jump must be
        sustain_thr_rel = max(4.0, 0.12*spread)  # POST mean above PRE median
        share_thr = 0.04                       # bass share must increase at least this

        if DEBUG_DROPS:
            print(f"   [auto] window bars {start_bar}-{end_bar-1} | quiet≈{quiet_thr:.2f} | spread≈{spread:.2f} | step≥{step_thr:.2f}")

        chosen_bar = None
        best_score = -1.0
        best_bar   = None

        # scan earliest → latest
        for s in range(start_bar + BASELINE_BARS, end_bar - SUSTAIN_BARS):
            pre_slice  = slice(s - BASELINE_BARS, s)
            post_slice = slice(s, s + SUSTAIN_BARS)

            pre_med  = float(np.median(bar_low[pre_slice]))
            pre_min  = float(np.min(bar_low[max(start_bar, s-VALLEY_LOOKBACK_BARS):s]))
            post_mean = float(np.mean(bar_low[post_slice]))

            step = post_mean - pre_med
            valley_ok = (pre_min <= min(quiet_thr, pre_med - max(2.0, 0.10*spread)))

            share_pre  = float(np.mean(bar_share[pre_slice]))
            share_post = float(np.mean(bar_share[post_slice]))
            share_jump = share_post - share_pre

            kicks = float(np.sum(bar_on[s:min(s+2, len(bar_on))]))  # first 1–2 bars
            onset_ok = kicks >= 1

            sustain_ok = (post_mean >= pre_med + sustain_thr_rel)

            if valley_ok and step >= step_thr and sustain_ok and share_jump >= share_thr and onset_ok:
                chosen_bar = s
                break

            # keep best candidate as fallback (score-based)
            step_norm  = max(0.0, step) / (spread + 1e-6)
            share_norm = max(0.0, share_jump) / 0.15
            onset_norm = min(1.0, kicks / 2.0)
            score = 0.6*step_norm + 0.3*share_norm + 0.1*onset_norm
            if valley_ok and score > best_score:
                best_score, best_bar = score, s

        if chosen_bar is None and best_bar is not None:
            chosen_bar = best_bar
            if DEBUG_DROPS:
                print(f"   [fallback] using best score at bar {chosen_bar} (score={best_score:.3f})")

        if chosen_bar is None:
            return None

        # ---- beat-level refine ----
        spb = 60.0 / max(1, bpm)
        beat_start = chosen_bar * 4
        beat_end   = beat_start + 4

        on_thr = float(np.median(on_beats) + 0.5*np.std(on_beats))
        pre_beats = slice(max(0, beat_start - 8), beat_start)
        low_med_b = float(np.median(low_db[pre_beats])) if beat_start > 0 else float(np.median(low_db[:4]))
        low_mad_b = _mad(low_db[pre_beats]) if beat_start > 0 else 0.5

        chosen_beat = beat_start
        for j in range(beat_start, min(beat_end, len(low_db))):
            z = (low_db[j] - low_med_b) / max(0.4, low_mad_b)
            if z >= 1.0 and on_beats[j] >= on_thr:
                chosen_beat = j
                break

        drop_beats = float(chosen_beat)
        if SNAP_TO_BEAT:
            drop_beats = round(drop_beats)
        drop_beats += float(OFFSET_BEATS)
        drop_sec = drop_beats * spb

        if DEBUG_DROPS:
            print(f"   [pick] bar {chosen_bar} (~bar {drop_beats/4:.2f}); beat {(chosen_beat % 4)+1}/4")

        return drop_sec, drop_beats

    except Exception as e:
        print(f"[WARN] Drop detection failed for {drums_path}: {e}")
        return None

# =================== ALS helpers ===================
def _set_clip_timing_fields(root: ET.Element, beat_value: float, loop_end_beats: Optional[float]=None):
    beat_str = f"{beat_value:.6f}"
    loop_end_str = f"{loop_end_beats:.6f}" if loop_end_beats is not None else None
    for elem in root.iter():
        tag = elem.tag
        if tag in ("StartMarker","LoopStart","HiddenLoopStart"):
            elem.set("Value", beat_str)
        elif loop_end_str and tag in ("LoopEnd","OutMarker"):
            elem.set("Value", loop_end_str)

def _add_or_update_warp_marker_for_all_clips(root: ET.Element, drop_sec: float, drop_beats: float):
    for clip in list(root.iter("Clip")):
        wm_parent = None
        for candidate in clip.iter("WarpMarkers"):
            wm_parent = candidate; break
        if wm_parent is None:
            wm_parent = ET.Element("WarpMarkers")
            clip.insert(0, wm_parent)

        example = None
        for c in wm_parent:
            if c.tag == "WarpMarker":
                example = c; break

        if example is not None:
            new_wm = ET.Element("WarpMarker")
            if "Time" in example.attrib or "BeatTime" in example.attrib:
                if "Time" in example.attrib:
                    new_wm.attrib["Time"] = f"{drop_sec:.6f}"
                if "BeatTime" in example.attrib:
                    new_wm.attrib["BeatTime"] = f"{drop_beats:.6f}"
            else:
                for child in list(example):
                    if child.tag.lower().startswith("time"):
                        ch = ET.Element(child.tag); ch.set("Value", f"{drop_sec:.6f}"); new_wm.append(ch)
                    elif child.tag.lower().startswith("beat"):
                        ch = ET.Element(child.tag); ch.set("Value", f"{drop_beats:.6f}"); new_wm.append(ch)
                    else:
                        ch = ET.Element(child.tag, child.attrib); ch.text = child.text; new_wm.append(ch)
            wm_parent.append(new_wm)
        else:
            wm_parent.append(ET.Element("WarpMarker", {"Time": f"{drop_sec:.6f}", "BeatTime": f"{drop_beats:.6f}"}))

# =================== rename + ALS wiring ===================
def _existing_energy_in_name(name: str) -> Optional[int]:
    m = _FN_ENERGY_RE.match(os.path.basename(name))
    if not m: return None
    try:
        e = int(m.group(2))
        if 1 <= e <= 12: return e
    except Exception: pass
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
        if os.path.exists(p): return p
    return None

def collect_track_names_for_folder(folder_abs: str) -> Dict[str, Optional[str]]:
    roles: Dict[str, Optional[str]] = {"drums": None, "inst": None, "vocals": None}
    for f in os.listdir(folder_abs):
        lf = f.lower()
        if not lf.endswith((".flac", ".wav", ".aiff", ".aif", ".mp3")):
            continue
        role = "drums" if lf.startswith("drums") else ("inst" if lf.startswith("inst") else ("vocals" if lf.startswith("vocals") else None))
        if not role: continue
        roles[role] = os.path.relpath(os.path.join(folder_abs, f), FLAC_FOLDER)
    return roles

def modify_als_file(input_path: Optional[str], target_folder: str, track_names: Dict[str, Optional[str]], bpm_value: Optional[int], force: bool=False) -> None:
    output_als = os.path.join(target_folder, "CH1.als")

    if os.path.exists(output_als) and SKIP_EXISTING and not force:
        return

    if input_path and os.path.abspath(input_path) != os.path.abspath(output_als):
        shutil.copy(input_path, output_als)
    elif not os.path.exists(output_als):
        print(f"[WARN] No template for BPM {bpm_value} and no existing CH1.als at {target_folder}; skipping.")
        return

    print(f"\n🎯 Creating/Updating ALS for {target_folder} (BPM {bpm_value})")

    drums_abs = None
    for f in os.listdir(target_folder):
        lf = f.lower()
        if lf.startswith("drums") and lf.endswith((".flac", ".wav", ".aiff", ".aif", ".mp3")):
            drums_abs = os.path.join(target_folder, f); break
    if not drums_abs and track_names.get("drums"):
        maybe_abs = os.path.join(FLAC_FOLDER, track_names["drums"])
        if os.path.exists(maybe_abs): drums_abs = maybe_abs
    if drums_abs:
        print(f"   🥁 Using drums: {os.path.basename(drums_abs)}")

    new_loop_end = None
    drop_sec = drop_beats = None
    if bpm_value and drums_abs and os.path.exists(drums_abs):
        new_loop_end = get_duration_in_beats(drums_abs, bpm_value)
        pair = detect_drop_beats(drums_abs, bpm_value)
        if pair:
            drop_sec, drop_beats = pair
            if DEBUG_DROPS:
                print(f"   → Drop ≈ bar {drop_beats/4:.2f}")
    else:
        if not bpm_value:
            print("   ℹ️  No BPM; cannot detect drop.")
        elif not drums_abs:
            print("   ℹ️  No drums stem found; skipping drop detection.")

    with gzip.open(output_als, "rb") as f:
        als_data = f.read()
    tree = ET.parse(BytesIO(als_data))
    root = tree.getroot()

    loop_end_beats = float(new_loop_end) if new_loop_end else None
    if drop_beats is not None:
        _set_clip_timing_fields(root, drop_beats, loop_end_beats=loop_end_beats)
        _add_or_update_warp_marker_for_all_clips(root, drop_sec=drop_sec, drop_beats=drop_beats)
        print(f"   ➕ Drop stamped to ALL clips at ~bar {drop_beats/4:.2f}.")
    else:
        if loop_end_beats is not None:
            _set_clip_timing_fields(root, beat_value=0.0, loop_end_beats=loop_end_beats)
        print("   ⚠️ Drop not detected. Only LoopEnd/OutMarker updated if available.")

    buf = BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    with gzip.open(output_als, "wb") as f_out:
        f_out.write(buf.getvalue())

    with gzip.open(output_als, "rb") as f:
        als_str = f.read().decode("utf-8")

    for role, rel_path in track_names.items():
        if not rel_path: continue
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

# =================== Moves / Flows ===================
def move_folder_to_target(stems_folder_abs: str, bpm: int, cam: str, energy: Optional[int]) -> Tuple[str, bool, bool, bool]:
    track_name = os.path.basename(stems_folder_abs)
    bpm_folder = os.path.join(destination_folder, str(bpm))
    cam_folder = os.path.join(bpm_folder, cam)
    os.makedirs(cam_folder, exist_ok=True)

    dest_path = os.path.join(cam_folder, track_name)
    moved = False
    if stems_folder_abs != dest_path:
        if os.path.exists(dest_path): shutil.rmtree(dest_path)
        shutil.move(stems_folder_abs, dest_path)
        print(f"[MOVE] {track_name}  →  {cam_folder}")
        moved = True

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

def process_new_stems_from_toBeOrganized(idx_exact: Dict[str,str]) -> None:
    if not os.path.isdir(htdemucs_source_folder):
        return
    for folder in os.listdir(htdemucs_source_folder):
        stems_abs = os.path.join(htdemucs_source_folder, folder)
        if not os.path.isdir(stems_abs): continue

        src_audio = find_source_for_track(idx_exact, folder)
        if not src_audio: continue

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key": continue

        move_folder_to_target(stems_abs, bpm, cam, energy)

def _snapshot_tracks(dest_folder: str):
    if not os.path.isdir(dest_folder):
        print(f"[INFO] Destination not found or empty: {dest_folder}")
        return []
    items = []
    for bpm_dir in os.listdir(dest_folder):
        if not bpm_dir.isdigit(): continue
        bpm_abs = os.path.join(dest_folder, bpm_dir)
        if not os.path.isdir(bpm_abs): continue
        for cam_dir in os.listdir(bpm_abs):
            cam_abs = os.path.join(bpm_abs, cam_dir)
            if not os.path.isdir(cam_abs): continue
            for track_folder in os.listdir(cam_abs):
                track_abs = os.path.join(cam_abs, track_folder)
                if os.path.isdir(track_abs):
                    items.append((int(bpm_dir), cam_dir, track_folder, track_abs))
    return items

def reconcile_existing_stems(idx_exact: Dict[str,str]) -> None:
    moved = regenerated = 0
    for cur_bpm, cur_cam, track_folder, track_abs in _snapshot_tracks(destination_folder):
        src_audio = find_source_for_track(idx_exact, track_folder)
        if not src_audio: continue

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key": continue

        _, did_move, added_energy, wrote_als = move_folder_to_target(track_abs, bpm, cam, energy)
        if did_move: moved += 1
        if wrote_als: regenerated += 1

    print(f"[INFO] Reconcile complete. Moved: {moved}, ALS updated: {regenerated}")

# =================== Main ===================
if __name__ == "__main__":
    if input("Ready? Type 'yes' to begin: ").strip().lower() == "yes":
        for p in (destination_folder, FLAC_FOLDER, ALS_FILES_FOLDER, htdemucs_source_folder):
            try: os.makedirs(p, exist_ok=True)
            except Exception: pass
        if not _HAS_LIBROSA:
            print("⚠️  librosa not found; drop detection will be skipped. Install: pip install librosa soundfile numpy")
        index_exact = index_source_library(mp3_source_folder)
        process_new_stems_from_toBeOrganized(index_exact)
        reconcile_existing_stems(index_exact)
        print("🎵 Done.")
    else:
        print("Operation cancelled.")
