#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, base64, html, math, shutil, gzip, subprocess, unicodedata, csv
from io import BytesIO
from typing import Optional, Dict, Tuple, List
import xml.etree.ElementTree as ET

import mutagen
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from mutagen.aiff import AIFF

import trackTime  # your module for duration

try:
    import numpy as np
except Exception:
    np = None

try:
    from edm_true_drop_detector import detect_true_drop
except Exception:
    detect_true_drop = None

try:
    from drop_signature_model import decode_audio_ffmpeg, signature_matrix_from_model, suggest_shift_by_signature
except Exception:
    decode_audio_ffmpeg = None
    signature_matrix_from_model = None
    suggest_shift_by_signature = None

try:
    from alsdrop.infer import run_predict as alsdrop_run_predict
except Exception:
    alsdrop_run_predict = None

# ========= USER SETTINGS =========
mp3_source_folder        = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder   = "/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized"
destination_folder       = "/Users/alirahimlou/Desktop/STEMS2"
ALS_FILES_FOLDER         = "alsFiles"  # folder containing <BPM>.als templates
FLAC_FOLDER              = destination_folder
SKIP_EXISTING            = True   # when False, force-regenerate ALS for every track folder
RENAME_TRACKS            = True
ONE_TIME_UNDO_RENAME     = False  # revert "Title - Artist" back to "Artist - Title" in destination_folder
DROP_ANCHOR_OFFSET_SEC   = 0.0    # manual nudge for drop anchor (negative = earlier, positive = later)
DROP_ANCHOR_OVERRIDES_SEC = {
    # Per-track fine offsets (folder name -> seconds). Keep empty for fully automatic mode.
}
DROP_MARKER_DB_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drop_marker_db.json")
DROP_SELF_LEARN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drop_self_learning_model.json")
USE_SELF_LEARN_MODEL     = True
USE_ALSDROP_MODEL        = True
_ALSDROP_ROOT            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alsdrop")
_ALSDROP_MODELS_DIR      = os.path.join(_ALSDROP_ROOT, "models")
_ALSDROP_OUTPUTS_DIR     = os.path.join(_ALSDROP_ROOT, "outputs")
ALSDROP_MODEL_PATH_ENV   = (os.environ.get("ALSDROP_MODEL_PATH") or "").strip()
ALSDROP_MODEL_BASELINE_PATH = os.path.join(_ALSDROP_MODELS_DIR, "model.pt")
ALSDROP_MODEL_PROD_PATH  = os.path.join(_ALSDROP_MODELS_DIR, "model_prod.pt")
ALSDROP_AUTO_MODEL_SELECT = str(os.environ.get("ALSDROP_AUTO_MODEL_SELECT", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALSDROP_MIN_TEST_DOWNBEAT = float(os.environ.get("ALSDROP_MIN_TEST_DOWNBEAT", "0.75"))
ALSDROP_MIN_CANDIDATE_RECALL = float(os.environ.get("ALSDROP_MIN_CANDIDATE_RECALL", "0.90"))
ALSDROP_REVIEW_THRESHOLD = 0.55
ALSDROP_MIN_CONFIDENCE   = 0.60
ALSDROP_USE_MADMOM       = True
ALSDROP_REVIEW_QUEUE_BASENAME = "alsdrop_review_queue"
ALSDROP_REQUIRE_MODEL_SELECTION = str(os.environ.get("ALSDROP_REQUIRE_MODEL_SELECTION", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALSDROP_ALLOW_SAFE_FALLBACK = str(os.environ.get("ALSDROP_ALLOW_SAFE_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "on"}
ALSDROP_SAFE_FALLBACK_MIN_CONF = float(os.environ.get("ALSDROP_SAFE_FALLBACK_MIN_CONF", "0.88"))
ALSDROP_SAFE_FALLBACK_MIN_MARGIN = float(os.environ.get("ALSDROP_SAFE_FALLBACK_MIN_MARGIN", "0.08"))

# ===== Stage-2 Micro Tuning Controls (edit these directly) =====
DROP_STAGE2_ENABLE = True
DROP_STAGE2_ATTACK_NUDGE_ENABLE = False
DROP_STAGE2_MAX_SHIFT_BEATS = 0.38
DROP_STAGE2_MIN_SHIFT_SEC = 0.008
DROP_STAGE2_FIXED_LATE_SEC = 0.0008
# By default we do not alter anchors learned from manual cues / DB labels.
DROP_STAGE2_APPLY_ON_DB_ANCHOR = False
DROP_STAGE2_APPLY_ON_MANUAL_ANCHOR = False
DROP_STAGE2_VERBOSE = True

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".mp4")
STEM_AUDIO_EXTS = (".flac", ".wav", ".aiff", ".aif", ".mp3")
BEATS_PER_BAR = 4
WARP_GRID_BEATS = 1.0 / 16.0

_DROP_DB_CACHE = None
_DROP_LEARN_MODEL_CACHE = None
_ALSDROP_WARNED = False
_ALSDROP_REVIEW_QUEUE: List[Dict[str, object]] = []
_ALSDROP_MODEL_RESOLVED_PATH: Optional[str] = None
_ALSDROP_MODEL_RESOLVED_REASON: str = ""
_DROP_ROLE_PREFIX_RE = re.compile(r"^(drums|inst|vocals)_", re.I)
_DROP_BPM_KEY_RE = re.compile(r"^\d{2,3}_[0-9]{1,2}[ab](?:_\d{1,2})?[-_]", re.I)
_DROP_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

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

# Case-insensitive energy pattern; match original name (no .lower())
_FN_ENERGY_RE   = re.compile(r"^(drums|inst|vocals)_\d+_[0-9]{1,2}[ab]_(\d{1,2})-", re.I)

def _as_float(x) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _clean_bpm_val(x) -> Optional[int]:
    v = _as_float(x)
    return int(round(v)) if v and v > 0 else None

def _load_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _metrics_candidates_for_model(model_path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    out = []
    out.append(os.path.join(_ALSDROP_OUTPUTS_DIR, f"train_metrics_{stem}.json"))
    out.append(os.path.join(_ALSDROP_OUTPUTS_DIR, f"eval_{stem}.json"))
    if stem.startswith("model_"):
        suffix = stem[len("model_") :]
        if suffix:
            out.append(os.path.join(_ALSDROP_OUTPUTS_DIR, f"train_metrics_{suffix}.json"))
            out.append(os.path.join(_ALSDROP_OUTPUTS_DIR, f"eval_{suffix}.json"))
    if stem == "model":
        out.extend(
            [
                os.path.join(_ALSDROP_OUTPUTS_DIR, "train_metrics_og_full.json"),
                os.path.join(_ALSDROP_OUTPUTS_DIR, "train_metrics.json"),
            ]
        )
    return out

def _load_model_metrics(model_path: str) -> Optional[dict]:
    for p in _metrics_candidates_for_model(model_path):
        if os.path.isfile(p):
            data = _load_json_file(p)
            if data:
                return data
    return None

def _get_model_quality(metrics: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(metrics, dict):
        return None, None
    test_downbeat = _as_float(((metrics.get("test_metrics") or {}).get("downbeat_acc")))
    if test_downbeat is None:
        test_downbeat = _as_float(((metrics.get("summary") or {}).get("downbeat_acc")))
    candidate_recall = _as_float(metrics.get("candidate_recall"))
    if candidate_recall is None:
        candidate_recall = _as_float(((metrics.get("build_stats") or {}).get("candidate_recall")))
    if candidate_recall is None:
        candidate_recall = _as_float(((metrics.get("summary") or {}).get("candidate_recall_100ms")))
    return test_downbeat, candidate_recall

def _resolve_alsdrop_model_path() -> Tuple[Optional[str], str]:
    if ALSDROP_MODEL_PATH_ENV:
        if os.path.exists(ALSDROP_MODEL_PATH_ENV):
            return ALSDROP_MODEL_PATH_ENV, "env_override"
        return None, f"env_override_missing:{ALSDROP_MODEL_PATH_ENV}"

    baseline = ALSDROP_MODEL_BASELINE_PATH
    prod = ALSDROP_MODEL_PROD_PATH

    if ALSDROP_AUTO_MODEL_SELECT and os.path.isfile(prod):
        prod_metrics = _load_model_metrics(prod)
        prod_test_downbeat, prod_candidate_recall = _get_model_quality(prod_metrics)
        if (
            prod_test_downbeat is not None
            and prod_candidate_recall is not None
            and prod_test_downbeat >= float(ALSDROP_MIN_TEST_DOWNBEAT)
            and prod_candidate_recall >= float(ALSDROP_MIN_CANDIDATE_RECALL)
        ):
            return (
                prod,
                f"auto_prod_pass(test={prod_test_downbeat:.3f},recall={prod_candidate_recall:.3f})",
            )
        if os.path.isfile(baseline):
            downbeat_txt = "n/a" if prod_test_downbeat is None else f"{prod_test_downbeat:.3f}"
            recall_txt = "n/a" if prod_candidate_recall is None else f"{prod_candidate_recall:.3f}"
            return (
                baseline,
                f"auto_baseline_prod_fail(test={downbeat_txt},recall={recall_txt})",
            )

    if os.path.isfile(baseline):
        return baseline, "baseline_default"
    if os.path.isfile(prod):
        return prod, "prod_only"

    for fallback_name in ("model_gold_v2.pt", "model_patch_smoke.pt", "model_tmp_cpu.pt"):
        p = os.path.join(_ALSDROP_MODELS_DIR, fallback_name)
        if os.path.isfile(p):
            return p, f"fallback:{fallback_name}"
    return None, "no_model_found"

def _drop_db_exact_key(path_or_name: str) -> str:
    base = nfc(os.path.basename(path_or_name or ""))
    stem, _ = os.path.splitext(base)
    return stem.lower().strip()

def _drop_db_slug_from_exact(exact_key: str) -> str:
    s = (exact_key or "").lower().strip()
    s = _DROP_ROLE_PREFIX_RE.sub("", s)
    s = _DROP_BPM_KEY_RE.sub("", s)
    s = _DROP_NON_ALNUM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def _drop_db_token_key(text: str) -> str:
    toks = [t for t in re.sub(r"\s+", " ", (text or "").strip().lower()).split(" ") if t]
    if not toks:
        return ""
    return " ".join(sorted(toks))

def _drop_db_bpm_from_exact(exact_key: str) -> Optional[int]:
    m = re.match(r"^(?:drums|inst|vocals)_(\d{2,3})_", (exact_key or ""), re.I)
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    return v if v > 0 else None

def _load_drop_marker_db() -> Dict[str, object]:
    global _DROP_DB_CACHE
    if _DROP_DB_CACHE is not None:
        return _DROP_DB_CACHE
    if not os.path.exists(DROP_MARKER_DB_PATH):
        _DROP_DB_CACHE = {}
        return _DROP_DB_CACHE
    try:
        with open(DROP_MARKER_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
        _DROP_DB_CACHE = data
    except Exception as e:
        print(f"[WARN] Could not load drop marker DB at {DROP_MARKER_DB_PATH}: {e}")
        _DROP_DB_CACHE = {}
    return _DROP_DB_CACHE

def _save_drop_marker_db(db: Dict[str, object]) -> bool:
    global _DROP_DB_CACHE
    try:
        with open(DROP_MARKER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=True, sort_keys=True)
            f.write("\n")
        _DROP_DB_CACHE = db
        return True
    except Exception as e:
        print(f"[WARN] Could not save drop marker DB at {DROP_MARKER_DB_PATH}: {e}")
        return False

def _upsert_drop_db_label(drums_path: str, drop_sec: float, bpm: Optional[int], source: str = "manual") -> bool:
    ex = _drop_db_exact_key(drums_path)
    if not ex or drop_sec < 0.0:
        return False
    sk = _drop_db_slug_from_exact(ex)
    db = _load_drop_marker_db()
    if not isinstance(db, dict):
        db = {}

    exact = db.setdefault("exact", {})
    slug = db.setdefault("slug", {})
    stats = db.setdefault("stats", {})
    if not isinstance(exact, dict) or not isinstance(slug, dict) or not isinstance(stats, dict):
        return False

    rec = exact.get(ex, {})
    if not isinstance(rec, dict):
        rec = {}
    old_drop = _as_float(rec.get("drop_sec"))
    old_obs = int(_as_float(rec.get("obs")) or 0)
    new_obs = old_obs + 1
    if old_drop is None:
        new_drop = float(drop_sec)
    else:
        # Running mean update.
        new_drop = ((old_drop * old_obs) + float(drop_sec)) / float(max(1, new_obs))

    rec.update({
        "drop_sec": round(float(new_drop), 6),
        "obs": int(new_obs),
        "obs_used": int(new_obs),
        "bpm": int(bpm) if bpm else _drop_db_bpm_from_exact(ex),
        "slug": sk,
        "sample_path": str(drums_path),
        "last_source": str(source),
    })
    exact[ex] = rec

    srows = slug.get(sk, [])
    if not isinstance(srows, list):
        srows = []
    found = False
    for row in srows:
        if isinstance(row, dict) and str(row.get("exact_key") or "") == ex:
            row["drop_sec"] = rec["drop_sec"]
            row["obs"] = rec["obs"]
            row["bpm"] = rec["bpm"]
            found = True
            break
    if not found:
        srows.append({
            "exact_key": ex,
            "drop_sec": rec["drop_sec"],
            "obs": rec["obs"],
            "bpm": rec["bpm"],
        })
    srows.sort(key=lambda r: (-int(_as_float(r.get("obs")) or 0), str(r.get("exact_key") or "")))
    slug[sk] = srows

    stats["unique_exact_keys"] = int(len(exact))
    stats["unique_slug_keys"] = int(len(slug))
    return _save_drop_marker_db(db)

def _lookup_drop_from_db(drums_path: str, bpm_hint: Optional[int]) -> Tuple[Optional[float], float, str]:
    db = _load_drop_marker_db()
    if not db:
        return None, 0.0, ""

    exact_map = db.get("exact", {})
    slug_map = db.get("slug", {})
    if not isinstance(exact_map, dict) or not isinstance(slug_map, dict):
        return None, 0.0, ""

    exact_key = _drop_db_exact_key(drums_path)
    if exact_key:
        rec = exact_map.get(exact_key)
        if isinstance(rec, dict):
            sec = _as_float(rec.get("drop_sec"))
            obs = int(_as_float(rec.get("obs")) or 1)
            if sec is not None and sec >= 0.0:
                if sec < 1.0 and obs < 3:
                    return None, 0.0, ""
                conf = min(0.99, 0.86 + (0.02 * min(obs, 6)))
                return float(sec), float(conf), f"exact:{exact_key} (obs={obs})"

    slug = _drop_db_slug_from_exact(exact_key)
    if not slug:
        return None, 0.0, ""
    cands = slug_map.get(slug, [])
    if (not isinstance(cands, list) or not cands) and isinstance(slug_map, dict):
        # Order-insensitive fallback for renamed folders:
        # "How We Hit It - ATYYA" <-> "ATYYA - How We Hit It"
        q_tok = _drop_db_token_key(slug)
        if q_tok:
            merged = []
            for sk, rows in slug_map.items():
                if not isinstance(rows, list) or not rows:
                    continue
                if _drop_db_token_key(str(sk)) == q_tok:
                    merged.extend(rows)
            cands = merged
    if not isinstance(cands, list) or not cands:
        return None, 0.0, ""

    rows: List[Tuple[float, int, str]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        sec = _as_float(c.get("drop_sec"))
        obs = int(_as_float(c.get("obs")) or 1)
        bpm = _clean_bpm_val(c.get("bpm"))
        if sec is None or sec < 0.0:
            continue
        bpm_pen = 0.0
        if bpm_hint and bpm:
            bpm_pen = 0.02 * abs(int(bpm_hint) - int(bpm))
        score = (0.12 * min(obs, 10)) - bpm_pen
        rows.append((score, obs, str(c.get("exact_key") or "")))
    if not rows:
        return None, 0.0, ""

    rows.sort(key=lambda x: x[0], reverse=True)
    best_exact = rows[0][2]
    rec = exact_map.get(best_exact, {})
    sec = _as_float(rec.get("drop_sec")) if isinstance(rec, dict) else None
    obs = int(_as_float(rec.get("obs")) or rows[0][1])
    if sec is None or sec < 0.0:
        return None, 0.0, ""
    if sec < 1.0 and obs < 3:
        return None, 0.0, ""
    conf = min(0.95, 0.78 + (0.02 * min(obs, 6)))
    return float(sec), float(conf), f"slug:{slug} -> {best_exact} (obs={obs})"

def _load_drop_self_learn_model() -> Dict[str, object]:
    global _DROP_LEARN_MODEL_CACHE
    if _DROP_LEARN_MODEL_CACHE is not None:
        return _DROP_LEARN_MODEL_CACHE
    if not os.path.exists(DROP_SELF_LEARN_MODEL_PATH):
        _DROP_LEARN_MODEL_CACHE = {}
        return _DROP_LEARN_MODEL_CACHE
    try:
        with open(DROP_SELF_LEARN_MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
        _DROP_LEARN_MODEL_CACHE = data
    except Exception as e:
        print(f"[WARN] Could not load self-learning model at {DROP_SELF_LEARN_MODEL_PATH}: {e}")
        _DROP_LEARN_MODEL_CACHE = {}
    return _DROP_LEARN_MODEL_CACHE

def _predict_drop_delta_beats_from_model(det_meta: Dict[str, float], bpm_hint: Optional[int]) -> Tuple[Optional[float], float, str]:
    if not USE_SELF_LEARN_MODEL:
        return None, 0.0, ""
    model = _load_drop_self_learn_model()
    if not model:
        return None, 0.0, ""
    stats = model.get("stats", {})
    mae = _as_float(stats.get("leave_one_out_mae_beats")) if isinstance(stats, dict) else None
    # Guard rail: ignore weak kNN models.
    if mae is not None and mae > 2.5:
        return None, 0.0, ""

    feats = model.get("feature_names")
    mean = model.get("feature_mean")
    std = model.get("feature_std")
    rows = model.get("rows")
    k = int(_as_float(model.get("k_neighbors")) or 15)
    max_abs = float(_as_float(model.get("max_abs_delta_beats")) or 8.0)
    if not isinstance(feats, list) or not isinstance(mean, list) or not isinstance(std, list) or not isinstance(rows, list):
        return None, 0.0, ""
    if not feats or len(feats) != len(mean) or len(feats) != len(std):
        return None, 0.0, ""

    x_raw: List[float] = []
    for f in feats:
        if str(f) == "tempo_bpm":
            x_raw.append(float(bpm_hint or det_meta.get("tempo_bpm") or 0.0))
        else:
            x_raw.append(float(det_meta.get(str(f), 0.0)))

    x = []
    for i in range(len(feats)):
        m = float(mean[i])
        s = float(std[i])
        if not math.isfinite(s) or s <= 1e-9:
            s = 1.0
        x.append((x_raw[i] - m) / s)

    dists: List[Tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rx = row.get("x")
        ry = _as_float(row.get("y_beats"))
        if not isinstance(rx, list) or ry is None or len(rx) != len(x):
            continue
        dist2 = 0.0
        for i in range(len(x)):
            dx = float(x[i]) - float(rx[i])
            dist2 += dx * dx
        dists.append((math.sqrt(dist2), float(ry)))
    if not dists:
        return None, 0.0, ""

    dists.sort(key=lambda t: t[0])
    top = dists[:max(1, min(k, len(dists)))]
    w_sum = 0.0
    y_sum = 0.0
    for dist, y in top:
        w = 1.0 / (dist + 1e-4)
        w_sum += w
        y_sum += w * y
    if w_sum <= 1e-9:
        return None, 0.0, ""
    pred = y_sum / w_sum
    pred = max(-max_abs, min(max_abs, pred))

    # Confidence from neighbor tightness.
    y_vals = [y for _, y in top]
    spread = float(np.std(np.asarray(y_vals, dtype=np.float32))) if np is not None and y_vals else 0.0
    conf = max(0.0, min(1.0, 1.0 - (spread / max(0.5, max_abs * 0.4))))
    return float(pred), float(conf), f"knn(k={len(top)})"

def _predict_drop_shift_beats_by_signature_model(drums_path: str, bpm: Optional[int], drop_sec: Optional[float]) -> Tuple[Optional[float], float, str]:
    if not USE_SELF_LEARN_MODEL:
        return None, 0.0, ""
    if not drums_path or not os.path.exists(drums_path) or not bpm or bpm <= 0 or drop_sec is None:
        return None, 0.0, ""
    if decode_audio_ffmpeg is None or signature_matrix_from_model is None or suggest_shift_by_signature is None:
        return None, 0.0, ""

    model = _load_drop_self_learn_model()
    if not model:
        return None, 0.0, ""
    stats = model.get("stats", {})
    if isinstance(stats, dict):
        mae = _as_float(stats.get("leave_one_out_mae_beats"))
        # Guard rail: ignore weak global models.
        if mae is not None and mae > 2.5:
            return None, 0.0, ""
    sig_mat = signature_matrix_from_model(model)
    if sig_mat is None or len(sig_mat) == 0:
        return None, 0.0, ""

    y = decode_audio_ffmpeg(drums_path, sr=22050)
    if y is None:
        return None, 0.0, ""
    shift_beats, conf, reason = suggest_shift_by_signature(
        y=y,
        sr=22050,
        bpm=int(bpm),
        anchor_sec=float(drop_sec),
        signature_matrix=sig_mat,
        max_shift_beats=8.0,
        step_beats=0.5,
        top_k=20,
    )
    return shift_beats, conf, reason


def _alsdrop_review_queue_paths() -> Tuple[str, str]:
    root = destination_folder if destination_folder else os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(root, ALSDROP_REVIEW_QUEUE_BASENAME)
    return f"{base}.jsonl", f"{base}.csv"


def _queue_alsdrop_review(pred: Dict[str, object]) -> None:
    if not isinstance(pred, dict):
        return
    needs_review = bool(pred.get("needs_manual_review", False))
    selected_by = str(pred.get("selected_by") or "").strip().lower()
    if not needs_review and selected_by == "model":
        return
    row: Dict[str, object] = {
        "audio_path": os.path.abspath(str(pred.get("audio_path") or "")),
        "predicted_sec": float(_as_float(pred.get("predicted_sec")) or 0.0),
        "selected_by": str(pred.get("selected_by") or ""),
        "confidence": float(_as_float(pred.get("confidence")) or 0.0),
        "score_margin": float(_as_float(pred.get("score_margin")) or 0.0),
        "region_window_sec": pred.get("region_window_sec") or [],
        "top_candidates": pred.get("top_candidates") or [],
        "model_path": str(pred.get("model_path") or ""),
    }
    if not row["audio_path"]:
        return
    _ALSDROP_REVIEW_QUEUE.append(row)


def _write_alsdrop_review_queue_exports() -> None:
    if not _ALSDROP_REVIEW_QUEUE:
        return

    dedup: Dict[str, Dict[str, object]] = {}
    for row in _ALSDROP_REVIEW_QUEUE:
        ap = str(row.get("audio_path") or "").strip()
        if not ap:
            continue
        dedup[ap] = row
    rows = [dedup[k] for k in sorted(dedup.keys())]
    if not rows:
        return

    jsonl_path, csv_path = _alsdrop_review_queue_paths()
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "predicted_sec",
                "selected_by",
                "confidence",
                "score_margin",
                "region_window_sec",
                "top_candidates",
                "model_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": str(row.get("audio_path") or ""),
                    "predicted_sec": f"{float(_as_float(row.get('predicted_sec')) or 0.0):.6f}",
                    "selected_by": str(row.get("selected_by") or ""),
                    "confidence": f"{float(_as_float(row.get('confidence')) or 0.0):.6f}",
                    "score_margin": f"{float(_as_float(row.get('score_margin')) or 0.0):.6f}",
                    "region_window_sec": json.dumps(row.get("region_window_sec") or [], ensure_ascii=True),
                    "top_candidates": json.dumps(row.get("top_candidates") or [], ensure_ascii=True),
                    "model_path": str(row.get("model_path") or ""),
                }
            )

    print(f"[ALSDrop] Manual review queue written: {jsonl_path}")
    print(f"[ALSDrop] Manual review queue written: {csv_path}")

def _predict_drop_with_alsdrop(drums_path: str, bpm: Optional[int]) -> Tuple[Optional[float], float, Dict[str, float]]:
    global _ALSDROP_WARNED, _ALSDROP_MODEL_RESOLVED_PATH, _ALSDROP_MODEL_RESOLVED_REASON
    if not USE_ALSDROP_MODEL:
        return None, 0.0, {}
    if alsdrop_run_predict is None:
        if not _ALSDROP_WARNED:
            print("[WARN] ALSDrop import unavailable; falling back to legacy drop detector.")
            _ALSDROP_WARNED = True
        return None, 0.0, {}
    if not drums_path or not os.path.exists(drums_path):
        return None, 0.0, {}

    if _ALSDROP_MODEL_RESOLVED_PATH is None:
        _ALSDROP_MODEL_RESOLVED_PATH, _ALSDROP_MODEL_RESOLVED_REASON = _resolve_alsdrop_model_path()
        if _ALSDROP_MODEL_RESOLVED_PATH:
            print(f"[ALSDrop] Using model: {_ALSDROP_MODEL_RESOLVED_PATH} ({_ALSDROP_MODEL_RESOLVED_REASON})")

    model_path = _ALSDROP_MODEL_RESOLVED_PATH
    if not model_path or not os.path.exists(model_path):
        if not _ALSDROP_WARNED:
            reason = _ALSDROP_MODEL_RESOLVED_REASON or "unresolved"
            print(f"[WARN] ALSDrop model not found ({reason}); falling back to legacy drop detector.")
            _ALSDROP_WARNED = True
        return None, 0.0, {}

    try:
        pred = alsdrop_run_predict(
            audio_path=drums_path,
            model_path=model_path,
            out_json=None,
            device="auto",
            use_madmom=bool(ALSDROP_USE_MADMOM),
            bpm_override=float(bpm) if bpm else None,
            review_threshold=float(ALSDROP_REVIEW_THRESHOLD),
        )
    except Exception as e:
        print(f"[WARN] ALSDrop model inference failed for {drums_path}: {e}")
        return None, 0.0, {}

    _queue_alsdrop_review(pred)

    sec = _as_float(pred.get("predicted_sec"))
    conf = float(_as_float(pred.get("confidence")) or 0.0)
    if sec is None or sec < 0.0:
        return None, 0.0, {}

    review = bool(pred.get("needs_manual_review", False))
    selected_by = str(pred.get("selected_by") or "").strip().lower()
    score_margin = float(_as_float(pred.get("score_margin")) or 0.0)
    safe_fallback_ok = (
        bool(ALSDROP_ALLOW_SAFE_FALLBACK)
        and selected_by == "safe_fallback"
        and conf >= float(ALSDROP_SAFE_FALLBACK_MIN_CONF)
        and score_margin >= float(ALSDROP_SAFE_FALLBACK_MIN_MARGIN)
        and (not review)
    )
    selection_ok = (selected_by == "model") or safe_fallback_ok
    if not bool(ALSDROP_REQUIRE_MODEL_SELECTION):
        selection_ok = selection_ok or (selected_by in {"model_rejected", "safe_fallback"} and not review)

    meta = {
        "alsdrop_source": 1.0,
        "alsdrop_conf": float(conf),
        "alsdrop_review": 1.0 if review else 0.0,
        "alsdrop_margin": float(score_margin),
        "alsdrop_safe_ok": 1.0 if safe_fallback_ok else 0.0,
        "alsdrop_selected_by_model": 1.0 if selected_by == "model" else 0.0,
        "alsdrop_candidates": float(_as_float(pred.get("n_candidates")) or 0.0),
        "alsdrop_selected_by": 1.0 if selected_by == "model" else 0.0,
        "tempo_bpm": float(_as_float(pred.get("bpm_used")) or _as_float(pred.get("tempo_est")) or float(bpm or 0.0)),
        # valid gates downstream self-learn adjustments.
        "valid": 1.0 if (conf >= float(ALSDROP_MIN_CONFIDENCE) and not review and selection_ok) else 0.0,
    }
    if selected_by != "model" and not safe_fallback_ok:
        print(
            f"[WARP] ALSDrop selection '{selected_by}' not trusted "
            f"(conf={conf:.2f}, margin={score_margin:.3f}); falling back."
        )
    return float(sec), float(conf), meta

def _detect_drop_anchor_sec(drums_path: str, bpm: int) -> Tuple[Optional[float], float, Dict[str, float]]:
    y_cached = None
    sr_cached = 0

    def _apply_wavefront_nudge(sec: Optional[float], meta: Dict[str, float]) -> Tuple[Optional[float], Dict[str, float]]:
        nonlocal y_cached, sr_cached
        if sec is None:
            return None, dict(meta or {})
        out_meta = dict(meta or {})
        try:
            if y_cached is None:
                y_cached, sr_cached = _load_audio_mono_f32_ffmpeg(drums_path, sr=22050)
            if y_cached is not None:
                nudged = _nudge_anchor_to_wavefront(y_cached, int(sr_cached), float(sec), int(bpm))
                if nudged > float(sec) + 1e-6:
                    out_meta["wavefront_nudge_sec"] = float(nudged - float(sec))
                    sec = float(nudged)
                if bool(DROP_STAGE2_ENABLE) and bool(DROP_STAGE2_ATTACK_NUDGE_ENABLE):
                    micro = _micro_nudge_to_attack(
                        y_cached,
                        int(sr_cached),
                        float(sec),
                        int(bpm),
                        max_shift_beats=float(DROP_STAGE2_MAX_SHIFT_BEATS),
                    )
                    if micro > float(sec) + float(DROP_STAGE2_MIN_SHIFT_SEC):
                        out_meta["stage2_micro_nudge_sec"] = float(micro - float(sec))
                        sec = float(micro)
        except Exception:
            pass
        return float(sec), out_meta

    model_sec, model_conf, model_meta = _predict_drop_with_alsdrop(drums_path, bpm)
    model_valid = float(_as_float((model_meta or {}).get("valid")) or 0.0) >= 0.5
    if model_sec is not None and model_conf >= float(ALSDROP_MIN_CONFIDENCE) and model_valid:
        model_sec, model_meta = _apply_wavefront_nudge(model_sec, model_meta)
        print(f"[WARP] ALSDrop anchor: {model_sec:.3f}s (conf={model_conf:.2f})")
        return float(model_sec), float(model_conf), dict(model_meta or {})
    if model_sec is not None:
        print(
            f"[WARP] ALSDrop rejected: sec={float(model_sec):.3f}s conf={float(model_conf):.2f} "
            f"valid={1 if model_valid else 0}; using legacy detector."
        )

    det_sec, det_conf, det_meta = _first_drop_sec_from_drums(drums_path, bpm)
    if det_sec is not None:
        out_meta = dict(det_meta or {})
        if model_sec is not None:
            out_meta["alsdrop_sec"] = float(model_sec)
            out_meta["alsdrop_conf"] = float(model_conf)
        det_sec, out_meta = _apply_wavefront_nudge(det_sec, out_meta)
        return float(det_sec), float(det_conf), out_meta

    if model_sec is not None:
        model_sec, model_meta = _apply_wavefront_nudge(model_sec, model_meta)
        print(f"[WARP] ALSDrop low-confidence fallback: {model_sec:.3f}s (conf={model_conf:.2f})")
        return float(model_sec), float(model_conf), dict(model_meta or {})
    return None, 0.0, {}

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

def swap_artist_title(text: str) -> str:
    parts = [p.strip() for p in text.split(" - ", 1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[1]} - {parts[0]}"
    return text

def desired_track_folder_name(track_name: str, src_basename: Optional[str], allow_rename: bool) -> str:
    if not (RENAME_TRACKS and allow_rename):
        return track_name
    if src_basename and nfc(track_name) == nfc(src_basename):
        return swap_artist_title(track_name)
    return track_name

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
        if ext in {".wav", ".aif", ".aiff"}:
            audio = WAVE(file_path) if ext == ".wav" else AIFF(file_path)
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
    if pl.endswith(".aiff"): return 3
    if pl.endswith(".aif"): return 4
    return 5

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
    hit = idx_exact.get(key)
    if hit:
        return hit
    if RENAME_TRACKS:
        swapped = swap_artist_title(track_folder_name)
        if swapped != track_folder_name:
            return idx_exact.get(nfc(swapped))
    return None

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
    sec = get_duration_seconds(track_path)
    if sec is None:
        return None
    return f"{(sec * bpm)/60.0:.6f}"

def _duration_seconds_ffprobe(track_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        track_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except Exception:
        return None
    out = (proc.stdout or b"").decode("utf-8", "ignore").strip()
    if not out:
        return None
    first = out.splitlines()[0].strip()
    try:
        sec = float(first)
    except Exception:
        return None
    if math.isfinite(sec) and sec > 0:
        return sec
    return None

def _duration_seconds_mutagen(track_path: str) -> Optional[float]:
    try:
        audio = mutagen.File(track_path)
    except Exception:
        return None
    if not audio or not getattr(audio, "info", None):
        return None
    try:
        sec = float(getattr(audio.info, "length", 0.0))
    except Exception:
        return None
    if math.isfinite(sec) and sec > 0:
        return sec
    return None

def get_duration_seconds(track_path: str) -> Optional[float]:
    if not os.path.exists(track_path):
        print(f"[ERROR] Duration for {track_path}: file not found")
        return None

    sec = _duration_seconds_ffprobe(track_path)
    if sec is not None:
        return sec

    sec = _duration_seconds_mutagen(track_path)
    if sec is not None:
        return sec

    # Keep legacy FLAC path as a last fallback.
    if track_path.lower().endswith(".flac"):
        try:
            sec = float(trackTime.get_track_duration(track_path))
            if sec > 0:
                return sec
        except Exception:
            pass

    print(f"[ERROR] Duration for {track_path}: unable to read duration (ffprobe/mutagen/trackTime).")
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
        if not f.lower().endswith((".flac", ".wav", ".aiff", ".aif")):
            continue
        low = f.lower()
        role = "drums" if low.startswith("drums") else ("inst" if low.startswith("inst") else ("vocals" if low.startswith("vocals") else None))
        if not role:
            continue
        rel_path = os.path.relpath(os.path.join(folder_abs, f), FLAC_FOLDER)
        cur = roles[role]
        if cur is None or _ext_priority(rel_path) < _ext_priority(cur):
            roles[role] = rel_path
    return roles

def _find_drums_stem_path(target_folder: str, track_names: Dict[str, Optional[str]]) -> Optional[str]:
    drums_rel = track_names.get("drums")
    if drums_rel:
        drums_abs = os.path.join(FLAC_FOLDER, drums_rel)
        if os.path.exists(drums_abs):
            return drums_abs
    try:
        for f in os.listdir(target_folder):
            low = f.lower()
            if low.startswith("drums") and low.endswith(STEM_AUDIO_EXTS):
                return os.path.join(target_folder, f)
    except Exception:
        pass
    return None

def _load_audio_mono_f32_ffmpeg(audio_path: str, sr: int = 22050):
    if np is None:
        return None, sr
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; skipping drop anchor stamping.")
        return None, sr
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "ignore").strip()
        print(f"[WARN] ffmpeg decode failed for {audio_path}: {err or e}")
        return None, sr
    if not proc.stdout:
        return None, sr
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=False)
    if y.size == 0:
        return None, sr
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y, sr

def _smooth_same(x, win: int):
    if np is None:
        return x
    if win <= 1 or x is None or len(x) < 3:
        return x
    w = int(max(1, win))
    if w % 2 == 0:
        w += 1
    kern = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kern, mode="same")

def _spectral_flux(y, sr: int, hop_sec: float = 0.01, n_fft: int = 2048):
    if np is None or y is None or len(y) < n_fft:
        return None, hop_sec
    hop = max(1, int(round(hop_sec * sr)))
    if len(y) < n_fft + hop:
        return None, hop_sec

    n_frames = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    prev_mag = None
    for i in range(n_frames):
        frame = y[i * hop:i * hop + n_fft]
        if len(frame) < n_fft:
            break
        mag = np.abs(np.fft.rfft(frame * win))
        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = float(np.sum(diff[diff > 0.0]))
        prev_mag = mag

    flux = _smooth_same(flux, 9)
    peak = float(np.max(flux)) if len(flux) else 0.0
    if peak > 0:
        flux = flux / peak
    return flux, hop_sec

def _estimate_beat_phase(flux, hop_sec: float, beat_sec: float, around_sec: float, dur_sec: float) -> float:
    if np is None or flux is None or len(flux) < 4:
        return 0.0
    win_start = max(6.0, around_sec - 16.0 * beat_sec)
    win_end = min(dur_sec - 2.0, around_sec + 32.0 * beat_sec)
    if win_end <= win_start:
        win_start = 0.0
        win_end = dur_sec

    phase_step = hop_sec
    phases = np.arange(0.0, beat_sec, phase_step)
    best_phase = 0.0
    best_score = -1.0
    for phase in phases:
        n0 = int(math.ceil((win_start - phase) / beat_sec))
        n1 = int(math.floor((win_end - phase) / beat_sec))
        if n1 < n0:
            continue
        score = 0.0
        count = 0
        for n in range(n0, n1 + 1):
            t = phase + n * beat_sec
            idx = int(round(t / hop_sec))
            if 1 <= idx < len(flux) - 1:
                # slight local integration to reduce jitter
                score += float(0.25 * flux[idx - 1] + 0.5 * flux[idx] + 0.25 * flux[idx + 1])
                count += 1
        if count > 0:
            score /= float(count)
            if score > best_score:
                best_score = score
                best_phase = float(phase)
    return best_phase

def _snap_to_beat_with_flux(coarse_sec: float, bpm: int, flux, hop_sec: float, phase_sec: float) -> float:
    if np is None or flux is None or len(flux) < 4:
        return coarse_sec
    beat_sec = 60.0 / max(1.0, float(bpm))
    base_n = int(round((coarse_sec - phase_sec) / beat_sec))
    local_win = 0.14 * beat_sec

    def _peak_near(t: float):
        i0 = max(1, int(round((t - local_win) / hop_sec)))
        i1 = min(len(flux) - 2, int(round((t + local_win) / hop_sec)))
        if i1 <= i0:
            idx = int(round(max(0.0, t) / hop_sec))
            idx = min(max(1, idx), len(flux) - 2)
            return idx, float(flux[idx])
        idx = i0
        val = float(flux[i0])
        for i in range(i0 + 1, i1 + 1):
            v = float(flux[i])
            if v > val:
                val = v
                idx = i
        return idx, val

    best = None
    for n in range(base_n - 2, base_n + 3):
        t = phase_sec + n * beat_sec
        if t < 0.0:
            continue
        idx, local_peak = _peak_near(t)

        # Continuity: true beats should keep finding strong peaks for several beats after.
        cont = 0.0
        hits = 0
        for k in range(1, 9):
            tk = t + k * beat_sec
            if tk >= (len(flux) * hop_sec):
                break
            _, pv = _peak_near(tk)
            cont += pv
            hits += 1
        cont_score = (cont / float(hits)) if hits > 0 else 0.0

        # Mild penalty for anchors clearly after the detected transition.
        late_penalty = max(0.0, (t - coarse_sec) / beat_sec) * 0.12
        score = (0.62 * local_peak) + (0.38 * cont_score) - late_penalty
        cand = (score, idx)
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return coarse_sec
    return float(best[1] * hop_sec)

def _onset_strength(y, sr: int, hop_sec: float = 0.01, win_sec: float = 0.02):
    if np is None or y is None or len(y) == 0:
        return None, hop_sec
    hop = max(1, int(round(hop_sec * sr)))
    win = max(1, int(round(win_sec * sr)))
    if len(y) <= win:
        return None, hop_sec

    n = 1 + (len(y) - win) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = y[i * hop:i * hop + win]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12)) if seg.size else 0.0
    onset = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    onset = _smooth_same(onset, 5)
    return onset, hop_sec

def _refine_anchor_to_onset(y, sr: int, anchor_sec: float, bpm: int) -> float:
    if np is None:
        return anchor_sec
    onset, hop_sec = _onset_strength(y, sr, hop_sec=0.01, win_sec=0.02)
    if onset is None or len(onset) < 4:
        return anchor_sec

    beat_sec = 60.0 / max(1.0, float(bpm))
    left = max(0.0, anchor_sec - 0.05 * beat_sec)
    right = min(float(len(y)) / float(sr), anchor_sec + 0.40 * beat_sec)
    i0 = max(1, int(round(left / hop_sec)))
    i1 = min(len(onset) - 2, int(round(right / hop_sec)))
    if i1 <= i0:
        return anchor_sec

    window = onset[i0:i1 + 1]
    if len(window) == 0:
        return anchor_sec
    local_thr = float(np.percentile(window, 65))
    global_thr = float(np.percentile(onset, 80)) * 0.45
    thr = max(local_thr, global_thr)

    # Prefer first meaningful transient at/after the snapped beat.
    anchor_i = int(round(anchor_sec / hop_sec))
    best_i = None
    for i in range(max(i0 + 1, anchor_i), i1):
        if onset[i] >= thr and onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]:
            best_i = i
            break

    # Fallback to strongest local transient in search window.
    if best_i is None:
        best_i = i0 + int(np.argmax(window))

    refined = float(best_i * hop_sec)
    if abs(refined - anchor_sec) > (0.45 * beat_sec):
        return anchor_sec
    return refined

def _nudge_anchor_to_wavefront(y, sr: int, anchor_sec: float, bpm: int) -> float:
    """
    Final anti-early correction:
    if anchor lands just before a low-to-high waveform entry, move to the first
    sustained edge of that entry. This only nudges forward and only by a tiny amount.
    """
    if np is None or y is None or len(y) < 8 or bpm <= 0:
        return float(anchor_sec)

    beat_sec = 60.0 / max(1.0, float(bpm))
    # High-resolution absolute envelope: better at finding first waveform entry.
    env = np.abs(y).astype(np.float32, copy=False)
    smooth = max(1, int(round(0.003 * sr)))  # ~3ms
    if smooth > 1:
        ker = np.ones(smooth, dtype=np.float32) / float(smooth)
        env = np.convolve(env, ker, mode="same").astype(np.float32, copy=False)
    if env is None or len(env) < 32:
        return float(anchor_sec)

    pre_sec = 0.30 * beat_sec
    max_fwd_sec = 0.60 * beat_sec
    i_anchor = int(round(float(anchor_sec) * float(sr)))
    i0 = max(1, int(round((float(anchor_sec) - pre_sec) * float(sr))))
    i1 = min(len(env) - 2, int(round((float(anchor_sec) + max_fwd_sec) * float(sr))))
    if i1 <= i0 or i_anchor >= i1:
        return float(anchor_sec)

    pre = env[i0:max(i0 + 1, i_anchor)]
    post = env[i_anchor:i1 + 1]
    if len(pre) < 16 or len(post) < 16:
        return float(anchor_sec)

    pre_base = float(np.percentile(pre, 60.0))
    pre_peak = float(np.percentile(pre, 95.0))
    post_peak = float(np.percentile(post, 98.0))
    # Only nudge when there is a clear "entry" after the anchor.
    if post_peak <= (pre_peak * 1.70 + 1e-8):
        return float(anchor_sec)

    dyn = max(1e-8, post_peak - pre_base)
    thr = pre_base + (0.08 * dyn)
    hold = max(1, int(round(0.004 * sr)))  # ~4ms sustain at sample resolution

    best_i = None
    for i in range(max(i0 + 1, i_anchor), i1 - hold):
        if env[i] < thr:
            continue
        if np.min(env[i:i + hold]) < thr:
            continue
        best_i = i
        break

    if best_i is None:
        return float(anchor_sec)
    refined = float(best_i) / float(sr)
    shift = refined - float(anchor_sec)
    if shift <= 0.002:  # <=2ms is negligible
        return float(anchor_sec)
    if shift > (0.45 * beat_sec):
        return float(anchor_sec)
    return float(refined)

def _micro_nudge_to_attack(y, sr: int, anchor_sec: float, bpm: int, max_shift_beats: float = 0.38) -> float:
    """
    Stage-2 micro alignment (optional):
    move a slightly-early anchor to the first sustained attack just after anchor.
    This keeps movement tiny and forward-only for DJ-safe cue behavior.
    """
    if np is None or y is None or len(y) < 64 or sr <= 0 or bpm <= 0:
        return float(anchor_sec)

    beat_sec = 60.0 / max(1.0, float(bpm))
    max_shift_sec = max(0.0, float(max_shift_beats)) * beat_sec
    max_shift_sec = min(max_shift_sec, 0.45 * beat_sec)
    if max_shift_sec <= 0.0:
        return float(anchor_sec)

    i_anchor = int(round(float(anchor_sec) * float(sr)))
    if i_anchor <= 1 or i_anchor >= (len(y) - 2):
        return float(anchor_sec)

    pre_sec = 0.35 * beat_sec
    i0 = max(2, int(round((float(anchor_sec) - pre_sec) * float(sr))))
    i1 = min(len(y) - 3, int(round((float(anchor_sec) + max_shift_sec) * float(sr))))
    if i1 <= i_anchor + 2:
        return float(anchor_sec)

    env = np.abs(y).astype(np.float32, copy=False)
    smooth = max(1, int(round(0.0025 * sr)))  # ~2.5ms
    if smooth > 1:
        ker = np.ones(smooth, dtype=np.float32) / float(smooth)
        env = np.convolve(env, ker, mode="same").astype(np.float32, copy=False)

    pre = env[i0:i_anchor]
    post = env[i_anchor:i1 + 1]
    if len(pre) < 16 or len(post) < 16:
        return float(anchor_sec)

    pre_med = float(np.percentile(pre, 60.0))
    pre_p90 = float(np.percentile(pre, 90.0))
    post_p98 = float(np.percentile(post, 98.0))
    # Require a meaningful entry after anchor.
    if post_p98 <= (pre_p90 * 1.05 + 1e-9):
        return float(anchor_sec)

    dyn = max(1e-9, post_p98 - pre_med)
    env_thr = pre_med + (0.04 * dyn)
    rise = np.maximum(0.0, np.diff(env, prepend=env[0])).astype(np.float32, copy=False)
    rise_post = rise[i_anchor:i1 + 1]
    if rise_post.size == 0:
        return float(anchor_sec)
    rise_thr = max(float(np.percentile(rise_post, 65.0)) * 0.45, float(np.mean(rise_post)) * 0.90)

    hold = max(1, int(round(0.003 * sr)))  # ~3ms
    sustain = max(1, int(round(0.035 * sr)))  # ~35ms

    best_i = None
    for i in range(i_anchor + 1, i1 - max(hold, sustain)):
        if env[i] < env_thr:
            continue
        if rise[i] < rise_thr:
            continue
        if np.min(env[i:i + hold]) < env_thr:
            continue
        # Keep entries where energy does not immediately collapse.
        early = float(np.mean(env[i:i + sustain]))
        late = float(np.mean(env[i + hold:i + hold + sustain]))
        if late < (0.72 * early):
            continue
        best_i = i
        break

    # Fallback for very subtle entries: small but clear rise from anchor level.
    if best_i is None:
        anchor_e = float(env[i_anchor])
        subtle_thr = max(anchor_e * 1.08, pre_med + (0.03 * dyn))
        subtle_hold = max(1, int(round(0.020 * sr)))  # ~20ms
        for i in range(i_anchor + 1, i1 - subtle_hold):
            if rise[i] <= 0.0:
                continue
            if env[i] < subtle_thr:
                continue
            if float(np.mean(env[i:i + subtle_hold])) < (anchor_e * 1.05):
                continue
            best_i = i
            break

    if best_i is None:
        return float(anchor_sec)

    refined = float(best_i) / float(sr)
    shift = refined - float(anchor_sec)
    if shift <= float(DROP_STAGE2_MIN_SHIFT_SEC):
        return float(anchor_sec)
    if shift > max_shift_sec:
        return float(anchor_sec)
    return float(refined)

def _refine_to_structural_drop(env, low_ratio, onset, hop_sec: float, beat_sec: float, anchor_sec: float) -> float:
    if np is None or env is None or len(env) < 20:
        return anchor_sec
    beat_frames = max(1, int(round(beat_sec / hop_sec)))
    if beat_frames < 1:
        return anchor_sec

    def _mean(arr, s_sec: float, e_sec: float) -> float:
        i0 = max(0, int(round(s_sec / hop_sec)))
        i1 = min(len(arr), int(round(e_sec / hop_sec)))
        if i1 <= i0:
            return 0.0
        return float(np.mean(arr[i0:i1]))

    def _beat_corr(arr, start_sec: float, beats: int = 16) -> float:
        i0 = max(0, int(round(start_sec / hop_sec)))
        i1 = min(len(arr), i0 + beats * beat_frames)
        seq = arr[i0:i1]
        return _periodicity_score(seq, beat_frames)

    cands = []
    # Search forward only: anchor can be a pre-drop pickup, but we avoid moving earlier.
    for off_beats in np.arange(0.0, 4.01, 0.25):
        c = anchor_sec + float(off_beats) * beat_sec
        pre = _mean(env, c - 8 * beat_sec, c - 2 * beat_sec)
        imp = _mean(env, c, c + 2 * beat_sec)
        sus = _mean(env, c + 2 * beat_sec, c + 12 * beat_sec)
        pre_low = _mean(low_ratio, c - 8 * beat_sec, c - 2 * beat_sec)
        post_low = _mean(low_ratio, c + 1 * beat_sec, c + 12 * beat_sec)
        bc = _beat_corr(onset, c, beats=16)

        if imp <= 1e-7 and sus <= 1e-7:
            continue
        # Fake-drop penalty only when sustain truly collapses.
        fake = 0.0
        if sus < (pre + 0.01):
            fake += min(1.0, (pre + 0.01 - sus) / (abs(pre) + 0.01))
        if sus < (0.60 * imp):
            fake += min(1.0, ((0.60 * imp) - sus) / (0.60 * imp + 1e-9))

        score = (
            0.42 * (sus - pre) +
            0.26 * (post_low - pre_low) +
            0.22 * (imp - pre) +
            0.10 * bc -
            0.30 * fake
        )
        cands.append((score, c, sus - pre, post_low - pre_low, imp - pre))

    if not cands:
        return anchor_sec

    cands.sort(key=lambda x: x[0], reverse=True)
    best_score, _, best_sus_gain, best_low_gain, best_imp_gain = cands[0]

    # Pick earliest candidate near the best score, as long as it still carries
    # strong impact and low-end gain. This avoids latching onto later repeats.
    near = [x for x in cands if x[0] >= (best_score - 0.025)]
    near = sorted(near, key=lambda x: x[1])
    for score, c, sus_gain, low_gain, imp_gain in near:
        if sus_gain < 0.0 or low_gain < 0.0 or imp_gain < 0.0:
            continue
        if best_imp_gain > 1e-9 and imp_gain < (0.68 * best_imp_gain):
            continue
        if best_low_gain > 1e-9 and low_gain < (0.85 * best_low_gain):
            continue
        if best_sus_gain > 1e-9 and sus_gain < (0.70 * best_sus_gain):
            continue
        return c

    return cands[0][1]

def _last_audible_sec(y, sr: int) -> Optional[float]:
    if np is None or y is None or len(y) == 0:
        return None
    hop_sec = 0.05
    win_sec = 0.10
    hop = max(1, int(round(hop_sec * sr)))
    win = max(1, int(round(win_sec * sr)))
    if len(y) <= win:
        return float(len(y)) / float(sr)

    n = 1 + (len(y) - win) // hop
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = y[i * hop:i * hop + win]
        env[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12)) if seg.size else 0.0

    p95 = float(np.percentile(env, 95))
    thr = max(0.005, p95 * 0.03)
    active = env >= thr
    idx = np.where(active)[0]
    if len(idx) == 0:
        return float(len(y)) / float(sr)

    # Ignore tiny clicks at the tail.
    min_len = max(1, int(round(0.25 / hop_sec)))
    last_end = int(idx[-1])
    run_start = last_end
    while run_start > 0 and active[run_start - 1]:
        run_start -= 1
    if (last_end - run_start + 1) < min_len:
        for k in range(run_start - 1, -1, -1):
            if active[k] and not active[min(k + 1, len(active) - 1)]:
                r_end = k
                r_start = k
                while r_start > 0 and active[r_start - 1]:
                    r_start -= 1
                if (r_end - r_start + 1) >= min_len:
                    last_end = r_end
                    break

    sec = float(last_end * hop_sec + win_sec)
    return min(float(len(y)) / float(sr), sec)

def _norm_by_quantiles(arr, q_lo: float = 20.0, q_hi: float = 80.0):
    if np is None or arr is None or len(arr) == 0:
        return arr
    lo = float(np.percentile(arr, q_lo))
    hi = float(np.percentile(arr, q_hi))
    span = hi - lo
    if span <= 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def _periodicity_score(x, lag: int) -> float:
    if np is None or x is None or lag <= 0 or len(x) < (lag + 8):
        return 0.0
    a = x[:-lag].astype(np.float32, copy=False)
    b = x[lag:].astype(np.float32, copy=False)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    corr = float(np.dot(a, b) / den)
    return max(0.0, min(1.0, corr))

def _env_transition_score(env, onset, hop_sec: float, t_sec: float, beat_sec: float) -> Tuple[Optional[float], bool]:
    if np is None or env is None or onset is None or len(env) < 16 or beat_sec <= 1e-9:
        return None, False
    i = int(round(float(t_sec) / float(hop_sec)))
    pre = max(0, int(round((float(t_sec) - (2.0 * beat_sec)) / float(hop_sec))))
    post = min(len(env), int(round((float(t_sec) + (4.0 * beat_sec)) / float(hop_sec))))
    post2 = min(len(env), int(round((float(t_sec) + (2.0 * beat_sec)) / float(hop_sec))))
    if i <= (pre + 1) or post <= (i + 1):
        return None, False

    pre_r = float(np.mean(env[pre:i]))
    post_r = float(np.mean(env[i:post]))
    oi0 = max(0, int(round((float(t_sec) - (0.15 * beat_sec)) / float(hop_sec))))
    oi1 = min(len(onset), int(round((float(t_sec) + (0.15 * beat_sec)) / float(hop_sec))))
    op = float(np.max(onset[oi0:oi1])) if oi1 > oi0 else 0.0

    floor = float(np.percentile(env, 20.0))
    no_dip = float(np.min(env[i:post2])) >= max(pre_r * 0.70, floor * 1.05)
    score = (0.75 * (post_r - pre_r)) + (0.25 * op) + (0.05 if no_dip else -0.05)
    return float(score), bool(no_dip)

def _pull_anchor_earlier_if_late(drums_path: str, bpm: int, anchor_sec: float, det_conf: float) -> float:
    if np is None or not drums_path or bpm <= 0 or anchor_sec <= 0.0:
        return float(anchor_sec)

    y, sr = _load_audio_mono_f32_ffmpeg(drums_path, sr=22050)
    if y is None or len(y) < (sr * 8):
        return float(anchor_sec)

    hop_sec = 0.05
    win_sec = 0.10
    hop = max(1, int(round(hop_sec * sr)))
    win = max(1, int(round(win_sec * sr)))
    if len(y) <= win:
        return float(anchor_sec)

    n_frames = 1 + (len(y) - win) // hop
    env = np.empty(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    prev_env = 0.0
    for i in range(n_frames):
        seg = y[i * hop:i * hop + win]
        env_i = float(np.sqrt(np.mean(seg * seg) + 1e-12)) if seg.size else 0.0
        env[i] = env_i
        onset[i] = max(0.0, env_i - prev_env)
        prev_env = env_i

    env = _smooth_same(env, max(5, int(round(0.80 / hop_sec))))
    onset = _smooth_same(onset, max(3, int(round(0.35 / hop_sec))))
    if env is None or onset is None or len(env) < 16:
        return float(anchor_sec)

    beat_sec = 60.0 / float(bpm)
    base_score, _ = _env_transition_score(env, onset, hop_sec, float(anchor_sec), beat_sec)
    if base_score is None:
        return float(anchor_sec)

    best_score = float(base_score)
    best_t = float(anchor_sec)
    # Scan up to 6 beats back; if transition is clearly stronger there, anchor was late.
    for off in (-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0):
        t = float(anchor_sec) + (float(off) * beat_sec)
        if t <= beat_sec:
            continue
        sc, no_dip = _env_transition_score(env, onset, hop_sec, t, beat_sec)
        if sc is None or not no_dip:
            continue
        if float(sc) > best_score:
            best_score = float(sc)
            best_t = float(t)

    gain = best_score - float(base_score)
    min_gain = 0.10 if det_conf >= 0.90 else 0.12
    if best_t < float(anchor_sec) and gain >= min_gain and best_score >= 0.18:
        # Be slightly conservative on large pulls to avoid overshooting.
        max_pull_beats = 4.25
        max_pull_sec = max_pull_beats * beat_sec
        if (float(anchor_sec) - float(best_t)) > max_pull_sec:
            best_t = float(anchor_sec) - max_pull_sec
        return float(best_t)
    return float(anchor_sec)

def _retime_anchor_local_bidirectional(drums_path: str, bpm: int, anchor_sec: float, det_conf: float) -> float:
    if np is None or not drums_path or bpm <= 0 or anchor_sec <= 0.0:
        return float(anchor_sec)

    y, sr = _load_audio_mono_f32_ffmpeg(drums_path, sr=22050)
    if y is None or len(y) < (sr * 8):
        return float(anchor_sec)

    hop_sec = 0.05
    hop = max(1, int(round(hop_sec * sr)))
    win = max(1, int(round(0.10 * sr)))
    if len(y) <= win:
        return float(anchor_sec)

    n_frames = 1 + (len(y) - win) // hop
    env = np.empty(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    prev_env = 0.0
    for i in range(n_frames):
        seg = y[i * hop:i * hop + win]
        env_i = float(np.sqrt(np.mean(seg * seg) + 1e-12)) if seg.size else 0.0
        env[i] = env_i
        onset[i] = max(0.0, env_i - prev_env)
        prev_env = env_i

    env = _smooth_same(env, max(5, int(round(0.80 / hop_sec))))
    onset = _smooth_same(onset, max(3, int(round(0.35 / hop_sec))))
    if env is None or onset is None or len(env) < 16:
        return float(anchor_sec)

    beat_sec = 60.0 / float(bpm)
    base_score, base_nodip = _env_transition_score(env, onset, hop_sec, float(anchor_sec), beat_sec)
    if base_score is None:
        return float(anchor_sec)

    cands: List[Tuple[float, float]] = []  # (t_sec, score)
    if base_nodip:
        cands.append((float(anchor_sec), float(base_score)))
    for off in (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        t = float(anchor_sec) + (float(off) * beat_sec)
        if t <= beat_sec:
            continue
        sc, no_dip = _env_transition_score(env, onset, hop_sec, t, beat_sec)
        if sc is None or not no_dip:
            continue
        cands.append((float(t), float(sc)))

    if not cands:
        return float(anchor_sec)

    chosen_t, chosen_sc = max(cands, key=lambda x: x[1])
    gain = float(chosen_sc - float(base_score))
    if gain <= 0.0:
        return float(anchor_sec)

    # Require only a small gain for local moves; larger shifts need stronger evidence.
    shift_beats = abs(float(chosen_t) - float(anchor_sec)) / max(1e-9, float(beat_sec))
    min_gain = 0.006 + (0.004 * max(0.0, shift_beats - 0.50))
    if gain >= min_gain:
        return float(chosen_t)
    return float(anchor_sec)

def _first_drop_sec_from_drums(drums_path: str, bpm: int) -> Tuple[Optional[float], float, Dict[str, float]]:
    if np is None:
        print("[WARN] numpy not installed; skipping drop anchor stamping.")
        return None, 0.0, {}
    if not bpm or bpm <= 0:
        return None, 0.0, {}

    primary_sec: Optional[float] = None
    primary_conf: float = 0.0
    primary_meta: Dict[str, float] = {}

    # Primary detector: bar-aligned "true drop" model tuned for EDM.
    if detect_true_drop is not None:
        try:
            b_hint = float(bpm)
            b_min = max(120.0, b_hint - 14.0)
            b_max = min(160.0, b_hint + 14.0)
            if b_max <= b_min:
                b_min, b_max = 120.0, 160.0
            res = detect_true_drop(
                drums_path,
                bpm_hint=b_hint,
                bpm_min=b_min,
                bpm_max=b_max,
            )
            if res is not None and res.drop_time_sec is not None:
                dbg = res.debug or {}
                meta = {
                    "det_conf": float(res.confidence),
                    "stage1_score": float(_as_float(dbg.get("stage1_score")) or 0.0),
                    "struct_score": float(_as_float(dbg.get("struct_score")) or 0.0),
                    "valid": float(_as_float(dbg.get("valid")) or 0.0),
                    "sustain8": float(_as_float(dbg.get("sustain8")) or 0.0),
                    "fake": float(_as_float(dbg.get("fake")) or 0.0),
                    "downbeat_offset": float(_as_float(dbg.get("downbeat_offset")) or 0.0),
                    "tempo_bpm": float(_as_float(dbg.get("tempo_bpm")) or bpm),
                }
                primary_sec = float(res.drop_time_sec)
                primary_conf = float(res.confidence)
                primary_meta = meta
                # Strong structural hit: trust primary detector directly.
                if float(meta.get("valid", 0.0)) >= 0.5 and primary_conf >= 0.82:
                    corrected = _pull_anchor_earlier_if_late(
                        drums_path=drums_path,
                        bpm=bpm,
                        anchor_sec=float(primary_sec),
                        det_conf=float(primary_conf),
                    )
                    if corrected < float(primary_sec) - 1e-3:
                        primary_meta["late_pull_sec"] = float(primary_sec - corrected)
                        primary_sec = float(corrected)
                    primary_sec = _retime_anchor_local_bidirectional(
                        drums_path=drums_path,
                        bpm=bpm,
                        anchor_sec=float(primary_sec),
                        det_conf=float(primary_conf),
                    )
                    # Final micro-snap to local onset for strong detections to avoid slight early anchors.
                    yy, ss = _load_audio_mono_f32_ffmpeg(drums_path, sr=22050)
                    if yy is not None and "late_pull_sec" not in primary_meta:
                        onset_refined = _refine_anchor_to_onset(yy, ss, float(primary_sec), int(bpm))
                        if onset_refined is not None and abs(float(onset_refined) - float(primary_sec)) <= (0.35 * (60.0 / float(bpm))):
                            accept_refine = True
                            # Guard: avoid pushing late when energy is already active before anchor.
                            if float(onset_refined) > float(primary_sec) + 1e-3:
                                beat_sec_local = 60.0 / float(bpm)
                                a0 = max(0, int(round((float(primary_sec) - (0.50 * beat_sec_local)) * float(ss))))
                                a1 = max(a0 + 1, int(round(float(primary_sec) * float(ss))))
                                b0 = a1
                                b1 = min(len(yy), max(b0 + 1, int(round((float(primary_sec) + (0.50 * beat_sec_local)) * float(ss)))))
                                pre_e = float(np.mean(np.abs(yy[a0:a1]))) if a1 > a0 else 0.0
                                post_e = float(np.mean(np.abs(yy[b0:b1]))) if b1 > b0 else 0.0
                                if pre_e >= (0.20 * max(1e-9, post_e)):
                                    accept_refine = False
                            if accept_refine:
                                if float(onset_refined) > float(primary_sec) + 1e-3:
                                    primary_meta["onset_refine_sec"] = float(onset_refined - float(primary_sec))
                                primary_sec = float(onset_refined)
                    return primary_sec, primary_conf, primary_meta
        except Exception as e:
            print(f"[WARN] true-drop detector failed for {drums_path}: {e}")

    y, sr = _load_audio_mono_f32_ffmpeg(drums_path, sr=22050)
    if y is None:
        return None, 0.0, {}
    dur_sec = float(len(y)) / float(sr)
    if dur_sec < 12.0:
        return None, 0.0, {}

    hop_sec = 0.05
    win_sec = 0.10
    hop = max(1, int(round(hop_sec * sr)))
    win = max(1, int(round(win_sec * sr)))
    if len(y) <= win:
        return None, 0.0, {}

    n_frames = 1 + (len(y) - win) // hop
    env = np.empty(n_frames, dtype=np.float32)
    low_ratio = np.empty(n_frames, dtype=np.float32)
    broad_energy = np.empty(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)

    n_fft = 2048
    fft_win = np.hanning(n_fft).astype(np.float32)
    if sr > 0:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    else:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / 22050.0)
    low_mask = (freqs >= 20.0) & (freqs <= 150.0)
    broad_mask = (freqs >= 20.0) & (freqs <= min(8000.0, float(sr) * 0.5))
    prev_mag = None
    prev_env = 0.0
    for i in range(n_frames):
        seg = y[i * hop:i * hop + win]
        env_i = float(np.sqrt(np.mean(seg * seg) + 1e-12)) if seg.size else 0.0
        env[i] = env_i
        onset[i] = max(0.0, env_i - prev_env)
        prev_env = env_i

        if len(seg) < n_fft:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:len(seg)] = seg.astype(np.float32, copy=False)
            frame = padded
        else:
            frame = seg[:n_fft].astype(np.float32, copy=False)
        mag = np.abs(np.fft.rfft(frame * fft_win)).astype(np.float32, copy=False)
        power = mag * mag
        broad = float(np.sum(power[broad_mask])) if np.any(broad_mask) else float(np.sum(power))
        low = float(np.sum(power[low_mask])) if np.any(low_mask) else 0.0
        broad_energy[i] = broad
        low_ratio[i] = (low / (broad + 1e-12)) if broad > 0 else 0.0

        if prev_mag is not None:
            d = mag - prev_mag
            flux[i] = float(np.sum(d[d > 0.0]))
        prev_mag = mag

    smooth_short = max(3, int(round(0.35 / hop_sec)))
    smooth_mid = max(5, int(round(0.80 / hop_sec)))
    env = _smooth_same(env, smooth_mid)
    low_ratio = _smooth_same(low_ratio, smooth_mid)
    flux = _smooth_same(flux, smooth_short)
    onset = _smooth_same(onset, smooth_short)
    broad_energy = _smooth_same(broad_energy, smooth_mid)
    if env is None or len(env) < 20:
        return None, 0.0, {}

    beat_sec = 60.0 / float(bpm)
    beat_frames = max(1, int(round(beat_sec / hop_sec)))
    pre_frames = max(beat_frames * 8, int(round(4.0 / hop_sec)))          # ~2 bars
    post_frames = max(beat_frames * 16, int(round(8.0 / hop_sec)))        # ~4 bars
    sustain_frames = max(beat_frames * 24, int(round(12.0 / hop_sec)))    # ~6 bars
    impact_frames = max(beat_frames * 4, int(round(2.0 / hop_sec)))       # ~1 bar

    start_sec = max(8.0, 2.0 * beat_sec * BEATS_PER_BAR)
    end_sec = max(start_sec + 2.0, dur_sec - max(8.0, 2.0 * beat_sec * BEATS_PER_BAR))
    i_start = max(pre_frames + 2, int(round(start_sec / hop_sec)))
    i_end = min(len(env) - sustain_frames - 2, int(round(end_sec / hop_sec)))
    if i_end <= i_start:
        return None, 0.0, {}

    step = max(1, beat_frames // 2)  # half-beat candidate spacing
    cand_i: List[int] = []
    rms_jump: List[float] = []
    low_jump: List[float] = []
    flux_jump: List[float] = []
    onset_jump: List[float] = []
    sustain_gain: List[float] = []
    beat_consistency: List[float] = []
    fake_penalty: List[float] = []

    global_env_med = float(np.percentile(env, 50))

    for i in range(i_start, i_end + 1, step):
        a0, a1 = i - pre_frames, i
        b0, b1 = i, i + post_frames
        s0, s1 = i + impact_frames, i + sustain_frames
        p0, p1 = max(0, i - impact_frames), i
        if a0 < 0 or b1 > len(env) or s1 > len(env):
            continue

        pre_rms = float(np.mean(env[a0:a1]))
        post_rms = float(np.mean(env[b0:b1]))
        imp_rms = float(np.mean(env[b0:b0 + impact_frames]))
        sus_rms = float(np.mean(env[s0:s1]))

        pre_low = float(np.mean(low_ratio[a0:a1]))
        post_low = float(np.mean(low_ratio[b0:b1]))

        pre_flux = float(np.percentile(flux[p0:p1], 75)) if p1 > p0 else 0.0
        post_flux = float(np.percentile(flux[b0:b0 + impact_frames], 75))
        pre_on = float(np.percentile(onset[p0:p1], 75)) if p1 > p0 else 0.0
        post_on = float(np.percentile(onset[b0:b0 + impact_frames], 75))

        beat_seq = onset[b0:min(len(onset), b0 + beat_frames * 16)]
        beat_corr = _periodicity_score(beat_seq, beat_frames)

        # Penalize impact-only fake drops: big hit then energy collapse.
        fake = 0.0
        if imp_rms > 1e-6:
            fake = max(0.0, (imp_rms - sus_rms) / imp_rms)
        if sus_rms < global_env_med:
            fake += 0.2

        cand_i.append(i)
        rms_jump.append(post_rms - pre_rms)
        low_jump.append(post_low - pre_low)
        flux_jump.append(post_flux - pre_flux)
        onset_jump.append(post_on - pre_on)
        sustain_gain.append(sus_rms - pre_rms)
        beat_consistency.append(beat_corr)
        fake_penalty.append(fake)

    if not cand_i:
        return None, 0.0, {}

    rms_jump = np.asarray(rms_jump, dtype=np.float32)
    low_jump = np.asarray(low_jump, dtype=np.float32)
    flux_jump = np.asarray(flux_jump, dtype=np.float32)
    onset_jump = np.asarray(onset_jump, dtype=np.float32)
    sustain_gain = np.asarray(sustain_gain, dtype=np.float32)
    beat_consistency = np.asarray(beat_consistency, dtype=np.float32)
    fake_penalty = np.asarray(fake_penalty, dtype=np.float32)

    nrms = _norm_by_quantiles(rms_jump)
    nlow = _norm_by_quantiles(low_jump)
    nflux = _norm_by_quantiles(flux_jump)
    nonset = _norm_by_quantiles(onset_jump)
    nsustain = _norm_by_quantiles(sustain_gain)
    nbeat = _norm_by_quantiles(beat_consistency, q_lo=15.0, q_hi=85.0)
    nfake = _norm_by_quantiles(fake_penalty)

    score = (
        0.28 * nrms +
        0.24 * nlow +
        0.17 * nflux +
        0.11 * nonset +
        0.14 * nsustain +
        0.16 * nbeat -
        0.24 * nfake
    )
    score = np.clip(score, 0.0, 1.0)

    # Strong true-drop gate, then earliest valid candidate.
    mask = (
        (score >= 0.62) &
        (nrms >= 0.45) &
        (nlow >= 0.35) &
        (nsustain >= 0.35) &
        (nbeat >= 0.20)
    )
    if np.any(mask):
        idx = int(np.where(mask)[0][0])
    else:
        idx = int(np.argmax(score))

    sorted_scores = np.sort(score)
    top = float(sorted_scores[-1]) if len(sorted_scores) else 0.0
    second = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
    margin = max(0.0, top - second)
    chosen = float(score[idx])
    chosen_fake = float(nfake[idx]) if len(nfake) > idx else 0.0
    confidence = float(np.clip(
        0.55 * chosen +
        0.25 * top +
        0.20 * margin -
        0.15 * chosen_fake,
        0.0, 1.0
    ))

    coarse = float(cand_i[idx] * hop_sec)
    flux10, flux_hop = _spectral_flux(y, sr, hop_sec=0.01, n_fft=2048)
    phase = _estimate_beat_phase(flux10, flux_hop, beat_sec, coarse, dur_sec)
    snapped = _snap_to_beat_with_flux(coarse, bpm, flux10, flux_hop, phase)
    snapped = _refine_to_structural_drop(env, low_ratio, onset, hop_sec, beat_sec, snapped)
    snapped = _refine_anchor_to_onset(y, sr, snapped, bpm)
    if abs(snapped - coarse) > (4.00 * beat_sec):
        snapped = coarse
    meta = {
        "det_conf": float(confidence),
        "stage1_score": float(chosen),
        "struct_score": float((0.45 * chosen) + (0.30 * top) + (0.25 * margin)),
        "valid": float(1.0 if (np.any(mask) and bool(mask[idx])) else 0.0),
        "sustain8": float(nsustain[idx]) if len(nsustain) > idx else 0.0,
        "fake": float(nfake[idx]) if len(nfake) > idx else 0.0,
        "downbeat_offset": 0.0,
        "tempo_bpm": float(bpm),
    }
    # If a primary detector result exists, prefer it unless fallback is clearly better.
    if primary_sec is not None:
        p_valid = float(primary_meta.get("valid", 0.0))
        f_valid = float(meta.get("valid", 0.0))
        choose_fallback = False
        if f_valid >= 0.5 and p_valid < 0.5:
            choose_fallback = True
        elif p_valid < 0.5 and confidence >= max(0.68, primary_conf + 0.05):
            choose_fallback = True
        elif confidence >= (primary_conf + 0.20):
            choose_fallback = True
        if choose_fallback:
            meta["primary_sec"] = float(primary_sec)
            meta["primary_conf"] = float(primary_conf)
            meta["primary_valid"] = float(p_valid)
            tuned = _retime_anchor_local_bidirectional(
                drums_path=drums_path,
                bpm=bpm,
                anchor_sec=float(snapped),
                det_conf=float(confidence),
            )
            if abs(float(tuned) - float(snapped)) > 1e-3:
                meta["local_tune_sec"] = float(tuned - float(snapped))
            return tuned, confidence, meta
        return float(primary_sec), float(primary_conf), primary_meta

    tuned = _retime_anchor_local_bidirectional(
        drums_path=drums_path,
        bpm=bpm,
        anchor_sec=float(snapped),
        det_conf=float(confidence),
    )
    if abs(float(tuned) - float(snapped)) > 1e-3:
        meta["local_tune_sec"] = float(tuned - float(snapped))
    return tuned, confidence, meta

def _sec_to_beats(sec: float, bpm: int) -> float:
    return (float(sec) * float(bpm)) / 60.0

def _quantize_beats(beats: float, grid: float = WARP_GRID_BEATS) -> float:
    if grid <= 0:
        return float(beats)
    return round(float(beats) / grid) * grid

def _compute_phase_to_drop_bar_one(drop_sec: float, bpm: int) -> float:
    drop_beats = _sec_to_beats(drop_sec, bpm)
    return -float(drop_beats)

def _drop_anchor_offset_for_track(target_folder: str) -> float:
    track_name = os.path.basename(target_folder)
    if track_name in DROP_ANCHOR_OVERRIDES_SEC:
        return float(DROP_ANCHOR_OVERRIDES_SEC[track_name])
    return float(DROP_ANCHOR_OFFSET_SEC)

def _manual_drop_from_project_als(target_folder: str, bpm: Optional[int]) -> Optional[float]:
    """
    If user manually inserted a transient/warp marker in:
      <track folder>/CH1 Project/CH1.als
    use the nearest marker before BeatTime=0 on the drums clip as manual drop.
    """
    proj_als = os.path.join(target_folder, "CH1 Project", "CH1.als")
    if not os.path.exists(proj_als):
        return None
    try:
        with gzip.open(proj_als, "rb") as f:
            data = f.read()
        root = ET.parse(BytesIO(data)).getroot()
    except Exception:
        return None

    for clip in root.iter("AudioClip"):
        names: List[str] = []
        for node in clip.iter("Name"):
            v = node.get("Value")
            if v:
                names.append(v)
        for node in clip.iter("EffectiveName"):
            v = node.get("Value")
            if v:
                names.append(v)
        joined = " | ".join(names).lower()
        if "drums_" not in joined:
            continue

        wm = clip.find("WarpMarkers")
        if wm is None:
            continue
        pts: List[Tuple[float, float]] = []
        for m in wm.findall("WarpMarker"):
            try:
                sec = float(m.get("SecTime"))
                beat = float(m.get("BeatTime"))
            except Exception:
                continue
            pts.append((sec, beat))
        if len(pts) < 4:
            # likely auto-generated (0, drop, end) with no manual marker
            continue
        pts.sort(key=lambda x: x[0])

        zero_sec = None
        for sec, beat in pts:
            if abs(beat) <= 1e-4:
                zero_sec = sec
                break
        if zero_sec is None:
            continue

        prev = [(sec, beat) for sec, beat in pts if sec < (zero_sec - 1e-6)]
        if not prev:
            continue
        sec_prev, beat_prev = prev[-1]
        if beat_prev >= -1e-6:
            # manual pre-drop marker should sit before 1.1.1 (negative beat)
            continue

        # sanity window: marker should be within up to ~8 beats before 1.1.1
        if bpm and bpm > 0:
            max_back = 8.0 * (60.0 / float(bpm))
            if (zero_sec - sec_prev) > max_back:
                continue
        if (zero_sec - sec_prev) < 0.05:
            continue
        return float(sec_prev)
    return None

def _max_duration_seconds(track_names: Dict[str, Optional[str]]) -> Optional[float]:
    max_sec = 0.0
    for rel in track_names.values():
        if not rel:
            continue
        p = os.path.join(FLAC_FOLDER, rel)
        sec = get_duration_seconds(p)
        if sec and sec > max_sec:
            max_sec = sec
    return max_sec if max_sec > 0 else None

def _max_audible_end_seconds(track_names: Dict[str, Optional[str]]) -> Optional[float]:
    if np is None:
        return _max_duration_seconds(track_names)
    max_sec = 0.0
    for rel in track_names.values():
        if not rel:
            continue
        p = os.path.join(FLAC_FOLDER, rel)
        if not os.path.exists(p):
            continue
        y, sr = _load_audio_mono_f32_ffmpeg(p, sr=22050)
        if y is None:
            sec = get_duration_seconds(p)
        else:
            sec = _last_audible_sec(y, sr)
        if sec and sec > max_sec:
            max_sec = sec
    if max_sec > 0:
        return max_sec
    return _max_duration_seconds(track_names)

def _stamp_drop_anchor_warpmarkers(root: ET.Element,
                                   bpm: Optional[int],
                                   drop_sec: Optional[float],
                                   end_sec: Optional[float]) -> int:
    if not bpm or bpm <= 0 or drop_sec is None:
        return 0
    if end_sec is None or end_sec <= 0:
        return 0

    phi = _compute_phase_to_drop_bar_one(drop_sec, bpm)
    anchors = [0.0, float(drop_sec), float(end_sec)]
    points = sorted(set(round(max(0.0, p), 6) for p in anchors))
    if len(points) < 2:
        return 0

    total = 0
    for aclip in root.iter("AudioClip"):
        for wm in list(aclip.findall("WarpMarkers")):
            aclip.remove(wm)
        wm_parent = ET.Element("WarpMarkers")
        for i, sec in enumerate(points):
            beats = _sec_to_beats(sec, bpm) + phi
            wm_parent.append(ET.Element("WarpMarker", {
                "Id": str(i),
                "SecTime": f"{sec:.6f}",
                "BeatTime": f"{beats:.6f}",
            }))
        aclip.insert(0, wm_parent)
        total += len(points)

    for node in root.iter("IsWarped"):
        node.set("Value", "true")
    return total

def modify_als_file(input_path: Optional[str], target_folder: str, track_names: Dict[str, Optional[str]], bpm_value: Optional[int], force: bool=False) -> None:
    output_als = os.path.join(target_folder, "CH1.als")

    # Respect SKIP_EXISTING unless forcing
    if os.path.exists(output_als) and SKIP_EXISTING and not force:
        return

    # If a template is provided and it's not the same file, copy it; otherwise edit in place
    if input_path and os.path.abspath(input_path) != os.path.abspath(output_als):
        shutil.copy(input_path, output_als)
    elif not os.path.exists(output_als):
        print(f"[WARN] No template for BPM {bpm_value} and no existing CH1.als at {target_folder}; skipping.")
        return

    print(f"\n🎯 Creating/Updating ALS for {target_folder} (BPM {bpm_value})")
    if bool(DROP_STAGE2_ENABLE) and bool(DROP_STAGE2_VERBOSE):
        print(
            f"[WARP] Stage2 active: attack_nudge={bool(DROP_STAGE2_ATTACK_NUDGE_ENABLE)}, "
            f"max_shift_beats={float(DROP_STAGE2_MAX_SHIFT_BEATS):.2f}, "
            f"fixed_late={float(DROP_STAGE2_FIXED_LATE_SEC):.3f}s, "
            f"apply_db={bool(DROP_STAGE2_APPLY_ON_DB_ANCHOR)}, "
            f"apply_manual={bool(DROP_STAGE2_APPLY_ON_MANUAL_ANCHOR)}"
        )

    new_loop_end = None
    drop_sec = None
    drop_conf = 0.0
    drop_meta: Dict[str, float] = {}
    end_sec = _max_audible_end_seconds(track_names)
    phi_beats = 0.0
    drums_source_path = None
    used_db_anchor = False
    used_manual_anchor = False

    if track_names.get("drums") and bpm_value:
        flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
        drums_source_path = flac_path if os.path.exists(flac_path) else None
        new_loop_end = get_duration_in_beats(flac_path, bpm_value)
        if os.path.exists(flac_path):
            drop_sec, drop_conf, drop_meta = _detect_drop_anchor_sec(flac_path, bpm_value)
    else:
        drums_abs = _find_drums_stem_path(target_folder, track_names)
        drums_source_path = drums_abs if drums_abs and os.path.exists(drums_abs) else None
        if drums_abs and bpm_value:
            drop_sec, drop_conf, drop_meta = _detect_drop_anchor_sec(drums_abs, bpm_value)

    # Ground-truth lookup DB (built from manually-cued ALS projects).
    if bpm_value and drums_source_path:
        db_sec, db_conf, db_reason = _lookup_drop_from_db(drums_source_path, bpm_value)
        if db_sec is not None:
            drop_sec = float(db_sec)
            drop_conf = max(float(drop_conf), float(db_conf))
            used_db_anchor = True
            print(f"[WARP] Using drop DB anchor: {drop_sec:.3f}s ({db_reason})")

    # Manual override from Ableton project (if user marked the desired pre-1.1.1 transient).
    if bpm_value:
        manual_sec = _manual_drop_from_project_als(target_folder, bpm_value)
        if manual_sec is not None:
            drop_sec = float(manual_sec)
            drop_conf = max(float(drop_conf), 0.99)
            used_manual_anchor = True
            if drums_source_path:
                if _upsert_drop_db_label(drums_source_path, float(manual_sec), bpm_value, source="manual_ch1_project"):
                    print("[WARP] Learned manual marker into drop DB.")
            print(f"[WARP] Using manual drop marker from CH1 Project: {drop_sec:.3f}s")

    # Self-learning model on unseen tracks:
    # 1) waveform signature shift (preferred), 2) detector-feature kNN fallback.
    if bpm_value and drop_sec is not None and (not used_db_anchor) and (not used_manual_anchor):
        if float(_as_float(drop_meta.get("alsdrop_source")) or 0.0) >= 0.5:
            pass
        else:
            det_valid = float(drop_meta.get("valid", 0.0)) if drop_meta else 0.0
            # Do not override strong structural detections with statistical shifts.
            if drop_conf >= 0.88 and det_valid >= 0.5:
                pass
            else:
                beat_sec = 60.0 / float(bpm_value)
                max_shift_sec = 4.0 * BEATS_PER_BAR * beat_sec
                applied = False

                if drums_source_path:
                    sig_beats, sig_conf, sig_reason = _predict_drop_shift_beats_by_signature_model(drums_source_path, bpm_value, drop_sec)
                    if sig_beats is not None and abs(float(sig_beats)) >= 0.20:
                        shifted = max(0.0, float(drop_sec) + (float(sig_beats) * beat_sec))
                        if abs(shifted - float(drop_sec)) <= max_shift_sec:
                            print(f"[WARP] Self-learn waveform shift: {sig_beats:+.2f} beats ({sig_reason}).")
                            drop_sec = shifted
                            drop_conf = max(float(drop_conf), min(0.95, 0.60 + (0.30 * float(sig_conf))))
                            applied = True

                if (not applied) and drop_meta:
                    delta_beats, ml_conf, ml_reason = _predict_drop_delta_beats_from_model(drop_meta, bpm_value)
                    if delta_beats is not None and abs(float(delta_beats)) >= 0.20:
                        shifted = max(0.0, float(drop_sec) + (float(delta_beats) * beat_sec))
                        if abs(shifted - float(drop_sec)) <= max_shift_sec:
                            print(f"[WARP] Self-learn feature shift: {delta_beats:+.2f} beats ({ml_reason}).")
                            drop_sec = shifted
                            drop_conf = max(float(drop_conf), min(0.95, 0.55 + (0.35 * float(ml_conf))))

    if bpm_value and drop_sec is not None:
        if bool(DROP_STAGE2_ENABLE):
            allow_db = bool(DROP_STAGE2_APPLY_ON_DB_ANCHOR)
            allow_manual = bool(DROP_STAGE2_APPLY_ON_MANUAL_ANCHOR)
            blocked = (used_db_anchor and (not allow_db)) or (used_manual_anchor and (not allow_manual))
            if blocked:
                if bool(DROP_STAGE2_VERBOSE):
                    src = "db" if used_db_anchor else "manual"
                    print(f"[WARP] Stage2 fixed late skipped (source={src}).")
            else:
                fixed_late = max(0.0, float(DROP_STAGE2_FIXED_LATE_SEC))
                if fixed_late > 0.0 and bpm_value > 0:
                    beat_sec = 60.0 / float(max(1, bpm_value))
                    max_push = min(0.18 * beat_sec, 0.060)  # keep this strictly micro
                    push = min(fixed_late, max_push)
                    if push > 0.0:
                        drop_sec = max(0.0, float(drop_sec) + float(push))
                        drop_meta["stage2_fixed_late_sec"] = float(push)
                        if bool(DROP_STAGE2_VERBOSE):
                            print(f"[WARP] Stage2 fixed late shift: +{push:.3f}s")
        off = _drop_anchor_offset_for_track(target_folder)
        if off:
            drop_sec = max(0.0, float(drop_sec) + off)
        phi_beats = _compute_phase_to_drop_bar_one(drop_sec, bpm_value)

    # Read + parse
    with gzip.open(output_als, "rb") as f:
        als_data = f.read()
    tree = ET.parse(BytesIO(als_data))
    root = tree.getroot()

    # Mirror manual workflow:
    # 1) place a marker on drums drop transient
    # 2) set that marker to 1.1.1
    # 3) apply same phase to all clips so drums/inst/vocals launch in sync
    stamped = _stamp_drop_anchor_warpmarkers(root, bpm_value, drop_sec, end_sec)
    if stamped and drop_sec is not None:
        print(f"[WARP] Drop anchor set at {drop_sec:.3f}s (1.1.1), confidence={drop_conf:.2f}.")
        if drop_conf < 0.45:
            print("[WARP] Low-confidence drop; consider adjusting DROP_ANCHOR_OFFSET_SEC for this track.")

    # Update loop/time markers from real end marker in warped timeline.
    if end_sec and bpm_value:
        end_beats = _sec_to_beats(end_sec, bpm_value) + phi_beats
        end_beats_str = f"{end_beats:.6f}"
        loop_updates = 0
        out_updates = 0
        for clip in root.iter("AudioClip"):
            for loop_node in clip.iter("LoopEnd"):
                loop_node.set("Value", end_beats_str)
                loop_updates += 1
            for out_node in clip.iter("OutMarker"):
                out_node.set("Value", end_beats_str)
                out_updates += 1

        # Fallback in case template structure differs.
        if loop_updates == 0:
            for loop_node in root.iter("LoopEnd"):
                loop_node.set("Value", end_beats_str)
        if out_updates == 0:
            for out_node in root.iter("OutMarker"):
                out_node.set("Value", end_beats_str)
    elif new_loop_end:
        for elem in root.iter():
            if elem.tag in ("LoopEnd", "OutMarker"):
                elem.set("Value", new_loop_end)

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

            old_name = os.path.splitext(old)[0]
            new_name = safe_xml_value(os.path.splitext(os.path.basename(rel_path))[0])
            for tag in ["MemorizedFirstClipName","UserName","Name","EffectiveName"]:
                pattern = rf'(<{tag}\s+Value="){re.escape(old_name)}(")'
                als_str = re.sub(pattern, rf"\1{new_name}\2", als_str)

    with gzip.open(output_als, "wb") as f_out:
        f_out.write(als_str.encode("utf-8"))
    print(f"✅ ALS saved: {output_als}")

# ========= MOVES =========
def move_folder_to_target(stems_folder_abs: str, bpm: int, cam: str, energy: Optional[int], src_basename: Optional[str], allow_rename: bool) -> Tuple[str, bool, bool, bool]:
    track_name = os.path.basename(stems_folder_abs)
    desired_name = desired_track_folder_name(track_name, src_basename, allow_rename)
    bpm_folder = os.path.join(destination_folder, str(bpm))
    cam_folder = os.path.join(bpm_folder, cam)
    os.makedirs(cam_folder, exist_ok=True)

    dest_path = os.path.join(cam_folder, desired_name)
    moved = False
    if stems_folder_abs != dest_path:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(stems_folder_abs, dest_path)
        print(f"[MOVE] {track_name}  →  {cam_folder}/{desired_name}")
        moved = True

    # Rename (adds energy if missing)
    energy_added = rename_stems_in_place(dest_path, bpm, cam, energy)

    output_als = os.path.join(dest_path, "CH1.als")

    # Key change: when SKIP_EXISTING is False, force regeneration
    force_regen = (not SKIP_EXISTING)
    need_als = force_regen or (not os.path.exists(output_als)) or moved or energy_added

    if not need_als:
        return dest_path, moved, energy_added, False

    blank_als   = select_blank_als(bpm)
    track_names = collect_track_names_for_folder(dest_path)

    # If no template available, but an ALS exists, modify in place
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
            print(f"[SKIP] No source match for '{folder}' in {mp3_source_folder}")
            continue

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key":
            reason = []
            if bpm is None:
                reason.append("missing BPM")
            if cam == "Unknown Key":
                reason.append("missing/unknown key")
            print(f"[SKIP] {folder} → {', '.join(reason)} from {os.path.basename(src_audio)}")
            continue

        src_base = os.path.splitext(os.path.basename(src_audio))[0]
        move_folder_to_target(stems_abs, bpm, cam, energy, src_base, allow_rename=True)

def _snapshot_tracks(dest_folder: str):
    items = []
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
            continue  # silent if no match

        bpm, key, energy = get_bpm_key_energy(src_audio)
        cam = to_camelot(key or "")
        if bpm is None or cam == "Unknown Key":
            continue

        src_base = os.path.splitext(os.path.basename(src_audio))[0]
        _, did_move, added_energy, wrote_als = move_folder_to_target(track_abs, bpm, cam, energy, src_base, allow_rename=False)
        if did_move: moved += 1
        if wrote_als: regenerated += 1

    print(f"[INFO] Reconcile complete. Moved: {moved}, ALS updated: {regenerated}")

def undo_renamed_folders(idx_exact: Dict[str,str]) -> None:
    source_names = set(idx_exact.keys())
    swapped_map: Dict[str, str] = {}
    for name in source_names:
        swapped = swap_artist_title(name)
        if swapped and swapped != name:
            swapped_map[swapped] = name

    renamed = skipped = 0
    for cur_bpm, cur_cam, track_folder, track_abs in _snapshot_tracks(destination_folder):
        orig_name = swapped_map.get(nfc(track_folder))
        if not orig_name:
            continue
        dest_path = os.path.join(os.path.dirname(track_abs), orig_name)
        if os.path.exists(dest_path):
            print(f"[SKIP] Undo rename target exists: {dest_path}")
            skipped += 1
            continue
        os.rename(track_abs, dest_path)
        print(f"[UNDO] {track_folder}  →  {orig_name}")
        renamed += 1

    print(f"[INFO] Undo rename complete. Renamed: {renamed}, Skipped: {skipped}")

# ========= MAIN =========
if __name__ == "__main__":
    if input("Ready? Type 'yes' to begin: ").strip().lower() == "yes":
        index_exact = index_source_library(mp3_source_folder)     # exact base-name index (NFC)
        if ONE_TIME_UNDO_RENAME:
            undo_renamed_folders(index_exact)
        process_new_stems_from_toBeOrganized(index_exact)         # place new folders
        reconcile_existing_stems(index_exact)                     # fix existing folders (force-regens if SKIP_EXISTING=False)
        _write_alsdrop_review_queue_exports()
        print("🎵 Done.")
    else:
        print("Operation cancelled.")
