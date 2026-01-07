#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, re, csv, sys, datetime, gzip
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# =========================
# ========== CONFIG =======
# =========================
STEMS_DIR         = "/Users/alirahimlou/Desktop/MUSIC/STEMS"

# Final combined file + CSV live here (no per-track copies)
OUTPUT_DIR        = Path("/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/Set_Festival")

# Base ALS that contains CH1/CH2/CH3(/CH4) tracks
BASE_ALS          = Path("/Users/alirahimlou/myapps/trackOrganizer/alsFiles/CH1.als")

# Optional: inject into an existing ALS that already has tracks/scenes
INJECT_INTO_EXISTING = True
INJECT_ALS          = Path("/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26.als")

SET_LEN           = 90           # how many tracks to pick
START_BPM         = 90         # None = random start; or int like 124
BPM_STEP_MAX      = 4            # max BPM increase per transition
MAX_PER_ARTIST    = 2            # soft cap per artist

# If True, restrict the track pool to a BPM range (inclusive)
USE_BPM_RANGE     = True
BPM_RANGE_MIN     = 80
BPM_RANGE_MAX     = 150

# If True, ignore BPM/harmonic/energy settings and use all tracks in DATE_RANGE (by CH1.als mtime)
USE_NEWEST_TRACKS = True
# Required when USE_NEWEST_TRACKS=True. Format: "12-19-25--12-21-25" or "2025-12-19--2025-12-21"
DATE_RANGE = "12-31-25--1-7-26"

# strict → same key, ±1 (same letter), relative (A↔B)
# energy → strict + energy boost/drop (±2, same letter)
# pro → energy + perfect 4th/5th (±7, same letter) + modal shift (swap A/B and ±1)
# open → pro + tritone (±6, same letter)
# Harmonic strategy: 'strict' | 'energy' | 'pro' | 'open'
HARMONIC_MODE     = "pro"

# ignore
# • Doesn’t use energy at all when choosing the next track.
# • Transitions are guided only by BPM and harmonic rules.
# • Scoring applies no real penalty for energy differences.
# • Use when you don’t trust (or don’t care about) energy tags.

# ramp (default)
# • Energy is non-decreasing, and can rise at most ENERGY_STEP_MAX per step.
# • With ENERGY_STEP_MAX = 2, allowed jumps are: +0, +1, +2; drops are blocked.
# • Example allowed: 4 → 5 → 5 → 7, not allowed: 6 → 4.
# • Start-track bias: tries to start at lower energy (≤4) if available.
# • If no candidates fit, the code temporarily relaxes the energy rule to keep the set going.

# wave
# • Energy can go up or down each step, but the size of the move is limited to ±ENERGY_STEP_MAX.
# • With ENERGY_STEP_MAX = 2, allowed: 5 → 4/5/6/7; not allowed: 5 → 8 or 5 → 2.
# • Good for gentle ebb-and-flow without big whiplash jumps.

# match
# • Requires same energy for each transition (e.g., 5 → 5 → 5).
# • Very consistent feel; best when energy tags are reliable and you want a fixed intensity lane.
# • If no match exists at a step, the script will (as a safety) relax the constraint to avoid dead-ends.

# A few extra notes:
# ENERGY_STEP_MAX only matters for ramp and wave; higher values allow bigger changes.
# Missing energy on either track is treated as “okay” (the script doesn’t block the transition).
# When the algorithm finds no candidates under the current energy rule, it automatically re-runs the search with the energy rule relaxed so your set can continue

# Energy sequencing: 'ignore' | 'ramp' | 'wave' | 'match'
ENERGY_STRATEGY   = "wave"
ENERGY_STEP_MAX   = 2

RANDOM_SEED       = None         # None = different every run; or int like 42
ENABLE_DEDUPING   = True

# If a file with the same name exists, append a timestamp to avoid overwrite
AVOID_OVERWRITE   = True

# =========================
# ========== SET ==========
# =========================
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

CAMELOT_RE = re.compile(r'^(\d{1,2})([ABab])$')
# stems like: drums_80_2A_5-Artist - Title.flac (role may be drums/inst/vocals)
STEM_RE    = re.compile(r'^(drums|inst|vocals)_(\d+)_(\d+[ABab])_(\d+)-', re.IGNORECASE)
SCENE_SORT_RE = re.compile(r'^\s*(\d{2,3})\s*[_-]\s*([0-9]{1,2}[ABab])(?:[_-](\d{1,2}))?')

def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\u2010-\u2015]', '-', s)
    s = re.sub(r'[_\(\)\[\]\{\}\-]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s\.&/+]', '', s)
    return s.strip()

def norm_title_from_folder(folder: str) -> str:
    base = Path(folder).name
    base = re.sub(r'\b(vip|edit|remix|mix|master|version|alt|vip mix)\b', '', base, flags=re.I)
    base = re.sub(r'\b(feat|ft|with)\b.*', '', base, flags=re.I)
    base = re.sub(r'\b\d{2}\b', '', base)  # drop "01 " etc
    return norm_text(base)

def guess_artist(folder: str) -> str:
    base = Path(folder).name
    cand = re.split(r'\s*-\s*', base, maxsplit=1)[0]
    cand = re.split(r',', cand)[0]
    return norm_text(cand)

def key_ok(key: str) -> bool:
    return bool(CAMELOT_RE.fullmatch(key.strip()))

def wrap12(x: int) -> int:
    while x < 1: x += 12
    while x > 12: x -= 12
    return x

def allowed_keys(key: str, mode: str) -> List[str]:
    m = CAMELOT_RE.fullmatch(key.strip())
    if not m: return []
    n = int(m.group(1)); L = m.group(2).upper()
    other = 'B' if L == 'A' else 'A'
    buckets: List[Set[str]] = []
    buckets.append({f"{n}{L}"})  # same
    buckets.append({f"{n}{other}"})  # relative
    buckets.append({f"{wrap12(n+1)}{L}", f"{wrap12(n-1)}{L}"})  # ±1
    if mode in ("energy", "pro", "open"):
        buckets.append({f"{wrap12(n+2)}{L}", f"{wrap12(n-2)}{L}"})  # ±2
    if mode in ("pro", "open"):
        buckets.append({f"{wrap12(n+1)}{'B' if L=='A' else 'A'}", f"{wrap12(n-1)}{'B' if L=='A' else 'A'}"})  # modal ±1
        buckets.append({f"{wrap12(n+7)}{L}", f"{wrap12(n-7)}{L}"})  # 4th/5th
    if mode == "open":
        buckets.append({f"{wrap12(n+6)}{L}", f"{wrap12(n-6)}{L}"})  # tritone
    out, seen = [], set()
    for b in buckets:
        for k in b:
            if k not in seen:
                seen.add(k); out.append(k)
    return out

def parse_energy_from_folder(folder: str) -> Optional[int]:
    prefer = {"drums": 0, "inst": 1, "vocals": 2}
    best: Tuple[int, Optional[int]] = (999, None)
    try:
        for name in os.listdir(folder):
            m = STEM_RE.match(name)
            if not m: continue
            role, _bpm, _key, energy_str = m.groups()
            rank = prefer.get(role.lower(), 99)
            energy = int(energy_str)
            if rank < best[0]:
                best = (rank, energy)
    except FileNotFoundError:
        return None
    return best[1]

def collect_tracks(stems_dir: str) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    for bpm_name in os.listdir(stems_dir):
        if not bpm_name.isdigit(): continue
        bpm = int(bpm_name)
        bpm_dir = os.path.join(stems_dir, bpm_name)
        if not os.path.isdir(bpm_dir): continue
        for key_name in os.listdir(bpm_dir):
            key_dir = os.path.join(bpm_dir, key_name)
            if not os.path.isdir(key_dir) or not key_ok(key_name): continue
            for root, _, files in os.walk(key_dir):
                if "CH1.als" in files:
                    track_folder = os.path.basename(root)
                    energy = parse_energy_from_folder(root)
                    src_path = os.path.join(root, "CH1.als")
                    try:
                        added_ts = os.path.getmtime(src_path)
                    except FileNotFoundError:
                        added_ts = 0.0
                    tracks.append({
                        "bpm": bpm,
                        "key": key_name.upper(),
                        "energy": energy,           # may be None
                        "folder": track_folder,
                        "src": src_path,
                        "artist": guess_artist(track_folder),
                        "title_norm": norm_title_from_folder(track_folder),
                        "added_ts": added_ts,
                    })
    def sort_key(t):
        m = CAMELOT_RE.fullmatch(t["key"])
        n = int(m.group(1)); L = 0 if m.group(2).upper()=="A" else 1
        energy_sort = t["energy"] if t["energy"] is not None else 999
        return (t["bpm"], n, L, energy_sort, norm_text(t["folder"]))
    tracks.sort(key=sort_key)
    return tracks

def sort_by_bpm_key_energy(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(t):
        m = CAMELOT_RE.fullmatch(t["key"])
        n = int(m.group(1)) if m else 99
        L = 0 if (m and m.group(2).upper() == "A") else 1
        energy_sort = t["energy"] if t["energy"] is not None else 999
        return (t["bpm"], n, L, energy_sort, norm_text(t["folder"]))
    return sorted(tracks, key=sort_key)

def dedupe(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_DEDUPING: return tracks
    tracks = sorted(tracks, key=lambda t: t.get("added_ts", 0.0), reverse=True)
    seen, out = set(), []
    for t in tracks:
        k = (t["bpm"], t["key"], t["title_norm"])
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

def filter_newest_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_NEWEST_TRACKS:
        return tracks
    if not DATE_RANGE:
        print("[ERROR] USE_NEWEST_TRACKS=True requires DATE_RANGE (e.g. \"12-19-25--12-21-25\").")
        sys.exit(1)
    rng = parse_date_range(DATE_RANGE)
    if not rng:
        print(f"[ERROR] DATE_RANGE not understood: {DATE_RANGE}")
        sys.exit(1)
    start_ts, end_ts = rng
    return [t for t in tracks if start_ts <= t.get("added_ts", 0.0) <= end_ts]

def parse_date_range(spec: str) -> Optional[Tuple[float, float]]:
    if not spec: return None
    if "--" not in spec: return None
    a, b = [s.strip() for s in spec.split("--", 1)]
    fmts = ["%Y-%m-%d", "%m-%d-%y"]
    def parse_one(s: str) -> Optional[datetime.datetime]:
        for f in fmts:
            try:
                return datetime.datetime.strptime(s, f)
            except ValueError:
                continue
        return None
    da = parse_one(a); db = parse_one(b)
    if da is None or db is None: return None
    if da > db: da, db = db, da
    start = da.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end = db.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
    return (start, end)

def filter_by_bpm_range(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_BPM_RANGE: return tracks
    lo = min(BPM_RANGE_MIN, BPM_RANGE_MAX)
    hi = max(BPM_RANGE_MIN, BPM_RANGE_MAX)
    return [t for t in tracks if lo <= t.get("bpm", 0) <= hi]

def build_indexes(tracks: List[Dict[str, Any]]):
    by_bpm: Dict[int, List[Dict[str, Any]]] = {}
    for t in tracks:
        by_bpm.setdefault(t["bpm"], []).append(t)
    return by_bpm

def choose_start(tracks: List[Dict[str, Any]], by_bpm: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not tracks: raise RuntimeError("No tracks found.")
    if ENERGY_STRATEGY == "ramp":
        low_energy = [t for t in tracks if (t.get("energy") is not None and t["energy"] <= 4)]
        if low_energy: return random.choice(low_energy)
    if START_BPM is None: return random.choice(tracks)
    candidate_bpms = sorted(by_bpm.keys(), key=lambda x: (abs(x-START_BPM), x))
    for b in candidate_bpms:
        bucket = by_bpm.get(b, [])
        if bucket: return random.choice(bucket)
    return random.choice(tracks)

def energy_ok(curr_e: Optional[int], next_e: Optional[int], strategy: str) -> bool:
    if strategy == "ignore": return True
    if curr_e is None or next_e is None: return True
    delta = next_e - curr_e
    if strategy == "ramp":  return (delta >= 0) and (delta <= ENERGY_STEP_MAX)
    if strategy == "wave":  return abs(delta) <= ENERGY_STEP_MAX
    if strategy == "match": return next_e == curr_e
    return True

def pick_next(curr: Dict[str, Any],
              by_bpm: Dict[int, List[Dict[str, Any]]],
              used_srcs: Set[str],
              artist_counts: Dict[str, int]) -> Optional[Dict[str, Any]]:

    curr_bpm = curr["bpm"]; curr_key = curr["key"]; curr_energy = curr.get("energy")

    key_ranked = allowed_keys(curr_key, HARMONIC_MODE)
    allowed_set = set(key_ranked)
    next_bpms = list(range(curr_bpm, curr_bpm + BPM_STEP_MAX + 1))

    def gather(relax_energy: bool = False) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        for b in next_bpms:
            for t in by_bpm.get(b, []):
                if t["src"] in used_srcs: continue
                if t["key"] not in allowed_set: continue
                if artist_counts.get(t["artist"], 0) >= MAX_PER_ARTIST: continue
                if not relax_energy and not energy_ok(curr_energy, t.get("energy"), ENERGY_STRATEGY): continue
                cands.append(t)
        if not cands:
            for b in range(max(70, curr_bpm-2), curr_bpm):
                for t in by_bpm.get(b, []):
                    if t["src"] in used_srcs: continue
                    if t["key"] not in allowed_set: continue
                    if artist_counts.get(t["artist"], 0) >= MAX_PER_ARTIST: continue
                    if not relax_energy and not energy_ok(curr_energy, t.get("energy"), ENERGY_STRATEGY): continue
                    cands.append(t)
        return cands

    candidates = gather(False) or gather(True)
    if not candidates: return None

    key_priority = {k:i for i,k in enumerate(key_ranked)}
    def score(t):
        bpm_jump = abs(t["bpm"] - curr_bpm)
        kp = key_priority.get(t["key"], 999)
        e = t.get("energy")
        if curr_energy is None or e is None:
            e_penalty = 1; e_diff = 9
        else:
            e_diff = abs(e - curr_energy)
            if ENERGY_STRATEGY == "ramp":
                e_penalty = (0 if e >= curr_energy else 3) + max(0, e - curr_energy - ENERGY_STEP_MAX)
            elif ENERGY_STRATEGY == "match":
                e_penalty = 0 if e == curr_energy else (1 if e_diff == 1 else 3)
            else:
                e_penalty = e_diff
        return (bpm_jump, e_penalty, kp, e_diff, norm_text(t["folder"]))
    candidates.sort(key=score)
    return random.choice(candidates[:5])

def build_set(tracks: List[Dict[str, Any]], set_len: int = SET_LEN) -> List[Dict[str, Any]]:
    by_bpm = build_indexes(tracks)
    current = choose_start(tracks, by_bpm)
    chosen = [current]
    used = {current["src"]}
    artist_counts = {current["artist"]: 1}
    while len(chosen) < set_len:
        nxt = pick_next(chosen[-1], by_bpm, used, artist_counts)
        if not nxt: break
        chosen.append(nxt)
        used.add(nxt["src"])
        artist_counts[nxt["artist"]] = artist_counts.get(nxt["artist"], 0) + 1
    return chosen

# =========================
# ===== ALS COMBINE =======
# =========================
def is_gz(b: bytes) -> bool:
    return len(b) >= 2 and b[:2] == b"\x1f\x8b"

def read_als_text(p: Path) -> str:
    d = p.read_bytes()
    return (gzip.decompress(d) if is_gz(d) else d).decode("utf-8", errors="replace")

def write_als_gz(p: Path, root: ET.Element):
    buf = BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True, method="xml")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(buf.getvalue()))
    # quick sanity: round-trip parse
    ET.fromstring(gzip.decompress(p.read_bytes()))

def deep_copy(e: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(e, encoding="utf-8"))

def liveset(root):
    if root.tag == "LiveSet": return root
    if root.tag == "Ableton": return root.find("LiveSet")
    return root.find("LiveSet")

def scenes_node(root):
    sc = root.find(".//Scenes")
    if sc is None:
        sc = root.find(".//SceneList")
    if sc is None:
        ls = liveset(root)
        if ls is None:
            ls = ET.SubElement(root, "LiveSet")
        sc = ET.SubElement(ls, "Scenes")
    return sc

def tracks_parent(root): 
    return root.find(".//Tracks")

def name_value(track: ET.Element) -> str:
    nm = track.find("./Name")
    if nm is None: return ""
    for tag in ("UserName","EffectiveName","Name"):
        el = nm.find(tag) if tag != "Name" else nm
        if el is not None and el.attrib.get("Value"): return el.attrib["Value"]
    return ""

def track_label(track: ET.Element) -> str:
    return name_value(track)

def scene_label(scene: ET.Element) -> str:
    nm = scene.find("./Name")
    if nm is None: return ""
    return nm.attrib.get("Value", "")

def ch_index_from_label(name: str):
    m = re.search(r'\bch(?:annel)?\s*([1-4])\b', name, flags=re.IGNORECASE)
    if m: return int(m.group(1))
    if re.fullmatch(r'CH[1-4]', name.strip(), flags=re.IGNORECASE): return int(name.strip()[-1])
    return None

def ch_tracks_map(root):
    tp = tracks_parent(root)
    if tp is None:
        return {}
    mapping = {}
    for t in list(tp):
        idx = ch_index_from_label(track_label(t))
        if idx in (1,2,3,4) and idx not in mapping:
            mapping[idx] = t
    return mapping

def role_index_from_name(name: str):
    n = name.lower()
    if "drum" in n: return 1
    if any(k in n for k in ("inst","instrument","synth","music")): return 2
    if any(k in n for k in ("voc","vox","vocal")): return 3
    return None

def src_role_map(root):
    tp = tracks_parent(root)
    if tp is None:
        return {}
    mapping = {}
    for t in list(tp):
        idx = role_index_from_name(track_label(t))
        if idx and idx not in mapping: mapping[idx] = t
    return mapping

def src_order_map(root):
    tp = tracks_parent(root)
    if tp is None:
        return {}
    tracks = [t for t in list(tp) if t.tag in ("AudioTrack","MidiTrack","GroupTrack")]
    mapping = {}
    for i, t in enumerate(tracks, start=1):
        if i > 4:
            break
        mapping[i] = t
    return mapping

def main_csl(track): 
    return track.find(".//DeviceChain/MainSequencer/ClipSlotList")

def freeze_csl(track):
    fr = track.find(".//FreezeSequencer")
    return fr.find("./ClipSlotList") if fr is not None else None

def slot_has_any_clip(slot: ET.Element) -> bool:
    for el in slot.iter():
        if el.tag != "ClipSlot" and el.tag.endswith("Clip"):
            return True
    return False

def first_filled_slot(track: ET.Element):
    csl = main_csl(track)
    if csl is None: return None
    for sl in csl.findall("./ClipSlot"):
        if slot_has_any_clip(sl): return sl
    return None

def remove_all_clips(elem: ET.Element):
    parent = {}
    stack = [elem]
    while stack:
        p = stack.pop()
        for c in list(p):
            parent[c] = p
            stack.append(c)
    for n in [n for n in parent if (n.tag != "ClipSlot" and n.tag.endswith("Clip"))]:
        parent[n].remove(n)

def next_attr_id(parent: ET.Element) -> int:
    mx = -1
    for child in list(parent):
        cur = child.attrib.get("Id")
        if cur and cur.isdigit(): mx = max(mx, int(cur))
    return mx + 1

def camelot_key_tuple(key: str) -> Optional[Tuple[int, int]]:
    m = CAMELOT_RE.fullmatch(key.strip()) if key else None
    if not m: return None
    n = int(m.group(1)); L = 0 if m.group(2).upper() == "A" else 1
    return (n, L)

def scene_sort_key_from_name(name: str) -> Optional[Tuple[int, int, int, int, str]]:
    m = SCENE_SORT_RE.match(name or "")
    if not m: return None
    bpm = int(m.group(1))
    key = m.group(2).upper()
    key_tuple = camelot_key_tuple(key)
    if not key_tuple: return None
    energy = int(m.group(3)) if (m.group(3) and m.group(3).isdigit()) else 999
    return (bpm, key_tuple[0], key_tuple[1], energy, norm_text(name))

def clip_name_from_slot(slot: ET.Element) -> str:
    for el in slot.iter():
        if el.tag != "ClipSlot" and el.tag.endswith("Clip"):
            nm = el.find("./Name")
            if nm is not None and nm.attrib.get("Value"):
                return nm.attrib["Value"]
    return ""

def scene_sort_key_from_clipname(name: str) -> Optional[Tuple[int, int, int, int, str]]:
    m = STEM_RE.match(name or "")
    if not m: return None
    _role, bpm_str, key_str, energy_str = m.groups()
    key_tuple = camelot_key_tuple(key_str.upper())
    if not key_tuple:
        return None
    bpm = int(bpm_str)
    energy = int(energy_str) if energy_str.isdigit() else 999
    return (bpm, key_tuple[0], key_tuple[1], energy, norm_text(name))

def track_slot_list(track: ET.Element) -> Optional[List[ET.Element]]:
    csl = main_csl(track)
    if csl is None:
        return None
    return list(csl.findall("./ClipSlot"))

def source_clip_names(src_path: Path) -> Dict[int, str]:
    try:
        sroot = ET.fromstring(read_als_text(src_path))
    except Exception:
        return {}
    src_map = ch_tracks_map(sroot) or src_role_map(sroot) or src_order_map(sroot)
    out: Dict[int, str] = {}
    for ch_idx, t in src_map.items():
        slot = first_filled_slot(t)
        if slot is None:
            continue
        nm = clip_name_from_slot(slot)
        if nm:
            out[ch_idx] = nm
    return out

def existing_clip_names(ch_map: dict[int, ET.Element], chs=(1,2,3)) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}
    for ch in chs:
        t = ch_map.get(ch)
        if t is None:
            continue
        slots = track_slot_list(t)
        if not slots:
            continue
        names = set()
        for sl in slots:
            nm = clip_name_from_slot(sl)
            if nm:
                names.add(norm_text(nm))
        out[ch] = names
    return out

def dedupe_existing_clips(ch_map: dict[int, ET.Element], chs=(1,2,3)):
    for ch in chs:
        t = ch_map.get(ch)
        if t is None:
            continue
        slots = track_slot_list(t)
        if not slots:
            continue
        seen = set()
        for sl in slots:
            nm = clip_name_from_slot(sl)
            if not nm:
                continue
            key = norm_text(nm)
            if key in seen:
                remove_all_clips(sl)
            else:
                seen.add(key)

def reorder_scene_rows(master_root: ET.Element, reference_track: Optional[ET.Element]):
    sc = scenes_node(master_root)
    scenes = list(sc.findall("./Scene"))
    if not scenes or reference_track is None:
        return
    slots = track_slot_list(reference_track)
    if not slots or len(slots) != len(scenes):
        return

    keys = []
    for i, sl in enumerate(slots):
        k = scene_sort_key_from_clipname(clip_name_from_slot(sl))
        keys.append((i, k))

    order = sorted(
        range(len(scenes)),
        key=lambda i: (0, keys[i][1]) if keys[i][1] is not None else (1, i)
    )
    if order == list(range(len(scenes))):
        return

    # Reorder Scenes
    sc[:] = [scenes[i] for i in order]

    # Reorder clip slots for all main tracks that match scene count
    tp = tracks_parent(master_root)
    if tp is None:
        return
    for t in list(tp):
        if t.tag not in ("AudioTrack", "MidiTrack", "GroupTrack"):
            continue
        csl = main_csl(t)
        if csl is None:
            continue
        slots = list(csl.findall("./ClipSlot"))
        if len(slots) != len(scenes):
            continue
        csl[:] = [slots[i] for i in order]

def scene_sort_key_from_track(t: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    key_tuple = camelot_key_tuple(t["key"]) or (99, 9)
    energy = t.get("energy") if t.get("energy") is not None else 999
    return (t["bpm"], key_tuple[0], key_tuple[1], energy, norm_text(t["folder"]))

def find_insert_index(sc: ET.Element, reference_track: Optional[ET.Element], new_key: Tuple[int, int, int, int, str]) -> int:
    scenes = list(sc.findall("./Scene"))
    parsed = []
    if reference_track is not None:
        csl = main_csl(reference_track)
        if csl is not None:
            slots = list(csl.findall("./ClipSlot"))
            for i, sl in enumerate(slots[:len(scenes)]):
                k = scene_sort_key_from_clipname(clip_name_from_slot(sl))
                if k is not None:
                    parsed.append((i, k))
    if not parsed:
        parsed = [(i, scene_sort_key_from_name(scene_label(s))) for i, s in enumerate(scenes)]
        parsed = [(i, k) for i, k in parsed if k is not None]
    if not parsed:
        return len(scenes)
    last_le = None
    for i, k in parsed:
        if k <= new_key:
            last_le = i
    if last_le is None:
        return parsed[0][0]
    return last_le + 1

def make_blank_slot(mcsl: ET.Element) -> ET.Element:
    prev = list(mcsl.findall("./ClipSlot"))
    if prev:
        slot = deep_copy(prev[-1]); remove_all_clips(slot)
    else:
        slot = ET.Element("ClipSlot")
    slot.attrib["Id"] = str(next_attr_id(mcsl))
    return slot

def insert_scene(master_root: ET.Element,
                 all_tracks: List[ET.Element],
                 ch_map: dict[int, ET.Element],
                 src_path: Path,
                 scene_name: str,
                 insert_idx: int):
    sc = scenes_node(master_root)

    # New Scene node (attribute Id only)
    scene_id = next_attr_id(sc)
    template_scene = sc.find("./Scene")
    if template_scene is not None:
        new_scene = deep_copy(template_scene)
        new_scene.attrib["Id"] = str(scene_id)
        nm = new_scene.find("./Name")
        if nm is None:
            nm = ET.SubElement(new_scene, "Name")
        nm.attrib["Value"] = scene_name
    else:
        new_scene = ET.Element("Scene"); new_scene.attrib["Id"] = str(scene_id)
        ET.SubElement(new_scene, "Name").attrib["Value"] = scene_name
    sc.insert(insert_idx, new_scene)

    # Load source ALS and map source tracks
    try:
        sroot = ET.fromstring(read_als_text(src_path))
    except Exception:
        sroot = None

    src_map = ch_tracks_map(sroot) if sroot is not None else {}
    if sroot is not None and not src_map:
        src_map = src_role_map(sroot)
    if sroot is not None and not src_map:
        src_map = src_order_map(sroot)

    ch_tracks = set(ch_map.values())
    for dest_track in all_tracks:
        if dest_track.tag == "ReturnTrack":
            continue
        mcsl = main_csl(dest_track)
        if mcsl is None:
            dc = dest_track.find("./DeviceChain")
            if dc is None:
                dc = ET.SubElement(dest_track, "DeviceChain")
            ms = dc.find("./MainSequencer")
            if ms is None:
                ms = ET.SubElement(dc, "MainSequencer")
            mcsl = ET.SubElement(ms, "ClipSlotList")

        new_slot = None
        if dest_track in ch_tracks:
            ch_idx = next((i for i, t in ch_map.items() if t is dest_track), None)
            src_track = src_map.get(ch_idx) if ch_idx is not None else None
            if src_track is not None:
                src_slot = first_filled_slot(src_track)
                if src_slot is not None:
                    new_slot = deep_copy(src_slot)

        if new_slot is None:
            new_slot = make_blank_slot(mcsl)
        else:
            new_slot.attrib["Id"] = str(next_attr_id(mcsl))

        slots = list(mcsl.findall("./ClipSlot"))
        if insert_idx >= len(slots):
            mcsl.append(new_slot)
        else:
            mcsl.insert(insert_idx, new_slot)

        fcsl = freeze_csl(dest_track)
        if fcsl is not None:
            fnew = make_blank_slot(fcsl)
            fslots = list(fcsl.findall("./ClipSlot"))
            if insert_idx >= len(fslots):
                fcsl.append(fnew)
            else:
                fcsl.insert(insert_idx, fnew)

def unique_save_path(directory: Path, base_name: str) -> Path:
    """
    If base_name exists in directory and AVOID_OVERWRITE=True, append _YYYYmmdd_HHMMSS
    """
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / base_name
    if not AVOID_OVERWRITE or not p.exists():
        return p
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem, suf = os.path.splitext(base_name)
    return directory / f"{stem}_{ts}{suf}"

def combine_in_order(chosen_tracks: List[Dict[str, Any]], base_als: Path, out_file: Path, insert_into_existing: bool = False):
    if not base_als.exists():
        print(f"[ERROR] Missing BASE_ALS: {base_als}")
        sys.exit(1)

    master_root = ET.fromstring(read_als_text(base_als))

    tp = tracks_parent(master_root)
    if tp is None:
        print("[ERROR] Base has no <Tracks>"); sys.exit(1)
    if not insert_into_existing:
        # Keep only the FIRST CH tracks from base (prevents horizontal dupes)
        first_for = {}
        for t in list(tp):
            idx = ch_index_from_label(track_label(t))
            if idx and idx not in first_for: first_for[idx] = t
        keep = set(first_for.values())
        for t in list(tp):
            if t not in keep: tp.remove(t)

    ch_map = ch_tracks_map(master_root)
    if not ch_map:
        print("[ERROR] No CH tracks (CH1..CH4) in base"); sys.exit(1)

    all_tracks = [t for t in list(tp) if t.tag in ("AudioTrack","MidiTrack","GroupTrack")]
    reference_track = ch_map.get(1) or ch_map.get(2) or ch_map.get(3) or ch_map.get(4)
    dedupe_existing_clips(ch_map, chs=(1,2,3))
    reorder_scene_rows(master_root, reference_track)
    existing_names = existing_clip_names(ch_map, chs=(1,2,3))
    print(f"[INFO] Inserting {len(chosen_tracks)} scene(s) into template...")
    for t in chosen_tracks:
        # Nice scene label: "BPM_KEY[_ENERGY]-TrackFolder"
        energy = t.get("energy")
        if energy is not None:
            scene_name = f"{t['bpm']}_{t['key']}_{energy}-{t['folder']}"
        else:
            scene_name = f"{t['bpm']}_{t['key']}-{t['folder']}"
        src_names = source_clip_names(Path(t["src"]))
        dup = False
        for ch in (1,2,3):
            nm = src_names.get(ch)
            if nm and norm_text(nm) in existing_names.get(ch, set()):
                dup = True
                break
        if dup:
            print(f"[SKIP] Duplicate found, skipping: {scene_name}")
            continue
        insert_idx = find_insert_index(scenes_node(master_root), reference_track, scene_sort_key_from_track(t))
        insert_scene(master_root, all_tracks, ch_map, Path(t["src"]), scene_name, insert_idx)
        for ch in (1,2,3):
            nm = src_names.get(ch)
            if nm:
                existing_names.setdefault(ch, set()).add(norm_text(nm))

    write_als_gz(out_file, master_root)
    print(f"[DONE] Wrote combined ALS → {out_file}")

# =========================
# ========== MAIN =========
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tracks = collect_tracks(STEMS_DIR)
    if not tracks:
        print("No CH1.als files found. Check /BPM/Key/Track/CH1.als")
        sys.exit(1)

    if USE_NEWEST_TRACKS:
        tracks = filter_newest_tracks(tracks)
        if not tracks:
            print("No tracks found in DATE_RANGE.")
            sys.exit(1)
        set_tracks = sort_by_bpm_key_energy(tracks)
    else:
        tracks = dedupe(tracks)
        tracks = filter_by_bpm_range(tracks)
        set_tracks = build_set(tracks)

    if (not USE_NEWEST_TRACKS) and len(set_tracks) < 2:
        print("Could not build a sequence with current constraints.\n"
              f"- HARMONIC_MODE: {HARMONIC_MODE}\n"
              f"- ENERGY_STRATEGY: {ENERGY_STRATEGY} (ENERGY_STEP_MAX={ENERGY_STEP_MAX})\n"
              "- Try increasing BPM_STEP_MAX / MAX_PER_ARTIST or expanding the library.")
        sys.exit(2)

    start_bpm = set_tracks[0]["bpm"]
    end_bpm   = set_tracks[-1]["bpm"]

    # Final ALS path
    if INJECT_INTO_EXISTING:
        final_als_path = INJECT_ALS
    else:
        final_als_name = f"{start_bpm}-{end_bpm}_Combined_CH_Scenes.als"
        final_als_path = unique_save_path(OUTPUT_DIR, final_als_name)

    # CSV log next to the ALS, same stem
    csv_path = final_als_path.with_suffix(".csv")

    # Write compact playlist log (no per-track ALS copies created)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["#", "BPM", "Key", "Energy", "TrackFolder", "SourcePath"])
        for i, t in enumerate(set_tracks, start=1):
            w.writerow([i, t["bpm"], t["key"], (t.get("energy") if t.get("energy") is not None else ""),
                        t["folder"], t["src"]])
    print(f"[LOG] Wrote playlist CSV → {csv_path}")

    # Build the combined ALS directly from original CH1.als sources
    base_als = INJECT_ALS if INJECT_INTO_EXISTING else BASE_ALS
    combine_in_order(set_tracks, base_als, final_als_path, insert_into_existing=INJECT_INTO_EXISTING)

    print("\n✅ Done.")
    print(f"   Final ALS : {final_als_path}")
    print(f"   Playlist  : {csv_path}")
    print("   (No per-track ALS copies were created.)")

if __name__ == "__main__":
    main()
