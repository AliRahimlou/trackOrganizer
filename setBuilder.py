#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, random, re, csv, sys, datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# ---------- CONFIG ----------
STEMS_DIR        = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
OUTPUT_ROOT_DIR  = "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/Set_Festival"  # parent folder for per-run subfolders

SET_LEN          = 85                 # how many tracks to pick
START_BPM        = None               # None = random start; or int like 124
BPM_STEP_MAX     = 4                  # max BPM increase per transition
MAX_PER_ARTIST   = 2                  # soft cap per artist

# strict → same key, ±1 (same letter), relative (A↔B)
# energy → strict + energy boost/drop (±2, same letter)
# pro → energy + perfect 4th/5th (±7, same letter) + modal shift (swap A/B and ±1)
# open → pro + tritone (±6, same letter)
# Harmonic strategy: 'strict' | 'energy' | 'pro' | 'open'
HARMONIC_MODE    = "pro"

# Energy usage in sequencing:
#   'ignore' → don't consider energy at all
#   'ramp'   → non-decreasing energy, allow at most +ENERGY_STEP_MAX per step (default)
#   'wave'   → allow +/-ENERGY_STEP_MAX around current energy
#   'match'  → prefer same energy (allows +/-1 fallback when stuck)
ENERGY_STRATEGY  = "ramp"
ENERGY_STEP_MAX  = 2

RANDOM_SEED      = None               # None = different every run; or int like 42
ENABLE_DEDUPING  = True               # de-dupe near-duplicate titles
# ----------------------------

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

CAMELOT_RE = re.compile(r'^(\d{1,2})([ABab])$')
# stems like: drums_80_2A_5-Artist - Title.flac (role may be drums/inst/vocals)
STEM_RE    = re.compile(r'^(drums|inst|vocals)_(\d+)_(\d+[ABab])_(\d+)-', re.IGNORECASE)

def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\u2010-\u2015]', '-', s)   # normalize dashes
    s = re.sub(r'[_\(\)\[\]\{\}\-]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s\.&/+]', '', s)
    return s.strip()

def norm_title_from_folder(folder: str) -> str:
    base = Path(folder).name
    base = re.sub(r'\b(vip|edit|remix|mix|master|version|alt|vip mix)\b', '', base, flags=re.I)
    base = re.sub(r'\b(feat|ft|with)\b.*', '', base, flags=re.I)
    base = re.sub(r'\b\d{2}\b', '', base)  # scrap track numbers like "01"
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
    """
    Return a ranked list of allowed next keys based on HARMONIC_MODE.
    Earlier items are considered 'nicer' for scoring.
    """
    m = CAMELOT_RE.fullmatch(key.strip())
    if not m: return []
    n = int(m.group(1)); L = m.group(2).upper()
    other = 'B' if L == 'A' else 'A'

    buckets: List[Set[str]] = []

    # 1) Same key (perfect)
    buckets.append({f"{n}{L}"})

    # 2) Relative major/minor (change letter, same number)
    buckets.append({f"{n}{other}"})

    # 3) ±1 around the wheel (same letter)
    buckets.append({f"{wrap12(n+1)}{L}", f"{wrap12(n-1)}{L}"})

    if mode in ("energy", "pro", "open"):
        # 4) Energy boost/drop ±2 (same letter)
        buckets.append({f"{wrap12(n+2)}{L}", f"{wrap12(n-2)}{L}"})

    if mode in ("pro", "open"):
        # 5) Modal shift neighbors (swap A/B and ±1 number)
        buckets.append({f"{wrap12(n+1)}{'B' if L=='A' else 'A'}", f"{wrap12(n-1)}{'B' if L=='A' else 'A'}"})
        # 6) Perfect fourth/fifth (±7, same letter)
        buckets.append({f"{wrap12(n+7)}{L}", f"{wrap12(n-7)}{L}"})

    if mode == "open":
        # 7) Tritone (±6, same letter)
        buckets.append({f"{wrap12(n+6)}{L}", f"{wrap12(n-6)}{L}"})

    out: List[str] = []
    seen: Set[str] = set()
    for b in buckets:
        for k in b:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out

# ---------- ENERGY EXTRACTION ----------
def parse_energy_from_folder(folder: str) -> Optional[int]:
    """
    Look for stems in the folder and extract the energy integer from filenames like:
      drums_80_2A_5-*.flac  (also inst_/vocals_)
    Prefer 'drums' > 'inst' > 'vocals' if multiple found.
    """
    prefer = {"drums": 0, "inst": 1, "vocals": 2}
    best: Tuple[int, Optional[int]] = (999, None)  # (rank, energy)

    try:
        for name in os.listdir(folder):
            m = STEM_RE.match(name)
            if not m:
                continue
            role, _bpm, _key, energy_str = m.groups()
            rank = prefer.get(role.lower(), 99)
            energy = int(energy_str)
            if rank < best[0]:
                best = (rank, energy)
    except FileNotFoundError:
        return None

    return best[1]

# ---------- COLLECT / DEDUPE ----------
def collect_tracks(stems_dir: str) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    for bpm_name in os.listdir(stems_dir):
        if not bpm_name.isdigit():
            continue
        bpm = int(bpm_name)
        bpm_dir = os.path.join(stems_dir, bpm_name)
        if not os.path.isdir(bpm_dir):
            continue
        for key_name in os.listdir(bpm_dir):
            key_dir = os.path.join(bpm_dir, key_name)
            if not os.path.isdir(key_dir) or not key_ok(key_name):
                continue
            for root, _, files in os.walk(key_dir):
                if "CH1.als" in files:
                    track_folder = os.path.basename(root)
                    energy = parse_energy_from_folder(root)  # <-- NEW
                    tracks.append({
                        "bpm": bpm,
                        "key": key_name.upper(),
                        "energy": energy,  # may be None
                        "folder": track_folder,
                        "src": os.path.join(root, "CH1.als"),
                        "artist": guess_artist(track_folder),
                        "title_norm": norm_title_from_folder(track_folder),
                    })
    def sort_key(t):
        m = CAMELOT_RE.fullmatch(t["key"])
        n = int(m.group(1)); L = 0 if m.group(2).upper()=="A" else 1
        # sort by BPM, key number/letter, then energy (None last), then name
        energy_sort = t["energy"] if t["energy"] is not None else 999
        return (t["bpm"], n, L, energy_sort, norm_text(t["folder"]))
    tracks.sort(key=sort_key)
    return tracks

def dedupe(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_DEDUPING:
        return tracks
    seen = set()
    out = []
    for t in tracks:
        k = (t["bpm"], t["key"], t["title_norm"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

def build_indexes(tracks: List[Dict[str, Any]]):
    by_bpm: Dict[int, List[Dict[str, Any]]] = {}
    for t in tracks:
        by_bpm.setdefault(t["bpm"], []).append(t)
    return by_bpm

# ---------- START / PICK ----------
def choose_start(tracks: List[Dict[str, Any]], by_bpm: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not tracks:
        raise RuntimeError("No tracks found.")
    # If ramping, bias to lower energy starts when available
    if ENERGY_STRATEGY == "ramp":
        low_energy = [t for t in tracks if (t.get("energy") is not None and t["energy"] <= 4)]
        if low_energy:
            return random.choice(low_energy)
    if START_BPM is None:
        return random.choice(tracks)
    candidate_bpms = sorted(by_bpm.keys(), key=lambda x: (abs(x-START_BPM), x))
    for b in candidate_bpms:
        bucket = by_bpm.get(b, [])
        if bucket:
            return random.choice(bucket)
    return random.choice(tracks)

def energy_ok(curr_e: Optional[int], next_e: Optional[int], strategy: str) -> bool:
    if strategy == "ignore":
        return True
    if curr_e is None or next_e is None:
        # If we don't know one of them, don't block the transition
        return True

    delta = next_e - curr_e
    adelta = abs(delta)

    if strategy == "ramp":
        return (delta >= 0) and (delta <= ENERGY_STEP_MAX)
    if strategy == "wave":
        return adelta <= ENERGY_STEP_MAX
    if strategy == "match":
        return next_e == curr_e

    return True  # fallback

def pick_next(curr: Dict[str, Any],
              by_bpm: Dict[int, List[Dict[str, Any]]],
              used_srcs: Set[str],
              artist_counts: Dict[str, int]) -> Optional[Dict[str, Any]]:

    curr_bpm = curr["bpm"]
    curr_key = curr["key"]
    curr_energy = curr.get("energy", None)

    key_ranked = allowed_keys(curr_key, HARMONIC_MODE)
    allowed_set = set(key_ranked)

    next_bpms = list(range(curr_bpm, curr_bpm + BPM_STEP_MAX + 1))

    def gather_candidates(relax_energy: bool = False) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        for b in next_bpms:
            for t in by_bpm.get(b, []):
                if t["src"] in used_srcs:
                    continue
                if t["key"] not in allowed_set:
                    continue
                if artist_counts.get(t["artist"], 0) >= MAX_PER_ARTIST:
                    continue
                if not relax_energy and not energy_ok(curr_energy, t.get("energy"), ENERGY_STRATEGY):
                    continue
                cands.append(t)
        if not cands:
            # small backstep in BPM if we have nothing
            down_bpms = list(range(max(70, curr_bpm-2), curr_bpm))
            for b in down_bpms:
                for t in by_bpm.get(b, []):
                    if t["src"] in used_srcs:
                        continue
                    if t["key"] not in allowed_set:
                        continue
                    if artist_counts.get(t["artist"], 0) >= MAX_PER_ARTIST:
                        continue
                    if not relax_energy and not energy_ok(curr_energy, t.get("energy"), ENERGY_STRATEGY):
                        continue
                    cands.append(t)
        return cands

    candidates = gather_candidates(relax_energy=False)
    if not candidates:
        # Relax energy rule as a safety valve
        candidates = gather_candidates(relax_energy=True)
        if not candidates:
            return None

    key_priority = {k:i for i,k in enumerate(key_ranked)}

    def score(t):
        bpm_jump = abs(t["bpm"] - curr_bpm)
        kp = key_priority.get(t["key"], 999)
        # Energy scoring
        e = t.get("energy", None)
        if curr_energy is None or e is None:
            e_penalty = 1  # unknown energy: mild penalty
            e_diff = 9
        else:
            e_diff = abs(e - curr_energy)
            if ENERGY_STRATEGY == "ramp":
                # prefer small positive increases (0..ENERGY_STEP_MAX)
                e_penalty = (0 if e >= curr_energy else 3) + max(0, e - curr_energy - ENERGY_STEP_MAX)
            elif ENERGY_STRATEGY == "match":
                e_penalty = 0 if e == curr_energy else (1 if e_diff == 1 else 3)
            else:  # wave/ignore handled above; smaller diff is better
                e_penalty = e_diff
        return (bpm_jump, e_penalty, kp, e_diff, norm_text(t["folder"]))

    candidates.sort(key=score)
    top = candidates[:5] if len(candidates) >= 5 else candidates
    return random.choice(top)

# ---------- BUILD SET ----------
def build_set(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_bpm = build_indexes(tracks)
    current = choose_start(tracks, by_bpm)
    chosen = [current]
    used = {current["src"]}
    artist_counts = {current["artist"]: 1}

    while len(chosen) < SET_LEN:
        nxt = pick_next(chosen[-1], by_bpm, used, artist_counts)
        if not nxt:
            break
        chosen.append(nxt)
        used.add(nxt["src"])
        artist_counts[nxt["artist"]] = artist_counts.get(nxt["artist"], 0) + 1

    return chosen

# ---- Per-run folder naming with the actual first track BPM/Key ----
def build_descriptor_word(start_bpm_actual: int, start_key_actual: str) -> str:
    """
    Format: START_BPM_START_KEY.HARMONIC_MODE-ENERGY_STRATEGY-l{SET_LEN}-step{BPM_STEP_MAX}
    Example: '124_8A.pro-ramp-l18-step4'
    """
    bpm_part  = f"{start_bpm_actual}"
    key_part  = start_key_actual
    mode_part = HARMONIC_MODE.lower()
    energy    = ENERGY_STRATEGY.lower()
    len_part  = f"l{SET_LEN}"
    step_part = f"step{BPM_STEP_MAX}"
    return f"{bpm_part}_{key_part}.{mode_part}-{energy}-{len_part}-{step_part}"

def make_unique_output_dir(start_bpm_actual: int, start_key_actual: str) -> Path:
    base = Path(OUTPUT_ROOT_DIR)
    base.mkdir(parents=True, exist_ok=True)
    descriptor = build_descriptor_word(start_bpm_actual, start_key_actual)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir_name = f"{descriptor}_{ts}"
    run_dir = base / run_dir_name
    (run_dir / "sets").mkdir(parents=True, exist_ok=True)
    return run_dir

# ---------- OUTPUT ----------
def write_out(set_tracks: List[Dict[str, Any]]):
    start_bpm_actual = set_tracks[0]["bpm"]
    start_key_actual = set_tracks[0]["key"]

    run_dir = make_unique_output_dir(start_bpm_actual, start_key_actual)
    out_dir = run_dir / "sets"
    csv_path = run_dir / "playlist.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "BPM", "Key", "Energy", "TrackFolder", "SourcePath", "DestFile"])
        for i, t in enumerate(set_tracks, start=1):
            safe_folder = t["folder"]
            energy = t.get("energy", None)
            if energy is not None:
                dst_name = f"{i:03d}_{t['bpm']}_{t['key']}_{energy}-{safe_folder}.als"
            else:
                dst_name = f"{i:03d}_{t['bpm']}_{t['key']}-{safe_folder}.als"
            dst = out_dir / dst_name
            shutil.copy(t["src"], dst)
            writer.writerow([i, t["bpm"], t["key"], energy if energy is not None else "", t["folder"], t["src"], str(dst)])
            print(f"{i:03d}: {t['src']} → {dst}")

    print(f"\nSaved set to: {run_dir}")
    print(f"Wrote {len(set_tracks)} tracks + CSV → {csv_path}")

# ---------- MAIN ----------
def main():
    tracks = collect_tracks(STEMS_DIR)
    if not tracks:
        print("No CH1.als files found. Check your STEMS_DIR and folder structure: /BPM/Key/Track/CH1.als")
        sys.exit(1)

    tracks = dedupe(tracks)
    set_tracks = build_set(tracks)

    if len(set_tracks) < 2:
        print("Could not build a sequence with current constraints.\n"
              f"- Try different HARMONIC_MODE (current: {HARMONIC_MODE})\n"
              f"- Try different ENERGY_STRATEGY (current: {ENERGY_STRATEGY}) or ENERGY_STEP_MAX={ENERGY_STEP_MAX}\n"
              "- Increase BPM_STEP_MAX\n"
              "- Increase MAX_PER_ARTIST\n"
              "- Or expand your library in the target BPM/key range.")
        sys.exit(2)

    write_out(set_tracks)

if __name__ == "__main__":
    main()
