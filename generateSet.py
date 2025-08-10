#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generateSet.py — append scenes DOWNWARD with CH1/CH2/CH3(/CH4) only.
Now sorts source .als files by BPM parsed from the filename (ascending),
so your scenes go 79 → … → 155 in order.

- Copies the entire ClipSlot from each source CH track (if it has a clip).
- No internal <…Id Value> renumbering; list-item attribute Ids only.
- Keeps only the first CH tracks present in BASE_ALS (prevents horizontal dupes).
"""

import gzip, re, sys
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

# --------- hard-coded paths ---------
SRC_DIR  = Path("/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/stuff")
OUT_FILE = Path("/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/Sets/Combined_CH_Scenes.als")
BASE_ALS = Path("/Users/alirahimlou/myapps/trackOrganizer/alsFiles/CH1.als")  # point to your base

MAX_FILES = 0
VERBOSE   = True

# ---------- utils ----------
def is_gz(b: bytes) -> bool: return len(b) >= 2 and b[:2] == b"\x1f\x8b"
def read_als_text(p: Path) -> str:
    d = p.read_bytes()
    return (gzip.decompress(d) if is_gz(d) else d).decode("utf-8", errors="replace")
def write_als_gz(p: Path, root: ET.Element):
    buf = BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True, method="xml")
    p.write_bytes(gzip.compress(buf.getvalue()))
    ET.fromstring(gzip.decompress(p.read_bytes()))  # sanity

def deep_copy(e: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(e, encoding="utf-8"))

# ---------- Ableton helpers ----------
def liveset(root): 
    if root.tag == "LiveSet": return root
    if root.tag == "Ableton": return root.find("LiveSet")
    return root.find("LiveSet")

def scenes_node(root):
    sc = root.find(".//Scenes") or root.find(".//SceneList")
    if sc is None:
        ls = liveset(root) or ET.SubElement(root, "LiveSet")
        sc = ET.SubElement(ls, "Scenes")
    return sc

def tracks_parent(root): return root.find(".//Tracks")

def name_value(track: ET.Element) -> str:
    nm = track.find("./Name")
    if nm is None: return ""
    for tag in ("UserName","EffectiveName","Name"):
        el = nm.find(tag) if tag != "Name" else nm
        if el is not None and el.attrib.get("Value"): return el.attrib["Value"]
    return ""

def track_label(track: ET.Element) -> str: return name_value(track)

def ch_index_from_label(name: str):
    m = re.search(r'\bch(?:annel)?\s*([1-4])\b', name, flags=re.IGNORECASE)
    if m: return int(m.group(1))
    if re.fullmatch(r'CH[1-4]', name.strip(), flags=re.IGNORECASE): return int(name.strip()[-1])
    return None

def ch_tracks_map(root):
    tp = tracks_parent(root) or []
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
    tp = tracks_parent(root) or []
    mapping = {}
    for t in list(tp):
        idx = role_index_from_name(track_label(t))
        if idx and idx not in mapping: mapping[idx] = t
    return mapping

def main_csl(track):   return track.find(".//DeviceChain/MainSequencer/ClipSlotList")
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

# ---------- BPM sorting ----------
_BPM_RX = re.compile(r'(?<!\d)(\d{2,3})(?=[ _\-\.])')  # first 2–3 digit number before delimiter

def bpm_from_filename(path: Path) -> int:
    """
    Extract a BPM-like number (e.g., '154' from '154_4A_Title.als').
    Falls back to a large number so files without a BPM sort to the end.
    """
    s = path.stem
    m = _BPM_RX.search(s)
    if m:
        try:
            bpm = int(m.group(1))
            # sanity clamp to common range
            if 40 <= bpm <= 220:
                return bpm
        except ValueError:
            pass
    return 10**9  # put unknowns at the end

# ---------- core ----------
def append_scene(master_root: ET.Element, ch_map: dict[int, ET.Element], src_path: Path):
    sc = scenes_node(master_root)

    # New Scene (attribute Id only)
    scene_id = next_attr_id(sc)
    new_scene = ET.Element("Scene"); new_scene.attrib["Id"] = str(scene_id)
    ET.SubElement(new_scene, "Name").attrib["Value"] = src_path.stem
    sc.append(new_scene)

    # Load source and build a per-channel source map (CHx or role fallback)
    try:
        sroot = ET.fromstring(read_als_text(src_path))
    except Exception:
        sroot = None
    src_map = ch_tracks_map(sroot) if sroot is not None else {}
    if sroot is not None and not src_map:
        src_map = src_role_map(sroot)

    for ch_idx, dest_track in sorted(ch_map.items()):
        mcsl = main_csl(dest_track)
        if mcsl is None:
            dc = dest_track.find("./DeviceChain") or ET.SubElement(dest_track, "DeviceChain")
            ms = dc.find("./MainSequencer") or ET.SubElement(dc, "MainSequencer")
            mcsl = ET.SubElement(ms, "ClipSlotList")

        # Build the slot to append
        src_track = src_map.get(ch_idx)
        new_slot = None
        if src_track is not None:
            src_slot = first_filled_slot(src_track)
            if src_slot is not None:
                new_slot = deep_copy(src_slot)  # full slot with AudioClip/MidiClip etc.

        if new_slot is None:
            # fall back: clone last dest slot but strip clips
            prev = list(mcsl.findall("./ClipSlot"))
            if prev:
                new_slot = deep_copy(prev[-1]); remove_all_clips(new_slot)
            else:
                new_slot = ET.Element("ClipSlot")

        # assign unique attribute Id (no child <Id>, no <SlotId>)
        new_slot.attrib["Id"] = str(next_attr_id(mcsl))
        mcsl.append(new_slot)

        # Keep FreezeSequencer counts aligned if base has them
        fcsl = freeze_csl(dest_track)
        if fcsl is not None:
            fprev = list(fcsl.findall("./ClipSlot"))
            if fprev:
                fnew = deep_copy(fprev[-1]); remove_all_clips(fnew)
            else:
                fnew = ET.Element("ClipSlot")
            fnew.attrib["Id"] = str(next_attr_id(fcsl))
            fcsl.append(fnew)

def main():
    if not SRC_DIR.exists(): print(f"[ERROR] Missing SRC_DIR: {SRC_DIR}"); sys.exit(1)
    if not BASE_ALS.exists(): print(f"[ERROR] Missing BASE_ALS: {BASE_ALS}"); sys.exit(1)

    master_root = ET.fromstring(read_als_text(BASE_ALS))

    # Keep only the FIRST CH tracks from base (prevents horizontal dupes)
    tp = tracks_parent(master_root)
    if tp is None: print("[ERROR] Base has no <Tracks>"); sys.exit(1)
    first_for = {}
    for t in list(tp):
        idx = ch_index_from_label(track_label(t))
        if idx and idx not in first_for: first_for[idx] = t
    keep = set(first_for.values())
    for t in list(tp):
        if t not in keep: tp.remove(t)

    ch_map = ch_tracks_map(master_root)
    if not ch_map: print("[ERROR] No CH tracks (CH1..CH4) in base"); sys.exit(1)

    # Gather and sort sources by BPM in filename (ascending)
    candidates = [
        p for p in SRC_DIR.rglob("*.als")
        if "__MACOSX" not in str(p) and not p.name.startswith("._")
           and p.resolve() != BASE_ALS.resolve()
    ]
    if MAX_FILES and MAX_FILES > 0:
        candidates = candidates[:MAX_FILES]
    als_paths = sorted(candidates, key=lambda p: (bpm_from_filename(p), p.name.lower()))

    if VERBOSE:
        preview = [f"{p.name} (bpm={bpm_from_filename(p) if bpm_from_filename(p)!=10**9 else '?'} )" for p in als_paths[:10]]
        print(f"[INFO] Base: {BASE_ALS.name} | Channels: {', '.join('CH'+str(k) for k in sorted(ch_map))}")
        print(f"[INFO] Appending {len(als_paths)} scene(s) in BPM order...")
        if preview: print("[INFO] First 10:", "; ".join(preview))

    for sp in als_paths:
        append_scene(master_root, ch_map, sp)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_als_gz(OUT_FILE, master_root)
    print(f"[DONE] Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
