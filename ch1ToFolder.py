#!/usr/bin/env python3
import os
import re
import shutil

stems_dir  = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
output_dir = "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/stuff"

# If you want the prefix (drums_/inst_/vocals_) in the ALS filename, keep this True
INCLUDE_ROLE_PREFIX = False

os.makedirs(output_dir, exist_ok=True)

# Regex to parse stems like: drums_80_2A_5-Artist - Title.flac
STEM_RE = re.compile(r'^(drums|inst|vocals)_(\d+)_(\d+[AB])_(\d+)-', re.IGNORECASE)

def find_role_and_energy(folder: str):
    """
    Return (role, energy) by scanning a folder for stems named
    <role>_<bpm>_<camelot>_<energy>-*.ext. Prefer drums > inst > vocals.
    """
    prefer = ["drums", "inst", "vocals"]
    found = {}

    for name in os.listdir(folder):
        m = STEM_RE.match(name)
        if not m:
            continue
        role, _bpm, _key, energy = m.groups()
        role = role.lower()
        found[role] = energy

    for role in prefer:
        if role in found:
            return role, found[role]

    return None, None

def safe_name(s: str) -> str:
    # Keep your original characters, but make sure slashes don’t break paths
    return s.replace("/", "_").replace("\\", "_")

for bpm in range(70, 181):
    bpm_folder = os.path.join(stems_dir, str(bpm))
    if not os.path.isdir(bpm_folder):
        continue

    for root, dirs, files in os.walk(bpm_folder):
        if "CH1.als" not in files:
            continue

        src = os.path.join(root, "CH1.als")

        track_name = os.path.basename(root)  # e.g., "CØNTRA, Saturna - What I Can Do"
        key_dir = os.path.basename(os.path.dirname(root))  # e.g., "2A"

        # Discover energy (and which role we found it from) by looking at stems in this folder
        role, energy = find_role_and_energy(root)

        # Build the filename
        prefix = (f"{role}_" if (INCLUDE_ROLE_PREFIX and role) else "")
        energy_part = (f"_{energy}" if energy else "")  # if not found, omit gracefully

        dst_filename = f"{prefix}{bpm}_{key_dir}{energy_part}-{safe_name(track_name)}.als"
        dst = os.path.join(output_dir, dst_filename)

        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
