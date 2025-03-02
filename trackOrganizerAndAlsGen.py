import os
import shutil
from mutagen.id3 import ID3, ID3NoHeaderError, error as ID3Error
import gzip
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import trackTime  # Assuming this module is available

# --- trackOrganizer.py Section ---

# Prompt the user for input
date_folder = input("Enter the date folder (e.g., 1-25-25): ")

# Source and destination paths
mp3_source_folder = f"/Users/alirahimlou/Desktop/PlaylistsByDate/{date_folder}"
htdemucs_source_folder = "/Users/alirahimlou/Desktop/STEMS/toBeOrganized"
destination_folder = "/Users/alirahimlou/Desktop/STEMS"

# Camelot Wheel Mapping
camelot_wheel = {
    "Amin": "8A", "A#min": "9A", "Bbmin": "3A", "Bmin": "10A", "Cmin": "5A",
    "C#min": "12A", "Dbmin": "12A", "Dmin": "7A", "D#min": "2A", "Ebmin": "2A",
    "Emin": "9A", "Fmin": "4A", "F#min": "11A", "Gbmin": "11A", "Gmin": "6A",
    "G#min": "3A", "Abmin": "3A", "Cmaj": "8B", "C#maj": "9B", "Dbmaj": "9B",
    "Dmaj": "10B", "D#maj": "11B", "Ebmaj": "11B", "Emaj": "12B", "Fmaj": "7B",
    "F#maj": "2B", "Gbmaj": "2B", "Gmaj": "9B", "G#maj": "4B", "Abmaj": "4B",
    "Amaj": "11B", "A#maj": "6B", "Bbmaj": "6B", "Bmaj": "1B"
}

# Fix for ambiguous keys
ambiguous_keys = {
    "Eb": "2A", "Ebm": "2A", "Ebmaj": "11B", "B": "1B"
}

def get_bpm_and_key(file_path):
    """Extract BPM and Key from an MP3 file's ID3 tags safely."""
    try:
        audio = ID3(file_path)
        bpm = audio.get('TBPM')
        bpm = bpm.text[0] if bpm and hasattr(bpm, 'text') else 'Unknown BPM'
        key = audio.get('TKEY')
        key = key.text[0].strip() if key and hasattr(key, 'text') else 'Unknown Key'
        return bpm, key
    except (ID3NoHeaderError, ID3Error, AttributeError, KeyError):
        return 'Unknown BPM', 'Unknown Key'

def convert_to_camelot(key):
    """Convert key to Camelot notation, ensuring ambiguous keys are handled."""
    if key == "Unknown Key":
        return "Unknown Key"
    key = key.strip().replace(" ", "").replace("m", "min").replace("#", "#")
    return ambiguous_keys.get(key, camelot_wheel.get(key, "Unknown Key"))

def rename_and_move_files(source_folder, bpm, camelot_key):
    """Rename 'drums', 'Inst', and 'vocals' files by inserting BPM and key immediately after the prefix."""
    for file in os.listdir(source_folder):
        if file.startswith(("drums", "Inst", "vocals")):
            old_path = os.path.join(source_folder, file)
            name_part, ext = os.path.splitext(file)
            if name_part.startswith("drums"):
                new_name = f"drums_{bpm}_{camelot_key}{name_part[5:]}{ext}"
            elif name_part.startswith("Inst"):
                new_name = f"Inst_{bpm}_{camelot_key}{name_part[4:]}{ext}"
            elif name_part.startswith("vocals"):
                new_name = f"vocals_{bpm}_{camelot_key}{name_part[6:]}{ext}"
            else:
                continue
            new_path = os.path.join(source_folder, new_name)
            os.rename(old_path, new_path)
            print(f"[INFO] Renamed '{file}' → '{new_name}'")

def move_folders_based_on_bpm_and_camelot(mp3_folder, htdemucs_folder, dest_folder):
    """Move folders based on BPM and Camelot Key while logging results."""
    print(f"\n[INFO] Organizing folders from '{htdemucs_folder}' to '{dest_folder}'\n")
    processed_tracks = []
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in os.listdir(mp3_folder):
        if file.endswith('.mp3'):
            file_path = os.path.join(mp3_folder, file)
            bpm, key = get_bpm_and_key(file_path)
            camelot_key = convert_to_camelot(key)
            processed_tracks.append((file, bpm, key, camelot_key))
            if bpm == "Unknown BPM" or camelot_key == "Unknown Key":
                print(f"[WARNING] Skipping {file} due to missing metadata.")
                continue
            bpm_folder = os.path.join(dest_folder, str(bpm))
            camelot_folder = os.path.join(bpm_folder, camelot_key)
            track_name = os.path.splitext(file)[0]
            source_track_folder = os.path.join(htdemucs_source_folder, track_name)
            if os.path.exists(source_track_folder):
                rename_and_move_files(source_track_folder, bpm, camelot_key)
                os.makedirs(camelot_folder, exist_ok=True)
                target_folder = os.path.join(camelot_folder, track_name)
                shutil.move(source_track_folder, target_folder)
                print(f"[INFO] Moved '{track_name}' → '{target_folder}'")
    print("\n[INFO] Processing Summary\n")
    print(f"{'Track Name':<40} {'BPM':<10} {'Camelot Key':<10}")
    print("="*80)
    for track, bpm, key, camelot in processed_tracks:
        print(f"{track[:37]:<40} {bpm:<10} {camelot:<10}")
    print("="*80)
    print("\n[INFO] Track organization complete.\n")

# --- alsGen.py Section ---

# 🛠 CONFIG: Paths
ALS_FILES_FOLDER = "alsFiles"  # Folder where BPM ALS templates are stored
FLAC_FOLDER = "/Users/alirahimlou/Desktop/STEMS"

# ✅ CONFIG: Skip or overwrite existing ALS files
SKIP_EXISTING = True  # Set to False if you want to overwrite existing ALS files

def find_flac_folders(directory):
    """Recursively search for folders in `directory` that contain .flac files."""
    results = []
    for root, dirs, files in os.walk(directory):
        flac_files = sorted([f for f in files if f.lower().endswith(".flac")])
        if flac_files:
            track_names = {"drums": None, "Inst": None, "vocals": None}
            for f in flac_files:
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                if "drums" in f.lower() and track_names["drums"] is None:
                    track_names["drums"] = rel_path
                elif "inst" in f.lower() and track_names["Inst"] is None:
                    track_names["Inst"] = rel_path
                elif "vocals" in f.lower() and track_names["vocals"] is None:
                    track_names["vocals"] = rel_path
            bpm_value = extract_bpm_from_path(root)
            if any(track_names.values()):
                results.append((root, track_names, bpm_value))
    return results

def extract_bpm_from_path(folder_path):
    """Extracts the BPM from the folder structure."""
    parts = folder_path.split(os.sep)
    try:
        bpm_value = int(parts[-3])
        return bpm_value
    except (IndexError, ValueError):
        print(f"   Warning: Could not extract BPM from path '{folder_path}'.")
        return None

def select_blank_als(bpm_value):
    """Dynamically selects the correct blank ALS file based on BPM value."""
    if bpm_value:
        bpm_als_path = os.path.join(ALS_FILES_FOLDER, f"{bpm_value}.als")
        if os.path.exists(bpm_als_path):
            return bpm_als_path
    print(f"⚠️ Warning: No ALS file found for BPM {bpm_value}. Skipping...")
    return None

def get_duration_in_beats(track_path, bpm):
    """Gets the duration of the drums FLAC file in beats using the provided BPM."""
    try:
        duration_seconds = trackTime.get_track_duration(track_path)
        print(f"   Drums Duration: {duration_seconds:.2f} seconds")
        duration_beats = (duration_seconds * bpm) / 60
        print(f"   Converted to {bpm} BPM: {duration_beats:.6f} beats")
        return f"{duration_beats:.6f}"
    except Exception as e:
        print(f"   Error getting duration: {e}")
        return None

def modify_als_file(input_path, target_folder, track_names, bpm_value):
    """Loads the selected ALS file, replaces FLAC references, updates LoopEnd and OutMarker, and saves."""
    try:
        if input_path is None:
            print(f"❌ Skipping folder '{target_folder}' due to missing ALS template.")
            return
        output_als = os.path.join(target_folder, "CH1.als")
        if os.path.exists(output_als) and SKIP_EXISTING:
            print(f"⏭️ Skipping '{target_folder}' – CH1.als already exists.")
            return
        shutil.copy(input_path, output_als)
        new_loop_end = None
        if track_names["drums"] and bpm_value:
            flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
            new_loop_end = get_duration_in_beats(flac_path, bpm_value)
        else:
            print("   No drums track or BPM value available; skipping LoopEnd modification.")
        with gzip.open(output_als, "rb") as f:
            als_data = f.read()
        tree = ET.parse(BytesIO(als_data))
        root = tree.getroot()
        modified_count = 0
        if new_loop_end:
            for elem in root.iter():
                if elem.tag == "LoopEnd":
                    old_value = elem.get("Value")
                    elem.set("Value", new_loop_end)
                    modified_count += 1
                    print(f"   Updated <LoopEnd> from {old_value} to {new_loop_end}")
                elif elem.tag == "OutMarker":
                    old_value = elem.get("Value")
                    elem.set("Value", new_loop_end)
                    print(f"   Updated <OutMarker> from {old_value} to {new_loop_end}")
        with gzip.open(output_als, "wb") as f_out:
            tree.write(f_out, encoding="utf-8", xml_declaration=True)
        with gzip.open(output_als, "rb") as f:
            als_data = f.read()
        als_str = als_data.decode("latin1")
        replacements = {
            "drums-Tape B - i won't be ur drug.flac": track_names["drums"] if track_names["drums"] else "",
            "Inst-Tape B - i won't be ur drug.flac": track_names["Inst"] if track_names["Inst"] else "",
            "vocals-Tape B - i won't be ur drug.flac": track_names["vocals"] if track_names["vocals"] else "",
        }
        for old, new in replacements.items():
            if new:
                als_str = als_str.replace(old, new)
                als_str = als_str.replace(f"../{old}", f"../{new}")
                als_str = als_str.replace(f"{target_folder}/{old}", f"{target_folder}/{new}")
                als_str = als_str.replace(old.replace(" ", "%20"), new.replace(" ", "%20"))
        for old, new in replacements.items():
            if new:
                old_track_name = old.replace(".flac", "")
                new_track_name = os.path.basename(new).replace(".flac", "")
                als_str = re.sub(rf'(<MemorizedFirstClipName Value="){re.escape(old_track_name)}(")', rf'\1{new_track_name}\2', als_str)
                als_str = re.sub(rf'(<UserName Value="){re.escape(old_track_name)}(")', rf'\1{new_track_name}\2', als_str)
                als_str = re.sub(rf'(<Name Value="){re.escape(old_track_name)}(")', rf'\1{new_track_name}\2', als_str)
                als_str = re.sub(rf'(<EffectiveName Value="){re.escape(old_track_name)}(")', rf'\1{new_track_name}\2', als_str)
        with gzip.open(output_als, "wb") as f:
            f.write(als_str.encode("latin1"))
        print(f"✅ Final modified ALS saved at: {output_als} (Modified {modified_count} LoopEnd elements)\n")
    except Exception as e:
        print(f"❌ Error modifying ALS in folder '{target_folder}': {e}")

def generate_als_files():
    """Generate ALS files for organized tracks."""
    folders = find_flac_folders(FLAC_FOLDER)
    if not folders:
        print("❌ No relevant FLAC files found in any folder!")
    else:
        for folder, track_names, bpm_value in folders:
            blank_als_path = select_blank_als(bpm_value)
            print(f"🎯 Processing folder: {folder} (BPM: {bpm_value or 'Unknown'})")
            print(f"   Using ALS template: {blank_als_path if blank_als_path else '⚠️ Skipping (No ALS file)'}")
            print("   Found track files:", track_names)
            modify_als_file(blank_als_path, folder, track_names, bpm_value)
        print("🎵 All ALS files generated successfully!")

if __name__ == "__main__":
    # Step 1: Organize tracks
    move_folders_based_on_bpm_and_camelot(mp3_source_folder, htdemucs_source_folder, destination_folder)
    # Step 2: Generate ALS files
    print("\n[INFO] Starting ALS file generation...\n")
    generate_als_files()