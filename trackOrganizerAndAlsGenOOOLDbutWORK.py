import os
import shutil
from mutagen.id3 import ID3, ID3NoHeaderError, error as ID3Error
from mutagen.flac import FLAC  # Added for FLAC support
from mutagen.wave import WAVE  # Added for WAV support
import gzip
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import trackTime  # Assuming this module is available

# --- trackOrganizer.py Section ---

# Source and destination paths
mp3_source_folder = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder = "/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized"
destination_folder = "/Users/alirahimlou/Desktop/MUSIC/STEMS"

# Expanded and Corrected Camelot Wheel Mapping (uppercase, standardized to correct Camelot codes)
camelot_wheel = {
    "ABMIN": "1A", "G#MIN": "1A",
    "EBMIN": "2A", "D#MIN": "2A",
    "BBMIN": "3A", "A#MIN": "3A",
    "FMIN": "4A",
    "CMIN": "5A",
    "GMIN": "6A",
    "DMIN": "7A",
    "AMIN": "8A",
    "EMIN": "9A",
    "BMIN": "10A",
    "F#MIN": "11A", "GBMIN": "11A",
    "C#MIN": "12A", "DBMIN": "12A",
    "BMAJ": "1B",
    "F#MAJ": "2B", "GBMAJ": "2B",
    "C#MAJ": "3B", "DBMAJ": "3B",
    "ABMAJ": "4B", "G#MAJ": "4B",
    "EBMAJ": "5B", "D#MAJ": "5B",
    "BBMAJ": "6B", "A#MAJ": "6B",
    "FMAJ": "7B",
    "CMAJ": "8B",
    "GMAJ": "9B",
    "DMAJ": "10B",
    "AMAJ": "11B",
    "EMAJ": "12B",
    # Additional variations for common tagging inconsistencies
    "ABMINOR": "1A", "G#MINOR": "1A",
    "EBMINOR": "2A", "D#MINOR": "2A",
    "BBMINOR": "3A", "A#MINOR": "3A",
    "FMINOR": "4A",
    "CMINOR": "5A",
    "GMINOR": "6A",
    "DMINOR": "7A",
    "AMINOR": "8A",
    "EMINOR": "9A",
    "BMINOR": "10A",
    "F#MINOR": "11A", "GBMINOR": "11A",
    "C#MINOR": "12A", "DBMINOR": "12A",
    "BMAJOR": "1B",
    "F#MAJOR": "2B", "GBMAJOR": "2B",
    "C#MAJOR": "3B", "DBMAJOR": "3B",
    "ABMAJOR": "4B", "G#MAJOR": "4B",
    "EBMAJOR": "5B", "D#MAJOR": "5B",
    "BBMAJOR": "6B", "A#MAJOR": "6B",
    "FMAJOR": "7B",
    "CMAJOR": "8B",
    "GMAJOR": "9B",
    "DMAJOR": "10B",
    "AMAJOR": "11B",
    "EMAJOR": "12B",
}

def normalize_key(key):
    """Normalize key to uppercase standardized form like 'EMAJ' or 'EMIN'."""
    key = re.sub(r'\s+', '', key.upper())  # Remove spaces and uppercase
    # Replace full words
    key = re.sub(r'MAJOR$', 'MAJ', key)
    key = re.sub(r'MINOR$', 'MIN', key)
    # Handle short minor notation (e.g., 'Cm' -> 'CMIN')
    if re.match(r'^[A-G](#|B)?M$', key):
        key = key[:-1] + 'MIN'
    # Assume major if no specifier (e.g., 'C' -> 'CMAJ')
    if re.match(r'^[A-G](#|B)?$', key):
        key += 'MAJ'
    return key

def get_bpm_and_key(file_path):
    """Extract BPM and Key from MP3, FLAC, or WAV files."""
    try:
        if file_path.lower().endswith('.mp3'):
            # Handle MP3 files with ID3 tags
            audio = ID3(file_path)
            bpm = audio.get('TBPM')
            bpm = bpm.text[0] if bpm and hasattr(bpm, 'text') else 'Unknown BPM'
            key = audio.get('TKEY')
            key = key.text[0].strip() if key and hasattr(key, 'text') else 'Unknown Key'
        elif file_path.lower().endswith('.flac'):
            # Handle FLAC files with Vorbis comments
            audio = FLAC(file_path)
            bpm = audio.get('BPM', ['Unknown BPM'])[0]  # Common Vorbis tag for BPM
            key = audio.get('KEY', audio.get('INITIALKEY', ['Unknown Key']))[0]  # Try KEY or INITIALKEY
        elif file_path.lower().endswith('.wav'):
            # Handle WAV files, which may have embedded ID3 tags
            audio = WAVE(file_path)
            if audio.tags is None:
                return 'Unknown BPM', 'Unknown Key'
            bpm = audio.tags.get('TBPM')
            bpm = bpm.text[0] if bpm and hasattr(bpm, 'text') else 'Unknown BPM'
            key = audio.tags.get('TKEY')
            key = key.text[0].strip() if key and hasattr(key, 'text') else 'Unknown Key'
        else:
            return 'Unknown BPM', 'Unknown Key'
        return bpm, key
    except Exception as e:
        print(f"[DEBUG] Error reading metadata for {file_path}: {e}")
        return 'Unknown BPM', 'Unknown Key'

def convert_to_camelot(key):
    """Convert key to Camelot notation using normalization."""
    if key == "Unknown Key":
        return "Unknown Key"
    normalized = normalize_key(key)
    camelot_key = camelot_wheel.get(normalized, "Unknown Key")
    if camelot_key == "Unknown Key":
        print(f"[DEBUG] Unknown normalized key: {normalized} (original: {key})")
    return camelot_key

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
    for root, dirs, files in os.walk(mp3_folder):
        for file in files:
            if file.lower().endswith(('.mp3', '.flac', '.wav')):
                file_path = os.path.join(root, file)
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
FLAC_FOLDER = "/Users/alirahimlou/Desktop/MUSIC/STEMS"

# ✅ CONFIG: Skip or overwrite existing ALS files
SKIP_EXISTING = False  # Set to False if you want to overwrite existing ALS files

def find_flac_folders(directory):
    """Recursively search for folders in directory that contain .flac files."""
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
        return None

def select_blank_als(bpm_value):
    """Dynamically selects the correct blank ALS file based on BPM value."""
    if bpm_value:
        bpm_als_path = os.path.join(ALS_FILES_FOLDER, f"{bpm_value}.als")
        if os.path.exists(bpm_als_path):
            return bpm_als_path
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
            return
        output_als = os.path.join(target_folder, "CH1.als")
        if os.path.exists(output_als) and SKIP_EXISTING:
            return  # Silently skip existing files
        
        # Only print when actually processing a new file
        print(f"🎯 Processing new ALS for: {target_folder} (BPM: {bpm_value or 'Unknown'})")
        print(f"   Using ALS template: {input_path}")
        print("   Found track files:", track_names)

        shutil.copy(input_path, output_als)
        new_loop_end = None
        if track_names["drums"] and bpm_value:
            flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
            new_loop_end = get_duration_in_beats(flac_path, bpm_value)
        
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
        print(f"✅ New ALS saved at: {output_als} (Modified {modified_count} LoopEnd elements)\n")

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
            modify_als_file(blank_als_path, folder, track_names, bpm_value)
        print("🎵 All ALS files generated successfully!")

if __name__ == "__main__":
    ready_input = input("Ready? Type 'yes' to begin: ")
    if ready_input.lower() == 'yes':
        # Step 1: Organize tracks
        move_folders_based_on_bpm_and_camelot(mp3_source_folder, htdemucs_source_folder, destination_folder)
        # Step 2: Generate ALS files
        print("\n[INFO] Starting ALS file generation...\n")
        generate_als_files()
    else:
        print("Operation cancelled.")