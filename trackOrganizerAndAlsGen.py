import os
import shutil
from mutagen.id3 import ID3
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import gzip
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import html
import trackTime  # Assuming this module is available

# Paths
mp3_source_folder = "/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate"
htdemucs_source_folder = "/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized"
destination_folder = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
ALS_FILES_FOLDER = "alsFiles"
FLAC_FOLDER = destination_folder
SKIP_EXISTING = True

# Camelot Wheel Mapping
camelot_wheel = {
    "ABMIN": "1A", "G#MIN": "1A", "EBMIN": "2A", "D#MIN": "2A", "BBMIN": "3A", "A#MIN": "3A",
    "FMIN": "4A", "CMIN": "5A", "GMIN": "6A", "DMIN": "7A", "AMIN": "8A", "EMIN": "9A",
    "BMIN": "10A", "F#MIN": "11A", "GBMIN": "11A", "C#MIN": "12A", "DBMIN": "12A",
    "BMAJ": "1B", "F#MAJ": "2B", "GBMAJ": "2B", "C#MAJ": "3B", "DBMAJ": "3B", "ABMAJ": "4B",
    "G#MAJ": "4B", "EBMAJ": "5B", "D#MAJ": "5B", "BBMAJ": "6B", "A#MAJ": "6B", "FMAJ": "7B",
    "CMAJ": "8B", "GMAJ": "9B", "DMAJ": "10B", "AMAJ": "11B", "EMAJ": "12B",
    "ABMINOR": "1A", "G#MINOR": "1A", "EBMINOR": "2A", "D#MINOR": "2A", "BBMINOR": "3A", "A#MINOR": "3A",
    "FMINOR": "4A", "CMINOR": "5A", "GMINOR": "6A", "DMINOR": "7A", "AMINOR": "8A", "EMINOR": "9A",
    "BMINOR": "10A", "F#MINOR": "11A", "GBMINOR": "11A", "C#MINOR": "12A", "DBMINOR": "12A",
    "BMAJOR": "1B", "F#MAJOR": "2B", "GBMAJOR": "2B", "C#MAJOR": "3B", "DBMAJOR": "3B",
    "ABMAJOR": "4B", "G#MAJOR": "4B", "EBMAJOR": "5B", "D#MAJOR": "5B", "BBMAJOR": "6B",
    "A#MAJOR": "6B", "FMAJOR": "7B", "CMAJOR": "8B", "GMAJOR": "9B", "DMAJOR": "10B",
    "AMAJOR": "11B", "EMAJOR": "12B"
}

def safe_xml_value(s):
    """Escape characters that might break XML in ALS files and convert non-latin1 chars to numeric entities."""
    if not s:
        return s

    result = []
    for char in s:
        code = ord(char)
        if code > 255:
            # Convert to XML numeric entity
            result.append(f"&#{code};")
        elif char == "'":
            result.append("&#39;")
        else:
            result.append(html.escape(char, quote=True))
    return ''.join(result)

def normalize_key(key):
    key = re.sub(r'\s+', '', key.upper())
    key = re.sub(r'MAJOR$', 'MAJ', key)
    key = re.sub(r'MINOR$', 'MIN', key)
    if re.match(r'^[A-G](#|B)?M$', key):
        key = key[:-1] + 'MIN'
    if re.match(r'^[A-G](#|B)?$', key):
        key += 'MAJ'
    return key

def get_bpm_and_key(file_path):
    try:
        if file_path.lower().endswith('.mp3'):
            audio = ID3(file_path)
            bpm = audio.get('TBPM')
            bpm = bpm.text[0] if bpm else 'Unknown BPM'
            key = audio.get('TKEY')
            key = key.text[0].strip() if key else 'Unknown Key'
        elif file_path.lower().endswith('.flac'):
            audio = FLAC(file_path)
            bpm = audio.get('BPM', ['Unknown BPM'])[0]
            key = audio.get('KEY', audio.get('INITIALKEY', ['Unknown Key']))[0]
        elif file_path.lower().endswith('.wav'):
            audio = WAVE(file_path)
            if audio.tags is None:
                return 'Unknown BPM', 'Unknown Key'
            bpm = audio.tags.get('TBPM')
            bpm = bpm.text[0] if bpm else 'Unknown BPM'
            key = audio.tags.get('TKEY')
            key = key.text[0].strip() if key else 'Unknown Key'
        else:
            return 'Unknown BPM', 'Unknown Key'
        return bpm, key
    except Exception as e:
        print(f"[DEBUG] Metadata error {file_path}: {e}")
        return 'Unknown BPM', 'Unknown Key'

def convert_to_camelot(key):
    if key == "Unknown Key":
        return "Unknown Key"
    normalized = normalize_key(key)
    camelot_key = camelot_wheel.get(normalized, "Unknown Key")
    if camelot_key == "Unknown Key":
        print(f"[DEBUG] Unknown normalized key: {normalized} (original: {key})")
    return camelot_key

def rename_and_move_files(source_folder, bpm, camelot_key):
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
    print(f"\n[INFO] Organizing folders from '{htdemucs_folder}' to '{dest_folder}'\n")
    processed_tracks = []
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for root, _, files in os.walk(mp3_folder):
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
                source_track_folder = os.path.join(htdemucs_folder, track_name)
                if os.path.exists(source_track_folder):
                    rename_and_move_files(source_track_folder, bpm, camelot_key)
                    os.makedirs(camelot_folder, exist_ok=True)
                    target_folder = os.path.join(camelot_folder, track_name)
                    shutil.move(source_track_folder, target_folder)
                    print(f"[INFO] Moved '{track_name}' → '{target_folder}'")
    print("\n[INFO] Track organization complete.\n")

def extract_bpm_from_path(folder_path):
    parts = folder_path.split(os.sep)
    try:
        return int(parts[-3])
    except (IndexError, ValueError):
        return None

def find_flac_folders(directory):
    results = []
    for root, _, files in os.walk(directory):
        flac_files = [f for f in files if f.lower().endswith(".flac")]
        if flac_files:
            track_names = {"drums": None, "Inst": None, "vocals": None}
            for f in flac_files:
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                if "drums" in f.lower() and not track_names["drums"]:
                    track_names["drums"] = rel_path
                elif "inst" in f.lower() and not track_names["Inst"]:
                    track_names["Inst"] = rel_path
                elif "vocals" in f.lower() and not track_names["vocals"]:
                    track_names["vocals"] = rel_path
            bpm_value = extract_bpm_from_path(root)
            if any(track_names.values()):
                results.append((root, track_names, bpm_value))
    return results

def select_blank_als(bpm_value):
    if bpm_value:
        bpm_als_path = os.path.join(ALS_FILES_FOLDER, f"{bpm_value}.als")
        if os.path.exists(bpm_als_path):
            return bpm_als_path
    return None

def get_duration_in_beats(track_path, bpm):
    try:
        duration_seconds = trackTime.get_track_duration(track_path)
        duration_beats = (duration_seconds * bpm) / 60
        return f"{duration_beats:.6f}"
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None

def modify_als_file(input_path, target_folder, track_names, bpm_value):
    if input_path is None:
        return
    output_als = os.path.join(target_folder, "CH1.als")
    if os.path.exists(output_als) and SKIP_EXISTING:
        return
    print(f"\n🎯 Creating ALS for {target_folder} (BPM {bpm_value})")

    shutil.copy(input_path, output_als)
    new_loop_end = None
    if track_names["drums"] and bpm_value:
        flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
        new_loop_end = get_duration_in_beats(flac_path, bpm_value)

    with gzip.open(output_als, "rb") as f:
        als_data = f.read()

    tree = ET.parse(BytesIO(als_data))
    root = tree.getroot()

    if new_loop_end:
        for elem in root.iter():
            if elem.tag in ("LoopEnd", "OutMarker"):
                elem.set("Value", new_loop_end)

    with gzip.open(output_als, "wb") as f_out:
        tree.write(f_out, encoding="utf-8", xml_declaration=True)

    with gzip.open(output_als, "rb") as f:
        als_str = f.read().decode("latin1")

    for role, rel_path in track_names.items():
        if rel_path:
            old = f"{role}-Tape B - i won't be ur drug.flac"
            new = safe_xml_value(rel_path)
            als_str = als_str.replace(old, new).replace(old.replace(" ", "%20"), new.replace(" ", "%20"))

            old_name = old.replace(".flac", "")
            new_name = safe_xml_value(os.path.basename(rel_path).replace(".flac", ""))
            for tag in ["MemorizedFirstClipName", "UserName", "Name", "EffectiveName"]:
                als_str = re.sub(rf'(<{tag} Value="){re.escape(old_name)}(")', rf'\1{new_name}\2', als_str)

    with gzip.open(output_als, "wb") as f:
        f.write(als_str.encode("latin1"))
    print(f"✅ ALS saved: {output_als}")

def generate_als_files():
    folders = find_flac_folders(FLAC_FOLDER)
    if not folders:
        print("❌ No FLAC tracks found.")
        return
    for folder, track_names, bpm_value in folders:
        blank_als = select_blank_als(bpm_value)
        modify_als_file(blank_als, folder, track_names, bpm_value)
    print("🎵 ALS generation complete.")

if __name__ == "__main__":
    if input("Ready? Type 'yes' to begin: ").lower() == 'yes':
        move_folders_based_on_bpm_and_camelot(mp3_source_folder, htdemucs_source_folder, destination_folder)
        generate_als_files()
    else:
        print("Operation cancelled.")