import os
import shutil
import gzip
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import trackTime  # Import the trackTime module

# 🛠 CONFIG: Paths
ALS_FILES_FOLDER = "alsFiles"  # Folder where BPM ALS templates are stored
FLAC_FOLDER = "/Users/alirahimlou/Desktop/STEMS"

# ✅ CONFIG: Skip or overwrite existing ALS files
SKIP_EXISTING = True  # Set to False if you want to overwrite existing ALS files

def find_flac_folders(directory):
    """
    Recursively search for folders in `directory` that contain .flac files.
    Returns a list of tuples: (folder_path, track_names, bpm_value).
    """
    results = []
    for root, dirs, files in os.walk(directory):
        flac_files = sorted([f for f in files if f.lower().endswith(".flac")])
        if flac_files:
            track_names = {
                "drums": None,
                "Inst": None,
                "vocals": None
            }
            for f in flac_files:
                rel_path = os.path.relpath(os.path.join(root, f), directory)  # Compute relative path
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
    """
    Extracts the BPM from the folder structure.
    Example: /Users/.../STEMS/133/5A/TrackName -> BPM = 133
    """
    parts = folder_path.split(os.sep)
    try:
        bpm_value = int(parts[-3])  # Extract BPM from third level from the end
        return bpm_value
    except (IndexError, ValueError):
        print(f"   Warning: Could not extract BPM from path '{folder_path}'.")
        return None

def select_blank_als(bpm_value):
    """
    Dynamically selects the correct blank ALS file based on BPM value.
    """
    if bpm_value:
        bpm_als_path = os.path.join(ALS_FILES_FOLDER, f"{bpm_value}.als")
        if os.path.exists(bpm_als_path):
            return bpm_als_path
    print(f"⚠️ Warning: No ALS file found for BPM {bpm_value}. Skipping...")
    return None

def get_duration_in_beats(track_path, bpm):
    """
    Gets the duration of the drums FLAC file in beats using the provided BPM.
    """
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
    """
    Loads the selected ALS file, replaces FLAC references, updates <LoopEnd> and <OutMarker> with the drums track duration,
    and saves the ALS as "CH1.als" in the target folder.
    """
    try:
        if input_path is None:
            print(f"❌ Skipping folder '{target_folder}' due to missing ALS template.")
            return

        output_als = os.path.join(target_folder, "CH1.als")

        if os.path.exists(output_als) and SKIP_EXISTING:
            print(f"⏭️ Skipping '{target_folder}' – CH1.als already exists.")
            return

        shutil.copy(input_path, output_als)

        # Get the drums duration in beats (if available)
        new_loop_end = None
        if track_names["drums"] and bpm_value:
            flac_path = os.path.join(FLAC_FOLDER, track_names["drums"])
            new_loop_end = get_duration_in_beats(flac_path, bpm_value)
        else:
            print("   No drums track or BPM value available; skipping LoopEnd modification.")

        # Modify the ALS file using XML parsing
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

        # FLAC reference replacements
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

if __name__ == "__main__":
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