import os

# Path to the testing directory
testing_folder = "/Users/alirahimlou/Desktop/STEMS/separated/testing"

# Camelot Wheel Mapping (Ensuring Eb is included correctly)
camelot_wheel = {
    "Amin": "8A", "A#min": "9A", "Bbmin": "3A", "Bmin": "10A", "Cmin": "5A",
    "C#min": "12A", "Dbmin": "12A", "Dmin": "7A", "D#min": "2A", "Ebmin": "2A",
    "Emin": "9A", "Fmin": "4A", "F#min": "11A", "Gbmin": "11A", "Gmin": "6A",
    "G#min": "3A", "Abmin": "3A", "Cmaj": "8B", "C#maj": "9B", "Dbmaj": "9B",
    "Dmaj": "10B", "D#maj": "11B", "Ebmaj": "11B", "Emaj": "12B", "Fmaj": "7B",
    "F#maj": "2B", "Gbmaj": "2B", "Gmaj": "9B", "G#maj": "4B", "Abmaj": "4B",
    "Amaj": "11B", "A#maj": "6B", "Bbmaj": "6B", "Bmaj": "1B"
}

def convert_to_camelot(folder_name):
    """Convert key folder names to Camelot notation, assuming default major for ambiguous keys."""
    normalized_key = folder_name.strip().replace("m", "min").replace("#", "#")

    # Handle ambiguous "Eb" case
    if normalized_key == "Eb":
        normalized_key = "Ebmaj"  # Default to major if no minor indicator

    return camelot_wheel.get(normalized_key, None)

def rename_key_folders(directory):
    """Rename folders inside /testing based on Camelot Key."""
    for bpm_folder in os.listdir(directory):
        bpm_path = os.path.join(directory, bpm_folder)

        if not os.path.isdir(bpm_path):
            continue  # Skip files, only process folders

        for key_folder in os.listdir(bpm_path):
            key_path = os.path.join(bpm_path, key_folder)

            if os.path.isdir(key_path):  # Ensure it's a folder
                camelot_key = convert_to_camelot(key_folder)

                if camelot_key:
                    new_key_path = os.path.join(bpm_path, camelot_key)

                    if not os.path.exists(new_key_path):  # Avoid overwriting
                        os.rename(key_path, new_key_path)
                        print(f"[INFO] Renamed: {key_folder} → {camelot_key}")
                    else:
                        print(f"[WARNING] Skipping {key_folder}, {camelot_key} already exists.")

if __name__ == "__main__":
    rename_key_folders(testing_folder)
