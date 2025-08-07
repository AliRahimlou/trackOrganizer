import os
import shutil
from mutagen.id3 import ID3, ID3NoHeaderError, error as ID3Error

# Prompt the user for input
date_folder = input("Enter the date folder (e.g., 1-25-25): ")
# Source and destination paths
mp3_source_folder = f"/Users/alirahimlou/Desktop/PlaylistsByDate/{date_folder}"
# mp3_source_folder = "/Users/alirahimlou/Desktop/PlaylistsByDate/1-25-25"
htdemucs_source_folder = "/Users/alirahimlou/Desktop/STEMS/separated/htdemucs/toBeOrganized"
destination_folder = "/Users/alirahimlou/Desktop/STEMS/separated/htdemucs"

# Camelot Wheel Mapping (Final Fix)
camelot_wheel = {
    "Amin": "8A", "A#min": "9A", "Bbmin": "3A", "Bmin": "10A", "Cmin": "5A",
    "C#min": "12A", "Dbmin": "12A", "Dmin": "7A", "D#min": "2A", "Ebmin": "2A",
    "Emin": "9A", "Fmin": "4A", "F#min": "11A", "Gbmin": "11A", "Gmin": "6A",
    "G#min": "3A", "Abmin": "3A", "Cmaj": "8B", "C#maj": "9B", "Dbmaj": "9B",
    "Dmaj": "10B", "D#maj": "11B", "Ebmaj": "11B", "Emaj": "12B", "Fmaj": "7B",
    "F#maj": "2B", "Gbmaj": "2B", "Gmaj": "9B", "G#maj": "4B", "Abmaj": "4B",
    "Amaj": "11B", "A#maj": "6B", "Bbmaj": "6B", "Bmaj": "1B"
}

# Fix for ambiguous keys (ensuring Eb → 2A)
ambiguous_keys = {
    "Eb": "2A",  # Default to minor
    "Ebm": "2A",  # Explicitly minor
    "Ebmaj": "11B",  # Explicitly major
    "B": "1B",  # Assume major
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

    # Check for ambiguous key cases
    return ambiguous_keys.get(key, camelot_wheel.get(key, "Unknown Key"))

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

            # Define the new folder structure
            bpm_folder = os.path.join(dest_folder, str(bpm))
            camelot_folder = os.path.join(bpm_folder, camelot_key)

            # Find corresponding folder in htdemucs
            track_name = os.path.splitext(file)[0]  # Remove .mp3 extension
            source_track_folder = os.path.join(htdemucs_source_folder, track_name)

            if os.path.exists(source_track_folder):
                # Ensure target directories exist
                os.makedirs(camelot_folder, exist_ok=True)

                # Move the folder
                target_folder = os.path.join(camelot_folder, track_name)
                shutil.move(source_track_folder, target_folder)
                print(f"[INFO] Moved '{track_name}' → '{target_folder}'")

    # Print the summary table
    print("\n[INFO] Processing Summary\n")
    print(f"{'Track Name':<40} {'BPM':<10} {'Camelot Key':<10}")
    print("="*80)
    for track, bpm, key, camelot in processed_tracks:
        print(f"{track[:37]:<40} {bpm:<10} {camelot:<10}")
    print("="*80)
    print("\n[INFO] Processing complete.\n")

if __name__ == "__main__":
    move_folders_based_on_bpm_and_camelot(mp3_source_folder, htdemucs_source_folder, destination_folder)
