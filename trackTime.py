# trackTime.py
import mutagen
from mutagen.flac import FLAC
import os

def get_track_duration(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    try:
        # Load the FLAC file
        audio = FLAC(file_path)
        # Return the duration in seconds
        return audio.info.length
    except mutagen.MutagenError as e:
        raise Exception(f"Could not process the FLAC file - {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")

if __name__ == "__main__":
    file_path = "drums-PHILDEL - The Wolf.flac"
    duration = get_track_duration(file_path)
    print(f"Duration: {duration:.2f} seconds")