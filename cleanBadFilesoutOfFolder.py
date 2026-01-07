import os
import re
import shutil
from datetime import datetime, date

# ================== CONFIG ==================
ROOT_DIR = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
DRY_RUN = False

# If True: in addition to incomplete track folders, also delete loose audio files
# sitting directly inside BPM/KEY (like your acapella .wav examples).
DELETE_LOOSE_AUDIO_IN_KEY_FOLDER = True

STEM_PREFIXES = ("drums", "inst", "vocals")
AUDIO_EXTS = {".flac", ".wav", ".aif", ".aiff", ".mp3", ".m4a"}

BPM_MIN = 70
BPM_MAX = 180
KEY_RE = re.compile(r"^(?:[1-9]|1[0-2])[AB]$", re.IGNORECASE)
# ============================================


def is_bpm_folder(name: str) -> bool:
    return name.isdigit() and BPM_MIN <= int(name) <= BPM_MAX


def is_key_folder(name: str) -> bool:
    return KEY_RE.match(name) is not None


def list_files_recursive(folder: str):
    """Return list of (filename, full_path) for all files under folder."""
    out = []
    try:
        for root, _, fnames in os.walk(folder):
            for fname in fnames:
                p = os.path.join(root, fname)
                if os.path.isfile(p):
                    out.append((fname, p))
    except (PermissionError, FileNotFoundError):
        pass
    return out


def scan_stems_in_folder_recursive(track_folder: str):
    """
    Looks for drums/inst/vocals by filename prefix anywhere inside folder.
    Uses startswith('drums') etc (not requiring underscore) to match your naming.
    """
    found = {k: False for k in STEM_PREFIXES}
    for name, _ in list_files_recursive(track_folder):
        lower = name.lower()
        for stem in STEM_PREFIXES:
            if lower.startswith(stem):
                found[stem] = True
    return found


def is_track_folder_candidate(folder_path: str) -> bool:
    """
    A folder we should evaluate as a "track folder" if it contains either:
    - CH1.als anywhere inside, or
    - any audio file anywhere inside
    """
    for name, _ in list_files_recursive(folder_path):
        lower = name.lower()
        if lower == "ch1.als":
            return True
        _, ext = os.path.splitext(lower)
        if ext in AUDIO_EXTS:
            return True
    return False


def list_loose_audio_files_in_key_folder(key_path: str):
    """
    Returns list of audio files directly inside BPM/KEY (not inside subfolders).
    """
    loose = []
    try:
        for entry in os.listdir(key_path):
            p = os.path.join(key_path, entry)
            if not os.path.isfile(p):
                continue
            lower = entry.lower()
            _, ext = os.path.splitext(lower)
            if ext in AUDIO_EXTS:
                loose.append((entry, p))
    except (PermissionError, FileNotFoundError):
        pass
    return loose


def is_stem_file_name(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.startswith(stem) for stem in STEM_PREFIXES)


def main():
    folders_to_delete = []  # (path, bpm, key, track_folder_name, missing_list)
    loose_files_to_delete = []  # (path, bpm, key, filename, reason)

    # --- Scan BPM folders ---
    try:
        bpm_entries = os.listdir(ROOT_DIR)
    except Exception as e:
        print(f"❌ Failed to read ROOT_DIR: {ROOT_DIR} ({e})")
        return

    for bpm_name in bpm_entries:
        bpm_path = os.path.join(ROOT_DIR, bpm_name)
        if not os.path.isdir(bpm_path) or not is_bpm_folder(bpm_name):
            continue

        for key_name in os.listdir(bpm_path):
            key_path = os.path.join(bpm_path, key_name)
            if not os.path.isdir(key_path) or not is_key_folder(key_name):
                continue

            # 1) Handle loose audio files directly under BPM/KEY (your acapella case)
            if DELETE_LOOSE_AUDIO_IN_KEY_FOLDER:
                for fname, fpath in list_loose_audio_files_in_key_folder(key_path):
                    # If it's not one of the three stem files, it can never be a full 3-part stem set
                    # (it’s usually an acapella or misc file), so mark for deletion.
                    if not is_stem_file_name(fname):
                        loose_files_to_delete.append((fpath, bpm_name, key_name, fname, "loose-audio-not-stem"))

            # 2) Handle track folders inside BPM/KEY
            for track_name in os.listdir(key_path):
                track_path = os.path.join(key_path, track_name)
                if not os.path.isdir(track_path):
                    continue

                # Only evaluate folders that actually look like track folders (have audio/CH1 somewhere)
                if not is_track_folder_candidate(track_path):
                    continue

                found = scan_stems_in_folder_recursive(track_path)
                has_all_three = all(found.values())
                if has_all_three:
                    continue

                missing = [k for k, v in found.items() if not v]
                folders_to_delete.append((track_path, bpm_name, key_name, track_name, missing))

    if not folders_to_delete and not loose_files_to_delete:
        print("✅ Nothing to delete: no incomplete track folders and no loose audio files matched.")
        return

    if folders_to_delete:
        print("\n⚠️  Incomplete TRACK folders to delete:\n")
        for path, bpm, key, track, missing in folders_to_delete:
            print(f"- {bpm}/{key}/{track}")
            print(f"  path: {path}")
            print(f"  missing: {', '.join(missing)}")

    if loose_files_to_delete:
        print("\n⚠️  Loose AUDIO files directly under BPM/KEY to delete:\n")
        for fpath, bpm, key, fname, reason in loose_files_to_delete:
            print(f"- {bpm}/{key}/{fname}")
            print(f"  path: {fpath}")
            print(f"  reason: {reason}")

    total = len(folders_to_delete) + len(loose_files_to_delete)

    if DRY_RUN:
        print(f"\n🟡 DRY RUN ENABLED — {total} items would be deleted.")
        print("Set DRY_RUN = False to actually delete these folders/files.")
        return

    print("\n🔴 DELETING...\n")

    for path, bpm, key, track, _ in folders_to_delete:
        try:
            shutil.rmtree(path)
            print(f"🗑️  Deleted folder: {bpm}/{key}/{track}")
        except Exception as e:
            print(f"❌ Failed to delete folder {path}: {e}")

    for fpath, bpm, key, fname, _ in loose_files_to_delete:
        try:
            os.remove(fpath)
            print(f"🗑️  Deleted file: {bpm}/{key}/{fname}")
        except Exception as e:
            print(f"❌ Failed to delete file {fpath}: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
