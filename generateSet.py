import os
import gzip
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

FLAC_FOLDER = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
TEMPLATE_PATH = "alsFiles/X1 TEMPLATE v7.als"
OUTPUT_PATH = "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/X1 Master Set.als"
BPM_MIN = 80
BPM_MAX = 90

def get_bpm_and_key(path: Path):
    try:
        bpm = int(path.parts[-3])
        key = path.parts[-2]
        return bpm, key
    except Exception:
        return 999, "ZZ"

def extract_sample_paths_from_ch1(ch1_path):
    try:
        with gzip.open(ch1_path, "rb") as f:
            ch1_data = f.read()
        ch1_tree = ET.parse(BytesIO(ch1_data))
        root = ch1_tree.getroot()

        sample_paths = []
        for track in root.findall(".//Tracks/*"):
            clip = track.find(".//Clip")
            if clip is not None:
                sample_ref = clip.find(".//SampleRef")
                if sample_ref is not None:
                    file_ref = sample_ref.find(".//FileRef")
                    if file_ref is not None:
                        name = file_ref.find("Name")
                        if name is not None and "Value" in name.attrib:
                            sample_paths.append(name.attrib["Value"])
        while len(sample_paths) < 3:
            sample_paths.append(sample_paths[-1] if sample_paths else "")
        return sample_paths[:3]
    except Exception as e:
        print(f"[ERROR] reading {ch1_path}: {e}")
        return ["", "", ""]

def update_sample_name(track_elem, scene_index, new_filename):
    device_chain = track_elem.find("DeviceChain")
    if device_chain is None:
        return
    main_seq = device_chain.find("MainSequencer")
    if main_seq is None:
        return
    clip_slots = main_seq.find("ClipSlotList")
    if clip_slots is None:
        return
    try:
        clip_slot = clip_slots.findall("ClipSlot")[scene_index]
        clip = clip_slot.find("Clip")
        if clip is not None:
            sample_ref = clip.find(".//SampleRef")
            if sample_ref is not None:
                name_elem = sample_ref.find(".//FileRef/Name")
                if name_elem is not None:
                    name_elem.attrib["Value"] = new_filename
    except IndexError:
        pass

def inject_into_template(template_path, ch1_folder, output_path):
    with gzip.open(template_path, "rb") as f:
        template_data = f.read()
    tree = ET.parse(BytesIO(template_data))
    root = tree.getroot()

    tracks = root.find(".//Tracks")
    if tracks is None or len(tracks.findall("AudioTrack")) < 4:
        print("❌ Template must have 4 tracks")
        return

    ch1_files = [
        p for p in Path(ch1_folder).rglob("CH1.als")
        if BPM_MIN <= get_bpm_and_key(p)[0] <= BPM_MAX
    ]
    ch1_files.sort(key=lambda p: get_bpm_and_key(p))

    scenes = root.find(".//Scenes") or root.find(".//SceneList")
    if scenes is None:
        print("❌ Template is missing <Scenes> or <SceneList>")
        return
    total_scenes = len(scenes.findall("Scene"))

    for i, ch1_path in enumerate(ch1_files):
        if i >= total_scenes:
            print(f"⚠️ Skipping extra CH1 file (no matching scene): {ch1_path}")
            continue
        sample_paths = extract_sample_paths_from_ch1(str(ch1_path))
        drums, inst, vocals = sample_paths
        update_sample_name(tracks.findall("AudioTrack")[0], i, drums)
        update_sample_name(tracks.findall("AudioTrack")[1], i, inst)
        update_sample_name(tracks.findall("AudioTrack")[2], i, vocals)
        update_sample_name(tracks.findall("AudioTrack")[3], i, vocals)
        print(f"✅ Scene {i+1} updated from {ch1_path.name}")

    final_bytes = BytesIO()
    tree.write(final_bytes, encoding="utf-8", xml_declaration=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f_out:
        f_out.write(final_bytes.getvalue())

    print(f"\n✅ Master ALS saved to: {output_path}")

if __name__ == "__main__":
    print("[INFO] Injecting sample names into X1 TEMPLATE v7.als...")
    inject_into_template(TEMPLATE_PATH, FLAC_FOLDER, OUTPUT_PATH)
