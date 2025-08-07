import os
import shutil

stems_dir = "/Users/alirahimlou/Desktop/MUSIC/STEMS"
output_dir = "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/stuff"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through BPM folders from 70 to 155
for bpm in range(70, 156):
    bpm_folder = os.path.join(stems_dir, str(bpm))
    if os.path.isdir(bpm_folder):
        for root, dirs, files in os.walk(bpm_folder):
            for file in files:
                if file == "CH1.als":
                    src = os.path.join(root, file)
                    # Get the track folder name
                    track_name = os.path.basename(root).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "").replace("&", "and")
                    # Get the key folder if exists
                    key_dir = os.path.basename(os.path.dirname(root))
                    dst_filename = f"{bpm}_{key_dir}_{track_name}_CH1.als"
                    dst = os.path.join(output_dir, dst_filename)
                    shutil.copy(src, dst)
                    print(f"Copied {src} to {dst}")