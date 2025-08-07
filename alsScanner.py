import gzip
import re

file_path = "/Users/alirahimlou/myapps/trackOrganizer/alsFiles/141.als"

def scan_als_for_138(path):
    try:
        with gzip.open(path, "rb") as f:
            raw_data = f.read()
        decoded_data = raw_data.decode("latin1", errors="replace")

        matches = list(re.finditer(r".{0,50}14.{0,50}", decoded_data))
        if not matches:
            print("✅ No matches for '138' found.")
        else:
            print(f"🔍 Found {len(matches)} matches for '138':\n")
            for i, match in enumerate(matches, 1):
                context = match.group().replace("\n", "\\n")
                print(f"{i:02d}. ...{context}...")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    scan_als_for_138(file_path)
