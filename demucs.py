#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demucs batch (GUI-matching config) -> FLAC 24-bit:

GUI parity:
  - Model: htdemucs
  - Device: mps
  - Segment: 7.8s (fallback to 7 if CLI needs int)
  - Overlap: 0.25
  - Shifts: 20
  - Clip mode: rescale
  - Output: FLAC int24 (libsndfile)
  - Naming: /Users/.../toBeOrganized/{track}/{stem}-{track}.flac

3-stem output:
  - drums = drums + bass
  - inst  = other
  - vocals = vocals

Run:  python demucs.py
Reqs: pip install demucs soundfile numpy
"""

import sys
import re
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

# ---------- PATHS ----------
SRC_DIR  = Path("/Users/alirahimlou/Desktop/MUSIC/PlaylistsByDate/music2")
DEST_DIR = Path("/Users/alirahimlou/Desktop/MUSIC/STEMS/toBeOrganized")

# ---------- DEMUCS (matches GUI) ----------
BASE_CMD = [
    "demucs",
    "-d", "mps",
    "-n", "htdemucs",        # GUI shows htdemucs (not _ft)
    "--shifts", "20",
    "--overlap", "0.25",
    "--clip-mode", "rescale",
    "--flac",                # ask demucs to emit FLAC
    "--int24",               # 24-bit
    "-j", "6",
]
RUN_MODES = [
    ["--segment", "7.8"],    # GUI exact
    ["--segment", "7"],      # fallback for CLIs that require integer
]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif"}

# ---------- HELPERS ----------
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str, max_len: int = 200) -> str:
    s = s.strip()
    s = re.sub(r'[\\/:*?"<>|]', "_", s)  # keep '-' and commas
    s = re.sub(r"\s+", " ", s)
    return s[:max_len].strip()

def list_audio_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def _find_track_out_dir(temp_root: Path, src_stem: str) -> Path | None:
    # Demucs will create: temp_root/htdemucs/<trackname>/
    model_dir = temp_root / "htdemucs"
    if model_dir.exists():
        exact = model_dir / src_stem
        if exact.exists(): return exact
        cand = sorted(model_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand: return cand[0]
    cand = sorted(temp_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None

def run_demucs_with_fallback(audio_path: Path, temp_root: Path) -> Path | None:
    last_err = ""
    for mode in RUN_MODES:
        cmd = BASE_CMD + mode + ["-o", str(temp_root), str(audio_path)]
        print(f"→ demucs {audio_path.name} [{' '.join(mode)}]")
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode == 0:
            out_dir = _find_track_out_dir(temp_root, audio_path.stem)
            if out_dir: return out_dir
        last_err = (res.stderr or "") + (res.stdout or "")
        # if 7.8 rejected as int, continue to 7
        if "argument --segment: invalid int value" in last_err:
            continue
    print(f"ERROR running Demucs on {audio_path}:\n{last_err}", file=sys.stderr)
    return None

def find_stem(track_dir: Path, stem: str) -> Path | None:
    for ext in (".flac", ".wav"):  # demucs asked to write flac; be robust anyway
        p = track_dir / f"{stem}{ext}"
        if p.exists(): return p
    hits = list(track_dir.glob(f"{stem}.*"))
    return hits[0] if hits else None

def read_stereo(path: Path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)  # decode to float32 for mixing
    return y, sr

def write_flac24(path: Path, audio_f32: np.ndarray, sr: int):
    """
    Match GUI: FLAC int24, clip-mode=rescale (i.e., prevent integer overflow by scaling).
    We also apply tiny TPDF dither for the float->24-bit step.
    """
    safe_mkdir(path.parent)
    a = np.asarray(audio_f32, dtype=np.float32, order="C")

    # rescale if > 0 dBFS (GUI 'rescale' semantics)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:
        a = a / peak * 0.999999

    # TPDF dither ~1 LSB p-p for 24-bit
    q = 1.0 / (2**23)
    dither = (np.random.random_sample(a.shape) - np.random.random_sample(a.shape)) * q
    a_d = a + dither

    sf.write(str(path), a_d, sr, subtype="PCM_24")

# ---------- CORE ----------
def process_track(audio_path: Path):
    track = safe_name(audio_path.stem)
    print(f"\n=== Processing: {track} ===")

    with tempfile.TemporaryDirectory(prefix="demucs_sep_") as tmp:
        tmp_root = Path(tmp)
        out_dir = run_demucs_with_fallback(audio_path, tmp_root)
        if out_dir is None:
            print(f"ERROR: No Demucs output for {track}", file=sys.stderr)
            return

        # Demucs stems (24-bit FLAC from CLI settings)
        drums_p = find_stem(out_dir, "drums")
        bass_p  = find_stem(out_dir, "bass")
        other_p = find_stem(out_dir, "other")
        vox_p   = find_stem(out_dir, "vocals")

        missing = [n for n, p in [("drums", drums_p), ("bass", bass_p), ("other", other_p), ("vocals", vox_p)] if p is None]
        if missing:
            print(f"WARNING: Missing stems for {track}: {missing}", file=sys.stderr)
            return

        # Read stems as float for mixdown
        drums, sr_d = read_stereo(drums_p)
        bass,  sr_b = read_stereo(bass_p)
        inst,  sr_i = read_stereo(other_p)
        vox,   sr_v = read_stereo(vox_p)
        if not (sr_d == sr_b == sr_i == sr_v):
            raise RuntimeError(f"Sample-rate mismatch among stems for {track}")

        # Length guard
        n = max(len(drums), len(bass), len(inst), len(vox))
        if len(drums) != n: drums = np.pad(drums, ((0, n-len(drums)), (0,0)))
        if len(bass)  != n: bass  = np.pad(bass,  ((0, n-len(bass)),  (0,0)))
        if len(inst)  != n: inst  = np.pad(inst,  ((0, n-len(inst)),  (0,0)))
        if len(vox)   != n: vox   = np.pad(vox,   ((0, n-len(vox)),   (0,0)))

        # drums = drums + bass (no limiting)
        drums_mix = drums + bass

        # Final 3-stem outputs (FLAC 24-bit, GUI naming)
        final_dir = DEST_DIR / track
        drums_out = final_dir / f"drums-{track}.flac"
        inst_out  = final_dir / f"inst-{track}.flac"
        vox_out   = final_dir / f"vocals-{track}.flac"

        write_flac24(drums_out, drums_mix, sr_d)
        write_flac24(inst_out,  inst,      sr_d)
        write_flac24(vox_out,   vox,       sr_d)

        print(f"✅ Wrote:\n  {drums_out}\n  {inst_out}\n  {vox_out}")

def main():
    if not SRC_DIR.exists():
        print(f"Source not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)
    safe_mkdir(DEST_DIR)

    files = list(list_audio_files(SRC_DIR))
    if not files:
        print(f"No audio files found in {SRC_DIR}")
        return

    print(f"Found {len(files)} file(s). Running Demucs (GUI-matching)…")
    for f in files:
        try:
            process_track(f)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"ERROR on {f.name}: {e}", file=sys.stderr)

    print("\nAll done.")

if __name__ == "__main__":
    main()
