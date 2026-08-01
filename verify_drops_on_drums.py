#!/usr/bin/env python3
"""Independently verify placed drops against the drums stem.

Reads a build report from build_fresh_visual_first_library_set.py and, for every
production-pass row, checks the written anchor directly against the drums PCM —
sharing no code with the detector's own proof stack:

- low-band (<=150 Hz) and broadband RMS jump across the anchor (a drop must slam)
- distance from the anchor to the nearest local energy onset (must be tight)
- Ali's zero-line rule: sample amplitude at the anchor relative to the local peak

Flags rows that look misplaced so they can be reviewed off the drums waveform,
which is the reference stem for drop placement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

PRE_WINDOW_SEC = (-0.45, -0.05)
POST_WINDOW_SEC = (0.005, 0.405)
ONSET_SEARCH_SEC = 0.035
CONTEXT_SEC = 1.6
LOW_BAND_HZ = 150.0

# Flag thresholds: a real drop entry on drums shows a strong low-band jump and an
# onset within a few milliseconds of the anchor.
MIN_LOW_JUMP = 2.0
MAX_ONSET_DISTANCE_MS = 15.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _window(x: np.ndarray, sr: int, center: int, lo_sec: float, hi_sec: float) -> np.ndarray:
    lo = max(0, center + int(lo_sec * sr))
    hi = min(x.size, center + int(hi_sec * sr))
    return x[lo:hi]


def _nearest_onset_ms(x: np.ndarray, sr: int, center: int) -> Optional[float]:
    """Distance from center to the strongest energy step within the search window."""
    half = int(ONSET_SEARCH_SEC * sr)
    frame = max(1, int(0.005 * sr))
    lo = max(frame * 6, center - half)
    hi = min(x.size - frame, center + half)
    if hi <= lo:
        return None
    energy = np.array([_rms(x[i : i + frame]) for i in range(lo, hi, frame // 2 or 1)])
    if energy.size < 4:
        return None
    steps = energy[2:] / np.maximum(np.minimum.reduce([energy[:-2], energy[1:-1]]), 1e-9)
    best = int(np.argmax(steps)) + 2
    onset_sample = lo + best * (frame // 2 or 1)
    return abs(onset_sample - center) / sr * 1000.0


def verify_row(drums_path: str, anchor_sec: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "drums_path": drums_path,
        "anchor_sec": float(anchor_sec),
        "ok": False,
        "low_jump": None,
        "broad_jump": None,
        "onset_distance_ms": None,
        "zero_line_ratio": None,
        "flags": [],
    }
    path = Path(drums_path)
    if not path.is_file():
        out["flags"].append("drums_missing")
        return out
    try:
        info = sf.info(str(path))
        sr = info.samplerate
        start = max(0, int((anchor_sec - CONTEXT_SEC) * sr))
        frames = int(2 * CONTEXT_SEC * sr)
        audio, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    except Exception as exc:
        out["flags"].append(f"read_error:{exc.__class__.__name__}")
        return out
    mono = audio.mean(axis=1)
    center = int(anchor_sec * sr) - start
    if center <= 0 or center >= mono.size:
        out["flags"].append("anchor_outside_file")
        return out

    sos = butter(4, LOW_BAND_HZ, btype="lowpass", fs=sr, output="sos")
    low = sosfilt(sos, mono)

    pre_low = _rms(_window(low, sr, center, *PRE_WINDOW_SEC))
    post_low = _rms(_window(low, sr, center, *POST_WINDOW_SEC))
    pre_b = _rms(_window(mono, sr, center, *PRE_WINDOW_SEC))
    post_b = _rms(_window(mono, sr, center, *POST_WINDOW_SEC))
    out["low_jump"] = round(post_low / max(pre_low, 1e-9), 3)
    out["broad_jump"] = round(post_b / max(pre_b, 1e-9), 3)

    out["onset_distance_ms"] = _nearest_onset_ms(mono, sr, center)
    local_peak = float(np.max(np.abs(mono[max(0, center - int(0.05 * sr)) : center + int(0.05 * sr)])) or 0.0)
    out["zero_line_ratio"] = round(abs(float(mono[center])) / max(local_peak, 1e-9), 4)

    weak = out["low_jump"] < MIN_LOW_JUMP and out["broad_jump"] < MIN_LOW_JUMP
    far = out["onset_distance_ms"] is not None and out["onset_distance_ms"] > MAX_ONSET_DISTANCE_MS
    if weak:
        out["flags"].append(f"weak_energy_jump:{out['low_jump']}/{out['broad_jump']}")
    if far:
        out["flags"].append(f"onset_far:{out['onset_distance_ms']:.1f}ms")
    # Tiering: an anchor on a tight drums transient is fine even when the
    # buildup keeps drums loud (weak jump alone is a note, not a suspect).
    if weak and far:
        out["tier"] = "HIGH"
    elif far:
        out["tier"] = "MEDIUM"
    elif weak:
        out["tier"] = "NOTE"
    else:
        out["tier"] = "OK"
    out["ok"] = out["tier"] in ("OK", "NOTE")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="Build report JSON from build_fresh_visual_first_library_set.py")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("processed_rows", [])
        if not row.get("production_gate_reasons") and row.get("drums_path")
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[drums-verify] checking {len(rows)} production-pass rows")

    results: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        anchor = row.get("als_anchor", {}).get("drop_sec") if isinstance(row.get("als_anchor"), Mapping) else None
        if anchor is None:
            anchor = row.get("marker")
        result = verify_row(str(row["drums_path"]), float(anchor))
        result["track"] = str((row.get("track") or {}).get("folder") or "")
        results.append(result)
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}")

    flagged = [r for r in results if not r["ok"]]
    out_csv = args.out_csv or Path(args.report).with_name(Path(args.report).stem + "_drums_verify.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["track", "drums_path", "anchor_sec", "tier", "ok", "low_jump", "broad_jump",
                        "onset_distance_ms", "zero_line_ratio", "flags"],
        )
        writer.writeheader()
        tier_order = {"HIGH": 0, "MEDIUM": 1, "NOTE": 2, "OK": 3}
        for r in sorted(results, key=lambda r: (tier_order.get(r.get("tier"), 9), r["track"])):
            writer.writerow({**{k: r.get(k) for k in writer.fieldnames}, "flags": ";".join(r["flags"])})

    from collections import Counter
    tiers = Counter(r.get("tier") for r in results)
    print(f"[drums-verify] tiers={dict(tiers)} -> {out_csv}")
    for r in [r for r in results if r.get("tier") == "HIGH"][:20]:
        print(f"  HIGH {r['track'][:60]:60s} {';'.join(r['flags'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
