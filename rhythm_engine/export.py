from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .types import RhythmEstimate


def beatgrid_rows(estimate: RhythmEstimate) -> List[Dict[str, Any]]:
    downbeat_set = {round(float(t), 6) for t in estimate.downbeats}
    rows: List[Dict[str, Any]] = []
    bar = 0
    beat_in_bar = 0
    for idx, time_sec in enumerate(estimate.beats):
        is_downbeat = round(float(time_sec), 6) in downbeat_set
        if is_downbeat:
            bar += 1
            beat_in_bar = 1
        elif bar > 0:
            beat_in_bar += 1
        else:
            beat_in_bar = 0
        rows.append(
            {
                "beat_index": int(idx),
                "time_sec": float(time_sec),
                "is_downbeat": bool(is_downbeat),
                "bar": int(bar),
                "beat_in_bar": int(beat_in_bar),
                "bpm": None if estimate.bpm is None else float(estimate.bpm),
                "provider": estimate.provider,
            }
        )
    return rows


def write_beatgrid_csv(estimate: RhythmEstimate, output_path: str) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = beatgrid_rows(estimate)
    fieldnames = ["beat_index", "time_sec", "is_downbeat", "bar", "beat_in_bar", "bpm", "provider"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def write_beatgrid_json(estimate: RhythmEstimate, output_path: str) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": estimate.provider,
        "bpm": estimate.bpm,
        "confidence": estimate.confidence,
        "beats": list(estimate.beats),
        "downbeats": list(estimate.downbeats),
        "rows": beatgrid_rows(estimate),
        "metadata": dict(estimate.metadata),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)
