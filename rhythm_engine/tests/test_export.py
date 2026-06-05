from __future__ import annotations

import json
from pathlib import Path

from rhythm_engine.export import beatgrid_rows, write_beatgrid_csv, write_beatgrid_json
from rhythm_engine.types import RhythmEstimate


def test_beatgrid_rows_marks_bars_and_downbeats() -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5, 1.0, 1.5, 2.0),
        downbeats=(0.0, 2.0),
        bpm=120.0,
        confidence=0.9,
    )

    rows = beatgrid_rows(estimate)

    assert rows[0]["is_downbeat"] is True
    assert rows[0]["bar"] == 1
    assert rows[1]["beat_in_bar"] == 2
    assert rows[4]["is_downbeat"] is True
    assert rows[4]["bar"] == 2


def test_write_beatgrid_csv_and_json(tmp_path: Path) -> None:
    estimate = RhythmEstimate(
        provider="test",
        beats=(0.0, 0.5),
        downbeats=(0.0,),
        bpm=120.0,
        confidence=0.9,
    )
    csv_path = tmp_path / "beats.csv"
    json_path = tmp_path / "beats.json"

    write_beatgrid_csv(estimate, str(csv_path))
    write_beatgrid_json(estimate, str(json_path))

    assert "beat_index,time_sec,is_downbeat" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "test"
    assert payload["beats"] == [0.0, 0.5]
    assert payload["rows"][0]["is_downbeat"] is True
