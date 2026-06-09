from __future__ import annotations

import json

from drop_aligner.historical_markers import load_historical_markers, slug_for_path


def test_historical_markers_prefer_latest_correction_over_marker_db(tmp_path):
    marker_db = tmp_path / "drop_marker_db.json"
    correction_log = tmp_path / "drop_corrections.jsonl"
    track = "/music/STEMS/100/2A/Artist - Track/drums_100_2A_7-Artist - Track.flac"
    marker_db.write_text(
        json.dumps(
            {
                "exact": {
                    "drums_100_2a_7-artist - track": {
                        "drop_sec": 12.0,
                        "sample_path": track,
                        "bpm": 100,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    correction_log.write_text(
        "\n".join(
            [
                json.dumps({"track": track, "user_pick": 24.0, "reviewed_from": "web_manual_marker"}),
                json.dumps({"track": track, "user_pick": 38.4, "reviewed_from": "web_candidate_pick"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index = load_historical_markers(correction_logs=[correction_log], marker_db_path=marker_db)
    marker = index.find(track, bpm=100)

    assert marker is not None
    assert marker.user_pick == 38.4
    assert marker.source == "correction_log"


def test_historical_markers_ignore_batch_auto_corrections(tmp_path):
    marker_db = tmp_path / "drop_marker_db.json"
    correction_log = tmp_path / "drop_corrections.jsonl"
    track = "/music/STEMS/140/4A/Artist - Track/drums_140_4A_7-Artist - Track.flac"
    marker_db.write_text(
        json.dumps(
            {
                "exact": {
                    "drums_140_4a_7-artist - track": {
                        "drop_sec": 16.0,
                        "sample_path": track,
                        "bpm": 140,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    correction_log.write_text(
        "\n".join(
            [
                json.dumps({"track": track, "user_pick": 41.142, "reviewed_from": "visual_first_batch_auto"}),
                json.dumps({"track": track, "user_pick": 41.571, "reviewed_from": "batch_auto"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index = load_historical_markers(correction_logs=[correction_log], marker_db_path=marker_db)
    marker = index.find(track, bpm=140)

    assert marker is not None
    assert marker.user_pick == 16.0
    assert marker.source == "drop_marker_db"


def test_historical_markers_ignore_zero_marker_db_entries(tmp_path):
    marker_db = tmp_path / "drop_marker_db.json"
    track = "/music/STEMS/100/2A/Artist - Track/drums_100_2A_7-Artist - Track.flac"
    marker_db.write_text(
        json.dumps(
            {
                "exact": {
                    "drums_100_2a_7-artist - track": {
                        "drop_sec": 0.0,
                        "sample_path": track,
                        "bpm": 100,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    index = load_historical_markers(marker_db_path=marker_db)

    assert index.find(track, bpm=100) is None


def test_slug_for_path_removes_stem_prefix():
    assert slug_for_path("/x/drums_140_2A_7-Klinical - Made It Clear.flac") == "klinical made it clear"
