import json
import struct
import wave
from pathlib import Path

from ableton_analysis_adapter import extract_ableton_warp_markers


def _write_silent_wav(path: Path, *, sample_rate: int = 1000, frames: int = 1000) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"".join(struct.pack("<h", 0) for _ in range(frames)))


def test_sdk_warp_marker_sidecar_seconds(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    _write_silent_wav(audio)
    sidecar = Path(str(audio) + ".ableton_warp_markers.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "track_organizer_ableton_warp_markers_v1",
                "audioPath": str(audio),
                "warpMarkers": [
                    {"sampleTime": 0.25, "beatTime": 1.0},
                    {"sampleTime": 0.50, "beatTime": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    markers = extract_ableton_warp_markers(str(audio))

    assert markers is not None
    assert markers.source == "sdk_warp_markers"
    assert markers.candidate_seconds == [0.25, 0.5]
    assert markers.candidate_samples == [250, 500]
    assert markers.metadata["sample_time_unit"] == "seconds"


def test_sdk_warp_marker_sidecar_sample_frames(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    _write_silent_wav(audio)
    sidecar = Path(str(audio) + ".ableton_warp_markers.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "track_organizer_ableton_warp_markers_v1",
                "audioPath": str(audio),
                "warpMarkers": [
                    {"sampleTime": 250, "beatTime": 1.0},
                    {"sampleTime": 500, "beatTime": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    markers = extract_ableton_warp_markers(str(audio))

    assert markers is not None
    assert markers.candidate_seconds == [0.25, 0.5]
    assert markers.candidate_samples == [250, 500]
    assert markers.metadata["sample_time_unit"] == "sample_frames"
