#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

from alsdrop.als_io import (
    WarpMarker,
    beat0_sec_from_markers,
    parse_warp_markers,
    read_als_root,
    rewrite_clip_warp_markers,
    validate_als,
    write_als_root,
)


def _make_root_with_clip() -> ET.Element:
    root = ET.Element("Ableton")
    clip = ET.SubElement(root, "AudioClip", {"Id": "10"})
    ET.SubElement(clip, "Name", {"Value": "test"})
    wm = ET.SubElement(clip, "WarpMarkers")
    ET.SubElement(wm, "WarpMarker", {"Id": "11", "SecTime": "2.000000", "BeatTime": "-4.000000"})
    ET.SubElement(wm, "WarpMarker", {"Id": "12", "SecTime": "4.000000", "BeatTime": "0.000000"})
    ET.SubElement(wm, "WarpMarker", {"Id": "13", "SecTime": "6.000000", "BeatTime": "4.000000"})
    return root


def test_beat0_exact():
    markers = [
        WarpMarker(marker_id=1, beat=-4.0, sec=2.0),
        WarpMarker(marker_id=2, beat=0.0, sec=4.0),
        WarpMarker(marker_id=3, beat=4.0, sec=6.0),
    ]
    sec, src = beat0_sec_from_markers(markers)
    assert sec == 4.0
    assert src == "exact"


def test_beat0_interpolation():
    markers = [
        WarpMarker(marker_id=1, beat=-2.0, sec=3.0),
        WarpMarker(marker_id=2, beat=2.0, sec=5.0),
    ]
    sec, src = beat0_sec_from_markers(markers)
    assert abs(float(sec) - 4.0) < 1e-6
    assert src == "interp"


def test_beat0_extrapolation():
    markers = [
        WarpMarker(marker_id=1, beat=4.0, sec=8.0),
        WarpMarker(marker_id=2, beat=8.0, sec=10.0),
    ]
    sec, src = beat0_sec_from_markers(markers)
    # linear extrapolation back to beat 0
    assert abs(float(sec) - 6.0) < 1e-6
    assert src == "extrap"


def test_write_and_validate_roundtrip():
    root = _make_root_with_clip()
    clip = root.find("AudioClip")
    assert clip is not None

    rewrite_clip_warp_markers(root=root, clip=clip, target_sec=12.5, bpm=140.0, duration_sec=180.0)

    check = validate_als(root, expected_target_sec=12.5)
    assert check["ok"] is True
    assert check["beat0_markers"] >= 1

    with tempfile.TemporaryDirectory() as td:
        out_als = os.path.join(td, "test.als")
        write_als_root(out_als, root)
        root2 = read_als_root(out_als)
        clip2 = root2.find("AudioClip")
        assert clip2 is not None
        markers = parse_warp_markers(clip2)
        beat0 = [m for m in markers if abs(float(m.beat)) < 1e-6]
        assert beat0, "BeatTime=0 marker missing after write/read"
