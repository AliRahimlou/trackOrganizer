#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lxml import etree


DEFAULT_TOLERANCE_SEC = 0.001
ZERO_TOLERANCE = 1e-6
BEAT_TOLERANCE = 0.001


def _value(node: Optional[etree._Element]) -> Optional[str]:
    if node is None:
        return None
    if "Value" in node.attrib:
        return node.attrib.get("Value")
    return node.text


def _marker_get(marker: etree._Element, name: str) -> Optional[float]:
    if name in marker.attrib:
        try:
            return float(marker.attrib[name])
        except (TypeError, ValueError):
            return None
    child = marker.find(name)
    raw = _value(child)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_root(als_path: str) -> etree._Element:
    with gzip.open(als_path, "rb") as fh:
        data = fh.read()
    parser = etree.XMLParser(remove_blank_text=False, recover=False)
    return etree.fromstring(data, parser=parser)


def _audio_clips(root: etree._Element) -> List[etree._Element]:
    return list(root.findall(".//AudioClip"))


def _clip_with_markers(clips: Sequence[etree._Element]) -> Tuple[Optional[etree._Element], Optional[etree._Element]]:
    for clip in clips:
        markers = clip.find("WarpMarkers")
        if markers is not None and markers.findall("WarpMarker"):
            return clip, markers
    if clips:
        return clips[0], clips[0].find("WarpMarkers")
    return None, None


def _marker_values(markers: etree._Element) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for marker in markers.findall("WarpMarker"):
        out.append(
            {
                "sec_time": _marker_get(marker, "SecTime"),
                "beat_time": _marker_get(marker, "BeatTime"),
                "id": marker.attrib.get("Id") or _value(marker.find("Id")),
            }
        )
    return out


def _monotonic(values: Sequence[Optional[float]]) -> bool:
    nums = [float(v) for v in values if v is not None]
    return all(nums[i] <= nums[i + 1] + ZERO_TOLERANCE for i in range(len(nums) - 1))


def _file_ref_texts(clip: etree._Element) -> List[str]:
    refs: List[str] = []
    for node in clip.findall(".//FileRef"):
        for child_name in ("Path", "RelativePath"):
            raw = _value(node.find(child_name))
            if raw and str(raw).strip():
                refs.append(str(raw).strip())
    return refs


def _file_ref_exists(clip: etree._Element, als_path: str) -> Tuple[bool, List[str]]:
    refs = _file_ref_texts(clip)
    if not refs:
        return False, []
    als_dir = Path(als_path).expanduser().resolve().parent
    for ref in refs:
        path = Path(ref).expanduser()
        if path.is_absolute() and path.exists():
            return True, refs
        if not path.is_absolute() and (als_dir / path).exists():
            return True, refs
    return False, refs


def _clip_value_float(clip: etree._Element, path: str) -> Optional[float]:
    raw = _value(clip.find(path))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _close_optional(a: Optional[float], b: Optional[float], tolerance: float = BEAT_TOLERANCE) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= float(tolerance)


def _candidate_json_drop(candidates_json: Optional[str]) -> Optional[float]:
    if not candidates_json:
        return None
    path = Path(candidates_json).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Candidates JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Candidates JSON is not an object: {path}")
    for key in ("user_pick", "corrected_drop_time", "final_ai_pick", "drop_sec", "detected_drop_time"):
        value = payload.get(key)
        if value is None:
            continue
        return float(value)
    raise ValueError(f"Candidates JSON does not contain final_ai_pick/drop_sec: {path}")


def _drop_markers(markers: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        marker
        for marker in markers
        if marker.get("sec_time") is not None
        and abs(float(marker["sec_time"])) > ZERO_TOLERANCE
        and marker.get("beat_time") is not None
        and abs(float(marker["beat_time"])) <= ZERO_TOLERANCE
    ]


def _add_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _clip_anchor_sec(clip: etree._Element) -> Optional[float]:
    """SecTime of the clip's BeatTime-0 warp marker (the 1.1.1 drop anchor)."""
    node = clip.find("WarpMarkers")
    if node is None:
        return None
    for marker in _marker_values(node):
        beat = marker.get("beat_time")
        sec = marker.get("sec_time")
        if beat is not None and abs(float(beat)) <= BEAT_TOLERANCE and sec is not None:
            return float(sec)
    return None


def verify_als(
    als_path: str,
    *,
    candidates_json: Optional[str] = None,
    allow_multiple: bool = False,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "als_path": str(Path(als_path).expanduser()),
        "candidates_json": str(Path(candidates_json).expanduser()) if candidates_json else None,
        "valid": False,
        "checks": checks,
        "errors": [],
        "audio_clip_count": 0,
        "warp_marker_count": 0,
        "drop_marker_count": 0,
        "drop_marker_time": None,
        "candidate_drop_time": None,
        "clip_timing": {},
    }

    try:
        root = _load_root(als_path)
        _add_check(checks, "valid_gzip_xml", True)
    except Exception as exc:
        _add_check(checks, "valid_gzip_xml", False, str(exc) or exc.__class__.__name__)
        report["errors"].append(str(exc) or exc.__class__.__name__)
        return report

    clips = _audio_clips(root)
    report["audio_clip_count"] = len(clips)
    _add_check(checks, "contains_audio_clip", bool(clips), f"audio_clips={len(clips)}")
    clip, markers_node = _clip_with_markers(clips)
    if clip is None:
        report["errors"].append("No AudioClip found.")
        return report

    has_markers = markers_node is not None and bool(markers_node.findall("WarpMarker"))
    _add_check(checks, "contains_warp_markers", bool(has_markers))
    if markers_node is None:
        report["errors"].append("No WarpMarkers node found.")
        return report

    expected_drop: Optional[float] = None
    expected_drop_error: Optional[str] = None
    if candidates_json:
        try:
            expected_drop = float(_candidate_json_drop(candidates_json))
            report["candidate_drop_time"] = float(expected_drop)
        except Exception as exc:
            expected_drop_error = str(exc) or exc.__class__.__name__

    markers = _marker_values(markers_node)
    report["warp_marker_count"] = len(markers)
    zero_markers = [m for m in markers if m["sec_time"] is not None and abs(float(m["sec_time"])) <= ZERO_TOLERANCE]
    zero_drop_markers = [
        m
        for m in zero_markers
        if m.get("beat_time") is not None and abs(float(m["beat_time"])) <= BEAT_TOLERANCE
    ]
    _add_check(checks, "preserves_sec_time_zero_marker", bool(zero_markers), f"zero_markers={len(zero_markers)}")

    drops = _drop_markers(markers)
    expected_zero_drop = bool(expected_drop is not None and abs(float(expected_drop)) <= float(tolerance_sec))
    report["drop_marker_count"] = len(drops) if not expected_zero_drop else len(zero_drop_markers)
    if expected_zero_drop and zero_drop_markers:
        report["drop_marker_time"] = 0.0
    elif drops:
        report["drop_marker_time"] = float(drops[0]["sec_time"])
    if expected_zero_drop:
        if allow_multiple:
            passed = bool(zero_drop_markers)
        else:
            passed = len(zero_drop_markers) == 1
        _add_check(
            checks,
            "contains_zero_drop_marker_for_zero_drop",
            passed,
            f"zero_drop_markers={len(zero_drop_markers)}; expected_drop={float(expected_drop):.9f}",
        )
    elif allow_multiple:
        _add_check(checks, "contains_nonzero_drop_marker", bool(drops), f"drop_markers={len(drops)}")
    else:
        _add_check(checks, "contains_exactly_one_nonzero_drop_marker", len(drops) == 1, f"drop_markers={len(drops)}")

    _add_check(checks, "warp_markers_sec_time_monotonic", _monotonic([m["sec_time"] for m in markers]))
    _add_check(checks, "warp_markers_beat_time_monotonic", _monotonic([m["beat_time"] for m in markers]))

    current_start = _clip_value_float(clip, "CurrentStart")
    current_end = _clip_value_float(clip, "CurrentEnd")
    loop_start = _clip_value_float(clip, "Loop/LoopStart")
    loop_end = _clip_value_float(clip, "Loop/LoopEnd")
    out_marker = _clip_value_float(clip, "Loop/OutMarker")
    hidden_start = _clip_value_float(clip, "Loop/HiddenLoopStart")
    hidden_end = _clip_value_float(clip, "Loop/HiddenLoopEnd")
    zero_sec_beat = None
    marker_beat_values = [m["beat_time"] for m in markers if m.get("beat_time") is not None]
    end_marker_beat = max((float(v) for v in marker_beat_values), default=None)
    for marker in markers:
        sec = marker.get("sec_time")
        beat = marker.get("beat_time")
        if sec is not None and abs(float(sec)) <= ZERO_TOLERANCE and beat is not None:
            zero_sec_beat = float(beat)
            break

    report["clip_timing"] = {
        "current_start": current_start,
        "current_end": current_end,
        "loop_start": loop_start,
        "loop_end": loop_end,
        "out_marker": out_marker,
        "hidden_loop_start": hidden_start,
        "hidden_loop_end": hidden_end,
        "zero_sec_marker_beat": zero_sec_beat,
        "end_marker_beat": end_marker_beat,
    }
    _add_check(
        checks,
        "clip_launch_start_is_bar_one",
        _close_optional(current_start, 0.0) and _close_optional(loop_start, 0.0),
        f"current_start={current_start}; loop_start={loop_start}",
    )
    _add_check(
        checks,
        "loop_out_marker_matches_loop_end",
        _close_optional(out_marker, loop_end),
        f"out_marker={out_marker}; loop_end={loop_end}",
    )
    _add_check(
        checks,
        "clip_current_end_matches_loop_end",
        _close_optional(current_end, loop_end),
        f"current_end={current_end}; loop_end={loop_end}",
    )
    loop_end_after_last_marker = loop_end is not None and end_marker_beat is not None and float(loop_end) + BEAT_TOLERANCE >= float(end_marker_beat)
    _add_check(
        checks,
        "loop_end_covers_last_warp_marker",
        loop_end_after_last_marker,
        f"loop_end={loop_end}; last_marker_beat={end_marker_beat}",
    )
    _add_check(
        checks,
        "hidden_start_matches_sample_zero_phase",
        _close_optional(hidden_start, zero_sec_beat),
        f"hidden_start={hidden_start}; zero_sec_marker_beat={zero_sec_beat}",
    )
    _add_check(
        checks,
        "hidden_loop_window_is_one_bar",
        _close_optional(hidden_end, (hidden_start + 4.0) if hidden_start is not None else None),
        f"hidden_start={hidden_start}; hidden_end={hidden_end}",
    )

    audio_exists, refs = _file_ref_exists(clip, als_path)
    report["audio_file_references"] = refs
    _add_check(checks, "audio_file_reference_exists", bool(audio_exists), "; ".join(refs[:4]))

    # Every stem clip must carry the same BeatTime-0 anchor: the drop lands on
    # 1.1.1 for drums, inst, and vocals alike or the set drifts out of phase.
    clip_anchors = [_clip_anchor_sec(other) for other in clips]
    report["clip_anchor_times"] = clip_anchors
    if len(clips) > 1:
        present = [t for t in clip_anchors if t is not None]
        _add_check(
            checks,
            "all_clips_have_drop_anchor",
            len(present) == len(clips),
            f"anchored_clips={len(present)}/{len(clips)}",
        )
        spread = (max(present) - min(present)) if present else None
        _add_check(
            checks,
            "all_clips_share_drop_anchor",
            spread is not None and float(spread) <= float(tolerance_sec),
            f"anchor_spread_sec={spread}; tolerance_sec={float(tolerance_sec):.9f}",
        )

    if candidates_json:
        try:
            if expected_drop_error:
                raise ValueError(expected_drop_error)
            expected = float(expected_drop) if expected_drop is not None else _candidate_json_drop(candidates_json)
            report["candidate_drop_time"] = float(expected)
            if expected_zero_drop:
                if not zero_drop_markers:
                    _add_check(checks, "drop_marker_matches_candidate_json", False, "no zero drop marker")
                else:
                    nearest_delta = min(abs(float(marker["sec_time"]) - float(expected)) for marker in zero_drop_markers)
                    _add_check(
                        checks,
                        "drop_marker_matches_candidate_json",
                        nearest_delta <= float(tolerance_sec),
                        f"nearest_delta_sec={nearest_delta:.9f}; tolerance_sec={float(tolerance_sec):.9f}",
                    )
            elif not drops:
                _add_check(checks, "drop_marker_matches_candidate_json", False, "no drop marker")
            else:
                nearest_delta = min(abs(float(marker["sec_time"]) - float(expected)) for marker in drops)
                _add_check(
                    checks,
                    "drop_marker_matches_candidate_json",
                    nearest_delta <= float(tolerance_sec),
                    f"nearest_delta_sec={nearest_delta:.9f}; tolerance_sec={float(tolerance_sec):.9f}",
                )
        except Exception as exc:
            _add_check(checks, "drop_marker_matches_candidate_json", False, str(exc) or exc.__class__.__name__)

    report["valid"] = all(check["passed"] for check in checks)
    report["errors"] = [check["detail"] or check["name"] for check in checks if not check["passed"]]
    return report


def _print_report(report: Mapping[str, Any]) -> None:
    status = "PASS" if report.get("valid") else "FAIL"
    print(f"ALS verification: {status}")
    print(f"ALS: {report.get('als_path')}")
    if report.get("candidates_json"):
        print(f"Candidates: {report.get('candidates_json')}")
    print(f"AudioClips: {report.get('audio_clip_count')}")
    print(f"WarpMarkers: {report.get('warp_marker_count')}")
    print(f"Drop markers: {report.get('drop_marker_count')}")
    if report.get("drop_marker_time") is not None:
        print(f"Drop marker time: {float(report['drop_marker_time']):.9f}s")
    for check in report.get("checks", []):
        prefix = "PASS" if check.get("passed") else "FAIL"
        detail = f" - {check.get('detail')}" if check.get("detail") else ""
        print(f"  {prefix}: {check.get('name')}{detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a generated Ableton Live ALS file.")
    parser.add_argument("als", help="Generated *_DROP_ALIGNED.als file")
    parser.add_argument("--candidates-json", help="Optional *_drop_candidates.json to compare drop marker time")
    parser.add_argument("--allow-multiple", action="store_true", help="Allow multiple non-zero BeatTime=0 drop markers")
    parser.add_argument("--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC, help="Candidate JSON time match tolerance")
    parser.add_argument("--json", help="Write verification report JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_als(
        args.als,
        candidates_json=args.candidates_json,
        allow_multiple=bool(args.allow_multiple),
        tolerance_sec=float(args.tolerance_sec),
    )
    _print_report(report)
    if args.json:
        path = Path(args.json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=True)
        print(f"JSON report: {path}")
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
