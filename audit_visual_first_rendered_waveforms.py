from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from drop_aligner.waveform import WaveformCache


DEFAULT_REPORT = Path(
    "/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/"
    "VISUAL_FIRST_FRESH_ALL_TRACKS_contract_hardened_v31_gui_sparse_pulse_contract_report.json"
)
DEFAULT_CACHE_DIR = Path("/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/.waveform_cache")
DEFAULT_OUT_DIR = Path("/Users/alirahimlou/Desktop/MUSIC/GeneratedSet/VisualFirstFresh/rendered_waveform_audit")

DEFAULT_VIEW_SEC = 6.0
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 220
MARKER_EDGE_TOLERANCE_SEC = 0.080
POST_BODY_SEC = 0.520
MIN_MARKER_DARK_PIXELS = 18
MIN_POST_DARK_PIXELS = 42


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _processed_rows(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rows = report.get("processed_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    rows = report.get("tracks")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _marker_x(marker: float, start: float, end: float, width: int) -> int:
    span = max(float(end) - float(start), 1e-9)
    return max(0, min(int(width) - 1, int(round(((float(marker) - float(start)) / span) * (int(width) - 1)))))


def _dark_mask(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"))
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    return (red < 95) & (green < 112) & (blue < 130)


def _count_dark(mask: np.ndarray, x0: int, x1: int) -> int:
    width = int(mask.shape[1]) if mask.ndim == 2 else 0
    if width <= 0:
        return 0
    left = max(0, min(width, int(x0)))
    right = max(left, min(width, int(x1)))
    if right <= left:
        return 0
    return int(np.count_nonzero(mask[:, left:right]))


def _audit_row(job: Mapping[str, Any]) -> Dict[str, Any]:
    row = job["row"]
    index = int(job["index"])
    width = int(job["width"])
    height = int(job["height"])
    view_sec = float(job["view_sec"])
    marker = _finite_float(row.get("marker"), default=float("nan"))
    drums_path = str(row.get("drums_path") or "")
    selected_by = str(row.get("selected_by") or "")
    base: Dict[str, Any] = {
        "index": index,
        "track": str(row.get("track") or ""),
        "drums_path": drums_path,
        "marker": marker if math.isfinite(marker) else "",
        "selected_by": selected_by,
        "fail_flags": "",
        "marker_dark_pixels": "",
        "post_dark_pixels": "",
        "total_dark_pixels": "",
        "start_sec": "",
        "end_sec": "",
        "marker_x": "",
    }
    if not drums_path or not math.isfinite(marker):
        return {**base, "fail_flags": "missing_marker_or_drums_path"}
    try:
        cache = WaveformCache(Path(str(job["cache_dir"])).expanduser())
        info = cache.info(drums_path)
        duration = _finite_float(info.get("duration"), default=0.0)
        if duration <= 0.0:
            return {**base, "fail_flags": "missing_audio_duration"}
        half = max(0.100, view_sec / 2.0)
        start = max(0.0, float(marker) - half)
        end = min(duration, float(marker) + half)
        if end - start < 0.250:
            end = min(duration, start + 0.250)
        payload = cache.render_png(
            drums_path,
            start_sec=start,
            end_sec=end,
            width=width,
            height=height,
            markers=(),
        )
        image = Image.open(BytesIO(payload))
        dark = _dark_mask(image)
        marker_x = _marker_x(marker, start, end, width)
        span = max(end - start, 1e-9)
        tolerance_px = max(4, int(math.ceil((MARKER_EDGE_TOLERANCE_SEC / span) * width)))
        marker_dark = _count_dark(dark, marker_x - tolerance_px, marker_x + tolerance_px + 1)
        post_px = max(tolerance_px + 1, int(math.ceil((POST_BODY_SEC / span) * width)))
        post_dark = _count_dark(dark, marker_x - tolerance_px, marker_x + post_px + 1)
        total_dark = int(np.count_nonzero(dark))
        fail_flags: List[str] = []
        if total_dark <= 0:
            fail_flags.append("rendered_png_has_no_boom_pixels")
        if marker_dark < MIN_MARKER_DARK_PIXELS:
            fail_flags.append("marker_not_on_rendered_dark_boom_edge")
        if post_dark < MIN_POST_DARK_PIXELS:
            fail_flags.append("no_rendered_post_boom_body_after_marker")
        return {
            **base,
            "fail_flags": ";".join(fail_flags),
            "marker_dark_pixels": int(marker_dark),
            "post_dark_pixels": int(post_dark),
            "total_dark_pixels": int(total_dark),
            "start_sec": f"{start:.6f}",
            "end_sec": f"{end:.6f}",
            "marker_x": int(marker_x),
        }
    except Exception as exc:  # pragma: no cover - CLI defensive boundary
        return {**base, "fail_flags": f"render_audit_error:{str(exc) or exc.__class__.__name__}"}


def _count_flags(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for flag in str(row.get("fail_flags") or "").split(";"):
            flag = flag.strip()
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def run_audit(
    report_path: Path,
    *,
    cache_dir: Path,
    out_dir: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    view_sec: float = DEFAULT_VIEW_SEC,
    workers: int = 1,
) -> Dict[str, Any]:
    report = json.loads(report_path.expanduser().read_text(encoding="utf-8"))
    rows = _processed_rows(report if isinstance(report, Mapping) else {})
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "index": index + 1,
            "row": row,
            "cache_dir": str(cache_dir.expanduser()),
            "width": int(width),
            "height": int(height),
            "view_sec": float(view_sec),
        }
        for index, row in enumerate(rows)
    ]
    results: List[Dict[str, Any]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            results.append(_audit_row(job))
            if index % 25 == 0 or index == len(jobs):
                print(f"render-audited {index}/{len(jobs)}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_audit_row, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index % 25 == 0 or index == len(futures):
                    print(f"render-audited {index}/{len(futures)}", file=sys.stderr, flush=True)
    results.sort(key=lambda row: int(row.get("index") or 0))
    fail_rows = [row for row in results if str(row.get("fail_flags") or "")]
    csv_path = out_dir / f"{report_path.stem}_rendered_waveform_audit.csv"
    fail_csv_path = out_dir / f"{report_path.stem}_rendered_waveform_failures.csv"
    fieldnames = [
        "index",
        "track",
        "marker",
        "selected_by",
        "fail_flags",
        "marker_dark_pixels",
        "post_dark_pixels",
        "total_dark_pixels",
        "start_sec",
        "end_sec",
        "marker_x",
        "drums_path",
    ]
    for path, selected in ((csv_path, results), (fail_csv_path, fail_rows)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
    summary = {
        "source_report": str(report_path.expanduser()),
        "audited_count": len(results),
        "failure_count": len(fail_rows),
        "all_rendered_waveforms_passed": len(fail_rows) == 0,
        "failure_flag_counts": _count_flags(fail_rows),
        "csv": str(csv_path),
        "failure_csv": str(fail_csv_path),
        "width": int(width),
        "height": int(height),
        "view_sec": float(view_sec),
    }
    json_path = out_dir / f"{report_path.stem}_rendered_waveform_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(json_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pixel-audit rendered Boom waveform PNGs around final markers.")
    parser.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--view-sec", type=float, default=DEFAULT_VIEW_SEC)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-no-failures", action="store_true")
    args = parser.parse_args(argv)
    summary = run_audit(
        Path(args.report),
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        width=int(args.width),
        height=int(args.height),
        view_sec=float(args.view_sec),
        workers=int(args.workers),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_no_failures and summary.get("failure_count"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
