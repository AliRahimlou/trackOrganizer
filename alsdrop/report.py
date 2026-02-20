#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import os
from typing import Dict, List, Sequence

from .utils import as_float


def _hist(vals: List[float], bins: int = 20) -> List[Dict[str, object]]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [{"start": lo, "end": hi, "count": len(vals)}]
    w = (hi - lo) / float(bins)
    counts = [0 for _ in range(bins)]
    for v in vals:
        i = int((v - lo) / w)
        if i >= bins:
            i = bins - 1
        if i < 0:
            i = 0
        counts[i] += 1
    rows = []
    for i, c in enumerate(counts):
        a = lo + (i * w)
        b = lo + ((i + 1) * w)
        rows.append({"start": a, "end": b, "count": c})
    return rows


def write_dataset_report(rows: Sequence[Dict[str, object]], out_html: str) -> None:
    vals = [float(v) for v in [as_float(r.get("target_sec")) for r in rows] if v is not None]
    bpm_vals = [float(v) for v in [as_float(r.get("bpm_hint")) for r in rows] if v and v > 0]
    src_count: Dict[str, int] = {}
    for r in rows:
        md = r.get("metadata")
        src = "unknown"
        if isinstance(md, dict):
            src = str(md.get("target_source") or "unknown")
        src_count[src] = src_count.get(src, 0) + 1

    hist = _hist(vals, bins=24)
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)

    lines: List[str] = []
    lines.append("<html><head><meta charset='utf-8'><title>ALS Drop Dataset Report</title>")
    lines.append("<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px}table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #333}.bar{height:10px;background:#4fc3f7}</style>")
    lines.append("</head><body>")
    lines.append("<h1>ALS Drop Dataset Report</h1>")
    lines.append(f"<p>Total labels: <b>{len(rows)}</b></p>")
    if vals:
        lines.append(
            f"<p>Target sec min/median/max: <b>{min(vals):.3f}</b> / <b>{sorted(vals)[len(vals)//2]:.3f}</b> / <b>{max(vals):.3f}</b></p>"
        )
    if bpm_vals:
        lines.append(
            f"<p>BPM hint min/median/max: <b>{min(bpm_vals):.1f}</b> / <b>{sorted(bpm_vals)[len(bpm_vals)//2]:.1f}</b> / <b>{max(bpm_vals):.1f}</b></p>"
        )

    lines.append("<h2>Target Source</h2><table><tr><th>Source</th><th>Count</th></tr>")
    for k, v in sorted(src_count.items(), key=lambda t: (-t[1], t[0])):
        lines.append(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>")
    lines.append("</table>")

    if hist:
        max_count = max(int(h["count"]) for h in hist) or 1
        lines.append("<h2>target_sec Histogram</h2><table><tr><th>Range (s)</th><th>Count</th><th>Density</th></tr>")
        for h in hist:
            c = int(h["count"])
            w = int(round((c / max_count) * 240))
            lines.append(
                f"<tr><td>{h['start']:.2f} - {h['end']:.2f}</td><td>{c}</td><td><div class='bar' style='width:{w}px'></div></td></tr>"
            )
        lines.append("</table>")

    lines.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
