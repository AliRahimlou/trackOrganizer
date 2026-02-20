#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def median_bpm(rows: Sequence[Dict[str, object]]) -> Optional[float]:
    vals: List[float] = []
    for r in rows:
        x = _f(r.get("bpm"))
        if x is not None and x > 0:
            vals.append(float(x))
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def _cluster_times(sorted_secs: List[float], gap_sec: float) -> List[List[float]]:
    if not sorted_secs:
        return []
    out: List[List[float]] = [[sorted_secs[0]]]
    for s in sorted_secs[1:]:
        if abs(float(s) - float(out[-1][-1])) <= float(gap_sec):
            out[-1].append(float(s))
        else:
            out.append([float(s)])
    return out


def select_primary_target_sec(rows: Sequence[Dict[str, object]]) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Choose one canonical 1.1.1 target for an audio file.
    Strategy:
    - Filter obvious bogus near-zero anchors first.
    - Cluster remaining timestamps by ~1 beat gap.
    - Prefer clusters with most support, then choose earliest cluster.
    """
    raw_vals: List[float] = []
    for r in rows:
        x = _f(r.get("target_sec"))
        if x is not None and x >= 0:
            raw_vals.append(float(x))
    if not raw_vals:
        return None, {"n_raw": 0.0}

    bpm = median_bpm(rows)
    beat_sec = (60.0 / float(bpm)) if bpm and bpm > 0 else 0.5
    min_valid = max(0.75, 1.5 * beat_sec)

    vals = sorted(v for v in raw_vals if v >= min_valid)
    if not vals:
        vals = sorted(raw_vals)
    if not vals:
        return None, {"n_raw": float(len(raw_vals))}

    cluster_gap = max(0.12, 1.0 * beat_sec)
    clusters = _cluster_times(vals, gap_sec=cluster_gap)
    if not clusters:
        return None, {"n_raw": float(len(raw_vals))}

    # Rank by support first, then choose the earliest plausible cluster.
    scored = []
    for c in clusters:
        center = float(np.median(np.asarray(c, dtype=np.float32)))
        scored.append((len(c), center, c))
    scored.sort(key=lambda t: (-int(t[0]), float(t[1])))

    max_n = max(int(t[0]) for t in scored)
    candidate_centers = [float(center) for n, center, _ in scored if int(n) >= max(1, int(round(max_n * 0.5)))]
    chosen = min(candidate_centers) if candidate_centers else float(scored[0][1])
    return float(chosen), {
        "n_raw": float(len(raw_vals)),
        "n_used": float(len(vals)),
        "n_clusters": float(len(clusters)),
        "bpm": float(bpm) if bpm else 0.0,
    }


def collapse_rows_by_audio(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_audio: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        ap = str(r.get("audio_path", "")).strip()
        if not ap:
            continue
        by_audio[ap].append(dict(r))

    out: List[Dict[str, object]] = []
    for ap in sorted(by_audio.keys()):
        group = by_audio[ap]
        target_sec, meta = select_primary_target_sec(group)
        if target_sec is None:
            continue
        bpm = median_bpm(group)
        # Keep representative metadata, but enforce canonical target.
        base = dict(group[0])
        base["audio_path"] = ap
        base["target_sec"] = float(target_sec)
        base["bpm"] = float(bpm) if bpm else base.get("bpm")
        base["target_source"] = "collapsed"
        base["obs"] = int(len(group))
        base["collapse_meta"] = {
            "n_raw": int(meta.get("n_raw", 0)),
            "n_used": int(meta.get("n_used", 0)),
            "n_clusters": int(meta.get("n_clusters", 0)),
        }
        out.append(base)
    return out

