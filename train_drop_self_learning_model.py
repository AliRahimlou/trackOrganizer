#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a simple self-learning drop correction model from manual 1.1.1 labels.

Model type:
- kNN over detector feature vectors
- target is correction in beats (manual_drop - detected_drop)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from edm_true_drop_detector import detect_true_drop
from drop_signature_model import extract_drop_signature_from_path


FEATURE_NAMES = [
    "det_conf",
    "stage1_score",
    "struct_score",
    "valid",
    "sustain8",
    "fake",
    "downbeat_offset",
    "tempo_bpm",
]


def _as_float(x) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _clean_bpm(v) -> Optional[int]:
    f = _as_float(v)
    if f is None or not math.isfinite(f) or f <= 0:
        return None
    return int(round(f))


def _resolve_sample_path(sample_path: str, exact_key: str, basename_index: Dict[str, str]) -> Optional[str]:
    if sample_path and os.path.exists(sample_path):
        return sample_path

    base = ""
    if sample_path:
        base = os.path.basename(sample_path).lower()
    if not base:
        # Try common stem extensions.
        for ext in (".flac", ".wav", ".aiff", ".aif", ".mp3"):
            cand = f"{exact_key}{ext}".lower()
            if cand in basename_index:
                return basename_index[cand]
        return None

    return basename_index.get(base)


def _build_basename_index(search_roots: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for root in search_roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for ext in ("*.flac", "*.wav", "*.aiff", "*.aif", "*.mp3"):
            patt = os.path.join(root, "**", ext)
            for p in glob.glob(patt, recursive=True):
                b = os.path.basename(p).lower()
                if b not in idx:
                    idx[b] = p
    return idx


def _feature_vec_from_detection(bpm: int, det) -> List[float]:
    dbg = det.debug or {}
    return [
        float(det.confidence),
        float(_as_float(dbg.get("stage1_score")) or 0.0),
        float(_as_float(dbg.get("struct_score")) or 0.0),
        float(_as_float(dbg.get("valid")) or 0.0),
        float(_as_float(dbg.get("sustain8")) or 0.0),
        float(_as_float(dbg.get("fake")) or 0.0),
        float(_as_float(dbg.get("downbeat_offset")) or 0.0),
        float(bpm),
    ]


def _knn_predict(x: np.ndarray, rows_x: np.ndarray, rows_y: np.ndarray, k: int, skip_idx: Optional[int] = None) -> float:
    dif = rows_x - x[None, :]
    dist = np.sqrt(np.sum(dif * dif, axis=1))
    if skip_idx is not None and 0 <= skip_idx < len(dist):
        dist[skip_idx] = 1e9
    order = np.argsort(dist)
    kk = max(1, min(k, len(order)))
    pick = order[:kk]
    w = 1.0 / (dist[pick] + 1e-4)
    ws = float(np.sum(w))
    if ws <= 1e-9:
        return float(np.mean(rows_y[pick]))
    return float(np.sum(w * rows_y[pick]) / ws)


def train_model(db: Dict[str, object], basename_index: Dict[str, str], max_rows: int, max_sig_rows: int) -> Dict[str, object]:
    exact = db.get("exact", {})
    if not isinstance(exact, dict):
        raise ValueError("DB missing 'exact' map")

    rows_x_raw: List[List[float]] = []
    rows_y_beats: List[float] = []
    rows_meta: List[Dict[str, object]] = []

    # Highest-observation labels first.
    items = []
    for ex_key, rec in exact.items():
        if not isinstance(rec, dict):
            continue
        obs = int(_as_float(rec.get("obs")) or 1)
        items.append((obs, str(ex_key), rec))
    items.sort(key=lambda t: (-t[0], t[1]))

    for _, ex_key, rec in items:
        if len(rows_x_raw) >= max_rows:
            break
        true_drop = _as_float(rec.get("drop_sec"))
        bpm = _clean_bpm(rec.get("bpm"))
        sample_path = str(rec.get("sample_path") or "")
        if true_drop is None or true_drop < 0 or not bpm:
            continue

        resolved = _resolve_sample_path(sample_path, ex_key, basename_index)
        if not resolved or not os.path.exists(resolved):
            continue

        try:
            det = detect_true_drop(
                resolved,
                bpm_hint=float(bpm),
                bpm_min=max(120.0, float(bpm) - 14.0),
                bpm_max=min(160.0, float(bpm) + 14.0),
            )
        except Exception:
            det = None
        if det is None or det.drop_time_sec is None:
            continue

        beat_sec = 60.0 / float(bpm)
        delta_beats = (float(true_drop) - float(det.drop_time_sec)) / beat_sec
        if not math.isfinite(delta_beats):
            continue
        # Keep outliers from polluting kNN.
        if abs(delta_beats) > 32.0:
            continue

        x = _feature_vec_from_detection(bpm, det)
        rows_x_raw.append(x)
        rows_y_beats.append(float(delta_beats))
        rows_meta.append({
            "exact_key": ex_key,
            "audio_path": resolved,
            "true_drop_sec": float(true_drop),
            "det_drop_sec": float(det.drop_time_sec),
            "delta_beats": float(delta_beats),
        })

    if len(rows_x_raw) < 8:
        raise ValueError(f"Not enough labeled rows for model training (got {len(rows_x_raw)}).")

    x_raw = np.asarray(rows_x_raw, dtype=np.float32)
    y = np.asarray(rows_y_beats, dtype=np.float32)
    feat_mean = np.mean(x_raw, axis=0)
    feat_std = np.std(x_raw, axis=0)
    feat_std = np.where(feat_std <= 1e-6, 1.0, feat_std).astype(np.float32)
    x = (x_raw - feat_mean[None, :]) / feat_std[None, :]

    k = min(15, max(5, int(round(math.sqrt(len(x))))))

    # Training diagnostic: leave-one-out MAE.
    errs = []
    for i in range(len(x)):
        yp = _knn_predict(x[i], x, y, k=k, skip_idx=i)
        errs.append(abs(float(yp) - float(y[i])))
    mae = float(np.mean(np.asarray(errs, dtype=np.float32))) if errs else 0.0

    model_rows = []
    for i in range(len(x)):
        model_rows.append({
            "x": [round(float(v), 6) for v in x[i].tolist()],
            "y_beats": round(float(y[i]), 6),
            "exact_key": str(rows_meta[i]["exact_key"]),
        })

    # Build waveform signature bank from manual true drops.
    signature_rows = []
    for row in rows_meta[:max(8, int(max_sig_rows))]:
        p = str(row.get("audio_path") or "")
        bpm = _clean_bpm(exact.get(str(row.get("exact_key")), {}).get("bpm")) if isinstance(exact, dict) else None
        if not p or not os.path.exists(p) or not bpm:
            continue
        sig = extract_drop_signature_from_path(p, center_sec=float(row["true_drop_sec"]), bpm=int(bpm))
        if sig is None:
            continue
        signature_rows.append({
            "exact_key": str(row.get("exact_key") or ""),
            "bpm": int(bpm),
            "sig": [round(float(v), 6) for v in sig.tolist()],
        })

    out = {
        "version": 1,
        "model": "knn_delta_beats",
        "feature_names": FEATURE_NAMES,
        "feature_mean": [round(float(v), 6) for v in feat_mean.tolist()],
        "feature_std": [round(float(v), 6) for v in feat_std.tolist()],
        "k_neighbors": int(k),
        "max_abs_delta_beats": 8.0,
        "rows": model_rows,
        "signature_rows": signature_rows,
        "stats": {
            "train_rows": int(len(model_rows)),
            "leave_one_out_mae_beats": round(mae, 6),
            "signature_rows": int(len(signature_rows)),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train self-learning drop correction model from drop_marker_db.json")
    ap.add_argument("--db", default="drop_marker_db.json", help="Path to drop marker DB JSON.")
    ap.add_argument("--out", default="drop_self_learning_model.json", help="Output model JSON path.")
    ap.add_argument("--search-root", action="append", default=[], help="Root to search for audio files (repeatable).")
    ap.add_argument("--max-rows", type=int, default=700, help="Max labeled tracks to use.")
    ap.add_argument("--max-sig-rows", type=int, default=320, help="Max tracks to add to waveform signature bank.")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"[ERROR] DB not found: {db_path}")
        return 2
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    roots = [os.path.abspath(r) for r in args.search_root if r]
    if not roots:
        # Fallbacks that commonly contain stems in this project.
        roots = [
            "/Users/alirahimlou/Desktop/STEMS2",
            "/Users/alirahimlou/Desktop/MUSIC/STEMS",
        ]
    basename_index = _build_basename_index(roots)
    print(f"[INFO] Indexed {len(basename_index)} audio basenames from {len(roots)} roots.")

    try:
        model = train_model(
            db=db,
            basename_index=basename_index,
            max_rows=max(32, int(args.max_rows)),
            max_sig_rows=max(32, int(args.max_sig_rows)),
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 3

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")

    st = model.get("stats", {})
    print(f"[OK] Wrote model: {out_path}")
    print(f"[OK] train_rows={st.get('train_rows', 0)}")
    print(f"[OK] loo_mae_beats={st.get('leave_one_out_mae_beats', 0)}")
    print(f"[OK] signature_rows={st.get('signature_rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
