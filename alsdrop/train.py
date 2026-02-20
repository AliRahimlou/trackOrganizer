#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .audio_features import load_feature_cache
from .candidates import CandidateSet, candidate_cache_path, generate_candidates, snap_to_nearest_downbeat
from .constants import DEFAULT_MAX_OFFSET_SEC, DEFAULT_SEED
from .model import (
    ContextConfig,
    build_candidate_contexts,
    build_model,
    calibrate_temperature,
    listwise_softmax_loss,
    pairwise_ranking_loss,
    sigmoid_np,
    temperature_scale_logits,
)
from .utils import as_float, iter_jsonl, seeded, write_json


def _require_torch():
    try:
        import torch
        from torch.cuda.amp import GradScaler, autocast
    except Exception as e:
        raise RuntimeError("PyTorch is required for training. Install with: pip install torch") from e
    return torch, GradScaler, autocast


def _device_auto(torch, device_arg: str) -> str:
    if device_arg != "auto":
        return str(device_arg)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _median(vals: Sequence[float], default: float = 0.0) -> float:
    if not vals:
        return float(default)
    arr = sorted(float(v) for v in vals)
    return float(arr[len(arr) // 2])


def _candidate_meta(feature_dict: Dict[str, np.ndarray], cand_times: np.ndarray, cand_conf: np.ndarray, tempo_bpm: float) -> np.ndarray:
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)
    dur = float(frame_times[-1]) if frame_times.size else 1.0
    onset = feature_dict.get("onset", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    low_ratio = feature_dict.get("low_ratio", np.zeros(frame_times.shape[0], dtype=np.float32)).astype(np.float32, copy=False)
    n = int(cand_times.shape[0])
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out

    out[:, 0] = cand_conf[:n].astype(np.float32, copy=False) if cand_conf.size else 0.0
    out[:, 1] = cand_times.astype(np.float32, copy=False) / max(1e-6, dur)

    if frame_times.size and low_ratio.size:
        out[:, 2] = np.interp(cand_times, frame_times, low_ratio, left=low_ratio[0], right=low_ratio[-1]).astype(np.float32, copy=False)
    if frame_times.size and onset.size:
        out[:, 3] = np.interp(cand_times, frame_times, onset, left=onset[0], right=onset[-1]).astype(np.float32, copy=False)
    if tempo_bpm > 0:
        out[:, 0] = np.clip(out[:, 0] + (float(tempo_bpm) / 220.0) * 0.05, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


@dataclass
class TrackItem:
    audio_path: str
    feature_cache: str
    target_sec: float
    bpm_hint: float
    duration_sec: float
    cand_times: np.ndarray
    cand_conf: np.ndarray
    downbeats: np.ndarray
    tempo_bpm: float
    labels: np.ndarray
    offset_targets: np.ndarray
    meta: np.ndarray
    pos_index: int
    nearest_candidate_error_sec: float
    sample_weight: float
    role: str


@dataclass
class BuildStats:
    scanned: int = 0
    with_features: int = 0
    with_target: int = 0
    with_candidates: int = 0
    candidate_hit: int = 0
    candidate_miss: int = 0
    skipped_missing_candidate: int = 0


class FeatureTensorLRU:
    def __init__(self, context_cfg: ContextConfig, capacity: int = 16):
        self.cfg = context_cfg
        self.capacity = max(4, int(capacity))
        self._store: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, key: str, feature_cache: str, cand_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if key in self._store:
            v = self._store.pop(key)
            self._store[key] = v
            return v

        feat = load_feature_cache(feature_cache)
        long_ctx, short_ctx = build_candidate_contexts(feat, cand_times.tolist(), cfg=self.cfg)

        if len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = (long_ctx, short_ctx)
        return long_ctx, short_ctx


def _role_from_audio_path(path: str) -> str:
    b = os.path.basename(path).lower()
    if b.startswith("drums_"):
        return "drums"
    if b.startswith("inst_"):
        return "inst"
    if b.startswith("vocals_"):
        return "vocals"
    return "other"


def _load_dataset_rows(dataset_jsonl: str, source_weight: float = 1.0, source_name: str = "primary") -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in iter_jsonl(dataset_jsonl):
        r = dict(row)
        r["_sample_weight"] = float(max(0.0, source_weight))
        r["_source_name"] = str(source_name)
        out.append(r)
    return out


def _load_manifest(manifest_jsonl: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in iter_jsonl(manifest_jsonl):
        ap = str(r.get("audio_path", "")).strip()
        cp = str(r.get("cache_path", "")).strip()
        if ap and cp:
            m[os.path.abspath(ap)] = os.path.abspath(cp)
    return m


def _choose_target(group: Sequence[Dict[str, object]]) -> Optional[float]:
    vals = [float(v) for v in [as_float(r.get("target_sec")) for r in group] if v is not None and v > 0]
    if not vals:
        return None
    return _median(vals)


def _choose_bpm(group: Sequence[Dict[str, object]], feature_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
    vals = [float(v) for v in [as_float(r.get("bpm_hint")) for r in group] if v is not None and v > 0]
    if vals:
        return _median(vals)
    if feature_dict is not None and "tempo_est" in feature_dict and len(feature_dict["tempo_est"]):
        t = float(feature_dict["tempo_est"][0])
        if t > 0:
            return t
    return 128.0


def _choose_sample_weight(group: Sequence[Dict[str, object]], default: float = 1.0) -> float:
    vals = [float(v) for v in [as_float(r.get("_sample_weight")) for r in group] if v is not None and v > 0]
    if not vals:
        return float(default)
    # If same audio appears in GOLD and SILVER, prefer stricter label weight.
    return float(max(vals))


def _group_key_for_path(audio_path: str) -> str:
    """
    Group-level key for split leakage control.
    Uses folder + normalized track stem so remixes/duplicates are less likely
    to leak across train/val/test.
    """
    ap = os.path.abspath(audio_path)
    folder = os.path.basename(os.path.dirname(ap)).strip().lower()
    stem = os.path.splitext(os.path.basename(ap))[0].strip().lower()
    # Strip common stem prefixes: drums_140_2A_7-Title -> Title
    if "-" in stem and "_" in stem:
        left, right = stem.split("-", 1)
        if left.startswith(("drums_", "inst_", "vocals_")) and right.strip():
            stem = right.strip()
    return f"{folder}::{stem}"


def _split_grouped(paths: Sequence[str], seed: int = 42, train: float = 0.8, val: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    by_group: Dict[str, List[str]] = {}
    for p in paths:
        by_group.setdefault(_group_key_for_path(p), []).append(p)

    groups = sorted(by_group.keys())
    rng = random.Random(int(seed))
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(round(float(train) * n)))
    n_val = max(1, int(round(float(val) * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)

    g_train = set(groups[:n_train])
    g_val = set(groups[n_train : n_train + n_val])
    g_test = set(groups[n_train + n_val : n_train + n_val + n_test])
    if not g_test and groups:
        g_test = {groups[-1]}

    train_items: List[str] = []
    val_items: List[str] = []
    test_items: List[str] = []
    for g, ps in by_group.items():
        if g in g_train:
            train_items.extend(ps)
        elif g in g_val:
            val_items.extend(ps)
        else:
            test_items.extend(ps)

    # Keep deterministic ordering for reproducibility.
    return sorted(train_items), sorted(val_items), sorted(test_items)


def _build_tracks(
    dataset_rows: List[Dict[str, object]],
    feature_manifest: Dict[str, str],
    candidates_dir: str,
    use_madmom: bool,
    force_candidates: bool,
    max_offset_sec: float,
    max_candidates: int,
    candidate_tol_sec: float,
    require_candidate_hit: bool,
    limit: int = 0,
) -> Tuple[List[TrackItem], BuildStats]:
    by_audio: Dict[str, List[Dict[str, object]]] = {}
    for r in dataset_rows:
        ap = str(r.get("audio_path", "")).strip()
        if not ap:
            continue
        by_audio.setdefault(os.path.abspath(ap), []).append(r)

    items = sorted(by_audio.items())
    if limit and limit > 0:
        items = items[: int(limit)]

    total = len(items)
    out: List[TrackItem] = []
    stats = BuildStats(scanned=int(total))
    os.makedirs(os.path.abspath(candidates_dir), exist_ok=True)
    for i, (ap, group) in enumerate(items, start=1):
        feature_cache = feature_manifest.get(ap)
        if not feature_cache or (not os.path.isfile(feature_cache)):
            continue
        stats.with_features += 1

        target = _choose_target(group)
        if target is None:
            continue
        stats.with_target += 1

        feat = load_feature_cache(feature_cache)
        bpm_hint = _choose_bpm(group, feature_dict=feat)
        sample_weight = _choose_sample_weight(group, default=1.0)
        role = _role_from_audio_path(ap)
        frame_times = feat.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
        duration_sec = float(frame_times[-1]) if frame_times.size else 0.0

        cpath = candidate_cache_path(candidates_dir, ap)
        cset: Optional[CandidateSet] = None
        if (not force_candidates) and os.path.isfile(cpath):
            try:
                cset = CandidateSet.from_npz(cpath)
            except Exception:
                cset = None

        if cset is None:
            cset = generate_candidates(feature_dict=feat, audio_path=ap, use_madmom=bool(use_madmom))
            cset.to_npz(cpath)

        if cset.times.size == 0:
            continue
        stats.with_candidates += 1

        times = cset.times.astype(np.float32, copy=False)
        conf = cset.confidence.astype(np.float32, copy=False)
        max_keep = int(max_candidates) if int(max_candidates) > 0 else 0
        if max_keep > 0 and times.shape[0] > max_keep:
            keep = np.unique(np.linspace(0, times.shape[0] - 1, num=max_keep, dtype=np.int32))
            # Always include the nearest candidate to the ALS label.
            pos_global = int(np.argmin(np.abs(times - float(target))))
            if pos_global not in keep:
                repl = int(np.argmin(np.abs(keep.astype(np.float32) - float(pos_global))))
                keep[repl] = int(pos_global)
                keep = np.unique(np.sort(keep))
            times = times[keep]
            conf = conf[keep]

        # Positive label is nearest candidate to ALS target after optional downsampling.
        pos_idx = int(np.argmin(np.abs(times - float(target))))
        pos_err = float(abs(float(times[pos_idx]) - float(target)))
        if pos_err <= float(candidate_tol_sec):
            stats.candidate_hit += 1
        else:
            stats.candidate_miss += 1
            if bool(require_candidate_hit):
                stats.skipped_missing_candidate += 1
                if (i % 25 == 0) or (i == total):
                    print(
                        f"[build] {i}/{total} scanned, usable={len(out)} hit={stats.candidate_hit} miss={stats.candidate_miss}",
                        flush=True,
                    )
                continue

        labels = np.zeros(times.shape[0], dtype=np.float32)
        labels[pos_idx] = 1.0

        offset = np.clip(float(target) - times, -float(max_offset_sec), float(max_offset_sec))
        offset_targets = np.zeros_like(offset, dtype=np.float32)
        offset_targets[pos_idx] = float(offset[pos_idx])

        meta = _candidate_meta(feat, times, conf, cset.tempo_bpm)

        out.append(
            TrackItem(
                audio_path=ap,
                feature_cache=feature_cache,
                target_sec=float(target),
                bpm_hint=float(bpm_hint),
                duration_sec=float(duration_sec),
                cand_times=times,
                cand_conf=conf,
                downbeats=cset.downbeats.astype(np.float32, copy=False),
                tempo_bpm=float(cset.tempo_bpm),
                labels=labels,
                offset_targets=offset_targets,
                meta=meta,
                pos_index=pos_idx,
                nearest_candidate_error_sec=float(pos_err),
                sample_weight=float(sample_weight),
                role=str(role),
            )
        )
        if (i % 25 == 0) or (i == total):
            print(
                f"[build] {i}/{total} scanned, usable={len(out)} hit={stats.candidate_hit} miss={stats.candidate_miss}",
                flush=True,
            )

    return out, stats


def _sample_indices(track: TrackItem, rng: random.Random, max_neg: int, max_hard: int) -> List[int]:
    n = int(track.cand_times.shape[0])
    pos = int(track.pos_index)
    all_idx = list(range(n))
    neg = [i for i in all_idx if i != pos]

    # Near-neighbor negatives keep boundary learning sharp.
    neigh = [i for i in range(max(0, pos - 8), min(n, pos + 9)) if i != pos]

    # Hard negatives:
    # 1) high-confidence/high-energy structural moments that are not the label
    # 2) "second drop" style moments after the positive candidate
    hard_score = (
        (0.45 * track.cand_conf.astype(np.float32, copy=False))
        + (0.25 * track.meta[:, 2].astype(np.float32, copy=False))
        + (0.20 * track.meta[:, 3].astype(np.float32, copy=False))
        + (0.10 * np.clip(track.meta[:, 1], 0.0, 1.0).astype(np.float32, copy=False))
    )
    hard_order = np.argsort(hard_score)[::-1].tolist()
    hard = [int(i) for i in hard_order if i != pos]

    bpm = float(track.tempo_bpm if track.tempo_bpm > 0 else track.bpm_hint)
    beat_sec = 60.0 / max(1e-9, bpm)
    second_drop_min_sec = float(track.cand_times[pos]) + (8.0 * beat_sec)
    second_drop = [int(i) for i in hard if float(track.cand_times[i]) >= second_drop_min_sec]

    picked: List[int] = [pos]
    for i in neigh:
        if i not in picked:
            picked.append(i)
        if len(picked) >= (1 + max(2, int(max_hard // 2))):
            break

    for i in second_drop:
        if i not in picked:
            picked.append(i)
        if len(picked) >= (1 + max_hard):
            break

    for i in hard:
        if i not in picked:
            picked.append(i)
        if len(picked) >= (1 + max_hard):
            break

    remain = [i for i in neg if i not in picked]
    rng.shuffle(remain)
    for i in remain[: max(0, int(max_neg))]:
        picked.append(int(i))

    picked = sorted(set(int(i) for i in picked if 0 <= int(i) < n))
    if pos not in picked:
        picked = [pos] + picked
    return picked


def _weighted_track_losses(
    torch_module,
    logits,
    offsets,
    labels,
    offset_targets,
    track_ids,
    track_weights,
    bce_loss_fn,
    rank_margin: float = 0.35,
):
    """
    Compute per-track losses and aggregate them with explicit sample weights.
    This keeps GOLD/SILVER weighting correct even when multiple tracks are
    packed into one candidate batch.
    """
    uniq = track_ids.unique()
    if uniq.numel() == 0:
        z = logits.new_zeros(())
        return z, z, z, z

    sum_bce = logits.new_zeros(())
    sum_rank = logits.new_zeros(())
    sum_list = logits.new_zeros(())
    sum_off = logits.new_zeros(())
    sum_w = logits.new_zeros(())

    for tid in uniq:
        m = track_ids == tid
        if not bool(m.any()):
            continue
        s = logits[m]
        y = labels[m]
        off = offsets[m]
        off_t = offset_targets[m]
        w = track_weights[m].mean()
        if float(w.detach().cpu()) <= 0.0:
            continue

        local_ids = y.new_zeros(y.shape, dtype=track_ids.dtype)
        l_bce = bce_loss_fn(s, y)
        l_rank = pairwise_ranking_loss(logits=s, labels=y, track_ids=local_ids, margin=float(rank_margin))
        l_list = listwise_softmax_loss(logits=s, labels=y, track_ids=local_ids)

        pos_mask = y > 0.5
        if bool(pos_mask.any()):
            l_off = torch_module.nn.functional.smooth_l1_loss(off[pos_mask], off_t[pos_mask])
        else:
            l_off = logits.new_zeros(())

        sum_bce = sum_bce + (w * l_bce)
        sum_rank = sum_rank + (w * l_rank)
        sum_list = sum_list + (w * l_list)
        sum_off = sum_off + (w * l_off)
        sum_w = sum_w + w

    sum_w = sum_w.clamp(min=1e-6)
    return (sum_bce / sum_w), (sum_rank / sum_w), (sum_list / sum_w), (sum_off / sum_w)


def _score_track(
    model,
    torch,
    track: TrackItem,
    tensor_cache: FeatureTensorLRU,
    device: str,
    temperature: float,
) -> Dict[str, object]:
    long_ctx, short_ctx = tensor_cache.get(track.audio_path, track.feature_cache, track.cand_times)
    xb_long = torch.from_numpy(long_ctx).to(device=device, dtype=torch.float32)
    xb_short = torch.from_numpy(short_ctx).to(device=device, dtype=torch.float32)
    xb_meta = torch.from_numpy(track.meta).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits, offsets = model(xb_long, xb_short, xb_meta)
        logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        offsets_np = offsets.detach().cpu().numpy().astype(np.float32, copy=False)

    probs = sigmoid_np(temperature_scale_logits(logits_np, float(temperature))).astype(np.float32, copy=False)
    order = np.argsort(probs)[::-1]
    j = int(order[0])
    top1 = float(probs[j])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = float(top1 - top2)
    cand_sec = float(track.cand_times[j])
    refined = float(cand_sec + offsets_np[j])

    bpm = float(track.tempo_bpm if track.tempo_bpm > 0 else track.bpm_hint)
    beat_sec = 60.0 / max(1e-9, bpm)
    if abs(refined - cand_sec) > (0.22 * beat_sec):
        refined = cand_sec

    snapped = snap_to_nearest_downbeat(refined, track.downbeats.tolist())

    truth_cand = float(track.cand_times[track.pos_index])
    downbeat_ok = abs(float(cand_sec) - float(truth_cand)) <= max(0.050, 0.12 * beat_sec)
    err_sec = abs(float(snapped) - float(track.target_sec))
    bar_err = abs(float(snapped) - float(track.target_sec)) / max(1e-9, (4.0 * beat_sec))

    top3 = set(int(i) for i in order[:3].tolist())
    candidate_hit = bool(abs(float(track.cand_times[track.pos_index]) - float(track.target_sec)) <= 0.100)

    return {
        "pred_sec": float(snapped),
        "candidate_sec": float(cand_sec),
        "target_sec": float(track.target_sec),
        "confidence": float(top1),
        "margin": float(margin),
        "error_sec": float(err_sec),
        "bar_error": float(bar_err),
        "downbeat_match": bool(downbeat_ok),
        "top3_match": bool(int(track.pos_index) in top3),
        "candidate_hit": bool(candidate_hit),
        "candidate_target_error_sec": float(track.nearest_candidate_error_sec),
        "candidate_norm_pos": float(cand_sec / max(1e-6, track.duration_sec)),
    }


def _aggregate_metrics(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "downbeat_acc": 0.0,
            "median_abs_error_ms": 0.0,
            "p95_abs_error_ms": 0.0,
            "mean_bar_error": 0.0,
            "top3_acc": 0.0,
            "mean_confidence": 0.0,
        }

    err = np.asarray([float(r["error_sec"]) for r in rows], dtype=np.float32)
    bar = np.asarray([float(r["bar_error"]) for r in rows], dtype=np.float32)
    conf = np.asarray([float(r["confidence"]) for r in rows], dtype=np.float32)
    db = np.asarray([1.0 if bool(r["downbeat_match"]) else 0.0 for r in rows], dtype=np.float32)
    t3 = np.asarray([1.0 if bool(r["top3_match"]) else 0.0 for r in rows], dtype=np.float32)

    return {
        "n": float(len(rows)),
        "downbeat_acc": float(np.mean(db)),
        "median_abs_error_ms": float(np.median(err) * 1000.0),
        "p95_abs_error_ms": float(np.percentile(err, 95.0) * 1000.0),
        "mean_bar_error": float(np.mean(bar)),
        "top3_acc": float(np.mean(t3)),
        "mean_confidence": float(np.mean(conf)),
    }


def _pick_review_threshold(val_rows: List[Dict[str, object]]) -> float:
    if not val_rows:
        return 0.55
    conf = np.asarray([float(r["confidence"]) for r in val_rows], dtype=np.float32)
    ok = np.asarray([1.0 if bool(r["downbeat_match"]) else 0.0 for r in val_rows], dtype=np.float32)
    best_thr = 0.55
    best_score = -1.0
    for thr in np.linspace(0.35, 0.90, num=24).tolist():
        pred_good = (conf >= float(thr)).astype(np.float32)
        tp = float(np.sum((pred_good > 0.5) & (ok > 0.5)))
        fp = float(np.sum((pred_good > 0.5) & (ok <= 0.5)))
        fn = float(np.sum((pred_good <= 0.5) & (ok > 0.5)))
        prec = tp / max(1e-6, tp + fp)
        rec = tp / max(1e-6, tp + fn)
        f1 = (2.0 * prec * rec) / max(1e-6, prec + rec)
        if f1 > best_score:
            best_score = f1
            best_thr = float(thr)
    return float(best_thr)


def _pick_guardrail_thresholds(val_rows: List[Dict[str, object]]) -> Tuple[float, float]:
    if not val_rows:
        return 0.58, 0.05
    conf = np.asarray([float(r.get("confidence", 0.0)) for r in val_rows], dtype=np.float32)
    margin = np.asarray([float(r.get("margin", 0.0)) for r in val_rows], dtype=np.float32)
    ok = np.asarray([1.0 if bool(r.get("downbeat_match")) else 0.0 for r in val_rows], dtype=np.float32)
    best = (-1.0, 0.58, 0.05)
    for c in np.linspace(0.35, 0.90, num=23).tolist():
        for m in np.linspace(0.01, 0.20, num=20).tolist():
            accept = ((conf >= float(c)) & (margin >= float(m))).astype(np.float32)
            tp = float(np.sum((accept > 0.5) & (ok > 0.5)))
            fp = float(np.sum((accept > 0.5) & (ok <= 0.5)))
            fn = float(np.sum((accept <= 0.5) & (ok > 0.5)))
            prec = tp / max(1e-6, tp + fp)
            rec = tp / max(1e-6, tp + fn)
            f1 = (2.0 * prec * rec) / max(1e-6, prec + rec)
            # Penalize very low acceptance.
            acc_rate = float(np.mean(accept))
            score = float(f1 - (0.08 * max(0.0, 0.45 - acc_rate)))
            if score > best[0]:
                best = (score, float(c), float(m))
    return float(best[1]), float(best[2])


def _profile_dataset_rows(
    rows: Sequence[Dict[str, object]],
    feature_manifest: Dict[str, str],
) -> Dict[str, object]:
    role_counts: Dict[str, int] = {}
    bpm_vals: List[float] = []
    dur_vals: List[float] = []
    tnorm_vals: List[float] = []
    src_counts: Dict[str, int] = {}

    dur_cache: Dict[str, float] = {}
    for r in rows:
        ap = os.path.abspath(str(r.get("audio_path", "")).strip())
        if not ap:
            continue
        role = _role_from_audio_path(ap)
        role_counts[role] = int(role_counts.get(role, 0) + 1)

        src = str(r.get("_source_name") or "unknown")
        src_counts[src] = int(src_counts.get(src, 0) + 1)

        b = as_float(r.get("bpm_hint"))
        if b is not None and b > 0:
            bpm_vals.append(float(b))

        t = as_float(r.get("target_sec"))
        if t is None or t <= 0:
            continue

        d = 0.0
        md = r.get("metadata")
        if isinstance(md, dict):
            d_md = as_float(md.get("duration_sec"))
            if d_md is not None and d_md > 0:
                d = float(d_md)
        if d <= 0:
            if ap not in dur_cache:
                cp = feature_manifest.get(ap)
                if cp and os.path.isfile(cp):
                    try:
                        feat = load_feature_cache(cp)
                        ft = feat.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
                        dur_cache[ap] = float(ft[-1]) if ft.size else 0.0
                    except Exception:
                        dur_cache[ap] = 0.0
                else:
                    dur_cache[ap] = 0.0
            d = float(dur_cache.get(ap, 0.0))
        if d > 0:
            dur_vals.append(float(d))
            tnorm_vals.append(float(np.clip(float(t) / max(1e-6, float(d)), 0.0, 1.0)))

    def _stats(vals: Sequence[float]) -> Dict[str, float]:
        if not vals:
            return {"n": 0.0, "min": 0.0, "median": 0.0, "max": 0.0, "p05": 0.0, "p95": 0.0}
        arr = np.asarray(list(vals), dtype=np.float32)
        return {
            "n": float(arr.size),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "p05": float(np.percentile(arr, 5.0)),
            "p95": float(np.percentile(arr, 95.0)),
        }

    return {
        "rows": int(len(rows)),
        "unique_audio": int(len({os.path.abspath(str(r.get('audio_path', '')).strip()) for r in rows if str(r.get('audio_path', '')).strip()})),
        "role_counts": {k: int(v) for k, v in sorted(role_counts.items())},
        "source_counts": {k: int(v) for k, v in sorted(src_counts.items())},
        "bpm": _stats(bpm_vals),
        "duration_sec": _stats(dur_vals),
        "target_norm": _stats(tnorm_vals),
    }


def run_train(
    dataset_jsonl: str,
    feature_manifest_jsonl: str,
    candidates_dir: str,
    out_model: str,
    out_metrics: str,
    silver_dataset_jsonl: str = "",
    use_madmom: bool = True,
    force_candidates: bool = False,
    epochs: int = 18,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_neg: int = 96,
    max_hard: int = 96,
    rank_weight: float = 0.5,
    listwise_weight: float = 0.85,
    bce_weight: float = 0.25,
    offset_weight: float = 0.35,
    primary_weight: float = 1.0,
    silver_weight: float = 0.35,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    limit_tracks: int = 0,
    tensor_cache_capacity: int = 0,
    tracks_per_step: int = 8,
    model_width: int = 128,
    model_dropout: float = 0.15,
    long_frames: int = 384,
    short_frames: int = 128,
    max_candidates: int = 192,
    candidate_tol_sec: float = 0.100,
    require_candidate_hit: bool = True,
) -> Dict[str, object]:
    torch, GradScaler, autocast = _require_torch()
    seeded(int(seed))

    feature_manifest = _load_manifest(feature_manifest_jsonl)
    primary_rows = _load_dataset_rows(dataset_jsonl, source_weight=float(primary_weight), source_name="primary")
    dataset_rows = list(primary_rows)
    silver_rows: List[Dict[str, object]] = []
    if silver_dataset_jsonl and os.path.isfile(silver_dataset_jsonl):
        silver_rows = _load_dataset_rows(silver_dataset_jsonl, source_weight=float(silver_weight), source_name="silver")
        dataset_rows.extend(silver_rows)
    profile_primary = _profile_dataset_rows(primary_rows, feature_manifest)
    profile_silver = _profile_dataset_rows(silver_rows, feature_manifest) if silver_rows else {}
    profile_merged = _profile_dataset_rows(dataset_rows, feature_manifest)

    context_cfg = ContextConfig(
        long_frames=int(max(96, int(long_frames))),
        short_frames=int(max(48, int(short_frames))),
    )
    tracks, build_stats = _build_tracks(
        dataset_rows=dataset_rows,
        feature_manifest=feature_manifest,
        candidates_dir=candidates_dir,
        use_madmom=bool(use_madmom),
        force_candidates=bool(force_candidates),
        max_offset_sec=float(DEFAULT_MAX_OFFSET_SEC),
        max_candidates=int(max_candidates),
        candidate_tol_sec=float(candidate_tol_sec),
        require_candidate_hit=bool(require_candidate_hit),
        limit=int(limit_tracks),
    )
    # If strict candidate-hit filtering leaves too little data, relax once.
    if len(tracks) < 12 and bool(require_candidate_hit):
        print(
            "[train] strict candidate-hit filtering left too few tracks; retrying without candidate-hit filter",
            flush=True,
        )
        tracks, build_stats = _build_tracks(
            dataset_rows=dataset_rows,
            feature_manifest=feature_manifest,
            candidates_dir=candidates_dir,
            use_madmom=bool(use_madmom),
            force_candidates=bool(force_candidates),
            max_offset_sec=float(DEFAULT_MAX_OFFSET_SEC),
            max_candidates=int(max_candidates),
            candidate_tol_sec=float(candidate_tol_sec),
            require_candidate_hit=False,
            limit=int(limit_tracks),
        )
    if len(tracks) < 12:
        raise RuntimeError(f"Need at least 12 labeled tracks with features+candidates. Found {len(tracks)}")

    track_paths = [t.audio_path for t in tracks]
    train_keys, val_keys, test_keys = _split_grouped(track_paths, seed=int(seed), train=0.8, val=0.1)
    by_key = {t.audio_path: t for t in tracks}
    train_tracks = [by_key[k] for k in train_keys if k in by_key]
    val_tracks = [by_key[k] for k in val_keys if k in by_key]
    test_tracks = [by_key[k] for k in test_keys if k in by_key]

    # infer model shape from one feature file
    probe = load_feature_cache(train_tracks[0].feature_cache)
    in_channels = int(probe["mel_db"].shape[1] + 9)
    meta_dim = 4

    model = build_model(
        in_channels=in_channels,
        meta_dim=meta_dim,
        width=int(max(32, int(model_width))),
        dropout=float(max(0.0, float(model_dropout))),
        max_offset_sec=float(DEFAULT_MAX_OFFSET_SEC),
    )

    dev = _device_auto(torch, device)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = GradScaler(enabled=(dev == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0], device=dev, dtype=torch.float32))

    cache_capacity = int(tensor_cache_capacity) if int(tensor_cache_capacity) > 0 else min(384, max(32, len(tracks)))
    tensor_cache = FeatureTensorLRU(context_cfg=context_cfg, capacity=cache_capacity)
    rng = random.Random(int(seed))

    cand_recall = float(build_stats.candidate_hit) / max(1.0, float(build_stats.with_candidates))
    w_vals = np.asarray([float(t.sample_weight) for t in tracks], dtype=np.float32) if tracks else np.asarray([], dtype=np.float32)
    print(f"[train] tracks total={len(tracks)} train={len(train_tracks)} val={len(val_tracks)} test={len(test_tracks)}", flush=True)
    if w_vals.size:
        print(
            f"[train] sample_weight min/median/max={float(np.min(w_vals)):.2f}/{float(np.median(w_vals)):.2f}/{float(np.max(w_vals)):.2f}",
            flush=True,
        )
    print(
        f"[train] candidate recall@{float(candidate_tol_sec)*1000:.0f}ms={cand_recall:.3f} "
        f"(hit={build_stats.candidate_hit}, miss={build_stats.candidate_miss}, scanned_with_candidates={build_stats.with_candidates})",
        flush=True,
    )
    if cand_recall < 0.95:
        print("[train][WARN] candidate recall below 0.95; candidate generator likely still dropping GT anchors.", flush=True)
    print(f"[train] tensor context cache capacity={cache_capacity}", flush=True)
    print(f"[train] max candidates per track={int(max_candidates)}", flush=True)

    history: List[Dict[str, float]] = []
    best_state = None
    best_score = float("inf")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        rng.shuffle(train_tracks)

        loss_rows = []
        opt.zero_grad(set_to_none=True)
        tps = max(1, int(tracks_per_step))
        n_steps = max(1, int(math.ceil(float(len(train_tracks)) / float(tps))))
        log_every = max(1, n_steps // 6)

        for step_i in range(n_steps):
            i0 = int(step_i * tps)
            i1 = int(min(len(train_tracks), (step_i + 1) * tps))
            chunk = train_tracks[i0:i1]
            if not chunk:
                continue
            print(
                f"[E{epoch:02d}] step {step_i + 1}/{n_steps} begin (tracks={len(chunk)})",
                flush=True,
            )

            np_long: List[np.ndarray] = []
            np_short: List[np.ndarray] = []
            np_meta: List[np.ndarray] = []
            np_y: List[np.ndarray] = []
            np_off_t: List[np.ndarray] = []
            np_tid: List[np.ndarray] = []
            np_w: List[np.ndarray] = []

            local_tid = 0
            for tr in chunk:
                idx = _sample_indices(tr, rng=rng, max_neg=int(max_neg), max_hard=int(max_hard))
                if not idx:
                    continue
                long_ctx, short_ctx = tensor_cache.get(tr.audio_path, tr.feature_cache, tr.cand_times)
                np_long.append(long_ctx[idx].astype(np.float32, copy=False))
                np_short.append(short_ctx[idx].astype(np.float32, copy=False))
                np_meta.append(tr.meta[idx].astype(np.float32, copy=False))
                np_y.append(tr.labels[idx].astype(np.float32, copy=False))
                np_off_t.append(tr.offset_targets[idx].astype(np.float32, copy=False))
                n_i = len(idx)
                np_tid.append(np.full((n_i,), int(local_tid), dtype=np.int64))
                np_w.append(np.full((n_i,), float(max(1e-6, tr.sample_weight)), dtype=np.float32))
                local_tid += 1

            if not np_long:
                continue

            x_long = torch.from_numpy(np.concatenate(np_long, axis=0)).to(device=dev, dtype=torch.float32)
            x_short = torch.from_numpy(np.concatenate(np_short, axis=0)).to(device=dev, dtype=torch.float32)
            x_meta = torch.from_numpy(np.concatenate(np_meta, axis=0)).to(device=dev, dtype=torch.float32)
            y = torch.from_numpy(np.concatenate(np_y, axis=0)).to(device=dev, dtype=torch.float32)
            off_t = torch.from_numpy(np.concatenate(np_off_t, axis=0)).to(device=dev, dtype=torch.float32)
            track_ids = torch.from_numpy(np.concatenate(np_tid, axis=0)).to(device=dev, dtype=torch.long)
            track_w = torch.from_numpy(np.concatenate(np_w, axis=0)).to(device=dev, dtype=torch.float32)

            with autocast(enabled=(dev == "cuda")):
                logits, off = model(x_long, x_short, x_meta)
                l_bce, l_rank, l_list, l_off = _weighted_track_losses(
                    torch_module=torch,
                    logits=logits,
                    offsets=off,
                    labels=y,
                    offset_targets=off_t,
                    track_ids=track_ids,
                    track_weights=track_w,
                    bce_loss_fn=bce,
                    rank_margin=0.35,
                )
                loss = (
                    (float(bce_weight) * l_bce)
                    + (float(rank_weight) * l_rank)
                    + (float(listwise_weight) * l_list)
                    + (float(offset_weight) * l_off)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            loss_rows.append(
                (
                    float(l_bce.detach().cpu()),
                    float(l_rank.detach().cpu()),
                    float(l_list.detach().cpu()),
                    float(l_off.detach().cpu()),
                    float(loss.detach().cpu()),
                )
            )
            if ((step_i + 1) % log_every == 0) or ((step_i + 1) == n_steps):
                print(f"[E{epoch:02d}] step {step_i + 1}/{n_steps} end", flush=True)

        # Gather validation logits for calibration.
        model.eval()
        val_logits_all: List[np.ndarray] = []
        val_labels_all: List[np.ndarray] = []
        with torch.no_grad():
            for tr in val_tracks:
                long_ctx, short_ctx = tensor_cache.get(tr.audio_path, tr.feature_cache, tr.cand_times)
                xb_long = torch.from_numpy(long_ctx).to(device=dev, dtype=torch.float32)
                xb_short = torch.from_numpy(short_ctx).to(device=dev, dtype=torch.float32)
                xb_meta = torch.from_numpy(tr.meta).to(device=dev, dtype=torch.float32)
                logits, _off = model(xb_long, xb_short, xb_meta)
                val_logits_all.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
                val_labels_all.append(tr.labels.astype(np.float32, copy=False))

        if val_logits_all:
            val_logits_np = np.concatenate(val_logits_all, axis=0)
            val_labels_np = np.concatenate(val_labels_all, axis=0)
            temperature = float(calibrate_temperature(val_logits_np, val_labels_np))
        else:
            temperature = 1.0

        val_rows = [_score_track(model, torch, tr, tensor_cache, dev, temperature=temperature) for tr in val_tracks]
        val_metrics = _aggregate_metrics(val_rows)

        train_loss = float(np.mean(np.asarray([r[4] for r in loss_rows], dtype=np.float32))) if loss_rows else 0.0
        mean_bce = float(np.mean(np.asarray([r[0] for r in loss_rows], dtype=np.float32))) if loss_rows else 0.0
        mean_rank = float(np.mean(np.asarray([r[1] for r in loss_rows], dtype=np.float32))) if loss_rows else 0.0
        mean_list = float(np.mean(np.asarray([r[2] for r in loss_rows], dtype=np.float32))) if loss_rows else 0.0
        mean_off = float(np.mean(np.asarray([r[3] for r in loss_rows], dtype=np.float32))) if loss_rows else 0.0
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_bce": mean_bce,
            "train_pairwise": mean_rank,
            "train_listwise": mean_list,
            "train_offset": mean_off,
            "val_downbeat_acc": float(val_metrics["downbeat_acc"]),
            "val_median_ms": float(val_metrics["median_abs_error_ms"]),
            "val_p95_ms": float(val_metrics["p95_abs_error_ms"]),
            "val_bar_error": float(val_metrics["mean_bar_error"]),
            "val_top3": float(val_metrics["top3_acc"]),
            "temperature": float(temperature),
        }
        history.append(row)

        score = float(val_metrics["median_abs_error_ms"]) + (120.0 * (1.0 - float(val_metrics["downbeat_acc"]))) + (40.0 * float(val_metrics["mean_bar_error"]))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[E{epoch:02d}] loss={train_loss:.4f} val_downbeat={val_metrics['downbeat_acc']:.3f} "
            f"val_med={val_metrics['median_abs_error_ms']:.1f}ms val_p95={val_metrics['p95_abs_error_ms']:.1f}ms",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final calibration + metrics.
    model.eval()
    final_val_logits: List[np.ndarray] = []
    final_val_labels: List[np.ndarray] = []
    with torch.no_grad():
        for tr in val_tracks:
            long_ctx, short_ctx = tensor_cache.get(tr.audio_path, tr.feature_cache, tr.cand_times)
            xb_long = torch.from_numpy(long_ctx).to(device=dev, dtype=torch.float32)
            xb_short = torch.from_numpy(short_ctx).to(device=dev, dtype=torch.float32)
            xb_meta = torch.from_numpy(tr.meta).to(device=dev, dtype=torch.float32)
            logits, _ = model(xb_long, xb_short, xb_meta)
            final_val_logits.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            final_val_labels.append(tr.labels.astype(np.float32, copy=False))

    if final_val_logits:
        final_temperature = float(calibrate_temperature(np.concatenate(final_val_logits), np.concatenate(final_val_labels)))
    else:
        final_temperature = 1.0

    val_rows = [_score_track(model, torch, tr, tensor_cache, dev, temperature=final_temperature) for tr in val_tracks]
    test_rows = [_score_track(model, torch, tr, tensor_cache, dev, temperature=final_temperature) for tr in test_tracks]
    val_metrics = _aggregate_metrics(val_rows)
    test_metrics = _aggregate_metrics(test_rows)
    review_threshold = _pick_review_threshold(val_rows)
    guardrail_conf, guardrail_margin = _pick_guardrail_thresholds(val_rows)

    target_norm = np.asarray(
        [float(t.target_sec) / max(1e-6, float(t.duration_sec)) for t in tracks if t.duration_sec > 0 and t.target_sec > 0],
        dtype=np.float32,
    )
    if target_norm.size:
        prior_p05 = float(np.clip(np.percentile(target_norm, 5.0), 0.0, 1.0))
        prior_p95 = float(np.clip(np.percentile(target_norm, 95.0), 0.0, 1.0))
    else:
        prior_p05, prior_p95 = 0.06, 0.90

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "in_channels": int(in_channels),
            "meta_dim": int(meta_dim),
            "width": int(max(32, int(model_width))),
            "dropout": float(max(0.0, float(model_dropout))),
            "max_offset_sec": float(DEFAULT_MAX_OFFSET_SEC),
            "sr": int(probe["sr"][0]) if "sr" in probe else 22050,
            "hop": int(probe["hop_length"][0]) if "hop_length" in probe else 256,
            "n_mels": int(probe["n_mels"][0]) if "n_mels" in probe else int(probe["mel_db"].shape[1]),
            "long_left_sec": float(context_cfg.long_left_sec),
            "long_right_sec": float(context_cfg.long_right_sec),
            "short_left_sec": float(context_cfg.short_left_sec),
            "short_right_sec": float(context_cfg.short_right_sec),
            "long_frames": int(context_cfg.long_frames),
            "short_frames": int(context_cfg.short_frames),
        },
        "temperature": float(final_temperature),
        "review_threshold": float(review_threshold),
        "guardrails": {
            "min_confidence": float(guardrail_conf),
            "min_margin": float(guardrail_margin),
            "candidate_tol_sec": float(candidate_tol_sec),
        },
        "drop_region_prior": {
            "norm_p05": float(prior_p05),
            "norm_p95": float(prior_p95),
        },
        "splits": {
            "train": [t.audio_path for t in train_tracks],
            "val": [t.audio_path for t in val_tracks],
            "test": [t.audio_path for t in test_tracks],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "tensor_cache_capacity": int(cache_capacity),
        "tracks_per_step": int(tps),
        "model_width": int(max(32, int(model_width))),
        "model_dropout": float(max(0.0, float(model_dropout))),
        "long_frames": int(context_cfg.long_frames),
        "short_frames": int(context_cfg.short_frames),
        "max_candidates": int(max_candidates),
        "dataset_profile": {
            "primary": profile_primary,
            "silver": profile_silver,
            "merged": profile_merged,
            "weights": {
                "primary_weight": float(primary_weight),
                "silver_weight": float(silver_weight),
            },
        },
        "build_stats": {
            "scanned": int(build_stats.scanned),
            "with_features": int(build_stats.with_features),
            "with_target": int(build_stats.with_target),
            "with_candidates": int(build_stats.with_candidates),
            "candidate_hit": int(build_stats.candidate_hit),
            "candidate_miss": int(build_stats.candidate_miss),
            "skipped_missing_candidate": int(build_stats.skipped_missing_candidate),
            "candidate_recall": float(cand_recall),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_model)), exist_ok=True)
    torch.save(ckpt, out_model)

    metrics = {
        "tracks_total": int(len(tracks)),
        "train_tracks": int(len(train_tracks)),
        "val_tracks": int(len(val_tracks)),
        "test_tracks": int(len(test_tracks)),
        "temperature": float(final_temperature),
        "review_threshold": float(review_threshold),
        "guardrail_conf": float(guardrail_conf),
        "guardrail_margin": float(guardrail_margin),
        "candidate_tol_sec": float(candidate_tol_sec),
        "candidate_recall": float(cand_recall),
        "build_stats": ckpt["build_stats"],
        "best_score": float(best_score),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "tensor_cache_capacity": int(cache_capacity),
        "tracks_per_step": int(tps),
        "model_width": int(max(32, int(model_width))),
        "model_dropout": float(max(0.0, float(model_dropout))),
        "long_frames": int(context_cfg.long_frames),
        "short_frames": int(context_cfg.short_frames),
        "max_candidates": int(max_candidates),
        "dataset_profile": ckpt["dataset_profile"],
    }
    write_json(out_metrics, metrics)
    return {
        "ok": True,
        "out_model": os.path.abspath(out_model),
        "out_metrics": os.path.abspath(out_metrics),
        **metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train ALS-supervised drop anchor model")
    ap.add_argument("--dataset", default="alsdrop/data/dataset.jsonl", help="Label dataset JSONL")
    ap.add_argument("--silver-dataset", default="", help="Optional SILVER dataset JSONL for weighted mixed training")
    ap.add_argument("--features", default="alsdrop/data/features_manifest.jsonl", help="Feature manifest JSONL")
    ap.add_argument("--candidates_dir", default="alsdrop/data/candidates", help="Candidate cache directory")
    ap.add_argument("--out", default="alsdrop/models/model.pt", help="Output model checkpoint")
    ap.add_argument("--metrics", default="alsdrop/outputs/train_metrics.json", help="Output metrics JSON")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-neg", type=int, default=96)
    ap.add_argument("--max-hard", type=int, default=96)
    ap.add_argument("--rank-weight", type=float, default=0.5)
    ap.add_argument("--listwise-weight", type=float, default=0.85)
    ap.add_argument("--bce-weight", type=float, default=0.25)
    ap.add_argument("--offset-weight", type=float, default=0.35)
    ap.add_argument("--primary-weight", type=float, default=1.0, help="Sample weight for --dataset rows")
    ap.add_argument("--silver-weight", type=float, default=0.35, help="Sample weight for --silver-dataset rows")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--limit-tracks", type=int, default=0)
    ap.add_argument("--tensor-cache-capacity", type=int, default=0)
    ap.add_argument("--tracks-per-step", type=int, default=8, help="How many tracks to fuse into one training step")
    ap.add_argument("--model-width", type=int, default=128, help="Model channel width; lower is faster")
    ap.add_argument("--model-dropout", type=float, default=0.15)
    ap.add_argument("--long-frames", type=int, default=384, help="Long context frames; lower is faster")
    ap.add_argument("--short-frames", type=int, default=128, help="Short context frames; lower is faster")
    ap.add_argument("--max-candidates", type=int, default=192)
    ap.add_argument("--candidate-tol-sec", type=float, default=0.100, help="GT coverage tolerance for candidate recall")
    ap.add_argument("--no-require-candidate-hit", action="store_true", help="Do not skip tracks missing close candidate")
    ap.add_argument("--no-madmom", action="store_true")
    ap.add_argument("--force-candidates", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_train(
        dataset_jsonl=args.dataset,
        feature_manifest_jsonl=args.features,
        candidates_dir=args.candidates_dir,
        out_model=args.out,
        out_metrics=args.metrics,
        silver_dataset_jsonl=args.silver_dataset,
        use_madmom=not bool(args.no_madmom),
        force_candidates=bool(args.force_candidates),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_neg=int(args.max_neg),
        max_hard=int(args.max_hard),
        rank_weight=float(args.rank_weight),
        listwise_weight=float(args.listwise_weight),
        bce_weight=float(args.bce_weight),
        offset_weight=float(args.offset_weight),
        primary_weight=float(args.primary_weight),
        silver_weight=float(args.silver_weight),
        seed=int(args.seed),
        device=args.device,
        limit_tracks=int(args.limit_tracks),
        tensor_cache_capacity=int(args.tensor_cache_capacity),
        tracks_per_step=int(args.tracks_per_step),
        model_width=int(args.model_width),
        model_dropout=float(args.model_dropout),
        long_frames=int(args.long_frames),
        short_frames=int(args.short_frames),
        max_candidates=int(args.max_candidates),
        candidate_tol_sec=float(args.candidate_tol_sec),
        require_candidate_hit=not bool(args.no_require_candidate_hit),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
