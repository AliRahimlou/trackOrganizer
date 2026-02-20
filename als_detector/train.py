#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from als_detector.model import build_model, extract_window, stack_channels
from als_detector.utils.audio import load_feature_cache
from als_detector.utils.candidates import (
    generate_candidate_indices,
    madmom_downbeats,
    sample_training_indices,
    snap_to_nearest_downbeat,
)
from als_detector.utils.labels import median_bpm, select_primary_target_sec


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for training. Install with: pip install torch") from e
    return torch, F, Dataset, DataLoader


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def _median_float(vals: List[float], fallback: float = 0.0) -> float:
    if not vals:
        return fallback
    return float(np.median(np.asarray(vals, dtype=np.float32)))


@dataclass
class TrackMeta:
    audio_path: str
    cache_path: str
    target_sec: float
    bpm: float


class FeatureLRU:
    def __init__(self, capacity: int = 32):
        self.capacity = max(4, int(capacity))
        self._d: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def get(self, cache_path: str) -> Dict[str, np.ndarray]:
        if cache_path in self._d:
            v = self._d.pop(cache_path)
            self._d[cache_path] = v
            return v
        z = load_feature_cache(cache_path)
        feat = {
            "mel_db": z["mel_db"],
            "rms": z["rms"],
            "onset": z["onset"],
            "low_energy": z["low_energy"],
            "frame_times": z["frame_times"],
            "beat_times": z.get("beat_times", np.asarray([], dtype=np.float32)),
        }
        feat["channels"] = stack_channels(feat)
        if len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        self._d[cache_path] = feat
        return feat


class WindowDataset:
    def __init__(self, samples: List[Tuple[str, int, int]], tracks: Dict[str, TrackMeta], window_frames: int, lru_capacity: int = 24):
        self.samples = samples
        self.tracks = tracks
        self.window_frames = int(window_frames)
        self.lru = FeatureLRU(capacity=lru_capacity)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        audio_path, frame_idx, label = self.samples[idx]
        tr = self.tracks[audio_path]
        feat = self.lru.get(tr.cache_path)
        x = extract_window(feat["channels"], center_idx=int(frame_idx), window_frames=self.window_frames)
        return x.astype(np.float32, copy=False), np.float32(label)


def _build_track_table(labels: List[Dict[str, object]], manifest: List[Dict[str, object]]) -> Dict[str, TrackMeta]:
    by_audio_labels: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in labels:
        ap = str(r.get("audio_path", "")).strip()
        if ap:
            by_audio_labels[os.path.abspath(ap)].append(r)

    m_by_audio: Dict[str, Dict[str, object]] = {}
    for r in manifest:
        ap = os.path.abspath(str(r.get("audio_path", "")).strip())
        if ap:
            m_by_audio[ap] = r

    out: Dict[str, TrackMeta] = {}
    for ap, group in by_audio_labels.items():
        m = m_by_audio.get(ap)
        if m is None:
            continue
        cache = str(m.get("cache_path", "")).strip()
        if not cache or not os.path.exists(cache):
            continue
        target, _meta = select_primary_target_sec(group)
        if target is None:
            continue
        bpm_m = median_bpm(group)
        bpm = float(bpm_m) if bpm_m and bpm_m > 0 else float(m.get("bpm_hint", 0.0) or 0.0)
        if bpm <= 0:
            bpm = float(m.get("tempo_est", 0.0) or 128.0)
        out[ap] = TrackMeta(audio_path=ap, cache_path=os.path.abspath(cache), target_sec=target, bpm=bpm)
    return out


def _split_tracks(track_paths: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    arr = list(track_paths)
    rnd = random.Random(seed)
    rnd.shuffle(arr)
    n = len(arr)
    n_train = max(1, int(round(0.80 * n)))
    n_val = max(1, int(round(0.10 * n)))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_train = max(1, n_train - 1)
    train = arr[:n_train]
    val = arr[n_train:n_train + n_val]
    test = arr[n_train + n_val:]
    if not test:
        test = val[-1:]
        val = val[:-1] if len(val) > 1 else val
    return train, val, test


def _build_samples_for_split(
    split_tracks: List[str],
    tracks: Dict[str, TrackMeta],
    seed: int,
) -> List[Tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    lru = FeatureLRU(capacity=16)
    out: List[Tuple[str, int, int]] = []
    for ap in split_tracks:
        tr = tracks[ap]
        feat = lru.get(tr.cache_path)
        rows = sample_training_indices(
            feature_dict=feat,
            target_sec=tr.target_sec,
            rng=rng,
            pos_radius_frames=1,
            n_random_neg=64,
            n_hard_neg=48,
        )
        out.extend((ap, int(i), int(y)) for i, y in rows)
    return out


def _predict_track(
    model,
    torch,
    track: TrackMeta,
    lru: FeatureLRU,
    window_frames: int,
    device: str,
    use_madmom: bool,
) -> Tuple[float, float]:
    feat = lru.get(track.cache_path)
    cand = generate_candidate_indices(
        feature_dict=feat,
        audio_path=track.audio_path,
        use_madmom=use_madmom,
    )
    if cand.size == 0:
        return float(track.target_sec), 0.0

    windows = np.stack([extract_window(feat["channels"], int(i), window_frames) for i in cand], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    j = int(np.argmax(probs))
    pred_raw = float(feat["frame_times"][int(cand[j])])

    downbeats = []
    if use_madmom:
        downbeats = [float(t) for t in madmom_downbeats(track.audio_path).tolist()]
    if not downbeats:
        downbeats = [float(t) for t in feat.get("beat_times", np.asarray([], dtype=np.float32)).tolist()]
    pred_snap = snap_to_nearest_downbeat(pred_raw, downbeats)
    return float(pred_snap), float(probs[j])


def _evaluate_localization(
    model,
    torch,
    split_tracks: List[str],
    tracks: Dict[str, TrackMeta],
    window_frames: int,
    device: str,
    use_madmom: bool,
) -> Dict[str, float]:
    lru = FeatureLRU(capacity=24)
    errs = []
    bar_errs = []
    probs = []
    for ap in split_tracks:
        tr = tracks[ap]
        pred_sec, p = _predict_track(
            model=model,
            torch=torch,
            track=tr,
            lru=lru,
            window_frames=window_frames,
            device=device,
            use_madmom=use_madmom,
        )
        err = abs(float(pred_sec) - float(tr.target_sec))
        errs.append(err)
        beat_sec = 60.0 / max(1e-6, float(tr.bpm))
        bar_errs.append(abs(err / beat_sec) / 4.0)
        probs.append(float(p))
    if not errs:
        return {"mae_sec": 0.0, "bar_error": 0.0, "acc_100ms": 0.0, "mean_prob": 0.0}
    errs_np = np.asarray(errs, dtype=np.float32)
    bar_np = np.asarray(bar_errs, dtype=np.float32)
    prob_np = np.asarray(probs, dtype=np.float32)
    return {
        "mae_sec": float(np.mean(errs_np)),
        "bar_error": float(np.mean(bar_np)),
        "acc_100ms": float(np.mean((errs_np <= 0.100).astype(np.float32))),
        "mean_prob": float(np.mean(prob_np)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ALS-trained 1.1.1 detector.")
    ap.add_argument("--dataset", default="als_detector/data/labels.jsonl")
    ap.add_argument("--manifest", default="als_detector/data/features_manifest.jsonl")
    ap.add_argument("--out-model", default="als_detector/models/anchor_cnn.pt")
    ap.add_argument("--out-metrics", default="als_detector/outputs/train_metrics.json")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window-sec", type=float, default=8.0, help="Feature window width in seconds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--use-madmom", action="store_true")
    args = ap.parse_args()

    torch, F, DatasetBase, DataLoader = _require_torch()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = _load_jsonl(args.dataset)
    manifest = _load_jsonl(args.manifest)
    tracks = _build_track_table(labels=labels, manifest=manifest)
    if len(tracks) < 6:
        print(f"[ERROR] Need at least 6 labeled tracks with features; got {len(tracks)}")
        return 2

    train_tracks, val_tracks, test_tracks = _split_tracks(sorted(tracks.keys()), seed=args.seed)
    print(f"[INFO] Tracks split train/val/test: {len(train_tracks)}/{len(val_tracks)}/{len(test_tracks)}")

    train_samples = _build_samples_for_split(train_tracks, tracks, seed=args.seed)
    val_samples = _build_samples_for_split(val_tracks, tracks, seed=args.seed + 1)
    if not train_samples or not val_samples:
        print("[ERROR] Empty training/validation samples.")
        return 3

    # Infer frame params from first track.
    first = load_feature_cache(tracks[train_tracks[0]].cache_path)
    frame_times = first["frame_times"]
    hop_sec = float(frame_times[1] - frame_times[0]) if len(frame_times) > 1 else (512.0 / 22050.0)
    sr = int(first["meta_sr"][0]) if "meta_sr" in first else 22050
    hop_length = int(first["meta_hop_length"][0]) if "meta_hop_length" in first else 512
    n_fft = int(first["meta_n_fft"][0]) if "meta_n_fft" in first else 2048
    n_mels = int(first["meta_n_mels"][0]) if "meta_n_mels" in first else int(first["mel_db"].shape[1])
    window_frames = max(32, int(round(float(args.window_sec) / hop_sec)))
    in_channels = int(first["mel_db"].shape[1] + 3)  # mel channels + rms/onset/low

    train_ds = WindowDataset(train_samples, tracks=tracks, window_frames=window_frames)
    val_ds = WindowDataset(val_samples, tracks=tracks, window_frames=window_frames)

    # Wrap with torch Dataset-compatible adapter.
    class _TorchDS(DatasetBase):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx):
            x, y = self.inner[idx]
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    t_train = _TorchDS(train_ds)
    t_val = _TorchDS(val_ds)

    train_loader = DataLoader(t_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(t_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model = build_model(in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    pos = sum(1 for _, _, y in train_samples if y == 1)
    neg = max(1, len(train_samples) - pos)
    pos_weight = float(neg / max(1, pos))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    history = []
    best_state = None
    best_score = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                logits = model(xb)
                loss = bce(logits, yb)
                va_losses.append(float(loss.detach().cpu()))

        loc = _evaluate_localization(
            model=model,
            torch=torch,
            split_tracks=val_tracks,
            tracks=tracks,
            window_frames=window_frames,
            device=device,
            use_madmom=bool(args.use_madmom),
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(np.asarray(tr_losses, dtype=np.float32))) if tr_losses else 0.0,
            "val_loss": float(np.mean(np.asarray(va_losses, dtype=np.float32))) if va_losses else 0.0,
            **loc,
        }
        history.append(row)
        print(
            f"[E{epoch:02d}] train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} mae={row['mae_sec']:.3f}s "
            f"bar={row['bar_error']:.3f} acc100={row['acc_100ms']:.3f}"
        )

        score = float(row["mae_sec"]) + (0.25 * float(row["bar_error"]))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loc = _evaluate_localization(
        model=model,
        torch=torch,
        split_tracks=test_tracks,
        tracks=tracks,
        window_frames=window_frames,
        device=device,
        use_madmom=bool(args.use_madmom),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_model)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "in_channels": in_channels,
                "window_frames": window_frames,
                "window_sec": float(args.window_sec),
                "hop_sec": hop_sec,
                "sr": sr,
                "hop_length": hop_length,
                "n_fft": n_fft,
                "n_mels": n_mels,
                "use_madmom": bool(args.use_madmom),
            },
            "splits": {
                "train": train_tracks,
                "val": val_tracks,
                "test": test_tracks,
            },
            "test_metrics": test_loc,
        },
        args.out_model,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_metrics)), exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_score": best_score,
                "test_metrics": test_loc,
                "n_tracks": len(tracks),
                "n_train_samples": len(train_samples),
                "n_val_samples": len(val_samples),
            },
            f,
            indent=2,
        )

    print(f"[OK] Model saved: {os.path.abspath(args.out_model)}")
    print(f"[OK] Test metrics: {test_loc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
