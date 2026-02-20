#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> str:
    p = os.path.abspath(path)
    os.makedirs(p, exist_ok=True)
    return p


def parent_dir(path: str) -> str:
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return d


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()[: max(4, int(n))]


def audio_id(audio_path: str) -> str:
    return stable_hash(os.path.abspath(audio_path), n=16)


def write_json(path: str, obj: Dict[str, object]) -> None:
    parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def iter_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj


def read_jsonl(path: str) -> List[Dict[str, object]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def as_float(v, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return float(x)


def as_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        x = int(round(float(v)))
    except Exception:
        return default
    return int(x)


def seeded(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def quantile_clip_norm(x: np.ndarray, q_lo: float = 10.0, q_hi: float = 90.0) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, q_lo))
    hi = float(np.percentile(x, q_hi))
    span = max(1e-9, hi - lo)
    y = (x - lo) / span
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def split_train_val_test(items: Sequence[str], seed: int = 42, train: float = 0.8, val: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    arr = list(items)
    rng = random.Random(seed)
    rng.shuffle(arr)
    n = len(arr)
    n_train = max(1, int(round(train * n)))
    n_val = max(1, int(round(val * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)
    train_items = arr[:n_train]
    val_items = arr[n_train : n_train + n_val]
    test_items = arr[n_train + n_val : n_train + n_val + n_test]
    if not test_items:
        test_items = arr[-1:]
    return train_items, val_items, test_items


def find_audio_files(audio_dir: str) -> List[str]:
    out: List[str] = []
    root = Path(audio_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg"}:
            out.append(str(p.resolve()))
    return sorted(set(out))
