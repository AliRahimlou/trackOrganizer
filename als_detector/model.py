#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def stack_channels(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    mel = feature_dict["mel_db"].astype(np.float32, copy=False).T  # [n_mels, T]
    rms = feature_dict["rms"].astype(np.float32, copy=False)[None, :]
    onset = feature_dict["onset"].astype(np.float32, copy=False)[None, :]
    low = feature_dict["low_energy"].astype(np.float32, copy=False)[None, :]

    # Robust normalize scalar channels track-wise.
    def _nz(v: np.ndarray) -> np.ndarray:
        lo = np.percentile(v, 10.0)
        hi = np.percentile(v, 90.0)
        span = max(1e-6, float(hi - lo))
        return np.clip((v - lo) / span, 0.0, 1.0).astype(np.float32, copy=False)

    rms = _nz(rms)
    onset = _nz(onset)
    low = _nz(low)

    x = np.concatenate([mel, rms, onset, low], axis=0).astype(np.float32, copy=False)
    return x


def extract_window(channels: np.ndarray, center_idx: int, window_frames: int) -> np.ndarray:
    c, t = channels.shape
    half = window_frames // 2
    a = int(center_idx) - half
    b = a + window_frames

    out = np.zeros((c, window_frames), dtype=np.float32)
    src_a = max(0, a)
    src_b = min(t, b)
    dst_a = src_a - a
    dst_b = dst_a + (src_b - src_a)
    if src_b > src_a:
        out[:, dst_a:dst_b] = channels[:, src_a:src_b]
    return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, p_drop: float = 0.1):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnchorCNN(nn.Module):
    def __init__(self, in_channels: int, width: int = 96, dropout: float = 0.15):
        super().__init__()
        self.b1 = ConvBlock(in_channels, width, k=7, p_drop=dropout)
        self.b2 = ConvBlock(width, width * 2, k=5, p_drop=dropout)
        self.b3 = ConvBlock(width * 2, width * 2, k=5, p_drop=dropout)
        self.b4 = ConvBlock(width * 2, width * 3, k=3, p_drop=dropout)
        self.head = nn.Sequential(
            nn.Linear((width * 3) * 2, width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.b1(x)
        z = self.b2(z)
        z = self.b3(z)
        z = self.b4(z)
        avg = z.mean(dim=-1)
        mx = z.amax(dim=-1)
        h = torch.cat([avg, mx], dim=-1)
        return self.head(h).squeeze(-1)


def build_model(in_channels: int) -> AnchorCNN:
    return AnchorCNN(in_channels=in_channels)

