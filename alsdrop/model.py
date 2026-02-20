#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_MAX_OFFSET_SEC, DEFAULT_SHORT_LEFT_SEC, DEFAULT_SHORT_RIGHT_SEC, DEFAULT_LONG_LEFT_SEC, DEFAULT_LONG_RIGHT_SEC
from .audio_features import channel_matrix, interpolate_context


@dataclass
class ContextConfig:
    long_left_sec: float = DEFAULT_LONG_LEFT_SEC
    long_right_sec: float = DEFAULT_LONG_RIGHT_SEC
    short_left_sec: float = DEFAULT_SHORT_LEFT_SEC
    short_right_sec: float = DEFAULT_SHORT_RIGHT_SEC
    long_frames: int = 384
    short_frames: int = 128


class ConvFrontEnd(nn.Module):
    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.front = ConvFrontEnd(in_channels=in_channels, width=width)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=nhead,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.front(x)  # [B, C, T]
        z = z.transpose(1, 2)  # [B, T, C]
        z = self.encoder(z)
        z = self.norm(z)
        mean_pool = torch.mean(z, dim=1)
        max_pool = torch.amax(z, dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)


class DropAnchorModel(nn.Module):
    """
    CNN+Transformer ranking model with an offset refinement head.
    Inputs:
      long_ctx  : [B, C, T_long]
      short_ctx : [B, C, T_short]
      meta      : [B, M]
    Outputs:
      rank_logit: [B]
      offset_sec: [B] (clipped to +/- max_offset_sec)
    """

    def __init__(
        self,
        in_channels: int,
        meta_dim: int = 4,
        width: int = 128,
        dropout: float = 0.15,
        max_offset_sec: float = DEFAULT_MAX_OFFSET_SEC,
    ):
        super().__init__()
        self.max_offset_sec = float(max_offset_sec)
        self.long_branch = BranchEncoder(in_channels=in_channels, width=width, nhead=4, layers=2, dropout=dropout)
        self.short_branch = BranchEncoder(in_channels=in_channels, width=width, nhead=4, layers=1, dropout=dropout)

        in_dim = (width * 2) + (width * 2) + int(meta_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rank_head = nn.Linear(width * 2, 1)
        self.offset_head = nn.Linear(width * 2, 1)

    def forward(self, long_ctx: torch.Tensor, short_ctx: torch.Tensor, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        long_vec = self.long_branch(long_ctx)
        short_vec = self.short_branch(short_ctx)
        h = torch.cat([long_vec, short_vec, meta], dim=-1)
        h = self.shared(h)
        rank_logit = self.rank_head(h).squeeze(-1)
        offset = torch.tanh(self.offset_head(h).squeeze(-1)) * self.max_offset_sec
        return rank_logit, offset


def build_model(
    in_channels: int,
    meta_dim: int = 4,
    width: int = 128,
    dropout: float = 0.15,
    max_offset_sec: float = DEFAULT_MAX_OFFSET_SEC,
) -> DropAnchorModel:
    return DropAnchorModel(
        in_channels=in_channels,
        meta_dim=meta_dim,
        width=width,
        dropout=dropout,
        max_offset_sec=max_offset_sec,
    )


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    track_ids: torch.Tensor,
    margin: float = 0.40,
) -> torch.Tensor:
    """
    Enforce positive candidate score > negative candidates within each track.
    """
    losses = []
    uniq = torch.unique(track_ids)
    for tid in uniq:
        m = track_ids == tid
        y = labels[m]
        s = logits[m]
        pos = s[y > 0.5]
        neg = s[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        p = torch.mean(pos)
        n = torch.max(neg)
        losses.append(F.relu(float(margin) - p + n))
    if not losses:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return torch.mean(torch.stack(losses))


def listwise_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    track_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Listwise ranking objective:
      - for each track group, maximize softmax probability at positive candidate.
      - labels should be one-hot per track (or sparse positives).
    """
    losses = []
    uniq = torch.unique(track_ids)
    for tid in uniq:
        m = track_ids == tid
        y = labels[m]
        s = logits[m]
        if s.numel() == 0:
            continue
        y_sum = torch.sum(y)
        if y_sum <= 0:
            continue
        log_p = F.log_softmax(s, dim=0)
        y_norm = y / torch.clamp(y_sum, min=1.0)
        losses.append(-torch.sum(y_norm * log_p))
    if not losses:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return torch.mean(torch.stack(losses))


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = max(1e-3, float(temperature))
    return logits / t


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Simple deterministic grid search temperature calibration on validation logits.
    """
    if logits.size == 0 or labels.size == 0:
        return 1.0
    y = labels.astype(np.float64, copy=False)
    z = logits.astype(np.float64, copy=False)

    def _nll(temp: float) -> float:
        p = sigmoid_np((z / temp).astype(np.float64, copy=False))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    temps = np.concatenate(
        [
            np.linspace(0.4, 1.2, num=24),
            np.linspace(1.2, 3.0, num=24),
            np.linspace(3.0, 6.0, num=12),
        ]
    )
    best_t = 1.0
    best_loss = float("inf")
    for t in temps.tolist():
        loss = _nll(float(t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return float(best_t)


def build_candidate_contexts(
    feature_dict: Dict[str, np.ndarray],
    candidate_times: Sequence[float],
    cfg: Optional[ContextConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = ContextConfig()

    channels = channel_matrix(feature_dict)
    frame_times = feature_dict["frame_times"].astype(np.float32, copy=False)

    long_ctx = []
    short_ctx = []
    for t in candidate_times:
        long_ctx.append(
            interpolate_context(
                channels=channels,
                frame_times=frame_times,
                center_sec=float(t),
                left_sec=float(cfg.long_left_sec),
                right_sec=float(cfg.long_right_sec),
                out_frames=int(cfg.long_frames),
            )
        )
        short_ctx.append(
            interpolate_context(
                channels=channels,
                frame_times=frame_times,
                center_sec=float(t),
                left_sec=float(cfg.short_left_sec),
                right_sec=float(cfg.short_right_sec),
                out_frames=int(cfg.short_frames),
            )
        )

    if not long_ctx:
        c = channels.shape[0]
        return (
            np.zeros((0, c, int(cfg.long_frames)), dtype=np.float32),
            np.zeros((0, c, int(cfg.short_frames)), dtype=np.float32),
        )

    return (
        np.stack(long_ctx, axis=0).astype(np.float32, copy=False),
        np.stack(short_ctx, axis=0).astype(np.float32, copy=False),
    )
