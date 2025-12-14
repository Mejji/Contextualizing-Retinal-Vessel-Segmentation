# -*- coding: utf-8 -*-
"""
AFF_Module.py
A compact Adaptive Feature Fusion (AFF) block for fusing DAU2 CNN maps with GNN tensors.

Design:
  - Channel alignment: 1x1 conv + BN + ReLU to map CNN and GNN features to the same 'mid_ch'.
  - Gating: μ = sigmoid(Conv([C||G])) produces a pixel-wise gate in [0,1] with mid_ch channels.
  - Fusion: F = μ*C + (1-μ)*G
  - Refinement: small 3x3 -> 3x3 conv stack + BN + ReLU
  - Head: 1x1 conv -> logits; sigmoid outside if you want probabilities

Also ships a small helper to rasterize node features onto an image grid.

Author: you asked for a minimal, dependency-free module. No TF, no utils.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_spatial(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Resize feature map to (H,W) if needed."""
    if x.shape[-2:] != hw:
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return x


class AFFModule(nn.Module):
    """
    Adaptive Feature Fusion (AFF).

    Args:
        in_ch_cnn:  channels of the CNN branch (DAU2 probmaps → usually 1)
        in_ch_gnn:  channels of the GNN branch (after last GAT layer)
        mid_ch:     internal aligned channels for both branches
        out_ch:     output channels (1 for binary vessel prob)

    Forward:
        Inputs:
            cnn_feat: [B, Cc, H, W] (can be [B,1,H,W] DAU2 probability map)
            gnn_feat: [B, Cg, H', W'] (dense rasterized GNN feature map)
        Returns:
            prob:     [B, out_ch, H, W]  -- sigmoid(logits)
            aux:      dict with ("mu", "fused", "refined", "logits")

    """
    def __init__(self, in_ch_cnn=1, in_ch_gnn=64, mid_ch=64, out_ch=1):
        super().__init__()
        self.cnn_align = nn.Sequential(
            nn.Conv2d(in_ch_cnn, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.gnn_align = nn.Sequential(
            nn.Conv2d(in_ch_gnn, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        # Gate μ: predict per-channel, per-pixel mixing weights
        self.gate = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        # Lightweight refinement
        self.refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def forward(self, cnn_feat: torch.Tensor, gnn_feat: torch.Tensor):
        if cnn_feat.dim() == 3:
            cnn_feat = cnn_feat.unsqueeze(1)  # [B,1,H,W]
        if gnn_feat.dim() == 3:
            gnn_feat = gnn_feat.unsqueeze(1)

        H, W = cnn_feat.shape[-2:]
        gnn_feat = _align_spatial(gnn_feat, (H, W))

        c = self.cnn_align(cnn_feat)
        g = self.gnn_align(gnn_feat)

        mu = self.gate(torch.cat([c, g], dim=1))           # [B, mid, H, W]
        fused = mu * c + (1.0 - mu) * g                    # [B, mid, H, W] # AFF CALCULATION for GNN and CNN features
        f = self.refine(fused)                              # [B, mid, H, W]
        logits = self.head(f)                               # [B, out, H, W]
        prob = torch.sigmoid(logits)
        return prob, {"mu": mu, "fused": fused, "refined": f, "logits": logits}


# -------------------------- Rasterizer (GNN -> dense map) --------------------------

@torch.no_grad()
def rasterize_gnn_features(
    verts: torch.Tensor,
    grid_hw: Tuple[int, int],
    cell: int,
    node_feat: torch.Tensor,
    out_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Scatter node features onto a coarse grid and upsample to full resolution.

    Args:
        verts:     [V, 2] long (y, x) in original image pixels
        grid_hw:   (Hc, Wc) grid size (ceil(H/cell), ceil(W/cell))
        cell:      cell size (usually 2*win_size or stride used by SRNS)
        node_feat: [V, C] node features (from the last GAT layer)
        out_hw:    (H, W) target spatial size

    Returns:
        dense:     [1, C, H, W] dense feature map
    """
    assert verts.ndim == 2 and verts.shape[1] == 2
    V, C = node_feat.shape
    Hc, Wc = int(grid_hw[0]), int(grid_hw[1])

    grid = torch.zeros(1, C, Hc, Wc, device=node_feat.device, dtype=node_feat.dtype)

    ys = torch.clamp(verts[:, 0] // int(cell), 0, Hc - 1)
    xs = torch.clamp(verts[:, 1] // int(cell), 0, Wc - 1)
    grid[0, :, ys, xs] = node_feat.t()  # scatter

    dense = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)
    return dense
