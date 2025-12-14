# -*- coding: utf-8 -*-
"""
gat_only.py

Pure PyTorch Graph Attention Network (GAT) baseline.

This module gives you:

  • GATLayer – multi-head graph attention layer (PyG-style interface).
  • GATOnlyNet – stack of GAT layers + final linear head for node-level tasks.

Expected input shapes:
  • x          : [V, F]          node features (float32/float16)
  • edge_index : [2, E] (long)   directed edges (src, dst) in COO format

Designed to be safe under AMP + torch.compile:
  - attention *scores* are kept in a single dtype (score_dtype)
  - scatter_reduce_ / scatter_add_ always see matching dtypes
  - feature aggregation stays in the feature dtype (usually half under AMP)
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Multi‑head Graph Attention Layer
# -------------------------------------------------------------------------
class GATLayer(nn.Module):
    """
    Basic multi-head GAT layer.

    Inputs:
      x          : [V, Fin]
      edge_index : [2, E]  (src, dst)

    Output:
      h          : [V, heads * Fout]
    """
    def __init__(self, in_ch: int, out_ch: int, heads: int = 4,
                 negative_slope: float = 0.2):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_ch, heads * out_ch, bias=False)

        # separate attention vectors per head (source & destination)
        self.a_src = nn.Parameter(torch.empty(heads, out_ch))
        self.a_dst = nn.Parameter(torch.empty(heads, out_ch))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [V, Fin]
        edge_index: [2, E] (src, dst)

        returns: [V, heads * Fout]
        """
        if x.dim() != 2:
            raise ValueError(f"GATLayer expects [V, F], got {x.shape}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")

        V = x.shape[0]
        H = self.heads
        Cout = self.W.out_features // H

        # Linear projection + reshape per head
        z = self.W(x).view(V, H, Cout)          # [V, H, Cout]

        src, dst = edge_index                  # [E], [E]
        z_src = z[src]                         # [E, H, Cout]
        z_dst = z[dst]                         # [E, H, Cout]

        # attention logits e_ij per head
        # NOTE: compute in a "score dtype" (usually float32 under AMP)
        e = (z_src * self.a_src).sum(-1) + (z_dst * self.a_dst).sum(-1)  # [E, H]
        e = self.leakyrelu(e)
        score_dtype = e.dtype

        # output tensor (features stay in z's dtype)
        out = torch.zeros(V, H, Cout, device=x.device, dtype=z.dtype)

        for h in range(H):
            e_h = e[:, h]                      # [E] in score_dtype

            # max per destination for numerical stability
            max_per_dst = torch.full(
                (V,),
                -1e9,
                device=x.device,
                dtype=score_dtype,             # <-- matches e_h.dtype
            )
            max_per_dst.scatter_reduce_(
                0, dst, e_h, reduce='amax', include_self=True
            )

            # softmax over incoming edges
            exp_e = torch.exp(e_h - max_per_dst[dst])  # [E], score_dtype
            denom = torch.zeros(V, device=x.device, dtype=score_dtype)
            denom.scatter_add_(0, dst, exp_e)
            alpha = exp_e / (denom[dst] + 1e-9)        # [E], score_dtype

            # cast attention weights to feature dtype before aggregation
            alpha_f = alpha.to(z_src.dtype)            # same as z_src.dtype
            agg = torch.zeros(V, Cout, device=x.device, dtype=z_src.dtype)
            agg.index_add_(0, dst, z_src[:, h, :] * alpha_f.unsqueeze(-1))
            out[:, h, :] = F.elu(agg)

        return out.reshape(V, H * Cout)        # [V, H * Cout]


# -------------------------------------------------------------------------
# GAT-only network
# -------------------------------------------------------------------------
class GATOnlyNet(nn.Module):
    """
    Stack of GAT layers + final linear head.

    This is a generic node-level model:
      • binary classification if out_dim == 1 (sigmoid on top)
      • multi-class classification if out_dim > 1 (softmax on top)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 3,
        heads: int = 4,
        out_dim: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.out_dim = out_dim
        self.residual = residual
        self.use_dropout = dropout > 0.0
        self.dropout = nn.Dropout(dropout) if self.use_dropout else None

        layers = []
        h_in = in_dim
        for _ in range(num_layers):
            layers.append(GATLayer(in_ch=h_in, out_ch=hidden_dim, heads=heads))
            h_in = hidden_dim * heads
        self.layers = nn.ModuleList(layers)

        self.head = nn.Linear(h_in, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.layers:
            if isinstance(m, GATLayer):
                nn.init.xavier_uniform_(m.W.weight)
                # a_src/a_dst already init'ed in ctor
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        """
        x:          [V, F]
        edge_index: [2, E]

        If return_dict=False:
            returns logits:
              • [V]    if out_dim == 1
              • [V, C] if out_dim > 1
        """
        if x.dim() != 2:
            raise ValueError(f"GATOnlyNet expects x [V, F], got {x.shape}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")

        h = x
        for li, layer in enumerate(self.layers):
            h_new = layer(h, edge_index)
            if self.residual and h_new.shape == h.shape:
                h = h_new + h
            else:
                h = h_new

            if self.use_dropout and li < len(self.layers) - 1:
                h = self.dropout(h)

        logits = self.head(h)  # [V, out_dim]
        if self.out_dim == 1:
            logits = logits.squeeze(-1)  # [V]

        if not return_dict:
            return logits
        return {"logits": logits, "node_feat": h}

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience helper:

        • For out_dim == 1 → returns sigmoid(logits)  [V]
        • For out_dim > 1  → returns softmax(logits) [V, C]
        """
        logits = self.forward(x, edge_index, return_dict=False)
        if self.out_dim == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


# -------------------------------------------------------------------------
# Small helper
# -------------------------------------------------------------------------
def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """
    Convert dense adjacency [V, V] to edge_index [2, E].

    adj: [V, V] (bool / 0-1 / weights > 0 treated as edges)
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency_to_edge_index expects [V, V], got {adj.shape}")
    # nonzero gives (row, col) = (dst, src) if you view as matrix;
    # we want (src, dst) so we flip.
    dst, src = (adj > 0).nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()
    return edge_index
