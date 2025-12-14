# -*- coding: utf-8 -*-
"""
Vessel Graph Network (VGN) - Pure PyTorch implementation, GPU ready.

This is extracted / cleaned from your mixed PyTorch+TF file and stripped
of all TensorFlow/Keras dependencies so it runs cleanly on modern GPUs
(e.g. RTX 4090) with current Python stacks.

Usage (GPU):

    import torch
    from model_vgn import VGN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGN(gnn_heads=4, gnn_hidden=16).to(device)
    model.eval()  # or .train() if you implement training

    # Dummy input: 1 RGB image, 256x256
    x = torch.randn(1, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(x, edge_mode="euclidean")  # or "geodesic"
        p_cnn = out["p_cnn"]  # [1, 1, H, W] CNN prob map
        p_vgn = out["p_vgn"]  # [1, 1, H, W] refined prob map
        p_gnn = out["p_gnn"]  # [V] node probs
        graph = out["graph"]  # verts, edges, etc.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#                Small CNN backbone (DRIU-lite)
# ============================================================

class _Conv(nn.Module):
    def __init__(self, c_in, c_out, ks=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, ks, s, p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _VGGBlock(nn.Module):
    def __init__(self, c_in, c_out, n=2):
        super().__init__()
        layers = []
        ch = c_in
        for _ in range(n):
            layers.append(_Conv(ch, c_out))
            ch = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DRIUBackbone(nn.Module):
    """
    DRIU-style CNN backbone producing:
      - p_cnn: coarse vessel prob map [B,1,H,W]
      - feats: multi-scale CNN feature maps (f1..f5)
    """
    def __init__(self):
        super().__init__()
        self.b1 = _VGGBlock( 3,  64, n=2); self.p1 = nn.MaxPool2d(2)
        self.b2 = _VGGBlock(64, 128, n=2); self.p2 = nn.MaxPool2d(2)
        self.b3 = _VGGBlock(128,256, n=3); self.p3 = nn.MaxPool2d(2)
        self.b4 = _VGGBlock(256,512, n=3); self.p4 = nn.MaxPool2d(2)
        self.b5 = _VGGBlock(512,512, n=3)

        # side outputs squeezed to 16 channels
        self.s1 = nn.Conv2d( 64, 16, 1)
        self.s2 = nn.Conv2d(128, 16, 1)
        self.s3 = nn.Conv2d(256, 16, 1)
        self.s4 = nn.Conv2d(512, 16, 1)

        # final 1x1 conv to a single vessel prob map
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]

        f1 = self.b1(x)
        f2 = self.b2(self.p1(f1))
        f3 = self.b3(self.p2(f2))
        f4 = self.b4(self.p3(f3))
        f5 = self.b5(self.p4(f4))

        # multi-scale side outputs, upsampled to input resolution
        s1 = F.interpolate(self.s1(f1), size=(h, w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.s2(f2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.s3(f3), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.s4(f4), size=(h, w), mode='bilinear', align_corners=False)

        fused = torch.cat([s1, s2, s3, s4], dim=1)  # [B, 64, H, W]
        p_cnn = torch.sigmoid(self.out(fused))      # [B, 1, H, W]

        feats = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
        return p_cnn, feats


# ============================================================
#                       Simple GAT layer
# ============================================================

class GATLayer(nn.Module):
    """
    Lightweight GAT layer implemented with dense tensors.

    x:          [V, C_in]
    edge_index: [2, E] (src, dst) indices
    Output:     [V, heads * out_ch]
    """
    def __init__(self, in_ch, out_ch, heads=4, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_ch, heads * out_ch, bias=False)

        self.a_src = nn.Parameter(torch.empty(heads, out_ch))
        self.a_dst = nn.Parameter(torch.empty(heads, out_ch))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x, edge_index):
        V = x.shape[0]
        H = self.heads
        Cout = self.W.out_features // H

        z = self.W(x).view(V, H, Cout)  # [V, H, Cout]
        src, dst = edge_index          # [E], [E]

        z_src = z[src]  # [E, H, Cout]
        z_dst = z[dst]  # [E, H, Cout]

        # Attention coefficients e_ij per head
        e = (z_src * self.a_src).sum(-1) + (z_dst * self.a_dst).sum(-1)  # [E, H]
        e = self.leakyrelu(e)

        out = torch.zeros(V, H, Cout, device=x.device)

        # Head-separable softmax aggregation
        for h in range(H):
            e_h = e[:, h]  # [E]

            # stable per-dst softmax
            max_per_dst = torch.full((V,), -1e9, device=x.device)
            max_per_dst.scatter_reduce_(0, dst, e_h, reduce='amax', include_self=True)
            exp_e = torch.exp(e_h - max_per_dst[dst])
            denom = torch.zeros(V, device=x.device)
            denom.scatter_add_(0, dst, exp_e)
            alpha = exp_e / (denom[dst] + 1e-9)  # [E]

            agg = torch.zeros(V, Cout, device=x.device)
            agg.index_add_(0, dst, z_src[:, h, :] * alpha.unsqueeze(-1))
            out[:, h, :] = F.elu(agg)

        return out.reshape(V, H * Cout)


# ============================================================
#               Graph building & utility functions
# ============================================================

def _align(x, ref_hw):
    """Resize feature map x to match spatial size ref_hw=(H,W)."""
    if x.shape[-2:] != ref_hw:
        x = F.interpolate(x, size=ref_hw, mode='bilinear', align_corners=False)
    return x


def _scale_verts_to_hw(verts, src_hw, dst_hw):
    """
    Scale integer vertex coordinates from src_hw -> dst_hw via linear scaling.
    verts: [V, 2] (y, x) in src_hw coordinates
    """
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    if (Hs, Ws) == (Hd, Wd):
        return verts

    vy = verts[:, 0].float() * (Hd - 1) / max(1.0, (Hs - 1))
    vx = verts[:, 1].float() * (Wd - 1) / max(1.0, (Ws - 1))
    vy = torch.clamp(torch.round(vy), 0, Hd - 1).long()
    vx = torch.clamp(torch.round(vx), 0, Wd - 1).long()
    return torch.stack([vy, vx], dim=1)


def srvs_vertices(prob, delta):
    """
    Spatially Regularized Vertex Sampling (SRVS).

    prob:  [1, H, W] probability map (single-channel, no batch dim)
    delta: sampling stride parameter (float or int)

    Returns:
        verts:   [V, 2] (y,x) sampled vertices
        grid_hw: (Hc, Wc) coarse grid size
        cell:    cell size used
    """
    _, H, W = prob.shape
    cell = 2 * int(delta)
    Hc = math.ceil(H / cell)
    Wc = math.ceil(W / cell)

    ys, xs = [], []

    with torch.no_grad():
        # replicate-pad to multiple of cell
        P = F.pad(prob, (0, Wc * cell - W, 0, Hc * cell - H), mode='replicate')
        # group into (Hc, Wc, cell*cell)
        P = P.view(1, Hc, cell, Wc, cell).permute(0, 1, 3, 2, 4).reshape(1, Hc, Wc, cell * cell)
        idx = P.argmax(dim=-1)  # [1, Hc, Wc]

        for iy in range(Hc):
            for ix in range(Wc):
                off = idx[0, iy, ix].item()
                dy, dx = divmod(off, cell)
                y = min(iy * cell + dy, H - 1)
                x = min(ix * cell + dx, W - 1)
                ys.append(y)
                xs.append(x)

    verts = torch.tensor(list(zip(ys, xs)), dtype=torch.long, device=prob.device)
    return verts, (Hc, Wc), cell


def edge_index_euclidean(verts, k=8):
    """
    k-NN graph in Euclidean pixel space.

    verts: [V, 2] (y,x)
    Returns edge_index: [2, E]
    """
    V = verts.shape[0]
    vy = verts[:, 0].float()
    vx = verts[:, 1].float()
    coords = torch.stack([vy, vx], dim=1)  # [V,2]

    d2 = torch.cdist(coords, coords)  # [V,V]
    knn = torch.topk(d2, k=k + 1, largest=False).indices[:, 1:]  # skip self
    src = torch.arange(V, device=verts.device).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([src.reshape(-1), knn.reshape(-1)], dim=0)  # [2,E]
    return edge_index


def edge_index_geodesic(prob, verts, d_thresh, k_fallback=8):
    """
    Geodesic graph construction in probability space (very expensive but faithful).

    prob:  [1, H, W] prob map (single-channel)
    verts: [V, 2]
    d_thresh: max geodesic distance to accept an edge
    """
    H, W = prob.shape[-2:]
    V = verts.shape[0]

    vy = verts[:, 0].float()
    vx = verts[:, 1].float()
    coords = torch.stack([vy, vx], dim=1)

    with torch.no_grad():
        d2 = torch.cdist(coords, coords)
        knn = torch.topk(d2, k=k_fallback + 1, largest=False).indices[:, 1:]
        edges = []
        P = prob[0]  # [H,W]

        def geo_cost(a, b):
            ay, ax = int(a[0]), int(a[1])
            by, bx = int(b[0]), int(b[1])
            y0, y1 = sorted([ay, by])
            x0, x1 = sorted([ax, bx])

            y0 = max(0, y0 - 8)
            y1 = min(H - 1, y1 + 8)
            x0 = max(0, x0 - 8)
            x1 = min(W - 1, x1 + 8)

            sub = P[y0:y1 + 1, x0:x1 + 1].clone()
            sy, sx = ay - y0, ax - x0
            ty, tx = by - y0, bx - x0

            inf = 1e9
            dist = torch.full_like(sub, inf)
            dist[sy, sx] = 0.0
            vis = torch.zeros_like(sub, dtype=torch.bool)

            # Dijkstra over local patch
            for _ in range(sub.numel()):
                cur = torch.where(~vis, dist, torch.tensor(inf, device=sub.device)).view(-1)
                idx = torch.argmin(cur).item()
                cy = idx // sub.shape[1]
                cx = idx % sub.shape[1]
                if vis[cy, cx]:
                    break
                vis[cy, cx] = True
                if cy == ty and cx == tx:
                    break

                p0 = sub[cy, cx]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < sub.shape[0] and 0 <= nx < sub.shape[1]:
                            w = (p0 - sub[ny, nx]).abs()
                            if dist[cy, cx] + w < dist[ny, nx]:
                                dist[ny, nx] = dist[cy, cx] + w

            return float(dist[ty, tx].item())

        for i in range(V):
            a = coords[i]
            for j in knn[i]:
                b = coords[j]
                gd = geo_cost(a, b)
                if gd <= d_thresh:
                    edges.append([i, int(j.item())])

        # fallback to simple chain if everything got filtered
        if not edges:
            for i in range(V - 1):
                edges.append([i, i + 1])

        return torch.tensor(edges, dtype=torch.long, device=prob.device).t().contiguous()


# ============================================================
#                 Inference (decoder) module
# ============================================================

class InferenceModule(nn.Module):
    """
    Upsampling decoder that fuses GNN features with CNN features.
    """
    def __init__(self, gnn_ch, cnn_chs=(64, 128, 256, 512, 512)):
        super().__init__()
        self.compress = nn.Conv2d(gnn_ch, 64, 1)

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.fuse4 = _VGGBlock(64 + cnn_chs[3], 64, n=2)

        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.fuse3 = _VGGBlock(64 + cnn_chs[2], 64, n=2)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.fuse2 = _VGGBlock(64 + cnn_chs[1], 64, n=2)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.fuse1 = _VGGBlock(64 + cnn_chs[0], 64, n=2)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, gnn_tensor, feats):
        gnn_tensor = _align(gnn_tensor, feats["f5"].shape[-2:])
        x = self.compress(gnn_tensor)

        x = self.up4(x)
        x = _align(x, feats["f4"].shape[-2:])
        x = self.fuse4(torch.cat([x, feats["f4"]], dim=1))

        x = self.up3(x)
        x = _align(x, feats["f3"].shape[-2:])
        x = self.fuse3(torch.cat([x, feats["f3"]], dim=1))

        x = self.up2(x)
        x = _align(x, feats["f2"].shape[-2:])
        x = self.fuse2(torch.cat([x, feats["f2"]], dim=1))

        x = self.up1(x)
        x = _align(x, feats["f1"].shape[-2:])
        x = self.fuse1(torch.cat([x, feats["f1"]], dim=1))

        return torch.sigmoid(self.out(x))


# ============================================================
#                          VGN model
# ============================================================

class VGN(nn.Module):
    """
    Full Vessel Graph Network in PyTorch.

    forward(x, cache_graph=None, delta=4, geo_thresh=10, edge_mode="geodesic")

    x: [1, 3, H, W] (batch = 1 enforced)
    Returns dict:
        {
          "p_cnn": CNN prob map,  [1,1,H,W]
          "p_vgn": refined map,   [1,1,H,W]
          "p_gnn": node probs,    [V]
          "graph": {
             "verts": [V,2] (y,x in original CNN grid),
             "edges": [2,E],
             "grid_hw": (Hc,Wc),
             "cell": cell_size,
             "src_hw": (H_src,W_src),
          }
        }
    """
    def __init__(self, gnn_heads=4, gnn_hidden=16):
        super().__init__()
        self.cnn = DRIUBackbone()

        self.gat1 = GATLayer(in_ch=5, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat2 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat3 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)

        self.gnn_pred = nn.Linear(gnn_hidden * gnn_heads, 1)
        self.infer = InferenceModule(gnn_ch=gnn_hidden * gnn_heads)

    @torch.no_grad()
    def _tensorize_gnn(self, verts, grid_hw, cell, gnn_feat, ref_feat_spatial):
        """
        Scatter node features onto a coarse grid, then upsample to ref_feat_spatial.

        verts:    [V,2]
        grid_hw:  (Hc,Wc)
        cell:     int
        gnn_feat: [V,Cg]
        """
        Hc, Wc = grid_hw
        Cg = gnn_feat.shape[1]

        grid = torch.zeros(1, Cg, Hc, Wc, device=gnn_feat.device)
        ys = torch.clamp(verts[:, 0] // cell, 0, Hc - 1)
        xs = torch.clamp(verts[:, 1] // cell, 0, Wc - 1)
        grid[0, :, ys, xs] = gnn_feat.T

        return F.interpolate(grid, size=ref_feat_spatial, mode='bilinear', align_corners=False)

    def forward(self, x, cache_graph=None, delta=4, geo_thresh=10, edge_mode="geodesic"):
        assert x.shape[0] == 1, "VGN assumes batch=1."

        # CNN backbone
        p_cnn, feats = self.cnn(x)
        H, W = p_cnn.shape[-2:]
        prob = p_cnn.detach()

        # Build or reuse graph
        if cache_graph is None:
            verts, grid_hw, cell = srvs_vertices(prob[0], delta)
            if edge_mode == "geodesic":
                edges = edge_index_geodesic(prob[0], verts, geo_thresh)
            else:
                edges = edge_index_euclidean(verts)

            cache_graph = {
                "verts": verts,
                "edges": edges,
                "grid_hw": grid_hw,
                "cell": cell,
                "src_hw": (H, W),
            }

        verts_raw = cache_graph["verts"]
        src_hw = cache_graph.get("src_hw", (H, W))
        verts = _scale_verts_to_hw(verts_raw, src_hw, (H, W))

        cell = 2 * int(delta)
        grid_hw = (math.ceil(H / cell), math.ceil(W / cell))
        edges = cache_graph["edges"]

        # Node features: [y_norm, x_norm, p, g, 1-p]
        with torch.no_grad():
            y = (verts[:, 0].float() / max(1, (H - 1))).unsqueeze(1)
            xk = (verts[:, 1].float() / max(1, (W - 1))).unsqueeze(1)
            p = prob[0, 0, verts[:, 0], verts[:, 1]].unsqueeze(1)
            g = x[0, 1, verts[:, 0], verts[:, 1]].unsqueeze(1)  # using G-channel as extra cue
            node_feat = torch.cat([y, xk, p, g, 1.0 - p], dim=1)  # [V,5]

        # GAT stack
        h1 = self.gat1(node_feat, edges)
        h2 = self.gat2(h1, edges)
        h3 = self.gat3(h2, edges)

        # Node classifier
        p_gnn = torch.sigmoid(self.gnn_pred(h3)).squeeze(1)  # [V]

        # Scatter GNN features to dense map, run decoder
        gnn_tensor = self._tensorize_gnn(verts, grid_hw, cell, h3, feats["f5"].shape[-2:])
        p_vgn = self.infer(gnn_tensor, feats)  # [1,1,H,W]

        graph_snapshot = {
            "verts": verts_raw,
            "edges": edges,
            "grid_hw": grid_hw,
            "cell": cell,
            "src_hw": src_hw,
        }

        return {
            "p_cnn": p_cnn,
            "p_vgn": p_vgn,
            "p_gnn": p_gnn,
            "graph": graph_snapshot,
        }


# ============================================================
#                      Quick smoke test
# ============================================================

if __name__ == "__main__":
    # This is just a quick shape check; on a 4090 you’ll run this with device="cuda".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = VGN(gnn_heads=4, gnn_hidden=16).to(device)
    model.eval()

    # Dummy input image
    x = torch.randn(1, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(x, edge_mode="euclidean")  # euclidean edges are cheaper than geodesic

    print("p_cnn:", out["p_cnn"].shape)
    print("p_vgn:", out["p_vgn"].shape)
    print("p_gnn:", out["p_gnn"].shape)
    print("num verts:", out["graph"]["verts"].shape[0])
    print("num edges:", out["graph"]["edges"].shape[1])
