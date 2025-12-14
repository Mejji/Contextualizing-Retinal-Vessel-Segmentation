#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_DAU2_AFF_VGN.py  (stabilized + AMP/EMA + resume + cosine LR)

What was broken:
- HRF split builder was feeding stems like '01_dr_label' into the prob cache,
  so it looked for '01_dr_label_prob.png' which doesn't exist.

Fixes:
- HRF split builder now strips *_label/*_mask/*_manual1 and keeps only *_dr bases.
- Prob cache strips those suffixes again and (last resort) maps *_g|*_h -> *_dr.
- Graph roots already point at /workspace/DAU_HRF/HRF/{train,test}.

Run example (HRF):
  python code/train_DAU2_AFF_VGN.py \
    --dataset HRF --win_size 10 --edge_dist_thresh 80 \
    --save_root /workspace/DAU2_HRF/VGN_AFF --run_name AFF_VGN_HRF --amp --ema_decay 0.999
"""

import os, sys, json, argparse, time, math, re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import skimage.io
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Prefer new AMP API; fall back for older torch
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old
    _USE_NEW_AMP = False

from config import cfg
import util as util
try:
    from Modules.gat_only import GATLayer  # legacy name
except ImportError:
    # fallback if module only defines the optimized layer
    from Modules.gat_only import GATLayerFast as GATLayer

# ----------------------------- Determinism / speed -----------------------------
def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ----------------------------- AFF -----------------------------
def _align_spatial(x: torch.Tensor, hw):
    if x.shape[-2:] != hw:
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return x

class AFFModule(nn.Module):
    def __init__(self, in_ch_cnn=1, in_ch_gnn=64, mid_ch=64, out_ch=1):
        super().__init__()
        self.cnn_align = nn.Sequential(
            nn.Conv2d(in_ch_cnn, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.gnn_align = nn.Sequential(
            nn.Conv2d(in_ch_gnn, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, cnn_feat: torch.Tensor, gnn_feat: torch.Tensor):
        if cnn_feat.dim() == 3: cnn_feat = cnn_feat.unsqueeze(1)
        if gnn_feat.dim() == 3: gnn_feat = gnn_feat.unsqueeze(1)
        H, W = cnn_feat.shape[-2:]
        gnn_feat = _align_spatial(gnn_feat, (H, W))
        c = self.cnn_align(cnn_feat)
        g = self.gnn_align(gnn_feat)
        mu = self.gate(torch.cat([c, g], dim=1))
        fused = mu * c + (1.0 - mu) * g
        f = self.refine(fused)
        logits = self.head(f)
        prob = torch.sigmoid(logits)
        return prob, {"mu": mu, "fused": fused, "refined": f, "logits": logits}

@torch.no_grad()
def rasterize_gnn_features(verts: torch.Tensor, grid_hw, cell: int,
                           node_feat: torch.Tensor, out_hw):
    V, C = node_feat.shape
    Hc, Wc = int(grid_hw[0]), int(grid_hw[1])
    grid = torch.zeros(1, C, Hc, Wc, device=node_feat.device, dtype=node_feat.dtype)
    ys = torch.clamp(verts[:, 0] // int(cell), 0, Hc - 1)
    xs = torch.clamp(verts[:, 1] // int(cell), 0, Wc - 1)
    grid[0, :, ys, xs] = node_feat.t()
    dense = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)
    return dense

# ----------------------------- Small utilities -----------------------------
def _resize_4d(x: torch.Tensor, size_hw, mode='bilinear', align_corners=False):
    if x is None: return None
    Ht, Wt = size_hw
    Hx, Wx = x.shape[-2:]
    if Hx == Ht and Wx == Wt: return x
    return F.interpolate(x, size=(Ht, Wt), mode=mode, align_corners=align_corners)

def _maybe_override_graph_dirs(train_probs, test_probs):
    if train_probs:
        os.makedirs(train_probs, exist_ok=True)
        setattr(cfg.PATHS, "GRAPH_TRAIN_DIR", train_probs)
        print(f"[CFG] GRAPH_TRAIN_DIR -> {cfg.PATHS.GRAPH_TRAIN_DIR}")
    if test_probs:
        os.makedirs(test_probs, exist_ok=True)
        setattr(cfg.PATHS, "GRAPH_TEST_DIR", test_probs)
        print(f"[CFG] GRAPH_TEST_DIR  -> {cfg.PATHS.GRAPH_TEST_DIR}")

def _default_dau_root(dataset: str) -> Path:
    dataset = dataset.upper()
    defaults = {
        'DRIVE': Path('/workspace/DAU2_DRIVE'),
        'CHASE_DB1': Path('/workspace/DAU2_CHASE/CHASE-DB1'),
        'HRF': Path('/workspace/DAU_HRF/HRF'),
    }
    return defaults.get(dataset, defaults['DRIVE'])

def _resolve_probmap_roots(dataset: str, dau_root: Path,
                           train_override: str, test_override: str) -> Tuple[Path, Path]:
    dataset = dataset.upper()
    def _pick(split: str, override: str) -> Path:
        if override:
            root = Path(override)
        else:
            candidate = dau_root / split
            prob_candidate = candidate / 'prob'
            root = prob_candidate if prob_candidate.exists() else candidate
        if not root.exists():
            raise FileNotFoundError(f"Prob-map root not found for {dataset} split '{split}': {root}")
        return root
    train_root = _pick('train', train_override)
    test_root  = _pick('test',  test_override)
    return train_root, test_root

def _hrf_dataset_root() -> Path:
    txt_path = getattr(cfg.TRAIN, 'HRF_SET_TXT_PATH', '/workspace/DATASETS/HRF/training/images/train_768.txt')
    p = Path(txt_path)
    try:
        return p.parents[2]  # .../HRF
    except IndexError:
        return Path('/workspace/DATASETS/HRF')

def _list_hrf_split() -> Tuple[List[str], List[str]]:
    """
    Build lists of canonical HRF bases (keep only *_dr). Strip *_label/*_mask/*_manual1 from filenames.
    Returns paths WITHOUT extension: <root>/<split>/images/<base>   where base like '01_dr'.
    """
    root = _hrf_dataset_root()
    def _collect(split: str) -> List[str]:
        img_dir = root / split / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f"HRF images dir missing: {img_dir}")
        bases = []
        for img_path in sorted(img_dir.glob('*.*')):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                continue
            stem0 = img_path.stem
            base  = re.sub(r'_(label|mask|manual1)$', '', stem0, flags=re.IGNORECASE)
            if not base.lower().endswith('_dr'):   # only DR subset has DAU2 probmaps
                continue
            bases.append(str((img_dir / base).with_suffix('')))  # path without extension
        # unique and stable order
        uniq = []
        seen = set()
        for b in bases:
            if b not in seen:
                uniq.append(b); seen.add(b)
        return uniq
    return _collect('training'), _collect('testing')

def _read_txt_lines(txt_path: str) -> List[str]:
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Split list not found: {txt_path}")
    with open(path) as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def _resolve_split_lists(dataset: str) -> Tuple[List[str], List[str]]:
    dataset = dataset.upper()
    if dataset == 'HRF':
        return _list_hrf_split()
    if dataset == 'DRIVE':
        train_txt, test_txt = cfg.TRAIN.DRIVE_SET_TXT_PATH, cfg.TEST.DRIVE_SET_TXT_PATH
    elif dataset == 'CHASE_DB1':
        train_txt, test_txt = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH, cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif dataset == 'STARE':
        train_txt = getattr(cfg.TRAIN, 'STARE_SET_TXT_PATH', '')
        test_txt  = getattr(cfg.TEST,  'STARE_SET_TXT_PATH', '')
        if not train_txt or not test_txt:
            raise ValueError("STARE split txt paths are not configured in cfg.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return _read_txt_lines(train_txt), _read_txt_lines(test_txt)

def _canon_prob_stem(stem: str) -> str:
    """Strip *_label/*_mask/*_manual1; last resort map *_g|*_h -> *_dr."""
    s = re.sub(r'_(label|mask|manual1)$', '', stem, flags=re.IGNORECASE)
    if re.search(r'_(g|h)$', s, flags=re.IGNORECASE):
        s = re.sub(r'_(g|h)$', '_dr', s, flags=re.IGNORECASE)
    return s

def _load_dau2_prob_592(probs_root: Path, stem: str) -> np.ndarray:
    s = _canon_prob_stem(stem)
    cands = [
        probs_root / f"{s}_prob.npy",
        probs_root / f"{s}_prob.png",
        probs_root / f"{s}.npy",
        probs_root / f"{s}.png",
    ]
    for p in cands:
        if p.exists():
            if p.suffix.lower() == ".npy":
                arr = np.load(str(p)).astype(np.float32)
            else:
                arr = skimage.io.imread(str(p)).astype(np.float32)
            if arr.ndim == 3: arr = arr[..., 0]
            if arr.max() > 1.0: arr = np.clip(arr, 0, 255) / 255.0
            if (not np.isfinite(arr).all()) or arr.min() < 0 or arr.max() > 1.0:
                raise ValueError(f"Bad probmap values in {p}")
            return arr
    raise FileNotFoundError(f"DAU2 prob not found for {stem} under {probs_root}")

def _graph_to_verts_edges(G: nx.Graph):
    H = nx.convert_node_labels_to_integers(G)
    V = H.number_of_nodes()
    verts = np.zeros((V, 2), dtype=np.int64)
    for i in range(V):
        d = H.nodes[i]
        verts[i, 0] = int(d.get("y", d.get("row", 0)))
        verts[i, 1] = int(d.get("x", d.get("col", 0)))
    edges_src, edges_dst = [], []
    for u, v in H.edges():
        edges_src.append(u); edges_dst.append(v)
        edges_src.append(v); edges_dst.append(u)
    if not edges_src and V > 1:
        for i in range(V - 1):
            edges_src.append(i); edges_dst.append(i + 1)
            edges_src.append(i + 1); edges_dst.append(i)
    return (
        torch.as_tensor(verts, dtype=torch.long),
        torch.stack([torch.as_tensor(edges_src, dtype=torch.long),
                     torch.as_tensor(edges_dst, dtype=torch.long)], dim=0)
    )

_GRAPH_TENSOR_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

def get_graph_tensors_cached(graph: nx.Graph, stem: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if stem in _GRAPH_TENSOR_CACHE:
        return _GRAPH_TENSOR_CACHE[stem]
    verts, edge_index = _graph_to_verts_edges(graph)
    _GRAPH_TENSOR_CACHE[stem] = (verts, edge_index)
    return verts, edge_index

def _build_node_features(verts, prob_map_t, img_t):
    device = prob_map_t.device
    H, W = prob_map_t.shape[-2:]
    vy = verts[:, 0].clamp(0, H - 1).long()
    vx = verts[:, 1].clamp(0, W - 1).long()
    y_norm = (vy.float() / max(1.0, (H - 1))).unsqueeze(1)
    x_norm = (vx.float() / max(1.0, (W - 1))).unsqueeze(1)
    p = prob_map_t[0, 0, vy, vx].unsqueeze(1)
    g = img_t[0, 1, vy, vx].unsqueeze(1)
    if g.max() > 1.0: g = g / 255.0
    node_feat = torch.cat([y_norm, x_norm, p, g, 1.0 - p], dim=1).to(device)
    return node_feat

def _class_weights_from_mask(label, fov):
    mask = (fov > 0).float()
    num = mask.sum().item() + 1e-6
    fg = ((label > 0.5).float() * mask).sum().item()
    bg = num - fg
    w_bg = fg / num
    w_fg = bg / num
    return float(w_bg), float(w_fg), mask

def _fmt_eta(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

class RunningStat:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def push(self, x: float):
        if x is None or not np.isfinite(x): return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def mean_std(self):
        if self.n < 2: return (self.mean, float("nan"))
        return (self.mean, math.sqrt(self.M2 / (self.n - 1)))

def _to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def _overlay_on_img(img_rgb, prob, fov=None, alpha=0.6):
    img = img_rgb.astype(np.float32)
    if img.max() > 1.0: img = img / 255.0
    p = np.clip(prob, 0.0, 1.0)
    if fov is not None:
        p = p * (fov.astype(bool).astype(np.float32))
    overlay = img * (1.0 - alpha * p[..., None])
    red = np.zeros_like(img); red[..., 0] = 1.0
    overlay = overlay + red * (alpha * p[..., None])
    return _to_uint8(overlay)

def _safe_imsave(path, arr):
    try:
        skimage.io.imsave(path, arr)
    except Exception:
        pass

def _save_probmaps(out_dir, stem, iter_no, pred_prob, cnn_prob, img_rgb, fov=None):
    os.makedirs(out_dir, exist_ok=True)
    base = f"{stem}_iter{int(iter_no):06d}"
    if fov is not None:
        m = fov.astype(bool)
        pred_png = np.where(m, pred_prob, 0.0)
        cnn_png  = np.where(m, cnn_prob,  0.0)
    else:
        pred_png = pred_prob; cnn_png = cnn_prob
    np.save(os.path.join(out_dir, base + "_post_prob.npy"), pred_prob.astype(np.float32))
    np.save(os.path.join(out_dir, base + "_cnn_prob.npy"),  cnn_prob.astype(np.float32))
    _safe_imsave(os.path.join(out_dir, base + "_post_prob.png"), _to_uint8(pred_png))
    _safe_imsave(os.path.join(out_dir, base + "_cnn_prob.png"),  _to_uint8(cnn_png))
    ov = _overlay_on_img(img_rgb, pred_prob, fov=fov, alpha=0.6)
    _safe_imsave(os.path.join(out_dir, base + "_overlay.png"), ov)

def _safe_log_line(log_dir: str, text: str, filename: str = 'train.log'):
    path = os.path.join(log_dir, filename)
    try:
        with open(path, 'a', encoding='utf-8') as f: f.write(text + '\n')
    except Exception:
        with open(path, 'a', encoding='ascii', errors='ignore') as f: f.write(text + '\n')

# ----------------------------- Model -----------------------------
class AFF_VGN(nn.Module):
    def __init__(self, gnn_hidden=16, gnn_heads=4, aff_mid_ch=64):
        super().__init__()
        in_node_ch = 5
        self.gat1 = GATLayer(in_ch=in_node_ch, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat2 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat3 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.node_pred = nn.Linear(gnn_hidden * gnn_heads, 1)
        self.aff = AFFModule(in_ch_cnn=1, in_ch_gnn=gnn_hidden * gnn_heads, mid_ch=aff_mid_ch, out_ch=1)

    def forward(self, node_feat, edge_index, verts, grid_hw, cell, img_hw, cnn_prob_map):
        H, W = img_hw
        h1 = self.gat1(node_feat, edge_index)
        h2 = self.gat2(h1, edge_index)
        h3 = self.gat3(h2, edge_index)                # [V, Cg]
        node_logits = self.node_pred(h3).squeeze(1)   # [V]
        p_node = torch.sigmoid(node_logits)

        gnn_dense = rasterize_gnn_features(verts, grid_hw, cell, h3, img_hw)  # [1,Cg,H,W]
        prob, aux = self.aff(cnn_prob_map, gnn_dense)
        logits = aux.get("logits", None)
        if logits is not None:
            logits = _resize_4d(logits, (H, W), mode='bilinear', align_corners=False)
            prob = torch.sigmoid(logits)
        else:
            prob = _resize_4d(prob, (H, W), mode='bilinear', align_corners=False)
            p = prob.clamp(1e-6, 1 - 1e-6)
            logits = torch.log(p) - torch.log1p(-p)
        gnn_dense = _resize_4d(gnn_dense, (H, W), mode='bilinear', align_corners=False)
        return prob, {"p_node": p_node, "gnn_dense": gnn_dense, "logits": logits}

# ----------------------------- EMA -----------------------------
class EMAHelper:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        if self.decay > 0:
            base = getattr(model, "_orig_mod", model)
            for n, p in base.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def _iter_params(self, model: nn.Module):
        base = getattr(model, "_orig_mod", model)
        for n, p in base.named_parameters():
            if p.requires_grad:
                yield n, p

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.decay <= 0: return
        for n, p in self._iter_params(model):
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        if self.decay <= 0: return None
        backup = {}
        base = getattr(model, "_orig_mod", model)
        for n, p in base.named_parameters():
            if not p.requires_grad: continue
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup):
        if self.decay <= 0 or backup is None: return
        base = getattr(model, "_orig_mod", model)
        for n, p in base.named_parameters():
            if not p.requires_grad: continue
            if n in backup:
                p.data.copy_(backup[n])

# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train AFF+GAT using DAU2 probmaps + SRNS graphs")
    p.add_argument('--dataset', default='HRF', choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--dau_root', type=str, default='/workspace/DAU_HRF/HRF', help='DAU2 base folder (contains train/test prob+graphs).')
    p.add_argument('--train_probs', type=str, default='/workspace/DAU_HRF/HRF/train', help='Override prob+graph root for train split.')
    p.add_argument('--test_probs', type=str, default='/workspace/DAU_HRF/HRF/test', help='Override prob+graph root for test split.')

    p.add_argument('--win_size', type=int, default=10)
    p.add_argument('--edge_dist_thresh', type=float, default=40,
                   help='Geodesic distance threshold used when loading SRNS graphs (e.g., 80 for HRF).')
    p.add_argument('--gnn_hidden', type=int, default=16)
    p.add_argument('--gnn_heads', type=int, default=4)
    p.add_argument('--aff_mid_ch', type=int, default=64)

    p.add_argument('--max_iters', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--display', type=int, default=100)
    p.add_argument('--snapshot_iters', type=int, default=5000)
    p.add_argument('--test_iters', type=int, default=2000)
    p.add_argument('--cosine_lr', action='store_true', help='Use cosine LR schedule over max_iters')

    p.add_argument('--save_root', type=str, default='/workspace/DAU_HRF/VGN')
    p.add_argument('--run_id', type=int, default=1)
    p.add_argument('--run_name', type=str, default='')

    # Stability knobs
    p.add_argument('--warmup_iters', type=int, default=1000, help='Pixel loss only early.')
    p.add_argument('--node_w_start', type=float, default=0.0)
    p.add_argument('--node_w_end', type=float, default=0.05)
    p.add_argument('--node_w_ramp', type=int, default=15000)
    p.add_argument('--aff_alpha_start', type=float, default=0.10, help='alpha=0 CNN logits; alpha=1 AFF logits.')
    p.add_argument('--aff_alpha_end', type=float, default=0.50)
    p.add_argument('--aff_alpha_ramp', type=int, default=12000)
    p.add_argument('--gat_lr_mult', type=float, default=0.5, help='LR multiplier for GAT+node head vs AFF.')
    p.add_argument('--min_auc_ckpt', type=float, default=0.85, help='Only accept new "best" if AUC >= this.')

    # Qualitative saving controls
    p.add_argument('--save_train_probmaps', action='store_true', default=True)
    p.add_argument('--save_test_probmaps', action='store_true', default=True)

    # AMP & EMA & resume
    p.add_argument('--amp', action='store_true', help='Enable mixed precision.')
    p.add_argument('--ema_decay', type=float, default=0.0, help='EMA decay in [0,1); 0 disables.')
    p.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from.')

    # Misc
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def make_autocast(device_type: str, enabled: bool):
    if _USE_NEW_AMP:
        return _autocast_new(device_type, enabled=enabled)
    else:
        return _autocast_old(enabled=enabled)

def make_scaler(device_type: str, enabled: bool):
    if _USE_NEW_AMP:
        return _GradScaler_new(device_type, enabled=enabled)
    else:
        return _GradScaler_old(enabled=enabled)

# ----------------------------- probmap cache -----------------------------
def build_prob_cache(probs_root: Path, img_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Pre-load DAU2 probmaps once into CPU tensors:
      stem -> [1,1,H,W] float32 in [0,1].
    """
    cache: Dict[str, torch.Tensor] = {}
    for p in img_names:
        stem0 = Path(p).stem
        stem = _canon_prob_stem(stem0)
        arr = _load_dau2_prob_592(probs_root, stem)
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        cache[stem] = t
    print(f"[PROB_CACHE] Loaded {len(cache)} probmaps from {probs_root}")
    return cache

# ----------------------------- Train -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    torch.set_float32_matmul_precision("high")

    dau_root = Path(args.dau_root) if args.dau_root else _default_dau_root(args.dataset)
    dau_root = dau_root.resolve()
    args.dau_root = str(dau_root)

    train_prob_root, test_prob_root = _resolve_probmap_roots(args.dataset, dau_root, args.train_probs, args.test_probs)
    train_prob_root = train_prob_root.resolve()
    test_prob_root  = test_prob_root.resolve()
    args.train_probs = str(train_prob_root)
    args.test_probs  = str(test_prob_root)

    setattr(cfg, 'USE_DAU2_GRAPH_PATHS', True)
    _maybe_override_graph_dirs(args.train_probs, args.test_probs)

    # Resolve lists (HRF now returns only *_dr bases with suffixes stripped)
    train_img_names, test_img_names = _resolve_split_lists(args.dataset)

    # Pre-load probmaps
    train_prob_cache = build_prob_cache(train_prob_root, train_img_names)
    test_prob_cache  = build_prob_cache(test_prob_root,  test_img_names)

    # Data layers (cached graph I/O)
    dl_train = util.GraphDataLayerCached(train_img_names, True, 'srns_geo_dist_binary',
                                         win_size=args.win_size, edge_geo_dist_thresh=args.edge_dist_thresh)
    dl_test  = util.GraphDataLayerCached(test_img_names,  False, 'srns_geo_dist_binary',
                                         win_size=args.win_size, edge_geo_dist_thresh=args.edge_dist_thresh)

    # Paths
    run_name  = args.run_name if args.run_name else f"AFF_VGN_run{int(args.run_id)}"
    run_root  = os.path.join(args.save_root, run_name)
    model_dir = os.path.join(run_root, 'train')
    os.makedirs(model_dir, exist_ok=True)
    train_prob_dir = os.path.join(model_dir, 'probmaps')
    test_prob_dir  = os.path.join(run_root, 'test', 'probmaps')
    if args.save_train_probmaps: os.makedirs(train_prob_dir, exist_ok=True)
    if args.save_test_probmaps:  os.makedirs(test_prob_dir,  exist_ok=True)

    # Log args
    with open(os.path.join(model_dir, "args.json"), "w", encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    # Model
    device = torch.device(args.device)
    net = AFF_VGN(gnn_hidden=args.gnn_hidden, gnn_heads=args.gnn_heads, aff_mid_ch=args.aff_mid_ch).to(device)

    # Use channels_last + TF32 on CUDA
    if device.type == 'cuda':
        net = net.to(memory_format=torch.channels_last)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Param groups
    gat_params = list(net.gat1.parameters()) + list(net.gat2.parameters()) + list(net.gat3.parameters()) + list(net.node_pred.parameters())
    aff_params = list(net.aff.parameters())
    optimizer = Adam([
        {'params': aff_params, 'lr': args.lr},
        {'params': gat_params, 'lr': args.lr * args.gat_lr_mult},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=0.1 * args.lr
    ) if args.cosine_lr else None

    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = make_scaler(amp_device_type, enabled=args.amp)
    ema = EMAHelper(net, decay=args.ema_decay)

    # Optional: torch.compile
    try:
        net = torch.compile(net)  # type: ignore[attr-defined]
        print("[INFO] torch.compile enabled.")
    except Exception:
        print("[INFO] torch.compile not available; skipping.")

    print("Training AFF+GAT...")

    best_dice = -1.0
    t0 = time.time()
    auc_tracker = RunningStat()
    start_iter = 0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        start_iter = int(ckpt.get("iter", 0))
        if "ema" in ckpt and ema.decay > 0:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        print(f"[RESUME] from {args.resume} @ iter {start_iter}")

    # ---------------- loop ----------------
    for it in range(start_iter, args.max_iters):
        img_list, blobs = dl_train.forward()
        if len(img_list) == 0:
            continue

        # ----- TRAIN BATCH -----
        img   = blobs['img'][0]
        label = blobs['label'][0]
        fov   = blobs['fov'][0]
        graph = blobs['graph']
        H, W, _ = img.shape

        stem = Path(img_list[0]).stem  # canonical base, e.g., 01_dr

        pm_t = train_prob_cache[_canon_prob_stem(stem)].to(device, non_blocking=True)

        # tensors to GPU, channels_last
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device, non_blocking=True)
        lbl_t = torch.from_numpy(label[..., 0]).float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        fov_t = torch.from_numpy(fov[..., 0]).float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

        lbl_t = _resize_4d(lbl_t, (H, W), mode='nearest')
        fov_t = _resize_4d(fov_t, (H, W), mode='nearest')
        pm_t  = _resize_4d(pm_t,  (H, W), mode='bilinear')

        if device.type == 'cuda':
            img_t = img_t.contiguous(memory_format=torch.channels_last)
            lbl_t = lbl_t.contiguous(memory_format=torch.channels_last)
            fov_t = fov_t.contiguous(memory_format=torch.channels_last)
            pm_t  = pm_t.contiguous(memory_format=torch.channels_last)

        # verts / edges cached per stem
        verts_cpu, edge_index_cpu = get_graph_tensors_cached(graph, stem)
        verts      = verts_cpu.to(device, non_blocking=True)
        edge_index = edge_index_cpu.to(device, non_blocking=True)
        node_feat  = _build_node_features(verts, pm_t, img_t)

        vy = verts[:, 0].clamp(0, H - 1).long()
        vx = verts[:, 1].clamp(0, W - 1).long()
        node_lbl = (lbl_t[0, 0, vy, vx] > 0.5).float()

        cell   = int(2 * args.win_size)
        grid_hw = (int(np.ceil(H / cell)), int(np.ceil(W / cell)))

        # forward
        net.train()
        with make_autocast(amp_device_type, enabled=args.amp):
            _, out = net(node_feat=node_feat, edge_index=edge_index,
                         verts=verts, grid_hw=grid_hw, cell=cell,
                         img_hw=(H, W), cnn_prob_map=pm_t)
            aff_logits = out["logits"]  # [1,1,H,W]

            # Logit-level blend (CNN vs AFF)
            pm_clamped = pm_t.clamp(1e-6, 1 - 1e-6)
            cnn_logits = torch.log(pm_clamped) - torch.log1p(-pm_clamped)
            it_done = it + 1
            if it_done <= args.warmup_iters:
                alpha = args.aff_alpha_start; node_w = 0.0
            else:
                t_after = it_done - args.warmup_iters
                alpha  = args.aff_alpha_start + min(1.0, t_after / max(1, args.aff_alpha_ramp)) * (args.aff_alpha_end - args.aff_alpha_start)
                node_w = args.node_w_start   + min(1.0, t_after / max(1, args.node_w_ramp))   * (args.node_w_end   - args.node_w_start)
            logits = alpha * aff_logits + (1.0 - alpha) * cnn_logits

            # Losses
            w_bg, w_fg, mask = _class_weights_from_mask(lbl_t, fov_t)
            bce = F.binary_cross_entropy_with_logits(logits, lbl_t, reduction='none')
            weights = torch.where(
                lbl_t > 0.5,
                torch.tensor(w_fg, device=logits.device),
                torch.tensor(w_bg, device=logits.device)
            )
            pixel_loss = (bce * weights * mask).sum() / (mask.sum() + 1e-6)

            p_node = out["p_node"].clamp(1e-6, 1 - 1e-6)
            node_logits = torch.log(p_node) - torch.log1p(-p_node)
            node_bce = F.binary_cross_entropy_with_logits(node_logits, node_lbl)

            loss = pixel_loss + node_w * node_bce

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None: scheduler.step()
        ema.update(net)

        # ---- Logging + TRAIN QUAL ----
        if it_done % args.display == 0:
            with torch.no_grad(), make_autocast(amp_device_type, enabled=False):
                pred = torch.sigmoid(logits).clamp(0, 1).detach().cpu().numpy()[0, 0]
                lbl  = lbl_t.cpu().numpy()[0, 0]
                fovn = fov_t.cpu().numpy()[0, 0].astype(bool)

                # ---- Robust AUC/AP guard ----
                y = lbl[fovn].ravel()
                s = pred[fovn].ravel()
                reason = None
                if y.size == 0:
                    reason = "empty-fov"
                    auc = ap = np.nan
                elif not np.isfinite(s).all():
                    reason = "nan-in-scores"
                    auc = ap = np.nan
                else:
                    u = np.unique(y)
                    if u.size < 2:
                        reason = f"single-class-{int(u[0])}"
                        auc = ap = np.nan
                    else:
                        try:
                            auc, ap = util.get_auc_ap_score(y, s)
                        except Exception:
                            reason = "sklearn-error"
                            auc = ap = np.nan

                if reason:
                    # Lightweight diagnostic so you know why AUC/AP were skipped
                    print(f"[METRICS] skipped AUC/AP: {reason} | "
                          f"pos={int((y>0.5).sum())} neg={int((y<=0.5).sum())} "
                          f"nan_scores={int(np.isnan(s).sum())}")

                # thresholded metrics (protect from NaN by zero-filling for threshold only)
                pred_for_thresh = np.where(np.isfinite(pred), pred, 0.0)
                pb = (pred_for_thresh >= 0.5)
                lb = (lbl >= 0.5)
                tp = float(np.logical_and(pb & fovn, lb & fovn).sum())
                tn = float(np.logical_and(~pb & fovn, ~lb & fovn).sum())
                fp = float(np.logical_and(pb & fovn, ~lb & fovn).sum())
                fn = float(np.logical_and(~pb & fovn, lb & fovn).sum())
                denom = max(tp + tn + fp + fn, 1.0)
                acc  = (tp + tn) / denom
                se   = tp / max(tp + fn, 1.0)
                sp   = tn / max(tn + fp, 1.0)
                dice = (2 * tp) / max(2 * tp + fp + fn, 1.0)

                if np.isfinite(auc): auc_tracker.push(float(auc))
                auc_mu, auc_std = auc_tracker.mean_std()

            elapsed = time.time() - t0
            ips = it_done / max(elapsed, 1e-6)
            remain_iters = max(0, args.max_iters - it_done)
            eta_total_sec = remain_iters / max(ips, 1e-9)

            acc_p = acc * 100.0
            sp_p  = sp  * 100.0
            se_p  = se  * 100.0
            auc_p = (auc * 100.0) if np.isfinite(auc) else np.nan
            auc_mu_p  = auc_mu * 100.0 if np.isfinite(auc_mu) else np.nan
            auc_std_p = auc_std * 100.0 if np.isfinite(auc_std) else np.nan

            msg = (f"[{it_done:06d}] loss={loss.item():.4f} | pix={pixel_loss.item():.4f} "
                   f"node={node_bce.item():.4f} | α={alpha:.2f} node_w={node_w:.3f} | "
                   f"Acc={acc_p:5.2f}% Sp={sp_p:5.2f}% Se={se_p:5.2f}% "
                   f"AUC={auc_p:5.2f}% (μ±σ={auc_mu_p:5.2f}±{auc_std_p:5.2f}%) "
                   f"| dice={dice:.4f} ap={ap if np.isfinite(ap) else float('nan'):.4f} "
                   f"| {elapsed/60.0:.1f} min | TOTAL ETA: {_fmt_eta(eta_total_sec)}")
            print(msg)
            _safe_log_line(model_dir, msg)

            if args.save_train_probmaps:
                pred_prob = pred
                cnn_prob  = pm_t.detach().cpu().numpy()[0, 0]
                fov_np    = fov_t.detach().cpu().numpy()[0, 0]
                _save_probmaps(train_prob_dir, stem, it_done, pred_prob, cnn_prob, img, fov_np)

        # snapshots
        if it_done % args.snapshot_iters == 0:
            ckpt_path = os.path.join(model_dir, f"iter_{it_done}.pth")
            to_save = {"model": net.state_dict(), "iter": it_done, "args": vars(args), "opt": optimizer.state_dict()}
            if ema.decay > 0: to_save["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
            torch.save(to_save, ckpt_path)
            print(f"[CKPT] saved {ckpt_path}")

        # ---- TEST TICK (quick sanity; saves probmaps) ----
        if it_done % args.test_iters == 0:
            backup = ema.apply_to(net) if ema.decay > 0 else None
            try:
                img_list_e, blobs_e = dl_test.forward()
                if len(img_list_e) > 0:
                    img_e   = blobs_e['img'][0]
                    label_e = blobs_e['label'][0]
                    fov_e   = blobs_e['fov'][0]
                    graph_e = blobs_e['graph']
                    He, We, _ = img_e.shape

                    stem_e = Path(img_list_e[0]).stem
                    img_te = torch.from_numpy(img_e.transpose(2, 0, 1)).float().unsqueeze(0).to(device, non_blocking=True)
                    lbl_te = torch.from_numpy(label_e[..., 0]).float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                    fov_te = torch.from_numpy(fov_e[..., 0]).float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                    pm_te  = test_prob_cache[_canon_prob_stem(stem_e)].to(device, non_blocking=True)

                    lbl_te = _resize_4d(lbl_te, (He, We), mode='nearest')
                    fov_te = _resize_4d(fov_te, (He, We), mode='nearest')
                    pm_te  = _resize_4d(pm_te,  (He, We), mode='bilinear')

                    if device.type == 'cuda':
                        img_te = img_te.contiguous(memory_format=torch.channels_last)
                        lbl_te = lbl_te.contiguous(memory_format=torch.channels_last)
                        fov_te = fov_te.contiguous(memory_format=torch.channels_last)
                        pm_te  = pm_te.contiguous(memory_format=torch.channels_last)

                    verts_e_cpu, edge_index_e_cpu = get_graph_tensors_cached(graph_e, stem_e)
                    verts_e      = verts_e_cpu.to(device, non_blocking=True)
                    edge_index_e = edge_index_e_cpu.to(device, non_blocking=True)
                    node_feat_e  = _build_node_features(verts_e, pm_te, img_te)
                    cell_e   = int(2 * args.win_size)
                    grid_hw_e = (int(np.ceil(He / cell_e)), int(np.ceil(We / cell_e)))

                    net.eval()
                    with torch.no_grad():
                        _, out_e = net(node_feat=node_feat_e, edge_index=edge_index_e,
                                       verts=verts_e, grid_hw=grid_hw_e, cell=cell_e,
                                       img_hw=(He, We), cnn_prob_map=pm_te)
                        aff_logits_e = out_e["logits"]
                        pmc_e = pm_te.clamp(1e-6, 1 - 1e-6)
                        cnn_logits_e = torch.log(pmc_e) - torch.log1p(-pmc_e)
                        alpha_e = alpha
                        logits_e = alpha_e * aff_logits_e + (1 - alpha_e) * cnn_logits_e
                        pred_e = torch.sigmoid(logits_e).clamp(0, 1).cpu().numpy()[0, 0]

                    if args.save_test_probmaps:
                        _save_probmaps(
                            test_prob_dir, stem_e, it_done,
                            pred_e, pm_te.cpu().numpy()[0, 0],
                            img_e, fov_te.cpu().numpy()[0, 0]
                        )
            finally:
                if ema.decay > 0:
                    ema.restore(net, backup)

    # Final save
    final = os.path.join(model_dir, f"iter_{args.max_iters}.pth")
    to_save = {"model": net.state_dict(), "iter": args.max_iters, "args": vars(args), "opt": optimizer.state_dict()}
    if ema.decay > 0:
        to_save["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save(to_save, final)
    print(f"[CKPT] final saved {final}")

if __name__ == "__main__":
    main()
