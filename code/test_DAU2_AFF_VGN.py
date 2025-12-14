#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_DAU2_AFF_VGN.py  (calibrated, DR+G+H for HRF)

What this does:
  • Pure-PyTorch GATLayer (Modules/gat_only.py), same as training.
  • AFFModule + rasterize inlined.
  • Uses ALL HRF subsets: *_dr, *_g, *_h (no filtering to DR only).
  • Loads separate probmaps for DR/G/H (no mapping _g/_h -> _dr).
  • Robust size alignment: pm/label/fov/preds forcibly matched.
  • Labels normalized to {0,1} with vessel=1 (handles 0/1 or 0/255 sources).
  • Orientation auto-detect: if AUC<0.5 we flip logits (1-p equivalent).
  • Dataset-level calibration: threshold sweep, alpha sweep, optional temp scaling.
  • Per-image metrics, alpha_sweep.csv, thr_sweep.csv written to results_root.
"""

import os
import re
import csv
import argparse
from pathlib import Path
import warnings
from typing import Dict, List, Tuple

import numpy as np
import skimage.io
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
import util as util
try:
    from Modules.gat_only import GATLayer  # legacy name
except ImportError:
    from Modules.gat_only import GATLayerFast as GATLayer


# ----------------------------- AFF (same as training) -----------------------------
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
        if cnn_feat.dim() == 3:
            cnn_feat = cnn_feat.unsqueeze(1)
        if gnn_feat.dim() == 3:
            gnn_feat = gnn_feat.unsqueeze(1)
        H, W = cnn_feat.shape[-2:]
        gnn_feat = _align_spatial(gnn_feat, (H, W))
        c = self.cnn_align(cnn_feat)
        g = self.gnn_align(gnn_feat)
        mu = self.gate(torch.cat([c, g], dim=1))
        fused = mu * c + (1.0 - mu) * g
        f = self.refine(fused)
        logits = self.head(f)  # [B,1,H,W]
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


# ----------------------------- Graph + node features -----------------------------
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

def _build_node_features(verts, prob_map_t, img_t):
    device = prob_map_t.device
    H, W = prob_map_t.shape[-2:]
    vy = verts[:, 0].clamp(0, H - 1).long()
    vx = verts[:, 1].clamp(0, W - 1).long()
    y_norm = (vy.float() / max(1.0, (H - 1))).unsqueeze(1)
    x_norm = (vx.float() / max(1.0, (W - 1))).unsqueeze(1)
    p = prob_map_t[0, 0, vy, vx].unsqueeze(1)
    g = img_t[0, 1, vy, vx].unsqueeze(1)
    if g.max() > 1.0:
        g = g / 255.0
    node_feat = torch.cat([y_norm, x_norm, p, g, 1.0 - p], dim=1).to(device)
    return node_feat


# ----------------------------- Robust size helpers -----------------------------
def pad_or_crop_2d(arr: np.ndarray, target_hw, fill=0):
    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    if arr.ndim != 2:
        raise ValueError(f"pad_or_crop_2d expects 2D array, got shape {arr.shape}")
    H, W = arr.shape
    out = np.full((Ht, Wt), fill, dtype=arr.dtype)
    h = min(H, Ht); w = min(W, Wt)
    out[:h, :w] = arr[:h, :w]
    return out

def align_three(pred2d: np.ndarray, lbl2d: np.ndarray, fov2d: np.ndarray):
    Hc = min(pred2d.shape[0], lbl2d.shape[0], fov2d.shape[0])
    Wc = min(pred2d.shape[1], lbl2d.shape[1], fov2d.shape[1])
    return pred2d[:Hc, :Wc], lbl2d[:Hc, :Wc], fov2d[:Hc, :Wc]


# ----------------------------- Canonical stems + I/O helpers -----------------------------
def _canon_prob_stem(stem: str) -> str:
    """
    Canonicalize file stems for prob/graph lookup:

    - Strip *_label / *_mask / *_manual1 suffixes.
    - KEEP the DR/G/H suffix: 06_dr, 06_g, 06_h all stay distinct.
    """
    return re.sub(r'_(label|mask|manual1)$', '', stem, flags=re.IGNORECASE)

def _load_dau2_prob_592(probs_root: Path, stem: str) -> np.ndarray:
    """
    Load a DAU2 probmap for the *exact* stem (after stripping label/mask).
    For HRF this means:
      06_dr -> 06_dr_prob.*
      06_g  -> 06_g_prob.*
      06_h  -> 06_h_prob.*
    """
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
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.max() > 1.0:
                arr = np.clip(arr, 0, 255) / 255.0
            return arr
    raise FileNotFoundError(f"DAU2 prob not found for stem {s} under {probs_root}")

def build_prob_cache(probs_root: Path, img_names):
    """
    Pre-load probmaps once into CPU tensors:
      canonical_stem -> [1,1,H,W]
    Supports HRF DR/G/H separately, assuming <stem>_prob.* exists for each.
    """
    cache = {}
    for p in img_names:
        stem0 = Path(p).stem
        stem  = _canon_prob_stem(stem0)
        arr = _load_dau2_prob_592(probs_root, stem)
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        cache[stem] = t
    print(f"[PROB_CACHE] Loaded {len(cache)} probmaps from {probs_root}")
    return cache


# ----------------------------- Dataset split resolution (HRF: DR+G+H) -----------------------------
def _hrf_dataset_root() -> Path:
    txt_path = getattr(cfg.TRAIN, 'HRF_SET_TXT_PATH', 'C:/Users/rog/THESIS/DATASETS/HRF/training/images/train_768.txt')
    p = Path(txt_path)
    try:
        return p.parents[2]
    except IndexError:
        return Path('C:/Users/rog/THESIS/DATASETS/HRF')

def _list_hrf_split():
    """
    Build lists of HRF bases for *all* subsets: *_dr, *_g, *_h.

    Returns paths WITHOUT extension:
      <root>/<split>/images/<base>
    where base is like '06_dr', '06_g', '06_h'.
    """
    root = _hrf_dataset_root()
    def _collect(split: str):
        img_dir = root / split / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f"HRF images dir missing: {img_dir}")
        bases = []
        for img_path in sorted(img_dir.glob('*.*')):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                continue
            stem0 = img_path.stem
            base  = re.sub(r'_(label|mask|manual1)$', '', stem0, flags=re.IGNORECASE)
            # NO filtering to *_dr here: we keep *_dr, *_g, *_h
            bases.append(str((img_dir / base).with_suffix('')))
        uniq, seen = [], set()
        for b in bases:
            if b not in seen:
                uniq.append(b); seen.add(b)
        return uniq
    return _collect('training'), _collect('testing')

def _read_txt_lines(txt_path: str):
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Split list not found: {txt_path}")
    with open(path) as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def _resolve_split_lists(dataset: str):
    dataset = dataset.upper()
    if dataset == 'HRF':
        return _list_hrf_split()
    if dataset == 'DRIVE':
        train_txt, test_txt = cfg.TRAIN.DRIVE_SET_TXT_PATH, cfg.TEST.DRIVE_SET_TXT_PATH
    elif dataset == 'CHASE_DB1':
        train_txt, test_txt = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH, cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif dataset == 'STARE':
        train_txt = getattr(cfg.TRAIN, 'STARE_SET_TXT_PATH', '')
        test_txt = getattr(cfg.TEST, 'STARE_SET_TXT_PATH', '')
        if not train_txt or not test_txt:
            raise ValueError("STARE split txt paths are not configured in cfg.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return _read_txt_lines(train_txt), _read_txt_lines(test_txt)

def _default_dau_root(dataset: str) -> Path:
    dataset = dataset.upper()
    defaults = {
        'DRIVE': Path('C:/Users/rog/THESIS/DAU2_DRIVE'),
        'CHASE_DB1': Path('C:/Users/rog/THESIS/DAU2_CHASE/CHASE-DB1'),
        'HRF': Path('C:/Users/rog/THESIS/DAU_HRF/HRF'),
        'STARE': Path('C:/Users/rog/THESIS/DAU2_STARE/STARE'),
    }
    return defaults.get(dataset, Path('C:/Users/rog/THESIS/DAU2_DRIVE'))

def _resolve_probmap_roots(dataset: str, dau_root: Path,
                           train_override: str, test_override: str):
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
    train_root = _pick('train', train_override or '')
    test_root = _pick('test',  test_override or '')
    return train_root, test_root


# ----------------------------- Blob normalization wrapper -----------------------------
def _ensure_graph_blobs_dict(blobs):
    if isinstance(blobs, dict):
        return blobs
    if isinstance(blobs, tuple):
        if len(blobs) == 3:
            im_blob, label_blob, fov_blob = blobs
            return {"img": im_blob, "label": label_blob, "fov": fov_blob}
        if len(blobs) == 8:
            (im_blob, label_blob, fov_blob, probmap_blob, graph,
             num_nodes_list, vec_aug_on, rot_angle) = blobs
            return {
                "img": im_blob, "label": label_blob, "fov": fov_blob,
                "probmap": probmap_blob, "graph": graph,
                "num_of_nodes_list": num_nodes_list,
                "vec_aug_on": vec_aug_on, "rot_angle": rot_angle,
            }
    raise TypeError("Unexpected blob format")


# ----------------------------- Inference model (returns logits) -----------------------------
class AFF_VGN_Infer(nn.Module):
    def __init__(self, gnn_hidden=16, gnn_heads=4, aff_mid_ch=64):
        super().__init__()
        in_node_ch = 5
        self.gat1 = GATLayer(in_ch=in_node_ch, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat2 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat3 = GATLayer(in_ch=gnn_hidden * gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.node_pred = nn.Linear(gnn_hidden * gnn_heads, 1)
        self.aff = AFFModule(in_ch_cnn=1, in_ch_gnn=gnn_hidden * gnn_heads, mid_ch=aff_mid_ch, out_ch=1)

    @torch.no_grad()
    def forward_logits(self, img_t, pm_t, G, win_size=4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits_aff: [1,1,H,W], logits from AFF branch (after alignment)
          logits_cnn: [1,1,H,W], logits of CNN probmap (platt from pm_t)
        """
        H, W = pm_t.shape[-2:]
        verts, edge_index = _graph_to_verts_edges(G)
        verts = verts.to(pm_t.device)
        edge_index = edge_index.to(pm_t.device)

        node_feat = _build_node_features(verts, pm_t, img_t)
        h1 = self.gat1(node_feat, edge_index)
        h2 = self.gat2(h1, edge_index)
        h3 = self.gat3(h2, edge_index)
        _ = torch.sigmoid(self.node_pred(h3))

        cell = int(2 * win_size)
        grid_hw = (int(np.ceil(H / cell)), int(np.ceil(W / cell)))
        gnn_dense = rasterize_gnn_features(verts, grid_hw, cell, h3, (H, W))  # [1,C,H,W]

        prob_aff, aux = self.aff(pm_t, gnn_dense)
        logits_aff = aux["logits"]
        if logits_aff.shape[-2:] != (H, W):
            logits_aff = F.interpolate(logits_aff, size=(H, W), mode="bilinear", align_corners=False)

        pmc = pm_t.clamp(1e-6, 1.0 - 1e-6)
        logits_cnn = torch.log(pmc) - torch.log1p(-pmc)
        return logits_aff, logits_cnn


# ----------------------------- Metrics -----------------------------
def _denom(x: float) -> float:
    return x if x > 0 else 1.0

def compute_confusion_and_metrics(labels_bin: np.ndarray, preds_bin: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(np.logical_and(preds_bin, labels_bin)))
    tn = float(np.sum(np.logical_and(~preds_bin, ~labels_bin)))
    fp = float(np.sum(np.logical_and(preds_bin, ~labels_bin)))
    fn = float(np.sum(np.logical_and(~preds_bin, labels_bin)))

    acc  = (tp + tn) / _denom(tp + tn + fp + fn)
    precision = tp / _denom(tp + fp)
    recall    = tp / _denom(tp + fn)
    specificity = tn / _denom(tn + fp)
    dice = (2 * tp) / _denom(2 * tp + fp + fn)
    iou  = tp / _denom(tp + fp + fn)
    iou_bg = tn / _denom(tn + fp + fn)
    miou = 0.5 * (iou + iou_bg)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "precision": precision, "recall": recall,
        "specificity": specificity, "dice": dice, "iou": iou, "miou": miou
    }

def binary_cross_entropy_np(labels: np.ndarray, preds: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(preds, eps, 1.0 - eps)
    y = np.clip(labels, 0.0, 1.0)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


# ----------------------------- Calibration helpers -----------------------------
def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "_orig_mod.") -> Dict[str, torch.Tensor]:
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _auto_flip_logits(labels: np.ndarray, logits: np.ndarray, allow_flip: bool) -> Tuple[np.ndarray, bool, float, float]:
    """Return possibly flipped logits to ensure AUC >= 0.5, and (flipped?, auc_raw, auc_used)."""
    probs = _sigmoid_np(logits)
    auc_raw, ap_raw = util.get_auc_ap_score(labels, probs)
    if allow_flip and (auc_raw < 0.5):
        logits = -logits
        probs = _sigmoid_np(logits)
        auc_used, _ = util.get_auc_ap_score(labels, probs)
        return logits, True, auc_raw, auc_used
    return logits, False, auc_raw, auc_raw

def _parse_thr_sweep(arg: str) -> List[float]:
    """
    'start:end:step' or comma list '0.4,0.5,0.6'
    """
    if not arg:
        return []
    if ":" in arg:
        a, b, c = arg.split(":")
        a = float(a); b = float(b); c = float(c)
        vals = []
        x = a
        while (x <= b + 1e-9) and len(vals) < 10000:
            vals.append(round(x, 6))
            x += c
        return vals
    else:
        return [float(t) for t in arg.split(",") if t.strip()]

def _sweep_thresholds(labels: np.ndarray, logits: np.ndarray, thresholds: List[float]) -> List[Dict[str, float]]:
    out = []
    probs = _sigmoid_np(logits)
    for thr in thresholds:
        pb = probs >= thr
        lb = labels >= 0.5
        m = compute_confusion_and_metrics(lb, pb)
        bce = binary_cross_entropy_np(labels, probs)
        auc, ap = util.get_auc_ap_score(labels, probs)
        out.append({
            "thr": thr,
            "dice": m["dice"],
            "iou": m["iou"],
            "miou": m["miou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "acc": m["acc"],
            "specificity": m["specificity"],
            "bce": bce,
            "auc": auc,
            "ap": ap
        })
    return out

def _best_by(results: List[Dict[str, float]], key: str, maximize: bool = True) -> Dict[str, float]:
    if not results:
        return {}
    return sorted(results, key=lambda r: r.get(key, -np.inf), reverse=maximize)[0]

def _learn_temperature(labels: np.ndarray, logits: np.ndarray, grid=(0.5, 4.0, 200)) -> float:
    """Brute-force 1D search over temperature T to minimize BCE(labels, sigmoid(logits/T))."""
    lo, hi, n = grid
    Ts = np.linspace(lo, hi, int(n))
    best_T, best_bce = 1.0, float("inf")
    for T in Ts:
        probs = _sigmoid_np(logits / T)
        bce = binary_cross_entropy_np(labels, probs)
        if bce < best_bce:
            best_bce, best_T = bce, T
    return float(best_T)


# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate + calibrate AFF+GAT on DAU2 probmaps + SRNS graphs")
    p.add_argument('--dataset', default='DRIVE', choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--dau_root', type=str, default='C:/Users/rog/THESIS/DAU2_DRIVE')
    p.add_argument('--test_probs', type=str, default='C:/Users/rog/THESIS/DAU2_DRIVE/test')
    p.add_argument('--win_size', type=int, default=8)
    p.add_argument('--edge_dist_thresh', type=float, default=20)

    p.add_argument('--model_path', type=str, default='C:/Users/rog/THESIS/DAU2_DRIVE/VGN_AFF/train/best.pth')
    p.add_argument('--results_root', type=str, default='C:/Users/rog/THESIS/DAU2_DRIVE/VGN')
    p.add_argument('--run_name', type=str, default='AFF_VGN_DRIVE2')

    # Calibration controls
    p.add_argument('--alpha', type=float, default=0.3, help='1.0=AFF only; 0.0=CNN only; blended in logit space')
    p.add_argument('--alpha_sweep', type=str, default='', help='Comma list of alphas to sweep, e.g. "0.0,0.25,0.5,0.75,1.0"')
    p.add_argument('--thr_sweep', type=str, default='0.05:0.95:0.05', help='Threshold grid "start:end:step" or comma list')
    p.add_argument('--opt_target', type=str, default='dice',
                   choices=['dice','iou','miou','precision','recall','acc','specificity','ap','auc','bce'],
                   help='Metric to pick best threshold (max for all except bce)')
    p.add_argument('--learn_temp', action='store_true', help='Fit a temperature scalar T to minimize BCE (dataset-level).')

    p.add_argument('--no_ema', action='store_true', help='Ignore EMA weights even if present')
    p.add_argument('--threshold', type=float, default=0.5, help='Baseline binarization threshold for reporting')

    p.add_argument('--no_auto_flip', action='store_true', help='Disable auto flip (1-p) if dataset AUC<0.5')
    p.add_argument('--save_both', action='store_true', help='Also save <stem>_aff_prob_inv.png (1 - prob)')

    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ----------------------------- Main -----------------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
    args = parse_args()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dataset = args.dataset.upper()
    dau_root = Path(args.dau_root) if args.dau_root else _default_dau_root(dataset)
    dau_root = dau_root.resolve()

    # Resolve split lists and prob roots
    _, test_img_names = _resolve_split_lists(dataset)
    _, test_prob_root = _resolve_probmap_roots(dataset, dau_root, '', args.test_probs)
    test_prob_root = test_prob_root.resolve()

    # Graph dir wiring
    try:
        setattr(cfg.PATHS, "GRAPH_TEST_DIR", str(test_prob_root))
    except Exception:
        pass

    # Data layer
    dl_test = util.GraphDataLayerCached(
        test_img_names, False, 'srns_geo_dist_binary',
        win_size=args.win_size, edge_geo_dist_thresh=args.edge_dist_thresh
    )

    # Prob cache (CPU)
    prob_cache = build_prob_cache(test_prob_root, test_img_names)

    # Model + robust loading
    ckpt = torch.load(args.model_path, map_location='cpu')
    margs = ckpt.get("args", {})
    net = AFF_VGN_Infer(
        gnn_hidden=int(margs.get("gnn_hidden", 16)),
        gnn_heads=int(margs.get("gnn_heads", 4)),
        aff_mid_ch=int(margs.get("aff_mid_ch", 64))
    ).to(args.device)

    base_state = _strip_prefix(ckpt.get("model", {}))
    ema_state  = _strip_prefix(ckpt.get("ema", {})) if (not args.no_ema) and isinstance(ckpt.get("ema", None), dict) else {}
    use_ema = (len(ema_state) > 0)
    if use_ema:
        print("[INFO] Using EMA params over raw model (BN buffers kept).")
    overlay_state = dict(base_state)
    overlay_state.update(ema_state)
    missing, unexpected = net.load_state_dict(overlay_state, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {unexpected}")
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    net.eval()

    # Output dirs
    res_dir = os.path.join(args.results_root, args.run_name, dataset)
    os.makedirs(res_dir, exist_ok=True)

    # Collect dataset-wide labels and LOGITS (aff & cnn separately)
    labels_all = np.zeros((0,), dtype=np.float32)
    laff_all   = np.zeros((0,), dtype=np.float32)
    lcnn_all   = np.zeros((0,), dtype=np.float32)

    # For saving PNGs once at baseline alpha
    want = set(_canon_prob_stem(Path(p).stem) for p in test_img_names)
    seen = set()
    loops = 0
    max_loops = len(want) * 5

    per_items = []  # per-image raw labels and blended logits @ baseline alpha (for per-image CSV)

    while len(seen) < len(want) and loops < max_loops:
        loops += 1
        img_list, blobs = dl_test.forward()
        if not img_list:
            continue
        raw_stem = Path(img_list[0]).stem
        stem = _canon_prob_stem(raw_stem)
        if stem in seen:
            continue
        blobs = _ensure_graph_blobs_dict(blobs)

        img   = blobs['img'][0]
        label = blobs['label'][0]
        fov   = blobs['fov'][0]
        G     = blobs.get('graph', None)
        if G is None:
            raise RuntimeError("Graph object missing from test data layer.")

        H, W, _ = img.shape
        lbl_2d = label[..., 0].astype(np.float32)
        fov_2d = fov[..., 0].astype(bool)
        if lbl_2d.max() > 1.0:
            lbl_2d = (lbl_2d >= 128).astype(np.float32)
        else:
            lbl_2d = np.clip(lbl_2d, 0.0, 1.0)

        pm_np = prob_cache[stem].cpu().numpy()[0, 0]
        if pm_np.shape != (H, W):
            pm_np = pad_or_crop_2d(pm_np, (H, W), fill=0.0)

        # Torch tensors
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(args.device)
        pm_t  = torch.from_numpy(pm_np).float().unsqueeze(0).unsqueeze(0).to(args.device)

        # Inference -> logits
        with torch.no_grad():
            logits_aff, logits_cnn = net.forward_logits(img_t, pm_t, G, win_size=args.win_size)
            logits_aff = logits_aff.squeeze().cpu().numpy().astype(np.float32)
            logits_cnn = logits_cnn.squeeze().cpu().numpy().astype(np.float32)

        # Align shapes against label/FOV
        if (logits_aff.shape != lbl_2d.shape) or (logits_aff.shape != fov_2d.shape):
            # We align on logits_aff; logits_cnn follows the same crop
            logits_aff, lbl_2d, fov_2d = align_three(logits_aff, lbl_2d, fov_2d)
            logits_cnn = logits_cnn[:logits_aff.shape[0], :logits_aff.shape[1]]

        # Baseline alpha prob (for saving)
        logits_blend = float(args.alpha) * logits_aff + (1.0 - float(args.alpha)) * logits_cnn
        prob_np = _sigmoid_np(logits_blend).astype(np.float32)

        # Save outputs (prob is independent of threshold)
        np.save(os.path.join(res_dir, stem + '_aff_prob.npy'), prob_np)
        try:
            skimage.io.imsave(os.path.join(res_dir, stem + '_aff_prob.png'), (prob_np * 255.0).astype(np.uint8))
            if args.save_both:
                inv = (1.0 - prob_np)
                skimage.io.imsave(os.path.join(res_dir, stem + '_aff_prob_inv.png'), (inv * 255.0).astype(np.uint8))
        except Exception:
            pass

        # Append FOV-masked vectors
        m = fov_2d.astype(bool)
        y = lbl_2d[m].reshape(-1).astype(np.float32)
        la = logits_aff[m].reshape(-1).astype(np.float32)
        lc = logits_cnn[m].reshape(-1).astype(np.float32)

        labels_all = np.concatenate([labels_all, y])
        laff_all   = np.concatenate([laff_all, la])
        lcnn_all   = np.concatenate([lcnn_all, lc])

        per_items.append({"stem": stem, "y": y, "logits_blend": logits_blend[m].reshape(-1).astype(np.float32)})
        seen.add(stem)

    if len(seen) < len(want):
        missing = sorted(list(want - seen))
        raise RuntimeError(f"Did not iterate all test images. Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # ---------- Calibration & Reporting ----------
    # Alpha list
    alphas = [args.alpha]
    if args.alpha_sweep.strip():
        alphas = sorted(set([float(x) for x in args.alpha_sweep.split(",") if x.strip() != ""]))

    # Threshold grid
    thr_grid = _parse_thr_sweep(args.thr_sweep)
    if not thr_grid:
        thr_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Alpha sweep results (dataset-level)
    alpha_rows = []
    best_alpha_pack = None

    for alpha in alphas:
        logits = alpha * laff_all + (1.0 - alpha) * lcnn_all

        # Auto flip by AUC if requested
        logits_adj, flipped, auc_raw, auc_used = _auto_flip_logits(labels_all, logits, allow_flip=(not args.no_auto_flip))

        # Optional temperature scaling (learn T on-the-fly for this alpha)
        T = 1.0
        if args.learn_temp:
            T = _learn_temperature(labels_all, logits_adj, grid=(0.5, 4.0, 200))
        logits_scaled = logits_adj / T

        # Baseline threshold metrics (user-provided threshold)
        probs_base = _sigmoid_np(logits_scaled)
        bce_base = binary_cross_entropy_np(labels_all, probs_base)
        auc_base, ap_base = util.get_auc_ap_score(labels_all, probs_base)

        # Threshold sweep -> pick best by target
        sweep = _sweep_thresholds(labels_all, logits_scaled, thr_grid)
        maximize = (args.opt_target != 'bce')
        best = _best_by(sweep, args.opt_target, maximize=maximize)

        # Store alpha row
        row = {
            "alpha": alpha,
            "flipped": int(flipped),
            "temp": T,
            "auc_raw": auc_raw,
            "auc_used": auc_used,
            "auc_base": auc_base,
            "ap_base": ap_base,
            "bce_base": bce_base,
            "best_thr": best.get("thr", float('nan')),
            "best_target": args.opt_target,
            "best_dice": best.get("dice", float('nan')),
            "best_iou": best.get("iou", float('nan')),
            "best_precision": best.get("precision", float('nan')),
            "best_recall": best.get("recall", float('nan')),
            "best_ap": best.get("ap", float('nan')),
            "best_auc": best.get("auc", float('nan')),
            "best_bce": best.get("bce", float('nan')),
        }
        alpha_rows.append(row)

        # Keep pack with highest dice by default
        if (best_alpha_pack is None) or (best.get("dice", -1.0) > best_alpha_pack["best"]["dice"]):
            best_alpha_pack = {
                "alpha": alpha,
                "flipped": flipped,
                "T": T,
                "best": best,
                "auc_raw": auc_raw,
                "auc_used": auc_used
            }

        # Print per-alpha summary
        print(f"[ALPHA {alpha:.2f}] flip={flipped} T={T:.3f} | "
              f"base(thr={args.threshold:.2f}) AUC={auc_base:.4f} AP={ap_base:.4f} BCE={bce_base:.6f} | "
              f"best@{args.opt_target}: thr={best.get('thr', np.nan):.2f} "
              f"Dice={best.get('dice', np.nan):.4f} Prec={best.get('precision', np.nan):.4f} Rec={best.get('recall', np.nan):.4f}")

    # Write alpha_sweep.csv
    alpha_csv = os.path.join(res_dir, 'alpha_sweep.csv')
    with open(alpha_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["alpha","flipped","temp","auc_raw","auc_used","auc_base","ap_base","bce_base",
                    "best_thr","best_target","best_dice","best_iou","best_precision","best_recall","best_ap","best_auc","best_bce"])
        for r in alpha_rows:
            w.writerow([r[k] for k in ["alpha","flipped","temp","auc_raw","auc_used","auc_base","ap_base","bce_base",
                                       "best_thr","best_target","best_dice","best_iou","best_precision","best_recall","best_ap","best_auc","best_bce"]])

    # Also write thr_sweep.csv for the *best alpha* (for inspection)
    if best_alpha_pack is not None:
        alpha_best = best_alpha_pack["alpha"]
        logits_best = alpha_best * laff_all + (1.0 - alpha_best) * lcnn_all
        logits_best, flipped_best, auc_raw_best, auc_used_best = _auto_flip_logits(labels_all, logits_best, allow_flip=(not args.no_auto_flip))
        T_best = best_alpha_pack["T"]
        logits_best = logits_best / T_best
        sweep_best = _sweep_thresholds(labels_all, logits_best, thr_grid)
        thr_csv = os.path.join(res_dir, 'thr_sweep.csv')
        with open(thr_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["thr","dice","iou","miou","precision","recall","acc","specificity","bce","auc","ap"])
            for r in sweep_best:
                w.writerow([r[k] for k in ["thr","dice","iou","miou","precision","recall","acc","specificity","bce","auc","ap"]])

    # ----- Final dataset-level report at baseline alpha + threshold -----
    alpha0 = float(args.alpha)
    logits0 = alpha0 * laff_all + (1.0 - alpha0) * lcnn_all
    logits0, flipped0, auc_raw0, auc_used0 = _auto_flip_logits(labels_all, logits0, allow_flip=(not args.no_auto_flip))
    T0 = _learn_temperature(labels_all, logits0, grid=(0.5, 4.0, 200)) if args.learn_temp else 1.0
    logits0 = logits0 / T0
    probs0 = _sigmoid_np(logits0)
    bce0 = binary_cross_entropy_np(labels_all, probs0)
    auc0, ap0 = util.get_auc_ap_score(labels_all, probs0)

    thr = float(args.threshold)
    preds_b = (probs0 >= thr)
    labels_b = (labels_all >= 0.5)
    m0 = compute_confusion_and_metrics(labels_b, preds_b)
    pos_ratio = float(np.mean(labels_b))
    mean_pred = float(np.mean(probs0))

    print('----- TEST (FOV ROI) — baseline alpha/threshold -----')
    if flipped0:
        print('[NOTE] Raw AUC was < 0.5 (%.4f). Metrics below are with logits flipped (1-p equivalently).' % auc_raw0)
    print('Dataset:         %s' % dataset)
    print('Pos-ratio:       %.2f%%' % (100.0 * pos_ratio))
    print('Mean pred:       %.4f' % mean_pred)
    print('BCE:             %.6f' % bce0)
    print('Accuracy:        %.4f' % m0["acc"])
    print('Specificity:     %.4f' % m0["specificity"])
    print('Sensitivity:     %.4f' % m0["recall"])
    print('Precision:       %.4f' % m0["precision"])
    print('F1 / Dice:       %.4f' % m0["dice"])
    print('IoU:             %.4f' % m0["iou"])
    print('mIoU:            %.4f' % m0["miou"])
    print('AUC:             %.4f' % auc0)
    print('AP:              %.4f' % ap0)
    print('BCE:             %.6f' % bce0)

    # Per-image metrics CSV (using baseline alpha / temp / flip, threshold=--threshold)
    csv_path = os.path.join(res_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            'image', 'N', 'PosRatio', 'MeanPred', 'BCE',
            'Accuracy', 'Specificity', 'Sensitivity', 'Precision',
            'Dice', 'IoU', 'mIoU', 'AUC', 'AP', 'TP', 'TN', 'FP', 'FN'
        ])
        for item in sorted(per_items, key=lambda x: x['stem']):
            y = item['y']
            logits_blend_vec = item['logits_blend']
            # Apply the same flip + temperature used above
            logits_eval = (-logits_blend_vec if flipped0 else logits_blend_vec) / T0
            p_eval = _sigmoid_np(logits_eval)

            N = int(y.size)
            posr = float(np.mean(y >= 0.5))
            meanp = float(np.mean(p_eval))
            bce_i = binary_cross_entropy_np(y, p_eval)
            try:
                auc_i, ap_i = util.get_auc_ap_score(y, p_eval)
            except Exception:
                auc_i, ap_i = (np.nan, np.nan)

            pb = p_eval >= thr
            lb = y >= 0.5
            m_i = compute_confusion_and_metrics(lb, pb)

            w.writerow([
                item['stem'], N, posr, meanp, bce_i,
                m_i["acc"], m_i["specificity"], m_i["recall"], m_i["precision"],
                m_i["dice"], m_i["iou"], m_i["miou"],
                auc_i, ap_i, m_i["tp"], m_i["tn"], m_i["fp"], m_i["fn"]
            ])

    # Log summary + best config from sweeps
    best_line = ""
    if best_alpha_pack is not None:
        b = best_alpha_pack["best"]
        best_line = (f"BEST alpha={best_alpha_pack['alpha']:.2f} flip={best_alpha_pack['flipped']} "
                     f"T={best_alpha_pack['T']:.3f} by {args.opt_target}: thr={b['thr']:.3f} "
                     f"Dice={b['dice']:.4f} Prec={b['precision']:.4f} Rec={b['recall']:.4f}")

    log_path = os.path.join(res_dir, 'log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"alpha (baseline): {args.alpha}\n")
        f.write(f"alpha_sweep: {args.alpha_sweep}\n")
        f.write(f"thr_sweep: {args.thr_sweep}\n")
        f.write(f"opt_target: {args.opt_target}\n")
        f.write(f"learn_temp: {args.learn_temp}\n")
        f.write(f"use_ema: {use_ema}\n")
        f.write(f"edge_dist_thresh: {args.edge_dist_thresh}\n")
        f.write(f"auto_flip: {not args.no_auto_flip}\n")
        f.write(f"baseline_flip: {flipped0}\n")
        f.write(f"baseline_T: {T0}\n")
        f.write(f"baseline_threshold: {args.threshold}\n")
        f.write(f"alpha_sweep_csv: {alpha_csv}\n")
        if best_line:
            f.write(best_line + "\n")
        f.write('Per-image CSV: ' + csv_path + '\n')

    print(best_line if best_line else "No sweep performed.")
    print(f"Per-image metrics written to: {csv_path}")
    print("Testing + calibration complete.")


if __name__ == "__main__":
    main()
