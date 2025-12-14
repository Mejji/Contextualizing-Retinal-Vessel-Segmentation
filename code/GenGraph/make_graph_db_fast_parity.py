#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_graph_db_fast_parity.py
--------------------------------
Shin-parity, faster graph generator for VGN-style datasets (DRIVE/STARE/CHASE_DB1/HRF).

This version adds robustness for softer probability maps (e.g., DAU2Net) and
a contrast-safe visualizer so graphs are actually visible on HRF masks.

What's new (vs. the parity baseline):
- Speed normalization pipeline (percentile clip + gamma + epsilon floor).
- Auto-scaling of geodesic threshold based on observed local vesselness.
- Tunable source gating (3×3 mean threshold).
- KD radius multiplier and optional KNN back-fill for sparse cases.
- High-contrast, fast LineCollection-based visualization (viz_mode=simple).
- Debug prints for effective thresholds, valid-node counts, edge counts.

How it's still fast **without** changing topology (when auto-scaling is off):
- We compute geodesic travel time (skfmm.travel_time) **only inside a local window**
  centered on each source node, with radius R = ceil(edge_dist_thresh) + margin.
- KD-tree pre-filtering in Euclidean space to avoid O(N^2) scans.

How it preserves Shin’s construction (when parity switches are at defaults):
- SRVS-style nodes (one per grid patch; center fallback when the patch sum is 0).
- Source gating: speed[y,x] > 0 and 3×3 mean >= min_local_mean (0.1 in Shin; default 0.05 here).
- Geodesic edges: skfmm.travel_time with `narrow=edge_dist_thresh_eff`.
- Euclidean fallback only if you explicitly set --force_knn > 0.

Recommended HRF settings if your maps are "soft":
  --dataset HRF --win_size 10 --edge_dist_thresh 80 --auto_scale_thresh
  --target_local_mean 0.20 --min_local_mean 0.05 --speed_norm percentile
  --p_high 99.5 --speed_eps 1e-4 --viz_mode simple
"""
import argparse
import os
from pathlib import Path
import pickle as pkl
import multiprocessing as mp
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import skimage.io
from skimage.transform import resize
import networkx as nx
import skfmm
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # for viz
from matplotlib.collections import LineCollection

# Project-local utilities
import _init_paths  # noqa: F401
from bwmorph import bwmorph
from config import cfg  # noqa: F401  (kept for project parity, not used here)
import util as util

DEBUG = False


# -------------------------
# Parameters / CLI
# -------------------------
@dataclass
class Params:
    dataset: str
    use_multiprocessing: bool
    processes: int
    source_type: str
    win_size: int
    edge_method: str
    edge_dist_thresh: float
    geo_window_margin: int = 1
    only_names: Tuple[str, ...] = tuple()

    # ---- New robustness knobs ----
    # Source gating
    min_local_mean: float = 0.05   # Shin used ~0.1; softer maps benefit from 0.03–0.08
    speed_eps: float = 1e-4        # floor for speed to avoid zero-speed masks in FMM

    # Speed normalization
    speed_norm: str = "none"       # none | percentile | zscore
    p_low: float = 0.0
    p_high: float = 100.0
    speed_gamma: float = 1.0       # gamma correction; <1 boosts low probs, >1 compresses
    clip_after_gamma: bool = True

    # Auto threshold scaling wrt observed vesselness (softer maps => higher effective d)
    auto_scale_thresh: bool = True
    target_local_mean: float = 0.20
    auto_thresh_min_scale: float = 0.5
    auto_thresh_max_scale: float = 6.0

    # KD-tree radius multiplier
    radius_factor: float = 1.0

    # Force a few edges even if geodesic is too strict (Euclidean KNN fallback)
    force_knn: int = 0  # 0 disables
    force_knn_radius: float = 0.0  # 0 => no radius cap; else Euclidean cap in px

    # Visualization
    viz_mode: str = "simple"       # simple | util
    viz_downsample_max: int = 1800 # longest side for saved PNG (to keep files viewable)
    edge_alpha: float = 1.0
    edge_lw: float = 0.5
    edge_color: str = "w"          # white edges on gray mask
    node_alpha: float = 0.9
    node_size_px: float = 0.0      # 0 to skip nodes; use small (e.g., 0.6) for sparse views
    max_edges_draw: int = 250_000  # cap drawn edges to keep plot fast

    # Debug
    print_stats: bool = True


def parse_args():
    p = argparse.ArgumentParser(description='Make a graph db (Shin-parity, fast windowed-FMM, robust viz)')
    p.add_argument('--dataset', default='DRIVE', type=str, help='DRIVE | STARE | CHASE_DB1 | HRF')
    p.add_argument('--use_multiprocessing', dest='use_multiprocessing',
                   action='store_true', default=True, help='Enable multiprocessing (default: on)')
    p.add_argument('--no_multiprocessing', dest='use_multiprocessing', action='store_false',
                   help='Disable multiprocessing')
    p.add_argument('--processes', type=int, default=8, help='Number of workers when MP is enabled')
    p.add_argument('--source_type', default='result', type=str, choices=['result', 'gt'],
                   help='Use CNN result prob map ("result") or ground-truth mask ("gt") as speed')
    p.add_argument('--win_size', default=None, type=int,
                   help='SRVS grid window size (defaults: DRIVE/STARE/CHASE=4, HRF=10)')
    p.add_argument('--edge_method', default='geo_dist', type=str, choices=['geo_dist', 'eu_dist'],
                   help='Edge construction method')
    p.add_argument('--edge_dist_thresh', default=None, type=float,
                   help='Distance threshold for edge construction (defaults: DRIVE/STARE=10, CHASE=40, HRF=80)')
    p.add_argument('--geo_window_margin', default=1, type=int,
                   help='Extra pixels added to the crop around radius R for FMM (default: 1)')
    p.add_argument('--only_names', default='', type=str,
                   help='Comma-separated basenames (without extension) to process only these files')

    # New robustness/viz controls
    p.add_argument('--min_local_mean', default=0.05, type=float, help='3x3 mean threshold for source gating')
    p.add_argument('--speed_eps', default=1e-4, type=float, help='Minimum speed passed into FMM')
    p.add_argument('--speed_norm', default='none', choices=['none', 'percentile', 'zscore'],
                   help='Speed normalization mode')
    p.add_argument('--p_low', default=0.0, type=float, help='Low percentile for percentile clipping')
    p.add_argument('--p_high', default=100.0, type=float, help='High percentile for percentile clipping')
    p.add_argument('--speed_gamma', default=1.0, type=float, help='Gamma correction for speed')
    p.add_argument('--no_clip_after_gamma', dest='clip_after_gamma', action='store_false',
                   help='Disable post-gamma clipping to [0,1]')
    p.add_argument('--auto_scale_thresh', dest='auto_scale_thresh', action='store_true', default=True,
                   help='Auto-scale geodesic threshold based on observed local means')
    p.add_argument('--no_auto_scale_thresh', dest='auto_scale_thresh', action='store_false',
                   help='Disable auto scaling')
    p.add_argument('--target_local_mean', default=0.20, type=float, help='Target 3x3 mean for auto-threshold scaling')
    p.add_argument('--auto_thresh_min_scale', default=0.5, type=float, help='Lower clamp for auto scaling factor')
    p.add_argument('--auto_thresh_max_scale', default=6.0, type=float, help='Upper clamp for auto scaling factor')
    p.add_argument('--radius_factor', default=1.0, type=float, help='Euclidean KD radius multiplier')
    p.add_argument('--force_knn', default=0, type=int, help='Back-fill KNN edges per source if geodesic adds none')
    p.add_argument('--force_knn_radius', default=0.0, type=float, help='Optional Euclidean radius cap for KNN back-fill')
    p.add_argument('--viz_mode', default='simple', choices=['simple', 'util'],
                   help='Visualization backend (simple=high-contrast, util=project util)')
    p.add_argument('--viz_downsample_max', default=1800, type=int,
                   help='Downsample longest side to this many px in saved PNGs (simple viz)')
    p.add_argument('--edge_alpha', default=1.0, type=float, help='Edge alpha for simple viz')
    p.add_argument('--edge_lw', default=0.5, type=float, help='Edge linewidth for simple viz (px)')
    p.add_argument('--edge_color', default='w', type=str, help='Edge color for simple viz (matplotlib color)')
    p.add_argument('--node_alpha', default=0.9, type=float, help='Node alpha for simple viz')
    p.add_argument('--node_size_px', default=0.0, type=float, help='Node radius in pixels for simple viz (0 disables)')
    p.add_argument('--max_edges_draw', default=250000, type=int, help='Max edges drawn in simple viz')
    p.add_argument('--no_print_stats', dest='print_stats', action='store_false', help='Silence debug stats')
    return p.parse_args()


# -------------------------
# I/O utilities
# -------------------------
def _resolve_image_path(path_no_ext: str):
    """Return existing file path, trying common extensions."""
    cand = Path(path_no_ext)
    if cand.suffix and cand.exists():
        return str(cand)
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    for ext in exts:
        chk = cand.with_suffix(ext)
        if chk.exists():
            return str(chk)
    matches = list(cand.parent.glob(cand.name + '.*'))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Image not found for base path: {path_no_ext}")


def _dataset_layout(dataset: str):
    """Return (im_ext, label_ext, H, W) based on dataset name."""
    dataset = dataset.upper()
    if dataset == 'DRIVE':
        return ('_image.tif', '_label.gif', 592, 592)
    elif dataset == 'CHASE_DB1':
        # Clarify actual orientation: height=960, width=999
        return ('.jpg', '_1stHO.png', 960, 999)
    elif dataset == 'HRF':
        # Native HRF images / prob-maps are 3504 x 2336 (height x width)
        return ('.jpg', '.tif', 2336, 3504)
    else:
        # fall back to DRIVE-like
        return ('_image.tif', '_label.gif', 592, 592)


def _infer_hrf_split(name: str) -> str:
    """Return 'training' or 'testing' split for HRF given a file stem like '06_dr'."""
    digits = []
    for ch in name:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        raise ValueError(f"Cannot infer HRF split from name '{name}' (no leading digits).")
    idx = int(''.join(digits))
    return 'training' if idx <= 5 else 'testing'


# -------------------------
# Image / speed preparation
# -------------------------
def _read_and_fix_sizes(cur_im_path, cur_gt_mask_path, cur_res_prob_path, H, W, source_type):
    im = skimage.io.imread(cur_im_path)
    # gt
    if os.path.exists(cur_gt_mask_path):
        gt_mask = skimage.io.imread(cur_gt_mask_path).astype(float) / 255.0
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]
        gt_mask = gt_mask >= 0.5
    else:
        # keep shape compatible
        gt_mask = np.zeros(im.shape[:2], dtype=bool)

    # speed / vesselness
    if source_type == 'gt':
        vesselness = gt_mask.astype(float)
    else:
        vesselness = skimage.io.imread(cur_res_prob_path).astype(float) / 255.0
        if vesselness.ndim == 3:
            vesselness = vesselness[..., 0]

    # pad/crop to canonical
    def _fix(arr, tgt_shape):
        out = np.zeros(tgt_shape, dtype=arr.dtype)
        slices = tuple(slice(0, min(arr.shape[i], tgt_shape[i])) for i in range(arr.ndim))
        out[slices] = arr[slices]
        return out

    im_fix = _fix(im, (H, W, 3) if im.ndim == 3 else (H, W))
    gt_fix = _fix(gt_mask, (H, W))
    ves_fix = _fix(vesselness, (H, W))
    if im_fix.ndim == 2:
        im_fix = np.repeat(im_fix[..., None], 3, axis=2)
    return im_fix, gt_fix, ves_fix


def _normalize_speed(speed: np.ndarray, params: Params) -> np.ndarray:
    s = speed.astype(np.float32).copy()

    if params.speed_norm == "percentile":
        lo = np.percentile(s, params.p_low)
        hi = np.percentile(s, params.p_high)
        if hi <= lo:
            hi = lo + 1e-6
        s = (s - lo) / (hi - lo)
        s = np.clip(s, 0.0, 1.0)
    elif params.speed_norm == "zscore":
        m = float(np.mean(s))
        sd = float(np.std(s)) + 1e-6
        s = (s - m) / sd
        # squash to [0,1] via sigmoid-ish map
        s = 1.0 / (1.0 + np.exp(-s))
    # gamma
    if params.speed_gamma != 1.0:
        s = np.power(np.clip(s, 0.0, 1.0), params.speed_gamma)
    if params.clip_after_gamma:
        s = np.clip(s, 0.0, 1.0)
    # epsilon floor
    s = np.maximum(s, float(params.speed_eps))
    return s


# -------------------------
# SRVS node sampling
# -------------------------
def _srvs_nodes(vesselness: np.ndarray, win_size: int) -> List[Tuple[int, int]]:
    """Semi-regular vertex sampling (SRVS):
    - One node per grid patch
    - If patch sum == 0 -> use patch center
    - Else -> argmax within the patch
    """
    H, W = vesselness.shape
    y_quan = sorted(set(range(0, H, win_size)) | {H})
    x_quan = sorted(set(range(0, W, win_size)) | {W})
    max_pos = []
    for y0, y1 in zip(y_quan[:-1], y_quan[1:]):
        for x0, x1 in zip(x_quan[:-1], x_quan[1:]):
            patch = vesselness[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if np.sum(patch) == 0:
                max_pos.append((y0 + patch.shape[0] // 2, x0 + patch.shape[1] // 2))
            else:
                rr, cc = np.unravel_index(np.argmax(patch), patch.shape)
                max_pos.append((y0 + int(rr), x0 + int(cc)))
    return max_pos


def _windowed_travel_time(speed: np.ndarray, sy: int, sx: int, thresh: float, margin: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Compute skfmm.travel_time in a local window around (sy, sx).
    Return (tt_win, (y0,y1,x0,x1)) such that tt in full image coords would be mapped by [cy-y0, cx-x0].
    """
    H, W = speed.shape
    R = int(np.ceil(thresh)) + int(max(0, margin))
    y0 = max(0, sy - R)
    y1 = min(H, sy + R + 1)
    x0 = max(0, sx - R)
    x1 = min(W, sx + R + 1)
    speed_win = speed[y0:y1, x0:x1]
    phi = np.ones_like(speed_win, dtype=float)
    phi[sy - y0, sx - x0] = -1.0
    tt = skfmm.travel_time(phi, speed_win, narrow=thresh)
    if isinstance(tt, np.ma.MaskedArray):
        tt = np.ma.filled(tt, np.inf)
    return tt, (y0, y1, x0, x1)


# -------------------------
# Edge builders
# -------------------------
def _build_edges_geo(max_pos: List[Tuple[int, int]],
                     speed: np.ndarray,
                     base_thresh: float,
                     margin: int,
                     params: Params) -> Tuple[List[Tuple[int, int, float]], float, int]:
    """Construct geodesic edges using windowed-FMM + KD-tree candidate pruning.
    Returns list of (i, j, weight), i<j, along with effective threshold used and number of valid sources.
    """
    H, W = speed.shape
    nodes = np.asarray(max_pos, dtype=int)
    ys = nodes[:, 0].astype(np.float32)
    xs = nodes[:, 1].astype(np.float32)
    n = len(nodes)

    # Source gating (tolerant)
    valid = []
    local_means = []
    for i in range(n):
        yy, xx = int(ys[i]), int(xs[i])
        if speed[yy, xx] <= params.speed_eps:
            continue
        y0, y1 = max(0, yy - 1), min(H, yy + 2)
        x0, x1 = max(0, xx - 1), min(W, xx + 2)
        m = float(speed[y0:y1, x0:x1].mean())
        if m < params.min_local_mean:
            continue
        valid.append(i)
        local_means.append(m)

    if len(valid) == 0:
        return [], base_thresh, 0

    # Auto-scale threshold to compensate for softer/harder speeds:
    # travel_time ~ path_length / speed. If median local speed is s_med,
    # then time inflates by ~1/s_med. So scale threshold ~ 1/s_med relative to target.
    eff_thresh = float(base_thresh)
    if params.auto_scale_thresh and len(local_means) > 0:
        s_med = float(np.median(local_means))
        if s_med <= 1e-6:
            scale = params.auto_thresh_max_scale
        else:
            scale = params.target_local_mean / s_med
            scale = float(np.clip(scale, params.auto_thresh_min_scale, params.auto_thresh_max_scale))
        eff_thresh = float(base_thresh) * scale

    # KD-tree for Euclidean candidate pruning within radius eff_thresh * radius_factor
    coords = np.stack([ys, xs], axis=1)
    tree = cKDTree(coords, leafsize=32)
    radius = float(eff_thresh) * float(max(1.0, params.radius_factor)) + 1e-6

    edges = []
    edges_added_per_i = 0
    for k, i in enumerate(valid, start=1):
        sy, sx = int(ys[i]), int(xs[i])
        cand = tree.query_ball_point([ys[i], xs[i]], r=radius)
        cand = [j for j in cand if j > i]
        if cand:
            tt_win, (y0, y1, x0, x1) = _windowed_travel_time(speed, sy, sx, eff_thresh, margin)
            added = 0
            for j in cand:
                cy, cx = int(ys[j]), int(xs[j])
                if cy < y0 or cy >= y1 or cx < x0 or cx >= x1:
                    continue
                geo = float(tt_win[cy - y0, cx - x0])
                if not np.isfinite(geo):
                    continue
                if geo < eff_thresh:
                    w = float(eff_thresh / (eff_thresh + geo))
                    edges.append((i, j, w))
                    added += 1
            if added == 0 and params.force_knn > 0:
                # Backfill KNN if geodesic added none
                _backfill_knn(i, coords, tree, params, edges, i_only=True)
        else:
            if params.force_knn > 0:
                _backfill_knn(i, coords, tree, params, edges, i_only=True)

        if DEBUG and (k % 500 == 0 or k == len(valid)):
            print(f'[DEBUG] geo edges: processed {k}/{len(valid)} sources; edges so far = {len(edges)}')

    return edges, eff_thresh, len(valid)


def _backfill_knn(i: int,
                  coords: np.ndarray,
                  tree: cKDTree,
                  params: Params,
                  edges: List[Tuple[int, int, float]],
                  i_only: bool = True):
    k = params.force_knn + 1  # include self
    dists, idxs = tree.query(coords[i], k=k)
    if np.isscalar(idxs):
        idxs = [int(idxs)]
        dists = [float(dists)]
    for j, d in zip(idxs, dists):
        if j == i:
            continue
        if j < 0:
            continue
        if params.force_knn_radius > 0.0 and float(d) > params.force_knn_radius:
            continue
        if j > i:
            edges.append((i, int(j), 1.0))
        elif not i_only and j < i:
            edges.append((int(j), i, 1.0))


def _build_edges_eu(max_pos: List[Tuple[int, int]], thresh: float, params: Params) -> List[Tuple[int, int, float]]:
    nodes = np.asarray(max_pos, dtype=int)
    ys = nodes[:, 0].astype(np.float32)
    xs = nodes[:, 1].astype(np.float32)
    coords = np.stack([ys, xs], axis=1)
    tree = cKDTree(coords, leafsize=32)
    radius = float(thresh) * float(max(1.0, params.radius_factor)) + 1e-6

    edges = []
    for i in range(len(nodes)):
        neigh = tree.query_ball_point([ys[i], xs[i]], r=radius)
        for j in neigh:
            if j > i:
                edges.append((i, j, 1.0))
        if params.force_knn > 0 and len([e for e in edges if e[0] == i or e[1] == i]) == 0:
            _backfill_knn(i, coords, tree, params, edges, i_only=True)
    return edges


# -------------------------
# Visualization
# -------------------------
def _simple_visualize(base: np.ndarray,
                      g: nx.Graph,
                      save_path: str,
                      params: Params):
    """
    High-contrast viz for large graphs using LineCollection.
    - base: image or mask; if boolean/0-1, we convert to gray background.
    - Downsamples to keep PNG <~ few thousand px on the long side.
    """
    if base.ndim == 2:
        base_vis = (base.astype(float))
        # map boolean mask to gray (background dark, vessels mid)
        if base_vis.max() <= 1.0:
            base_vis = base_vis * 0.7  # vessel-ish mid-gray on black
        base_rgb = np.dstack([base_vis, base_vis, base_vis])
    else:
        base_rgb = base.astype(float)
        if base_rgb.max() > 1.0:
            base_rgb /= 255.0

    H, W = base_rgb.shape[:2]
    scale = 1.0
    longest = max(H, W)
    if params.viz_downsample_max > 0 and longest > params.viz_downsample_max:
        scale = params.viz_downsample_max / float(longest)
        newH = max(1, int(H * scale))
        newW = max(1, int(W * scale))
        base_rgb = resize(base_rgb, (newH, newW), preserve_range=True, anti_aliasing=True).astype(np.float32)
    else:
        newH, newW = H, W

    # Collect edges into segments
    segs = []
    for (u, v) in g.edges():
        yu, xu = g.nodes[u]['y'], g.nodes[u]['x']
        yv, xv = g.nodes[v]['y'], g.nodes[v]['x']
        segs.append([(xu * scale, yu * scale), (xv * scale, yv * scale)])

    if len(segs) == 0 and params.node_size_px <= 0:
        # nothing to draw
        skimage.io.imsave(save_path, (np.clip(base_rgb, 0, 1) * 255).astype(np.uint8))
        return

    # Sample edges if too many
    if len(segs) > params.max_edges_draw:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(segs), size=params.max_edges_draw, replace=False)
        segs = [segs[i] for i in idx]

    fig = plt.figure(figsize=(newW / 100.0, newH / 100.0), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(np.clip(base_rgb, 0, 1), interpolation='nearest')
    ax.set_axis_off()

    if len(segs) > 0:
        lc = LineCollection(segs, colors=params.edge_color, linewidths=params.edge_lw, alpha=params.edge_alpha, antialiaseds=True)
        ax.add_collection(lc)

    if params.node_size_px > 0.0:
        ys = [g.nodes[i]['y'] * scale for i in g.nodes]
        xs = [g.nodes[i]['x'] * scale for i in g.nodes]
        ax.scatter(xs, ys, s=(params.node_size_px ** 2), c='r', alpha=params.node_alpha, edgecolors='none')

    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# -------------------------
# Per-image worker
# -------------------------
def generate_graph_using_srns(args_tuple: Tuple[str, str, str, Params]):
    """Worker for one image: (img_name, im_root_path, cnn_result_root_path, params)."""
    img_name, im_root_path, cnn_result_root_path, params = args_tuple

    im_ext, label_ext, H, W = _dataset_layout(params.dataset)
    # File stems
    override_im_path = None
    prob_base = img_name
    if isinstance(img_name, tuple):
        prob_base, override_im_path = img_name
    tail_idx = util.find(prob_base, '/')
    cur_filename = prob_base[tail_idx[-1] + 1:] if len(tail_idx) > 0 else prob_base

    # Paths
    if params.dataset == 'DRIVE' and (('training' in cur_filename) or ('test' in cur_filename)):
        dataset_subdir = 'training' if 'training' in cur_filename else 'test'
        id_prefix = cur_filename.split('_')[0]
        cur_im_path = os.path.join(im_root_path, dataset_subdir, 'images', cur_filename + '.tif')
        cur_gt_mask_path = os.path.join(im_root_path, dataset_subdir, '1st_manual', id_prefix + '_manual1.gif')
    elif params.dataset == 'CHASE_DB1':
        if override_im_path is not None:
            cur_im_path = override_im_path if os.path.splitext(override_im_path)[1] else _resolve_image_path(override_im_path)
        else:
            cur_im_path = _resolve_image_path(os.path.join(im_root_path, 'training', 'images', cur_filename))
        im_path = Path(cur_im_path)
        cur_gt_mask_path = str(im_path.with_name(im_path.stem + label_ext))
    elif params.dataset == 'HRF':
        split_leaf = Path(cnn_result_root_path).name.lower()
        if 'train' in split_leaf:
            dataset_subdir = 'training'
        elif 'test' in split_leaf:
            dataset_subdir = 'testing'
        else:
            dataset_subdir = _infer_hrf_split(cur_filename)
        cur_im_path = _resolve_image_path(os.path.join(im_root_path, dataset_subdir, 'images', cur_filename))
        cur_gt_mask_path = _resolve_image_path(os.path.join(im_root_path, dataset_subdir, 'manual1', cur_filename + label_ext))
    else:
        cur_im_path = os.path.join(im_root_path, cur_filename + im_ext)
        cur_gt_mask_path = os.path.join(im_root_path, cur_filename + label_ext)

    if params.source_type == 'gt':
        cur_res_prob_path = cur_gt_mask_path
    else:
        cur_res_prob_path = os.path.join(cnn_result_root_path, cur_filename + '_prob.png')

    win_size_str = f'{int(params.win_size):02d}_{int(params.edge_dist_thresh):02d}'
    if params.source_type == 'gt':
        win_size_str += '_gt'

    out_im_png = os.path.join(cnn_result_root_path, f'{cur_filename}_{win_size_str}_vis_graph_res_on_im.png')
    out_mask_png = os.path.join(cnn_result_root_path, f'{cur_filename}_{win_size_str}_vis_graph_res_on_mask.png')
    out_graph = os.path.join(cnn_result_root_path, f'{cur_filename}_{win_size_str}.graph_res')

    # Log
    print(f"[{datetime.now().strftime('%H:%M:%S')}] processing {cur_filename}", flush=True)

    # Read inputs
    im, gt_mask, vesselness = _read_and_fix_sizes(cur_im_path, cur_gt_mask_path, cur_res_prob_path, H, W, params.source_type)

    # Normalize speed map
    if params.source_type == 'gt':
        speed = bwmorph(gt_mask.astype(float), 'dilate', n_iter=1).astype(float)
        speed = _normalize_speed(speed, params)  # will floor by eps
    else:
        speed = _normalize_speed(vesselness, params)

    # Nodes (SRVS)
    max_pos = _srvs_nodes(speed, params.win_size)

    # Init graph + add nodes
    g = nx.Graph()
    for idx, (yy, xx) in enumerate(max_pos):
        g.add_node(idx, kind='MP', y=int(yy), x=int(xx), label=int(idx))
        if DEBUG:
            print('node label', idx, 'pos', (int(yy), int(xx)), 'added')

    # Edges
    if params.edge_method == 'geo_dist':
        edges, eff_thresh, n_valid = _build_edges_geo(max_pos, speed, params.edge_dist_thresh, params.geo_window_margin, params)
    else:  # eu_dist
        edges = _build_edges_eu(max_pos, params.edge_dist_thresh, params)
        eff_thresh = params.edge_dist_thresh
        n_valid = len(max_pos)

    for i, j, w in edges:
        g.add_edge(int(i), int(j), weight=float(w))

    if params.print_stats:
        print(f"  nodes={g.number_of_nodes():,d} valid_sources~{n_valid:,d} edges={g.number_of_edges():,d} "
              f"| win={params.win_size} base_d={params.edge_dist_thresh} eff_d={eff_thresh:.2f} "
              f"| min_local_mean={params.min_local_mean} speed_eps={params.speed_eps} "
              f"| norm={params.speed_norm} p=({params.p_low},{params.p_high}) gamma={params.speed_gamma} "
              f"| auto_scale={params.auto_scale_thresh} target_local_mean={params.target_local_mean}")

    # Visualize
    if params.viz_mode == "util":
        util.visualize_graph(im, g, show_graph=False, save_graph=True,
                             num_nodes_each_type=[0, g.number_of_nodes()], save_path=out_im_png)
        util.visualize_graph(gt_mask, g, show_graph=False, save_graph=True,
                             num_nodes_each_type=[0, g.number_of_nodes()], save_path=out_mask_png)
    else:
        # simple, high-contrast overlays
        _simple_visualize(im, g, out_im_png, params)
        # For mask view, show boolean as gray
        _simple_visualize(gt_mask.astype(float), g, out_mask_png, params)

    # Save graph
    os.makedirs(os.path.dirname(out_graph), exist_ok=True)
    if hasattr(nx, 'write_gpickle'):
        nx.write_gpickle(g, out_graph, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with open(out_graph, 'wb') as f:
            pkl.dump(g, f, protocol=pkl.HIGHEST_PROTOCOL)

    g.clear()
    return out_graph


# -------------------------
# Dataset listing
# -------------------------
def _load_lists_and_roots(dataset: str):
    if dataset == 'DRIVE':
        # Local prob-map directories
        train_probmap_root = "C:/Users/rog/THESIS/DRIU_DRIVE/train"
        test_probmap_root = "C:/Users/rog/THESIS/DRIU_DRIVE/test"
        im_root_path = "C:/Users/rog/THESIS/DATASETS/DRIVE"

        def list_prob_basenames(root_dir, name_contains=None):
            names = []
            if os.path.isdir(root_dir):
                for fname in os.listdir(root_dir):
                    if fname.lower().endswith('_prob.png'):
                        base = fname[:-9]
                        if name_contains and (name_contains not in base):
                            continue
                        names.append(base)
            return sorted(names)

        train_img_names = list_prob_basenames(train_probmap_root, 'training')
        test_img_names = list_prob_basenames(test_probmap_root, 'test')
        return (train_img_names, test_img_names, im_root_path, (train_probmap_root, test_probmap_root))

    if dataset == 'CHASE_DB1':
        train_probmap_root = Path("C:/Users/rog/THESIS/DRIU_CHASE_DB1/train")
        test_probmap_root = Path("C:/Users/rog/THESIS/DRIU_CHASE_DB1/test")
        image_root = Path("C:/Users/rog/THESIS/DATASETS/CHASE_DB1")

        if not train_probmap_root.is_dir():
            raise FileNotFoundError(f"CHASE train prob dir missing: {train_probmap_root}")
        if not test_probmap_root.is_dir():
            raise FileNotFoundError(f"CHASE test prob dir missing: {test_probmap_root}")

        def _prob_base_from_name(name: str) -> str:
            stem = Path(name).stem
            return stem[:-5] if stem.endswith('_prob') else stem

        def _find_image_path(image_name: str) -> str:
            for split in ('training', 'test'):
                cand = image_root / split / 'images' / image_name
                try:
                    return _resolve_image_path(str(cand))
                except FileNotFoundError:
                    continue
            raise FileNotFoundError(f"CHASE image not found for {image_name}")

        def _train_base_to_image(base: str) -> str:
            stem_lower = base.lower()
            if stem_lower.startswith('image_'):
                return _find_image_path(base)
            num_part = base.split('_')[0]
            if not num_part.isdigit():
                raise ValueError(f"Unexpected CHASE train basename: {base}")
            idx = int(num_part)
            rel = idx - 21
            if rel < 0:
                raise ValueError(f"Train basename outside expected range 21-40: {base}")
            subject = rel // 2 + 1
            side = 'L' if idx % 2 == 1 else 'R'
            image_name = f"Image_{subject:02d}{side}"
            return _find_image_path(image_name)

        train_records: List[Tuple[str, str]] = []
        for prob_path in sorted(train_probmap_root.glob('*_prob.png')):
            base = _prob_base_from_name(prob_path.name)
            img_path = _train_base_to_image(base)
            train_records.append((base, img_path))

        test_records: List[Tuple[str, str]] = []
        for prob_path in sorted(test_probmap_root.glob('*_prob.png')):
            base = _prob_base_from_name(prob_path.name)
            img_path = _find_image_path(base)
            test_records.append((base, img_path))

        return (train_records, test_records, "", (str(train_probmap_root), str(test_probmap_root)))

    elif dataset == 'HRF':
        im_root_path = "C:/Users/rog/THESIS/DATASETS/HRF"
        train_probmap_root = "C:/Users/rog/THESIS/DRIU_HRF/train"
        test_probmap_root = "C:/Users/rog/THESIS/DRIU_HRF/test"

        def list_hrf_prob_bases(root_dir: str):
            if not os.path.isdir(root_dir):
                raise FileNotFoundError(f"HRF prob-map root not found: {root_dir}")
            names = []
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith('_prob.png'):
                    names.append(fname[:-9])
            return names

        train_img_names = list_hrf_prob_bases(train_probmap_root)
        test_img_names = list_hrf_prob_bases(test_probmap_root)
        return (train_img_names, test_img_names, im_root_path, (train_probmap_root, test_probmap_root))
    else:
        raise ValueError('Unknown dataset: ' + dataset)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    dataset_key = args.dataset.upper()
    default_ws = {'DRIVE': 4, 'STARE': 4, 'CHASE_DB1': 8, 'HRF': 10}
    default_thresh = {'DRIVE': 10, 'STARE': 10, 'CHASE_DB1': 40, 'HRF': 80}
    if args.win_size is None:
        args.win_size = default_ws.get(dataset_key, 4)
    if args.edge_dist_thresh is None:
        args.edge_dist_thresh = default_thresh.get(dataset_key, 40)
    only_names = tuple([s.strip() for s in args.only_names.split(',') if len(s.strip()) > 0])
    params = Params(
        dataset=args.dataset,
        use_multiprocessing=args.use_multiprocessing,
        processes=int(args.processes),
        source_type=args.source_type,
        win_size=int(args.win_size),
        edge_method=args.edge_method,
        edge_dist_thresh=float(args.edge_dist_thresh),
        geo_window_margin=int(args.geo_window_margin),
        only_names=only_names,

        min_local_mean=float(args.min_local_mean),
        speed_eps=float(args.speed_eps),
        speed_norm=str(args.speed_norm),
        p_low=float(args.p_low),
        p_high=float(args.p_high),
        speed_gamma=float(args.speed_gamma),
        clip_after_gamma=bool(args.clip_after_gamma),
        auto_scale_thresh=bool(args.auto_scale_thresh),
        target_local_mean=float(args.target_local_mean),
        auto_thresh_min_scale=float(args.auto_thresh_min_scale),
        auto_thresh_max_scale=float(args.auto_thresh_max_scale),
        radius_factor=float(args.radius_factor),
        force_knn=int(args.force_knn),
        force_knn_radius=float(args.force_knn_radius),
        viz_mode=str(args.viz_mode),
        viz_downsample_max=int(args.viz_downsample_max),
        edge_alpha=float(args.edge_alpha),
        edge_lw=float(args.edge_lw),
        edge_color=str(args.edge_color),
        node_alpha=float(args.node_alpha),
        node_size_px=float(args.node_size_px),
        max_edges_draw=int(args.max_edges_draw),
        print_stats=bool(args.print_stats)
    )

    print('Called with args:')
    print(args)

    train_img_names, test_img_names, im_root_path, prob_roots = _load_lists_and_roots(params.dataset)

    # Optionally filter by --only_names
    if params.only_names:
        filt = set(params.only_names)

        def _name_in_filter(entry):
            name = entry[0] if isinstance(entry, tuple) else entry
            tail_idx = util.find(name, '/')
            base = name[tail_idx[-1] + 1:] if tail_idx else name
            return base in filt or name in filt

        train_img_names = [n for n in train_img_names if _name_in_filter(n)]
        test_img_names = [n for n in test_img_names if _name_in_filter(n)]

    if isinstance(prob_roots, (tuple, list)):
        train_root, test_root = prob_roots
    else:
        train_root = test_root = prob_roots

    func_arg_train = [(train_img_names[i], im_root_path, train_root, params) for i in range(len(train_img_names))]
    func_arg_test = [(test_img_names[i], im_root_path, test_root, params) for i in range(len(test_img_names))]

    all_args: List[Tuple[str, str, str, Params]] = []
    all_args.extend(func_arg_train)
    all_args.extend(func_arg_test)

    total = len(all_args)
    if total == 0:
        print('No images to process. Exiting.')
        return

    start_ts = time.time()

    def _eta_str(done: int, now_ts: float) -> str:
        elapsed = max(1e-6, now_ts - start_ts)
        rate = done / elapsed
        remain = max(0, total - done)
        eta_s = remain / max(1e-6, rate)
        if eta_s >= 3600:
            h = int(eta_s // 3600)
            m = int((eta_s % 3600) // 60)
            s = int(eta_s % 60)
            return f"{h}:{m:02d}:{s:02d}"
        elif eta_s >= 60:
            m = int(eta_s // 60)
            s = int(eta_s % 60)
            return f"{m:02d}:{s:02d}"
        else:
            return f"{eta_s:.1f}s"

    if params.use_multiprocessing:
        with mp.Pool(processes=params.processes) as pool:
            for idx, out_path in enumerate(pool.imap_unordered(generate_graph_using_srns, all_args, chunksize=1), start=1):
                now = time.time()
                elapsed = now - start_ts
                eta = _eta_str(idx, now)
                base = os.path.basename(out_path) if out_path else 'N/A'
                print(f"[done {idx}/{total}] {base} | elapsed {elapsed:.1f}s | ETA {eta}", flush=True)
    else:
        for idx, x in enumerate(all_args, start=1):
            img_name = x[0]
            tail_idx = util.find(img_name, '/')
            cur_filename = img_name[tail_idx[-1] + 1:] if len(tail_idx) > 0 else img_name
            print(f"[{idx}/{total}] starting {cur_filename}", flush=True)
            out_path = generate_graph_using_srns(x)
            now = time.time()
            elapsed = now - start_ts
            eta = _eta_str(idx, now)
            base = os.path.basename(out_path) if out_path else 'N/A'
            print(f"[done {idx}/{total}] {base} | elapsed {elapsed:.1f}s | ETA {eta}", flush=True)


if __name__ == '__main__':
    main()
