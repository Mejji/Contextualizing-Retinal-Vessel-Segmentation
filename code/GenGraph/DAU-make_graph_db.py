#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_graph_db_fast_parity.py
--------------------------------
Shin-parity, faster graph generator for VGN-style datasets (DRIVE/STARE/CHASE_DB1/HRF).

Supports **DRIU** and **DA-U²Net** prob maps:
- DRIU: "<stem>_prob.png" already in canonical canvas -> align='top_left'
- DA-U²Net: "<stem>.png" at original HxW -> align='center' into canonical canvas

Speed-ups (without topology changes):
- Local-window FMM (skfmm.travel_time) near each source (radius ~ threshold)
- KD-tree candidate pruning in Euclidean space
"""
import argparse
import os
import pickle as pkl
import multiprocessing as mp
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import skimage.io
import networkx as nx
import skfmm
from scipy.spatial import cKDTree

# Project-local utilities
import _init_paths  # noqa: F401
from bwmorph import bwmorph
from config import cfg  # noqa: F401  (kept for project parity, not used here)
import util as util

DEBUG = False


# --------------------- Config / CLI ---------------------
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
    # NEW:
    producer: str = 'dau2net'          # 'driu' or 'dau2net'
    prob_suffix: str = '.png'           # e.g., '_prob.png' (driu) or '.png' (dau2net)
    align: str = 'center'         # 'top_left' | 'center' | 'resize'
    prob_root_train: str = 'C:/Users/rog/THESIS/DAU2_DRIVE/train/prob'       # where train prob maps live
    prob_root_test: str = 'C:/Users/rog/THESIS/DAU2_DRIVE/test/prob'        # where test prob maps live


def parse_args():
    p = argparse.ArgumentParser(description='Make a graph db (Shin-parity, fast windowed-FMM)')
    p.add_argument('--dataset', default='DRIVE', type=str,
                   help='DRIVE | STARE | CHASE_DB1 | HRF')
    p.add_argument('--use_multiprocessing', dest='use_multiprocessing',
                   action='store_true', default=True, help='Enable multiprocessing (default: on)')
    p.add_argument('--no_multiprocessing', dest='use_multiprocessing',
                   action='store_false', help='Disable multiprocessing')
    p.add_argument('--processes', type=int, default=20, help='Number of workers when MP is enabled')
    p.add_argument('--source_type', default='result', type=str, choices=['result', 'gt'],
                   help='Use CNN result prob map ("result") or ground-truth mask ("gt") as speed')
    p.add_argument('--win_size', default=4, type=int, help='SRVS grid window size (default: 4)')
    p.add_argument('--edge_method', default='geo_dist', type=str, choices=['geo_dist', 'eu_dist'],
                   help='Edge construction method')
    p.add_argument('--edge_dist_thresh', default=10.0, type=float,
                   help='Distance threshold for edge construction')
    p.add_argument('--geo_window_margin', default=1, type=int,
                   help='Extra pixels for the crop around radius R for FMM (default: 1)')
    p.add_argument('--only_names', default='', type=str,
                   help='Comma-separated basenames (without extension) to process only these files')

    # NEW: producer / alignment / paths
    p.add_argument('--producer', default='dau2net', choices=['driu', 'dau2net'],
                   help='Prob-map producer. Controls filename pattern + default alignment.')
    p.add_argument('--prob_suffix', default='.png', type=str,
                   help='Override prob filename suffix. Default: _prob.png (driu) | .png (dau2net).')
    p.add_argument('--align', default='center', choices=[None, 'top_left', 'center', 'resize'],
                   help='How to fit arrays to canonical size. Default: driu=top_left, dau2net=center.')
    p.add_argument('--prob_root_train', default='C:/Users/rog/THESIS/DAU2_DRIVE/train/prob', type=str,
                   help='Folder containing TRAIN prob maps (defaults to DA-U²Net path).')
    p.add_argument('--prob_root_test', default='C:/Users/rog/THESIS/DAU2_DRIVE/test/prob', type=str,
                   help='Folder containing TEST prob maps (defaults to DA-U²Net path).')
    p.add_argument('--im_root_path', default='C:/Users/rog/THESIS/DATASETS/DRIVE', type=str,
                   help='Root of original dataset images (expects DRIVE-like structure).')
    return p.parse_args()


# --------------------- Dataset utils ---------------------
def _dataset_layout(img_name: str):
    """Return (im_ext, label_ext, H, W) based on dataset token in the name."""
    if 'DRIVE' in img_name or True:  # default to DRIVE-like if unsure
        return ('_image.tif', '_label.gif', 592, 592)
    elif 'STARE' in img_name:
        return ('.ppm', '.ah.ppm', 704, 704)
    elif 'CHASE_DB1' in img_name:
        return ('.jpg', '_1stHO.png', 1024, 1024)
    elif 'HRF' in img_name:
        return ('.bmp', '.tif', 768, 768)
    else:
        return ('_image.tif', '_label.gif', 592, 592)


def _fit_to_canvas(arr, tgt_shape, mode='top_left'):
    """
    Fit 2D or 3D array to tgt_shape=(H,W) or (H,W,C).
    modes:
      - 'top_left': copy into (0,0) and crop overflow
      - 'center'  : center-pad/crop around the middle
      - 'resize'  : resize to (H,W)
    """
    if mode == 'resize':
        H, W = (tgt_shape[:2] if len(tgt_shape) == 3 else tgt_shape)
        return skimage.transform.resize(arr, (H, W) if arr.ndim == 2 else (H, W, arr.shape[2]),
                                        order=1, preserve_range=True, anti_aliasing=True).astype(arr.dtype)

    H, W = (tgt_shape[:2] if len(tgt_shape) == 3 else tgt_shape)
    out = np.zeros(tgt_shape, dtype=arr.dtype)
    h, w = arr.shape[:2]
    if mode == 'top_left':
        y0 = x0 = 0
    elif mode == 'center':
        y0 = max(0, (H - h) // 2)
        x0 = max(0, (W - w) // 2)
    else:
        raise ValueError(f'Unknown align mode: {mode}')

    ys0, ys1 = y0, min(y0 + h, H)
    xs0, xs1 = x0, min(x0 + w, W)
    ys0_s, ys1_s = 0, ys1 - ys0
    xs0_s, xs1_s = 0, xs1 - xs0
    out[ys0:ys1, xs0:xs1, ...] = arr[ys0_s:ys1_s, xs0_s:xs1_s, ...]
    return out


def _read_and_fix_sizes(cur_im_path, cur_gt_mask_path, cur_res_prob_path, H, W, source_type, align_mode):
    im = skimage.io.imread(cur_im_path)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)

    # GT
    if os.path.exists(cur_gt_mask_path):
        gt_mask = skimage.io.imread(cur_gt_mask_path)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]
        gt_mask = (gt_mask.astype(float) / 255.0) >= 0.5
    else:
        gt_mask = np.zeros(im.shape[:2], dtype=bool)

    # Speed / vesselness
    if source_type == 'gt':
        vesselness = gt_mask.astype(float)
    else:
        vesselness = skimage.io.imread(cur_res_prob_path).astype(float)
        if vesselness.ndim == 3:
            vesselness = vesselness[..., 0]
        if vesselness.max() > 1.0:
            vesselness /= 255.0

    # Align all into canonical canvas
    im_fix = _fit_to_canvas(im, (H, W, 3), mode=align_mode)
    gt_fix = _fit_to_canvas(gt_mask.astype(np.uint8), (H, W), mode=align_mode).astype(bool)
    ves_fix = _fit_to_canvas(vesselness, (H, W), mode=align_mode).astype(float)
    return im_fix, gt_fix, ves_fix


def _srvs_nodes(vesselness: np.ndarray, win_size: int) -> List[Tuple[int, int]]:
    """Semi-regular vertex sampling: one node per patch; argmax or center fallback."""
    H, W = vesselness.shape
    y_quan = sorted(set(range(0, H, win_size)) | {H})
    x_quan = sorted(set(range(0, W, win_size)) | {W})
    max_pos = []
    for y0, y1 in zip(y_quan[:-1], y_quan[1:]):
        for x0, x1 in zip(x_quan[:-1], x_quan[1:]):
            patch = vesselness[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if float(np.sum(patch)) == 0:
                max_pos.append((y0 + patch.shape[0] // 2, x0 + patch.shape[1] // 2))
            else:
                rr, cc = np.unravel_index(np.argmax(patch), patch.shape)
                max_pos.append((y0 + int(rr), x0 + int(cc)))
    return max_pos


def _windowed_travel_time(speed: np.ndarray, sy: int, sx: int, thresh: float, margin: int):
    """FMM in a local window centered at (sy,sx)."""
    H, W = speed.shape
    R = int(np.ceil(thresh)) + int(max(0, margin))
    y0 = max(0, sy - R); y1 = min(H, sy + R + 1)
    x0 = max(0, sx - R); x1 = min(W, sx + R + 1)
    speed_win = speed[y0:y1, x0:x1]
    phi = np.ones_like(speed_win, dtype=float)
    phi[sy - y0, sx - x0] = -1.0
    tt = skfmm.travel_time(phi, speed_win, narrow=thresh)
    if isinstance(tt, np.ma.MaskedArray):
        tt = np.ma.filled(tt, np.inf)
    return tt, (y0, y1, x0, x1)


def _build_edges_geo(max_pos, speed, thresh, margin):
    H, W = speed.shape
    nodes = np.asarray(max_pos, dtype=int)
    ys = nodes[:, 0].astype(np.float32)
    xs = nodes[:, 1].astype(np.float32)
    n = len(nodes)

    # Source gating
    valid = []
    for i in range(n):
        yy, xx = int(ys[i]), int(xs[i])
        if speed[yy, xx] == 0:
            continue
        y0, y1 = max(0, yy - 1), min(H, yy + 2)
        x0, x1 = max(0, xx - 1), min(W, xx + 2)
        if float(speed[y0:y1, x0:x1].mean()) < 0.1:
            continue
        valid.append(i)
    if len(valid) == 0:
        return []

    # KD-tree candidate pruning within radius thresh
    coords = np.stack([ys, xs], axis=1)
    tree = cKDTree(coords, leafsize=32)
    radius = float(thresh) + 1e-6

    edges = []
    for k, i in enumerate(valid, start=1):
        sy, sx = int(ys[i]), int(xs[i])
        cand = tree.query_ball_point([ys[i], xs[i]], r=radius)
        cand = [j for j in cand if j > i]
        if not cand:
            continue
        tt_win, (y0, y1, x0, x1) = _windowed_travel_time(speed, sy, sx, thresh, margin)
        for j in cand:
            cy, cx = int(ys[j]), int(xs[j])
            if cy < y0 or cy >= y1 or cx < x0 or cx >= x1:
                continue
            geo = float(tt_win[cy - y0, cx - x0])
            if not np.isfinite(geo):
                continue
            if geo < thresh:
                w = float(thresh / (thresh + geo))
                edges.append((i, j, w))
        if DEBUG and (k % 500 == 0 or k == len(valid)):
            print(f'[DEBUG] geo edges: processed {k}/{len(valid)} sources; edges so far = {len(edges)}')
    return edges


def _build_edges_eu(max_pos, thresh):
    nodes = np.asarray(max_pos, dtype=int)
    ys = nodes[:, 0].astype(np.float32)
    xs = nodes[:, 1].astype(np.float32)
    coords = np.stack([ys, xs], axis=1)
    tree = cKDTree(coords, leafsize=32)
    radius = float(thresh) + 1e-6
    edges = []
    for i in range(len(nodes)):
        neigh = tree.query_ball_point([ys[i], xs[i]], r=radius)
        for j in neigh:
            if j > i:
                edges.append((i, j, 1.0))
    return edges


# --------------------- Listing DA-U²Net / DRIU files ---------------------
def _choose_latest_epoch(files: List[str]) -> str:
    """
    Given file paths like ".../ep100_01_training.png" and ".../ep120_01_training.png",
    return the one with the highest ep number. Falls back to the last item if parse fails.
    """
    best = None
    best_ep = -1
    for f in files:
        base = os.path.basename(f)
        if base.startswith('ep'):
            try:
                ep = int(base.split('_', 1)[0][2:])
            except Exception:
                ep = -1
        else:
            # plain file without ep -> treat as epoch infinity so it wins
            ep = 1_000_000_000
        if ep > best_ep:
            best = f
            best_ep = ep
    return best or (files[-1] if files else None)


def _list_items_daunet(root_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (img_name, prob_path) for DA-U²Net.
    Handles both "01_test.png" and "ep100_01_training.png" by mapping to
    canonical img_name "01_test" / "01_training" while keeping the actual prob file path.
    """
    out: Dict[str, List[str]] = {}
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith('.png'):
            continue
        if fname.lower().startswith('overlay') or fname.lower().startswith('.'):
            continue
        stem = os.path.splitext(fname)[0]
        # Accept "01_test" or "01_training"
        if '_' in stem and stem.split('_')[-1] in ('test', 'training') and stem.split('_')[0].isdigit():
            key = stem
        else:
            # Accept "epXYZ_01_test" or "epXYZ_01_training"
            if stem.startswith('ep') and '_' in stem:
                rest = stem.split('_', 1)[1]
                if '_' in rest and rest.split('_')[-1] in ('test', 'training') and rest.split('_')[0].isdigit():
                    key = rest
                else:
                    continue
            else:
                continue
        out.setdefault(key, []).append(os.path.join(root_dir, fname))
    # choose latest per key
    items = []
    for key, paths in out.items():
        pick = _choose_latest_epoch(sorted(paths))
        if pick:
            items.append((key, pick))
    items.sort(key=lambda x: x[0])
    return items


def _list_items_driu(root_dir: str, suffix='_prob.png') -> List[Tuple[str, str]]:
    """
    Return list of (img_name, prob_path) for DRIU-style outputs "<stem>_prob.png".
    """
    items = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(suffix):
            continue
        stem = fname[:-len(suffix)]
        items.append((stem, os.path.join(root_dir, fname)))
    items.sort(key=lambda x: x[0])
    return items


def _load_lists_and_roots(params: Params):
    """
    Returns:
        train_items: List[(img_name, prob_path)]
        test_items:  List[(img_name, prob_path)]
        im_root_path: str
    """
    im_root_path = 'C:/Users/rog/THESIS/DATASETS/DRIVE'  # default for DRIVE

    if params.dataset == 'DRIVE':
        train_root = params.prob_root_train
        test_root = params.prob_root_test
        if params.producer == 'dau2net':
            train_items = _list_items_daunet(train_root)
            test_items = _list_items_daunet(test_root)
        else:
            suffix = params.prob_suffix if params.prob_suffix else '_prob.png'
            train_items = _list_items_driu(train_root, suffix=suffix)
            test_items = _list_items_driu(test_root, suffix=suffix)
        return train_items, test_items, im_root_path
    else:
        raise ValueError('This script currently ships with DRIVE defaults; adapt _dataset_layout/_paths for other datasets.')


# --------------------- Per-image worker ---------------------
def generate_graph_using_srns(args_tuple: Tuple[str, str, str, Params]):
    """
    Worker for one image: (img_name, im_root_path, cur_res_prob_path, params).
    """
    img_name, im_root_path, cur_res_prob_path, params = args_tuple

    im_ext, label_ext, H, W = _dataset_layout(img_name)

    # Resolve dataset-specific image & GT paths (DRIVE)
    if params.dataset == 'DRIVE' and (('training' in img_name) or ('test' in img_name)):
        dataset_subdir = 'training' if 'training' in img_name else 'test'
        id_prefix = img_name.split('_')[0]
        cur_im_path = os.path.join(im_root_path, dataset_subdir, 'images', img_name + '.tif')
        cur_gt_mask_path = os.path.join(im_root_path, dataset_subdir, '1st_manual', id_prefix + '_manual1.gif')
    else:
        cur_im_path = os.path.join(im_root_path, img_name + im_ext)
        cur_gt_mask_path = os.path.join(im_root_path, img_name + label_ext)

    # Output root = where the prob map lives
    cnn_result_root_path = os.path.dirname(cur_res_prob_path)
    win_size_str = f'{int(params.win_size):02d}_{int(params.edge_dist_thresh):02d}'
    if params.source_type == 'gt':
        win_size_str += '_gt'
    out_im_png = os.path.join(cnn_result_root_path, f'{img_name}_{win_size_str}_vis_graph_res_on_im.png')
    out_mask_png = os.path.join(cnn_result_root_path, f'{img_name}_{win_size_str}_vis_graph_res_on_mask.png')
    out_graph = os.path.join(cnn_result_root_path, f'{img_name}_{win_size_str}.graph_res')

    # Log
    print(f"[{datetime.now().strftime('%H:%M:%S')}] processing {img_name}", flush=True)

    # Read inputs (with alignment)
    im, gt_mask, vesselness = _read_and_fix_sizes(
        cur_im_path, cur_gt_mask_path, cur_res_prob_path, H, W, params.source_type, params.align
    )

    # Nodes
    max_pos = _srvs_nodes(vesselness, params.win_size)

    # Graph init
    g = nx.Graph()
    for idx, (yy, xx) in enumerate(max_pos):
        g.add_node(idx, kind='MP', y=int(yy), x=int(xx), label=int(idx))

    # Speed map (parity with Shin when using GT)
    speed = vesselness.copy()
    if params.source_type == 'gt':
        speed = bwmorph(speed, 'dilate', n_iter=1).astype(float)

    # Edges
    if params.edge_method == 'geo_dist':
        edges = _build_edges_geo(max_pos, speed, params.edge_dist_thresh, params.geo_window_margin)
    else:
        edges = _build_edges_eu(max_pos, params.edge_dist_thresh)
    for i, j, w in edges:
        g.add_edge(int(i), int(j), weight=float(w))

    # Visualize (project util)
    util.visualize_graph(im, g, show_graph=False, save_graph=True,
                         num_nodes_each_type=[0, g.number_of_nodes()], save_path=out_im_png)
    util.visualize_graph(gt_mask, g, show_graph=False, save_graph=True,
                         num_nodes_each_type=[0, g.number_of_nodes()], save_path=out_mask_png)

    # Save graph
    os.makedirs(os.path.dirname(out_graph), exist_ok=True)
    if hasattr(nx, 'write_gpickle'):
        nx.write_gpickle(g, out_graph, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with open(out_graph, 'wb') as f:
            pkl.dump(g, f, protocol=pkl.HIGHEST_PROTOCOL)

    g.clear()
    return out_graph


# --------------------- Main ---------------------
def main():
    args = parse_args()

    # Derive defaults from producer
    default_suffix = '.png' if args.producer == 'dau2net' else '_prob.png'
    align_mode = args.align if args.align is not None else ('center' if args.producer == 'dau2net' else 'top_left')

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
        producer=args.producer,
        prob_suffix=(args.prob_suffix or default_suffix),
        align=align_mode,
        prob_root_train=args.prob_root_train,
        prob_root_test=args.prob_root_test,
    )

    print('Called with args:')
    print(args)

    # List files
    train_items, test_items, im_root_path = _load_lists_and_roots(params)

    # Optional filter by --only_names (match canonical stems like "01_test"/"01_training")
    if params.only_names:
        filt = set(params.only_names)
        train_items = [it for it in train_items if it[0] in filt]
        test_items = [it for it in test_items if it[0] in filt]

    # Pack worker args
    func_arg_train = [(name, im_root_path, prob_path, params) for (name, prob_path) in train_items]
    func_arg_test  = [(name, im_root_path, prob_path, params) for (name, prob_path) in test_items]

    all_args: List[Tuple[str, str, str, Params]] = func_arg_train + func_arg_test
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
            h = int(eta_s // 3600); m = int((eta_s % 3600) // 60); s = int(eta_s % 60)
            return f"{h}:{m:02d}:{s:02d}"
        elif eta_s >= 60:
            m = int(eta_s // 60); s = int(eta_s % 60)
            return f"{m:02d}:{s:02d}"
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
            print(f"[{idx}/{total}] starting {img_name}", flush=True)
            out_path = generate_graph_using_srns(x)
            now = time.time()
            elapsed = now - start_ts
            eta = _eta_str(idx, now)
            base = os.path.basename(out_path) if out_path else 'N/A'
            print(f"[done {idx}/{total}] {base} | elapsed {elapsed:.1f}s | ETA {eta}", flush=True)


if __name__ == '__main__':
    main()
