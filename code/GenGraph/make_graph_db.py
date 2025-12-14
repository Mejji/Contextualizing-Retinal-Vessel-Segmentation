
#!/usr/bin/env python3
"""
Baseline graph database generator (Python 3) — matches the original Python-2 logic.

Key points:
- Uses scikit-fmm (skfmm.travel_time) exactly like the original for geodesic edges.
- Keeps the original node proposal grid, source gating, and edge construction.
- Fixes Python-3 compatibility (print, xrange, tuple-unpacking, NetworkX API).
- Uses only arguments passed into the worker (no reliance on global args), so it
  works on Windows 'spawn' mode too.

CLI example:
  python make_graph_db_baseline_py3.py --dataset DRIVE --source_type result --edge_method geo_dist --edge_dist_thresh 10 --win_size 4

This will write *.graph_res and visualization PNGs into the CNN result directory.
"""

import argparse
import os
import pickle as pkl
import multiprocessing
import time
import numpy as np
import skimage.io
import networkx as nx
import matplotlib.pyplot as plt
import skfmm
try:
    # NetworkX 3.x: import module then access function to satisfy static analyzers
    from networkx.readwrite import gpickle as nx_gpickle
    nx_write_gpickle = nx_gpickle.write_gpickle
except Exception:
    nx_write_gpickle = None

# Project-local utilities (same as original)
import _init_paths  # noqa: F401
from bwmorph import bwmorph
from config import cfg  # not explicitly used here, but kept to match project layout
import util as util

DEBUG = False


def parse_args():
    """Parse input arguments (same flags as the original)."""
    parser = argparse.ArgumentParser(description='Make a graph db (baseline, Python 3)')
    parser.add_argument('--dataset', default='DRIVE',
                        help='Dataset to use: DRIVE | STARE | CHASE_DB1 | HRF', type=str)
    # Provide robust toggles for multiprocessing; default True (like original intent)
    parser.add_argument('--use_multiprocessing', dest='use_multiprocessing', action='store_true', default=True,
                        help='Enable multiprocessing (default: True)')
    parser.add_argument('--no_multiprocessing', dest='use_multiprocessing', action='store_false',
                        help='Disable multiprocessing')
    parser.add_argument('--source_type', default='result',
                        help='Source to be used: result | gt', type=str)
    parser.add_argument('--win_size', default=4,
                        help='Window size for SRNS grid', type=int)  # [4,8,16]
    parser.add_argument('--edge_method', default='geo_dist',
                        help='Edge construction method: geo_dist | eu_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=10.0,
                        help='Distance threshold for edge construction', type=float)  # [10,20,40]
    parser.add_argument('--processes', default=20, type=int,
                        help='Number of worker processes if multiprocessing is enabled (default: 20)')
    parser.add_argument('--only_names', default='22_training', type=str,
                        help='Comma-separated basenames to process (e.g., "21_training,01_test"). If empty, process all found.')
    return parser.parse_args()


def generate_graph_using_srns(arg_tuple):
    """Worker: build a graph for one image (exact baseline behavior).

    arg_tuple = (idx, total, img_name, im_root_path, cnn_result_root_path, params) or (img_name, im_root_path, cnn_result_root_path, params)
    where params is the argparse.Namespace with all CLI fields.
    """
    # Support both (idx,total,...) and legacy 4-tuple
    if isinstance(arg_tuple[0], int):
        idx, total, img_name, im_root_path, cnn_result_root_path, params = arg_tuple
    else:
        img_name, im_root_path, cnn_result_root_path, params = arg_tuple
        idx, total = None, None

    t0 = time.time()

    win_size_str = f"{int(params.win_size):02d}_{int(params.edge_dist_thresh):02d}"
    if params.source_type == 'gt':
        win_size_str = win_size_str + '_gt'

    # Dataset-specific file layout
    if 'DRIVE' in img_name:
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        len_y = 592
        len_x = 592
    elif 'STARE' in img_name:
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
        len_y = 704
        len_x = 704
    elif 'CHASE_DB1' in img_name:
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        len_y = 1024
        len_x = 1024
    elif 'HRF' in img_name:
        im_ext = '.bmp'
        label_ext = '.tif'
        len_y = 768
        len_x = 768
    else:
        # Fallback: assume DRIVE-like naming
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        len_y = 592
        len_x = 592

    # Extract simple filename tail
    slash_positions = util.find(img_name, '/')
    tail_start = slash_positions[-1] + 1 if len(slash_positions) > 0 else 0
    cur_filename = img_name[tail_start:]
    if idx is not None and total is not None:
        print(f"[STATUS] Start [{idx}/{total}] {cur_filename}")
    else:
        print('processing ' + cur_filename)

    # Support dataset layouts where images are under subfolders, e.g. DRIVE/{training,test}
    dataset_subdir = 'training' if 'training' in cur_filename else ('test' if 'test' in cur_filename else '')
    if params.dataset == 'DRIVE' and dataset_subdir:
        # DRIVE structure: DATASETS/DRIVE/{training|test}/{images|1st_manual|mask}
        id_prefix = cur_filename.split('_')[0]
        cur_im_path = os.path.join(im_root_path, dataset_subdir, 'images', cur_filename + '.tif')
        # Use 1st_manual vessel labels as GT
        cur_gt_mask_path = os.path.join(im_root_path, dataset_subdir, '1st_manual', id_prefix + '_manual1.gif')
    elif dataset_subdir:
        cur_im_path = os.path.join(im_root_path, dataset_subdir, cur_filename + im_ext)
        cur_gt_mask_path = os.path.join(im_root_path, dataset_subdir, cur_filename + label_ext)
    else:
        cur_im_path = os.path.join(im_root_path, cur_filename + im_ext)
        cur_gt_mask_path = os.path.join(im_root_path, cur_filename + label_ext)

    if params.source_type == 'gt':
        cur_res_prob_path = cur_gt_mask_path
    else:
        cur_res_prob_path = os.path.join(cnn_result_root_path, cur_filename + '_prob.png')

    cur_vis_res_im_savepath = os.path.join(cnn_result_root_path, cur_filename + '_' + win_size_str + '_vis_graph_res_on_im.png')
    cur_vis_res_mask_savepath = os.path.join(cnn_result_root_path, cur_filename + '_' + win_size_str + '_vis_graph_res_on_mask.png')
    cur_res_graph_savepath = os.path.join(cnn_result_root_path, cur_filename + '_' + win_size_str + '.graph_res')
    # No difference in paths across edge methods

    # Read inputs
    im = skimage.io.imread(cur_im_path)

    gt_mask = skimage.io.imread(cur_gt_mask_path)
    gt_mask = gt_mask.astype(float) / 255.0
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    gt_mask = gt_mask >= 0.5

    vesselness = skimage.io.imread(cur_res_prob_path)
    vesselness = vesselness.astype(float) / 255.0
    if vesselness.ndim == 3:
        vesselness = vesselness[:, :, 0]

    # Canonical pad/crop (same sizes as original)
    temp = np.copy(im)
    im_fix = np.zeros((len_y, len_x) + ((3,) if im.ndim == 3 else ()), dtype=temp.dtype)
    im_fix[:temp.shape[0], :temp.shape[1], ...] = temp

    temp = np.copy(gt_mask)
    gt_mask_fix = np.zeros((len_y, len_x), dtype=temp.dtype)
    gt_mask_fix[:temp.shape[0], :temp.shape[1]] = temp

    temp = np.copy(vesselness)
    vessel_fix = np.zeros((len_y, len_x), dtype=temp.dtype)
    vessel_fix[:temp.shape[0], :temp.shape[1]] = temp

    im = im_fix
    gt_mask = gt_mask_fix
    vesselness = vessel_fix

    # Find local maxima (grid patches)
    im_y, im_x = im.shape[0], im.shape[1]
    y_quan = sorted(set(list(range(0, im_y, params.win_size))) | {im_y})
    x_quan = sorted(set(list(range(0, im_x, params.win_size))) | {im_x})

    max_pos = []
    max_val = []
    for y_idx in range(len(y_quan) - 1):
        for x_idx in range(len(x_quan) - 1):
            y0, y1 = y_quan[y_idx], y_quan[y_idx + 1]
            x0, x1 = x_quan[x_idx], x_quan[x_idx + 1]
            cur_patch = vesselness[y0:y1, x0:x1]
            if np.sum(cur_patch) == 0:
                max_val.append(0.0)
                max_pos.append((y0 + cur_patch.shape[0] // 2, x0 + cur_patch.shape[1] // 2))
            else:
                max_val.append(float(np.amax(cur_patch)))
                rr, cc = np.unravel_index(np.argmax(cur_patch), cur_patch.shape)
                max_pos.append((y0 + int(rr), x0 + int(cc)))

    graph = nx.Graph()

    # Add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=int(node_y), x=int(node_x), label=int(node_idx))
        if DEBUG:
            print('node label', node_idx, 'pos', (int(node_y), int(node_x)), 'added')

    # Speed map
    speed = vesselness.copy()
    if params.source_type == 'gt':
        speed = bwmorph(speed, 'dilate', n_iter=1)
        speed = speed.astype(float)

    edge_dist_thresh_sq = float(params.edge_dist_thresh) ** 2

    node_list = list(graph.nodes)
    for i, n in enumerate(node_list):
        yy = graph.nodes[n]['y']
        xx = graph.nodes[n]['x']

        if speed[yy, xx] == 0:
            continue
        y0, y1 = max(0, yy - 1), min(im_y, yy + 2)
        x0, x1 = max(0, xx - 1), min(im_x, xx + 2)
        neighbor = speed[y0:y1, x0:x1]

        if float(np.mean(neighbor)) < 0.1:
            continue

        if params.edge_method == 'geo_dist':
            phi = np.ones_like(speed, dtype=float)
            phi[yy, xx] = -1.0
            # Travel time (FMM)
            tt = skfmm.travel_time(phi, speed, narrow=params.edge_dist_thresh)
            # Fill masked (unreachable) with +inf to avoid warnings and NaNs
            if isinstance(tt, np.ma.MaskedArray):
                tt = np.ma.filled(tt, np.inf)

            if DEBUG:
                plt.figure()
                plt.imshow(tt, interpolation='nearest')
                plt.show()
                plt.cla(); plt.clf(); plt.close()

            for n_comp in node_list[i+1:]:
                cy = graph.nodes[n_comp]['y']
                cx = graph.nodes[n_comp]['x']
                geo_dist = float(tt[cy, cx])
                if not np.isfinite(geo_dist):
                    continue
                if geo_dist < params.edge_dist_thresh:
                    w = params.edge_dist_thresh / (params.edge_dist_thresh + geo_dist)
                    graph.add_edge(n, n_comp, weight=float(w))
                    if DEBUG:
                        print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')

        elif params.edge_method == 'eu_dist':
            for n_comp in node_list[i+1:]:
                dy = float(graph.nodes[n_comp]['y'] - yy)
                dx = float(graph.nodes[n_comp]['x'] - xx)
                eu_dist = dy * dy + dx * dx
                if eu_dist < edge_dist_thresh_sq:
                    graph.add_edge(n, n_comp, weight=1.0)
                    if DEBUG:
                        print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
        else:
            raise NotImplementedError('Unknown edge_method: ' + str(params.edge_method))

    # Visualize the constructed graph
    util.visualize_graph(im, graph, show_graph=False,
                         save_graph=True, num_nodes_each_type=[0, graph.number_of_nodes()],
                         save_path=cur_vis_res_im_savepath)
    util.visualize_graph(gt_mask, graph, show_graph=False,
                         save_graph=True, num_nodes_each_type=[0, graph.number_of_nodes()],
                         save_path=cur_vis_res_mask_savepath)

    # Save
    os.makedirs(os.path.dirname(cur_res_graph_savepath), exist_ok=True)
    # Robust save across NetworkX versions
    if hasattr(nx, 'write_gpickle'):
        nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    elif nx_write_gpickle is not None:
        nx_write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        # Fallback to raw pickle if NetworkX helpers are unavailable
        with open(cur_res_graph_savepath, 'wb') as f:
            pkl.dump(graph, f, protocol=pkl.HIGHEST_PROTOCOL)
    print('Saved graph:', cur_res_graph_savepath)
    duration = time.time() - t0
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    graph.clear()

    return {
        'idx': int(idx) if idx is not None else None,
        'total': int(total) if total is not None else None,
        'img_name': cur_filename,
        'nodes': int(nodes),
        'edges': int(edges),
        'duration_sec': float(duration),
        'out_path': cur_res_graph_savepath,
    }


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    # Dataset roots (updated for local folders where needed)
    if args.dataset == 'DRIVE':
        # Use local prob-map directories directly instead of *.txt file lists
        train_probmap_root = 'C:/Users/rog/THESIS/DRIU_DRIVE/train'
        test_probmap_root = 'C:/Users/rog/THESIS/DRIU_DRIVE/test'
        # Base root for original images/labels; assumes subfolders 'training' and 'test'
        im_root_path = 'C:/Users/rog/THESIS/DATASETS/DRIVE'
    elif args.dataset == 'STARE':
        train_set_txt_path = '../../STARE/train.txt'
        test_set_txt_path = '../../STARE/test.txt'
        im_root_path = '../../STARE/all'
        cnn_result_root_path = '../STARE_cnn/res_resized'
    elif args.dataset == 'CHASE_DB1':
        train_set_txt_path = '../../CHASE_DB1/train.txt'
        test_set_txt_path = '../../CHASE_DB1/test.txt'
        im_root_path = '../../CHASE_DB1/all'
        cnn_result_root_path = '../CHASE_cnn/test_resized_graph_gen'
    elif args.dataset == 'HRF':
        train_set_txt_path = '../../HRF/train_768.txt'
        test_set_txt_path = '../../HRF/test_768.txt'
        im_root_path = '../../HRF/all_768'
        cnn_result_root_path = '../HRF_cnn/test'
    else:
        raise ValueError('Unsupported dataset: ' + args.dataset)

    # Read lists
    if args.dataset == 'DRIVE':
        # Derive basenames from available prob-map files: <name>_prob.png
        def list_prob_basenames(root_dir, name_contains=None):
            names = []
            if os.path.isdir(root_dir):
                for fname in os.listdir(root_dir):
                    if fname.lower().endswith('_prob.png'):
                        base = fname[:-9]  # strip '_prob.png'
                        if name_contains and (name_contains not in base):
                            continue
                        names.append(base)
            return sorted(names)

        train_img_names = list_prob_basenames(train_probmap_root, 'training')
        test_img_names = list_prob_basenames(test_probmap_root, 'test')
    else:
        with open(train_set_txt_path, 'r') as f:
            train_img_names = [x.strip() for x in f.readlines() if x.strip()]
        with open(test_set_txt_path, 'r') as f:
            test_img_names = [x.strip() for x in f.readlines() if x.strip()]

    # Optional filtering by specific basenames
    if args.only_names:
        wanted = {n.strip() for n in args.only_names.split(',') if n.strip()}
        train_img_names = [n for n in train_img_names if n in wanted]
        test_img_names = [n for n in test_img_names if n in wanted]

    len_train = len(train_img_names)
    len_test = len(test_img_names)

    func = generate_graph_using_srns
    if args.dataset == 'DRIVE':
        # Pass train/test probmap roots separately
        func_arg_train = [(train_img_names[x], im_root_path, train_probmap_root, args) for x in range(len_train)]
        func_arg_test = [(test_img_names[x], im_root_path, test_probmap_root, args) for x in range(len_test)]
    else:
        func_arg_train = [(train_img_names[x], im_root_path, cnn_result_root_path, args) for x in range(len_train)]
        func_arg_test = [(test_img_names[x], im_root_path, cnn_result_root_path, args) for x in range(len_test)]

    # Build a single list and tag with indices for progress
    all_args = []
    total = len_train + len_test
    for i, a in enumerate(func_arg_train, start=1):
        all_args.append((i, total) + a)
    for j, a in enumerate(func_arg_test, start=len_train + 1):
        all_args.append((j, total) + a)

    if args.use_multiprocessing:
        start = time.time()
        completed = 0
        with multiprocessing.Pool(processes=int(args.processes)) as pool:
            for res in pool.imap_unordered(func, all_args):
                completed += 1
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = total - completed
                eta_sec = (remaining / rate) if rate > 0 else 0.0
                hh = int(eta_sec // 3600)
                mm = int((eta_sec % 3600) // 60)
                ss = int(eta_sec % 60)
                pct = 100.0 * completed / total if total > 0 else 100.0
                print(f"[PROGRESS] {completed}/{total} ({pct:.1f}%) | ETA {hh:02d}:{mm:02d}:{ss:02d} | last: {res['img_name']} nodes={res['nodes']} edges={res['edges']} dur={res['duration_sec']:.1f}s")
    else:
        start = time.time()
        completed = 0
        total = len(all_args)
        avg = None
        for arg in all_args:
            res = func(arg)
            completed += 1
            # Exponential moving average for per-image time
            if avg is None:
                avg = res['duration_sec']
            else:
                avg = 0.9 * avg + 0.1 * res['duration_sec']
            remaining = total - completed
            eta_sec = remaining * (avg if avg else 0.0)
            hh = int(eta_sec // 3600)
            mm = int((eta_sec % 3600) // 60)
            ss = int(eta_sec % 60)
            pct = 100.0 * completed / total if total > 0 else 100.0
            print(f"[PROGRESS] {completed}/{total} ({pct:.1f}%) | ETA {hh:02d}:{mm:02d}:{ss:02d} | last: {res['img_name']} nodes={res['nodes']} edges={res['edges']} dur={res['duration_sec']:.1f}s")


if __name__ == '__main__':
    main()
