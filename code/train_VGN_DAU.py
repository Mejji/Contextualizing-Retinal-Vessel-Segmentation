#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_VGN_DAU.py — VGN training wired to DAU2Net folder layout + precomputed graphs.

Highlights / fixes:
  • Pure TF1 graph (tf.compat.v1) with eager disabled.
  • Uses external DAU2Net prob maps (no TF CNN run) by default.
  • No illegal feeding of non‑placeholder tensors (only placeholders are fed).
  • Correct eval pipeline (uses *_e variables and the right prob source for the split).
  • Robust checkpoint discovery + optional CNN warm start from TF checkpoint.
"""

import os
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import skimage.io
import networkx as nx
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from config import cfg
from Modules.model import vessel_segm_vgn  # TF1 compat inside
import util as util


# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train VGN (TF1) using DAU2Net folder layout + precomputed graphs')

    # Dataset / split
    p.add_argument('--dataset', default='HRF', type=str,
                   choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--use_fov_mask', default=True, type=bool)
    p.add_argument('--eval_split', default='test', choices=['test', 'val', 'none'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--limit_train_images', type=int, default=0)

    # external cnn probs
    p.add_argument('--use_external_prob', action='store_true', default=True,
                   help='Use DAU2Net prob map as CNN output (Option 1).')

    # DAU2Net layout (we override cfg.PATHS.GRAPH_* to these)
    p.add_argument('--dau_root', type=str, default='C:/Users/rog/THESIS/DAU2_HRF',
                   help='Root of DAU2_DRIVE (containing train/ and test/).')
    p.add_argument('--train_probs', type=str, default=cfg.PATHS.DAU2_PROBMAP_TRAIN_DIR,
                   help='Override train probs dir (default: <dau_root>/train/probs).')
    p.add_argument('--test_probs', type=str, default=cfg.PATHS.DAU2_PROBMAP_TEST_DIR,
                   help='Override test probs dir  (default: <dau_root>/test/probs).')

    # Model
    p.add_argument('--cnn_model', default='driu', choices=['driu', 'driu_large'])
    p.add_argument('--norm_type', default='GN', choices=['GN', 'BN', 'none'])
    p.add_argument('--win_size', type=int, default=10)
    p.add_argument('--edge_geo_dist_thresh', type=int, default=80)
    p.add_argument('--graph_geo_dist', type=int, default=None,
                   help='Optional override for graph geodesic distance suffix (e.g., 40 for CHASE).')

    # GAT head (TF model expects lists)
    p.add_argument('--gat_hid_units', type=int, nargs='+', default=[16, 16])
    p.add_argument('--gat_n_heads', type=int, nargs='+', default=[4, 4, 1])
    p.add_argument('--gat_use_residual', action='store_true', default=True)

    # Loss toggles / weights
    p.add_argument('--cnn_loss_on', action='store_true', default=True)
    p.add_argument('--gnn_loss_on', action='store_true', default=True)
    p.add_argument('--gnn_loss_weight', type=float, default=1.0)
    p.add_argument('--infer_module_kernel_size', type=int, default=3)
    p.add_argument('--use_enc_layer', action='store_true', default=False)

    # Optim / LR
    p.add_argument('--opt', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--lr', type=float, default=5e-5)  # smaller LR for quick fine-tune
    p.add_argument('--max_iters', type=int, default=6000)  # shorter fine-tune run
    p.add_argument('--lr_scheduling', default='pc', choices=['pc', 'fixed', 'exp'])
    p.add_argument('--lr_decay_tp', type=float, default=0.8)
    p.add_argument('--old_net_ft_lr', type=float, default=0.0,
                   help='0.0 ⇒ freeze old CNN; train post_cnn (+gnn).')
    p.add_argument('--new_net_lr', type=float, default=1e-4)
    p.add_argument('--infer_module_grad_weight', type=float, default=1.0)
    p.add_argument('--do_simul_training', action='store_true', default=False)

    # Dropouts
    p.add_argument('--gnn_feat_dropout', type=float, default=0.1)
    p.add_argument('--gnn_att_dropout', type=float, default=0.1)
    p.add_argument('--post_cnn_dropout', type=float, default=0.1)

    # Save / runs
    p.add_argument('--save_root', default='C:/Users/rog/THESIS/DAU2_HRF/VGN', type=str)
    p.add_argument('--run_id', type=int, default=1)
    p.add_argument('--run_name', type=str, default='')

    # Resume full VGN training
    p.add_argument('--resume', default='none', choices=['none', 'latest', 'iter', 'best'])
    p.add_argument('--resume_iter', type=int, default=0)
    p.add_argument('--resume_dir', type=str, default='')

    # CNN init — TF checkpoint ONLY (NOT .pth)
    p.add_argument('--cnn_init_mode', default='auto', choices=['auto', 'best', 'iter', 'path', 'none'])
    p.add_argument('--cnn_init_iter', type=int, default=50000)
    p.add_argument('--cnn_init_dir', type=str, default=str(cfg.PATHS.DAU2_DRIVE_ROOT))
    #p.add_argument('--cnn_init_path', type=str,
     #              default='C:/Users/rog/THESIS/DAU2_DRIVE/checkpoints/best.pth',
     #              help='TF checkpoint prefix for DAU2Net export (e.g., .../best.ckpt).')

    # Eval cadence
    p.add_argument('--display', type=int, default=200)
    p.add_argument('--test_iters', type=int, default=1000)
    p.add_argument('--snapshot_iters', type=int, default=2000)

    return p.parse_args()


# ----------------------------- Helpers -----------------------------
def _get_split_txt_paths(dataset):
    if dataset == 'DRIVE':
        return cfg.TRAIN.DRIVE_SET_TXT_PATH, cfg.TEST.DRIVE_SET_TXT_PATH
    if dataset == 'STARE':
        return cfg.TRAIN.STARE_SET_TXT_PATH, cfg.TEST.STARE_SET_TXT_PATH
    if dataset == 'CHASE_DB1':
        return cfg.TRAIN.CHASE_DB1_SET_TXT_PATH, cfg.TEST.CHASE_DB1_SET_TXT_PATH
    if dataset == 'HRF':
        return cfg.TRAIN.HRF_SET_TXT_PATH, cfg.TEST.HRF_SET_TXT_PATH
    raise ValueError(f"Unknown dataset: {dataset}")


def _discover_tf_checkpoint(ckpt_dir: str, mode: str, iter_num: int = 0) -> str:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return ''
    if mode == 'latest':
        latest = tf.train.latest_checkpoint(ckpt_dir)
        return latest or ''
    if mode == 'iter':
        if iter_num <= 0:
            return ''
        prefix = os.path.join(ckpt_dir, f'iter_{iter_num}.ckpt')
        return prefix if os.path.exists(prefix + '.index') else ''
    if mode == 'best':
        prefix = os.path.join(ckpt_dir, 'best.ckpt')
        return prefix if os.path.exists(prefix + '.index') else ''
    return ''


def _discover_cnn_init(args) -> str:
    """Pick a TF CNN checkpoint to partially restore into VGN."""
    mode = args.cnn_init_mode
    if mode == 'none':
        return ''
    if mode == 'path':
        p = args.cnn_init_path
        return p if (p and os.path.exists(p + '.index')) else ''
    if mode == 'iter':
        return _discover_tf_checkpoint(args.cnn_init_dir, 'iter', args.cnn_init_iter)
    if mode == 'best':
        return _discover_tf_checkpoint(args.cnn_init_dir, 'best')
    # auto: try dir/best, then dir/iter_N
    p = _discover_tf_checkpoint(args.cnn_init_dir, 'best')
    if not p:
        p = _discover_tf_checkpoint(args.cnn_init_dir, 'iter', args.cnn_init_iter)
    return p or ''


def _parse_iter_from_ckpt_path(ckpt_path: str) -> int:
    import re
    m = re.search(r'iter_(\d+)\.ckpt', os.path.basename(ckpt_path) if ckpt_path else '')
    return int(m.group(1)) if m else 0


def _graph_to_sparse_tensor(graph: nx.Graph):
    A = nx.adjacency_matrix(graph)  # scipy sparse
    indices, values, shape = util.preprocess_graph_gat(A)
    from tensorflow.python.framework import sparse_tensor
    return sparse_tensor.SparseTensorValue(
        indices=indices.astype(np.int64),
        values=values.astype(np.float32),
        dense_shape=np.array(shape, dtype=np.int64)
    )


def _format_eta(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _restore_cnn_from_ckpt(sess, net: vessel_segm_vgn, ckpt_path: str):
    """Partial‑restore CNN layers from TF checkpoint (maps 'output/*' -> 'img_output/*')."""
    if not ckpt_path:
        print('[CNN-INIT] Skipped (no TF checkpoint path). If you only have .pth, the CNN stays frozen.')
        return False
    if ckpt_path.endswith(('.pth', '.pt')):
        print('[CNN-INIT][WARN] PyTorch checkpoint is not loadable by TF. Provide a TF .ckpt instead.')
        return False

    print(f'[CNN-INIT] Restoring CNN layers from: {ckpt_path}')
    reader = tf.train.NewCheckpointReader(ckpt_path)
    ckpt_vars = reader.get_variable_to_shape_map()

    gvars = tf.global_variables()
    wanted_scopes = list(getattr(net, 'var_to_restore', []))

    var_map = {}
    for v in gvars:
        name = v.op.name
        scope = name.split('/')[0]
        if scope not in wanted_scopes:
            continue
        ck_scope = 'output' if scope == 'img_output' else scope
        ck_name = ck_scope + '/' + name.split('/')[-1]
        if ck_name in ckpt_vars and list(ckpt_vars[ck_name]) == list(v.shape):
            var_map[ck_name] = v

    if not var_map:
        print('[CNN-INIT][WARN] No matching variables found to restore.')
        return False

    saver_part = tf.train.Saver(var_list=var_map)
    saver_part.restore(sess, ckpt_path)
    print(f'[CNN-INIT] Restored {len(var_map)} tensors into VGN CNN.')
    return True


def _maybe_override_graph_dirs(args):
    """Override cfg graph dirs to DAU2Net probs folders so util.GraphDataLayer finds *.graph_res."""
    train_probs = args.train_probs or (os.path.join(args.dau_root, 'train', 'probs') if args.dau_root else '')
    test_probs = args.test_probs or (os.path.join(args.dau_root, 'test', 'probs') if args.dau_root else '')
    if train_probs:
        os.makedirs(train_probs, exist_ok=True)
        setattr(cfg.PATHS, 'GRAPH_TRAIN_DIR', train_probs)
        print(f'[CFG] GRAPH_TRAIN_DIR -> {cfg.PATHS.GRAPH_TRAIN_DIR}')
    if test_probs:
        os.makedirs(test_probs, exist_ok=True)
        setattr(cfg.PATHS, 'GRAPH_TEST_DIR', test_probs)
        print(f'[CFG] GRAPH_TEST_DIR  -> {cfg.PATHS.GRAPH_TEST_DIR}')


def _dataset_canvas_hw(dataset: str):
    """Return (H, W) canvas dims matching util.GraphDataLayer padding."""
    try:
        _, h, w = util._dataset_specs(dataset)
        return int(h), int(w)
    except Exception:
        return 592, 592


def _pad_to_canvas(x, canvas_hw):
    Ht, Wt = int(canvas_hw[0]), int(canvas_hw[1])
    if x.ndim == 2:
        h, w = x.shape
        out_h = max(Ht, h)
        out_w = max(Wt, w)
        out = np.zeros((out_h, out_w), dtype=x.dtype)
        out[:h, :w] = x
        return out
    if x.ndim == 3:
        h, w, c = x.shape
        out_h = max(Ht, h)
        out_w = max(Wt, w)
        out = np.zeros((out_h, out_w, c), dtype=x.dtype)
        out[:h, :w, :] = x
        return out
    raise ValueError(f"pad_to_canvas: unsupported ndim {x.ndim}")


def _try_load_external_prob_padded(probs_root: Path, stem: str, canvas_hw):
    cands = [
        probs_root / f"{stem}_prob.png",
        probs_root / f"{stem}_prob.npy",
        probs_root / f"{stem}.png",
        probs_root / f"{stem}.npy",
    ]
    for p in cands:
        if p.exists():
            if p.suffix.lower() == '.npy':
                arr = np.load(str(p)).astype(np.float32)
            else:
                arr = skimage.io.imread(str(p)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.max() > 1.0:
                arr = np.clip(arr, 0, 255) / 255.0
            return arr, _pad_to_canvas(arr, canvas_hw)
    return None, None


def _resolve_prob_root(explicit_path: str, dau_root: str, split: str) -> Path:
    """Pick an existing prob/graph directory for the requested split."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
    # Fallback: prefer <split>/prob, else <split>
    candidate = Path(dau_root) / split / 'prob'
    if candidate.exists():
        return candidate
    return Path(dau_root) / split


def _filter_samples_with_graphs(img_names, prob_root: Path, win_size: int, edge_geo_dist_thresh: int):
    """Keep only samples whose graph file exists; return filtered list and missing stems."""
    keep, missing = [], []
    for p in img_names:
        stem = Path(p).stem
        gpath = prob_root / f"{stem}_{int(win_size):02d}_{int(edge_geo_dist_thresh):02d}.graph_res"
        if gpath.exists():
            keep.append(p)
        else:
            missing.append(stem)
    return keep, missing


# --------------------------- Main ---------------------------
if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)

    if args.graph_geo_dist is not None:
        args.edge_geo_dist_thresh = int(args.graph_geo_dist)

    dataset_key = args.dataset.upper()
    default_drive_root = getattr(cfg.PATHS, 'DAU2_DRIVE_ROOT', '')
    drive_train_prob = getattr(cfg.PATHS, 'DAU2_PROBMAP_TRAIN_DIR', '')
    drive_test_prob = getattr(cfg.PATHS, 'DAU2_PROBMAP_TEST_DIR', '')
    default_train_prob = str(Path(drive_train_prob)) if drive_train_prob else ''
    default_test_prob = str(Path(drive_test_prob)) if drive_test_prob else ''

    if dataset_key == 'CHASE_DB1':
        chase_root = str(Path('C:/Users/rog/THESIS/DAU2_CHASE'))
        if (not args.dau_root) or (default_drive_root and Path(args.dau_root).resolve() == Path(default_drive_root).resolve()):
            args.dau_root = chase_root
        if (not args.train_probs) or (default_train_prob and Path(args.train_probs).resolve() == Path(default_train_prob).resolve()):
            args.train_probs = str(Path(args.dau_root) / 'train')
        if (not args.test_probs) or (default_test_prob and Path(args.test_probs).resolve() == Path(default_test_prob).resolve()):
            args.test_probs = str(Path(args.dau_root) / 'test')
        if args.win_size == 4:
            args.win_size = 8
        if args.edge_geo_dist_thresh == 10:
            args.edge_geo_dist_thresh = 40

    if dataset_key == 'HRF':
        hrf_root = str(Path('C:/Users/rog/THESIS/DAU_HRF'))
        # Force HRF root unless the user explicitly set a DAU_HRF path
        bad_root = any(tok in str(args.dau_root) for tok in ['DAU2_CHASE', 'DAU2_DRIVE', 'DAU2_HRF'])
        if (not args.dau_root) or bad_root or (not Path(args.dau_root).exists()):
            args.dau_root = hrf_root
        bad_train = any(tok in str(args.train_probs) for tok in ['DAU2_CHASE', 'DAU2_DRIVE', 'DAU2_HRF'])
        bad_test = any(tok in str(args.test_probs) for tok in ['DAU2_CHASE', 'DAU2_DRIVE', 'DAU2_HRF'])
        if (not args.train_probs) or (default_train_prob and Path(args.train_probs).resolve() == Path(default_train_prob).resolve()) or bad_train:
            args.train_probs = str(Path(args.dau_root) / 'train')
        if (not args.test_probs) or (default_test_prob and Path(args.test_probs).resolve() == Path(default_test_prob).resolve()) or bad_test:
            args.test_probs = str(Path(args.dau_root) / 'test')
        # HRF precomputed graphs/probmaps are suffixed *_10_80, so align defaults.
        if args.win_size == 4:
            args.win_size = 10
        if args.edge_geo_dist_thresh == 40:
            args.edge_geo_dist_thresh = 80
        # Save/run roots: keep HRF runs under DAU_HRF/VGN when user kept the default CHASE path.
        default_save_root = str(Path('C:/Users/rog/THESIS/DAU2_CHASE/VGN'))
        if args.save_root == default_save_root:
            args.save_root = str(Path(hrf_root) / 'VGN')

    train_prob_root = _resolve_prob_root(args.train_probs, args.dau_root, 'train')
    test_prob_root = _resolve_prob_root(args.test_probs, args.dau_root, 'test')
    args.train_probs = str(train_prob_root)
    args.test_probs = str(test_prob_root)

    cfg.USE_DAU2_GRAPH_PATHS = True  # custom flag for util.GraphDataLayer

    # Wire cfg to DAU2Net graph locations
    _maybe_override_graph_dirs(args)

    # Resolve data lists
    train_txt, test_txt = _get_split_txt_paths(args.dataset)
    with open(train_txt) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_txt) as f:
        test_img_names = [x.strip() for x in f.readlines()]

    # Filter to samples that have matching precomputed graphs/probmaps
    filtered_train, missing_train = _filter_samples_with_graphs(train_img_names, train_prob_root,
                                                                args.win_size, args.edge_geo_dist_thresh)
    filtered_test, missing_test = _filter_samples_with_graphs(test_img_names, test_prob_root,
                                                              args.win_size, args.edge_geo_dist_thresh)
    if missing_train:
        print(f"[WARN] Skipping {len(missing_train)} train samples without graphs: {missing_train}")
    if missing_test:
        print(f"[WARN] Skipping {len(missing_test)} test samples without graphs: {missing_test}")
    train_img_names = filtered_train
    test_img_names = filtered_test
    if len(train_img_names) == 0:
        raise RuntimeError(f"No training samples left after graph filtering. Check graph dir: {train_prob_root}")

    if args.limit_train_images and args.limit_train_images > 0:
        train_img_names = train_img_names[:args.limit_train_images]
        print(f"[SMOKE] limit_train_images -> {len(train_img_names)}")

    # Build eval split
    if args.eval_split == 'val':
        rs = np.random.RandomState(args.seed)
        n = len(train_img_names)
        idxs = np.arange(n)
        rs.shuffle(idxs)
        n_val = max(1, int(np.ceil(0.1 * n)))
        val_idx = set(idxs[:n_val].tolist())
        val_img_names = [x for i, x in enumerate(train_img_names) if i in val_idx]
        train_img_names = [x for i, x in enumerate(train_img_names) if i not in val_idx]
        eval_img_names = val_img_names
        eval_tag = 'val'
        prob_root_eval = train_prob_root
    elif args.eval_split == 'test':
        eval_img_names = test_img_names
        eval_tag = 'test'
        prob_root_eval = test_prob_root
    else:
        eval_img_names = []
        eval_tag = 'none'
        prob_root_eval = None

    # Data layers (graph-aware; graphs loaded, not generated)
    dl_train = util.GraphDataLayer(train_img_names, is_training=True,
                                   edge_type='srns_geo_dist_binary',
                                   win_size=args.win_size,
                                   edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    dl_eval = util.GraphDataLayer(eval_img_names, is_training=False,
                                  edge_type='srns_geo_dist_binary',
                                  win_size=args.win_size,
                                  edge_geo_dist_thresh=args.edge_geo_dist_thresh) if eval_img_names else None
    canvas_hw = _dataset_canvas_hw(args.dataset)

    # Run paths
    run_name = args.run_name if args.run_name else f"VGN_DAU_run{int(args.run_id)}"
    run_root = os.path.join(args.save_root, run_name) if args.save_root else run_name
    os.makedirs(run_root, exist_ok=True)
    model_save_path = os.path.join(run_root, 'train')
    os.makedirs(model_save_path, exist_ok=True)
    probmap_save_dir = os.path.join(model_save_path, 'probmaps')
    os.makedirs(probmap_save_dir, exist_ok=True)
    eval_save_dir = os.path.join(run_root, (cfg.TEST.RES_SAVE_PATH if eval_tag == 'test' else 'val'))
    if eval_tag != 'none':
        os.makedirs(eval_save_dir, exist_ok=True)

    # Build network (TF1 compat is handled in model.py)
    class _P:
        pass
    _p = _P()
    for k, v in vars(args).items():
        setattr(_p, k, v)
    _p.gat_hid_units = list(args.gat_hid_units)
    _p.gat_n_heads = list(args.gat_n_heads)
    _p.gat_use_residual = bool(args.gat_use_residual)
    _p.norm_type = None if str(args.norm_type).lower() == 'none' else args.norm_type

    net = vessel_segm_vgn(_p, None)

    # Session / IO
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver_all = tf.train.Saver(max_to_keep=100)
    summary_writer = tf.summary.FileWriter(model_save_path, sess.graph)

    # Profiler CSV
    profiler_csv_path = os.path.join(model_save_path, 'profiler.csv')
    if not os.path.exists(profiler_csv_path):
        with open(profiler_csv_path, 'w') as pf:
            pf.write('iter,step_time_s,avg_step_time_s,elapsed_s,loss,cnn_loss,gnn_loss,post_loss,acc,dice,auc\n')

    # Init
    sess.run(tf.global_variables_initializer())

    # Resume VGN if requested
    start_iter = 0
    did_resume = False
    if args.resume != 'none':
        ckpt_dir = args.resume_dir if args.resume_dir else model_save_path
        ckpt_path = _discover_tf_checkpoint(ckpt_dir, args.resume, args.resume_iter)
        if ckpt_path:
            try:
                saver_all.restore(sess, ckpt_path)
                parsed_iter = _parse_iter_from_ckpt_path(ckpt_path)
                start_iter = parsed_iter if parsed_iter > 0 else 0
                did_resume = True
                print(f'[RESUME] Restored VGN from {ckpt_path}. start_iter={start_iter}')
            except Exception as e:
                print(f'[RESUME] Failed to restore full VGN: {e}')

    # If not resuming full VGN, initialize CNN part from external TF checkpoint
    if not did_resume:
        cnn_ckpt = _discover_cnn_init(args)
        _restore_cnn_from_ckpt(sess, net, cnn_ckpt)

    # Training loop
    print("Training VGN...")
    best_metric = -1.0  # Dice
    train_wall_start = time.time()
    timer = util.Timer()
    train_loss_hist, train_cnn_loss_hist, train_gnn_loss_hist, train_post_loss_hist = [], [], [], []
    train_cnn_acc_hist, train_gnn_acc_hist, train_post_acc_hist = [], [], []

    for it in range(start_iter, args.max_iters):
        timer.tic()

        img_list, blobs = dl_train.forward()
        imgs = np.asarray(blobs['img'], dtype=np.float32)
        labels = np.asarray(blobs['label'], dtype=np.int64)
        fovs = np.asarray(blobs['fov'], dtype=np.int64) if args.use_fov_mask else np.ones_like(labels, dtype=np.int64)
        graph = blobs['graph']
        nlist = blobs['num_of_nodes_list']
        vec_aug_on = blobs.get('vec_aug_on', np.zeros((7,), dtype=bool))
        rot_angle = int(blobs.get('rot_angle', 0))

        node_byxs = util.get_node_byx_from_graph(graph, nlist).astype(np.int32)
        # Graph sparse tensor from NetworkX adjacency
        A = nx.adjacency_matrix(graph, weight=None).astype(float)
        indices, values, shape = util.preprocess_graph_gat(A)
        from tensorflow.python.framework import sparse_tensor
        adj_sp = sparse_tensor.SparseTensorValue(
            indices=indices.astype(np.int64),
            values=values.astype(np.float32),
            dense_shape=np.array(shape, dtype=np.int64)
        )

        pixel_weights = fovs.astype(np.float32)
        lr_flip = bool(vec_aug_on[0])
        ud_flip = bool(vec_aug_on[1])
        rot90_num = float((rot_angle // 90) % 4)

        # === EXTERNAL PROB BATCH (Option 1) ===
        ext_batch = []
        proot = train_prob_root
        for pth in img_list:
            stem = Path(pth).stem
            _, ext_canvas = _try_load_external_prob_padded(proot, stem, canvas_hw)
            if ext_canvas is None:
                raise FileNotFoundError(f"[DAU2 prob missing] {stem} under {proot}")
            ext_batch.append(ext_canvas.astype(np.float32))
        ext_batch = np.stack(ext_batch, axis=0)[..., np.newaxis]  # [B, H, W, 1]

        # apply SAME augmentation as the batch
        if lr_flip:
            ext_batch = ext_batch[:, :, ::-1, :]
        if ud_flip:
            ext_batch = ext_batch[:, ::-1, :, :]
        if int(rot90_num) % 4:
            k = int(rot90_num) % 4
            ext_batch = np.array(
                [np.rot90(ext_batch[i, ..., 0], k).copy()[..., None] for i in range(ext_batch.shape[0])],
                dtype=np.float32
            )

        feed = {
            net.imgs: imgs,
            net.labels: labels,
            net.fov_masks: fovs,
            net.node_byxs: node_byxs,
            net.adj: adj_sp,
            net.pixel_weights: pixel_weights,
            net.is_lr_flipped: lr_flip,
            net.is_ud_flipped: ud_flip,
            net.rot90_num: rot90_num,
            net.gnn_feat_dropout: float(args.gnn_feat_dropout),
            net.gnn_att_dropout: float(args.gnn_att_dropout),
            net.post_cnn_dropout: float(args.post_cnn_dropout),
            net.lr_ph: float(args.new_net_lr if args.old_net_ft_lr == 0.0 else args.old_net_ft_lr),
            # external prob
            net.external_cnn_prob: ext_batch,
        }

        outs = sess.run(
            [net.train_op, net.loss, net.cnn_loss, net.gnn_loss, net.post_cnn_loss,
             net.cnn_accuracy, net.gnn_accuracy, net.post_cnn_accuracy,
             net.post_cnn_img_fg_prob],
            feed_dict=feed
        )
        _, loss_total, loss_cnn, loss_gnn, loss_post, acc_cnn, acc_gnn, acc_post, prob_map = outs
        timer.toc()
        train_loss_hist.append(loss_total)
        train_cnn_loss_hist.append(loss_cnn)
        train_gnn_loss_hist.append(loss_gnn)
        train_post_loss_hist.append(loss_post)
        train_cnn_acc_hist.append(acc_cnn)
        train_gnn_acc_hist.append(acc_gnn)
        train_post_acc_hist.append(acc_post)

        # Save current probmaps (qual)
        for b_idx, img_path in enumerate(img_list):
            base = os.path.basename(str(img_path))
            arr = prob_map[b_idx, ..., 0] if prob_map.ndim == 4 else prob_map[b_idx]
            arr = np.squeeze(arr)
            np.save(os.path.join(probmap_save_dir, base + '_prob.npy'), arr.astype(np.float32))
            try:
                skimage.io.imsave(
                    os.path.join(probmap_save_dir, base + '_prob.png'),
                    (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                )
            except Exception:
                pass

        # Logging
        if (it + 1) % args.display == 0:
            avg_step = timer.average_time if timer.average_time > 1e-9 else timer.diff
            eta_seconds = (args.max_iters - (it + 1)) * max(avg_step, 0.0)
            eta_str = _format_eta(eta_seconds)
            print(f"iter: {it+1}/{args.max_iters}, "
                  f"loss: {loss_total:.4f} (cnn {loss_cnn:.4f}, gnn {loss_gnn:.4f}, post {loss_post:.4f}), "
                  f"acc(cnn/gnn/post): {acc_cnn:.4f}/{acc_gnn:.4f}/{acc_post:.4f}, "
                  f"step: {avg_step:.3f}s, eta: {eta_str}")
        try:
            elapsed_s = time.time() - train_wall_start
            with open(profiler_csv_path, 'a') as pf:
                pf.write(f"{it+1},{timer.diff:.6f},{timer.average_time:.6f},{elapsed_s:.2f},"
                         f"{loss_total:.6f},{loss_cnn:.6f},{loss_gnn:.6f},{loss_post:.6f},"
                         f"{acc_post:.6f},nan,nan\n")
        except Exception:
            pass

        # Snapshot
        if (it + 1) % args.snapshot_iters == 0:
            ck = os.path.join(model_save_path, f'iter_{it+1}.ckpt')
            saver_all.save(sess, ck)
            print(f'[CKPT] Wrote snapshot: {ck}')

        # Periodic eval
        if (eval_tag != 'none') and ((it + 1) % args.test_iters == 0):
            all_labels = np.zeros((0,))
            all_preds_post = np.zeros((0,))
            all_preds_cnn = np.zeros((0,))
            all_node_labels = np.zeros((0,))
            all_node_probs = np.zeros((0,))
            eval_losses, eval_cnn_losses, eval_gnn_losses, eval_post_losses = [], [], [], []
            eval_cnn_accs, eval_gnn_accs, eval_post_accs = [], [], []

            num_batches = int(np.ceil(float(len(eval_img_names)) / cfg.TRAIN.BATCH_SIZE))
            for _ in range(num_batches):
                img_list_e, blobs_e = dl_eval.forward()
                if len(img_list_e) == 0:
                    continue
                imgs_e = np.asarray(blobs_e['img'], dtype=np.float32)
                labels_e = np.asarray(blobs_e['label'], dtype=np.int64)
                fovs_e = np.asarray(blobs_e['fov'], dtype=np.int64) if args.use_fov_mask else np.ones_like(labels_e, dtype=np.int64)
                graph_e = blobs_e['graph']
                nlist_e = blobs_e['num_of_nodes_list']
                vec_aug_e = blobs_e.get('vec_aug_on', np.zeros((7,), dtype=bool))
                rot_ang_e = int(blobs_e.get('rot_angle', 0))

                node_byxs_e = util.get_node_byx_from_graph(graph_e, nlist_e).astype(np.int32)
                A_e = nx.adjacency_matrix(graph_e, weight=None).astype(float)
                idx, val, shp = util.preprocess_graph_gat(A_e)
                from tensorflow.python.framework import sparse_tensor
                adj_sp_e = sparse_tensor.SparseTensorValue(
                    indices=idx.astype(np.int64),
                    values=val.astype(np.float32),
                    dense_shape=np.array(shp, dtype=np.int64)
                )
                lr_flip_e = bool(vec_aug_e[0])
                ud_flip_e = bool(vec_aug_e[1])
                rot90_num_e = float((rot_ang_e // 90) % 4)

                # EXTERNAL PROB for EVAL (use the split‑appropriate prob root)
                ext_batch_e = []
                proot_e = prob_root_eval
                for pth in img_list_e:
                    stem = Path(pth).stem
                    _, ext_canvas_e = _try_load_external_prob_padded(proot_e, stem, canvas_hw)
                    if ext_canvas_e is None:
                        raise FileNotFoundError(f"[DAU2 prob missing] {stem} under {proot_e}")
                    ext_batch_e.append(ext_canvas_e.astype(np.float32))
                ext_batch_e = np.stack(ext_batch_e, axis=0)[..., np.newaxis]

                # apply SAME augmentation as the batch
                if lr_flip_e:
                    ext_batch_e = ext_batch_e[:, :, ::-1, :]
                if ud_flip_e:
                    ext_batch_e = ext_batch_e[:, ::-1, :, :]
                if int(rot90_num_e) % 4:
                    k = int(rot90_num_e) % 4
                    ext_batch_e = np.array(
                        [np.rot90(ext_batch_e[i, ..., 0], k).copy()[..., None] for i in range(ext_batch_e.shape[0])],
                        dtype=np.float32
                    )

                feed_e = {
                    net.imgs: imgs_e,
                    net.labels: labels_e,
                    net.fov_masks: fovs_e,
                    net.node_byxs: node_byxs_e,
                    net.adj: adj_sp_e,
                    net.pixel_weights: fovs_e.astype(np.float32),
                    net.is_lr_flipped: lr_flip_e,
                    net.is_ud_flipped: ud_flip_e,
                    net.rot90_num: rot90_num_e,
                    net.gnn_feat_dropout: 0.0,
                    net.gnn_att_dropout: 0.0,
                    net.post_cnn_dropout: 0.0,
                    net.external_cnn_prob: ext_batch_e,
                }

                (loss_e, cnn_loss_e, gnn_loss_e, post_loss_e,
                 cnn_acc_e, gnn_acc_e, post_acc_e,
                 prob_map_e, cnn_prob_map_e,
                 gnn_prob_e, node_labels_e) = sess.run(
                    [net.loss, net.cnn_loss, net.gnn_loss, net.post_cnn_loss,
                     net.cnn_accuracy, net.gnn_accuracy, net.post_cnn_accuracy,
                     net.post_cnn_img_fg_prob, net.img_fg_prob,
                     net.gnn_prob, net.node_labels],
                    feed_dict=feed_e
                )
                eval_losses.append(loss_e)
                eval_cnn_losses.append(cnn_loss_e)
                eval_gnn_losses.append(gnn_loss_e)
                eval_post_losses.append(post_loss_e)
                eval_cnn_accs.append(cnn_acc_e)
                eval_gnn_accs.append(gnn_acc_e)
                eval_post_accs.append(post_acc_e)
                if node_labels_e.size > 0:
                    all_node_labels = np.concatenate((all_node_labels, node_labels_e.reshape(-1)))
                    all_node_probs = np.concatenate((all_node_probs, gnn_prob_e.reshape(-1)))

                # flatten metrics inside FOV only (avoid padding/background skew)
                mask_e = fovs_e.astype(bool)
                all_labels = np.concatenate((all_labels, labels_e[mask_e].reshape(-1)))
                all_preds_post = np.concatenate((all_preds_post, prob_map_e[mask_e].reshape(-1)))
                all_preds_cnn = np.concatenate((all_preds_cnn, cnn_prob_map_e[mask_e].reshape(-1)))

                # qualitative save
                for i_path, pm in zip(img_list_e, prob_map_e):
                    stem = os.path.splitext(os.path.basename(str(i_path)))[0]
                    try:
                        skimage.io.imsave(os.path.join(eval_save_dir, f"{stem}_post_prob.png"),
                                          (np.clip(pm[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8))
                    except Exception:
                        pass

            # metrics
            def _metr(labels, preds, thr=0.5):
                labels = labels.astype(np.int64).ravel()
                preds = preds.astype(np.float32).ravel()
                predb = preds >= thr
                lblb = labels.astype(bool)
                tp = np.sum(np.logical_and(predb, lblb))
                tn = np.sum(np.logical_and(~predb, ~lblb))
                fp = np.sum(np.logical_and(predb, ~lblb))
                fn = np.sum(np.logical_and(~predb, lblb))
                denom = lambda x: x if x > 0 else 1
                acc = (tp + tn) / denom(tp + tn + fp + fn)
                dice = (2 * tp) / denom(2 * tp + fp + fn)
                iou = tp / denom(tp + fp + fn)
                try:
                    auc, ap = util.get_auc_ap_score(labels.astype(np.float32), preds.astype(np.float32))
                except Exception:
                    auc, ap = np.nan, np.nan
                return dict(accuracy=acc, dice=dice, iou=iou, auc=auc, ap=ap)

            post_metrics = _metr(all_labels, all_preds_post, 0.5)
            cnn_metrics = _metr(all_labels, all_preds_cnn, 0.5)

            # TensorBoard
            def _mean_or_zero(seq):
                return float(np.mean(seq)) if seq else 0.0

            cur_lr = sess.run(net.lr_handler)
            prefix = eval_tag if eval_tag else 'eval'
            summary = tf.Summary()
            summary.value.add(tag="train_loss", simple_value=_mean_or_zero(train_loss_hist))
            summary.value.add(tag="train_cnn_loss", simple_value=_mean_or_zero(train_cnn_loss_hist))
            summary.value.add(tag="train_gnn_loss", simple_value=_mean_or_zero(train_gnn_loss_hist))
            summary.value.add(tag="train_infer_module_loss", simple_value=_mean_or_zero(train_post_loss_hist))
            summary.value.add(tag="train_cnn_acc", simple_value=_mean_or_zero(train_cnn_acc_hist))
            summary.value.add(tag="train_gnn_acc", simple_value=_mean_or_zero(train_gnn_acc_hist))
            summary.value.add(tag="train_infer_module_acc", simple_value=_mean_or_zero(train_post_acc_hist))
            summary.value.add(tag=f"{prefix}_loss", simple_value=_mean_or_zero(eval_losses))
            summary.value.add(tag=f"{prefix}_cnn_loss", simple_value=_mean_or_zero(eval_cnn_losses))
            summary.value.add(tag=f"{prefix}_gnn_loss", simple_value=_mean_or_zero(eval_gnn_losses))
            summary.value.add(tag=f"{prefix}_infer_module_loss", simple_value=_mean_or_zero(eval_post_losses))
            summary.value.add(tag=f"{prefix}_cnn_acc", simple_value=cnn_metrics['accuracy'])
            summary.value.add(tag=f"{prefix}_cnn_auc", simple_value=cnn_metrics['auc'])
            summary.value.add(tag=f"{prefix}_cnn_ap", simple_value=cnn_metrics['ap'])
            summary.value.add(tag=f"{prefix}_infer_module_acc", simple_value=post_metrics['accuracy'])
            summary.value.add(tag=f"{prefix}_infer_module_auc", simple_value=post_metrics['auc'])
            summary.value.add(tag=f"{prefix}_infer_module_ap", simple_value=post_metrics['ap'])
            summary.value.add(tag="lr", simple_value=float(cur_lr))
            summary_writer.add_summary(summary, global_step=it + 1)
            summary_writer.flush()
            train_loss_hist.clear()
            train_cnn_loss_hist.clear()
            train_gnn_loss_hist.clear()
            train_post_loss_hist.clear()
            train_cnn_acc_hist.clear()
            train_gnn_acc_hist.clear()
            train_post_acc_hist.clear()

            print(f"{eval_tag.upper()}: iter {it+1}/{args.max_iters}, "
                  f"post-acc: {post_metrics['accuracy']:.4f}, post-auc: {post_metrics['auc']:.4f}, "
                  f"post-ap: {post_metrics['ap']:.4f}, post-dice: {post_metrics['dice']:.4f}, "
                  f"post-iou: {post_metrics['iou']:.4f}")

            # Save Dice-best
            if post_metrics['dice'] > best_metric:
                best_metric = float(post_metrics['dice'])
                best_prefix = os.path.join(model_save_path, 'best.ckpt')
                saver_all.save(sess, best_prefix)
                print(f'[CKPT] Saved new best: {best_prefix} (dice={best_metric:.4f})')
                try:
                    best_meta = dict(run=run_name, type='iteration', split=eval_tag,
                                     best_iter=int(it + 1), metric='dice',
                                     metric_value=best_metric,
                                     timestamp=datetime.now().isoformat())
                    with open(os.path.join(model_save_path, 'best_meta.json'), 'w') as fmeta:
                        json.dump(best_meta, fmeta, indent=2)
                except Exception:
                    pass

    # Final snapshot
    final_path = os.path.join(model_save_path, f'iter_{args.max_iters}.ckpt')
    saver_all.save(sess, final_path)
    print(f'[CKPT] Wrote final snapshot: {final_path}')

    total_elapsed = time.time() - train_wall_start
    print('Training completed in {:.2f} seconds ({}).'.format(
        total_elapsed, str(timedelta(seconds=int(total_elapsed)))))
    sess.close()
    print("VGN training complete.")
