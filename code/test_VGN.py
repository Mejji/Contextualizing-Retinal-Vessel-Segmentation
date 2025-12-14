#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_VGN.py — TensorFlow 1 (tf.compat.v1) test script for vessel_segm_vgn

Wired to train_VGN.py run layout:
  • Auto-discovers checkpoints under <save_root>/<run_name or run{run_id}>/train
    (prefers best.ckpt, else latest iter_*.ckpt)
  • Full-restore first; on failure, partial-restore overlapping vars
    (remaps output/* -> img_output/*)

Hardened for:
  • Full TF1 checkpoints: directory, prefix, or any of .meta/.index/.data-00000-of-00001
  • Modern NetworkX (2.x/3.x) API (no .node)
  • skimage >= 0.19 dtype rules for imsave

USAGE EXAMPLES

# Auto-discover from the training run folder you used in train_VGN.py
python test_VGN.py --dataset DRIVE --save_root <same_as_train> --run_id 1

# Or point directly to a dir/prefix/any file from the triplet:
python test_VGN.py --dataset DRIVE --model_path C:/path/to/run1/train/

# If you changed graph params during training, pass the same here:
python test_VGN.py --dataset DRIVE --win_size 4 --edge_geo_dist_thresh 10
"""

from __future__ import print_function

import os
import glob
import argparse
import pickle as pkl
import multiprocessing
import warnings
import json
from pathlib import Path
import numpy as np
import skimage.io
import networkx as nx
import skfmm

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from config import cfg
from Modules.model import vessel_segm_vgn  # must match your training graph
import util


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _warn_filter():
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


def resolve_checkpoint_path(model_path):
    """
    Accepts:
      - a directory containing TF 'checkpoint' file or *.ckpt*.index files
      - a checkpoint prefix (e.g., '.../iter_50000.ckpt')
      - a concrete file among the triplet (.meta | .index | .data-00000-of-00001)

    Returns the checkpoint *prefix* suitable for tf.train.Saver.restore(...).
    """
    p = os.path.expanduser(model_path)

    # (1) Directory -> try tf.latest_checkpoint, then newest *.index
    if os.path.isdir(p):
        ckpt = tf.train.latest_checkpoint(p)
        if ckpt is not None:
            return ckpt
        idxs = sorted(glob.glob(os.path.join(p, "*.ckpt*.index")),
                      key=os.path.getmtime, reverse=True)
        if idxs:
            return os.path.splitext(idxs[0])[0]
        raise FileNotFoundError("No checkpoint found in directory: {}".format(p))

    # (2) If it's a file path, normalize to prefix
    if os.path.isfile(p):
        base, ext = os.path.splitext(p)
        # Handle .data-00000-of-00001
        if ".data-" in p and p.endswith(".of-00001"):
            return p[: p.rfind(".data-")]
        # Handle .index/.meta
        if ext in (".index", ".meta"):
            return base

    # (3) Treat given path as a prefix and verify existence
    if os.path.exists(p + ".index") or os.path.exists(p + ".meta"):
        return p

    raise FileNotFoundError(
        "Checkpoint not found. Tried directory/latest, file triplet, or prefix around: {}".format(model_path)
    )


def discover_from_train_run(save_root, run_name, prefer="best"):
    """
    Reproduce train_VGN run layout:
      run_root = <save_root>/<run_name>
      ckpt_dir = run_root/train
      prefer best.ckpt, else latest iter_*.ckpt
    """
    run_root = os.path.join(save_root, run_name) if save_root else run_name
    ckpt_dir = os.path.join(run_root, "train")
    if not os.path.isdir(ckpt_dir):
        # allow pointing directly at "train" to be explicit
        if os.path.isdir(run_root) and os.path.basename(run_root) == "train":
            ckpt_dir = run_root
        else:
            raise FileNotFoundError(f"No train directory at: {ckpt_dir}")

    if prefer == "best":
        best = os.path.join(ckpt_dir, "best.ckpt")
        if os.path.exists(best + ".index"):
            return best

    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        return latest

    # last resort: newest *.index
    idxs = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt*.index")),
                  key=os.path.getmtime, reverse=True)
    if idxs:
        return os.path.splitext(idxs[0])[0]

    raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")


def smart_restore(sess, net, ckpt_prefix):
    """
    Try full restore first. If NOT_FOUND, do partial restore:
      - collect overlap by exact var name (without ':0') and matching shape
      - special-case mapping: output/* in ckpt -> img_output/* in current graph
    """
    skip_tokens = ('Adam', 'Adadelta', 'Adagrad', 'Momentum', 'RMSProp',
                   'beta1_power', 'beta2_power', 'power')
    def _is_optimizer_var(name):
        return any(tok in name for tok in skip_tokens)

    all_vars = [v for v in tf.compat.v1.global_variables() if not _is_optimizer_var(v.op.name)]
    saver = tf.compat.v1.train.Saver(var_list=all_vars)
    try:
        saver.restore(sess, ckpt_prefix)
        print("[RESTORE] Full restore succeeded.")
        return "full"
    except Exception as e:
        print(f"[RESTORE] Full restore failed ({type(e).__name__}): {e}")
        print("[RESTORE] Attempting partial restore of overlapping variables...")

    reader = tf.compat.v1.train.NewCheckpointReader(ckpt_prefix)
    ck_vars = reader.get_variable_to_shape_map()

    # Build name->var map for current graph
    name_to_var = {v.op.name: v for v in all_vars}

    # Build mapping var->ckpt_name
    var_map = {}
    for name, var in name_to_var.items():
        # direct match
        if name in ck_vars and list(ck_vars[name]) == list(var.shape):
            var_map[name] = var
            continue
        # output/* (old CNN) -> img_output/* (new VGN)
        if name.startswith("img_output/"):
            alt = "output/" + name.split("/", 1)[1]
            if alt in ck_vars and list(ck_vars[alt]) == list(var.shape):
                var_map[alt] = var
                continue

    if not var_map:
        raise RuntimeError("[RESTORE] Partial restore found zero matching tensors. Wrong checkpoint.")

    saver_part = tf.compat.v1.train.Saver(var_list=var_map)
    saver_part.restore(sess, ckpt_prefix)
    print(f"[RESTORE] Partial restore loaded {len(var_map)} tensors.")
    return "partial"


def _compute_binary_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(np.int64).ravel()
    probs = np.asarray(probs).astype(np.float32).ravel()
    preds = probs >= float(threshold)
    lbl_bool = labels.astype(bool)
    tp = np.sum(np.logical_and(preds, lbl_bool))
    tn = np.sum(np.logical_and(~preds, ~lbl_bool))
    fp = np.sum(np.logical_and(preds, ~lbl_bool))
    fn = np.sum(np.logical_and(~preds, lbl_bool))
    denom = lambda x: x if x > 0 else 1
    precision = tp / denom(tp + fp)
    recall = tp / denom(tp + fn)
    specificity = tn / denom(tn + fp)
    f1 = (2 * precision * recall) / denom(precision + recall)
    dice = (2 * tp) / denom(2 * tp + fp + fn)
    iou = tp / denom(tp + fp + fn)
    accuracy = (tp + tn) / denom(tp + tn + fp + fn)
    return dict(precision=precision, recall=recall, specificity=specificity,
                f1=f1, dice=dice, iou=iou, accuracy=accuracy)


def _graph_to_sparse_tensor(A):
    """Match train_VGN: util.preprocess_graph_gat(A) -> SparseTensorValue."""
    indices, values, shape = util.preprocess_graph_gat(A)
    from tensorflow.python.framework import sparse_tensor
    return sparse_tensor.SparseTensorValue(
        indices=indices.astype(np.int64),
        values=values.astype(np.float32),
        dense_shape=np.array(shape, dtype=np.int64)
    )


# ----------------------------------------------------------------------
# Argument parsing (aligned with train_VGN defaults)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Test Vessel Graph Network (TF1) — wired to train_VGN runs')

    # Dataset / split
    parser.add_argument('--dataset', default=cfg.DEFAULT_DATASET, type=str,
                        choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])

    # Graph params (must match your train params)
    parser.add_argument('--win_size', type=int, default=4)
    parser.add_argument('--edge_type', default='srns_geo_dist_binary',
                        choices=['srns_geo_dist_binary', 'srns_geo_dist_weighted'])
    parser.add_argument('--edge_geo_dist_thresh', type=float, default=10)

    # Model params (match train)
    parser.add_argument('--cnn_model', default='driu', choices=['driu', 'driu_large'])
    parser.add_argument('--norm_type', default='GN', choices=['GN', 'BN', 'none', None])
    parser.add_argument('--cnn_loss_on', action='store_true', default=True)
    parser.add_argument('--gnn_loss_on', action='store_true', default=True)
    parser.add_argument('--gnn_loss_weight', type=float, default=1.0)
    parser.add_argument('--gat_hid_units', type=int, nargs='+', default=[16, 16])
    parser.add_argument('--gat_n_heads', type=int, nargs='+', default=[4, 4, 1])
    parser.add_argument('--gat_use_residual', action='store_true', default=True)
    parser.add_argument('--infer_module_kernel_size', type=int, default=3)

    # Dropouts are forced to 0 in test, but keep flags for completeness
    parser.add_argument('--gnn_feat_dropout', type=float, default=0.0)
    parser.add_argument('--gnn_att_dropout', type=float, default=0.0)
    parser.add_argument('--post_cnn_dropout', type=float, default=0.0)

    # Restore options
    parser.add_argument('--model_path', default='C:/Users/rog/THESIS/DRIU_DRIVE/VGN/run1/train/best.ckpt',
                        help=('Checkpoint: directory, prefix, or any of (.meta/.index/.data-00000-of-00001). '
                              'Example: C:/.../iter_50000.ckpt OR C:/.../run1/train/'), type=str)

    # Train-run wiring (preferred)
    parser.add_argument('--save_root', default=cfg.VGN_SAVE_DIR, type=str,
                        help='Same save_root you used in training (contains run folders).')
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='',
                        help='If empty, uses run{run_id}')
    parser.add_argument('--prefer_ckpt', type=str, default='best', choices=['best', 'latest'],
                        help='Which checkpoint to pick from the run dir.')

    # I/O
    parser.add_argument('--results_root',
                        default=os.path.join(cfg.VGN_SAVE_DIR, 'test_results'),
                        help='Root path to save test results', type=str)

    # Multiprocessing for SRNS graphifying
    parser.add_argument('--use_multiprocessing', default=True, type=bool)
    parser.add_argument('--multiprocessing_num_proc', default=8, type=int)

    # Legacy knobs (declared for API parity, not used for testing)
    parser.add_argument('--do_simul_training', default=True, type=bool)
    parser.add_argument('--max_iters', default=50000, type=int)
    parser.add_argument('--old_net_ft_lr', default=0., type=float)
    parser.add_argument('--new_net_lr', default=1e-02, type=float)
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--lr_scheduling', default='pc', type=str)
    parser.add_argument('--lr_decay_tp', default=1., type=float)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Graph builder (Python 3 / NetworkX 2+ safe)
# ----------------------------------------------------------------------
def make_graph_using_srns(arg_tuple):
    """
    arg_tuple: (fg_prob_map, edge_type, win_size, edge_geo_dist_thresh, graph_path)
    """
    (fg_prob_map, edge_type, win_size, edge_geo_dist_thresh, graph_path) = arg_tuple

    graph_path = Path(graph_path)
    if graph_path.exists():
        return

    if 'srns' not in edge_type:
        raise NotImplementedError("Only srns-based edge types are supported here.")

    vesselness = fg_prob_map
    im_y, im_x = vesselness.shape[:2]

    y_quan = sorted(set(list(range(0, im_y, win_size)) + [im_y]))
    x_quan = sorted(set(list(range(0, im_x, win_size)) + [im_x]))

    max_pos = []
    for y_idx in range(len(y_quan) - 1):
        for x_idx in range(len(x_quan) - 1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx + 1],
                                   x_quan[x_idx]:x_quan[x_idx + 1]]
            if np.sum(cur_patch) == 0:
                max_pos.append((
                    y_quan[y_idx] + cur_patch.shape[0] // 2,
                    x_quan[x_idx] + cur_patch.shape[1] // 2
                ))
            else:
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((
                    y_quan[y_idx] + int(temp[0]),
                    x_quan[x_idx] + int(temp[1])
                ))

    G = nx.Graph()

    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        G.add_node(node_idx, kind='MP', y=int(node_y), x=int(node_x), label=int(node_idx))
        print('node label', node_idx, 'pos', (node_y, node_x), 'added')

    speed = vesselness
    node_list = list(G.nodes)

    for i, n in enumerate(node_list):
        phi = np.ones_like(speed, dtype=np.float32)
        y_n = G.nodes[n]['y']; x_n = G.nodes[n]['x']
        phi[y_n, x_n] = -1
        if speed[y_n, x_n] == 0:
            continue

        neighbor = speed[max(0, y_n - 1):min(im_y, y_n + 2),
                         max(0, x_n - 1):min(im_x, x_n + 2)]
        if float(np.mean(neighbor)) < 0.1:
            continue

        tt = skfmm.travel_time(phi, speed, narrow=edge_geo_dist_thresh)

        for n_comp in node_list[i + 1:]:
            y_c = G.nodes[n_comp]['y']; x_c = G.nodes[n_comp]['x']
            geo_dist = tt[y_c, x_c]
            if geo_dist < edge_geo_dist_thresh:
                w = edge_geo_dist_thresh / (edge_geo_dist_thresh + geo_dist)
                G.add_edge(n, n_comp, weight=float(w))
                print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')

    # save graph
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(G, str(graph_path), protocol=pkl.HIGHEST_PROTOCOL)
    G.clear()
    print('generated a graph for ' + img_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':

    _warn_filter()
    args = parse_args()

    # Normalize norm_type
    if isinstance(args.norm_type, str) and args.norm_type.lower() == 'none':
        args.norm_type = None

    print('Called with args:')
    print(args)

    # dataset-specific list files
    if args.dataset == 'DRIVE':
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
    elif args.dataset == 'STARE':
        test_set_txt_path = cfg.TEST.STARE_SET_TXT_PATH
    elif args.dataset == 'CHASE_DB1':
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif args.dataset == 'HRF':
        test_set_txt_path = cfg.TEST.HRF_SET_TXT_PATH
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    # multiprocessing
    pool = None
    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=int(args.multiprocessing_num_proc))

    # results directory
    run_name = args.run_name if args.run_name else f"run{int(args.run_id)}"
    res_save_path = os.path.join(args.results_root, run_name, args.dataset)
    os.makedirs(res_save_path, exist_ok=True)

    # test list
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    len_test = len(test_img_names)

    # data layer (uses your util.py; enforces (H,W,1) masks and padding)
    data_layer_test = util.DataLayer(
        test_img_names,
        is_training=False,
        use_padding=True
    )

    # Build graph
    # model expects lists for GAT params
    args.gat_hid_units = list(args.gat_hid_units)
    args.gat_n_heads = list(args.gat_n_heads)
    network = vessel_segm_vgn(args, None)

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # Init first (so partial restore can skip missing)
    sess.run(tf.global_variables_initializer())

    # --- Resolve checkpoint (prefer train run wiring) ---
    ckpt_prefix = None
    if args.model_path:
        ckpt_prefix = resolve_checkpoint_path(args.model_path)
    else:
        try:
            ckpt_prefix = discover_from_train_run(args.save_root, run_name, prefer=args.prefer_ckpt)
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve checkpoint from run (save_root={args.save_root}, run_name={run_name}). "
                f"Either pass --model_path or fix your paths. Original: {e}"
            )

    print("Restoring from checkpoint prefix:", ckpt_prefix)
    restore_mode = smart_restore(sess, network, ckpt_prefix)
    print(f"Checkpoint restored ({restore_mode}).")

    # Log file
    os.makedirs(res_save_path, exist_ok=True)
    f_log = open(os.path.join(res_save_path, 'log.txt'), 'w')
    f_log.write(json.dumps(vars(args), indent=2) + '\n'); f_log.flush()

    print("Testing the model...")

    # === make CNN results ===
    res_list = []
    batch_size = getattr(cfg.TRAIN, 'GRAPH_BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
    num_batches = int(np.ceil(float(len_test) / batch_size))
    for _ in range(num_batches):

        # get one batch
        img_list, blobs_test = data_layer_test.forward()

        img = blobs_test['img']
        label = blobs_test['label']
        fov = blobs_test['fov']

        (conv_feats,
         fg_prob_tensor,
         cnn_feat_dict,
         cnn_feat_spatial_sizes_dict) = sess.run(
            [
                network.conv_feats,
                network.img_fg_prob,
                network.cnn_feat,
                network.cnn_feat_spatial_sizes
            ],
            feed_dict={
                network.imgs: img,
                network.labels: label
            })

        cur_batch_size = len(img_list)
        for img_idx in range(cur_batch_size):
            cur_res = {}
            cur_res['img_path'] = img_list[img_idx]
            cur_res['img'] = img[[img_idx], :, :, :]
            cur_res['label'] = label[[img_idx], :, :, :]
            cur_res['conv_feats'] = conv_feats[[img_idx], :, :, :]
            cur_res['cnn_fg_prob_map'] = fg_prob_tensor[img_idx, :, :, 0]
            cur_res['cnn_feat'] = {k: v[[img_idx], :, :, :] for k, v in cnn_feat_dict.items()}
            cur_res['cnn_feat_spatial_sizes'] = cnn_feat_spatial_sizes_dict
            cur_res['graph'] = None
            cur_res['final_fg_prob_map'] = cur_res['cnn_fg_prob_map']
            cur_res['ap_list'] = []

            if args.dataset == 'DRIVE':
                mask = fov[img_idx, :, :, 0]
                cur_res['mask'] = mask

                cur_label = label[img_idx, :, :, 0]
                label_roi = cur_label[mask.astype(bool)].reshape((-1,))
                fg_prob_map_roi = cur_res['cnn_fg_prob_map'][mask.astype(bool)].reshape((-1,))
                _, cur_cnn_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
                cur_res['ap'] = cur_cnn_ap
                cur_res['ap_list'].append(cur_cnn_ap)
            else:
                cur_label = label[img_idx, :, :, 0].reshape((-1,))
                fg_prob_map = cur_res['cnn_fg_prob_map'].reshape((-1,))
                _, cur_cnn_ap = util.get_auc_ap_score(cur_label, fg_prob_map)
                cur_res['ap'] = cur_cnn_ap
                cur_res['ap_list'].append(cur_cnn_ap)

            res_list.append(cur_res)

    # === make graphs ===
    graph_root = Path(cfg.PATHS.GRAPH_TEST_DIR)
    graph_root.mkdir(parents=True, exist_ok=True)
    func_arg = []
    for img_idx in range(len(res_list)):
        temp_fg_prob_map = res_list[img_idx]['final_fg_prob_map']
        base_name = Path(res_list[img_idx]['img_path']).name
        graph_path = graph_root / (base_name + '_%.2d_%.2d.graph_res' %
                                   (args.win_size, args.edge_geo_dist_thresh))
        func_arg.append(
            (
                temp_fg_prob_map,
                args.edge_type,
                args.win_size,
                args.edge_geo_dist_thresh,
                str(graph_path)
            )
        )

    if args.use_multiprocessing and pool is not None:
        pool.map(make_graph_using_srns, func_arg)
    else:
        for x in func_arg:
            make_graph_using_srns(x)

    # === load graphs ===
    for img_idx in range(len(res_list)):
        base_name = Path(res_list[img_idx]['img_path']).name
        loadpath = graph_root / (base_name + '_%.2d_%.2d.graph_res' %
                                 (args.win_size, args.edge_geo_dist_thresh))
        with open(loadpath, 'rb') as gf:
            temp_graph = pkl.load(gf)
        res_list[img_idx]['graph'] = temp_graph

    # === final results with VGN/inference module ===
    for img_idx in range(len(res_list)):

        cur_img = res_list[img_idx]['img']
        cur_conv_feats = res_list[img_idx]['conv_feats']
        cur_cnn_feat = res_list[img_idx]['cnn_feat']
        cur_cnn_feat_spatial_sizes = res_list[img_idx]['cnn_feat_spatial_sizes']
        cur_graph = res_list[img_idx]['graph']

        # relabel nodes [0..N-1]
        cur_graph = nx.convert_node_labels_to_integers(cur_graph)
        node_byxs = util.get_node_byx_from_graph(
            cur_graph, [cur_graph.number_of_nodes()]
        )

        # adjacency (binary or weighted) -> SparseTensorValue
        if 'geo_dist_weighted' in args.edge_type:
            A = nx.adjacency_matrix(cur_graph)  # weighted
        else:
            A = nx.adjacency_matrix(cur_graph, weight=None).astype(float)  # binary
        adj_sp = _graph_to_sparse_tensor(A)

        cur_feed_dict = {
            network.imgs: cur_img,
            network.conv_feats: cur_conv_feats,
            network.node_byxs: node_byxs,
            network.adj: adj_sp,
            network.is_lr_flipped: False,
            network.is_ud_flipped: False,
            network.gnn_feat_dropout: float(args.gnn_feat_dropout),
            network.gnn_att_dropout: float(args.gnn_att_dropout),
            network.post_cnn_dropout: float(args.post_cnn_dropout),
        }
        cur_feed_dict.update({
            network.cnn_feat[cur_key]: cur_cnn_feat[cur_key]
            for cur_key in network.cnn_feat.keys()
        })
        cur_feed_dict.update({
            network.cnn_feat_spatial_sizes[cur_key]: cur_cnn_feat_spatial_sizes[cur_key]
            for cur_key in network.cnn_feat_spatial_sizes.keys()
        })

        res_prob_map = sess.run(
            [network.post_cnn_img_fg_prob],
            feed_dict=cur_feed_dict
        )[0]

        res_prob_map = res_prob_map.reshape(
            (res_prob_map.shape[1], res_prob_map.shape[2])
        )

        # AP computation
        if args.dataset == 'DRIVE':
            cur_label = res_list[img_idx]['label']
            cur_label = np.squeeze(cur_label)
            cur_mask = res_list[img_idx]['mask']
            label_roi = cur_label[cur_mask.astype(bool)].reshape((-1,))
            fg_prob_map_roi = res_prob_map[cur_mask.astype(bool)].reshape((-1,))
            _, cur_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
            res_prob_map = res_prob_map * cur_mask
        else:
            cur_label = res_list[img_idx]['label']
            cur_label = np.squeeze(cur_label)
            _, cur_ap = util.get_auc_ap_score(
                cur_label.reshape((-1,)),
                res_prob_map.reshape((-1,))
            )

        res_list[img_idx]['ap'] = cur_ap
        res_list[img_idx]['ap_list'].append(cur_ap)
        res_list[img_idx]['final_fg_prob_map'] = res_prob_map

    # === dataset-level metrics ===
    all_labels = np.zeros((0,))
    all_preds = np.zeros((0,))
    # accumulators for detailed metrics
    cnn_acc_batch = []
    cnn_prec_batch = []
    cnn_rec_batch = []
    test_loss_list = []

    for img_idx in range(len(res_list)):

        cur_label = res_list[img_idx]['label']
        cur_label = np.squeeze(cur_label)
        cur_pred = res_list[img_idx]['final_fg_prob_map']

        img_path = Path(res_list[img_idx]['img_path'])
        base_name = img_path.name

        # save qualitative results (uint8)
        temp_output = (np.clip(cur_pred, 0.0, 1.0) * 255.0).astype(np.uint8)
        cur_save_path = os.path.join(res_save_path, base_name + '_prob_final.png')
        skimage.io.imsave(cur_save_path, temp_output)

        cur_save_path = os.path.join(res_save_path, base_name + '.npy')
        np.save(cur_save_path, cur_pred.astype(np.float32))

        temp_output = ((1. - np.clip(cur_pred, 0.0, 1.0)) * 255.0).astype(np.uint8)
        cur_save_path = os.path.join(res_save_path, base_name + '_prob_final_inv.png')
        skimage.io.imsave(cur_save_path, temp_output)

        if args.dataset == 'DRIVE':
            cur_mask = res_list[img_idx]['mask']
            cur_label = cur_label[cur_mask.astype(bool)]
            cur_pred = cur_pred[cur_mask.astype(bool)]

        all_labels = np.concatenate((all_labels, np.reshape(cur_label, (-1,))))
        all_preds = np.concatenate((all_preds, np.reshape(cur_pred, (-1,))))
        # batch metrics
        pred_bin = cur_pred >= 0.5
        label_bin = cur_label.astype(bool)
        correct = (pred_bin == label_bin).astype(np.float32)
        if correct.size > 0:
            cnn_acc_batch.append(float(np.mean(correct)))
        tp = np.sum(np.logical_and(pred_bin, label_bin))
        fp = np.sum(np.logical_and(pred_bin, np.logical_not(label_bin)))
        fn = np.sum(np.logical_and(np.logical_not(pred_bin), label_bin))
        if tp + fp > 0:
            cnn_prec_batch.append(float(tp / (tp + fp)))
        if tp + fn > 0:
            cnn_rec_batch.append(float(tp / (tp + fn)))
        test_loss_list.append(float(np.mean((cur_pred - cur_label) ** 2)))

        print('AP list for', res_list[img_idx]['img_path'], ':', res_list[img_idx]['ap_list'])
        f_log.write(
            'AP list for ' + res_list[img_idx]['img_path'] + ' : ' +
            str(res_list[img_idx]['ap_list']) + '\n'
        )

    auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
    all_labels_bin = np.copy(all_labels).astype(np.bool_)
    all_preds_bin = all_preds >= 0.5
    all_correct = all_labels_bin == all_preds_bin
    acc_test = np.mean(all_correct.astype(np.float32))

    ext = _compute_binary_metrics(all_labels, all_preds, threshold=0.5)
    final_test_avg_acc = float(np.mean(cnn_acc_batch)) if cnn_acc_batch else 0.0
    final_test_avg_prec = float(np.mean(cnn_prec_batch)) if cnn_prec_batch else 0.0
    final_test_avg_rec = float(np.mean(cnn_rec_batch)) if cnn_rec_batch else 0.0
    print('Accuracy: %.4f' % acc_test)
    print('Specificity: %.4f' % ext['specificity'])
    print('Recall/Sensitivity: %.4f' % ext['recall'])
    print('F1/Dice: %.4f' % ext['f1'])
    print('AUC: %.4f' % auc_test)
    print('AP: %.4f' % ap_test)
    print('IoU: %.4f' % ext['iou'])
    print('mIoU: %.4f' % ext['iou'])

    f_log.write('test_loss ' + str(np.mean(test_loss_list)) + '\n')
    f_log.write('Accuracy ' + str(acc_test) + '\n')
    f_log.write('Specificity ' + str(ext['specificity']) + '\n')
    f_log.write('Recall/Sensitivity ' + str(ext['recall']) + '\n')
    f_log.write('F1_Dice ' + str(ext['f1']) + '\n')
    f_log.write('AUC ' + str(auc_test) + '\n')
    f_log.write('AP ' + str(ap_test) + '\n')
    f_log.write('IoU ' + str(ext['iou']) + '\n')
    f_log.write('mIoU ' + str(ext['iou']) + '\n')
    f_log.flush()
    f_log.close()

    sess.close()
    if args.use_multiprocessing and pool is not None:
        pool.terminate()
    print("Testing complete.")
