# train_VGN.py
import os
import argparse
import json
import time
from datetime import datetime, timedelta

import numpy as np
import skimage.io
import tensorflow as tf
import networkx as nx
import scipy.sparse as sp

from config import cfg
from Modules.model import vessel_segm_vgn  # uses TF1 compat shims internally
import util as util


# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train Vessel Graph Network (TF1) with CNN init from best/iter/path')

    # Dataset / split
    p.add_argument('--dataset', default=cfg.DEFAULT_DATASET, type=str,
                   choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--use_fov_mask', default=True, type=bool)
    p.add_argument('--eval_split', default='test', choices=['test', 'val', 'none'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--limit_train_images', type=int, default=0)

    # Model
    p.add_argument('--cnn_model', default='driu', choices=['driu', 'driu_large'])
    p.add_argument('--norm_type', default='GN', choices=['GN', 'BN', 'none'])
    p.add_argument('--win_size', type=int, default=4)
    p.add_argument('--edge_geo_dist_thresh', type=int, default=10)

    # GAT head (TF version in model.py expects these lists)
    p.add_argument('--gat_hid_units', type=int, nargs='+', default=[16, 16])
    p.add_argument('--gat_n_heads', type=int, nargs='+', default=[4, 4, 1])
    p.add_argument('--gat_use_residual', action='store_true', default=True)

    # Loss toggles / weights
    p.add_argument('--cnn_loss_on', action='store_true', default=True)
    p.add_argument('--gnn_loss_on', action='store_true', default=True)
    p.add_argument('--gnn_loss_weight', type=float, default=1.0)
    p.add_argument('--infer_module_kernel_size', type=int, default=3)
    p.add_argument('--use_enc_layer', action='store_true', default=False,
                   help='1x1 conv-encode CNN skip feats before fusion in infer module')

    # Optim
    p.add_argument('--opt', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--lr', type=float, default=1e-4)  # global lr placeholder (model uses it)
    p.add_argument('--max_iters', type=int, default=cfg.MAX_ITERS)
    p.add_argument('--lr_scheduling', default='pc', choices=['pc', 'fixed', 'exp'])
    p.add_argument('--lr_decay_tp', type=float, default=0.8)
    p.add_argument('--old_net_ft_lr', type=float, default=0.0) 
    p.add_argument('--new_net_lr', type=float, default=1e-4)
    p.add_argument('--infer_module_grad_weight', type=float, default=1.0)
    p.add_argument('--do_simul_training', action='store_true', default=False)

    # Dropouts
    p.add_argument('--gnn_feat_dropout', type=float, default=0.1)
    p.add_argument('--gnn_att_dropout', type=float, default=0.1)
    p.add_argument('--post_cnn_dropout', type=float, default=0.1)

    # Save / runs
    p.add_argument('--save_root', default=cfg.VGN_SAVE_DIR, type=str)
    p.add_argument('--run_id', type=int, default=1)
    p.add_argument('--run_name', type=str, default='')
    p.add_argument('--no_save_checkpoints', action='store_true', default=False)

    # Resume full VGN training (overrides cnn_init if used)
    p.add_argument('--resume', default='none', choices=['none', 'latest', 'iter', 'best'])
    p.add_argument('--resume_iter', type=int, default=0)
    p.add_argument('--resume_dir', type=str, default='')

    # CNN init from existing CNN checkpoint
    p.add_argument('--cnn_init_mode', default='auto', choices=['auto', 'best', 'iter', 'path', 'none'])
    p.add_argument('--cnn_init_iter', type=int, default=50000)
    p.add_argument('--cnn_init_dir', type=str, default=str(cfg.PATHS.DRIU_DRIVE_ROOT))
    p.add_argument('--cnn_init_path', type=str, default=str(cfg.CNN_INIT))

    # Eval cadence
    p.add_argument('--display', type=int, default=cfg.TRAIN.DISPLAY)
    p.add_argument('--test_iters', type=int, default=cfg.TRAIN.TEST_ITERS)
    p.add_argument('--snapshot_iters', type=int, default=cfg.TRAIN.SNAPSHOT_ITERS)

    return p.parse_args()


# ----------------------------- Utils -----------------------------
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
    """Pick a CNN checkpoint to partially restore into VGN (conv/spe/resized_spe/* + output→img_output)."""
    mode = args.cnn_init_mode
    if mode == 'none':
        return ''
    if mode == 'path':
        p = args.cnn_init_path
        if p and os.path.exists(p + '.index'):
            return p
        if p and os.path.exists(p):
            # if user gave prefix without ext, return it
            return p if os.path.exists(p + '.index') else ''
        return ''
    if mode == 'iter':
        return _discover_tf_checkpoint(args.cnn_init_dir, 'iter', args.cnn_init_iter)
    if mode == 'best':
        return _discover_tf_checkpoint(args.cnn_init_dir, 'best')
    # auto: try dir/best, then dir/iter_N, then cfg.CNN_INIT
    p = _discover_tf_checkpoint(args.cnn_init_dir, 'best')
    if not p:
        p = _discover_tf_checkpoint(args.cnn_init_dir, 'iter', args.cnn_init_iter)
    if not p and cfg.CNN_INIT and os.path.exists(str(cfg.CNN_INIT) + '.index'):
        p = str(cfg.CNN_INIT)
    return p or ''


def _parse_iter_from_ckpt_path(ckpt_path: str) -> int:
    import re
    m = re.search(r'iter_(\d+)\.ckpt', os.path.basename(ckpt_path) if ckpt_path else '')
    return int(m.group(1)) if m else 0


def _graph_to_sparse_tensor(graph: nx.Graph):
    """Return tf.SparseTensorValue for adjacency with self-loops & binarization."""
    A = nx.adjacency_matrix(graph)  # scipy sparse
    indices, values, shape = util.preprocess_graph_gat(A)
    # util.preprocess_graph_gat already adds self-loops and binarizes
    from tensorflow.python.framework import sparse_tensor
    return sparse_tensor.SparseTensorValue(indices=indices.astype(np.int64),
                                           values=values.astype(np.float32),
                                           dense_shape=np.array(shape, dtype=np.int64))


def compute_binary_metrics(labels, probs, mask=None, threshold=0.5):
    """Common binary metrics on numpy arrays."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    probs = np.asarray(probs).astype(np.float32).ravel()
    if mask is not None:
        m = np.asarray(mask).astype(np.int64).ravel()
        keep = m > 0
        if keep.any():
            labels = labels[keep]; probs = probs[keep]
    preds = (probs >= float(threshold)).astype(np.int64)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    denom = lambda x: (x if x > 0 else 1)
    precision = tp / denom(tp + fp)
    recall    = tp / denom(tp + fn)
    specificity = tn / denom(tn + fp)
    accuracy  = (tp + tn) / denom(tp + tn + fp + fn)
    f1   = (2 * precision * recall) / denom(precision + recall)
    dice = (2 * tp) / denom(2 * tp + fp + fn)
    iou  = tp / denom(tp + fp + fn)
    try:
        auc, ap = util.get_auc_ap_score(labels.astype(np.float32), probs.astype(np.float32))
    except Exception:
        auc, ap = np.nan, np.nan
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=float(accuracy), precision=float(precision), recall=float(recall),
                specificity=float(specificity), f1=float(f1), dice=float(dice), iou=float(iou),
                auc=float(auc) if np.isfinite(auc) else np.nan,
                ap=float(ap) if np.isfinite(ap) else np.nan)


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
    """Partially restore CNN weights from an external CNN checkpoint.
       Maps 'output/*' in CNN → 'img_output/*' in VGN.
    """
    if not ckpt_path:
        print('[CNN-INIT] Skipped (no checkpoint path).')
        return False

    print(f'[CNN-INIT] Restoring CNN layers from: {ckpt_path}')
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
    ckpt_vars = reader.get_variable_to_shape_map()

    gvars = tf.compat.v1.global_variables()
    wanted_scopes = list(getattr(net, 'var_to_restore', []))  # CNN scopes inside VGN
    if 'img_output' in wanted_scopes and 'output' not in wanted_scopes:
        pass  # handled via mapping

    var_map = {}
    for v in gvars:
        name = v.op.name  # e.g., 'conv1_1/W'
        scope = name.split('/')[0]
        if scope not in wanted_scopes:
            continue
        # Map 'img_output' (VGN) -> 'output' (CNN checkpoint)
        ck_scope = 'output' if scope == 'img_output' else scope
        ck_name = ck_scope + '/' + name.split('/')[-1]
        if ck_name in ckpt_vars and list(ckpt_vars[ck_name]) == list(v.shape):
            var_map[ck_name] = v

    if not var_map:
        print('[CNN-INIT] No matching variables found to restore.')
        return False

    saver_part = tf.compat.v1.train.Saver(var_list=var_map)
    saver_part.restore(sess, ckpt_path)
    print(f'[CNN-INIT] Restored {len(var_map)} tensors into VGN CNN (with output→img_output remap).')
    return True


def _save_eval_vis(save_dir, img_list, prob_map_batch):
    """Save qualitative predictions for a batch."""
    os.makedirs(save_dir, exist_ok=True)
    bs = len(img_list)
    for i in range(bs):
        stem = os.path.basename(str(img_list[i])); base = os.path.splitext(stem)[0]
        prob = prob_map_batch[i]
        if prob.ndim == 3 and prob.shape[-1] == 1: prob = np.squeeze(prob, -1)
        skimage.io.imsave(os.path.join(save_dir, f"{base}_prob.png"),
                          (prob * 255).clip(0, 255).astype(np.uint8))
        out = (prob >= 0.5).astype(np.uint8) * 255
        skimage.io.imsave(os.path.join(save_dir, f"{base}_output.png"), out)


# --------------------------- Main ---------------------------
if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)

    # Resolve data lists
    train_txt, test_txt = _get_split_txt_paths(args.dataset)
    with open(train_txt) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_txt) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    if args.dataset == 'HRF':
        test_img_names = [test_img_names[x] for x in range(7, len(test_img_names), 20)]

    if args.limit_train_images and args.limit_train_images > 0:
        train_img_names = train_img_names[:args.limit_train_images]
        print(f"[SMOKE] limit_train_images -> {len(train_img_names)}")

    # Build eval split
    if args.eval_split == 'val':
        rs = np.random.RandomState(args.seed)
        n = len(train_img_names)
        idxs = np.arange(n); rs.shuffle(idxs)
        n_val = max(1, int(np.ceil(0.1 * n)))
        val_idx = set(idxs[:n_val].tolist())
        val_img_names = [x for i, x in enumerate(train_img_names) if i in val_idx]
        train_img_names = [x for i, x in enumerate(train_img_names) if i not in val_idx]
        eval_img_names = val_img_names
        eval_tag = 'val'
    elif args.eval_split == 'test':
        eval_img_names = test_img_names
        eval_tag = 'test'
    else:
        eval_img_names = []
        eval_tag = 'none'

    # Data layers (graph-aware)
    dl_train = util.GraphDataLayer(train_img_names, is_training=True,
                                   edge_type='srns_geo_dist_binary',
                                   win_size=args.win_size,
                                   edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    dl_eval = util.GraphDataLayer(eval_img_names, is_training=False,
                                  edge_type='srns_geo_dist_binary',
                                  win_size=args.win_size,
                                  edge_geo_dist_thresh=args.edge_geo_dist_thresh) if eval_img_names else None

    # Run paths
    run_name = args.run_name if args.run_name else f"run{int(args.run_id)}"
    run_root = os.path.join(args.save_root, run_name) if args.save_root else run_name
    os.makedirs(run_root, exist_ok=True)
    model_save_path = os.path.join(run_root, 'train'); os.makedirs(model_save_path, exist_ok=True)
    probmap_save_dir = os.path.join(model_save_path, 'probmaps'); os.makedirs(probmap_save_dir, exist_ok=True)
    eval_save_dir = os.path.join(run_root, (cfg.TEST.RES_SAVE_PATH if eval_tag == 'test' else 'val'))
    if eval_tag != 'none':
        os.makedirs(eval_save_dir, exist_ok=True)

    # Build network (TF1 compat is handled in model.py)
    # Pack params object with attributes the VGN class expects
    class _P: pass
    _p = _P()
    # copy args into expected attr names
    for k, v in vars(args).items():
        setattr(_p, k, v)
    # ensure lists for the GAT
    _p.gat_hid_units = list(args.gat_hid_units)
    _p.gat_n_heads = list(args.gat_n_heads)
    _p.gat_use_residual = bool(args.gat_use_residual)

    # Normalization type mapping
    _p.norm_type = None if args.norm_type.lower() == 'none' else args.norm_type

    net = vessel_segm_vgn(_p, None)

    # Session / IO
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
    saver_all = tf.compat.v1.train.Saver(max_to_keep=100)
    summary_writer = tf.compat.v1.summary.FileWriter(model_save_path, sess.graph)

    # Profiler CSV
    profiler_csv_path = os.path.join(model_save_path, 'profiler.csv')
    if not os.path.exists(profiler_csv_path):
        with open(profiler_csv_path, 'w') as pf:
            pf.write('iter,step_time_s,avg_step_time_s,elapsed_s,loss,cnn_loss,gnn_loss,post_loss,acc,dice,auc\n')

    # Init
    sess.run(tf.compat.v1.global_variables_initializer())

    # Resume VGN if requested
    start_iter = 0; did_resume = False
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

    # If not resuming full VGN, initialize CNN part from external CNN checkpoint (best/iter/path)
    if not did_resume:
        cnn_ckpt = _discover_cnn_init(args)
        _restore_cnn_from_ckpt(sess, net, cnn_ckpt)

    # Training loop
    print("Training VGN...")
    best_metric = -1.0  # Dice
    train_wall_start = time.time()
    timer = util.Timer()
    train_loss_hist = []
    train_cnn_loss_hist = []
    train_gnn_loss_hist = []
    train_post_loss_hist = []
    train_cnn_acc_hist = []
    train_gnn_acc_hist = []
    train_post_acc_hist = []

    for it in range(start_iter, args.max_iters):
        timer.tic()

        img_list, blobs = dl_train.forward()
        # numpy casts
        imgs   = np.asarray(blobs['img'], dtype=np.float32)
        labels = np.asarray(blobs['label'], dtype=np.int64)
        fovs   = np.asarray(blobs['fov'], dtype=np.int64) if args.use_fov_mask else np.ones_like(labels, dtype=np.int64)
        graph  = blobs['graph']  # networkx.Graph over disjoint union
        nlist  = blobs['num_of_nodes_list']
        vec_aug_on = blobs.get('vec_aug_on', np.zeros((7,), dtype=bool))
        rot_angle  = int(blobs.get('rot_angle', 0))

        # node positions [N,3] (subgraph_idx, y, x)
        node_byxs = util.get_node_byx_from_graph(graph, nlist).astype(np.int32)
        adj_sp = _graph_to_sparse_tensor(graph)

        # weights & aug flags
        pixel_weights = fovs.astype(np.float32)
        lr_flip = bool(vec_aug_on[0]); ud_flip = bool(vec_aug_on[1])
        rot90_num = float((rot_angle // 90) % 4)

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
        }

        # Train step
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

        # Save current probmaps for this batch
        for b_idx, img_path in enumerate(img_list):
            base = os.path.basename(os.path.normpath(str(img_path)))
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
                # metrics like dice/auc are computed during eval; write NaN here
                pf.write(f"{it+1},{timer.diff:.6f},{timer.average_time:.6f},{elapsed_s:.2f},"
                         f"{loss_total:.6f},{loss_cnn:.6f},{loss_gnn:.6f},{loss_post:.6f},"
                         f"{acc_post:.6f},nan,nan\n")
        except Exception:
            pass

        # Snapshot
        if not args.no_save_checkpoints and (it + 1) % args.snapshot_iters == 0:
            ck = os.path.join(model_save_path, f'iter_{it+1}.ckpt')
            saver_all.save(sess, ck)
            print(f'[CKPT] Wrote snapshot: {ck}')

        # Periodic eval
        if (eval_tag != 'none') and ((it + 1) % args.test_iters == 0):
            all_labels = np.zeros((0,))
            all_preds_post  = np.zeros((0,))
            all_preds_cnn = np.zeros((0,))
            all_node_labels = np.zeros((0,))
            all_node_probs = np.zeros((0,))
            eval_losses = []
            eval_cnn_losses = []
            eval_gnn_losses = []
            eval_post_losses = []
            eval_cnn_accs = []
            eval_gnn_accs = []
            eval_post_accs = []

            # one pass over eval set
            num_batches = int(np.ceil(float(len(eval_img_names)) / cfg.TRAIN.BATCH_SIZE))
            for _ in range(num_batches):
                img_list_e, blobs_e = dl_eval.forward()
                if len(img_list_e) == 0:
                    continue
                imgs_e   = np.asarray(blobs_e['img'], dtype=np.float32)
                labels_e = np.asarray(blobs_e['label'], dtype=np.int64)
                fovs_e   = np.asarray(blobs_e['fov'], dtype=np.int64) if args.use_fov_mask else np.ones_like(labels_e, dtype=np.int64)
                graph_e  = blobs_e['graph']; nlist_e = blobs_e['num_of_nodes_list']
                vec_aug_e= blobs_e.get('vec_aug_on', np.zeros((7,), dtype=bool))
                rot_ang_e= int(blobs_e.get('rot_angle', 0))

                node_byxs_e = util.get_node_byx_from_graph(graph_e, nlist_e).astype(np.int32)
                adj_sp_e = _graph_to_sparse_tensor(graph_e)
                lr_flip_e = bool(vec_aug_e[0]); ud_flip_e = bool(vec_aug_e[1])
                rot90_num_e = float((rot_ang_e // 90) % 4)

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

                # flatten metrics (mask outside FOV to zero before flattening)
                all_labels = np.concatenate((all_labels, labels_e.reshape(-1)))
                weighted_mask = fovs_e.astype(float)
                pm_post = prob_map_e * weighted_mask
                pm_cnn = cnn_prob_map_e * weighted_mask
                all_preds_post  = np.concatenate((all_preds_post, pm_post.reshape(-1)))
                all_preds_cnn = np.concatenate((all_preds_cnn, pm_cnn.reshape(-1)))

                # qualitative save (parity with your CNN code)
                _save_eval_vis(eval_save_dir, img_list_e, prob_map_e)

            # metrics
            post_metrics = compute_binary_metrics(all_labels, all_preds_post, mask=None, threshold=0.5)
            cnn_metrics = compute_binary_metrics(all_labels, all_preds_cnn, mask=None, threshold=0.5)
            gnn_metrics = None
            if all_node_labels.size > 0 and all_node_probs.size > 0:
                gnn_metrics = compute_binary_metrics(all_node_labels, all_node_probs,
                                                     mask=None, threshold=0.5)
            mean_eval_loss = float(np.mean(eval_losses)) if eval_losses else np.nan

            # TensorBoard (keep tags consistent)
            def _mean_or_zero(seq):
                return float(np.mean(seq)) if seq else 0.0

            cur_lr = sess.run(net.lr_handler)
            prefix = eval_tag if eval_tag else 'eval'
            summary = tf.compat.v1.Summary()
            summary.value.add(tag="train_loss", simple_value=_mean_or_zero(train_loss_hist))
            summary.value.add(tag="train_cnn_loss", simple_value=_mean_or_zero(train_cnn_loss_hist))
            summary.value.add(tag="train_gnn_loss", simple_value=_mean_or_zero(train_gnn_loss_hist))
            summary.value.add(tag="train_infer_module_loss", simple_value=_mean_or_zero(train_post_loss_hist))
            summary.value.add(tag="train_cnn_acc", simple_value=_mean_or_zero(train_cnn_acc_hist))
            summary.value.add(tag="train_gnn_acc", simple_value=_mean_or_zero(train_gnn_acc_hist))
            summary.value.add(tag="train_infer_module_acc", simple_value=_mean_or_zero(train_post_acc_hist))
            summary.value.add(tag=f"{prefix}_loss", simple_value=mean_eval_loss)
            summary.value.add(tag=f"{prefix}_cnn_loss", simple_value=_mean_or_zero(eval_cnn_losses))
            summary.value.add(tag=f"{prefix}_gnn_loss", simple_value=_mean_or_zero(eval_gnn_losses))
            summary.value.add(tag=f"{prefix}_infer_module_loss", simple_value=_mean_or_zero(eval_post_losses))
            summary.value.add(tag=f"{prefix}_cnn_acc", simple_value=cnn_metrics['accuracy'])
            summary.value.add(tag=f"{prefix}_cnn_auc", simple_value=cnn_metrics['auc'])
            summary.value.add(tag=f"{prefix}_cnn_ap", simple_value=cnn_metrics['ap'])
            gnn_acc_val = gnn_metrics['accuracy'] if gnn_metrics is not None else 0.0
            gnn_auc_val = gnn_metrics['auc'] if gnn_metrics is not None else 0.0
            gnn_ap_val = gnn_metrics['ap'] if gnn_metrics is not None else 0.0
            summary.value.add(tag=f"{prefix}_gnn_acc", simple_value=gnn_acc_val)
            summary.value.add(tag=f"{prefix}_gnn_auc", simple_value=gnn_auc_val)
            summary.value.add(tag=f"{prefix}_gnn_ap", simple_value=gnn_ap_val)
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

            print(f"{eval_tag.upper()}: iter {it+1}/{args.max_iters}, loss: {mean_eval_loss:.4f}, "
                  f"acc: {post_metrics['accuracy']:.4f}, auc: {post_metrics['auc']:.4f}, "
                  f"ap: {post_metrics['ap']:.4f}, dice: {post_metrics['dice']:.4f}, "
                  f"iou: {post_metrics['iou']:.4f}")

            # Save Dice-best
            if not args.no_save_checkpoints and post_metrics['dice'] > best_metric:
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

            # Save a few train-set visualizations too (optional)
            train_vis_dir = os.path.join(run_root, 'train')
            os.makedirs(train_vis_dir, exist_ok=True)
            _save_eval_vis(train_vis_dir, img_list, prob_map)

            train_loss_hist = []

    # Final snapshot
    if not args.no_save_checkpoints:
        final_path = os.path.join(model_save_path, f'iter_{args.max_iters}.ckpt')
        saver_all.save(sess, final_path)
        print(f'[CKPT] Wrote final snapshot: {final_path}')

    total_elapsed = time.time() - train_wall_start
    print('Training completed in {:.2f} seconds ({}).'.format(
        total_elapsed, str(timedelta(seconds=int(total_elapsed)))))
    sess.close()
    print("VGN training complete.")
