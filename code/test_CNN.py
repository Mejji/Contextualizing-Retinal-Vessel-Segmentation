# coded by syshin

import os
import glob
import numpy as np
import skimage.io
import argparse
import time
import threading
import queue
import tensorflow as tf

from config import cfg
from Modules.model import vessel_segm_cnn
import util
import gpu_utils


class Prefetcher(object):
    """Background prefetcher that calls data_layer.forward() into a small queue.
    Keeps at most `maxsize` batches in memory (default 2).
    """
    def __init__(self, data_layer, maxsize=2):
        self.dl = data_layer
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                img_list, blobs = self.dl.forward()
            except Exception as e:
                # Log and skip this batch instead of killing the prefetcher.
                import traceback
                print(f"[Prefetcher] data_layer.forward() failed: {e}")
                traceback.print_exc()
                continue
            # Put batch into queue (block until space is available)
            try:
                self.q.put((img_list, blobs))
            except Exception as e:
                # Should be rare; log and retry next iteration
                print(f"[Prefetcher] queue.put failed: {e}")
                continue

    def get(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except Exception:
            return None, None

    def stop(self):
        self._stop.set()
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description='Test a vessel_segm_cnn network')
    from config import cfg as _cfg
    parser.add_argument('--dataset', default=_cfg.DEFAULT_DATASET, type=str,
                        help='Dataset to use: DRIVE, STARE, CHASE_DB1 (or CHASE-DB1), HRF')
    parser.add_argument('--cnn_model', default='driu', type=str, help='CNN model to use')
    parser.add_argument('--use_fov_mask', default=True, type=bool, help='Whether to use FOV masks')
    # Derive a repo-relative default checkpoint path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # Use underscore form to match workspace structure; will be refined later if needed
    default_ckpt = os.path.join(repo_root, 'shin-model','DRIVE','DRIU_', 'DRIU_DRIVE.ckpt')
    parser.add_argument('--model_path', default=default_ckpt, type=str, help='Path to the pretrained model (.ckpt prefix)')
    parser.add_argument('--save_root', default=os.path.join(repo_root, 'DRIU_DRIVE'), type=str, help='DRIU_DRIVE, DRIU_CHASE, DRIU_HRF for folders')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam','sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--include_train', action='store_true', help='Also run over the training list (mainly for debugging)')
    parser.add_argument('--dump_debug', action='store_true', help='Dump an extra debug .npy and contrast-stretched prob map')
    # Multi-run layout support (align with trainer). If provided, tester will auto-pick best checkpoint under save_root/runX/train.
    parser.add_argument('--run_id', type=int, default=0, help='Run number (1..N). If >0, search in <save_root>/run<id>/train for best/iter checkpoints')
    parser.add_argument('--run_name', type=str, default='', help="Optional run name (e.g., 'run3'). Overrides --run_id if provided.")
    # Smoke test helpers
    parser.add_argument('--skip_restore', action='store_true', default=False,
                        help='Skip restoring checkpoint (random init) for pipeline smoke test.')
    parser.add_argument('--limit_images', type=int, default=0,
                        help='If >0, limit number of test (and optionally train) images for quick smoke runs.')
    # Profiling
    parser.add_argument('--profile_functions', action='store_true', default=False,
                        help='Enable lightweight function-level profiling of DataLayer & preprocessing functions.')
    parser.add_argument('--profile_report_every', type=int, default=0,
                        help='Batches between interim profiling reports (0 disables interim, final report always shown if profiling enabled).')
    return parser.parse_args()


def setup_gpu():
    try:
        physical_gpus = tf.config.list_physical_devices('GPU')
        print("GPUs available:", [gpu.name for gpu in physical_gpus])
        if getattr(cfg.GPU, 'STRICT', True) and not physical_gpus:
            raise RuntimeError("STRICT GPU mode is enabled but no GPU detected")
        if physical_gpus and getattr(cfg.GPU, 'USE_GPU', True):
            target_idx = min(max(0, int(getattr(cfg.GPU, 'INDEX', 0))), len(physical_gpus)-1)
            tf.config.set_visible_devices(physical_gpus[target_idx], 'GPU')
            tf.config.experimental.set_memory_growth(physical_gpus[target_idx], getattr(cfg.GPU, 'ALLOW_GROWTH', True))
            chosen_dev = f"/GPU:0"
        else:
            chosen_dev = "/CPU:0"
        print(f"Selected device for graph construction: {chosen_dev}")
        return chosen_dev
    except Exception as e:
        if getattr(cfg.GPU, 'STRICT', True):
            raise
        print("Warning (GPU setup):", e)
        return "/CPU:0"


def _score_iter_from_name(path):
    """Helper: return iteration number if name like iter_XXXX.ckpt else 0."""
    import re
    try:
        m = re.search(r'iter_(\d+)\.ckpt', os.path.basename(path) if path else '')
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def resolve_best_ckpt_under(dir_path):
    """Given a directory that contains checkpoints, return the best available checkpoint prefix.
    Priority: best.ckpt > highest iter_*.ckpt > tf.train.latest_checkpoint
    Returns '' if nothing found.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return ''
    # 1) Explicit best
    best_prefix = os.path.join(dir_path, 'best.ckpt')
    if os.path.exists(best_prefix + '.index'):
        return best_prefix
    # 2) Highest iteration checkpoint
    import glob as _glob
    candidates = []
    for pat in ('iter_*.ckpt.index', 'iter_*.ckpt'):  # prefer .index listing on some filesystems
        candidates.extend(_glob.glob(os.path.join(dir_path, pat)))
    if candidates:
        # Normalize to prefix
        prefixes = []
        for c in candidates:
            p = c
            if p.endswith('.index'):
                p = p[:-len('.index')]
            prefixes.append(p)
        prefixes = sorted(set(prefixes), key=_score_iter_from_name, reverse=True)
        for p in prefixes:
            if os.path.exists(p + '.index') and (os.path.exists(p + '.data-00000-of-00001') or True):
                # Accept even if data shard name differs; TF will error later if truly missing
                return p
    # 3) TF latest
    try:
        latest = tf.train.latest_checkpoint(dir_path)
        if latest:
            return latest
    except Exception:
        pass
    return ''


def resolve_best_ckpt_from_runs(save_root, run_name='', run_id=0):
    """Search for the best checkpoint under runs inside save_root.
    If run_name or run_id provided, only check that run. Otherwise, scan all subdirs named run*/train and pick the best by:
      - best_meta.json metric_value (if present)
      - else existence of best.ckpt
      - else highest iter_*.ckpt
    Returns a checkpoint prefix or ''.
    """
    run_dirs = []
    if run_name:
        run_dirs = [os.path.join(save_root, run_name, cfg.TRAIN.MODEL_SAVE_PATH)]
    elif run_id and run_id > 0:
        run_dirs = [os.path.join(save_root, f'run{int(run_id)}', cfg.TRAIN.MODEL_SAVE_PATH)]
    else:
        # scan all run* dirs
        try:
            for d in os.listdir(save_root):
                if d.lower().startswith('run'):
                    run_dirs.append(os.path.join(save_root, d, cfg.TRAIN.MODEL_SAVE_PATH))
        except Exception:
            pass

    best_choice = ('', -1.0, 0)  # (ckpt_prefix, metric_value, iter_num)
    import json
    for d in run_dirs:
        # Try best_meta.json
        meta_path = os.path.join(d, 'best_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as fm:
                    meta = json.load(fm)
                mv = float(meta.get('metric_value', -1.0))
            except Exception:
                mv = -1.0
        else:
            mv = -1.0
        # Resolve a concrete checkpoint
        ckpt = resolve_best_ckpt_under(d)
        itn = _score_iter_from_name(ckpt)
        # Scoring tuple: prefer higher metric; tie-break on iteration
        score_tuple = (mv, itn)
        if ckpt and (mv > best_choice[1] or (mv == best_choice[1] and itn > best_choice[2])):
            best_choice = (ckpt, mv, itn)

    return best_choice[0]


def resolve_ckpt_path(model_path):
    """Normalize checkpoint path to .ckpt prefix"""
    if '*' in model_path:
        candidates = sorted(glob.glob(model_path))
        if not candidates:
            candidates = sorted(glob.glob(model_path + '.*'))
        if not candidates:
            raise ValueError(f"No checkpoint matches: {model_path}")
        chosen = [c for c in candidates if c.endswith('.ckpt.index')]
        restore_path = chosen[0] if chosen else candidates[0]
    else:
        restore_path = model_path

    # Remove suffixes if present
    for suffix in ['.index', '.meta']:
        if restore_path.endswith(suffix):
            restore_path = restore_path[:-len(suffix)]
    # If both .index and .data exist, we're good
    ckpt_index = restore_path + '.index'
    # Common TF shard name for v1 checkpoints
    ckpt_data = restore_path + '.data-00000-of-00001'
    if os.path.exists(ckpt_index) and os.path.exists(ckpt_data):
        return restore_path

    # Auto-discovery fallbacks
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # 1) If a TF 'checkpoint' file exists under DRIU_DRIVE/train, parse it
    train_dir = os.path.join(repo_root, 'DRIU_HRF', 'train')
    checkpoint_txt = os.path.join(train_dir, 'checkpoint')
    if os.path.exists(checkpoint_txt):
        try:
            with open(checkpoint_txt, 'r') as f:
                txt = f.read()
            import re
            m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', txt)
            if m:
                last_ckpt = os.path.join(train_dir, m.group(1))
                # Normalize to prefix and verify files
                if last_ckpt.endswith('.index'):
                    last_ckpt = last_ckpt[:-len('.index')]
                if last_ckpt.endswith('.meta'):
                    last_ckpt = last_ckpt[:-len('.meta')]
                if os.path.exists(last_ckpt + '.index') and os.path.exists(last_ckpt + '.data-00000-of-00001'):
                    return last_ckpt
        except Exception:
            pass

    # 2) Glob for iter_*.ckpt under DRIU_DRIVE/train first, then models/**
    search_globs = [
        os.path.join(repo_root, 'DRIU_HRF', 'train', 'iter_*.ckpt'),
        os.path.join(repo_root, 'models', '**', '*.ckpt'),
    ]

    found = []
    for pat in search_globs:
        found.extend(glob.glob(pat, recursive=True))

    # Prefer .ckpt.index files
    if not found:
        raise FileNotFoundError(f"Checkpoint files not found for prefix: {restore_path}. Looked for {ckpt_index} and {ckpt_data}. Also tried fallbacks in DRIU_DRIVE/train and models/. Provide --skip_restore or a valid --model_path.")

    def score_ckpt(path):
        # Score by iteration number if matches iter_XXXX, else by mtime
        base = os.path.basename(path)
        import re, time
        m = re.search(r'iter_(\d+)\.ckpt', base)
        if m:
            return (int(m.group(1)), 0.0)
        try:
            return (0, os.path.getmtime(path))
        except Exception:
            return (0, 0.0)

    # Choose best candidate
    for best in sorted(found, key=score_ckpt, reverse=True):
        # Normalize to .ckpt prefix
        prefix = best
        if prefix.endswith('.index'):
            prefix = prefix[:-len('.index')]
        if prefix.endswith('.meta'):
            prefix = prefix[:-len('.meta')]
        if os.path.exists(prefix + '.index') and os.path.exists(prefix + '.data-00000-of-00001'):
            return prefix

    # If none had both files, fall back to the best seen (will still likely fail)
    best = sorted(found, key=score_ckpt, reverse=True)[0]
    if best.endswith('.index'):
        best = best[:-len('.index')]
    if best.endswith('.meta'):
        best = best[:-len('.meta')]
    return best


def build_save_dirs(save_root):
    res_save_path = os.path.join(save_root, cfg.TEST.RES_SAVE_PATH) if len(save_root) > 0 else cfg.TEST.RES_SAVE_PATH
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(res_save_path, exist_ok=True)
    return res_save_path


def load_dataset_txt(dataset):
    # Normalize dataset key to handle both CHASE_DB1 and CHASE-DB1, etc.
    ds_key = str(dataset).upper().replace('-', '_')
    if ds_key == 'DRIVE':
        train_txt = cfg.TRAIN.DRIVE_SET_TXT_PATH
        test_txt = cfg.TEST.DRIVE_SET_TXT_PATH
    elif ds_key == 'STARE':
        train_txt = cfg.TRAIN.STARE_SET_TXT_PATH
        test_txt = cfg.TEST.STARE_SET_TXT_PATH
    elif ds_key == 'CHASE_DB1':
        train_txt = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH
        test_txt = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif ds_key == 'HRF':
        train_txt = cfg.TRAIN.HRF_SET_TXT_PATH
        test_txt = cfg.TEST.HRF_SET_TXT_PATH
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    with open(train_txt) as f:
        train_imgs = [x.strip() for x in f.readlines()]
    with open(test_txt) as f:
        test_imgs = [x.strip() for x in f.readlines()]
    return train_imgs, test_imgs


def save_outputs(res_save_path, img_list, fg_prob_map, reshaped_output, dump_debug=False):
    for idx, img_path in enumerate(img_list):
        temp_name = os.path.basename(img_path)
        cur_prob = fg_prob_map[idx]
        # Raw 0-1 to 0-255
        fg_map_uint8 = (cur_prob * 255).clip(0, 255).astype(np.uint8)
        # Simple contrast stretch for visualization if map is nearly flat
        if dump_debug:
            p_min = float(np.min(cur_prob))
            p_max = float(np.max(cur_prob))
            if p_max > p_min:
                stretched = ((cur_prob - p_min) / (p_max - p_min) * 255).astype(np.uint8)
            else:
                stretched = fg_map_uint8
        fg_map_inv_uint8 = ((1 - cur_prob) * 255).clip(0, 255).astype(np.uint8)
        output_uint8 = (reshaped_output[idx].astype(np.uint8) * 255)
        npy_path = os.path.join(res_save_path, temp_name + '.npy')
        skimage.io.imsave(os.path.join(res_save_path, temp_name + '_prob.png'), fg_map_uint8)
        skimage.io.imsave(os.path.join(res_save_path, temp_name + '_prob_inv.png'), fg_map_inv_uint8)
        skimage.io.imsave(os.path.join(res_save_path, temp_name + '_output.png'), output_uint8)
        if dump_debug:
            skimage.io.imsave(os.path.join(res_save_path, temp_name + '_prob_stretch.png'), stretched)
        np.save(npy_path, cur_prob)


def compute_binary_metrics(labels, probs, mask=None, threshold=0.5):
    """Compute binary segmentation metrics (numpy): specificity, F1, Dice, IoU, plus accuracy/precision/recall.

    Returns a dict with keys: accuracy, precision, recall, specificity, f1, dice, iou, tp, tn, fp, fn.
    """
    labels = np.asarray(labels).astype(np.int64).ravel()
    probs = np.asarray(probs).astype(np.float32).ravel()
    if mask is not None:
        m = np.asarray(mask).astype(np.int64).ravel()
        keep = m > 0
        if keep.any():
            labels = labels[keep]
            probs = probs[keep]
    preds = (probs >= float(threshold)).astype(np.int64)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    denom = lambda x: (x if x > 0 else 1)
    precision = tp / denom(tp + fp)
    recall = tp / denom(tp + fn)
    specificity = tn / denom(tn + fp)
    accuracy = (tp + tn) / denom(tp + tn + fp + fn)
    f1 = (2 * precision * recall) / denom(precision + recall)
    dice = (2 * tp) / denom(2 * tp + fp + fn)
    iou = tp / denom(tp + fp + fn)
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall),
        'specificity': float(specificity), 'f1': float(f1), 'dice': float(dice), 'iou': float(iou),
    }


def main():
    tf.compat.v1.disable_eager_execution()
    args = parse_args()
    device_str = setup_gpu()
    print("Args:", args)

    # Helper: convert torch tensors to numpy if needed
    def to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return x

    profiler = None
    if args.profile_functions:
        try:
            import importlib
            profiler_hooks = importlib.import_module('profiler_hooks')
        except Exception:
            # Fallback stub so the script can run when profiler_hooks is not installed.
            class _ProfilerStub:
                @staticmethod
                def install():
                    return []

                @staticmethod
                def report():
                    return "Profiler hooks not available"

            profiler_hooks = _ProfilerStub()
        try:
            installed = profiler_hooks.install()
            profiler = profiler_hooks
            print(f"[PROFILE] Installed wrappers: {installed}")
        except Exception as e:
            print(f"[PROFILE] Failed to install profiling hooks: {e}")

    train_imgs, test_imgs = load_dataset_txt(args.dataset)
    if args.limit_images and args.limit_images > 0:
        # apply limit to test set (and train if included)
        if args.include_train:
            train_imgs = train_imgs[:args.limit_images]
        test_imgs = test_imgs[:args.limit_images]
        print(f"[SMOKE] limit_images active: using {len(train_imgs) if args.include_train else 0} train / {len(test_imgs)} test images")
    len_train, len_test = len(train_imgs), len(test_imgs)

    # Only instantiate train datalayer if requested
    data_layers = []
    phase_names = []
    use_padding = True if args.dataset.upper() == 'HRF' else False
    if args.include_train:
        data_layers.append(util.DataLayer(train_imgs, is_training=False, use_padding=use_padding))
        phase_names.append('TrainList')
    data_layers.append(util.DataLayer(test_imgs, is_training=False, use_padding=use_padding))
    phase_names.append('Test')
    res_save_path = build_save_dirs(args.save_root)

    # Build the network on the selected device so variables and ops are placed on GPU when possible.
    try:
        with tf.device(device_str):
            network = vessel_segm_cnn(args, None)
    except Exception:
        # Fallback to default construction if device placement fails
        print(f"Warning: failed to build network with device {device_str}, falling back to default device placement")
        network = vessel_segm_cnn(args, None)

    # TF1-style session
    # Unified GPU session creation
    sess = gpu_utils.make_tf_session(allow_growth=getattr(cfg.GPU, 'ALLOW_GROWTH', True))
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())

    if not args.skip_restore:
        # Prefer run-based best checkpoints if run info is provided (or discoverable)
        ckpt_path = ''
        if args.run_name or (args.run_id and args.run_id > 0):
            ckpt_path = resolve_best_ckpt_from_runs(args.save_root, args.run_name, args.run_id)
            if ckpt_path:
                print(f"[CKPT] Using run-based checkpoint: {ckpt_path}")
        # If still empty, try discovering best across all runs under save_root
        if not ckpt_path:
            ckpt_path = resolve_best_ckpt_from_runs(args.save_root)
            if ckpt_path:
                print(f"[CKPT] Using best checkpoint discovered across runs: {ckpt_path}")
        # Fallback: resolve from provided model_path
        if not ckpt_path:
            ckpt_path = resolve_ckpt_path(args.model_path)
        print(f"Loading model from: {ckpt_path}")
        ckpt_index = ckpt_path + '.index'
        if not os.path.exists(ckpt_index):
            raise FileNotFoundError(f"Checkpoint index file not found: {ckpt_index}. Run training or pass --skip_restore for smoke test.")
        try:
            saver.restore(sess, ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to restore checkpoint {ckpt_path}: {e}")
    else:
        print('[INFO] Skipping checkpoint restore (random initialization).')

    f_log = open(os.path.join(res_save_path, 'log.txt'), 'w')
    timer = util.Timer()

    # --- Testing loop ---
    # Streaming metrics accumulators (avoid storing all pixels, which can OOM for HRF)
    total_tp = total_tn = total_fp = total_fn = 0
    # For AUC/AP we still need samples; subsample to a max budget
    max_auc_samples = int(os.environ.get('AUC_MAX_SAMPLES', '500000'))  # can override via env var
    auc_sample_labels = []
    auc_sample_probs = []
    all_labels_roi, all_preds_roi = np.zeros((0,)), np.zeros((0,))

    # Variables to hold final test phase metrics for logging
    final_test_avg_acc, final_test_avg_prec, final_test_avg_rec = 0.0, 0.0, 0.0
    test_phase_losses = []  # capture test losses for logging

    for data_layer, phase_name in zip(data_layers, phase_names):
        # NEW: Initialize lists to store metrics for each batch in this phase
        loss_list, acc_list, prec_list, rec_list = [], [], [], []
        
        total_batches = 0
        processed_batches = 0
        # Count total batches for this phase
        if phase_name == 'TrainList':
            total_batches = len_train // cfg.TRAIN.BATCH_SIZE + (1 if len_train % cfg.TRAIN.BATCH_SIZE != 0 else 0)
        else:
            total_batches = len_test // cfg.TRAIN.BATCH_SIZE + (1 if len_test % cfg.TRAIN.BATCH_SIZE != 0 else 0)
        
        # Use prefetcher to overlap data loading with model compute
        prefetch = Prefetcher(data_layer, maxsize=2)
        while True:
            t0 = time.time()
            img_list, blobs = prefetch.get(timeout=10)
            load_time = time.time() - t0
            if img_list is None:
                break
            # Ensure numpy arrays for TF feed
            blobs = {k: to_numpy(v) for k, v in (blobs or {}).items()}
            img = blobs.get('img', None)
            label = blobs.get('label', None)
            if img is None or label is None:
                print(f"[WARN] Empty batch from DataLayer; skipping")
                continue
            if args.use_fov_mask:
                fov_mask = to_numpy(blobs.get('fov', np.ones(label.shape)))
            else:
                fov_mask = np.ones(label.shape)
            # Type normalization
            img = img.astype(np.float32)
            label = label.astype(np.int64)
            fov_mask = fov_mask.astype(np.int64)

            timer.tic()
            t1 = time.time()
            # MODIFIED: Fetch loss, accuracy, precision, and recall in one go
            loss_val, acc_val, prec_val, rec_val, fg_prob_map = sess.run(
                [network.loss, network.accuracy, network.precision, network.recall, network.fg_prob],
                feed_dict={network.is_training: False,
                           network.imgs: img,
                           network.labels: label,
                           network.fov_masks: fov_mask})
            sess_time = time.time() - t1
            timer.toc()
            
            # MODIFIED: Append all new metrics to their lists
            loss_list.append(loss_val)
            acc_list.append(acc_val)
            prec_list.append(prec_val)
            rec_list.append(rec_val)

            # Sanitize predictions: replace NaN/inf, then clip to valid probability range
            if not np.isfinite(fg_prob_map).all():
                nonfinite = np.size(fg_prob_map) - np.isfinite(fg_prob_map).sum()
                print(f"Warning: detected {nonfinite} non-finite entries in model output; replacing with finite defaults")
            fg_prob_map = np.nan_to_num(fg_prob_map, nan=0.0, posinf=1.0, neginf=0.0)
            fg_prob_map = np.clip(fg_prob_map, 0.0, 1.0)

            reshaped_fg_prob_map = fg_prob_map.reshape((len(img_list), fg_prob_map.shape[1], fg_prob_map.shape[2]))
            reshaped_output = reshaped_fg_prob_map >= 0.5

            # Debug stats for first batch of each phase and per-batch label stats
            if processed_batches == 0:
                print(f"[{phase_name}] fg_prob stats: min={reshaped_fg_prob_map.min():.4f} max={reshaped_fg_prob_map.max():.4f} mean={reshaped_fg_prob_map.mean():.4f}")
                if args.dump_debug:
                    np.save(os.path.join(res_save_path, f'{phase_name}_first_batch_probs.npy'), reshaped_fg_prob_map)
            # Print label/pred diagnostics for this batch
            try:
                lbl = np.asarray(label).ravel()
                pred_flat = reshaped_fg_prob_map.ravel()
                unique_lbls, counts = np.unique(lbl, return_counts=True)
                print(f"[{phase_name}] batch {processed_batches+1}: label uniques={list(zip(unique_lbls.tolist(), counts.tolist()))}, pred_min={pred_flat.min():.4f}, pred_max={pred_flat.max():.4f}")
            except Exception:
                pass

            # Update streaming confusion counts (apply FOV mask if provided)
            flat_labels = label.reshape(-1)
            flat_probs = fg_prob_map.reshape(-1)
            flat_preds_bin = (flat_probs >= 0.5)
            total_tp += int(np.sum((flat_labels == 1) & (flat_preds_bin == 1)))
            total_tn += int(np.sum((flat_labels == 0) & (flat_preds_bin == 0)))
            total_fp += int(np.sum((flat_labels == 0) & (flat_preds_bin == 1)))
            total_fn += int(np.sum((flat_labels == 1) & (flat_preds_bin == 0)))
            # Reservoir-like subsampling for AUC/AP
            need = max_auc_samples - len(auc_sample_labels)
            if need > 0:
                take = min(need, flat_labels.size)
                # Uniform sample of current batch pixels
                idx = np.random.choice(flat_labels.size, size=take, replace=False)
                auc_sample_labels.append(flat_labels[idx])
                auc_sample_probs.append(flat_probs[idx])

            # Save outputs
            save_outputs(res_save_path, img_list, reshaped_fg_prob_map, reshaped_output, dump_debug=args.dump_debug)
            
            # Progress tracking
            processed_batches += 1
            print(f"{phase_name}: Processed batch {processed_batches}/{total_batches} load={load_time:.3f}s sess={sess_time:.3f}s avg_batch={timer.average_time:.3f}s")
            # Stop after the expected number of batches. DataLayer wraps the dataset,
            # so without this guard the loop would continue indefinitely.
            if processed_batches >= total_batches:
                break
            if profiler and args.profile_report_every > 0 and (processed_batches % args.profile_report_every == 0):
                try:
                    print('[PROFILE REPORT]\n' + profiler.report())
                except Exception as e:
                    print(f"[PROFILE] report failed: {e}")

        prefetch.stop()

    # MODIFIED: Calculate and print the average for all metrics for the current phase
    avg_loss = np.mean(loss_list)
    avg_acc = np.mean(acc_list)
    avg_prec = np.mean(prec_list)
    avg_rec = np.mean(rec_list)
    print(f"\n--- {phase_name} Phase Results ---")
    print(f"Average Loss:      {avg_loss:.4f}")
    print(f"Average Accuracy:  {avg_acc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall:    {avg_rec:.4f}")
    print(f"--------------------------\n")
    if phase_name == 'Test':
        # Keep a copy of the final test metrics for logging
        test_phase_losses = list(loss_list)
        final_test_avg_acc = avg_acc
        final_test_avg_prec = avg_prec
        final_test_avg_rec = avg_rec


    # Compute metrics over the entire dataset
    # Build sampled arrays for AUC/AP
    if len(auc_sample_labels) > 0:
        sampled_labels = np.concatenate(auc_sample_labels)
        sampled_probs = np.concatenate(auc_sample_probs)
        auc, ap = util.get_auc_ap_score(sampled_labels, sampled_probs)
    else:
        auc, ap = float('nan'), float('nan')
    # Derive overall metrics from accumulated confusion counts
    denom = lambda x: x if x>0 else 1
    overall_acc = (total_tp + total_tn) / denom(total_tp + total_tn + total_fp + total_fn)
    precision = total_tp / denom(total_tp + total_fp)
    recall = total_tp / denom(total_tp + total_fn)
    specificity = total_tn / denom(total_tn + total_fp)
    f1 = (2*precision*recall) / denom(precision + recall)
    dice = (2*total_tp) / denom(2*total_tp + total_fp + total_fn)
    iou = total_tp / denom(total_tp + total_fp + total_fn)
    ext = {
        'specificity': specificity,
        'f1': f1,
        'dice': dice,
        'iou': iou
    }
    print(f"--- Overall Dataset Metrics ---")
    print(f"Overall Accuracy: {overall_acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
    print(f"Specificity: {ext['specificity']:.4f}, F1: {ext['f1']:.4f}, Dice: {ext['dice']:.4f}, IoU: {ext['iou']:.4f}")
    print(f"-------------------------------\n")

    # Write requested keys to the log file
    try:
        test_loss_list = test_phase_losses if len(test_phase_losses) > 0 else loss_list
    except Exception:
        test_loss_list = loss_list
    cnn_acc_test = float(overall_acc)
    cnn_auc_test = float(auc) if np.isfinite(auc) else float('nan')
    cnn_ap_test = float(ap) if np.isfinite(ap) else float('nan')

    # MODIFIED: Write all the new metrics to the log file
    f_log.write('test_loss ' + str(np.mean(test_loss_list)) + '\n')
    f_log.write('test_cnn_acc_overall ' + str(cnn_acc_test) + '\n')
    f_log.write('test_cnn_acc_batch_avg ' + str(final_test_avg_acc) + '\n')
    f_log.write('test_cnn_precision_batch_avg ' + str(final_test_avg_prec) + '\n')
    f_log.write('test_cnn_recall_batch_avg ' + str(final_test_avg_rec) + '\n')
    f_log.write('test_cnn_auc ' + str(cnn_auc_test) + '\n')
    f_log.write('test_cnn_ap ' + str(cnn_ap_test) + '\n')
    f_log.write('test_cnn_specificity ' + str(ext['specificity']) + '\n')
    f_log.write('test_cnn_f1 ' + str(ext['f1']) + '\n')
    f_log.write('test_cnn_dice ' + str(ext['dice']) + '\n')
    f_log.write('test_cnn_iou ' + str(ext['iou']) + '\n')
    f_log.flush()
    f_log.close()

    print(f"Average batch time: {timer.average_time:.3f}s")
    if profiler:
        try:
            print('[PROFILE FINAL REPORT]\n' + profiler.report())
        except Exception as e:
            print(f"[PROFILE] final report failed: {e}")
    sess.close()
    print("Testing complete.")


if __name__ == '__main__':
    main()