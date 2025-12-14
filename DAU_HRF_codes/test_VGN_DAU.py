#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_VGN_DAU.py — PyTorch test script for VGN

- Uses the PyTorch VGN model (Modules.model_vgn.VGN).
- Uses util_vgn.TorchDataLayer to load DRIVE/STARE/CHASE_DB1/HRF.
- Loads a .pth checkpoint (model_path or from save_root/run_id).
- Evaluates both:
    * CNN head  -> p_cnn
    * VGN head  -> p_vgn
- Uses the same resize_to / delta as training by default
  (read from checkpoint['args']), unless overridden via CLI.
- For 'grid' graphs, delta is fixed (no auto-scaling).
  For 'euclidean', auto-delta is used if you don't override.
- Saves per-image VGN prob maps (.png + .npy) and a CSV of per-image metrics:
    image_id, delta_eff, accuracy, specificity, sensitivity, precision,
    f1, iou, miou, auc, ap, bce
"""

import os
import math
import argparse
from datetime import datetime
import csv

import numpy as np
import numpy as _np
import torch
import torch.nn.functional as F
import skimage.io
import torch.serialization

from config_vgn import cfg
import util_vgn as util
from Modules.model_vgn import VGN


# ----------------------------- Helpers -----------------------------
def _get_split_txt_paths(dataset: str):
    dataset = dataset.upper()
    if dataset == 'DRIVE':
        return cfg.TRAIN.DRIVE_SET_TXT_PATH, cfg.TEST.DRIVE_SET_TXT_PATH
    if dataset == 'STARE':
        return cfg.TRAIN.STARE_SET_TXT_PATH, cfg.TEST.STARE_SET_TXT_PATH
    if dataset == 'CHASE_DB1':
        return cfg.TRAIN.CHASE_DB1_SET_TXT_PATH, cfg.TEST.CHASE_DB1_SET_TXT_PATH
    if dataset == 'HRF':
        return cfg.TRAIN.HRF_SET_TXT_PATH, cfg.TEST.HRF_SET_TXT_PATH
    raise ValueError(f"Unknown dataset: {dataset}")


def _compute_seg_metrics(labels_flat, probs_flat, thr=0.5):
    """
    labels_flat: 1D np.array of {0,1}
    probs_flat : 1D np.array of [0,1]

    Returns:
        accuracy, specificity, sensitivity, precision,
        dice (F1), iou, miou, auc, ap, bce
    """
    eps = 1e-8

    labels = np.asarray(labels_flat).astype(np.float32).ravel()
    probs = np.asarray(probs_flat).astype(np.float32).ravel()

    # Ensure binary labels {0,1}
    labels_bin = (labels >= 0.5).astype(np.int64)
    preds = probs >= float(thr)
    lblb = labels_bin.astype(bool)

    tp = np.sum(np.logical_and(preds, lblb))
    tn = np.sum(np.logical_and(~preds, ~lblb))
    fp = np.sum(np.logical_and(preds, ~lblb))
    fn = np.sum(np.logical_and(~preds, lblb))

    denom = lambda x: x if x > 0 else 1.0

    acc  = (tp + tn) / denom(tp + tn + fp + fn)
    prec = tp / denom(tp + fp)
    rec  = tp / denom(tp + fn)  # sensitivity / recall
    spec = tn / denom(tn + fp)
    dice = (2 * tp) / denom(2 * tp + fp + fn)
    iou  = tp / denom(tp + fp + fn)
    miou = iou  # binary foreground IoU

    # BCE with clamping
    probs_clamped = np.clip(probs, eps, 1.0 - eps)
    bce = -np.mean(labels_bin * np.log(probs_clamped) +
                   (1.0 - labels_bin) * np.log(1.0 - probs_clamped))

    # AUC & AP (may fail if labels are constant)
    try:
        auc, ap = util.get_auc_ap_score(labels_bin.astype(np.float32),
                                        probs.astype(np.float32))
    except Exception:
        auc, ap = np.nan, np.nan

    return dict(
        accuracy=acc,
        specificity=spec,
        sensitivity=rec,
        precision=prec,
        dice=dice,
        iou=iou,
        miou=miou,
        auc=auc,
        ap=ap,
        bce=bce,
    )


def _resolve_device(arg_device: str):
    if arg_device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        return torch.device('cpu')
    device = torch.device(arg_device if arg_device in ['cuda', 'cpu'] else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    return device


def _estimate_num_verts(H, W, delta):
    cell = 2 * int(delta)
    Hc = math.ceil(H / cell)
    Wc = math.ceil(W / cell)
    return Hc * Wc


def _auto_delta(H, W, base_delta=4, max_verts=40000):
    """
    Pick a delta such that SRVS vertex count V ~= Hc*Wc stays under max_verts.
    Intended for 'euclidean' mode where torch.cdist can blow up.
    """
    V0 = _estimate_num_verts(H, W, base_delta)
    if V0 <= max_verts:
        return base_delta, V0

    # Solve cell^2 >= (H*W)/max_verts  =>  delta >= 0.5 * sqrt((H*W)/max_verts)
    cell_needed = math.sqrt((H * W) / float(max_verts))
    delta_needed = int(math.ceil(cell_needed / 2.0))
    delta_eff = max(base_delta, delta_needed)
    V_eff = _estimate_num_verts(H, W, delta_eff)
    return delta_eff, V_eff


def _maybe_resize_batch(imgs, labels, fovs, target_long_side):
    """
    Downscale batch so that max(H,W) <= target_long_side (keeping aspect ratio).
    If target_long_side <= 0, returns tensors unchanged.
    """
    if target_long_side is None or target_long_side <= 0:
        return imgs, labels, fovs

    B, C, H, W = imgs.shape
    cur_long = max(H, W)
    if cur_long <= target_long_side:
        return imgs, labels, fovs

    scale = float(target_long_side) / float(cur_long)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    imgs_res = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    labels_res = F.interpolate(labels.float(), size=(new_h, new_w), mode='nearest').long()
    fovs_res = F.interpolate(fovs.float(), size=(new_h, new_w), mode='nearest').long()
    return imgs_res, labels_res, fovs_res


# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Test VGN (PyTorch) on DRIVE/STARE/CHASE_DB1/HRF')

    # Dataset / split
    p.add_argument('--dataset', default='HRF', type=str,
                   choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--use_fov_mask', type=bool, default=True)

    # Random
    p.add_argument('--seed', type=int, default=1337)

    # Run / checkpoint
    p.add_argument('--save_root', type=str, default='/workspace/VGN',
                   help='Root where training runs are stored (same as train_VGN_DAU.py).')
    p.add_argument('--run_id', type=int, default=1,
                   help='Run ID (used if run_name is empty).')
    p.add_argument('--run_name', type=str, default='',
                   help='If empty, uses VGN_DAU_run{run_id}.')
    p.add_argument('--model_path', type=str, default='',
                   help='Full path to .pth checkpoint. If empty, use save_root/run/train/ckpt.')
    p.add_argument('--ckpt', type=str, default='best.pth',
                   help='Checkpoint file name under <save_root>/<run_name>/train if model_path is empty.')

    # Graph / GAT
    p.add_argument('--edge_mode', type=str, default='grid',
                   choices=['grid', 'euclidean'],
                   help='Edge construction mode for the graph (match training; grid is fastest).')
    p.add_argument('--base_delta', type=int, default=4,
                   help='Base SRVS sampling delta (used only if we need auto-delta).')
    p.add_argument('--max_verts', type=int, default=40000,
                   help='Max allowed SRVS vertices for auto-delta (euclidean mode).')
    p.add_argument('--delta', type=int, default=None,
                   help='If set, use this SRVS delta directly (no auto-delta).')

    # Device
    p.add_argument('--device', type=str, default='cuda',
                   help="Device to use: 'cuda' or 'cpu'.")

    # Output
    p.add_argument('--results_subdir', type=str, default='test_results',
                   help='Subdirectory inside the run folder to save outputs.')

    # Resize (test-time)
    p.add_argument('--resize_to', type=int, default=-1,
                   help='Test-time max(H,W). '
                        '-1: use value stored in checkpoint (training). '
                        ' 0: no resizing (native). '
                        '>0: explicit long-side.')

    return p.parse_args()


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # cuDNN autotune for speed at fixed resolution
    torch.backends.cudnn.benchmark = True

    # Device
    device = _resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Dataset splits
    dataset_key = args.dataset.upper()
    _, test_txt = _get_split_txt_paths(dataset_key)
    with open(test_txt) as f:
        test_img_names = [x.strip() for x in f.readlines() if x.strip()]
    print(f"[DATA] Test set: {len(test_img_names)} images")

    # Data layer
    dl_test = util.TorchDataLayer(test_img_names, is_training=False,
                                  use_padding=False, device=device)

    # Run / paths
    run_name = args.run_name if args.run_name else f"VGN_DAU_run{int(args.run_id)}"
    run_root = os.path.join(args.save_root, run_name) if args.save_root else run_name
    model_dir = os.path.join(run_root, 'train')

    # Resolve checkpoint path
    if args.model_path:
        ckpt_path = args.model_path
    else:
        ckpt_path = os.path.join(model_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[CKPT] Loading checkpoint: {ckpt_path}")

    # Results dir
    results_root = os.path.join(run_root, args.results_subdir, dataset_key)
    os.makedirs(results_root, exist_ok=True)
    log_path = os.path.join(results_root, 'log.txt')
    with open(log_path, 'w') as f_log:
        f_log.write(f"Test run at {datetime.now().isoformat()}\n")
        f_log.write(str(args) + "\n")
        f_log.write(f"Checkpoint: {ckpt_path}\n")

    # Model
    model = VGN().to(device)
    torch.serialization.add_safe_globals([_np.core.multiarray.scalar])

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[CKPT] Loaded model. Stored iter={checkpoint.get('iter', 'unknown')}")
        if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
            train_args = checkpoint['args']
    else:
        model.load_state_dict(checkpoint)
        print("[CKPT] Loaded raw state_dict (no metadata)")

    model.eval()

    # Pull training-time delta / resize_to if available
    train_delta = None
    train_resize_to = 0
    if train_args is not None:
        train_delta = train_args.get('delta', None)
        train_resize_to = train_args.get('resize_to', 0)

    # Effective test-time resize_to
    if args.resize_to == -1:
        resize_to = train_resize_to
    else:
        resize_to = max(0, int(args.resize_to))

    print(f"[INFO] Test resize_to={resize_to} "
          f"(training resize_to={train_resize_to})")

    # Effective delta base
    if args.delta is not None:
        delta_base = int(args.delta)
        print(f"[INFO] Using CLI delta={delta_base}")
    elif train_delta is not None:
        delta_base = int(train_delta)
        print(f"[INFO] Using training delta={delta_base}")
    else:
        delta_base = int(args.base_delta)
        print(f"[INFO] Using base_delta={delta_base} (no delta in checkpoint)")

    # Eval loop
    batch_size = getattr(cfg.TRAIN, 'BATCH_SIZE', 1)
    num_batches = int(np.ceil(float(len(test_img_names)) / max(1, batch_size)))
    print(f"[EVAL] Running {num_batches} batches (batch_size={batch_size})")

    all_labels = []
    all_probs_vgn = []
    all_probs_cnn = []

    per_image_rows = []

    # cache for per-(H,W) delta logging
    delta_cache = {}

    for _ in range(num_batches):
        img_list, blobs = dl_test.forward()
        if not img_list:
            continue

        imgs = blobs['img']        # [B,3,H,W]
        labels = blobs['label']    # [B,1,H,W]
        fovs = blobs['fov']        # [B,1,H,W]

        # Match training scale by default
        imgs, labels, fovs = _maybe_resize_batch(imgs, labels, fovs, resize_to)

        B = imgs.shape[0]

        with torch.no_grad():
            for b in range(B):
                x = imgs[b:b+1]              # [1,3,H,W]
                y = labels[b:b+1]            # [1,1,H,W]
                fov = fovs[b:b+1] if args.use_fov_mask else torch.ones_like(y)

                H, W = x.shape[-2:]
                hw_key = (H, W)

                if hw_key in delta_cache:
                    delta_eff = delta_cache[hw_key]
                else:
                    if args.edge_mode == 'grid':
                        # Just use the training/base delta; grid is cheap.
                        delta_eff = delta_base
                        V_est = _estimate_num_verts(H, W, delta_eff)
                        print(f"[GRAPH] GRID: delta={delta_eff} for {H}x{W} (V≈{V_est})")
                    else:
                        # euclidean: possibly heavy, so run auto-delta
                        delta_eff, V_est = _auto_delta(
                            H, W,
                            base_delta=delta_base,
                            max_verts=args.max_verts
                        )
                        if delta_eff != delta_base:
                            print(f"[GRAPH] EUCLIDEAN auto-delta: base={delta_base} -> {delta_eff} "
                                  f"for {H}x{W} (V≈{V_est})")
                        else:
                            print(f"[GRAPH] EUCLIDEAN: using delta={delta_eff} "
                                  f"for {H}x{W} (V≈{V_est})")
                    delta_cache[hw_key] = delta_eff

                out = model(x, delta=delta_eff, edge_mode=args.edge_mode)
                p_cnn = out['p_cnn']         # [1,1,H,W]
                p_vgn = out['p_vgn']         # [1,1,H,W]

                # --- flatten inside FOV for metrics ---
                probs_cnn_np = p_cnn.squeeze(0).squeeze(0).cpu().numpy().ravel()
                probs_vgn_np = p_vgn.squeeze(0).squeeze(0).cpu().numpy().ravel()
                labels_np = y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32).ravel()
                if args.use_fov_mask:
                    fov_np = (fov.squeeze(0).squeeze(0).cpu().numpy().astype(bool).ravel())
                else:
                    fov_np = np.ones_like(labels_np, dtype=bool)

                labels_masked = labels_np[fov_np]
                probs_cnn_masked = probs_cnn_np[fov_np]
                probs_vgn_masked = probs_vgn_np[fov_np]

                all_labels.append(labels_masked)
                all_probs_cnn.append(probs_cnn_masked)
                all_probs_vgn.append(probs_vgn_masked)

                # --- per-image metrics ---
                metr_cnn = _compute_seg_metrics(labels_masked, probs_cnn_masked, thr=0.5)
                metr_vgn = _compute_seg_metrics(labels_masked, probs_vgn_masked, thr=0.5)

                img_path = str(img_list[b])
                base = os.path.splitext(os.path.basename(img_path))[0]

                # --- save VGN prob maps only (CNN maps skipped for speed) ---
                vgn_map = p_vgn.squeeze(0).squeeze(0).cpu().numpy()

                np.save(os.path.join(results_root, base + '_vgn_prob.npy'),
                        vgn_map.astype(np.float32))

                try:
                    skimage.io.imsave(
                        os.path.join(results_root, base + '_vgn_prob.png'),
                        (np.clip(vgn_map, 0.0, 1.0) * 255.0).astype(np.uint8)
                    )
                except Exception:
                    pass

                # Per-image row for CSV (VGN head only)
                per_image_rows.append({
                    'image_id': base,
                    'delta_eff': int(delta_eff),
                    'accuracy': float(metr_vgn['accuracy']),
                    'specificity': float(metr_vgn['specificity']),
                    'sensitivity': float(metr_vgn['sensitivity']),
                    'precision': float(metr_vgn['precision']),
                    'f1': float(metr_vgn['dice']),
                    'iou': float(metr_vgn['iou']),
                    'miou': float(metr_vgn['miou']),
                    'auc': float(metr_vgn['auc']),
                    'ap': float(metr_vgn['ap']),
                    'bce': float(metr_vgn['bce']),
                })

                print(f"[IMG] {base}: "
                      f"CNN dice={metr_cnn['dice']:.4f}, "
                      f"VGN dice={metr_vgn['dice']:.4f}, "
                      f"VGN acc={metr_vgn['accuracy']:.4f}, "
                      f"delta={delta_eff}")

    # ---------------- Dataset-level metrics ----------------
    if not all_labels:
        raise RuntimeError("No test labels collected. Dataloader is doing nothing.")

    labels_cat = np.concatenate(all_labels)
    probs_cnn_cat = np.concatenate(all_probs_cnn)
    probs_vgn_cat = np.concatenate(all_probs_vgn)

    metr_cnn = _compute_seg_metrics(labels_cat, probs_cnn_cat, thr=0.5)
    metr_vgn = _compute_seg_metrics(labels_cat, probs_vgn_cat, thr=0.5)

    print(f"===== {dataset_key} TEST (FOV ROI) =====")
    print("[CNN] "
          f"Acc={metr_cnn['accuracy']:.4f}, "
          f"Dice={metr_cnn['dice']:.4f}, "
          f"IoU={metr_cnn['iou']:.4f}, "
          f"AUC={metr_cnn['auc']:.4f}, "
          f"AP={metr_cnn['ap']:.4f}, "
          f"BCE={metr_cnn['bce']:.4f}")

    print(f"----- {dataset_key} TEST (FOV ROI, VGN head) -----")
    print("Accuracy:      %.4f" % metr_vgn['accuracy'])
    print("Specificity:   %.4f" % metr_vgn['specificity'])
    print("Recall/Sens.:  %.4f" % metr_vgn['sensitivity'])
    print("F1/Dice:       %.4f" % metr_vgn['dice'])
    print("AUC:           %.4f" % metr_vgn['auc'])
    print("AP:            %.4f" % metr_vgn['ap'])
    print("Precision:     %.4f" % metr_vgn['precision'])
    print("IoU:           %.4f" % metr_vgn['iou'])
    print("mIoU:          %.4f" % metr_vgn['miou'])
    print("BCE:           %.4f" % metr_vgn['bce'])

    with open(log_path, 'a') as f_log:
        f_log.write("\n===== DATASET METRICS (CNN head, FOV ROI) =====\n")
        for k in ['accuracy', 'specificity', 'sensitivity',
                  'precision', 'dice', 'iou', 'miou', 'auc', 'ap', 'bce']:
            f_log.write(f"{k} {metr_cnn[k]:.6f}\n")

        f_log.write("\n===== DATASET METRICS (VGN head, FOV ROI) =====\n")
        f_log.write('Accuracy ' + str(metr_vgn['accuracy']) + '\n')
        f_log.write('Specificity ' + str(metr_vgn['specificity']) + '\n')
        f_log.write('Recall/Sensitivity ' + str(metr_vgn['sensitivity']) + '\n')
        f_log.write('F1_Dice ' + str(metr_vgn['dice']) + '\n')
        f_log.write('AUC ' + str(metr_vgn['auc']) + '\n')
        f_log.write('AP ' + str(metr_vgn['ap']) + '\n')
        f_log.write('Precision ' + str(metr_vgn['precision']) + '\n')
        f_log.write('IoU ' + str(metr_vgn['iou']) + '\n')
        f_log.write('mIoU ' + str(metr_vgn['miou']) + '\n')
        f_log.write('BCE ' + str(metr_vgn['bce']) + '\n')

    # Per-image CSV (VGN metrics)
    if per_image_rows:
        csv_path = os.path.join(results_root, 'per_image_metrics.csv')
        fieldnames = ['image_id', 'delta_eff',
                      'accuracy', 'specificity', 'sensitivity',
                      'precision', 'f1', 'iou', 'miou',
                      'auc', 'ap', 'bce']
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_image_rows:
                writer.writerow(row)
        print(f"[CSV] Per-image metrics saved to {csv_path}")

    print("[DONE] Testing complete.")


if __name__ == '__main__':
    main()
