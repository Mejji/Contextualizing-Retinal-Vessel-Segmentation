#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_VGN_DAU.py — PyTorch training script for VGN (optimized for high-end GPUs)

Key points:
  * Uses util_vgn.TorchDataLayer (pure PyTorch, no TF).
  * Optional down-scaling with --resize_to (for HRF default is 1024).
  * Fast graph construction: edge_mode='grid' or 'euclidean' (no geodesic).
  * Per-image graph caching so each graph is built only once.
  * Training+eval on CUDA if available. Heavy geometric augs are disabled
    to keep graph caching valid.
"""

import os
import argparse
import json
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F

from config_vgn import cfg
import util_vgn as util
from Modules.model_vgn import VGN


# ----------------------------- Args -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train VGN (PyTorch) with on-the-fly graph + CNN backbone')

    # Dataset / split
    p.add_argument('--dataset', default='HRF', type=str,
                   choices=['DRIVE', 'STARE', 'CHASE_DB1', 'HRF'])
    p.add_argument('--use_fov_mask', type=bool, default=True)
    p.add_argument('--eval_split', default='test', choices=['test', 'val', 'none'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--limit_train_images', type=int, default=0)

    # Model / loss
    p.add_argument('--cnn_loss_on', action='store_true', default=True,
                   help='If set, also supervise the CNN head p_cnn in addition to p_vgn.')
    p.add_argument('--cnn_loss_weight', type=float, default=0.5,
                   help='Weight of CNN loss relative to VGN loss.')

    # Graph options
    p.add_argument('--edge_mode', type=str, default='grid',
                   choices=['grid', 'euclidean'],
                   help="Graph connectivity: 'grid' (fast) or 'euclidean' (k-NN with cdist).")
    p.add_argument('--delta', type=int, default=8,
                   help='SRVS sampling stride; larger -> fewer vertices (default 8).')
    p.add_argument('--geo_thresh', type=float, default=10.0,
                   help='Kept for API compatibility; not used in grid/euclidean modes.')

    # Optional down-scaling
    p.add_argument('--resize_to', type=int, default=0,
                   help='Downscale so that max(H,W) <= resize_to. '
                        '0 = no resizing. For HRF, 1024 is a good default.')

    # Optim / LR
    p.add_argument('--opt', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_iters', type=int, default=cfg.MAX_ITERS)

    # Save / runs
    p.add_argument('--save_root', default='/workspace/VGN', type=str)
    p.add_argument('--run_id', type=int, default=1)
    p.add_argument('--run_name', type=str, default='')

    # Logging / eval cadence
    p.add_argument('--display', type=int, default=cfg.TRAIN.DISPLAY)
    p.add_argument('--test_iters', type=int, default=cfg.TRAIN.TEST_ITERS)
    p.add_argument('--snapshot_iters', type=int, default=cfg.TRAIN.SNAPSHOT_ITERS)

    # Device
    p.add_argument('--device', type=str, default='cuda',
                   help="Device to use: 'cuda' or 'cpu' (default: 'cuda' if available).")

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


def _format_eta(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _compute_seg_metrics(labels_flat, probs_flat, thr=0.5):
    """
    labels_flat: 1D np.array of {0,1}
    probs_flat : 1D np.array of [0,1]
    """
    labels = labels_flat.astype(np.int64).ravel()
    probs = probs_flat.astype(np.float32).ravel()
    preds = probs >= thr
    lblb = labels.astype(bool)

    tp = np.sum(np.logical_and(preds, lblb))
    tn = np.sum(np.logical_and(~preds, ~lblb))
    fp = np.sum(np.logical_and(preds, ~lblb))
    fn = np.sum(np.logical_and(~preds, lblb))

    denom = lambda x: x if x > 0 else 1.0
    acc = (tp + tn) / denom(tp + tn + fp + fn)
    dice = (2 * tp) / denom(2 * tp + fp + fn)
    iou = tp / denom(tp + fp + fn)

    try:
        auc, ap = util.get_auc_ap_score(labels.astype(np.float32), probs.astype(np.float32))
    except Exception:
        auc, ap = np.nan, np.nan

    return dict(accuracy=acc, dice=dice, iou=iou, auc=auc, ap=ap)


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


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    # cuDNN autotune for speed
    torch.backends.cudnn.benchmark = True

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if args.device in ['cuda', 'cpu'] else 'cuda')
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')

    print(f"[INFO] Using device: {device}")

    # HRF-specific default resize if user didn't specify
    if args.resize_to <= 0 and args.dataset.upper() == 'HRF':
        args.resize_to = 1024
    if args.resize_to > 0:
        print(f"[INFO] Will downscale batches so max(H,W) <= {args.resize_to}")

    # Disable heavy geometric augmentations so graph caching stays valid
    for name in ['USE_LR_FLIPPED', 'USE_UD_FLIPPED', 'USE_ROTATION',
                 'USE_SCALING', 'USE_CROPPING']:
        if hasattr(cfg.TRAIN, name):
            setattr(cfg.TRAIN, name, False)

    # Dataset splits
    train_txt, test_txt = _get_split_txt_paths(args.dataset)
    with open(train_txt) as f:
        train_img_names = [x.strip() for x in f.readlines() if x.strip()]
    with open(test_txt) as f:
        test_img_names = [x.strip() for x in f.readlines() if x.strip()]

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
    elif args.eval_split == 'test':
        eval_img_names = test_img_names
        eval_tag = 'test'
    else:
        eval_img_names = []
        eval_tag = 'none'

    print(f"[DATA] Train: {len(train_img_names)} images; Eval({eval_tag}): {len(eval_img_names)} images")

    # Data layers
    dl_train = util.TorchDataLayer(train_img_names, is_training=True,
                                   use_padding=False, device=device)
    dl_eval = util.TorchDataLayer(eval_img_names, is_training=False,
                                  use_padding=False, device=device) if eval_img_names else None

    # Run paths
    run_name = args.run_name if args.run_name else f"VGN_DAU_run{int(args.run_id)}"
    run_root = os.path.join(args.save_root, run_name) if args.save_root else run_name
    os.makedirs(run_root, exist_ok=True)
    model_save_path = os.path.join(run_root, 'train')
    os.makedirs(model_save_path, exist_ok=True)
    eval_save_dir = os.path.join(run_root, eval_tag if eval_tag != 'none' else 'eval')
    if eval_tag != 'none':
        os.makedirs(eval_save_dir, exist_ok=True)

    profiler_csv_path = os.path.join(model_save_path, 'profiler.csv')
    if not os.path.exists(profiler_csv_path):
        with open(profiler_csv_path, 'w') as pf:
            pf.write('iter,step_time_s,avg_step_time_s,elapsed_s,loss,loss_vgn,loss_cnn,dice,auc\n')

    # Model
    model = VGN().to(device)
    print(f"[MODEL] VGN parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer
    weight_decay = getattr(cfg.TRAIN, 'WEIGHT_DECAY_RATE', 0.0)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        momentum = getattr(cfg.TRAIN, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )

    # Training loop
    print("[TRAIN] Starting training...")
    best_dice = -1.0
    timer = util.Timer()
    train_wall_start = time.time()

    # Graph cache: image_path -> graph snapshot
    graph_cache = {}

    for it in range(args.max_iters):
        model.train()
        timer.tic()

        img_list, blobs = dl_train.forward()
        imgs = blobs['img']        # [B,3,H,W]
        labels = blobs['label']    # [B,1,H,W]
        fovs = blobs['fov']        # [B,1,H,W]

        imgs, labels, fovs = _maybe_resize_batch(imgs, labels, fovs, args.resize_to)

        B = imgs.shape[0]

        optimizer.zero_grad()

        total_loss = 0.0
        total_loss_vgn = 0.0
        total_loss_cnn = 0.0

        # VGN assumes batch=1 internally; handle per-sample loop
        for b in range(B):
            x = imgs[b:b+1]                # [1,3,H,W]
            y = labels[b:b+1]              # [1,1,H,W]
            fov = fovs[b:b+1] if args.use_fov_mask else torch.ones_like(y)
            img_id = img_list[b]

            cache = graph_cache.get(img_id, None)
            out = model(x,
                        cache_graph=cache,
                        delta=args.delta,
                        geo_thresh=args.geo_thresh,
                        edge_mode=args.edge_mode)
            if cache is None:
                graph_cache[img_id] = out['graph']

            p_cnn = out['p_cnn']           # [1,1,H,W]
            p_vgn = out['p_vgn']           # [1,1,H,W]

            target = y.float()             # [1,1,H,W]
            fov_mask = fov.float()         # [1,1,H,W]

            # VGN loss (final head)
            loss_vgn_map = F.binary_cross_entropy(p_vgn, target, reduction='none')
            loss_vgn = (loss_vgn_map * fov_mask).sum() / (fov_mask.sum() + 1e-8)

            # Optional CNN loss
            if args.cnn_loss_on:
                loss_cnn_map = F.binary_cross_entropy(p_cnn, target, reduction='none')
                loss_cnn = (loss_cnn_map * fov_mask).sum() / (fov_mask.sum() + 1e-8)
            else:
                loss_cnn = torch.tensor(0.0, device=device)

            loss = loss_vgn + args.cnn_loss_weight * loss_cnn
            loss = loss / B  # normalize over batch
            loss.backward()

            total_loss += float(loss.detach().cpu().item())
            total_loss_vgn += float(loss_vgn.detach().cpu().item())
            total_loss_cnn += float(loss_cnn.detach().cpu().item())

        optimizer.step()
        step_time = timer.toc(average=False)
        avg_step = timer.average_time
        elapsed_s = time.time() - train_wall_start

        # Simple train metrics from the last sample of this batch
        with torch.no_grad():
            probs_last = p_vgn.detach().cpu().numpy().ravel()
            labels_last = labels[-1].detach().cpu().numpy().ravel()
            if args.use_fov_mask:
                fov_last = (fovs[-1].detach().cpu().numpy().ravel() > 0)
            else:
                fov_last = np.ones_like(labels_last, dtype=bool)
            labels_masked = labels_last[fov_last]
            probs_masked = probs_last[fov_last]
            train_metrics = _compute_seg_metrics(labels_masked, probs_masked, thr=0.5)

        if (it + 1) % args.display == 0:
            eta_seconds = (args.max_iters - (it + 1)) * max(avg_step, 0.0)
            eta_str = _format_eta(eta_seconds)
            print(f"iter: {it+1}/{args.max_iters}, "
                  f"loss: {total_loss:.4f} (vgn {total_loss_vgn:.4f}, cnn {total_loss_cnn:.4f}), "
                  f"dice: {train_metrics['dice']:.4f}, "
                  f"step: {avg_step:.3f}s, eta: {eta_str}")

        # Write profiler line
        try:
            with open(profiler_csv_path, 'a') as pf:
                pf.write(f"{it+1},{step_time:.6f},{avg_step:.6f},{elapsed_s:.2f},"
                         f"{total_loss:.6f},{total_loss_vgn:.6f},{total_loss_cnn:.6f},"
                         f"{train_metrics['dice']:.6f},{train_metrics['auc']:.6f}\n")
        except Exception:
            pass

        # Snapshot
        if (it + 1) % args.snapshot_iters == 0:
            ck = os.path.join(model_save_path, f'iter_{it+1}.pth')
            torch.save({
                'iter': it + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, ck)
            print(f'[CKPT] Wrote snapshot: {ck}')

        # Periodic eval
        if dl_eval is not None and (it + 1) % args.test_iters == 0:
            model.eval()
            all_labels = []
            all_probs = []

            num_eval_batches = int(np.ceil(len(eval_img_names) / cfg.TRAIN.BATCH_SIZE)) if len(eval_img_names) > 0 else 0
            for _ in range(num_eval_batches):
                img_list_e, blobs_e = dl_eval.forward()
                if len(img_list_e) == 0:
                    continue
                imgs_e = blobs_e['img']      # [B,3,H,W]
                labels_e = blobs_e['label']  # [B,1,H,W]
                fovs_e = blobs_e['fov']      # [B,1,H,W]

                imgs_e, labels_e, fovs_e = _maybe_resize_batch(imgs_e, labels_e, fovs_e, args.resize_to)
                B_e = imgs_e.shape[0]

                with torch.no_grad():
                    for b in range(B_e):
                        x_e = imgs_e[b:b+1]
                        y_e = labels_e[b:b+1]
                        fov_e = fovs_e[b:b+1] if args.use_fov_mask else torch.ones_like(y_e)
                        img_id_e = img_list_e[b]

                        cache_e = graph_cache.get(img_id_e, None)
                        out_e = model(x_e,
                                      cache_graph=cache_e,
                                      delta=args.delta,
                                      geo_thresh=args.geo_thresh,
                                      edge_mode=args.edge_mode)
                        if cache_e is None:
                            graph_cache[img_id_e] = out_e['graph']

                        p_vgn_e = out_e['p_vgn']  # [1,1,H,W]

                        probs_e = p_vgn_e.squeeze(0).squeeze(0).cpu().numpy().ravel()
                        labels_np = y_e.squeeze(0).cpu().numpy().ravel()
                        if args.use_fov_mask:
                            fov_np = (fov_e.squeeze(0).cpu().numpy().ravel() > 0)
                        else:
                            fov_np = np.ones_like(labels_np, dtype=bool)

                        all_labels.append(labels_np[fov_np])
                        all_probs.append(probs_e[fov_np])

            if all_labels:
                labels_cat = np.concatenate(all_labels)
                probs_cat = np.concatenate(all_probs)
                eval_metrics = _compute_seg_metrics(labels_cat, probs_cat, thr=0.5)
                print(f"[EVAL-{eval_tag.upper()}] iter {it+1}/{args.max_iters}, "
                      f"dice: {eval_metrics['dice']:.4f}, "
                      f"acc: {eval_metrics['accuracy']:.4f}, "
                      f"auc: {eval_metrics['auc']:.4f}, "
                      f"ap: {eval_metrics['ap']:.4f}, "
                      f"iou: {eval_metrics['iou']:.4f}")

                # Save best dice checkpoint
                if eval_metrics['dice'] > best_dice:
                    best_dice = float(eval_metrics['dice'])
                    best_path = os.path.join(model_save_path, 'best.pth')
                    torch.save({
                        'iter': it + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': vars(args),
                        'metrics': eval_metrics,
                    }, best_path)
                    print(f"[CKPT] Saved new best model to {best_path} (dice={best_dice:.4f})")

                    # Write metadata
                    try:
                        best_meta = dict(run=run_name, type='iteration', split=eval_tag,
                                         best_iter=int(it + 1), metric='dice',
                                         metric_value=best_dice,
                                         timestamp=datetime.now().isoformat())
                        with open(os.path.join(model_save_path, 'best_meta.json'), 'w') as fmeta:
                            json.dump(best_meta, fmeta, indent=2)
                    except Exception:
                        pass

            model.train()

    # Final snapshot
    final_path = os.path.join(model_save_path, f'iter_{args.max_iters}.pth')
    torch.save({
        'iter': args.max_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f'[CKPT] Wrote final snapshot: {final_path}')

    total_elapsed = time.time() - train_wall_start
    print('Training completed in {:.2f} seconds ({}).'.format(
        total_elapsed, str(timedelta(seconds=int(total_elapsed)))))


if __name__ == '__main__':
    main()
