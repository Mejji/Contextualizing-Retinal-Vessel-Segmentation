#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_CNN_fixed.py — Strictly wired to Shin's Train_CNN flow,
with NHWC safety and TF1-on-TF2 compatibility.

Paths/layout:
  models  -> <save_root>/<cfg.TRAIN.MODEL_SAVE_PATH>
  results -> <save_root>/<cfg.TEST.RES_SAVE_PATH>

Uses the vessel_segm_cnn API (imgs, labels, fov_masks, is_training,
loss/accuracy/precision/recall, fg_prob) defined in your model.  👇
"""

import os
import argparse
import numpy as np
import skimage.io
import tensorflow as tf

from config import cfg
import util

# ---- import vessel_segm_cnn irrespective of where you expose it ----
try:
    # Your repo variant: Modules/model.py
    from Modules.model import vessel_segm_cnn
except Exception:
    # Shin's original: model.py
    from Modules.model import vessel_segm_cnn

# ------------------------------ Args ------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description='Train a vessel_segm_cnn network (Shin-wired)')
    parser.add_argument('--dataset', default='DRIVE', type=str,
                        help='DRIVE | STARE | CHASE_DB1 | HRF')
    parser.add_argument('--cnn_model', default='driu', type=str,
                        help='driu | driu_large')
    parser.add_argument('--use_fov_mask', default=True, type=bool,
                        help='Whether to use FOV masks')
    parser.add_argument('--opt', default='adam', type=str,
                        help='sgd | adam')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='Initial LR (graph uses this)')
    parser.add_argument('--lr_decay', default='pc', type=str,
                        help='const | pc | exp (handled in model)')
    parser.add_argument('--max_iters', default=50000, type=int,
                        help='Maximum number of iterations')
    parser.add_argument('--pretrained_model', default='../pretrained_model/VGG_imagenet.npy', type=str,
                        help='Path to VGG .npy (dict style)')
    parser.add_argument('--save_root', default='DRIU_DRIVE', type=str,
                        help='Root to save models/results (Shin layout)')
    return parser.parse_args()

# ------------------------------ Utils ------------------------------ #
def _ensure_nhwc(arr):
    """If 4D and looks like NCHW (C in dim 1), transpose to NHWC."""
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[1] in (1, 3) and a.shape[-1] not in (1, 3):
        return np.transpose(a, (0, 2, 3, 1))
    return a

def load_pretrained(data_path, session, ignore_missing=True):
    """Load VGG-style .npy dict into TF1 variables (Shin loader, Py3-safe)."""
    data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item()
    for key in data_dict:
        with tf.compat.v1.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                target = 'W' if subkey == 'weights' else ('b' if subkey == 'biases' else None)
                if target is None:
                    continue
                try:
                    var = tf.compat.v1.get_variable(target)
                    session.run(var.assign(data_dict[key][subkey]))
                    print(f"assign pretrain model {subkey} to {key}")
                except ValueError:
                    print(f"ignore {key}/{subkey}")
                    if not ignore_missing:
                        raise

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

# ------------------------------ Main ------------------------------ #
if __name__ == '__main__':
    # TF1 behavior on TF2 runtimes
    tf.compat.v1.disable_eager_execution()

    args = parse_args()
    print('Called with args:')
    print(args)

    # --- dataset lists
    train_txt, test_txt = _get_split_txt_paths(args.dataset)
    with open(train_txt) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_txt) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    if args.dataset == 'HRF':
        # Same odd sampling Shin does
        test_img_names = [test_img_names[x] for x in range(7, len(test_img_names), 20)]

    len_train = len(train_img_names)
    len_test  = len(test_img_names)

    data_layer_train = util.DataLayer(train_img_names, is_training=True)
    data_layer_test  = util.DataLayer(test_img_names,  is_training=False)

    # --- Shin layout (no run subdir)
    model_save_path = (args.save_root + '/' + cfg.TRAIN.MODEL_SAVE_PATH) if len(args.save_root) > 0 else cfg.TRAIN.MODEL_SAVE_PATH
    res_save_path   = (args.save_root + '/' + cfg.TEST.RES_SAVE_PATH)   if len(args.save_root) > 0 else cfg.TEST.RES_SAVE_PATH

    if len(args.save_root) > 0 and not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.isdir(res_save_path):
        os.mkdir(res_save_path)

    # --- network (your TF1 graph, same API as Shin)
    network = vessel_segm_cnn(args, None)  # imgs/labels/fov_masks/is_training/loss/metrics present in your model.  # noqa
    #                                                                                                               ^^^
    #                                                                                         Required API confirmed. :contentReference[oaicite:5]{index=5}

    # --- session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    summary_writer = tf.compat.v1.summary.FileWriter(model_save_path, sess.graph)

    sess.run(tf.compat.v1.global_variables_initializer())
    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        print("Loading pretrained model...")
        load_pretrained(args.pretrained_model, sess, ignore_missing=True)

    f_log = open(os.path.join(model_save_path, 'log.txt'), 'w')
    last_snapshot_iter = -1
    timer = util.Timer()

    train_loss_list = []
    test_loss_list  = []

    print("Training the model...")
    for it in range(int(args.max_iters)):
        timer.tic()

        # ---- one batch
        _, blobs_train = data_layer_train.forward()
        imgs_np   = _ensure_nhwc(blobs_train['img']).astype(np.float32)
        labels_np = _ensure_nhwc(blobs_train['label']).astype(np.int64)
        if args.use_fov_mask:
            fov_np = _ensure_nhwc(blobs_train['fov']).astype(np.int64)
        else:
            fov_np = np.ones(labels_np.shape, dtype=np.int64)

        _, loss_val, acc_val, pre_val, rec_val = sess.run(
            [network.train_op, network.loss, network.accuracy, network.precision, network.recall],
            feed_dict={
                network.is_training: True,
                network.imgs:        imgs_np,
                network.labels:      labels_np,
                network.fov_masks:   fov_np
            })

        timer.toc()
        train_loss_list.append(loss_val)

        if (it + 1) % cfg.TRAIN.DISPLAY == 0:
            print('iter: %d / %d, loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' %
                  (it + 1, args.max_iters, loss_val, acc_val, pre_val, rec_val))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        # ---- snapshot
        if (it + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = it
            filename = os.path.join(model_save_path, f'iter_{it + 1}.ckpt')
            saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename))

        # ---- eval
        if (it + 1) % cfg.TRAIN.TEST_ITERS == 0:
            all_labels = np.zeros((0,))
            all_preds  = np.zeros((0,))
            test_loss_list.clear()

            num_eval_batches = int(np.ceil(float(len_test) / cfg.TRAIN.BATCH_SIZE))
            for _ in range(num_eval_batches):
                img_list, blobs_test = data_layer_test.forward()
                imgs   = _ensure_nhwc(blobs_test['img']).astype(np.float32)
                labels = _ensure_nhwc(blobs_test['label']).astype(np.int64)
                if args.use_fov_mask:
                    fovs = _ensure_nhwc(blobs_test['fov']).astype(np.int64)
                else:
                    fovs = np.ones(labels.shape, dtype=np.int64)

                loss_e, fg_prob_map = sess.run(
                    [network.loss, network.fg_prob],
                    feed_dict={
                        network.is_training: False,
                        network.imgs:        imgs,
                        network.labels:      labels,
                        network.fov_masks:   fovs
                    })

                test_loss_list.append(loss_e)

                # mask both labels & preds by FOV before flatten
                keep = (fovs.reshape(-1) > 0)
                all_labels = np.concatenate([all_labels, labels.reshape(-1)[keep]])
                all_preds  = np.concatenate([all_preds,  fg_prob_map.reshape(-1)[keep]])

                # qualitative saves (match Shin naming)
                cur_bs = len(img_list)
                fg = fg_prob_map
                if fg.ndim == 4 and fg.shape[-1] == 1:
                    fg = np.squeeze(fg, -1)  # [B,H,W]
                for i in range(cur_bs):
                    base = os.path.basename(img_list[i])
                    # preserve Shin's temp_name -> base without dir
                    prob_img = (fg[i] * 255).clip(0, 255).astype(np.uint8)
                    out_img  = ((fg[i] >= 0.5).astype(np.uint8) * 255)
                    skimage.io.imsave(os.path.join(res_save_path, base + '_prob.png'),   prob_img)
                    skimage.io.imsave(os.path.join(res_save_path, base + '_output.png'), out_img)

            # Metrics
            auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
            all_labels_bin = all_labels.astype(np.bool_)
            all_preds_bin  = (all_preds >= 0.5)
            acc_test = np.mean((all_labels_bin == all_preds_bin).astype(np.float32))

            # TB summary
            summary = tf.compat.v1.Summary()
            summary.value.add(tag="train_loss", simple_value=float(np.mean(train_loss_list)))
            summary.value.add(tag="test_loss",  simple_value=float(np.mean(test_loss_list)))
            summary.value.add(tag="test_acc",   simple_value=float(acc_test))
            summary.value.add(tag="test_auc",   simple_value=float(auc_test))
            summary.value.add(tag="test_ap",    simple_value=float(ap_test))
            summary_writer.add_summary(summary, global_step=it + 1)
            summary_writer.flush()

            print('iter: %d / %d, train_loss: %.4f' %
                  (it + 1, args.max_iters, np.mean(train_loss_list)))
            print('iter: %d / %d, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, test_ap: %.4f' %
                  (it + 1, args.max_iters, np.mean(test_loss_list), acc_test, auc_test, ap_test))

            f_log.write(f'iter: {it + 1} / {args.max_iters}\n')
            f_log.write(f'train_loss {np.mean(train_loss_list)}\n')
            f_log.write(f'test_loss {np.mean(test_loss_list)}\n')
            f_log.write(f'test_acc {acc_test}\n')
            f_log.write(f'test_auc {auc_test}\n')
            f_log.write(f'test_ap {ap_test}\n')
            f_log.flush()

            train_loss_list = []  # reset after reporting

    # final snapshot if last iter wasn't snapped
    if last_snapshot_iter != it:
        filename = os.path.join(model_save_path, f'iter_{it + 1}.ckpt')
        saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    f_log.close()
    sess.close()
    print("Training complete.")
