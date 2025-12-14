#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_VGN_DAU.py – VGN test wired to DAU2Net/DAU2 dataset layout + precomputed SRNS graphs.

Key fixes:
  • Clean split: no training code embedded.
  • Lazy import of skfmm (only if we must regenerate graphs).
  • No illegal feeds (we do NOT feed tensors like conv_feats / cnn_feat_spatial_sizes).
  • Correct sess.run usage (no list-wrapping of tensors).
  • Robust file extension resolution and checkpoint discovery.
  Possible for DRIVE AND CHASE ONLY!!!!
"""

from __future__ import print_function

import os
import glob
import argparse
import pickle as pkl
import warnings
from pathlib import Path
import csv
import pickle

import numpy as np
import skimage.io
import networkx as nx

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from config import cfg
from Modules.model import vessel_segm_vgn
import util


# ----------------------------- Utilities -----------------------------
def _get_split_txt_paths(dataset):
    if dataset == 'DRIVE':
        return cfg.TRAIN.DRIVE_SET_TXT_PATH, cfg.TEST.DRIVE_SET_TXT_PATH
    if dataset == 'CHASE_DB1':
        return cfg.TRAIN.CHASE_DB1_SET_TXT_PATH, cfg.TEST.CHASE_DB1_SET_TXT_PATH
    if dataset == 'HRF':
        return cfg.TRAIN.HRF_SET_TXT_PATH, cfg.TEST.HRF_SET_TXT_PATH
    raise ValueError(f"Unknown dataset: {dataset}")


def resolve_image_path(path):
    """Given a path or a basestem, return an existing image file path (any common ext)."""
    exts = (".jpg", ".png", ".tif", ".tiff", ".bmp", ".jpeg", ".gif", ".JPG")

    def _try_path(p: Path):
        if p.exists():
            return str(p)
        for ext in exts:
            cand = p.with_suffix(ext)
            if cand.exists():
                return str(cand)
        return None

    p = Path(path)

    # HRF convenience: if path is e.g., .../HRF/testing/06_dr, also try .../images/06_dr.ext
    if p.name.lower().startswith(tuple(str(d) for d in range(10))):
        # dir form
        if p.is_dir():
            img_dir = p / "images"
            if img_dir.exists():
                found = _try_path(img_dir / p.name)
                if found:
                    return found
        # stem form under testing/training
        if ("hrf" in [q.lower() for q in p.parts]) and ("testing" in [q.lower() for q in p.parts] or "training" in [q.lower() for q in p.parts]):
            img_dir = p.parent / "images"
            if img_dir.exists():
                found = _try_path(img_dir / p.name)
                if found:
                    return found

    # Direct or same-dir extension
    found = _try_path(p)
    if found:
        return found

    # Fix common malformed CHASE paths missing a slash (e.g., /workspaceCHASE_DB1 -> /workspace/DATASETS/CHASE_DB1)
    s = str(path)
    if "workspaceCHASE_DB1" in s:
        s_fix = s.replace("/workspaceCHASE_DB1", "/workspace/DATASETS/CHASE_DB1")
        s_fix = s_fix.replace("workspaceCHASE_DB1", "workspace/DATASETS/CHASE_DB1")
        found = _try_path(Path(s_fix))
        if found:
            return found

    # If linux-style /workspace/DATASETS paths are present but you're on Windows, remap to C:\Users\rog\THESIS\DATASETS
    if s.startswith("/workspace/DATASETS/"):
        win_path = Path("C:/Users/rog/THESIS/DATASETS") / Path(s).relative_to("/workspace/DATASETS")
        found = _try_path(win_path)
        if found:
            return found

    # Fallback: try dataset roots inferred from config (CHASE/DRIVE/HRF)
    candidates = []
    try:
        candidates.append(Path(cfg.TEST.CHASE_DB1_SET_TXT_PATH).resolve().parent.parent)  # .../CHASE_DB1
    except Exception:
        pass
    try:
        candidates.append(Path(cfg.TEST.DRIVE_SET_TXT_PATH).resolve().parent.parent)  # .../DRIVE
    except Exception:
        pass
    try:
        candidates.append(Path(cfg.TEST.HRF_SET_TXT_PATH).resolve().parent.parent)  # .../HRF
    except Exception:
        pass
    # explicit common roots if config is wrong
    candidates.append(Path("/workspace/DATASETS/CHASE_DB1"))
    candidates.append(Path("C:/Users/rog/THESIS/DATASETS/CHASE_DB1"))

    stem = p.stem
    for root in candidates:
        for split in ("test", "testing", "train", "training"):
            for sub in ("images",):
                for ext in exts:
                    cand = root / split / sub / f"{stem}{ext}"
                    if cand.exists():
                        return str(cand)

    raise FileNotFoundError(f"No image found for base path: {path}")


def _warn_filter():
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


def resolve_checkpoint_path(model_path):
    p = os.path.expanduser(model_path)
    if os.path.isdir(p):
        ckpt = tf.train.latest_checkpoint(p)
        if ckpt is not None:
            return ckpt
        idxs = sorted(glob.glob(os.path.join(p, "*.ckpt*.index")),
                      key=os.path.getmtime, reverse=True)
        if idxs:
            return os.path.splitext(idxs[0])[0]
        raise FileNotFoundError("No checkpoint found in directory: {}".format(p))
    if os.path.isfile(p):
        base, ext = os.path.splitext(p)
        if ".data-" in p and p.endswith(".of-00001"):
            return p[: p.rfind(".data-")]
        if ext in (".index", ".meta"):
            return base
    if os.path.exists(p + ".index") or os.path.exists(p + ".meta"):
        return p
    raise FileNotFoundError("Checkpoint not found around: {}".format(model_path))


def discover_from_train_run(save_root, run_name, prefer="best"):
    run_root = os.path.join(save_root, run_name) if save_root else run_name
    ckpt_dir = os.path.join(run_root, "train")
    if not os.path.isdir(ckpt_dir):
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
    idxs = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt*.index")),
                  key=os.path.getmtime, reverse=True)
    if idxs:
        return os.path.splitext(idxs[0])[0]
    raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")


def smart_restore(sess, net, ckpt_prefix):
    skip_tokens = ('Adam', 'Adadelta', 'Adagrad', 'Momentum', 'RMSProp',
                   'beta1_power', 'beta2_power', 'power')

    def _is_opt(name): return any(tok in name for tok in skip_tokens)

    all_vars = [v for v in tf.global_variables() if not _is_opt(v.op.name)]
    saver = tf.train.Saver(var_list=all_vars)
    try:
        saver.restore(sess, ckpt_prefix)
        print("[RESTORE] Full restore succeeded.")
        return "full"
    except Exception as e:
        print(f"[RESTORE] Full restore failed ({type(e).__name__}): {e}")
        print("[RESTORE] Attempting partial restore of overlapping variables...")

    reader = tf.train.NewCheckpointReader(ckpt_prefix)
    ck_vars = reader.get_variable_to_shape_map()
    name_to_var = {v.op.name: v for v in all_vars}
    var_map = {}
    for name, var in name_to_var.items():
        if name in ck_vars and list(ck_vars[name]) == list(var.shape):
            var_map[name] = var
            continue
        if name.startswith("img_output/"):
            alt = "output/" + name.split("/", 1)[1]
            if alt in ck_vars and list(ck_vars[alt]) == list(var.shape):
                var_map[alt] = var
                continue
    if not var_map:
        raise RuntimeError("[RESTORE] Partial restore found zero matching tensors.")
    saver_part = tf.train.Saver(var_list=var_map)
    saver_part.restore(sess, ckpt_prefix)
    print(f"[RESTORE] Partial restore loaded {len(var_map)} tensors.")
    return "partial"


def _graph_to_sparse_tensor(A):
    indices, values, shape = util.preprocess_graph_gat(A)
    from tensorflow.python.framework import sparse_tensor
    return sparse_tensor.SparseTensorValue(
        indices=indices.astype(np.int64),
        values=values.astype(np.float32),
        dense_shape=np.array(shape, dtype=np.int64)
    )


def _dataset_canvas_hw(dataset: str):
    try:
        _, h, w = util._dataset_specs(dataset)
        return int(h), int(w)
    except Exception:
        return 592, 592


def _pad_to_canvas(x, canvas_hw):
    h_target, w_target = int(canvas_hw[0]), int(canvas_hw[1])
    if x.ndim == 2:
        h, w = x.shape
        out = np.zeros((max(h_target, h), max(w_target, w)), dtype=x.dtype)
        out[:h, :w] = x
        return out
    if x.ndim == 3:
        h, w, c = x.shape
        out = np.zeros((max(h_target, h), max(w_target, w), c), dtype=x.dtype)
        out[:h, :w, :] = x
        return out
    raise ValueError("Unsupported ndim for pad_to_canvas: {}".format(x.ndim))


def _crop_to_raw(x_canvas, raw_h, raw_w):
    if x_canvas.ndim == 2:
        return x_canvas[:raw_h, :raw_w]
    if x_canvas.ndim == 3:
        return x_canvas[:raw_h, :raw_w, :]
    raise ValueError("Unsupported ndim for crop_to_raw: {}".format(x_canvas.ndim))


def _get_raw_hw(img_path):
    """Read the original DRIVE RGB image to get its raw (H,W)."""
    arr = skimage.io.imread(str(img_path))
    if arr.ndim == 3:
        return int(arr.shape[0]), int(arr.shape[1])
    elif arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    else:
        raise ValueError(f"Unexpected image shape at {img_path}: {arr.shape}")


def _try_load_external_prob_padded(probs_root: Path, stem: str, canvas_hw):
    """
    Load external DAU2Net prob map (baseline AP). Returns (pm_raw, pm_canvas).
    Tries: <stem>_prob.(png|npy) then <stem>.(png|npy). Values in [0,1].
    If not found, returns (None, None).
    """
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
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if arr.max() > 1.0:
                    arr = np.clip(arr, 0, 255) / 255.0
                return arr, _pad_to_canvas(arr, canvas_hw)
            else:
                arr = skimage.io.imread(str(p)).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if arr.max() > 1.0:
                    arr = np.clip(arr, 0, 255) / 255.0
                return arr, _pad_to_canvas(arr, canvas_hw)
    return None, None


def _resolve_prob_root(explicit_path: str, dau_root: str, split: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
    if dau_root:
        cand = Path(dau_root) / split / 'prob'
        if cand.exists():
            return cand
        fallback = Path(dau_root) / split
        if fallback.exists():
            return fallback
    raise ValueError(f"Provide a valid --dau_root/--test_probs for split '{split}' so we can load probmaps/graphs.")


def binarize_label_vessel_positive(lbl_raw: np.ndarray,
                                   dataset_key: str = "",
                                   debug: bool = False) -> np.ndarray:
    """
    Force labels into {0,1} with 1 = vessel (foreground), 0 = background.
    Heuristic: vessels are typically the minority class; pick the least frequent value as vessel.
    """
    lbl = np.asarray(lbl_raw)
    vals, counts = np.unique(lbl, return_counts=True)
    if len(vals) < 2:
        raise ValueError(f"Label image appears constant: values={vals}")

    vessel_val = vals[int(np.argmin(counts))]
    bg_val = vals[int(np.argmax(counts))]
    lbl_bin = (lbl == vessel_val).astype(np.float32)

    if debug:
        vessel_frac = float(lbl_bin.mean()) if lbl_bin.size else 0.0
        print(f"[LABEL] dataset={dataset_key}, vals={vals}, counts={counts}, "
              f"vessel_val={vessel_val}, bg_val={bg_val}, "
              f"vessel_frac={vessel_frac:.4f}")

    return lbl_bin


# ----------------------------- Fallback SRNS graph builder -----------------------------
def make_graph_using_srns(prob_map_canvas, edge_type, win_size, edge_geo_dist_thresh, save_path):
    """
    prob_map_canvas: 2D float32 in [0,1], padded canvas used by VGN inputs
    """
    # Lazy import to avoid hard dependency during normal testing
    import skfmm  # noqa: F401

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if prob_map_canvas.ndim != 2:
        raise ValueError("SRNS regen expects a 2D probability map.")
    vesselness = prob_map_canvas
    im_y, im_x = vesselness.shape[:2]
    y_quan = sorted(set(list(range(0, im_y, win_size)) + [im_y]))
    x_quan = sorted(set(list(range(0, im_x, win_size)) + [im_x]))

    # SRVS nodes
    max_pos = []
    for y_idx in range(len(y_quan) - 1):
        for x_idx in range(len(x_quan) - 1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx + 1],
                                   x_quan[x_idx]:x_quan[x_idx + 1]]
            if np.sum(cur_patch) == 0:
                max_pos.append((y_quan[y_idx] + cur_patch.shape[0] // 2,
                                x_quan[x_idx] + cur_patch.shape[1] // 2))
            else:
                rr, cc = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx] + int(rr), x_quan[x_idx] + int(cc)))

    G = nx.Graph()
    for node_idx, (ny, nx_) in enumerate(max_pos):
        G.add_node(node_idx, kind='MP', y=int(ny), x=int(nx_), label=int(node_idx))

    # Geodesic edges via FMM (windowless here—full image for simplicity)
    speed = vesselness.astype(np.float32)
    node_list = list(G.nodes)
    for i, n in enumerate(node_list):
        y_n = G.nodes[n]['y']
        x_n = G.nodes[n]['x']
        if speed[y_n, x_n] == 0:
            continue
        # 3×3 neighborhood gating
        y0, y1 = max(0, y_n - 1), min(im_y, y_n + 2)
        x0, x1 = max(0, x_n - 1), min(im_x, x_n + 2)
        if float(np.mean(speed[y0:y1, x0:x1])) < 0.1:
            continue

        # narrow band distance transform
        phi = np.ones_like(speed, dtype=np.float32)
        phi[y_n, x_n] = -1
        import skfmm
        tt = skfmm.travel_time(phi, speed, narrow=edge_geo_dist_thresh)

        for j in range(i + 1, len(node_list)):
            m = node_list[j]
            y_c = G.nodes[m]['y']
            x_c = G.nodes[m]['x']
            geo = float(tt[y_c, x_c])
            if np.isfinite(geo) and (geo < edge_geo_dist_thresh):
                if 'weighted' in edge_type:
                    w = float(edge_geo_dist_thresh / (edge_geo_dist_thresh + geo))
                else:
                    w = 1.0
                G.add_edge(n, m, weight=w)

    nx.write_gpickle(G, str(save_path), protocol=pkl.HIGHEST_PROTOCOL)
    G.clear()


# ----------------------------- CLI -----------------------------
def parse_args():
    default_save_root = 'C:/Users/rog/THESIS/DAU2_CHASE/VGN'
    default_results_root = os.path.join(default_save_root, 'test_results')

    parser = argparse.ArgumentParser(description='VGN test with DAU2/DAU2Net layout + precomputed graphs')

    # Dataset
    parser.add_argument('--dataset', default='DRIVE', choices=['DRIVE', 'CHASE_DB1', 'HRF'])

    # DAU2Net layout
    parser.add_argument('--dau_root', type=str, default='C:/Users/rog/THESIS/DAU2_CHASE',
                        help='Root of DAU2_CHASE containing train/ and test/.')
    parser.add_argument('--test_probs', type=str, default=cfg.PATHS.DAU2_PROBMAP_TEST_DIR,
                        help='Override test probs dir (default: <dau_root>/test/probs).')
    parser.add_argument('--use_external_cnn_probs', action='store_true', default=True,
                        help='If True, baseline AP uses DAU2Net prob maps from probs dir.')

    # Graph params (must match training)
    parser.add_argument('--win_size', type=int, default=8)
    parser.add_argument('--edge_type', default='srns_geo_dist_binary',
                        choices=['srns_geo_dist_binary', 'srns_geo_dist_weighted'])
    parser.add_argument('--edge_geo_dist_thresh', type=float, default=40)
    parser.add_argument('--graph_geo_dist', type=float, default=40,
                        help='Optional override for graph geodesic distance suffix (e.g., 40 for CHASE).')

    # Model params
    parser.add_argument('--cnn_model', default='driu', choices=['driu', 'driu_large'])
    parser.add_argument('--norm_type', default='GN', choices=['GN', 'BN', 'none', None])
    parser.add_argument('--cnn_loss_on', action='store_true', default=True)
    parser.add_argument('--gnn_loss_on', action='store_true', default=True)
    parser.add_argument('--gnn_loss_weight', type=float, default=1.0)
    parser.add_argument('--gat_hid_units', type=int, nargs='+', default=[16, 16])
    parser.add_argument('--gat_n_heads', type=int, nargs='+', default=[4, 4, 1])
    parser.add_argument('--gat_use_residual', action='store_true', default=True)
    parser.add_argument('--infer_module_kernel_size', type=int, default=3)

    # Optimizers (dummy; needed so model.build_optimizer() doesn't crash)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iters', default=50000, type=int)
    parser.add_argument('--do_simul_training', type=bool, default=False)
    parser.add_argument('--old_net_ft_lr', type=float, default=0.0)
    parser.add_argument('--new_net_lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduling', type=str, default='pc')
    parser.add_argument('--lr_decay_tp', type=float, default=0.8)

    # Dropouts (keep 0 in test)
    parser.add_argument('--gnn_feat_dropout', type=float, default=0.0)
    parser.add_argument('--gnn_att_dropout', type=float, default=0.0)
    parser.add_argument('--post_cnn_dropout', type=float, default=0.0)

    # Restore options
    parser.add_argument('--model_path', default='C:\\Users\\rog\\THESIS\\DAU2_HRF\\VGN\\VGN_DAU_run1\\train\\iter_9500.ckpt',
                        help='VGN checkpoint: dir or prefix, or any of (.meta/.index/.data-00000-of-00001).')

    # Train-run discovery (preferred)
    parser.add_argument('--save_root', default='C:\\Users\\rog\\THESIS\\DAU2_CHASE\\train2',
                        help='save_root used in training (contains run folders).')
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='',
                        help='If empty, uses VGN_DAU_run{run_id}')
    parser.add_argument('--prefer_ckpt', type=str, default='best', choices=['best', 'latest'])

    # I/O
    parser.add_argument('--results_root', default='C:\\Users\\rog\\THESIS\\DAU2_CHASE\\train2',
                        help='Root path to save test results')

    # Optional regen
    parser.add_argument('--regen_missing_graphs', action='store_true', default=True)
    parser.add_argument('--prob_shift', type=float, default=0.0,
                        help='Subtract this from logits before sigmoid at test (e.g., 0.3 to darken background).')
    parser.add_argument('--eval_threshold', type=float, default=0.5,
                        help='Threshold for binarizing probs when computing ACC/Spec/Dice/IoU.')
    parser.add_argument('--sweep_thresholds', action='store_true',
                        help='If set, sweep thresholds to report best Dice/balanced-acc on test set.')
    parser.add_argument('--sweep_min', type=float, default=0.4,
                        help='Min threshold for sweep (inclusive).')
    parser.add_argument('--sweep_max', type=float, default=0.8,
                        help='Max threshold for sweep (inclusive).')
    parser.add_argument('--sweep_steps', type=int, default=9,
                        help='Number of thresholds to sweep between min/max.')
    parser.add_argument('--invert_output', action='store_true', default=False,
                        help='If set, treat network output as background prob and invert to vessel prob.')
    parser.add_argument('--invert_external_prob', action='store_true', default=False,
                        help='Invert external CNN prob maps/graphs if they were saved as background-prob.')

    return parser.parse_args()


# ----------------------------- Main -----------------------------
if __name__ == '__main__':
    _warn_filter()
    args = parse_args()

    dataset_key = args.dataset.upper()
    args.dataset = dataset_key
    if args.graph_geo_dist is not None:
        args.edge_geo_dist_thresh = float(args.graph_geo_dist)

    default_drive_root = 'C:/Users/rog/THESIS/DAU2_DRIVE'
    default_test_prob = cfg.PATHS.DAU2_PROBMAP_TEST_DIR
    if dataset_key == 'CHASE_DB1':
        chase_root = str(Path('C:/Users/rog/THESIS/DAU2_CHASE'))
        if (not args.dau_root) or (Path(args.dau_root).resolve() == Path(default_drive_root).resolve()):
            args.dau_root = chase_root
        if (not args.test_probs) or (Path(args.test_probs).resolve() == Path(default_test_prob).resolve()):
            args.test_probs = str(Path(args.dau_root) / 'test')
        if args.win_size == 4:
            args.win_size = 8
        if args.edge_geo_dist_thresh == 10:
            args.edge_geo_dist_thresh = 40

    if dataset_key == 'HRF':
        hrf_root = str(Path('C:/Users/rog/THESIS/DAU_HRF'))
        bad_root = any(tok in str(args.dau_root) for tok in ['DAU2_CHASE', 'DAU2_DRIVE', 'DAU2_HRF'])
        if (not args.dau_root) or bad_root or (not Path(args.dau_root).exists()):
            args.dau_root = hrf_root
        bad_test = any(tok in str(args.test_probs) for tok in ['DAU2_CHASE', 'DAU2_DRIVE', 'DAU2_HRF'])
        if (not args.test_probs) or (Path(args.test_probs).resolve() == Path(default_test_prob).resolve()) or bad_test:
            args.test_probs = str(Path(args.dau_root) / 'test')
        if args.win_size == 4:
            args.win_size = 10
        if args.edge_geo_dist_thresh == 40:
            args.edge_geo_dist_thresh = 80
        # adjust save/result/model roots if user left CHASE defaults
        default_save_root = 'C:/Users/rog/THESIS/DAU2_CHASE/VGN'
        default_results_root = os.path.join(default_save_root, 'test_results')
        default_model = r'C:\Users\rog\THESIS\DAU2_CHASE\VGN\VGN_DAU_run1\train\iter_10000.ckpt'
        if args.results_root == default_results_root:
            args.results_root = str(Path(hrf_root) / 'VGN' / 'test_results')
        if args.model_path == default_model:
            args.model_path = str(Path(hrf_root) / 'VGN' / 'VGN_DAU_run1' / 'train' / 'iter_10000.ckpt')
        if args.save_root == default_save_root:
            args.save_root = str(Path(hrf_root) / 'VGN')

    if isinstance(args.norm_type, str) and args.norm_type.lower() == 'none':
        args.norm_type = None

    print('Called with args:')
    print(args)

    # Test list
    _, test_set_txt_path = _get_split_txt_paths(dataset_key)
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    # Fix common malformed CHASE paths missing a slash
    def _fix_path(s: str) -> str:
        if "workspaceCHASE_DB1" in s:
            s = s.replace("/workspaceCHASE_DB1", "/workspace/DATASETS/CHASE_DB1")
            s = s.replace("workspaceCHASE_DB1", "workspace/DATASETS/CHASE_DB1")
        return s
    test_img_names = [_fix_path(p) for p in test_img_names]
    len_test = len(test_img_names)

    # Results dir
    run_name = args.run_name if args.run_name else f"VGN_DAU_run{int(args.run_id)}"
    res_save_path = os.path.join(args.results_root, run_name, dataset_key)
    os.makedirs(res_save_path, exist_ok=True)
    log_path = os.path.join(res_save_path, 'log.txt')
    with open(log_path, 'w') as f_log:
        f_log.write(str(args) + '\n')

    # Data layer (pads to dataset canvas)
    # right before building the DataLayer
    use_pad = False if dataset_key == 'CHASE_DB1' else True
    data_layer_test = util.DataLayer(test_img_names, is_training=False, use_padding=use_pad)

    # Build network
    args.gat_hid_units = list(args.gat_hid_units)
    args.gat_n_heads = list(args.gat_n_heads)
    network = vessel_segm_vgn(args, None)

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())

    # Resolve checkpoint
    if args.model_path:
        ckpt_prefix = resolve_checkpoint_path(args.model_path)
    else:
        run_name_eff = run_name
        ckpt_prefix = discover_from_train_run(args.save_root, run_name_eff, prefer=args.prefer_ckpt)
    print("Restoring from checkpoint prefix:", ckpt_prefix)
    mode = smart_restore(sess, network, ckpt_prefix)
    print(f"Checkpoint restored ({mode}).")

    # Test loop
    probs_root = _resolve_prob_root(args.test_probs, args.dau_root, 'test')
    canvas_hw = _dataset_canvas_hw(dataset_key)
    batch_size = getattr(cfg.TRAIN, 'BATCH_SIZE', 1)
    num_batches = int(np.ceil(float(len_test) / max(1, batch_size)))

    # Accumulators for dataset metrics
    all_labels = np.zeros((0,), dtype=np.float32)
    all_preds = np.zeros((0,), dtype=np.float32)
    per_image_rows = []
    per_image_auc = []
    per_image_ap = []

    for _ in range(num_batches):
        img_list, blobs = data_layer_test.forward()
        if not img_list:
            continue

        # Padded tensors (dataset canvas)
        img = blobs['img']
        label = blobs['label']
        fov = blobs['fov']

        # === Build external prob batch ===
        paths_resolved = [resolve_image_path(p) for p in img_list]
        ext_batch = []
        ext_per_item = []
        for pth in paths_resolved:
            stem = Path(pth).stem
            _, ext_canvas = _try_load_external_prob_padded(probs_root, stem, canvas_hw)
            if ext_canvas is None:
                raise FileNotFoundError(f"[DAU2 prob missing] {stem} under {probs_root}")
            if args.invert_external_prob:
                ext_canvas = 1.0 - ext_canvas
            ext_per_item.append(ext_canvas.astype(np.float32))
            ext_batch.append(ext_canvas.astype(np.float32))
        ext_batch = np.stack(ext_batch, axis=0)[..., np.newaxis]  # [B,H,W,1]

        cur_batch = len(img_list)
        for b in range(cur_batch):
            rec = {}
            img_path = resolve_image_path(img_list[b])
            stem = Path(img_path).stem
            raw_h, raw_w = _get_raw_hw(img_path)

            rec['stem'] = stem
            rec['img_path'] = img_path
            rec['raw_hw'] = (raw_h, raw_w)
            rec['img_canvas'] = img[[b], :, :, :]
            rec['label_canvas'] = label[[b], :, :, :]
            rec['fov_canvas'] = fov[[b], :, :, :]

            # External CNN prob snapshots
            ext_raw = _crop_to_raw(ext_per_item[b], raw_h, raw_w)
            ext_canvas = ext_per_item[b]
            rec['cnn_fg_prob_raw'] = ext_raw
            rec['cnn_fg_prob_canvas'] = ext_canvas

            # Baseline AP (inside FOV)
            mask_canvas = rec['fov_canvas'][0, :, :, 0].astype(bool)
            label_canvas_raw = rec['label_canvas'][0, :, :, 0].astype(np.float32)
            label_canvas_bin = binarize_label_vessel_positive(label_canvas_raw,
                                                              dataset_key=dataset_key,
                                                              debug=False)
            fg_canvas = np.asarray(rec['cnn_fg_prob_canvas'], dtype=np.float32)
            try:
                _, ap_base = util.get_auc_ap_score(label_canvas_bin[mask_canvas], fg_canvas[mask_canvas])
            except Exception:
                ap_base = np.nan
            rec['ap_baseline'] = float(ap_base)

            # ---------- Graph: load or regenerate ----------
            graph_path = probs_root / f"{stem}_{int(args.win_size):02d}_{int(args.edge_geo_dist_thresh):02d}.graph_res"
            if not graph_path.exists():
                if not args.regen_missing_graphs:
                    raise FileNotFoundError(f"Missing graph {graph_path} and regen disabled.")
                canvas_desc = f"{canvas_hw[0]}x{canvas_hw[1]}"
                print(f"[GRAPH] Missing {graph_path.name}; regenerating from prob map ({canvas_desc}).")
                pm_canvas = rec['cnn_fg_prob_canvas']
                if pm_canvas.ndim == 3:
                    pm_canvas = pm_canvas[..., 0]
                make_graph_using_srns(pm_canvas, args.edge_type, int(args.win_size),
                                      int(args.edge_geo_dist_thresh), graph_path)
            with open(str(graph_path), 'rb') as gf:
                G = pkl.load(gf)
            rec['graph'] = G

            # ---------- VGN inference ----------
            cur_graph = nx.convert_node_labels_to_integers(rec['graph'])
            node_byxs = util.get_node_byx_from_graph(cur_graph, [cur_graph.number_of_nodes()])
            if 'weighted' in args.edge_type:
                A = nx.adjacency_matrix(cur_graph)
            else:
                A = nx.adjacency_matrix(cur_graph, weight=None).astype(float)
            adj_sp = _graph_to_sparse_tensor(A)

            feed = {
                network.imgs: rec['img_canvas'],
                network.node_byxs: node_byxs,
                network.adj: adj_sp,
                network.is_lr_flipped: False,
                network.is_ud_flipped: False,
                network.gnn_feat_dropout: float(args.gnn_feat_dropout),
                network.gnn_att_dropout: float(args.gnn_att_dropout),
                network.post_cnn_dropout: float(args.post_cnn_dropout),
                network.external_cnn_prob: rec['cnn_fg_prob_canvas'][None, ..., None],
            }

            # Fetch final probability (no list wrapping)
            res_prob_canvas = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed)
            res_prob_canvas = np.squeeze(res_prob_canvas)  # H×W
            if abs(getattr(args, "prob_shift", 0.0)) > 1e-9:
                eps = 1e-6
                p = np.clip(res_prob_canvas, eps, 1.0 - eps)
                logit = np.log(p / (1.0 - p))
                res_prob_canvas = 1.0 / (1.0 + np.exp(-(logit - float(args.prob_shift))))
            if args.invert_output:
                # Some checkpoints were trained with background=1 labels; invert to interpret as vessel prob.
                res_prob_canvas = 1.0 - res_prob_canvas

            # ---------- Crop back to raw & save ----------
            ph, pw = rec['raw_hw']
            pred_raw = _crop_to_raw(res_prob_canvas, ph, pw).astype(np.float32)
            lbl_raw = _crop_to_raw(rec['label_canvas'][0, :, :, 0], ph, pw).astype(np.float32)
            lbl_bin = binarize_label_vessel_positive(lbl_raw,
                                                     dataset_key=dataset_key,
                                                     debug=False)
            fov_raw = _crop_to_raw(rec['fov_canvas'][0, :, :, 0], ph, pw).astype(bool)

            # Save raw-dimension outputs
            base = rec['stem']
            out_u8 = (np.clip(pred_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
            skimage.io.imsave(os.path.join(res_save_path, base + '_prob_final.png'), out_u8)
            np.save(os.path.join(res_save_path, base + '.npy'), pred_raw.astype(np.float32))
            out_inv = ((1. - np.clip(pred_raw, 0.0, 1.0)) * 255.0).astype(np.uint8)
            skimage.io.imsave(os.path.join(res_save_path, base + '_prob_final_inv.png'), out_inv)

            # ---------- Metrics (DRIVE: inside FOV) ----------
            all_labels = np.concatenate((all_labels, lbl_bin[fov_raw].reshape(-1).astype(np.float32)))
            all_preds = np.concatenate((all_preds, pred_raw[fov_raw].reshape(-1).astype(np.float32)))

            # Per-image APs (baseline + final)
            try:
                auc_final, ap_final = util.get_auc_ap_score(lbl_bin[fov_raw], pred_raw[fov_raw])
            except Exception:
                auc_final = np.nan
                ap_final = np.nan
            per_image_auc.append(float(auc_final))
            per_image_ap.append(float(ap_final))
            print(f"{base}: baseline_AP={rec['ap_baseline']:.4f} | final_AP={ap_final:.4f}")

            # Per-image BCE + metrics inside FOV
            lbl_flat = lbl_bin[fov_raw].reshape(-1).astype(np.float32)
            pred_flat = pred_raw[fov_raw].reshape(-1).astype(np.float32)
            eps = 1e-6
            pred_clamped = np.clip(pred_flat, eps, 1.0 - eps)
            bce = -np.mean(lbl_flat * np.log(pred_clamped) + (1.0 - lbl_flat) * np.log(1.0 - pred_clamped))

            preds_bin = pred_flat >= float(args.eval_threshold)
            lbls_bin = lbl_flat >= 0.5
            tp_img = float(np.sum(np.logical_and(preds_bin, lbls_bin)))
            tn_img = float(np.sum(np.logical_and(~preds_bin, ~lbls_bin)))
            fp_img = float(np.sum(np.logical_and(preds_bin, ~lbls_bin)))
            fn_img = float(np.sum(np.logical_and(~preds_bin, lbls_bin)))
            denom = lambda x: x if x > 0 else 1.0

            acc_img = (tp_img + tn_img) / denom(tp_img + tn_img + fp_img + fn_img)
            prec_img = tp_img / denom(tp_img + fp_img)
            rec_img = tp_img / denom(tp_img + fn_img)
            spec_img = tn_img / denom(tn_img + fp_img)

            # F1-score (harmonic mean of precision and recall)
            if (prec_img + rec_img) > 0:
                f1_img = 2.0 * prec_img * rec_img / (prec_img + rec_img)
            else:
                f1_img = 0.0

            # Dice coefficient using 2 * |P∩T| / (|P| + |T|)
            pred_pos_img = tp_img + fp_img
            actual_pos_img = tp_img + fn_img
            if (pred_pos_img + actual_pos_img) > 0:
                dice_img = 2.0 * tp_img / (pred_pos_img + actual_pos_img)
            else:
                dice_img = 0.0

            iou_img = tp_img / denom(tp_img + fp_img + fn_img)

            per_image_rows.append({
                'image': base,
                'bce': float(bce),
                'ap_baseline': float(rec['ap_baseline']),
                'ap_final': float(ap_final),
                'accuracy': float(acc_img),
                'specificity': float(spec_img),
                'sensitivity': float(rec_img),
                'precision': float(prec_img),
                'recall': float(rec_img),
                'f1': float(f1_img),
                'dice': float(dice_img),
                'iou': float(iou_img),
            })

    # ---------------- Dataset metrics ----------------
    # Use balanced sampling inside util.get_auc_ap_score; no extra subsample here.
    overall_auc, overall_ap = util.get_auc_ap_score(all_labels, all_preds)
    # Also report mean per-image AUC/AP (nan-safe) to reduce dominance from huge images
    auc_test = float(np.nanmean(per_image_auc)) if per_image_auc else overall_auc
    ap_test = float(np.nanmean(per_image_ap)) if per_image_ap else overall_ap

    labels_b = all_labels >= 0.5
    preds_b = all_preds >= float(args.eval_threshold)
    eps = 1e-6
    preds_clamped = np.clip(all_preds, eps, 1.0 - eps)
    bce_overall = -np.mean(all_labels * np.log(preds_clamped) + (1.0 - all_labels) * np.log(1.0 - preds_clamped))

    tp = float(np.sum(np.logical_and(preds_b, labels_b)))
    tn = float(np.sum(np.logical_and(~preds_b, ~labels_b)))
    fp = float(np.sum(np.logical_and(preds_b, ~labels_b)))
    fn = float(np.sum(np.logical_and(~preds_b, labels_b)))
    denom = lambda x: x if x > 0 else 1.0

    acc = (tp + tn) / denom(tp + tn + fp + fn)
    prec = tp / denom(tp + fp)
    rec = tp / denom(tp + fn)
    spec = tn / denom(tn + fp)

    # F1-score from precision and recall (harmonic mean)
    if (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    # Dice coefficient using cardinalities of prediction/ground-truth sets
    pred_pos = tp + fp
    actual_pos = tp + fn
    if (pred_pos + actual_pos) > 0:
        dice = 2.0 * tp / (pred_pos + actual_pos)
    else:
        dice = 0.0

    iou = tp / denom(tp + fp + fn)
    miou = iou  # binary foreground IoU

    # Optional threshold sweep
    best_eval_from_sweep = None
    if args.sweep_thresholds:
        thr_list = np.linspace(args.sweep_min, args.sweep_max, args.sweep_steps)
        best_dice = -1.0
        best_bal_acc = -1.0
        best_row_dice = None
        best_row_bal = None
        for th in thr_list:
            pb = all_preds >= float(th)
            tp_ = float(np.sum(np.logical_and(pb, labels_b)))
            tn_ = float(np.sum(np.logical_and(~pb, ~labels_b)))
            fp_ = float(np.sum(np.logical_and(pb, ~labels_b)))
            fn_ = float(np.sum(np.logical_and(~pb, labels_b)))
            acc_ = (tp_ + tn_) / denom(tp_ + tn_ + fp_ + fn_)
            rec_ = tp_ / denom(tp_ + fn_)
            spec_ = tn_ / denom(tn_ + fp_)
            pred_pos_ = tp_ + fp_
            actual_pos_ = tp_ + fn_
            if (pred_pos_ + actual_pos_) > 0:
                dice_ = 2.0 * tp_ / (pred_pos_ + actual_pos_)
            else:
                dice_ = 0.0
            bal_acc_ = 0.5 * (rec_ + spec_)
            if dice_ > best_dice:
                best_dice = dice_
                best_row_dice = (th, acc_, spec_, rec_, dice_, bal_acc_)
            if bal_acc_ > best_bal_acc:
                best_bal_acc = bal_acc_
                best_row_bal = (th, acc_, spec_, rec_, dice_, bal_acc_)

        if best_row_dice:
            th, acc_b, spec_b, rec_b, dice_b, bal_b = best_row_dice
            print(f'[SWEEP] Best Dice @ thr={th:.3f}: acc={acc_b:.4f} spec={spec_b:.4f} rec={rec_b:.4f} dice={dice_b:.4f} bal-acc={bal_b:.4f}')
        if best_row_bal:
            th, acc_b, spec_b, rec_b, dice_b, bal_b = best_row_bal
            print(f'[SWEEP] Best Bal-Acc @ thr={th:.3f}: acc={acc_b:.4f} spec={spec_b:.4f} rec={rec_b:.4f} dice={dice_b:.4f} bal-acc={bal_b:.4f}')
            best_eval_from_sweep = th

    print(f'----- {dataset_key} TEST (FOV ROI) -----')
    print('Accuracy:      %.4f' % acc)
    print('Specificity:   %.4f' % spec)
    print('Recall/Sens.:  %.4f' % rec)
    print('F1-score:      %.4f' % f1)
    print('Dice Coeff.:   %.4f' % dice)
    print('AUC:           %.4f' % auc_test)
    print('AP:            %.4f' % ap_test)
    print('IoU:           %.4f' % iou)
    print('mIoU:          %.4f' % miou)
    print('BCE:           %.4f' % bce_overall)

    with open(log_path, 'a') as f_log:
        f_log.write('Accuracy ' + str(acc) + '\n')
        f_log.write('Specificity ' + str(spec) + '\n')
        f_log.write('Recall/Sensitivity ' + str(rec) + '\n')
        f_log.write('F1 ' + str(f1) + '\n')
        f_log.write('Dice ' + str(dice) + '\n')
        f_log.write('AUC ' + str(auc_test) + '\n')
        f_log.write('AP ' + str(ap_test) + '\n')
        f_log.write('IoU ' + str(iou) + '\n')
        f_log.write('mIoU ' + str(miou) + '\n')
        f_log.write('BCE ' + str(bce_overall) + '\n')

    if per_image_rows:
        csv_path = os.path.join(res_save_path, 'per_image_metrics.csv')
        fieldnames = ['image', 'bce', 'ap_baseline', 'ap_final',
                      'accuracy', 'specificity', 'sensitivity',
                      'precision', 'recall', 'f1', 'dice', 'iou']
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_image_rows:
                writer.writerow(row)
        print(f"[CSV] Per-image metrics saved to {csv_path}")

    sess.close()
    print("Testing complete.")
