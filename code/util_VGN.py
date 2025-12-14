# -*- coding: utf-8 -*-
""" Common util file — Python3/TF2-safe & Windows-path wired (Mj)

This is the clean, de-duplicated version of util.py. It's named util_fixed.py
to avoid legacy collisions. Callers should import it as `import util_fixed as util`.
"""

from __future__ import division, print_function

# ---- Py2/3 shims ------------------------------------------------------------
try:  # keep legacy code that used xrange happy
    xrange  # type: ignore[name-defined]
except NameError:  # Python 3
    xrange = range  # noqa: F401

import os
import time
import pickle
from typing import List

import numpy as np
import numpy.random as npr
import skimage.io
import skimage.transform
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import _init_paths
from config import cfg

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------
STR_ELEM = np.array([[1,1,1],[1,1,0],[0,0,0]], dtype=bool)  # 8-neighbors
VIS_FIG_SIZE = (10,10)
VIS_NODE_SIZE = 50
VIS_ALPHA = 0.5
VIS_NODE_COLOR = ['b','r','y','g']  # tp/fp/fn/tn
VIS_EDGE_COLOR = ['b','g','r']      # tp/fn/fp
DEBUG = False

# -----------------------------------------------------------------------------
# helpers for dataset root resolution (your Windows folder layout)
# -----------------------------------------------------------------------------
def _dataset_dir_name(ds: str) -> str:
    return 'CHASE-DB1' if ds.upper() == 'CHASE_DB1' else ds.upper()

def _meta_for_dataset(ds: str):
    ds = ds.upper()
    if ds == 'DRIVE':
        return dict(im_ext='_image.tif', label_ext='_label.gif', fov_ext='_mask.gif',
                    pixel_mean=cfg.PIXEL_MEAN_DRIVE, size=(592,592))
    if ds == 'STARE':
        return dict(im_ext='.ppm', label_ext='.ah.ppm', fov_ext='_mask.png',
                    pixel_mean=cfg.PIXEL_MEAN_STARE, size=(704,704))
    if ds == 'CHASE_DB1':
        return dict(im_ext='.jpg', label_ext='_1stHO.png', fov_ext='_mask.tif',
                    pixel_mean=cfg.PIXEL_MEAN_CHASE_DB1, size=(1024,1024))
    if ds == 'HRF':
        return dict(im_ext='.bmp', label_ext='.tif', fov_ext='_mask.tif',
                    pixel_mean=cfg.PIXEL_MEAN_HRF, size=(768,768))
    raise ValueError('Unknown dataset: {}'.format(ds))

def _pick_graph_path(base: str, win_size: int, geo_thr: float) -> str:
    """Prefer `{base}_{win}_{thr}.graph_res`, otherwise fallback to `{base}.graph_res`
    or the first matching `*.graph_res` beside it."""
    win_str = f'_{win_size:02d}_{int(geo_thr):02d}'
    cand1 = base + win_str + '.graph_res'
    if os.path.isfile(cand1):
        return cand1
    cand2 = base + '.graph_res'
    if os.path.isfile(cand2):
        return cand2
    # last resort: any graph next to the base
    import glob
    found = glob.glob(base + '_*.graph_res')
    if found:
        return found[0]
    raise FileNotFoundError('No graph_res next to {}'.format(base))

# -----------------------------------------------------------------------------
# minibatch & datalayers
# -----------------------------------------------------------------------------
class DataLayer(object):
    def __init__(self, db: List[str], is_training: bool, use_padding: bool=False):
        self._db = db
        self._is_training = is_training
        self._use_padding = use_padding
        (self._shuffle_db_inds() if self._is_training else self._db_inds())

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _db_inds(self):
        self._perm = np.arange(len(self._db)); self._cur = 0

    def _get_next_minibatch_inds(self):
        cur_batch_size = cfg.TRAIN.BATCH_SIZE
        if self._is_training:
            if self._cur + cfg.TRAIN.BATCH_SIZE > len(self._db):
                self._shuffle_db_inds()
        else:
            rem = len(self._db) - self._cur
            cur_batch_size = cfg.TRAIN.BATCH_SIZE if rem >= cfg.TRAIN.BATCH_SIZE else rem
        db_inds = self._perm[self._cur:self._cur+cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur >= len(self._db)):
            self._db_inds()
        return db_inds

    def _get_next_minibatch(self):
        inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in inds]
        return minibatch_db, get_minibatch(minibatch_db, self._is_training, use_padding=self._use_padding)

    def forward(self):
        return self._get_next_minibatch()

class GraphDataLayer(object):
    def __init__(self, db: List[str], is_training: bool,
                 edge_type='srns_geo_dist_binary', win_size=8, edge_geo_dist_thresh=20,
                 dataset_name: str=None, datasets_root: str=None):
        self._db = db
        self._is_training = is_training
        self._edge_type = edge_type
        self._win_size = int(win_size)
        self._edge_geo_dist_thresh = float(edge_geo_dist_thresh)
        self._dataset_name = dataset_name
        self._datasets_root = datasets_root
        (self._shuffle_db_inds() if self._is_training else self._db_inds())

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db))); self._cur = 0
    def _db_inds(self):
        self._perm = np.arange(len(self._db)); self._cur = 0

    def _get_next_minibatch_inds(self):
        cur_batch_size = cfg.TRAIN.GRAPH_BATCH_SIZE
        if self._is_training:
            if self._cur + cfg.TRAIN.GRAPH_BATCH_SIZE > len(self._db):
                self._shuffle_db_inds()
        else:
            rem = len(self._db) - self._cur
            cur_batch_size = cfg.TRAIN.GRAPH_BATCH_SIZE if rem >= cfg.TRAIN.GRAPH_BATCH_SIZE else rem
        db_inds = self._perm[self._cur:self._cur+cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur >= len(self._db)):
            self._db_inds()
        return db_inds

    def _get_next_minibatch(self):
        inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in inds]
        return minibatch_db, get_minibatch(
            minibatch_db, self._is_training,
            is_about_graph=True, edge_type=self._edge_type,
            win_size=self._win_size, edge_geo_dist_thresh=self._edge_geo_dist_thresh,
            dataset_name=self._dataset_name, datasets_root=self._datasets_root)

    def forward(self):
        return self._get_next_minibatch()

    def reinit(self, db, is_training, edge_type='srns_geo_dist_binary',
               win_size=8, edge_geo_dist_thresh=20,
               dataset_name: str=None, datasets_root: str=None):
        self._db = db; self._is_training = is_training
        self._edge_type = edge_type; self._win_size = int(win_size)
        self._edge_geo_dist_thresh = float(edge_geo_dist_thresh)
        self._dataset_name = dataset_name; self._datasets_root = datasets_root
        (self._shuffle_db_inds() if self._is_training else self._db_inds())

# -----------------------------------------------------------------------------
# minibatch builders
# -----------------------------------------------------------------------------
def get_minibatch(minibatch_db: List[str], is_training: bool,
                  is_about_graph=False, edge_type='srns_geo_dist_binary',
                  win_size=8, edge_geo_dist_thresh=20, use_padding=False,
                  dataset_name: str=None, datasets_root: str=None):
    if not is_about_graph:
        im_blob, label_blob, fov_blob = _get_image_fov_blob(minibatch_db, is_training, use_padding=use_padding)
        return {'img': im_blob, 'label': label_blob, 'fov': fov_blob}

    blobs = _get_graph_fov_blob(
        minibatch_db, is_training, edge_type, win_size, edge_geo_dist_thresh,
        dataset_name=dataset_name, datasets_root=datasets_root)
    (im_blob, label_blob, fov_blob, probmap_blob,
     all_union_graph, num_of_nodes_list, vec_aug_on, rot_angle) = blobs
    return {
        'img': im_blob, 'label': label_blob, 'fov': fov_blob, 'probmap': probmap_blob,
        'graph': all_union_graph, 'num_of_nodes_list': num_of_nodes_list,
        'vec_aug_on': vec_aug_on, 'rot_angle': rot_angle
    }

def _get_image_fov_blob(minibatch_db: List[str], is_training: bool, use_padding=False):
    """Original image+label+fov loader used by test code (kept)."""
    num_images = len(minibatch_db)
    processed_ims, processed_labels, processed_fovs = [], [], []

    # Infer dataset from path
    probe = minibatch_db[0]
    if 'DRIVE' in probe:
        im_ext, label_ext, fov_ext = '_image.tif', '_label.gif', '_mask.gif'
        pixel_mean, (len_y, len_x) = cfg.PIXEL_MEAN_DRIVE, (592, 592)
    elif 'STARE' in probe:
        im_ext, label_ext, fov_ext = '.ppm', '.ah.ppm', '_mask.png'
        pixel_mean, (len_y, len_x) = cfg.PIXEL_MEAN_STARE, (704, 704)
    elif 'CHASE_DB1' in probe:
        im_ext, label_ext, fov_ext = '.jpg', '_1stHO.png', '_mask.tif'
        pixel_mean, (len_y, len_x) = cfg.PIXEL_MEAN_CHASE_DB1, (1024, 1024)
    elif 'HRF' in probe:
        im_ext, label_ext, fov_ext = '.bmp', '.tif', '_mask.tif'
        pixel_mean, (len_y, len_x) = cfg.PIXEL_MEAN_HRF, (768, 768)
    else:
        raise ValueError("Unknown dataset in path: {}".format(probe))

    for i in xrange(num_images):
        im = skimage.io.imread(minibatch_db[i]+im_ext)
        label = skimage.io.imread(minibatch_db[i]+label_ext).reshape((-1, -1, 1))
        fov = skimage.io.imread(minibatch_db[i]+fov_ext)
        fov = fov if fov.ndim == 3 else fov[..., None]
        if use_padding:
            im_p = np.zeros((len_y, len_x, 3), dtype=im.dtype); im_p[:im.shape[0], :im.shape[1], :] = im; im = im_p
            lab_p = np.zeros((len_y, len_x, 1), dtype=label.dtype); lab_p[:label.shape[0], :label.shape[1], :] = label; label = lab_p
            fov_p = np.zeros((len_y, len_x, 1), dtype=fov.dtype); fov_p[:fov.shape[0], :fov.shape[1], :] = fov; fov = fov_p
        proc_im, proc_lab, proc_fov, _ = prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training)
        processed_ims.append(proc_im); processed_labels.append(proc_lab); processed_fovs.append(proc_fov)

    return im_list_to_blob(processed_ims), im_list_to_blob(processed_labels), im_list_to_blob(processed_fovs)

def _get_graph_fov_blob(minibatch_db: List[str], is_training: bool,
                        edge_type='srns_geo_dist_binary', win_size=8, edge_geo_dist_thresh=20,
                        dataset_name: str=None, datasets_root: str=None):
    """Build an input blob from the graphs in the minibatch."""
    num_graphs = len(minibatch_db)
    processed_ims, processed_labels, processed_fovs, processed_probmaps = [], [], [], []
    all_graphs, num_of_nodes_list = [], []

    # infer split from base path ('train' vs 'test')
    base0 = minibatch_db[0].replace('\\', '/').lower()
    split = 'training' if ('/train/' in base0 or '/training/' in base0) else 'testing'
    ds = dataset_name or ('DRIVE' if 'drive' in base0 else
                          'STARE' if 'stare' in base0 else
                          'CHASE_DB1' if 'chase' in base0 else
                          'HRF')
    meta = _meta_for_dataset(ds)
    ds_root = datasets_root or r"C:/Users/rog/THESIS/DATASETS"
    im_root_path = os.path.join(ds_root, _dataset_dir_name(ds), split)

    for i in xrange(num_graphs):
        cur_base = minibatch_db[i]
        cur_name = os.path.basename(cur_base)

        im = skimage.io.imread(os.path.join(im_root_path, cur_name + meta['im_ext']))
        label = skimage.io.imread(os.path.join(im_root_path, cur_name + meta['label_ext'])).reshape((-1, -1, 1))
        fov = skimage.io.imread(os.path.join(im_root_path, cur_name + meta['fov_ext']))
        fov = fov if fov.ndim == 3 else fov[..., None]
        probmap = skimage.io.imread(cur_base + '_prob.png')
        probmap = probmap.reshape((probmap.shape[0], probmap.shape[1], 1))

        # pad to canonical size
        len_y, len_x = meta['size']
        def pad_to(arr, tgt_hw, ch=1):
            out = np.zeros((tgt_hw[0], tgt_hw[1], arr.shape[2] if arr.ndim == 3 else ch), dtype=arr.dtype)
            out[:arr.shape[0], :arr.shape[1], ...] = arr
            return out
        im = pad_to(im, (len_y, len_x), 3)
        label = pad_to(label, (len_y, len_x), 1)
        fov = pad_to(fov, (len_y, len_x), 1)
        probmap = pad_to(probmap, (len_y, len_x), 1)

        # graph
        if 'srns' not in edge_type:
            raise NotImplementedError
        graph_path = _pick_graph_path(cur_base, win_size, edge_geo_dist_thresh)
        with open(graph_path, 'rb') as gf:
            union_graph = pickle.load(gf)
        union_graph = nx.convert_node_labels_to_integers(union_graph)
        n_nodes = union_graph.number_of_nodes()

        # map nodes into current crop/frame (no crop here; still keep code shape)
        node_idx_map = np.zeros(im.shape[:2], dtype=np.int32)
        for j in range(n_nodes):
            y, x = union_graph.nodes[j]['y'], union_graph.nodes[j]['x']
            node_idx_map[y, x] = j + 1

        proc_im, proc_lab, proc_fov, proc_prob, proc_node_idx_map, vec_aug_on, (cy1, cy2, cx1, cx2), rot_angle = \
            prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, meta['pixel_mean'], is_training, win_size)

        processed_ims.append(proc_im); processed_labels.append(proc_lab)
        processed_fovs.append(proc_fov); processed_probmaps.append(proc_prob)

        # relocate nodes after crop/augs
        ys, xs = np.where(proc_node_idx_map)
        for j in range(len(ys)):
            idx = proc_node_idx_map[ys[j], xs[j]] - 1
            union_graph.nodes[idx]['y'] = ys[j]; union_graph.nodes[idx]['x'] = xs[j]
        union_graph = nx.convert_node_labels_to_integers(union_graph)
        all_graphs.append(union_graph)
        num_of_nodes_list.append(union_graph.number_of_nodes())

    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels)
    fov_blob = im_list_to_blob(processed_fovs)
    probmap_blob = im_list_to_blob(processed_probmaps)
    all_union_graph = nx.algorithms.operators.all.disjoint_union_all(all_graphs)
    # simple flags (no per-image randomness exposed back out except in this vector)
    vec_aug_on = np.zeros((7,), dtype=np.bool_)  # placeholder; real flags come from prep_*
    rot_angle = 0
    return im_blob, label_blob, fov_blob, probmap_blob, all_union_graph, num_of_nodes_list, vec_aug_on, rot_angle

# -----------------------------------------------------------------------------
# image prep & augmentation (kept from original, Python3-safe)
# -----------------------------------------------------------------------------
def prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training):
    im = im.astype(np.float32, copy=False)/255.; label = label.astype(np.float32)/255.; fov = fov.astype(np.float32)/255.
    vec_aug_on = np.zeros((7,), dtype=np.bool_)
    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True; im = im[:, ::-1, :]; label = label[:, ::-1, :]; fov = fov[:, ::-1, :]
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True; im = im[::-1, :, :]; label = label[::-1, :, :]; fov = fov[::-1, :, :]
        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            angle = np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE, cfg.TRAIN.ROTATION_MAX_ANGLE)
            im = np.dstack([skimage.transform.rotate(im[:,:,c], angle, cval=pixel_mean[c]/255.) for c in range(3)])
            label = skimage.transform.rotate(label, angle, order=0, cval=0.)
            fov = skimage.transform.rotate(fov, angle, order=0, cval=0.)
        if cfg.TRAIN.USE_SCALING:
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0], cfg.TRAIN.SCALING_RANGE[1])
            im = skimage.transform.rescale(im, scale, preserve_range=True, channel_axis=2, anti_aliasing=False)
            label = skimage.transform.rescale(label, scale, order=0, preserve_range=True, channel_axis=None, anti_aliasing=False)
            fov = skimage.transform.rescale(fov, scale, order=0, preserve_range=True, channel_axis=None, anti_aliasing=False)
        if cfg.TRAIN.USE_CROPPING:
            vec_aug_on[4] = True
            h = int(np.random.randint(int(im.shape[0]*0.5), int(im.shape[0]*0.8)))
            w = int(np.random.randint(int(im.shape[1]*0.5), int(im.shape[1]*0.8)))
            y1 = int(np.random.randint(0, im.shape[0]-h)); x1 = int(np.random.randint(0, im.shape[1]-w))
            y2 = y1+h; x2 = x1+w
            im, label, fov = im[y1:y2, x1:x2, :], label[y1:y2, x1:x2, :], fov[y1:y2, x1:x2, :]
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            vec_aug_on[5] = True; im = np.clip(im + np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA, cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA), 0, 1)
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            vec_aug_on[6] = True; m = np.mean(im); im = np.clip((im-m)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR, cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR)+m, 0, 1)
    im = (im - np.array(pixel_mean)/255.) * 255.
    label = (label >= 0.5); fov = (fov >= 0.5)
    return im, label, fov, vec_aug_on

def prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, pixel_mean, is_training, win_size):
    # Normalize & augs; match the original behavior but with modern skimage args
    im = im.astype(np.float32, copy=False)/255.; label = label.astype(np.float32)/255.; fov = fov.astype(np.float32)/255.; probmap = probmap.astype(np.float32)/255.
    vec_aug_on = np.zeros((7,), dtype=np.bool_)
    rot_angle = 0
    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True; im = im[:, ::-1, :]; label = label[:, ::-1, :]; fov = fov[:, ::-1, :]; probmap = probmap[:, ::-1, :]; node_idx_map = node_idx_map[:, ::-1]
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True; im = im[::-1, :, :]; label = label[::-1, :, :]; fov = fov[::-1, :, :]; probmap = probmap[::-1, :, :]; node_idx_map = node_idx_map[::-1, :]
        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            rot_angle = int(np.random.choice([0, 90, 180, 270]))
            def _rot(a, order=1, ch_axis=None):
                return skimage.transform.rotate(a, rot_angle, cval=0., order=order, preserve_range=True, resize=True)
            im_r = _rot(im[:,:,0], order=1); im_g = _rot(im[:,:,1], order=1); im_b = _rot(im[:,:,2], order=1); im = np.dstack((im_r, im_g, im_b))
            label = _rot(label, order=0); fov = _rot(fov, order=0); probmap = _rot(probmap, order=1); node_idx_map = _rot(node_idx_map, order=0)
            # crop back to original size
            im = im[:label.shape[0], :label.shape[1], :]; fov = fov[:label.shape[0], :label.shape[1], :]; probmap = probmap[:label.shape[0], :label.shape[1], :]; node_idx_map = node_idx_map[:label.shape[0], :label.shape[1]]
        if cfg.TRAIN.USE_SCALING:
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0], cfg.TRAIN.SCALING_RANGE[1])
            im = skimage.transform.rescale(im, scale, preserve_range=True, channel_axis=2, anti_aliasing=False)
            label = skimage.transform.rescale(label, scale, order=0, preserve_range=True, channel_axis=None, anti_aliasing=False)
            fov = skimage.transform.rescale(fov, scale, order=0, preserve_range=True, channel_axis=None, anti_aliasing=False)
            probmap = skimage.transform.rescale(probmap, scale, preserve_range=True, channel_axis=None, anti_aliasing=False)
            node_idx_map = skimage.transform.rescale(node_idx_map, scale, order=0, preserve_range=True, channel_axis=None, anti_aliasing=False)
        if cfg.TRAIN.USE_CROPPING:
            vec_aug_on[4] = True
            # crop sizes aligned to win_size
            h = (int(np.random.randint(int(im.shape[0]*0.5), int(im.shape[0]*0.8))) // win_size) * win_size
            w = (int(np.random.randint(int(im.shape[1]*0.5), int(im.shape[1]*0.8))) // win_size) * win_size
            y1 = int(np.random.randint(0, im.shape[0]-h)); x1 = int(np.random.randint(0, im.shape[1]-w))
            y2, x2 = y1+h, x1+w
            im, label, fov, probmap, node_idx_map = im[y1:y2, x1:x2, :], label[y1:y2, x1:x2, :], fov[y1:y2, x1:x2, :], probmap[y1:y2, x1:x2, :], node_idx_map[y1:y2, x1:x2]
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            vec_aug_on[5] = True; im = np.clip(im + np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA, cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA), 0, 1)
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            vec_aug_on[6] = True; m = np.mean(im); im = np.clip((im-m)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR, cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR)+m, 0, 1)
    im = (im - np.array(pixel_mean)/255.) * 255.
    label = (label >= 0.5); fov = (fov >= 0.5)
    return im, label, fov, probmap, node_idx_map, vec_aug_on, (0, 0, 0, 0), rot_angle  # crop coords unused by caller

# -----------------------------------------------------------------------------
# misc utils that the training code relies on
# -----------------------------------------------------------------------------
def im_list_to_blob(ims: List[np.ndarray]) -> np.ndarray:
    """Stack a list of HxWxC images into a 4D blob (N,H,W,C)."""
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    blob = np.zeros((len(ims), max_shape[0], max_shape[1], max_shape[2]), dtype=np.float32)
    for i, im in enumerate(ims):
        blob[i, :im.shape[0], :im.shape[1], :im.shape[2]] = im
    return blob

def find(s: str, sub: str) -> List[int]:
    """Return all start indices of substring sub in s (inclusive)."""
    out, i = [], s.find(sub)
    while i != -1:
        out.append(i); i = s.find(sub, i+1)
    return out

def sparse_to_tuple(sparse_mx):
    """Convert a scipy.sparse matrix to a tuple representation for tf.SparseTensor feed."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape, dtype=np.int64)
    return coords, values, shape

def preprocess_graph_gat(adj):
    """Add self-loops (I+A) and return tf.SparseTensor feed tuple."""
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='coo')
    return sparse_to_tuple(adj)

def visualize_graph(img, graph, show_graph=False, save_graph=False, save_path='graph.png',
                    num_nodes_each_type=None, tp_edges=None, fn_edges=None, fp_edges=None):
    plt.figure(figsize=VIS_FIG_SIZE)
    plt.imshow(img.astype(np.uint8))
    pos = {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) for n in graph.nodes()}
    node_color = VIS_NODE_COLOR[0]
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=VIS_NODE_SIZE, alpha=VIS_ALPHA)
    if tp_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=tp_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[0])
    if fn_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=fn_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[1])
    if fp_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=fp_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[2])
    if save_graph: plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_graph: plt.show()
    plt.close()

def get_auc_ap_score(labels, preds):
    return roc_auc_score(labels, preds), average_precision_score(labels, preds)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def get_node_byx_from_graph(graph: nx.Graph, num_of_nodes_list: List[int]):
    node_byxs = np.zeros((graph.number_of_nodes(), 3), dtype=np.int32)
    node_idx = 0
    for sub_graph_idx, cur_num_nodes in enumerate(num_of_nodes_list):
        for i in range(node_idx, node_idx+cur_num_nodes):
            node_byxs[i, :] = [sub_graph_idx, graph.nodes[i]['y'], graph.nodes[i]['x']]
        node_idx += cur_num_nodes
    return node_byxs

class Timer(object):
    def __init__(self):
        self.total_time = 0.; self.calls = 0; self.start_time = 0.; self.diff = 0.; self.average_time = 0.
    def tic(self): self.start_time = time.time()
    def toc(self, average=True):
        self.diff = time.time() - self.start_time; self.total_time += self.diff; self.calls += 1
        self.average_time = self.total_time / self.calls if average else self.diff
        return self.diff
