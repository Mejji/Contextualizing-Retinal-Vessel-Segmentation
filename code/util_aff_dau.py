# -*- coding: utf-8 -*-
"""
util.py — HRF/DRIVE/CHASE path + shape sanity, cache, and graph loaders.

Key fixes for your layout:
- HRF resolver now supports labels/masks colocated in `.../images/` as <base>_label.* & <base>_mask.*,
  not only the classic `manual1/` and `mask/` subfolders.
- Graph tag formatting centralized via _graph_win_tag(int,int) -> "_10_80".
- Cached datalayer returns a dict (uniform downstream handling).
- Shape coercion prevents (H,W,1) channel confusion during skimage ops.

This file is drop-in and backward compatible with your training script.
"""

import os
import re
import time
import pickle
from pathlib import Path

import numpy as np
import numpy.random as npr
import skimage.io
import skimage.transform
import skimage.draw
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

import GenGraph._init_paths  # noqa: F401
from config import cfg

# ---- morphology / viz defaults ----
STR_ELEM = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [0, 0, 0]], dtype=bool)  # eight-neighbors

VIS_FIG_SIZE = (10, 10)
VIS_NODE_SIZE = 50
VIS_ALPHA = 0.5
VIS_NODE_COLOR = ['b', 'r', 'y', 'g']
VIS_EDGE_COLOR = ['b', 'g', 'r']

DEBUG = False  # flip to True to get assertions and shape logs

# ---- Caches for expensive disk I/O ----
_IMAGE_BASE_CACHE = {}   # sample_id -> dict(im_pad, label_pad, fov_pad, prob_pad, graph)
_IMAGE_SIMPLE_CACHE = {} # sample_id -> dict(im, label, fov)


# -------------------------------------------------------------------------
# Dataset helpers
# -------------------------------------------------------------------------
def _infer_dataset_from_string(s: str) -> str:
    u = str(s).upper()
    if 'DRIVE' in u: return 'DRIVE'
    if 'STARE' in u: return 'STARE'
    if 'CHASE' in u: return 'CHASE_DB1'
    if 'HRF'   in u: return 'HRF'
    return getattr(cfg, 'DEFAULT_DATASET', 'DRIVE')


def _dataset_specs(ds: str):
    """Return (pixel_mean(list), canvas_h, canvas_w) for padding use-cases."""
    ds = ds.upper()
    if ds == 'DRIVE':     return cfg.PIXEL_MEAN_DRIVE, 592, 592
    if ds == 'STARE':     return cfg.PIXEL_MEAN_STARE, 704, 704
    if ds == 'CHASE_DB1': return cfg.PIXEL_MEAN_CHASE_DB1, 1024, 1024  # CHASE images ≈999x960; use 1024 canvas
    if ds == 'HRF':       return cfg.PIXEL_MEAN_HRF, 2336, 3504
    return cfg.PIXEL_MEAN_DRIVE, 592, 592


def _find_file_with_exts(dir_path: Path, stem: str, exts=None):
    """Return first existing file in dir_path matching stem + ext (case-sensitive)."""
    if exts:
        for ext in exts:
            cand = dir_path / f"{stem}{ext}"
            if cand.exists():
                return cand
    matches = sorted(dir_path.glob(stem + '.*'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"[util] file not found: {dir_path}/{stem}.*")


# ----------------------------- DRIVE -----------------------------
def _resolve_drive_files(sample_path: str):
    p = Path(sample_path)
    s = str(p).replace('\\', '/').lower()
    subset = 'training' if '/training/' in s or s.endswith('/training') else (
             'test' if '/test/' in s or s.endswith('/test') else None)

    if p.suffix.lower() == '.tif':
        fname = p.name
        m_train = re.match(r'^(\d+)_training\.tif$', fname, flags=re.IGNORECASE)
        m_test  = re.match(r'^(\d+)_test\.tif$', fname, flags=re.IGNORECASE)
        if m_train: img_id, subset = m_train.group(1), 'training'
        elif m_test: img_id, subset = m_test.group(1), 'test'
        else: raise FileNotFoundError(f"[util] Unrecognized DRIVE image filename: {fname}")
        root = p.parent.parent
        img_path = p
    else:
        base = p.name
        m_train = re.match(r'^(\d+)_training$', base, flags=re.IGNORECASE)
        m_test  = re.match(r'^(\d+)_test$', base, flags=re.IGNORECASE)
        if m_train:
            img_id, subset = m_train.group(1), 'training'
        elif m_test:
            img_id, subset = m_test.group(1), 'test'
        else:
            m_any = re.search(r'(\d+)', base)
            if not m_any or subset is None:
                raise FileNotFoundError(f"[util] Can't infer DRIVE id/subset from: {sample_path}")
            img_id = m_any.group(1)
        if 'images' in [q.name.lower() for q in p.parents]:
            cur = p
            while cur.name.lower() != 'images' and cur.parent != cur:
                cur = cur.parent
            root = cur.parent
            img_path = cur / f"{img_id}_{subset}.tif"
        else:
            root = p.parent if p.is_dir() else p.parent.parent
            img_path = root / 'images' / f"{img_id}_{subset}.tif"

    ds_root = root.parent
    label_path = ds_root / subset / '1st_manual' / f"{img_id}_manual1.gif"
    fov_path   = ds_root / subset / 'mask'       / f"{img_id}_{subset}_mask.gif"
    if not fov_path.exists():
        alt = ds_root / subset / 'mask' / f"{img_id}_training_mask.gif"
        if alt.exists(): fov_path = alt

    if not img_path.exists():   raise FileNotFoundError(f"[util] Missing image file: {img_path}")
    if not label_path.exists(): raise FileNotFoundError(f"[util] Missing label file: {label_path}")
    if not fov_path.exists():   raise FileNotFoundError(f"[util] Missing FOV mask file: {fov_path}")
    return str(img_path), str(label_path), str(fov_path)


# ----------------------------- HRF -----------------------------
def _resolve_hrf_files(sample_path: str):
    """
    Robust HRF resolver. Supports two common layouts:

    A) Classic:
       HRF/
         training/
           images/<base>.jpg|tif
           manual1/<base>_manual1.png|tif
           mask/<base>_mask.png|tif
    B) Colocated labels/masks in images/:
           images/<base>.*
           images/<base>_label.*   (or <base>_manual1.*)
           images/<base>_mask.*

    <base> examples: 01_dr, 02_g, 03_h
    """
    p = Path(sample_path)

    # subset
    subset = None
    for part in p.parts:
        low = part.lower()
        if low in ('training', 'train'):
            subset = 'training'; break
        if low in ('testing', 'test'):
            subset = 'testing'; break
    if subset is None:
        subset = 'training'

    # find HRF root
    root = None
    for ancestor in [p] + list(p.parents):
        if ancestor.name.lower() == 'hrf':
            root = ancestor
            break
    if root is None:
        # fallback to your default path
        root = Path('/workspace/DATASETS/HRF')

    # prefer images dir if present; otherwise allow bare <base> folder/file usage
    images_dir = root / subset / 'images'
    manual_dir = root / subset / 'manual1'
    mask_dir   = root / subset / 'mask'

    # canonical base from input (strip *_label/_mask/_manual1 if passed)
    base = p.stem if p.suffix else p.name
    base = re.sub(r'_(label|mask|manual1)$', '', base, flags=re.IGNORECASE)

    # image
    img_search_dirs = [images_dir, root / subset]
    img_exts = ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    img_path = None
    for d in img_search_dirs:
        try:
            img_path = _find_file_with_exts(d, base, img_exts)
            break
        except FileNotFoundError:
            continue
    if img_path is None:
        raise FileNotFoundError(f"[util] HRF image not found for base '{base}' under {img_search_dirs}")

    # label (try manual1/<base>_manual1.*, then images/<base>_label.*)
    label_exts = ['.tif', '.tiff', '.png', '.gif']
    label_candidates = []
    if manual_dir.exists():
        label_candidates.append((manual_dir, f"{base}_manual1"))
    label_candidates.append((images_dir, f"{base}_label"))
    label_path = None
    for d, s in label_candidates:
        try:
            label_path = _find_file_with_exts(d, s, label_exts)
            break
        except FileNotFoundError:
            continue
    if label_path is None:
        raise FileNotFoundError(f"[util] HRF label not found for base '{base}' "
                                f"(tried {[(str(d), s) for d, s in label_candidates]})")

    # FOV mask (try mask/<base>_mask.*, then images/<base>_mask.*)
    fov_exts = ['.tif', '.tiff', '.png']
    fov_candidates = []
    if mask_dir.exists():
        fov_candidates.append((mask_dir, f"{base}_mask"))
    fov_candidates.append((images_dir, f"{base}_mask"))
    fov_path = None
    for d, s in fov_candidates:
        try:
            fov_path = _find_file_with_exts(d, s, fov_exts)
            break
        except FileNotFoundError:
            continue
    if fov_path is None:
        raise FileNotFoundError(f"[util] HRF FOV/mask not found for base '{base}' "
                                f"(tried {[(str(d), s) for d, s in fov_candidates]})")

    return str(img_path), str(label_path), str(fov_path)


# ----------------------------- CHASE_DB1 -----------------------------
def _resolve_chase_files(sample_path: str):
    """
    CHASE_DB1 layout:
      /workspace/DATASETS/CHASE_DB1/{training|train, testing|test}/images/
         Image_01L.jpg(.png)
         Image_01L_1stHO.png   (GT)
         Image_01L_mask.tif    (FOV)
    """
    p = Path(sample_path)

    # find root
    root = None
    for anc in [p] + list(p.parents):
        if anc.name.upper() == 'CHASE_DB1':
            root = anc
            break
    if root is None:
        root = Path('/workspace/DATASETS/CHASE_DB1')

    # subset + images dir
    s = str(p).lower()
    subset = 'training' if ('/training/' in s or '/train/' in s) else ('testing' if ('/testing/' in s or '/test/' in s) else 'training')
    cand_img_dirs = [root / 'training' / 'images', root / 'train' / 'images'] if subset.startswith('train') else \
                    [root / 'testing'  / 'images', root / 'test'  / 'images']
    img_dir = None
    for d in cand_img_dirs:
        if d.exists():
            img_dir = d; break
    if img_dir is None:
        raise FileNotFoundError(f"[util] CHASE images dir missing under {root}")

    # stems
    listed_stem = p.stem  # possibly 'Image_01L_1stHO'
    base_stem = listed_stem[:-6] if listed_stem.lower().endswith('_1stho') else \
                listed_stem[:-5] if listed_stem.lower().endswith('_mask') else \
                listed_stem

    # resolve image
    img_path = None
    for stem_try in (base_stem, listed_stem):
        try:
            img_path = _find_file_with_exts(img_dir, stem_try, ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
            break
        except FileNotFoundError:
            continue
    if img_path is None:
        img_path = _find_file_with_exts(img_dir, base_stem, ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

    # GT and mask
    label_stems = [base_stem + '_1stHO', listed_stem] if not listed_stem.lower().endswith('_mask') else [listed_stem.replace('_mask', '_1stHO')]
    fov_stems   = [base_stem + '_mask',  listed_stem] if not listed_stem.lower().endswith('_1stHO') else [listed_stem.replace('_1stHO', '_mask')]

    label_path = None
    for ls in label_stems:
        try:
            label_path = _find_file_with_exts(img_dir, ls, ['.png', '.tif', '.tiff', '.gif'])
            break
        except FileNotFoundError:
            continue
    if label_path is None:
        raise FileNotFoundError(f"[util] CHASE label not found for stems {label_stems} in {img_dir}")

    fov_path = None
    for ms in fov_stems:
        try:
            fov_path = _find_file_with_exts(img_dir, ms, ['.png', '.tif', '.tiff'])
            break
        except FileNotFoundError:
            continue
    if fov_path is None:
        raise FileNotFoundError(f"[util] CHASE mask not found for stems {fov_stems} in {img_dir}")

    return str(img_path), str(label_path), str(fov_path)


# -------------------------------------------------------------------------
# skimage transform wrappers (channel-safe)
# -------------------------------------------------------------------------
def _rotate_img(x, angle, order=1, cval=0.0, resize=False):
    kw = dict(order=order, mode='constant', cval=cval, resize=resize, preserve_range=True)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        try:
            return skimage.transform.rotate(x, angle, channel_axis=-1, **kw)
        except TypeError:
            return skimage.transform.rotate(x, angle, multichannel=True, **kw)
    else:
        try:
            return skimage.transform.rotate(x, angle, channel_axis=None, **kw)
        except TypeError:
            return skimage.transform.rotate(x, angle, multichannel=False, **kw)


def _rescale_img(x, scale, order=1):
    kw = dict(scale=scale, order=order, preserve_range=True, anti_aliasing=False)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        try:
            return skimage.transform.rescale(x, **kw, channel_axis=-1)
        except TypeError:
            return skimage.transform.rescale(x, **kw, multichannel=True)
    else:
        try:
            return skimage.transform.rescale(x, **kw, channel_axis=None)
        except TypeError:
            return skimage.transform.rescale(x, **kw, multichannel=False)


def _resize_2d(x, out_hw, order=0):
    H, W = out_hw
    if x.ndim == 2:
        return skimage.transform.resize(x, (H, W), order=order, preserve_range=True, anti_aliasing=False)
    elif x.ndim == 3 and x.shape[-1] in (1, 3):
        try:
            return skimage.transform.resize(x, (H, W), order=order, preserve_range=True,
                                            anti_aliasing=False, channel_axis=-1)
        except TypeError:
            return skimage.transform.resize(x, (H, W), order=order, preserve_range=True,
                                            anti_aliasing=False, multichannel=True)
    else:
        return skimage.transform.resize(np.squeeze(x), (H, W), order=order, preserve_range=True, anti_aliasing=False)


# -------------------------------------------------------------------------
# Shape coercion helpers
# -------------------------------------------------------------------------
def _to_hw1(x):
    a = np.asarray(x)
    if a.ndim == 2:
        return a[..., None]
    if a.ndim == 3:
        if a.shape[-1] in (1, 3):
            return a[..., :1]
        if a.shape[0] == 1:
            return np.transpose(a, (1, 2, 0))
        if a.shape[1] == 1:
            return np.transpose(a, (0, 2, 1))
        return a[..., :1]
    raise ValueError(f"[util] Cannot coerce shape {a.shape} to (H,W,1)")


def _fit_mask_to_image(mask, ref_hw, name="mask"):
    H, W = ref_hw
    m = _to_hw1(mask)
    h, w = m.shape[:2]
    if (h, w) == (H, W):
        return m
    if (h, w) == (W, H):
        return np.transpose(m, (1, 0, 2))
    return _resize_2d(m, (H, W), order=0)[..., :1]


def _fit_rgb_to_image(im):
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"[util] Unexpected image shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


# -------------------------------------------------------------------------
# Graph tag helpers
# -------------------------------------------------------------------------
def _graph_win_tag(win_size, edge_geo_dist_thresh):
    return '_%.2d_%.2d' % (int(win_size), int(edge_geo_dist_thresh))


def _stems_for_graph(base_stem: str):
    """Return valid stems for graph lookup. Only strip _1stHO/_mask suffixes."""
    s = base_stem
    stems = [s]
    low = s.lower()
    if low.endswith('_1stho'):
        stems.append(s[:-6])
    elif low.endswith('_mask'):
        stems.append(s[:-5])
    return stems


def _graph_roots_in_order(is_training: bool):
    """Prefer the split dir, but also try the opposite split as fallback."""
    train_root = Path(getattr(cfg.PATHS, 'GRAPH_TRAIN_DIR', ''))
    test_root  = Path(getattr(cfg.PATHS, 'GRAPH_TEST_DIR',  ''))
    roots = []
    if is_training:
        if str(train_root): roots.append(train_root)
        if str(test_root):  roots.append(test_root)
    else:
        if str(test_root):  roots.append(test_root)
        if str(train_root): roots.append(train_root)
    # Dedup
    seen, uniq = set(), []
    for r in roots:
        if r and r not in seen:
            uniq.append(r); seen.add(r)
    return uniq


# -------------------------------------------------------------------------
# Data layers
# -------------------------------------------------------------------------
class DataLayer(object):
    def __init__(self, db, is_training, use_padding=False):
        self._db = db
        self._is_training = is_training
        self._use_padding = use_padding
        self.img_names = list(db)
        if self._is_training: self._shuffle_db_inds()
        else: self._db_inds()

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _db_inds(self):
        self._perm = np.arange(len(self._db))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        cur_batch_size = cfg.TRAIN.BATCH_SIZE
        if self._is_training:
            if self._cur + cfg.TRAIN.BATCH_SIZE > len(self._db):
                self._shuffle_db_inds()
        else:
            rem = len(self._db) - self._cur
            cur_batch_size = cfg.TRAIN.BATCH_SIZE if rem >= cfg.TRAIN.BATCH_SIZE else rem
        db_inds = self._perm[self._cur:self._cur + cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur >= len(self._db)):
            self._db_inds()
        return db_inds

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return minibatch_db, get_minibatch(minibatch_db, self._is_training, use_padding=self._use_padding)

    def forward(self):
        img_list, blobs = self._get_next_minibatch()
        return img_list, blobs


class GraphDataLayer(object):
    """Original graph datalayer (no caching)."""
    def __init__(self, db, is_training,
                 edge_type='srns_geo_dist_binary',
                 win_size=8, edge_geo_dist_thresh=20):
        self._db = db
        self._is_training = is_training
        self._edge_type = edge_type
        self._win_size = win_size
        self._edge_geo_dist_thresh = edge_geo_dist_thresh
        if self._is_training: self._shuffle_db_inds()
        else: self._db_inds()

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _db_inds(self):
        self._perm = np.arange(len(self._db))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        cur_batch_size = getattr(cfg.TRAIN, 'GRAPH_BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
        if self._is_training:
            if self._cur + cur_batch_size > len(self._db):
                self._shuffle_db_inds()
        else:
            rem = len(self._db) - self._cur
            cur_batch_size = cur_batch_size if rem >= cur_batch_size else rem
        db_inds = self._perm[self._cur:self._cur + cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur >= len(self._db)):
            self._db_inds()
        return db_inds

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return minibatch_db, get_minibatch(minibatch_db, self._is_training,
                                           is_about_graph=True,
                                           edge_type=self._edge_type,
                                           win_size=self._win_size,
                                           edge_geo_dist_thresh=self._edge_geo_dist_thresh)

    def forward(self):
        img_list, blobs = self._get_next_minibatch()
        return img_list, blobs

    def reinit(self, db, is_training,
               edge_type='srns_geo_dist_binary',
               win_size=8, edge_geo_dist_thresh=20):
        self._db = db
        self._is_training = is_training
        self._edge_type = edge_type
        self._win_size = win_size
        self._edge_geo_dist_thresh = edge_geo_dist_thresh
        if self._is_training: self._shuffle_db_inds()
        else: self._db_inds()


class GraphDataLayerCached(GraphDataLayer):
    """
    Cached graph datalayer:
      - loads image/label/fov/probmap/graph ONCE per sample
      - stores padded base arrays + graph in RAM
      - applies augmentation per-iteration
      - returns a dict
    """
    def __init__(self, db, is_training,
                 edge_type='srns_geo_dist_binary',
                 win_size=8, edge_geo_dist_thresh=20):
        super().__init__(db, is_training, edge_type, win_size, edge_geo_dist_thresh)
        self._base_cache = _IMAGE_BASE_CACHE

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        blobs = self._get_graph_fov_blob_cached(
            minibatch_db,
            self._is_training,
            self._edge_type,
            self._win_size,
            self._edge_geo_dist_thresh
        )
        return minibatch_db, blobs

    def _get_graph_fov_blob_cached(self, minibatch_db, is_training,
                                   edge_type='srns_geo_dist_binary',
                                   win_size=8, edge_geo_dist_thresh=20):
        num_graphs = len(minibatch_db)
        processed_ims, processed_labels, processed_fovs, processed_probmaps = [], [], [], []
        all_graphs, num_of_nodes_list = [], []

        ds = _infer_dataset_from_string(minibatch_db[0])
        pixel_mean, len_y, len_x = _dataset_specs(ds)

        vec_aug_on = np.zeros((7,), dtype=bool)
        rot_angle = 0

        for i in range(num_graphs):
            cur_path = minibatch_db[i]

            # -------- base cache --------
            if cur_path in self._base_cache:
                base = self._base_cache[cur_path]
                im_pad   = base['im_pad']
                label_pad= base['label_pad']
                fov_pad  = base['fov_pad']
                prob_pad = base['prob_pad']
                union_graph = base['graph']
            else:
                # resolve files per dataset
                if ds == 'DRIVE':
                    im_path, label_path, fov_path = _resolve_drive_files(cur_path)
                elif ds == 'HRF':
                    im_path, label_path, fov_path = _resolve_hrf_files(cur_path)
                elif ds == 'CHASE_DB1':
                    im_path, label_path, fov_path = _resolve_chase_files(cur_path)
                else:
                    raise NotImplementedError(f"GraphDataLayerCached not wired for dataset: {ds}")

                im = skimage.io.imread(im_path)
                label = skimage.io.imread(label_path)
                fov = skimage.io.imread(fov_path)

                im = _fit_rgb_to_image(im)
                H, W = im.shape[:2]
                label = _fit_mask_to_image(label, (H, W), name="label")
                fov   = _fit_mask_to_image(fov,   (H, W), name="fov")

                # Probmap heuristic (rare; this is NOT the DAU2 prob)
                base_no_ext = str(Path(im_path).with_suffix(''))
                prob_guess = base_no_ext + '_prob.png'
                if Path(prob_guess).exists():
                    pm = skimage.io.imread(prob_guess)
                    probmap = _fit_mask_to_image(pm, (H, W), name="probmap")
                else:
                    probmap = np.zeros((H, W, 1), dtype=np.uint8)

                # Pad to canonical canvas
                def _pad(x, target_hw):
                    Ty, Tx = target_hw
                    out = np.zeros((Ty, Tx, x.shape[-1]), dtype=x.dtype)
                    out[:x.shape[0], :x.shape[1], :] = x
                    return out

                im_pad     = _pad(im,     (len_y, len_x))
                label_pad  = _pad(label,  (len_y, len_x))
                fov_pad    = _pad(fov,    (len_y, len_x))
                prob_pad   = _pad(probmap,(len_y, len_x))

                # Graph loading
                if 'srns' not in edge_type:
                    raise NotImplementedError("Only 'srns_*' edge types are supported")

                win_size_str = _graph_win_tag(win_size, edge_geo_dist_thresh)
                base_stem = Path(cur_path).stem
                stems_try = _stems_for_graph(base_stem)

                graph_path = None
                for root in _graph_roots_in_order(is_training):
                    for stem in stems_try:
                        cand = Path(root) / f"{stem}{win_size_str}.graph_res"
                        if cand.exists():
                            graph_path = cand
                            break
                    if graph_path is not None:
                        break
                if graph_path is None:
                    tried = [str(Path(r) / f"{s}{win_size_str}.graph_res")
                             for r in _graph_roots_in_order(is_training)
                             for s in stems_try]
                    raise FileNotFoundError("[util] Missing graph file for stems "
                                            f"{stems_try}.\nTried:\n  " + "\n  ".join(tried))

                with open(str(graph_path), 'rb') as gf:
                    graph = pickle.load(gf)

                union_graph = nx.convert_node_labels_to_integers(graph)

                self._base_cache[cur_path] = dict(
                    im_pad=im_pad,
                    label_pad=label_pad,
                    fov_pad=fov_pad,
                    prob_pad=prob_pad,
                    graph=union_graph
                )

            # -------- per-iteration augmentation --------
            (processed_im, processed_label, processed_fov, processed_probmap, _node_idx_map,
             vec_aug_on, (crop_y1, crop_y2, crop_x1, crop_x2), rot_angle) = \
                prep_im_label_fov_probmap_for_blob(
                    im_pad.copy(), label_pad.copy(), fov_pad.copy(), prob_pad.copy(),
                    np.zeros(im_pad.shape[:2], dtype=np.int32),
                    pixel_mean, is_training, win_size
                )

            processed_ims.append(processed_im)
            processed_labels.append(_fit_mask_to_image(processed_label, processed_im.shape[:2], name="label_aug"))
            processed_fovs.append(_fit_mask_to_image(processed_fov, processed_im.shape[:2], name="fov_aug"))
            processed_probmaps.append(_fit_mask_to_image(processed_probmap, processed_im.shape[:2], name="prob_aug"))

            all_graphs.append(union_graph)
            num_of_nodes_list.append(union_graph.number_of_nodes())

        im_blob = im_list_to_blob(processed_ims)
        label_blob = im_list_to_blob(processed_labels).astype(np.int64)
        fov_blob = im_list_to_blob(processed_fovs).astype(np.int64)
        probmap_blob = im_list_to_blob(processed_probmaps)
        all_union_graph = nx.algorithms.operators.all.disjoint_union_all(all_graphs)

        # Return **dict** for uniform downstream handling
        return {
            'img': im_blob,
            'label': label_blob,
            'fov': fov_blob,
            'probmap': probmap_blob,
            'graph': all_union_graph,
            'num_of_nodes_list': num_of_nodes_list,
            'vec_aug_on': vec_aug_on,
            'rot_angle': rot_angle
        }


# -------------------------------------------------------------------------
# Minibatch builders (kept same API)
# -------------------------------------------------------------------------
def get_minibatch(minibatch_db, is_training,
                  is_about_graph=False,
                  edge_type='srns_geo_dist_binary',
                  win_size=8, edge_geo_dist_thresh=20,
                  use_padding=False):
    if not is_about_graph:
        im_blob, label_blob, fov_blob = _get_image_fov_blob(minibatch_db, is_training, use_padding=use_padding)
        blobs = {'img': im_blob, 'label': label_blob, 'fov': fov_blob}
    else:
        (im_blob, label_blob, fov_blob, probmap_blob,
         all_union_graph,
         num_of_nodes_list, vec_aug_on, rot_angle) = _get_graph_fov_blob(minibatch_db, is_training,
                                                                         edge_type, win_size, edge_geo_dist_thresh)
        blobs = {'img': im_blob, 'label': label_blob, 'fov': fov_blob, 'probmap': probmap_blob,
                 'graph': all_union_graph,
                 'num_of_nodes_list': num_of_nodes_list,
                 'vec_aug_on': vec_aug_on,
                 'rot_angle': rot_angle}
    return blobs


def _get_image_fov_blob(minibatch_db, is_training, use_padding=False):
    num_images = len(minibatch_db)
    processed_ims, processed_labels, processed_fovs = [], []

    ds = _infer_dataset_from_string(minibatch_db[0])
    pixel_mean, len_y, len_x = _dataset_specs(ds)

    for i in range(num_images):
        base = minibatch_db[i]

        if base in _IMAGE_SIMPLE_CACHE:
            entry = _IMAGE_SIMPLE_CACHE[base]
            im = entry['im'].copy()
            label = entry['label'].copy()
            fov = entry['fov'].copy()
        else:
            if ds == 'DRIVE':
                im_path, label_path, fov_path = _resolve_drive_files(base)
            elif ds == 'HRF':
                im_path, label_path, fov_path = _resolve_hrf_files(base)
            elif ds == 'CHASE_DB1':
                im_path, label_path, fov_path = _resolve_chase_files(base)
            else:
                raise NotImplementedError(f"Image path resolver not wired for dataset: {ds}")

            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)

            im = _fit_rgb_to_image(im)
            H, W = im.shape[:2]
            label = _fit_mask_to_image(label, (H, W), name="label")
            fov   = _fit_mask_to_image(fov,   (H, W), name="fov")

            _IMAGE_SIMPLE_CACHE[base] = dict(im=im, label=label, fov=fov)
            im = im.copy(); label = label.copy(); fov = fov.copy()

        if use_padding:
            H, W = im.shape[:2]
            canvas = np.zeros((len_y, len_x, 3), dtype=im.dtype); canvas[:H, :W, :] = im; im = canvas
            canv1 = np.zeros((len_y, len_x, 1), dtype=label.dtype); canv1[:H, :W, :] = label; label = canv1
            canv2 = np.zeros((len_y, len_x, 1), dtype=fov.dtype);   canv2[:H, :W, :] = fov;   fov   = canv2

        processed_im, processed_label, processed_fov, _ = prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training)
        Hi, Wi = processed_im.shape[:2]
        processed_label = _fit_mask_to_image(processed_label, (Hi, Wi), name="label_aug")
        processed_fov   = _fit_mask_to_image(processed_fov,   (Hi, Wi), name="fov_aug")

        processed_ims.append(processed_im)
        processed_labels.append(processed_label)
        processed_fovs.append(processed_fov)

    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels).astype(np.int64)
    fov_blob = im_list_to_blob(processed_fovs).astype(np.int64)
    return im_blob, label_blob, fov_blob


def _get_graph_fov_blob(minibatch_db, is_training, edge_type='srns_geo_dist_binary',
                        win_size=8, edge_geo_dist_thresh=20):
    """Legacy no-cache graph loader (kept for API compatibility)."""
    num_graphs = len(minibatch_db)
    processed_ims, processed_labels, processed_fovs, processed_probmaps = [], [], [], []
    all_graphs, num_of_nodes_list = [], []

    ds = _infer_dataset_from_string(minibatch_db[0])
    pixel_mean, len_y, len_x = _dataset_specs(ds)

    for i in range(num_graphs):
        cur_path = minibatch_db[i]

        if ds == 'DRIVE':
            im_path, label_path, fov_path = _resolve_drive_files(cur_path)
        elif ds == 'HRF':
            im_path, label_path, fov_path = _resolve_hrf_files(cur_path)
        elif ds == 'CHASE_DB1':
            im_path, label_path, fov_path = _resolve_chase_files(cur_path)
        else:
            raise NotImplementedError(f"Graph path resolver not wired for dataset: {ds}")

        im = skimage.io.imread(im_path)
        label = skimage.io.imread(label_path)
        fov = skimage.io.imread(fov_path)

        im = _fit_rgb_to_image(im)
        H, W = im.shape[:2]
        label = _fit_mask_to_image(label, (H, W), name="label")
        fov   = _fit_mask_to_image(fov,   (H, W), name="fov")

        # Probmap heuristic (rare; not DAU2)
        base_no_ext = str(Path(im_path).with_suffix(''))
        prob_guess = base_no_ext + '_prob.png'
        if Path(prob_guess).exists():
            pm = skimage.io.imread(prob_guess)
            probmap = _fit_mask_to_image(pm, (H, W), name="probmap")
        else:
            probmap = np.zeros((H, W, 1), dtype=np.uint8)

        # Pad to canvas
        def _pad(x, target_hw, fill=0):
            Ty, Tx = target_hw
            out = np.zeros((Ty, Tx, x.shape[-1]), dtype=x.dtype)
            out[:x.shape[0], :x.shape[1], :] = x
            return out

        im_pad     = _pad(im,     (len_y, len_x))
        label_pad  = _pad(label,  (len_y, len_x))
        fov_pad    = _pad(fov,    (len_y, len_x))
        prob_pad   = _pad(probmap,(len_y, len_x))

        (processed_im, processed_label, processed_fov, processed_probmap, processed_node_idx_map,
         vec_aug_on, (crop_y1, crop_y2, crop_x1, crop_x2), rot_angle) = \
            prep_im_label_fov_probmap_for_blob(im_pad, label_pad, fov_pad, prob_pad,
                                               np.zeros(im_pad.shape[:2], dtype=np.int32),
                                               pixel_mean, is_training, win_size)

        processed_ims.append(processed_im)
        processed_labels.append(_fit_mask_to_image(processed_label, processed_im.shape[:2], name="label_aug"))
        processed_fovs.append(_fit_mask_to_image(processed_fov, processed_im.shape[:2], name="fov_aug"))
        processed_probmaps.append(_fit_mask_to_image(processed_probmap, processed_im.shape[:2], name="prob_aug"))

        # Graph loading
        if 'srns' not in edge_type:
            raise NotImplementedError("Only 'srns_*' edge types are supported")

        win_size_str = _graph_win_tag(win_size, edge_geo_dist_thresh)
        base_stem = Path(cur_path).stem
        stems_try = _stems_for_graph(base_stem)

        graph_path = None
        for root in _graph_roots_in_order(is_training):
            for stem in stems_try:
                cand = Path(root) / f"{stem}{win_size_str}.graph_res"
                if cand.exists():
                    graph_path = cand
                    break
            if graph_path is not None:
                break
        if graph_path is None:
            tried = [str(Path(r) / f"{s}{win_size_str}.graph_res")
                     for r in _graph_roots_in_order(is_training)
                     for s in stems_try]
            raise FileNotFoundError("[util] Missing graph file for stems "
                                    f"{stems_try}.\nTried:\n  " + "\n  ".join(tried))

        with open(str(graph_path), 'rb') as gf:
            graph = pickle.load(gf)

        union_graph = nx.convert_node_labels_to_integers(graph)
        all_graphs.append(union_graph)
        num_of_nodes_list.append(union_graph.number_of_nodes())

    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels).astype(np.int64)
    fov_blob = im_list_to_blob(processed_fovs).astype(np.int64)
    probmap_blob = im_list_to_blob(processed_probmaps)
    all_union_graph = nx.algorithms.operators.all.disjoint_union_all(all_graphs)
    return im_blob, label_blob, fov_blob, probmap_blob, all_union_graph, num_of_nodes_list, vec_aug_on, rot_angle


# -------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------
def prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training):
    im = im.astype(np.float32, copy=False) / 255.
    label = label.astype(np.float32, copy=False) / 255.
    fov = fov.astype(np.float32, copy=False) / 255.

    vec_aug_on = np.zeros((7,), dtype=bool)

    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]; label = label[:, ::-1, :]; fov = fov[:, ::-1, :]

        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]; label = label[::-1, :, :]; fov = fov[::-1, :, :]

        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            rot_angle = np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE, cfg.TRAIN.ROTATION_MAX_ANGLE)
            im = _rotate_img(im, rot_angle, order=1, cval=0.)
            label = _rotate_img(label, rot_angle, order=0, cval=0.)
            fov = _rotate_img(fov, rot_angle, order=0, cval=0.)

        if getattr(cfg.TRAIN, 'USE_SCALING', False):
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0], cfg.TRAIN.SCALING_RANGE[1])
            im = _rescale_img(im, scale, order=1)
            label = _rescale_img(label, scale, order=0)
            fov = _rescale_img(fov, scale, order=0)

        if getattr(cfg.TRAIN, 'USE_CROPPING', False):
            vec_aug_on[4] = True
            h, w = im.shape[:2]
            h1 = int(round(h * 0.5)); h2 = int(round(h * 0.8))
            w1 = int(round(w * 0.5)); w2 = int(round(w * 0.8))
            cur_h = np.random.randint(h1, max(h1+1, h2 + 1)) if h2 >= h1 else h
            cur_w = np.random.randint(w1, max(w1+1, w2 + 1)) if w2 >= w1 else w
            cur_h = min(cur_h, h); cur_w = min(cur_w, w)
            y1 = np.random.randint(0, max(1, h - cur_h + 1)); x1 = np.random.randint(0, max(1, w - cur_w + 1))
            y2 = y1 + cur_h; x2 = x1 + cur_w
            im = im[y1:y2, x1:x2, :]
            label = label[y1:y2, x1:x2, :]
            fov = fov[y1:y2, x1:x2, :]

        if getattr(cfg.TRAIN, 'USE_BRIGHTNESS_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_BRIGHTNESS', False)):
            vec_aug_on[5] = True
            delta = getattr(cfg.TRAIN, 'BRIGHTNESS_ADJUSTMENT_MAX_DELTA', getattr(cfg.TRAIN, 'BRIGHTNESS_DELTA', 0.05))
            im += np.random.uniform(-delta, delta); im = np.clip(im, 0, 1)

        if getattr(cfg.TRAIN, 'USE_CONTRAST_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_CONTRAST', False)):
            vec_aug_on[6] = True
            lower = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_LOWER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_LOWER', 0.9))
            upper = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_UPPER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_UPPER', 1.1))
            mm = np.mean(im); im = (im - mm) * np.random.uniform(lower, upper) + mm; im = np.clip(im, 0, 1)

    im -= np.array(cfg.PIXEL_MEAN_DRIVE if len(pixel_mean) == 0 else pixel_mean) / 255.
    im = im * 255.

    label = (label >= 0.5).astype(np.int64)
    fov = (fov >= 0.5).astype(np.int64)
    return im, label, fov, vec_aug_on


def prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, pixel_mean, is_training, win_size):
    im = im.astype(np.float32, copy=False) / 255.
    label = label.astype(np.float32, copy=False) / 255.
    fov = fov.astype(np.float32, copy=False) / 255.
    probmap = probmap.astype(np.float32, copy=False) / 255.

    vec_aug_on = np.zeros((7,), dtype=bool)
    cur_y1 = cur_y2 = cur_x1 = cur_x2 = 0
    rot_angle = 0

    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]; label = label[:, ::-1, :]; fov = fov[:, ::-1, :]; probmap = probmap[:, ::-1, :]; node_idx_map = node_idx_map[:, ::-1]
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]; label = label[::-1, :, :]; fov = fov[::-1, :, :]; probmap = probmap[::-1, :, :]; node_idx_map = node_idx_map[::-1, :]
        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            len_ori_y, len_ori_x = im.shape[:2]
            rot_angle = int(np.random.choice([0, 90, 180, 270]))
            im      = _rotate_img(im,      rot_angle, order=1, cval=0., resize=True)
            label   = _rotate_img(label,   rot_angle, order=0, cval=0., resize=True)
            fov     = _rotate_img(fov,     rot_angle, order=0, cval=0., resize=True)
            probmap = _rotate_img(probmap, rot_angle, order=0, cval=0., resize=True)
            node_idx_map = _resize_2d(node_idx_map, (im.shape[0], im.shape[1]), order=0)
            im      = im[:len_ori_y, :len_ori_x, :]
            label   = label[:len_ori_y, :len_ori_x, :]
            fov     = fov[:len_ori_y, :len_ori_x, :]
            probmap = probmap[:len_ori_y, :len_ori_x, :]
            node_idx_map = node_idx_map[:len_ori_y, :len_ori_x]
        if getattr(cfg.TRAIN, 'USE_SCALING', False):
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0], cfg.TRAIN.SCALING_RANGE[1])
            im      = _rescale_img(im,      scale, order=1)
            label   = _rescale_img(label,   scale, order=0)
            fov     = _rescale_img(fov,     scale, order=0)
            probmap = _rescale_img(probmap, scale, order=0)
            node_idx_map = _resize_2d(node_idx_map, (im.shape[0], im.shape[1]), order=0)
        if getattr(cfg.TRAIN, 'USE_CROPPING', False):
            vec_aug_on[4] = True
            h, w = im.shape[:2]
            h1 = int(round(h * 0.5)); h2 = int(round(h * 0.8))
            w1 = int(round(w * 0.5)); w2 = int(round(w * 0.8))
            cur_h = max(win_size, (np.random.randint(h1, max(h2,h1)+1) // win_size) * win_size) if win_size > 0 else np.random.randint(h1, max(h2,h1)+1)
            cur_w = max(win_size, (np.random.randint(w1, max(w2,w1)+1) // win_size) * win_size) if win_size > 0 else np.random.randint(w1, max(w2,w1)+1)
            cur_h = min(cur_h, h); cur_w = min(cur_w, w)
            y1 = np.random.randint(0, max(1, h - cur_h + 1)); x1 = np.random.randint(0, max(1, w - cur_w + 1))
            cur_y1, cur_y2, cur_x1, cur_x2 = y1, y1 + cur_h, x1, x1 + cur_w
            im      = im[cur_y1:cur_y2, cur_x1:cur_x2, :]
            label   = label[cur_y1:cur_y2, cur_x1:cur_x2, :]
            fov     = fov[cur_y1:cur_y2, cur_x1:cur_x2, :]
            probmap = probmap[cur_y1:cur_y2, cur_x1:cur_x2, :]
            node_idx_map = node_idx_map[cur_y1:cur_y2, cur_x1:cur_x2]
        if getattr(cfg.TRAIN, 'USE_BRIGHTNESS_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_BRIGHTNESS', False)):
            vec_aug_on[5] = True
            delta = getattr(cfg.TRAIN, 'BRIGHTNESS_ADJUSTMENT_MAX_DELTA', getattr(cfg.TRAIN, 'BRIGHTNESS_DELTA', 0.05))
            im += np.random.uniform(-delta, delta); im = np.clip(im, 0, 1)
        if getattr(cfg.TRAIN, 'USE_CONTRAST_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_CONTRAST', False)):
            vec_aug_on[6] = True
            lower = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_LOWER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_LOWER', 0.9))
            upper = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_UPPER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_UPPER', 1.1))
            mm = np.mean(im); im = (im - mm) * np.random.uniform(lower, upper) + mm; im = np.clip(im, 0, 1)

    im -= np.array(cfg.PIXEL_MEAN_DRIVE if len(pixel_mean) == 0 else pixel_mean) / 255.
    im = im * 255.

    label = (label >= 0.5).astype(np.int64)
    fov = (fov >= 0.5).astype(np.int64)

    return im, label, fov, probmap, node_idx_map, vec_aug_on, (cur_y1, cur_y2, cur_x1, cur_x2), rot_angle


# -------------------------------------------------------------------------
# Blob builder / misc
# -------------------------------------------------------------------------
def im_list_to_blob(ims):
    for k, x in enumerate(ims):
        if x.ndim != 3:
            raise ValueError(f"[util] Blob expects 3D arrays, got {x.shape} at index {k}")
        if x.shape[-1] not in (1, 3):
            raise ValueError(f"[util] Last channel must be 1 or 3, got {x.shape} at index {k}")

    max_h = max(im.shape[0] for im in ims)
    max_w = max(im.shape[1] for im in ims)
    ch    = ims[0].shape[2]
    if any(im.shape[2] != ch for im in ims):
        raise ValueError("[util] Mixed channel counts in blob inputs")

    num_images = len(ims)
    blob = np.zeros((num_images, int(max_h), int(max_w), int(ch)), dtype=ims[0].dtype)
    for i in range(num_images):
        im = ims[i]
        blob[i, :im.shape[0], :im.shape[1], :] = im
    return blob


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        return self.average_time if average else self.diff


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph_gat(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return indices, adj.data, adj.shape


def visualize_graph(im, graph, show_graph=False, save_graph=True,
                    num_nodes_each_type=None, custom_node_color=None,
                    tp_edges=None, fn_edges=None, fp_edges=None,
                    save_path='graph.png'):
    plt.figure(figsize=VIS_FIG_SIZE)
    if im.dtype == bool or im.dtype == np.bool_:
        bg = im.astype(int) * 255
    else:
        bg = im
    if len(bg.shape) == 2:
        plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    elif len(bg.shape) == 3:
        plt.imshow(bg)
    plt.axis('off')

    pos = {}
    node_list = list(graph.nodes)
    for i in node_list:
        pos[i] = [graph.nodes[i]['x'], graph.nodes[i]['y']]

    if custom_node_color is not None:
        node_color = custom_node_color
    else:
        if num_nodes_each_type is None:
            node_color = 'b'
        else:
            if not (graph.number_of_nodes() == np.sum(np.array(num_nodes_each_type))):
                raise ValueError('Wrong number of nodes')
            node_color = [VIS_NODE_COLOR[0]] * num_nodes_each_type[0] + [VIS_NODE_COLOR[1]] * num_nodes_each_type[1]

    nx.draw(graph, pos, node_color='green', edge_color='blue', width=1, node_size=10, alpha=VIS_ALPHA)

    if tp_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=tp_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[0])
    if fn_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=fn_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[1])
    if fp_edges is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=fp_edges, width=3, alpha=VIS_EDGE_COLOR[2])

    if save_graph: plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_graph: plt.show()
    plt.cla(); plt.clf(); plt.close()


def get_auc_ap_score(labels, preds):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)
    return auc_score, ap_score


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def get_node_byx_from_graph(graph, num_of_nodes_list):
    node_byxs = np.zeros((graph.number_of_nodes(), 3), dtype=np.int32)
    node_idx = 0
    for sub_graph_idx, cur_num_nodes in enumerate(num_of_nodes_list):
        for i in range(node_idx, node_idx + cur_num_nodes):
            node_byxs[i, :] = [sub_graph_idx, graph.nodes[i]['y'], graph.nodes[i]['x']]
        node_idx = node_idx + cur_num_nodes
    return node_byxs
