# -*- coding: utf-8 -*-
"""
util.py — FIXED for DRIVE/STARE/CHASE/HRF shape sanity + TF feed safety
- Forces label/FOV to (H, W, 1) to match the image before batching
- Uses skimage rotate/rescale with explicit channel axis (prevents (H,W,1) → (1,H,W) disasters)
- Adds hard checks + gentle fallbacks (transpose/resize) when shapes drift
- Python 3 clean
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

import _init_paths
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
    if ds == 'CHASE_DB1': return cfg.PIXEL_MEAN_CHASE_DB1, 960, 999
    if ds == 'HRF':       return cfg.PIXEL_MEAN_HRF, 2336, 3504
    return cfg.PIXEL_MEAN_DRIVE, 592, 592


def _resolve_drive_files(sample_path: str):
    """
    Resolve DRIVE paths from any of these inputs:
      - .../DRIVE/training/images/32_training(.tif)
      - .../DRIVE/test/images/02_test(.tif)
    Returns: (img_path, label_path, fov_path)
    """
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


def _resolve_hrf_files(sample_path: str):
    """
    Resolve HRF image/label/FOV paths whether sample_path is the image file,
    a directory, or a basename without extension.
    """
    p = Path(sample_path)
    subset = None
    for part in p.parts:
        low = part.lower()
        if low in ('training', 'train'):
            subset = 'training'
            break
        if low in ('testing', 'test'):
            subset = 'testing'
            break
    if subset is None:
        subset = 'training'

    root = None
    for ancestor in [p] + list(p.parents):
        if ancestor.name.lower() == 'hrf':
            root = ancestor
            break
    if root is None:
        root = Path('C:/Users/rog/THESIS/DATASETS/HRF')

    img_dir = root / subset / 'images'
    manual_dir = root / subset / 'manual1'
    mask_dir = root / subset / 'mask'
    # HRF variants often name labels as *_label.* and masks as *_mask.* in the images folder.
    # If manual1/mask folders are missing or file lookup fails, fall back to the images dir.
    if not manual_dir.exists():
        manual_dir = root / subset / 'images'
    if not mask_dir.exists():
        mask_dir = root / subset / 'images'
    if not img_dir.exists():
        raise FileNotFoundError(f"[util] HRF images dir missing: {img_dir}")

    base = p.stem if p.suffix else p.name
    img_path = _find_file_with_exts(img_dir, base, ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
    # Labels: try manual1/<base>.*, then images/<base>_label.*
    try:
        label_path = _find_file_with_exts(manual_dir, base, ['.tif', '.tiff', '.png'])
    except FileNotFoundError:
        label_path = _find_file_with_exts(img_dir, base + '_label', ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg'])
    # Masks: try mask/<base>_mask.*, then images/<base>_mask.*
    try:
        fov_path = _find_file_with_exts(mask_dir, base + '_mask', ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg'])
    except FileNotFoundError:
        fov_path = _find_file_with_exts(img_dir, base + '_mask', ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg'])

    return str(img_path), str(label_path), str(fov_path)


def _resolve_chase_db1_files(sample_path: str):
    """
    Resolve CHASE_DB1 image/label/FOV paths whether sample_path is the image file
    or a basename without extension.
    """
    p = Path(sample_path)
    subset = None
    for part in p.parts:
        low = part.lower()
        if low in ('training', 'train'):
            subset = 'training'
            break
        if low in ('testing', 'test'):
            subset = 'test'
            break
    if subset is None:
        subset = 'training'

    root = None
    for ancestor in [p] + list(p.parents):
        low = ancestor.name.lower()
        if low == 'chase_db1':
            root = ancestor
            break
    if root is None:
        root = Path('C:/Users/rog/THESIS/DATASETS/CHASE_DB1')
    else:
        img_check = root / subset / 'images'
        if not img_check.exists():
            parent = root.parent
            if parent.name.lower() == 'chase_db1' and (parent / subset / 'images').exists():
                root = parent

    img_dir = root / subset / 'images'
    if not img_dir.exists():
        alt_root = Path('C:/Users/rog/THESIS/DATASETS/CHASE_DB1')
        alt_img_dir = alt_root / subset / 'images'
        if alt_img_dir.exists():
            root = alt_root
            img_dir = alt_img_dir
        else:
            raise FileNotFoundError(f"[util] CHASE_DB1 images dir missing: {img_dir}")

    label_dir = root / subset / '1stHO'
    if not label_dir.exists():
        label_dir = img_dir
    mask_dir = root / subset / 'mask'
    if not mask_dir.exists():
        mask_dir = img_dir
    if not img_dir.exists():
        raise FileNotFoundError(f"[util] CHASE_DB1 images dir missing: {img_dir}")

    base = p.stem if p.suffix else p.name
    img_path = _find_file_with_exts(img_dir, base, ['.jpg', '.png', '.tif', '.tiff'])

    # CHASE labels are named *_1stHO.* (sometimes variations); fall back to the plain stem if needed.
    label_path = None
    label_stems = [
        base + '_1stHO', base + '_1stho',
        base + '_1stHO1', base + '_1stho1',
        base + '_1st_manual', base + '_manual1',
        base,  # plain stem last as a weak fallback
    ]
    for stem in label_stems:
        try:
            label_path = _find_file_with_exts(label_dir, stem, ['.png', '.tif', '.tiff'])
            break
        except FileNotFoundError:
            continue
    if label_path is None:
        raise FileNotFoundError(f"[util] Missing CHASE_DB1 label for {base} under {label_dir}")

    mask_path = _find_file_with_exts(mask_dir, base + '_mask', ['.tif', '.tiff', '.png'])

    return str(img_path), str(label_path), str(mask_path)


# -------------------------------------------------------------------------
# skimage transform wrappers (channel-safe)
# -------------------------------------------------------------------------
def _rotate_img(x, angle, order=1, cval=0.0, resize=False):
    """Rotate with explicit channel axis if x is (H,W,C)."""
    kw = dict(order=order, mode='constant', cval=cval, resize=resize, preserve_range=True)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        try:
            return skimage.transform.rotate(x, angle, channel_axis=-1, **kw)
        except TypeError:
            return skimage.transform.rotate(x, angle, multichannel=True, **kw)
    else:
        # 2D or weird; treat as single-channel plane
        try:
            return skimage.transform.rotate(x, angle, channel_axis=None, **kw)
        except TypeError:
            return skimage.transform.rotate(x, angle, multichannel=False, **kw)


def _rescale_img(x, scale, order=1):
    """Rescale spatial dims only; keep channels intact."""
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
    """Resize 2D or (H,W,1)/(H,W,3) to out_hw; keep channels."""
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
        # Last resort: treat weird shapes as 2D by squeezing
        return skimage.transform.resize(np.squeeze(x), (H, W), order=order, preserve_range=True, anti_aliasing=False)


# -------------------------------------------------------------------------
# Shape coercion helpers
# -------------------------------------------------------------------------
def _to_hw1(x):
    """Return array shaped (H,W,1). Accepts (H,W), (H,W,1), (H,W,3),
       or botched (W,H,1)/(1,H,W)/(H,1,W) and fixes when obvious."""
    a = np.asarray(x)
    if a.ndim == 2:
        return a[..., None]
    if a.ndim == 3:
        # If channels last & OK
        if a.shape[-1] in (1, 3):
            return a[..., :1]  # keep first channel
        # If obviously swapped: (1,H,W) -> (H,W,1)
        if a.shape[0] == 1:
            return np.transpose(a, (1, 2, 0))
        # If (H,1,W) -> (H,W,1)
        if a.shape[1] == 1:
            return np.transpose(a, (0, 2, 1))
        # If (W,H,1) -> (H,W,1) (we'll detect with a ref later)
        return a[..., :1]
    # 1D or other oddities: flatten is unsafe—caller must fix with ref_hw
    raise ValueError(f"[util] Cannot coerce shape {a.shape} to (H,W,1)")


def _fit_mask_to_image(mask, ref_hw, name="mask"):
    """Coerce mask to (H,W,1) and match ref image size with transpose/resize if needed."""
    H, W = ref_hw
    m = _to_hw1(mask)
    h, w = m.shape[:2]

    if (h, w) == (H, W):
        return m
    # Common swap: (W,H,1) → transpose
    if (h, w) == (W, H):
        m2 = np.transpose(m, (1, 0, 2))
        if DEBUG: print(f"[util] transposed {name}: {(h,w,'1')} -> {(H,W,'1')}")
        return m2
    # Fallback: resize (nearest) to image spatial size
    m2 = _resize_2d(m, (H, W), order=0)[..., :1]
    if DEBUG: print(f"[util] resized {name}: {(h,w,'1')} -> {(H,W,'1')}")
    return m2


def _fit_rgb_to_image(im):
    """Ensure image is float32 RGB (H,W,3)."""
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


def _to_float01_mask(x):
    """
    Cast masks/probmaps to float32 in [0,1].
    - If values are already 0/1 or in [0,1], leave them.
    - If values look like 0..255, scale down.
    """
    arr = x.astype(np.float32, copy=False)
    maxv = float(np.max(arr)) if arr.size else 0.0
    minv = float(np.min(arr)) if arr.size else 0.0
    if maxv > 1.0 or minv < 0.0:
        arr = np.clip(arr, 0.0, 255.0) / 255.0
    return arr


# -------------------------------------------------------------------------
# Data layers
# -------------------------------------------------------------------------
class DataLayer(object):
    def __init__(self, db, is_training, use_padding=False):
        self._db = db
        self._is_training = is_training
        self._use_padding = use_padding
        # expose for optional visualization helpers
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


# -------------------------------------------------------------------------
# Minibatch builders
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
    processed_ims, processed_labels, processed_fovs = [], [], []

    ds = _infer_dataset_from_string(minibatch_db[0])
    pixel_mean, len_y, len_x = _dataset_specs(ds)

    for i in range(num_images):
        base = minibatch_db[i]

        if ds == 'DRIVE':
            im_path, label_path, fov_path = _resolve_drive_files(base)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        elif ds == 'CHASE_DB1':
            im_path, label_path, fov_path = _resolve_chase_db1_files(base)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        elif ds == 'HRF':
            im_path, label_path, fov_path = _resolve_hrf_files(base)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        else:
            # Legacy fallback; adjust extensions per dataset as needed
            if ds == 'STARE':
                im_ext, label_ext, fov_ext = '.ppm', '.ah.ppm', '_mask.png'
            elif ds == 'CHASE_DB1':
                im_ext, label_ext, fov_ext = '.jpg', '_1stHO.png', '_mask.tif'
            elif ds == 'HRF':
                im_ext, label_ext, fov_ext = '.bmp', '.tif', '_mask.tif'
            else:
                im_ext, label_ext, fov_ext = '_image.tif', '_label.gif', '_mask.gif'
            im = skimage.io.imread(base + im_ext)
            label = skimage.io.imread(base + label_ext)
            fov = skimage.io.imread(base + fov_ext)

        im = _fit_rgb_to_image(im)
        H, W = im.shape[:2]
        label = _fit_mask_to_image(label, (H, W), name="label")
        fov   = _fit_mask_to_image(fov,   (H, W), name="fov")


        if use_padding:
            # pad to canonical canvas
            canvas = np.zeros((len_y, len_x, 3), dtype=im.dtype); canvas[:H, :W, :] = im; im = canvas
            canv1 = np.zeros((len_y, len_x, 1), dtype=label.dtype); canv1[:H, :W, :] = label; label = canv1
            canv2 = np.zeros((len_y, len_x, 1), dtype=fov.dtype);   canv2[:H, :W, :] = fov;   fov   = canv2

        processed_im, processed_label, processed_fov, _ = prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training)

        # Final sanity: re-fit masks to image after aug
        Hi, Wi = processed_im.shape[:2]
        processed_label = _fit_mask_to_image(processed_label, (Hi, Wi), name="label_aug")
        processed_fov   = _fit_mask_to_image(processed_fov,   (Hi, Wi), name="fov_aug")

        processed_ims.append(processed_im)
        processed_labels.append(processed_label)
        processed_fovs.append(processed_fov)

        if DEBUG:
            assert processed_label.shape[:2] == processed_im.shape[:2], f"Label HW mismatch {processed_label.shape} vs {processed_im.shape}"
            assert processed_fov.shape[:2]   == processed_im.shape[:2], f"FOV   HW mismatch {processed_fov.shape} vs {processed_im.shape}"

    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels).astype(np.int64)
    fov_blob = im_list_to_blob(processed_fovs).astype(np.int64)
    return im_blob, label_blob, fov_blob


def _get_graph_fov_blob(minibatch_db, is_training, edge_type='srns_geo_dist_binary',
                        win_size=8, edge_geo_dist_thresh=20):
    num_graphs = len(minibatch_db)
    processed_ims, processed_labels, processed_fovs, processed_probmaps = [], [], [], []
    all_graphs, num_of_nodes_list = [], []

    ds = _infer_dataset_from_string(minibatch_db[0])
    pixel_mean, len_y, len_x = _dataset_specs(ds)

    for i in range(num_graphs):
        cur_path = minibatch_db[i]

        if ds == 'DRIVE':
            im_path, label_path, fov_path = _resolve_drive_files(cur_path)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        elif ds == 'CHASE_DB1':
            im_path, label_path, fov_path = _resolve_chase_db1_files(cur_path)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        elif ds == 'HRF':
            im_path, label_path, fov_path = _resolve_hrf_files(cur_path)
            im = skimage.io.imread(im_path)
            label = skimage.io.imread(label_path)
            fov = skimage.io.imread(fov_path)
        else:
            raise NotImplementedError("Graph path resolver for non-DRIVE datasets is not wired.")

        im = _fit_rgb_to_image(im)
        H, W = im.shape[:2]
        label = _fit_mask_to_image(label, (H, W), name="label")
        fov   = _fit_mask_to_image(fov,   (H, W), name="fov")

        # Probmap heuristic (optional)
        base_no_ext = re.sub(r'\.tif$', '', im_path, flags=re.IGNORECASE)
        prob_guess = base_no_ext + '_prob.png'
        if Path(prob_guess).exists():
            pm = skimage.io.imread(prob_guess)
            probmap = _fit_mask_to_image(pm, (H, W), name="probmap")
        else:
            probmap = np.zeros((H, W, 1), dtype=np.uint8)

        # Pad to canonical canvas (to match Shin's graph code assumptions)
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

        # Graph loading (unchanged)
        if 'srns' not in edge_type:
            raise NotImplementedError("Only 'srns_*' edge types are supported")
        
        # ---------------------------
        # RESPONSIVE GRAPH FILE LOADING (DRIU OR DAU2NET)
        # ---------------------------
        win_size_str = '_%.2d_%.2d' % (win_size, edge_geo_dist_thresh)

        # Case 1 — ORIGINAL SHIN DRIU MODE
        # Looks beside the image file: <image>.tif → <image>_04_10.graph_res
        default_graph_path = Path(str(cur_path) + win_size_str + '.graph_res')

        # Case 2 — DAU2Net MODE (flag enabled by train_VGN_DAU.py)
        if getattr(cfg, 'USE_DAU2_GRAPH_PATHS', False):
            base = Path(cur_path).stem   # e.g., '21_training'
            graph_dir = Path(cfg.PATHS.GRAPH_TRAIN_DIR if is_training else cfg.PATHS.GRAPH_TEST_DIR)
            graph_path = graph_dir / (base + win_size_str + '.graph_res')
        else:
            graph_path = default_graph_path

        # Fallback: if primary fails, try the fallback dir
        if not graph_path.exists():
            fallback_root = Path(cfg.PATHS.GRAPH_TRAIN_DIR if is_training else cfg.PATHS.GRAPH_TEST_DIR)
            alt_path = fallback_root / (Path(cur_path).stem + win_size_str + '.graph_res')
            if alt_path.exists():
                graph_path = alt_path
            else:
                raise FileNotFoundError(f"[util] Missing graph file: {graph_path}")

        # Load graph
        with open(str(graph_path), 'rb') as gf:
            graph = pickle.load(gf)

            
        union_graph = nx.convert_node_labels_to_integers(graph)
        n_nodes_in_graph = union_graph.number_of_nodes()
        node_idx_map = np.zeros(processed_im.shape[:2], dtype=np.int32)
        for j in range(n_nodes_in_graph):
            node_idx_map[union_graph.nodes[j]['y'], union_graph.nodes[j]['x']] = j + 1
        union_graph = nx.convert_node_labels_to_integers(union_graph)
        all_graphs.append(union_graph)
        num_of_nodes_list.append(union_graph.number_of_nodes())

    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels).astype(np.int64)
    fov_blob = im_list_to_blob(processed_fovs).astype(np.int64)
    probmap_blob = im_list_to_blob(processed_probmaps)
    all_union_graph = nx.algorithms.operators.all.disjoint_union_all(all_graphs)
    # vec_aug_on / rot_angle are set in the last loop; fine for logging parity
    return im_blob, label_blob, fov_blob, probmap_blob, all_union_graph, num_of_nodes_list, vec_aug_on, rot_angle


# -------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------
def prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training):
    """Preprocess images for use in a blob."""
    im = im.astype(np.float32, copy=False) / 255.
    label = _to_float01_mask(label)
    fov = _to_float01_mask(fov)

    vec_aug_on = np.zeros((7,), dtype=bool)

    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]
            label = label[:, ::-1, :]
            fov = fov[:, ::-1, :]

        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]
            label = label[::-1, :, :]
            fov = fov[::-1, :, :]

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
            y1 = np.random.randint(0, max(1, h - cur_h + 1))
            x1 = np.random.randint(0, max(1, w - cur_w + 1))
            y2 = y1 + cur_h; x2 = x1 + cur_w
            im = im[y1:y2, x1:x2, :]
            label = label[y1:y2, x1:x2, :]
            fov = fov[y1:y2, x1:x2, :]

        if getattr(cfg.TRAIN, 'USE_BRIGHTNESS_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_BRIGHTNESS', False)):
            vec_aug_on[5] = True
            delta = getattr(cfg.TRAIN, 'BRIGHTNESS_ADJUSTMENT_MAX_DELTA', getattr(cfg.TRAIN, 'BRIGHTNESS_DELTA', 0.05))
            im += np.random.uniform(-delta, delta)
            im = np.clip(im, 0, 1)

        if getattr(cfg.TRAIN, 'USE_CONTRAST_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_CONTRAST', False)):
            vec_aug_on[6] = True
            lower = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_LOWER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_LOWER', 0.9))
            upper = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_UPPER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_UPPER', 1.1))
            mm = np.mean(im)
            im = (im - mm) * np.random.uniform(lower, upper) + mm
            im = np.clip(im, 0, 1)

    # VGG-style mean subtraction (re-scaled)
    im -= np.array(pixel_mean) / 255.
    im = im * 255.

    # binarize and cast to int64
    label = (label >= 0.5).astype(np.int64)
    fov = (fov >= 0.5).astype(np.int64)
    return im, label, fov, vec_aug_on


def prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, pixel_mean, is_training, win_size):
    """Preprocess images + graph maps for use in a blob."""
    im = im.astype(np.float32, copy=False) / 255.
    label = _to_float01_mask(label)
    fov = _to_float01_mask(fov)
    probmap = _to_float01_mask(probmap)

    vec_aug_on = np.zeros((7,), dtype=bool)
    cur_y1 = cur_y2 = cur_x1 = cur_x2 = 0
    rot_angle = 0

    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]
            label = label[:, ::-1, :]
            fov = fov[:, ::-1, :]
            probmap = probmap[:, ::-1, :]
            node_idx_map = node_idx_map[:, ::-1]

        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]
            label = label[::-1, :, :]
            fov = fov[::-1, :, :]
            probmap = probmap[::-1, :, :]
            node_idx_map = node_idx_map[::-1, :]

        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            len_ori_y, len_ori_x = im.shape[:2]
            rot_angle = int(np.random.choice([0, 90, 180, 270]))
            im      = _rotate_img(im,      rot_angle, order=1, cval=0., resize=True)
            label   = _rotate_img(label,   rot_angle, order=0, cval=0., resize=True)
            fov     = _rotate_img(fov,     rot_angle, order=0, cval=0., resize=True)
            probmap = _rotate_img(probmap, rot_angle, order=0, cval=0., resize=True)
            node_idx_map = _resize_2d(node_idx_map, (im.shape[0], im.shape[1]), order=0)
            # Clip back to original canvas
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
            im += np.random.uniform(-delta, delta)
            im = np.clip(im, 0, 1)

        if getattr(cfg.TRAIN, 'USE_CONTRAST_ADJUSTMENT', getattr(cfg.TRAIN, 'RANDOM_CONTRAST', False)):
            vec_aug_on[6] = True
            lower = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_LOWER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_LOWER', 0.9))
            upper = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_UPPER_FACTOR', getattr(cfg.TRAIN, 'CONTRAST_UPPER', 1.1))
            mm = np.mean(im)
            im = (im - mm) * np.random.uniform(lower, upper) + mm
            im = np.clip(im, 0, 1)

    im -= np.array(pixel_mean) / 255.
    im = im * 255.

    label = (label >= 0.5).astype(np.int64)
    fov = (fov >= 0.5).astype(np.int64)

    return im, label, fov, probmap, node_idx_map, vec_aug_on, (cur_y1, cur_y2, cur_x1, cur_x2), rot_angle


# -------------------------------------------------------------------------
# Blob builder / misc
# -------------------------------------------------------------------------
def im_list_to_blob(ims):
    """Convert a list of (H,W,C) arrays into a [N,Hmax,Wmax,C] blob."""
    # Coerce shapes before stacking
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
    """A simple timer."""
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
        nx.draw_networkx_edges(graph, pos, edgelist=fp_edges, width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[2])

    if save_graph: plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_graph: plt.show()
    plt.cla(); plt.clf(); plt.close()


def get_auc_ap_score(labels, preds, max_samples=200000):
    """
    Compute AUC/AP with a balanced subsample to avoid OOM and skew from class imbalance.
    - max_samples: cap on total samples used for ROC (balanced pos/neg).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    lbl = np.asarray(labels).ravel()
    prd = np.asarray(preds).ravel()
    # Drop NaNs in preds if any
    keep = ~np.isnan(prd)
    lbl = lbl[keep]
    prd = prd[keep]

    pos_idx = np.where(lbl == 1)[0]
    neg_idx = np.where(lbl == 0)[0]

    # Fallback if only one class
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        auc_score = float('nan')
        ap_score = float('nan')
        return auc_score, ap_score

    # Balanced subsample for ROC to avoid domination by negatives
    half_cap = max_samples // 2 if max_samples else min(len(pos_idx), len(neg_idx))
    k = min(len(pos_idx), len(neg_idx), half_cap)
    rng = np.random.default_rng(12345)
    pos_sel = rng.choice(pos_idx, size=k, replace=len(pos_idx) < k)
    neg_sel = rng.choice(neg_idx, size=k, replace=len(neg_idx) < k)
    sel = np.concatenate([pos_sel, neg_sel])

    auc_score = roc_auc_score(lbl[sel], prd[sel])
    # AP is more stable; compute on full set
    ap_score = average_precision_score(lbl, prd)
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
