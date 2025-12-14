# -*- coding: utf-8 -*-
"""
util_vgn.py — PyTorch rewire of your data / blob utilities
- No TensorFlow, no Keras 3 nonsense
- Augmentations roughly match your original TF pipeline
- Everything runs on CPU or GPU depending on tensor.device
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import numpy.random as npr
import skimage.io
import skimage.transform

import torch
import torch.nn.functional as F

from config_vgn import cfg


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


def _find_file_with_exts(dir_path: Path, stem: str, exts=None):
    if exts:
        for ext in exts:
            cand = dir_path / f"{stem}{ext}"
            if cand.exists():
                return cand
    matches = sorted(dir_path.glob(stem + '.*'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"[util_vgn] file not found: {dir_path}/{stem}.*")


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
        else: raise FileNotFoundError(f"[util_vgn] Unrecognized DRIVE filename: {fname}")
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
                raise FileNotFoundError(f"[util_vgn] Can't infer DRIVE id/subset from: {sample_path}")
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

    if not img_path.exists():   raise FileNotFoundError(f"[util_vgn] Missing image file: {img_path}")
    if not label_path.exists(): raise FileNotFoundError(f"[util_vgn] Missing label file: {label_path}")
    if not fov_path.exists():   raise FileNotFoundError(f"[util_vgn] Missing FOV file: {fov_path}")
    return str(img_path), str(label_path), str(fov_path)


def _resolve_hrf_files(sample_path: str):
    """
    HRF resolver with correct train/test split:
      * 01_* .. 05_*  -> training
      * 06_* .. 15_*  -> testing
    If the path already encodes 'training' / 'testing', we still override it
    using this numeric rule.
    """
    p = Path(sample_path)

    # Infer id from basename (e.g., "06_dr" -> 6)
    base_name = p.stem if p.suffix else p.name
    m_id = re.match(r'^(\d+)', base_name)
    if m_id:
        idx = int(m_id.group(1))
        subset_from_id = 'training' if idx <= 5 else 'testing'
    else:
        subset_from_id = None

    # Look for explicit train/test tokens in the path
    subset = None
    for part in p.parts:
        low = part.lower()
        if low in ('training', 'train'):
            subset = 'training'
            break
        if low in ('testing', 'test'):
            subset = 'testing'
            break

    # Override / fill with numeric rule
    if subset_from_id is not None:
        subset = subset_from_id
    if subset is None:
        subset = 'training'

    # Find HRF root
    root = None
    for ancestor in [p] + list(p.parents):
        if ancestor.name.lower() == 'hrf':
            root = ancestor
            break
    if root is None:
        root = Path('/workspace/DATASETS/HRF')

    # Robustly find subset dir (training/train vs testing/test)
    def _resolve_subset_dir(root_dir: Path, subset_name: str):
        cand_names = []
        if subset_name == 'training':
            cand_names = ['training', 'train']
        elif subset_name == 'testing':
            cand_names = ['testing', 'test']
        else:
            cand_names = [subset_name]

        for cand in cand_names:
            img_dir = root_dir / cand / 'images'
            if img_dir.exists():
                return img_dir, cand

        # if not found, try both training/testing as a last resort
        for cand in ['training', 'train', 'testing', 'test']:
            img_dir = root_dir / cand / 'images'
            if img_dir.exists():
                return img_dir, cand

        raise FileNotFoundError(f"[util_vgn] HRF images dir missing under {root_dir} for subset {subset_name}")

    img_dir, subset_resolved = _resolve_subset_dir(root, subset)
    subset = subset_resolved

    manual_dir = root / subset / 'manual1'
    mask_dir   = root / subset / 'mask'
    if not manual_dir.exists():
        manual_dir = img_dir
    if not mask_dir.exists():
        mask_dir = img_dir

    if not img_dir.exists():
        raise FileNotFoundError(f"[util_vgn] HRF images dir missing: {img_dir}")

    base = base_name
    img_path = _find_file_with_exts(img_dir, base,
                                    ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
    try:
        label_path = _find_file_with_exts(manual_dir, base, ['.tif', '.tiff', '.png'])
    except FileNotFoundError:
        label_path = _find_file_with_exts(
            img_dir, base + '_label',
            ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']
        )
    try:
        fov_path = _find_file_with_exts(
            mask_dir, base + '_mask',
            ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']
        )
    except FileNotFoundError:
        fov_path = _find_file_with_exts(
            img_dir, base + '_mask',
            ['.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg']
        )

    return str(img_path), str(label_path), str(fov_path)


def _resolve_chase_db1_files(sample_path: str):
    p = Path(sample_path)
    subset = None
    for part in p.parts:
        low = part.lower()
        if low in ('training', 'train'): subset = 'training'; break
        if low in ('testing', 'test'):   subset = 'test';      break
    if subset is None: subset = 'training'

    root = None
    for ancestor in [p] + list(p.parents):
        low = ancestor.name.lower()
        if low == 'chase_db1':
            root = ancestor
            break
    if root is None:
        root = Path('/workspace/DATASETS/CHASE_DB1')
    else:
        img_check = root / subset / 'images'
        if not img_check.exists():
            parent = root.parent
            if parent.name.lower() == 'chase_db1' and (parent / subset / 'images').exists():
                root = parent

    img_dir = root / subset / 'images'
    if not img_dir.exists():
        alt_root = Path('/workspace/DATASETS/CHASE_DB1')
        alt_img_dir = alt_root / subset / 'images'
        if alt_img_dir.exists():
            root = alt_root
            img_dir = alt_img_dir
        else:
            raise FileNotFoundError(f"[util_vgn] CHASE_DB1 images dir missing: {img_dir}")

    label_dir = root / subset / '1stHO'
    if not label_dir.exists(): label_dir = img_dir
    mask_dir = root / subset / 'mask'
    if not mask_dir.exists():  mask_dir  = img_dir

    base = p.stem if p.suffix else p.name
    img_path = _find_file_with_exts(img_dir, base, ['.jpg', '.png', '.tif', '.tiff'])

    label_path = None
    label_stems = [
        base + '_1stHO', base + '_1stho',
        base + '_1stHO1', base + '_1stho1',
        base + '_1st_manual', base + '_manual1',
        base,
    ]
    for stem in label_stems:
        try:
            label_path = _find_file_with_exts(label_dir, stem, ['.png', '.tif', '.tiff'])
            break
        except FileNotFoundError:
            continue
    if label_path is None:
        raise FileNotFoundError(f"[util_vgn] Missing CHASE_DB1 label for {base} under {label_dir}")

    mask_path = _find_file_with_exts(mask_dir, base + '_mask', ['.tif', '.tiff', '.png'])

    return str(img_path), str(label_path), str(mask_path)


# -------------------------------------------------------------------------
# skimage transform wrappers -> numpy, then tensors
# -------------------------------------------------------------------------
def _rotate_img_np(x, angle, order=1, cval=0.0, resize=False):
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


def _rescale_img_np(x, scale, order=1):
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


def _resize_2d_np(x, out_hw, order=0):
    H, W = out_hw
    if x.ndim == 2:
        return skimage.transform.resize(x, (H, W), order=order,
                                        preserve_range=True, anti_aliasing=False)
    elif x.ndim == 3 and x.shape[-1] in (1, 3):
        try:
            return skimage.transform.resize(x, (H, W), order=order,
                                            preserve_range=True,
                                            anti_aliasing=False, channel_axis=-1)
        except TypeError:
            return skimage.transform.resize(x, (H, W), order=order,
                                            preserve_range=True,
                                            anti_aliasing=False, multichannel=True)
    else:
        return skimage.transform.resize(np.squeeze(x), (H, W),
                                        order=order, preserve_range=True,
                                        anti_aliasing=False)


# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------
def _fit_rgb_np(im):
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"[util_vgn] unexpected image shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _to_hw1_np(x):
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
    raise ValueError(f"[util_vgn] cannot coerce {a.shape} to (H,W,1)")


def _fit_mask_to_image_np(mask, ref_hw):
    H, W = ref_hw
    m = _to_hw1_np(mask)
    h, w = m.shape[:2]
    if (h, w) == (H, W):
        return m
    if (h, w) == (W, H):
        return np.transpose(m, (1, 0, 2))
    m2 = _resize_2d_np(m, (H, W), order=0)[..., :1]
    return m2


def _to_float01_mask_np(x):
    arr = x.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    maxv = float(np.max(arr))
    minv = float(np.min(arr))
    if maxv > 1.0 or minv < 0.0:
        arr = np.clip(arr, 0.0, 255.0) / 255.0
    return arr


# -------------------------------------------------------------------------
# Core minibatch builder (NP -> Torch)
# -------------------------------------------------------------------------
def _load_single_np(path: str, ds: str, use_padding=False):
    pixel_mean, len_y, len_x = _dataset_specs(ds)

    if ds == 'DRIVE':
        im_path, label_path, fov_path = _resolve_drive_files(path)
        im  = skimage.io.imread(im_path)
        lab = skimage.io.imread(label_path)
        fov = skimage.io.imread(fov_path)
    elif ds == 'CHASE_DB1':
        im_path, label_path, fov_path = _resolve_chase_db1_files(path)
        im  = skimage.io.imread(im_path)
        lab = skimage.io.imread(label_path)
        fov = skimage.io.imread(fov_path)
    elif ds == 'HRF':
        im_path, label_path, fov_path = _resolve_hrf_files(path)
        im  = skimage.io.imread(im_path)
        lab = skimage.io.imread(label_path)
        fov = skimage.io.imread(fov_path)
    else:
        raise NotImplementedError(f"[util_vgn] dataset resolver not wired for {ds}")

    im = _fit_rgb_np(im)
    H, W = im.shape[:2]
    lab = _fit_mask_to_image_np(lab, (H, W))
    fov = _fit_mask_to_image_np(fov, (H, W))

    if use_padding:
        canvas = np.zeros((len_y, len_x, 3), dtype=im.dtype)
        canvas[:H, :W, :] = im
        im = canvas

        canv1 = np.zeros((len_y, len_x, 1), dtype=lab.dtype)
        canv1[:H, :W, :] = lab
        lab = canv1

        canv2 = np.zeros((len_y, len_x, 1), dtype=fov.dtype)
        canv2[:H, :W, :] = fov
        fov = canv2

    return im, lab, fov, pixel_mean


def _augment_np(im, label, fov, pixel_mean, is_training: bool):
    """
    Rough PyTorch-ish clone of your TF `prep_im_fov_for_blob`
    but done in NumPy, returning np arrays.
    """

    im = im.astype(np.float32) / 255.0
    label = _to_float01_mask_np(label)
    fov   = _to_float01_mask_np(fov)

    if is_training:
        # LR flip
        if getattr(cfg.TRAIN, 'USE_LR_FLIPPED', False) and npr.random_sample() >= 0.5:
            im    = im[:, ::-1, :]
            label = label[:, ::-1, :]
            fov   = fov[:, ::-1, :]

        # UD flip
        if getattr(cfg.TRAIN, 'USE_UD_FLIPPED', False) and npr.random_sample() >= 0.5:
            im    = im[::-1, :, :]
            label = label[::-1, :, :]
            fov   = fov[::-1, :, :]

        # Rotation
        if getattr(cfg.TRAIN, 'USE_ROTATION', False):
            angle = np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE,
                                       cfg.TRAIN.ROTATION_MAX_ANGLE)
            im    = _rotate_img_np(im,    angle, order=1, cval=0.0)
            label = _rotate_img_np(label, angle, order=0, cval=0.0)
            fov   = _rotate_img_np(fov,   angle, order=0, cval=0.0)

        # Scaling
        if getattr(cfg.TRAIN, 'USE_SCALING', False):
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0],
                                      cfg.TRAIN.SCALING_RANGE[1])
            im    = _rescale_img_np(im,    scale, order=1)
            label = _rescale_img_np(label, scale, order=0)
            fov   = _rescale_img_np(fov,   scale, order=0)

        # Random patch cropping
        if getattr(cfg.TRAIN, 'USE_CROPPING', False):
            h, w = im.shape[:2]
            h1 = int(round(h * 0.5)); h2 = int(round(h * 0.8))
            w1 = int(round(w * 0.5)); w2 = int(round(w * 0.8))
            cur_h = np.random.randint(h1, max(h1+1, h2+1)) if h2 >= h1 else h
            cur_w = np.random.randint(w1, max(w1+1, w2+1)) if w2 >= w1 else w
            cur_h = min(cur_h, h); cur_w = min(cur_w, w)
            y1 = np.random.randint(0, max(1, h - cur_h + 1))
            x1 = np.random.randint(0, max(1, w - cur_w + 1))
            y2 = y1 + cur_h; x2 = x1 + cur_w
            im    = im[y1:y2, x1:x2, :]
            label = label[y1:y2, x1:x2, :]
            fov   = fov[y1:y2, x1:x2, :]

        # Brightness
        if getattr(cfg.TRAIN, 'USE_BRIGHTNESS_ADJUSTMENT',
                   getattr(cfg.TRAIN, 'RANDOM_BRIGHTNESS', False)):
            delta = getattr(cfg.TRAIN, 'BRIGHTNESS_ADJUSTMENT_MAX_DELTA',
                            getattr(cfg.TRAIN, 'BRIGHTNESS_DELTA', 0.05))
            im += np.random.uniform(-delta, delta)
            im = np.clip(im, 0.0, 1.0)

        # Contrast
        if getattr(cfg.TRAIN, 'USE_CONTRAST_ADJUSTMENT',
                   getattr(cfg.TRAIN, 'RANDOM_CONTRAST', False)):
            lower = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_LOWER_FACTOR',
                            getattr(cfg.TRAIN, 'CONTRAST_LOWER', 0.9))
            upper = getattr(cfg.TRAIN, 'CONTRAST_ADJUSTMENT_UPPER_FACTOR',
                            getattr(cfg.TRAIN, 'CONTRAST_UPPER', 1.1))
            mm = np.mean(im)
            im = (im - mm) * np.random.uniform(lower, upper) + mm
            im = np.clip(im, 0.0, 1.0)

    # VGG-style mean subtraction rescaled
    im -= np.array(pixel_mean, dtype=np.float32) / 255.0
    im = im * 255.0

    # binarize
    label = (label >= 0.5).astype(np.int64)
    fov   = (fov   >= 0.5).astype(np.int64)
    return im, label, fov


def _np_to_torch(im_np, lab_np, fov_np, device):
    """
    Turn single-sample numpy arrays into torch tensors:
      im:  (H,W,3)   -> (1,3,H,W)
      lab:(H,W,1)    -> (1,1,H,W)
      fov:(H,W,1)    -> (1,1,H,W)
    """
    im_t  = torch.from_numpy(im_np.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    lab_t = torch.from_numpy(lab_np.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.long)
    fov_t = torch.from_numpy(fov_np.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.long)
    return im_t, lab_t, fov_t


def get_minibatch_torch(paths, is_training: bool, use_padding: bool = False, device=None):
    """
    PyTorch version of `_get_image_fov_blob` + `im_list_to_blob`.
    paths: list of strings (image IDs / basenames / paths)
    Returns:
        img  : FloatTensor [N,3,Hmax,Wmax]
        label: LongTensor  [N,1,Hmax,Wmax]
        fov  : LongTensor  [N,1,Hmax,Wmax]
    """
    if device is None:
        device = torch.device('cpu')

    assert len(paths) > 0
    ds = _infer_dataset_from_string(paths[0])

    ims_t, labs_t, fovs_t = [], [], []

    for p in paths:
        im_np, lab_np, fov_np, pixel_mean = _load_single_np(p, ds, use_padding=use_padding)
        im_np, lab_np, fov_np            = _augment_np(im_np, lab_np, fov_np, pixel_mean, is_training)
        im_t, lab_t, fov_t              = _np_to_torch(im_np, lab_np, fov_np, device)
        ims_t.append(im_t)
        labs_t.append(lab_t)
        fovs_t.append(fov_t)

    # pad to max H,W like your original im_list_to_blob
    max_h = max(x.shape[2] for x in ims_t)
    max_w = max(x.shape[3] for x in ims_t)

    N = len(ims_t)
    img_blob   = torch.zeros(N, 3, max_h, max_w, device=device, dtype=torch.float32)
    label_blob = torch.zeros(N, 1, max_h, max_w, device=device, dtype=torch.long)
    fov_blob   = torch.zeros(N, 1, max_h, max_w, device=device, dtype=torch.long)

    for i in range(N):
        _, _, h, w = ims_t[i].shape
        img_blob[i, :, :h, :w]   = ims_t[i][0]
        label_blob[i, :, :h, :w] = labs_t[i][0]
        fov_blob[i, :, :h, :w]   = fovs_t[i][0]

    return img_blob, label_blob, fov_blob


# -------------------------------------------------------------------------
# Simple DataLayer wrapper (PyTorch version)
# -------------------------------------------------------------------------
class TorchDataLayer(object):
    def __init__(self, db, is_training, use_padding=False, device=None):
        """
        db: list of image base paths (same as your old TF db)
        """
        self._db = db
        self._is_training = is_training
        self._use_padding = use_padding
        self.img_names = list(db)
        self.device = device or torch.device('cpu')

        if self._is_training:
            self._shuffle_db_inds()
        else:
            self._db_inds()

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _db_inds(self):
        self._perm = np.arange(len(self._db))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        cur_batch_size = cfg.TRAIN.BATCH_SIZE
        if self._is_training:
            if self._cur + cur_batch_size > len(self._db):
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
        img, label, fov = get_minibatch_torch(minibatch_db,
                                              is_training=self._is_training,
                                              use_padding=self._use_padding,
                                              device=self.device)
        blobs = {'img': img, 'label': label, 'fov': fov}
        return minibatch_db, blobs

    def forward(self):
        img_list, blobs = self._get_next_minibatch()
        return img_list, blobs


# -------------------------------------------------------------------------
# Timer (unchanged semantics)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Metrics helper for AUC / AP (used by train_VGN_DAU.py)
# -------------------------------------------------------------------------
def get_auc_ap_score(labels, preds):
    from sklearn.metrics import roc_auc_score, average_precision_score
    labels = np.asarray(labels).astype(np.float32).ravel()
    preds = np.asarray(preds).astype(np.float32).ravel()
    auc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)
    return auc_score, ap_score
