# -*- coding: utf-8 -*-
# Train AFF fusion on folders of {probmap PNG, graph_res} pairs.
# Strict pairing from *real* graph files (.graph_res/.npz/.npy/.pkl/.gpickle).
# Probmaps must be PNG/JPG that contain 'prob' or 'cnn' and NOT 'graph_res'.
# HRF/DRIVE/CHASE_DB1 GT+mask auto-discovery. Saves fused probmaps and μ maps.

import os, re, glob, math, argparse, pickle, hashlib, warnings
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as cp

from AFF_Module import AFFModule, rasterize_gnn_features  # unchanged

try:
    import networkx as nx
except Exception:
    nx = None

# -------------------- constants --------------------
GRAPH_EXTS = ('.graph_res', '.npz', '.npy', '.gpickle', '.pkl')
IMG_EXTS   = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# -------------------- tiny utils --------------------
def _imread_gray_float(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32)
    if arr.max() > 1.0: arr = arr / 255.0
    return arr

def _imread_mask01(path: str) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
    if arr.max() > 1.0: arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)

def _save_png01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(np.asarray(arr01, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def _is_graph_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in GRAPH_EXTS

def _file_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _prefix_from_stem(stem: str) -> str:
    """Return dataset prefix from a stem. Order matters."""
    s = stem.lower()
    # HRF: 01_dr_10_80 -> 01_dr  /  07_g_... -> 07_g  /  12_h_... -> 12_h
    m = re.match(r'^(\d+_(dr|g|h))', s)
    if m: return m.group(1)
    # DRIVE: 21_training_04_10 -> 21_training
    m = re.match(r'^(\d+_(training|test))', s)
    if m: return m.group(1)
    # CHASE_DB1: Image_01L_08_40 -> Image_01L
    m = re.match(r'^(image_\d+[lr])', s)
    if m: return m.group(1)
    # fallback
    return stem

def _collect_graphs(root: str) -> List[str]:
    files = [p for p in glob.glob(os.path.join(root, '*')) if _is_graph_file(p)]
    return sorted(files)

def _collect_probmaps_for_prefix(root: str, prefix: str) -> List[str]:
    """Find CNN probmaps that start with the prefix and contain 'prob' or 'cnn' (NOT 'graph_res')."""
    out = []
    for p in glob.glob(os.path.join(root, f'{prefix}*')):
        name = os.path.basename(p).lower()
        if any(name.endswith(ext) for ext in IMG_EXTS) and ('graph_res' not in name) and ('prob' in name or 'cnn' in name):
            out.append(p)
    return sorted(out)

def _cache_key(path: str, out_hw: Tuple[int,int]) -> str:
    h = hashlib.sha1()
    h.update(path.encode('utf-8'))
    h.update(f'|{int(out_hw[0])}x{int(out_hw[1])}'.encode('utf-8'))
    return h.hexdigest()[:20]

def _from_np_array_to_dense(arr: np.ndarray, out_hw: Tuple[int,int]) -> torch.Tensor:
    H, W = out_hw
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2: arr = arr[None, ...]
    ten = torch.from_numpy(arr).float().unsqueeze(0)  # [1,C,H,W] or [1,1,H,W]
    ten = F.interpolate(ten, size=(H,W), mode='bilinear', align_corners=False)
    return ten.cpu()

def _read_graph_dense_cpu(path: str, out_hw: Tuple[int,int],
                          cache_dir: Optional[str]=None,
                          default_cell: int = 8) -> torch.Tensor:
    """
    Return CPU tensor [1,C,H,W] float32. Supports .graph_res/.pkl (dict or networkx), .npz, .npy.
    """
    H, W = out_hw
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(os.path.abspath(path), out_hw)
        cfile = os.path.join(cache_dir, f'{key}.pt')
        if os.path.exists(cfile):
            try:
                ten = torch.load(cfile, map_location='cpu')
                if isinstance(ten, torch.Tensor) and ten.ndim == 4 and ten.shape[-2:] == (H, W):
                    return ten.float().cpu()
            except Exception:
                pass

    suf = os.path.splitext(path)[1].lower()
    ten = None

    # pickle formats: dict with 'dense' or verts/node_feat … OR networkx graph
    if suf in ('.graph_res', '.pkl', '.gpickle'):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                dense = obj.get('dense') or obj.get('gnn_dense') or obj.get('feat_map')
                if dense is not None:
                    ten = _from_np_array_to_dense(dense, out_hw)
                else:
                    verts = obj.get('verts') or obj.get('positions') or obj.get('yx') or obj.get('xy')
                    node_feat = obj.get('node_feat') or obj.get('features') or obj.get('h') or obj.get('feat')
                    grid_hw = obj.get('grid_hw') or obj.get('grid_size') or (math.ceil(H/default_cell), math.ceil(W/default_cell))
                    cell = int(obj.get('cell') or obj.get('stride') or obj.get('patch') or default_cell)
                    if verts is None or node_feat is None:
                        raise ValueError("pickle dict lacks verts/node_feat")
                    verts = torch.as_tensor(np.asarray(verts), dtype=torch.long, device='cpu')
                    node_feat = torch.as_tensor(np.asarray(node_feat), dtype=torch.float32, device='cpu')
                    ten = rasterize_gnn_features(verts, (int(grid_hw[0]), int(grid_hw[1])), cell, node_feat, (H,W)).cpu()
            elif nx is not None and hasattr(obj, 'nodes'):
                verts, feats = [], []
                for _, a in obj.nodes(data=True):
                    pos = a.get('pos') or (a.get('y'), a.get('x'))
                    if pos is None: continue
                    feat = a.get('feat') or a.get('features') or a.get('h') or a.get('prob') or [1.0]
                    verts.append(pos)
                    feats.append(np.atleast_1d(np.asarray(feat, dtype=np.float32)))
                verts = torch.as_tensor(np.asarray(verts), dtype=torch.long, device='cpu')
                node_feat = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device='cpu')
                ten = rasterize_gnn_features(verts, (math.ceil(H/default_cell), math.ceil(W/default_cell)),
                                             default_cell, node_feat, (H,W)).cpu()
        except Exception as e:
            ten = None

    if suf == '.npz' and ten is None:
        data = np.load(path, allow_pickle=True)
        dense = data.get('dense') or data.get('gnn_dense') or data.get('feat_map')
        if dense is not None:
            ten = _from_np_array_to_dense(dense, out_hw)
        else:
            verts = data.get('verts') or data.get('positions') or data.get('yx') or data.get('xy')
            node_feat = data.get('node_feat') or data.get('features') or data.get('feat') or data.get('h')
            grid_hw = data.get('grid_hw') or data.get('grid_size') or (math.ceil(H/default_cell), math.ceil(W/default_cell))
            cell = int(data.get('cell') or data.get('stride') or data.get('patch') or default_cell)
            if verts is None or node_feat is None:
                raise ValueError(f"{path}: no verts/node_feat or dense map")
            verts = torch.as_tensor(np.asarray(verts), dtype=torch.long, device='cpu')
            node_feat = torch.as_tensor(np.asarray(node_feat), dtype=torch.float32, device='cpu')
            ten = rasterize_gnn_features(verts, (int(grid_hw[0]), int(grid_hw[1])), cell, node_feat, (H,W)).cpu()

    if suf == '.npy' and ten is None:
        arr = np.load(path, allow_pickle=True)
        ten = _from_np_array_to_dense(arr, out_hw)

    if ten is None:
        raise ValueError(f"Unsupported graph_res format or unreadable file: {path}")

    if cache_dir:
        try: torch.save(ten, cfile)
        except Exception: pass
    return ten

# -------------------- GT/mask discovery --------------------
def _drive_guess_gt(root: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    s = prefix.lower()
    set_name = 'test' if ('_test' in s or s.endswith('_test')) else 'training'
    m = re.search(r'(\d+)', prefix)
    if not m: return None, None
    num = m.group(1)
    lab_cands = [
        f"{root}/{set_name}/1st_manual/{num}_manual1.gif",
        f"{root}/{set_name}/1st_manual/{num}_manual1.png",
        f"{root}/{set_name}/1st_manual/{num}_manual1.tif",
    ]
    mask_cands = [
        f"{root}/{set_name}/mask/{num}_{set_name}_mask.gif",
        f"{root}/{set_name}/mask/{num}_{set_name}_mask.png",
        f"{root}/{set_name}/mask/{num}_{set_name}_mask.tif",
    ]
    lab = next((p for p in lab_cands if os.path.exists(p)), None)
    fov = next((p for p in mask_cands if os.path.exists(p)), None)
    return lab, fov

def _chase_guess_gt(root: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    # Examples: Image_01L_1stHO.png, Image_01L_mask.tif (user examples)
    base = prefix  # already like 'Image_01L'
    # Common layouts people use:
    cands_lab = [
        f"{root}/training/1st_manual/{base}_1stHO.png",
        f"{root}/test/1st_manual/{base}_1stHO.png",
        f"{root}/{base}_1stHO.png",
        f"{root}/images/{base}_1stHO.png",
        f"{root}/gt/{base}_1stHO.png",
    ]
    cands_mask = [
        f"{root}/training/mask/{base}_mask.tif",
        f"{root}/test/mask/{base}_mask.tif",
        f"{root}/{base}_mask.tif",
        f"{root}/images/{base}_mask.tif",
        f"{root}/mask/{base}_mask.tif",
    ]
    lab = next((p for p in cands_lab if os.path.exists(p)), None)
    fov = next((p for p in cands_mask if os.path.exists(p)), None)
    return lab, fov

def _hrf_guess_gt(root: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    # Your spec: GT = 01_dr_label.tif ; Mask = 01_dr_mask.tif
    cands_lab = [
        f"{root}/{prefix}_label.tif", f"{root}/labels/{prefix}_label.tif",
        f"{root}/gt/{prefix}_label.tif", f"{root}/manual1/{prefix}.tif",
        f"{root}/manual1/{prefix}.png",
    ]
    cands_mask = [
        f"{root}/{prefix}_mask.tif", f"{root}/mask/{prefix}_mask.tif",
        f"{root}/masks/{prefix}_mask.tif", f"{root}/mask/{prefix}.png",
        f"{root}/mask/{prefix}.tif",
    ]
    lab = next((p for p in cands_lab if os.path.exists(p)), None)
    fov = next((p for p in cands_mask if os.path.exists(p)), None)
    if lab is None or fov is None:
        # recursive fallback
        all_lab = glob.glob(os.path.join(root, '**', f'{prefix}_label.*'), recursive=True)
        lab = lab or (all_lab[0] if all_lab else None)
        all_mask = glob.glob(os.path.join(root, '**', f'{prefix}_mask.*'), recursive=True)
        fov = fov or (all_mask[0] if all_mask else None)
    return lab, fov

def _guess_gt(root: Optional[str], prefix: str) -> Tuple[Optional[str], Optional[str]]:
    if not root or not os.path.isdir(root):
        return None, None
    sroot = os.path.basename(os.path.normpath(root)).lower()
    # pick function by hint OR by prefix pattern
    if 'hrf' in sroot or re.match(r'^\d+_(dr|g|h)$', prefix.lower()):
        return _hrf_guess_gt(root, prefix)
    if 'drive' in sroot or re.match(r'^\d+_(training|test)$', prefix.lower()):
        return _drive_guess_gt(root, prefix)
    if 'chase' in sroot or prefix.lower().startswith('image_'):
        return _chase_guess_gt(root, prefix)
    # try all
    for fn in (_hrf_guess_gt, _drive_guess_gt, _chase_guess_gt):
        lab, fov = fn(root, prefix)
        if lab or fov: return lab, fov
    return None, None

# -------------------- Dataset --------------------
class FolderPairs(Dataset):
    """
    Build pairs (prob, graph) strictly from graph files in root.
    If a list is provided, keep only graph stems whose dataset prefix is in that list.
    """
    def __init__(self, root: str, id_list: Optional[str], dataset_root: Optional[str],
                 mode: str, augment: bool, fixed_hw: Optional[Tuple[int,int]]=None):
        super().__init__()
        self.root = root
        self.mode = mode
        self.augment = augment
        self.dataset_root = dataset_root
        self.fixed_hw = fixed_hw

        allowed_prefixes: Optional[set] = None
        if id_list and os.path.exists(id_list):
            allowed_prefixes = set()
            with open(id_list, 'r') as f:
                for ln in f:
                    bx = os.path.basename(ln.strip())
                    allowed_prefixes.add(os.path.splitext(bx)[0].lower())

        graph_files = _collect_graphs(root)
        if not graph_files:
            raise RuntimeError(f"No graph files found in {root}. Expect one of {GRAPH_EXTS}")

        self.samples = []
        for gpath in graph_files:
            stem = _file_stem(gpath)
            prefix = _prefix_from_stem(stem)
            if allowed_prefixes and prefix.lower() not in allowed_prefixes:
                continue
            prob_candidates = _collect_probmaps_for_prefix(root, prefix)
            if not prob_candidates:
                continue
            prob = prob_candidates[0]
            lab, fov = _guess_gt(dataset_root, prefix)
            self.samples.append(dict(id=stem, prefix=prefix, prob=prob, graph=gpath, label=lab, fov=fov))

        if not self.samples:
            raise RuntimeError(f"No (prob,graph) pairs found in {root}. Check filenames.")

        print(f"[{mode}] Found {len(self.samples)} pairs in {root}")
        for k in range(min(3, len(self.samples))):
            s = self.samples[k]
            print(f"[{mode}] sample#{k}: id={s['id']}  prefix={s['prefix']}")
            print(f"      prob={os.path.basename(s['prob'])}")
            print(f"      graph={os.path.basename(s['graph'])}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        it = self.samples[idx]
        prob_p, graph_p = it['prob'], it['graph']
        label_p, fov_p  = it['label'], it['fov']

        prob = _imread_gray_float(prob_p)
        H, W = prob.shape

        label = None
        if label_p:
            label = _imread_mask01(label_p)
            if label.shape != prob.shape:
                label = np.array(Image.fromarray((label*255).astype(np.uint8)).resize((W, H), Image.NEAREST), dtype=np.float32)/255.0
                label = (label > 0.5).astype(np.float32)

        if fov_p and os.path.exists(fov_p):
            fov = _imread_mask01(fov_p)
            if fov.shape != prob.shape:
                fov = np.array(Image.fromarray((fov*255).astype(np.uint8)).resize((W, H), Image.NEAREST), dtype=np.float32)/255.0
                fov = (fov > 0.5).astype(np.float32)
        else:
            fov = np.ones_like(prob, dtype=np.float32)

        # optional fixed resize (VRAM control)
        if self.fixed_hw is not None:
            th, tw = int(self.fixed_hw[0]), int(self.fixed_hw[1])
            if (H, W) != (th, tw):
                prob = np.array(Image.fromarray((prob*255).astype(np.uint8)).resize((tw, th), Image.BILINEAR), dtype=np.float32)/255.0
                if label is not None:
                    label = np.array(Image.fromarray((label*255).astype(np.uint8)).resize((tw, th), Image.NEAREST), dtype=np.float32)/255.0
                    label = (label > 0.5).astype(np.float32)
                fov = np.array(Image.fromarray((fov*255).astype(np.uint8)).resize((tw, th), Image.NEAREST), dtype=np.float32)/255.0
                fov = (fov > 0.5).astype(np.float32)
                H, W = th, tw

        return {
            'id': it['id'], 'prefix': it['prefix'],
            'prob': torch.from_numpy(prob)[None, ...],  # [1,H,W]
            'graph_path': graph_p,
            'H': H, 'W': W,
            'label': torch.from_numpy(label)[None, ...] if label is not None else None,
            'fov': torch.from_numpy(fov)[None, ...],
        }

# -------------------- collate + pad --------------------
def _pad_to_hw(t: torch.Tensor, target_hw):
    H, W = t.shape[-2:]
    th, tw = int(target_hw[0]), int(target_hw[1])
    pad_h, pad_w = th - H, tw - W
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    return t

def collate_pad(batch):
    maxH = max([b['prob'].shape[-2] for b in batch])
    maxW = max([b['prob'].shape[-1] for b in batch])
    tgt_hw = (maxH, maxW)

    out = {'id':[b['id'] for b in batch],
           'graph_path':[b['graph_path'] for b in batch],
           'H':[b['H'] for b in batch], 'W':[b['W'] for b in batch]}
    out['prob'] = torch.stack([_pad_to_hw(b['prob'], tgt_hw) for b in batch], dim=0)
    out['label'] = [(_pad_to_hw(b['label'], tgt_hw) if b['label'] is not None else None) for b in batch]
    out['fov']   = [_pad_to_hw(b['fov'], tgt_hw) for b in batch]
    return out

# -------------------- loss / metrics --------------------
def dice_loss_with_logits(logits, target, weights=None, eps=1e-6):
    probs = torch.sigmoid(logits)
    if weights is None: weights = torch.ones_like(target)
    dims = (1,2,3)
    num = (2.0 * probs*target*weights).sum(dim=dims)
    den = ((probs*weights).sum(dim=dims) + (target*weights).sum(dim=dims)).clamp_min(eps)
    return 1.0 - (num/den).mean()

def bce_with_logits_masked_posw(logits, target, weights):
    with torch.no_grad():
        pos = (target * weights).sum()
        neg = (weights.sum() - pos)
        pos_weight = (neg / (pos + 1e-6)).clamp(1.0, 50.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=pos_weight)
    loss = (loss * weights).sum() / (weights.sum() + 1e-6)
    return loss

# -------------------- AFF wrapper (μ learns) --------------------
class AFFWrapper(nn.Module):
    def __init__(self, in_ch_cnn=1, in_ch_gnn=64, mid_ch=64, out_ch=1,
                 mu_mode='dynamic', static_mu=0.5,
                 use_prior_skip=True, prior_gain=0.0,
                 use_grad_ckpt=False):
        super().__init__()
        self.aff = AFFModule(in_ch_cnn, in_ch_gnn, mid_ch, out_ch)
        self.mu_mode = mu_mode
        self.register_buffer('mu_const', torch.tensor(float(static_mu)).view(1,1,1,1))
        self.use_prior_skip = bool(use_prior_skip)
        self.register_buffer('prior_gain', torch.tensor(float(prior_gain)))
        self.use_grad_ckpt = bool(use_grad_ckpt)

        # zero head so prior skip defines start if enabled
        if hasattr(self.aff, 'head'):
            if hasattr(self.aff.head, 'weight'): nn.init.zeros_(self.aff.head.weight)
            if hasattr(self.aff.head, 'bias') and self.aff.head.bias is not None: nn.init.zeros_(self.aff.head.bias)
        # neutral μ init
        last_conv = None
        for m in self.aff.gate.modules():
            if isinstance(m, nn.Conv2d): last_conv = m
        if last_conv is not None and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.0)

    def _refine_head(self, fused):
        f = self.aff.refine(fused); return self.aff.head(f)

    def forward(self, cnn_feat, gnn_feat, cnn_logits: Optional[torch.Tensor]=None):
        if cnn_feat.dim() == 3: cnn_feat = cnn_feat.unsqueeze(1)
        if gnn_feat.dim() == 3: gnn_feat = gnn_feat.unsqueeze(1)

        H, W = cnn_feat.shape[-2:]
        gnn_feat = F.interpolate(gnn_feat, size=(H,W), mode='bilinear', align_corners=False)

        c = self.aff.cnn_align(cnn_feat)
        g = self.aff.gnn_align(gnn_feat)

        mu = self.aff.gate(torch.cat([c, g], dim=1)) if self.mu_mode == 'dynamic' else self.mu_const.expand_as(c)

        fused = mu * c + (1.0 - mu) * g
        logits = cp.checkpoint(self._refine_head, fused, use_reentrant=False) if (self.use_grad_ckpt and self.training) else self._refine_head(fused)

        logits_total = logits + (self.prior_gain * cnn_logits) if (self.use_prior_skip and (cnn_logits is not None)) else logits
        prob_total = torch.sigmoid(logits_total)
        aux = {"mu": mu, "fused": fused, "logits": logits, "logits_total": logits_total}
        return prob_total, aux

# -------------------- helpers --------------------
def _parse_hw(hwstr: str) -> Optional[Tuple[int,int]]:
    if not hwstr: return None
    m = re.match(r'(\d+)[xX](\d+)', hwstr)
    return (int(m.group(1)), int(m.group(2))) if m else None

def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return True
    return str(v).lower() in ('1','true','t','yes','y')

def _logit_from_prob(p: torch.Tensor, eps=1e-5):
    return torch.logit(p.float().clamp(eps, 1.0 - eps))

def _get_lr(optim):
    return optim.param_groups[0]['lr']

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser("Train AFF (strict pairing, cached GNN).")
    # *** HRF defaults (yours) ***
    ap.add_argument('--train_root', type=str, default='/workspace/DRIU_HRF/train')
    ap.add_argument('--test_root',  type=str, default='/workspace/DRIU_HRF/test')
    ap.add_argument('--train_list', type=str, default='/workspace/DATASETS/HRF/train.txt')
    ap.add_argument('--val_list',   type=str, default='/workspace/DATASETS/HRF/test.txt')
    ap.add_argument('--dataset_root', type=str, default='/workspace/DATASETS/HRF')

    ap.add_argument('--save_dir', type=str, default='/workspace/DRIU_HRF/aff')
    ap.add_argument('--epochs', type=int, default=450)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--mid_ch', type=int, default=64)
    ap.add_argument('--mu_mode', choices=['dynamic','static'], default='dynamic')
    ap.add_argument('--static_mu', type=float, default=0.5)
    ap.add_argument('--mu_reg', type=float, default=0.0)
    ap.add_argument('--use_prior_skip', type=str2bool, nargs='?', const=True, default=True)
    ap.add_argument('--prior_gain', type=float, default=0.0)

    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--prefetch_factor', type=int, default=4)
    # HRF is big; use safe default. Set '' to keep native size.
    ap.add_argument('--fixed_hw', type=str, default='1024x1536')

    # AMP / ckpt
    ap.add_argument('--amp', type=str2bool, nargs='?', const=True, default=True)
    ap.add_argument('--amp_dtype', choices=['fp16','bf16'], default='bf16')
    ap.add_argument('--grad_ckpt', type=str2bool, nargs='?', const=True, default=True)

    # caching + saving
    ap.add_argument('--cache_gnn_dir', type=str, default='')
    ap.add_argument('--save_probmaps', type=str2bool, nargs='?', const=True, default=True)
    ap.add_argument('--save_every', type=int, default=100)
    ap.add_argument('--val_every', type=int, default=5)

    # training niceties
    ap.add_argument('--grad_clip', type=float, default=5.0)
    ap.add_argument('--accum_steps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1337)
    return ap.parse_args()

# -------------------- main --------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    out_prob_dir = os.path.join(args.save_dir, 'probmaps'); os.makedirs(out_prob_dir, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    fixed_hw = _parse_hw(args.fixed_hw)
    ds_tr = FolderPairs(args.train_root, args.train_list, args.dataset_root,
                        mode='train', augment=False, fixed_hw=fixed_hw)
    ds_va = FolderPairs(args.test_root, args.val_list, args.dataset_root,
                        mode='val', augment=False, fixed_hw=fixed_hw) if args.test_root else None

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=bool(args.num_workers > 0),
                       prefetch_factor=(args.prefetch_factor if args.num_workers>0 else None),
                       collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0,
                       collate_fn=collate_pad) if ds_va else None

    # infer GNN C from first sample (CPU)
    first = ds_tr[0]
    with torch.no_grad():
        g_dense_cpu = _read_graph_dense_cpu(first['graph_path'], (first['H'], first['W']),
                                            cache_dir=args.cache_gnn_dir)
        in_ch_gnn = g_dense_cpu.shape[1]

    model = AFFWrapper(in_ch_cnn=1, in_ch_gnn=in_ch_gnn, mid_ch=args.mid_ch,
                       out_ch=1, mu_mode=args.mu_mode, static_mu=args.static_mu,
                       use_prior_skip=args.use_prior_skip, prior_gain=args.prior_gain,
                       use_grad_ckpt=args.grad_ckpt).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=25, min_lr=1e-5)
    amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16

    best_dice = -1.0
    did_debug_dump = False
    step_in_accum = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0

        for batch in dl_tr:
            cnn = batch['prob'].to(device, non_blocking=True).to(amp_dtype).contiguous(memory_format=torch.channels_last)
            H, W = cnn.shape[-2], cnn.shape[-1]

            # Build dense GNN batch on CPU → one .to(device)
            g_list_cpu, labels, fovs = [], [], []
            for b in range(cnn.size(0)):
                g_cpu = _read_graph_dense_cpu(batch['graph_path'][b], (H,W), cache_dir=args.cache_gnn_dir)
                g_list_cpu.append(g_cpu)
                labels.append(batch['label'][b] if batch['label'][b] is not None else None)
                fovs.append(batch['fov'][b])
            gnn = torch.cat(g_list_cpu, dim=0).to(device, non_blocking=True).to(amp_dtype).contiguous(memory_format=torch.channels_last)

            if any(l is None for l in labels):
                raise RuntimeError("No labels discovered. Provide --dataset_root for GT/masks (DRIVE, CHASE_DB1, HRF).")

            tgt = torch.stack([x.to(device, non_blocking=True) for x in labels], dim=0).to(amp_dtype).contiguous(memory_format=torch.channels_last)
            wts = torch.stack([x.to(device, non_blocking=True) for x in fovs],   dim=0).to(amp_dtype).contiguous(memory_format=torch.channels_last)

            if not did_debug_dump:
                os.makedirs(os.path.join(args.save_dir, 'debug'), exist_ok=True)
                for j in range(min(cnn.size(0), 3)):
                    pid = batch['id'][j]
                    cn = cnn[j,0].detach().float().cpu().numpy()
                    lb = tgt[j,0].detach().float().cpu().numpy()
                    fv = wts[j,0].detach().float().cpu().numpy()
                    print(f"[DEBUG] {pid}: prob[min/max/mean]={cn.min():.3f}/{cn.max():.3f}/{cn.mean():.3f} "
                          f"label(pos%)={(lb>0.5).mean()*100:.2f}  fov(%)={(fv>0.5).mean()*100:.2f}")
                    _save_png01(os.path.join(args.save_dir, 'debug', f'{pid}_prob.png'), cn)
                    _save_png01(os.path.join(args.save_dir, 'debug', f'{pid}_gt.png'), (lb>0.5).astype(np.float32))
                    _save_png01(os.path.join(args.save_dir, 'debug', f'{pid}_fov.png'), (fv>0.5).astype(np.float32))
                did_debug_dump = True

            with torch.amp.autocast('cuda', enabled=args.amp, dtype=amp_dtype):
                cnn_logits = _logit_from_prob(cnn)
                prob, aux = model(cnn, gnn, cnn_logits=cnn_logits)
                logits_total = aux['logits_total']

                loss_bce  = bce_with_logits_masked_posw(logits_total, tgt, wts)
                loss_dice = dice_loss_with_logits(logits_total, tgt, weights=wts)
                loss = 0.3*loss_bce + 0.7*loss_dice
                if args.mu_mode == 'dynamic' and ('mu' in aux) and (args.mu_reg > 0.0):
                    loss = loss + float(args.mu_reg) * (1.0 - aux['mu']).mean()
                loss = loss / max(int(args.accum_steps), 1)

            scaler.scale(loss).backward()
            step_in_accum += 1
            if step_in_accum >= max(int(args.accum_steps), 1):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step_in_accum = 0

            running += float(loss.item()) * cnn.size(0) * max(int(args.accum_steps), 1)

        tr_loss = running / len(ds_tr)
        log = f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  lr={_get_lr(opt):.2e}"

        # ---- validation ----
        do_val = (dl_va is not None) and ((epoch % max(int(args.val_every),1)) == 0 or epoch==1)
        if do_val:
            model.eval()
            dices_aff, dices_cnn = [], []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp, dtype=amp_dtype):
                for batch in dl_va:
                    stem = batch['id'][0]
                    cnn = batch['prob'].to(device).to(amp_dtype)
                    H, W = cnn.shape[-2], cnn.shape[-1]
                    gnn = _read_graph_dense_cpu(batch['graph_path'][0], (H,W), cache_dir=args.cache_gnn_dir).to(device).to(amp_dtype)
                    cnn_logits = _logit_from_prob(cnn)
                    prob, aux = model(cnn, gnn, cnn_logits=cnn_logits)
                    pr_aff = prob[0,0].float().detach().cpu().numpy()
                    pr_cnn = cnn[0,0].float().detach().cpu().numpy()

                    # scheduled saving
                    if args.save_probmaps and ((epoch % max(int(args.save_every),1)) == 0 or epoch==1):
                        _save_png01(os.path.join(out_prob_dir, f"{stem}_e{epoch:03d}_aff_prob.png"), pr_aff)
                        _save_png01(os.path.join(out_prob_dir, f"{stem}_e{epoch:03d}_cnn_prob.png"), pr_cnn)
                        if args.mu_mode == 'dynamic' and 'mu' in aux:
                            mu0 = aux['mu'][0,0].float().detach().cpu().numpy()
                            _save_png01(os.path.join(out_prob_dir, f"{stem}_e{epoch:03d}_mu.png"), mu0)

                    if batch['label'][0] is not None:
                        gt = batch['label'][0][0].numpy()
                        fov = batch['fov'][0][0].numpy()
                        # quick dice
                        yp = (pr_aff >= 0.5).astype(np.uint8); yt = (gt >= 0.5).astype(np.uint8); mk = (fov>0.5)
                        tp = int(((yp==1)&(yt==1)&mk).sum()); fp = int(((yp==1)&(yt==0)&mk).sum()); fn = int(((yp==0)&(yt==1)&mk).sum())
                        dice = (2*tp) / max(2*tp+fp+fn,1)
                        # baseline
                        yp2 = (pr_cnn >= 0.5).astype(np.uint8)
                        tp2 = int(((yp2==1)&(yt==1)&mk).sum()); fp2 = int(((yp2==1)&(yt==0)&mk).sum()); fn2 = int(((yp2==0)&(yt==1)&mk).sum())
                        dice2 = (2*tp2) / max(2*tp2+fp2+fn2,1)
                        dices_aff.append(dice); dices_cnn.append(dice2)

            if dices_aff:
                val_dice = float(np.mean(dices_aff))
                val_cnn  = float(np.mean(dices_cnn)) if dices_cnn else float('nan')
                log += f"  val_dice_aff={val_dice:.4f}  val_dice_cnn={val_cnn:.4f}"
                scheduler.step(val_dice)
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save({'model': model.state_dict(),
                                'in_ch_gnn': in_ch_gnn,
                                'mid_ch': args.mid_ch,
                                'mu_mode': args.mu_mode,
                                'static_mu': args.static_mu,
                                'use_prior_skip': args.use_prior_skip,
                                'prior_gain': args.prior_gain},
                               os.path.join(args.save_dir, 'best_aff.ckpt'))

        print(log)

    # final save
    torch.save({'model': model.state_dict(),
                'in_ch_gnn': in_ch_gnn,
                'mid_ch': args.mid_ch,
                'mu_mode': args.mu_mode,
                'static_mu': args.static_mu,
                'use_prior_skip': args.use_prior_skip,
                'prior_gain': args.prior_gain},
               os.path.join(args.save_dir, 'last_aff.ckpt'))

if __name__ == "__main__":
    # For 24GB: keep torch.compile OFF (bloat) and use expandable_segments.
    #   export TORCH_COMPILE_DISABLE=1
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    main()
