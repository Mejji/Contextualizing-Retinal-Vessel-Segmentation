# -*- coding: utf-8 -*-
# Test AFF fusion on a folder of HRF/DRIVE/CHASE probmaps + graph_res.
# Robust pairing + case-insensitive CHASE_DB1 GT lookup + proper HRF nested /images/ discovery + optional FOV fallback.

import os, re, glob, math, argparse, pickle, warnings, csv, fnmatch
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.AFF_Module import AFFModule, rasterize_gnn_features

try:
    import networkx as nx
except Exception:
    nx = None

GRAPH_EXTS = ('.graph_res', '.npz', '.npy', '.gpickle', '.pkl')
IMG_EXTS   = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif')

def _save_png01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(np.asarray(arr01, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def _imread_gray_float(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32)
    if arr.max() > 1.0: arr = arr / 255.0
    return arr

def _imread_mask01(path: str) -> np.ndarray:
    # Accepts tif/png/gif; if 3ch, take first; binarize >0.5
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
    if arr.max() > 1.0: arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)

def _is_graph_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in GRAPH_EXTS

def _file_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _prefix_from_stem(stem: str) -> str:
    s = stem.lower()
    # HRF: 01_dr_10_80 -> prefix 01_dr
    m = re.match(r'^(\d+_(dr|g|h))', s)
    if m: return m.group(1)
    # DRIVE: 21_training_04_10 -> 21_training ; 07_test_XX -> 07_test
    m = re.match(r'^(\d+_(training|test))', s)
    if m: return m.group(1)
    # CHASE_DB1: Image_01L_08_40 -> Image_01L
    m = re.match(r'^(image_\d+[lr])', s)
    if m: return m.group(1)
    return stem

def _collect_pairs(root: str, require_graph_ext: Optional[str], loose_pairing: bool=False):
    """
    Return list of tuples: (stem, prob_path, graph_path).
    Graph-driven strict pairing; avoids 'graph_res.png'.
    """
    pairs = []
    # index probmaps by prefix
    prob_by_prefix: Dict[str, List[str]] = {}
    for p in glob.glob(os.path.join(root, '*')):
        name = os.path.basename(p).lower()
        if any(name.endswith(ext) for ext in IMG_EXTS) and ('graph_res' not in name) and ('prob' in name or 'cnn' in name):
            stem = _file_stem(p)
            prefix = _prefix_from_stem(stem)
            prob_by_prefix.setdefault(prefix, []).append(p)

    # graph-driven strict
    graphs = [p for p in glob.glob(os.path.join(root, '*')) if _is_graph_file(p)]
    if require_graph_ext:
        req = require_graph_ext.lower().strip()
        graphs = [g for g in graphs if g.lower().endswith(req)]

    for g in sorted(graphs):
        stem = _file_stem(g)
        prefix = _prefix_from_stem(stem)
        cands = prob_by_prefix.get(prefix, [])
        if not cands and loose_pairing:
            for k, vlist in prob_by_prefix.items():
                if k in stem: cands.extend(vlist)
        if not cands:
            raise RuntimeError(f"No CNN probmap for stem '{stem}'. Expected something like '{prefix}_prob.png' in {root} (loose_pairing={'on' if loose_pairing else 'off'}).")
        pairs.append((stem, sorted(cands)[0], g))
    return pairs

def _from_np_array_to_dense(arr: np.ndarray, out_hw) -> torch.Tensor:
    H, W = out_hw
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2: arr = arr[None, ...]
    ten = torch.from_numpy(arr).float().unsqueeze(0)
    ten = F.interpolate(ten, size=(H,W), mode='bilinear', align_corners=False)
    return ten.cpu()

def _read_graph_dense(path: str, out_hw, device, cache_dir: Optional[str]=None, default_cell: int=8) -> torch.Tensor:
    H, W = out_hw
    suf = os.path.splitext(path)[1].lower()
    ten = None
    if suf in ('.graph_res', '.pkl', '.gpickle'):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                dense = obj.get('dense') or obj.get('gnn_dense') or obj.get('feat_map')
                if dense is not None:
                    ten = _from_np_array_to_dense(dense, (H,W))
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
                ten = rasterize_gnn_features(verts, (math.ceil(H/default_cell), math.ceil(W/default_cell)), default_cell, node_feat, (H,W)).cpu()
        except Exception:
            ten = None
    if suf == '.npz' and ten is None:
        data = np.load(path, allow_pickle=True)
        dense = data.get('dense') or data.get('gnn_dense') or data.get('feat_map')
        if dense is not None:
            ten = _from_np_array_to_dense(dense, (H,W))
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
        ten = _from_np_array_to_dense(arr, (H,W))
    if ten is None:
        raise ValueError(f"Unsupported graph_res format: {path}")
    return ten.to(device)

# -------- metrics --------
def _bce_loss_np(y_true, y_prob, mask=None, eps=1e-7):
    y = y_true.astype(np.float32).ravel()
    p = np.clip(y_prob.astype(np.float32).ravel(), eps, 1.0-eps)
    if mask is not None:
        m = (mask.ravel() > 0)
        y, p = y[m], p[m]
    return float(- (y*np.log(p) + (1-y)*np.log(1-p)).mean())

def _bin_counts(y_true, y_pred, mask=None):
    y = (y_true.ravel() > 0.5); yp = (y_pred.ravel() > 0.5)
    if mask is not None:
        m = (mask.ravel() > 0); y = y[m]; yp = yp[m]
    tp = int((y & yp).sum()); tn = int((~y & ~yp).sum()); fp = int((~y & yp).sum()); fn = int((y & ~yp).sum())
    return tp, tn, fp, fn

def _pr_rc_curve(y_true, y_prob, mask=None, num=256):
    y = (y_true.ravel() > 0.5).astype(np.uint8)
    p = y_prob.ravel().astype(np.float32)
    if mask is not None:
        m = (mask.ravel() > 0)
        y, p = y[m], p[m]
    thrs = np.linspace(0.0, 1.0, num=num)
    P, R = [], []
    for t in thrs:
        yp = (p >= t).astype(np.uint8)
        tp = int(((y==1)&(yp==1)).sum()); fp = int(((y==0)&(yp==1)).sum()); fn = int(((y==1)&(yp==0)).sum())
        prec = tp / max(tp+fp,1); rec = tp / max(tp+fn,1)
        P.append(prec); R.append(rec)
    order = np.argsort(R)
    R_sorted = np.array(R)[order]; P_sorted = np.array(P)[order]
    ap = float(np.trapz(P_sorted, R_sorted))
    return np.array(P), np.array(R), ap

def _roc_curve(y_true, y_prob, mask=None, num=256):
    y = (y_true.ravel() > 0.5).astype(np.uint8)
    p = y_prob.ravel().astype(np.float32)
    if mask is not None:
        m = (mask.ravel() > 0)
        y, p = y[m], p[m]
    thrs = np.linspace(0.0, 1.0, num=num)
    TPR, FPR = [], []
    for t in thrs:
        yp = (p >= t).astype(np.uint8)
        tp = int(((y==1)&(yp==1)).sum()); fp = int(((y==0)&(yp==1)).sum())
        tn = int(((y==0)&(yp==0)).sum()); fn = int(((y==1)&(yp==0)).sum())
        tpr = tp / max(tp+fn,1); fpr = fp / max(fp+tn,1)
        TPR.append(tpr); FPR.append(fpr)
    order = np.argsort(FPR)
    FPRs = np.array(FPR)[order]; TPRs = np.array(TPR)[order]
    auc = float(np.trapz(TPRs, FPRs))
    return np.array(FPRs), np.array(TPRs), auc

# -------- model wrapper --------
class AFFWrapper(nn.Module):
    def __init__(self, in_ch_cnn=1, in_ch_gnn=64, mid_ch=64, out_ch=1,
                 mu_mode='dynamic', static_mu=0.5,
                 use_prior_skip=True, prior_gain=0.0):
        super().__init__()
        self.aff = AFFModule(in_ch_cnn, in_ch_gnn, mid_ch, out_ch)
        self.mu_mode = mu_mode
        self.register_buffer('mu_const', torch.tensor(float(static_mu)).view(1,1,1,1))
        self.use_prior_skip = bool(use_prior_skip)
        self.register_buffer('prior_gain', torch.tensor(float(prior_gain)))

        if hasattr(self.aff, 'head') and hasattr(self.aff.head, 'weight'):
            nn.init.zeros_(self.aff.head.weight)
            if self.aff.head.bias is not None: nn.init.zeros_(self.aff.head.bias)

    def forward(self, cnn_feat, gnn_feat, cnn_logits=None):
        if cnn_feat.dim() == 3: cnn_feat = cnn_feat.unsqueeze(1)
        if gnn_feat.dim() == 3: gnn_feat = gnn_feat.unsqueeze(1)
        H, W = cnn_feat.shape[-2:]
        gnn_feat = F.interpolate(gnn_feat, size=(H,W), mode='bilinear', align_corners=False)

        c = self.aff.cnn_align(cnn_feat)
        g = self.aff.gnn_align(gnn_feat)
        mu = self.aff.gate(torch.cat([c, g], dim=1)) if self.mu_mode == 'dynamic' else self.mu_const.expand_as(c)
        fused = mu * c + (1.0 - mu) * g
        logits = self.aff.head(self.aff.refine(fused))
        logits_total = logits + (self.prior_gain * cnn_logits) if (self.use_prior_skip and (cnn_logits is not None)) else logits
        prob_total = torch.sigmoid(logits_total)
        return prob_total, {'mu': mu, 'logits_total': logits_total}

def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return True
    return str(v).lower() in ('1','true','t','yes','y')

def _parse_hw(hwstr: str) -> Optional[Tuple[int,int]]:
    if not hwstr: return None
    m = re.match(r'(\d+)[xX](\d+)', hwstr)
    return (int(m.group(1)), int(m.group(2))) if m else None

# ---------- robust, case-insensitive file discovery ----------

def _anycase_files(root: str) -> List[str]:
    # Cache all files (full paths) for any-case matching
    return [os.path.join(dp, f) for dp, _, files in os.walk(root) for f in files]

def _find_anycase(all_files: List[str], pattern_basename: str) -> Optional[str]:
    pat = pattern_basename.lower()
    for p in all_files:
        if fnmatch.fnmatch(os.path.basename(p).lower(), pat):
            return p
    return None

def _find_anycase_all(all_files: List[str], pattern_basename: str) -> List[str]:
    pat = pattern_basename.lower()
    hits = []
    for p in all_files:
        if fnmatch.fnmatch(os.path.basename(p).lower(), pat):
            hits.append(p)
    return hits

def main():
    ap = argparse.ArgumentParser("Test AFF (folder-based, robust pairing).")
    ap.add_argument('--root', type=str, required=True, help='folder with probmaps + graph files')
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--require_graph_ext', type=str, default='', help="e.g. '.graph_res' to force type")
    ap.add_argument('--dataset_root', type=str, default='', help='for GT/mask discovery (optional)')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--amp', type=str2bool, nargs='?', const=True, default=True)
    ap.add_argument('--amp_dtype', choices=['fp16','bf16','fp32'], default='bf16')
    ap.add_argument('--save_mu', type=str2bool, nargs='?', const=True, default=True)
    ap.add_argument('--force_prior_gain', type=float, default=None, help='override ckpt prior_gain if set')
    ap.add_argument('--fixed_hw', type=str, default='')  # match train if you resized
    ap.add_argument('--loose_pairing', type=str2bool, nargs='?', const=True, default=False)
    # NEW: per-image reporting
    ap.add_argument('--csv_path', type=str, default='', help='optional path for per-image CSV (defaults to <save_dir>/per_image_metrics.csv)')
    ap.add_argument('--per_image_stdout', type=str2bool, nargs='?', const=True, default=True)
    # NEW: FOV fallback behavior
    ap.add_argument('--fov_fallback', choices=['ones','none'], default='ones', help="if no FOV found, 'ones' uses full image; 'none' disables metrics.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if not args.csv_path:
        args.csv_path = os.path.join(args.save_dir, 'per_image_metrics.csv')

    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.amp_dtype]

    # load ckpt
    ck = torch.load(args.ckpt, map_location='cpu')
    in_ch_gnn = int(ck.get('in_ch_gnn', 64))
    mid_ch = int(ck.get('mid_ch', 64))
    mu_mode = ck.get('mu_mode', 'dynamic')
    prior_skip = bool(ck.get('use_prior_skip', True))
    prior_gain = float(ck.get('prior_gain', 0.0))
    if args.force_prior_gain is not None:
        prior_gain = float(args.force_prior_gain)
    print(f"[INFO] Loaded ckpt args: in_ch_gnn={in_ch_gnn} mid_ch={mid_ch} mu_mode={mu_mode} prior_skip={prior_skip} prior_gain={prior_gain}")

    # collect pairs
    pairs = _collect_pairs(args.root, args.require_graph_ext, args.loose_pairing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AFFWrapper(in_ch_cnn=1, in_ch_gnn=in_ch_gnn, mid_ch=mid_ch, out_ch=1,
                       mu_mode=mu_mode, use_prior_skip=prior_skip, prior_gain=prior_gain).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    fixed_hw = _parse_hw(args.fixed_hw)

    # Pre-index files under dataset_root for any-case matching
    all_ds_files = _anycase_files(args.dataset_root) if args.dataset_root and os.path.isdir(args.dataset_root) else []

    def _prefer_images_dir(candidates: List[str]) -> Optional[str]:
        if not candidates: return None
        # Prefer a path containing '/images/' (or '\images\') if present
        def score(p):
            pl = p.replace('\\', '/').lower()
            return (('/images/' in pl) or pl.endswith('/images')) or ('/manual1/' in pl)
        # sort: True first
        candidates_sorted = sorted(candidates, key=lambda p: (not score(p), len(p)))
        return candidates_sorted[0]

    def _guess_gt(root: str, prefix: str):
        """
        Try HRF/DRIVE/CHASE_DB1 patterns.
        - HRF: accept nested .../<split>/images/<prefix>_{label,mask}.*
        - DRIVE: manual/mask folders by number
        - CHASE_DB1: case-insensitive search anywhere for *_1stHO.* and *_mask.*
        """
        lab, fov = None, None

        # --- HRF canonical (top-level quick checks) ---
        for cand in [
            f"{prefix}_label.tif", f"labels/{prefix}_label.tif",
            f"manual1/{prefix}.tif", f"gt/{prefix}_label.tif"
        ]:
            p = os.path.join(root, cand)
            if os.path.exists(p): lab = p; break

        for cand in [
            f"{prefix}_mask.tif", f"mask/{prefix}_mask.tif",
            f"mask/{prefix}.png", f"masks/{prefix}_mask.tif"
        ]:
            p = os.path.join(root, cand)
            if os.path.exists(p): fov = p; break

        # --- HRF recursive search: nested /images/ (training/testing) ---
        if all_ds_files:
            if lab is None:
                # Prefer images/ directory matches; allow any extension
                hits = []
                for pat in [f"{prefix}_label.*", f"{prefix}.*"]:  # second covers manual1/<prefix>.tif
                    hits.extend(_find_anycase_all(all_ds_files, pat))
                if hits:
                    # Filter to likely labels (suffix _label or manual1)
                    labelish = [h for h in hits if re.search(r'(_label\.)', os.path.basename(h).lower()) or '/manual1/' in h.replace('\\','/').lower()]
                    lab = _prefer_images_dir(labelish or hits)

            if fov is None:
                hits = []
                for pat in [f"{prefix}_mask.*", f"{prefix}*mask.*", f"{prefix}_fov.*"]:
                    hits.extend(_find_anycase_all(all_ds_files, pat))
                if hits:
                    fov = _prefer_images_dir(hits)

        # --- DRIVE fallback (by image number) ---
        if (lab is None) or (fov is None):
            s_lower = prefix.lower()
            set_name = 'test' if ('_test' in s_lower or s_lower.endswith('_test')) else 'training'
            mnum = re.search(r'(\d+)', prefix)
            if mnum:
                num = mnum.group(1)
                for cand in [
                    f"{set_name}/1st_manual/{num}_manual1.gif",
                    f"{set_name}/1st_manual/{num}_manual1.png",
                ]:
                    p = os.path.join(root, cand)
                    if (lab is None) and os.path.exists(p): lab = p
                for cand in [
                    f"{set_name}/mask/{num}_{set_name}_mask.gif",
                    f"{set_name}/mask/{num}_{set_name}_mask.png",
                ]:
                    p = os.path.join(root, cand)
                    if (fov is None) and os.path.exists(p): fov = p

        # --- CHASE_DB1 fallback: case-insensitive search anywhere ---
        if lab is None and all_ds_files:
            for base in [f"{prefix}_1stho.*", f"{prefix}_2ndho.*"]:
                hit = _find_anycase(all_ds_files, base)
                if hit: lab = hit; break
            if lab is None:
                loose = _find_anycase(all_ds_files, f"{prefix}*_*1stho.*")
                if loose: lab = loose

        if fov is None and all_ds_files:
            for base in [f"{prefix}_mask.*", f"{prefix}*mask.*", f"{prefix}_fov.*"]:
                hit = _find_anycase(all_ds_files, base)
                if hit: fov = hit; break

        return lab, fov

    # metric accumulators
    S = {'acc':[], 'sp':[], 'se':[], 'f1':[], 'iou':[], 'miou':[], 'auc':[], 'ap':[], 'bce':[]}
    S_cnn = {'acc':[], 'sp':[], 'se':[], 'f1':[], 'iou':[], 'miou':[], 'auc':[], 'ap':[], 'bce':[]}

    # per-image rows for CSV
    rows: List[Dict] = []
    header = [
        'id','H','W','thr','kind','acc','sp','se','f1','iou','miou','auc','ap','bce',
        'gt_path','fov_path','mu_mean','mu_min','mu_max'
    ]

    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(args.amp and device.type=='cuda'))

    for stem, prob_p, gpath in pairs:
        pr = _imread_gray_float(prob_p)  # HxW 0..1
        H, W = pr.shape
        Ht, Wt = (H, W)
        if fixed_hw is not None and (H,W) != fixed_hw:
            Ht, Wt = fixed_hw

        lab_p, fov_p = (None, None)
        if args.dataset_root:
            lab_p, fov_p = _guess_gt(args.dataset_root, _prefix_from_stem(stem))
            if lab_p: print(f"[INFO] GT found for {stem}: {lab_p}")
            if fov_p: print(f"[INFO] FOV found for {stem}: {fov_p}")
            if (not fov_p) and args.fov_fallback == 'ones':
                fov_p = 'AUTO-ONES'  # sentinel

        cnn = torch.from_numpy(pr)[None,None,...]
        if (Ht,Wt)!=(H,W):
            cnn = F.interpolate(cnn, size=(Ht,Wt), mode='bilinear', align_corners=False)
        cnn = cnn.to(device)
        cnn_logits = torch.logit(cnn.float().clamp(1e-5, 1-1e-5))

        gnn = _read_graph_dense(gpath, (Ht,Wt), device=device)

        with torch.no_grad(), autocast_ctx:
            prob, aux = model(cnn.to(amp_dtype), gnn.to(amp_dtype), cnn_logits=cnn_logits.to(amp_dtype))
            pr_aff = prob[0,0].float().detach().cpu().numpy()
            pr_cnn = cnn[0,0].float().detach().cpu().numpy()

        # save predictions
        _save_png01(os.path.join(args.save_dir, f"{stem}_aff_prob.png"), pr_aff)
        _save_png01(os.path.join(args.save_dir, f"{stem}_cnn_prob.png"), pr_cnn)
        mu_mean = mu_min = mu_max = float('nan')
        if args.save_mu and 'mu' in aux and isinstance(aux['mu'], torch.Tensor):
            mu = aux['mu'][0].float().detach().cpu().numpy()  # [C,H,W]
            mu_mean = float(mu.mean())
            mu_min  = float(mu.min())
            mu_max  = float(mu.max())
            _save_png01(os.path.join(args.save_dir, f"{stem}_mu.png"), mu[0])

        # metrics (if GT present)
        have_lab = (lab_p is not None) and (os.path.exists(lab_p) if lab_p != 'AUTO' else True)
        have_fov = (fov_p is not None) and ((fov_p=='AUTO-ONES') or os.path.exists(fov_p))
        if lab_p and have_lab and have_fov:
            gt = _imread_mask01(lab_p).astype(np.float32)
            if fov_p=='AUTO-ONES':
                fov = np.ones_like(gt, dtype=np.float32)
            else:
                fov = _imread_mask01(fov_p).astype(np.float32)
            if (Ht,Wt)!=(gt.shape[0],gt.shape[1]):
                gt  = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((Wt,Ht), Image.NEAREST))/255.0
                gt  = (gt>0.5).astype(np.float32)
                fov = np.array(Image.fromarray((fov*255).astype(np.uint8)).resize((Wt,Ht), Image.NEAREST))/255.0
                fov = (fov>0.5).astype(np.float32)

            def _pack(y_prob):
                tp, tn, fp, fn = _bin_counts(gt, (y_prob>=args.thr).astype(np.float32), fov)
                acc = (tp+tn)/max(tp+tn+fp+fn,1)
                sp  = tn/max(tn+fp,1)
                se  = tp/max(tp+fn,1)
                f1  = (2*tp)/max(2*tp+fp+fn,1)
                iou = tp/max(tp+fp+fn,1)
                miou = (iou + tn/max(tn+fp+fn,1)) / 2.0
                FPR, TPR, auc = _roc_curve(gt, y_prob, fov)
                P, R, ap = _pr_rc_curve(gt, y_prob, fov)
                bce = _bce_loss_np(gt, y_prob, fov)
                return acc, sp, se, f1, iou, miou, auc, ap, bce

            # AFF
            acc, sp, se, f1, iou, miou, auc, ap, bce = _pack(pr_aff)
            for k,v in zip(S.keys(), [acc,sp,se,f1,iou,miou,auc,ap,bce]): S[k].append(v)
            rows.append({
                'id': stem, 'H': Ht, 'W': Wt, 'thr': args.thr, 'kind': 'AFF',
                'acc': acc, 'sp': sp, 'se': se, 'f1': f1, 'iou': iou, 'miou': miou, 'auc': auc, 'ap': ap, 'bce': bce,
                'gt_path': lab_p, 'fov_path': fov_p, 'mu_mean': mu_mean, 'mu_min': mu_min, 'mu_max': mu_max
            })
            if args.per_image_stdout:
                print(f"[{stem}] AFF  F1={f1:.4f} IoU={iou:.4f} Se={se:.4f} Sp={sp:.4f} Acc={acc:.4f} AUC={auc:.4f} AP={ap:.4f} BCE={bce:.6f}  μ(mean/min/max)={mu_mean:.3f}/{mu_min:.3f}/{mu_max:.3f}")

            # CNN
            acc, sp, se, f1, iou, miou, auc, ap, bce = _pack(pr_cnn)
            for k,v in zip(S_cnn.keys(), [acc,sp,se,f1,iou,miou,auc,ap,bce]): S_cnn[k].append(v)
            rows.append({
                'id': stem, 'H': Ht, 'W': Wt, 'thr': args.thr, 'kind': 'CNN',
                'acc': acc, 'sp': sp, 'se': se, 'f1': f1, 'iou': iou, 'miou': miou, 'auc': auc, 'ap': ap, 'bce': bce,
                'gt_path': lab_p, 'fov_path': fov_p, 'mu_mean': mu_mean, 'mu_min': mu_min, 'mu_max': mu_max
            })
            if args.per_image_stdout:
                print(f"[{stem}] CNN  F1={f1:.4f} IoU={iou:.4f} Se={se:.4f} Sp={sp:.4f} Acc={acc:.4f} AUC={auc:.4f} AP={ap:.4f} BCE={bce:.6f}")

        else:
            if args.per_image_stdout:
                print(f"[{stem}] GT/mask not found → saved probmaps (AFF + CNN){' and μ' if args.save_mu else ''}.")
            rows.append({
                'id': stem, 'H': Ht, 'W': Wt, 'thr': args.thr, 'kind': 'AFF',
                'acc': np.nan, 'sp': np.nan, 'se': np.nan, 'f1': np.nan, 'iou': np.nan, 'miou': np.nan, 'auc': np.nan, 'ap': np.nan, 'bce': np.nan,
                'gt_path': lab_p, 'fov_path': fov_p, 'mu_mean': mu_mean, 'mu_min': mu_min, 'mu_max': mu_max
            })

    # write per-image CSV
    try:
        with open(args.csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in header})
        print(f"[INFO] Wrote per-image metrics to: {args.csv_path}")
    except Exception as e:
        print(f"[WARN] Failed to write CSV at {args.csv_path}: {e}")

    # print summary if GT present
    if S['f1']:
        def _m(d): return {k: float(np.mean(v)) if len(v)>0 else float('nan') for k,v in d.items()}
        A = _m(S); C = _m(S_cnn)
        print("\n--- AFF (fused) ---")
        print(f"Acc:\t  {A['acc']:.4f}")
        print(f"SP:\t  {A['sp']:.4f}")
        print(f"SE/Recall:{A['se']:.4f}")
        print(f"F1/Dice:\t{A['f1']:.4f}")
        print(f"IoU:\t  {A['iou']:.4f}")
        print(f"mIoU:\t  {A['miou']:.4f}")
        print(f"AUC:\t  {A['auc']:.4f}")
        print(f"AP:\t  {A['ap']:.4f}")
        print(f"BCE loss: {A['bce']:.6f}")

        print("\n--- CNN (baseline probmap) ---")
        print(f"Acc:\t  {C['acc']:.4f}")
        print(f"SP:\t  {C['sp']:.4f}")
        print(f"SE/Recall:{C['se']:.4f}")
        print(f"F1/Dice:\t{C['f1']:.4f}")
        print(f"IoU:\t  {C['iou']:.4f}")
        print(f"mIoU:\t  {C['miou']:.4f}")
        print(f"AUC:\t  {C['auc']:.4f}")
        print(f"AP:\t  {C['ap']:.4f}")
        print(f"BCE loss: {C['bce']:.6f}")
    else:
        print("[INFO] GT/mask not found anywhere. Only saved probmaps (AFF + CNN) and μ images.")

if __name__ == "__main__":
    main()
