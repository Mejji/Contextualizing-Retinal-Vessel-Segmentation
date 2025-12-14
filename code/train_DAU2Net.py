import os, sys, math, glob, random, argparse, re, time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image

from Modules.DAU2Net import DAU2Net


# -----------------------------
# Utilities: reproducibility
# -----------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Image I/O & preprocessing
# -----------------------------

def read_rgb(path):
    # support TIF/PNG/JPG; always return RGB float32 [0..1]
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def preprocess_green_clahe_gamma(img_rgb, clip=2.0, tile=8, gamma=1.2):
    """
    Paper preprocessing: Project to green channel, CLAHE, gamma correction.
    Returns 3-channel normalized image by repeating enhanced green channel.
    """
    g = img_rgb[..., 1]
    g8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    g_clahe = clahe.apply(g8).astype(np.float32) / 255.0
    # gamma
    g_gamma = np.power(np.clip(g_clahe, 0, 1), 1.0 / gamma)
    out = np.stack([g_gamma, g_gamma, g_gamma], axis=0)  # CHW
    return out


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    # Normalize to {0,1}
    m = (m > 127).astype(np.float32)
    return m


def pad_resize(img_chw, mask_hw, size=None, long_side=None):
    """
    Keep aspect ratio: pad to square then resize to (size,size).
    Used inside Dataset (training) — does NOT return geometry meta.
    """
    c, h, w = img_chw.shape
    m = mask_hw
    s = max(h, w)
    pad_h = s - h
    pad_w = s - w
    img_pad = np.pad(img_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    m_pad = np.pad(m, ((0, pad_h), (0, pad_w)), mode='constant')
    if size is not None:
        img_pad = torch.from_numpy(img_pad).unsqueeze(0)
        m_pad = torch.from_numpy(m_pad).unsqueeze(0).unsqueeze(0)
        img_res = torch.nn.functional.interpolate(img_pad, size=(size, size), mode='bilinear', align_corners=False)
        m_res = torch.nn.functional.interpolate(m_pad, size=(size, size), mode='nearest')
        return img_res.squeeze(0).numpy(), m_res.squeeze(0).squeeze(0).numpy()
    return img_pad, m_pad


# --- NEW: pad/resize helper for dumping (with geometry meta)
def pad_resize_img_meta(img_chw, size=None):
    """
    Same pad+resize as pad_resize, but for images only and returns:
      img_sq: (C,size,size)
      orig_hw: (h,w) of original image
      pads: (pad_h,pad_w) that were added before resizing
    """
    c, h, w = img_chw.shape
    s = max(h, w)
    pad_h = s - h
    pad_w = s - w
    img_pad = np.pad(img_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    if size is not None:
        t = torch.from_numpy(img_pad).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
        img_sq = t.squeeze(0).numpy()
        return img_sq, (h, w), (pad_h, pad_w)
    return img_pad, (h, w), (pad_h, pad_w)


def unpad_resize(prob_sq, orig_hw, pads, out_h=None, out_w=None):
    """
    Inverse of pad_resize_img_meta:
      prob_sq: (size,size) prob map from network
      orig_hw: original (h,w)
      pads: (pad_h,pad_w) used to create the square
    Returns prob map in original image coordinates (h×w), or resized to (out_h,out_w) if specified.
    """
    h, w = orig_hw
    pad_h, pad_w = pads
    s = h + pad_h  # == w + pad_w == max(h,w)
    prob_s = cv2.resize(prob_sq, (s, s), interpolation=cv2.INTER_LINEAR)
    prob_hw = prob_s[:h, :w]
    if out_h is not None and out_w is not None and (out_h, out_w) != (h, w):
        prob_hw = cv2.resize(prob_hw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return prob_hw


# -----------------------------
# Augmentations
# -----------------------------

def random_augment(img_chw, mask_hw, translate=50, rotate=0, scale_jitter=0.0):
    c, h, w = img_chw.shape
    # flips
    if random.random() < 0.5:
        img_chw = img_chw[:, :, ::-1].copy()
        mask_hw = mask_hw[:, ::-1].copy()
    if random.random() < 0.5:
        img_chw = img_chw[:, ::-1, :].copy()
        mask_hw = mask_hw[::-1, :].copy()

    # translations
    tx = random.randint(-translate, translate)
    ty = random.randint(-translate, translate)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_nhw = np.transpose(img_chw, (1, 2, 0))
    img_nhw = cv2.warpAffine(img_nhw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_hw = cv2.warpAffine(mask_hw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img_chw = np.transpose(img_nhw, (2, 0, 1))

    # optional scale jitter
    if scale_jitter and scale_jitter > 0:
        s = 1.0 + random.uniform(-scale_jitter, scale_jitter)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        img_nhw = np.transpose(img_chw, (1, 2, 0))
        img_nhw = cv2.resize(img_nhw, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_hw = cv2.resize(mask_hw, (nw, nh), interpolation=cv2.INTER_NEAREST)
        # center pad/crop to (h,w)
        pad_h = max(0, h - nh)
        pad_w = max(0, w - nw)
        if pad_h > 0 or pad_w > 0:
            img_nhw = cv2.copyMakeBorder(img_nhw, pad_h // 2, h - nh - pad_h // 2,
                                         pad_w // 2, w - nw - pad_w // 2, cv2.BORDER_REFLECT101)
            mask_hw = cv2.copyMakeBorder(mask_hw, pad_h // 2, h - nh - pad_h // 2,
                                         pad_w // 2, w - nw - pad_w // 2, cv2.BORDER_CONSTANT, value=0)
        img_nhw = img_nhw[:h, :w]
        mask_hw = mask_hw[:h, :w]
        img_chw = np.transpose(img_nhw, (2, 0, 1))

    # optional small rotations
    if rotate and random.random() < 0.5:
        angle = random.uniform(-rotate, rotate)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_nhw = np.transpose(img_chw, (1, 2, 0))
        img_nhw = cv2.warpAffine(img_nhw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask_hw = cv2.warpAffine(mask_hw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img_chw = np.transpose(img_nhw, (2, 0, 1))

    return img_chw, mask_hw


# -----------------------------
# Dataset
# -----------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


def default_drive_split(root):
    """
    Expect DRIVE-like layout:
      root/
        training/images/*.tif
        training/1st_manual/*.gif
        training/mask/*.gif
        test/images/*.tif
        test/1st_manual/*.gif
        test/mask/*.gif
    """
    root = Path(root)

    def _filter_files(paths, exts):
        exts = {e.lower() for e in exts}
        out = []
        for p in paths:
            try:
                if p.is_file() and p.suffix.lower() in exts:
                    out.append(p)
            except Exception:
                continue
        return sorted(out)

    img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    gt_exts = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")
    tr_imgs = _filter_files((root / "training/images").glob("*"), img_exts)
    tr_gts = _filter_files((root / "training/1st_manual").glob("*"), gt_exts)
    tr_msks = _filter_files((root / "training/mask").glob("*"), gt_exts)
    te_imgs = _filter_files((root / "test/images").glob("*"), img_exts)
    te_gts = _filter_files((root / "test/1st_manual").glob("*"), gt_exts)
    te_msks = _filter_files((root / "test/mask").glob("*"), gt_exts)
    return (tr_imgs, tr_gts, tr_msks), (te_imgs, te_gts, te_msks)


def _align_by_numeric_id(imgs, gts):
    """Pair images and GTs by shared numeric id within the filename (e.g., '01')."""

    def key(p):
        s = Path(p).stem
        ids = re.findall(r"\d+", s)
        return ids[0] if ids else s

    im_map = {key(p): p for p in imgs if Path(p).is_file()}
    gt_map = {key(p): p for p in gts if Path(p).is_file()}
    common = sorted(set(im_map.keys()) & set(gt_map.keys()), key=lambda k: (len(k), k))
    return [im_map[k] for k in common], [gt_map[k] for k in common]


class VesselSet(Dataset):
    def __init__(self, img_paths, gt_paths, fov_masks=None, size=512,
                 train=True, translate=50, rotate=0, scale_jitter=0.0, preprocess=True):
        self.imgs = img_paths
        self.gts = gt_paths
        self.fovs = fov_masks if fov_masks and len(fov_masks) == len(img_paths) else [None] * len(img_paths)
        self.size = size
        self.train = train
        self.translate = translate
        self.rotate = rotate
        self.scale_jitter = scale_jitter
        self.preprocess = preprocess

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = read_rgb(self.imgs[i])
        gt = load_mask(self.gts[i])

        if self.preprocess:
            img = preprocess_green_clahe_gamma(img)  # CHW float
        else:
            img = np.transpose(img, (2, 0, 1))       # CHW

        # resize with pad to square
        img, gt = pad_resize(img, gt, size=self.size)

        if self.train:
            img, gt = random_augment(img, gt, translate=self.translate,
                                     rotate=self.rotate, scale_jitter=self.scale_jitter)

        img_t = torch.from_numpy(img).float()
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        return img_t, gt_t


# -----------------------------
# Loss & metrics
# -----------------------------

class DiceBCELoss(nn.Module):
    """
    Combine Dice loss with BCE to address detail saturation.
    """
    def __init__(self, pos_weight=1.5, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.register_buffer('pos_weight_buf', torch.tensor([pos_weight], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)

    def forward(self, logits, targets):
        self.bce.pos_weight = self.pos_weight_buf.to(logits.device)
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2. * torch.sum(probs * targets) + self.smooth
        den = torch.sum(probs + targets) + self.smooth
        dice = 1. - num / den
        return bce + dice


def compute_metrics(probs, gts, thresh=0.5, eps=1e-7):
    pred = (probs >= thresh).astype(np.uint8)
    gt = (gts >= 0.5).astype(np.uint8)
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    se = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    f1 = 2 * prec * se / (prec + se + eps)
    iou = tp / (tp + fp + fn + eps)
    return dict(Acc=acc, SP=sp, SE=se, F1=f1, IoU=iou)


def _confusion(y_true, y_pred, eps=1e-7):
    y_true = (y_true > 0.5).float()
    y_pred = (y_pred > 0.5).float()
    tp = (y_true * y_pred).sum().item()
    tn = ((1 - y_true) * (1 - y_pred)).sum().item()
    fp = ((1 - y_true) * y_pred).sum().item()
    fn = (y_true * (1 - y_pred)).sum().item()
    return tp, tn, fp, fn


def _metrics_from_counts(tp, tn, fp, fn):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sp = tn / max(tn + fp, 1)
    se = tp / max(tp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)  # Dice = F1
    iou_v = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou = 0.5 * (iou_v + iou_bg)
    return acc, sp, se, f1, miou, iou_v


# -----------------------------
# Training
# -----------------------------

def save_prob_map(prob_t, path):
    p = prob_t.squeeze(0).cpu().numpy()
    p8 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(p8).save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="C:/Users/rog/THESIS/DATASETS/DRIVE",
                    help="Path to dataset root (expects training/ and test/ subfolders like DRIVE).")
    ap.add_argument("--dataset", type=str, default="DRIVE", choices=["DRIVE", "CHASEDB1", "HRF", "CUSTOM"])
    ap.add_argument("--train_imgs", type=str, default="", help="(CUSTOM) glob or dir for train images")
    ap.add_argument("--train_gts", type=str, default="", help="(CUSTOM) glob or dir for train gts")
    ap.add_argument("--val_imgs", type=str, default="", help="(CUSTOM) glob or dir for val images")
    ap.add_argument("--val_gts", type=str, default="", help="(CUSTOM) glob or dir for val gts")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--translate", type=int, default=50)
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--scale_jitter", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="C:/Users/rog/THESIS/DAU2_DRIVE")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--pos_weight", type=float, default=1.5)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--cudnn_benchmark", action="store_true")
    ap.add_argument("--dump_train_every", type=int, default=0,
                    help="0 = dump only at final epoch; N>0 = dump every N epochs as well as final.")
    ap.add_argument("--dump_thresh", type=float, default=0.5,
                    help="Threshold for saving training bin masks.")
    # Explicit DRIVE defaults
    ap.add_argument("--drive_train_images", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/training/images")
    ap.add_argument("--drive_train_masks", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/training/1st_manual")
    ap.add_argument("--drive_train_fov", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/training/mask")
    ap.add_argument("--drive_val_images", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/test/images")
    ap.add_argument("--drive_val_masks", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/test/1st_manual")
    ap.add_argument("--drive_val_fov", type=str,
                    default="C:/Users/rog/THESIS/DATASETS/DRIVE/test/mask")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(f"{args.out}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.out}/val_probs", exist_ok=True)
    os.makedirs(f"{args.out}/train/prob", exist_ok=True)
    os.makedirs(f"{args.out}/train/bin", exist_ok=True)

    # ---------------- data split ----------------
    if args.dataset == "DRIVE":
        def _filter_files(paths, exts):
            exts = {e.lower() for e in exts}
            out = []
            for p in paths:
                p = Path(p)
                try:
                    if p.is_file() and p.suffix.lower() in exts:
                        out.append(p)
                except Exception:
                    continue
            return sorted(out)

        img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        gt_exts = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")

        tr_i = _filter_files(Path(args.drive_train_images).glob("*"), img_exts)
        tr_g = _filter_files(Path(args.drive_train_masks).glob("*"), gt_exts)
        te_i = _filter_files(Path(args.drive_val_images).glob("*"), img_exts)
        te_g = _filter_files(Path(args.drive_val_masks).glob("*"), gt_exts)

        train_imgs, train_gts = _align_by_numeric_id(tr_i, tr_g)
        val_imgs, val_gts = _align_by_numeric_id(te_i, te_g)
    elif args.dataset == "CUSTOM":
        def expand(p):
            if os.path.isdir(p):
                return sorted(Path(p).glob("*"))
            return sorted(glob.glob(p))

        train_imgs = expand(args.train_imgs)
        train_gts = expand(args.train_gts)
        val_imgs = expand(args.val_imgs)
        val_gts = expand(args.val_gts)
    else:
        root = Path(args.data_root)
        train_imgs = sorted((root / "train/images").glob("*"))
        train_gts = sorted((root / "train/labels").glob("*"))
        val_imgs = sorted((root / "val/images").glob("*"))
        val_gts = sorted((root / "val/labels").glob("*"))
        if not train_imgs:
            print("Please arrange CHASEDB1/HRF as train/images & train/labels etc., or use --CUSTOM.")
            sys.exit(1)

    assert len(train_imgs) == len(train_gts), "Mismatch train images/masks"
    assert len(val_imgs) == len(val_gts), "Mismatch val images/masks"

    train_set = VesselSet(train_imgs, train_gts, size=args.size, train=True,
                          translate=args.translate, rotate=args.rotate,
                          scale_jitter=args.scale_jitter, preprocess=True)
    val_set = VesselSet(val_imgs, val_gts, size=args.size, train=False, preprocess=True)

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    # We keep train_eval_loader for potential debugging, but the dump now uses the raw images.
    train_eval_set = VesselSet(train_imgs, train_gts, size=args.size, train=False, preprocess=True)
    train_eval_loader = DataLoader(train_eval_set, batch_size=1, shuffle=False,
                                   num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    # ---------------- model & optim ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAU2Net(in_ch=3, out_ch=1).to(device)
    if torch.cuda.is_available() and args.cudnn_benchmark:
        cudnn.benchmark = True

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = DiceBCELoss(pos_weight=args.pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    tmax = max(1, args.epochs - args.warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=args.eta_min)

    def multiscale_loss(fuse, sides, gt):
        loss = loss_fn(fuse, gt)
        for s in sides:
            loss = loss + 0.5 * loss_fn(s, gt)
        return loss

    best_f1 = 0.0
    t0 = time.time()

    # ---------------- training loop ----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for img_t, gt_t in train_loader:
            img_t = img_t.to(device, non_blocking=True)
            gt_t = gt_t.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                fuse, sides = model(img_t)
                loss = multiscale_loss(fuse, sides, gt_t)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach().cpu())

        # ---- validation ----
        model.eval()
        metrics_accum = dict(Acc=[], SP=[], SE=[], F1=[], IoU=[])
        agg_tp = agg_tn = agg_fp = agg_fn = 0.0
        with torch.no_grad():
            for vi, (img_t, gt_t) in enumerate(val_loader):
                img_t = img_t.to(device, non_blocking=True)
                gt_t = gt_t.to(device, non_blocking=True)
                fuse, sides = model(img_t)
                prob = torch.sigmoid(fuse)

                if vi < 5 and (epoch % args.save_every == 0 or epoch == 1):
                    save_prob_map(prob[0, 0], Path(f"{args.out}/val_probs") / f"ep{epoch:03d}_idx{vi:02d}.png")

                p = prob[0, 0].cpu().numpy()
                g = gt_t[0, 0].cpu().numpy()
                m = compute_metrics(p, g, thresh=0.5)
                for k, v in m.items():
                    metrics_accum[k].append(v)

                tp_i, tn_i, fp_i, fn_i = _confusion(gt_t, prob)
                agg_tp += tp_i
                agg_tn += tn_i
                agg_fp += fp_i
                agg_fn += fn_i

            m_epoch = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
            avg_loss = run_loss / max(1, len(train_loader))
            acc, sp, se, f1, miou, iou_v = _metrics_from_counts(agg_tp, agg_tn, agg_fp, agg_fn)

            elapsed = time.time() - t0
            avg_ep = elapsed / epoch
            rem_ep = max(0, args.epochs - epoch)
            eta_sec = rem_ep * avg_ep

            def _fmt(secs):
                h = int(secs // 3600)
                m = int((secs % 3600) // 60)
                s = int(secs % 60)
                return f"{h:02d}h {m:02d}m {s:02d}s"

            cur_lr = opt.param_groups[0]['lr']
            print(f"[Ep {epoch:03d}] loss={avg_loss:.4f}  LR={cur_lr:.6f}  "
                  f"Acc={m_epoch['Acc']:.4f}  SE={m_epoch['SE']:.4f}  SP={m_epoch['SP']:.4f}  "
                  f"F1={m_epoch['F1']:.4f}  IoU={m_epoch['IoU']:.4f}")
            print(f"[Val] Acc={acc:.4f}  SP={sp:.4f}  SE={se:.4f}  F1/Dice={f1:.4f}  "
                  f"mIoU={miou:.4f}  IoU={iou_v:.4f}  |  ETA total: {_fmt(eta_sec)}")

        # LR schedule step
        if epoch <= args.warmup_epochs:
            warm_lr = args.lr * epoch / max(1, args.warmup_epochs)
            for g in opt.param_groups:
                g['lr'] = warm_lr
        else:
            scheduler.step()

        # checkpoint
        ckpt_path = f"{args.out}/checkpoints/ep{epoch:03d}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       f"{args.out}/checkpoints/best.pth")

        # ---- dump training outputs (geometry-correct prob + bin) ----
        do_dump = (args.dump_train_every > 0 and (epoch % args.dump_train_every == 0)) or (epoch == args.epochs)
        if do_dump:
            model.eval()
            out_prob = Path(f"{args.out}/train/prob")
            out_bin = Path(f"{args.out}/train/bin")
            with torch.no_grad():
                for ti, img_path in enumerate(train_imgs):
                    img_rgb = read_rgb(img_path)
                    img_chw = preprocess_green_clahe_gamma(img_rgb)  # CHW

                    # Apply same pad+resize as training, but record geometry
                    img_sq, orig_hw, pads = pad_resize_img_meta(img_chw, size=args.size)

                    img_t = torch.from_numpy(img_sq).unsqueeze(0).to(device, non_blocking=True)
                    fuse, sides = model(img_t)
                    prob_sq = torch.sigmoid(fuse)[0, 0].cpu().numpy()

                    # Restore to original DRIVE resolution (no augmentation)
                    prob_hw = unpad_resize(prob_sq, orig_hw, pads)

                    stem = Path(img_path).stem
                    p8 = np.clip(prob_hw * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(p8).save(out_prob / f"ep{epoch:03d}_{stem}.png")

                    bin_m = (prob_hw >= args.dump_thresh).astype(np.uint8)
                    Image.fromarray((bin_m * 255).astype(np.uint8)).save(out_bin / f"ep{epoch:03d}_{stem}.png")

    print("Training complete. Best F1:", best_f1)


if __name__ == "__main__":
    main()
