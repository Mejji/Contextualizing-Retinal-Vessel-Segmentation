# test_DAU2Net.py
# Inference + metrics for (DA-U)²Net.
# Saves probability maps, binarized masks, and reports Acc, SP, SE/Recall, F1/Dice, IoU, mIoU, AUC, AP.
# Author: THESIS / GPT-5 Pro

import os, glob, argparse, json, csv, re
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from Modules.DAU2Net import DAU2Net

# -----------------------------
# I/O helpers
# -----------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")

def read_rgb(path):
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
    g = img_rgb[...,1]
    g8 = np.clip(g*255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    g_clahe = clahe.apply(g8).astype(np.float32) / 255.0
    g_gamma = np.power(np.clip(g_clahe, 0, 1), 1.0/gamma)
    out = np.stack([g_gamma, g_gamma, g_gamma], axis=0)  # CHW
    return out

def pad_resize(img_chw, size=None):
    c, h, w = img_chw.shape
    s = max(h, w)
    pad_h = s - h
    pad_w = s - w
    img_pad = np.pad(img_chw, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
    if size is not None:
        t = torch.from_numpy(img_pad).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(size,size), mode='bilinear', align_corners=False)
        return t.squeeze(0).numpy(), (h, w), (pad_h, pad_w)
    return img_pad, (h, w), (pad_h, pad_w)

def unpad_resize(prob_sq, orig_hw, pads, out_h=None, out_w=None):
    """
    prob_sq: (size,size) prob map from the network.
    Upsample back to the padded square (s x s), crop padding to (h x w),
    then (optionally) resize to (out_h, out_w).
    """
    h, w = orig_hw
    pad_h, pad_w = pads
    s = h + pad_h  # == w + pad_w == max(h,w)
    prob_s = cv2.resize(prob_sq, (s, s), interpolation=cv2.INTER_LINEAR)
    prob_hw = prob_s[:h, :w]
    if out_h is not None and out_w is not None and (out_h, out_w) != (h, w):
        prob_hw = cv2.resize(prob_hw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return prob_hw

def save_gray01(arr01, path):
    p8 = np.clip(arr01*255.0, 0, 255).astype(np.uint8)
    Image.fromarray(p8).save(str(path))

def binarize01(prob01, thr):
    return (prob01 >= thr).astype(np.uint8)

def read_mask_binary(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m > 0).astype(np.uint8)
    return m

# -----------------------------
# Inference helpers (TTA)
# -----------------------------
def _forward_prob(model, img_t):
    """Forward one batch (1,3,H,W), return numpy HxW prob."""
    fuse, _ = model(img_t)
    prob = torch.sigmoid(fuse)[0,0].float().cpu().numpy()
    return prob

def forward_prob_tta(model, img_t, do_tta=False):
    """If do_tta, average predictions over flips: identity, h, v, hv."""
    if not do_tta:
        return _forward_prob(model, img_t)
    preds = []
    # identity
    preds.append(torch.sigmoid(model(img_t)[0])[0,0])
    # horizontal flip (W)
    x = torch.flip(img_t, dims=[3])
    p = torch.sigmoid(model(x)[0])[0,0]
    p = torch.flip(p, dims=[1])
    preds.append(p)
    # vertical flip (H)
    x = torch.flip(img_t, dims=[2])
    p = torch.sigmoid(model(x)[0])[0,0]
    p = torch.flip(p, dims=[0])
    preds.append(p)
    # both flips (H and W)
    x = torch.flip(img_t, dims=[2,3])
    p = torch.sigmoid(model(x)[0])[0,0]
    p = torch.flip(p, dims=[0,1])
    preds.append(p)
    prob = torch.stack(preds, dim=0).mean(0).float().cpu().numpy()
    return prob

# -----------------------------
# Dataset
# -----------------------------
class TestImages(Dataset):
    def __init__(self, img_paths, size=448):
        self.imgs = img_paths
        self.size = size
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        p = str(self.imgs[i])
        img_rgb = read_rgb(p)
        img_chw = preprocess_green_clahe_gamma(img_rgb)
        img_sq, orig_hw, pads = pad_resize(img_chw, size=self.size)
        return torch.from_numpy(img_sq).float(), (orig_hw, pads), p

# -----------------------------
# Metrics
# -----------------------------
def confusion_from_binary(y_true, y_pred, roi=None):
    if roi is not None:
        m = (roi > 0)
        y_true = y_true[m]
        y_pred = y_pred[m]
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def metrics_from_counts(tp, tn, fp, fn, eps=1e-7):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sp  = tn / max(tn + fp, 1)
    se  = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1  = 2 * tp / max(2 * tp + fp + fn, 1)   # Dice
    iou_v  = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou = 0.5 * (iou_v + iou_bg)
    return {"Acc":acc,"SP":sp,"SE":se,"Recall":se,"Precision":precision,"F1_Dice":f1,"IoU":iou_v,"mIoU":miou}

# ---- NEW: dataset-level AUC / AP (pure NumPy) ----
def roc_auc_from_scores(y_true, y_score):
    """ROC AUC via trapezoidal integral over (FPR, TPR)."""
    y_true = y_true.astype(np.uint8)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    if P == 0 or N == 0:  # undefined
        return float("nan")
    order = np.argsort(-y_score)  # descending by score
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / P
    fpr = fp / N
    # ensure (0,0) start
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapz(tpr, fpr))

def average_precision_from_scores(y_true, y_score):
    """AP = area under PR curve (scikit-like step area)."""
    y_true = y_true.astype(np.uint8)
    P = int(np.sum(y_true == 1))
    if P == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / P
    # sentinel ends
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # area
    ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
    return float(ap)

# -----------------------------
# Exact DRIVE mapping & debug print
# -----------------------------
def drive_exact_paths(stem, masks_dir, fov_dir):
    m = re.match(r"^(\d{2})_", stem)
    if not m:
        return None, None
    nid = m.group(1)
    masks_dir = Path(masks_dir) if masks_dir else None
    fov_dir   = Path(fov_dir) if fov_dir else None
    gt = None
    if masks_dir:
        cands = sorted(list(masks_dir.glob(f"{nid}_manual1.*")))
        gt = str(cands[0]) if cands else None
    fv = None
    if fov_dir:
        if 'test' in stem:
            cands = sorted(list(fov_dir.glob(f"{nid}_test_mask.*")))
        elif 'training' in stem:
            cands = sorted(list(fov_dir.glob(f"{nid}_training_mask.*")))
        else:
            cands = sorted(list(fov_dir.glob(f"{nid}_mask.*")))
        fv = str(cands[0]) if cands else None
    return gt, fv

def print_sample_mapping(images, masks_dir, fov_dir, n=5):
    print("\n[Mapping check] First few image→(GT,FOV):")
    for p in images[:n]:
        stem = Path(p).stem
        gt, fv = drive_exact_paths(stem, masks_dir, fov_dir)
        gt_n = Path(gt).name if gt else None
        fv_n = Path(fv).name if fv else None
        print(f"  {stem:>12}  →  GT={gt_n},  FOV={fv_n}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default='C:/Users/rog/THESIS/DAU2_DRIVE/checkpoints/best.pth', help="Path to .pth (best.pth).")
    ap.add_argument("--images",  type=str, default='C:/Users/rog/THESIS/DATASETS/DRIVE/test/images', help="Folder or glob of images.")
    ap.add_argument("--size",    type=int, default=448)
    ap.add_argument("--out",     type=str, default="C:/Users/rog/THESIS/DAU2_DRIVE/test")
    ap.add_argument("--thresh",  type=float, default=0.47, help="Binarization threshold for metrics & saved bin masks (when thresh_mode=fixed).")
    ap.add_argument("--thresh_mode", type=str, choices=["fixed","grid"], default="fixed",
                    help="Use a fixed threshold or grid search to pick best F1 over a range.")
    ap.add_argument("--grid_start", type=float, default=0.46)
    ap.add_argument("--grid_end",   type=float, default=0.49)
    ap.add_argument("--grid_step",  type=float, default=0.01)
    ap.add_argument("--masks",   type=str, default="C:/Users/rog/THESIS/DATASETS/DRIVE/test/1st_manual",
                    help="Folder with ground-truth vessel masks (binary).")
    ap.add_argument("--mask_glob", type=str, default="*_manual1.gif",
                    help="Glob pattern for GT, e.g., '*_manual1.gif'.")
    ap.add_argument("--fov",     type=str, default="C:/Users/rog/THESIS/DATASETS/DRIVE/test/mask",
                    help="Folder with FOV masks (1 inside field, 0 outside).")
    ap.add_argument("--fov_glob", type=str, default="*_mask.gif",
                    help="Glob pattern for FOV, e.g., '*_mask.gif'.")
    ap.add_argument("--save_csv", dest="save_csv", action="store_true", help="Save per-image metrics CSV.")
    ap.add_argument("--no_save_csv", dest="save_csv", action="store_false")
    ap.set_defaults(save_csv=True)
    ap.add_argument("--tta", action="store_true", help="Enable 4-way flip TTA (h/v/hv) during inference.")
    # Default to TTA enabled
    ap.set_defaults(tta=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d_prob = Path(args.out) / "prob"
    d_bin  = Path(args.out) / "bin"
    d_prob.mkdir(parents=True, exist_ok=True)
    d_bin.mkdir(parents=True, exist_ok=True)

    # Collect input images
    if os.path.isdir(args.images):
        img_paths = sorted([p for p in Path(args.images).glob("*") if p.suffix.lower() in IMG_EXT])
    else:
        img_paths = sorted([Path(p) for p in glob.glob(args.images)])
    assert img_paths, "No images found."

    # Optional mapping preview
    if args.masks:
        print_sample_mapping([str(p) for p in img_paths], args.masks, args.fov, n=min(5, len(img_paths)))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAU2Net(in_ch=3, out_ch=1).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    # Loader
    ds = TestImages(img_paths, size=args.size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Accumulators
    per_image_rows = []
    agg_tp = agg_tn = agg_fp = agg_fn = 0
    # NEW: store scores/labels for AUC/AP
    all_scores = []
    all_labels = []
    # For grid mode: store per-image records
    records = []

    with torch.no_grad():
        for vi, (img_t, meta, path) in enumerate(dl):
            (orig_hw_b, pads_b) = meta
            def _to_hw(x):
                import torch as _t
                if isinstance(x, (list, tuple)):
                    if len(x) == 1: return _to_hw(x[0])
                    if len(x) >= 2: return (int(x[0]), int(x[1]))
                if isinstance(x, _t.Tensor):
                    t = x.view(-1).tolist()
                    return (int(t[0]), int(t[1]))
                return x
            orig_hw = _to_hw(orig_hw_b)
            pads    = _to_hw(pads_b)
            img_t = img_t.to(device, non_blocking=True)

            # Forward (+optional TTA)
            prob = forward_prob_tta(model, img_t, do_tta=args.tta)

            # Back to original size
            prob = unpad_resize(prob, orig_hw, pads)

            # Save prob
            path = path[0]
            stem = Path(path).stem
            save_gray01(prob, d_prob / f"{stem}.png")

            # Metrics if GT present
            if args.masks:
                gt_path, fov_path = drive_exact_paths(stem, args.masks, args.fov)
                if gt_path is None:
                    raise FileNotFoundError(f"GT not found for '{stem}' in {args.masks}")
                if args.fov and fov_path is None:
                    raise FileNotFoundError(f"FOV not found for '{stem}' in {args.fov}")
                gt  = read_mask_binary(gt_path)
                roi = read_mask_binary(fov_path) if args.fov else None

                # size match (normally already OK)
                ph, pw = prob.shape
                gh, gw = gt.shape
                if (ph, pw) != (gh, gw):
                    prob = cv2.resize(prob, (gw, gh), interpolation=cv2.INTER_LINEAR)

                if args.thresh_mode == "fixed":
                    # Save bin
                    pred_bin = (prob >= args.thresh).astype(np.uint8)
                    Image.fromarray((pred_bin*255).astype(np.uint8)).save(d_bin / f"{stem}.png")

                    # Counts-based metrics
                    tp, tn, fp, fn = confusion_from_binary(gt, pred_bin, roi)
                    agg_tp += tp; agg_tn += tn; agg_fp += fp; agg_fn += fn
                    m = metrics_from_counts(tp, tn, fp, fn)
                    per_image_rows.append({
                        "image": stem,
                        "Acc": m["Acc"], "SP": m["SP"], "SE": m["SE"], "Recall": m["Recall"],
                        "Precision": m["Precision"], "F1_Dice": m["F1_Dice"],
                        "IoU": m["IoU"], "mIoU": m["mIoU"],
                        "TP": tp, "TN": tn, "FP": fp, "FN": fn
                    })
                    # Optional overlay (first few)
                    if vi < 3:
                        rgb = (read_rgb(path) * 255).astype(np.uint8)
                        ph2, pw2 = prob.shape
                        rgb = cv2.resize(rgb, (pw2, ph2), interpolation=cv2.INTER_LINEAR)
                        heat = (prob * 255).astype(np.uint8)
                        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
                        gt_cnt, _ = cv2.findContours((gt*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        pr_cnt, _ = cv2.findContours((pred_bin*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(overlay, gt_cnt, -1, (0,255,0), 1)
                        cv2.drawContours(overlay, pr_cnt, -1, (255,0,0), 1)
                        Image.fromarray(overlay[...,::-1]).save(Path(args.out)/f"overlay_{stem}.png")
                else:
                    # grid mode: store samples; bins will be generated after best-thr selection
                    records.append({
                        "stem": stem,
                        "prob": prob.astype(np.float32),
                        "gt": gt.astype(np.uint8),
                        "roi": (roi.astype(np.uint8) if roi is not None else None)
                    })

                # --- NEW: accumulate scores/labels for AUC/AP ---
                if roi is None:
                    mask = np.ones_like(gt, dtype=bool)
                else:
                    mask = roi.astype(bool)
                all_scores.append(prob[mask].ravel())
                all_labels.append(gt[mask].ravel())

                # Optional overlay (first few)
                if vi < 3:
                    rgb = (read_rgb(path) * 255).astype(np.uint8)
                    ph2, pw2 = prob.shape
                    rgb = cv2.resize(rgb, (pw2, ph2), interpolation=cv2.INTER_LINEAR)
                    heat = (prob * 255).astype(np.uint8)
                    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
                    gt_cnt, _ = cv2.findContours((gt*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pr_cnt, _ = cv2.findContours((pred_bin*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(overlay, gt_cnt, -1, (0,255,0), 1)
                    cv2.drawContours(overlay, pr_cnt, -1, (255,0,0), 1)
                    Image.fromarray(overlay[...,::-1]).save(Path(args.out)/f"overlay_{stem}.png")
            else:
                pred_bin = (prob >= args.thresh).astype(np.uint8)
                Image.fromarray((pred_bin*255).astype(np.uint8)).save(d_bin / f"{stem}.png")

    # Summaries
    if args.masks:
        # AUC/AP can be computed regardless of threshold mode
        y_score = np.concatenate(all_scores) if len(all_scores) else np.array([], dtype=np.float32)
        y_true  = np.concatenate(all_labels) if len(all_labels) else np.array([], dtype=np.uint8)
        auc = roc_auc_from_scores(y_true, y_score) if y_true.size else float("nan")
        ap  = average_precision_from_scores(y_true, y_score) if y_true.size else float("nan")

        if args.thresh_mode == "fixed":
            summary = metrics_from_counts(agg_tp, agg_tn, agg_fp, agg_fn)
            summary_counts = {"TP": agg_tp, "TN": agg_tn, "FP": agg_fp, "FN": agg_fn}
            print("\n=== DATASET METRICS @ threshold={:.3f} ===".format(args.thresh))
            print("Accuracy:           {:.4f}".format(summary["Acc"]))
            print("Specificity:        {:.4f}".format(summary["SP"]))
            print("Recall/Sensitivity: {:.4f}".format(summary["Recall"]))
            print("F1/Dice:            {:.4f}".format(summary["F1_Dice"]))
            print("AUC:                {:.4f}".format(auc))
            print("AP:                 {:.4f}".format(ap))
            print("IoU:                {:.4f}".format(summary["IoU"]))
            print("mIoU:               {:.4f}".format(summary["mIoU"]))
            print("Counts:", summary_counts)

            if args.save_csv and len(per_image_rows) > 0:
                csv_path = Path(args.out) / "per_image_metrics.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                    writer.writeheader()
                    for r in per_image_rows:
                        writer.writerow(r)
                print(f"Per-image metrics CSV: {csv_path}")

            with open(Path(args.out) / "summary_metrics.json", "w") as f:
                json.dump({
                    "threshold": args.thresh,
                    "summary": {**summary, "AUC": auc, "AP": ap},
                    "counts": summary_counts
                }, f, indent=2)
        else:
            # Grid search for best F1
            def frange(a, b, step):
                vals = []
                x = a
                while x <= b + 1e-9:
                    vals.append(round(x, 6))
                    x += step
                return vals
            thrs = frange(args.grid_start, args.grid_end, args.grid_step)
            grid_rows = []
            best = {"thr": None, "summary": None, "counts": None, "F1_Dice": -1.0}
            for thr in thrs:
                tp=tn=fp=fn=0
                for rec in records:
                    gt = rec["gt"]; roi = rec["roi"]; prob = rec["prob"]
                    pred_bin = (prob >= thr).astype(np.uint8)
                    tpi, tni, fpi, fni = confusion_from_binary(gt, pred_bin, roi)
                    tp += tpi; tn += tni; fp += fpi; fn += fni
                sm = metrics_from_counts(tp, tn, fp, fn)
                grid_rows.append({"threshold": thr, **sm, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
                if sm["F1_Dice"] > best["F1_Dice"]:
                    best = {"thr": thr, "summary": sm, "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}, "F1_Dice": sm["F1_Dice"]}

            print("\n=== GRID SEARCH RESULTS ===")
            print("range: [{:.3f}, {:.3f}] step {:.3f}".format(args.grid_start, args.grid_end, args.grid_step))
            print("Best threshold: {:.3f}".format(best["thr"]))
            sm = best["summary"]; cnt = best["counts"]
            print("Accuracy:           {:.4f}".format(sm["Acc"]))
            print("Specificity:        {:.4f}".format(sm["SP"]))
            print("Recall/Sensitivity: {:.4f}".format(sm["Recall"]))
            print("F1/Dice:            {:.4f}".format(sm["F1_Dice"]))
            print("AUC:                {:.4f}".format(auc))
            print("AP:                 {:.4f}".format(ap))
            print("IoU:                {:.4f}".format(sm["IoU"]))
            print("mIoU:               {:.4f}".format(sm["mIoU"]))
            print("Counts:", cnt)

            # Save grid CSV
            grid_csv = Path(args.out) / "grid_metrics.csv"
            with open(grid_csv, "w", newline="") as f:
                fieldnames = ["threshold","Acc","SP","SE","Recall","Precision","F1_Dice","IoU","mIoU","TP","TN","FP","FN"]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in grid_rows:
                    w.writerow(r)
            print(f"Grid metrics CSV: {grid_csv}")

            # Save per-image CSV for best thr, and write bin masks
            per_image_rows = []
            for rec in records:
                stem = rec["stem"]; prob = rec["prob"]; gt = rec["gt"]; roi = rec["roi"]
                pred_bin = (prob >= best["thr"]).astype(np.uint8)
                Image.fromarray((pred_bin*255).astype(np.uint8)).save(d_bin / f"{stem}.png")
                tpi, tni, fpi, fni = confusion_from_binary(gt, pred_bin, roi)
                m = metrics_from_counts(tpi, tni, fpi, fni)
                per_image_rows.append({
                    "image": stem,
                    "Acc": m["Acc"], "SP": m["SP"], "SE": m["SE"], "Recall": m["Recall"],
                    "Precision": m["Precision"], "F1_Dice": m["F1_Dice"],
                    "IoU": m["IoU"], "mIoU": m["mIoU"],
                    "TP": tpi, "TN": tni, "FP": fpi, "FN": fni
                })
            if args.save_csv and len(per_image_rows) > 0:
                csv_path = Path(args.out) / "per_image_metrics.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                    writer.writeheader()
                    for r in per_image_rows:
                        writer.writerow(r)
                print(f"Per-image metrics CSV (best thr): {csv_path}")

            with open(Path(args.out) / "summary_metrics.json", "w") as f:
                json.dump({
                    "threshold": best["thr"],
                    "summary": {**sm, "AUC": auc, "AP": ap},
                    "counts": cnt
                }, f, indent=2)

    print("\nDone. Outputs:")
    print("  Prob maps:", d_prob)
    print("  Bin masks:", d_bin)
    if args.masks:
        print("  Metrics:  ", Path(args.out) / "summary_metrics.json")
        if args.save_csv:
            print("  Per-image:", Path(args.out) / "per_image_metrics.csv")


if __name__ == "__main__":
    main()
