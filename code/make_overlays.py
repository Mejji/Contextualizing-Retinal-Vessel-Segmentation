# make_overlays.py
# Create labeled overlays for DRIU and DA-U2Net against DRIVE GT.
# Saves per-model overlays and a side-by-side panel.
# Author: MJ ruthless mentor edition

import os, re, glob, argparse
from pathlib import Path
import numpy as np
import cv2

IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif"}

# -----------------------
# IO helpers
# -----------------------
def imread_gray01(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(p)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img > 0).astype(np.uint8)  # treat nonzero as 1
    return img

def imread_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(p)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def ensure_size(img, hw):
    h, w = hw
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if img.ndim==2 else cv2.INTER_LINEAR)
    return img

def write_png(p, arr):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), arr)

# -----------------------
# DRIVE mapping
# -----------------------
def id2_drive_gt_paths(stem_or_path, gt_dir, allow_no1=False):
    """
    Map '01_test' or file path to GT like '01_manual1.gif' (or '01_manual.gif' if allow_no1).
    """
    stem = Path(stem_or_path).stem
    m = re.match(r"^(\d{2})_", stem)  # '01_test'
    if not m:  # fallback: bare '01'
        m = re.match(r"^(\d{2})$", stem)
    if not m:
        # last resort: take first 2 consecutive digits anywhere
        m = re.search(r"(\d{2})", stem)
    if not m:
        return None
    nid = m.group(1)
    gt_dir = Path(gt_dir)
    pats = [f"{nid}_manual1.*"]
    if allow_no1:
        pats.append(f"{nid}_manual.*")
    for pat in pats:
        cands = sorted(gt_dir.glob(pat))
        if cands:
            return str(cands[0])
    return None

# -----------------------
# Prediction locators
# -----------------------
def find_driu_pred(stem, driu_dir, use_prob=False, thr=0.5):
    """
    Find DRIU prediction for a given 'stem' (e.g., 01_test).
    Prefers *_output.png. If not found and use_prob==True, thresholds *_prob.png.
    Returns binary mask (uint8 0/1) or None.
    """
    d = Path(driu_dir)
    # DRIU saved files usually look like '<origname>_output.png' where <origname> may itself contain '.png'
    # Search by substring match on stem.
    out_cands = sorted([p for p in d.glob("*_output.png") if stem in p.name])
    if out_cands:
        pred = imread_gray01(out_cands[0])
        return pred

    if use_prob:
        prob_cands = sorted([p for p in d.glob("*_prob.png") if stem in p.name])
        if prob_cands:
            prob = cv2.imread(str(prob_cands[0]), cv2.IMREAD_UNCHANGED)
            if prob is None: return None
            if prob.ndim == 3:
                prob = cv2.cvtColor(prob, cv2.COLOR_BGR2GRAY)
            pred = (prob.astype(np.float32) / 255.0 >= float(thr)).astype(np.uint8)
            return pred
    return None

def find_dau2_pred(stem, bin_dir):
    """
    DAU2 test saves bin/<stem>.png (foreground=255). Match by stem or by test id (01_*).
    """
    b = Path(bin_dir)
    cands = []
    # exact stem
    cands += list(b.glob(f"{stem}.png"))
    # try just numeric id
    m = re.match(r"^(\d{2})_", stem)
    if m:
        nid = m.group(1)
        cands += list(b.glob(f"{nid}_*.png"))
    if not cands:
        # brute fallback: any file starting with same two digits
        two = re.findall(r"\d{2}", stem)
        if two:
            cands += list(b.glob(f"{two[0]}*.png"))
    if not cands:
        return None
    img = cv2.imread(str(cands[0]), cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 0).astype(np.uint8)

# -----------------------
# Overlay + legend
# -----------------------
COLORS = {
    "TP":   (255, 255, 255), # white
    "FP":   (255,   0, 255), # magenta
    "FN":   (  0, 255, 255), # yellow/cyan-ish for visibility (OpenCV BGR)
    "GT":   (  0, 255,   0), # green contour
    "PRED": (  0,   0, 255), # red contour
}

def make_overlay(rgb, gt, pred, alpha=0.55):
    """Return overlay (BGR) with TP/FP/FN color-coded + GT and Pred contours."""
    h, w = gt.shape
    rgb = ensure_size(rgb, (h, w))
    base = rgb.copy()

    # masks
    tp = ((gt == 1) & (pred == 1)).astype(np.uint8)
    fp = ((gt == 0) & (pred == 1)).astype(np.uint8)
    fn = ((gt == 1) & (pred == 0)).astype(np.uint8)

    # paint TP/FP/FN as solid color layers then alpha-blend
    paint = np.zeros_like(base)
    for m, color in [(tp,COLORS["TP"]), (fp,COLORS["FP"]), (fn,COLORS["FN"])]:
        if np.any(m):
            col = np.zeros_like(base); col[:] = color
            mask3 = np.repeat(m[:, :, None], 3, axis=2)
            paint[mask3>0] = col[mask3>0]
    overlay = cv2.addWeighted(base, 1.0, paint, alpha, 0)

    # draw contours
    def _draw_cnt(img, msk, color, thick=1):
        cnts, _ = cv2.findContours((msk*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, color, thick)

    _draw_cnt(overlay, gt,   COLORS["GT"],   1)
    _draw_cnt(overlay, pred, COLORS["PRED"], 1)

    # legend
    legend = [
        ("GT (green edge)",   COLORS["GT"]),
        ("Pred (red edge)",   COLORS["PRED"]),
        ("TP (white)",        COLORS["TP"]),
        ("FP (magenta)",      COLORS["FP"]),
        ("FN (yellow)",       COLORS["FN"]),
    ]
    x0, y0, sw, sh, pad = 10, 10, 16, 16, 6
    for i, (txt, col) in enumerate(legend):
        y = y0 + i*(sh+pad)
        cv2.rectangle(overlay, (x0,y), (x0+sw,y+sh), col, thickness=-1)
        cv2.putText(overlay, txt, (x0+sw+8, y+sh-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return overlay

def side_by_side(a, b):
    h = max(a.shape[0], b.shape[0])
    w1, w2 = a.shape[1], b.shape[1]
    A = cv2.copyMakeBorder(a, 0, h-a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    B = cv2.copyMakeBorder(b, 0, h-b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    canvas = np.zeros((h, w1+w2, 3), dtype=np.uint8)
    canvas[:h, :w1] = A
    canvas[:h, w1:w1+w2] = B
    # titles
    cv2.putText(canvas, "DRIU", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "DA-U2Net", (w1+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return canvas

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of DRIVE images (RGB).")
    ap.add_argument("--gt_dir", required=True, help="DRIVE GT folder (e.g., .../test/1st_manual).")
    ap.add_argument("--driu_dir", required=True, help="DRIU test outputs folder that contains *_output.png (and *_prob.png).")
    ap.add_argument("--driu_use_prob", action="store_true", help="If *_output.png missing, threshold *_prob.png.")
    ap.add_argument("--driu_thr", type=float, default=0.5, help="Threshold for *_prob.png if --driu_use_prob.")
    ap.add_argument("--dau2_bin", required=True, help="DAU2Net bin folder (predicted masks).")
    ap.add_argument("--out", required=True, help="Output folder for overlays.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Collect image list
    img_list = sorted([p for p in Path(args.images).glob("*") if p.suffix.lower() in IMG_EXTS])
    if not img_list:
        raise RuntimeError("No images found in --images")

    n_ok = n_skip = 0
    for img_path in img_list:
        stem = Path(img_path).stem  # e.g., 01_test
        rgb = imread_rgb(img_path)

        # GT
        gt_path = id2_drive_gt_paths(stem, args.gt_dir, allow_no1=True)
        if gt_path is None:
            print(f"[skip] GT not found for {stem}")
            n_skip += 1; continue
        gt = imread_gray01(gt_path)

        # DRIU pred
        driu_pred = find_driu_pred(stem, args.driu_dir, use_prob=args.driu_use_prob, thr=args.driu_thr)
        if driu_pred is None:
            print(f"[warn] DRIU pred not found for {stem}; skipping this image for DRIU overlay.")
        else:
            driu_pred = ensure_size(driu_pred, gt.shape)
            driu_overlay = make_overlay(ensure_size(rgb, gt.shape), gt, driu_pred)
            write_png(out / f"{stem}_overlay_driu.png", driu_overlay)

        # DAU2 pred
        dau2_pred = find_dau2_pred(stem, args.dau2_bin)
        if dau2_pred is None:
            print(f"[warn] DAU2Net pred not found for {stem}; skipping this image for DAU2 overlay.")
        else:
            dau2_pred = ensure_size(dau2_pred, gt.shape)
            dau2_overlay = make_overlay(ensure_size(rgb, gt.shape), gt, dau2_pred)
            write_png(out / f"{stem}_overlay_dau2.png", dau2_overlay)

        # side-by-side if both exist
        if driu_pred is not None and dau2_pred is not None:
            panel = side_by_side(driu_overlay, dau2_overlay)
            write_png(out / f"{stem}_overlay_compare.png", panel)

        n_ok += 1

    print(f"Done. Saved overlays to: {out}")
    print(f"Processed: {n_ok} images, Skipped: {n_skip} (missing GT or preds).")
    print("Color code: GT edge=green, Pred edge=red, TP=white, FP=magenta, FN=yellow.")
    
if __name__ == "__main__":
    main()
