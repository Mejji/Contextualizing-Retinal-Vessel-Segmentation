#!/usr/bin/env python3
"""
Quick post-process for saved probability maps:
- Loads *.npy prob maps from a directory.
- Produces min-max stretched visualization (background -> 0, vessels -> 1).
- Produces a hard binary mask at a chosen threshold (optional).
- Optionally runs hysteresis (strong/weak) to keep faint vessels and bridge small gaps.
- Optional post modes:
  * skeleton: 1-pixel skeleton
  * thin: ~1-pixel thinning
  * med: light closing + erosion to get moderately thin, connected vessels
  * light: closing only (connectivity-first, minimal slimming)
  * bridge: tiny dilate->erode to reconnect 1-2 px gaps without much thickening
Extra:
  * minmax-bin: save a binary from the min-max map (no slimming), useful if you want a solid mask matching the min-max visualization

Usage:
  python postprocess_prob_maps.py --root C:/Users/rog/THESIS/DAU_HRF/VGN/test_results/VGN_DAU_run1/HRF
  python postprocess_prob_maps.py --root <dir> --threshold 0.35 --skeletonize
"""

import argparse
import glob
import os

import numpy as np
import skimage.io
from skimage.morphology import skeletonize, thin, disk, binary_dilation, binary_closing, erosion


def hysteresis_mask(prob_norm, strong_thr=0.5, weak_thr=0.2):
    """Keep strong pixels, and weak pixels connected to strong ones."""
    strong = prob_norm > strong_thr
    weak = prob_norm > weak_thr
    # Dilate strong to reach weak neighborhoods, then union and close small gaps
    reach = binary_dilation(strong, disk(2))
    mask = strong | (weak & reach)
    mask = binary_closing(mask, disk(1))
    return mask


def process_file(path, threshold, do_skel, do_thin, do_med, do_light, do_bridge, do_minmax_bin, hyst):
    p = np.load(path).astype(np.float32)
    p_min, p_max = float(p.min()), float(p.max())
    p_mm = (p - p_min) / (p_max - p_min + 1e-6)

    base = os.path.splitext(path)[0]
    out_mm = base + "_viz_minmax.png"

    skimage.io.imsave(out_mm, (np.clip(p_mm, 0, 1) * 255).astype(np.uint8))

    if threshold is not None:
        if do_minmax_bin:
            if hyst:
                mm_bin = hysteresis_mask(p_mm, strong_thr=threshold, weak_thr=threshold * 0.6)
            else:
                mm_bin = (p_mm >= threshold)
            skimage.io.imsave(base + "_viz_minmax_bin.png", (mm_bin.astype(np.uint8) * 255))

        if hyst:
            # Use hysteresis: strong=threshold, weak=threshold*0.6
            mask = hysteresis_mask(p_mm, strong_thr=threshold, weak_thr=threshold * 0.6)
        else:
            mask = (p_mm >= threshold)
        skimage.io.imsave(base + "_viz_bin.png", (mask.astype(np.uint8) * 255))

        if do_skel:
            skel = skeletonize(mask)
            skimage.io.imsave(base + "_viz_skel.png", (skel.astype(np.uint8) * 255))
        if do_thin:
            th = thin(mask)
            skimage.io.imsave(base + "_viz_thin.png", (th.astype(np.uint8) * 255))
        if do_light:
            # Connectivity-first: closing only
            light = binary_closing(mask, disk(1))
            skimage.io.imsave(base + "_viz_light.png", (light.astype(np.uint8) * 255))
        if do_med:
            # Light closing to fill tiny gaps, then a single erosion to slim
            med = binary_closing(mask, disk(1))
            med = erosion(med, disk(1))
            skimage.io.imsave(base + "_viz_med.png", (med.astype(np.uint8) * 255))
        if do_bridge:
            # Tiny bridge: dilate then erode to reconnect 1-2 px gaps
            bridged = binary_dilation(mask, disk(1))
            bridged = erosion(bridged, disk(1))
            skimage.io.imsave(base + "_viz_bridge.png", (bridged.astype(np.uint8) * 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing *.npy prob maps.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for binary mask; set negative to skip binary output.")
    ap.add_argument("--skeletonize", action="store_true",
                    help="Also save a 1-pixel skeleton of the binary mask.")
    ap.add_argument("--hysteresis", action="store_true",
                    help="Use hysteresis (strong/weak) to keep faint vessels connected.")
    ap.add_argument("--thin", action="store_true",
                    help="Save a thinned (~1px) version of the binary mask (_viz_thin.png).")
    ap.add_argument("--medium", action="store_true",
                    help="Save a lightly thinned/connected mask (_viz_med.png) via closing+erosion.")
    ap.add_argument("--light", action="store_true",
                    help="Save a connectivity-first mask (_viz_light.png) via closing only.")
    ap.add_argument("--bridge", action="store_true",
                    help="Save a bridged mask (_viz_bridge.png) via tiny dilate->erode to reconnect small gaps.")
    ap.add_argument("--minmax-bin", action="store_true",
                    help="Also save a binary mask from the min-max map (_viz_minmax_bin.png) using the same threshold/hysteresis.")
    args = ap.parse_args()

    npys = sorted(glob.glob(os.path.join(args.root, "*.npy")))
    if not npys:
        raise SystemExit(f"No .npy files found under {args.root}")

    do_bin = args.threshold >= 0.0
    for p in npys:
        process_file(p, args.threshold if do_bin else None,
                     args.skeletonize, args.thin, args.medium, args.light, args.bridge, args.minmax_bin, args.hysteresis)

    print(f"Processed {len(npys)} files under {args.root}")


if __name__ == "__main__":
    main()
