import os
import numpy as np
import skimage.io

"""Compute and print per-channel mean RGB values for supported datasets.
Restored utility.
Update dataset_root_base if paths differ on this machine.
"""

def compute_mean_rgb(dataset_root_path, pattern_func, image_dir, suffix):
    img_dir = os.path.join(dataset_root_path, image_dir)
    names = [pattern_func(f) for f in os.listdir(img_dir) if f.endswith(suffix)]
    names = sorted(list(set(names)))
    rgb_cum = np.zeros(3, dtype=np.float64)
    n_pixels = 0.0
    for n in names:
        path = os.path.join(img_dir, f"{n}{suffix}")
        img = skimage.io.imread(path)
        rgb_cum += img[..., :3].reshape(-1, 3).sum(axis=0)
        n_pixels += img.shape[0] * img.shape[1]
    return rgb_cum / n_pixels

if __name__ == "__main__":
    # Adjust these base paths as needed.
    drive_root = 'C:/Users/rog/THESIS2/DRIVE'
    stare_root = 'C:/Users/rog/THESIS2/STARE'

    if os.path.isdir(drive_root):
        drive_mean = compute_mean_rgb(drive_root, lambda f: f.split('_')[0], 'training/images', '_training.tif')
        print('DRIVE mean RGB:', drive_mean)
    else:
        print('DRIVE root not found:', drive_root)

    if os.path.isdir(stare_root):
        stare_img_dir = os.path.join(stare_root, 'STARE_training')
        names = sorted([f for f in os.listdir(stare_img_dir) if f.lower().endswith('.ppm')])
        rgb_cum = np.zeros(3, dtype=np.float64)
        n_pixels = 0.0
        for f in names:
            p = os.path.join(stare_img_dir, f)
            img = skimage.io.imread(p)
            rgb_cum += img[..., :3].reshape(-1, 3).sum(axis=0)
            n_pixels += img.shape[0] * img.shape[1]
        if n_pixels > 0:
            print('STARE mean RGB:', rgb_cum / n_pixels)
        else:
            print('No STARE images found.')
    else:
        print('STARE root not found:', stare_root)
