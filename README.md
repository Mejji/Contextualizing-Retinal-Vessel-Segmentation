# Contextualizing-Retinal-Vessel-Segmentation
Contextualizing Retinal Vessel Segmentation:  An Attention-based Adaptive Fusion of Local Features and Global Topology

# Retinal Vessel Segmentation (CNN + VGN + AFF)

End-to-end code for retinal vessel segmentation that pairs a TensorFlow CNN (DRIU variants) with a graph refinement module (Vessel Graph Network). The repo includes preprocessing utilities, training/inference scripts, evaluation, and a small Tkinter demo for qualitative browsing.

## Repository Layout
- [code/](code/) – CNN and VGN training/testing, utilities (DataLayer, graph helpers), configs.
- [system-demo/](system-demo/) – Tkinter viewer for qualitative comparisons; expects pre-rendered assets.
- [Boot Straps/](Boot%20Straps/) – analysis notebooks and CSV outputs for experiments.
- [DATASETS/](DATASETS/) – expected location for DRIVE/CHASE_DB1/HRF datasets (not bundled).
- [pretrained_model/VGG_imagenet.npy](pretrained_model/VGG_imagenet.npy) – VGG weights for CNN initialization.
- [results/](results/) and `*/VGN*` – optional output roots for saved probability maps, segmentations, graphs.

## Data Setup
- Default dataset root is `THESIS_DATASETS_ROOT` (falls back to `C:/Users/rog/THESIS/DATASETS`). Adjust paths in [code/config.py](code/config.py) or set env vars:
  - `THESIS_DATASETS_ROOT` for DRIVE/CHASE_DB1/HRF
  - `THESIS_DRIU_DRIVE_ROOT` for CNN/VGN outputs (e.g., DRIU_DRIVE)
  - `THESIS_VGN_CKPT_ROOT` and `THESIS_VGN_DAU2_CKPT_ROOT` for graph checkpoints
- Expected dataset structure (DRIVE example):
  ```
  DATASETS/
    DRIVE/
      training/images/*.tif, training/1st_manual/*.gif, training/mask/*.gif
      test/images/*.tif, test/1st_manual/*.gif, test/mask/*.gif
    CHASE_DB1/train.txt, test.txt (image paths) and image/label/mask folders
    HRF/train.txt, test.txt (image paths) and images/mask/manual1 folders
  ```
- `train.txt` / `test.txt` should list absolute or repo-relative image paths (one per line). See the defaults in [code/config.py](code/config.py).

## Environment
Target Python 3.9–3.10. GPU is recommended for full training; CPU is fine for smoke tests.

### Conda
```bash
conda env create -f environment.yml
conda activate thesis
```

### Pip
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch CUDA builds: install from the official index, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

## Quick Smoke Tests
Small sanity runs that do not need full datasets (limit images, skip restores):
```bash
# CNN inference smoke (no checkpoint restore)
python code/test_CNN.py --dataset DRIVE --skip_restore --limit_images 2 --save_root tmp_test_results/run1

# CNN training smoke (short run)
python code/train_CNN.py --dataset DRIVE --max_iters 25 --save_root tmp_train_run
```

## CNN Training / Inference (TensorFlow 1 on TF2 runtime)
- Training: saves checkpoints under `<save_root>/train` and validation outputs under `<save_root>/test`.
```bash
python code/train_CNN.py --dataset DRIVE --cnn_model driu --max_iters 50000 \
  --save_root DRIU_DRIVE --pretrained_model pretrained_model/VGG_imagenet.npy
```
- Evaluation / export probability maps and binaries (writes to `<save_root>/test`):
```bash
python code/test_CNN.py --dataset DRIVE --save_root DRIU_DRIVE --limit_images 0
```
Useful flags: `--run_id/--run_name` to pick a specific training run, `--skip_restore` for pipeline smoke, `--dump_debug` to save stretched prob maps.

## Graph Network (VGN) Training / Inference
- Train VGN with CNN init (reads DRIVE prob maps, writes to `<save_root>/runX/train`):
```bash
python code/train_VGN.py --dataset DRIVE --save_root DRIU_DRIVE/VGN \
  --cnn_init_path DRIU_DRIVE/train/iter_50000.ckpt --run_id 1
```
- Test VGN (auto-discovers best/iter checkpoints under run folder):
```bash
python code/test_VGN.py --dataset DRIVE --save_root DRIU_DRIVE/VGN --run_id 1
```
Match graph params (`--win_size`, `--edge_geo_dist_thresh`, etc.) to the training run. Metrics and PNGs are written beside the checkpoint directory.

## Demo Viewer
Tkinter viewer for staged outputs (fundus, CNN map, graph, final seg, GT).
```bash
python system-demo/demo.py
```
Drop per-model assets under `system-demo/assets/<MODEL>/<DATASET>/` with filenames containing `fundus`, `cnn`, `graph`, `seg`, `gt` so the viewer can align cases.

## Docker (CPU baseline)
Build a CPU-only image and mount data/checkpoints at runtime:
```bash
docker build -t thesis-vessels .
docker run --rm -it -v /abs/path/to/DATASETS:/workspace/DATASETS thesis-vessels \ 
  python code/test_CNN.py --dataset DRIVE --skip_restore --limit_images 2
```
GPU (CUDA/DirectML) builds are not baked in; prefer native installs for GPU training.

## Expected Outputs
- CNN training: checkpoints and logs in `<save_root>/train` plus probability maps and binaries in `<save_root>/test`.
- CNN testing: `_prob.png`, `_prob_inv.png`, `_output.png`, and `.npy` per image under `<save_root>/test`.
- VGN training/testing: graph results, metrics, and checkpoints under `<save_root>/run*/train`.
- Demo: on-screen visualization only; asset folders remain unchanged.

## Troubleshooting
- Set `TF_CPP_MIN_LOG_LEVEL=2` to reduce TF verbosity.
- If a checkpoint is not found, pass `--skip_restore` (smoke) or point `--model_path` to a concrete `.ckpt` prefix.
- HRF images are large; ensure enough RAM/GPU memory or set `AUC_MAX_SAMPLES` env var to downsample metrics.

<img width="895" height="614" alt="{0EC1331D-F5EF-4FCF-BD2B-BD689E300D80}" src="https://github.com/user-attachments/assets/6e4f5793-279a-4dec-984a-ea482e0a927e" />
<img width="895" height="622" alt="{EA9BFE22-ED96-4118-A6E0-DB02DA04475E}" src="https://github.com/user-attachments/assets/eead9c37-507a-4d83-8582-f2431bd8f90d" />
<img width="898" height="621" alt="{65AB6D21-A281-4DE4-94FF-C1603DD12C76}" src="https://github.com/user-attachments/assets/b9318b27-d58d-40c8-b194-a1c0c8efbe04" />
<img width="897" height="618" alt="{366B94B1-E0D0-4052-AA5C-610E3B6902E4}" src="https://github.com/user-attachments/assets/d443adc3-6f71-4af9-8e85-75b47d39fbbf" />
<img width="898" height="622" alt="{BB6CD699-97FE-4D83-A46A-06D5B716B5A2}" src="https://github.com/user-attachments/assets/280443d0-632f-4de0-a9cf-f5ce37d7cbcf" />




