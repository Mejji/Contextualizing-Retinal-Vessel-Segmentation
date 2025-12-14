# -*- coding: utf-8 -*-
"""
Project-wide config (Windows-safe paths + TF CNN + PyTorch VGN)
Shin-parity keys included.
"""
import os
from pathlib import Path
from easydict import EasyDict as edict

cfg = edict()

# roots
_REPO     = Path(__file__).resolve().parents[0].parent
_DATASETS = Path(os.getenv('THESIS_DATASETS_ROOT', '/workspace/DATASETS'))
_DRIU     = Path(os.getenv('THESIS_DRIU_DRIVE_ROOT', '/workspace/DRIU_DRIVE'))
_VGN_CKPT = Path(os.getenv('THESIS_VGN_CKPT_ROOT', '/workspace/DRIU_DRIVE/VGN'))

#DAU DRIVE paths
_DAU2    = Path(os.getenv('THESIS_DAU2_DRIVE_ROOT', '/workspace/DAU2_DRIVE'))
_VGN_DAU2_CKPT = Path(os.getenv('THESIS_VGN_DAU2_CKPT_ROOT', '/workspace/DAU2_DRIVE/VGN'))


# ================= CNN (TensorFlow) =================
cfg.DEFAULT_DATASET = 'DRIVE'

# Paths for train/test lists (Shin style)
cfg.TRAIN = edict()
cfg.TEST  = edict()
cfg.TRAIN.DRIVE_SET_TXT_PATH      = str(_DATASETS/'DRIVE'/'training'/'images'/'train.txt')
cfg.TEST .DRIVE_SET_TXT_PATH      = str(_DATASETS/'DRIVE'/'test'    /'images'/'test.txt')
cfg.TRAIN.CHASE_DB1_SET_TXT_PATH  = str(_DATASETS/'CHASE_DB1'/'train.txt')
cfg.TEST .CHASE_DB1_SET_TXT_PATH  = str(_DATASETS/'CHASE_DB1'/'test.txt')
cfg.TRAIN.HRF_SET_TXT_PATH        = str(_DATASETS/'HRF'/'train.txt')
cfg.TEST .HRF_SET_TXT_PATH        = str(_DATASETS/'HRF'/'test.txt')

# Where to save
cfg.TRAIN.MODEL_SAVE_PATH = 'train'         # Shin uses 'train'
cfg.TEST.RES_SAVE_PATH    = 'test'
cfg.TEST.WHOLE_IMG_RES_SAVE_PATH = 'test_whole_img'

# Schedule / logging (match Shin defaults)
cfg.TRAIN.BATCH_SIZE     = 1
cfg.TRAIN.DISPLAY        = 10
cfg.TRAIN.TEST_ITERS     = 500
cfg.TRAIN.SNAPSHOT_ITERS = 500
cfg.TRAIN.GRAPH_BATCH_SIZE = 1  # not used by CNN but kept for parity

# Optim / graph constants used by Modules.model
cfg.TRAIN.WEIGHT_DECAY_RATE = 5e-4
cfg.TRAIN.MOMENTUM          = 0.9
cfg.TRAIN.DROPOUT_KEEP_PROB = 0.90
cfg.EPSILON                 = 1e-3  # Shin uses 1e-03

# Data augmentation knobs (Shin)
cfg.TRAIN.USE_LR_FLIPPED = True
cfg.TRAIN.USE_UD_FLIPPED = False
cfg.TRAIN.USE_ROTATION   = False
cfg.TRAIN.ROTATION_MAX_ANGLE = 45
cfg.TRAIN.USE_SCALING    = False
cfg.TRAIN.SCALING_RANGE  = [1.0, 1.25]
cfg.TRAIN.USE_CROPPING   = False
cfg.TRAIN.CROPPING_MAX_MARGIN = 0.05
cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT = True
cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA = 0.2
cfg.TRAIN.USE_CONTRAST_ADJUSTMENT = True
cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR = 0.5
cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR = 1.5
# (aliases some forks expect)
cfg.TRAIN.RANDOM_BRIGHTNESS = cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT
cfg.TRAIN.BRIGHTNESS_DELTA  = cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA
cfg.TRAIN.RANDOM_CONTRAST   = cfg.TRAIN.USE_CONTRAST_ADJUSTMENT
cfg.TRAIN.CONTRAST_LOWER    = cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR
cfg.TRAIN.CONTRAST_UPPER    = cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR
cfg.TRAIN.USE_PADDING       = True
cfg.TRAIN.PADDING_MODE      = 'constant'

# Feature normalization flags seen in Shin config
cfg.USE_BRN       = True
cfg.GN_MIN_NUM_G  = 8
cfg.GN_MIN_CHS_PER_G = 16

# Dataset-specific pixel means (Shin)
cfg.PIXEL_MEAN_DRIVE     = [126.837, 69.015, 41.422]
cfg.PIXEL_MEAN_CHASE_DB1 = [113.953, 39.807, 6.880]
cfg.PIXEL_MEAN_HRF       = [164.420, 51.826, 27.130]
# keep std neutral
cfg.PIXEL_STD_DRIVE      = [1.0, 1.0, 1.0]
cfg.PIXEL_STD_CHASE_DB1  = [1.0, 1.0, 1.0]
cfg.PIXEL_STD_HRF        = [1.0, 1.0, 1.0]

# Canonical sizes / FOV thresh
cfg.DRIVE_INPUT_SHAPE     = (592, 592)
cfg.CHASE_DB1_INPUT_SHAPE = (960, 999)
cfg.HRF_INPUT_SHAPE       = (2336, 3504)
cfg.FOV_BIN_THRESH = 127

# ================= VGN (PyTorch) =================
cfg.VGN_DATA_ROOT = str(_DATASETS / 'DRIVE')
cfg.PATHS = edict()
cfg.PATHS.DRIU_DRIVE_ROOT   = str(_DRIU)
cfg.PATHS.PROBMAP_TRAIN_DIR = str(_DRIU / 'train')
cfg.PATHS.PROBMAP_TEST_DIR  = str(_DRIU / 'test')
cfg.PATHS.GRAPH_TRAIN_DIR   = cfg.PATHS.PROBMAP_TRAIN_DIR
cfg.PATHS.GRAPH_TEST_DIR    = cfg.PATHS.PROBMAP_TEST_DIR

#===== DAU2 Net paths (overrides) =====
cfg.PATHS.DAU2_DRIVE_ROOT   = str(_DAU2)
cfg.PATHS.DAU2_PROBMAP_TRAIN_DIR = str(_DAU2 / 'train' / 'prob')
cfg.PATHS.DAU2_PROBMAP_TEST_DIR  = str(_DAU2 / 'test'  / 'prob')
cfg.PATHS.GRAPH_TRAIN_DIR   = cfg.PATHS.DAU2_PROBMAP_TRAIN_DIR
cfg.PATHS.GRAPH_TEST_DIR    = cfg.PATHS.DAU2_PROBMAP_TEST_DIR
#========================================

cfg.MAX_ITERS      = 50_000
cfg.PRETRAIN_ITERS = 0
cfg.LOG_INTERVAL   = 100
cfg.CKPT_INTERVAL  = 5_000
cfg.BATCH_SIZE     = 4
cfg.NUM_WORKERS    = 8
cfg.SEED           = 1337
cfg.DEVICE         = 'cuda'
cfg.GPU            = '0'

cfg.LR_JOINT_CNN    = 1e-2
cfg.LR_JOINT_OTHERS = 1e-2
cfg.WEIGHT_DECAY    = 5e-4

cfg.DELTA           = 4
cfg.GEODESIC_THRESH = 10
cfg.EDGE_MODE       = 'geodesic'

#==DRIU===
cfg.VGN_SAVE_DIR = str(_VGN_CKPT)
cfg.CNN_INIT     = str(Path(cfg.PATHS.PROBMAP_TRAIN_DIR) / 'iter_50000.ckpt')


#===DAU2Net===
cfg.VGN_SAVE_DIR_DAU = str(_VGN_DAU2_CKPT)
cfg.CNN_INIT_DAU     = str(Path(cfg.PATHS.DAU2_PROBMAP_TRAIN_DIR) / 'checkpoints' / 'best.pth')

cfg.USE_PRECOMPUTED_GRAPH = True
cfg.STRICT_USE_PRECOMP    = True
cfg.PRECOMP = edict()
cfg.PRECOMP.TRAIN_DIR   = cfg.PATHS.GRAPH_TRAIN_DIR
cfg.PRECOMP.TRAIN_RANGE = (21, 40)
cfg.PRECOMP.TEST_DIR    = cfg.PATHS.GRAPH_TEST_DIR
cfg.PRECOMP.TEST_RANGE  = (1, 20)
