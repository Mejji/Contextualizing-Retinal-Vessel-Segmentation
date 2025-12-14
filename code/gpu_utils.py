"""
Lightweight GPU utility shims for TF1-style session code.
Provides a compatible make_tf_session used by test scripts in this repo.
Safe on systems without CUDA or with TensorFlow-DirectML.
"""
from __future__ import annotations
import os
from typing import Iterable, Optional, Union

import tensorflow as tf


def list_gpus():
    """Return TensorFlow physical GPU devices (may be empty)."""
    try:
        return tf.config.list_physical_devices('GPU')
    except Exception:
        return []


def _normalize_visible_gpus(visible_gpus: Optional[Union[str, int, Iterable[int], Iterable[str]]]) -> Optional[str]:
    if visible_gpus is None:
        return None
    if isinstance(visible_gpus, str):
        return visible_gpus
    if isinstance(visible_gpus, int):
        return str(visible_gpus)
    try:
        return ",".join(str(x) for x in visible_gpus)
    except Exception:
        return None


def set_visible_gpus(visible_gpus: Optional[Union[str, int, Iterable[int], Iterable[str]]]) -> None:
    """Set CUDA_VISIBLE_DEVICES early to limit GPU visibility.
    No-op on platforms/backends that ignore CUDA (e.g., DirectML).
    """
    norm = _normalize_visible_gpus(visible_gpus)
    if norm is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = norm


def make_tf_session(
    allow_growth: bool = True,
    per_process_gpu_memory_fraction: Optional[float] = None,
    visible_gpus: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
) -> tf.compat.v1.Session:
    """
    Create a TF1 Session with common GPU options.

    Parameters
    - allow_growth: enable incremental GPU memory allocation
    - per_process_gpu_memory_fraction: cap memory usage (0..1), if provided
    - visible_gpus: indices or csv string to restrict visible GPUs via env var
    """
    # Best-effort: limit visibility before creating the session
    try:
        set_visible_gpus(visible_gpus)
    except Exception:
        pass

    # Build a TF1-style ConfigProto
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False

    try:
        config.gpu_options.allow_growth = bool(allow_growth)
        if per_process_gpu_memory_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = float(per_process_gpu_memory_fraction)
    except Exception:
        # Some backends (e.g., CPU-only, DirectML) may ignore/omit gpu_options
        pass

    # Ensure v2 behavior is disabled for session execution
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    return tf.compat.v1.Session(config=config)
