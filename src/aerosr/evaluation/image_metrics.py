"""Image reconstruction quality metrics (PSNR, SSIM)."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(gt_bgr: np.ndarray, pred_bgr: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two uint8 images."""
    if gt_bgr.shape != pred_bgr.shape:
        raise ValueError(f"Shape mismatch: {gt_bgr.shape} vs {pred_bgr.shape}")
    return float(peak_signal_noise_ratio(gt_bgr, pred_bgr, data_range=255))


def compute_ssim(gt_bgr: np.ndarray, pred_bgr: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two uint8 images."""
    if gt_bgr.shape != pred_bgr.shape:
        raise ValueError(f"Shape mismatch: {gt_bgr.shape} vs {pred_bgr.shape}")
    return float(structural_similarity(gt_bgr, pred_bgr, data_range=255, channel_axis=2))
