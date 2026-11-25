"""Unit tests for image quality and detection evaluation utilities."""

import numpy as np

from aerosr.evaluation import categorize_box_size, compute_psnr, compute_ssim
from aerosr.evaluation.detection_metrics import _compute_iou_matrix


def test_psnr_ssim_identical():
    img = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert compute_psnr(img, img) == float("inf")
    assert np.isclose(compute_ssim(img, img), 1.0)


def test_categorize_box_size():
    assert categorize_box_size(0.001) == "small"
    assert categorize_box_size(0.01) == "medium"
    assert categorize_box_size(0.10) == "large"


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    iou = _compute_iou_matrix(a, b)
    assert iou.shape == (1, 2)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 25.0 / 175.0)
