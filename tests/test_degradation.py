"""Unit tests for the controlled degradation framework."""

import numpy as np

from aerosr.degradation import DegradationConfig, DegradationPipeline, degrade_image


def test_degradation_determinism():
    img = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    cfg = DegradationConfig(scale_factor=4, noise_sigma=15.0, blur_kernel=5, apply_haze=True, seed=123)
    pipe = DegradationPipeline(cfg)

    lr1, up1 = pipe(img)
    lr2, up2 = pipe(img)
    assert np.array_equal(lr1, lr2)
    assert np.array_equal(up1, up2)
    assert lr1.shape == (32, 32, 3)
    assert up1.shape == (128, 128, 3)


def test_individual_components():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)

    # Noise only (no downsampling, no blur, no haze)
    cfg_noise = DegradationConfig(
        apply_downsample=False, apply_blur=False, apply_haze=False, apply_noise=True, noise_sigma=10.0, seed=7
    )
    lr_n, up_n = DegradationPipeline(cfg_noise)(img)
    assert lr_n.shape == (64, 64, 3)
    assert up_n.shape == (64, 64, 3)
    assert not np.array_equal(lr_n, img)

    # JPEG only
    cfg_jpeg = DegradationConfig(
        apply_downsample=False, apply_blur=False, apply_haze=False, apply_noise=False, apply_jpeg=True, jpeg_quality=30
    )
    lr_j, _ = DegradationPipeline(cfg_jpeg)(img)
    assert lr_j.shape == (64, 64, 3)


def test_convenience_wrapper():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    lr, bicubic = degrade_image(img, scale_factor=4, seed=42)
    assert lr.shape == (20, 20, 3)
    assert bicubic.shape == (80, 80, 3)
