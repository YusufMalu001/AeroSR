"""Configurable and deterministic synthetic degradation pipeline for aerial imagery.

Supports independently configurable:
- Downsampling (bicubic, bilinear, area, lanczos)
- Upsampling (bicubic, bilinear, nearest, lanczos)
- Gaussian blur (HR-space or legacy LR-space)
- Additive Gaussian noise (with explicit numpy.random.Generator)
- Atmospheric haze (uniform scattering model)
- JPEG compression artifacts
- Severity presets and randomized parameter ranges for training
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

INTERPOLATION_MAP: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


@dataclass
class DegradationConfig:
    """Configuration for synthetic image degradation."""

    scale_factor: int = 4
    downsample_mode: str = "bicubic"
    upsample_mode: str = "bicubic"
    apply_downsample: bool = True

    # Blur
    apply_blur: bool = True
    blur_kernel: int = 5
    blur_sigma: float = 0.0
    blur_in_hr: bool = True

    # Noise
    apply_noise: bool = True
    noise_sigma: float = 15.0

    # Haze
    apply_haze: bool = True
    haze_t: float = 0.85
    haze_A: float = 240.0

    # JPEG compression
    apply_jpeg: bool = False
    jpeg_quality: int = 75

    # Randomized ranges (used when randomize=True)
    randomize: bool = False
    blur_kernel_choices: tuple[int, ...] = (3, 5, 7)
    noise_sigma_range: tuple[float, float] = (5.0, 25.0)
    haze_t_range: tuple[float, float] = (0.65, 0.95)
    jpeg_quality_range: tuple[int, int] = (50, 95)

    seed: int | None = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DegradationConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        for key in ("blur_kernel_choices", "noise_sigma_range", "haze_t_range", "jpeg_quality_range"):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        return cls(**filtered)


def synthesize_haze(image: np.ndarray, t: float = 0.85, A: float = 240.0) -> np.ndarray:
    """Apply uniform atmospheric scattering model: I(x) = J(x)*t + A*(1 - t)."""
    img_float = image.astype(np.float32)
    hazed = img_float * t + A * (1.0 - t)
    return np.clip(hazed, 0.0, 255.0).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG compression and decompression in memory."""
    quality = int(np.clip(quality, 1, 100))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode(".jpg", image, encode_param)
    if not success:
        raise RuntimeError("Failed to encode image for JPEG degradation.")
    decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode JPEG-compressed image.")
    return decoded


class DegradationPipeline:
    """Deterministic, configurable degradation pipeline."""

    def __init__(self, config: DegradationConfig | None = None) -> None:
        self.config = config or DegradationConfig()

    def _sample_params(self, rng: np.random.Generator) -> tuple[int, float, float, int]:
        cfg = self.config
        if not cfg.randomize:
            return cfg.blur_kernel, cfg.noise_sigma, cfg.haze_t, cfg.jpeg_quality

        blur_k = int(rng.choice(cfg.blur_kernel_choices))
        noise_s = float(rng.uniform(cfg.noise_sigma_range[0], cfg.noise_sigma_range[1]))
        haze_t = float(rng.uniform(cfg.haze_t_range[0], cfg.haze_t_range[1]))
        jpeg_q = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))
        return blur_k, noise_s, haze_t, jpeg_q

    def __call__(
        self,
        image: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Degrade an HR image and return (lr_or_degraded, upscaled_baseline).

        If apply_downsample is False, lr_or_degraded has the same spatial size as input
        and upscaled_baseline is a copy of lr_or_degraded.
        """
        if image is None or image.ndim != 3:
            raise ValueError("Input image must be a 3D (H, W, C) numpy array.")

        cfg = self.config
        effective_seed = seed if seed is not None else cfg.seed
        rng = np.random.default_rng(effective_seed)

        blur_k, noise_s, haze_t, jpeg_q = self._sample_params(rng)
        h, w = image.shape[:2]
        work = image.copy()

        # 1. Optional HR-space blur (physically realistic optical/motion blur before sensor sampling)
        if cfg.apply_blur and cfg.blur_in_hr and blur_k > 0:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            work = cv2.GaussianBlur(work, (k, k), cfg.blur_sigma)

        # 2. Optional downsampling
        if cfg.apply_downsample and cfg.scale_factor > 1:
            interp_down = INTERPOLATION_MAP.get(cfg.downsample_mode, cv2.INTER_CUBIC)
            lr_w = max(1, w // cfg.scale_factor)
            lr_h = max(1, h // cfg.scale_factor)
            work = cv2.resize(work, (lr_w, lr_h), interpolation=interp_down)

        # 3. Optional LR-space blur (legacy compatibility)
        if cfg.apply_blur and not cfg.blur_in_hr and blur_k > 0:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            work = cv2.GaussianBlur(work, (k, k), cfg.blur_sigma)

        # 4. Atmospheric haze
        if cfg.apply_haze and haze_t < 1.0:
            work = synthesize_haze(work, t=haze_t, A=cfg.haze_A)

        # 5. Additive Gaussian sensor noise
        if cfg.apply_noise and noise_s > 0:
            noise = rng.normal(0.0, noise_s, work.shape).astype(np.float32)
            work = np.clip(work.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        # 6. Optional JPEG compression artifacts
        if cfg.apply_jpeg:
            work = apply_jpeg_compression(work, quality=jpeg_q)

        # 7. Upscaled baseline (same spatial dimensions as HR input)
        if cfg.apply_downsample and cfg.scale_factor > 1:
            interp_up = INTERPOLATION_MAP.get(cfg.upsample_mode, cv2.INTER_CUBIC)
            upscaled = cv2.resize(work, (w, h), interpolation=interp_up)
        else:
            upscaled = work.copy()

        return work, upscaled


def degrade_image(
    image: np.ndarray,
    scale_factor: int = 4,
    noise_sigma: float = 15.0,
    blur_kernel: int = 5,
    haze_t: float = 0.85,
    haze_A: float = 240.0,
    apply_haze: bool = True,
    blur_in_hr: bool = True,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around DegradationPipeline."""
    cfg = DegradationConfig(
        scale_factor=scale_factor,
        noise_sigma=noise_sigma,
        blur_kernel=blur_kernel,
        haze_t=haze_t,
        haze_A=haze_A,
        apply_haze=apply_haze,
        blur_in_hr=blur_in_hr,
        seed=seed,
    )
    return DegradationPipeline(cfg)(image, seed=seed)
