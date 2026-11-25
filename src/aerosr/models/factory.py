"""Model factory and parameter/size inspection utilities."""

from __future__ import annotations

import torch.nn as nn

from .architectures import ESPCN, FSRCNN, SRCNN, ResidualSRNet

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "espcn": ESPCN,
    "srcnn": SRCNN,
    "fsrcnn": FSRCNN,
    "residual_sr": ResidualSRNet,
}


def build_model(name: str, upscale_factor: int = 4, channels: int = 3) -> nn.Module:
    """Instantiate a super-resolution model by registry name."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[key](upscale_factor=upscale_factor, channels=channels)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Estimate model parameter memory footprint in megabytes (float32)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024.0 * 1024.0)
