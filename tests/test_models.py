"""Unit tests for super-resolution model architectures."""

import pytest
import torch

from aerosr.models import MODEL_REGISTRY, build_model, count_parameters, model_size_mb


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY.keys()))
def test_model_forward_shape(model_name: str):
    model = build_model(model_name, upscale_factor=4, channels=3)
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 3, 64, 64)
    assert count_parameters(model) > 0
    assert model_size_mb(model) > 0.0
