"""Restoration inference utilities and latency/memory benchmarking."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from aerosr.models import count_parameters, model_size_mb


@torch.no_grad()
def restore_image(
    model: nn.Module,
    lr_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Restore a BGR uint8 LR image using a PyTorch SR model."""
    model.eval()
    lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
    lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    sr_tensor = model(lr_tensor)
    sr_rgb = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def benchmark_restoration_model(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 200, 200),
    device: torch.device | None = None,
    warmup: int = 3,
    runs: int = 10,
) -> dict[str, Any]:
    """Measure parameter count, model size, latency (ms), and throughput (FPS)."""
    dev = device or torch.device("cpu")
    model = model.to(dev).eval()
    dummy = torch.randn(*input_shape, device=dev)

    for _ in range(warmup):
        _ = model(dummy)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        if dev.type == "cuda"
        else 0.0
    )

    return {
        "params": count_parameters(model),
        "size_mb": model_size_mb(model),
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "throughput_fps": fps,
        "peak_gpu_mem_mb": peak_mem_mb,
    }
